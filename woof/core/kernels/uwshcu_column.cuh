// uwshcu_column.cuh -- the UW shallow cumulus column (WRF v4.6.1
// shcu_physics = 2): compute_uwshcu and the module's private routines from
// phys/module_shcu_camuwshcu.F, transcribed in WRF's operation order.
// ArWen's own work, Apache-2.0; WRF is public domain.
//
// Lowering rules (the reference build is WRF's own scalar build, gfortran
// -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize; see
// tools/uwshcu_wrf461_oracle/README.md):
//   * x**2 with an INTEGER 2 is x*x; wue**2._r8 (line 2540, a REAL 2.0) is
//     also x*x at -O2; every other real exponent ((p/p00)**rovcp,
//     PiLCL**(1/rovcp)) is a pow call -> uw_pow.
//   * Constant sub-expressions gfortran folds at compile time carry the
//     oracle's own words (tools/uwshcu_wrf461_oracle/probe_constants.F90),
//     written here as hex.
//   * The one default-REAL literal in the r8 code, 1.e-8 at line 5019, is
//     the float32 rounding widened: R8(1.e-8f).
//   * MAX/MIN: a > b ? a : b.  On a tie between +0 and -0 gfortran -O2's
//     maxsd/minsd operand order is a code-generation accident (the oracle's
//     own -O0 and snan builds disagree with it on the sign of some exact
//     zeros), so the grade counts +0 and -0 as equal and reports sign-only
//     differences separately.
//
// What is not transcribed, because no output depends on it (each is a
// dead store in WRF): the diagnostic *_out/outfld arrays and the exit_/
// limit_ flags; the updraft tracer profiles tru/trflx/trsrc and the
// tracer-positivity loop over m = 4..ncnst (with WRF's five CAM
// constituents m = 4 and 5 are NUMLIQ/NUMICE and the loop body never runs;
// only the NUMLIQ/NUMICE tendencies leave the scheme); cldfrct/concldfrct
// and dpdry0 (copied in, never read); s0 and positive_moisture_single's
// s update (s0 reaches no output); u0_s/v0_s/tr0_s and their *_o copies
// (computed but never written back to u0/v0/tr0, so restoring them is the
// identity); tw0 from findsp.
#ifndef UWSHCU_COLUMN_CUH
#define UWSHCU_COLUMN_CUH

// 0-based column view for the Fortran (0:mkx) arrays.
struct VZ {
    double* p;
    int s;
    __device__ __forceinline__ R8& operator()(int k) const {
        return *reinterpret_cast<R8*>(p + (long long)k * s);
    }
};
__device__ __forceinline__ VZ uwsc_vz(Ws& ws, int mkx) {
    V v = ws.r8(mkx + 1);
    return VZ{v.p, v.s};
}

// Folded constants (gfortran front end, probe_constants.F90).
#define UWSC_DEN_LIQ   0x1.b8d025327b18ep-37   // 4*3.14159*(10.e-6**3)*997
#define UWSC_DEN_ICE   0x1.afc6df1b50ce7p-31   // 4*3.14159*(50.e-6**3)*500
#define UWSC_A1        0x1.144f69ff9ffc5p+4    // 7.5_r8*log(10._r8)
#define UWSC_SQRT_PI6  0x1.c5bf86036d4acp+0    // sqrt(3.141592_r8)
#define UWSC_QRMF2     0x1.8ffffffffffffp+4    // 0.25_r8/rmaxfrac**2
#define UWSC_TWO_25066 0x1.9885813c5beadp-1    // (2._r8/2.5066_r8)

// Module constants, as init_uwshcu sets them (module_shcu_camuwshcu.F:
// 319-328) from module_physics_init.F:4312's actual arguments.
struct UwscK {
    R8 xlv, xlf, xls, cp, zvir, r, g, ep2, p00, rovcp, rpen;
};
__device__ __forceinline__ UwscK uwsc_constants() {
    UwscK c;
    c.xlv = R8(UW_LATVAP);
    c.xlf = R8(UW_LATICE);
    c.xls = c.xlv + c.xlf;
    c.cp = R8(UW_CPAIR);
    c.zvir = R8(UW_ZVIR);
    c.r = R8(UW_RAIR);
    c.g = R8(UW_GRAVIT);
    c.ep2 = R8(UW_EPSILO);                 // mwh2o/mwdry
    c.p00 = R8(1.e5);
    c.rovcp = c.r / c.cp;
    c.rpen = R8(10.0);                     // uwshcu_readnl under WRF_PORT
    return c;
}

__device__ __forceinline__ R8 uwsc_exnf(const UwscK& c, R8 p) {
    return uw_pow(p / c.p00, c.rovcp);
}

// qsat = fqsatd -> vqsatd with len = 1 (uwpbl_wvsat.cuh).
__device__ __forceinline__ void uwsc_qsat(R8 t, R8 p, R8& es, R8& qs, R8& gam) {
    uw_fqsatd(t, p, es, qs, gam);
}

// Exact live prefix of fqsatd for callers that discard its derivative.
__device__ __forceinline__ void uwsc_qsat_only(R8 t, R8 p, R8& es, R8& qs) {
    R8 omeps = R8(1) - R8(UW_EPSILO);
    es = uw_estblf(t);
    qs = R8(UW_EPSILO) * es / (p - omeps * es);
    qs = uw_min(R8(1), qs);
    if (qs < R8(0)) { qs = R8(1); es = p; }
}

// subroutine conden (lines 4730-4797)
__device__ void uwsc_conden_exn(const UwscK& c, R8 p, R8 exn, R8 thl, R8 qt, R8& th,
                            R8& qv, R8& ql, R8& qi, R8& rvls, int& id_check) {
    R8 tc, temps, leff, nu, qc, es, qs, gam;
    tc = thl * exn;
    nu = uw_max(uw_min((R8(268.0) - tc) / R8(20.0), R8(1.0)), R8(0.0));
    leff = (R8(1.0) - nu) * c.xlv + nu * c.xls;
    temps = tc;
    uwsc_qsat_only(temps, p, es, qs);
    rvls = qs;
    if (qs >= qt) {
        id_check = 0;
        qv = qt;
        qc = R8(0.0);
        ql = R8(0.0);
        qi = R8(0.0);
        th = tc / exn;
    } else {
        for (int iteration = 1; iteration <= 10; ++iteration) {
            R8 previous_temps = temps;
            temps = temps + ((tc - temps) * c.cp / leff + qt - rvls) /
                    (c.cp / leff + c.ep2 * leff * rvls / c.r / temps / temps);
            uwsc_qsat_only(temps, p, es, qs);
            rvls = qs;
            // Every remaining iteration has identical input words at an
            // exact fixed point. This preserves the native ten-iteration
            // result; no convergence tolerance or iteration cap is reduced.
            if (__double_as_longlong(temps.v) ==
                    __double_as_longlong(previous_temps.v)) break;
        }
        qc = uw_max(qt - qs, R8(0.0));
        qv = qt - qc;
        ql = qc * (R8(1.0) - nu);
        qi = nu * qc;
        th = temps / exn;
        if (uw_abs((temps - (leff / c.cp) * qc) - tc) >= R8(1.0)) id_check = 1;
        else id_check = 0;
    }
}

// The column already owns Exner factors at fixed mass/interface pressures.
// Reuse those words across calls; dynamic parcel pressures compute them here.
__device__ void uwsc_conden(const UwscK& c, R8 p, R8 thl, R8 qt, R8& th,
                            R8& qv, R8& ql, R8& qi, R8& rvls, int& id_check) {
    uwsc_conden_exn(c, p, uwsc_exnf(c, p), thl, qt, th, qv, ql, qi, rvls, id_check);
}

// subroutine roots (lines 4799-4841)
__device__ void uwsc_roots(R8 a, R8 b, R8 cc, R8& r1, R8& r2, int& status) {
    R8 q;
    status = 0;
    if (a == R8(0.0)) {
        if (b == R8(0.0)) {
            status = 1;
        } else {
            r1 = -(cc / b);
        }
        r2 = r1;
    } else {
        if (b == R8(0.0)) {
            if (a * cc > R8(0.0)) {
                status = 2;
            } else {
                r1 = uw_sqrt(-(cc / a));
            }
            r2 = -r1;
        } else {
            if ((b * b - R8(4.0) * a * cc) < R8(0.0)) {
                status = 3;
            } else {
                q = -(R8(0.5) * (b + uw_sign(R8(1.0), b) *
                                 uw_sqrt(b * b - R8(4.0) * a * cc)));
                r1 = q / a;
                r2 = cc / q;
            }
        }
    }
}

// function slope (lines 4843-4873)
__device__ void uwsc_slope(int mkx, V field, V p0, V out) {
    R8 below, above;
    below = (field(2) - field(1)) / (p0(2) - p0(1));
    for (int k = 2; k <= mkx; ++k) {
        above = (field(k) - field(k - 1)) / (p0(k) - p0(k - 1));
        if (above > R8(0.0)) {
            out(k - 1) = uw_max(R8(0.0), uw_min(above, below));
        } else {
            out(k - 1) = uw_min(R8(0.0), uw_max(above, below));
        }
        below = above;
    }
    out(mkx) = out(mkx - 1);
}

// function qsinvert (lines 4875-4950)
__device__ R8 uwsc_qsinvert(const UwscK& c, R8 qt, R8 thl, R8 psfc) {
    R8 ps, Pis, Ts, err, dlnqsdT, dTdPis, dPisdps, dlnqsdps, derrdps, dps;
    R8 Ti, rhi, TLCL, PiLCL, psmin, dpsmax, es, qs, gam, leff, nu;
    psmin = R8(100.0) * R8(100.0);
    dpsmax = R8(1.0);
    Ti = thl * uw_pow(psfc / c.p00, c.rovcp);
    uwsc_qsat_only(Ti, psfc, es, qs);
    rhi = qt / qs;
    if (rhi <= R8(0.01)) return psmin;
    TLCL = R8(55.0) + R8(1.0) / (R8(1.0) / (Ti - R8(55.0)) - uw_log(rhi) / R8(2840.0));
    PiLCL = TLCL / thl;
    ps = c.p00 * uw_pow(PiLCL, R8(1.0) / c.rovcp);
    for (int i = 1; i <= 10; ++i) {
        Pis = uw_pow(ps / c.p00, c.rovcp);
        Ts = thl * Pis;
        uwsc_qsat(Ts, ps, es, qs, gam);
        err = qt - qs;
        nu = uw_max(uw_min((R8(268.0) - Ts) / R8(20.0), R8(1.0)), R8(0.0));
        leff = (R8(1.0) - nu) * c.xlv + nu * c.xls;
        dlnqsdT = gam * (c.cp / leff) / qs;
        dTdPis = thl;
        dPisdps = c.rovcp * Pis / ps;
        dlnqsdps = -(R8(1.0) / (ps - (R8(1.0) - c.ep2) * es));
        derrdps = -(qs * (dlnqsdT * dTdPis * dPisdps + dlnqsdps));
        dps = -(err / derrdps);
        ps = ps + dps;
        if (ps < R8(0.0)) return psmin;
        if (uw_abs(dps) <= dpsmax) return ps;
    }
    return psmin;
}

// function compute_alpha (lines 4952-4971)
__device__ R8 uwsc_compute_alpha(R8 del_CIN, R8 ke) {
    R8 x0 = R8(0.0), x1;
    for (int iteration = 1; iteration <= 10; ++iteration) {
        x1 = x0 - (uw_exp(-(x0 * ke * del_CIN)) - x0) /
                  (-(ke * del_CIN * uw_exp(-(x0 * ke * del_CIN))) - R8(1.0));
        x0 = x1;
    }
    return x0;
}

// function compute_mumin2 (lines 4973-5000); rmaxfrac is a dummy argument
// here, so rmaxfrac**2 is evaluated at run time.
__device__ R8 uwsc_compute_mumin2(R8 mulcl, R8 rmaxfrac, R8 mulow) {
    R8 x0, x1, ex, ef, exf, f, fs;
    x0 = mulow;
    for (int iteration = 1; iteration <= 10; ++iteration) {
        ex = uw_exp(-(x0 * x0));
        ef = uwsc_erfc(x0);
        exf = ex / ef;
        R8 t1 = ex / R8(2.0) / rmaxfrac;
        R8 t2 = mulcl * R8(2.5066) / R8(2.0);
        f = R8(0.5) * (exf * exf) - R8(0.5) * (t1 * t1) - t2 * t2;
        fs = (R8(2.0) * (exf * exf)) * (exf / R8(UWSC_SQRT_PI6) - x0) +
             (R8(0.5) * x0 * (ex * ex)) / (rmaxfrac * rmaxfrac);
        x1 = x0 - f / fs;
        x0 = x1;
    }
    return x0;
}

// function compute_ppen (lines 5002-5049)
__device__ R8 uwsc_compute_ppen(R8 wtwb, R8 D, R8 bogbot, R8 bogtop, R8 rho0j,
                                R8 dpen) {
    R8 x0, x1, f, fs, SB, s00;
    SB = (bogtop - bogbot) / dpen;
    s00 = bogbot / rho0j - D * wtwb;
    if (D * dpen < R8(1.e-8f)) {               // default-REAL literal
        if (s00 >= R8(0.0)) {
            x0 = dpen;
        } else {
            x0 = uw_max(R8(0.0), uw_min(dpen, -(R8(0.5) * wtwb / s00)));
        }
    } else {
        if (s00 >= R8(0.0)) x0 = dpen;
        else x0 = R8(0.0);
        for (int iteration = 1; iteration <= 5; ++iteration) {
            f = uw_exp(-(R8(2.0) * D * x0)) *
                    (wtwb - (bogbot - SB / (R8(2.0) * D)) / (D * rho0j)) +
                (SB * x0 + bogbot - SB / (R8(2.0) * D)) / (D * rho0j);
            fs = -(R8(2.0) * D * uw_exp(-(R8(2.0) * D * x0)) *
                   (wtwb - (bogbot - SB / (R8(2.0) * D)) / (D * rho0j))) +
                 (SB) / (D * rho0j);
            if (fs >= R8(0.0)) fs = uw_max(fs, R8(1.e-10));
            else fs = uw_min(fs, R8(-1.e-10));
            x1 = x0 - f / fs;
            x0 = x1;
        }
    }
    return -uw_max(R8(0.0), uw_min(dpen, x0));
}

// subroutine fluxbelowinv (lines 5051-5124), WRF_PORT arm
__device__ void uwsc_fluxbelowinv(const UwscK& c, R8 cbmf, VZ ps0, int mkx,
                                  int kinv, R8 dt, R8 xsrc, R8 xmean,
                                  R8 xtopin, R8 xbotin, VZ xflx) {
    R8 rcbmf, rpeff, dp, rr, xtop, xbot, pinv, xtop_ori, xbot_ori;
    for (int k = 0; k <= mkx; ++k) xflx(k) = R8(0.0);
    dp = ps0(kinv - 1) - ps0(kinv);
    xbot = xbotin;
    xtop = xtopin;
    xtop_ori = xtop;
    xbot_ori = xbot;
    rcbmf = (cbmf * c.g * dt) / dp;
    rpeff = (xmean - xtop) / uw_max(R8(1.e-13), (xbot - xtop));
    rpeff = uw_abs(rpeff);
    rpeff = uw_min(uw_max(R8(0.0), rpeff), R8(1.0));
    if (rpeff == R8(0.0) || rpeff == R8(1.0)) {
        xbot = xmean;
        xtop = xmean;
    }
    rr = rpeff / rcbmf;
    pinv = ps0(kinv - 1) - rpeff * dp;
    for (int k = 0; k <= kinv - 1; ++k) {
        xflx(k) = cbmf * (xsrc - xbot) * (ps0(0) - ps0(k)) / (ps0(0) - pinv);
    }
    if (rr <= R8(1.0)) {
        xflx(kinv - 1) = xflx(kinv - 1) - (R8(1.0) - rr) * cbmf * (xtop_ori - xbot_ori);
    }
}

// subroutine positive_moisture_single (lines 5126-5195), without the dead
// s/sten-of-s bookkeeping (s reaches no output; sten's update below reads
// only dql/dqi).
__device__ void uwsc_positive_moisture(const UwscK& c, int mkx, R8 dt,
                                       R8 qvmin, R8 qlmin, R8 qimin, V dp,
                                       V qv, V ql, V qi, V qvten, V qlten,
                                       V qiten, V sten) {
    R8 dql, dqi, dqv = R8(0.0), sum, aa, dum;
    for (int k = mkx; k >= 1; --k) {
        dql = uw_max(R8(0.0), R8(1.0) * qlmin - ql(k));
        dqi = uw_max(R8(0.0), R8(1.0) * qimin - qi(k));
        qlten(k) = qlten(k) + dql / dt;
        qiten(k) = qiten(k) + dqi / dt;
        qvten(k) = qvten(k) - (dql + dqi) / dt;
        sten(k) = sten(k) + c.xlv * (dql / dt) + c.xls * (dqi / dt);
        ql(k) = ql(k) + dql;
        qi(k) = qi(k) + dqi;
        qv(k) = qv(k) - dql - dqi;
        dqv = uw_max(R8(0.0), R8(1.0) * qvmin - qv(k));
        qvten(k) = qvten(k) + dqv / dt;
        qv(k) = qv(k) + dqv;
        if (k != 1) {
            qv(k - 1) = qv(k - 1) - dqv * dp(k) / dp(k - 1);
            qvten(k - 1) = qvten(k - 1) - dqv * dp(k) / dp(k - 1) / dt;
        }
        qv(k) = uw_max(qv(k), qvmin);
        ql(k) = uw_max(ql(k), qlmin);
        qi(k) = uw_max(qi(k), qimin);
    }
    if (dqv > R8(1.e-20)) {
        sum = R8(0.0);
        for (int k = 1; k <= mkx; ++k) {
            if (qv(k) > R8(2.0) * qvmin) sum = sum + qv(k) * dp(k);
        }
        aa = dqv * dp(1) / uw_max(R8(1.e-20), sum);
        if (aa < R8(0.5)) {
            for (int k = 1; k <= mkx; ++k) {
                if (qv(k) > R8(2.0) * qvmin) {
                    dum = aa * qv(k);
                    qv(k) = qv(k) - dum;
                    qvten(k) = qvten(k) - dum / dt;
                }
            }
        }
    }
}

// subroutine findsp (lines 5197-5520) for one column, returning qsp (the
// wet-bulb specific humidity compute_uwshcu reads as qw0_in).  Its two
// endrun calls (no convergence; enthalpy not conserved) set *fatal.
__device__ void uwsc_findsp(int pver, V q, V t, V p, V qsp, int* fatal) {
    const R8 epsqs(UW_EPSILO), hlatv(UW_LATVAP), hlatf(UW_LATICE);
    const R8 rgasv(UW_RH2O), cp(UW_CPAIR), ttrice(UWSC_TTRICE);
    const R8 tmin(UWSC_TMIN);
    R8 omeps, trinv, es, desdt, dqsdt, dgdt, g, weight, hlatsb, hlatvp;
    R8 hltalt, tterm, qs, tc, tt0, t1, q1, dt, dq, dtm, dqm, qvd, a1, tmp;
    R8 rair, r1b, c1, c2, c3, denom, dttol, dqtol, enin, enout, tlim, tsp;
    int doit, iter;
    bool lflg;
    omeps = R8(1.0) - epsqs;
    trinv = R8(1.0) / ttrice;
    a1 = R8(UWSC_A1);
    rair = R8(287.04);
    c3 = rair * a1 / cp;
    dtm = R8(0.0);
    dqm = R8(0.0);
    dttol = R8(1.e-4);
    dqtol = R8(1.e-4);
    tt0 = R8(273.15);
    iter = 8;
    for (int k = 1; k <= pver; ++k) {
        tlim = uw_min(uw_max(t(k), R8(173.0)), R8(373.0));
        es = uw_estblf(tlim);
        denom = p(k) - omeps * es;
        qs = epsqs * es / denom;
        doit = 0;
        enout = R8(1.0);
        if (p(k) > R8(5.0) * es && qs > R8(0.0) && qs < R8(0.5)) {
            qs = uw_min(epsqs * es / denom, R8(1.0));
            tc = tlim - tt0;
            lflg = (tc >= -ttrice && tc < R8(0.0));
            weight = uw_min(-(tc * trinv), R8(1.0));
            hlatsb = hlatv + weight * hlatf;
            hlatvp = hlatv - R8(2369.0) * tc;
            if (tlim < tt0) hltalt = hlatsb;
            else hltalt = hlatvp;
            enin = cp * tlim + hltalt * q(k);
            tmp = q(k) - qs;
            c1 = hltalt * c3;
            c2 = (tlim + R8(36.0)) * (tlim + R8(36.0));
            r1b = c2 / (c2 + c1 * qs);
            qvd = r1b * tmp;
            tsp = tlim + ((hltalt / cp) * qvd);
            es = uw_estblf(tsp);
            qsp(k) = uw_min(epsqs * es / (p(k) - omeps * es), R8(1.0));
        } else {
            doit = 1;
            tsp = tlim;
            qsp(k) = q(k);
            enin = R8(1.0);
        }
        for (int l = 1; l <= iter; ++l) {
            dtm = R8(0.0);
            dqm = R8(0.0);
            if (doit == 0) {
                es = uw_estblf(tsp);
                qs = uw_min(epsqs * es / (p(k) - omeps * es), R8(1.0));
                tc = tsp - tt0;
                lflg = (tc >= -ttrice && tc < R8(0.0));
                weight = uw_min(-(tc * trinv), R8(1.0));
                hlatsb = hlatv + weight * hlatf;
                hlatvp = hlatv - R8(2369.0) * tc;
                if (tsp < tt0) hltalt = hlatsb;
                else hltalt = hlatvp;
                if (lflg) {
                    tterm = R8(UWSC_PCF1) + tc * (R8(UWSC_PCF2) + tc * (R8(UWSC_PCF3) +
                            tc * (R8(UWSC_PCF4) + tc * R8(UWSC_PCF5))));
                } else {
                    tterm = R8(0.0);
                }
                desdt = hltalt * es / (rgasv * tsp * tsp) + tterm * trinv;
                dqsdt = (epsqs + omeps * qs) / (p(k) - omeps * es) * desdt;
                g = enin - (cp * tsp + hltalt * qsp(k));
                dgdt = -(cp + hltalt * dqsdt);
                t1 = tsp - g / dgdt;
                dt = uw_abs(t1 - tsp) / t1;
                tsp = uw_max(t1, tmin);
                es = uw_estblf(tsp);
                q1 = uw_min(epsqs * es / (p(k) - omeps * es), R8(1.0));
                dq = uw_abs(q1 - qsp(k)) / uw_max(q1, R8(1.e-12));
                qsp(k) = q1;
                dtm = uw_max(dtm, dt);
                dqm = uw_max(dqm, dq);
                if (dt < dttol && dq < dqtol) doit = 2;
                enout = cp * tsp + hltalt * qsp(k);
                if (tsp < R8(174.16)) doit = 4;
            }
            if (dtm < dttol && dqm < dqtol) break;      // go to 10
        }
        if (dtm > dttol || dqm > dqtol) {
            if (doit == 0) *fatal = 1;                   // endrun('FINDSP')
        }
        if (doit == 2 && uw_abs((enin - enout) / (enin + enout)) > R8(1.e-4)) {
            *fatal = 1;                                  // endrun('FINDSP')
        }
    }
}

// single_cin (lines 4716-4727)
__device__ __forceinline__ R8 uwsc_single_cin(const UwscK& c, R8 pbot,
                                              R8 thv0bot, R8 ptop, R8 thv0top,
                                              R8 thvubot, R8 thvutop) {
    return ((R8(1.0) - thvubot / thv0bot) + (R8(1.0) - thvutop / thv0top)) *
           (pbot - ptop) /
           (pbot / (c.r * thv0bot * uwsc_exnf(c, pbot)) +
            ptop / (c.r * thv0top * uwsc_exnf(c, ptop)));
}

// getbuoy (lines 4684-4714)
__device__ void uwsc_getbuoy(const UwscK& c, R8 pbot, R8 thv0bot, R8 ptop,
                             R8 thv0top, R8 thvubot, R8 thvutop, R8& plfc,
                             R8& cin) {
    R8 frc;
    if (thvubot > thv0bot && thvutop > thv0top) {
        plfc = pbot;
        return;
    } else if (thvubot <= thv0bot && thvutop <= thv0top) {
        cin = cin - ((thvubot / thv0bot - R8(1.0)) + (thvutop / thv0top - R8(1.0))) *
                    (pbot - ptop) /
                    (pbot / (c.r * thv0bot * uwsc_exnf(c, pbot)) +
                     ptop / (c.r * thv0top * uwsc_exnf(c, ptop)));
    } else if (thvubot > thv0bot && thvutop <= thv0top) {
        frc = (thvutop / thv0top - R8(1.0)) /
              ((thvutop / thv0top - R8(1.0)) - (thvubot / thv0bot - R8(1.0)));
        cin = cin - (thvutop / thv0top - R8(1.0)) * ((ptop + frc * (pbot - ptop)) - ptop) /
                    (pbot / (c.r * thv0bot * uwsc_exnf(c, pbot)) +
                     ptop / (c.r * thv0top * uwsc_exnf(c, ptop)));
    } else {
        frc = (thvubot / thv0bot - R8(1.0)) /
              ((thvubot / thv0bot - R8(1.0)) - (thvutop / thv0top - R8(1.0)));
        plfc = pbot - frc * (pbot - ptop);
        cin = cin - (thvubot / thv0bot - R8(1.0)) * (pbot - plfc) /
                    (pbot / (c.r * thv0bot * uwsc_exnf(c, pbot)) +
                     ptop / (c.r * thv0top * uwsc_exnf(c, ptop)));
    }
}

// thv = th*(1 + zvir*qv - ql - qi), WRF's left-to-right grouping
__device__ __forceinline__ R8 uwsc_thv(const UwscK& c, R8 th, R8 qv, R8 ql, R8 qi) {
    return th * (R8(1.0) + c.zvir * qv - ql - qi);
}

// ---- the column -----------------------------------------------------------
// Inputs are compute_uwshcu's own (layer 1 and interface 0 at the surface).
// The caller fills the working arrays the column then updates in place:
// qv0/ql0/qi0/t0 (the implicit-CIN pass replaces them), p0/z0/dp0/u0/v0/
// trl/tri/ps0/zs0/tke (read only).  qw0 is findsp's qsp of the INPUT qv0/t0
// (compute_uwshcu calls findsp before the column loop).
struct UwscCol {
    // inputs
    V p0, z0, dp0, u0, v0, qv0, ql0, qi0, t0, trl, tri, qw0;
    VZ ps0, zs0, tke;
    R8 pblh, cush, dt;
    // outputs (compute_uwshcu's *_out for this column)
    VZ umf_out, slflx_out, qtflx_out, flxprc1_out, flxsnow1_out;
    V qvten_out, qlten_out, qiten_out, sten_out, uten_out, vten_out,
      qrten_out, qsten_out, evapc_out, cufrc_out, qcu_out, qlu_out, qiu_out,
      qc_out, trtl_out, trti_out;
    R8 precip_out, snow_out, cbmf_out, rliq_out, cnt_out, cnb_out, cush_out;
};

__device__ void uwsc_column(Ws& ws, int mkx, UwscCol& io, int* fatal) {
    WsMark mark(ws);
    const UwscK c = uwsc_constants();
    const R8 r = c.r, g = c.g, cp = c.cp, zvir = c.zvir, xlv = c.xlv, xls = c.xls;
    const R8 dt = io.dt;
    // parameters (lines 1044-1142)
    const int niter_xc = 2, iter_cin = 2;
    const R8 rle(0.1), rkm(14.0), rkfre(1.0), rmaxfrac(0.10), mumin1(0.906);
    const R8 rbuoy(1.0), rdrag(1.0), epsvarw(5.e-4), PGFc(0.7);
    const R8 criqc(0.7e-3), frc_rasn(1.0), kevp(2.e-6);
    const R8 qmin1(1.e-12), qmin2(0.0), qmin3(0.0);   // CAM_INIT's cnst_add qmin

    V p0 = io.p0, z0 = io.z0, dp0 = io.dp0, u0 = io.u0, v0 = io.v0;
    V qv0 = io.qv0, ql0 = io.ql0, qi0 = io.qi0, t0 = io.t0, trl = io.trl, tri = io.tri;
    VZ ps0 = io.ps0, zs0 = io.zs0, tke = io.tke;

    // ---- working arrays (automatic arrays of compute_uwshcu) ----
    V qt0 = ws.r8(mkx), thl0 = ws.r8(mkx), thvl0 = ws.r8(mkx);
    V ssthl0 = ws.r8(mkx), ssqt0 = ws.r8(mkx), ssu0 = ws.r8(mkx), ssv0 = ws.r8(mkx);
    V sstrl = ws.r8(mkx), sstri = ws.r8(mkx);
    V thv0bot = ws.r8(mkx), thv0top = ws.r8(mkx), thvl0bot = ws.r8(mkx), thvl0top = ws.r8(mkx);
    V exn0 = ws.r8(mkx);
    VZ exns0 = uwsc_vz(ws, mkx);
    V qv0_o = ws.r8(mkx), ql0_o = ws.r8(mkx), qi0_o = ws.r8(mkx), t0_o = ws.r8(mkx);
    V qt0_o = ws.r8(mkx), thl0_o = ws.r8(mkx), thvl0_o = ws.r8(mkx);
    V ssthl0_o = ws.r8(mkx), ssqt0_o = ws.r8(mkx);
    V thv0bot_o = ws.r8(mkx), thv0top_o = ws.r8(mkx), thvl0bot_o = ws.r8(mkx), thvl0top_o = ws.r8(mkx);
    VZ umf = uwsc_vz(ws, mkx), emf = uwsc_vz(ws, mkx), slflx = uwsc_vz(ws, mkx);
    VZ qtflx = uwsc_vz(ws, mkx), uflx = uwsc_vz(ws, mkx), vflx = uwsc_vz(ws, mkx);
    VZ ufrc = uwsc_vz(ws, mkx), thlu = uwsc_vz(ws, mkx), qtu = uwsc_vz(ws, mkx);
    VZ uu = uwsc_vz(ws, mkx), vu = uwsc_vz(ws, mkx), wu = uwsc_vz(ws, mkx);
    VZ thvu = uwsc_vz(ws, mkx), thlu_emf = uwsc_vz(ws, mkx), qtu_emf = uwsc_vz(ws, mkx);
    VZ uu_emf = uwsc_vz(ws, mkx), vu_emf = uwsc_vz(ws, mkx);
    VZ trl_emf = uwsc_vz(ws, mkx), tri_emf = uwsc_vz(ws, mkx);
    VZ uemf = uwsc_vz(ws, mkx), flxrain = uwsc_vz(ws, mkx), flxsnow = uwsc_vz(ws, mkx);
    VZ xflx = uwsc_vz(ws, mkx);
    VZ umf_s = uwsc_vz(ws, mkx), slflx_s = uwsc_vz(ws, mkx), qtflx_s = uwsc_vz(ws, mkx);
    V qvten = ws.r8(mkx), qlten = ws.r8(mkx), qiten = ws.r8(mkx), sten = ws.r8(mkx);
    V uten = ws.r8(mkx), vten = ws.r8(mkx), qrten = ws.r8(mkx), qsten = ws.r8(mkx);
    V dwten = ws.r8(mkx), diten = ws.r8(mkx), evapc = ws.r8(mkx), cufrc = ws.r8(mkx);
    V qcu = ws.r8(mkx), qlu = ws.r8(mkx), qiu = ws.r8(mkx), fer = ws.r8(mkx), fdr = ws.r8(mkx);
    V qc = ws.r8(mkx), qc_l = ws.r8(mkx), qc_i = ws.r8(mkx), qtten = ws.r8(mkx), slten = ws.r8(mkx);
    V trtl = ws.r8(mkx), trti = ws.r8(mkx), rei = ws.r8(mkx), uf = ws.r8(mkx), vf = ws.r8(mkx);
    V comsub = ws.r8(mkx), qlten_sink = ws.r8(mkx), qiten_sink = ws.r8(mkx);
    V nlten_sink = ws.r8(mkx), niten_sink = ws.r8(mkx), ntraprd = ws.r8(mkx), ntsnprd = ws.r8(mkx);
    V qv0_star = ws.r8(mkx), ql0_star = ws.r8(mkx), qi0_star = ws.r8(mkx);
    V qvten_s = ws.r8(mkx), qlten_s = ws.r8(mkx), qiten_s = ws.r8(mkx), sten_s = ws.r8(mkx);
    V uten_s = ws.r8(mkx), vten_s = ws.r8(mkx), qrten_s = ws.r8(mkx), qsten_s = ws.r8(mkx);
    V evapc_s = ws.r8(mkx), cufrc_s = ws.r8(mkx), qcu_s = ws.r8(mkx), qlu_s = ws.r8(mkx);
    V qiu_s = ws.r8(mkx), qc_s = ws.r8(mkx), trtl_s = ws.r8(mkx), trti_s = ws.r8(mkx);

    // ---- scalars (all declared up front so the Fortran GO TOs are legal) ----
    int k, kk, km1, kp1, iter, iter_scaleh, iter_xc, id_check, status;
    int klcl, kinv, krel, klfc, kbup, kpen, kinv_o, klcl_o, klfc_o;
    bool id_exit, forcedCu;
    R8 thlsrc, qtsrc, usrc, vsrc, thvlsrc, uplus, vplus, plcl, plfc, prel, wrel;
    R8 ee2, ud2, wtw, wtwb, xc, cridis, scaleh, tscaleh, sigmaw, tkeavg, dpsum;
    R8 dpi, thvlmin, thlxsat, qtxsat, thvxsat, x_cu, x_en, thv_x0, thv_x1;
    R8 thj, qvj, qlj, qij, thvj, thv0j, rho0j, rhos0j, qse, cin, cinlcl;
    R8 pe, dpe, exne, thvebot, thle, qte, ue, ve, thlue, qtue, wue;
    R8 mu, mumin0, mumin2, mulcl, mulclstar, cbmf, wcrit, winv, wlcl, ufrcinv;
    R8 ufrclcl, exql, exqi, ppen, thl0top, thl0bot, qt0bot, qt0top, thvubot;
    R8 thvutop, thlu_top, qtu_top, qlu_top, qiu_top, qlu_mid, qiu_mid, exntop;
    R8 thl0lcl, qt0lcl, thv0lcl, thv0rel, rho0inv, aquad, bquad, cquad, xc1, xc2;
    R8 excessu, excess0, xsat, xs1, xs2, bogbot, bogtop, delbog, drage, expfac;
    R8 qcubelow, qlubelow, qiubelow, rainflx, snowflx, es, qs, gam, qsat_arg;
    R8 xsrc, xmean, xtop, xbot, tmp1, tmp2, rliq, cnt, cnb, cush, pblh, precip, snow;
    R8 qc_lm, qc_im, nc_lm, nc_im, ql_emf_kbup, qi_emf_kbup, nl_emf_kbup, ni_emf_kbup;
    R8 thlten_sub, qtten_sub, qlten_sub, qiten_sub, nlten_sub, niten_sub, thl_prog, qt_prog;
    R8 snowmlt, subsat, evprain, evpsnow, evplimit, evplimit_rain, evplimit_snow;
    R8 evpint_rain, evpint_snow, cbmflimit, ufrcinvbase, winvbase;
    R8 cush_s, precip_s, snow_s, rliq_s, cbmf_s, cnt_s, cnb_s;
    R8 cin_i, cin_f, del_CIN, ke, alpha, cinlcl_i;
    R8 tkeavg_o, thvlmin_o, qtsrc_o, thvlsrc_o, thlsrc_o, usrc_o, vsrc_o, plcl_o;
    R8 plfc_o, thv0lcl_o;
    // values the Fortran leaves undefined on paths the cases never take
    xs1 = R8(0.0); xs2 = R8(0.0); x_cu = R8(0.0); x_en = R8(0.0);
    drage = R8(0.0); bogbot = R8(0.0); bogtop = R8(0.0); rho0j = R8(1.0);
    qlubelow = R8(0.0); qiubelow = R8(0.0); qlj = R8(0.0); qij = R8(0.0);
    thlu_top = R8(0.0); qtu_top = R8(0.0); ke = R8(0.0); cin_i = R8(0.0);
    cinlcl_i = R8(0.0); tkeavg_o = R8(0.0); thvlmin_o = R8(0.0); qtsrc_o = R8(0.0);
    thvlsrc_o = R8(0.0); thlsrc_o = R8(0.0); usrc_o = R8(0.0); vsrc_o = R8(0.0);
    plcl_o = R8(0.0); plfc_o = R8(0.0); thv0lcl_o = R8(0.0); kinv_o = 1; klcl_o = 1;
    klfc_o = 1; cush_s = R8(0.0); precip_s = R8(0.0); snow_s = R8(0.0); rliq_s = R8(0.0);
    cbmf_s = R8(0.0); cnt_s = R8(0.0); cnb_s = R8(0.0); wtwb = R8(0.0); ppen = R8(0.0);
    kbup = 1; kpen = 1; krel = 1; kinv = 1; klcl = 1; klfc = 1;

    // compute_uwshcu's output initialization (lines 1157-1185)
    for (k = 0; k <= mkx; ++k) {
        io.umf_out(k) = R8(0.0); io.slflx_out(k) = R8(0.0); io.qtflx_out(k) = R8(0.0);
        io.flxprc1_out(k) = R8(0.0); io.flxsnow1_out(k) = R8(0.0);
    }
    for (k = 1; k <= mkx; ++k) {
        io.qvten_out(k) = R8(0.0); io.qlten_out(k) = R8(0.0); io.qiten_out(k) = R8(0.0);
        io.sten_out(k) = R8(0.0); io.uten_out(k) = R8(0.0); io.vten_out(k) = R8(0.0);
        io.qrten_out(k) = R8(0.0); io.qsten_out(k) = R8(0.0); io.evapc_out(k) = R8(0.0);
        io.cufrc_out(k) = R8(0.0); io.qcu_out(k) = R8(0.0); io.qlu_out(k) = R8(0.0);
        io.qiu_out(k) = R8(0.0); io.qc_out(k) = R8(0.0);
        io.trtl_out(k) = R8(0.0); io.trti_out(k) = R8(0.0);
    }
    io.precip_out = R8(0.0); io.snow_out = R8(0.0); io.cbmf_out = R8(0.0);
    io.rliq_out = R8(0.0); io.cnt_out = uw_real(mkx); io.cnb_out = R8(0.0);

    id_exit = false;
    pblh = io.pblh;
    cush = io.cush;

    // lines 1319-1363: environment
    for (k = 1; k <= mkx; ++k) exn0(k) = uw_pow(p0(k) / c.p00, c.rovcp);
    for (k = 0; k <= mkx; ++k) exns0(k) = uw_pow(ps0(k) / c.p00, c.rovcp);
    for (k = 1; k <= mkx; ++k) {
        qt0(k) = (qv0(k) + ql0(k) + qi0(k));
        thl0(k) = (t0(k) - xlv * ql0(k) / cp - xls * qi0(k) / cp) / exn0(k);
        thvl0(k) = (R8(1.0) + zvir * qt0(k)) * thl0(k);
    }
    uwsc_slope(mkx, thl0, p0, ssthl0);
    uwsc_slope(mkx, qt0, p0, ssqt0);
    uwsc_slope(mkx, u0, p0, ssu0);
    uwsc_slope(mkx, v0, p0, ssv0);
    uwsc_slope(mkx, trl, p0, sstrl);
    uwsc_slope(mkx, tri, p0, sstri);
    for (k = 1; k <= mkx; ++k) {
        thl0bot = thl0(k) + ssthl0(k) * (ps0(k - 1) - p0(k));
        qt0bot = qt0(k) + ssqt0(k) * (ps0(k - 1) - p0(k));
        uwsc_conden_exn(c, ps0(k - 1), exns0(k - 1), thl0bot, qt0bot, thj, qvj, qlj, qij, qse, id_check);
        if (id_check == 1) { id_exit = true; goto L333; }
        thv0bot(k) = uwsc_thv(c, thj, qvj, qlj, qij);
        thvl0bot(k) = thl0bot * (R8(1.0) + zvir * qt0bot);
        thl0top = thl0(k) + ssthl0(k) * (ps0(k) - p0(k));
        qt0top = qt0(k) + ssqt0(k) * (ps0(k) - p0(k));
        uwsc_conden_exn(c, ps0(k), exns0(k), thl0top, qt0top, thj, qvj, qlj, qij, qse, id_check);
        if (id_check == 1) { id_exit = true; goto L333; }
        thv0top(k) = uwsc_thv(c, thj, qvj, qlj, qij);
        thvl0top(k) = thl0top * (R8(1.0) + zvir * qt0top);
    }
    // lines 1370-1391: the original environment (*_o)
    for (k = 1; k <= mkx; ++k) {
        qv0_o(k) = qv0(k); ql0_o(k) = ql0(k); qi0_o(k) = qi0(k); t0_o(k) = t0(k);
        qt0_o(k) = qt0(k); thl0_o(k) = thl0(k); thvl0_o(k) = thvl0(k);
        ssthl0_o(k) = ssthl0(k); ssqt0_o(k) = ssqt0(k);
        thv0bot_o(k) = thv0bot(k); thv0top_o(k) = thv0top(k);
        thvl0bot_o(k) = thvl0bot(k); thvl0top_o(k) = thvl0top(k);
    }
    // lines 1397-1472: zero the working fields
    for (k = 0; k <= mkx; ++k) {
        umf(k) = R8(0.0); emf(k) = R8(0.0); slflx(k) = R8(0.0); qtflx(k) = R8(0.0);
        uflx(k) = R8(0.0); vflx(k) = R8(0.0); ufrc(k) = R8(0.0);
        thlu(k) = R8(0.0); qtu(k) = R8(0.0); uu(k) = R8(0.0); vu(k) = R8(0.0);
        wu(k) = R8(0.0); thvu(k) = R8(0.0); thlu_emf(k) = R8(0.0); qtu_emf(k) = R8(0.0);
        uu_emf(k) = R8(0.0); vu_emf(k) = R8(0.0); uemf(k) = R8(0.0);
        trl_emf(k) = R8(0.0); tri_emf(k) = R8(0.0);
        flxrain(k) = R8(0.0); flxsnow(k) = R8(0.0);
    }
    for (k = 1; k <= mkx; ++k) {
        qvten(k) = R8(0.0); qlten(k) = R8(0.0); qiten(k) = R8(0.0); sten(k) = R8(0.0);
        uten(k) = R8(0.0); vten(k) = R8(0.0); qrten(k) = R8(0.0); qsten(k) = R8(0.0);
        dwten(k) = R8(0.0); diten(k) = R8(0.0); evapc(k) = R8(0.0); cufrc(k) = R8(0.0);
        qcu(k) = R8(0.0); qlu(k) = R8(0.0); qiu(k) = R8(0.0); fer(k) = R8(0.0);
        fdr(k) = R8(0.0); qc(k) = R8(0.0); qc_l(k) = R8(0.0); qc_i(k) = R8(0.0);
        qtten(k) = R8(0.0); slten(k) = R8(0.0); comsub(k) = R8(0.0);
        qlten_sink(k) = R8(0.0); qiten_sink(k) = R8(0.0); nlten_sink(k) = R8(0.0);
        niten_sink(k) = R8(0.0); trtl(k) = R8(0.0); trti(k) = R8(0.0);
        rei(k) = R8(0.0); ntraprd(k) = R8(0.0); ntsnprd(k) = R8(0.0);
    }
    precip = R8(0.0); snow = R8(0.0); cin = R8(0.0); cbmf = R8(0.0); rliq = R8(0.0);
    cnt = uw_real(mkx); cnb = R8(0.0);
    ufrcinvbase = R8(0.0); ufrclcl = R8(0.0); winvbase = R8(0.0); wlcl = R8(0.0);
    cbmflimit = R8(0.0);
    cinlcl = R8(0.0);

    for (iter = 1; iter <= iter_cin; ++iter) {
        tscaleh = cush;
        cush = R8(-1.0);

        // lines 1519-1532: kinv
        for (k = mkx - 1; k >= 1; --k) {
            if ((pblh + R8(5.0) - zs0(k)) * (pblh + R8(5.0) - zs0(k + 1)) < R8(0.0)) {
                kinv = k + 1;
                goto L15;
            }
        }
        kinv = 1;
    L15:
        if (kinv <= 1) { id_exit = true; goto L333; }

        // lines 1548-1563: PBL-mean tke, minimum thvl
        dpsum = R8(0.0);
        tkeavg = R8(0.0);
        thvlmin = R8(1000.0);
        for (k = 0; k <= kinv - 1; ++k) {
            if (k == 0) {
                dpi = ps0(0) - p0(1);
            } else if (k == (kinv - 1)) {
                dpi = p0(kinv - 1) - ps0(kinv - 1);
            } else {
                dpi = p0(k) - p0(k + 1);
            }
            dpsum = dpsum + dpi;
            tkeavg = tkeavg + dpi * tke(k);
            if (k != 0) thvlmin = uw_min(thvlmin, uw_min(thvl0bot(k), thvl0top(k)));
        }
        tkeavg = tkeavg / dpsum;

        // lines 1573-1580: source air
        qtsrc = qt0(1);
        thvlsrc = thvlmin;
        thlsrc = thvlsrc / (R8(1.0) + zvir * qtsrc);
        usrc = u0(kinv - 1) + ssu0(kinv - 1) * (ps0(kinv - 1) - p0(kinv - 1));
        vsrc = v0(kinv - 1) + ssv0(kinv - 1) * (ps0(kinv - 1) - p0(kinv - 1));

        // lines 1597-1631: LCL
        plcl = uwsc_qsinvert(c, qtsrc, thlsrc, ps0(0));
        for (k = 0; k <= mkx; ++k) {
            if (ps0(k) < plcl) {
                klcl = k;
                goto L25;
            }
        }
        klcl = mkx;
    L25:
        klcl = uw_imax(1, klcl);
        if (plcl < R8(30000.0)) { id_exit = true; goto L333; }
        thl0lcl = thl0(klcl) + ssthl0(klcl) * (plcl - p0(klcl));
        qt0lcl = qt0(klcl) + ssqt0(klcl) * (plcl - p0(klcl));
        uwsc_conden(c, plcl, thl0lcl, qt0lcl, thj, qvj, qlj, qij, qse, id_check);
        if (id_check == 1) { id_exit = true; goto L333; }
        thv0lcl = uwsc_thv(c, thj, qvj, qlj, qij);

        // lines 1656-1749: CIN
        cin = R8(0.0);
        cinlcl = R8(0.0);
        plfc = R8(0.0);
        klfc = mkx;
        if (klcl >= kinv) {
            for (k = kinv; k <= mkx - 1; ++k) {
                if (k < klcl) {
                    thvubot = thvlsrc;
                    thvutop = thvlsrc;
                    cin = cin + uwsc_single_cin(c, ps0(k - 1), thv0bot(k), ps0(k), thv0top(k),
                                                thvubot, thvutop);
                } else if (k == klcl) {
                    thvubot = thvlsrc;
                    thvutop = thvlsrc;
                    cin = cin + uwsc_single_cin(c, ps0(k - 1), thv0bot(k), plcl, thv0lcl,
                                                thvubot, thvutop);
                    cinlcl = uw_max(cin, R8(0.0));
                    cin = cinlcl;
                    thvubot = thvlsrc;
                    uwsc_conden_exn(c, ps0(k), exns0(k), thlsrc, qtsrc, thj, qvj, qlj, qij, qse, id_check);
                    if (id_check == 1) { id_exit = true; goto L333; }
                    thvutop = uwsc_thv(c, thj, qvj, qlj, qij);
                    uwsc_getbuoy(c, plcl, thv0lcl, ps0(k), thv0top(k), thvubot, thvutop,
                                 plfc, cin);
                    if (plfc > R8(0.0)) {
                        klfc = k;
                        goto L35;
                    }
                } else {
                    thvubot = thvutop;
                    uwsc_conden_exn(c, ps0(k), exns0(k), thlsrc, qtsrc, thj, qvj, qlj, qij, qse, id_check);
                    if (id_check == 1) { id_exit = true; goto L333; }
                    thvutop = uwsc_thv(c, thj, qvj, qlj, qij);
                    uwsc_getbuoy(c, ps0(k - 1), thv0bot(k), ps0(k), thv0top(k), thvubot,
                                 thvutop, plfc, cin);
                    if (plfc > R8(0.0)) {
                        klfc = k;
                        goto L35;
                    }
                }
            }
        } else {
            cinlcl = R8(0.0);
            for (k = kinv; k <= mkx - 1; ++k) {
                uwsc_conden_exn(c, ps0(k - 1), exns0(k - 1), thlsrc, qtsrc, thj, qvj, qlj, qij, qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                thvubot = uwsc_thv(c, thj, qvj, qlj, qij);
                uwsc_conden_exn(c, ps0(k), exns0(k), thlsrc, qtsrc, thj, qvj, qlj, qij, qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                thvutop = uwsc_thv(c, thj, qvj, qlj, qij);
                uwsc_getbuoy(c, ps0(k - 1), thv0bot(k), ps0(k), thv0top(k), thvubot, thvutop,
                             plfc, cin);
                if (plfc > R8(0.0)) {
                    klfc = k;
                    goto L35;
                }
            }
        }
    L35:
        cin = uw_max(R8(0.0), cin);
        if (klfc >= mkx) {
            klfc = mkx;
            id_exit = true;
            goto L333;
        }

        // lines 1757-1777
        if (iter == 1) {
            cin_i = cin;
            cinlcl_i = cinlcl;
            ke = rbuoy / (rkfre * tkeavg + epsvarw);
            kinv_o = kinv; klcl_o = klcl; klfc_o = klfc;
            plcl_o = plcl; plfc_o = plfc; tkeavg_o = tkeavg; thvlmin_o = thvlmin;
            qtsrc_o = qtsrc; thvlsrc_o = thvlsrc; thlsrc_o = thlsrc;
            usrc_o = usrc; vsrc_o = vsrc; thv0lcl_o = thv0lcl;
        }

        // lines 1800-2069: implicit CIN
        if (iter != 1) {
            cin_f = cin;
            del_CIN = cin_f - cin_i;          // use_CINcin
            if (del_CIN > R8(0.0)) {
                alpha = uwsc_compute_alpha(del_CIN, ke);
                cin = cin_i + alpha * del_CIN;
                cinlcl = cinlcl_i;
                kinv = kinv_o; klcl = klcl_o; klfc = klfc_o;
                plcl = plcl_o; plfc = plfc_o; tkeavg = tkeavg_o; thvlmin = thvlmin_o;
                qtsrc = qtsrc_o; thvlsrc = thvlsrc_o; thlsrc = thlsrc_o;
                usrc = usrc_o; vsrc = vsrc_o; thv0lcl = thv0lcl_o;
                for (k = 1; k <= mkx; ++k) {
                    qv0(k) = qv0_o(k); ql0(k) = ql0_o(k); qi0(k) = qi0_o(k);
                    t0(k) = t0_o(k); qt0(k) = qt0_o(k); thl0(k) = thl0_o(k);
                    thvl0(k) = thvl0_o(k); ssthl0(k) = ssthl0_o(k); ssqt0(k) = ssqt0_o(k);
                    thv0bot(k) = thv0bot_o(k); thv0top(k) = thv0top_o(k);
                    thvl0bot(k) = thvl0bot_o(k); thvl0top(k) = thvl0top_o(k);
                }
                for (k = 0; k <= mkx; ++k) {
                    umf(k) = R8(0.0); emf(k) = R8(0.0); slflx(k) = R8(0.0);
                    qtflx(k) = R8(0.0); uflx(k) = R8(0.0); vflx(k) = R8(0.0);
                    ufrc(k) = R8(0.0); thlu(k) = R8(0.0); qtu(k) = R8(0.0);
                    uu(k) = R8(0.0); vu(k) = R8(0.0); wu(k) = R8(0.0); thvu(k) = R8(0.0);
                    thlu_emf(k) = R8(0.0); qtu_emf(k) = R8(0.0); uu_emf(k) = R8(0.0);
                    vu_emf(k) = R8(0.0); trl_emf(k) = R8(0.0); tri_emf(k) = R8(0.0);
                }
                for (k = 1; k <= mkx; ++k) {
                    qvten(k) = R8(0.0); qlten(k) = R8(0.0); qiten(k) = R8(0.0);
                    sten(k) = R8(0.0); uten(k) = R8(0.0); vten(k) = R8(0.0);
                    qrten(k) = R8(0.0); qsten(k) = R8(0.0); dwten(k) = R8(0.0);
                    diten(k) = R8(0.0); evapc(k) = R8(0.0); cufrc(k) = R8(0.0);
                    qcu(k) = R8(0.0); qlu(k) = R8(0.0); qiu(k) = R8(0.0);
                    fer(k) = R8(0.0); fdr(k) = R8(0.0); qc(k) = R8(0.0);
                    qc_l(k) = R8(0.0); qc_i(k) = R8(0.0); qtten(k) = R8(0.0);
                    slten(k) = R8(0.0); trtl(k) = R8(0.0); trti(k) = R8(0.0);
                }
                precip = R8(0.0); snow = R8(0.0); rliq = R8(0.0); cbmf = R8(0.0);
                cnt = uw_real(mkx); cnb = R8(0.0);
                ufrcinvbase = R8(0.0); ufrclcl = R8(0.0); winvbase = R8(0.0);
                wlcl = R8(0.0); cbmflimit = R8(0.0);
            } else {
                // explicit CIN: the first pass's saved results are the answer
                for (k = 0; k <= mkx; ++k) {
                    io.umf_out(k) = umf_s(k); io.slflx_out(k) = slflx_s(k);
                    io.qtflx_out(k) = qtflx_s(k);
                }
                for (k = 1; k <= mkx; ++k) {
                    io.qvten_out(k) = qvten_s(k); io.qlten_out(k) = qlten_s(k);
                    io.qiten_out(k) = qiten_s(k); io.sten_out(k) = sten_s(k);
                    io.uten_out(k) = uten_s(k); io.vten_out(k) = vten_s(k);
                    io.qrten_out(k) = qrten_s(k); io.qsten_out(k) = qsten_s(k);
                    io.evapc_out(k) = evapc_s(k); io.cufrc_out(k) = cufrc_s(k);
                    io.qcu_out(k) = qcu_s(k); io.qlu_out(k) = qlu_s(k);
                    io.qiu_out(k) = qiu_s(k); io.qc_out(k) = qc_s(k);
                    io.trtl_out(k) = trtl_s(k); io.trti_out(k) = trti_s(k);
                }
                io.precip_out = precip_s; io.snow_out = snow_s; cush = cush_s;
                io.cbmf_out = cbmf_s; io.rliq_out = rliq_s;
                io.cnt_out = cnt_s; io.cnb_out = cnb_s;
                id_exit = false;
                goto L333;
            }
        }

        // lines 2090-2098: release level
        if (klcl < kinv) {
            krel = kinv;
            prel = ps0(krel - 1);
            thv0rel = thv0bot(krel);
        } else {
            krel = klcl;
            prel = plcl;
            thv0rel = thv0lcl;
        }

        // lines 2147-2231: cloud-base mass flux and updraft fraction
        wcrit = uw_sqrt(R8(2.0) * cin * rbuoy);         // use_CINcin
        sigmaw = uw_sqrt(rkfre * tkeavg + epsvarw);
        mu = wcrit / sigmaw / R8(1.4142);
        if (mu >= R8(3.0)) { id_exit = true; goto L333; }
        rho0inv = ps0(kinv - 1) / (r * thv0top(kinv - 1) * exns0(kinv - 1));
        cbmf = (rho0inv * sigmaw / R8(2.5066)) * uw_exp(-(mu * mu));
        cbmflimit = R8(0.9) * dp0(kinv - 1) / g / dt;
        mumin0 = R8(0.0);
        if (cbmf > cbmflimit) mumin0 = uw_sqrt(-uw_log(R8(2.5066) * cbmflimit / rho0inv / sigmaw));
        mu = uw_max(uw_max(mu, mumin0), mumin1);
        mulcl = uw_sqrt(R8(2.0) * cinlcl * rbuoy) / R8(1.4142) / sigmaw;
        {
            R8 e1 = uw_exp(-(mu * mu)) / R8(2.5066);
            R8 f1 = uwsc_erfc(mu);
            mulclstar = uw_sqrt(uw_max(R8(0.0), R8(2.0) * (e1 * e1) *
                                (R8(1.0) / (f1 * f1) - R8(UWSC_QRMF2))));
        }
        if (mulcl > R8(1.e-8) && mulcl > mulclstar) {
            mumin2 = uwsc_compute_mumin2(mulcl, rmaxfrac, mu);
            if (mu > mumin2) { *fatal |= 2; }           // endrun: 'Critical error in mu'
            mu = uw_max(mu, mumin2);
        }
        cbmf = (rho0inv * sigmaw / R8(2.5066)) * uw_exp(-(mu * mu));
        winv = sigmaw * R8(UWSC_TWO_25066) * uw_exp(-(mu * mu)) / uwsc_erfc(mu);
        ufrcinv = cbmf / winv / rho0inv;
        wtw = winv * winv - R8(2.0) * cinlcl * rbuoy;
        if (wtw <= R8(0.0)) { id_exit = true; goto L333; }
        wlcl = uw_sqrt(wtw);
        ufrclcl = cbmf / wlcl / rho0inv;
        wrel = wlcl;
        if (ufrclcl <= R8(0.0001)) { id_exit = true; goto L333; }
        ufrc(krel - 1) = ufrclcl;
        ufrcinvbase = ufrcinv;
        winvbase = winv;
        for (k = kinv - 1; k <= krel - 1; ++k) {
            umf(k) = cbmf;
            wu(k) = winv;
        }

        // lines 2252-2302: release-level updraft
        emf(krel - 1) = R8(0.0);
        umf(krel - 1) = cbmf;
        wu(krel - 1) = wrel;
        thlu(krel - 1) = thlsrc;
        qtu(krel - 1) = qtsrc;
        uwsc_conden(c, prel, thlsrc, qtsrc, thj, qvj, qlj, qij, qse, id_check);
        if (id_check == 1) { id_exit = true; goto L333; }
        thvu(krel - 1) = uwsc_thv(c, thj, qvj, qlj, qij);
        uplus = R8(0.0);
        vplus = R8(0.0);
        if (krel == kinv) {
            uplus = PGFc * ssu0(kinv) * (prel - ps0(kinv - 1));
            vplus = PGFc * ssv0(kinv) * (prel - ps0(kinv - 1));
        } else {
            for (k = kinv; k <= uw_imax(krel - 1, kinv); ++k) {
                uplus = uplus + PGFc * ssu0(k) * (ps0(k) - ps0(k - 1));
                vplus = vplus + PGFc * ssv0(k) * (ps0(k) - ps0(k - 1));
            }
            uplus = uplus + PGFc * ssu0(krel) * (prel - ps0(krel - 1));
            vplus = vplus + PGFc * ssv0(krel) * (prel - ps0(krel - 1));
        }
        uu(krel - 1) = usrc + uplus;
        vu(krel - 1) = vsrc + vplus;
        pe = R8(0.5) * (prel + ps0(krel));
        dpe = prel - ps0(krel);
        exne = uwsc_exnf(c, pe);
        thvebot = thv0rel;
        thle = thl0(krel) + ssthl0(krel) * (pe - p0(krel));
        qte = qt0(krel) + ssqt0(krel) * (pe - p0(krel));
        ue = u0(krel) + ssu0(krel) * (pe - p0(krel));
        ve = v0(krel) + ssv0(krel) * (pe - p0(krel));

        // lines 2343-3005: the updraft, three passes for the scale height
        scaleh = tscaleh;
        if (tscaleh < R8(0.0)) scaleh = R8(1000.0);
        for (iter_scaleh = 1; iter_scaleh <= 3; ++iter_scaleh) {
            kbup = krel;
            kpen = krel;
            wtw = wlcl * wlcl;
            pe = R8(0.5) * (prel + ps0(krel));
            dpe = prel - ps0(krel);
            exne = uwsc_exnf(c, pe);
            thvebot = thv0rel;
            thle = thl0(krel) + ssthl0(krel) * (pe - p0(krel));
            qte = qt0(krel) + ssqt0(krel) * (pe - p0(krel));
            ue = u0(krel) + ssu0(krel) * (pe - p0(krel));
            ve = v0(krel) + ssv0(krel) * (pe - p0(krel));

            for (k = krel; k <= mkx - 1; ++k) {
                km1 = k - 1;
                thlue = thlu(km1);
                qtue = qtu(km1);
                wue = wu(km1);
                wtwb = wtw;
                for (iter_xc = 1; iter_xc <= niter_xc; ++iter_xc) {
                    wtw = wu(km1) * wu(km1);
                    uwsc_conden(c, pe, thle, qte, thj, qvj, qlj, qij, qse, id_check);
                    if (id_check == 1) { id_exit = true; goto L333; }
                    thv0j = uwsc_thv(c, thj, qvj, qlj, qij);
                    rho0j = pe / (r * thv0j * exne);
                    qsat_arg = thle * exne;
                    uwsc_qsat_only(qsat_arg, pe, es, qs);
                    excess0 = qte - qs;
                    uwsc_conden(c, pe, thlue, qtue, thj, qvj, qlj, qij, qse, id_check);
                    if (id_check == 1) { id_exit = true; goto L333; }
                    if ((qlj + qij) > criqc) {
                        exql = ((qlj + qij) - criqc) * qlj / (qlj + qij);
                        exqi = ((qlj + qij) - criqc) * qij / (qlj + qij);
                        qtue = qtue - exql - exqi;
                        thlue = thlue + (xlv / cp / exne) * exql + (xls / cp / exne) * exqi;
                    }
                    uwsc_conden(c, pe, thlue, qtue, thj, qvj, qlj, qij, qse, id_check);
                    if (id_check == 1) { id_exit = true; goto L333; }
                    thvj = uwsc_thv(c, thj, qvj, qlj, qij);
                    qsat_arg = thlue * exne;
                    uwsc_qsat_only(qsat_arg, pe, es, qs);
                    excessu = qtue - qs;
                    cridis = rle * scaleh;
                    if ((excessu <= R8(0.0) && excess0 <= R8(0.0)) ||
                        (excessu >= R8(0.0) && excess0 >= R8(0.0))) {
                        // wue**2._r8: x*x at the reference flags
                        xc = uw_min(R8(1.0), uw_max(R8(0.0), R8(1.0) - R8(2.0) * rbuoy * g *
                             cridis / (wue * wue) * (R8(1.0) - thvj / thv0j)));
                        aquad = R8(0.0);
                        bquad = R8(0.0);
                        cquad = R8(0.0);
                    } else {
                        xsat = excessu / (excessu - excess0);
                        thlxsat = thlue + xsat * (thle - thlue);
                        qtxsat = qtue + xsat * (qte - qtue);
                        uwsc_conden(c, pe, thlxsat, qtxsat, thj, qvj, qlj, qij, qse, id_check);
                        if (id_check == 1) { id_exit = true; goto L333; }
                        thvxsat = uwsc_thv(c, thj, qvj, qlj, qij);
                        for (kk = 1; kk <= 2; ++kk) {
                            if (kk == 1) {
                                thv_x0 = thvj;
                                thv_x1 = (R8(1.0) - R8(1.0) / xsat) * thvj + (R8(1.0) / xsat) * thvxsat;
                            } else {
                                thv_x1 = thv0j;
                                thv_x0 = (xsat / (xsat - R8(1.0))) * thv0j +
                                         (R8(1.0) / (R8(1.0) - xsat)) * thvxsat;
                            }
                            aquad = wue * wue;
                            bquad = R8(2.0) * rbuoy * g * cridis * (thv_x1 - thv_x0) / thv0j -
                                    R8(2.0) * (wue * wue);
                            cquad = R8(2.0) * rbuoy * g * cridis * (thv_x0 - thv0j) / thv0j +
                                    (wue * wue);
                            if (kk == 1) {
                                if ((bquad * bquad - R8(4.0) * aquad * cquad) >= R8(0.0)) {
                                    uwsc_roots(aquad, bquad, cquad, xs1, xs2, status);
                                    x_cu = uw_min(R8(1.0), uw_max(R8(0.0), uw_min(xsat, uw_min(xs1, xs2))));
                                } else {
                                    x_cu = xsat;
                                }
                            } else {
                                if ((bquad * bquad - R8(4.0) * aquad * cquad) >= R8(0.0)) {
                                    uwsc_roots(aquad, bquad, cquad, xs1, xs2, status);
                                    x_en = uw_min(R8(1.0), uw_max(R8(0.0), uw_max(xsat, uw_min(xs1, xs2))));
                                } else {
                                    x_en = R8(1.0);
                                }
                            }
                        }
                        if (x_cu == xsat) xc = uw_max(x_cu, x_en);
                        else xc = x_cu;
                    }
                    ee2 = xc * xc;
                    ud2 = R8(1.0) - R8(2.0) * xc + xc * xc;
                    rei(k) = (R8(0.5) * rkm / z0(k) / g / rho0j);
                    if (xc > R8(0.5)) {
                        rei(k) = uw_min(rei(k), R8(0.9) * uw_log(dp0(k) / g / dt / umf(km1) + R8(1.0)) /
                                                dpe / (R8(2.0) * xc - R8(1.0)));
                    }
                    fer(k) = rei(k) * ee2;
                    fdr(k) = rei(k) * ud2;
                    umf(k) = umf(km1) * uw_exp(dpe * (fer(k) - fdr(k)));
                    emf(k) = R8(0.0);
                    if (fer(k) * dpe < R8(1.e-4)) {
                        thlu(k) = thlu(km1) + (thle + ssthl0(k) * dpe / R8(2.0) - thlu(km1)) * fer(k) * dpe;
                        qtu(k) = qtu(km1) + (qte + ssqt0(k) * dpe / R8(2.0) - qtu(km1)) * fer(k) * dpe;
                        uu(k) = uu(km1) + (ue + ssu0(k) * dpe / R8(2.0) - uu(km1)) * fer(k) * dpe -
                                PGFc * ssu0(k) * dpe;
                        vu(k) = vu(km1) + (ve + ssv0(k) * dpe / R8(2.0) - vu(km1)) * fer(k) * dpe -
                                PGFc * ssv0(k) * dpe;
                    } else {
                        R8 ef = uw_exp(-(fer(k) * dpe));
                        thlu(k) = (thle + ssthl0(k) / fer(k) - ssthl0(k) * dpe / R8(2.0)) -
                                  (thle + ssthl0(k) * dpe / R8(2.0) - thlu(km1) + ssthl0(k) / fer(k)) * ef;
                        qtu(k) = (qte + ssqt0(k) / fer(k) - ssqt0(k) * dpe / R8(2.0)) -
                                 (qte + ssqt0(k) * dpe / R8(2.0) - qtu(km1) + ssqt0(k) / fer(k)) * ef;
                        uu(k) = (ue + (R8(1.0) - PGFc) * ssu0(k) / fer(k) - ssu0(k) * dpe / R8(2.0)) -
                                (ue + ssu0(k) * dpe / R8(2.0) - uu(km1) + (R8(1.0) - PGFc) * ssu0(k) / fer(k)) * ef;
                        vu(k) = (ve + (R8(1.0) - PGFc) * ssv0(k) / fer(k) - ssv0(k) * dpe / R8(2.0)) -
                                (ve + ssv0(k) * dpe / R8(2.0) - vu(km1) + (R8(1.0) - PGFc) * ssv0(k) / fer(k)) * ef;
                    }
                    uwsc_conden_exn(c, ps0(k), exns0(k), thlu(k), qtu(k), thj, qvj, qlj, qij, qse, id_check);
                    if (id_check == 1) { id_exit = true; goto L333; }
                    if ((qlj + qij) > criqc) {
                        exql = ((qlj + qij) - criqc) * qlj / (qlj + qij);
                        exqi = ((qlj + qij) - criqc) * qij / (qlj + qij);
                        qtu(k) = qtu(k) - exql - exqi;
                        thlu(k) = thlu(k) + (xlv / cp / exns0(k)) * exql + (xls / cp / exns0(k)) * exqi;
                        dwten(k) = exql;
                        diten(k) = exqi;
                    } else {
                        dwten(k) = R8(0.0);
                        diten(k) = R8(0.0);
                    }
                    uwsc_conden_exn(c, ps0(k), exns0(k), thlu(k), qtu(k), thj, qvj, qlj, qij, qse, id_check);
                    if (id_check == 1) { id_exit = true; goto L333; }
                    thvu(k) = uwsc_thv(c, thj, qvj, qlj, qij);
                    bogbot = rbuoy * (thvu(km1) / thvebot - R8(1.0));
                    bogtop = rbuoy * (thvu(k) / thv0top(k) - R8(1.0));
                    delbog = bogtop - bogbot;
                    drage = fer(k) * (R8(1.0) + rdrag);
                    expfac = uw_exp(-(R8(2.0) * drage * dpe));
                    wtwb = wtw;
                    if (drage * dpe > R8(1.e-3)) {
                        wtw = wtw * expfac + (delbog + (R8(1.0) - expfac) *
                              (bogbot + delbog / (-(R8(2.0) * drage * dpe)))) / (rho0j * drage);
                    } else {
                        wtw = wtw + dpe * (bogbot + bogtop) / rho0j;
                    }
                    if (wtw > R8(0.0)) {
                        thlue = R8(0.5) * (thlu(km1) + thlu(k));
                        qtue = R8(0.5) * (qtu(km1) + qtu(k));
                        wue = R8(0.5) * uw_sqrt(uw_max(wtwb + wtw, R8(0.0)));
                    } else {
                        goto L111;
                    }
                }
            L111:
                if (umf(k) == R8(0.0)) wtw = R8(-1.0);
                if (bogtop > R8(0.0) && wtw > R8(0.0)) kbup = k;
                if (wtw <= R8(0.0)) {
                    kpen = k;
                    goto L45;
                }
                wu(k) = uw_sqrt(wtw);
                if (wu(k) > R8(100.0)) { id_exit = true; goto L333; }
                rhos0j = ps0(k) / (r * R8(0.5) * (thv0bot(k + 1) + thv0top(k)) * exns0(k));
                ufrc(k) = umf(k) / (rhos0j * wu(k));
                if (ufrc(k) > rmaxfrac) {
                    ufrc(k) = rmaxfrac;
                    umf(k) = rmaxfrac * rhos0j * wu(k);
                    fdr(k) = fer(k) - uw_log(umf(k) / umf(km1)) / dpe;
                }
                pe = p0(k + 1);
                dpe = dp0(k + 1);
                exne = exn0(k + 1);
                thvebot = thv0bot(k + 1);
                thle = thl0(k + 1);
                qte = qt0(k + 1);
                ue = u0(k + 1);
                ve = v0(k + 1);
            }
        L45:
            // lines 2929-2958: penetration depth
            if (drage == R8(0.0)) {
                aquad = (bogtop - bogbot) / (ps0(kpen) - ps0(kpen - 1));
                bquad = R8(2.0) * bogbot;
                cquad = -((wu(kpen - 1) * wu(kpen - 1)) * rho0j);
                uwsc_roots(aquad, bquad, cquad, xc1, xc2, status);
                if (status == 0) {
                    if (xc1 <= R8(0.0) && xc2 <= R8(0.0)) {
                        ppen = uw_max(xc1, xc2);
                        ppen = uw_min(R8(0.0), uw_max(-dp0(kpen), ppen));
                    } else if (xc1 > R8(0.0) && xc2 > R8(0.0)) {
                        ppen = -dp0(kpen);
                    } else {
                        ppen = uw_min(xc1, xc2);
                        ppen = uw_min(R8(0.0), uw_max(-dp0(kpen), ppen));
                    }
                } else {
                    ppen = -dp0(kpen);
                }
            } else {
                ppen = uwsc_compute_ppen(wtwb, drage, bogbot, bogtop, rho0j, dp0(kpen));
            }
            // lines 2968-3003: updraft top properties, cush
            if (fer(kpen) * (-ppen) < R8(1.e-4)) {
                thlu_top = thlu(kpen - 1) + (thl0(kpen) + ssthl0(kpen) * (-ppen) / R8(2.0) -
                           thlu(kpen - 1)) * fer(kpen) * (-ppen);
                qtu_top = qtu(kpen - 1) + (qt0(kpen) + ssqt0(kpen) * (-ppen) / R8(2.0) -
                          qtu(kpen - 1)) * fer(kpen) * (-ppen);
            } else {
                R8 ef = uw_exp(-(fer(kpen) * (-ppen)));
                thlu_top = (thl0(kpen) + ssthl0(kpen) / fer(kpen) - ssthl0(kpen) * (-ppen) / R8(2.0)) -
                           (thl0(kpen) + ssthl0(kpen) * (-ppen) / R8(2.0) - thlu(kpen - 1) +
                            ssthl0(kpen) / fer(kpen)) * ef;
                qtu_top = (qt0(kpen) + ssqt0(kpen) / fer(kpen) - ssqt0(kpen) * (-ppen) / R8(2.0)) -
                          (qt0(kpen) + ssqt0(kpen) * (-ppen) / R8(2.0) - qtu(kpen - 1) +
                           ssqt0(kpen) / fer(kpen)) * ef;
            }
            uwsc_conden(c, ps0(kpen - 1) + ppen, thlu_top, qtu_top, thj, qvj, qlj, qij, qse, id_check);
            if (id_check == 1) { id_exit = true; goto L333; }
            exntop = uw_pow((ps0(kpen - 1) + ppen) / c.p00, c.rovcp);
            if ((qlj + qij) > criqc) {
                dwten(kpen) = ((qlj + qij) - criqc) * qlj / (qlj + qij);
                diten(kpen) = ((qlj + qij) - criqc) * qij / (qlj + qij);
                qtu_top = qtu_top - dwten(kpen) - diten(kpen);
                thlu_top = thlu_top + (xlv / cp / exntop) * dwten(kpen) + (xls / cp / exntop) * diten(kpen);
            } else {
                dwten(kpen) = R8(0.0);
                diten(kpen) = R8(0.0);
            }
            rhos0j = ps0(kpen - 1) / (r * R8(0.5) * (thv0bot(kpen) + thv0top(kpen - 1)) * exns0(kpen - 1));
            cush = zs0(kpen - 1) - ppen / rhos0j / g;
            scaleh = cush;
        }

        // lines 3017-3069
        forcedCu = (kbup == krel);
        if (forcedCu) { id_exit = true; goto L333; }
        for (k = kpen; k <= mkx; ++k) {
            umf(k) = R8(0.0);
            emf(k) = R8(0.0);
            ufrc(k) = R8(0.0);
        }
        for (k = kpen + 1; k <= mkx; ++k) {
            dwten(k) = R8(0.0); diten(k) = R8(0.0); fer(k) = R8(0.0); fdr(k) = R8(0.0);
        }

        // lines 3108-3236: penetrative entrainment
        for (k = 0; k <= mkx; ++k) {
            thlu_emf(k) = thlu(k);
            qtu_emf(k) = qtu(k);
            uu_emf(k) = uu(k);
            vu_emf(k) = vu(k);
        }
        for (k = kpen - 1; k >= kbup; --k) {
            rhos0j = ps0(k) / (r * R8(0.5) * (thv0bot(k + 1) + thv0top(k)) * exns0(k));
            if (k == kpen - 1) {
                emf(k) = uw_max(uw_max(umf(k) * ppen * rei(kpen) * c.rpen, -(R8(0.1) * rhos0j)),
                                -(R8(0.9) * dp0(kpen) / g / dt));
                thlu_emf(k) = thl0(kpen) + ssthl0(kpen) * (ps0(k) - p0(kpen));
                qtu_emf(k) = qt0(kpen) + ssqt0(kpen) * (ps0(k) - p0(kpen));
                uu_emf(k) = u0(kpen) + ssu0(kpen) * (ps0(k) - p0(kpen));
                vu_emf(k) = v0(kpen) + ssv0(kpen) * (ps0(k) - p0(kpen));
                trl_emf(k) = trl(kpen) + sstrl(kpen) * (ps0(k) - p0(kpen));
                tri_emf(k) = tri(kpen) + sstri(kpen) * (ps0(k) - p0(kpen));
            } else {
                emf(k) = uw_max(uw_max(emf(k + 1) - umf(k) * dp0(k + 1) * rei(k + 1) * c.rpen,
                                       -(R8(0.1) * rhos0j)),
                                -(R8(0.9) * dp0(k + 1) / g / dt));
                if (uw_abs(emf(k)) > uw_abs(emf(k + 1))) {
                    thlu_emf(k) = (thlu_emf(k + 1) * emf(k + 1) + thl0(k + 1) * (emf(k) - emf(k + 1))) / emf(k);
                    qtu_emf(k) = (qtu_emf(k + 1) * emf(k + 1) + qt0(k + 1) * (emf(k) - emf(k + 1))) / emf(k);
                    uu_emf(k) = (uu_emf(k + 1) * emf(k + 1) + u0(k + 1) * (emf(k) - emf(k + 1))) / emf(k);
                    vu_emf(k) = (vu_emf(k + 1) * emf(k + 1) + v0(k + 1) * (emf(k) - emf(k + 1))) / emf(k);
                    trl_emf(k) = (trl_emf(k + 1) * emf(k + 1) + trl(k + 1) * (emf(k) - emf(k + 1))) / emf(k);
                    tri_emf(k) = (tri_emf(k + 1) * emf(k + 1) + tri(k + 1) * (emf(k) - emf(k + 1))) / emf(k);
                } else {
                    thlu_emf(k) = thl0(k + 1);
                    qtu_emf(k) = qt0(k + 1);
                    uu_emf(k) = u0(k + 1);
                    vu_emf(k) = v0(k + 1);
                    trl_emf(k) = trl(k + 1);
                    tri_emf(k) = tri(k + 1);
                }
            }
        }

        // lines 3308-3343: fluxes at and below the PBL top
        xsrc = qtsrc;
        xmean = qt0(kinv);
        xtop = qt0(kinv + 1) + ssqt0(kinv + 1) * (ps0(kinv) - p0(kinv + 1));
        xbot = qt0(kinv - 1) + ssqt0(kinv - 1) * (ps0(kinv - 1) - p0(kinv - 1));
        uwsc_fluxbelowinv(c, cbmf, ps0, mkx, kinv, dt, xsrc, xmean, xtop, xbot, xflx);
        for (k = 0; k <= kinv - 1; ++k) qtflx(k) = xflx(k);
        xsrc = thlsrc;
        xmean = thl0(kinv);
        xtop = thl0(kinv + 1) + ssthl0(kinv + 1) * (ps0(kinv) - p0(kinv + 1));
        xbot = thl0(kinv - 1) + ssthl0(kinv - 1) * (ps0(kinv - 1) - p0(kinv - 1));
        uwsc_fluxbelowinv(c, cbmf, ps0, mkx, kinv, dt, xsrc, xmean, xtop, xbot, xflx);
        for (k = 0; k <= kinv - 1; ++k) slflx(k) = cp * exns0(k) * xflx(k);
        xsrc = usrc;
        xmean = u0(kinv);
        xtop = u0(kinv + 1) + ssu0(kinv + 1) * (ps0(kinv) - p0(kinv + 1));
        xbot = u0(kinv - 1) + ssu0(kinv - 1) * (ps0(kinv - 1) - p0(kinv - 1));
        uwsc_fluxbelowinv(c, cbmf, ps0, mkx, kinv, dt, xsrc, xmean, xtop, xbot, xflx);
        for (k = 0; k <= kinv - 1; ++k) uflx(k) = xflx(k);
        xsrc = vsrc;
        xmean = v0(kinv);
        xtop = v0(kinv + 1) + ssv0(kinv + 1) * (ps0(kinv) - p0(kinv + 1));
        xbot = v0(kinv - 1) + ssv0(kinv - 1) * (ps0(kinv - 1) - p0(kinv - 1));
        uwsc_fluxbelowinv(c, cbmf, ps0, mkx, kinv, dt, xsrc, xmean, xtop, xbot, xflx);
        for (k = 0; k <= kinv - 1; ++k) vflx(k) = xflx(k);

        // lines 3353-3366: PBL top to the release level
        uplus = R8(0.0);
        vplus = R8(0.0);
        for (k = kinv; k <= krel - 1; ++k) {
            kp1 = k + 1;
            qtflx(k) = cbmf * (qtsrc - (qt0(kp1) + ssqt0(kp1) * (ps0(k) - p0(kp1))));
            slflx(k) = cbmf * (thlsrc - (thl0(kp1) + ssthl0(kp1) * (ps0(k) - p0(kp1)))) * cp * exns0(k);
            uplus = uplus + PGFc * ssu0(k) * (ps0(k) - ps0(k - 1));
            vplus = vplus + PGFc * ssv0(k) * (ps0(k) - ps0(k - 1));
            uflx(k) = cbmf * (usrc + uplus - (u0(kp1) + ssu0(kp1) * (ps0(k) - p0(kp1))));
            vflx(k) = cbmf * (vsrc + vplus - (v0(kp1) + ssv0(kp1) * (ps0(k) - p0(kp1))));
        }
        // lines 3375-3384: release level to kbup-1
        for (k = krel; k <= kbup - 1; ++k) {
            kp1 = k + 1;
            slflx(k) = cp * exns0(k) * umf(k) * (thlu(k) - (thl0(kp1) + ssthl0(kp1) * (ps0(k) - p0(kp1))));
            qtflx(k) = umf(k) * (qtu(k) - (qt0(kp1) + ssqt0(kp1) * (ps0(k) - p0(kp1))));
            uflx(k) = umf(k) * (uu(k) - (u0(kp1) + ssu0(kp1) * (ps0(k) - p0(kp1))));
            vflx(k) = umf(k) * (vu(k) - (v0(kp1) + ssv0(kp1) * (ps0(k) - p0(kp1))));
        }
        // lines 3445-3454: penetrative entrainment fluxes
        for (k = kbup; k <= kpen - 1; ++k) {
            slflx(k) = cp * exns0(k) * emf(k) * (thlu_emf(k) - (thl0(k) + ssthl0(k) * (ps0(k) - p0(k))));
            qtflx(k) = emf(k) * (qtu_emf(k) - (qt0(k) + ssqt0(k) * (ps0(k) - p0(k))));
            uflx(k) = emf(k) * (uu_emf(k) - (u0(k) + ssu0(k) * (ps0(k) - p0(k))));
            vflx(k) = emf(k) * (vu_emf(k) - (v0(k) + ssv0(k) * (ps0(k) - p0(k))));
        }

        // lines 3469-3480: compensating subsidence mass flux
        for (k = 0; k <= mkx; ++k) uemf(k) = R8(0.0);
        for (k = 0; k <= kinv - 2; ++k) {
            uemf(k) = cbmf * (ps0(0) - ps0(k)) / (ps0(0) - ps0(kinv - 1));
        }
        for (k = kinv - 1; k <= krel - 1; ++k) uemf(k) = cbmf;
        for (k = krel; k <= kbup - 1; ++k) uemf(k) = umf(k);
        for (k = kbup; k <= kpen - 1; ++k) uemf(k) = emf(k);
        for (k = 1; k <= mkx; ++k) comsub(k) = R8(0.0);
        for (k = 1; k <= kpen; ++k) comsub(k) = R8(0.5) * (uemf(k) + uemf(k - 1));

        // lines 3482-3529: condensate sinks by compensating subsidence
        for (k = 1; k <= kpen; ++k) {
            if (comsub(k) >= R8(0.0)) {
                if (k == mkx) {
                    thlten_sub = R8(0.0); qtten_sub = R8(0.0); qlten_sub = R8(0.0);
                    qiten_sub = R8(0.0); nlten_sub = R8(0.0); niten_sub = R8(0.0);
                } else {
                    thlten_sub = g * comsub(k) * (thl0(k + 1) - thl0(k)) / (p0(k) - p0(k + 1));
                    qtten_sub = g * comsub(k) * (qt0(k + 1) - qt0(k)) / (p0(k) - p0(k + 1));
                    qlten_sub = g * comsub(k) * (ql0(k + 1) - ql0(k)) / (p0(k) - p0(k + 1));
                    qiten_sub = g * comsub(k) * (qi0(k + 1) - qi0(k)) / (p0(k) - p0(k + 1));
                    nlten_sub = g * comsub(k) * (trl(k + 1) - trl(k)) / (p0(k) - p0(k + 1));
                    niten_sub = g * comsub(k) * (tri(k + 1) - tri(k)) / (p0(k) - p0(k + 1));
                }
            } else {
                if (k == 1) {
                    thlten_sub = R8(0.0); qtten_sub = R8(0.0); qlten_sub = R8(0.0);
                    qiten_sub = R8(0.0); nlten_sub = R8(0.0); niten_sub = R8(0.0);
                } else {
                    thlten_sub = g * comsub(k) * (thl0(k) - thl0(k - 1)) / (p0(k - 1) - p0(k));
                    qtten_sub = g * comsub(k) * (qt0(k) - qt0(k - 1)) / (p0(k - 1) - p0(k));
                    qlten_sub = g * comsub(k) * (ql0(k) - ql0(k - 1)) / (p0(k - 1) - p0(k));
                    qiten_sub = g * comsub(k) * (qi0(k) - qi0(k - 1)) / (p0(k - 1) - p0(k));
                    nlten_sub = g * comsub(k) * (trl(k) - trl(k - 1)) / (p0(k - 1) - p0(k));
                    niten_sub = g * comsub(k) * (tri(k) - tri(k - 1)) / (p0(k - 1) - p0(k));
                }
            }
            thl_prog = thl0(k) + thlten_sub * dt;
            qt_prog = uw_max(qt0(k) + qtten_sub * dt, R8(1.e-12));
            uwsc_conden_exn(c, p0(k), exn0(k), thl_prog, qt_prog, thj, qvj, qlj, qij, qse, id_check);
            if (id_check == 1) { id_exit = true; goto L333; }
            qlten_sink(k) = uw_max(qlten_sub, -(ql0(k) / dt));
            qiten_sink(k) = uw_max(qiten_sub, -(qi0(k) / dt));
            nlten_sink(k) = uw_max(nlten_sub, -(trl(k) / dt));
            niten_sink(k) = uw_max(niten_sub, -(tri(k) / dt));
        }

        // lines 3541-3554: momentum
        for (k = 1; k <= kpen; ++k) {
            km1 = k - 1;
            uten(k) = (uflx(km1) - uflx(k)) * g / dp0(k);
            vten(k) = (vflx(km1) - vflx(k)) * g / dp0(k);
            uf(k) = u0(k) + uten(k) * dt;
            vf(k) = v0(k) + vten(k) * dt;
        }

        // lines 3564-3847: thermodynamic tendencies and detrainment
        rliq = R8(0.0);
        rainflx = R8(0.0);
        snowflx = R8(0.0);
        for (k = 1; k <= kpen; ++k) {
            km1 = k - 1;
            dwten(k) = dwten(k) * R8(0.5) * (umf(k - 1) + umf(k)) * g / dp0(k);
            diten(k) = diten(k) * R8(0.5) * (umf(k - 1) + umf(k)) * g / dp0(k);
            qrten(k) = frc_rasn * dwten(k);
            qsten(k) = frc_rasn * diten(k);
            rainflx = rainflx + qrten(k) * dp0(k) / g;
            snowflx = snowflx + qsten(k) * dp0(k) / g;
            slten(k) = (slflx(km1) - slflx(k)) * g / dp0(k);
            if (k == 1) {
                slten(k) = slten(k) - g / R8(4.0) / dp0(k) * (
                           uflx(k) * (uf(k + 1) - uf(k) + u0(k + 1) - u0(k)) +
                           vflx(k) * (vf(k + 1) - vf(k) + v0(k + 1) - v0(k)));
            } else if (k >= 2 && k <= kpen - 1) {
                slten(k) = slten(k) - g / R8(4.0) / dp0(k) * (
                           uflx(k) * (uf(k + 1) - uf(k) + u0(k + 1) - u0(k)) +
                           uflx(k - 1) * (uf(k) - uf(k - 1) + u0(k) - u0(k - 1)) +
                           vflx(k) * (vf(k + 1) - vf(k) + v0(k + 1) - v0(k)) +
                           vflx(k - 1) * (vf(k) - vf(k - 1) + v0(k) - v0(k - 1)));
            } else if (k == kpen) {
                slten(k) = slten(k) - g / R8(4.0) / dp0(k) * (
                           uflx(k - 1) * (uf(k) - uf(k - 1) + u0(k) - u0(k - 1)) +
                           vflx(k - 1) * (vf(k) - vf(k - 1) + v0(k) - v0(k - 1)));
            }
            qtten(k) = (qtflx(km1) - qtflx(k)) * g / dp0(k);

            if (k < krel || k > kpen) {
                qlu_mid = R8(0.0);
                qiu_mid = R8(0.0);
                qlj = R8(0.0);
                qij = R8(0.0);
            } else if (k == krel) {
                uwsc_conden(c, prel, thlu(krel - 1), qtu(krel - 1), thj, qvj, qlj, qij, qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                qlubelow = qlj;
                qiubelow = qij;
                uwsc_conden_exn(c, ps0(k), exns0(k), thlu(k), qtu(k), thj, qvj, qlj, qij, qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                qlu_mid = R8(0.5) * (qlubelow + qlj) * (prel - ps0(k)) / (ps0(k - 1) - ps0(k));
                qiu_mid = R8(0.5) * (qiubelow + qij) * (prel - ps0(k)) / (ps0(k - 1) - ps0(k));
            } else if (k == kpen) {
                uwsc_conden(c, ps0(k - 1) + ppen, thlu_top, qtu_top, thj, qvj, qlj, qij, qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                qlu_mid = R8(0.5) * (qlubelow + qlj) * (-ppen) / (ps0(k - 1) - ps0(k));
                qiu_mid = R8(0.5) * (qiubelow + qij) * (-ppen) / (ps0(k - 1) - ps0(k));
                qlu_top = qlj;
                qiu_top = qij;
            } else {
                uwsc_conden_exn(c, ps0(k), exns0(k), thlu(k), qtu(k), thj, qvj, qlj, qij, qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                qlu_mid = R8(0.5) * (qlubelow + qlj);
                qiu_mid = R8(0.5) * (qiubelow + qij);
            }
            qlubelow = qlj;
            qiubelow = qij;

            qc_l(k) = (R8(1.0) - frc_rasn) * dwten(k);
            qc_i(k) = (R8(1.0) - frc_rasn) * diten(k);
            if (k <= kbup) {
                qc_l(k) = qc_l(k) + g * R8(0.5) * (umf(k - 1) + umf(k)) * fdr(k) * qlu_mid;
                qc_i(k) = qc_i(k) + g * R8(0.5) * (umf(k - 1) + umf(k)) * fdr(k) * qiu_mid;
                qc_lm = -(g * R8(0.5) * (umf(k - 1) + umf(k)) * fdr(k) * ql0(k));
                qc_im = -(g * R8(0.5) * (umf(k - 1) + umf(k)) * fdr(k) * qi0(k));
                nc_lm = -(g * R8(0.5) * (umf(k - 1) + umf(k)) * fdr(k) * trl(k));
                nc_im = -(g * R8(0.5) * (umf(k - 1) + umf(k)) * fdr(k) * tri(k));
            } else {
                qc_lm = R8(0.0);
                qc_im = R8(0.0);
                nc_lm = R8(0.0);
                nc_im = R8(0.0);
            }
            if (k == kbup) {
                qc_l(k) = qc_l(k) + g * umf(k) * qlj / (ps0(k - 1) - ps0(k));
                qc_i(k) = qc_i(k) + g * umf(k) * qij / (ps0(k - 1) - ps0(k));
                qc_lm = qc_lm - g * umf(k) * ql0(k) / (ps0(k - 1) - ps0(k));
                qc_im = qc_im - g * umf(k) * qi0(k) / (ps0(k - 1) - ps0(k));
                nc_lm = nc_lm - g * umf(k) * trl(k) / (ps0(k - 1) - ps0(k));
                nc_im = nc_im - g * umf(k) * tri(k) / (ps0(k - 1) - ps0(k));
            }
            if (k == kbup) {
                uwsc_conden_exn(c, p0(k), exn0(k), thlu_emf(k), qtu_emf(k), thj, qvj, ql_emf_kbup, qi_emf_kbup,
                            qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                if (ql_emf_kbup > R8(0.0)) nl_emf_kbup = trl_emf(k);
                else nl_emf_kbup = R8(0.0);
                if (qi_emf_kbup > R8(0.0)) ni_emf_kbup = tri_emf(k);
                else ni_emf_kbup = R8(0.0);
                qc_lm = qc_lm - g * emf(k) * (ql_emf_kbup - ql0(k)) / (ps0(k - 1) - ps0(k));
                qc_im = qc_im - g * emf(k) * (qi_emf_kbup - qi0(k)) / (ps0(k - 1) - ps0(k));
                nc_lm = nc_lm - g * emf(k) * (nl_emf_kbup - trl(k)) / (ps0(k - 1) - ps0(k));
                nc_im = nc_im - g * emf(k) * (ni_emf_kbup - tri(k)) / (ps0(k - 1) - ps0(k));
            }
            // use_expconten = .true., use_unicondet = .false.
            qlten(k) = qc_l(k) + frc_rasn * dwten(k) +
                       (uw_max(R8(0.0), ql0(k) + (qc_lm + qlten_sink(k)) * dt) - ql0(k)) / dt;
            qiten(k) = qc_i(k) + frc_rasn * diten(k) +
                       (uw_max(R8(0.0), qi0(k) + (qc_im + qiten_sink(k)) * dt) - qi0(k)) / dt;
            trtl(k) = uw_max(nc_lm + nlten_sink(k), -(trl(k) / dt));
            trti(k) = uw_max(nc_im + niten_sink(k), -(tri(k) / dt));
            qvten(k) = qtten(k) - qlten(k) - qiten(k);
            sten(k) = slten(k) + xlv * qlten(k) + xls * qiten(k);
            qc(k) = qc_l(k) + qc_i(k);
            rliq = rliq + qc(k) * dp0(k) / g / R8(1000.0);
        }
        precip = rainflx + snowflx;
        snow = snowflx;

        // lines 3861-3987: evaporation of precipitation
        evpint_rain = R8(0.0);
        evpint_snow = R8(0.0);
        for (k = 0; k <= mkx; ++k) { flxrain(k) = R8(0.0); flxsnow(k) = R8(0.0); }
        for (k = 1; k <= mkx; ++k) { ntraprd(k) = R8(0.0); ntsnprd(k) = R8(0.0); }
        for (k = mkx; k >= 1; --k) {
            if (t0(k) > R8(273.16)) snowmlt = uw_max(R8(0.0), flxsnow(k) * g / dp0(k));
            else snowmlt = R8(0.0);
            uwsc_qsat_only(t0(k), p0(k), es, qs);
            subsat = uw_max((R8(1.0) - qv0(k) / qs), R8(0.0));
            evprain = kevp * subsat * uw_sqrt(flxrain(k) + snowmlt * dp0(k) / g);
            evpsnow = kevp * subsat * uw_sqrt(uw_max(flxsnow(k) - snowmlt * dp0(k) / g, R8(0.0)));
            evplimit = uw_max(R8(0.0), (io.qw0(k) - qv0(k)) / dt);
            evplimit_rain = uw_min(evplimit, (flxrain(k) + snowmlt * dp0(k) / g) * g / dp0(k));
            evplimit_rain = uw_min(evplimit_rain, (rainflx - evpint_rain) * g / dp0(k));
            evprain = uw_max(R8(0.0), uw_min(evplimit_rain, evprain));
            evplimit_snow = uw_min(evplimit, uw_max(flxsnow(k) - snowmlt * dp0(k) / g, R8(0.0)) * g / dp0(k));
            evplimit_snow = uw_min(evplimit_snow, (snowflx - evpint_snow) * g / dp0(k));
            evpsnow = uw_max(R8(0.0), uw_min(evplimit_snow, evpsnow));
            if ((evprain + evpsnow) > evplimit) {
                tmp1 = evprain * evplimit / (evprain + evpsnow);
                tmp2 = evpsnow * evplimit / (evprain + evpsnow);
                evprain = tmp1;
                evpsnow = tmp2;
            }
            evapc(k) = evprain + evpsnow;
            evpint_rain = evpint_rain + evprain * dp0(k) / g;
            evpint_snow = evpint_snow + evpsnow * dp0(k) / g;
            ntraprd(k) = qrten(k) - evprain + snowmlt;
            ntsnprd(k) = qsten(k) - evpsnow - snowmlt;
            flxrain(k - 1) = flxrain(k) + ntraprd(k) * dp0(k) / g;
            flxsnow(k - 1) = flxsnow(k) + ntsnprd(k) * dp0(k) / g;
            flxrain(k - 1) = uw_max(flxrain(k - 1), R8(0.0));
            if (flxrain(k - 1) == R8(0.0)) ntraprd(k) = -(flxrain(k) * g / dp0(k));
            flxsnow(k - 1) = uw_max(flxsnow(k - 1), R8(0.0));
            if (flxsnow(k - 1) == R8(0.0)) ntsnprd(k) = -(flxsnow(k) * g / dp0(k));
            qlten(k) = qlten(k) - qrten(k);
            qiten(k) = qiten(k) - qsten(k);
            qvten(k) = qvten(k) + evprain + evpsnow;
            qtten(k) = qlten(k) + qiten(k) + qvten(k);
            sten(k) = sten(k) - xlv * evprain - xls * evpsnow - (xls - xlv) * snowmlt;
            slten(k) = sten(k) - xlv * qlten(k) - xls * qiten(k);
        }
        precip = (flxrain(0) + flxsnow(0)) / R8(1000.0);
        snow = flxsnow(0) / R8(1000.0);

        // lines 4019-4034: reserved condensate out of the layer tendencies
        for (k = 1; k <= kpen; ++k) {
            qtten(k) = qtten(k) - qc(k);
            qlten(k) = qlten(k) - qc_l(k);
            qiten(k) = qiten(k) - qc_i(k);
            slten(k) = slten(k) + (xlv * qc_l(k) + xls * qc_i(k));
            sten(k) = sten(k) - (xls - xlv) * qc_i(k);
        }

        // lines 4048-4055: positive moisture
        for (k = 1; k <= mkx; ++k) {
            qv0_star(k) = qv0(k) + qvten(k) * dt;
            ql0_star(k) = ql0(k) + qlten(k) * dt;
            qi0_star(k) = qi0(k) + qiten(k) * dt;
        }
        uwsc_positive_moisture(c, mkx, dt, qmin1, qmin2, qmin3, dp0, qv0_star, ql0_star,
                               qi0_star, qvten, qlten, qiten, sten);
        for (k = 1; k <= mkx; ++k) {
            qtten(k) = qvten(k) + qlten(k) + qiten(k);
            slten(k) = sten(k) - xlv * qlten(k) - xls * qiten(k);
        }

        // lines 4124-4190: in-cumulus condensate and cloud fraction
        uwsc_conden(c, prel, thlu(krel - 1), qtu(krel - 1), thj, qvj, qlj, qij, qse, id_check);
        if (id_check == 1) { id_exit = true; goto L333; }
        qcubelow = qlj + qij;
        qlubelow = qlj;
        qiubelow = qij;
        for (k = krel; k <= kpen; ++k) {
            if (k == kpen) {
                uwsc_conden(c, ps0(k - 1) + ppen, thlu_top, qtu_top, thj, qvj, qlj, qij, qse, id_check);
            } else {
                uwsc_conden_exn(c, ps0(k), exns0(k), thlu(k), qtu(k), thj, qvj, qlj, qij, qse, id_check);
            }
            if (id_check == 1) { id_exit = true; goto L333; }
            qcu(k) = R8(0.5) * (qcubelow + qlj + qij);
            qlu(k) = R8(0.5) * (qlubelow + qlj);
            qiu(k) = R8(0.5) * (qiubelow + qij);
            cufrc(k) = (ufrc(k - 1) + ufrc(k));
            if (k == krel) {
                cufrc(k) = (ufrclcl + ufrc(k)) * (prel - ps0(k)) / (ps0(k - 1) - ps0(k));
            } else if (k == kpen) {
                cufrc(k) = (ufrc(k - 1) + R8(0.0)) * (-ppen) / (ps0(k - 1) - ps0(k));
                if ((qlj + qij) > criqc) {
                    qcu(k) = R8(0.5) * (qcubelow + criqc);
                    qlu(k) = R8(0.5) * (qlubelow + criqc * qlj / (qlj + qij));
                    qiu(k) = R8(0.5) * (qiubelow + criqc * qij / (qlj + qij));
                }
            }
            qcubelow = qlj + qij;
            qlubelow = qlj;
            qiubelow = qij;
        }
        cnt = uw_real(kpen);
        cnb = uw_real(krel - 1);

        // lines 4201-4361: first pass: keep its answer, update the environment
        if (iter != iter_cin) {
            for (k = 0; k <= mkx; ++k) {
                umf_s(k) = umf(k); slflx_s(k) = slflx(k); qtflx_s(k) = qtflx(k);
            }
            for (k = 1; k <= mkx; ++k) {
                R8 qv0_s = qv0(k) + qvten(k) * dt;
                R8 ql0_s = ql0(k) + qlten(k) * dt;
                R8 qi0_s = qi0(k) + qiten(k) * dt;
                R8 t0_s = t0(k) + sten(k) * dt / cp;
                qvten_s(k) = qvten(k); qlten_s(k) = qlten(k); qiten_s(k) = qiten(k);
                sten_s(k) = sten(k); uten_s(k) = uten(k); vten_s(k) = vten(k);
                qrten_s(k) = qrten(k); qsten_s(k) = qsten(k); evapc_s(k) = evapc(k);
                cufrc_s(k) = cufrc(k); qcu_s(k) = qcu(k); qlu_s(k) = qlu(k);
                qiu_s(k) = qiu(k); qc_s(k) = qc(k); trtl_s(k) = trtl(k); trti_s(k) = trti(k);
                qv0(k) = qv0_s; ql0(k) = ql0_s; qi0(k) = qi0_s; t0(k) = t0_s;
            }
            precip_s = precip; snow_s = snow; cush_s = cush; cbmf_s = cbmf;
            rliq_s = rliq; cnt_s = cnt; cnb_s = cnb;
            for (k = 1; k <= mkx; ++k) {
                qt0(k) = (qv0(k) + ql0(k) + qi0(k));
                thl0(k) = (t0(k) - xlv * ql0(k) / cp - xls * qi0(k) / cp) / exn0(k);
                thvl0(k) = (R8(1.0) + zvir * qt0(k)) * thl0(k);
            }
            uwsc_slope(mkx, thl0, p0, ssthl0);
            uwsc_slope(mkx, qt0, p0, ssqt0);
            // ssu0/ssv0/sstr0 recomputed from unchanged u0/v0/tr0: identical
            for (k = 1; k <= mkx; ++k) {
                thl0bot = thl0(k) + ssthl0(k) * (ps0(k - 1) - p0(k));
                qt0bot = qt0(k) + ssqt0(k) * (ps0(k - 1) - p0(k));
                uwsc_conden_exn(c, ps0(k - 1), exns0(k - 1), thl0bot, qt0bot, thj, qvj, qlj, qij, qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                thv0bot(k) = uwsc_thv(c, thj, qvj, qlj, qij);
                thvl0bot(k) = thl0bot * (R8(1.0) + zvir * qt0bot);
                thl0top = thl0(k) + ssthl0(k) * (ps0(k) - p0(k));
                qt0top = qt0(k) + ssqt0(k) * (ps0(k) - p0(k));
                uwsc_conden_exn(c, ps0(k), exns0(k), thl0top, qt0top, thj, qvj, qlj, qij, qse, id_check);
                if (id_check == 1) { id_exit = true; goto L333; }
                thv0top(k) = uwsc_thv(c, thj, qvj, qlj, qij);
                thvl0top(k) = thl0top * (R8(1.0) + zvir * qt0top);
            }
        }
    }

    // lines 4369-4399: the column's answer
    for (k = 0; k <= mkx; ++k) {
        io.umf_out(k) = umf(k); io.slflx_out(k) = slflx(k); io.qtflx_out(k) = qtflx(k);
        io.flxprc1_out(k) = flxrain(k) + flxsnow(k);
        io.flxsnow1_out(k) = flxsnow(k);
    }
    for (k = 1; k <= mkx; ++k) {
        io.qvten_out(k) = qvten(k); io.qlten_out(k) = qlten(k); io.qiten_out(k) = qiten(k);
        io.sten_out(k) = sten(k); io.uten_out(k) = uten(k); io.vten_out(k) = vten(k);
        io.qrten_out(k) = qrten(k); io.qsten_out(k) = qsten(k); io.evapc_out(k) = evapc(k);
        io.cufrc_out(k) = cufrc(k); io.qcu_out(k) = qcu(k); io.qlu_out(k) = qlu(k);
        io.qiu_out(k) = qiu(k); io.qc_out(k) = qc(k);
        io.trtl_out(k) = trtl(k); io.trti_out(k) = trti(k);
    }
    io.precip_out = precip; io.snow_out = snow; io.cbmf_out = cbmf;
    io.rliq_out = rliq; io.cnt_out = cnt; io.cnb_out = cnb;

L333:
    if (id_exit) {
        // lines 4470-4501: exit without cumulus convection
        for (k = 0; k <= mkx; ++k) {
            io.umf_out(k) = R8(0.0); io.slflx_out(k) = R8(0.0); io.qtflx_out(k) = R8(0.0);
        }
        for (k = 1; k <= mkx; ++k) {
            io.qvten_out(k) = R8(0.0); io.qlten_out(k) = R8(0.0); io.qiten_out(k) = R8(0.0);
            io.sten_out(k) = R8(0.0); io.uten_out(k) = R8(0.0); io.vten_out(k) = R8(0.0);
            io.qrten_out(k) = R8(0.0); io.qsten_out(k) = R8(0.0); io.evapc_out(k) = R8(0.0);
            io.cufrc_out(k) = R8(0.0); io.qcu_out(k) = R8(0.0); io.qlu_out(k) = R8(0.0);
            io.qiu_out(k) = R8(0.0); io.qc_out(k) = R8(0.0);
            io.trtl_out(k) = R8(0.0); io.trti_out(k) = R8(0.0);
        }
        io.precip_out = R8(0.0); io.snow_out = R8(0.0);
        cush = R8(-1.0);
        io.cbmf_out = R8(0.0); io.rliq_out = R8(0.0);
        io.cnt_out = R8(1.0); io.cnb_out = uw_real(mkx);
    }
    io.cush_out = cush;
}

#endif  // UWSHCU_COLUMN_CUH
