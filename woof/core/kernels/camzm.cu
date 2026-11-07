// camzm.cu -- WRF v4.6.1 cu_physics = 7 (CAMZMSCHEME): the CAM5
// Zhang-McFarlane deep convection, transcribed for the GPU.  WOOF's own
// work, Apache-2.0.  Source of record: WRF v4.6.1 (public domain)
// phys/module_cu_camzm_driver.F (camzm_driver, get_tpert, zm_conv_init) and
// phys/module_cu_camzm.F (zm_convi, zm_convr, buoyan_dilute, parcel_dilute,
// entropy, ientropy, cldprp, closure, q1q2_pjr, zm_conv_evap, convtran,
// momtran); line numbers below are those files'.
//
// The loader prepends glibc_flt32.cuh, glibc_flt64.cuh, uwpbl_common.cuh,
// uwpbl_wvsat.cuh and cam5_support.cuh (gpuwm/core/kernels/__init__.py
// _EXTRA_HEADERS["camzm"]).  From them: R8, the binary64 type whose every
// operation is a rounding-pinned intrinsic (so neither --fmad nor the
// strict arm can change a word); uw_max/uw_min, gfortran's MAX/MIN; uw_exp,
// uw_log, uw_pow, WOOF's own binary64 libm words; gfk_exp/gfk_pow, WOOF's
// own float32 libm words; uw_aqsat, wv_saturation's table lookup; and the
// shared CAM5 layer (cam_cldwat_fice, the float32 vocabulary).
//
// ONE THREAD, ONE WRF COLUMN, AS IN WRF.  camzm_driver hands zm_convr a
// one-column chunk (pcols = 1, ncol = 1), so CAM's gathered arrays are the
// column itself (lengath is 0 or 1, ideep(1) = 1) and every per-column
// integer array is a scalar here.  CAM's automatic arrays live in the
// per-column global pool of uwpbl_common.cuh (Ws); views are 1-based and
// in CAM's vertical order (k = 1 at the model top) exactly like the
// Fortran, and the driver's flips (kflip = kte-k+1) are written as such.
//
// Preserve the stock WRF branches, including the XLAND test at driver
// :459-463 and the expression grouping in momtran :1633. A corrected
// referee is a diagnostic and cannot certify this WRF selector.

#define CZ_LIMCNV 2            // zm_conv_init :839
#define CZ_MSG (CZ_LIMCNV - 1) // zm_convr :547
#define CZ_CAPELMT 70.0        // :59
#define CZ_C0_LND 0.0059       // zmconv_readnl :136 (0.0059D0)
#define CZ_C0_OCN 0.0450       // :137 (0.0450D0)
#define CZ_KE 0x1.0c6f7ap-20   // :138, ke = 1.0E-6: a default-REAL literal
#define CZ_C1 6.112            // :66
#define CZ_C2 17.67            // :67
#define CZ_C3 243.5            // :68
#define CZ_TIEDKE_ADD 0.5      // :85
// camzm_driver :713-714 denominators as the compiler folded them
// (8.e-6_r8**3 and 25.e-6_r8**3 are folded with one rounding, which is not
// the two-multiply chain for 25e-6); recorded by the oracle (const/qcn_den,
// const/qin_den) and checked against it by the parity test.
#define CZ_QCN_DEN 0x1.c3648033b1403p-38
#define CZ_QIN_DEN 0x1.afc6df1b50ce7p-34
// module_model_constants words get_tpert reads (default REAL, oracle const/mc_*)
#define CZ_MC_CP 0x1.f64p+9f
#define CZ_MC_EP_1 0x1.377b48p-1f
#define CZ_MC_EP_2 0x1.3e5612p-1f
#define CZ_MC_G 0x1.39eb86p+3f
#define CZ_MC_R_D 0x1.1fp+8f
#define CZ_MC_RCP 0x1.24924ap-2f
#define CZ_MC_SVP1 0x1.38ef34p-1f
#define CZ_MC_SVP2 0x1.1ab852p+4f
#define CZ_MC_SVP3 0x1.da6666p+4f
#define CZ_MC_SVPT0 0x1.112666p+8f
// WRF Registry scheme IDs get_tpert compares against (stub_wrf.F90 cites them)
#define CZ_MYJSFCSCHEME 2
#define CZ_MYJPBLSCHEME 2
#define CZ_CAMUWPBLSCHEME 9
// status words (the launcher names each breakage)
#define CZ_ERR_POOL 1
#define CZ_ERR_IENTROPY 3
#define CZ_ERR_PBLH 4
#define CZ_ERR_ARM 5

// Module constants zm_convi sets from physconst (:171-185).
struct CzK {
    R8 tfreez, eps1, rl, cpres, rgrav, rgas, grav, cp, tau;
};

__device__ __forceinline__ void cz_flag(int* err, int code) {
    atomicCAS(err, 0, code);
}

// ---------------------------------------------------------------------------
// log(1) is positive zero in both native and CUDA math. Keep the multiply
// at the call site, including its IEEE treatment of nonfinite operands.
__device__ __forceinline__ R8 cz_water_log(R8 qv, R8 qsat) {
    return qv == qsat && qsat > R8(0.0) && isfinite(qsat.v)
        ? R8(0.0) : uw_log(qv / qsat);
}
// qv == qt has identical pressure-log inputs across the Newton endpoints
// and subsequent iterations. Cache its first evaluation without factoring
// or changing any arithmetic operation. Other qv values evaluate normally.
__device__ __forceinline__ R8 cz_pressure_log(const CzK& K, R8 p, R8 e,
        R8 qv, R8 qt, R8& cached, bool& valid) {
    bool same = __double_as_longlong(qv.v) == __double_as_longlong(qt.v);
    if (same && valid) return cached;
    R8 value = uw_log((p - e) / R8(1000.0));
    if (same) { cached = value; valid = true; }
    return value;
}

// entropy (:3692-3718), Raymond and Blyth 1992
__device__ R8 cz_entropy(const CzK& K, R8 tk, R8 p, R8 qtot) {
    const R8 pref = R8(1000.0);
    R8 L = K.rl - (R8(CAM_CPLIQ) - R8(CAM_CPWV)) * (tk - K.tfreez);
    R8 esat = R8(CZ_C1) * uw_exp(R8(CZ_C2) * (tk - K.tfreez) / ((R8(CZ_C3) + tk) - K.tfreez));
    R8 qsat = K.eps1 * esat / (p - esat);
    R8 qv = uw_min(qtot, qsat);
    R8 e = qv * p / (K.eps1 + qv);
    return (((K.cpres + qtot * R8(CAM_CPLIQ)) * uw_log(tk / K.tfreez)
             - K.rgas * uw_log((p - e) / pref)) + L * qv / tk)
           - qv * R8(UW_RH2O) * cz_water_log(qv, qsat);
}

// ientropy (:3722-3813): Newton inversion of entropy for T and qsat.
// Returns 0, or CZ_ERR_IENTROPY where WRF calls endrun (:3798).
__device__ int cz_ientropy(const CzK& K, R8 s, R8 p, R8 qt, R8& T, R8& qsat, R8 tfg) {
    const int LOOPMAX = 100;
    const R8 pref = R8(1000.0);
    R8 Ts = tfg; R8 pressure_cached; bool pressure_valid = false;
    for (int i = 0; i <= LOOPMAX; ++i) {
        R8 L = K.rl - (R8(CAM_CPLIQ) - R8(CAM_CPWV)) * (Ts - K.tfreez);
        R8 esat = R8(CZ_C1) * uw_exp(R8(CZ_C2) * (Ts - K.tfreez) / ((R8(CZ_C3) + Ts) - K.tfreez));
        qsat = K.eps1 * esat / (p - esat);
        R8 qv = uw_min(qt, qsat);
        R8 e = qv * p / (K.eps1 + qv);
        R8 fs1 = ((((K.cpres + qt * R8(CAM_CPLIQ)) * uw_log(Ts / K.tfreez)
                    - K.rgas * cz_pressure_log(K, p, e, qv, qt, pressure_cached, pressure_valid)) + L * qv / Ts)
                  - qv * R8(UW_RH2O) * cz_water_log(qv, qsat)) - s;
        L = K.rl - (R8(CAM_CPLIQ) - R8(CAM_CPWV)) * ((Ts - R8(1.0)) - K.tfreez);
        esat = R8(CZ_C1) * uw_exp(R8(CZ_C2) * ((Ts - R8(1.0)) - K.tfreez)
                                  / (((R8(CZ_C3) + Ts) - R8(1.0)) - K.tfreez));
        qsat = K.eps1 * esat / (p - esat);
        qv = uw_min(qt, qsat);
        e = qv * p / (K.eps1 + qv);
        R8 fs2 = ((((K.cpres + qt * R8(CAM_CPLIQ)) * uw_log((Ts - R8(1.0)) / K.tfreez)
                    - K.rgas * cz_pressure_log(K, p, e, qv, qt, pressure_cached, pressure_valid)) + L * qv / (Ts - R8(1.0)))
                  - qv * R8(UW_RH2O) * cz_water_log(qv, qsat)) - s;
        R8 dTs = fs1 / (fs2 - fs1);
        Ts = Ts + dTs;
        if (uw_abs(dTs) < R8(0.001)) break;
        if (i == LOOPMAX - 1) return CZ_ERR_IENTROPY;
    }
    R8 esat = R8(CZ_C1) * uw_exp(R8(CZ_C2) * (Ts - K.tfreez) / ((R8(CZ_C3) + Ts) - K.tfreez));
    qsat = K.eps1 * esat / (p - esat);
    T = Ts;
    return 0;
}

// ---------------------------------------------------------------------------
// parcel_dilute (:3376-3689), one column; klaunch = mx.
__device__ int cz_parcel_dilute(const CzK& K, Ws& ws, int pver, int msg, int klaunch,
                                V p, V t, V q, R8 tpert, V tp, V tpv, V qstp,
                                R8& pl, R8& tl, int& lcl) {
    WsMark mark(ws);
    V tmix = ws.r8(pver), qtmix = ws.r8(pver), qsmix = ws.r8(pver), smix = ws.r8(pver);
    V xsh2o = ws.r8(pver), ds_xsh2o = ws.r8(pver), ds_freeze = ws.r8(pver);
    const int nit_lheat = 2;
    const R8 dmpdz = R8(-1.e-3);
    const R8 lwmax = R8(1.e-3);
    const R8 tscool = R8(0.0);
    for (int k = 1; k <= pver; ++k) {
        qtmix(k) = R8(0.0); smix(k) = R8(0.0);
        xsh2o(k) = R8(0.0); ds_xsh2o(k) = R8(0.0); ds_freeze(k) = R8(0.0);
    }
    R8 qtp0 = R8(0.0), sp0 = R8(0.0), mp0 = R8(0.0);
    R8 qtp = R8(0.0), sp = R8(0.0), mp = R8(0.0);
    R8 new_q = R8(0.0), new_s = R8(0.0);
    R8 tfguess;
    int st;
    for (int k = pver; k >= msg + 1; --k) {
        if (k == klaunch) {
            qtp0 = q(k);
            sp0 = cz_entropy(K, t(k), p(k), qtp0);
            mp0 = R8(1.0);
            smix(k) = sp0;
            qtmix(k) = qtp0;
            tfguess = t(k);
            R8 tm, qs;
            st = cz_ientropy(K, smix(k), p(k), qtmix(k), tm, qs, tfguess);
            if (st) return st;
            tmix(k) = tm; qsmix(k) = qs;
        }
        if (k < klaunch) {
            R8 dp = p(k) - p(k + 1);
            R8 qtenv = R8(0.5) * (q(k) + q(k + 1));
            R8 tenv = R8(0.5) * (t(k) + t(k + 1));
            R8 penv = R8(0.5) * (p(k) + p(k + 1));
            R8 senv = cz_entropy(K, tenv, penv, qtenv);
            R8 dpdz = -((penv * K.grav) / (K.rgas * tenv));
            R8 dzdp = R8(1.0) / dpdz;
            R8 dmpdp = dmpdz * dzdp;
            sp = sp - dmpdp * dp * senv;
            qtp = qtp - dmpdp * dp * qtenv;
            mp = mp - dmpdp * dp;
            smix(k) = (sp0 + sp) / (mp0 + mp);
            qtmix(k) = (qtp0 + qtp) / (mp0 + mp);
            tfguess = tmix(k + 1);
            R8 tm, qs;
            st = cz_ientropy(K, smix(k), p(k), qtmix(k), tm, qs, tfguess);
            if (st) return st;
            tmix(k) = tm; qsmix(k) = qs;
            if (qsmix(k) <= qtmix(k) && qsmix(k + 1) > qtmix(k + 1)) {
                lcl = k;
                R8 qxsk = qtmix(k) - qsmix(k);
                R8 qxskp1 = qtmix(k + 1) - qsmix(k + 1);
                R8 dqxsdp = (qxsk - qxskp1) / dp;
                pl = p(k + 1) - qxskp1 / dqxsdp;
                R8 dsdp = (smix(k) - smix(k + 1)) / dp;
                R8 dqtdp = (qtmix(k) - qtmix(k + 1)) / dp;
                R8 slcl = smix(k + 1) + dsdp * (pl - p(k + 1));
                R8 qtlcl = qtmix(k + 1) + dqtdp * (pl - p(k + 1));
                tfguess = tmix(k);
                R8 qslcl;
                st = cz_ientropy(K, slcl, pl, qtlcl, tl, qslcl, tfguess);
                if (st) return st;
            }
        }
    }
    for (int k = pver; k >= msg + 1; --k) {
        if (k == klaunch) {
            tp(k) = tmix(k);
            qstp(k) = q(k);
            tpv(k) = (tp(k) + tpert) * (R8(1.0) + R8(1.608) * qstp(k)) / (R8(1.0) + qstp(k));
        }
        if (k < klaunch) {
            for (int ii = 0; ii <= nit_lheat - 1; ++ii) {
                xsh2o(k) = uw_max(R8(0.0), (qtmix(k) - qsmix(k)) - lwmax);
                ds_xsh2o(k) = ds_xsh2o(k + 1)
                    - R8(CAM_CPLIQ) * uw_log(tmix(k) / K.tfreez) * uw_max(R8(0.0), xsh2o(k) - xsh2o(k + 1));
                if (tmix(k) <= K.tfreez + tscool && ds_freeze(k + 1) == R8(0.0)) {
                    ds_freeze(k) = (R8(UW_LATICE) / tmix(k))
                        * uw_max(R8(0.0), (qtmix(k) - qsmix(k)) - xsh2o(k));
                }
                if (tmix(k) <= K.tfreez + tscool && ds_freeze(k + 1) != R8(0.0)) {
                    ds_freeze(k) = ds_freeze(k + 1)
                        + (R8(UW_LATICE) / tmix(k)) * uw_max(R8(0.0), qsmix(k + 1) - qsmix(k));
                }
                new_s = (smix(k) + ds_xsh2o(k)) + ds_freeze(k);
                new_q = qtmix(k) - xsh2o(k);
                tfguess = tmix(k);
                R8 tm, qs;
                st = cz_ientropy(K, new_s, p(k), new_q, tm, qs, tfguess);
                if (st) return st;
                tmix(k) = tm; qsmix(k) = qs;
            }
            tp(k) = tmix(k);
            if (new_q > qsmix(k)) qstp(k) = qsmix(k);
            else qstp(k) = new_q;
            tpv(k) = (tp(k) + tpert) * (R8(1.0) + R8(1.608) * qstp(k)) / (R8(1.0) + new_q);
        }
    }
    return 0;
}

// buoyan_dilute (:3126-3374), one column.
__device__ int cz_buoyan_dilute(const CzK& K, Ws& ws, int pver, int msg,
                                V q, V t, V p, V z, V pf, V tp, V qstp,
                                R8& tl, R8& cape, R8 pblt, int& lcl, int& lel,
                                int& mx, R8 tpert) {
    WsMark mark(ws);
    V tv = ws.r8(pver), tpv = ws.r8(pver), buoy = ws.r8(pver);
    R8 capeten[5];
    int lelten[5];
    for (int n = 0; n < 5; ++n) { lelten[n] = pver; capeten[n] = R8(0.0); }
    int lon = pver;
    int knt = 0;
    lel = pver;
    mx = lon;
    cape = R8(0.0);
    R8 hmax = R8(0.0);
    for (int k = 1; k <= pver; ++k) {
        tp(k) = t(k);
        qstp(k) = q(k);
        tv(k) = t(k) * (R8(1.0) + R8(1.608) * q(k)) / (R8(1.0) + q(k));
        tpv(k) = tv(k);
        buoy(k) = R8(0.0);
    }
    // nint(pblt): pblt holds an integer value, so the rounding is exact.
    const int npblt = (int)rint(pblt.v);
    for (int k = pver; k >= msg + 1; --k) {
        R8 hmn = (K.cp * t(k) + K.grav * z(k)) + K.rl * q(k);
        if (k >= npblt && k <= lon && hmn > hmax) {
            hmax = hmn;
            mx = k;
        }
    }
    lcl = mx;
    tl = t(mx);
    R8 pl = p(mx);
    int st = cz_parcel_dilute(K, ws, pver, msg, mx, p, t, q, tpert, tp, tpv, qstp, pl, tl, lcl);
    if (st) return st;
    const bool plge600 = pl >= R8(600.0);
    for (int k = pver; k >= msg + 1; --k) {
        if (k <= mx && plge600) {
            tv(k) = t(k) * (R8(1.0) + R8(1.608) * q(k)) / (R8(1.0) + q(k));
            buoy(k) = (tpv(k) - tv(k)) + R8(CZ_TIEDKE_ADD);
        } else {
            qstp(k) = q(k);
            tp(k) = t(k);
            tpv(k) = tv(k);
        }
    }
    for (int k = msg + 2; k <= pver; ++k) {
        if (k < lcl && plge600) {
            if (buoy(k + 1) > R8(0.0) && buoy(k) <= R8(0.0)) {
                knt = min(5, knt + 1);
                lelten[knt - 1] = k;
            }
        }
    }
    for (int n = 0; n < 5; ++n) {
        for (int k = msg + 1; k <= pver; ++k) {
            if (plge600 && k <= mx && k > lelten[n]) {
                capeten[n] = capeten[n] + K.rgas * buoy(k) * uw_log(pf(k + 1) / pf(k));
            }
        }
    }
    for (int n = 0; n < 5; ++n) {
        if (capeten[n] > cape) {
            cape = capeten[n];
            lel = lelten[n];
        }
    }
    cape = uw_max(cape, R8(0.0));
    return 0;
}

// ---------------------------------------------------------------------------
// cldprp (:2114-2779), one gathered column (il2g = 1).  Outputs are views
// the caller owns; jt, jlcl, j0, jd are written through the references.
struct CzCld {
    V mu, eu, du, md, ed, sd, qd, mc, qu, su, qst, hmn, hsat, ql, cmeg;
    V pflx, evp, cu, rprd;   // pflx is pverp long
};

__device__ void cz_cldprp(const CzK& K, Ws& ws, int pver, int msg, int limcnv,
                          V q, V t, V p, V z, V s, V zf, V shat,
                          int jb, int lel, int mx, int& jt, int& jlcl,
                          int& j0, int& jd, R8 landfrac, CzCld& o) {
    WsMark mark(ws);
    V gamma = ws.r8(pver), dz = ws.r8(pver), iprm = ws.r8(pver), hu = ws.r8(pver);
    V hd = ws.r8(pver), eps = ws.r8(pver), f = ws.r8(pver), k1 = ws.r8(pver);
    V i2 = ws.r8(pver), ihat = ws.r8(pver), i3 = ws.r8(pver), idag = ws.r8(pver);
    V i4 = ws.r8(pver), qsthat = ws.r8(pver), hsthat = ws.r8(pver);
    V gamhat = ws.r8(pver), qds = ws.r8(pver);
    const R8 rl = K.rl, rd = K.rgas, cp = K.cp, grav = K.grav;
    R8 ftemp = R8(0.0), expnum = R8(0.0), expdif = R8(0.0);
    const R8 c0mask = R8(CZ_C0_OCN) * (R8(1.0) - landfrac) + R8(CZ_C0_LND) * landfrac;
    for (int k = 1; k <= pver; ++k) dz(k) = zf(k) - zf(k + 1);
    o.pflx(1) = R8(0.0);
    for (int k = 1; k <= pver; ++k) {
        k1(k) = R8(0.0); i2(k) = R8(0.0); i3(k) = R8(0.0); i4(k) = R8(0.0);
        o.mu(k) = R8(0.0); f(k) = R8(0.0); eps(k) = R8(0.0); o.eu(k) = R8(0.0);
        o.du(k) = R8(0.0); o.ql(k) = R8(0.0); o.cu(k) = R8(0.0); o.evp(k) = R8(0.0);
        o.cmeg(k) = R8(0.0); qds(k) = q(k); o.md(k) = R8(0.0); o.ed(k) = R8(0.0);
        o.sd(k) = s(k); o.qd(k) = q(k); o.mc(k) = R8(0.0); o.qu(k) = q(k);
        o.su(k) = s(k);
        R8 est = R8(CZ_C1) * uw_exp((R8(CZ_C2) * (t(k) - K.tfreez)) / ((t(k) - K.tfreez) + R8(CZ_C3)));
        if (p(k) - est > R8(0.0)) o.qst(k) = K.eps1 * est / (p(k) - est);
        else o.qst(k) = R8(1.0);
        gamma(k) = o.qst(k) * (R8(1.0) + o.qst(k) / K.eps1) * K.eps1 * rl / (rd * uw_sq(t(k))) * rl / cp;
        o.hmn(k) = (cp * t(k) + grav * z(k)) + rl * q(k);
        o.hsat(k) = (cp * t(k) + grav * z(k)) + rl * o.qst(k);
        hu(k) = o.hmn(k);
        hd(k) = o.hmn(k);
        o.rprd(k) = R8(0.0);
    }
    for (int k = 1; k <= msg; ++k) o.rprd(k) = R8(0.0);
    hsthat(msg + 1) = o.hsat(msg + 1);
    qsthat(msg + 1) = o.qst(msg + 1);
    gamhat(msg + 1) = gamma(msg + 1);
    R8 totpcp = R8(0.0), totevp = R8(0.0);
    for (int k = msg + 2; k <= pver; ++k) {
        if (uw_abs(o.qst(k - 1) - o.qst(k)) > R8(1.E-6)) {
            qsthat(k) = uw_log(o.qst(k - 1) / o.qst(k)) * o.qst(k - 1) * o.qst(k) / (o.qst(k - 1) - o.qst(k));
        } else {
            qsthat(k) = o.qst(k);
        }
        hsthat(k) = cp * shat(k) + rl * qsthat(k);
        if (uw_abs(gamma(k - 1) - gamma(k)) > R8(1.E-6)) {
            gamhat(k) = uw_log(gamma(k - 1) / gamma(k)) * gamma(k - 1) * gamma(k) / (gamma(k - 1) - gamma(k));
        } else {
            gamhat(k) = gamma(k);
        }
    }
    jt = uw_imax(lel, limcnv + 1);
    jt = uw_imin(jt, pver);
    jd = pver;
    jlcl = lel;
    R8 hmin = R8(1.E6);
    for (int k = msg + 1; k <= pver; ++k) {
        if (o.hsat(k) <= hmin && k >= jt && k <= jb) {
            hmin = o.hsat(k);
            j0 = k;
        }
    }
    j0 = uw_imin(j0, jb - 2);
    j0 = uw_imax(j0, jt + 2);
    j0 = uw_imin(j0, pver);
    for (int k = msg + 1; k <= pver; ++k) {
        if (k >= jt && k <= jb) {
            hu(k) = o.hmn(mx) + cp * R8(CZ_TIEDKE_ADD);
            o.su(k) = s(mx) + R8(CZ_TIEDKE_ADD);
        }
    }
    for (int k = pver - 1; k >= msg + 1; --k) {
        if (k < jb && k >= jt) {
            k1(k) = k1(k + 1) + (o.hmn(mx) - o.hmn(k)) * dz(k);
            ihat(k) = R8(0.5) * (k1(k + 1) + k1(k));
            i2(k) = i2(k + 1) + ihat(k) * dz(k);
            idag(k) = R8(0.5) * (i2(k + 1) + i2(k));
            i3(k) = i3(k + 1) + idag(k) * dz(k);
            iprm(k) = R8(0.5) * (i3(k + 1) + i3(k));
            i4(k) = i4(k + 1) + iprm(k) * dz(k);
        }
    }
    hmin = R8(1.E6);
    for (int k = msg + 1; k <= pver; ++k) {
        if (k >= j0 && k <= jb && o.hmn(k) <= hmin) {
            hmin = o.hmn(k);
            expdif = o.hmn(mx) - hmin;
        }
    }
    for (int k = msg + 2; k <= pver; ++k) {
        expnum = R8(0.0);
        ftemp = R8(0.0);
        if (k < jt || k >= jb) {
            k1(k) = R8(0.0);
            expnum = R8(0.0);
        } else {
            expnum = o.hmn(mx) - (o.hsat(k - 1) * (zf(k) - z(k)) + o.hsat(k) * (z(k - 1) - zf(k))) / (z(k - 1) - z(k));
        }
        if ((expdif > R8(100.0) && expnum > R8(0.0)) && k1(k) > expnum * dz(k)) {
            ftemp = expnum / k1(k);
            // integer exponents 2, 3, 4 are multiplication chains:
            // x*x, (x*x)*x and (x*x)*(x*x), as gfortran expands them
            R8 ft2 = uw_sq(ftemp), ft3 = uw_sq(ftemp) * ftemp, ft4 = uw_sq(ftemp) * uw_sq(ftemp);
            R8 k1sq = uw_sq(k1(k)), k1cu = uw_sq(k1(k)) * k1(k);
            R8 i2sq = uw_sq(i2(k)), i2cu = uw_sq(i2(k)) * i2(k);
            f(k) = ((ftemp + i2(k) / k1(k) * ft2)
                    + (R8(2.0) * i2sq - k1(k) * i3(k)) / k1sq * ft3)
                   + ((-(R8(5.0) * k1(k) * i2(k) * i3(k)) + R8(5.0) * i2cu) + k1sq * i4(k)) / k1cu * ft4;
            f(k) = uw_max(f(k), R8(0.0));
            f(k) = uw_min(f(k), R8(0.0002));
        }
    }
    if (j0 < jb) {
        if (f(j0) < R8(1.E-6) && f(j0 + 1) > f(j0)) j0 = j0 + 1;
    }
    for (int k = msg + 2; k <= pver; ++k) {
        if (k >= jt && k <= j0) f(k) = uw_max(f(k), f(k - 1));
    }
    const R8 eps0 = f(j0);
    eps(jb) = eps0;
    for (int k = pver; k >= msg + 1; --k) {
        if (k >= j0 && k <= jb) eps(k) = f(j0);
    }
    for (int k = pver; k >= msg + 1; --k) {
        if (k < j0 && k >= jt) eps(k) = f(k);
    }
    if (eps0 > R8(0.0)) {
        o.mu(jb) = R8(1.0);
        o.eu(jb) = o.mu(jb) / dz(jb);
    }
    for (int k = pver; k >= msg + 1; --k) {
        if (eps0 > R8(0.0) && (k >= jt && k < jb)) {
            R8 zuef = zf(k) - zf(jb);
            R8 rmue = (R8(1.0) / eps0) * (uw_exp(eps(k + 1) * zuef) - R8(1.0)) / zuef;
            o.mu(k) = (R8(1.0) / eps0) * (uw_exp(eps(k) * zuef) - R8(1.0)) / zuef;
            o.eu(k) = (rmue - o.mu(k + 1)) / dz(k);
            o.du(k) = (rmue - o.mu(k)) / dz(k);
        }
    }
    int khighest = pver + 1;
    int klowest = 1;
    khighest = uw_imin(khighest, lel);
    klowest = uw_imax(klowest, jb);
    for (int k = klowest - 1; k >= khighest; --k) {
        if (k <= jb - 1 && k >= lel && eps0 > R8(0.0)) {
            if (o.mu(k) < R8(0.01)) {
                hu(k) = hu(jb);
                o.mu(k) = R8(0.0);
                o.eu(k) = R8(0.0);
                o.du(k) = o.mu(k + 1) / dz(k);
            } else {
                hu(k) = o.mu(k + 1) / o.mu(k) * hu(k + 1)
                        + dz(k) / o.mu(k) * (o.eu(k) * o.hmn(k) - o.du(k) * o.hsat(k));
            }
        }
    }
    bool doit = true;
    for (int k = klowest - 2; k >= khighest - 1; --k) {
        if (doit && k <= jb - 2 && k >= lel - 1) {
            if (hu(k) <= hsthat(k) && hu(k + 1) > hsthat(k + 1) && o.mu(k) >= R8(0.02)) {
                if (hu(k) - hsthat(k) < R8(-2000.0)) {
                    jt = k + 1;
                    doit = false;
                } else {
                    jt = k;
                    if (eps0 <= R8(0.0)) doit = false;
                }
            } else if (hu(k) > hu(jb) || o.mu(k) < R8(0.01)) {
                jt = k + 1;
                doit = false;
            }
        }
    }
    for (int k = pver; k >= msg + 1; --k) {
        if (k >= lel && k <= jt && eps0 > R8(0.0)) {
            o.mu(k) = R8(0.0);
            o.eu(k) = R8(0.0);
            o.du(k) = R8(0.0);
            hu(k) = hu(jb);
        }
        if (k == jt && eps0 > R8(0.0)) {
            o.du(k) = o.mu(k + 1) / dz(k);
            o.eu(k) = R8(0.0);
            o.mu(k) = R8(0.0);
        }
    }
    const R8 alfa = R8(0.1);
    jt = uw_imin(jt, jb - 1);
    jd = uw_imax(j0, jt + 1);
    jd = uw_imin(jd, jb);
    hd(jd) = o.hmn(jd - 1);
    R8 epsm = R8(0.0);
    if (jd < jb && eps0 > R8(0.0)) {
        epsm = eps0;
        o.md(jd) = -(alfa * epsm / eps0);
    }
    for (int k = msg + 1; k <= pver; ++k) {
        if ((k > jd && k <= jb) && eps0 > R8(0.0)) {
            R8 zdef = zf(jd) - zf(k);
            o.md(k) = -(alfa / (R8(2.0) * eps0) * (uw_exp(R8(2.0) * epsm * zdef) - R8(1.0)) / zdef);
        }
    }
    for (int k = msg + 1; k <= pver; ++k) {
        if ((k >= jt && k <= jb) && eps0 > R8(0.0) && jd < jb) {
            R8 ratmjb = uw_min(uw_abs(o.mu(jb) / o.md(jb)), R8(1.0));
            o.md(k) = o.md(k) * ratmjb;
        }
    }
    const R8 small = R8(1.e-20);
    for (int k = msg + 1; k <= pver; ++k) {
        if ((k >= jt && k <= pver) && eps0 > R8(0.0)) {
            o.ed(k - 1) = (o.md(k - 1) - o.md(k)) / dz(k - 1);
            R8 mdt = uw_min(o.md(k), -small);
            hd(k) = (o.md(k - 1) * hd(k - 1) - dz(k - 1) * o.ed(k - 1) * o.hmn(k - 1)) / mdt;
        }
    }
    for (int k = msg + 2; k <= pver; ++k) {
        if ((k >= jd && k <= jb) && eps0 > R8(0.0) && jd < jb) {
            qds(k) = qsthat(k) + gamhat(k) * (hd(k) - hsthat(k)) / (rl * (R8(1.0) + gamhat(k)));
        }
    }
    bool done = false;
    for (int k = pver; k >= msg + 2; --k) {
        if ((!done && k > jt && k < jb) && eps0 > R8(0.0)) {
            o.su(k) = o.mu(k + 1) / o.mu(k) * o.su(k + 1)
                      + dz(k) / o.mu(k) * (o.eu(k) - o.du(k)) * s(k);
            o.qu(k) = o.mu(k + 1) / o.mu(k) * o.qu(k + 1)
                      + dz(k) / o.mu(k) * (o.eu(k) * q(k) - o.du(k) * o.qst(k));
            R8 tu = o.su(k) - grav / cp * zf(k);
            R8 estu = R8(CZ_C1) * uw_exp((R8(CZ_C2) * (tu - K.tfreez)) / ((tu - K.tfreez) + R8(CZ_C3)));
            R8 qstu = K.eps1 * estu / ((p(k) + p(k - 1)) / R8(2.0) - estu);
            if (o.qu(k) >= qstu) {
                jlcl = k;
                done = true;
            }
        }
        if (done) break;   // kount >= il2g (:2655): the one column is done
    }
    for (int k = msg + 2; k <= pver; ++k) {
        if (k == jb && eps0 > R8(0.0)) {
            o.qu(k) = q(mx);
            o.su(k) = (hu(k) - rl * o.qu(k)) / cp;
        }
        if ((k > jt && k <= jlcl) && eps0 > R8(0.0)) {
            o.su(k) = shat(k) + (hu(k) - hsthat(k)) / (cp * (R8(1.0) + gamhat(k)));
            o.qu(k) = qsthat(k) + gamhat(k) * (hu(k) - hsthat(k)) / (rl * (R8(1.0) + gamhat(k)));
        }
    }
    for (int k = pver; k >= msg + 2; --k) {
        if (k >= jt && k < jb && eps0 > R8(0.0)) {
            o.cu(k) = ((o.mu(k) * o.su(k) - o.mu(k + 1) * o.su(k + 1)) / dz(k)
                       - (o.eu(k) - o.du(k)) * s(k)) / (rl / cp);
            if (k == jt) o.cu(k) = R8(0.0);
            o.cu(k) = uw_max(R8(0.0), o.cu(k));
        }
    }
    for (int k = pver; k >= msg + 2; --k) {
        o.rprd(k) = R8(0.0);
        if (k >= jt && k < jb && eps0 > R8(0.0) && o.mu(k) >= R8(0.0)) {
            if (o.mu(k) > R8(0.0)) {
                R8 ql1 = R8(1.0) / o.mu(k) * ((o.mu(k + 1) * o.ql(k + 1)
                                                - dz(k) * o.du(k) * o.ql(k + 1)) + dz(k) * o.cu(k));
                o.ql(k) = ql1 / (R8(1.0) + dz(k) * c0mask);
            } else {
                o.ql(k) = R8(0.0);
            }
            totpcp = totpcp + dz(k) * (o.cu(k) - o.du(k) * o.ql(k + 1));
            o.rprd(k) = c0mask * o.mu(k) * o.ql(k);
        }
    }
    o.qd(jd) = qds(jd);
    o.sd(jd) = (hd(jd) - rl * o.qd(jd)) / cp;
    for (int k = msg + 2; k <= pver; ++k) {
        if (k >= jd && k < jb && eps0 > R8(0.0)) {
            o.qd(k + 1) = qds(k + 1);
            o.evp(k) = -(o.ed(k) * q(k)) + (o.md(k) * o.qd(k) - o.md(k + 1) * o.qd(k + 1)) / dz(k);
            o.evp(k) = uw_max(o.evp(k), R8(0.0));
            R8 mdt = uw_min(o.md(k + 1), -small);
            o.sd(k + 1) = ((rl / cp * o.evp(k) - o.ed(k) * s(k)) * dz(k) + o.md(k) * o.sd(k)) / mdt;
            totevp = totevp - dz(k) * o.ed(k) * q(k);
        }
    }
    totevp = (totevp + o.md(jd) * o.qd(jd)) - o.md(jb) * o.qd(jb);
    totpcp = uw_max(totpcp, R8(0.0));
    totevp = uw_max(totevp, R8(0.0));
    for (int k = msg + 2; k <= pver; ++k) {
        if (totevp > R8(0.0) && totpcp > R8(0.0)) {
            o.md(k) = o.md(k) * uw_min(R8(1.0), totpcp / (totevp + totpcp));
            o.ed(k) = o.ed(k) * uw_min(R8(1.0), totpcp / (totevp + totpcp));
            o.evp(k) = o.evp(k) * uw_min(R8(1.0), totpcp / (totevp + totpcp));
        } else {
            o.md(k) = R8(0.0);
            o.ed(k) = R8(0.0);
            o.evp(k) = R8(0.0);
        }
        o.cmeg(k) = o.cu(k) - o.evp(k);
        o.rprd(k) = o.rprd(k) - o.evp(k);
    }
    o.pflx(1) = R8(0.0);
    for (int k = 2; k <= pver + 1; ++k) o.pflx(k) = o.pflx(k - 1) + o.rprd(k - 1) * dz(k - 1);
    for (int k = msg + 1; k <= pver; ++k) o.mc(k) = o.mu(k) + o.md(k);
}

// ---------------------------------------------------------------------------
// closure (:2781-2996), one gathered column; returns mb.
__device__ R8 cz_closure(const CzK& K, Ws& ws, int pver, int msg,
                         V q, V t, V p, V s, V tp, V qu, V su, V mc, V du,
                         V mu, V md, V qd, V sd, V qhat, V shat, V dp,
                         V qstp, V zf, V ql, R8 dsubcld, R8 cape, R8 tl,
                         int lcl, int lel, int jt, int mx) {
    WsMark mark(ws);
    V dtpdt = ws.r8(pver), dqsdtp = ws.r8(pver), dtmdt = ws.r8(pver);
    V dqmdt = ws.r8(pver), dboydt = ws.r8(pver), thetavp = ws.r8(pver);
    V thetavm = ws.r8(pver);
    const R8 rd = K.rgas, cp = K.cp, rl = K.rl, grav = K.grav, eps1 = K.eps1;
    R8 mb = R8(0.0);
    R8 eb = p(mx) * q(mx) / (eps1 + q(mx));
    R8 dtbdt = (R8(1.0) / dsubcld) * (mu(mx) * (shat(mx) - su(mx)) + md(mx) * (shat(mx) - sd(mx)));
    R8 dqbdt = (R8(1.0) / dsubcld) * (mu(mx) * (qhat(mx) - qu(mx)) + md(mx) * (qhat(mx) - qd(mx)));
    R8 debdt = eps1 * p(mx) / uw_sq(eps1 + q(mx)) * dqbdt;
    R8 dtldt = -(R8(2840.0) * (R8(3.5) / t(mx) * dtbdt - debdt / eb)
                 / uw_sq((R8(3.5) * uw_log(t(mx)) - uw_log(eb)) - R8(4.805)));
    for (int k = msg + 1; k <= pver; ++k) {
        dtmdt(k) = R8(0.0);
        dqmdt(k) = R8(0.0);
    }
    for (int k = msg + 1; k <= pver - 1; ++k) {
        if (k == jt) {
            dtmdt(k) = (R8(1.0) / dp(k)) * (mu(k + 1) * ((su(k + 1) - shat(k + 1)) - rl / cp * ql(k + 1))
                                           + md(k + 1) * (sd(k + 1) - shat(k + 1)));
            dqmdt(k) = (R8(1.0) / dp(k)) * (mu(k + 1) * ((qu(k + 1) - qhat(k + 1)) + ql(k + 1))
                                           + md(k + 1) * (qd(k + 1) - qhat(k + 1)));
        }
    }
    const R8 beta = R8(0.0);
    for (int k = msg + 1; k <= pver - 1; ++k) {
        if (k > jt && k < mx) {
            dtmdt(k) = (mc(k) * (shat(k) - s(k)) + mc(k + 1) * (s(k) - shat(k + 1))) / dp(k)
                       - rl / cp * du(k) * (beta * ql(k) + (R8(1.0) - beta) * ql(k + 1));
            dqmdt(k) = ((((mu(k + 1) * ((qu(k + 1) - qhat(k + 1)) + cp / rl * (su(k + 1) - s(k))))
                          - mu(k) * ((qu(k) - qhat(k)) + cp / rl * (su(k) - s(k))))
                         + md(k + 1) * ((qd(k + 1) - qhat(k + 1)) + cp / rl * (sd(k + 1) - s(k))))
                        - md(k) * ((qd(k) - qhat(k)) + cp / rl * (sd(k) - s(k)))) / dp(k)
                       + du(k) * (beta * ql(k) + (R8(1.0) - beta) * ql(k + 1));
        }
    }
    for (int k = msg + 1; k <= pver; ++k) {
        if (k >= lel && k <= lcl) {
            thetavp(k) = tp(k) * uw_pow(R8(1000.0) / p(k), rd / cp)
                         * ((R8(1.0) + R8(1.608) * qstp(k)) - q(mx));
            thetavm(k) = t(k) * uw_pow(R8(1000.0) / p(k), rd / cp) * (R8(1.0) + R8(0.608) * q(k));
            dqsdtp(k) = qstp(k) * (R8(1.0) + qstp(k) / eps1) * eps1 * rl / (rd * uw_sq(tp(k)));
            dtpdt(k) = tp(k) / (R8(1.0) + rl / cp * (dqsdtp(k) - qstp(k) / tp(k)))
                       * (dtbdt / t(mx) + rl / cp * (dqbdt / tl - q(mx) / uw_sq(tl) * dtldt));
            dboydt(k) = ((dtpdt(k) / tp(k) + R8(1.0) / ((R8(1.0) + R8(1.608) * qstp(k)) - q(mx))
                          * (R8(1.608) * dqsdtp(k) * dtpdt(k) - dqbdt))
                         - (dtmdt(k) / t(k) + R8(0.608) / (R8(1.0) + R8(0.608) * q(k)) * dqmdt(k)))
                        * grav * thetavp(k) / thetavm(k);
        }
    }
    for (int k = msg + 1; k <= pver; ++k) {
        if (k > lcl && k < mx) {
            thetavp(k) = tp(k) * uw_pow(R8(1000.0) / p(k), rd / cp) * (R8(1.0) + R8(0.608) * q(mx));
            thetavm(k) = t(k) * uw_pow(R8(1000.0) / p(k), rd / cp) * (R8(1.0) + R8(0.608) * q(k));
            dboydt(k) = (((dtbdt / t(mx) + R8(0.608) / (R8(1.0) + R8(0.608) * q(mx)) * dqbdt)
                          - dtmdt(k) / t(k))
                         - R8(0.608) / (R8(1.0) + R8(0.608) * q(k)) * dqmdt(k))
                        * grav * thetavp(k) / thetavm(k);
        }
    }
    R8 dadt = R8(0.0);
    const int kmin = lel;
    const int kmax = mx - 1;
    for (int k = kmin; k <= kmax; ++k) {
        if (k >= lel && k <= mx - 1) dadt = dadt + dboydt(k) * (zf(k) - zf(k + 1));
    }
    R8 dltaa = R8(-1.0) * (cape - R8(CZ_CAPELMT));
    if (dadt != R8(0.0)) mb = uw_max(dltaa / K.tau / dadt, R8(0.0));
    return mb;
}

// q1q2_pjr (:2998-3124), one gathered column.
__device__ void cz_q1q2_pjr(const CzK& K, int pver, int msg, V dqdt, V dsdt,
                            V qu, V su, V du, V qhat, V shat, V dp, V mu, V md,
                            V sd, V qd, V ql, R8 dsubcld, int jt, int mx,
                            V dl, V evp, V cu) {
    const R8 cp = K.cp, rl = K.rl;
    for (int k = msg + 1; k <= pver; ++k) {
        dsdt(k) = R8(0.0);
        dqdt(k) = R8(0.0);
        dl(k) = R8(0.0);
    }
    const int ktm = uw_imin(pver, jt);
    const int kbm = uw_imin(pver, mx);
    for (int k = ktm; k <= pver - 1; ++k) {
        R8 emc = -cu(k) + evp(k);
        dsdt(k) = -(rl / cp * emc)
                  + (((mu(k + 1) * (su(k + 1) - shat(k + 1)) - mu(k) * (su(k) - shat(k)))
                      + md(k + 1) * (sd(k + 1) - shat(k + 1))) - md(k) * (sd(k) - shat(k))) / dp(k);
        dqdt(k) = emc
                  + (((mu(k + 1) * (qu(k + 1) - qhat(k + 1)) - mu(k) * (qu(k) - qhat(k)))
                      + md(k + 1) * (qd(k + 1) - qhat(k + 1))) - md(k) * (qd(k) - qhat(k))) / dp(k);
        dl(k) = du(k) * ql(k + 1);
    }
    for (int k = kbm; k <= pver; ++k) {
        if (k == mx) {
            dsdt(k) = (R8(1.0) / dsubcld) * (-(mu(k) * (su(k) - shat(k))) - md(k) * (sd(k) - shat(k)));
            dqdt(k) = (R8(1.0) / dsubcld) * (-(mu(k) * (qu(k) - qhat(k))) - md(k) * (qd(k) - qhat(k)));
        } else if (k > mx) {
            dsdt(k) = dsdt(k - 1);
            dqdt(k) = dqdt(k - 1);
        }
    }
}

// ---------------------------------------------------------------------------
// zm_convr (:272-894) for the one-column chunk camzm_driver passes.
struct CzConvr {
    // inputs (CAM order)
    V t, qh, pap, paph, dpp, zm, zi;
    R8 geos, pblh, tpert, landfrac, delt;
    // outputs
    V qtnd, heat, mcon, dlf, pflx, cme, zdu, rprd, mu, md, du, eu, ed, dp, ql;
    R8 cape, dsubcld, jctop, jcbot, prec, rliq;
    int jt, maxg, lengath;
};

__device__ int cz_zm_convr(const CzK& K, Ws& ws, int pver, CzConvr& c) {
    WsMark mark(ws);
    const int msg = CZ_MSG;
    const int pverp = pver + 1;
    V q = ws.r8(pver), p = ws.r8(pver), z = ws.r8(pver), s = ws.r8(pver);
    V tp = ws.r8(pver), zf = ws.r8(pverp), pf = ws.r8(pverp), qstp = ws.r8(pver);
    V dqdt = ws.r8(pver), dsdt = ws.r8(pver), sd = ws.r8(pver), qd = ws.r8(pver);
    V mc = ws.r8(pver), qhat = ws.r8(pver), qu = ws.r8(pver), su = ws.r8(pver);
    V qs = ws.r8(pver), shat = ws.r8(pver), hmn = ws.r8(pver), hsat = ws.r8(pver);
    V qlg = ws.r8(pver), dlg = ws.r8(pver), pflxg = ws.r8(pverp), cug = ws.r8(pver);
    V evpg = ws.r8(pver), cmeg = ws.r8(pver), rprdg = ws.r8(pver);
    for (int k = 1; k <= pver; ++k) {
        c.qtnd(k) = R8(0.0); c.heat(k) = R8(0.0);
    }
    for (int k = 1; k <= pverp; ++k) c.mcon(k) = R8(0.0);
    c.rliq = R8(0.0);
    c.prec = R8(0.0);
    for (int k = 1; k <= pver; ++k) {
        dqdt(k) = R8(0.0); dsdt(k) = R8(0.0);
        c.pflx(k) = R8(0.0); pflxg(k) = R8(0.0); c.cme(k) = R8(0.0);
        c.rprd(k) = R8(0.0); c.zdu(k) = R8(0.0); c.ql(k) = R8(0.0);
        qlg(k) = R8(0.0); c.dlf(k) = R8(0.0); dlg(k) = R8(0.0);
    }
    c.pflx(pverp) = R8(0.0);
    pflxg(pverp) = R8(0.0);
    R8 pblt = uw_real(pver);
    c.dsubcld = R8(0.0);
    c.jt = pver;
    c.jctop = uw_real(pver);
    c.jcbot = R8(1.0);
    const R8 zs = c.geos * K.rgrav;
    pf(pverp) = c.paph(pverp) * R8(0.01);
    zf(pverp) = c.zi(pverp) + zs;
    for (int k = 1; k <= pver; ++k) {
        p(k) = c.pap(k) * R8(0.01);
        pf(k) = c.paph(k) * R8(0.01);
        z(k) = c.zm(k) + zs;
        zf(k) = c.zi(k) + zs;
    }
    for (int k = pver - 1; k >= msg + 1; --k) {
        if (uw_abs((z(k) - zs) - c.pblh) < (zf(k) - zf(k + 1)) * R8(0.5)) pblt = uw_real(k);
    }
    for (int k = 1; k <= pver; ++k) {
        q(k) = c.qh(k);
        s(k) = c.t(k) + (K.grav / K.cpres) * z(k);
        tp(k) = R8(0.0);
        shat(k) = s(k);
        qhat(k) = q(k);
    }
    R8 tl = R8(400.0);
    int lcl = 1, lel = pver, maxi = 1;
    c.maxg = 1;
    int st = cz_buoyan_dilute(K, ws, pver, msg, q, c.t, p, z, pf, tp, qstp, tl,
                              c.cape, pblt, lcl, lel, maxi, c.tpert);
    if (st) return st;
    c.lengath = (c.cape > R8(CZ_CAPELMT)) ? 1 : 0;
    if (c.lengath == 0) return 0;
    // gathered arrays: the one column (ideep(1) = 1); qg, tg, pg, zg, sg,
    // tpg, zfg, qstpg are q, t, p, z, s, tp, zf, qstp themselves.
    for (int k = 1; k <= pver; ++k) c.dp(k) = R8(0.01) * c.dpp(k);
    c.maxg = maxi;
    const int jb = maxi;
    for (int k = msg + 1; k <= pver; ++k) {
        if (k >= c.maxg) c.dsubcld = c.dsubcld + c.dp(k);
    }
    for (int k = msg + 2; k <= pver; ++k) {
        R8 sdifr = R8(0.0), qdifr = R8(0.0);
        if (s(k) > R8(0.0) || s(k - 1) > R8(0.0))
            sdifr = uw_abs((s(k) - s(k - 1)) / uw_max(s(k - 1), s(k)));
        if (q(k) > R8(0.0) || q(k - 1) > R8(0.0))
            qdifr = uw_abs((q(k) - q(k - 1)) / uw_max(q(k - 1), q(k)));
        if (sdifr > R8(1.E-6)) shat(k) = uw_log(s(k - 1) / s(k)) * s(k - 1) * s(k) / (s(k - 1) - s(k));
        else shat(k) = R8(0.5) * (s(k) + s(k - 1));
        if (qdifr > R8(1.E-6)) qhat(k) = uw_log(q(k - 1) / q(k)) * q(k - 1) * q(k) / (q(k - 1) - q(k));
        else qhat(k) = R8(0.5) * (q(k) + q(k - 1));
    }
    CzCld o;
    o.mu = c.mu; o.eu = c.eu; o.du = c.du; o.md = c.md; o.ed = c.ed;
    o.sd = sd; o.qd = qd; o.mc = mc; o.qu = qu; o.su = su; o.qst = qs;
    o.hmn = hmn; o.hsat = hsat; o.ql = qlg; o.cmeg = cmeg; o.pflx = pflxg;
    o.evp = evpg; o.cu = cug; o.rprd = rprdg;
    int jt = pver, jlcl = 0, j0 = 0, jd = pver;
    cz_cldprp(K, ws, pver, msg, CZ_LIMCNV, q, c.t, p, z, s, zf, shat,
              c.maxg, lel, c.maxg, jt, jlcl, j0, jd, c.landfrac, o);
    for (int k = msg + 1; k <= pver; ++k) {
        R8 thick = zf(k) - zf(k + 1);
        c.du(k) = c.du(k) * thick / c.dp(k);
        c.eu(k) = c.eu(k) * thick / c.dp(k);
        c.ed(k) = c.ed(k) * thick / c.dp(k);
        cug(k) = cug(k) * thick / c.dp(k);
        cmeg(k) = cmeg(k) * thick / c.dp(k);
        rprdg(k) = rprdg(k) * thick / c.dp(k);
        evpg(k) = evpg(k) * thick / c.dp(k);
    }
    R8 mb = cz_closure(K, ws, pver, msg, q, c.t, p, s, tp, qu, su, mc, c.du,
                       c.mu, c.md, qd, sd, qhat, shat, c.dp, qstp, zf, qlg,
                       c.dsubcld, c.cape, tl, lcl, lel, jt, c.maxg);
    R8 mumax = R8(0.0);
    for (int k = msg + 2; k <= pver; ++k) mumax = uw_max(mumax, c.mu(k) / c.dp(k));
    if (mumax > R8(0.0)) mb = uw_min(mb, R8(0.5) / (c.delt * mumax));
    else mb = R8(0.0);
    // no_deep_pbl = .true. (zm_conv_init passes NO_DEEP_PBL_IN=.true.)
    if (c.zm(jt) < c.pblh) mb = R8(0.0);
    for (int k = msg + 1; k <= pver; ++k) {
        c.mu(k) = c.mu(k) * mb;
        c.md(k) = c.md(k) * mb;
        mc(k) = mc(k) * mb;
        c.du(k) = c.du(k) * mb;
        c.eu(k) = c.eu(k) * mb;
        c.ed(k) = c.ed(k) * mb;
        cmeg(k) = cmeg(k) * mb;
        rprdg(k) = rprdg(k) * mb;
        cug(k) = cug(k) * mb;
        evpg(k) = evpg(k) * mb;
        pflxg(k + 1) = pflxg(k + 1) * mb * R8(100.0) / K.grav;
    }
    cz_q1q2_pjr(K, pver, msg, dqdt, dsdt, qu, su, c.du, qhat, shat, c.dp,
                c.mu, c.md, sd, qd, qlg, c.dsubcld, jt, c.maxg, dlg, evpg, cug);
    for (int k = msg + 1; k <= pver; ++k) {
        q(k) = c.qh(k) + R8(2.0) * c.delt * dqdt(k);
        c.qtnd(k) = dqdt(k);
        c.cme(k) = cmeg(k);
        c.rprd(k) = rprdg(k);
        c.zdu(k) = c.du(k);
        c.mcon(k) = mc(k);
        c.heat(k) = dsdt(k) * K.cpres;
        c.dlf(k) = dlg(k);
        c.pflx(k) = pflxg(k);
        c.ql(k) = qlg(k);
    }
    c.jt = jt;
    c.jctop = uw_real(jt);
    c.jcbot = uw_real(c.maxg);
    c.pflx(pverp) = pflxg(pverp);
    for (int k = pver; k >= msg + 1; --k) {
        c.prec = (c.prec - c.dpp(k) * (q(k) - c.qh(k))) - c.dpp(k) * c.dlf(k) * R8(2.0) * c.delt;
    }
    c.prec = K.rgrav * uw_max(c.prec, R8(0.0)) / (R8(2.0) * c.delt) / R8(1000.0);
    for (int k = 1; k <= pver; ++k) c.rliq = c.rliq + c.dlf(k) * c.dpp(k) / R8(UW_GRAVIT);
    c.rliq = c.rliq / R8(1000.0);
    return 0;
}

// ---------------------------------------------------------------------------
// zm_conv_evap (:897-1079), one column.  prec is in/out (m/s); snow out.
__device__ void cz_zm_conv_evap(Ws& ws, int pver, V t, V pmid, V pdel, V q,
                                V tend_s, V tend_s_snwprd, V tend_s_snwevmlt,
                                V tend_q, V prdprec, V cldfrc, R8 deltat,
                                R8& prec, R8& snow, V ntprprd, V ntsnprd,
                                V flxprec, V flxsnow) {
    WsMark mark(ws);
    V est = ws.r8(pver), qsat = ws.r8(pver);
    const R8 gravit = R8(UW_GRAVIT), latice = R8(UW_LATICE), latvap = R8(UW_LATVAP);
    prec = prec * R8(1000.0);
    uw_aqsat(t, pmid, est, qsat, 1, pver);
    flxprec(1) = R8(0.0);
    flxsnow(1) = R8(0.0);
    R8 evpvint = R8(0.0);
    for (int k = 1; k <= pver; ++k) {
        R8 fice, fsnow_conv;
        cam_cldwat_fice(t(k), fice, fsnow_conv);
        R8 flxsntm, snowmlt;
        if (t(k) > R8(UW_TMELT)) {
            flxsntm = R8(0.0);
            snowmlt = flxsnow(k) * gravit / pdel(k);
        } else {
            flxsntm = flxsnow(k);
            snowmlt = R8(0.0);
        }
        R8 evplimit = uw_max(R8(1.0) - q(k) / qsat(k), R8(0.0));
        R8 evpprec = R8(CZ_KE) * (R8(1.0) - cldfrc(k)) * evplimit * uw_sqrt(flxprec(k));
        evplimit = uw_max(R8(0.0), (qsat(k) - q(k)) / deltat);
        evplimit = uw_min(evplimit, flxprec(k) * gravit / pdel(k));
        evplimit = uw_min(evplimit, (prec - evpvint) * gravit / pdel(k));
        evpprec = uw_min(evplimit, evpprec);
        R8 evpsnow, work1;
        if (flxprec(k) > R8(0.0)) {
            work1 = uw_min(uw_max(R8(0.0), flxsntm / flxprec(k)), R8(1.0));
            evpsnow = evpprec * work1;
        } else {
            evpsnow = R8(0.0);
        }
        evpvint = evpvint + evpprec * pdel(k) / gravit;
        ntprprd(k) = prdprec(k) - evpprec;
        if (flxprec(k) > R8(0.0)) work1 = uw_min(uw_max(R8(0.0), flxsnow(k) / flxprec(k)), R8(1.0));
        else work1 = R8(0.0);
        R8 work2 = uw_max(fsnow_conv, work1);
        if (snowmlt > R8(0.0)) work2 = R8(0.0);
        ntsnprd(k) = (prdprec(k) * work2 - evpsnow) - snowmlt;
        tend_s_snwprd(k) = prdprec(k) * work2 * latice;
        tend_s_snwevmlt(k) = -(evpsnow + snowmlt) * latice;
        flxprec(k + 1) = flxprec(k) + ntprprd(k) * pdel(k) / gravit;
        flxsnow(k + 1) = flxsnow(k) + ntsnprd(k) * pdel(k) / gravit;
        flxprec(k + 1) = uw_max(flxprec(k + 1), R8(0.0));
        flxsnow(k + 1) = uw_max(flxsnow(k + 1), R8(0.0));
        tend_s(k) = -(evpprec * latvap) + ntsnprd(k) * latice;
        tend_q(k) = evpprec;
    }
    prec = flxprec(pver + 1) / R8(1000.0);
    snow = flxsnow(pver + 1) / R8(1000.0);
}

// ---------------------------------------------------------------------------
// convtran (:1083-1378) for ONE constituent of the 'wet' type (CAM_INIT adds
// all five without mixtype; cam5_support.CONSTITUENT_TYPE), fracis = 1
// (camzm_driver :673-680), so dptmp/dutmp/eutmp/edtmp are dp/du/eu/ed and
// fisg's factor of one is an exact identity.  lengath = il2g.
__device__ void cz_convtran_one(Ws& ws, int pver, int lengath, V q, V mu, V md,
                                V du, V eu, V ed, V dp, int jt, int mx, V dqdt) {
    if (lengath == 0) {
        for (int k = 1; k <= pver; ++k) dqdt(k) = R8(0.0);
        return;
    }
    WsMark mark(ws);
    V chat = ws.r8(pver), cond = ws.r8(pver), cnst = ws.r8(pver);
    V conu = ws.r8(pver), dcondt = ws.r8(pver);
    const R8 small = R8(1.e-36), mbsth = R8(1.e-15);
    const int ktm = uw_imin(pver, jt);
    const int kbm = uw_imin(pver, mx);
    for (int k = 1; k <= pver; ++k) cnst(k) = q(k);
    for (int k = 1; k <= pver; ++k) {
        int km1 = uw_imax(1, k - 1);
        R8 minc = uw_min(cnst(km1), cnst(k));
        R8 maxc = uw_max(cnst(km1), cnst(k));
        R8 cdifr;
        if (minc < R8(0.0)) cdifr = R8(0.0);
        else cdifr = uw_abs(cnst(k) - cnst(km1)) / uw_max(maxc, small);
        if (cdifr > R8(1.E-6)) {
            R8 cabv = uw_max(cnst(km1), maxc * R8(1.e-12));
            R8 cbel = uw_max(cnst(k), maxc * R8(1.e-12));
            chat(k) = uw_log(cabv / cbel) / (cabv - cbel) * cabv * cbel;
        } else {
            chat(k) = R8(0.5) * (cnst(k) + cnst(km1));
        }
        conu(k) = chat(k);
        cond(k) = chat(k);
        dcondt(k) = R8(0.0);
    }
    {
        const int k = 2, km1 = 1, kk = pver;
        R8 mupdudp = mu(kk) + du(kk) * dp(kk);
        if (mupdudp > mbsth) conu(kk) = (eu(kk) * cnst(kk) * dp(kk)) / mupdudp;
        if (md(k) < -mbsth) cond(k) = (-(ed(km1) * cnst(km1) * dp(km1))) / md(k);
    }
    for (int kk = pver - 1; kk >= 1; --kk) {
        int kkp1 = uw_imin(pver, kk + 1);
        R8 mupdudp = mu(kk) + du(kk) * dp(kk);
        if (mupdudp > mbsth) {
            conu(kk) = (mu(kkp1) * conu(kkp1) + eu(kk) * cnst(kk) * dp(kk)) / mupdudp;
        }
    }
    for (int k = 3; k <= pver; ++k) {
        int km1 = uw_imax(1, k - 1);
        if (md(k) < -mbsth) {
            cond(k) = (md(km1) * cond(km1) - ed(km1) * cnst(km1) * dp(km1)) / md(k);
        }
    }
    for (int k = ktm; k <= pver; ++k) {
        int km1 = uw_imax(1, k - 1);
        int kp1 = uw_imin(pver, k + 1);
        R8 fluxin = (mu(kp1) * conu(kp1) + mu(k) * uw_min(chat(k), cnst(km1)))
                    - (md(k) * cond(k) + md(kp1) * uw_min(chat(kp1), cnst(kp1)));
        R8 fluxout = (mu(k) * conu(k) + mu(kp1) * uw_min(chat(kp1), cnst(k)))
                     - (md(kp1) * cond(kp1) + md(k) * uw_min(chat(k), cnst(k)));
        R8 netflux = fluxin - fluxout;
        if (uw_abs(netflux) < uw_max(fluxin, fluxout) * R8(1.e-12)) netflux = R8(0.0);
        dcondt(k) = netflux / dp(k);
    }
    for (int k = kbm; k <= pver; ++k) {
        int km1 = uw_imax(1, k - 1);
        if (k == mx) {
            R8 fluxin = mu(k) * uw_min(chat(k), cnst(km1)) - md(k) * cond(k);
            R8 fluxout = mu(k) * conu(k) - md(k) * uw_min(chat(k), cnst(k));
            R8 netflux = fluxin - fluxout;
            if (uw_abs(netflux) < uw_max(fluxin, fluxout) * R8(1.e-12)) netflux = R8(0.0);
            dcondt(k) = netflux / dp(k);
        } else if (k > mx) {
            dcondt(k) = R8(0.0);
        }
    }
    for (int k = 1; k <= pver; ++k) dqdt(k) = dcondt(k);
}

// ---------------------------------------------------------------------------
// momtran (:1382-1795), u and v (ncnst = 2, both transported), one column.
// q/dqdt/pguall/pgdall/icwu/icwd hold component m at slot (m-1)*pver + k.
__device__ void cz_momtran(Ws& ws, int pver, int lengath, V q, V mu, V md,
                           V du, V eu, V ed, V dp, int jt, int mx, R8 dt,
                           V dqdt, V pguall, V pgdall, V icwu, V icwd, V seten) {
    WsMark mark(ws);
    const int pverp = pver + 1;
    V chat = ws.r8(pver), cond = ws.r8(pver), cnst = ws.r8(pver);
    V conu = ws.r8(pver), dcondt = ws.r8(pver), mududp = ws.r8(pver);
    V mddudp = ws.r8(pver), pgu = ws.r8(pver), pgd = ws.r8(pver);
    V gseten = ws.r8(pver), mflux = ws.r8(2 * pverp), wind0 = ws.r8(2 * pver);
    V windf = ws.r8(2 * pver);
    auto at = [](V a, int n, int m, int k) -> R8& { return a((m - 1) * n + k); };
    for (int m = 1; m <= 2; ++m) {
        for (int k = 1; k <= pver; ++k) {
            at(pguall, pver, m, k) = R8(0.0);
            at(pgdall, pver, m, k) = R8(0.0);
            at(icwu, pver, m, k) = at(q, pver, m, k);
            at(icwd, pver, m, k) = at(q, pver, m, k);
            at(wind0, pver, m, k) = R8(0.0);
            at(windf, pver, m, k) = R8(0.0);
        }
        for (int k = 1; k <= pverp; ++k) at(mflux, pverp, m, k) = R8(0.0);
    }
    for (int k = 1; k <= pver; ++k) {
        seten(k) = R8(0.0);
        gseten(k) = R8(0.0);
    }
    const R8 momcu = R8(0.4), momcd = R8(0.4);
    const R8 mbsth = R8(1.e-15);
    int ktm = pver, kbm = pver;
    if (lengath == 1) {
        ktm = uw_imin(ktm, jt);
        kbm = uw_imin(kbm, mx);
    }
    for (int m = 1; m <= 2; ++m) {
        // dqdt(:,:,m) = 0 happens below, after the gathered work (:1710)
        if (lengath == 0) {
            for (int k = 1; k <= pver; ++k) at(dqdt, pver, m, k) = R8(0.0);
            continue;
        }
        for (int k = 1; k <= pver; ++k) {
            cnst(k) = at(q, pver, m, k);
            at(wind0, pver, m, k) = cnst(k);
        }
        for (int k = 1; k <= pver; ++k) {
            int km1 = uw_imax(1, k - 1);
            chat(k) = R8(0.5) * (cnst(k) + cnst(km1));
            conu(k) = chat(k);
            cond(k) = chat(k);
            dcondt(k) = R8(0.0);
        }
        pgu(1) = R8(0.0);
        pgd(1) = R8(0.0);
        for (int k = 2; k <= pver - 1; ++k) {
            int km1 = uw_imax(1, k - 1);
            int kp1 = uw_imin(pver, k + 1);
            mududp(k) = mu(k) * (cnst(k) - cnst(km1)) / dp(km1)
                        + mu(kp1) * (cnst(kp1) - cnst(k)) / dp(k);
            pgu(k) = -(momcu * R8(0.5) * mududp(k));
            mddudp(k) = md(k) * (cnst(k) - cnst(km1)) / dp(km1)
                        + md(kp1) * (cnst(kp1) - cnst(k)) / dp(k);
            pgd(k) = -(momcd * R8(0.5) * mddudp(k));
        }
        {
            const int k = pver;
            const int km1 = uw_imax(1, k - 1);
            mududp(k) = mu(k) * (cnst(k) - cnst(km1)) / dp(km1);
            pgu(k) = -(momcu * mududp(k));
            mddudp(k) = md(k) * (cnst(k) - cnst(km1)) / dp(km1);
            pgd(k) = -(momcd * mddudp(k));
        }
        {
            const int k = 2, km1 = 1, kk = pver;
            R8 mupdudp = mu(kk) + du(kk) * dp(kk);
            if (mupdudp > mbsth) {
                conu(kk) = (eu(kk) * cnst(kk) * dp(kk) + pgu(kk) * dp(kk)) / mupdudp;
            }
            if (md(k) < -mbsth) {
                // WRF :1633, including its original grouping.
                cond(k) = (-(ed(km1) * cnst(km1) * dp(km1))) - pgd(km1) * dp(km1) / md(k);
            }
        }
        for (int kk = pver - 1; kk >= 1; --kk) {
            int kkp1 = uw_imin(pver, kk + 1);
            R8 mupdudp = mu(kk) + du(kk) * dp(kk);
            if (mupdudp > mbsth) {
                conu(kk) = ((mu(kkp1) * conu(kkp1) + eu(kk) * cnst(kk) * dp(kk)) + pgu(kk) * dp(kk)) / mupdudp;
            }
        }
        for (int k = 3; k <= pver; ++k) {
            int km1 = uw_imax(1, k - 1);
            if (md(k) < -mbsth) {
                cond(k) = ((md(km1) * cond(km1) - ed(km1) * cnst(km1) * dp(km1)) - pgd(km1) * dp(km1)) / md(k);
            }
        }
        for (int k = ktm; k <= pver; ++k) {
            int kp1 = uw_imin(pver, k + 1);
            dcondt(k) = (((mu(kp1) * (conu(kp1) - chat(kp1)) - mu(k) * (conu(k) - chat(k)))
                          + md(kp1) * (cond(kp1) - chat(kp1))) - md(k) * (cond(k) - chat(k))) / dp(k);
        }
        for (int k = kbm; k <= pver; ++k) {
            if (k == mx) {
                dcondt(k) = (R8(1.0) / dp(k)) * (-(mu(k) * (conu(k) - chat(k))) - md(k) * (cond(k) - chat(k)));
            }
        }
        for (int k = 1; k <= pver; ++k) {
            at(dqdt, pver, m, k) = dcondt(k);
            at(pguall, pver, m, k) = -pgu(k);
            at(pgdall, pver, m, k) = -pgd(k);
            at(icwu, pver, m, k) = conu(k);
            at(icwd, pver, m, k) = cond(k);
        }
        for (int k = ktm; k <= pver; ++k) {
            at(mflux, pverp, m, k) = -(mu(k) * (conu(k) - chat(k))) - md(k) * (cond(k) - chat(k));
        }
        for (int k = ktm; k <= pver; ++k) {
            int kp1 = k + 1;
            at(windf, pver, m, k) = cnst(k)
                - (at(mflux, pverp, m, kp1) - at(mflux, pverp, m, k)) * dt / dp(k);
        }
    }
    if (lengath == 1) {
        for (int k = ktm; k <= pver; ++k) {
            int km1 = uw_imax(1, k - 1);
            int kp1 = uw_imin(pver, k + 1);
            R8 utop = (at(wind0, pver, 1, k) + at(wind0, pver, 1, km1)) / R8(2.0);
            R8 vtop = (at(wind0, pver, 2, k) + at(wind0, pver, 2, km1)) / R8(2.0);
            R8 ubot = (at(wind0, pver, 1, kp1) + at(wind0, pver, 1, k)) / R8(2.0);
            R8 vbot = (at(wind0, pver, 2, kp1) + at(wind0, pver, 2, k)) / R8(2.0);
            R8 fket = utop * at(mflux, pverp, 1, k) + vtop * at(mflux, pverp, 2, k);
            R8 fkeb = ubot * at(mflux, pverp, 1, k + 1) + vbot * at(mflux, pverp, 2, k + 1);
            R8 ketend_cons = (fket - fkeb) / dp(k);
            R8 ketend = ((uw_sq(at(windf, pver, 1, k)) + uw_sq(at(windf, pver, 2, k)))
                         - (uw_sq(at(wind0, pver, 1, k)) + uw_sq(at(wind0, pver, 2, k)))) * R8(0.5) / dt;
            gseten(k) = ketend_cons - ketend;
        }
        for (int k = 1; k <= pver; ++k) seten(k) = gseten(k);
    }
}

// ---------------------------------------------------------------------------
// get_tpert (camzm_driver :873-1022), default REAL as in WRF.  Returns the
// status word; tpert is real(r8) in WRF (its RHS is default REAL).
__device__ int cz_get_tpert(int blpbl, int sfclay, float dx, float mavail,
                            int kpbl, float pblh, float dzlowest, float psfc,
                            float qvlowest, float t_phylowest, float thlowest,
                            float tsk, const float* tke_pbl, long long tke_stride,
                            int kte, float ust, float u_phylowest,
                            float v_phylowest, float hfx, float qfx,
                            float tpert_camuwpbl, R8& tpert) {
    const float fak = 8.5f, tfac = 1.f, wfac = 1.f, wpertmin = 1.e-6f;
    if (blpbl == CZ_CAMUWPBLSCHEME) {
        tpert = cam_widen(tpert_camuwpbl);
        return 0;
    }
    if (blpbl != CZ_MYJPBLSCHEME) return CZ_ERR_ARM;
    bool unstable_or_neutral = false;
    if (sfclay == CZ_MYJSFCSCHEME) {
        if (pblh <= 0.f) return CZ_ERR_PBLH;
        float za = cam_fmul(0.5f, dzlowest);
        float e1 = cam_fmul(CZ_MC_SVP1, gfk_exp(cam_fdiv(cam_fmul(CZ_MC_SVP2, cam_fsub(tsk, CZ_MC_SVPT0)),
                                                         cam_fsub(tsk, CZ_MC_SVP3))));
        float psfccmb = cam_fdiv(psfc, 1000.f);
        float qsfc = cam_fdiv(cam_fmul(CZ_MC_EP_2, e1), cam_fsub(psfccmb, e1));
        float thgb = cam_fmul(tsk, gfk_pow(cam_fdiv(100.f, psfccmb), CZ_MC_RCP));
        float tskv = cam_fmul(thgb, cam_fadd(1.f, cam_fmul(cam_fmul(CZ_MC_EP_1, qsfc), mavail)));
        float tvcon = cam_fadd(1.f, cam_fmul(CZ_MC_EP_1, qvlowest));
        float thv = cam_fmul(thlowest, tvcon);
        float dthvdz = cam_fsub(thv, tskv);
        float govrth = cam_fdiv(CZ_MC_G, thlowest);
        float rhox = cam_fdiv(psfc, cam_fmul(cam_fmul(CZ_MC_R_D, t_phylowest), tvcon));
        float flux = cam_fmax(cam_fadd(cam_fdiv(cam_fdiv(hfx, rhox), CZ_MC_CP),
                                       cam_fdiv(cam_fmul(cam_fmul(CZ_MC_EP_1, tskv), qfx), rhox)), 0.f);
        float vconv = gfk_pow(cam_fmul(cam_fmul(cam_fdiv(CZ_MC_G, tsk), pblh), flux), 0.33f);
        float vsgd = cam_fmul(0.32f, gfk_pow(cam_fmax(cam_fsub(cam_fdiv(dx, 5000.f), 1.f), 0.f), 0.33f));
        float wspd = cam_fsqrt(cam_fadd(cam_fmul(u_phylowest, u_phylowest), cam_fmul(v_phylowest, v_phylowest)));
        wspd = cam_fsqrt(cam_fadd(cam_fadd(cam_fmul(wspd, wspd), cam_fmul(vconv, vconv)), cam_fmul(vsgd, vsgd)));
        wspd = cam_fmax(wspd, 0.1f);
        float br2 = cam_fdiv(cam_fmul(cam_fmul(govrth, za), dthvdz), cam_fmul(wspd, wspd));
        if (br2 <= 0.f) unstable_or_neutral = true;
    } else {
        return CZ_ERR_ARM;
    }
    float tvcon = cam_fadd(1.f, cam_fmul(CZ_MC_EP_1, qvlowest));
    float rhox = cam_fdiv(psfc, cam_fmul(cam_fmul(CZ_MC_R_D, t_phylowest), tvcon));
    float tp;
    if (unstable_or_neutral) {
        float ebrk = 0.f;
        const int ktop = kpbl < kte + 1 ? kpbl : kte + 1;
        for (int k = 1; k <= ktop; ++k) ebrk = cam_fadd(ebrk, tke_pbl[(long long)(k - 1) * tke_stride]);
        ebrk = cam_fdiv(ebrk, (float)kpbl);
        float wpert = cam_fmax(cam_fmul(wfac, cam_fsqrt(ebrk)), wpertmin);
        tp = cam_fmax(cam_fdiv(cam_fmul(fabsf(cam_fdiv(cam_fdiv(hfx, rhox), CZ_MC_CP)), tfac), wpert), 0.f);
    } else {
        tp = cam_fmax(cam_fdiv(cam_fmul(cam_fdiv(cam_fdiv(hfx, rhox), CZ_MC_CP), fak), ust), 0.f);
    }
    tpert = cam_widen(tp);
    return 0;
}

// ---------------------------------------------------------------------------
// The kernel's argument tables.  Their order is gpuwm/core/camzm.py's
// CAMZM_INPUTS / CAMZM_OUTPUTS / CAMZM_INT_OUTPUTS (a test reads these
// defines back and holds the two equal).
#define CZ_IN_TH 0
#define CZ_IN_T 1
#define CZ_IN_P 2
#define CZ_IN_PI 3
#define CZ_IN_QV 4
#define CZ_IN_QC 5
#define CZ_IN_QI 6
#define CZ_IN_Z 7
#define CZ_IN_DZ8W 8
#define CZ_IN_U 9
#define CZ_IN_V 10
#define CZ_IN_CLDFRA 11
#define CZ_IN_P8W 12
#define CZ_IN_Z_AT_W 13
#define CZ_IN_TKE_PBL 14
#define CZ_IN_TSK 15
#define CZ_IN_UST 16
#define CZ_IN_MAVAIL 17
#define CZ_IN_PBLH 18
#define CZ_IN_XLAND 19
#define CZ_IN_HT 20
#define CZ_IN_PSFC 21
#define CZ_IN_HFX 22
#define CZ_IN_QFX 23
#define CZ_IN_TPERT 24
#define CZ_NIN 25

#define CZ_OUT_RTHCUTEN 0
#define CZ_OUT_RQVCUTEN 1
#define CZ_OUT_RQCCUTEN 2
#define CZ_OUT_RQICUTEN 3
#define CZ_OUT_RQCNCUTEN 4
#define CZ_OUT_RQINCUTEN 5
#define CZ_OUT_RUCUTEN 6
#define CZ_OUT_RVCUTEN 7
#define CZ_OUT_ZMMU 8
#define CZ_OUT_ZMMD 9
#define CZ_OUT_ZMDT 10
#define CZ_OUT_ZMDQ 11
#define CZ_OUT_DLF 12
#define CZ_OUT_EVAPTZM 13
#define CZ_OUT_FZSNTZM 14
#define CZ_OUT_EVSNTZM 15
#define CZ_OUT_EVAPQZM 16
#define CZ_OUT_ZMFLXPRC 17
#define CZ_OUT_ZMFLXSNW 18
#define CZ_OUT_ZMNTPRPD 19
#define CZ_OUT_ZMNTSNPD 20
#define CZ_OUT_ZMEIHEAT 21
#define CZ_OUT_ZMMTU 22
#define CZ_OUT_ZMMTV 23
#define CZ_OUT_ZMUPGU 24
#define CZ_OUT_ZMUPGD 25
#define CZ_OUT_ZMVPGU 26
#define CZ_OUT_ZMVPGD 27
#define CZ_OUT_ZMICUU 28
#define CZ_OUT_ZMICUD 29
#define CZ_OUT_ZMICVU 30
#define CZ_OUT_ZMICVD 31
#define CZ_OUT_ZMDICE 32
#define CZ_OUT_ZMDLIQ 33
#define CZ_OUT_EVAPCDP3D 34
#define CZ_OUT_ICWMRDP3D 35
#define CZ_OUT_RPRDDP3D 36
#define CZ_OUT_DP3D 37
#define CZ_OUT_DU3D 38
#define CZ_OUT_ED3D 39
#define CZ_OUT_EU3D 40
#define CZ_OUT_MD3D 41
#define CZ_OUT_MU3D 42
#define CZ_OUT_CMFMC 43
#define CZ_OUT_CMFMCDZM 44
#define CZ_OUT_RAINCV 45
#define CZ_OUT_PRATEC 46
#define CZ_OUT_CAPE 47
#define CZ_OUT_RLIQ 48
#define CZ_OUT_PCONVB 49
#define CZ_OUT_PCONVT 50
#define CZ_OUT_CUBOT 51
#define CZ_OUT_CUTOP 52
#define CZ_OUT_PRECCDZM 53
#define CZ_OUT_PRECZ 54
#define CZ_OUT_DSUBCLD2D 55
#define CZ_NOUT 56

#define CZ_IOUT_IDEEP2D 0
#define CZ_IOUT_JT2D 1
#define CZ_IOUT_MAXG2D 2
#define CZ_IOUT_LENGATH2D 3
#define CZ_NIOUT 4

struct CzCtx {
    const unsigned long long* in;
    const unsigned long long* out;
    const unsigned long long* iout;
    long long ncols;
    long long col;
};

__device__ __forceinline__ float cz_in(const CzCtx& c, int f, int k) {
    return reinterpret_cast<const float*>(c.in[f])[(long long)(k - 1) * c.ncols + c.col];
}
__device__ __forceinline__ float cz_in2(const CzCtx& c, int f) {
    return reinterpret_cast<const float*>(c.in[f])[c.col];
}
// Outputs a launch did not ask for are null and skipped.
__device__ __forceinline__ void cz_out(const CzCtx& c, int f, int k, float v) {
    float* p = reinterpret_cast<float*>(c.out[f]);
    if (p) p[(long long)(k - 1) * c.ncols + c.col] = v;
}
__device__ __forceinline__ void cz_out2(const CzCtx& c, int f, float v) {
    float* p = reinterpret_cast<float*>(c.out[f]);
    if (p) p[c.col] = v;
}
__device__ __forceinline__ void cz_iout(const CzCtx& c, int f, int v) {
    int* p = reinterpret_cast<int*>(c.iout[f]);
    if (p) p[c.col] = v;
}

// camzm_driver (:46-760) for one column, a step on which it computes.
__device__ int cz_driver(const CzCtx& c, Ws& ws, int nk, float dt, float dx,
                         float cudt, int itimestep, int blpbl, int sfclay,
                         const int* kpbl) {
    const int pver = nk, pverp = nk + 1, kte = nk;
    // zm_convi's module constants (:171-185, :228)
    CzK K;
    K.tfreez = R8(UW_TMELT);
    K.eps1 = R8(UW_EPSILO);
    K.rl = R8(UW_LATVAP);
    K.cpres = R8(UW_CPAIR);
    K.rgrav = R8(1.0) / R8(UW_GRAVIT);
    K.rgas = R8(UW_RAIR);
    K.grav = R8(UW_GRAVIT);
    K.cp = K.cpres;
    {
        const R8 deltax = cam_widen(dx);
        K.tau = uw_max(R8(600.0), R8(3600.0) * uw_min(R8(1.0), deltax / R8(275000.0)));
    }
    R8 cudts, hcudts;
    if (cudt == 0.f) {
        cudts = cam_widen(dt);
        hcudts = cudts * R8(0.5);
    } else {
        cudts = cam_widen(cudt) * R8(60.0);
        hcudts = cudts * R8(0.5);
    }

    V t8 = ws.r8(pver), qh8 = ws.r8(pver), pdel8 = ws.r8(pver), pmid8 = ws.r8(pver);
    V ql8prm = ws.r8(pver), qi8 = ws.r8(pver), zm8 = ws.r8(pver), cld8 = ws.r8(pver);
    V pint8 = ws.r8(pverp), zi8 = ws.r8(pverp);
    CzConvr cv;
    cv.qtnd = ws.r8(pver); cv.heat = ws.r8(pver); cv.mcon = ws.r8(pverp);
    cv.dlf = ws.r8(pver); cv.pflx = ws.r8(pverp); cv.cme = ws.r8(pver);
    cv.zdu = ws.r8(pver); cv.rprd = ws.r8(pver); cv.mu = ws.r8(pver);
    cv.md = ws.r8(pver); cv.du = ws.r8(pver); cv.eu = ws.r8(pver);
    cv.ed = ws.r8(pver); cv.dp = ws.r8(pver); cv.ql = ws.r8(pver);
    V tend_snwprd = ws.r8(pver), tend_snwevmlt = ws.r8(pver);
    V ntprprd = ws.r8(pver), ntsnprd = ws.r8(pver);
    V flxprec = ws.r8(pverp), flxsnow = ws.r8(pverp);
    V winds = ws.r8(2 * pver), wind_tends = ws.r8(2 * pver);
    V pguall = ws.r8(2 * pver), pgdall = ws.r8(2 * pver);
    V icwu = ws.r8(2 * pver), icwd = ws.r8(2 * pver);
    V cloud2 = ws.r8(pver), cloud3 = ws.r8(pver);
    V cloudtnd2 = ws.r8(pver), cloudtnd3 = ws.r8(pver);
    if (*ws.err) return 0;   // the pool overflowed: the launcher refuses the step

    // :418-448 flip the layer middles (driver zeroes mu/md/du/eu/ed/dp)
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        float qv = cz_in(c, CZ_IN_QV, k);
        cld8(kflip) = cam_widen(cz_in(c, CZ_IN_CLDFRA, k));
        if (itimestep == 1) cld8(kflip) = R8(0.0);
        pdel8(kflip) = cam_widen(cam_fsub(cz_in(c, CZ_IN_P8W, k), cz_in(c, CZ_IN_P8W, k + 1)));
        pmid8(kflip) = cam_widen(cz_in(c, CZ_IN_P, k));
        qh8(kflip) = cam_widen(cam_fmax(cam_fdiv(qv, cam_fadd(1.f, qv)), 1e-30f));
        ql8prm(kflip) = cam_widen(cam_fdiv(cz_in(c, CZ_IN_QC, k), cam_fadd(1.f, qv)));
        qi8(kflip) = cam_widen(cam_fdiv(cz_in(c, CZ_IN_QI, k), cam_fadd(1.f, qv)));
        t8(kflip) = cam_widen(cz_in(c, CZ_IN_T, k));
        zm8(kflip) = cam_widen(cam_fsub(cz_in(c, CZ_IN_Z, k), cz_in2(c, CZ_IN_HT)));
        cv.dp(kflip) = R8(0.0); cv.du(kflip) = R8(0.0); cv.ed(kflip) = R8(0.0);
        cv.eu(kflip) = R8(0.0); cv.md(kflip) = R8(0.0); cv.mu(kflip) = R8(0.0);
    }
    for (int k = 1; k <= kte + 1; ++k) {
        const int kflip = kte - k + 2;
        pint8(kflip) = cam_widen(cz_in(c, CZ_IN_P8W, k));
        zi8(kflip) = cam_widen(cam_fsub(cz_in(c, CZ_IN_Z_AT_W, k), cz_in2(c, CZ_IN_HT)));
    }
    // :459-463, stock WRF's XLAND test.
    const float xland = cz_in2(c, CZ_IN_XLAND);
    const R8 landfrac = (xland == 2.f) ? R8(1.0) : R8(0.0);
    const float ht = cz_in2(c, CZ_IN_HT);
    const float pblh = cz_in2(c, CZ_IN_PBLH);
    cv.pblh = cam_widen(pblh);
    cv.geos = cam_widen(ht) * R8(UW_GRAVIT);
    {
        const float* tke = reinterpret_cast<const float*>(c.in[CZ_IN_TKE_PBL]) + c.col;
        int st = cz_get_tpert(blpbl, sfclay, dx, cz_in2(c, CZ_IN_MAVAIL), kpbl[c.col], pblh,
                              cz_in(c, CZ_IN_DZ8W, 1), cz_in2(c, CZ_IN_PSFC),
                              cz_in(c, CZ_IN_QV, 1), cz_in(c, CZ_IN_T, 1),
                              cz_in(c, CZ_IN_TH, 1), cz_in2(c, CZ_IN_TSK), tke,
                              c.ncols, kte, cz_in2(c, CZ_IN_UST), cz_in(c, CZ_IN_U, 1),
                              cz_in(c, CZ_IN_V, 1), cz_in2(c, CZ_IN_HFX),
                              cz_in2(c, CZ_IN_QFX), cz_in2(c, CZ_IN_TPERT), cv.tpert);
        if (st) return st;
    }
    cv.t = t8; cv.qh = qh8; cv.pap = pmid8; cv.paph = pint8; cv.dpp = pdel8;
    cv.zm = zm8; cv.zi = zi8; cv.landfrac = landfrac; cv.delt = hcudts;
    {
        int st = cz_zm_convr(K, ws, pver, cv);
        if (st) return st;
    }
    const bool deep = cv.lengath == 1;
    // :492-498
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        cv.dlf(kflip) = uw_max(uw_min(R8(1.e-6), cv.dlf(kflip)), R8(0.0));
        cz_out(c, CZ_OUT_DLF, k, cam_narrow(cv.dlf(kflip)));
    }
    cz_out2(c, CZ_OUT_CAPE, cam_narrow(cv.cape));
    cz_out2(c, CZ_OUT_RLIQ, cam_narrow(cv.rliq));
    for (int k = 1; k <= pverp; ++k) cv.mcon(k) = cv.mcon(k) * R8(100.0) / R8(UW_GRAVIT);
    // :405-408 zero, then :505-512 the gathered mass fluxes
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        float mu_out = 0.f, md_out = 0.f;
        if (deep) {
            mu_out = cam_narrow(cv.mu(kflip) * R8(100.0) / R8(UW_GRAVIT));
            md_out = cam_narrow(cv.md(kflip) * R8(100.0) / R8(UW_GRAVIT));
        }
        cz_out(c, CZ_OUT_ZMMU, k, mu_out);
        cz_out(c, CZ_OUT_ZMMD, k, md_out);
    }
    // :514-549
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        const float zmdt = cam_narrow(cv.heat(kflip) / R8(UW_CPAIR));
        const float zmdq = cam_narrow(cv.qtnd(kflip));
        const float pi = cz_in(c, CZ_IN_PI, k);
        const float qv = cz_in(c, CZ_IN_QV, k);
        cz_out(c, CZ_OUT_ZMDT, k, zmdt);
        cz_out(c, CZ_OUT_ZMDQ, k, zmdq);
        // rthcuten/rqvcuten are carried in their float32 output words; the
        // next three updates read them back from there (WRF's own arrays).
        cz_out(c, CZ_OUT_RTHCUTEN, k, cam_fdiv(zmdt, pi));
        cz_out(c, CZ_OUT_RQVCUTEN, k, cam_narrow(cam_widen(zmdq) * uw_sq(R8(1.0) + cam_widen(qv))));
        t8(kflip) = t8(kflip) + cam_widen(zmdt) * cudts;
        qh8(kflip) = qh8(kflip) + cam_widen(zmdq) * cudts;
    }
    {
        float pconvt = cz_in(c, CZ_IN_P8W, 1), pconvb = cz_in(c, CZ_IN_P8W, 1);
        if (deep && cv.maxg > cv.jt) {
            pconvt = cam_narrow(pmid8(cv.jt));
            pconvb = cam_narrow(pmid8(cv.maxg));
        }
        cz_out2(c, CZ_OUT_PCONVT, pconvt);
        cz_out2(c, CZ_OUT_PCONVB, pconvb);
        cz_out2(c, CZ_OUT_CUTOP, cam_narrow(cv.jctop));
        cz_out2(c, CZ_OUT_CUBOT, cam_narrow(cv.jcbot));
    }
    // :555-561 zm_conv_evap with fresh tendencies
    for (int k = 1; k <= pver; ++k) {
        cv.qtnd(k) = R8(0.0);
        cv.heat(k) = R8(0.0);
    }
    R8 prec = cv.prec, snow;
    cz_zm_conv_evap(ws, pver, t8, pmid8, pdel8, qh8, cv.heat, tend_snwprd,
                    tend_snwevmlt, cv.qtnd, cv.rprd, cld8, cudts, prec, snow,
                    ntprprd, ntsnprd, flxprec, flxsnow);
    // :565-610
    const float pratec = cam_narrow(prec * R8(1000.0));
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        const float evaptzm = cam_narrow(cv.heat(kflip) / R8(UW_CPAIR));
        const float evapqzm = cam_narrow(cv.qtnd(kflip));
        cz_out(c, CZ_OUT_EVAPTZM, k, evaptzm);
        cz_out(c, CZ_OUT_FZSNTZM, k, cam_narrow(tend_snwprd(kflip) / R8(UW_CPAIR)));
        cz_out(c, CZ_OUT_EVSNTZM, k, cam_narrow(tend_snwevmlt(kflip) / R8(UW_CPAIR)));
        cz_out(c, CZ_OUT_EVAPQZM, k, evapqzm);
        cz_out(c, CZ_OUT_ZMFLXPRC, k, cam_narrow(flxprec(kflip)));
        cz_out(c, CZ_OUT_ZMFLXSNW, k, cam_narrow(flxsnow(kflip)));
        cz_out(c, CZ_OUT_ZMNTPRPD, k, cam_narrow(ntprprd(kflip)));
        cz_out(c, CZ_OUT_ZMNTSNPD, k, cam_narrow(ntsnprd(kflip)));
        cz_out(c, CZ_OUT_ZMEIHEAT, k, cam_narrow(cv.heat(kflip)));
        const float pi = cz_in(c, CZ_IN_PI, k);
        const float qv = cz_in(c, CZ_IN_QV, k);
        const float onepq = cam_fadd(1.f, qv);
        float* rth = reinterpret_cast<float*>(c.out[CZ_OUT_RTHCUTEN]) + (long long)(k - 1) * c.ncols + c.col;
        float* rqv = reinterpret_cast<float*>(c.out[CZ_OUT_RQVCUTEN]) + (long long)(k - 1) * c.ncols + c.col;
        *rth = cam_fadd(*rth, cam_fdiv(evaptzm, pi));
        *rqv = cam_fadd(*rqv, cam_fmul(evapqzm, cam_fmul(onepq, onepq)));
        t8(kflip) = t8(kflip) + cam_widen(evaptzm) * cudts;
        qh8(kflip) = qh8(kflip) + cam_widen(evapqzm) * cudts;
    }
    cz_out2(c, CZ_OUT_PRECCDZM, cam_narrow(prec));
    cz_out2(c, CZ_OUT_PRECZ, cam_narrow(prec));
    cz_out2(c, CZ_OUT_PRATEC, pratec);
    cz_out2(c, CZ_OUT_RAINCV, cam_fmul(pratec, dt));
    for (int k = 1; k <= kte + 1; ++k) {
        const int kflip = kte - k + 2;
        cz_out(c, CZ_OUT_CMFMC, k, cam_narrow(cv.mcon(kflip)));
        cz_out(c, CZ_OUT_CMFMCDZM, k, cam_narrow(cv.mcon(kflip)));
    }
    // :612-646 momentum transport
    for (int k = 1; k <= pver; ++k) cv.heat(k) = R8(0.0);
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        winds(k) = cam_widen(cz_in(c, CZ_IN_U, kflip));
        winds(pver + k) = cam_widen(cz_in(c, CZ_IN_V, kflip));
    }
    cz_momtran(ws, pver, cv.lengath, winds, cv.mu, cv.md, cv.du, cv.eu, cv.ed,
               cv.dp, cv.jt, cv.maxg, hcudts, wind_tends, pguall, pgdall, icwu,
               icwd, cv.heat);
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        const float pi = cz_in(c, CZ_IN_PI, k);
        cz_out(c, CZ_OUT_RUCUTEN, k, cam_narrow(wind_tends(kflip)));
        cz_out(c, CZ_OUT_RVCUTEN, k, cam_narrow(wind_tends(pver + kflip)));
        float* rth = reinterpret_cast<float*>(c.out[CZ_OUT_RTHCUTEN]) + (long long)(k - 1) * c.ncols + c.col;
        *rth = cam_narrow(cam_widen(*rth) + cv.heat(kflip) / R8(UW_CPAIR) / cam_widen(pi));
        t8(kflip) = t8(kflip) + cv.heat(kflip) / R8(UW_CPAIR) * cudts;
    }
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        cz_out(c, CZ_OUT_ZMMTU, k, cam_narrow(wind_tends(kflip)));
        cz_out(c, CZ_OUT_ZMMTV, k, cam_narrow(wind_tends(pver + kflip)));
        cz_out(c, CZ_OUT_ZMUPGU, k, cam_narrow(pguall(kflip)));
        cz_out(c, CZ_OUT_ZMUPGD, k, cam_narrow(pgdall(kflip)));
        cz_out(c, CZ_OUT_ZMVPGU, k, cam_narrow(pguall(pver + kflip)));
        cz_out(c, CZ_OUT_ZMVPGD, k, cam_narrow(pgdall(pver + kflip)));
        cz_out(c, CZ_OUT_ZMICUU, k, cam_narrow(icwu(kflip)));
        cz_out(c, CZ_OUT_ZMICUD, k, cam_narrow(icwd(kflip)));
        cz_out(c, CZ_OUT_ZMICVU, k, cam_narrow(icwu(pver + kflip)));
        cz_out(c, CZ_OUT_ZMICVD, k, cam_narrow(icwd(pver + kflip)));
    }
    // :666-696 convective transport of cloud liquid and ice (moist mixing
    // ratios, 'wet', fracis = 1); water vapour (l_qt(1)) is not transported
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        cloud2(kflip) = ql8prm(kflip);
        cloud3(kflip) = qi8(kflip);
    }
    cz_convtran_one(ws, pver, cv.lengath, cloud2, cv.mu, cv.md, cv.du, cv.eu,
                    cv.ed, cv.dp, cv.jt, cv.maxg, cloudtnd2);
    cz_convtran_one(ws, pver, cv.lengath, cloud3, cv.mu, cv.md, cv.du, cv.eu,
                    cv.ed, cv.dp, cv.jt, cv.maxg, cloudtnd3);
    // :700-756 detrainment split and the remaining outputs
    for (int k = 1; k <= kte; ++k) {
        const int kflip = kte - k + 1;
        R8 dum1;
        if (t8(kflip) > R8(268.15)) dum1 = R8(0.0);
        else if (t8(kflip) < R8(238.15)) dum1 = R8(1.0);
        else dum1 = (R8(268.15) - t8(kflip)) / R8(30.0);
        const R8 dlf = cv.dlf(kflip);
        const R8 qcten_det = dlf * (R8(1.0) - dum1);
        const R8 qiten_det = dlf * dum1;
        const R8 qcnten_det = R8(3.0) * (dlf * (R8(1.0) - dum1)) / R8(CZ_QCN_DEN);
        const R8 qinten_det = R8(3.0) * (dlf * dum1) / R8(CZ_QIN_DEN);
        const R8 qsten_det = dlf * dum1 * R8(UW_LATICE);
        const float pi = cz_in(c, CZ_IN_PI, k);
        const R8 onepq = cam_widen(cam_fadd(1.f, cz_in(c, CZ_IN_QV, k)));
        cz_out(c, CZ_OUT_EVAPCDP3D, k, cam_narrow(cv.qtnd(kflip)));
        cz_out(c, CZ_OUT_ICWMRDP3D, k, cam_narrow(cv.ql(kflip)));
        cz_out(c, CZ_OUT_RPRDDP3D, k, cam_narrow(cv.rprd(kflip)));
        cz_out(c, CZ_OUT_ZMDICE, k, cam_narrow(cloudtnd3(kflip) + qiten_det));
        cz_out(c, CZ_OUT_ZMDLIQ, k, cam_narrow(cloudtnd2(kflip) + qcten_det));
        float* rth = reinterpret_cast<float*>(c.out[CZ_OUT_RTHCUTEN]) + (long long)(k - 1) * c.ncols + c.col;
        *rth = cam_narrow(cam_widen(*rth) + qsten_det / R8(UW_CPAIR) / cam_widen(pi));
        cz_out(c, CZ_OUT_RQCCUTEN, k, cam_narrow((cloudtnd2(kflip) + qcten_det) * onepq));
        cz_out(c, CZ_OUT_RQICUTEN, k, cam_narrow((cloudtnd3(kflip) + qiten_det) * onepq));
        cz_out(c, CZ_OUT_RQCNCUTEN, k, cam_narrow(qcnten_det * onepq));
        cz_out(c, CZ_OUT_RQINCUTEN, k, cam_narrow(qinten_det * onepq));
        cz_out(c, CZ_OUT_DP3D, k, cam_narrow(cv.dp(kflip)));
        cz_out(c, CZ_OUT_DU3D, k, cam_narrow(cv.du(kflip)));
        cz_out(c, CZ_OUT_ED3D, k, cam_narrow(cv.ed(kflip)));
        cz_out(c, CZ_OUT_EU3D, k, cam_narrow(cv.eu(kflip)));
        cz_out(c, CZ_OUT_MD3D, k, cam_narrow(cv.md(kflip)));
        cz_out(c, CZ_OUT_MU3D, k, cam_narrow(cv.mu(kflip)));
    }
    cz_out2(c, CZ_OUT_DSUBCLD2D, cam_narrow(cv.dsubcld));
    cz_iout(c, CZ_IOUT_IDEEP2D, deep ? 1 : -1);
    cz_iout(c, CZ_IOUT_JT2D, deep ? cv.jt : -1);
    // zm_convr initializes MAXG=1 before its LENGATH=0 return (:632).
    // IDEEP and JT have no corresponding inactive-column assignment.
    cz_iout(c, CZ_IOUT_MAXG2D, cv.maxg);
    cz_iout(c, CZ_IOUT_LENGATH2D, cv.lengath);
    return 0;
}

// One thread per column of a chunk [col0, col0 + n) of the ncols columns.
// Field f's word for WRF level k (1-based) of column g is at
// in[f][(k-1)*ncols + g]; the workspace pool holds chunk columns.
// RTHCUTEN and RQVCUTEN are accumulated in their output words, so those two
// outputs are always requested (the launcher never passes them null).
extern "C" __global__ void camzm_columns(
    const unsigned long long* in, const int* kpbl,
    const unsigned long long* out, const unsigned long long* iout,
    long long ncols, long long col0, int n, int chunk, int nz,
    float dt, float dx, float cudt, int itimestep, int blpbl, int sfclay,
    double* r8pool, int r8cap, int* err) {
    const int lcol = blockIdx.x * blockDim.x + threadIdx.x;
    if (lcol >= n) return;
    CzCtx c{in, out, iout, ncols, col0 + lcol};
    Ws ws{r8pool + lcol, nullptr, chunk, 0, r8cap, 0, 0, err};
    int st = cz_driver(c, ws, nz, dt, dx, cudt, itimestep, blpbl, sfclay, kpbl);
    if (st) {
        cz_flag(err, st);
        atomicCAS(err + 1, -1, (int)(col0 + lcol));
    }
}
