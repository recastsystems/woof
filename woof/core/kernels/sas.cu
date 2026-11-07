// gpuwm/core/kernels/sas.cu
//
// The SAS-family cumulus column core.  First member: KIAPS SAS
// (cu_physics = 14), WRF v4.6.1 phys/module_cu_ksas.F -- the cu_ksas driver
// body (module_cu_ksas.F:162-295), the nsas2d deep column (:303-2252) and the
// inlined fpvs (:2254-2294), transcribed in WRF's operation order.  WRF is
// public domain; this is a clean transcription.
//
// SHAPE.  One thread per column.  The scheme's 2-D (its:ite, kts:kte)
// automatics become one workspace slot each, level-major with the column
// fastest ([slot][k][column]), so a warp's loads at one level coalesce the
// way the house kernels' (nz, ncol) fields do.  Fortran's 1-based level
// index is kept verbatim (K(k) maps level k to row k-1) so every loop bound
// below reads exactly as module_cu_ksas.F prints it.
//
// THE ROW-WIDE RETURNS ARE PER-COLUMN EXITS.  nsas2d returns from the whole
// j-row when no column is still convecting (`if(totflg) return`).  A column
// that has lost cnvflg is never written again by anything that reaches an
// output (every later write is guarded by cnvflg), so stopping it at the
// point it drops out is the same answer as carrying it to the row's end.
//
// ARITHMETIC. Strict numerics preserves WRF's operation order, uses the
// existing native-libm helpers, disables contraction and preserves subnormals.
// GPUWM_WRF_GLIBC selects the referee's atanf generation. Default fast numerics
// uses CUDA math and permits contraction. Integer powers keep WRF's multiply
// chains in both modes. The replay compares raw words, including signed zero.

#define SAS_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define SAS_MIN(a, b) (((a) < (b)) ? (a) : (b))

// ---------------------------------------------------------------- fpvs
// module_cu_ksas.F:2254-2294, called everywhere with ice = 1.
__device__ float sas_fpvs(float t, float rd, float rv, float cvap, float cliq,
                          float cice, float hvap, float hsub, float psat,
                          float t0c)
{
    (void)rd;
    float ttp = t0c + 0.01f;
    float dldt = cvap - cliq;
    float xa = -dldt / rv;
    float xb = xa + hvap / (rv * ttp);
    float dldti = cvap - cice;
    float xai = -dldti / rv;
    float xbi = xai + hsub / (rv * ttp);
    float tr = ttp / t;
    float f;
    if (t < ttp) {
        f = psat * gfk_pow(tr, xai) * gfk_exp(xbi * (1.f - tr));
    } else {
        f = psat * gfk_pow(tr, xa) * gfk_exp(xb * (1.f - tr));
    }
    if (t < 180.f) {
        tr = ttp / 180.f;
        if (t < ttp) {
            f = psat * gfk_pow(tr, xai) * gfk_exp(xbi * (1.f - tr));
        } else {
            f = psat * gfk_pow(tr, xa) * gfk_exp(xb * (1.f - tr));
        }
    }
    if (t >= 330.f) {
        tr = ttp / 330.f;   // ttp/330: an INTEGER literal promoted to REAL
        if (t < ttp) {
            f = psat * gfk_pow(tr, xai) * gfk_exp(xbi * (1.f - tr));
        } else {
            f = psat * gfk_pow(tr, xa) * gfk_exp(xb * (1.f - tr));
        }
    }
    return f;
}

// Workspace slots, one (nz + 1)-level row each.
enum {
    S_DEL = 0, S_ZL, S_ZI, S_DOT, S_T1, S_Q1, S_U1, S_V1, S_QC2, S_QI2,
    S_P, S_UO, S_VO, S_TO, S_QO, S_HCKO, S_QCKO, S_ETA, S_ETAD, S_QRCDO,
    S_PWO, S_PWDO, S_C0T, S_WU2, S_BUO, S_DRAG, S_QESO, S_HEO, S_HESO,
    S_QRCD, S_DELLAH, S_DELLAQ, S_XLAMB, S_QCIRS, S_DELLAL, S_FRH, S_UCKO,
    S_VCKO, S_UCDO, S_VCDO, S_DELLAU, S_DELLAV, S_HCDO, S_QCDO, S_XHCD,
    S_XQCD, S_PGCON, S_DBYO, S_ZII,
    SAS_KSAS_SLOTS
};

// The KSAS tunable constants (module_cu_ksas.F:421-434), as written.
#define KSAS_BETAL 0.05f
#define KSAS_BETAS 0.05f
#define KSAS_C0 0.002f
#define KSAS_C1 0.002f
#define KSAS_XLAMDD 1.0e-4f
#define KSAS_XLAMDE 1.0e-4f
#define KSAS_CLAM 0.1f
#define KSAS_CXLAMU 1.0e-3f
#define KSAS_AAFAC 0.1f
#define KSAS_DTHK 25.f
#define KSAS_CINPCRMX 240.f
#define KSAS_CINPCRMN 120.f
#define KSAS_CINACRMX (-120.f)
#define KSAS_BET1 1.875f
#define KSAS_CD1 0.506f
#define KSAS_F1 2.0f
#define KSAS_GAM1 0.5f
#define KSAS_TFAC 1.0f
#define KSAS_DX1KM 1000.f
#define KSAS_DX5KM 5000.f
#define KSAS_EDTMAXL 0.3f
#define KSAS_EDTMAXS 0.3f
#define KSAS_EVFACTS 0.3f
#define KSAS_EVFACTL 0.3f
#define KSAS_TF 233.16f
#define KSAS_TCR 273.16f

// ------------------------------------------------------------------ cu_ksas
// One column per thread over a chunk of `ncw` columns starting at `col0` of
// fields whose column stride is `ncs`.  Level fields are [nz][ncs], interface
// fields [nz + 1][ncs], surface fields [ncs].  `ncloud` is the driver's
// mp_physics mapping (module_cu_ksas.F:145-151), resolved on the host.
extern "C" __global__ void sas_ksas_column(
    // driver inputs
    const float* __restrict__ t3d, const float* __restrict__ qv3d,
    const float* __restrict__ qc3d, const float* __restrict__ qi3d,
    const float* __restrict__ p3d, const float* __restrict__ pi3d,
    const float* __restrict__ rho3d, const float* __restrict__ dz8w,
    const float* __restrict__ u3d, const float* __restrict__ v3d,
    const float* __restrict__ w, const float* __restrict__ znu,
    const float* __restrict__ xland, const float* __restrict__ hpbl,
    const float* __restrict__ hpbl_hold,
    // outputs
    float* __restrict__ rthcuten, float* __restrict__ rqvcuten,
    float* __restrict__ rqccuten, float* __restrict__ rqicuten,
    float* __restrict__ rucuten, float* __restrict__ rvcuten,
    float* __restrict__ raincv, float* __restrict__ pratec,
    float* __restrict__ hbot, float* __restrict__ htop,
    // workspace [SAS_KSAS_SLOTS][nz + 1][ncw]
    float* __restrict__ ws,
    // scalars
    float dt, float dx, int stepcu, int ncloud, int f_qc, int f_qi,
    int nz, int col0, int ncw, int ncs,
    // WRF physical constants as the driver passes them
    float cp, float cliq, float cpv, float g, float xlv, float r_d, float r_v,
    float ep_1, float ep_2, float cice, float xls, float psat)
{
    const int ic = blockIdx.x * blockDim.x + threadIdx.x;
    if (ic >= ncw) return;
    const int col = col0 + ic;
    const int lev = nz + 1;

#define W_(s, k) ws[((size_t)(s) * (size_t)lev + (size_t)((k) - 1)) * (size_t)ncw + (size_t)ic]
#define F_(a, k) a[(size_t)((k) - 1) * (size_t)ncs + (size_t)col]
#define del(k) W_(S_DEL, k)
#define zl(k) W_(S_ZL, k)
#define zi(k) W_(S_ZI, k)
#define zii(k) W_(S_ZII, k)
#define dot(k) W_(S_DOT, k)
#define t1(k) W_(S_T1, k)
#define q1(k) W_(S_Q1, k)
#define u1(k) W_(S_U1, k)
#define v1(k) W_(S_V1, k)
#define qc2(k) W_(S_QC2, k)
#define qi2(k) W_(S_QI2, k)
#define p(k) W_(S_P, k)
#define uo(k) W_(S_UO, k)
#define vo(k) W_(S_VO, k)
#define to(k) W_(S_TO, k)
#define qo(k) W_(S_QO, k)
#define hcko(k) W_(S_HCKO, k)
#define qcko(k) W_(S_QCKO, k)
#define eta(k) W_(S_ETA, k)
#define etad(k) W_(S_ETAD, k)
#define qrcdo(k) W_(S_QRCDO, k)
#define pwo(k) W_(S_PWO, k)
#define pwdo(k) W_(S_PWDO, k)
#define c0t(k) W_(S_C0T, k)
#define wu2(k) W_(S_WU2, k)
#define buo(k) W_(S_BUO, k)
#define drag(k) W_(S_DRAG, k)
#define qeso(k) W_(S_QESO, k)
#define heo(k) W_(S_HEO, k)
#define heso(k) W_(S_HESO, k)
#define qrcd(k) W_(S_QRCD, k)
#define dellah(k) W_(S_DELLAH, k)
#define dellaq(k) W_(S_DELLAQ, k)
#define xlamb(k) W_(S_XLAMB, k)
#define qcirs(k) W_(S_QCIRS, k)
#define dellal(k) W_(S_DELLAL, k)
#define frh(k) W_(S_FRH, k)
#define ucko(k) W_(S_UCKO, k)
#define vcko(k) W_(S_VCKO, k)
#define ucdo(k) W_(S_UCDO, k)
#define vcdo(k) W_(S_VCDO, k)
#define dellau(k) W_(S_DELLAU, k)
#define dellav(k) W_(S_DELLAV, k)
#define hcdo(k) W_(S_HCDO, k)
#define qcdo(k) W_(S_QCDO, k)
#define xhcd(k) W_(S_XHCD, k)
#define xqcd(k) W_(S_XQCD, k)
#define pgcon(k) W_(S_PGCON, k)
#define dbyo(k) W_(S_DBYO, k)
#define FPVS(tt) sas_fpvs((tt), rd_, rv_, cvap_, cliq_, cice, xlv0, xls, psat, t0c_)

    const int kts = 1, kte = nz;
    int k;

    // ------------------------------------------------- cu_ksas (driver body)
    // :170-177 -- the znu-derived search tops.
    int kbmax = kte, kbm = kte, kmax = kte;
    for (k = kts; k <= kte; ++k) {
        if (znu[k - 1] > 0.45f) kbmax = k + 1;
        if (znu[k - 1] > 0.70f) kbm = k + 1;
        if (znu[k - 1] > 0.05f) kmax = k + 1;
    }
    const float delt = dt * (float)stepcu;
    const float rdelt = 1.f / delt;

    // :182-222
    for (k = kts; k <= kte; ++k) {
        const int kp = k + 1;
        dot(k) = -5.0e-4f * g * F_(rho3d, k) * (F_(w, k) + F_(w, kp));
    }
    zii(1) = 0.0f;
    for (k = kts; k <= kte; ++k) zii(k + 1) = zii(k) + F_(dz8w, k);
    for (k = kts; k <= kte; ++k) zl(k) = 0.5f * (zii(k) + zii(k + 1));
    for (k = kts; k <= kte; ++k) {
        const float prsll = F_(p3d, k) * 0.001f;
        del(k) = prsll * g / r_d * F_(dz8w, k) / F_(t3d, k);
        u1(k) = F_(u3d, k);
        v1(k) = F_(v3d, k);
        q1(k) = F_(qv3d, k);
        t1(k) = F_(t3d, k);
        qi2(k) = F_(qi3d, k);
        qc2(k) = F_(qc3d, k);
    }

    // ------------------------------------------------------------- nsas2d
    const float delx = dx;
    const float slimsk = xland[col];
    const float hpbl_i = hpbl[col];
    const float hpbl_hold_i = hpbl_hold[col];
    const float cp_ = cp, cliq_ = cliq, cvap_ = cpv, g_ = g, hvap_ = xlv;
    const float rd_ = r_d, rv_ = r_v, fv_ = ep_1;
    (void)ep_2;
    const float tcrf = 1.0f / (KSAS_TCR - KSAS_TF);

    // :563-585
    const float pi_ = 3.14159f;
    const float qmin_ = 1.0e-30f;
    const float t0c_ = 273.15f;
    const float xlv0 = hvap_;
    const float rcs = 1.f;
    const float el2orc = hvap_ * hvap_ / (rv_ * cp_);
    const float eps = rd_ / rv_;
    const float fact1 = (cvap_ - cliq_) / rv_;
    const float fact2 = hvap_ / rv_ - fact1 * t0c_;
    const int kts1 = kts + 1;
    const int kte1 = kte - 1;
    const float dt2 = delt;
    const float dtmin = SAS_MAX(dt2, 600.f);
    const float dtmax = SAS_MAX(dt2, 10800.f);
    const float mbdt = dt2;
    const float sigma_con = glibc_tanf(0.4f * pi_) / (KSAS_DX5KM - KSAS_DX1KM);
    float sigma = (1.f - 1.f / pi_ * (wrf_atan_f32(sigma_con * (delx - KSAS_DX5KM)) + pi_ / 2.f));
    if (delx < KSAS_DX5KM) {
        sigma = SAS_MIN(sigma - 0.01684f * delx / 1000.f + 0.0842f, 1.0f);
    }
    const float cinpcr = (KSAS_CINPCRMN + 0.5f * (KSAS_CINPCRMX - KSAS_CINPCRMN)) * (1.f - sigma);

    // :589-613
    float rain = 0.0f;
    int kbot = kte + 1, ktop = 0;
    bool cnvflg = true;
    float pdot = 0.0f, edto = 0.0f, edtx = 0.0f, xmbmax = 0.3f, aa2 = 0.0f;
    float qlko_ktcon = 0.0f, edt = 0.0f, cina = 0.0f, frh_sum = 0.0f;
    float cinpcri = 0.0f, apbl = 0.0f, dtpbl = 0.0f;
    int lmin = 1, jmin = 1;
    for (k = kts; k <= kte; ++k) pgcon(k) = 0.5f;

    // :618-620
    kmax = SAS_MIN(kmax, kte);
    const int kmax1 = kmax - 1;
    kbm = SAS_MIN(kbm, kte);

    // :624-643
    const float c1t = KSAS_C1 * sigma;
    for (k = kts; k <= kte; ++k) {
        c0t(k) = KSAS_C0;
        qcirs(k) = 0.0f;
        wu2(k) = 0.0f;
        buo(k) = 0.0f;
        drag(k) = 0.0f;
        pwo(k) = 0.0f;
        pwdo(k) = 0.0f;
        dellal(k) = 0.0f;
        hcko(k) = 0.0f;
        qcko(k) = 0.0f;
        hcdo(k) = 0.0f;
        qcdo(k) = 0.0f;
    }
    // :645-661
    for (k = kts; k <= kmax; ++k) {
        p(k) = F_(p3d, k) * 0.001f * 10.f;
        to(k) = t1(k);
        qo(k) = q1(k);
        dbyo(k) = 0.0f;
        frh(k) = 0.0f;
        ucko(k) = 0.0f;
        vcko(k) = 0.0f;
        ucdo(k) = 0.0f;
        vcdo(k) = 0.0f;
        uo(k) = u1(k) * rcs;
        vo(k) = v1(k) * rcs;
    }

    // :671-679
    for (k = kts; k <= kmax; ++k) {
        qeso(k) = 0.01f * FPVS(to(k));
        qeso(k) = eps * qeso(k) / (p(k) + (eps - 1.f) * qeso(k));
        qeso(k) = SAS_MAX(qeso(k), qmin_);
        qo(k) = SAS_MAX(qo(k), 1.e-10f);
    }
    // :683-688
    for (k = kts; k <= kmax; ++k) {
        heo(k) = g_ * zl(k) + cp_ * to(k) + hvap_ * qo(k);
        heso(k) = g_ * zl(k) + cp_ * to(k) + hvap_ * qeso(k);
    }
    // :693-705
    float hmax = heo(1);
    int kb = 1;
    for (k = kts1; k <= kbm; ++k) {
        if (heo(k) > hmax) { kb = k; hmax = heo(k); }
    }
    // :707-709
    if (qo(kb) < qmin_) cnvflg = false;
    // :711-741
    if (cnvflg) {
        for (k = kts; k <= kmax1; ++k) {
            float dz = .5f * (zl(k + 1) - zl(k));
            float dp = .5f * (p(k + 1) - p(k));
            float es = 0.01f * FPVS(to(k + 1));
            float pprime = p(k + 1) + (eps - 1.f) * es;
            float qs = eps * es / pprime;
            float dqsdp = -qs / pprime;
            float desdt = es * (fact1 / to(k + 1) + fact2 / (to(k + 1) * to(k + 1)));
            float dqsdt = qs * p(k + 1) * desdt / (es * pprime);
            float gamma = el2orc * qeso(k + 1) / (to(k + 1) * to(k + 1));
            float dtt = (g_ * dz + hvap_ * dqsdp * dp) / (cp_ * (1.f + gamma));
            float dq = dqsdt * dtt + dqsdp * dp;
            to(k) = to(k + 1) + dtt;
            qo(k) = qo(k + 1) + dq;
            float po = .5f * (p(k) + p(k + 1));
            qeso(k) = 0.01f * FPVS(to(k));
            qeso(k) = eps * qeso(k) / (po + (eps - 1.f) * qeso(k));
            qeso(k) = SAS_MAX(qeso(k), qmin_);
            qo(k) = SAS_MAX(qo(k), 1.e-10f);
            frh(k) = 1.f - SAS_MIN(qo(k) / qeso(k), 1.f);
            heo(k) = .5f * g_ * (zl(k) + zl(k + 1)) + cp_ * to(k) + hvap_ * qo(k);
            heso(k) = .5f * g_ * (zl(k) + zl(k + 1)) + cp_ * to(k) + hvap_ * qeso(k);
            uo(k) = .5f * (uo(k) + uo(k + 1));
            vo(k) = .5f * (vo(k) + vo(k + 1));
        }
    }

    // :745-772 level of free convection
    float hkbo = 0.f, qkbo = 0.f;
    if (cnvflg) { hkbo = heo(kb); qkbo = qo(kb); }
    bool flg = cnvflg;
    int kbcon = kmax;
    for (k = kts; k <= kbmax; ++k) {
        if (flg && k > kb) {
            float hsbar = heso(k);
            if (hkbo > hsbar) { flg = false; kbcon = k; }
        }
    }
    if (kbcon == kmax) cnvflg = false;
    // The row-wide returns, as per-column exits: `break` leaves this block.
    do {
    if (!cnvflg) break;

    // :780-796
    for (k = kts; k <= kmax1; ++k) {
        if (k >= kb && k <= kbcon) frh_sum = frh_sum + (1.f - frh(k));
    }
    {
        float tem1 = p(kb) - p(kbcon);
        cinpcri = cinpcr * frh_sum / (float)(kbcon - kb + 1);
        if (tem1 > cinpcri) cnvflg = false;
    }
    if (!cnvflg) break;

    // :804-834 entrainment
    for (k = kts1; k <= kte; ++k) zi(k) = 0.5f * (zl(k - 1) + zl(k));
    for (k = kts; k <= kte1; ++k) xlamb(k) = KSAS_CLAM / zi(k + 1);
    for (k = kts1; k <= kmax1; ++k) {
        if (k > kbcon) xlamb(k) = xlamb(kbcon);
    }
    float xlamud;
    xlamud = xlamb(kbcon);
    // :839-860 -- fent1/fent2 are read only by the next loop and depend only
    // on qeso, so they are formed where they are read.  At k = kbcon they
    // keep their initial 1.0 (:651-652).
    for (k = kts1; k <= kmax1; ++k) {
        if (k >= kbcon) {
            float fent1 = 1.0f, fent2 = 1.0f;
            if (k > kbcon) {
                float tem = qeso(k) / qeso(kbcon);
                fent1 = tem * tem;
                fent2 = tem * tem * tem;
            }
            float tem = KSAS_CXLAMU * frh(k) * fent2;
            xlamb(k) = xlamb(k) * fent1 + tem;
        }
    }

    // :864-896 updraft mass flux
    for (k = kts; k <= kte; ++k) eta(k) = 1.f;
    for (k = kbmax; k >= kts1; --k) {
        if (k < kbcon && k >= kb) {
            float dz = zi(k + 2) - zi(k + 1);
            float ptem = 0.5f * (xlamb(k) + xlamb(k + 1)) - xlamud;
            eta(k) = eta(k + 1) / (1.f + ptem * dz);
        }
    }
    for (k = kts1; k <= kmax1; ++k) {
        if (k > kbcon) {
            float dz = zi(k + 1) - zi(k);
            float ptem = 0.5f * (xlamb(k) + xlamb(k - 1)) - xlamud;
            eta(k) = eta(k - 1) * (1.f + ptem * dz);
        }
    }
    {
        float dz = zi(3) - zi(2);
        float ptem = 0.5f * (xlamb(1) + xlamb(2)) - xlamud;
        eta(1) = eta(2) / (1.f + ptem * dz);
    }

    // :900-925 updraft properties
    float pwavo;
    hcko(kb) = hkbo;
    qcko(kb) = qkbo;
    ucko(kb) = uo(kb);
    vcko(kb) = vo(kb);
    pwavo = 0.f;
    for (k = kts1; k <= kmax1; ++k) {
        if (k > kb) {
            float dz = zi(k + 1) - zi(k);
            float tem = 0.5f * (xlamb(k) + xlamb(k - 1)) * dz;
            float tem1 = 0.5f * xlamud * dz;
            float factor = 1.f + tem - tem1;
            hcko(k) = ((1.f - tem1) * hcko(k - 1) + tem * 0.5f * (heo(k) + heo(k - 1))) / factor;
            dbyo(k) = hcko(k) - heso(k);
        }
    }

    // :930-963
    flg = cnvflg;
    int kbcon1;
    kbcon1 = kmax;
    for (k = kts1; k <= kmax; ++k) {
        if (flg && k >= kbcon && dbyo(k) > 0.f) { kbcon1 = k; flg = false; }
    }
    if (kbcon1 == kmax) cnvflg = false;
    if (cnvflg) {
        float tem = p(kbcon) - p(kbcon1);
        if (tem > KSAS_DTHK) cnvflg = false;
    }
    if (!cnvflg) break;

    // :967-993 convective inhibition
    for (k = kts1; k <= kmax1; ++k) {
        if (k > kb && k < kbcon1) {
            float dz1 = (zi(k + 1) - zi(k));
            float gamma = el2orc * qeso(k) / (to(k) * to(k));
            float rfact = 1.f + fv_ * cp_ * gamma * to(k) / hvap_;
            cina = cina + dz1 * (g_ / (cp_ * to(k))) * dbyo(k) / (1.f + gamma) * rfact;
            cina = cina + dz1 * g_ * fv_ * SAS_MAX(0.f, (qeso(k) - qo(k)));
        }
    }
    if (cina < KSAS_CINACRMX) cnvflg = false;
    if (!cnvflg) break;

    // :997-1024 cloud top
    flg = cnvflg;
    int ktcon;
    ktcon = 1;
    for (k = kts1; k <= kmax1; ++k) {
        if (dbyo(k) < 0.f && flg && k > kbcon1) { ktcon = k; flg = false; }
    }
    if ((p(kbcon) - p(ktcon)) < 150.f) cnvflg = false;
    if (!cnvflg) break;

    // :1028-1054 downdraft source level
    float hmin;
    hmin = heo(kbcon1);
    lmin = kbmax;
    jmin = kbmax;
    for (k = kts1; k <= kbmax; ++k) {
        if (k > kbcon1 && heo(k) < hmin) { lmin = k + 1; hmin = heo(k); }
    }
    jmin = SAS_MIN(lmin, ktcon - 1);
    jmin = SAS_MAX(jmin, kbcon1 + 1);
    if (jmin >= ktcon) cnvflg = false;
    if (jmin <= kbcon) cnvflg = false;

    // :1058-1064 -- reached by the columns this row still carries, so a
    // column that lost cnvflg just above stops here rather than at the next
    // row-wide return; nothing below writes an output for it.
    if (!cnvflg) break;
    {
        int kk = kbcon;
        float dp = 1000.f * del(kk);
        xmbmax = dp / (g_ * dt2);
    }

    // :1068-1129 cloud moisture, precipitation, buoyancy
    for (k = kts1; k <= kmax; ++k) {
        if (k > kb) {
            float alpha1 = SAS_MIN((-0.7f * gfk_log(100.f) + 24.f) * 0.0001f, KSAS_C0);
            float beta1 = 0.07f;
            float c0fac;
            if (to(k) > t0c_) {
                c0fac = alpha1;
            } else {
                c0fac = alpha1 * gfk_exp(beta1 * (to(k) - t0c_));
            }
            c0fac = SAS_MAX(0.0f, c0fac);
            c0t(k) = c0fac;
        }
        if (k > kb && k < ktcon) {
            float dz1 = (zi(k + 1) - zi(k));
            float gamma = el2orc * qeso(k) / (to(k) * to(k));
            float qrch = qeso(k) + gamma * dbyo(k) / (hvap_ * (1.f + gamma));
            float tem = 0.5f * (xlamb(k) + xlamb(k - 1)) * dz1;
            float tem1 = 0.5f * xlamud * dz1;
            float factor = 1.f + tem - tem1;
            qcko(k) = ((1.f - tem1) * qcko(k - 1) + tem * 0.5f * (qo(k) + qo(k - 1))) / factor;
            qcirs(k) = eta(k) * qcko(k) - eta(k) * qrch;
            if (qcirs(k) > 0.f && k >= kbcon) {
                float etah = .5f * (eta(k) + eta(k - 1));
                float qlk;
                if (ncloud > 0 && k > jmin) {
                    float dp = 1000.f * del(k);
                    float ptem = c0t(k) + c1t;
                    qlk = qcirs(k) / (eta(k) + etah * ptem * dz1);
                    dellal(k) = etah * c1t * dz1 * qlk * g_ / dp;
                } else {
                    qlk = qcirs(k) / (eta(k) + etah * c0t(k) * dz1);
                }
                pwo(k) = etah * c0t(k) * dz1 * qlk;
                float qc = qlk + qrch;
                qcko(k) = qc;
                pwavo = pwavo + pwo(k);
                buo(k) = buo(k) - g_ * qlk;
                if (k >= kbcon) {
                    float rfact = 1.f + fv_ * cp_ * gamma * to(k) / hvap_;
                    buo(k) = buo(k) + (g_ / (cp_ * to(k))) * dbyo(k) / (1.f + gamma) * rfact;
                    buo(k) = buo(k) + g_ * fv_ * SAS_MAX(0.f, (qeso(k) - qo(k)));
                    drag(k) = SAS_MAX(xlamb(k), xlamud);
                }
            }
        }
    }

    // :1133-1158 cloud work function
    float aa1;
    aa1 = 0.f;
    for (k = kts1; k <= kmax; ++k) {
        if (k >= kbcon && k < ktcon) {
            float dz1 = zl(k + 1) - zl(k);
            aa1 = aa1 + buo(k) * dz1;
        }
    }
    if (aa1 <= 0.f) cnvflg = false;
    if (!cnvflg) break;

    // :1164-1191 overshooting
    aa2 = KSAS_AAFAC * aa1;
    flg = cnvflg;
    int ktcon1;
    ktcon1 = kmax1;
    for (k = kts1; k <= kmax; ++k) {
        if (flg) {
            if (k >= ktcon && k < kmax) {
                float dz1 = zl(k + 1) - zl(k);
                float gamma = el2orc * qeso(k) / (to(k) * to(k));
                float rfact = 1.f + fv_ * cp_ * gamma * to(k) / hvap_;
                aa2 = aa2 + dz1 * (g_ / (cp_ * to(k))) * dbyo(k) / (1.f + gamma) * rfact;
                if (aa2 < 0.f) { ktcon1 = k; flg = false; }
            }
        }
    }

    // :1196-1229 overshooting-layer moisture
    for (k = kts1; k <= kmax; ++k) {
        if (k >= ktcon && k < ktcon1) {
            float dz = (zi(k + 1) - zi(k));
            float gamma = el2orc * qeso(k) / (to(k) * to(k));
            float qrch = qeso(k) + gamma * dbyo(k) / (hvap_ * (1.f + gamma));
            float tem = 0.5f * (xlamb(k) + xlamb(k - 1)) * dz;
            float tem1 = 0.5f * xlamud * dz;
            float factor = 1.f + tem - tem1;
            qcko(k) = ((1.f - tem1) * qcko(k - 1) + tem * 0.5f * (qo(k) + qo(k - 1))) / factor;
            qcirs(k) = eta(k) * qcko(k) - eta(k) * qrch;
            if (qcirs(k) > 0.f) {
                float etah = .5f * (eta(k) + eta(k - 1));
                float qlk;
                if (ncloud > 0) {
                    float dp = 1000.f * del(k);
                    qlk = qcirs(k) / (eta(k) + etah * (c0t(k) + c1t) * dz);
                    dellal(k) = etah * c1t * dz * qlk * g_ / dp;
                } else {
                    qlk = qcirs(k) / (eta(k) + etah * c0t(k) * dz);
                }
                pwo(k) = etah * c0t(k) * dz * qlk;
                float qc = qlk + qrch;
                qcko(k) = qc;
                pwavo = pwavo + pwo(k);
            }
        }
    }

    // :1233-1306 updraft velocity
    const float bb1 = 2.f * (1.f + KSAS_BET1 * KSAS_CD1);
    const float bb2 = 2.f / (KSAS_F1 * (1.f + KSAS_GAM1));
    {
        int kk = kbcon1;
        float po = .5f * (p(kk) + p(kk + 1));
        float tem = po / (rd_ * to(kk));
        float wucb = -10.f * dot(kk) / (tem * g_);
        if (wucb > 0.f) {
            wu2(kk) = wucb * wucb;
        } else {
            wu2(kk) = 0.f;
        }
    }
    for (k = kts1; k <= kmax; ++k) {
        if (k > kbcon1 && k < ktcon) {
            float dz = zi(k + 1) - zi(k);
            float tem = 0.25f * bb1 * (drag(k) + drag(k - 1)) * dz;
            float tem1 = 0.5f * bb2 * (buo(k) + buo(k - 1)) * dz;
            float ptem = (1.f - tem) * wu2(k - 1);
            float ptem1 = 1.f + tem;
            wu2(k) = (ptem + tem1) / ptem1;
            wu2(k) = SAS_MAX(wu2(k), 0.f);
        }
    }
    float wc, wbar, sumx;
    wc = 0.f; wbar = 0.f; sumx = 0.f;
    {
        const float ptem = -0.5f * rd_ / g_;
        for (k = kts1; k <= kmax; ++k) {
            if (k > kbcon1 && k < ktcon) {
                float dz = zi(k + 1) - zi(k);
                float tem = 0.5f * (sqrtf(wu2(k)) + sqrtf(wu2(k - 1)));
                wc = wc + tem * dz;
                // po1(k) = .5*(p(k)+p(k+1)), :1274-1278
                float po1k = .5f * (p(k) + p(k + 1));
                float po1km = .5f * (p(k - 1) + p(k));
                tem = 10.f * dot(k) * to(k) / po1k;
                float tem1 = 10.f * dot(k - 1) * to(k - 1) / po1km;
                wbar = wbar + ptem * (tem + tem1) * dz;
                sumx = sumx + dz;
            }
        }
    }
    if (sumx == 0.f) {
        cnvflg = false;
    } else {
        wc = wc / sumx;
        wbar = wbar / sumx;
    }
    if (wc < 1.e-4f) cnvflg = false;

    // :1312-1320 cloud core fraction
    float clear = 0.f;
    if (cnvflg) {
        float tem = wbar / wc;
        tem = SAS_MAX(tem, 0.f);
        clear = 1.f - tem;
        clear = SAS_MAX(SAS_MIN(clear, 1.0f), 0.f);
        if (wbar > 0.f && wbar > wc) cnvflg = false;
    }
    if (!cnvflg) break;

    // :1324-1334 exchange ktcon with ktcon1
    {
        int kk = ktcon;
        ktcon = ktcon1;
        ktcon1 = kk;
    }
    if (ktcon <= kb) cnvflg = false;
    if (!cnvflg) break;

    // :1336-1348 pressure-gradient factor
    for (k = kts; k <= kte; ++k) {
        if (k <= ktcon) {
            pgcon(k) = 0.5f + 0.5f * gfk_exp(3.f * (zl(k) - zl(ktcon)) / zl(ktcon));
        } else {
            pgcon(k) = 1.0f;
        }
        pgcon(k) = SAS_MIN(pgcon(k), 1.0f);
        pgcon(k) = SAS_MAX(pgcon(k), 0.5f);
    }
    // :1350-1365 updraft momentum
    for (k = kts1; k <= kmax1; ++k) {
        if (k > kb) {
            float dz = zi(k + 1) - zi(k);
            float tem = 0.5f * (xlamb(k) + xlamb(k - 1)) * dz;
            float tem1 = 0.5f * xlamud * dz;
            float factor = 1.f + tem - tem1;
            float ptem = 0.5f * tem + pgcon(k);
            float ptem1 = 0.5f * tem - pgcon(k);
            ucko(k) = ((1.f - tem1) * ucko(k - 1) + ptem * uo(k) + ptem1 * uo(k - 1)) / factor;
            vcko(k) = ((1.f - tem1) * vcko(k - 1) + ptem * vo(k) + ptem1 * vo(k - 1)) / factor;
        }
    }
    // :1369-1389 cloud-top liquid/vapour split
    if (ncloud > 0) {
        int kk = ktcon - 1;
        float gamma = el2orc * qeso(kk) / (to(kk) * to(kk));
        float qrch = qeso(kk) + gamma * dbyo(kk) / (hvap_ * (1.f + gamma));
        float dq = qcko(kk) - qrch;
        if (dq > 0.f) {
            qlko_ktcon = dq * sigma;
            qcko(kk) = qrch + dq * (1.f - sigma);
        }
    }

    // :1395-1428 downdraft strength from shear
    float vshear, edt_s;
    vshear = 0.f;
    for (k = kts1; k <= kmax; ++k) {
        if (k > kb && k <= ktcon) {
            float du = uo(k) - uo(k - 1);
            float dv = vo(k) - vo(k - 1);
            float shear = sqrtf(du * du + dv * dv);
            vshear = vshear + shear;
        }
    }
    {
        vshear = 1.e3f * vshear / (zi(ktcon + 1) - zi(kb + 1));
        float e1 = 1.591f - .639f * vshear + .0953f * (vshear * vshear)
                   - .00496f * (vshear * vshear * vshear);
        edt = 1.f - e1;
        float ccn_f = 1.0f;
        edt_s = edt;
        edt_s = SAS_MIN(edt_s, .9f);
        edt_s = SAS_MAX(edt_s, .0f);
        edt = SAS_MIN(edt, .9f);
        edt = SAS_MAX(edt, .0f);
        edt = edt * ccn_f;
        edto = edt;
        edtx = edt;
    }

    // :1432-1457 detrainment between 1 and kbdtr
    sumx = 0.f;
    for (k = kts; k <= kmax1; ++k) {
        if (k >= 1 && k < kbcon) {
            float dz = zi(k + 2) - zi(k + 1);
            sumx = sumx + dz;
        }
    }
    float xlamd;
    {
        float beta = KSAS_BETAS;
        if (slimsk == 1.f) beta = KSAS_BETAL;
        float dz = (sumx + zi(2)) / (float)kbcon;
        float tem = 1.f / (float)kbcon;
        xlamd = (1.f - gfk_pow(beta, tem)) / dz;
    }

    // :1461-1485 downdraft mass flux
    for (k = kts; k <= kmax; ++k) {
        etad(k) = 1.f;
        qrcdo(k) = 0.f;
        qrcd(k) = 0.f;
    }
    for (k = kmax1; k >= kts; --k) {
        if (k < jmin && k >= kbcon) {
            float dz = (zi(k + 2) - zi(k + 1));
            float ptem = KSAS_XLAMDD - KSAS_XLAMDE;
            etad(k) = etad(k + 1) * (1.f - ptem * dz);
        } else if (k < kbcon) {
            float dz = (zi(k + 2) - zi(k + 1));
            float ptem = xlamd + KSAS_XLAMDD - KSAS_XLAMDE;
            etad(k) = etad(k + 1) * (1.f - ptem * dz);
        }
    }

    // :1489-1554 downdraft moisture
    float pwevo;
    pwevo = 0.f;
    {
        int jmn = jmin;
        hcdo(jmn) = heo(jmn);
        qcdo(jmn) = qo(jmn);
        qrcdo(jmn) = qeso(jmn);
        ucdo(jmn) = uo(jmn);
        vcdo(jmn) = vo(jmn);
    }
    for (k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            float dz = zi(k + 2) - zi(k + 1);
            float tem, tem1;
            if (k >= kbcon) {
                tem = KSAS_XLAMDE * dz;
                tem1 = 0.5f * KSAS_XLAMDD * dz;
            } else {
                tem = KSAS_XLAMDE * dz;
                tem1 = 0.5f * (xlamd + KSAS_XLAMDD) * dz;
            }
            float factor = 1.f + tem - tem1;
            float ptem = 0.5f * tem - pgcon(k);
            float ptem1 = 0.5f * tem + pgcon(k);
            hcdo(k) = ((1.f - tem1) * hcdo(k + 1) + tem * 0.5f * (heo(k) + heo(k + 1))) / factor;
            ucdo(k) = ((1.f - tem1) * ucdo(k + 1) + ptem * uo(k + 1) + ptem1 * uo(k)) / factor;
            vcdo(k) = ((1.f - tem1) * vcdo(k + 1) + ptem * vo(k + 1) + ptem1 * vo(k)) / factor;
            dbyo(k) = hcdo(k) - heso(k);
        }
    }
    for (k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            float dq = qeso(k);
            float dtt = to(k);
            float gamma = el2orc * dq / (dtt * dtt);
            qrcdo(k) = dq + (1.f / hvap_) * (gamma / (1.f + gamma)) * dbyo(k);
            float dz = zi(k + 2) - zi(k + 1);
            float tem, tem1;
            if (k >= kbcon) {
                tem = KSAS_XLAMDE * dz;
                tem1 = 0.5f * KSAS_XLAMDD * dz;
            } else {
                tem = KSAS_XLAMDE * dz;
                tem1 = 0.5f * (xlamd + KSAS_XLAMDD) * dz;
            }
            float factor = 1.f + tem - tem1;
            qcdo(k) = ((1.f - tem1) * qcdo(k + 1) + tem * 0.5f * (qo(k) + qo(k + 1))) / factor;
            pwdo(k) = etad(k + 1) * qcdo(k) - etad(k + 1) * qrcdo(k);
            qcdo(k) = qrcdo(k);
            pwevo = pwevo + pwdo(k);
        }
    }

    // :1560-1571 final downdraft strength
    {
        float edtmax = KSAS_EDTMAXL;
        if (slimsk == 2.f) edtmax = KSAS_EDTMAXS;
        if (pwevo < 0.f) {
            edto = -edto * pwavo / pwevo;
            edto = SAS_MIN(edto, edtmax);
        } else {
            edto = 0.f;
        }
    }

    // :1575-1599 downdraft cloud work
    for (k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            float gamma = el2orc * qeso(k) / (to(k) * to(k));
            float dhh = hcdo(k);
            float dtt = to(k);
            float dg = gamma;
            float dh = heso(k);
            float dz = -1.f * (zl(k + 1) - zl(k));
            aa1 = aa1 + edto * dz * (g_ / (cp_ * dtt)) * ((dhh - dh) / (1.f + dg))
                  * (1.f + fv_ * cp_ * dg * dtt / hvap_);
            aa1 = aa1 + edto * dz * g_ * fv_ * SAS_MAX(0.f, (qeso(k) - qo(k)));
        }
    }
    if (aa1 <= 0.f) cnvflg = false;
    if (!cnvflg) break;

    // :1604-1627 unit-mass-flux changes, surface layer
    for (k = kts; k <= kmax; ++k) {
        dellah(k) = 0.f;
        dellaq(k) = 0.f;
        dellau(k) = 0.f;
        dellav(k) = 0.f;
    }
    {
        float dp = 1000.f * del(1);
        dellah(1) = edto * etad(1) * (hcdo(1) - heo(1)) * g_ / dp;
        dellaq(1) = edto * etad(1) * (qcdo(1) - qo(1)) * g_ / dp;
        dellau(1) = edto * etad(1) * (ucdo(1) - uo(1)) * g_ / dp;
        dellav(1) = edto * etad(1) * (vcdo(1) - vo(1)) * g_ / dp;
    }
    // :1631-1693 subsidence and entrainment
    for (k = kts1; k <= kmax1; ++k) {
        if (k < ktcon) {
            float aup = 1.f;
            if (k <= kb) aup = 0.f;
            float adw = 1.f;
            if (k > jmin) adw = 0.f;
            float dv1 = heo(k);
            float dv2 = .5f * (heo(k) + heo(k - 1));
            float dv3 = heo(k - 1);
            float dv1q = qo(k);
            float dv2q = .5f * (qo(k) + qo(k - 1));
            float dv3q = qo(k - 1);
            float dv1u = uo(k);
            float dv2u = .5f * (uo(k) + uo(k - 1));
            float dv3u = uo(k - 1);
            float dv1v = vo(k);
            float dv2v = .5f * (vo(k) + vo(k - 1));
            float dv3v = vo(k - 1);
            float dp = 1000.f * del(k);
            float dz = zi(k + 1) - zi(k);
            float tem = 0.5f * (xlamb(k) + xlamb(k - 1));
            float tem1 = xlamud;
            float ptem, ptem1;
            if (k <= kbcon) {
                ptem = KSAS_XLAMDE;
                ptem1 = xlamd + KSAS_XLAMDD;
            } else {
                ptem = KSAS_XLAMDE;
                ptem1 = KSAS_XLAMDD;
            }
            dellah(k) = dellah(k) +
                ((aup * eta(k) - adw * edto * etad(k)) * dv1
                 - (aup * eta(k - 1) - adw * edto * etad(k - 1)) * dv3
                 - (aup * tem * eta(k - 1) + adw * edto * ptem * etad(k)) * dv2 * dz
                 + aup * tem1 * eta(k - 1) * .5f * (hcko(k) + hcko(k - 1)) * dz
                 + adw * edto * ptem1 * etad(k) * .5f * (hcdo(k) + hcdo(k - 1)) * dz) * g_ / dp;
            dellaq(k) = dellaq(k) +
                ((aup * eta(k) - adw * edto * etad(k)) * dv1q
                 - (aup * eta(k - 1) - adw * edto * etad(k - 1)) * dv3q
                 - (aup * tem * eta(k - 1) + adw * edto * ptem * etad(k)) * dv2q * dz
                 + aup * tem1 * eta(k - 1) * .5f * (qcko(k) + qcko(k - 1)) * dz
                 + adw * edto * ptem1 * etad(k) * .5f * (qrcdo(k) + qrcdo(k - 1)) * dz) * g_ / dp;
            dellau(k) = dellau(k) +
                ((aup * eta(k) - adw * edto * etad(k)) * dv1u
                 - (aup * eta(k - 1) - adw * edto * etad(k - 1)) * dv3u
                 - (aup * tem * eta(k - 1) + adw * edto * ptem * etad(k)) * dv2u * dz
                 + aup * tem1 * eta(k - 1) * .5f * (ucko(k) + ucko(k - 1)) * dz
                 + adw * edto * ptem1 * etad(k) * .5f * (ucdo(k) + ucdo(k - 1)) * dz
                 - pgcon(k) * (aup * eta(k - 1) - adw * edto * etad(k)) * (dv1u - dv3u)) * g_ / dp;
            dellav(k) = dellav(k) +
                ((aup * eta(k) - adw * edto * etad(k)) * dv1v
                 - (aup * eta(k - 1) - adw * edto * etad(k - 1)) * dv3v
                 - (aup * tem * eta(k - 1) + adw * edto * ptem * etad(k)) * dv2v * dz
                 + aup * tem1 * eta(k - 1) * .5f * (vcko(k) + vcko(k - 1)) * dz
                 + adw * edto * ptem1 * etad(k) * .5f * (vcdo(k) + vcdo(k - 1)) * dz
                 - pgcon(k) * (aup * eta(k - 1) - adw * edto * etad(k)) * (dv1v - dv3v)) * g_ / dp;
        }
    }
    // :1697-1718 cloud top
    {
        int indx = ktcon;
        float dp = 1000.f * del(indx);
        float dv1 = heo(indx - 1);
        dellah(indx) = eta(indx - 1) * (hcko(indx - 1) - dv1) * g_ / dp;
        float dv1q = qo(indx - 1);
        dellaq(indx) = eta(indx - 1) * (qcko(indx - 1) - dv1q) * g_ / dp;
        float dv1u = uo(indx - 1);
        dellau(indx) = eta(indx - 1) * (ucko(indx - 1) - dv1u) * g_ / dp;
        float dv1v = vo(indx - 1);
        dellav(indx) = eta(indx - 1) * (vcko(indx - 1) - dv1v) * g_ / dp;
        dellal(indx) = eta(indx - 1) * qlko_ktcon * g_ / dp;
    }
    // :1722-1735 changed environment per unit mass flux
    for (k = kts; k <= kmax; ++k) {
        if (k > ktcon) {
            qo(k) = q1(k);
            to(k) = t1(k);
        }
        if (k <= ktcon) {
            qo(k) = dellaq(k) * mbdt + q1(k);
            float dellat = (dellah(k) - hvap_ * dellaq(k)) / cp_;
            to(k) = dellat * mbdt + t1(k);
            qo(k) = SAS_MAX(qo(k), 1.0e-10f);
        }
    }

    // :1749-1797 environment again
    for (k = kts; k <= kmax; ++k) {
        qeso(k) = 0.01f * FPVS(to(k));
        qeso(k) = eps * qeso(k) / (p(k) + (eps - 1.f) * qeso(k));
        qeso(k) = SAS_MAX(qeso(k), qmin_);
    }
    for (k = kts; k <= kmax1; ++k) {
        float dz = .5f * (zl(k + 1) - zl(k));
        float dp = .5f * (p(k + 1) - p(k));
        float es = 0.01f * FPVS(to(k + 1));
        float pprime = p(k + 1) + (eps - 1.f) * es;
        float qs = eps * es / pprime;
        float dqsdp = -qs / pprime;
        float desdt = es * (fact1 / to(k + 1) + fact2 / (to(k + 1) * to(k + 1)));
        float dqsdt = qs * p(k + 1) * desdt / (es * pprime);
        float gamma = el2orc * qeso(k + 1) / (to(k + 1) * to(k + 1));
        float dtt = (g_ * dz + hvap_ * dqsdp * dp) / (cp_ * (1.f + gamma));
        float dq = dqsdt * dtt + dqsdp * dp;
        to(k) = to(k + 1) + dtt;
        qo(k) = qo(k + 1) + dq;
        float po = .5f * (p(k) + p(k + 1));
        qeso(k) = 0.01f * FPVS(to(k));
        qeso(k) = eps * qeso(k) / (po + (eps - 1.f) * qeso(k));
        qeso(k) = SAS_MAX(qeso(k), qmin_);
        qo(k) = SAS_MAX(qo(k), 1.0e-10f);
        heo(k) = .5f * g_ * (zl(k) + zl(k + 1)) + cp_ * to(k) + hvap_ * qo(k);
        heso(k) = .5f * g_ * (zl(k) + zl(k + 1)) + cp_ * to(k) + hvap_ * qeso(k);
    }
    {
        int kk = kmax;
        heo(kk) = g_ * zl(kk) + cp_ * to(kk) + hvap_ * qo(kk);
        heso(kk) = g_ * zl(kk) + cp_ * to(kk) + hvap_ * qeso(kk);
    }
    float xaa0, xpwav;
    xaa0 = 0.f;
    xpwav = 0.f;
    {
        int indx = kb;
        float xhkb = heo(indx);
        float xqkb = qo(indx);
        hcko(indx) = xhkb;
        qcko(indx) = xqkb;
    }

    // :1815-1884 static control
    for (k = kts1; k <= kmax1; ++k) {
        if (k > kb && k <= ktcon) {
            float dz = zi(k + 1) - zi(k);
            float tem = 0.5f * (xlamb(k) + xlamb(k - 1)) * dz;
            float tem1 = 0.5f * xlamud * dz;
            float factor = 1.f + tem - tem1;
            hcko(k) = ((1.f - tem1) * hcko(k - 1) + tem * 0.5f * (heo(k) + heo(k - 1))) / factor;
        }
    }
    for (k = kts1; k <= kmax1; ++k) {
        if (k > kb && k < ktcon) {
            float alpha1 = SAS_MIN((-0.7f * gfk_log(100.f) + 24.f) * 0.0001f, KSAS_C0);
            float beta1 = 0.07f;
            float c0fac;
            if (to(k) > t0c_) {
                c0fac = alpha1;
            } else {
                c0fac = alpha1 * gfk_exp(beta1 * (to(k) - t0c_));
            }
            c0fac = SAS_MAX(0.0f, c0fac);
            c0t(k) = c0fac;
            float dz = zi(k + 1) - zi(k);
            float gamma = el2orc * qeso(k) / (to(k) * to(k));
            float xdby = hcko(k) - heso(k);
            float xqrch = qeso(k) + gamma * xdby / (hvap_ * (1.f + gamma));
            float tem = 0.5f * (xlamb(k) + xlamb(k - 1)) * dz;
            float tem1 = 0.5f * xlamud * dz;
            float factor = 1.f + tem - tem1;
            qcko(k) = ((1.f - tem1) * qcko(k - 1) + tem * 0.5f * (qo(k) + qo(k - 1))) / factor;
            float dq = eta(k) * qcko(k) - eta(k) * xqrch;
            if (k >= kbcon && dq > 0.f) {
                float etah = .5f * (eta(k) + eta(k - 1));
                float qlk;
                if (ncloud > 0 && k > jmin) {
                    qlk = dq / (eta(k) + etah * (c0t(k) + c1t) * dz);
                } else {
                    qlk = dq / (eta(k) + etah * c0t(k) * dz);
                }
                float xpw = etah * c0t(k) * dz * qlk;
                if (k < ktcon1) {
                    xaa0 = xaa0 - (zl(k + 1) - zl(k)) * g_ * qlk;
                }
                qcko(k) = qlk + xqrch;
                xpwav = xpwav + xpw;
            }
        }
        if (k >= kbcon && k < ktcon1) {
            float dz1 = zl(k + 1) - zl(k);
            float gamma = el2orc * qeso(k) / (to(k) * to(k));
            float rfact = 1.f + fv_ * cp_ * gamma * to(k) / hvap_;
            float xdby = hcko(k) - heso(k);
            xaa0 = xaa0 + dz1 * (g_ / (cp_ * to(k))) * xdby / (1.f + gamma) * rfact;
            xaa0 = xaa0 + dz1 * g_ * fv_ * SAS_MAX(0.f, (qeso(k) - qo(k)));
        }
    }

    // :1890-1958 downdraft, perturbed environment
    float xpwev;
    xpwev = 0.f;
    {
        int jmn = jmin;
        xhcd(jmn) = heo(jmn);
        xqcd(jmn) = qo(jmn);
        qrcd(jmn) = qeso(jmn);
    }
    for (k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            float dz = zi(k + 2) - zi(k + 1);
            float tem, tem1;
            if (k >= kbcon) {
                tem = KSAS_XLAMDE * dz;
                tem1 = 0.5f * KSAS_XLAMDD * dz;
            } else {
                tem = KSAS_XLAMDE * dz;
                tem1 = 0.5f * (xlamd + KSAS_XLAMDD) * dz;
            }
            float factor = 1.f + tem - tem1;
            xhcd(k) = ((1.f - tem1) * xhcd(k + 1) + tem * 0.5f * (heo(k) + heo(k + 1))) / factor;
        }
    }
    for (k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            float dq = qeso(k);
            float dtt = to(k);
            float gamma = el2orc * dq / (dtt * dtt);
            float dh = xhcd(k) - heso(k);
            qrcd(k) = dq + (1.f / hvap_) * (gamma / (1.f + gamma)) * dh;
            float dz = zi(k + 2) - zi(k + 1);
            float tem, tem1;
            if (k >= kbcon) {
                tem = KSAS_XLAMDE * dz;
                tem1 = 0.5f * KSAS_XLAMDD * dz;
            } else {
                tem = KSAS_XLAMDE * dz;
                tem1 = 0.5f * (xlamd + KSAS_XLAMDD) * dz;
            }
            float factor = 1.f + tem - tem1;
            xqcd(k) = ((1.f - tem1) * xqcd(k + 1) + tem * 0.5f * (qo(k) + qo(k + 1))) / factor;
            float xpwd = etad(k + 1) * (xqcd(k) - qrcd(k));
            xqcd(k) = qrcd(k);
            xpwev = xpwev + xpwd;
        }
    }
    {
        float edtmax = KSAS_EDTMAXL;
        if (slimsk == 2.f) edtmax = KSAS_EDTMAXS;
        if (xpwev >= 0.f) {
            edtx = 0.f;
        } else {
            edtx = -edtx * xpwav / xpwev;
            edtx = SAS_MIN(edtx, edtmax);
        }
    }
    // :1962-1977
    for (k = kmax1; k >= kts; --k) {
        if (k < jmin) {
            float gamma = el2orc * qeso(k) / (to(k) * to(k));
            float dhh = xhcd(k);
            float dtt = to(k);
            float dg = gamma;
            float dh = heso(k);
            float dz = -1.f * (zl(k + 1) - zl(k));
            xaa0 = xaa0 + edtx * dz * (g_ / (cp_ * dtt)) * ((dhh - dh) / (1.f + dg))
                   * (1.f + fv_ * cp_ * dg * dtt / hvap_);
            xaa0 = xaa0 + edtx * dz * g_ * fv_ * SAS_MAX(0.f, (qeso(k) - qo(k)));
        }
    }

    // :1981-1987 critical cloud work function timescale
    float dtconv;
    {
        float tem = zi(ktcon1) - zi(kbcon1);
        dtconv = KSAS_TFAC * tem / wc;
        dtconv = SAS_MAX(dtconv, dtmin);
        dtconv = SAS_MIN(dtconv, dtmax);
    }
    // :1993-2020 large-scale forcing timescale over water
    if (slimsk == 2.f) {
        wc = 0.f;
        sumx = 0.f;
        for (k = kts; k <= kbmax; ++k) {
            if (k > kb && k <= kbcon) {
                float dz = zi(k + 1) - zi(k);
                float tem = sqrtf(uo(k) * uo(k) + vo(k) * vo(k));
                wc = wc + tem * dz;
                sumx = sumx + dz;
            }
        }
        wc = wc / sumx;
        dtpbl = zl(kbcon) / wc;
    } else {
        dtpbl = dtconv;
    }
    // :2022-2027 boundary-layer growth
    apbl = g_ * (hpbl_i - hpbl_hold_i) / dt2 * dtpbl;
    apbl = SAS_MIN(SAS_MAX(apbl, 0.0f), aa1);

    // :2029-2055 closure
    float xmb = 0.f;
    {
        float f = (aa1 - apbl) / dtconv;
        if (f <= 0.f) cnvflg = false;
        if (cnvflg) {
            float xk = (xaa0 - aa1) / mbdt;
            if (xk >= 0.f) cnvflg = false;
            if (cnvflg) {
                xmb = -f / xk;
                xmb = xmb * clear * (1.f - sigma);
                xmb = SAS_MIN(xmb, xmbmax);
            }
        }
        pdot = 10.f * dot(kbcon);
        if (cnvflg && xmb < pdot / g_) cnvflg = false;
    }
    if (!cnvflg) break;

    // :2059-2071 restore
    for (k = kts; k <= kmax; ++k) {
        to(k) = t1(k);
        qo(k) = q1(k);
        uo(k) = u1(k);
        vo(k) = v1(k);
        qeso(k) = 0.01f * FPVS(t1(k));
        qeso(k) = eps * qeso(k) / (p(k) + (eps - 1.f) * qeso(k));
        qeso(k) = SAS_MAX(qeso(k), qmin_);
    }

    // :2086-2116 feedback
    for (k = kts; k <= kmax; ++k) {
        if (k <= ktcon && dellaq(k) <= 0.f) {
            if (q1(k) > 0.f) {
                float tem = dellaq(k) * xmb * dt2;
                dellaq(k) = SAS_MAX(tem, -q1(k)) / (xmb * dt2);
            } else {
                dellaq(k) = 0.0f;
            }
        }
    }
    for (k = kts; k <= kmax; ++k) {
        if (k <= ktcon) {
            float dellat = (dellah(k) - hvap_ * dellaq(k)) / cp_;
            t1(k) = t1(k) + dellat * xmb * dt2;
            q1(k) = q1(k) + dellaq(k) * xmb * dt2;
            float tem = 1.f / rcs;
            u1(k) = u1(k) + dellau(k) * xmb * dt2 * tem;
            v1(k) = v1(k) + dellav(k) * xmb * dt2 * tem;
            // delhbar/delqbar/deltbar/delubar/delvbar (:2108-2113) are
            // column budgets nothing downstream reads.
        }
    }
    for (k = kts; k <= kmax; ++k) {
        if (k <= ktcon) {
            qeso(k) = 0.01f * FPVS(t1(k));
            qeso(k) = eps * qeso(k) / (p(k) + (eps - 1.f) * qeso(k));
            qeso(k) = SAS_MAX(qeso(k), qmin_);
        }
    }

    // :2128-2149 total rain
    {
        float rntot = 0.f, delqev = 0.f, delq2 = 0.f;
        bool flg2 = cnvflg;
        for (k = kmax; k >= kts; --k) {
            if (k < ktcon) {
                float aup = 1.f;
                if (k <= kb) aup = 0.f;
                float adw = 1.f;
                if (k >= jmin) adw = 0.f;
                rntot = rntot + (aup * pwo(k) + adw * edto * pwdo(k)) * xmb * .001f * dt2;
            }
        }
        // :2153-2203 rain and its evaporation
        for (k = kmax; k >= kts; --k) {
            float qevap = 0.0f;
            if (k < ktcon) {
                float aup = 1.f;
                if (k <= kb) aup = 0.f;
                float adw = 1.f;
                if (k >= jmin) adw = 0.f;
                rain = rain + (aup * pwo(k) + adw * edto * pwdo(k)) * xmb * .001f * dt2;
            }
            if (flg2 && k < ktcon) {
                float ccn_f = 1.0f;
                float evef = edt_s * KSAS_EVFACTS * ccn_f;
                if (slimsk == 1.f) evef = edt_s * KSAS_EVFACTL * ccn_f;
                float qcond = evef * (q1(k) - qeso(k)) / (1.f + el2orc * qeso(k) / (t1(k) * t1(k)));
                float dp = 1000.f * del(k);
                if (rain > 0.f && qcond < 0.f) {
                    qevap = -qcond * (1.f - gfk_exp(-.32f * sqrtf(dt2 * rain)));
                    qevap = SAS_MIN(qevap, rain * 1000.f * g_ / dp);
                    delq2 = delqev + .001f * qevap * dp / g_;
                    if (delq2 > rntot) {
                        qevap = 1000.f * g_ * (rntot - delqev) / dp;
                        flg2 = false;
                    }
                }
                if (rain > 0.f && qevap > 0.f) {
                    q1(k) = q1(k) + qevap;
                    t1(k) = t1(k) - (hvap_ / cp_) * qevap;
                    rain = rain - .001f * qevap * dp / g_;
                    delqev = delqev + .001f * dp * qevap / g_;
                }
                // dellaq/delqbar/deltbar updates (:2193-2195) feed nothing
                // that reaches an output.
            }
            // qrs/qci (:2198-2201) are locals nothing downstream reads.
        }
        // :2207-2218
        if (rain < 0.f && !flg2) rain = 0.f;
        if (rain <= 0.f) {
            rain = 0.f;
        } else {
            ktop = ktcon;
            kbot = kbcon;
        }
    }
    // :2220-2229
    if (rain <= 0.f) {
        for (k = kts; k <= kmax; ++k) {
            t1(k) = to(k);
            q1(k) = qo(k);
            u1(k) = uo(k);
            v1(k) = vo(k);
        }
    }
    // :2233-2250 detrained condensate
    if (ncloud > 0) {
        if (rain > 0.f) {
            for (k = kts; k <= kmax; ++k) {
                if (k >= kbcon && k <= ktcon) {
                    float tem = dellal(k) * xmb * dt2;
                    float tem1 = SAS_MAX(0.0f, SAS_MIN(1.0f, (KSAS_TCR - t1(k)) * tcrf));
                    if (ncloud >= 2) {
                        qi2(k) = qi2(k) + tem * tem1;
                        qc2(k) = qc2(k) + tem * (1.0f - tem1);
                    } else {
                        qc2(k) = qc2(k) + tem;
                    }
                }
            }
        }
    }

    } while (0);

    // ------------------------------------------------ cu_ksas, :245-293
    {
        const float pratec1 = rain * 1000.f / ((float)stepcu * dt);
        const float raincv1 = rain * 1000.f / ((float)stepcu);
        raincv[col] = raincv1;
        pratec[col] = pratec1;
        hbot[col] = (float)kbot;
        htop[col] = (float)ktop;
        for (k = kts; k <= kte; ++k) {
            F_(rthcuten, k) = (t1(k) - F_(t3d, k)) / F_(pi3d, k) * rdelt;
            F_(rqvcuten, k) = (q1(k) - F_(qv3d, k)) * rdelt;
            F_(rucuten, k) = (u1(k) - F_(u3d, k)) * rdelt;
            F_(rvcuten, k) = (v1(k) - F_(v3d, k)) * rdelt;
            if (f_qi) F_(rqicuten, k) = (qi2(k) - F_(qi3d, k)) * rdelt;
            if (f_qc) F_(rqccuten, k) = (qc2(k) - F_(qc3d, k)) * rdelt;
        }
    }
#undef W_
#undef F_
#undef FPVS
}
