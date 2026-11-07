// EEPS: the TKE + TKE-dissipation (e-epsilon) PBL, bl_pbl_physics = 16.
//
// A transcription of WRF v4.6.1 phys/module_bl_eepsilon.F (module_bl_eeps):
// the EEPS driver (:58-360), the eeps2d column core (:364-1048) and its
// tridiag solver (:1053-1090), one CUDA thread per column, in WRF's own
// operation order. Strict numerics uses the pinned scalar gfk_pow/exp/log;
// the default uses CUDA powf/expf/logf and permits contraction. Integer
// powers retain the multiply chains in the native scalar source.
//
// Array convention.  The engine's arrays are bottom-up (k = 0 at the
// surface), (nz, ny, nx), column ordinal col = j*nx + i, element k*n + col.
// The column core works top-down exactly as eeps2d does: index kk = 1..km
// is level km+1-kk of WRF's bottom-up 1-based column (kk = 1 the model top,
// kk = km the lowest level), so every Fortran subscript below is the
// Fortran subscript.
//
// Native undefined values: RADSUM (:748) has no initialization in WRF.
// It starts at zero here, as the explicit stock-zero/fixed oracle witness
// records. An active cloud-top source with KMINRAD < 3 would read outside
// SZV at :734. Report that condition through the device health ledger;
// do not silently replace the native expression with a clamped stencil.
//
// EEPS_NZ is the native column depth. Work arrays occupy a coalesced device
// workspace, bounded by the resident column tile rather than CUDA context
// thread capacity. This avoids the full-forecast local-memory allocation failure.

#ifndef EEPS_NZ
#error "EEPS_NZ must be defined by the launcher"
#endif
#define EEPS_KM EEPS_NZ
#define EEPS_DIM (EEPS_NZ + 2)

#ifdef GPUWM_WRF_EXACT
#define eeps_pow gfk_pow
#define eeps_exp gfk_exp
#define eeps_log gfk_log
#else
#define eeps_pow powf
#define eeps_exp expf
#define eeps_log logf
#endif

// share/module_model_constants.F, as gfortran folds them (bit patterns are
// the column oracle's constants.txt; tests/test_eeps_port.py pins both).
#define EEPS_RGAS    __int_as_float(0x438F8000)   // r_d     287.
#define EEPS_T13     __int_as_float(0x3EAAAAAB)   // Prandtl 1./3.0
#define EEPS_EP1     __int_as_float(0x3F1BBDA4)   // EP_1    R_v/R_d-1.
#define EEPS_EP2     __int_as_float(0x3F1F2B09)   // EP_2    R_d/R_v
#define EEPS_GRV     __int_as_float(0x411CF5C3)   // g       9.81
#define EEPS_AKV     __int_as_float(0x3ECCCCCD)   // KARMAN  0.4
#define EEPS_CP      __int_as_float(0x447B2000)   // cp      7.*r_d/2.
#define EEPS_RCP     __int_as_float(0x3E924925)   // rcp     r_d/cp
#define EEPS_XLV     __int_as_float(0x4A189680)   // XLV     2.5E6
#define EEPS_XLS     __int_as_float(0x4A2DF340)   // XLS     2.85E6
#define EEPS_P1000MB __int_as_float(0x47C35000)   // p1000mb 100000.

// module_bl_eeps private parameters (:30-51).
#define EEPS_CS2 17.67f
#define EEPS_CS3 273.15f
#define EEPS_CS4 29.65f
#define EEPS_CI2 22.514f
#define EEPS_CI3 273.16f
#define EEPS_CI4 0.0f
#define EEPS_HGFR 233.15f
#define EEPS_T0 273.15f
#define EEPS_EPSL 1.e-20f
#define EEPS_US2MIN 0.1f
#define EEPS_AKVMX 1600.f
#define EEPS_RICR 0.25f
#define EEPS_BT 5.0f
#define EEPS_MINPEK 1.0e-4f
#define EEPS_MINPEP 1.0e-7f
#define EEPS_MAXPEK 160.f
#define EEPS_MAXPEP 25.0f
#define EEPS_ZFMIN 0.01f
#define EEPS_C1 1.35f
#define EEPS_C2 0.09f
#define EEPS_C3 1.44f
#define EEPS_C4 1.92f
#define EEPS_C5 0.77f
// bl_eeps_tkeadv = 1 and bl_eeps_topdown = 1 (:42-44) are PARAMETERs in
// WRF; their arms are the only arms transcribed.

// Fortran MIN/MAX on two reals: the first argument unless the second is
// strictly beyond it.
__device__ __forceinline__ float eeps_max(float a, float b)
{
    return (b > a) ? b : a;
}
__device__ __forceinline__ float eeps_min(float a, float b)
{
    return (b < a) ? b : a;
}

// Each work-array level spans a tile of columns for coalesced warp access.
//
// Pass structure.  Every value below is the native expression, evaluated
// once, in WRF's operand order; only the loops are fused.  Each tridiagonal
// coefficient row is formed inside the descending elimination sweep that
// consumes it (tridiag :1053-1090 reads row k only at step k), so the
// coefficient arrays of the native source are never stored.  Arrays that are
// plain copies of the inputs (T3D, U3D, V3D) are read from the inputs.  The
// strict replay is the proof that fusing changed no word.
#define EEPS_WS_SLOTS 22
#define EEPS_WS_LANES 32
struct EepsColumn {
    float* data;
    __device__ __forceinline__ float& operator[](int k) const {
        return data[(size_t)k * EEPS_WS_LANES];
    }
};

// :782-794, the stability function of one Richardson number.
__device__ __forceinline__ float eeps_dum2(float rich)
{
    float richf = 0.1620092f;
    if (rich < 0.145f) {
        richf = 0.6588f * (rich + 0.1776f
                           - sqrtf(rich * rich - 0.3221f * rich
                                   + 0.03156f));
        richf = eeps_max(-1000.0f, eeps_min(0.1620092f, richf));
    }
    float dum2 = 1.116893f;
    if (richf < 0.1620092f)
        dum2 = 1.318f * (0.2231f - richf) / (0.2341f - richf);
    return dum2;
}

extern "C" __global__ void eeps_column(
    // bottom-up (nz, ny, nx) column inputs
    const float* __restrict__ u3d, const float* __restrict__ v3d,
    const float* __restrict__ t3d, const float* __restrict__ qv3d,
    const float* __restrict__ qc3d, const float* __restrict__ qi3d,
    const float* __restrict__ qr3d, const float* __restrict__ qs3d,
    const float* __restrict__ qg3d, const float* __restrict__ p3d,
    const float* __restrict__ pi3d, const float* __restrict__ rho3d,
    const float* __restrict__ rthraten3d, const float* __restrict__ dz8w,
    // surface inputs (ny, nx)
    const float* __restrict__ psfc, const float* __restrict__ qsfc,
    const float* __restrict__ tsk, const float* __restrict__ ust,
    const float* __restrict__ rmol, const float* __restrict__ wspd,
    const float* __restrict__ hfx, const float* __restrict__ qfx,
    // carried state: PEK_PBL/PEP_PBL (in: read only on the first step; out)
    // and the advected PEK_ADV/PEP_ADV (in and out)
    float* __restrict__ pek3d, float* __restrict__ pep3d,
    float* __restrict__ pek_adv, float* __restrict__ pep_adv,
    // outputs
    float* __restrict__ rublten, float* __restrict__ rvblten,
    float* __restrict__ rthblten, float* __restrict__ rqvblten,
    float* __restrict__ rqcblten, float* __restrict__ rqiblten,
    float* __restrict__ exch_h, float* __restrict__ exch_m,   // (nz+1, ...)
    float* __restrict__ pblh_out, int* __restrict__ kpbl_out,
    unsigned int* __restrict__ status,
    const float dt, const int first_step, const int n,
    float* __restrict__ workspace, const int column_offset)
{
    const int lane = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = column_offset + lane;
    if (col >= n) return;
    const int km = EEPS_KM;
    const size_t work_block = (size_t)(lane / EEPS_WS_LANES) * EEPS_WS_SLOTS * EEPS_DIM * EEPS_WS_LANES;
    const int work_lane = lane % EEPS_WS_LANES;
    unsigned int flags = 0;
    // Bottom-up 1-based level k of this column -> flat element.
#define AT(arr, k) (arr)[(size_t)((k) - 1) * (size_t)n + col]
    // Top-down eeps2d index kk -> bottom-up level km + 1 - kk.
#define TD(arr, kk) AT(arr, km + 1 - (kk))
#define SLOT(s) EepsColumn{workspace + work_block + (size_t)(s) * EEPS_DIM * EEPS_WS_LANES + work_lane}

    // zi is bottom-up (driver :291); everything else is eeps2d's top-down
    // column.  pu..pqi hold the final diffusion's right-hand sides and its
    // eliminated values; yy/qw the TKE and epsilon solves'; qh the heat
    // solve's second factor set.
    const EepsColumn zi = SLOT(0), poz = SLOT(1), pu = SLOT(2), pv = SLOT(3);
    const EepsColumn pqv = SLOT(4), pqc = SLOT(5), pqi = SLOT(6);
    const EepsColumn pek = SLOT(7), pep = SLOT(8), psp = SLOT(9);
    const EepsColumn pt = SLOT(10), cpm = SLOT(11), szv = SLOT(12);
    const EepsColumn sh = SLOT(13), st = SLOT(14), rich = SLOT(15);
    const EepsColumn dum2 = SLOT(16), am = SLOT(17), src = SLOT(18);
    const EepsColumn yy = SLOT(19), qw = SLOT(20), qh = SLOT(21);

    // ---------------- EEPS driver: heights ------------------------------
    zi[1] = 0.0f;
    for (int k = 1; k <= km; ++k) zi[k + 1] = zi[k] + AT(dz8w, k);
    // dzz (:438-447) through ghti's zz(kk) = zi(km + 2 - kk) (:291-298).
#define DZZ(k) (zi[km + 2 - (k)] - zi[km + 1 - (k)])
#define DOZ(k) (poz[k] - poz[(k) + 1])

    const float psfcpa = psfc[col], ust_c = ust[col], rmol_c = rmol[col];
    const float wspd_c = wspd[col], qsfc_c = qsfc[col], hfx_c = hfx[col];
    const float qfx_c = qfx[col], tsk_c = tsk[col];

    const int mdt = 1 + (int)(sqrtf(dt / 9.0f) + 0.00001f);       // :434
    const float dtt = dt / (float)mdt;

    // ---------------- the reversed column, :472-507, :513-536 and the
    // stability functions :782-794 in one top-down pass.  Level kk-1's
    // Richardson number needs only levels kk-1 and kk.  k700/k950 (:705-712)
    // are the deepest qualifying levels, which the ascending scan keeps last.
    const float prskm = TD(p3d, km);
    int k700 = km, k950 = km;
    float u_p = 0.f, v_p = 0.f, tz_p = 0.f, pqv_p = 0.f, qvs_p = 0.f;
    float szv_p = 0.f, pt_p = 0.f, axl_p = 0.f, cx_p = 0.f, poz_p = 0.f;
    for (int kk = 1; kk <= km; ++kk) {
        const int k = km + 1 - kk;
        const float u = AT(u3d, k), v = AT(v3d, k), tz = AT(t3d, k);
        const float qv = AT(qv3d, k), qc = AT(qc3d, k), qi = AT(qi3d, k);
        const float qr = AT(qr3d, k), qs = AT(qs3d, k), qg = AT(qg3d, k);
        const float pqvk = qv / (1.f + qv);
        const float pqck = qc / (1.f + qc);
        const float pqik = qi / (1.f + qi);
        // q4..q6 enter eeps2d only through FAC (:481).  An absent optional
        // QR3D/QS3D/QG3D is 0. there (:261-275), and a zero field gives the
        // same 0. through 0./(1.+0.).
        const float q4 = qr / (1.f + qr), q5 = qs / (1.f + qs), q6 = qg / (1.f + qg);
        const float fack = pqck + pqik + q4 + q5 + q6;
        const float pozk = 0.5f * (zi[k] + zi[k + 1]);
        // bl_eeps_tkeadv == 1 (:280-282), after the itimestep == 1 copy
        // PEK_ADV = PEK, PEP_ADV = PEP (:227-230).
        const float ka = first_step ? AT(pek3d, k) : AT(pek_adv, k);
        const float ea = first_step ? AT(pep3d, k) : AT(pep_adv, k);
        pek[kk] = eeps_min(EEPS_MAXPEK, eeps_max(EEPS_MINPEK, ka));
        pep[kk] = eeps_min(EEPS_MAXPEP, eeps_max(EEPS_MINPEP, ea));
        pqv[kk] = pqvk;
        pqc[kk] = pqck;
        pqi[kk] = pqik;
        poz[kk] = pozk;

        const float prs = AT(p3d, k);
        const float pspk = eeps_pow(prs / EEPS_P1000MB, EEPS_RCP);
        const float ptk = tz / pspk;
        cpm[kk] = EEPS_CP * (1.f + 0.8f * pqvk);
        const float dum = 1.0f + EEPS_EP1 * pqvk;
        const float szvk = ptk * dum;
        const float cxk = 1.0f + pqvk + fack;
        const float alh = 3.1484e6f - 2.37e3f * tz;
        float eqp = 611.2f * eeps_exp(EEPS_CS2 * (tz - EEPS_CS3)
                                     / (tz - EEPS_CS4));
        eqp = eeps_min(0.5f * prs, eqp);
        const float qvsw = 0.622f * eqp / (prs - eqp);
        eqp = 611.0f * eeps_exp(EEPS_CI2 * (tz - EEPS_CI3)
                               / (tz - EEPS_CI4));
        eqp = eeps_min(0.5f * prs, eqp);
        const float qvsi = 0.622f * eqp / (prs - eqp);
        float faf;
        if (tz >= EEPS_T0) {
            faf = 0.0f;
        } else if (tz <= EEPS_HGFR) {
            faf = 1.0f;
        } else {
            if (pqik <= EEPS_EPSL) {
                faf = 0.0f;
            } else if (pqck <= EEPS_EPSL) {
                faf = 1.0f;
            } else {
                faf = pqik / (pqck + pqik);
            }
        }
        const float qvsk = (1.0f - faf) * qvsw + faf * qvsi;
        const float axlk = (1.0f - faf) * alh + faf * EEPS_XLS;
        psp[kk] = pspk;
        pt[kk] = ptk;
        szv[kk] = szvk;
        if (kk >= 2) {
            if ((prskm - prs) > 3.0e4f) k700 = kk;
            if ((prskm - prs) > 5.e3f) k950 = kk;
            // :513-536 at level kr = kk - 1, over levels kr and kr + 1
            const int kr = kk - 1;
            const float rdz = 1.0f / (poz_p - pozk);
            const float du = u_p - u;
            const float dv = v_p - v;
            const float shv = (du * du + dv * dv) * (rdz * rdz);
            float stv, dss;
            if ((pqv_p < qvs_p) || (pqvk < qvsk)) {
                dss = eeps_log(szv_p / szvk);
                stv = EEPS_GRV * dss * rdz;
            } else {
                dss = eeps_log(pt_p / ptk);
                const float tk = 0.5f * (tz + tz_p);
                const float qvk = 0.5f * (pqvk + pqv_p);
                const float alk = 0.5f * (axlk + axl_p);
                const float af = alk * qvk / (EEPS_RGAS * tk);
                const float ag = alk / (EEPS_CP * tk);
                const float aks = EEPS_EP2 * af * ag;
                const float aco = (1.0f + af) / (1.0f + aks);
                stv = (aco * (dss + ag * (pqv_p - pqvk))
                       - (cx_p - cxk)) * EEPS_GRV * rdz;
            }
            const float richv = stv / eeps_max(1.0e-7f, shv);
            sh[kr] = shv;
            st[kr] = stv;
            rich[kr] = richv;
            dum2[kr] = eeps_dum2(richv);
        }
        u_p = u; v_p = v; tz_p = tz; pqv_p = pqvk; qvs_p = qvsk;
        szv_p = szvk; pt_p = ptk; axl_p = axlk; cx_p = cxk; poz_p = pozk;
    }

    // :539-562 first-guess PBLH (richb recomputed per level: same bits)
#define UP2(k) (EEPS_GRV / eeps_max(EEPS_US2MIN, TD(u3d, k) * TD(u3d, k) + TD(v3d, k) * TD(v3d, k)))
    float pblh = poz[km];
    int kpbl = km;
    {
        const float szvkm = szv[km];
#define RICHB0(k) (poz[k] * (szv[k] / szvkm - 1.0f) * UP2(k))
        int lvl = km;
        for (int k = km - 1; k >= 5; --k) {
            if (RICHB0(k) >= EEPS_RICR) { lvl = k; break; }
        }
        if (lvl < km) {
            float rl = RICHB0(lvl), rl1 = RICHB0(lvl + 1);
            float dum = (poz[lvl] - poz[lvl + 1]) / (rl - rl1);
            pblh = eeps_max(poz[lvl + 1], poz[lvl] - (rl - EEPS_RICR) * dum);
            pblh = eeps_min(pblh, poz[lvl]);
            kpbl = lvl;
        }
#undef RICHB0
    }

    // :566-598 surface values and the lowest-level TKE/epsilon
    const float tkm = TD(t3d, km), zkm = poz[km], qkm = pqv[km];
    const float ptkm = pt[km], szvkm = szv[km], cpmkm = cpm[km];
    const float sfa = tsk_c * eeps_pow(EEPS_P1000MB / psfcpa, EEPS_RCP);
    const float dens = psfcpa / (EEPS_RGAS * tkm * (1.f + EEPS_EP1 * qkm));
    // sh(km) = st(km) = 0 (:575-576) are never read: every reader stops at
    // km - 1.
    const float rm2 = ust_c * ust_c / wspd_c;
    const float hs = hfx_c / (dens * cpmkm);
    const float hq = qfx_c / dens;
    const float duj = ust_c * ust_c;
    float pepkm = duj * ust_c / (EEPS_AKV * zkm);
    pepkm = eeps_min(EEPS_MAXPEP, eeps_max(EEPS_MINPEP, pepkm));
    pep[km] = pepkm;
    const float govrth = EEPS_GRV / ptkm;
    const float sflux = hs + hq * EEPS_EP1 * ptkm;
    float bfx0 = eeps_max(sflux, 0.f);
    const float wstar3 = govrth * bfx0 * pblh;
    const float wstar2 = eeps_pow(wstar3 * wstar3, EEPS_T13);    // (w**2)**t13
    // wspd**2. folds to wspd*wspd (a REAL exponent of 2.0)
    const float richkm = EEPS_GRV * zkm
        * eeps_log(szvkm / (sfa * (1.f + EEPS_EP1 * qsfc_c)))
        / (wspd_c * wspd_c);
    rich[km] = richkm;
    dum2[km] = eeps_dum2(richkm);
    const bool unstable = (rmol_c < 0.0f) && (sflux > 0.0f);
    float pekkm;
    if (unstable) {
        float zr = -zkm * rmol_c;
        pekkm = 3.75f * duj + 0.2f * wstar2
            + duj * eeps_pow(zr * zr, EEPS_T13);
    } else {
        pekkm = 3.75f * duj;
    }
    pekkm = eeps_min(EEPS_MAXPEK, eeps_max(EEPS_MINPEK, pekkm));
    pek[km] = pekkm;

    // :601-623 PBLH again with the thermal excess
    if (unstable) {
        float dss = szvkm + eeps_min(5.0f, EEPS_BT * sflux
                                     / (eeps_max(0.1f, sqrtf(wstar2))));
#define RICHB1(k) (poz[k] * (szv[k] / dss - 1.0f) * UP2(k))
        int lvl = km;
        for (int k = km - 1; k >= 5; --k) {
            if (RICHB1(k) >= EEPS_RICR) { lvl = k; break; }
        }
        if (lvl < km) {
            float rl = RICHB1(lvl), rl1 = RICHB1(lvl + 1);
            float dum = (poz[lvl] - poz[lvl + 1]) / (rl - rl1);
            pblh = eeps_max(poz[lvl + 1], poz[lvl] - (rl - EEPS_RICR) * dum);
            pblh = eeps_min(pblh, poz[lvl]);
            kpbl = lvl;
        }
#undef RICHB1
    }

    // :626-635 countergradient terms
    float gamat, gamaq;
    if (unstable) {
        float du = eeps_max(0.1f, sqrtf(wstar2));
        gamat = eeps_max(EEPS_BT * hs / (pblh * du), 0.f);
        gamaq = eeps_max(EEPS_BT * hq / (pblh * du), 0.f);
    } else {
        gamat = 0.0f;
        gamaq = 0.0f;
    }

    // :639-777 top-down source under stratocumulus (bl_eeps_topdown = 1).
    // Every loop after the EIS test acts only while SCUFLG holds, so a
    // column that fails it leaves TKEprodTD at its zero start (:447).
    bool scuflg = true;
    int kcld = km;
    float zminrad = poz[km];
    const float zl1 = poz[km];
    float tkp_coef = 0.f;
    {
        bool flg = true;
        float minrad = 100.f;
        int kminrad = km;
        float radsum = 0.f;     // EEPS_RADSUM_INIT: uninitialised in WRF
        float ent_eff = 0.f, wm3 = 0.f;
        // ktop = minval(k700) over WRF's tile bounds only the k loops below,
        // and every condition inside them already requires k >= kcld >=
        // k700; this column's own k700 is the same bound.
        const int ktop = k700;
#define HE(k) (EEPS_CP * TD(t3d, k) * (1.f + 5.87f * (pqv[k] + (pqc[k] + pqi[k]))) \
               - EEPS_XLV * pqc[k] - EEPS_XLS * pqi[k] + EEPS_GRV * poz[k])
        {
            float he1 = HE(k700);
            float he2 = HE(k950);
            float he0 = HE(km);
            float eis = eeps_max(he1 - he2, he2 - he0);
            eis = eis / EEPS_CP;
            if (eis < 5.0f) scuflg = false;
        }
#undef HE
        if (scuflg) {
            flg = scuflg;
            for (int k = ktop; k <= km; ++k) {
                if (flg && k >= k700 && (pqc[k] + pqi[k]) >= 1.e-6f) {
                    kcld = k;
                    flg = false;
                }
            }
            for (int k = ktop; k <= km; ++k) {
                if (k >= kcld && (pqc[k] + pqi[k]) >= 1.e-6f) {
                    const float rr = TD(rthraten3d, k);
                    if (rr < minrad) {
                        minrad = rr;
                        kminrad = k;
                        zminrad = poz[k];
                    }
                }
            }
            if (scuflg && minrad >= 0.f) scuflg = false;
            if (scuflg && kcld < kpbl) scuflg = false;
            if (scuflg && kminrad >= (km - 1)) scuflg = false;
        }

        if (scuflg) {
            const int k = kminrad;
            const float prs = TD(p3d, k);
            const float pci = pqc[k] + pqi[k];
            float templ = TD(t3d, k);
            float rvls = 100.f * 6.112f
                * eeps_exp(17.67f * (templ - 273.16f) / (templ - 29.65f))
                * (EEPS_EP2 / prs);
            float temps = templ + (pqv[k] + pci - rvls)
                / (EEPS_CP / EEPS_XLV
                   + EEPS_EP2 * EEPS_XLV * rvls / (EEPS_RGAS * (templ * templ)));
            rvls = 100.f * 6.112f
                * eeps_exp(17.67f * (temps - 273.15f) / (temps - 29.65f))
                * (EEPS_EP2 / prs);
            float rcldb = eeps_max(pqv[k] + pci - rvls, 0.f);
            // Native SZV(K-2) is undefined when K < 3. Refuse this
            // column at the next health drain without an invalid load.
            if (k < 3) flags |= 2u;
            float dthvx = (k >= 3) ? szv[k - 2] - szv[k] : nanf("");
            dthvx = eeps_max(dthvx, 0.1f);
            float tmp1 = EEPS_XLV / EEPS_CP * rcldb / (psp[k] * dthvx);
            ent_eff = 0.2f + 0.2f * 8.f * tmp1;
            for (int kk = km; kk >= ktop; --kk) {
                if (kk <= (kminrad + 1) && kk >= kcld) {
                    float radflux = TD(rthraten3d, kk) * psp[kk];
                    radflux = radflux * EEPS_CP / EEPS_GRV
                        * (TD(p3d, kk) - TD(p3d, kk - 1));
                    if (radflux < 0.0f) radsum = fabsf(radflux) + radsum;
                }
            }
            radsum = eeps_min(radsum, 120.0f);
            float b0 = eeps_max(radsum / TD(rho3d, k) / EEPS_CP, 0.f);
            wm3 = EEPS_GRV / szv[k] * b0 * eeps_min(pblh, 1500.f);
            // :770-776 evaluates 2*ent_eff*wm3/max(pblh,100)*zfacent left
            // to right; its leading product is the same for every level.
            tkp_coef = 2.f * ent_eff * wm3 / eeps_max(pblh, 100.f);
        }
    }
    // TKEprodTD(k) (:770-776), formed where it is read.
#define TKEPROD(k, pozk) ((scuflg && (k) >= kcld) ? eeps_tkeprod(pozk) : 0.0f)
    auto eeps_tkeprod = [&](float pozk) -> float {
        float zfac = eeps_min(eeps_max(
            (1.f - (pozk - zl1) / (zminrad - zl1)), EEPS_ZFMIN), 1.f);
        float om = 1.f - zfac;
        float zfacent = 10.f
            * eeps_max((zminrad - pozk) / zminrad, 0.0f)
            * ((om * om) * om);
        float t = tkp_coef * zfacent;
        return eeps_max(t, 0.0f);
    };

    // :796-898 time integration of TKE and epsilon, MDT substeps.  The
    // source terms and AM of a substep come from its starting PEK/PEP, so
    // the epsilon sweep forms them (and stores the two the TKE solve needs)
    // before any PEK/PEP of the substep changes.
    const int m = km - 1;
    // Row j's clamped KM, SRC and SRC1 (:800-830).
    auto eeps_row = [&](int j, float& akm_o, float& src_o, float& src1_o) {
        const float pekj = pek[j];
        float akmj = EEPS_C2 * pekj * pekj / pep[j];
        float akhj = dum2[j] * akmj;
        akmj = 0.1f + akmj;          // sgk1 = 0.1
        akhj = 0.01f + akhj;         // sgk2 = 0.01
        akmj = eeps_min(EEPS_AKVMX, akmj);
        akhj = eeps_min(EEPS_AKVMX, akhj);
        const float stj = st[j], shj = sh[j];
        float edrain;
        if (stj > 0.0f) {
            float bvf = sqrtf(stj);
            float rbdts = eeps_min(1.f, sqrtf(rich[j] / 0.145f));
            edrain = 0.1f / EEPS_C3 * rbdts * bvf * pekj;
        } else {
            edrain = 0.0f;
        }
        const float pozj = poz[j];
        const float tkp = TKEPROD(j, pozj);
        if (pozj > pblh) {
            src_o = akmj * shj - akhj * stj + tkp;
            src1_o = akmj * shj - akhj * stj + edrain + tkp;
        } else {
            float cg = 2.0f * EEPS_GRV * gamat / (szv[j + 1] + szv[j]);
            src_o = akmj * shj - akhj * (stj - cg) + tkp;
            src1_o = akmj * shj - akhj * (stj - cg) + edrain + tkp;
        }
        akm_o = akmj;
    };
    for (int lvl = 1; lvl <= mdt; ++lvl) {
        // epsilon (:846-867): rows m..1 of the descending elimination.
        // akm(km) is the bare C2*PEK**2/PEP (:832); am(km) mixes it with
        // the clamped akm(km-1) (:836-838).
        const float pekm_s = pek[km];
        const float akm_top = EEPS_C2 * pekm_s * pekm_s / pep[km];
        float akm_k, src_k, src1_k;
        eeps_row(m, akm_k, src_k, src1_k);
        float am_k1 = 0.5f * (akm_k + akm_top) / DZZ(km);   // am(k+1)
        am[km] = am_k1;
        const float yy_m_add = EEPS_C5 * (dtt / DOZ(km - 1)) * am_k1 * pep[km];
        float q_prev = 0.f, d_prev = 0.f;
        for (int k = m; k >= 1; --k) {
            float am_k, akm_n = 0.f, src_n = 0.f, src1_n = 0.f;
            if (k >= 2) {
                eeps_row(k - 1, akm_n, src_n, src1_n);
                am_k = 0.5f * (akm_n + akm_k) / DZZ(k);
            } else {
                am_k = 0.5f * akm_k / DZZ(1);
            }
            am[k] = am_k;
            src[k] = src_k;
            const float amt = dtt / DOZ(k);
            const float pepk = pep[k];
            float dum = (EEPS_C3 * src1_k - EEPS_C4 * pepk) * dtt / pek[k];
            const float ax = -EEPS_C5 * amt * am_k1;
            const float cc = (k == 1) ? 0.0f : -EEPS_C5 * amt * am_k;
            float bx, y;
            if (dum <= 0.5f) {
                bx = 1.0f + EEPS_C5 * amt * (am_k + am_k1) - dum;
                y = pepk;
            } else {
                bx = 1.0f + EEPS_C5 * amt * (am_k + am_k1);
                y = pepk + dum * pepk;
            }
            if (k == m) {
                y = y + yy_m_add;
                q_prev = -cc / bx;
                d_prev = y / bx;
            } else {
                const float p = 1.f / (bx + ax * q_prev);
                q_prev = -cc * p;
                d_prev = (y - ax * d_prev) * p;
            }
            qw[k] = q_prev;
            yy[k] = d_prev;
            akm_k = akm_n;
            src_k = src_n;
            src1_k = src1_n;
            am_k1 = am_k;
        }
        // back substitution; PEP clamped as the TKE loop does (:873).
        {
            float d = yy[1];
            pep[1] = eeps_min(EEPS_MAXPEP, eeps_max(EEPS_MINPEP, d));
            for (int k = 2; k <= m; ++k) {
                d = yy[k] + qw[k] * d;
                pep[k] = eeps_min(EEPS_MAXPEP, eeps_max(EEPS_MINPEP, d));
            }
        }

        // TKE (:872-895)
        {
            float amk1 = am[km];
            const float yy_m_tke = EEPS_C1 * (dtt / DOZ(km - 1)) * amk1 * pek[km];
            float q_p = 0.f, d_p = 0.f;
            for (int k = m; k >= 1; --k) {
                const float amk = am[k];
                const float amt = dtt / DOZ(k);
                float dum = (src[k] - pep[k]) * dtt;
                const float ax = -EEPS_C1 * amt * amk1;
                const float bx = 1.0f + EEPS_C1 * amt * (amk + amk1);
                const float cc = (k == 1) ? 0.0f : -EEPS_C1 * amt * amk;
                float y = pek[k] + dum;
                if (k == m) {
                    y = y + yy_m_tke;
                    q_p = -cc / bx;
                    d_p = y / bx;
                } else {
                    const float p = 1.f / (bx + ax * q_p);
                    q_p = -cc * p;
                    d_p = (y - ax * d_p) * p;
                }
                qw[k] = q_p;
                yy[k] = d_p;
                amk1 = amk;
            }
            float d = yy[1];
            pek[1] = eeps_min(EEPS_MAXPEK, eeps_max(EEPS_MINPEK, d));
            for (int k = 2; k <= m; ++k) {
                d = yy[k] + qw[k] * d;
                pek[k] = eeps_min(EEPS_MAXPEK, eeps_max(EEPS_MINPEK, d));
            }
        }
    }

    // :905-927 diffusivities at interface k, :931-1010 the momentum, heat
    // and moisture solves, rows km..1 of the descending elimination.  The
    // momentum rows serve U and V, the heat rows PT, QV, QC and QI
    // (tridiag leaves A and B alone; the C(1) it zeroes is never read).
    // exch_h(k) = k_h(km + 2 - k) on WRF's kts..kte+1 full levels, with
    // k_h(1) = k_h(km+1) = 0 and k_h(k) = akh(k)*doz(k-1) (:1034-1046).
    {
        auto eeps_iface = [&](int k, float& akm_o, float& akh_o, float& pnt_o) {
            const float pekk = pek[k - 1];
            float akmk = EEPS_C2 * pekk * pekk / pep[k - 1];
            float akhk = dum2[k] * akmk;
            akmk = 0.1f + akmk;
            akhk = 0.01f + akhk;
            akmk = eeps_min(EEPS_AKVMX, akmk);
            akhk = eeps_min(EEPS_AKVMX, akhk);
            float pntk = akhk;
            if (poz[k] > pblh) pntk = 0.f;
            const float doz = DOZ(k - 1);
            akm_o = akmk / doz;
            akh_o = akhk / doz;
            pnt_o = pntk;
        };
        exch_h[(size_t)0 * (size_t)n + col] = 0.f;
        exch_m[(size_t)0 * (size_t)n + col] = 0.f;
        exch_h[(size_t)km * (size_t)n + col] = 0.f;
        exch_m[(size_t)km * (size_t)n + col] = 0.f;
        float akm1 = 0.f, akh1 = 0.f, pnt1 = 0.f;    // interface k + 1
        float akmk, akhk, pntk;
        eeps_iface(km, akmk, akhk, pntk);
        float qm_p = 0.f, qh_p = 0.f;
        float du_p = 0.f, dv_p = 0.f, dt_p = 0.f, dq_p = 0.f, dc_p = 0.f, di_p = 0.f;
        for (int k = km; k >= 1; --k) {
            {
                const float kh = (k >= 2) ? akhk * DOZ(k - 1) : 0.f;
                const float kmv = (k >= 2) ? akmk * DOZ(k - 1) : 0.f;
                if (!(isfinite(kh) && isfinite(kmv))) flags |= 1u;
                if (k >= 2) {
                    exch_h[(size_t)(km + 1 - k) * (size_t)n + col] = kh;
                    exch_m[(size_t)(km + 1 - k) * (size_t)n + col] = kmv;
                }
            }
            const float alt = dt / DZZ(k);
            // momentum (:931-950)
            const float ax = -alt * akm1;
            float bx = 1.0f + alt * (akmk + akm1);
            const float cc = (k == 1) ? 0.0f : -alt * akmk;
            if (k == km) bx = bx + (dt / DZZ(km)) * rm2;
            // heat and moisture (:954-1010)
            const float ah = -alt * akh1;
            const float bh = 1.0f + alt * (akhk + akh1);
            const float ch = (k == 1) ? 0.0f : -alt * akhk;
            float dum0 = 0.0f;
            if (k > 1) dum0 = pep[k - 1];
            float dum = pep[k];
            if (dum0 <= 1.e-6f) dum0 = 0.0f;
            if (dum <= 1.e-6f) dum = 0.0f;
            const float disht = 0.5f * (dum0 + dum) / cpm[k];
            float rt = pt[k] + disht * dt / psp[k]
                + alt * gamat * (pnt1 - pntk);
            float rq = pqv[k] + alt * gamaq * (pnt1 - pntk);
            if (k == km) {
                rt = rt + (dt / DZZ(km)) * hs;
                rq = rq + (dt / DZZ(km)) * hq;
            }
            const float ru = TD(u3d, k), rv = TD(v3d, k);
            const float rc = pqc[k], ri = pqi[k];
            if (k == km) {
                qm_p = -cc / bx;
                du_p = ru / bx;
                dv_p = rv / bx;
                qh_p = -ch / bh;
                dt_p = rt / bh;
                dq_p = rq / bh;
                dc_p = rc / bh;
                di_p = ri / bh;
            } else {
                const float p = 1.f / (bx + ax * qm_p);
                qm_p = -cc * p;
                du_p = (ru - ax * du_p) * p;
                dv_p = (rv - ax * dv_p) * p;
                const float ph = 1.f / (bh + ah * qh_p);
                qh_p = -ch * ph;
                dt_p = (rt - ah * dt_p) * ph;
                dq_p = (rq - ah * dq_p) * ph;
                dc_p = (rc - ah * dc_p) * ph;
                di_p = (ri - ah * di_p) * ph;
            }
            qw[k] = qm_p;
            qh[k] = qh_p;
            pu[k] = du_p;
            pv[k] = dv_p;
            pt[k] = dt_p;
            pqv[k] = dq_p;
            pqc[k] = dc_p;
            pqi[k] = di_p;
            akm1 = akmk; akh1 = akhk; pnt1 = pntk;
            if (k >= 2) {
                if (k - 1 >= 2) {
                    eeps_iface(k - 1, akmk, akhk, pntk);
                } else {
                    akmk = 0.f; akhk = 0.f; pntk = 0.f;     // :937-939
                }
            }
        }
    }

    // back substitution, :1012-1032 back to temperature with immediate
    // melting of cloud ice, and the driver's return to the bottom-up grid.
    const float rdelt = 1.f / dt;
    float du_p = 0.f, dv_p = 0.f, dt_p = 0.f, dq_p = 0.f, dc_p = 0.f, di_p = 0.f;
    for (int kk = 1; kk <= km; ++kk) {
        float uu = pu[kk], vv = pv[kk], tt = pt[kk];
        float q1 = pqv[kk], q2 = pqc[kk], q3 = pqi[kk];
        if (kk >= 2) {
            const float qm = qw[kk], qq = qh[kk];
            uu = uu + qm * du_p;
            vv = vv + qm * dv_p;
            tt = tt + qq * dt_p;
            q1 = q1 + qq * dq_p;
            q2 = q2 + qq * dc_p;
            q3 = q3 + qq * di_p;
        }
        du_p = uu; dv_p = vv; dt_p = tt; dq_p = q1; dc_p = q2; di_p = q3;
        const float cpmk = cpm[kk];
        float tz = tt * psp[kk];
        q1 = eeps_max(EEPS_EPSL, q1);
        q2 = eeps_max(EEPS_EPSL, q2);
        q3 = eeps_max(EEPS_EPSL, q3);
        if (tz >= EEPS_T0 && q3 > EEPS_EPSL) {
            if (q2 <= EEPS_EPSL) {
                q2 = q3;
            } else {
                q2 = q2 + q3;
            }
            float alh = 3.1484e6f - 2.37e3f * tz;
            tz = tz - (EEPS_XLS - alh) * q3 / cpmk;
            q3 = EEPS_EPSL;
        }
        const int k = km + 1 - kk;
        const float pekk = pek[kk], pepk = pep[kk];
        const float tth = (tz - AT(t3d, k)) / AT(pi3d, k) * rdelt;
        const float tqv = (q1 / (1.f - q1) - AT(qv3d, k)) * rdelt;
        const float tqc = (q2 / (1.f - q2) - AT(qc3d, k)) * rdelt;
        const float tqi = (q3 / (1.f - q3) - AT(qi3d, k)) * rdelt;
        const float tu = (uu - AT(u3d, k)) * rdelt;
        const float tv = (vv - AT(v3d, k)) * rdelt;
        AT(rthblten, k) = tth;
        AT(rqvblten, k) = tqv;
        AT(rqcblten, k) = tqc;
        AT(rqiblten, k) = tqi;
        AT(rublten, k) = tu;
        AT(rvblten, k) = tv;
        AT(pek3d, k) = pekk;
        AT(pep3d, k) = pepk;
        AT(pek_adv, k) = pekk;      // bl_eeps_tkeadv == 1 (:348-355)
        AT(pep_adv, k) = pepk;
        if (!(isfinite(tu) && isfinite(tv) && isfinite(tth) && isfinite(tqv)
              && isfinite(tqc) && isfinite(tqi)
              && isfinite(pekk) && isfinite(pepk))) flags |= 1u;
    }
    pblh_out[col] = pblh;
    kpbl_out[col] = km - kpbl + 1;
    if (!isfinite(pblh)) flags |= 1u;
    if (flags) atomicOr(status, flags);
#undef AT
#undef TD
#undef SLOT
#undef UP2
#undef DZZ
#undef DOZ
#undef TKEPROD
}
