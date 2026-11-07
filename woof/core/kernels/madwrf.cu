// MAD-WRF (mp_physics = 96): cloud initialization and tracer nudging.
//
// WRF v4.7.1 f52c197ed39d12e087d02c50f412d90d418f6186.
// Cloud module byte-identical to v4.6.1; reuse of that port.
// WRF ships the original under its public-domain statement.
//   phys/module_madwrf.F:71-204   Init_madwrf_clouds (one thread per column)
//     :241-558  cal_cldfra3_madwrf    :560-626  find_k / find_thresh_k_*
//     :633-797  find_cloudLayers      :799-868  Calc_tropo_height
//     :870-992  adjust_cloudIce/H2O/Final       :23-69  Calc_cldtopz_from_brtemp
//   dyn_em/module_big_step_utilities_em.F:6711-6756  Cloud_tracer_nudge
//   phys/module_mp_thompson.F:5383-5444  RSLF / RSIF
//
// FP32 in WRF's operation order.  Compiled WITHOUT contraction and WITHOUT
// flush-to-zero (the shared gpuwm/core/kernels loader compiles this unit with
// --fmad=false --ftz=false), so every product and sum rounds once, as
// gfortran's do, and a subnormal hydrometeor word is kept, as WRF keeps it.
// REAL ** REAL is WOOF's own float32 powf, gfk_pow (glibc_flt32.cuh, which
// the loader prepends); sqrtf and '/' are IEEE correctly rounded.
//
// Column arrays: state fields are (k, j, i) with plane stride SP = ny*nx;
// the intermediates (ts, dzs, theta, qvs, rh, rhoa) live in a caller-owned
// scratch of (nl, ncol_chunk) with stride SS.  Level indices below are
// WRF's 1-based k throughout; nl is the number of mass layers, kte = nl+1.
//
// The one declared divergence (gpuwm/core/madwrf.py): find_k_from_z_agl
// searches k = 1..nl, not WRF's 1..kte, whose last iteration reads dz one
// element past Init_madwrf_clouds' automatic array.

#define MW_MISSING (-999.9f)
#define MW_NO_LEVEL (-999)

struct MwV {
    float* a;
    long long s;
    __device__ __forceinline__ float& operator()(int k) const {
        return a[(long long)(k - 1) * s];
    }
};

__device__ __forceinline__ float mw_rslf(float p, float t) {
    const float c0 = .611583699E03f, c1 = .444606896E02f,
                c2 = .143177157E01f, c3 = .264224321E-1f,
                c4 = .299291081E-3f, c5 = .203154182E-5f,
                c6 = .702620698E-8f, c7 = .379534310E-11f,
                c8 = -.321582393E-13f;
    const float x = fmaxf(-80.f, t - 273.16f);
    float esl = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5
        + x * (c6 + x * (c7 + x * c8)))))));
    esl = fminf(esl, p * 0.15f);
    return .622f * esl / (p - esl);
}

__device__ __forceinline__ float mw_rsif(float p, float t) {
    const float c0 = .609868993E03f, c1 = .499320233E02f,
                c2 = .184672631E01f, c3 = .402737184E-1f,
                c4 = .565392987E-3f, c5 = .521693933E-5f,
                c6 = .307839583E-7f, c7 = .105785160E-9f,
                c8 = .161444444E-12f;
    const float x = fmaxf(-80.f, t - 273.16f);
    float esi = c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5
        + x * (c6 + x * (c7 + x * c8)))))));
    esi = fminf(esi, p * 0.15f);
    return .622f * esi / fmaxf(1.e-4f, p - esi);
}

// Fortran MAX/MIN on finite operands.
__device__ __forceinline__ float mw_max(float a, float b) { return a > b ? a : b; }
__device__ __forceinline__ float mw_min(float a, float b) { return a < b ? a : b; }

__device__ void mw_theta(const MwV& t, const MwV& p, const MwV& th, int nl) {
    const float expo = 287.05f / 1004.f;
    for (int k = nl; k >= 1; --k)
        th(k) = t(k) * gfk_pow(100000.0f / p(k), expo);
}

// :799-868 Calc_tropo_height.  th must hold theta(t, p) for k = 1..nl.
__device__ void mw_tropo(const MwV& p, const MwV& dz, const MwV& th, int nl,
                         int& k_tropo, float& tropo_z) {
    const int kte = nl + 1;
    int k_p200 = 0;
    for (int k = kte - 1; k >= 1; --k)
        if (p(k) > 19999.0f && k_p200 == 0) k_p200 = k;
    if ((kte - k_p200) < 3) k_p200 = kte - 3;
    const float lapse = 10.f / 1500.f;
    int k;
    for (k = k_p200 - 2; k >= 1; --k) {
        const float theta1 = th(k), theta2 = th(k + 2);
        const float delz = 0.5f * dz(k) + dz(k + 1) + 0.5f * dz(k + 2);
        if (((theta2 - theta1) / delz) < lapse || p(k) > 70000.f) break;
    }
    k_tropo = max(3, min(k + 2, kte - 1));
    if (k_tropo > k_p200) {
        for (k = kte - 3; k >= k_p200 - 2; --k) {
            const float theta1 = th(k), theta2 = th(k + 2);
            const float delz = 0.5f * dz(k) + dz(k + 1) + 0.5f * dz(k + 2);
            if (((theta2 - theta1) / delz) < lapse && p(k) > 9500.f) break;
        }
        k_tropo = max(k_p200 - 1, min(k + 2, kte - 1));
    }
    float s = 0.0f;
    for (int kk = 1; kk <= k_tropo; ++kk) s = s + dz(kk);
    tropo_z = s;
}

// :560-584 over the model's own layers (the declared divergence).
__device__ void mw_find_k(float z_agl, int& k_lev, const MwV& dz, int nl) {
    float z_full = 0.0f;
    for (int k = 1; k <= nl; ++k) {
        const float z_this = z_full;
        const float z_next = z_this + dz(k);
        if (z_agl > 0.0f && z_agl > z_this && z_agl <= z_next) k_lev = k;
        z_full = z_next;
    }
}

// :870-948 adjust_cloudIce (qs non-null) / adjust_cloudH2O (qs null).
__device__ void mw_adjust(const MwV& cfr, const MwV& q, const MwV* qs,
                          const MwV& qvs, const MwV& t, const MwV& dz,
                          int k1, int k2, float t_min) {
    const float entr = 0.5f;
    float tdz = 0.f;
    for (int k = k1; k <= k2; ++k) tdz = tdz + dz(k);
    float max_c = fabsf(qvs(k2) - qvs(k1));
    for (int k = k1; k <= k2; ++k)
        max_c = mw_max(1.E-6f, max_c - (qs ? q(k) + (*qs)(k) : q(k)));
    max_c = mw_min(1.E-3f, max_c);
    float this_dz = 0.0f;
    for (int k = k1; k <= k2; ++k) {
        if (k == k1) this_dz = this_dz + 0.5f * dz(k);
        else this_dz = this_dz + dz(k);
        const float this_c = max_c * this_dz / tdz;
        const float c = mw_max(1.E-6f, this_c * (1.f - entr));
        if (cfr(k) > 0.0f && cfr(k) < 1.0f && t(k) >= t_min)
            q(k) = q(k) + cfr(k) * cfr(k) * c;
    }
}

// :633-797 find_cloudLayers, called without ktropo.  Returns false where
// WRF calls wrf_error_fatal (:673-681).
__device__ bool mw_find_cloud_layers(const MwV& qvs, const MwV& cfr,
        const MwV& t, const MwV& p, const MwV& dz, const MwV& qc,
        const MwV& qi, const MwV& qs, const MwV& th, int nl, bool keep) {
    const int kte = nl + 1;
    int k_m12C = 0, k_m40C = 0;
    mw_theta(t, p, th, nl);
    for (int k = kte - 1; k >= 1; --k) {
        if (t(k) - 273.16f > -40.0f && p(k) > 7000.0f) k_m40C = max(k_m40C, k);
        if (t(k) - 273.16f > -12.0f && p(k) > 10100.0f) k_m12C = max(k_m12C, k);
    }
    if (k_m40C <= 1) k_m40C = 1;
    if (k_m12C <= 1) k_m12C = 1;
    if (k_m40C > kte - 2 || k_m12C > kte - 3) return false;
    int k_tropo;
    float tropo_z;
    mw_tropo(p, dz, th, nl, k_tropo, tropo_z);
    for (int k = k_tropo + 1; k <= kte - 1; ++k)
        if (cfr(k) > 0.0f && cfr(k) < 1.0f) cfr(k) = 0.f;
    int kbot = 3, k;
    for (k = kbot; k <= k_m12C; ++k)
        if ((th(k) - th(k - 1)) > 0.010E-3f * dz(k)) break;
    kbot = max(2, k - 2);
    if (!keep) {
        for (k = 1; k <= kbot; ++k)
            if (cfr(k) > 0.0f && cfr(k) < 1.0f) cfr(k) = 0.5f * cfr(k);
    } else {
        kbot = 2;
    }
    // Ice layers, :722-759.
    int k_cldb = k_tropo, k_cldt;
    bool in_cloud = false;
    k = k_tropo;
    while (!in_cloud && k > k_m12C + 1) {
        k_cldt = 0;
        if (cfr(k) >= 0.01f) { in_cloud = true; k_cldt = max(k_cldt, k); }
        if (in_cloud) {
            for (int k2 = k_cldt - 1; k2 >= k_m12C; --k2)
                if (cfr(k2) < 0.01f || k2 == k_m12C) { k_cldb = k2 + 1; break; }
            in_cloud = false;
        }
        if ((k_cldt - k_cldb + 1) >= 2) {
            mw_adjust(cfr, qi, &qs, qvs, t, dz, k_cldb, k_cldt, 203.16f);
            k = k_cldb;
        } else if ((k_cldt - k_cldb + 1) == 1) {
            if (cfr(k_cldb) > 0.f && cfr(k_cldb) < 1.f)
                qi(k_cldb) = 0.05f * qvs(k_cldb) * cfr(k_cldb) * cfr(k_cldb);
            k = k_cldb;
        }
        k = k - 1;
    }
    // Water layers, :762-795.
    k_cldb = k_m12C + 3;
    in_cloud = false;
    k = k_m12C + 2;
    while (!in_cloud && k > kbot) {
        k_cldt = 0;
        if (cfr(k) >= 0.01f) { in_cloud = true; k_cldt = max(k_cldt, k); }
        if (in_cloud) {
            for (int k2 = k_cldt - 1; k2 >= kbot; --k2)
                if (cfr(k2) < 0.01f || k2 == kbot) { k_cldb = k2 + 1; break; }
            in_cloud = false;
        }
        if ((k_cldt - k_cldb + 1) >= 2) {
            mw_adjust(cfr, qc, nullptr, qvs, t, dz, k_cldb, k_cldt, 253.16f);
            k = k_cldb;
        } else if ((k_cldt - k_cldb + 1) == 1) {
            if (cfr(k_cldb) > 0.f && cfr(k_cldb) < 1.f)
                qc(k_cldb) = 0.05f * qvs(k_cldb) * cfr(k_cldb) * cfr(k_cldb);
            k = k_cldb;
        }
        k = k - 1;
    }
    return true;
}

// :955-992 adjust_cloudFinal.
__device__ void mw_adjust_final(const MwV& cfr, const MwV& qc, const MwV& qi,
                                const MwV& rho, const MwV& dz, int nl) {
    float lwp = 0.f, iwp = 0.f;
    for (int k = 1; k <= nl; ++k)
        if (cfr(k) > 0.0f && cfr(k) < 1.0f) {
            lwp = lwp + qc(k) * rho(k) * dz(k);
            iwp = iwp + qi(k) * rho(k) * dz(k);
        }
    if (lwp > 1.0f) {
        const float xfac = 1.0f / lwp;
        for (int k = 1; k <= nl; ++k)
            if (cfr(k) > 0.0f && cfr(k) < 1.0f) qc(k) = qc(k) * xfac;
    }
    if (iwp > 1.0f) {
        const float xfac = 1.0f / iwp;
        for (int k = 1; k <= nl; ++k)
            if (cfr(k) > 0.0f && cfr(k) < 1.0f) qi(k) = qi(k) * xfac;
    }
}

// :23-69 Calc_cldtopz_from_brtemp.
__device__ float mw_cldtopz_from_brtemp(float cldmask, const MwV& ts,
        const MwV& dzs, float brtemp, float tropoz, float ht, int nl) {
    const int kte = nl + 1;
    float rv = MW_MISSING;
    if (brtemp == MW_MISSING) return rv;
    if (cldmask > 0.0f) {
        float s = 0.0f;
        for (int k = 1; k <= kte - 1; ++k) s = s + dzs(k);
        float this_height = s + ht;
        for (int k = kte - 2; k >= 1; --k) {
            this_height = this_height - dzs(k + 1);
            if (this_height > tropoz) continue;
            if (ts(k) > brtemp) {
                rv = this_height + dzs(k) * (brtemp - ts(k)) / (ts(k + 1) - ts(k));
                if ((rv < this_height) || (rv > this_height + dzs(k))) rv = tropoz;
                break;
            }
        }
    }
    return rv;
}

// :241-558 cal_cldfra3_madwrf with modify_qvapor=.true., max_relh=1.5,
// no k_tropo; mode = Init_madwrf_clouds' insert_clouds.
__device__ bool mw_cal_cldfra3(const MwV& cldfra, const MwV& qv, const MwV& qc,
        const MwV& qi, const MwV& qs, const MwV& dz, const MwV& p,
        const MwV& t, float xland, float gridkm, int mode, float cldmask,
        float cldtopz, float cldbasez, const MwV& qvs, const MwV& rh,
        const MwV& rhoa, const MwV& th, int nl) {
    const int kte = nl + 1;
    const bool imp_mask = mode == 1, imp_top = mode == 2, imp_base = mode == 3;
    int cldtopk = MW_NO_LEVEL, cldbasek = MW_NO_LEVEL, thresh_k = MW_NO_LEVEL;
    bool keep = false;
    if (imp_mask && cldmask == 0.0f) {
        for (int k = 1; k <= kte - 1; ++k) {
            cldfra(k) = 0.0f; qc(k) = 0.0f; qi(k) = 0.0f; qs(k) = 0.0f;
        }
        return true;
    }
    if (imp_top || imp_base) {
        if (cldmask == 0.0f) {
            for (int k = 1; k <= kte - 1; ++k) {
                cldfra(k) = 0.0f; qc(k) = 0.0f; qi(k) = 0.0f; qs(k) = 0.0f;
            }
            return true;
        } else if (cldmask > 0.0f && cldmask <= 1.0f && cldtopz > 0.0f) {
            mw_find_k(cldtopz, cldtopk, dz, nl);
            if (cldtopk > 0 && cldtopk < kte)
                for (int k = cldtopk + 1; k <= kte - 1; ++k) {
                    cldfra(k) = 0.0f; qc(k) = 0.0f; qi(k) = 0.0f; qs(k) = 0.0f;
                }
            // CLDTHICK_DEF = -999.9 < 0: the thickness branch never runs.
        }
    }
    if (imp_base) {
        if (cldmask > 0.0f && cldmask <= 1.0f && cldbasez > 0.0f) {
            mw_find_k(cldbasez, cldbasek, dz, nl);
            if (cldtopk < cldbasek) cldbasek = MW_NO_LEVEL;
            if (cldbasek > 0 && cldbasek > 1)
                for (int k = 1; k <= cldbasek - 1; ++k) {
                    cldfra(k) = 0.0f; qc(k) = 0.0f; qi(k) = 0.0f; qs(k) = 0.0f;
                }
        }
    }
    for (int k = 1; k <= kte - 1; ++k) {
        cldfra(k) = 0.0f;
        float tc = t(k) - 273.15f;
        const float qvsw = mw_rslf(p(k), t(k));
        float qvsi;
        if (tc < 0.0f) qvsi = mw_rsif(p(k), t(k) + 0.025f);
        else qvsi = qvsw;
        if (tc >= -12.0f) qvs(k) = qvsw;
        else if (tc < -35.0f) qvs(k) = qvsi;
        else qvs(k) = qvsw - __fdiv_rn((qvsw - qvsi) * (-12.0f - tc), (-12.0f + 35.f));
        if (qc(k) > 1.E-8f) { qv(k) = mw_max(qv(k), qvsw); qvs(k) = qvsw; }
        if (qc(k) <= 1.E-8f && qi(k) >= 1.E-9f) {
            qv(k) = mw_max(qv(k), qvsi); qvs(k) = qvsi;
        }
        rh(k) = mw_max(0.01f, qv(k) / qvs(k));
        rhoa(k) = p(k) / (287.0f * t(k));
    }
    const bool ocean = (xland - 1.5f) > 0.0f;
    if ((imp_top || imp_base) && ocean && cldtopk > 0 && cldtopk < kte) {
        if (cldbasek < 0 || cldbasek > cldtopk) {
            if (rh(cldtopk) < 0.3f) cldbasek = cldtopk + 1;
            else cldbasek = cldtopk;
        } else {
            float tmp = (float)(cldtopk + 1);
            for (int k = cldtopk; k >= 1; --k) {
                if (rh(k) < 0.3f) break;
                tmp = (float)k;
            }
            if (tmp > (float)cldbasek) cldbasek = (int)tmp;
        }
    }
    for (int k = 1; k <= kte - 1; ++k) {
        const float delz = mw_max(100.f, dz(k));
        const float rh00l = 0.65f + sqrtf(1.f / (25.0f + gridkm * gridkm * delz * 0.01f));
        const float rh00o = 0.81f + sqrtf(1.f / (50.0f + gridkm * gridkm * delz * 0.01f));
        float rhum = rh(k);
        if (qc(k) > 1.E-7f || qi(k) >= 1.E-7f || (qs(k) > 1.E-6f && t(k) < 273.f)) {
            cldfra(k) = 1.0f;
        } else {
            float rh00 = ocean ? rh00o : rh00l;
            const float tc = t(k) - 273.15f;
            if (tc < -12.0f) rh00 = rh00l;
            float c;
            if (tc >= 25.0f) {
                c = 0.0f;
            } else if (tc >= -12.0f) {
                rhum = mw_min(rh(k), 1.0f);
                c = mw_max(0.f, 1.0f - sqrtf((1.005f - rhum) / (1.005f - rh00)));
            } else {
                rhum = mw_min(rh(k), 1.45f);
                rh00 = rh00 + __fdiv_rn((1.45f - rh00) * (-12.0f - tc), (-12.0f + 100.f));
                c = mw_max(0.f, 1.0f - sqrtf((1.5f - rhum) / (1.5f - rh00)));
            }
            if (c > 0.f) c = mw_max(0.01f, mw_min(c, 0.9f));
            cldfra(k) = c;
        }
    }
    if ((imp_top || imp_base) && cldtopk > 0 && cldtopk < kte) {
        for (int k = cldtopk + 1; k <= kte; ++k) cldfra(k) = 0.0f;
        keep = true;
        for (int k = cldtopk; k >= 1; --k)
            if (cldfra(k) > 0.0f) { thresh_k = k; break; }
        if (thresh_k > 0) {
            if (thresh_k < cldtopk)
                for (int k = thresh_k + 1; k <= cldtopk; ++k) cldfra(k) = 0.5f;
        }
    }
    if (imp_base && cldbasek > 1 && cldbasek <= kte) {
        for (int k = 1; k <= cldbasek - 1; ++k) cldfra(k) = 0.0f;
        thresh_k = MW_NO_LEVEL;
        for (int k = cldbasek; k <= kte; ++k)
            if (cldfra(k) > 0.0f) { thresh_k = k; break; }
        if (thresh_k > cldbasek)
            for (int k = cldbasek; k <= thresh_k - 1; ++k)
                if (rh(k) > 0.3f) cldfra(k) = 0.5f;
    }
    if (!mw_find_cloud_layers(qvs, cldfra, t, p, dz, qc, qi, qs, th, nl, keep))
        return false;
    mw_adjust_final(cldfra, qc, qi, rhoa, dz, nl);
    for (int k = 1; k <= kte - 1; ++k)
        if (cldfra(k) > 0.20f && cldfra(k) < 1.0f) qv(k) = mw_max(qv(k), qvs(k));
    return true;
}

// :71-204 Init_madwrf_clouds for one chunk of rows [j0, j0+nrows).
extern "C" __global__ void madwrf_cldinit(
        const float* __restrict__ t, const float* __restrict__ p,
        const float* __restrict__ ph_2, const float* __restrict__ phb,
        float* qv, float* qc, float* qi, float* qs, float* cldfra,
        const float* __restrict__ xland, const float* __restrict__ ht,
        float* cldmask, float* cldtopz, const float* __restrict__ cldbasez,
        const float* __restrict__ brtemp,
        float* scratch, float dx, float dy, float p00,
        int flag_cldmask, int flag_cldtopz, int flag_brtemp, int mode,
        int em_width, int hold_ups, int nl, int ny, int nx, int j0,
        int nrows, int* status) {
    const int sc = blockIdx.x * blockDim.x + threadIdx.x;
    const int ncol = nrows * nx;
    if (sc >= ncol) return;
    const int j = j0 + sc / nx, i = sc % nx;
    // share/module_soil_pre.F:3084-3110, in 1-based i/j with WRF's
    // ide = nx+1, jde = ny+1.
    if (!(hold_ups || nx + 1 == 3 || ny + 1 == 3)) {
        const int ii = i + 1, jj = j + 1;
        if (ii >= 1 + em_width && ii <= nx - em_width &&
            jj >= 1 + em_width && jj <= ny - em_width) return;
    }
    const long long SP = (long long)ny * nx, SS = ncol;
    const long long c = (long long)j * nx + i;
    const MwV T{const_cast<float*>(t) + c, SP}, P{const_cast<float*>(p) + c, SP};
    const MwV PH{const_cast<float*>(ph_2) + c, SP}, PB{const_cast<float*>(phb) + c, SP};
    const MwV QV{qv + c, SP}, QC{qc + c, SP}, QI{qi + c, SP}, QS{qs + c, SP};
    const MwV CF{cldfra + c, SP};
    const MwV TS{scratch + 0 * SS * nl + sc, SS}, DZ{scratch + 1 * SS * nl + sc, SS};
    const MwV TH{scratch + 2 * SS * nl + sc, SS}, QVS{scratch + 3 * SS * nl + sc, SS};
    const MwV RH{scratch + 4 * SS * nl + sc, SS}, RHOA{scratch + 5 * SS * nl + sc, SS};
    const float gridkm = sqrtf(dx * dy) * 0.001f;
    const float rcp = 287.f / (7.f * 287.f / 2.f);
    for (int k = 1; k <= nl; ++k) {
        TS(k) = (T(k) + 300.f) / gfk_pow(p00 / P(k), rcp);
        DZ(k) = __fdiv_rn((PH(k + 1) + PB(k + 1) - (PH(k) + PB(k))), 9.81f);
    }
    mw_theta(TS, P, TH, nl);
    int k_tropo;
    float tropoz;
    mw_tropo(P, DZ, TH, nl, k_tropo, tropoz);
    const float h = ht[c];
    tropoz = tropoz + h;
    float cm = cldmask[c], ctz = cldtopz[c], agl = MW_MISSING;
    if (mode > 0) {
        if (flag_cldtopz == 1) {
            if (ctz > 0.0f) cm = 1.0f;
            else if (ctz == 0.0f) cm = 0.0f;
            else cm = MW_MISSING;
            if (ctz > 0.0f) agl = mw_max(0.0f, ctz - h);
        } else if (flag_brtemp == 1) {
            ctz = mw_cldtopz_from_brtemp(cm, TS, DZ, brtemp[c], tropoz, h, nl);
            if (ctz > 0.0f) agl = mw_max(0.0f, ctz - h);
        }
        cldmask[c] = cm;
        cldtopz[c] = ctz;
    }
    if (!mw_cal_cldfra3(CF, QV, QC, QI, QS, DZ, P, TS, xland[c], gridkm, mode,
                        cm, agl, cldbasez[c], QVS, RH, RHOA, TH, nl))
        atomicExch(status, 1);
}

// dyn_em/module_big_step_utilities_em.F:6711-6756, the nudging statement.
extern "C" __global__ void madwrf_nudge(
        float* qc, float* qi, float* qs, const float* __restrict__ tr_qc,
        const float* __restrict__ tr_qi, const float* __restrict__ tr_qs,
        float dt, float dt_relax, long long n, int nx, int ny, int sz, int periodic_x) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const int i = idx % nx;
    const int j = (idx / nx) % ny;
    if (sz && (j < sz || j >= ny-sz || (!periodic_x && (i < sz || i >= nx-sz)))) return;
    qc[idx] = qc[idx] + (tr_qc[idx] - qc[idx]) * dt / dt_relax;
    qi[idx] = qi[idx] + (tr_qi[idx] - qi[idx]) * dt / dt_relax;
    qs[idx] = qs[idx] + (tr_qs[idx] - qs[idx]) * dt / dt_relax;
}
