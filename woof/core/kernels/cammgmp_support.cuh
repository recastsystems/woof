// cammgmp_support.cuh -- shared vocabulary of the CAMMGMP (WRF v4.6.1
// mp_physics = 11) port: Fortran -O2 lowering, the saturation functions, the
// WOOF special-function calls. WOOF's own work, Apache-2.0.
//
// The loader prepends glibc_flt64.cuh (WOOF's float64 libm: exp, log, pow,
// cos, acos), uwpbl_common.cuh (the rounding-pinned R8 type, column views,
// the workspace pool, CAM physconst words and the estbl table) and
// uwpbl_wvsat.cuh (wv_saturation estblf) before this header, then
// cammgmp_words.cuh (the CAMMGMP modules' run-time words).
//
// Referee: the SCALAR WRF build (gfortran 13.3 -O2 -fno-fast-math
// -ffp-contract=off -fno-tree-vectorize).  Its lowering, measured on the
// oracle host (tools/cammgmp_wrf461_oracle):
//   * x**2._r8 is x*x, x**1._r8 is x, x**(-1._r8) is 1/x; every other REAL
//     exponent (0.5_r8, 3._r8, ...) calls pow.  -> mg_pow
//   * x**n for an INTEGER constant n is an inline multiply chain:
//       n=2 x*x, n=3 (x*x)*x, n=4 (x*x)*(x*x), n=5 (x*(x*x))*(x*x),
//       n=6 ((x*x)*x)*((x*x)*x).                         -> ip2..ip6
//   * A constant expression (4._r8*atan(1._r8), log(2._r8), ...) is folded
//     at compile time, correctly rounded; it arrives here as a word.
//   * A literal without _r8 is a default (single) REAL: its value is the
//     float32 rounding of the decimal, widened exactly.        -> SP(x)
//   * MAX/MIN lower to maxsd/minsd; for non-NaN operands that are not a
//     +0/-0 pair every lowering agrees, and the column oracle measures the
//     rest.                                                    -> uw_max/min
#ifndef CAMMGMP_SUPPORT_CUH
#define CAMMGMP_SUPPORT_CUH

#define SP(x) R8((double)(x##f))

__device__ __forceinline__ R8 ip2(R8 x) { return x * x; }
__device__ __forceinline__ R8 ip3(R8 x) { R8 t = x * x; return t * x; }
__device__ __forceinline__ R8 ip4(R8 x) { R8 t = x * x; return t * t; }
__device__ __forceinline__ R8 ip5(R8 x) { R8 t = x * x; R8 u = x * t; return u * t; }
__device__ __forceinline__ R8 ip6(R8 x) { R8 t = x * x; R8 u = t * x; return u * u; }

__device__ __forceinline__ R8 mg_pow(R8 x, R8 y) { return uw_pow(x, y); }
__device__ __forceinline__ R8 mg_exp(R8 x) { return uw_exp(x); }
__device__ __forceinline__ R8 mg_log(R8 x) { return uw_log(x); }
__device__ __forceinline__ R8 mg_sqrt(R8 x) { return uw_sqrt(x); }
__device__ __forceinline__ R8 mg_max(R8 a, R8 b) { return uw_max(a, b); }
__device__ __forceinline__ R8 mg_min(R8 a, R8 b) { return uw_min(a, b); }

// log10 for the saturation polynomial, composed from WOOF's float64 log.
// For x in [0.5, 2) it is RN(log(x) * RN(1/ln 10)); above that the exponent
// split log10(2) + log10(x/2) with RN(log10 2).  Measured on the oracle host
// against its libm (tools/cammgmp_wrf461_oracle/README.md): bitwise equal on
// every sampled argument in [0.5, 2) -- which holds all of polysvp's
// 273.16/T and 373.16/T for T >= 186.6 K -- and on 99.56 % of [2, 4)
// (373.16/T for T < 186.6 K); that residual is the declared log10 divergence.
#define MG_INV_LN10 0x1.bcb7b1526e50ep-2   // RN(1/ln 10)
#define MG_LOG10_2  0x1.34413509f79ffp-2   // RN(log10 2)
__device__ __forceinline__ R8 mg_log10(R8 x) {
    if (x.v >= 2.0 && x.v < 4.0)
        return mg_log(x / R8(2.0)) * R8(MG_INV_LN10) + R8(MG_LOG10_2);
    return mg_log(x) * R8(MG_INV_LN10);
}

// wv_saturation polysvp (module_cam_wv_saturation.F:717-754): Goff-Gratch.
// log10(6.1071_r8) and log10(1013.246_r8) are constant arguments, folded by
// the compiler to their correctly rounded words.
#define MG_LOG10_61071   0x1.9258f81f79246p-1   // log10(6.1071_r8)
#define MG_LOG10_1013246 0x1.80bb4409e541bp+1   // log10(1013.246_r8)
__device__ R8 mg_polysvp(R8 t, int type) {
    R8 out = R8(0.0);
    if (type == 1) {
        R8 e = -(R8(9.09718) * (R8(273.16) / t - R8(1.0)))
               - R8(3.56654) * mg_log10(R8(273.16) / t)
               + R8(0.876793) * (R8(1.0) - t / R8(273.16))
               + R8(MG_LOG10_61071);
        out = mg_pow(R8(10.0), e) * R8(100.0);
    }
    if (type == 0) {
        R8 e = -(R8(7.90298) * (R8(373.16) / t - R8(1.0)))
               + R8(5.02808) * mg_log10(R8(373.16) / t)
               - R8(1.3816e-7) * (mg_pow(R8(10.0), R8(11.344) * (R8(1.0) - t / R8(373.16))) - R8(1.0))
               + R8(8.1328e-3) * (mg_pow(R8(10.0), -(R8(3.49149) * (R8(373.16) / t - R8(1.0)))) - R8(1.0))
               + R8(MG_LOG10_1013246);
        out = mg_pow(R8(10.0), e) * R8(100.0);
    }
    return out;
}

// Reuse only complete binary64 inputs. Pressure, ice mass, land and snow
// are fixed for each point during this call. The strict preprocessor path
// retains every original saturation and cloud-fraction call.
#ifdef GPUWM_CAM_FAST
struct MgSaturation {
    V pool;
    int nz;
    __device__ __forceinline__ R8& at(int slot, int k) const {
        return pool(slot * nz + k);
    }
    __device__ __forceinline__ static bool same(R8 a, R8 b) {
        return a.v == b.v && __double_as_longlong(a.v) == __double_as_longlong(b.v);
    }
    __device__ void init() {
        const R8 invalid(__longlong_as_double(0x7ff8000000000000LL));
        for (int k = 1; k <= nz; ++k) {
            at(0, k) = invalid; at(2, k) = invalid; at(4, k) = invalid;
        }
    }
    __device__ __forceinline__ R8 saturation(int k, R8 t, int type) {
        const int slot = type == 0 ? 0 : 2;
        if (same(at(slot, k), t)) return at(slot + 1, k);
        const R8 value = mg_polysvp(t, type);
        at(slot, k) = t;
        at(slot + 1, k) = value;
        return value;
    }
};
#define MG_SAT_DECL , MgSaturation& sat
#define MG_SAT_ARG , sat
#define MG_SAT_POINT_DECL , int sat_k, MgSaturation& sat
#define MG_SAT_POINT_ARG(k) , (k), sat
#define MG_SAT(k,t,type) sat.saturation((k), (t), (type))
#else
#define MG_SAT_DECL
#define MG_SAT_ARG
#define MG_SAT_POINT_DECL
#define MG_SAT_POINT_ARG(k)
#define MG_SAT(k,t,type) mg_polysvp(t, type)
#endif

// wv_saturation vqsatd_water for one point (module_cam_wv_saturation.F:
// 641-715).  Only es and qs reach the CAMMGMP outputs; gam is returned for
// completeness of the call.
__device__ void mg_vqsatd_water(R8 t, R8 p, R8& es, R8& qs, R8& gam MG_SAT_POINT_DECL) {
    R8 omeps = R8(1.0) - R8(WV_EPSQS);
    es = MG_SAT(sat_k, t, 0);
    qs = R8(WV_EPSQS) * es / (p - omeps * es);
    qs = mg_min(R8(1.0), qs);
    if (qs < R8(0.0)) { qs = R8(1.0); es = p; }
    R8 hlatvp = R8(WV_HLATV) - R8(2369.0) * (t - R8(WV_TMELT));
    R8 hlatsb = R8(WV_HLATV);
    R8 hltalt = (t < R8(WV_TMELT)) ? hlatsb : hlatvp;
    R8 desdt = hltalt * es / (R8(WV_RGASV) * t * t);
    gam = hltalt * qs * p * desdt / (R8(WV_CP) * es * (p - omeps * es));
    if (qs == R8(1.0)) gam = R8(0.0);
}

// Use the error_function module WRF actually imports, with native rounding.
__device__ R8 cam_erf(R8 arg) { return cam5_calerf_r8(arg, false); }
// cldwat2m_micro::GAMMA, WRF public source lines 3557-3813. This is the
// module's Cody routine, not the Fortran intrinsic or a libm approximation.
// MG's call sites pass positive arguments (pgam is bounded to [2,15]).
__device__ R8 mg_gamma(R8 x, int* bad) {
    if (!(x.v > 0.0)) { *bad = 1; return R8(3.4e38); }
    const R8 p[8] = {
        R8(-1.71618513886549492533811E+0), R8(2.47656508055759199108314E+1),
        R8(-3.79804256470945635097577E+2), R8(6.29331155312818442661052E+2),
        R8(8.66966202790413211295064E+2), R8(-3.14512729688483675254357E+4),
        R8(-3.61444134186911729807069E+4), R8(6.64561438202405440627855E+4)};
    const R8 q[8] = {
        R8(-3.08402300119738975254353E+1), R8(3.15350626979604161529144E+2),
        R8(-1.01515636749021914166146E+3), R8(-3.10777167157231109440444E+3),
        R8(2.25381184209801510330112E+4), R8(4.75584627752788110767815E+3),
        R8(-1.34659959864969306392456E+5), R8(-1.15132259675553483497211E+5)};
    const R8 c[7] = {
        R8(-1.910444077728E-03), R8(8.4171387781295E-04),
        R8(-5.952379913043012E-04), R8(7.93650793500350248E-04),
        R8(-2.777777777777681622553E-03), R8(8.333333333333333331554247E-02),
        R8(5.7083835261E-03)};
    R8 y = x;
    if (y < R8(1.19E-7))
        return y >= R8(1.18E-38) ? R8(1.0) / y : R8(3.4E38);
    if (y < R8(12.0)) {
        R8 y1 = y, z;
        int n = 0;
        if (y < R8(1.0)) {
            z = y;
            y = y + R8(1.0);
        } else {
            n = uw_int(y) - 1;
            y = y - uw_real(n);
            z = y - R8(1.0);
        }
        R8 xnum(0.0), xden(1.0);
        for (int i = 0; i < 8; ++i) {
            xnum = (xnum + p[i]) * z;
            xden = xden * z + q[i];
        }
        R8 res = xnum / xden + R8(1.0);
        if (y1 < y) res = res / y1;
        else if (y1 > y) {
            for (int i = 0; i < n; ++i) {
                res = res * y;
                y = y + R8(1.0);
            }
        }
        return res;
    }
    if (y > R8(35.040)) return R8(3.4E38);
    R8 ysq = y * y;
    R8 sum = c[6];
    for (int i = 0; i < 6; ++i) sum = sum / ysq + c[i];
    sum = sum / y - y + R8(0.9189385332046727417803297);
    sum = sum + (y - R8(0.5)) * mg_log(y);
    return mg_exp(sum);
}

#endif  // CAMMGMP_SUPPORT_CUH
