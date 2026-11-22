"""Fast numerics for the TEMPO v3.2.5 column: binary32 arithmetic and CUDA binary32 maths.

Strict numerics never reaches this module: a strict process compiles
``kernels/tempo325`` exactly as written (binary64, WOOF's glibc-word libm),
which is the code the column oracle holds bitwise to native TEMPO v3.2.5
(tools/tempo_ref/oracle/run_woof_oracle.py --strict).  The pinned v3.0.3
generation is never transformed either.

THE BREAKAGE THIS ENDS: the default (fast) forecast ran TEMPO as the strict
column with contraction allowed: every value binary64 (1/64 rate on the
GeForce and RTX PRO cards) and every exp/log/pow the double-double glibc
replica that exists only to reproduce the x86-64 C library's words.  On the
3 km HRRR recipe (Iowa 92x92x50, RTX 4090) one TEMPO call cost about 20 ms
and the whole step ran 1.6-1.9x Thompson mp=28's (owner bar: a default is
within 1.2x of the fastest equivalent route).  A single column alone took
4.5 ms, so the call was bound by the binary64 pipe, not by the grid size.

What the fast module changes, by a checked source transform of the same
headers (no second copy of the physics):

* every TEMPO ``double`` is ``float`` and every floating literal in the
  TEMPO headers is a binary32 literal, so no expression is promoted back to
  binary64;
* the table blob is binary32 (converted once on the host; a table value that
  does not fit binary32 is refused by name);
* tempo_exp / tempo_log / tempo_log10 / tempo_pow are CUDA's binary32
  intrinsics ``__expf`` / ``__logf`` / ``__log10f`` and ``exp2f(y*__log2f(x))``
  for positive bases (``powf`` otherwise), the maths hex's fast TEMPO runs
  (hexcore tempo_allfloat_v303); measured 7 % of the call over the IEEE
  library functions;
* the per-column work area and packed column state are tiled: 32
  neighbouring columns share a tile and element k of a column array sits
  ``TEMPO_TILE`` words after element k-1, so a warp's access to one level is
  one contiguous line instead of 32 lines 40 kB apart.  A segment profile of
  the binary32 column (clock64 markers, 3 km HRRR recipe, RTX 4090) put
  70 % of the call in work-area traffic (zeroing alone 19 %: ~600 cycles per
  store), only 8 % in the process rates.  Every column array argument of the
  TEMPO routines is a strided view (``TSP``); a thread's private arrays are
  views of stride 1.  The arithmetic and its order are unchanged by the tiling.

Fast words change by design.  The fast referee is the forecast against
observations and beside the strict column (maps, rain and echo totals),
as for hex's fast TEMPO (hexcore tempo_allfloat_v303).
"""
from __future__ import annotations

import re
from pathlib import Path

#: The device-maths block of tempo325/tempo_common.cuh that the fast module
#: replaces: the glibc-word include, the fdlibm log10 and the four wrappers.
_MATH_START = '#if defined(__CUDACC__) || defined(__CUDACC_RTC__)\n#include "../glibc_flt64.cuh"\n#endif\n'
_MATH_END = "#ifndef TEMPO_NZ_MAX"
_MATH_FAST = """// Fast numerics (gpuwm.core.tempo_fast): CUDA binary32 intrinsics, as hex's fast TEMPO.
// Positive bases: pow = exp2(y log2 x); zero, negative and NaN bases: powf, out of line.
__device__ __noinline__ float tempo_pow_slow(float x, float y) { return powf(x, y); }
TEMPO_HD float tempo_exp(float x) { return __expf(x); }
TEMPO_HD float tempo_log(float x) { return __logf(x); }
TEMPO_HD float tempo_pow(float x, float y) { return x > 0.0f ? exp2f(y * __log2f(x)) : tempo_pow_slow(x, y); }
TEMPO_HD float tempo_log10(float x) { return __log10f(x); }

#ifndef TEMPO_TILE
#define TEMPO_TILE 32
#endif
// A column array argument: element k lives s elements after element k-1 (a tile
// column, or s = 1 for a thread's private array).
template <class T> struct TSP {
  T* p; int s;
  TEMPO_HD TSP() : p(nullptr), s(1) {}
  TEMPO_HD TSP(decltype(nullptr)) : p(nullptr), s(1) {}
  TEMPO_HD TSP(T* q) : p(q), s(1) {}
  TEMPO_HD TSP(T* q, int st) : p(q), s(st) {}
  template <class U> TEMPO_HD TSP(const TSP<U>& o) : p(o.p), s(o.s) {}
  TEMPO_HD T& operator[](int k) const { return p[k * s]; }
  TEMPO_HD T& operator*() const { return *p; }
  TEMPO_HD explicit operator bool() const { return p != nullptr; }
  TEMPO_HD bool operator==(decltype(nullptr)) const { return p == nullptr; }
  TEMPO_HD bool operator!=(decltype(nullptr)) const { return p != nullptr; }
};
"""
#: A C/C++ floating literal that is not part of an identifier or a hex float.
#: The tile column type, after TEMPO_NZP is defined (tempo_common.cuh).
_NZP_DEFINE = "#define TEMPO_NZP (TEMPO_NZ_MAX + 3)\n"
_TILE_TYPES = """// One column array of a tile: TEMPO_TILE columns share it, 4 bytes per column and
// level, so the lane's element k is 4*TEMPO_TILE bytes after its element k-1.  A
// thread addresses its tile through a view shifted by 4*lane bytes (TEMPO_VIEW):
// every member address of the view is then a constant offset from one register.
#define TEMPO_LEVEL_BYTES (4 * TEMPO_TILE)
template <class T> struct TCol {
  alignas(16) unsigned char raw[TEMPO_NZP * TEMPO_LEVEL_BYTES];
  TEMPO_HD T& operator[](int k) { return *reinterpret_cast<T*>(raw + k * TEMPO_LEVEL_BYTES); }
  TEMPO_HD const T& operator[](int k) const {
    return *reinterpret_cast<const T*>(raw + k * TEMPO_LEVEL_BYTES);
  }
  TEMPO_HD operator TSP<T>() { return TSP<T>(&(*this)[0], TEMPO_LEVEL_BYTES / (int)sizeof(T)); }
  TEMPO_HD operator TSP<const T>() const {
    return TSP<const T>(&(*this)[0], TEMPO_LEVEL_BYTES / (int)sizeof(T));
  }
};
#define TEMPO_VIEW(type, tile, lane) \\
  (*reinterpret_cast<type*>(reinterpret_cast<char*>(tile) + 4 * (lane)))
#define TEMPO_CVIEW(type, tile, lane) \\
  (*reinterpret_cast<const type*>(reinterpret_cast<const char*>(tile) + 4 * (lane)))

"""
_FLOAT_LITERAL = re.compile(r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])")
_DOUBLE = re.compile(r"\bdouble\b")
_LINE_COMMENT = re.compile(r"//[^\n]*")

#: The fast module's runtime kernels: the entry points of tempo_runtime._KERNEL
#: (same names and arguments) over the tiled column state and work area.
KERNEL = r"""
#include "tempo_column.cuh"
#include "tempo_cloud_ml.cuh"
#include "../glibc_flt32.cuh"
#define NFLD 17
struct ColTile { TCol<float> f[NFLD]; };
#define TEMPO_COL(s, j, l) (s).f[j]

template <typename T>
__device__ void pack_impl(const T* const* fields, int nz, int ncol_total, int col0, int ncol,
                          ColTile* S) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncol) return;
  long long g = col0 + c;
  ColTile& s = TEMPO_VIEW(ColTile, &S[c / TEMPO_TILE], c % TEMPO_TILE);
  for (int j = 0; j < NFLD; ++j) {
    const T* a = fields[j];
    s.f[j][0] = 0.0f;
    for (int k = 1; k <= nz; ++k) s.f[j][k] = (float)a[(long long)(k - 1) * ncol_total + g];
    for (int k = nz + 1; k < TEMPO_NZP; ++k) s.f[j][k] = 0.0f;
  }
}
template <typename T>
__device__ void unpack_impl(T* const* fields, const int* which, int nwhich, int nz,
                            int ncol_total, int col0, int ncol, const ColTile* S,
                            const TempoWork* WT, double* sfc, double* coldiag) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncol) return;
  long long g = col0 + c;
  const ColTile& s = TEMPO_CVIEW(ColTile, &S[c / TEMPO_TILE], c % TEMPO_TILE);
  const TempoWork& w = TEMPO_CVIEW(TempoWork, &WT[c / TEMPO_TILE], c % TEMPO_TILE);
  for (int m = 0; m < nwhich; ++m) {
    T* a = fields[m];
    int j = which[m];
    for (int k = 1; k <= nz; ++k) a[(long long)(k - 1) * ncol_total + g] = (T)s.f[j][k];
  }
  float mh = w.max_hail_diameter[1];
  for (int k = 2; k <= nz; ++k) mh = w.max_hail_diameter[k] > mh ? w.max_hail_diameter[k] : mh;
  double v[8] = {w.rain_precip, w.ice_liquid_equiv_precip, w.snow_liquid_equiv_precip,
                 w.graupel_liquid_equiv_precip, w.frz_rain_precip, w.frozen_fraction,
                 w.max_hail_diameter[1], mh};
  for (int m = 0; m < 8; ++m) sfc[(long long)m * ncol_total + g] = v[m];
  long long plane = (long long)nz * ncol_total;
  for (int k = 1; k <= nz; ++k) {
    long long o = (long long)(k - 1) * ncol_total + g;
    coldiag[o] = w.refl10cm[k];
    coldiag[plane + o] = w.re_cloud[k];
    coldiag[2 * plane + o] = w.re_ice[k];
    coldiag[3 * plane + o] = w.re_snow[k];
  }
}
extern "C" __global__ void tempo_pack_f32(const float* const* f, int nz, int nt, int c0, int n, ColTile* S) { pack_impl<float>(f, nz, nt, c0, n, S); }
extern "C" __global__ void tempo_pack_f64(const double* const* f, int nz, int nt, int c0, int n, ColTile* S) { pack_impl<double>(f, nz, nt, c0, n, S); }
extern "C" __global__ void tempo_unpack_f32(float* const* f, const int* wh, int nw, int nz, int nt, int c0, int n, const ColTile* S, const TempoWork* W, double* sfc, double* cd) { unpack_impl<float>(f, wh, nw, nz, nt, c0, n, S, W, sfc, cd); }
extern "C" __global__ void tempo_unpack_f64(double* const* f, const int* wh, int nw, int nz, int nt, int c0, int n, const ColTile* S, const TempoWork* W, double* sfc, double* cd) { unpack_impl<double>(f, wh, nw, nz, nt, c0, n, S, W, sfc, cd); }

// Host float32 theta/Exner and m3/kg volume -> TEMPO temperature and L/kg in the column.
extern "C" __global__ void tempo_pack_adapter(const float* const* f, const float* pii,
        const float* emission, double dt, int nz, int nt, int c0, int n, ColTile* S) {
  pack_impl<float>(f, nz, nt, c0, n, S);
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  ColTile& s = TEMPO_VIEW(ColTile, &S[c / TEMPO_TILE], c % TEMPO_TILE);
  for (int k = 1; k <= nz; ++k) {
    long long o = (long long)(k - 1) * nt + c0 + c;
    s.f[0][k] *= pii[o];
    s.f[16][k] *= 1000.0f;
  }
  if (emission) s.f[13][1] += emission[c0 + c] * (float)dt;
}
extern "C" __global__ void tempo_unpack_adapter(float* const* f, const int* wh, int nw,
        const float* pii, const float* emission, double dt,
        int nz, int nt, int c0, int n, ColTile* S, const TempoWork* W,
        double* sfc, double* cd) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  ColTile& s = TEMPO_VIEW(ColTile, &S[c / TEMPO_TILE], c % TEMPO_TILE);
  // tempo_aerosol_surface_emissions after tempo_run, before float32 storage.
  if (emission) s.f[13][1] += emission[c0 + c] * (float)dt;
  for (int k = 1; k <= nz; ++k) {
    long long o = (long long)(k - 1) * nt + c0 + c;
    s.f[0][k] /= pii[o];
    s.f[16][k] *= 1.0e-3f;
  }
  unpack_impl<float>(f, wh, nw, nz, nt, c0, n, S, W, sfc, cd);
}
extern "C" __global__ void tempo_adapter_exner(const float* p, float* pii, int n,
                                               float p0, float rcp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pii[i] = gfk_pow(__fdiv_rn(p[i], p0), rcp);
}

extern "C" __global__ void tempo_columns(const float* blob, TempoCtx C, int ncol, int nz,
                                         int first_global_col, ColTile* S, TempoWork* WT,
                                         const float* qc_bl, const float* fraction_bl, int nt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ncol) return;
  TempoParams P{reinterpret_cast<const TempoBlob*>(blob)};
  TempoCtx c = C;
  ColTile& s = TEMPO_VIEW(ColTile, &S[i / TEMPO_TILE], i % TEMPO_TILE);
  TempoWork& W = TEMPO_VIEW(TempoWork, &WT[i / TEMPO_TILE], i % TEMPO_TILE);
  tempo_main_column(P, c, nz, W, TEMPO_COL(s, 4, l), TEMPO_COL(s, 5, l), TEMPO_COL(s, 7, l),
                    TEMPO_COL(s, 6, l), TEMPO_COL(s, 8, l), TEMPO_COL(s, 9, l), TEMPO_COL(s, 16, l),
                    TEMPO_COL(s, 10, l), TEMPO_COL(s, 11, l), TEMPO_COL(s, 12, l), TEMPO_COL(s, 15, l),
                    TEMPO_COL(s, 13, l), TEMPO_COL(s, 14, l), TEMPO_COL(s, 0, l), TEMPO_COL(s, 1, l),
                    TEMPO_COL(s, 2, l), TEMPO_COL(s, 3, l));
  if (qc_bl && fraction_bl)
    tempo_subgrid_cloud_radii(P, c, nz, W, TEMPO_COL(s, 5, l), TEMPO_COL(s, 6, l), TEMPO_COL(s, 7, l),
                             TEMPO_COL(s, 8, l), TEMPO_COL(s, 12, l), TEMPO_COL(s, 2, l),
                             qc_bl, fraction_bl, nt, first_global_col + i);
}
// Initial-state radar diagnosis (tempo_runtime._KERNEL's, over the tiles).
extern "C" __global__ void tempo_initial_refl(const float* blob, TempoCtx C,
    int n, int nz, int c0, int nt, ColTile* S, TempoWork* work, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  TempoParams P{reinterpret_cast<const TempoBlob*>(blob)};
  ColTile& s = TEMPO_VIEW(ColTile, &S[i / TEMPO_TILE], i % TEMPO_TILE);
  TempoWork& W = TEMPO_VIEW(TempoWork, &work[i / TEMPO_TILE], i % TEMPO_TILE);
  for (int k = 1; k <= nz; ++k) {
    W.temp[k] = s.f[0][k];
    W.qv[k] = tmax(PS(min_qv), s.f[4][k]);
    W.pres[k] = s.f[1][k];
    W.rho[k] = PS(roverrv) * W.pres[k] /
      (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    W.refl10cm[k] = -35.0f;
  }
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, TEMPO_COL(s, 6, l), TEMPO_COL(s, 11, l),
    W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, TEMPO_COL(s, 8, l), W.rs, W.qsten);
  float sq = 0.0f, sn = 0.0f, sb = 0.0f;
  for (int k = 1; k <= nz; ++k) {
    sq += s.f[9][k]; sn += s.f[15][k]; sb += s.f[16][k];
    if (W.l_qs[k]) {
      float tc = tmin(-0.1f, W.temp[k] - PS(t0));
      snow_moments(P, W.rs[k], tc, &W.smob[k], &W.smoc[k], nullptr,
        nullptr, nullptr, &W.smo2[k], nullptr, nullptr, nullptr, &W.smoz[k]);
    }
  }
  if (sq > PS(r1) && sn < PS(eps) && sb < PS(eps))
    graupel_init(P, nz, W.rho, TEMPO_COL(s, 9, l), TEMPO_COL(s, 15, l), TEMPO_COL(s, 16, l));
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, TEMPO_COL(s, 9, l), TEMPO_COL(s, 15, l),
    TEMPO_COL(s, 16, l), W.rg, W.ng, W.rb, W.idx_bg, W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  // The initial state carries no split hail: graupel holds it all.
  for (int k = 0; k < TEMPO_NZP; ++k) W.l_qh[k] = false;
  reflectivity_10cm(P, C, nz, C.refl10cm_from_melting_flag, W.temp,
    W.l_qr, W.rr, W.nr, W.ilamr, W.l_qs, W.rs, W.smoc, W.smob, W.smoz,
    W.l_qg, W.rg, W.ng, W.idx_bg, W.ilamg, W.l_qh, W.nh, W.ilamh, W.refl10cm);
  for (int k = 1; k <= nz; ++k)
    out[(long long)(k - 1) * nt + c0 + i] = W.refl10cm[k];
}
// Bytes per column (a tile holds TEMPO_TILE columns; the host rounds to whole tiles).
extern "C" __global__ void tempo_sizes(long long* out) {
  out[0] = (sizeof(TempoWork) + TEMPO_TILE - 1) / TEMPO_TILE;
  out[1] = (sizeof(ColTile) + TEMPO_TILE - 1) / TEMPO_TILE;
  out[2] = sizeof(TempoCtx);
  out[3] = TEMPO_NZ_MAX;
}
extern "C" __global__ void tempo_mpas_emission_init(const float* blob,
    const float* nwfa, const double* depth, const double* area, float* emission, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  TempoParams P{reinterpret_cast<const TempoBlob*>(blob)};
  double dz = depth[i] > 9.0 ? depth[i] : 9.0;
  double e = (double)nwfa[i];
  e = e < PS(aero_max) ? e : PS(aero_max);
  e = e > PS(nwfa_default) ? e : PS(nwfa_default);
  emission[i] = (float)(e * 0.000196 * (5.0 / dz) * (area[i] / 9.0e6));
}
"""
#: Entry points both kernels define (the runtime looks them up by name).
ENTRY_POINTS = ("tempo_pack_f32", "tempo_pack_f64", "tempo_unpack_f32", "tempo_unpack_f64",
                "tempo_pack_adapter", "tempo_unpack_adapter", "tempo_adapter_exner", "tempo_columns",
                "tempo_initial_refl", "tempo_sizes", "tempo_mpas_emission_init")


_TEND_OLD = """struct TempoTend {
#define TEMPO_TT_MEMBER(name) double name[TEMPO_NZP];
  TEMPO_TEND_FIELDS(TEMPO_TT_MEMBER)
#undef TEMPO_TT_MEMBER
};
"""
_TEND_NEW = """struct TempoTend {
#define TEMPO_TT_MEMBER(name) TCol<double> name;
  TEMPO_TEND_FIELDS(TEMPO_TT_MEMBER)
#undef TEMPO_TT_MEMBER
};
"""
_WORK_HEAD = "struct TempoWork {\n"
_WORK_MEMBERS = """#define TEMPO_WA_MEMBER(name) double name[TEMPO_NZP];
  TEMPO_WORK_ARRAYS(TEMPO_WA_MEMBER)
#undef TEMPO_WA_MEMBER
  bool l_qc[TEMPO_NZP], l_qi[TEMPO_NZP], l_qr[TEMPO_NZP], l_qs[TEMPO_NZP], l_qg[TEMPO_NZP],
      l_qh[TEMPO_NZP];
  int idx_bg[TEMPO_NZP];
  TempoTend T;
  // ty_tempo_main_diags scalars
  double rain_precip, cloud_precip, ice_liquid_equiv_precip, snow_liquid_equiv_precip,
      graupel_liquid_equiv_precip, hail_liquid_equiv_precip, frozen_fraction, frz_rain_precip;
  int did_micro;  // 0 when tempo_main returned early (no hydrometeors, no supersaturation)
};
"""
_FLAGS = ("l_qc", "l_qi", "l_qr", "l_qs", "l_qg", "l_qh")
_SCALARS = ("rain_precip", "cloud_precip", "ice_liquid_equiv_precip", "snow_liquid_equiv_precip",
            "graupel_liquid_equiv_precip", "hail_liquid_equiv_precip", "frozen_fraction", "frz_rain_precip")
_WORK_NEW = "".join([
    "// A tile of TEMPO_TILE columns; a column sees it through TEMPO_VIEW(TempoWork, tile, lane).\n",
    "struct TempoWork {\n",
    "#define TEMPO_WA_MEMBER(name) TCol<double> name;\n",
    "  TEMPO_WORK_ARRAYS(TEMPO_WA_MEMBER)\n",
    "#undef TEMPO_WA_MEMBER\n",
    *[f"  TCol<bool> {n};\n" for n in _FLAGS],
    "  TCol<int> idx_bg;\n",
    "  TempoTend T;\n",
    *[f"  double {n}; float {n}_lanes[TEMPO_TILE - 1];\n" for n in _SCALARS],
    "  int did_micro; int did_micro_lanes[TEMPO_TILE - 1];\n",
    "};\n",
])
#: tempo_any as TEMPO writes it (early return on the first true level) and its
#: branch-free fast form.  BREAKAGE THIS ENDS: on sm_120 (RTX PRO 6000 Blackwell,
#: RTX 5090) the early-return loop over the tiled bool arrays cost 5.2 M cycles per
#: column in the do_micro test alone (34 % of the call; 76 k cycles on sm_89), so a
#: TEMPO call took 15.3 ms on an RTX PRO 6000 against 5.5 ms on an RTX 4090.  The
#: OR over every level is the same value (tempo_any has no side effects); the call
#: drops to 6.0 ms with every output word unchanged (Iowa 92x92x50 capture).
_ANY_OLD = """TEMPO_HD bool tempo_any(const bool* l, int nz) {
  for (int k = 1; k <= nz; ++k)
    if (l[k]) return true;
  return false;
}
"""
_ANY_NEW = """TEMPO_HD bool tempo_any(const bool* l, int nz) {
  bool a = false;  // woof.core.tempo_fast: branch-free (sm_120), same value
  for (int k = 1; k <= nz; ++k) a = a | l[k];
  return a;
}
"""
#: A column-array declaration (parameter or local): ``[const] double|bool|int * name``.
_POINTER = re.compile(r"\b(const\s+)?(double|bool|int)\s*\*\s*(\w+)")


def _tile_structs(text: str, name: str) -> str:
    """The tiled work area (TempoWorkTile / TempoTendTile) and its per-column views."""
    if name == "tempo_common.cuh":
        if text.count(_TEND_OLD) != 1:
            raise ValueError("TEMPO fast: TempoTend struct not found once (header drift)")
        text = text.replace(_TEND_OLD, _TEND_NEW)
    if name == "tempo_column.cuh":
        if text.count(_ANY_OLD) != 1:
            raise ValueError("TEMPO fast: tempo_any not found once (header drift)")
        text = text.replace(_ANY_OLD, _ANY_NEW)
        if text.count(_WORK_HEAD) != 1 or text.count(_WORK_MEMBERS) != 1:
            raise ValueError("TEMPO fast: TempoWork struct not found once (header drift)")
        a = text.index(_WORK_HEAD)
        b = text.index(_WORK_MEMBERS)
        arrays = text[a + len(_WORK_HEAD):b]          # the TEMPO_WORK_ARRAYS X-macro
        if not arrays.startswith("#define TEMPO_WORK_ARRAYS(WA)"):
            raise ValueError("TEMPO fast: TEMPO_WORK_ARRAYS does not open TempoWork (header drift)")
        text = text[:a] + arrays + _WORK_NEW + text[b + len(_WORK_MEMBERS):]
    return _POINTER.sub(lambda m: f"TSP<{m.group(1) or ''}{m.group(2)}> {m.group(3)}", text)


def _float_header(text: str, name: str) -> str:
    text = _tile_structs(text, name)
    if name == "tempo_common.cuh":
        if text.count(_MATH_START) != 1 or text.count(_MATH_END) != 1:
            raise ValueError("TEMPO fast: tempo_common.cuh maths block not found once (header drift)")
        a = text.index(_MATH_START)
        b = text.index(_MATH_END)
        if not a < b:
            raise ValueError("TEMPO fast: tempo_common.cuh maths block markers out of order")
        text = text[:a] + _MATH_FAST + text[b:]
        if text.count(_NZP_DEFINE) != 1:
            raise ValueError("TEMPO fast: TEMPO_NZP definition not found once (header drift)")
        text = text.replace(_NZP_DEFINE, _NZP_DEFINE + _TILE_TYPES)
    if re.search(r"\b__(?:d(?:add|mul|div|sub|sqrt)|fma)_rn\b|__double_as|__longlong_as", _LINE_COMMENT.sub("", text)):
        raise ValueError(f"TEMPO fast: {name} carries a binary64 intrinsic the transform cannot narrow")
    text = _DOUBLE.sub("float", text)
    return _FLOAT_LITERAL.sub(lambda m: m.group(1) + "f", text)


def kernel_source(directory: Path) -> str:
    """:data:`KERNEL` with every TEMPO header of ``directory`` expanded in binary32, tiled.

    Non-TEMPO headers (``../glibc_flt32.cuh``, the exner's binary32 power) are
    expanded as written; each header is expanded once, as ``#pragma once`` does.
    """
    from woof.core.tempo_runtime import _QUOTED_INCLUDE

    kernel = KERNEL
    directory = directory.resolve()
    seen: set[Path] = set()

    def expand(source: str, base: Path) -> str:
        def one(match: re.Match) -> str:
            path = (base / match.group(1)).resolve()
            if path in seen:
                return ""
            seen.add(path)
            text = path.read_text(encoding="utf-8")
            if path.parent == directory:
                text = _float_header(text, path.name)
            return expand(text, path.parent)
        return _QUOTED_INCLUDE.sub(one, source)

    return expand(kernel, directory)


def float_blob(blob, layout):
    """The binary64 tempo_init blob as binary32; a value that does not fit is refused by name."""
    import numpy as np

    with np.errstate(over="ignore"):
        out = np.asarray(blob, dtype=np.float64).astype(np.float32)
    bad = ~np.isfinite(out) & np.isfinite(np.asarray(blob, dtype=np.float64))
    if bad.any():
        names, o = [], 0
        for name, n in layout:
            if bad[o:o + n].any():
                names.append(name)
            o += n
        raise ValueError("TEMPO fast: table values overflow binary32 in " + ", ".join(names))
    return out
