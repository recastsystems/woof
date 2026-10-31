"""NSSL sedimentation across levels and columns (rrfs/nssl-fast stage ``sed``): fast numerics only.

Strict numerics never sees this module (``mp_nssl2m.cuda_source`` applies it only in FAST numerics). Native
f_sediment1d (RRFS module_mp_nssl_2mom.F, ``nssl_sediment``'s column thread) computes one column's fall speeds over
every level (f_ziegfall1d gathers up to 128 active levels into 128-point work arrays), then for every species runs
its fallout substeps one level after another. THE BREAKAGE THIS ENDS: that one thread per column was the slot's
largest phase after the collection fix (86 of 226 ms of the mv3km call-120 replay, RTX 4090, 2026-10-29),
and half of it was the fall speeds walking 128-point work arrays in device memory.

The fast route splits the sedimentation into three launches (the shape TEMPO's fast sedimentation took,
rrfs/tempo-yolo 7c041466):

* ``nssl_sedpre`` (one thread per column): the sediment phase up to the f_sediment1d call, then it records the
  column's array descriptors for the fall-speed launch;
* ``nssl_sedfall`` (one thread per point, consecutive threads are consecutive columns at one level): native
  f_ziegfall1d for ONE level (``f_ziegfall1d_lvl``), its work arrays one point long in the thread's own local
  scratch. Fall speeds are pointwise in native (each gathered point reads and writes only itself), so a point
  computes exactly what native computes for it inside the 128-point gather. Inactive points (no hydrometeor above
  qxmin, native's gather test) do nothing;
* ``nssl_sedlanes`` (one warp per column, level k on lane (k-1)%32, row (k-1)/32): native's species loop. Native
  updates the fallout bottom to top while level k+1 is still old, so every level's update reads old fluxes only:
  each lane updates its own levels with the flux of level k+1 from a shuffle. calczgr1d and calcnfromz1d are
  pointwise and run per level (``*_lvl``). The species order, each species' substep count and dtfrac, every
  moment's fallout and the number limiters are native's; lane 0 then runs native's surface accumulation.

Same arithmetic per point and per flux as the column route (the referee is the word comparison against it).
Not on this route, which falls back to native's column thread (``nssl_sediment``) at run time: do_accurate_
sedimentation (fall speeds refreshed inside the substeps), more than 128 levels, or more than 16 species.
"""
from __future__ import annotations

import os
import re

ROWS = 4          # levels up to 32 * ROWS on the lane route
MAX_SPECIES = 16  # lhab - lc + 1 up to this
SCRATCH = 1536    # bytes of thread-local scratch for one point's fall-speed work arrays (1,240 needed at 128 levels, 16 species)

_SED_CALL = ("f_sediment1d(dtp, nx, ny, nz, an, na, nor, nor, xfall, dn1, dz2d, dz2dinv, t0, t7, infdo, jy, its, jts, "
             "timesed1, timesed2, timesed3, zmaxsed, timesetvt);")


def fall_mode() -> str:
    """``point`` (default): one fall-speed thread per point; ``column``: the column thread walks its levels."""
    mode = os.environ.get("WOOF_HEX_NSSL_SEDFALL", "point")
    if mode not in ("point", "column"):
        raise ValueError(f"WOOF_HEX_NSSL_SEDFALL: {mode!r} (point or column)")
    return mode


def fall_blocks(multiprocessors: int) -> int:
    """Grid of the point fall-speed launch (128-thread blocks, grid stride): WOOF_HEX_NSSL_SEDFALL_BLOCKS (per SM, 4)."""
    return max(1, int(multiprocessors) * int(os.environ.get("WOOF_HEX_NSSL_SEDFALL_BLOCKS", "4")))


def _block_end(text: str, start: int) -> int:
    opening = text.index("{", start)
    depth, i = 1, opening + 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return i


def _routine(text: str, name: str) -> tuple[int, int]:
    head = "__device__ __forceinline__ void " + name + "("
    if text.count(head) != 1:
        raise ValueError(f"NSSL fast sed: expected one {name}, found {text.count(head)}")
    start = text.index(head)
    return start, _block_end(text, start)


def _one(text: str, old: str, new: str, what: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"NSSL fast sed {what}: expected one {old[:80]!r}, found {text.count(old)}")
    return text.replace(old, new)


def _level_variants(body: str) -> str:
    """Per-level copies of the pointwise routines; the originals stay for the column fallback."""
    out = []
    s, e = _routine(body, "f_ziegfall1d")
    z = body[s:e].replace("void f_ziegfall1d(", "void f_ziegfall1d_lvl(int selected_level, ", 1)
    z = _one(z, "nzmpb = 1;", "nzmpb = selected_level;", "ziegfall level")
    z = _one(z, "nssl_first_3=((kz=nzmpb),0), nssl_end_3=(nz),", "nssl_first_3=((kz=nzmpb),0), nssl_end_3=(selected_level),",
             "ziegfall gather")
    out.append(z)
    for name in ("f_calczgr1d", "f_calcnfromz1d"):
        s, e = _routine(body, name)
        r = body[s:e].replace(f"void {name}(", f"void {name}_lvl(int selected_level, ", 1)
        r, n = re.subn(r"\(\(kz=1\),0\), (nssl_end_\d+)=\(kze\)", r"((kz=selected_level),0), \1=(selected_level)", r)
        if n != 2 or re.search(r"\(kz=(?!selected_level)", r):
            raise ValueError(f"NSSL fast sed: {name} level loops changed shape ({n})")
        out.append(r)
    return "\n".join(out)


DEVICE_GLOBALS = r"""
// rrfs/nssl-fast sed: buffers of the split sedimentation, bound once per run (nssl_fast_sed_bind).
struct NsslFastSedDesc { NsslArray<float, 4> an; NsslArray<float, 3> dn, t0, t7; float dtp; int nz, na, nor, infdo; };
__device__ float* nssl_fast_xvt = nullptr;          // fall speeds [(il-lc)*3 + j-1][kz-1][column]
__device__ unsigned char* nssl_fast_act = nullptr;  // native's gather test per point [kz-1][column]
__device__ NsslFastSedDesc* nssl_fast_desc = nullptr;
__device__ int nssl_fast_cols = 0;
#ifdef __CUDA_ARCH__
#define NSSL_FAST_LANE ((int)(threadIdx.x & 31))
#else
#define NSSL_FAST_LANE 0
#endif
"""

MEMBERS = r"""
// rrfs/nssl-fast sed: native f_sediment1d's fall speeds (its f_ziegfall1d call and the lzh limit before it) for ONE
// point. The ngs work arrays are one point long in this thread's local scratch; ziegfall's own frames continue there.
__device__ __forceinline__ void f_fast_sed_point(int kz, int col, NsslArray<float, 4> an, NsslArray<float, 3> dn,
    NsslArray<float, 3> t0, NsslArray<float, 3> t7, float dtp, int nz, int na, int nor, int infdo) {
constexpr int ng1 = 1;
constexpr int ngs = 1;
const int norz = nor;
const int nx = 1, ny = 1, ix = 1, jy = 1, jgs = 1;
an.rebase({(((-(nor))) + (1)), (((-(nor))) + (1)), (((-(norz))) + (1)), 1});
dn.rebase({(((-(nor))) + (1)), (((-(nor))) + (1)), (((-(norz))) + (1))});
t0.rebase({(((-(nor))) + (1)), (((-(nor))) + (1)), (((-(norz))) + (1))});
t7.rebase({(((-(nor))) + (1)), (((-(nor))) + (1)), (((-(norz))) + (1))});
if (((lzh) > (1))) { an(ix, jy, kz, lzh) = nssl_max((float)(0.f), (float)(an(ix, jy, kz, lzh))); }
const unsigned long long at = (unsigned long long)(kz - 1) * nssl_fast_cols + col;
bool flag = false;
for (int il = lc; il <= lhab; ++il) { flag = ((flag) || (((an(ix, jy, kz, il)) > (qxmin(il))))); }
nssl_fast_act[at] = flag ? 1 : 0;
if (!flag) return;
alignas(16) unsigned char scratch[NSSL_FAST_SCRATCH];
NsslArena la{scratch, 0, 1, 0, 0, 0};
NsslArray<float, 2> qx = nssl_array<float, 2>(la, {1, lv}, {ngs, lhab});
NsslArray<float, 2> qxw = nssl_array<float, 2>(la, {1, ls}, {ngs, lhab});
NsslArray<float, 2> cx = nssl_array<float, 2>(la, {1, lc}, {ngs, lhab});
NsslArray<float, 2> xv = nssl_array<float, 2>(la, {1, lc}, {ngs, lhab});
NsslArray<float, 3> vtxbar = nssl_array<float, 3>(la, {1, lc, 1}, {ngs, lhab, 3});
NsslArray<float, 2> xmas = nssl_array<float, 2>(la, {1, lc}, {ngs, lhab});
NsslArray<float, 2> xdn = nssl_array<float, 2>(la, {1, lc}, {ngs, lhab});
NsslArray<float, 3> xdia = nssl_array<float, 3>(la, {1, lc, 1}, {ngs, lhab, 3});
NsslArray<float, 2> vx = nssl_array<float, 2>(la, {1, li}, {ngs, lhab});
NsslArray<float, 2> alpha = nssl_array<float, 2>(la, {1, lc}, {ngs, lhab});
NsslArray<float, 2> zx = nssl_array<float, 2>(la, {1, lr}, {ngs, lhab});
NsslArray<int, 1> igs = nssl_array<int, 1>(la, {1}, {ngs});
NsslArray<int, 1> kgs = nssl_array<int, 1>(la, {1}, {ngs});
NsslArray<float, 1> rho0 = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> temcg = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> temg = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> rhovt = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> cwnc = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> cinc = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> fadvisc = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> cwdia = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> cipmas = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> cnina = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> cimas = nssl_array<float, 1>(la, {1}, {ngs});
NsslArray<float, 1> cnostmp = nssl_array<float, 1>(la, {1}, {ngs});
for (unsigned long long i = 0; i < la.offset; i += 8) *reinterpret_cast<unsigned long long*>(scratch + i) = 0ULL;
float xl[3 * NSSL_FAST_MAX_SPECIES];
for (int i = 0; i < 3 * (lhab - lc + 1); ++i) xl[i] = 0.0f;
NsslArray<float, 4> xvt;
xvt.data = xl; xvt.lo[0] = 1; xvt.lo[1] = 1; xvt.lo[2] = 1; xvt.lo[3] = lc;
xvt.hi[0] = nz + 1; xvt.hi[1] = nx; xvt.hi[2] = 3; xvt.hi[3] = lhab;
xvt.stride[0] = 0; xvt.stride[1] = 0; xvt.stride[2] = 1; xvt.stride[3] = 3;
float rv = nssl_sqrt(nssl_fmul(rho00, nssl_min((float)(nssl_fdiv(1.0f, 0.05f)), (float)(nssl_fdiv(1.f, dn(ix, jy, kz))))));
NsslArray<float, 2> rhovtzx;
rhovtzx.data = &rv; rhovtzx.lo[0] = 1; rhovtzx.lo[1] = 1; rhovtzx.hi[0] = nz; rhovtzx.hi[1] = nx;
rhovtzx.stride[0] = 0; rhovtzx.stride[1] = 0;
NsslArena saved_arena = arena;
arena = la;
f_ziegfall1d_lvl(kz, nx, ny, nz, nor, norz, na, dtp, jgs, ix, xvt, rhovtzx, an, dn, ipconc, t0, t7, cwmasn, cwmasx, cwradn, qxmin, xdnmx, xdnmn, cdx, cno, xdn0, xvmn, xvmx, ngs, qx, qxw, cx, xv, vtxbar, xmas, xdn, xdia, vx, alpha, zx, igs, kgs, rho0, temcg, temg, rhovt, cwnc, cinc, fadvisc, cwdia, cipmas, cnina, cimas, cnostmp, infdo, 0, {});
arena = saved_arena;
const unsigned long long plane = (unsigned long long)nz * nssl_fast_cols;
for (int i = 0; i < 3 * (lhab - lc + 1); ++i) nssl_fast_xvt[i * plane + at] = xl[i];
}
// rrfs/nssl-fast sed (fall stage ``point``): the column's arrays for the per-point fall-speed launch.
__device__ __forceinline__ void f_fast_sed_export(float dtp, int nz, int na, int nor, int infdo, NsslArray<float, 4> an,
    NsslArray<float, 3> dn, NsslArray<float, 3> t0, NsslArray<float, 3> t7) {
NsslFastSedDesc& d = nssl_fast_desc[fast_column];
d.an = an; d.dn = dn; d.t0 = t0; d.t7 = t7; d.dtp = dtp; d.nz = nz; d.na = na; d.nor = nor; d.infdo = infdo;
}
// rrfs/nssl-fast sed (fall stage ``column``): the column thread computes every level's fall speeds itself.
__device__ __forceinline__ void f_fast_sed_fall_column(float dtp, int nz, int na, int nor, int infdo, NsslArray<float, 4> an,
    NsslArray<float, 3> dn, NsslArray<float, 3> t0, NsslArray<float, 3> t7) {
for (int k = 1; k <= nz; ++k) f_fast_sed_point(k, fast_column, an, dn, t0, t7, dtp, nz, na, nor, infdo);
}
// One native f_fallout1d pass (F:4082-4189) with level k on lane (k-1)%32, row (k-1)/32. Native updates bottom to top
// while level k+1 is still old, so every update reads old fluxes; the flux of level k+1 arrives by shuffle. Lane 0
// adds the bottom outflow to *xf when native's kmn is 1 (the first nonzero level is 1 or 2, or nz <= 2).
template<class Ref> __device__ __forceinline__ void fast_sed_fallout(Ref ref, int nz, const bool (&live)[NSSL_FAST_ROWS],
    const int (&kk)[NSSL_FAST_ROWS], const float (&vt)[NSSL_FAST_ROWS], const float (&dtz)[NSSL_FAST_ROWS],
    const float (&db)[NSSL_FAST_ROWS], int id, float dtp, float dtfrac, float* xf) {
#ifdef __CUDA_ARCH__
const unsigned full = 0xffffffffu;
const int lane = NSSL_FAST_LANE;
float a[NSSL_FAST_ROWS], f[NSSL_FAST_ROWS];
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) {
a[r] = live[r] ? ref(kk[r]) : 0.0f;
f[r] = live[r] ? (((id) == (1)) ? nssl_fmul(nssl_fmul(a[r], vt[r]), db[r]) : nssl_fmul(a[r], vt[r])) : 0.0f;
}
if (xf != nullptr) {
const float a1 = __shfl_sync(full, a[0], 0), a2 = __shfl_sync(full, a[0], 1), v1 = __shfl_sync(full, vt[0], 0);
if (lane == 0 && (nz <= 2 || a1 != 0.0f || a2 != 0.0f)) *xf = nssl_fadd(*xf, nssl_fmul(nssl_fmul(a1, v1), dtfrac));
}
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) {
const float below = __shfl_down_sync(full, f[r], 1);
const float wrap = __shfl_sync(full, f[(r + 1 < NSSL_FAST_ROWS) ? r + 1 : r], 0);
float next = lane < 31 ? below : ((r + 1 < NSSL_FAST_ROWS) ? wrap : 0.0f);
if (kk[r] >= nz) next = 0.0f;
if (live[r]) ref(kk[r]) = nssl_fadd(a[r], nssl_fmul(nssl_fmul(dtp, dtz[r]), nssl_fsub(next, f[r])));
}
#endif
}
// rrfs/nssl-fast sed: native f_sediment1d's species loop (after the fall speeds) for one column, one warp, level k on
// lane (k-1)%32, row (k-1)/32. Same arguments as f_sediment1d.
__device__ __forceinline__ void f_fast_sed_lanes(float dtp, int nx, int ny, int nz, NsslArray<float, 4> an, int na, int nor, int norz, NsslArray<float, 3> xfall, NsslArray<float, 3> dn, NsslArray<float, 3> dz3d, NsslArray<float, 3> dz3dinv, NsslArray<float, 3> t0, NsslArray<float, 3> t7, int infdo, int jslab, int its, int jts, float timesed1, float timesed2, float timesed3, float zmaxsed, float timesetvt) {
#ifdef __CUDA_ARCH__
NsslFrame nssl_frame(arena);
constexpr int ng1 = 1;
an.rebase({(((-(nor))) + (1)), (((-(nor))) + (1)), (((-(norz))) + (1)), 1});
xfall.rebase({1, 1, 1});
dn.rebase({(((-(nor))) + (1)), (((-(nor))) + (1)), (((-(norz))) + (1))});
dz3dinv.rebase({(((-(nor))) + (1)), (((-(nor))) + (1)), (((-(norz))) + (1))});
const unsigned full = 0xffffffffu;
const int lane = NSSL_FAST_LANE;
const int jy = 1, jgs = 1, ix = 1, ixe = nx, kze = nz;
// Native's column temporaries; every lane allocates them alike, so they are the column's, each lane on its levels.
NsslArray<float, 3> tmpn = nssl_array<float, 3>(arena, {(((-(nor))) + (ng1)), (((-(nor))) + (ng1)), (((-(norz))) + (ng1))}, {((nx) + (nor)), ((ny) + (nor)), ((nz) + (norz))});
NsslArray<float, 3> tmpn2 = nssl_array<float, 3>(arena, {(((-(nor))) + (ng1)), (((-(nor))) + (ng1)), (((-(norz))) + (ng1))}, {((nx) + (nor)), ((ny) + (nor)), ((nz) + (norz))});
NsslArray<float, 3> z = nssl_array<float, 3>(arena, {(((-(nor))) + (ng1)), (((-(norz))) + (ng1)), lr}, {((nx) + (nor)), ((nz) + (norz)), lhab});
NsslArray<float, 2> db1 = nssl_array<float, 2>(arena, {1, 1}, {nx, ((nz) + (1))});
const unsigned long long plane = (unsigned long long)nz * nssl_fast_cols;
bool live[NSSL_FAST_ROWS], act[NSSL_FAST_ROWS];
int kk[NSSL_FAST_ROWS];
unsigned long long at[NSSL_FAST_ROWS];
float dt0[NSSL_FAST_ROWS], dt1[NSSL_FAST_ROWS], db[NSSL_FAST_ROWS];
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) {
kk[r] = 32 * r + lane + 1;
live[r] = kk[r] <= nz;
at[r] = (unsigned long long)(kk[r] - 1) * nssl_fast_cols + fast_column;
act[r] = live[r] && nssl_fast_act[at[r]] != 0;
dt0[r] = 0.0f; dt1[r] = 0.0f; db[r] = 0.0f;
if (live[r]) {
const int k = kk[r];
db1(ix, k) = dn(ix, jy, k);
db[r] = db1(ix, k);
const float inv = nssl_fdiv(1.f, dn(ix, jy, k));
dt0[r] = dz3dinv(ix, jy, k);
dt1[r] = nssl_fmul(dz3dinv(ix, jy, k), inv);
}
}
auto species = [&](int ia) { return [&, ia](int k) -> float& { return an(ix, jgs, k, ia); }; };
auto column1 = [&](NsslArray<float, 3>& t) { return [&](int k) -> float& { return t(ix, jgs, k); }; };
for (int il = lc; il <= lhab; ++il) {
if (((ido(il)) == (0))) { continue; }
float v1[NSSL_FAST_ROWS], v2[NSSL_FAST_ROWS], v3[NSSL_FAST_ROWS];
float vtmax = 0.0f;
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) {
v1[r] = 0.0f; v2[r] = 0.0f; v3[r] = 0.0f;
if (act[r]) {
v1[r] = nssl_fast_xvt[((il - lc) * 3 + 0) * plane + at[r]];
v2[r] = nssl_fast_xvt[((il - lc) * 3 + 1) * plane + at[r]];
v3[r] = nssl_fast_xvt[((il - lc) * 3 + 2) * plane + at[r]];
}
if (live[r]) {
v1[r] = nssl_min((float)(vtmaxsed), (float)(v1[r]));
v2[r] = nssl_min((float)(vtmaxsed), (float)(v2[r]));
v3[r] = nssl_min((float)(vtmaxsed), (float)(v3[r]));
vtmax = nssl_max((float)(vtmax), (float)(nssl_fmul(v1[r], dt0[r])));
vtmax = nssl_max((float)(vtmax), (float)(nssl_fmul(v2[r], dt0[r])));
vtmax = nssl_max((float)(vtmax), (float)(nssl_fmul(v3[r], dt0[r])));
}
}
for (int o = 16; o > 0; o >>= 1) vtmax = fmaxf(vtmax, __shfl_xor_sync(full, vtmax, o));
if (((vtmax) == (0.0f))) { continue; }
int ndfall;
if (((nssl_fmul(dtp, vtmax)) < (0.7f))) {
ndfall = 1;
} else {
if (((dtp) > (20.0f))) {
ndfall = nssl_max((int)(2), (int)(((((int)(nssl_fdiv(nssl_fmul(dtp, vtmax), 0.7f)))) + (1))));
} else {
ndfall = ((1) + (((int)(nssl_fadd(nssl_fmul(dtp, vtmax), 0.301f)))));
}
}
float dtptmp;
if (((ndfall) > (1))) {
dtptmp = nssl_fdiv(dtp, ((float)(ndfall)));
} else {
dtptmp = dtp;
}
const float dtfrac = nssl_fdiv(dtptmp, dtp);
for (int n = 1; n <= ndfall; ++n) {
if (((((((il) >= (lr))) && (((((infall) == (3))) || (((infall) == (4))))))) && (((ln(il)) > (0))))) {
if (((((((((il) == (lr))) && (((irfall) == (infall))))) && (((lzr) < (1))))) || (((((il) >= (lh))) && (((lz(il)) < (1))))))) {
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) if (live[r]) f_calczgr1d_lvl(kk[r], nx, ny, nz, nor, na, an, ixe, kze, z, db1, jgs, ipconc, dnu(il), il, ln(il), qxmin(il), xvmn(il), xvmx(il), lvol(il), xdn0(il), ix);
}
}
fast_sed_fallout(species(il), nz, live, kk, v1, dt1, db, 1, dtptmp, dtfrac, &xfall(ix, jy, il));
if (((ldovol) && (((il) >= (li))))) {
if (((lvol(il)) > (1))) {
fast_sed_fallout(species(lvol(il)), nz, live, kk, v1, dt0, db, 0, dtptmp, dtfrac, &xfall(ix, jy, lvol(il)));
}
}
if (((ipconc) >= (6))) {
if (((lz(il)) > (1))) {
fast_sed_fallout(species(lz(il)), nz, live, kk, v3, dt0, db, 0, dtptmp, dtfrac, &xfall(ix, jy, lz(il)));
}
}
if (((ipconc) > (0))) {
if (((ipconc) >= (ipc(il)))) {
if (((((((infall) >= (2))) || (((((infall) == (0))) && (((il) < (lh))))))) && (((lz(il)) < (1))))) {
if (((((((infall) == (3))) || (((infall) == (4))))) && (((((((il) == (lh))) || (((il) == (lhl))))) || (((((il) == (lr))) && (((irfall) == (infall))))))))) {
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) if (live[r]) { tmpn2(ix, jy, kk[r]) = z(ix, kk[r], il); tmpn(ix, jy, kk[r]) = an(ix, jy, kk[r], ln(il)); }
} else {
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) if (live[r]) { tmpn(ix, jy, kk[r]) = an(ix, jy, kk[r], ln(il)); }
}
}
int in = 2;
if (((infall) == (1))) { in = 1; }
fast_sed_fallout(species(ln(il)), nz, live, kk, in == 1 ? v1 : v2, dt0, db, 0, dtptmp, dtfrac, &xfall(ix, jy, ln(il)));
if (((lz(il)) < (1))) {
if (((((((((infall) >= (2))) || (((infall) == (3))))) && ((!(((((infall) == (0))) && (((il) >= (lh))))))))) && (((((il) == (lr))) || (((((il) >= (li))) && (((il) <= (lhab))))))))) {
// native's xfall0 (these passes' outflow) is never read: no outflow is kept for them.
if (((((((infall) == (3))) || (((infall) == (4))))) && (((((il) >= (lh))) || (((((il) == (lr))) && (((irfall) == (infall))))))))) {
fast_sed_fallout(column1(tmpn2), nz, live, kk, v3, dt0, db, 0, dtptmp, dtfrac, nullptr);
fast_sed_fallout(column1(tmpn), nz, live, kk, v1, dt0, db, 0, dtptmp, dtfrac, nullptr);
} else {
fast_sed_fallout(column1(tmpn), nz, live, kk, v1, dt0, db, 0, dtptmp, dtfrac, nullptr);
}
if (((((((infall) == (3))) || (((infall) == (4))))) && (((((((il) == (lr))) && (((irfall) == (infall))))) || (((il) >= (lh))))))) {
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) if (live[r]) f_calcnfromz1d_lvl(kk[r], nx, ny, nz, nor, na, an, tmpn2, ixe, kze, z, db1, jgs, ipconc, dnu(il), il, ln(il), qxmin(il), xvmn(il), xvmx(il), tmpn, lvol(il), rho_qh, infall, ix);
} else if (((((((infall) == (5))) && (((il) >= (lh))))) || (((((il) == (lr))) && (((irfall) == (5))))))) {
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) if (live[r]) { const int k = kk[r]; an(ix, jgs, k, ln(il)) = nssl_max((float)(an(ix, jgs, k, ln(il))), (float)(nssl_fmul(0.5f, nssl_fadd(an(ix, jgs, k, ln(il)), tmpn(ix, jy, k))))); }
} else if ((!(((((il) == (lr))) && (((irfall) == (0))))))) {
#pragma unroll
for (int r = 0; r < NSSL_FAST_ROWS; ++r) if (live[r]) { const int k = kk[r]; an(ix, jgs, k, ln(il)) = nssl_max((float)(an(ix, jgs, k, ln(il))), (float)(tmpn(ix, jy, k))); }
}
}
}
}
}
}
}
__syncwarp();
#endif
}
"""

ENTRY = r"""
// rrfs/nssl-fast sed: launches of the split sedimentation (fast numerics only).
extern "C" __global__ void nssl_fast_sed_info(unsigned char* parameters, int* out) {
const NsslState& s = *reinterpret_cast<const NsslState*>(parameters);
out[0] = s.lc; out[1] = s.lhab; out[2] = s.do_accurate_sedimentation ? 1 : 0; out[3] = (int)sizeof(NsslFastSedDesc);
}
extern "C" __global__ void nssl_fast_sed_bind(float* xvt, unsigned char* act, unsigned char* desc, int cols) {
nssl_fast_xvt = xvt; nssl_fast_act = act; nssl_fast_desc = reinterpret_cast<NsslFastSedDesc*>(desc); nssl_fast_cols = cols;
}
// Points in grid stride, consecutive threads on consecutive columns at one level; each thread copies the module
// context once for all its points. A native STOP inside the fall speeds (f_gamma_sp sets native_error) reaches the
// column's error word as in the column route.
extern "C" __global__ void nssl_sedfall(int first, int count, int nz, unsigned char* parameters, int* errors) {
NsslArena unused{nullptr, 0, 1, 0, 0, 0};
NsslColumn column{{}, unused};
column.copy_globals(*reinterpret_cast<const NsslState*>(parameters));
const long long total = (long long)count * nz;
for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
const int col = (int)(t % count);
const int kz = (int)(t / count) + 1;
const NsslFastSedDesc& d = nssl_fast_desc[col];
column.native_error = 0;
column.f_fast_sed_point(kz, col, d.an, d.dn, d.t0, d.t7, d.dtp, d.nz, d.na, d.nor, d.infdo);
if (column.native_error != 0) atomicMax(errors + first + col, column.native_error);
}
}
"""


def apply(body: str, entry: str, mode: str | None = None) -> tuple[str, str]:
    """Add the split sedimentation to the assembled fast module (body) and entry; the column route stays."""
    mode = fall_mode() if mode is None else mode
    if "f_fast_sed_lanes" in body:
        raise ValueError("NSSL fast sed: already applied")
    # Device buffers and the column's batch index (set by the phase entry) for the fall-speed launch.
    body = _one(body, "struct NsslColumn : NsslState {\nNsslArena& arena;",
                DEVICE_GLOBALS + "struct NsslColumn : NsslState {\nNsslArena& arena;\nint fast_column = 0;", "column struct")
    # Starts a line: the shared maths text before the module ends without a newline.
    constants = (f"\n#define NSSL_FAST_ROWS {ROWS}\n#define NSSL_FAST_MAX_SPECIES {MAX_SPECIES}\n"
                 f"#define NSSL_FAST_SCRATCH {SCRATCH}\n")
    body = constants + body
    s, _ = _routine(body, "f_sediment1d")
    body = body[:s] + _level_variants(body) + "\n" + MEMBERS + "\n" + body[s:]

    # Phase 7 = the sediment phase up to f_sediment1d (+ the fall-speed hand-off); phase 6 = the lanes and native's
    # surface accumulation after f_sediment1d, run by lane 0. Phase 1 (native's whole sediment phase) stays.
    t = body.index("template<int Phase> __device__ __forceinline__ void f_nssl_2mom_phase(")
    t_end = _block_end(body, t)
    phase = body[t:t_end]
    p1 = phase.index("if constexpr (Phase == 1) {")
    p1_end = _block_end(phase, p1)
    block = phase[phase.index("{", p1) + 1:p1_end - 1]
    if block.count(_SED_CALL) != 1:
        raise ValueError("NSSL fast sed: the sediment phase must call f_sediment1d once")
    call = block.index(_SED_CALL)
    pre, post = block[:call], block[call + len(_SED_CALL):]
    handoff = ("f_fast_sed_export(dtp, nz, na, nor, infdo, an, dn1, t0, t7);" if mode == "point"
               else "f_fast_sed_fall_column(dtp, nz, na, nor, infdo, an, dn1, t0, t7);")
    lanes = _SED_CALL.replace("f_sediment1d(", "f_fast_sed_lanes(", 1)
    added = ("if constexpr (Phase == 7) {" + pre + handoff + "\n}\n"
             "if constexpr (Phase == 6) {\n" + lanes + "\nif (NSSL_FAST_LANE == 0) {" + post + "\n}\n}\n")
    p4 = phase.index("if constexpr (Phase == 4) {")
    p4_end = _block_end(phase, p4)
    save = phase[p4_end:phase.rindex("}")]
    if not save.strip().startswith("saved.") or save.count("saved.") < 50:
        raise ValueError("NSSL fast sed: the phase driver's state save moved")
    # Lanes 1-31 of phase 6 hold copies of lane 0's driver state; only lane 0 saves it.
    phase = (phase[:p4_end] + "\n" + added + "if (Phase != 6 || NSSL_FAST_LANE == 0) {" + save + "}\n}")
    body = body[:t] + phase + body[t_end:]

    # Entry: the column's batch index for the hand-off, lane-0-only context write-back in phase 6, three kernels.
    b = entry.index("template<int Phase> __device__ __forceinline__ void nssl_phase_body(")
    b_end = _block_end(entry, b)
    pb = entry[b:b_end]
    pb = _one(pb, "NsslColumn column{{},arena};", "NsslColumn column{{},arena}; column.fast_column=workspace_column;",
              "phase body column")
    tail = "if constexpr(Phase == 0) { context.globals=column; } else {"
    pb = _one(pb, tail, "if constexpr(Phase == 6) {\n#ifdef __CUDA_ARCH__\ncolumn.native_error=__reduce_max_sync(0xffffffffu,"
              "column.native_error);\n#endif\nif (NSSL_FAST_LANE != 0) return;\n}\n" + tail, "phase body tail")
    entry = entry[:b] + pb + entry[b_end:]
    k = entry.index('extern "C" __global__ void nssl_sediment(')
    kernel = entry[k:_block_end(entry, k)]
    pre_kernel = kernel.replace("void nssl_sediment(", "void nssl_sedpre(", 1).replace("nssl_phase_body<1>(", "nssl_phase_body<7>(", 1)
    lane_kernel = kernel.replace("void nssl_sediment(", "void nssl_sedlanes(", 1).replace("nssl_phase_body<1>(", "nssl_phase_body<6>(", 1)
    lane_kernel = _one(lane_kernel, "int thread=blockIdx.x*blockDim.x+threadIdx.x;",
                       "int thread=(blockIdx.x*blockDim.x+threadIdx.x)>>5;", "lane kernel thread map")
    entry += "\n" + pre_kernel + "\n" + lane_kernel + "\n" + ENTRY
    return body, entry
