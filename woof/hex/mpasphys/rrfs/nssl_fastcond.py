"""NSSL condensation per point (rrfs/nssl-fast stage ``cond``): fast numerics only, after stage ``sed``.

Native f_nucond (RRFS module_mp_nssl_2mom.F; ``nssl_condense``'s column thread) first walks the column once (t0 and
the supersaturation ssfilt at every level, from the state before any update), then gathers the points that condense
into ngs-point work arrays in device memory and updates them, and finally walks the column again for the number
limits. A gathered point reads other levels only through ssfilt (k-1, k+1) and the immutable w, and the final walk
is per level, so a point computes the same when the first walk is done for the whole column before it.
THE BREAKAGE THIS ENDS: one thread per column walking ngs-point work arrays in device memory was the slot's third
largest phase (28.8 of 164 ms, mv3km call-120 replay, RTX 4090, 2026-10-29).

* ``nssl_condpre`` (thread per column): the condensation phase with f_nucond replaced by its first walk, then the
  column's f_nucond arguments for the per-point launch;
* ``nssl_condpoint`` (grid stride over points, consecutive threads on consecutive columns at one level, the module
  context copied once per thread): native f_nucond after its first walk for ONE level (``f_nucond_lvl``), its work
  arrays and frames in thread-local scratch. renucfrac, a module scalar native sets at run time, comes from the
  column's phase context.

Native's condensation phase (``nssl_condense``) stays and runs wherever the split sedimentation does not apply.
"""
from __future__ import annotations

import re

from woof.hex.mpasphys.rrfs.nssl_fastsed import _block_end, _one, _routine

SCRATCH = 3072  # bytes of thread-local scratch for one point's f_nucond work arrays and frames (2,488 needed at 128 levels, 16 species)

_CALL = ("f_nucond(nx, ny, nz, na, jy, nor, nor, dtp, nx, dz2d, t0, t9, an, dn1, t77, pn, wn, axtra2d, makediag, "
         "ssat, t00, t77, flag_qndrop);")
_WALK = "for (int nssl_first_1=((kz=1),0), nssl_end_1=(((nz) - (kstag))), "


def _variants(body: str) -> str:
    s, e = _routine(body, "f_nucond")
    nucond = body[s:e]
    walk = nucond.index(_WALK)
    walk_end = _block_end(nucond, walk)
    # The first walk alone (nssl_condpre): native up to and through it.
    after = nucond.index("jgs = jy;", walk_end)
    pre = nucond[:after].replace("void f_nucond(", "void f_nucond_pre(", 1)
    if pre.count("goto nssl_label_2200;") != 1:
        raise ValueError("NSSL fast cond: the first walk's skip moved")
    pre = pre.replace("goto nssl_label_2200;", "return;") + "return;\n}"
    if re.search(r"goto nssl_label_\d+;", pre):
        raise ValueError("NSSL fast cond: the first walk jumps elsewhere")
    # One level after the first walk (nssl_condpoint).
    lvl = (nucond[:walk] + nucond[walk_end:]).replace("void f_nucond(", "void f_nucond_lvl(int selected_level, ", 1)
    for old, new in (("const int ngs = nssl_min(500, nx*nz);", "constexpr int ngs = 1;"),
                     ("\nnzmpb = 1;\n", "\nnzmpb = selected_level;\n"),
                     ("numgs = ((((nxz) / (ngs))) + (1));", "numgs = 1;"),
                     ("kze = ((nz) - (kstag));", "kze = selected_level;"),
                     ("nssl_first_24=((kz=1),0), nssl_end_24=(nz),",
                      "nssl_first_24=((kz=selected_level),0), nssl_end_24=(selected_level),")):
        lvl = _one(lvl, old, new, "nucond level")
    if re.search(r"\(\(kz=1\),0\)", lvl):
        raise ValueError("NSSL fast cond: f_nucond has another full-column walk")
    return pre + "\n" + lvl


GLOBALS = r"""
// rrfs/nssl-fast cond: the column's f_nucond arguments for the per-point launch.
struct NsslFastCondDesc { NsslArray<float, 3> dz3d, t0, t9, dn, p2, pn, w, ssfilt, t00, t77; NsslArray<float, 4> an, axtra;
  float dtp; int nx, ny, nz, na, jyslab, nor, norz, nxi; bool io_flag, flag_qndrop; };
__device__ NsslFastCondDesc* nssl_fast_cond = nullptr;
"""

MEMBERS = r"""
// rrfs/nssl-fast cond (nssl_condpre): native f_nucond's first walk for this column, then its arguments for the
// per-point launch. Same arguments as f_nucond.
__device__ __forceinline__ void f_fast_cond_column(int nx, int ny, int nz, int na, int jyslab, int nor, int norz, float dtp, int nxi, NsslArray<float, 3> dz3d, NsslArray<float, 3> t0, NsslArray<float, 3> t9, NsslArray<float, 4> an, NsslArray<float, 3> dn, NsslArray<float, 3> p2, NsslArray<float, 3> pn, NsslArray<float, 3> w, NsslArray<float, 4> axtra, bool io_flag, NsslArray<float, 3> ssfilt, NsslArray<float, 3> t00, NsslArray<float, 3> t77, bool flag_qndrop) {
f_nucond_pre(nx, ny, nz, na, jyslab, nor, norz, dtp, nxi, dz3d, t0, t9, an, dn, p2, pn, w, axtra, io_flag, ssfilt, t00, t77, flag_qndrop);
NsslFastCondDesc& d = nssl_fast_cond[fast_column];
d.dz3d = dz3d; d.t0 = t0; d.t9 = t9; d.dn = dn; d.p2 = p2; d.pn = pn; d.w = w; d.ssfilt = ssfilt; d.t00 = t00; d.t77 = t77;
d.an = an; d.axtra = axtra; d.dtp = dtp; d.nx = nx; d.ny = ny; d.nz = nz; d.na = na; d.jyslab = jyslab; d.nor = nor;
d.norz = norz; d.nxi = nxi; d.io_flag = io_flag; d.flag_qndrop = flag_qndrop;
}
// rrfs/nssl-fast cond (nssl_condpoint): native f_nucond after its first walk for ONE level; its work arrays and frames
// in this thread's local scratch.
__device__ __forceinline__ void f_fast_cond_point(int kz, const NsslFastCondDesc& d) {
alignas(16) unsigned char scratch[NSSL_FAST_COND_SCRATCH];
NsslArena la{scratch, 0, 1, 0, 0, 0};
NsslArena saved_arena = arena;
arena = la;
f_nucond_lvl(kz, d.nx, d.ny, d.nz, d.na, d.jyslab, d.nor, d.norz, d.dtp, d.nxi, d.dz3d, d.t0, d.t9, d.an, d.dn, d.p2, d.pn, d.w, d.axtra, d.io_flag, d.ssfilt, d.t00, d.t77, d.flag_qndrop);
arena = saved_arena;
}
"""

ENTRY = r"""
// rrfs/nssl-fast cond: launches of the per-point condensation (fast numerics only).
extern "C" __global__ void nssl_fast_cond_bind(unsigned char* desc) {
nssl_fast_cond = reinterpret_cast<NsslFastCondDesc*>(desc);
}
extern "C" __global__ void nssl_fast_cond_info(int* out) { out[0] = (int)sizeof(NsslFastCondDesc); }
// Points in grid stride, consecutive threads on consecutive columns at one level; the module context is copied once
// per thread and each column's renucfrac comes from its phase context.
extern "C" __global__ void nssl_condpoint(int first, int count, int nz, unsigned char* parameters,
    const NsslPhaseState* contexts, int* errors) {
NsslArena unused{nullptr, 0, 1, 0, 0, 0};
NsslColumn column{{}, unused};
column.copy_globals(*reinterpret_cast<const NsslState*>(parameters));
const long long total = (long long)count * nz;
for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long)gridDim.x * blockDim.x) {
const int col = (int)(t % count);
const int kz = (int)(t / count) + 1;
column.renucfrac = contexts[col].globals.renucfrac;
column.native_error = 0;
column.f_fast_cond_point(kz, nssl_fast_cond[col]);
if (column.native_error != 0) atomicMax(errors + first + col, column.native_error);
}
}
"""


def apply(body: str, entry: str) -> tuple[str, str]:
    """Add the per-point condensation to the assembled fast module; native's condensation phase stays."""
    if "f_fast_cond_column" in body:
        raise ValueError("NSSL fast cond: already applied")
    if "int fast_column = 0;" not in body:
        raise ValueError("NSSL fast cond: needs the split sedimentation's column index (stage sed)")
    body = _one(body, "struct NsslColumn : NsslState {\n", GLOBALS + "struct NsslColumn : NsslState {\n", "cond globals")
    body = f"\n#define NSSL_FAST_COND_SCRATCH {SCRATCH}\n" + body
    s, _ = _routine(body, "f_nucond")
    body = body[:s] + _variants(body) + "\n" + MEMBERS + "\n" + body[s:]
    t = body.index("template<int Phase> __device__ __forceinline__ void f_nssl_2mom_phase(")
    t_end = _block_end(body, t)
    phase = body[t:t_end]
    p3 = phase.index("if constexpr (Phase == 3) {")
    p3_end = _block_end(phase, p3)
    block = phase[phase.index("{", p3) + 1:p3_end - 1]
    if block.count(_CALL) != 1:
        raise ValueError("NSSL fast cond: the condensation phase must call f_nucond once")
    added = ("if constexpr (Phase == 9) {" + block.replace(_CALL, _CALL.replace("f_nucond(", "f_fast_cond_column(", 1))
             + "\n}\n")
    phase = phase[:p3_end] + "\n" + added + phase[p3_end:]
    body = body[:t] + phase + body[t_end:]
    k = entry.index('extern "C" __global__ void nssl_condense(')
    kernel = entry[k:_block_end(entry, k)]
    entry += ("\n" + kernel.replace("void nssl_condense(", "void nssl_condpre(", 1)
              .replace("nssl_phase_body<3>(", "nssl_phase_body<9>(", 1) + "\n" + ENTRY)
    return body, entry
