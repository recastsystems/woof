"""Fast TEMPO as Thompson-style phase kernels: cell-local processes parallel over (level, cell).

Fast numerics only. Strict TEMPO is untouched: tempo_v303 builds the strict module from ``_COLUMN`` byte for
byte and never imports this file in strict mode.

Shape of one microphysics substep (n_microp loops it):

* phase A, B, C (module P): one thread per (level, cell); a warp is 32 neighbouring cells at one level, so a
  species branch is taken by the whole warp or by none of it (the warp-per-column kernel paid for every
  lane of both 32-level passes whenever any level had the species).
  - A: pack (theta * exner), zeroing, the first check-and-updates, snow moments, thermo; column flag
    "micro" (any hydrometeor or ice supersaturation) as a benign same-value store.
  - B (micro columns): every process rate, depletion, tendency sum, the second check-and-updates, thermo;
    the cloud-condensation stop level (first level from the ground with |ssatw| < eps) by atomicMin.
  - C (micro columns): cloud condensation below the stop level, rain evaporation, their check-and-updates
    and thermo.
* tail (module S, one warp per column, lanes across levels): sedimentation (vertically dependent), the
  final update and check-and-updates, diagnostics, PBL subgrid radii, the step accumulations and the
  write-back to the pools. The arrays the tail reads are gathered from the (level, cell) work area into
  the warp's contiguous work area first.

Every statement is the fast column's own (fast_source(fast_dp_source(_COLUMN)) with the same operands per
level), so the words match the fast warp kernel except where the parameter storage changes them: the
table blob is held in binary32 (the fast column already read every parameter and table as binary32; the
binary64 storage only cost a binary64 load and conversion per read). The fast maths shim's binary32
overloads also stop comparing against binary64 literals.

THE BREAKAGE THIS ENDS: at 3 km (mv3km, 206,726 columns, 59 levels) the fast TEMPO slot cost 143 ms per
call against Thompson's 22 ms for comparable physics; one 255-register warp-per-column kernel ran at the
binary64 pipe's pace and paid whole warp instructions for lanes at levels without the species.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache

import numpy as np

from woof.hex import numerics as _hex_numerics

#: Tile width: the (level, cell) work area holds 32 neighbouring cells per level contiguously.
TILE = 32
#: Tiles per phase chunk (65,536 cells): bounds the work area (about 3 GB at 59 levels) on any mesh.
CHUNK_TILES = 2048
#: Phase kernels: tiles per block (blockDim.x = 32 * TILES_PER_BLOCK, grid.y = levels).
TILES_PER_BLOCK = 2


# ---------------------------------------------------------------------------------------------------
# Source transforms
# ---------------------------------------------------------------------------------------------------

def _float_blob(source: str) -> str:
    """The table blob in binary32 (fast only): every TK_ kind is already binary32 in the fast column."""
    old = "#define TEMPO_TX_MEMBER(name, d1, d2, d3, d4, d5) double name[(d1) * (d2) * (d3) * (d4) * (d5)];"
    if source.count(old) != 1:
        raise ValueError("TEMPO phase: blob member macro not found once")
    source = source.replace(old, old.replace("double name", "float name"))
    if re.search(r"^using TK_\w+ = double;$", source, re.M):
        raise ValueError("TEMPO phase: a binary64 table kind survived the fast column")
    return source


def _float_shim(source: str) -> str:
    """The fast maths shim's binary32 overloads compare in binary32 (no binary64 compare or widening)."""
    out, hits = [], 0
    for line in source.split("\n"):
        if line.startswith("TEMPO_HD float tempo_") and ("1.2e-38" in line or "(double)y" in line):
            new = (line.replace("1.2e-38 ", "1.2e-38f ").replace("3.0e38)", "3.0e38f)")
                   .replace("fabs((double)y) <= 128.0", "fabsf(y) <= 128.0f"))
            new = new.replace("x > 1.2e-38f && x < 3.0e38 ", "x > 1.2e-38f && x < 3.0e38f ")
            hits += new != line
            line = new
        out.append(line)
    if hits < 3:
        raise ValueError(f"TEMPO phase: binary32 maths overloads not found ({hits})")
    return "\n".join(out)


@lru_cache(maxsize=None)
def _fast_column(tiny: str) -> str:
    from woof.hex.mpasphys.rrfs import tempo_v303 as t
    return t._column_source("fast")


def fast_column() -> str:
    """The fast column of tempo_v303 (rrfs/tempo-yolo: fixed-speed register sedimentation, a1 binary32
    arithmetic, the all-binary32 layer with its binary32 blob, constant-bank parameters and intrinsics, tiny
    graupel absent). The all-float layer already holds the blob in binary32, so the phase split's own blob and
    shim edits (_float_blob, _float_shim) are not applied over it."""
    import os
    return _fast_column(os.environ.get("WOOF_HEX_TEMPO_TINY_MASS", "graupel").strip().lower())


_SHAPE_OLD_START = "#ifndef TEMPO_WARP\n#define TEMPO_WARP 0\n#endif"
_SHAPE_OLD_END = "#define TEMPO_SLOTS(k, lo, hi) for (int k = (lo) + TEMPO_LANE; k < (hi); k += TEMPO_STRIDE)"
_SHAPE_NEW = r"""#define TEMPO_WARP 0
#define TEMPO_PHASE 1
// Phase shape: one thread per (level, cell); the level is blockIdx.y + 1, the cell is the x index.
// Every level loop runs exactly this thread's level (if it is in range).
#define tempo_pk ((int)blockIdx.y + 1)
#define TEMPO_LANE 0
#define TEMPO_STRIDE 1
#define TEMPO_SYNC() ((void)0)
#define TEMPO_LEAD (tempo_pk == 1)
#define TEMPO_LEVELS(k, n) for (int k = tempo_pk; k <= (n); k = 0x7ffffff0)
#define TEMPO_LEVELS_DOWN(k, n) TEMPO_LEVELS(k, n)
#define TEMPO_SLOTS(k, lo, hi) for (int k = tempo_pk; k >= (lo) && k < (hi); k = 0x7ffffff0)
// A level view of the (level, cell) work area: element k of this cell lives TEMPO_TILE words after k-1.
#define TEMPO_TILE 32
template<class T> struct TPV {
  T* p;
  __device__ __forceinline__ TPV() : p(nullptr) {}
  __device__ __forceinline__ TPV(decltype(nullptr)) : p(nullptr) {}
  __device__ __forceinline__ explicit TPV(T* q) : p(q) {}
  __device__ __forceinline__ T& operator[](long long k) const { return p[k * TEMPO_TILE]; }
  __device__ __forceinline__ bool operator==(decltype(nullptr)) const { return p == nullptr; }
  __device__ __forceinline__ bool operator!=(decltype(nullptr)) const { return p != nullptr; }
};"""

#: Routines whose pointer arguments are not level arrays (local radar bins) or that only the warp shape uses.
_KEEP_POINTERS = frozenset(("tempo_next_active", "tempo_simpson", "reflectivity_from_melting_graupel",
                            "reflectivity_from_melting_snow", "rayleigh_soak_wetgraupel"))
_FUNC = re.compile(r"^(?:TEMPO_HDN inline|TEMPO_HD) [\w:<> ]+?\b(\w+)\(", re.M)
_PARAM = re.compile(r"^(\s*)(?:const\s+)?(float|double|bool|int)\s*\*\s*(\w+)(\s*)$", re.S)


def _matching(text: str, start: int, open_: str, close: str) -> int:
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_:
            depth += 1
        elif ch == close:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("TEMPO phase: unbalanced source")


def _split_params(params: str):
    out, depth, cur = [], 0, []
    for ch in params:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _view_params(source: str):
    """Level-array pointer parameters become TPV level views; scalar out-parameters (dereferenced) stay pointers."""
    pieces, pos, changed = [], 0, 0
    for match in _FUNC.finditer(source):
        name = match.group(1)
        lp = match.end() - 1
        rp = _matching(source, lp, "(", ")")
        after = source[rp + 1:rp + 40].lstrip()
        if not after.startswith("{") or name in _KEEP_POINTERS:
            continue
        lb = source.index("{", rp)
        rb = _matching(source, lb, "{", "}")
        body = source[lb:rb + 1]
        params = _split_params(source[lp + 1:rp])
        new = []
        for prm in params:
            m = _PARAM.match(prm)
            if not m:
                new.append(prm); continue
            lead, kind, pname, tail = m.groups()
            indexed = re.search(r"\b" + pname + r"\s*\[", body) is not None
            deref = re.search(r"\*\s*" + pname + r"\b(?!\s*\[)", body) is not None
            if deref and not indexed:
                new.append(prm); continue
            new.append(f"{lead}TPV<{kind}> {pname}{tail}")
            changed += 1
        if lp + 1 < pos:
            continue
        pieces.append(source[pos:lp + 1])
        pieces.append(",".join(new))
        pos = rp
    pieces.append(source[pos:])
    return "".join(pieces), changed


def _cut(source: str, start_marker: str, end_marker: str) -> str:
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError(f"TEMPO phase: cut markers not found once: {start_marker[:40]!r} / {end_marker[:40]!r}")
    a, b = source.index(start_marker), source.index(end_marker)
    if b <= a:
        raise ValueError("TEMPO phase: cut markers out of order")
    return source[:a] + source[b:]


def _replace_once(source: str, old: str, new: str, what: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"TEMPO phase: {what} found {source.count(old)} times")
    return source.replace(old, new)


def phase_column(column: str) -> str:
    """The fast column for the (level, cell) phase kernels: level views, phase level loops, column steps out."""
    a = column.index(_SHAPE_OLD_START)
    b = column.index(_SHAPE_OLD_END) + len(_SHAPE_OLD_END)
    src = column[:a] + _SHAPE_NEW + column[b:]
    src = _replace_once(src, "#define TEMPO_TT_MEMBER(name) float name[TEMPO_NZP];",
                        "#define TEMPO_TT_MEMBER(name) TPV<float> name;", "TempoTend member macro")
    # Sedimentation and fall speeds are column work (the tail kernel); the main column is replaced by the phases.
    src = _cut(src, "TEMPO_HDN inline void sedimentation(", "template<class T> TEMPO_HD void tempo_copy(")
    src = _cut(src, "// ---------------------------------------------------------------------------\n"
                    "// tempo_main, lines 74-825", "// TEMPO 9adf9ef MPAS-Model-rrfs/src/core_atmosphere/physics/"
                    "physics_noaa/TEMPO/src/module_mp_tempo_ml.F90:28-81,104-203")
    src = src[:src.index("// Native TEMPO main:801-823")]
    # TempoWork: level arrays become level views.
    w0 = src.index("struct TempoWork {")
    w1 = _matching(src, w0, "{", "}")
    work = src[w0:w1]
    work, nwork = re.subn(r"^(float|double|int) (\w+)\[TEMPO_NZP\];$", r"TPV<\1> \2;", work, flags=re.M)
    work = _replace_once(work, "  bool l_qc[TEMPO_NZP], l_qi[TEMPO_NZP], l_qr[TEMPO_NZP], l_qs[TEMPO_NZP], l_qg[TEMPO_NZP];",
                         "  TPV<bool> l_qc, l_qi, l_qr, l_qs, l_qg;", "TempoWork flags")
    work = _replace_once(work, "  int idx_bg[TEMPO_NZP];", "  TPV<int> idx_bg;", "TempoWork idx_bg")
    src = src[:w0] + work + src[w1:]
    # tempo_copy / tempo_fill take views.
    src = _replace_once(src, "template<class T> TEMPO_HD void tempo_copy(T* dst,const T* src,int n) {TEMPO_LEVELS(k,n)dst[k]=src[k];}",
                        "template<class D,class S> TEMPO_HD void tempo_copy(D dst,S src,int n) {TEMPO_LEVELS(k,n)dst[k]=src[k];}",
                        "tempo_copy")
    src = _replace_once(src, "template<class T,class V> TEMPO_HD void tempo_fill(T* dst,V v,int n) {TEMPO_LEVELS(k,n)dst[k]=T(v);}",
                        "template<class T,class V> TEMPO_HD void tempo_fill(TPV<T> dst,V v,int n) {TEMPO_LEVELS(k,n)dst[k]=T(v);}",
                        "tempo_fill")
    # ice_processes / melting: t1_subl of this level only (no 103-word local array, no zeroing loop).
    get_call = "get_t1_subl(P, C, nz, rho, temp, qvsi, tcond, diffu, ssati, t1_subl);"
    if src.count("  float t1_subl[TEMPO_NZP];") != 2 or src.count(get_call) != 2 \
            or src.count("  for (int k = 0; k < TEMPO_NZP; ++k) t1_subl[k] = 0.0f;") != 2:
        raise ValueError("TEMPO phase: t1_subl sites changed")
    src = src.replace("  float t1_subl[TEMPO_NZP];", "  float t1_subl_v[1] = {0.0f};")
    src = src.replace("  for (int k = 0; k < TEMPO_NZP; ++k) t1_subl[k] = 0.0f;\n", "")
    src = src.replace(get_call, "get_t1_subl_one(P, nz, rho, temp, qvsi, tcond, diffu, ssati, t1_subl_v);")
    for fname in ("ice_processes", "melting"):
        s0 = src.index(f"TEMPO_HDN inline void {fname}(")
        lb = src.index("{", _matching(src, src.index("(", s0), "(", ")"))
        rb = _matching(src, lb, "{", "}")
        body = src[lb:rb]
        body, n = re.subn(r"\bt1_subl\[k\]", "t1_subl_v[0]", body)
        if n < 1 or re.search(r"\bt1_subl\b", body):
            raise ValueError(f"TEMPO phase: t1_subl uses in {fname} changed")
        src = src[:lb] + body + src[rb:]
    one = r"""
// Phase shape: get_t1_subl for this thread's level into one word.
TEMPO_HDN inline void get_t1_subl_one(const TempoParams& P, int nz, TPV<float> rho, TPV<float> temp,
    TPV<float> qvsi, TPV<float> tcond, TPV<float> diffu, TPV<float> ssati, float* t1) {
  float otemp; float  rvs; float  rvs_p; float  rvs_pp; float  gamsc; float  alphsc; float  xsat;
  const float lsub = PS(lsub); const float  orv = PS(orv);
  TEMPO_LEVELS(k, nz) {
    otemp = 1.0f / temp[k];
    rvs = rho[k] * qvsi[k];
    rvs_p = rvs * otemp * (lsub * otemp * orv - 1.0f);
    rvs_pp = rvs * (otemp * (lsub * otemp * orv - 1.0f) * otemp * (lsub * otemp * orv - 1.0f)
           + (-2.f * lsub * otemp * otemp * otemp * orv) + otemp * otemp);
    gamsc = lsub * diffu[k] / tcond[k] * rvs_p;
    alphsc = 0.5f * (gamsc / (1.0f + gamsc)) * (gamsc / (1.0f + gamsc)) * rvs_pp / rvs_p * rvs / rvs_p;
    alphsc = tmax(1.e-9f, alphsc);
    xsat = ssati[k];
    if (fabs(xsat) < 1.e-9f) xsat = 0.0f;
    *t1 = 4.0f * PS(pi) * (1.0f - alphsc * xsat + 2.0f * alphsc * alphsc * xsat * xsat
               - 5.0f * alphsc * alphsc * alphsc * xsat * xsat * xsat) / (1.0f + gamsc);
  }
}
"""
    marker = "TEMPO_HDN inline void ice_processes("
    cut = src.rfind("\n//", 0, src.index(marker))
    cut = src.rfind("\n", 0, cut) if cut > 0 else src.index(marker)
    src = src[:cut] + one + src[cut:]
    # cloud_condensation: the serial RETURN at the first level with |ssatw| < eps, as a column stop level.
    src = _replace_once(src, "    if (fabs(ssatw[k]) < PS(eps)) return;",
                        "    if (k >= tempo_kstop) return;", "cloud_condensation RETURN")
    src = _replace_once(src, "    const float* rc, const float* nc, TempoTend& T) {\n  float clap;",
                        "    const float* rc, const float* nc, TempoTend& T, int tempo_kstop) {\n  float clap;",
                        "cloud_condensation signature")
    src, changed = _view_params(src)
    if changed < 200:
        raise ValueError(f"TEMPO phase: only {changed} level-array parameters became views")
    return src


# ---------------------------------------------------------------------------------------------------
# Work-area layout: [array][tile][level 0..NZP-1][32 lanes] words; flags in a byte area of the same shape.
# ---------------------------------------------------------------------------------------------------

def _list(source: str, macro: str):
    m = re.search(r"#define " + macro + r"\((\w+)\) \\\n(.*?)\n(?!\s)", source, re.S)
    if not m:
        raise ValueError(f"TEMPO phase: {macro} not found")
    return re.findall(r"\b" + m.group(1) + r"\((\w+)\)", m.group(2))


def layout(source: str):
    """Word arrays (work, rates, idx_bg) and the 17 state carriers, in work-area order."""
    work = _list(source, "TEMPO_WORK_ARRAYS")
    tend = _list(source, "TEMPO_TEND_FIELDS")
    return work, tend


_FLAGS = ("l_qc", "l_qi", "l_qr", "l_qs", "l_qg")


def layout_macros(words: int, tile_major: bool) -> str:
    """Addresses in the work areas. Array-major ([array][tile][level][lane]): an array's stride is the chunk
    size (a run-time value). Tile-major ([tile][array][level][lane]): every array of a tile sits at a
    compile-time offset from the tile base, so one base register and immediates address them all."""
    head = f"#define TPH_WORDS {words}\n#define TPH_N ((long long)TEMPO_NZP * 32)\n"
    if tile_major:
        return head + (
            "#define TPH_WT(t) ((long long)(t) * TPH_WORDS * TPH_N)\n"
            "#define TPH_BT(t) ((long long)(t) * 5 * TPH_N)\n"
            "#define TPH_ST(t) ((long long)(t) * 17 * TPH_N)\n"
            "#define TPH_W(i) ((long long)(i) * TPH_N)\n")
    return head + (
        "#define TPH_WT(t) ((long long)(t) * TPH_N)\n"
        "#define TPH_BT(t) TPH_WT(t)\n"
        "#define TPH_ST(t) TPH_WT(t)\n"
        "#define TPH_W(i) ((long long)(i) * ntiles * TPH_N)\n")


def _work_binding(work, tend) -> str:
    lines = ["TEMPO_HD void tempo_phase_bind(TempoWork& W, float* wbuf, bool* bbuf, int ntiles, int tile, int lane) {",
             "  float* base = wbuf + TPH_WT(tile) + lane;",
             "  bool* bb = bbuf + TPH_BT(tile) + lane;"]
    ia = 0
    for name in work:
        lines.append(f"  W.{name} = TPV<float>(base + TPH_W({ia}));"); ia += 1
    for name in tend:
        lines.append(f"  W.T.{name} = TPV<float>(base + TPH_W({ia}));"); ia += 1
    lines.append(f"  W.idx_bg = TPV<int>(reinterpret_cast<int*>(base + TPH_W({ia})));"); ia += 1
    for j, name in enumerate(_FLAGS):
        lines.append(f"  W.{name} = TPV<bool>(bb + TPH_W({j}));")
    lines.append("}")
    return "\n".join(lines) + "\n", ia


_PHASES = r"""
// ------------------------------------------------------------------------------------------------
// Phase kernels (fast TEMPO, module_mp_tempo_main.F90:73-825 split at its column steps).
// x: 32 * TILES_PER_BLOCK cells (whole tiles), y: level.  sbuf holds the 17 carriers in the work-area
// shape (slot 0 = temperature); colflag/kstop are per cell of the chunk.
// ------------------------------------------------------------------------------------------------
#define TPH_PROLOGUE                                                                               \
  const int lane = threadIdx.x & 31;                                                               \
  const int tile = (int)blockIdx.x * ((int)blockDim.x >> 5) + ((int)threadIdx.x >> 5);              \
  if (tile >= ntiles) return;                                                                      \
  const int cl = tile * 32 + lane;                                                                 \
  const int c = c0 + cl;                                                                           \
  if (c >= cells) return;                                                                          \
  const int k = tempo_pk;                                                                          \
  (void)k;                                                                                         \
  TempoParams P = {blob};                                                                          \
  TempoCtx C = {dt, 1.0f / dt, aero, hail, 0, melt, 0, 1, 1, 1, 1, 1, 1, 0};                       \
  TempoWork W;                                                                                     \
  tempo_phase_bind(W, wbuf, bbuf, ntiles, tile, lane);                                             \
  TempoTend& T = W.T;                                                                              \
  (void)T;                                                                                         \
  float* sb = sbuf + TPH_ST(tile) + lane;                                                          \
  const TPV<float> t1d(sb + TPH_W(0));                                                          \
  const TPV<float> p1d(sb + TPH_W(1)), w1d(sb + TPH_W(2)), dz1d(sb + TPH_W(3));             \
  const TPV<float> qv1d(sb + TPH_W(4)), qc1d(sb + TPH_W(5)), qr1d(sb + TPH_W(6));           \
  const TPV<float> qi1d(sb + TPH_W(7)), qs1d(sb + TPH_W(8)), qg1d(sb + TPH_W(9));           \
  const TPV<float> ni1d(sb + TPH_W(10)), nr1d(sb + TPH_W(11));                               \
  const TPV<float> nc1d = (presence & 1) ? TPV<float>(sb + TPH_W(12)) : TPV<float>();            \
  const TPV<float> nwfa1d = (presence & 2) ? TPV<float>(sb + TPH_W(13)) : TPV<float>();          \
  const TPV<float> nifa1d = (presence & 4) ? TPV<float>(sb + TPH_W(14)) : TPV<float>();          \
  const TPV<float> ng1d = (presence & 8) ? TPV<float>(sb + TPH_W(15)) : TPV<float>();            \
  const TPV<float> qb1d = (presence & 16) ? TPV<float>(sb + TPH_W(16)) : TPV<float>();           \
  const TPV<float> ncsave = (presence & 1) ? TPV<float>() : W.ncsave;                               \
  bool supersaturated = false;                                                                     \
  (void)supersaturated;                                                                            \
  const float t0 = PS(t0);                                                                         \
  float tc0;                                                                                       \
  (void)tc0;

#define TPH_ARGS int nz, int cells, int c0, int ntiles, float** f, const TempoBlob* blob, int aero, \
  int hail, int melt, int presence, float dt, float* wbuf, bool* bbuf, float* sbuf, int* colflag, int* kstop

#define TEMPO_THERMO()                                                                      \
  thermo_vars(P, nz, W.qv, W.temp, W.pres, W.rho, W.rhof, W.rhof2, W.qvs, W.delqvs, W.qvsi, \
              W.satw, W.sati, W.ssatw, W.ssati, W.diffu, W.visco, W.vsc2, W.ocp, W.lvap,    \
              W.tcond, W.lvt2, &supersaturated)

extern "C" __global__ void __launch_bounds__(64) tempo_phase_a(TPH_ARGS) {
  TPH_PROLOGUE
  // tempo_pack: theta * exner in slot 0, absent carriers zero.
  {
    const long long x = (long long)(k - 1) * cells + c;
    float* s0 = sb + (long long)k * TEMPO_TILE;
    s0[0] = f[0][x] * f[17][x];
    for (int j = 1; j < 17; ++j) s0[TPH_W(j)] = f[j] ? f[j][x] : 0.0f;
  }
  W.tten[k] = 0.0f; W.qvten[k] = 0.0f; W.qcten[k] = 0.0f; W.qiten[k] = 0.0f; W.qrten[k] = 0.0f;
  W.qsten[k] = 0.0f; W.qgten[k] = 0.0f; W.ngten[k] = 0.0f; W.qbten[k] = 0.0f; W.niten[k] = 0.0f;
  W.nrten[k] = 0.0f; W.ncten[k] = 0.0f; W.nwfaten[k] = 0.0f; W.nifaten[k] = 0.0f;
  W.smo0[k] = 0.0f; W.smo1[k] = 0.0f; W.smo2[k] = 0.0f; W.smob[k] = 0.0f; W.smoc[k] = 0.0f;
  W.smoe[k] = 0.0f; W.smof[k] = 0.0f; W.smog[k] = 0.0f; W.smoz[k] = 0.0f; W.ns[k] = 0.0f;
  W.vtboost[k] = 1.0f;

  W.temp[k] = t1d[k];
  W.qv[k] = tmax(PS(min_qv), qv1d[k]);
  W.pres[k] = p1d[k];
  W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));

  aerosol_check_and_update(P, C, nz, W.rho, nwfa1d, nifa1d, W.nwfa, W.nifa, W.nwfaten, W.nifaten);
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, qr1d, nr1d, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
  ice_check_and_update(P, C, nz, W.rho, W.l_qi, qi1d, ni1d, W.ri, W.ni, W.qiten, W.niten, W.ilami);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, qs1d, W.rs, W.qsten);
  if (W.l_qs[k]) {
    tc0 = tmin(-0.1f, W.temp[k] - t0);
    snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], &W.ns[k], &W.smo0[k], &W.smo1[k],
                 &W.smo2[k], &W.smoe[k], &W.smof[k], &W.smog[k], nullptr);
  }
  if (nc1d == nullptr) {
    W.nc[k] = PS(nt_c_l);  // get_constant_cloud_number, land absent (ml_for_nc off at this pin)
    W.ncsave[k] = W.nc[k];
  }
  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, qc1d, nc1d, ncsave, W.rc, W.nc, W.qcten,
                         W.ncten, W.ilamc, W.mvd_c);
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  W.qcten[k] = 0.0f; W.qiten[k] = 0.0f; W.qrten[k] = 0.0f; W.qsten[k] = 0.0f; W.qgten[k] = 0.0f;
  W.ngten[k] = 0.0f; W.qbten[k] = 0.0f; W.niten[k] = 0.0f; W.nrten[k] = 0.0f; W.ncten[k] = 0.0f;
  TEMPO_THERMO();
  if (W.l_qc[k] || W.l_qr[k] || W.l_qi[k] || W.l_qs[k] || W.l_qg[k] || supersaturated) colflag[cl] = 1;
}

extern "C" __global__ void __launch_bounds__(64) tempo_phase_b(TPH_ARGS) {
  TPH_PROLOGUE
  if (!colflag[cl]) return;
#define TEMPO_TT_ZERO(name) T.name[k] = 0.0f;
  TEMPO_TEND_FIELDS(TEMPO_TT_ZERO)
#undef TEMPO_TT_ZERO
  if (!C.turn_off_micro_flag) {
    warm_rain(P, C, nz, W.rhof, W.l_qc, W.rc, W.nc, W.ilamc, W.mvd_c, W.l_qr, W.rr, W.nr, W.mvd_r, T);
    rain_snow_rain_graupel(P, C, nz, W.temp, W.l_qr, W.rr, W.nr, W.ilamr, W.l_qs, W.rs, W.l_qg,
                           W.rg, W.ng, W.ilamg, W.idx_bg, T);
    ice_nucleation(P, C, nz, W.temp, W.rho, w1d, W.qv, W.qvsi, W.ssati, W.ssatw, W.nifa, W.nwfa,
                   W.ni, W.smo0, W.rc, W.nc, W.rr, W.nr, W.ilamr, T);
    ice_processes(P, C, nz, W.rhof, W.rhof2, W.rho, w1d, W.temp, W.qv, W.qvsi, W.tcond, W.diffu,
                  W.vsc2, W.ssati, W.l_qi, W.ri, W.ni, W.ilami, W.l_qs, W.rs, W.smoe, W.smof,
                  W.smo1, W.rr, W.nr, W.ilamr, W.mvd_r, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);
    riming(P, C, nz, W.temp, W.rhof, W.visco, W.l_qc, W.rc, W.nc, W.ilamc, W.mvd_c, W.l_qs, W.rs,
           W.smo0, W.smob, W.smoc, W.smoe, W.vtboost, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);
    melting(P, C, nz, W.rhof2, W.rho, W.temp, W.qvsi, W.tcond, W.diffu, W.vsc2, W.ssati, W.delqvs,
            W.l_qs, W.rs, W.smof, W.smo0, W.smo1, W.l_qg, W.rg, W.ng, W.ilamg, W.idx_bg, T);
    aerosol_scavenging(P, C, nz, W.temp, W.rho, W.rhof, W.visco, W.nwfa, W.nifa, W.l_qr, W.nr,
                       W.ilamr, W.mvd_r, W.l_qs, W.rs, W.smob, W.smoc, W.smoe, W.l_qg, W.rg, W.ng,
                       W.ilamg, W.idx_bg, T);
  }
  check_over_depletion(P, C, nz, W.rho, W.temp, W.qvsi, W.qv, W.l_qc, W.rc, W.l_qi, W.ri, W.l_qr,
                       W.rr, W.l_qs, W.rs, W.l_qg, W.rg, T);
  sum_tendencies(P, nz, W.rho, W.temp, W.idx_bg, W.lvap, W.ocp, T, W.tten, W.qvten, W.qcten,
                 W.ncten, W.qiten, W.niten, W.qsten, W.qrten, W.nrten, W.qgten, W.ngten, W.qbten);
  {
    W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
    W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
    W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    W.nwfaten[k] = W.nwfaten[k] - (T.pna_rca[k] + T.pna_sca[k] + T.pna_gca[k] + T.pni_iha[k]) / W.rho[k];
    W.nifaten[k] = W.nifaten[k] - (T.pnd_rcd[k] + T.pnd_scd[k] + T.pnd_gcd[k]) / W.rho[k];
  }
  aerosol_check_and_update(P, C, nz, W.rho, nwfa1d, nifa1d, W.nwfa, W.nifa, W.nwfaten, W.nifaten);
  TPV<float> xncx, xngx, xqbx;
  tempo_copy(W.xrx, qc1d, nz);
  if (nc1d != nullptr) { tempo_copy(W.xncx, nc1d, nz); xncx = W.xncx; }
  cloud_check_and_update(P, C, nz, W.rho, W.l_qc, W.xrx, xncx, ncsave, W.rc, W.nc, W.qcten,
                         W.ncten, W.ilamc, W.mvd_c);
  tempo_copy(W.xrx, qr1d, nz);
  tempo_copy(W.xnx, nr1d, nz);
  rain_check_and_update(P, C, nz, W.rho, W.l_qr, W.xrx, W.xnx, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
  tempo_copy(W.xrx, qi1d, nz);
  tempo_copy(W.xnx, ni1d, nz);
  ice_check_and_update(P, C, nz, W.rho, W.l_qi, W.xrx, W.xnx, W.ri, W.ni, W.qiten, W.niten, W.ilami);
  tempo_copy(W.xrx, qs1d, nz);
  snow_check_and_update(P, C, nz, W.rho, W.l_qs, W.xrx, W.rs, W.qsten);
  W.smo0[k] = 0.0f; W.smo1[k] = 0.0f; W.smo2[k] = 0.0f; W.smob[k] = 0.0f; W.smoc[k] = 0.0f;
  W.smoe[k] = 0.0f; W.smof[k] = 0.0f; W.smog[k] = 0.0f; W.ns[k] = 0.0f;
  if (W.l_qs[k]) {
    tc0 = tmin(-0.1f, W.temp[k] - t0);
    snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], nullptr, nullptr, nullptr,
                 &W.smo2[k], nullptr, nullptr, nullptr, nullptr);
  }
  tempo_copy(W.xrx, qg1d, nz);
  if (ng1d != nullptr && qb1d != nullptr) {
    tempo_copy(W.xngx, ng1d, nz);
    tempo_copy(W.xqbx, qb1d, nz);
    xngx = W.xngx;
    xqbx = W.xqbx;
  }
  graupel_check_and_update(P, C, nz, W.rho, W.l_qg, W.xrx, xngx, xqbx, W.rg, W.ng, W.rb, W.idx_bg,
                           W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
  TEMPO_THERMO();
  // cloud_condensation's serial RETURN: the first level from the ground with |ssatw| < eps.
  if (fabs(W.ssatw[k]) < PS(eps)) atomicMin(&kstop[cl], k);
}

extern "C" __global__ void __launch_bounds__(64) tempo_phase_c(TPH_ARGS) {
  TPH_PROLOGUE
  if (!colflag[cl]) return;
  TPV<float> xncx;
  if (!C.turn_off_micro_flag && C.cloud_condensation_flag) {
    cloud_condensation(P, C, nz, W.rho, W.temp, w1d, W.ssatw, W.lvap, W.tcond, W.diffu, W.lvt2,
                       W.nwfa, W.qv, W.qvs, W.l_qc, W.rc, W.nc, T, kstop[cl]);
    {
      W.qvten[k] = W.qvten[k] - T.prw_vcd[k];
      W.qcten[k] = W.qcten[k] + T.prw_vcd[k];
      W.ncten[k] = W.ncten[k] + T.pnc_wcd[k];
      W.nwfaten[k] = W.nwfaten[k] - T.pnc_wcd[k];
      W.tten[k] = W.tten[k] + W.lvap[k] * W.ocp[k] * T.prw_vcd[k];
    }
    tempo_copy(W.xrx, qc1d, nz);
    if (nc1d != nullptr) { tempo_copy(W.xncx, nc1d, nz); xncx = W.xncx; }
    cloud_check_and_update(P, C, nz, W.rho, W.l_qc, W.xrx, xncx, ncsave, W.rc, W.nc, W.qcten,
                           W.ncten, W.ilamc, W.mvd_c);
    {
      W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
      W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
      W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    }
    TEMPO_THERMO();
  }
  if (!C.turn_off_micro_flag) {
    rain_evaporation(P, C, nz, W.rho, W.temp, W.ssatw, W.lvap, W.tcond, W.diffu, W.vsc2, W.rhof2,
                     W.qv, W.qvs, W.l_qr, W.rr, W.nr, W.ilamr, T);
    {
      W.qrten[k] = W.qrten[k] - T.prv_rev[k];
      W.qvten[k] = W.qvten[k] + T.prv_rev[k];
      W.nrten[k] = W.nrten[k] - T.pnr_rev[k];
      W.nwfaten[k] = W.nwfaten[k] + T.pnr_rev[k];
      W.tten[k] = W.tten[k] - W.lvap[k] * W.ocp[k] * T.prv_rev[k];
    }
    tempo_copy(W.xrx, qr1d, nz);
    tempo_copy(W.xnx, nr1d, nz);
    rain_check_and_update(P, C, nz, W.rho, W.l_qr, W.xrx, W.xnx, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
    {
      W.qv[k] = tmax(PS(min_qv), qv1d[k] + W.qvten[k] * C.global_dt);
      W.temp[k] = t1d[k] + W.tten[k] * C.global_dt;
      W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    }
    TEMPO_THERMO();
  }
}
#undef TEMPO_THERMO

// Phase D (fast TEMPO, module_mp_tempo_main.F90 after sedimentation, then the warp driver): every column.
// Micro columns: ice freezing/melting, the final update and check-and-updates, the diagnostics and the
// PBL subgrid radii for this level.  Every column: the unpack to the pools and the profiles.
extern "C" __global__ void __launch_bounds__(64) tempo_phase_d(TPH_ARGS, int last, int radii, int ml_bl,
    const float* bl, float** profiles) {
  TPH_PROLOGUE
  const int re = last && radii;
  C.refl10cm_flag = last; C.re_cloud_flag = re; C.re_ice_flag = re; C.re_snow_flag = re;
  C.max_hail_diameter_flag = last;
  const long long x = (long long)(k - 1) * cells + c;
  if (colflag[cl]) {
    if (!C.turn_off_micro_flag) {
      freeze_cloud_melt_ice(P, C, nz, W.temp, W.rho, W.ocp, W.lvap, qi1d, ni1d, W.qiten, W.niten,
                            qc1d, nc1d, W.qcten, W.ncten, W.tten, ncsave);
    }
    t1d[k] = t1d[k] + W.tten[k] * C.global_dt;
    qv1d[k] = tmax(PS(min_qv), (qv1d[k] + W.qvten[k] * C.global_dt));
    W.rho[k] = PS(roverrv) * W.pres[k] / (PS(rdry) * W.temp[k] * (W.qv[k] + PS(roverrv)));
    if (nwfa1d != nullptr)
      nwfa1d[k] = tmax(PS(nwfa_default), tmin(PS(aero_max), (nwfa1d[k] + W.nwfaten[k] * C.global_dt)));
    if (nifa1d != nullptr)
      nifa1d[k] = tmax(PS(nifa_default), tmin(PS(aero_max), (nifa1d[k] + W.nifaten[k] * C.global_dt)));
    cloud_check_and_update(P, C, nz, W.rho, W.l_qc, qc1d, nc1d, ncsave, W.rc, W.nc, W.qcten,
                           W.ncten, W.ilamc, W.mvd_c);
    rain_check_and_update(P, C, nz, W.rho, W.l_qr, qr1d, nr1d, W.rr, W.nr, W.qrten, W.nrten, W.ilamr, W.mvd_r);
    ice_check_and_update(P, C, nz, W.rho, W.l_qi, qi1d, ni1d, W.ri, W.ni, W.qiten, W.niten, W.ilami);
    snow_check_and_update(P, C, nz, W.rho, W.l_qs, qs1d, W.rs, W.qsten);
    W.smo0[k] = 0.0f; W.smo1[k] = 0.0f; W.smo2[k] = 0.0f; W.smob[k] = 0.0f; W.smoc[k] = 0.0f;
    W.smoe[k] = 0.0f; W.smof[k] = 0.0f; W.smog[k] = 0.0f; W.smoz[k] = 0.0f; W.ns[k] = 0.0f;
    if (W.l_qs[k]) {
      tc0 = tmin(-0.1f, W.temp[k] - t0);
      snow_moments(P, W.rs[k], tc0, &W.smob[k], &W.smoc[k], nullptr, nullptr, nullptr,
                   &W.smo2[k], nullptr, nullptr, nullptr, &W.smoz[k]);
    }
    graupel_check_and_update(P, C, nz, W.rho, W.l_qg, qg1d, ng1d, qb1d, W.rg, W.ng, W.rb, W.idx_bg,
                             W.qgten, W.ngten, W.qbten, W.ilamg, W.mvd_g);
    if (C.max_hail_diameter_flag) {
      max_hail_diam(P, C, nz, W.rho, W.rg, W.ng, W.ilamg, W.idx_bg, W.max_hail_diameter);
    } else {
      W.max_hail_diameter[k] = 0.0f;
    }
    if (C.refl10cm_flag) {
      reflectivity_10cm(P, C, nz, C.refl10cm_from_melting_flag, W.temp, W.l_qr, W.rr, W.nr, W.ilamr,
                        W.l_qs, W.rs, W.smoc, W.smob, W.smoz, W.l_qg, W.rg, W.ng, W.idx_bg, W.ilamg,
                        W.refl10cm);
    }
    if (C.re_cloud_flag && C.re_ice_flag && C.re_snow_flag) {
      effective_radius(P, C, nz, W.temp, W.l_qc, W.nc, W.ilamc, W.l_qi, W.ilami, W.l_qs, W.rs,
                       W.re_cloud, W.re_ice, W.re_snow);
    }
    // tempo_subgrid_cloud_radii (native TEMPO main:801-823) for this level; did_micro is this column flag.
    if (ml_bl && bl != nullptr && re) {
      W.xrx[k] = qc1d[k];
      const float qcb = bl[x], frb = bl[(long long)nz * cells + x];
      if (qc1d[k] <= PS(r1) && qcb > 1.0e-9f && frb > 0.0f) W.xrx[k] += qcb / frb;
      if (W.xrx[k] <= 1.0e-12f) W.xrx[k] = 0.0f;
      W.xncx[k] = tempo_cloud_number(W.xrx[k], qr1d[k], qi1d[k], qs1d[k], W.pres[k], W.temp[k], w1d[k]);
      W.qcten[k] = 0.0f;
      W.ncten[k] = 0.0f;
      cloud_check_and_update(P, C, nz, W.rho, W.l_qc, W.xrx, W.xncx, TPV<float>(),
                             W.rc, W.nc, W.qcten, W.ncten, W.ilamc, W.mvd_c);
      effective_radius(P, C, nz, W.temp, W.l_qc, W.nc, W.ilamc, W.l_qi, W.ilami,
                       W.l_qs, W.rs, W.re_cloud, W.re_ice, W.re_snow);
    }
  } else {
    W.max_hail_diameter[k] = 0.0f;
    W.refl10cm[k] = -35.0f; W.re_cloud[k] = 0.0f; W.re_ice[k] = 0.0f; W.re_snow[k] = 0.0f;
  }
  f[0][x] = t1d[k] / f[17][x];
  for (int j = 4; j < 17; ++j) if (f[j]) f[j][x] = sb[(long long)k * TEMPO_TILE + TPH_W(j)];
  if (last) {
    profiles[0][x] = W.refl10cm[k];
    if (radii) { profiles[1][x] = W.re_cloud[k]; profiles[2][x] = W.re_ice[k]; profiles[3][x] = W.re_snow[k]; }
  }
}

// Phase E: the column scalars (frozen fraction, freezing rain, hail maximum) and tempo_step_diags.
extern "C" __global__ void tempo_phase_e(int nz, int cells, int c0, int ntiles, float** s, const float* wbuf,
    const float* colsc, const int* colflag, const TempoBlob* blob, int ia_temp, int ia_hail) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = c0 + i;
  if (i >= ntiles * 32 || c >= cells) return;
  TempoParams P = {blob};
  const long long tb0 = TPH_WT(i >> 5) + (i & 31);
  float rain = 0.0f, cloud = 0.0f, ice = 0.0f, snow = 0.0f, graupel = 0.0f, frozen = 0.0f, frz = 0.0f;
  if (colflag[i]) {
    rain = colsc[5LL * i + 0]; cloud = colsc[5LL * i + 1]; ice = colsc[5LL * i + 2];
    snow = colsc[5LL * i + 3]; graupel = colsc[5LL * i + 4];
    frozen = (ice + snow + graupel) / (ice + snow + graupel + rain + PS(r1));
    freezing_rain(P, wbuf[TPH_W(ia_temp) + tb0 + 1LL * TEMPO_TILE], rain, cloud, &frz);
  }
  const float* hail = wbuf + TPH_W(ia_hail) + tb0;
  float maxhail = hail[1LL * TEMPO_TILE];
  for (int k = 2; k <= nz; ++k) maxhail = tmax(maxhail, hail[(long long)k * TEMPO_TILE]);
  s[2][c] = (s[2][c] + ice) + snow; s[3][c] = (s[3][c] + ice) + snow;
  s[4][c] += graupel; s[5][c] += graupel;
  s[0][c] = (((s[0][c] + rain) + ice) + snow) + graupel; s[1][c] = (((s[1][c] + rain) + ice) + snow) + graupel;
  s[6][c] = frozen; s[7][c] += frz;
  s[8][c] = hail[1LL * TEMPO_TILE]; s[9][c] = maxhail;
}
"""


def _inline(source: str, on: bool) -> str:
    """Force-inline the column routines (TEMPO_HDN): no call frames or stack in the phase kernels."""
    if not on:
        return source
    source = _replace_once(source, "#define TEMPO_HDN __device__" + chr(10),
                           "#define TEMPO_HDN __device__ __forceinline__" + chr(10), "TEMPO_HDN")
    return source.replace("TEMPO_HDN inline ", "TEMPO_HDN ")


def inline_mode() -> str:
    """Build variant, WOOF_HEX_TEMPO_PHASE_INLINE: P / S force-inline the column routines in module P / S,
    MBn caps the tail kernel to n resident blocks per SM (registers); e.g. "PS,MB12". Empty: none."""
    import os
    return os.environ.get("WOOF_HEX_TEMPO_PHASE_INLINE", DEFAULT_INLINE).upper().replace("NONE", "")


def _tail_minblocks(variant: str) -> int:
    m = re.search(r"MB(\d+)", variant)
    return int(m.group(1)) if m else 0


#: Default force-inline set (measured, see inline_mode).
DEFAULT_INLINE = ""


def phase_source(device_math_source: str, max_levels: int, inline: str = "") -> str:
    """Module P: the phase column and the (level, cell) phase kernels."""
    col = _inline(phase_column(fast_column()), "P" in inline.split("MB")[0].replace("TM", ""))
    work, tend = layout(col)
    binding, words = _work_binding(work, tend)
    return (f"#define TEMPO_NZ_MAX {max_levels}\n" + _shared_pow(device_math_source) + device_math_source
            + col + layout_macros(words, "TM" in inline) + binding + _PHASES)


def _shared_pow(device_math_source: str) -> str:
    from woof.hex.mpasphys.rrfs import tempo_v303 as t
    return t._shared_pow_define(device_math_source)


# ---------------------------------------------------------------------------------------------------
# Module S: the warp-per-column tail (sedimentation onward) and the binary32-blob warp kernel.
# ---------------------------------------------------------------------------------------------------

_TAIL_START = "  // sedimentation: rain\n"
#: With the fixed-speed sedimentation (tempo_fastsed_v303) the tail starts at its warp-shape block.
_TAIL_START_FASTSED = "#if TEMPO_WARP\n  // Fast numerics (tempo_fastsed_v303)"
_TAIL_END = "#undef TEMPO_THERMO\n}\n"


def _tail_text(column: str) -> str:
    a = column.index(_TAIL_START_FASTSED) if _TAIL_START_FASTSED in column else column.index(_TAIL_START)
    b = column.index(_TAIL_END, a)
    return column[a:b]


_POST_START = "  if (!C.turn_off_micro_flag) {\n    freeze_cloud_melt_ice("
#: Tendencies sedimentation changes and the post-sedimentation phase (D) reads back.
SED_OUT = ("qrten", "nrten", "qgten", "ngten", "qbten", "qsten", "qiten", "niten", "qcten", "ncten")
_SCALARS = ("rain_precip", "cloud_precip", "ice_liquid_equiv_precip", "snow_liquid_equiv_precip",
            "graupel_liquid_equiv_precip", "frz_rain_precip", "frozen_fraction", "did_micro", "T")
#: Read before written in the tail and not gathered: zeroed per column (main zeroes them at its start).
_ZEROED = ("vtrr", "vtnr", "vtrs", "vtri", "vtni", "vtrg", "vtng", "vtrc", "vtnc", "sed_r")
_UNREAD = {"refl10cm", "re_cloud", "re_ice", "re_snow", "max_hail_diameter", "rain_med_vol_diam",
           "graupel_med_vol_diam", "cloud_number_mixing_ratio", "xrx", "xnx", "xncx", "xngx", "xqbx"}


def _index(work, tend):
    ia = {n: i for i, n in enumerate(work)}
    for i, n in enumerate(tend):
        ia["T." + n] = len(work) + i
    ia["idx_bg"] = len(work) + len(tend)
    return ia


def _gather(names, tnames, ia):
    out = []
    for n in sorted(names):
        if n in _UNREAD or n in _ZEROED or n in _SCALARS:
            continue
        if n in _FLAGS:
            out.append(f"W.{n}[k] = bbuf[TPH_W({_FLAGS.index(n)}) + tbb];")
        elif n == "idx_bg":
            out.append(f"W.idx_bg[k] = reinterpret_cast<const int*>(wbuf)[TPH_W({ia[n]}) + tb];")
        elif n in ia:
            out.append(f"W.{n}[k] = wbuf[TPH_W({ia[n]}) + tb];")
        else:
            raise ValueError(f"TEMPO phase: tail reads unknown work array {n}")
    for n in sorted(tnames):
        out.append(f"W.T.{n}[k] = wbuf[TPH_W({ia['T.' + n]}) + tb];")
    return "\n        ".join(out)


#: Work arrays sedimentation reads per species (rain, graupel, snow, ice, cloud); the rest is always read.
_SED_SPECIES = (("a_r", ("nr", "qrten", "nrten", "ilamr", "mvd_r")),
                ("a_g", ("rg", "ng", "rb", "idx_bg", "ilamg", "mvd_g", "qgten", "ngten", "qbten", "visco")),
                ("a_s", ("rs", "smob", "smoc", "vtboost", "qsten", "T.prr_sml")),
                ("a_i", ("ri", "ni", "ilami", "qiten", "niten")),
                ("a_c", ("rc", "nc", "ilamc", "qcten", "ncten")))
_SED_ALWAYS = ("rho", "rhof", "rr")
#: State carriers sedimentation reads (w, dz, qr, qg, nr, ng, qb).
_SED_STATE = (2, 3, 6, 9, 11, 15, 16)


def _one(n, ia):
    if n in _FLAGS:
        return f"W.{n}[k] = bbuf[TPH_W({_FLAGS.index(n)}) + tbb];"
    if n == "idx_bg":
        return f"W.idx_bg[k] = reinterpret_cast<const int*>(wbuf)[TPH_W({ia[n]}) + tb];"
    if n.startswith("T."):
        return f"W.{n}[k] = wbuf[TPH_W({ia[n]}) + tb];"
    return f"W.{n}[k] = wbuf[TPH_W({ia[n]}) + tb];"


def _sed_gather(sed: str, ia) -> str:
    """Flags and the always-read arrays, then each present species' arrays (the column's any-flags)."""
    used = set(re.findall(r"\bW\.(\w+)", sed)) | {"T." + n for n in re.findall(r"\bT\.(\w+)", sed)}
    covered = set(_FLAGS) | set(_SED_ALWAYS) | {n for _, names in _SED_SPECIES for n in names}
    missing = {n for n in used if n not in covered and n not in _UNREAD and n not in _ZEROED and n not in _SCALARS}
    if missing:
        raise ValueError(f"TEMPO phase: sedimentation reads arrays outside the species map: {sorted(missing)}")
    first = " ".join(_one(n, ia) for n in _FLAGS + _SED_ALWAYS)
    lines = [f"TEMPO_LEVELS(k, nz) {{ const long long tb = tb0 + (long long)k * 32, tbb = tbb0 + (long long)k * 32; {first} }}",
             "TEMPO_SYNC();",
             "bool h_r = false, h_g = false, h_s = false, h_i = false, h_c = false;",
             "TEMPO_LEVELS(k, nz) { h_r = h_r || W.l_qr[k]; h_g = h_g || W.l_qg[k]; h_s = h_s || W.l_qs[k];"
             " h_i = h_i || W.l_qi[k]; h_c = h_c || W.l_qc[k]; }",
             "a_r = __any_sync(TEMPO_FULL, h_r); a_g = __any_sync(TEMPO_FULL, h_g); a_s = __any_sync(TEMPO_FULL, h_s);",
             "a_i = __any_sync(TEMPO_FULL, h_i); a_c = __any_sync(TEMPO_FULL, h_c);",
             "TEMPO_LEVELS(k, nz) {", "  const long long tb = tb0 + (long long)k * 32, tbb = tbb0 + (long long)k * 32;"]
    for flag, names in _SED_SPECIES:
        lines.append(f"  if ({flag}) {{ " + " ".join(_one(n, ia) for n in names) + " }")
    lines.append("}")
    return "\n        ".join(lines)


def _sed_scatter(ia) -> str:
    out = []
    for flag, names in _SED_SPECIES:
        tens = [n for n in names if n in SED_OUT]
        out.append(f"if ({flag}) {{ " + " ".join(f"wbuf[TPH_W({ia[n]}) + tb] = W.{n}[k];" for n in tens) + " }")
    return " ".join(out)


def _tail_kernel(column: str, work, tend) -> str:
    body = _tail_text(column)
    cut = body.index(_POST_START)
    sed, post = body[:cut], body[cut:]
    ia = _index(work, tend)
    radii = ("xrx", "xncx", "pres", "temp", "rho", "l_qc", "rc", "nc", "qcten", "ncten", "ilamc", "mvd_c",
             "l_qi", "ilami", "l_qs", "rs", "re_cloud", "re_ice", "re_snow")
    sed_names = set(re.findall(r"\bW\.(\w+)", sed))
    post_names = (set(re.findall(r"\bW\.(\w+)", post)) | set(radii) | {"ncsave"}) - sed_names
    sed_t = set(re.findall(r"\bT\.(\w+)", sed))
    gather_sed = _gather(sed_names, sed_t, ia)
    gather_post = _gather(post_names, set(re.findall(r"\bT\.(\w+)", post)) - sed_t, ia)
    zero = " ".join(f"W.{n}[k] = 0.0f;" for n in _ZEROED)
    scatter = _sed_scatter(ia)
    sed_gather = _sed_gather(sed, ia)
    sed_state = " ".join(f"st.f[{j}][k] = sbuf[TPH_W({j}) + tbs];" for j in _SED_STATE)
    return r"""
// ------------------------------------------------------------------------------------------------
// Phase tail (fast TEMPO): one warp per column, lanes across levels. module_mp_tempo_main.F90 from the
// rain sedimentation on. post=1: through the diagnostics, then the warp driver PBL radii, step
// diagnostics and unpack (every column).  post=0: micro columns only, sedimentation alone; the changed
// tendencies go back to the work area and the precipitation scalars to colsc (phases D and E finish).
// ------------------------------------------------------------------------------------------------
#ifndef TEMPO_TAIL_LB
#define TEMPO_TAIL_LB
#endif
extern "C" __global__ void TEMPO_TAIL_LB tempo_phase_tail(int nz, int cells, int c0, int ntiles, int last, int post,
    float** f, const float* bl, float** s, float** profiles, const TempoBlob* blob,
    int aero, int hail, int ml_bl, int melt, int presence, float dt, int radii,
    float* wbuf, const bool* bbuf, const float* sbuf, const int* colflag, float* colsc,
    TempoState* states, TempoWork* work, float* blwork) {
  const int lane = threadIdx.x & 31;
  const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int warps = (int)((gridDim.x * blockDim.x) >> 5);
  TempoState& st = states[warp]; TempoWork& W = work[warp]; float* b = blwork + (long long)warp * nz * 2;
  TempoParams P = {blob};
  const bool has_bl = bl != nullptr;
  const int chunk_cells = imin(ntiles * 32, cells - c0);
  (void)lane;
  for (int i = warp; i < chunk_cells; i += warps) {
    const int c = c0 + i;
    const int tile = i >> 5, lc = i & 31;
    const long long tb0 = TPH_WT(tile) + lc, tbb0 = TPH_BT(tile) + lc, ts0 = TPH_ST(tile) + lc;
    (void)tbb0;
    const int micro = colflag[i];
    if (!post && !micro) continue;
    TEMPO_LEVELS(k, nz) {
      const long long tbs = ts0 + (long long)k * 32;
      if (post) { for (int j = 0; j < 17; ++j) st.f[j][k] = sbuf[TPH_W(j) + tbs]; }
      else { """ + sed_state + r""" }
      if (post && has_bl) { const long long x = (long long)(k - 1) * cells + c; b[k - 1] = bl[x]; b[nz + k - 1] = bl[(long long)nz * cells + x]; }
    }
    if (micro) { TEMPO_LEVELS(k, nz + 1) { """ + zero + r""" } }
    TEMPO_LEVELS(k, nz) W.max_hail_diameter[k] = 0.0f;
    TEMPO_SYNC();
    const int re = last && radii;
    TempoCtx C = {dt, 1.0f / dt, aero, hail, 0, melt, 0, 1, last, re, re, re, last, 0};
    float* qv1d = st.f[4]; float* qc1d = st.f[5]; float* qr1d = st.f[6]; float* qi1d = st.f[7];
    float* qs1d = st.f[8]; float* qg1d = st.f[9]; float* ni1d = st.f[10]; float* nr1d = st.f[11];
    float* nc1d = (presence & 1) ? st.f[12] : nullptr; float* nwfa1d = (presence & 2) ? st.f[13] : nullptr;
    float* nifa1d = (presence & 4) ? st.f[14] : nullptr; float* ng1d = (presence & 8) ? st.f[15] : nullptr;
    float* qb1d = (presence & 16) ? st.f[16] : nullptr;
    float* t1d = st.f[0]; const float* p1d = st.f[1]; const float* w1d = st.f[2]; const float* dz1d = st.f[3];
    (void)p1d; (void)qv1d; (void)nwfa1d; (void)nifa1d; (void)t1d;
    bool a_r = true, a_g = true, a_s = true, a_i = true, a_c = true;
    if (micro && !post) {
        """ + sed_gather + r"""
    }
    if (micro) {
      if (post) {
        TEMPO_LEVELS(k, nz) {
          const long long tb = tb0 + (long long)k * 32, tbb = tbb0 + (long long)k * 32;
          """ + gather_sed + r"""
        }
      }
      if (post) {
        TEMPO_LEVELS(k, nz) {
          const long long tb = tb0 + (long long)k * 32, tbb = tbb0 + (long long)k * 32;
          """ + gather_post + r"""
        }
      }
      if (TEMPO_LEAD) {
        W.did_micro = 1;
        W.rain_precip = 0.0f; W.cloud_precip = 0.0f; W.ice_liquid_equiv_precip = 0.0f;
        W.snow_liquid_equiv_precip = 0.0f; W.graupel_liquid_equiv_precip = 0.0f;
        W.frz_rain_precip = 0.0f; W.frozen_fraction = 0.0f;
      }
      TEMPO_SYNC();
      TempoTend& T = W.T;
      const int nz1 = nz + 1;
      int substeps_sedi, ktop_sedi, n;
      float semi_sedi_factor; float tc0;
      const float t0 = PS(t0);
      const float* ncsave = (nc1d == nullptr) ? W.ncsave : nullptr;
      float* xncx = nullptr; float* xngx = nullptr; float* xqbx = nullptr;
      (void)nz1; (void)n; (void)semi_sedi_factor; (void)xncx; (void)T; (void)tc0; (void)t0; (void)ncsave;
""" + sed + r"""
      TEMPO_SYNC();
      if (!post) {
        TEMPO_LEVELS(k, nz) {
          const long long tb = tb0 + (long long)k * 32, tbb = tbb0 + (long long)k * 32;
          """ + scatter + r"""
        }
        if (TEMPO_LEAD) {
          colsc[5LL * i + 0] = W.rain_precip; colsc[5LL * i + 1] = W.cloud_precip;
          colsc[5LL * i + 2] = W.ice_liquid_equiv_precip; colsc[5LL * i + 3] = W.snow_liquid_equiv_precip;
          colsc[5LL * i + 4] = W.graupel_liquid_equiv_precip;
        }
        TEMPO_SYNC();
        continue;
      }
""" + post + r"""
      TEMPO_SYNC();
    } else {
      TEMPO_LEVELS(k, nz) {
        W.refl10cm[k] = -35.0f; W.re_cloud[k] = 0.0f; W.re_ice[k] = 0.0f; W.re_snow[k] = 0.0f;
      }
      if (TEMPO_LEAD) {
        W.did_micro = 0;
        W.rain_precip = 0.0f; W.cloud_precip = 0.0f; W.ice_liquid_equiv_precip = 0.0f;
        W.snow_liquid_equiv_precip = 0.0f; W.graupel_liquid_equiv_precip = 0.0f;
        W.frz_rain_precip = 0.0f; W.frozen_fraction = 0.0f;
      }
      TEMPO_SYNC();
    }
    if (ml_bl && has_bl && re) {
      tempo_subgrid_cloud_radii(P, C, nz, W, st.f[5], st.f[6], st.f[7], st.f[8],
          (presence & 1) ? st.f[12] : nullptr, st.f[2], b, b + nz, 1, 0);
      TEMPO_SYNC();
    }
    // tempo_step_diags, in its order of operations.
    if (TEMPO_LEAD) {
      float maxhail = W.max_hail_diameter[1];
      for (int k = 2; k <= nz; ++k) maxhail = tmax(maxhail, W.max_hail_diameter[k]);
      const float r = W.rain_precip, i = W.ice_liquid_equiv_precip,
                  sn = W.snow_liquid_equiv_precip, g = W.graupel_liquid_equiv_precip;
      s[2][c] = (s[2][c] + i) + sn; s[3][c] = (s[3][c] + i) + sn;
      s[4][c] += g; s[5][c] += g; s[0][c] = (((s[0][c] + r) + i) + sn) + g; s[1][c] = (((s[1][c] + r) + i) + sn) + g;
      s[6][c] = W.frozen_fraction; s[7][c] += W.frz_rain_precip;
      s[8][c] = W.max_hail_diameter[1]; s[9][c] = maxhail;
    }
    // unpack (theta = T / exner) and the carriers back to the pools.
    TEMPO_LEVELS(k, nz) {
      const long long x = (long long)(k - 1) * cells + c;
      f[0][x] = st.f[0][k] / f[17][x];
      for (int j = 4; j < 17; ++j) if (f[j]) f[j][x] = st.f[j][k];
      if (last) {
        profiles[0][x] = W.refl10cm[k];
        if (radii) { profiles[1][x] = W.re_cloud[k]; profiles[2][x] = W.re_ice[k]; profiles[3][x] = W.re_snow[k]; }
      }
    }
    TEMPO_SYNC();
  }
}
"""


def tail_source(device_math_source: str, max_levels: int, inline: str = "") -> str:
    """Module S: the binary32-blob fast column in the warp shape, the warp kernel and the phase tail."""
    from woof.hex.mpasphys.rrfs import tempo_v303 as t
    col = _inline(fast_column(), "S" in inline.split("MB")[0].replace("TM", ""))
    work, tend = layout(col)
    words = len(work) + len(tend) + 1
    mb = _tail_minblocks(inline)
    lb = f"#define TEMPO_TAIL_LB __launch_bounds__(32, {mb})\n" if mb else ""
    return (f"#define TEMPO_NZ_MAX {max_levels}\n#define TEMPO_WARP 1\n" + lb + _shared_pow(device_math_source)
            + device_math_source + col + t._WARP + layout_macros(words, "TM" in inline) + _tail_kernel(col, work, tend)
            + t._mpas_source("fast"))


# ---------------------------------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------------------------------

def _options():
    from woof.hex.mpasphys.rrfs import tempo_v303 as t
    return _hex_numerics.options(t.COMPILE_OPTIONS)


@lru_cache(maxsize=None)
def _modules(device: int, device_math_source: str, max_levels: int, inline: str = ""):
    import cupy as cp
    from cupy.cuda import compiler, driver, function, runtime
    from woof.hex.cuda_backend.aot import direct_image
    from woof.hex.mpasphys.rrfs import tempo_v303 as t
    with cp.cuda.Device(device):
        arch = compiler._get_arch()
        mods = []
        for src in (phase_source(device_math_source, max_levels, inline), tail_source(device_math_source, max_levels, inline)):
            image = direct_image(src, _options(), arch)
            module = function.Module()
            module.load(image)
            mods.append(module)
        mp, ms = mods
        sizes = cp.empty(3, dtype=cp.uint64)
        ms.get_function("tempo_warp_sizes")((1,), (1,), (sizes,))
        sizes = tuple(map(int, sizes.get()))
        sms = runtime.deviceGetAttribute(runtime.cudaDevAttrMultiProcessorCount, device)
        resident = {}
        for name in ("tempo_phase_tail", "tempo_warp_steps"):
            kern = ms.get_function(name)
            blocks = driver.occupancyMaxActiveBlocksPerMultiprocessor(kern.ptr, t.WARP_BLOCK, 0)
            resident[name] = max(1, sms * blocks) * (t.WARP_BLOCK // 32)
        col = fast_column()
        work, tend = layout(col)
        words = len(work) + len(tend) + 1
        return mp, ms, sizes, resident, words


_LOCK = threading.Lock()
_BLOBS = {}
_AREAS = {}


def float_blob(tables):
    """The table blob in binary32, converted once per device blob (fast numerics' parameter storage; the
    all-float conversion tempo_v303.fast_tables). A binary32 blob (converted at the bind) is returned as it is."""
    import cupy as cp
    from woof.hex.mpasphys.rrfs import tempo_v303 as t
    if tables.dtype == cp.float32:
        return tables
    key = (tables.device.id, tables.data.ptr, tables.size)
    with _LOCK:
        got = _BLOBS.get(key)
        if got is None:
            for old in [k for k in _BLOBS if k[0] == key[0]]:
                del _BLOBS[old]
            got = _BLOBS[key] = t.fast_tables(tables)
        return got


@lru_cache(maxsize=None)
def _work_index(name: str) -> int:
    """Position of a work array in the (level, cell) work area."""
    work, tend = layout(fast_column())
    return _index(work, tend)[name]


def _areas(device: int, nz: int, ntiles: int, words: int):
    import cupy as cp
    key = (device, cp.cuda.get_current_stream().ptr, nz, ntiles, words)
    with _LOCK:
        if key not in _AREAS:
            for old in [k for k in _AREAS if k[:2] == key[:2]]:
                del _AREAS[old]
            per = ntiles * (nz + 3) * 32
            _AREAS[key] = (cp.empty(words * per, dtype=cp.float32), cp.empty(len(_FLAGS) * per, dtype=cp.bool_),
                           cp.empty(17 * per, dtype=cp.float32), cp.empty(ntiles * 32, dtype=cp.int32),
                           cp.empty(ntiles * 32, dtype=cp.int32), cp.empty(5 * ntiles * 32, dtype=cp.float32))
        return _AREAS[key]


def launch_steps(*, device, nz, cells, steps, dt, f, bl, s, profiles, tables, flags, first, presence, radii,
                 device_math_source, force_warp=False):
    """One fast TEMPO slot (all n_microp substeps): phases A, B, C per chunk, then the warp tail.

    Falls back to the binary32-blob warp kernel where the column has serial setup work the phases do not
    carry: the first call (aerosol and graupel first-call initialisation) or absent aerosol carriers
    (the aerosol height profiles are rebuilt every call).
    """
    import cupy as cp
    from woof.hex.mpasphys.rrfs import tempo_v303 as t
    mp, ms, sizes, resident, words = _modules(device, device_math_source, nz, inline_mode())
    blob = float_blob(tables)
    aero, hail, ml_bl, melt = (int(bool(v)) for v in flags)
    tiles = -(-cells // 32)
    if force_warp or first or (presence & 6) != 6:
        blocks = -(-min(cells, t.WARP_WAVES * resident["tempo_warp_steps"]) * 32 // t.WARP_BLOCK)
        warps = blocks * (t.WARP_BLOCK // 32)
        states, work, blwork = t._warp_work(device, nz, warps, sizes)
        ms.get_function("tempo_warp_steps")((blocks,), (t.WARP_BLOCK,), (
            np.int32(nz), np.int32(cells), np.int32(steps), np.float32(dt), f,
            bl if bl is not None else np.uint64(0), s, profiles, blob,
            np.int32(aero), np.int32(hail), np.int32(ml_bl), np.int32(melt), np.int32(bool(first)), np.int32(presence),
            states, work, blwork, np.uint64(0), np.uint64(0), np.int32(bool(radii))))
        return "warp"
    chunk = min(tiles, CHUNK_TILES)
    wbuf, bbuf, sbuf, colflag, kstop, colsc = _areas(device, nz, chunk, words)
    tail_blocks = -(-min(chunk * 32, t.WARP_WAVES * resident["tempo_phase_tail"]) * 32 // t.WARP_BLOCK)
    tail_warps = tail_blocks * (t.WARP_BLOCK // 32)
    states, work, blwork = t._warp_work(device, nz, tail_warps, sizes)
    pa, pb, pc, pd, pe = (mp.get_function(n) for n in ("tempo_phase_a", "tempo_phase_b", "tempo_phase_c",
                                                        "tempo_phase_d", "tempo_phase_e"))
    tail = ms.get_function("tempo_phase_tail")
    # The melting-level reflectivity reads other levels: keep the whole post-sedimentation part in the tail.
    post = int(bool(melt))
    ia_temp, ia_hail = _work_index("temp"), _work_index("max_hail_diameter")
    import os
    prof = os.environ.get("WOOF_HEX_TEMPO_PHASE_PROFILE") == "1"  # measurement only: events between the kernels
    marks = []
    def mark(tag):
        if prof:
            ev = cp.cuda.Event(); ev.record(); marks.append((tag, ev))
    mark("start")
    for sub in range(steps):
        last = int(sub + 1 == steps)
        for t0 in range(0, tiles, chunk):
            nt = min(chunk, tiles - t0)
            c0 = t0 * 32
            grid = (-(-nt // TILES_PER_BLOCK), nz)
            args = (np.int32(nz), np.int32(cells), np.int32(c0), np.int32(nt), f, blob, np.int32(aero),
                    np.int32(hail), np.int32(melt), np.int32(presence), np.float32(dt), wbuf, bbuf, sbuf,
                    colflag, kstop)
            colflag.fill(0)
            kstop.fill(nz + 1)
            for tag, kern in (("A", pa), ("B", pb), ("C", pc)):
                kern(grid, (32 * TILES_PER_BLOCK,), args)
                mark(tag)
            tail((tail_blocks,), (t.WARP_BLOCK,), (
                np.int32(nz), np.int32(cells), np.int32(c0), np.int32(nt), np.int32(last), np.int32(post),
                f, bl if bl is not None else np.uint64(0), s, profiles, blob,
                np.int32(aero), np.int32(hail), np.int32(ml_bl), np.int32(melt), np.int32(presence),
                np.float32(dt), np.int32(bool(radii)), wbuf, bbuf, sbuf, colflag, colsc, states, work, blwork))
            mark("tail")
            if not post:
                pd(grid, (32 * TILES_PER_BLOCK,), args + (np.int32(last), np.int32(bool(radii)), np.int32(ml_bl),
                                                         bl if bl is not None else np.uint64(0), profiles))
                mark("D")
                pe((-(-nt * 32 // 128),), (128,), (np.int32(nz), np.int32(cells), np.int32(c0), np.int32(nt), s, wbuf,
                                                   colsc, colflag, blob, np.int32(ia_temp), np.int32(ia_hail)))
                mark("E")
    if prof:
        marks[-1][1].synchronize()
        acc = {}
        for (_, a), (tag, b) in zip(marks, marks[1:]):
            acc[tag] = acc.get(tag, 0.0) + cp.cuda.get_elapsed_time(a, b)
        print("TEMPO_PHASE_PROFILE " + " ".join(f"{k}={v:.2f}ms" for k, v in acc.items()), flush=True)
    return "phase"
