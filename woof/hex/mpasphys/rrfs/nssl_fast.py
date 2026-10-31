"""NSSL fast numerics (rrfs/nssl-fast): applied by ``mp_nssl2m`` in FAST numerics only.

Strict numerics never sees this module: the strict source is ``mp_nssl2m.cuda_source`` byte for byte, and the strict
launch shape is unchanged. Fast words change by design; the fast referee is the forecast (maps, rain and echo-area
totals), not words. The recipe is TEMPO's (``tempo_allfloat_v303``, rrfs/tempo-yolo):

* operators: the round-to-nearest intrinsics (``__fadd_rn`` ...) that forbid contraction become plain operators, so
  ``--fmad=true`` fuses them;
* maths: the glibc-exact device replicas (``mynn_*``) become CUDA's binary32 intrinsics (``__expf``, ``__logf``,
  ``pow(x, y) = exp2f(y * __log2f(x))`` for positive bases, binary32 ``powf`` otherwise);
* binary32: every routine on the step path computes in binary32. The initializer (``f_nssl_2mom_init`` and the
  binary64 gamma helpers it calls) keeps binary64, and its two binary64 tables (``gmoi``, ``gamxinflu``) are converted
  once to binary32 shadows (``gmoi_f``, ``gamxinflu_f``) that the step path reads;
* launch shape (``launch_shape``): whole warps per block and a workspace batch that fills the card, instead of
  8-column blocks and a 2 GiB arena budget (3,206 columns per batch at 59 levels, 65 batches on a 3 km mesh);
* registers (``register_scalars``): the collection-level routine's 734 one-point work arrays (native extent ngs, fixed
  at 1 on the per-level route) are held as scalars instead of device-memory arena cells. Same words as without it;
* column collection (``collect_by_column``): one collection thread per column runs its levels in order on one copy of
  the column context, instead of one thread per (column, level) each copying it. Same words as without it;
* sedimentation across points and levels (``nssl_fastsed``, stage ``sed``): fall speeds one thread per point,
  the species' fallout substeps one warp per column with a level on each lane;
* condensation per point (``nssl_fastcond``, stage ``cond``): native f_nucond after its first column walk, one
  thread per point;
* register caps (``bounds``) on the column-phase kernels.

``WOOF_HEX_NSSL_FAST=0`` gives the fast compiler flags over the strict source text (the pre-change fast route);
``WOOF_HEX_NSSL_FAST_STAGES`` (default ``ops,maths,float,shape,regs,column,sed,cond,bounds``) selects stages for measurement.
"""
from __future__ import annotations

import os
import re

import numpy as np

STAGES = ("ops", "maths", "float", "shape", "regs", "column", "sed", "cond", "bounds")

_OPS = (
    ("__device__ float nssl_fadd(float a,float b) { return __fadd_rn(a,b); }",
     "__device__ __forceinline__ float nssl_fadd(float a,float b) { return a+b; }"),
    ("__device__ float nssl_fsub(float a,float b) { return __fsub_rn(a,b); }",
     "__device__ __forceinline__ float nssl_fsub(float a,float b) { return a-b; }"),
    ("__device__ float nssl_fmul(float a,float b) { return __fmul_rn(a,b); }",
     "__device__ __forceinline__ float nssl_fmul(float a,float b) { return a*b; }"),
    ("__device__ float nssl_fdiv(float a,float b) { return __fdiv_rn(a,b); }",
     "__device__ __forceinline__ float nssl_fdiv(float a,float b) { return a/b; }"),
    # Mixed-kind templates: binary64 stays binary64 in the initializer, binary32 operands stay binary32.
    ("__device__ double nssl_dadd(double a,double b) { return __dadd_rn(a,b); }",
     "template<class A,class B> __device__ __forceinline__ auto nssl_dadd(A a,B b) { return a+b; }"),
    ("__device__ double nssl_dsub(double a,double b) { return __dsub_rn(a,b); }",
     "template<class A,class B> __device__ __forceinline__ auto nssl_dsub(A a,B b) { return a-b; }"),
    ("__device__ double nssl_dmul(double a,double b) { return __dmul_rn(a,b); }",
     "template<class A,class B> __device__ __forceinline__ auto nssl_dmul(A a,B b) { return a*b; }"),
    ("__device__ double nssl_ddiv(double a,double b) { return __ddiv_rn(a,b); }",
     "template<class A,class B> __device__ __forceinline__ auto nssl_ddiv(A a,B b) { return a/b; }"),
    ("__device__ float nssl_sqrt(float a) { return __fsqrt_rn(a); }",
     "__device__ __forceinline__ float nssl_sqrt(float a) { return sqrtf(a); }"),
    ("__device__ double nssl_sqrt(double a) { return __dsqrt_rn(a); }",
     "__device__ __forceinline__ double nssl_sqrt(double a) { return sqrt(a); }"),
)
# GNU powi NaN payload canonicalisation (strict identity only): a branch per integer power.
_IPOW_NAN = """    if constexpr(N>=2 && sizeof(T)==4) {
        // CUDA float multiply canonicalizes NaNs; strict GNU powi preserves
        // the input sign/payload and quiets signaling NaNs. Retire the measured
        // 576-word CUDA primitive defect without changing any finite operand.
        unsigned int bits=__float_as_uint(a);
        if((bits & 0x7fffffffU)>0x7f800000U)
            return __uint_as_float(bits | 0x00400000U);
    }
"""

_MATHS = (
    ("__device__ float nssl_exp(float a) { return mynn_exp(a); }",
     "__device__ __forceinline__ float nssl_exp(float a) { return __expf(a); }"),
    ("__device__ float nssl_log(float a) { return mynn_log(a); }",
     "__device__ __forceinline__ float nssl_log(float a) { return __logf(a); }"),
    ("__device__ float nssl_pow(float a, float b) { return mynn_pow(a,b); }",
     "__device__ __noinline__ float nssl_pow_slow(float a, float b) { return powf(a,b); }\n"
     "__device__ __forceinline__ float nssl_pow(float a, float b) { return a > 0.0f ? exp2f(b*__log2f(a)) : nssl_pow_slow(a,b); }"),
    ("__device__ double nssl_exp(double a) { return mynn_exp(a); }",
     "__device__ __forceinline__ double nssl_exp(double a) { return exp(a); }"),
    ("__device__ double nssl_log(double a) { return mynn_log(a); }",
     "__device__ __forceinline__ double nssl_log(double a) { return log(a); }"),
    ("__device__ double nssl_pow(double a, double b) { return mynn_pow(a,b); }",
     "__device__ __forceinline__ double nssl_pow(double a, double b) { return pow(a,b); }"),
    ("__device__ float nssl_log10(float a) { return mynn_log10(a); }",
     "__device__ __forceinline__ float nssl_log10(float a) { return __log10f(a); }"),
    ("__device__ float nssl_atan(float a) { return mynn_atan(a); }",
     "__device__ __forceinline__ float nssl_atan(float a) { return atanf(a); }"),
    ("__device__ float nssl_tanh(float a) { return mynn_tanh(a); }",
     "__device__ __forceinline__ float nssl_tanh(float a) { return tanhf(a); }"),
)

#: Routines that only the initializer reaches (binary64 tables), kept binary64.
_INIT_ROUTINES = ("f_nssl_2mom_init", "f_gamma_dpr", "f_gamxinfdp", "f_gamma_dp", "f_gammadp", "f_beta")
_TABLES = (("gmoi", 1), ("gamxinflu", 4))


def stages() -> tuple[str, ...]:
    if os.environ.get("WOOF_HEX_NSSL_FAST", "1") == "0":
        return ()
    chosen = tuple(s.strip() for s in os.environ.get("WOOF_HEX_NSSL_FAST_STAGES", ",".join(STAGES)).split(",") if s.strip())
    unknown = set(chosen) - set(STAGES)
    if unknown:
        raise ValueError(f"WOOF_HEX_NSSL_FAST_STAGES: unknown stage(s) {sorted(unknown)}")
    return chosen


def _replace_once(text: str, old: str, new: str, what: str) -> str:
    found = text.count(old)
    if found != 1:
        raise ValueError(f"NSSL fast {what}: expected one '{old[:70]}', found {found}")
    return text.replace(old, new)


def _block_end(text: str, start: int) -> int:
    opening = text.index("{", start)
    depth, i = 1, opening + 1
    while depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return i


def _routine_spans(text: str, names) -> list[tuple[int, int]]:
    spans = []
    for name in names:
        hits = [m.start() for m in re.finditer(r"__device__ (?:__noinline__ |__forceinline__ )?\w+ " + name + r"\(", text)]
        if len(hits) != 1:
            raise ValueError(f"NSSL fast binary32: expected one definition of {name}, found {len(hits)}")
        spans.append((hits[0], _block_end(text, hits[0])))
    return sorted(spans)


def all_float(body: str, entry: str) -> tuple[str, str]:
    """Binary32 on the step path; the initializer keeps binary64 and its tables get binary32 shadows."""
    start = body.index("struct NsslState {")
    end = body.index("\n};", start) + 3
    state = body[start:end]
    shadows = "".join(f"NsslArray<float, {rank}> {name}_f{{}};\n" for name, rank in _TABLES)
    # Step-path reads go through {name}_r: the binary32 shadow once it exists, else the binary64 table. THE BREAKAGE
    # THIS ENDS: the initializer itself calls the binary32 f_delbk/f_delabk, which read gmoi before the shadow is
    # made (CUDA_ERROR_ILLEGAL_ADDRESS at initialization: mv3km forecast 2026-10-29 05:24Z, RTX 4090 replay 05:40Z).
    shadows += "".join(f"template<class... I> __device__ __forceinline__ float {name}_r(I... i) const "
                       f"{{ return {name}_f.data ? {name}_f(i...) : (float){name}(i...); }}\n" for name, _ in _TABLES)
    for name, rank in _TABLES:
        if state.count(f"NsslArray<double, {rank}> {name}{{}};") != 1:
            raise ValueError(f"NSSL fast binary32: expected one binary64 table {name}")
    # The module's binary64 constants (dgam, dgami, zscale, zscaleinv) hold binary32-exact values; as binary64 they
    # promoted every step-path expression that used them to binary64. Binary32-exact, so the initializer is unchanged.
    for name, value in re.findall(r"static constexpr double (\w+) = (0x[0-9a-f.]+p[-+]\d+);", state):
        if float(np.float32(float.fromhex(value))) != float.fromhex(value):
            raise ValueError(f"NSSL fast binary32: constant {name} = {value} is not binary32-exact")
    state = state.replace("static constexpr double ", "static constexpr float ")
    state = state[:-3] + "\n" + shadows + state[-3:]
    body = body[:start] + state + body[end:]
    for name, _ in _TABLES:
        body = _replace_once(body, f"{name} = other.{name};", f"{name} = other.{name}; {name}_f = other.{name}_f;",
                             f"copy_globals {name}")
    for keep_text in _KEEP_DOUBLE[:2]:
        if body.count(keep_text) != 1:
            raise ValueError(f"NSSL fast binary32: expected one '{keep_text}'")
    if entry.count(_KEEP_DOUBLE[2]) != 1:
        raise ValueError("NSSL fast binary32: expected one binary64 initializer kernel signature")
    keep = _routine_spans(body, _INIT_ROUTINES)
    keep.append((body.index("struct NsslState {"), body.index("\n};", body.index("struct NsslState {")) + 3))
    keep.sort()
    out, last = [], 0
    for a, b in keep:
        out.append(_float_text(body[last:a]))
        out.append(body[a:b])
        last = b
    out.append(_float_text(body[last:]))
    body = "".join(out)
    # The step entry: binary32 everywhere except the initializer's binary64 override vector.
    init_end = _block_end(entry, entry.index('extern "C" __device__ void nssl_initialize_body('))
    entry = entry[:init_end] + _float_text(entry[init_end:])
    entry += FAST_TABLES_KERNEL
    return body, entry


#: Binary64 words the binary32 rewrite keeps: the module-storage allocation of the two initializer tables and the
#: initializer kernel's binary64 namelist override vector.
_KEEP_DOUBLE = ("gmoi = nssl_array<double, 1>", "gamxinflu = nssl_array<double, 4>",
                'extern "C" __global__ void nssl_initialize(unsigned char* parameters,unsigned char* storage,'
                'unsigned long long* used,const double* overrides')


_DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d*(?:[eE][-+]?\d+)?)(?![\w.])")


def _binary32_literal(text: str) -> bool:
    value = abs(float(text))
    return value == 0.0 or 1e-37 <= value <= 3e38


def _float_text(text: str) -> str:
    for i, keep in enumerate(_KEEP_DOUBLE):
        text = text.replace(keep, keep.replace("double", f"NSSL_KEEP_BINARY64_{i}"))
    text = re.sub(r"\bdouble\b", "float", text)
    # Unsuffixed decimal literals (Fortran d0 constants) are binary64 in C++ and promote the expression around them.
    text = _DECIMAL.sub(lambda m: m[1] + "f" if _binary32_literal(m[1]) else m[1], text)
    for i, keep in enumerate(_KEEP_DOUBLE):
        text = text.replace(keep.replace("double", f"NSSL_KEEP_BINARY64_{i}"), keep)
    for name, _ in _TABLES:
        text = re.sub(r"\b" + name + r"\(", name + "_r(", text)
    return text


FAST_TABLES_KERNEL = r'''
// rrfs/nssl-fast: binary32 shadows of the initializer's binary64 tables, converted once after initialization.
// A template, not a generic lambda: NVRTC 12.9 and 13.0 take an unannotated lambda in a kernel for a host
// function and refuse the whole module ("host functions are not allowed in JIT mode").
template <class D, class F>
__device__ __forceinline__ void nssl_fast_shadow(const D& d, F& f, float* storage, unsigned long long& offset) {
    f = {};
    if(!d.present()) return;
    int n = d.size();
    f.data = storage + offset;
    for(int r=0;r<(int)(sizeof(f.lo)/sizeof(int));++r) { f.lo[r]=d.lo[r]; f.hi[r]=d.hi[r]; f.stride[r]=d.stride[r]; }
    for(int i=0;i<n;++i) f.data[i] = (float)d.data[i];
    offset += n;
}
extern "C" __global__ void nssl_fast_tables(unsigned char* parameters, float* storage) {
    NsslState& s = *reinterpret_cast<NsslState*>(parameters);
    unsigned long long offset = 0;
    if(threadIdx.x==0 && blockIdx.x==0) {
        nssl_fast_shadow(s.gmoi, s.gmoi_f, storage, offset);
        nssl_fast_shadow(s.gamxinflu, s.gamxinflu_f, storage, offset);
    }
}
extern "C" __global__ void nssl_fast_table_words(unsigned char* parameters, unsigned long long* words) {
    NsslState& s = *reinterpret_cast<NsslState*>(parameters);
    words[0] = (s.gmoi.present() ? s.gmoi.size() : 0) + (s.gamxinflu.present() ? s.gamxinflu.size() : 0);
}
'''


def fast_source(source: str, chosen=None) -> str:
    """Apply the operator and maths stages to the assembled fast source text."""
    chosen = stages() if chosen is None else chosen
    if "ops" in chosen:
        for old, new in _OPS:
            source = _replace_once(source, old, new, "operator")
        source = _replace_once(source, _IPOW_NAN, "", "powi NaN canonicalisation")
    if "maths" in chosen:
        for old, new in _MATHS:
            source = _replace_once(source, old, new, "maths")
    if "bounds" in chosen:
        for kernel, (threads, blocks) in launch_bounds().items():
            head = f'extern "C" __global__ void nssl_{kernel}('
            if head in source:
                source = _replace_once(source, head, f'extern "C" __global__ void __launch_bounds__({threads}, {blocks}) '
                                       f'nssl_{kernel}(', "launch bounds")
    return source


#: Register caps for the column-phase kernels (``bounds``): (threads per block, resident blocks per SM). At 255
#: registers a 32-thread block allows 8 resident warps per SM, and those phases wait on memory with nothing to switch
#: to. Measurement knob WOOF_HEX_NSSL_BOUNDS="setup:32:16,..." replaces the table; "none" empties it.
LAUNCH_BOUNDS = {"setup": (32, 16), "sedpre": (32, 16), "condense": (32, 16), "condpre": (32, 16),
                 "finish": (32, 16)}


def launch_bounds() -> dict:
    knob = os.environ.get("WOOF_HEX_NSSL_BOUNDS")
    if knob is None:
        return dict(LAUNCH_BOUNDS)
    table = {}
    for item in filter(None, knob.split(",")):
        if item == "none":
            continue
        kernel, threads, blocks = item.split(":")
        table[kernel] = (int(threads), int(blocks))
    return table


#: The collection-level routine's one-point work arrays (native gather/scatter arrays sized ngs, which the per-level
#: collection route fixes at constexpr 1) become register-resident scalars. THE BREAKAGE THIS ENDS: each of them was an
#: arena allocation in device memory, so every read and write of 721 per-point temporaries was a global-memory access
#: (collection = 550 of 730 ms of the mv3km slot on an RTX 4090 host, 2026-10-29 05:40Z replay profile).
SCALAR1 = r"""
// rrfs/nssl-fast regs: a one-element work array (native extent ngs, here 1) held as a scalar.
template<class T> struct NsslScalar1View {
    T& v;
    int n;
    __device__ __forceinline__ int size() const { return n; }
    __device__ __forceinline__ T& linear(int) const { return v; }
};
template<class T> struct NsslScalar1 {
    T v{};
    int lo[1] = {1};
    int hi[1] = {1};
    __device__ __forceinline__ T& operator()(int) { return v; }
    __device__ __forceinline__ NsslScalar1View<T> slice(NsslRange r) { return {v, r.hi - r.lo + 1}; }
};
"""
_LEVEL = "void f_nssl_2mom_level("
_SCALAR_DECL = re.compile(r"\nNsslArray<(\w+), 1> (\w+) = nssl_array<\1, 1>\(arena, \{1\}, \{ngs\}\);")


def _section_end(text: str, at: int) -> int:
    """End of the parenthesised argument list that opens at text[at] == '('."""
    depth, i = 0, at
    while True:
        depth += (text[i] == "(") - (text[i] == ")")
        i += 1
        if depth == 0:
            return i


def register_scalars(body: str) -> str:
    """Hold the collection-level routine's ngs work arrays as scalars, except arrays handed whole to a callee.

    With ngs = 1 every index and section of such an array names its one element (sections 1:ngscnt run over at most
    one point), so element reads, element writes and section loops keep their meaning."""
    if body.count(_LEVEL) != 1:
        raise ValueError("NSSL fast regs: expected one per-level collection routine")
    start = body.index(_LEVEL)
    end = _block_end(body, start)
    level = body[start:end]
    if level.count("constexpr int ngs = 1;") != 1:
        raise ValueError("NSSL fast regs: the collection-level routine must fix ngs at 1")
    held = 0
    for match in list(_SCALAR_DECL.finditer(level)):
        kind, name = match[1], match[2]
        whole = False
        for m in re.finditer(r"\b" + name + r"\b", level):
            rest = level[m.end():m.end() + 40]
            if rest.startswith("(") or rest.startswith(".lo[0]") or rest.startswith(".hi[0]") \
                    or rest.startswith(" = nssl_array<"):
                continue
            if rest.startswith(".slice(NsslRange{"):
                close = _section_end(level, m.end() + len(".slice"))
                tail = level[close:close + 9]
                if tail.startswith(".linear(") or tail.startswith(").size()") or tail.startswith(".size()"):
                    continue
            whole = True
            break
        if whole:
            continue
        level = level.replace(match[0], f"\nNsslScalar1<{kind}> {name};", 1)
        held += 1
    if held < 700:
        raise ValueError(f"NSSL fast regs: expected about 734 one-point work arrays, held {held}")
    return SCALAR1 + body[:start] + level + body[end:]


_COLLECT_POINT = ("int point=point_first+thread; int workspace_column=point%count; int selected_level=point/count+1; "
                  "int active=first+workspace_column; if(thread<point_count) nssl_collect_level_body(selected_level,")
_COLLECT_COLUMN = ("int workspace_column=point_first+thread; int active=first+workspace_column; "
                   "if(thread<point_count) nssl_collect_level_body(0,")


def collect_by_column(entry: str) -> str:
    """One collection thread per column, running the column's levels in order on one copy of its context.

    THE BREAKAGE THIS ENDS: one thread per (column, level) copied the column's whole module context (NsslState) and
    rebuilt the routine's prologue constants for every level: 135 of the 200 ms collection time on the mv3km slot is
    that per-point overhead (RTX 4090 replay probe, collection returning at entry, 2026-10-29 06:04Z). Same words:
    a level reads other levels only from immutable t7 and w, and the module scalars a level writes (ibinhmlr,
    ibinhlmlr, iehlw, ipelec, renucfrac) are set from constants or immutable globals, the same at every level, and
    native_error only rises; the column's written-back context is still level 1's."""
    entry = _replace_once(entry, _COLLECT_POINT, _COLLECT_COLUMN, "collection thread map")
    start = entry.index("void nssl_collect_level_body(")
    end = _block_end(entry, start)
    body = entry[start:end]
    call = body.index("column.f_nssl_collect_level(selected_level,collection_arena,saved_driver,substep,")
    close = body.index(";", call) + 1
    output = body.index("if(selected_level==1) {", close)
    output_end = body.index("}", output) + 1
    if body[close:output].strip():
        raise ValueError("NSSL fast column collection: unexpected text between the level call and its write-back")
    level_call = body[call:close].replace("saved_driver,substep,", "level_driver,substep,", 1)
    loop = ("for (int selected_level=1; selected_level<=nz; ++selected_level) { NsslDriverState level_driver=saved_driver; "
            + level_call + "\n" + body[output:output_end] + " }")
    body = body[:call] + loop + body[output_end:]
    return entry[:start] + body + entry[end:]


def collect_columns() -> bool:
    return "column" in stages()


def split_sedimentation() -> bool:
    return "sed" in stages()


def point_condensation() -> bool:
    return "sed" in stages() and "cond" in stages()


def launch_shape(count: int):
    """(columns per block) for the column phases: whole warps on a mesh large enough to fill the card."""
    if "shape" not in stages() or count < 16384:
        return 8
    return 32


def workspace_budget(free_bytes: int) -> int:
    """The arena budget: half the free memory, up to WOOF_HEX_NSSL_WORKSPACE_GIB (16) GiB (strict: min(2 GiB, free/4))."""
    if "shape" not in stages():
        return min(2 * 1024**3, max(0, int(free_bytes)) // 4)
    cap = float(os.environ.get("WOOF_HEX_NSSL_WORKSPACE_GIB", "16"))
    return min(int(cap * 1024**3), max(0, int(free_bytes)) // 2)


def collection_budget(free_bytes: int) -> int:
    """The collection-level workspace budget: a quarter of free memory, up to 8 GiB (strict: up to 2 GiB)."""
    return min(8 * 1024**3, max(0, int(free_bytes)) // 4)


def profiling() -> bool:
    return os.environ.get("WOOF_HEX_NSSL_PHASE_PROFILE", "0") == "1"


class PhaseProfile:
    """WOOF_HEX_NSSL_PHASE_PROFILE=1 (measurement runs only): CUDA events between phase launches, summed per phase."""

    def __init__(self, cp):
        self.cp, self.events, self.names = cp, [], []

    def mark(self, name):
        e = self.cp.cuda.Event(); e.record(); self.events.append(e); self.names.append(name)

    def report(self, **extra):
        import json, sys
        end = self.cp.cuda.Event(); end.record(); end.synchronize()
        totals = {}
        for i, name in enumerate(self.names):
            nxt = self.events[i + 1] if i + 1 < len(self.events) else end
            totals[name] = totals.get(name, 0.0) + self.cp.cuda.get_elapsed_time(self.events[i], nxt)
        print("NSSL_PHASE_PROFILE " + json.dumps({**extra, **{k: round(v, 3) for k, v in totals.items()}}),
              file=sys.stderr, flush=True)


#: Column-phase arena bound (bytes per column, 8 bytes per element), intercept + slope * nz: the workspace_bound.py
#: audit's largest root-to-leaf stack of f_nssl_2mom_driver EXCLUDING f_nssl_2mom_gs, which the phased route runs
#: only in the collection kernel, on its own collection arena. At 59 levels: 188,552 bytes against the
#: 669,696-byte all-routines stride. Recomputed by tests/test_rrfs_nssl_fast.py.
COLUMN_BOUND_INTERCEPT = 7776
COLUMN_BOUND_SLOPE = 3064


def column_stride(nz: int) -> int:
    """Fast column arena stride: the narrow (four-byte) bank, 3/4 of the stride, holds the whole column-phase bound.

    The binary32 stage leaves the step path no binary64 local, so the eight-byte bank stays empty; without it the
    strict stride is kept. The bound counts eight bytes per element, twice what binary32 cells use.
    THE BREAKAGE THIS ENDS: the all-routines stride sized every column for the collection routine's 514 KB stack,
    which never runs on this arena, so a 3 km mesh ran 65 workspace batches of 3,206 columns each.
    """
    if "shape" not in stages() or "float" not in stages():
        return 65536 + 10240 * nz
    narrow = COLUMN_BOUND_INTERCEPT + COLUMN_BOUND_SLOPE * nz
    return -(-(narrow * 4 // 3 + 1) // 256) * 256
