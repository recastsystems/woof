"""The gates the Grell-Freitas column workspace owns.

The breakage each one prevents, named:

1. ``test_the_gf_frame_stays_under_the_default_stack`` -- the whole point.
   GF's column arrays used to live in the per-thread local frame, and CUDA
   sizes ONE per-context local-memory backing store to the widest frame in
   the context times the card's RESIDENT-THREAD CAPACITY, not times the
   occupancy the kernel achieves.  MEASURED on test host 1 (RTX 5070 Ti, 70 SMs
   x 1,536, sm_120): a 22,416 B frame took 2,200.0 MiB at first launch and
   never gave it back.  Anything that puts a column array back on the stack
   -- a new ``float x[GF_KP]``, a capture sink that stops being null, a
   compiler that stops promoting -- brings that reservation back, and this
   is the only assertion that catches it on a platform with no recorded
   row (the RTX 3080 among them).

2. ``test_the_workspace_slot_map_fits_the_declared_regions`` -- gf.cu's
   slot ids are literal because NVRTC has no ``__COUNTER__``.  A duplicate
   id makes two column arrays alias, which is a wrong forecast and not a
   crash; an id past its region's cap runs into the next region's slots,
   same result.  This reads the ids straight out of the source.

3. ``test_the_python_side_mirrors_the_kernels_workspace_geometry`` -- the
   launcher allocates the workspace and the kernel indexes it.  If the two
   disagree about slot count, lane width or per-block stride, the kernel
   writes past the allocation.

4. ``test_the_workspace_is_free_of_residue`` (GPU) -- the workspace is
   allocated with ``cp.empty`` and reused across tiles, exactly as the
   local frame was reused across launches.  That is only sound if no
   column array is ever read before it is written.  This fills the
   workspace with two different garbage patterns and asserts the outputs
   are bit-identical.

5. ``test_the_no_gpu_oracle_harness_still_compiles`` -- the workspace
   argument is a KERNEL SIGNATURE, and one caller of those signatures is
   not Python.  ``tools/gf_wrf461_oracle/gf_host_harness.cpp`` ``#include``s
   gf.cu and compiles it as plain C++ so the exact device source can be
   graded against the committed WRF v4.6.1 fixtures with no GPU anywhere.
   Nothing in the suite used to compile it, and this cut is what that cost:
   inserting ``float *ws`` after ``isca`` in all three entry points left the
   harness calling the old arities, the no-GPU grader stopped building --
   6 errors -- and every suite stayed green through it.  This node builds
   the harness for real, so the next signature change cannot disarm the
   grader in silence.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from woof.core.gf import (                                  # noqa: E402
    GF_BLOCK, GFWS_SLOT_COUNT_COL, GFWS_SLOT_COUNT_DRV, GFWS_SLOTS,
    gf_workspace_floats,
)

_CU = os.path.join(_ROOT, "woof", "core", "kernels", "gf.cu")

#: The CUDA default per-thread stack.  A frame at or under this reserves
#: nothing, because the driver's backing store is
#: ``(frame - default stack) x SMs x threads/SM`` floored at zero.
DEFAULT_STACK_BYTES = 1024


def _source() -> str:
    with open(_CU, encoding="utf-8") as fh:
        return fh.read()


def _define(name: str) -> int:
    m = re.search(rf"^#define {name} (\d+)$", _source(), re.M)
    assert m, f"gf.cu has no #define {name}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# source-only gates
# ---------------------------------------------------------------------------
def test_the_workspace_slot_map_fits_the_declared_regions():
    text = _source().split("\n")
    owners = {
        "gfd_deep_column": ("gfws", GFWS_SLOT_COUNT_COL),
        "gfd_shallow_column": ("gfws", GFWS_SLOT_COUNT_COL),
        "gf_deep_stage": ("gfws_own", GFWS_SLOT_COUNT_DRV),
        "gf_shallow_stage": ("gfws_own", GFWS_SLOT_COUNT_DRV),
        "gf_gfdrv_stage": ("gfws_own", GFWS_SLOT_COUNT_DRV),
    }
    seen = {name: [] for name in owners}
    current = None
    depth = 0
    for line in text:
        if line.startswith("__device__") or line.startswith('extern "C"'):
            m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
            current = m.group(1) if m and m.group(1) in owners else None
            depth = 0
        if current:
            for hit in re.finditer(
                    r"GFWS_AT(?:_I)?\((gfws|gfws_own), (\d+)\)", line):
                base, idx = hit.group(1), int(hit.group(2))
                assert base == owners[current][0], (
                    f"{current} indexes {base}, not {owners[current][0]}")
                seen[current].append(idx)
            depth += line.count("{") - line.count("}")
            if depth < 0:
                current = None
    for name, (_base, cap) in owners.items():
        ids = seen[name]
        assert ids, f"{name} declares no workspace column arrays"
        assert len(ids) == len(set(ids)), (
            f"{name} reuses a workspace slot id: two column arrays would "
            f"alias.  ids={sorted(ids)}")
        assert max(ids) < cap, (
            f"{name} uses slot {max(ids)} of a {cap}-slot region: it would "
            "run into the next region's arrays")


def test_the_python_side_mirrors_the_kernels_workspace_geometry():
    assert _define("GFWS_SLOT_COUNT_COL") == GFWS_SLOT_COUNT_COL
    assert _define("GFWS_SLOT_COUNT_DRV") == GFWS_SLOT_COUNT_DRV
    assert _define("GFWS_LANES") == GF_BLOCK, (
        "gf.cu indexes the workspace by lane within a block of GFWS_LANES "
        "threads; a launch at any other block width aliases lanes")
    # One whole block of columns at the shipped tier, both sides.
    kp = _define("GF_KMAX") + 9
    assert gf_workspace_floats(_define("GF_KMAX"), GF_BLOCK) == (
        GFWS_SLOTS * kp * GF_BLOCK)
    # A partial block still allocates a whole block's region.
    assert (gf_workspace_floats(_define("GF_KMAX"), 1)
            == gf_workspace_floats(_define("GF_KMAX"), GF_BLOCK))


def test_the_capture_sinks_are_never_read_back():
    """The driver path passes a NULL capture sink, and a null write is a
    no-op.  That is only sound while the slabs are write-only."""
    text = _source()
    reads = [line for line in text.split("\n")
             if re.search(r"(?<![A-Za-z_])(LEVB|SCAB|ISCB)\s*\[", line)
             and not re.search(
                 r"(?<![A-Za-z_])(LEVB|SCAB|ISCB)\s*\[[^\]]*\]\s*=", line)]
    assert reads == [], (
        "a capture sink is read back; with GfSink{nullptr} on the driver "
        f"path that read has no defined value: {reads[:3]}")


# ---------------------------------------------------------------------------
# host-compile gate
# ---------------------------------------------------------------------------
_HARNESS = os.path.join(
    _ROOT, "tools", "gf_wrf461_oracle", "gf_host_harness.cpp")
_KERNEL_DIR = os.path.dirname(_CU)

#: gf_host_parity.py's own recipe, so this builds what the grader builds.
_HARNESS_FLAGS = ("-O2", "-std=c++17", "-ffp-contract=off",
                  "-fno-unsafe-math-optimizations", "-shared", "-fPIC")

#: A translation unit that compiles anywhere a C++17 toolchain works.  It is
#: the discriminator this gate turns on: if THIS fails there is no toolchain
#: and skipping is accurate, and if it passes then a harness failure is the
#: harness's and must be reported as a failure.
_TRIVIAL = ("#include <vector>\n"
            "int main() { return (int)std::vector<float>(4).size(); }\n")


def _run(argv):
    """(returncode, combined output).  Compiler diagnostics are UTF-8 even
    where the Windows locale is not, so they are decoded explicitly."""
    done = subprocess.run(argv, capture_output=True, check=False)
    return done.returncode, (done.stdout + done.stderr).decode(
        "utf-8", "replace")


def _native_builders():
    """``(build, label)`` for every C++ compiler on PATH.

    All of them, not the first: a candidate that is present but cannot
    compile must not consume the box's only chance at a toolchain.
    """
    found = []
    for name in ("g++", "c++", "clang++"):
        path = shutil.which(name)
        if path is None:
            continue

        def build(source, out, include, _cxx=path):
            return _run([_cxx, *_HARNESS_FLAGS, "-I", str(include),
                         str(source), "-o", str(out), "-lm"])

        found.append((build, path))
    return found


def _wsl_builder():
    """The same, through WSL's default distro.

    The Windows box carries no native POSIX C++ toolchain; the oracle's own
    (gcc 13.3 / glibc 2.39) lives in WSL, which is where gf_host_parity.py is
    run from, so that is where this gate has a compiler to find.
    """
    if os.name != "nt":
        return None
    wsl = shutil.which("wsl.exe")
    if wsl is None:
        return None
    if _run([wsl, "--exec", "g++", "-dumpversion"])[0] != 0:
        return None

    def unix(path):
        code, text = _run([wsl, "--exec", "wslpath", "-a", str(path)])
        assert code == 0, f"wslpath failed on {path}: {text}"
        return text.strip()

    def build(source, out, include):
        return _run([wsl, "--exec", "g++", *_HARNESS_FLAGS,
                     "-I", unix(include), unix(source),
                     "-o", unix(out), "-lm"])

    return build, f"{wsl} --exec g++"


@pytest.fixture(scope="module")
def cxx(tmp_path_factory):
    """A toolchain PROVEN able to compile, or a skip that cannot hide a
    compile error.

    A bare ``shutil.which`` would let a half-installed compiler turn a real
    break in gf.cu into a skip.  Every candidate builds ``_TRIVIAL`` with the
    harness's own flags first; only a candidate that passes that is returned,
    so from here on a nonzero return code is the source's fault.
    """
    probe_dir = tmp_path_factory.mktemp("cxx-probe")
    source = probe_dir / "trivial.cpp"
    source.write_text(_TRIVIAL, encoding="ascii", newline="\n")
    tried = []
    candidates = list(_native_builders())
    wsl = _wsl_builder()
    if wsl is not None:
        candidates.append(wsl)
    for build, label in candidates:
        code, text = build(source, probe_dir / "trivial.out", probe_dir)
        if code == 0:
            return build
        tried.append(f"{label}: rc {code}\n{text}")
    pytest.skip("no working C++17 toolchain for the GF oracle harness"
                + ("; tried " + " | ".join(tried) if tried else ""))


def test_the_no_gpu_oracle_harness_still_compiles(cxx, tmp_path):
    """gf.cu's kernel signatures and the oracle harness's calls cannot drift.

    Named breakage: a change to ``gf_deep_stage`` / ``gf_shallow_stage`` /
    ``gf_gfdrv_stage``'s parameters -- exactly what the workspace cut did by
    inserting ``float *ws`` after ``isca`` -- stops
    ``tools/gf_wrf461_oracle/gf_host_harness.cpp`` compiling.  That harness
    is the ENTIRE no-GPU grader: gf_host_parity.py loads the ``.so`` it
    produces and grades gf.cu against the byte-frozen WRF v4.6.1 fixtures
    (83 level fields, 69 float scalars, 39 integer fields bitwise over 216
    columns, plus the four libm sweeps).  A harness that will not build
    grades nothing, and nothing else in the suite compiles it, so the grader
    can be disarmed by an edit that leaves every other test green.
    """
    out = tmp_path / "gf_host_harness.so"
    code, text = cxx(_HARNESS, out, _KERNEL_DIR)
    assert code == 0, (
        "the no-GPU GF oracle harness no longer compiles against gf.cu, so "
        "tools/gf_wrf461_oracle/gf_host_parity.py can grade nothing:\n"
        + text)
    assert out.exists() and out.stat().st_size > 0, (
        f"the harness build reported success but produced no artifact at "
        f"{out}")


#: A probe translation unit: the same CUDA language shims and the same
#: include order ``gf_host_harness.cpp`` uses, cut down to one device
#: routine.  ``GfColC`` converts from ``GfCol`` only, and ``GfCol`` strides
#: by ``GFWS_LANES``, so the backing vectors are that much wider.
_AA0_PROBE = r"""
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#define __device__
#define __global__
#define __constant__ static const
#define __forceinline__ inline
#define __restrict__
struct GfS3 { unsigned x, y, z; };
static GfS3 blockIdx = {0, 0, 0};
static GfS3 blockDim = {1, 1, 1};
static GfS3 threadIdx = {0, 0, 0};
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fsqrt_rn(float a) { return sqrtf(a); }
static inline double __dadd_rn(double a, double b) { return a + b; }
static inline double __dsub_rn(double a, double b) { return a - b; }
static inline double __dmul_rn(double a, double b) { return a * b; }
static inline double __ddiv_rn(double a, double b) { return a / b; }
static inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
static inline float __uint_as_float(unsigned int u)
{ float f; std::memcpy(&f, &u, 4); return f; }
static inline unsigned int __float_as_uint(float f)
{ unsigned int u; std::memcpy(&u, &f, 4); return u; }
static inline float __int_as_float(int i)
{ float f; std::memcpy(&f, &i, 4); return f; }
static inline long long __double_as_longlong(double d)
{ long long l; std::memcpy(&l, &d, 8); return l; }
static inline double __longlong_as_double(long long l)
{ double d; std::memcpy(&d, &l, 8); return d; }
static inline float __double2float_rn(double d) { return (float)d; }
#include "common.cuh"
#include "glibc_flt32.cuh"
#include "gf.cu"

extern "C" float gf_probe_aa0(int kbcon, int ktop, int ktf, float dby_below)
{
    const int kp = GF_KMAX + 9;
    const int n = kp * GFWS_LANES;
    std::vector<float> zb(n, 0.0f), zub(n, 0.0f), dbyb(n, 0.0f),
                       gamb(n, 0.0f), tcb(n, 0.0f);
    GfCol z{zb.data()}, zu{zub.data()}, dby{dbyb.data()}, gam{gamb.data()},
          tc{tcb.data()};
    for (int k = 1; k < kp; k++) {
        z[k]   = 500.0f * (float)k;
        zu[k]  = 1.0f;
        gam[k] = 0.5f;
        tc[k]  = 280.0f;
        dby[k] = 0.0f;
    }
    dby[kbcon - 1] = dby_below;
    return gfd_cup_up_aa0(z, zu, dby, gam, tc, kbcon, ktop, 0, ktf);
}
"""


def test_cup_up_aa0_keeps_the_k_equals_kbcon_layer(cxx, tmp_path):
    """``module_cu_gf_deep.F:3024`` is ``IF(K.LT.KBCON(I))GO TO 100``.

    ``k == kbcon`` is INSIDE WRF's accumulation, and its term reads
    ``dby(kbcon-1)``.  gf.cu carried ``k <= kbcon``, which dropped that
    layer; ``woof/verify/gf_deep_ref.py:971`` -- this repository's own CPU
    transcription of the same subroutine -- has always read ``k < kbcon``,
    so the two implementations of one scheme disagreed.

    Named breakage, and it is discrete rather than perturbative: a column
    whose only positive buoyancy sits at ``k == kbcon`` returns ``aa1 = 0``,
    and ``module_cu_gf_deep.F:1129-1135`` turns ``aa1 == 0`` into
    ``ierr = 17``, switching convection off in that column entirely.  The
    committed 216-column fixture cannot see this -- ``dby(kbcon-1)`` is
    negative on all 60 of its live columns, so ``max(0., da)`` is zero
    there and ``gf_host_parity.py`` grades green either way.  Hence a
    constructed column, and hence this test rather than the oracle.

    The expected word is not a constant typed in here: it comes from the
    CPU reference on the same inputs, so the gate is the seam between the
    two transcriptions and cannot drift away from WRF on its own.
    """
    from woof.verify.gf_deep_ref import cup_up_aa0

    source = tmp_path / "gf_aa0_probe.cpp"
    source.write_text(_AA0_PROBE, encoding="ascii", newline="\n")
    out = tmp_path / "gf_aa0_probe.so"
    code, text = cxx(source, out, _KERNEL_DIR)
    assert code == 0, "the gfd_cup_up_aa0 probe will not build:\n" + text

    lib = ctypes.CDLL(str(out))
    lib.gf_probe_aa0.restype = ctypes.c_float
    lib.gf_probe_aa0.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_float)

    kbcon, ktop, ktf = 5, 8, 12
    kp = _define("GF_KMAX") + 9
    z = [np.float32(0.0)] * kp
    zu = [np.float32(0.0)] * kp
    dby = [np.float32(0.0)] * kp
    gam = [np.float32(0.0)] * kp
    tc = [np.float32(0.0)] * kp
    for k in range(1, kp):
        z[k] = np.float32(500.0 * k)
        zu[k] = np.float32(1.0)
        gam[k] = np.float32(0.5)
        tc[k] = np.float32(280.0)
    dby[kbcon - 1] = np.float32(3.0)

    want = np.float32(cup_up_aa0(z=z, zu=zu, dby=dby, gamma_cup=gam,
                                 t_cup=tc, kbcon=kbcon, ktop=ktop, ierr=0,
                                 ktf=ktf))
    got = np.float32(lib.gf_probe_aa0(kbcon, ktop, ktf, np.float32(3.0)))

    assert want != np.float32(0.0), (
        "the constructed column no longer exercises the k == kbcon layer, "
        "so this gate would pass on a kernel that still drops it")
    assert got.tobytes() == want.tobytes(), (
        f"gf.cu's cup_up_aa0 returned {got!r} where WRF's loop bound gives "
        f"{want!r}: the k == kbcon layer is being dropped")


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_the_gf_frame_stays_under_the_default_stack():
    cp = pytest.importorskip("cupy")
    from woof.core.kernels import load_module

    module = load_module("gf")
    for name in ("gf_gfdrv_stage", "gf_deep_stage", "gf_shallow_stage"):
        frame = module.get_function(name).local_size_bytes
        assert frame <= DEFAULT_STACK_BYTES, (
            f"{name} compiles a {frame} B per-thread local frame on this "
            "platform.  Over the 1,024 B default stack the driver reserves "
            f"(frame - 1024) x SMs x threads/SM of device memory at first "
            "launch and never returns it -- 2,200.0 MiB on a 70-SM RTX "
            "5070 Ti at the 22,416 B frame this cut removed.  A column "
            "array is back on the stack.")
    del cp


@pytest.mark.gpu
def test_the_workspace_is_free_of_residue():
    cp = pytest.importorskip("cupy")
    sys.path.insert(0, os.path.join(_ROOT, "tools", "gf_wrf461_oracle"))
    from gf_field_lists import (
        DRV_IN_LEV, DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS,
        drv_scalar_inputs,
    )
    from woof.core.kernels import load_module
    from woof.verify.gf_oracle import GF_NZ, load_gf_oracle

    fx = load_gf_oracle()
    n, nz = fx.ncol, GF_NZ
    lv = np.zeros((n, len(DRV_IN_LEV), nz), dtype=np.float32)
    # fzu COMPUTED (the shipped forecast path); residue is a property of the
    # workspace, not of gamma, so the unpinned arm is the one to probe.
    sc = drv_scalar_inputs(fx, False)
    ii = np.zeros((n, 3), dtype=np.int32)
    for j, name in enumerate(DRV_IN_LEV):
        lv[:, j, :] = fx.levels[name]
    ii[:, 0] = fx.surface["kpbl"].astype(np.int32)
    ii[:, 1] = fx.surface["ishallow"].astype(np.int32)
    ii[:, 2] = fx.surface["ichoice"].astype(np.int32)
    d_lv, d_sc, d_ii = cp.asarray(lv), cp.asarray(sc), cp.asarray(ii)

    fn = load_module("gf").get_function("gf_gfdrv_stage")
    words = gf_workspace_floats(nz, n)
    out = []
    for fill in (np.float32(0.0), np.float32(-7.0e30)):
        ws = cp.full(words, fill, dtype=cp.float32)
        lev = cp.zeros((n, len(DRV_LEV_FIELDS), nz), dtype=cp.float32)
        sca = cp.zeros((n, len(DRV_SCA_FIELDS)), dtype=cp.float32)
        isc = cp.zeros((n, len(DRV_ISCA_FIELDS)), dtype=cp.int32)
        fn(((n + GF_BLOCK - 1) // GF_BLOCK,), (GF_BLOCK,),
           (d_lv, d_sc, d_ii, lev, sca, isc, ws,
            np.int32(0), np.int32(n), np.int32(nz)))
        cp.cuda.Stream.null.synchronize()
        out.append((cp.asnumpy(lev), cp.asnumpy(sca), cp.asnumpy(isc)))
    for a, b, what in zip(out[0], out[1], ("lev", "sca", "isc")):
        av = np.ascontiguousarray(a)
        bv = np.ascontiguousarray(b)
        if av.dtype == np.float32:
            av, bv = av.view(np.uint32), bv.view(np.uint32)
        bad = int((av.ravel() != bv.ravel()).sum())
        assert bad == 0, (
            f"{what}: {bad} words move when the workspace starts from "
            "different residue, so some column array is read before it is "
            "written and the result depends on the previous tile")
