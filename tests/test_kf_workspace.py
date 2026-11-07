"""The gates the Kain-Fritsch column workspace owns.

The breakage each one prevents, named:

1. ``test_the_kf_frame_stays_under_the_default_stack`` -- the whole point.
   ``kf_column``'s column arrays used to live in the per-thread local frame,
   and CUDA sizes ONE per-context local-memory backing store to the widest
   frame in the context times the card's RESIDENT-THREAD CAPACITY, not times
   the occupancy the kernel achieves.  MEASURED on test host 1 (RTX 5070 Ti, 70
   SMs x 1,536, sm_120, NVRTC 13.0.48): the 9,216 B frame at nz = 49 took
   840.0 MiB at first launch and never gave it back.  Anything that puts a
   column array back on the stack brings that reservation back, and this
   is the only assertion that catches it on a platform with no recorded
   row.

2. ``test_the_workspace_slot_map_fits_the_declared_region`` -- kf.cu's slot
   ids are literal because NVRTC has no ``__COUNTER__``.  A duplicate id
   makes two column arrays alias, which is a wrong forecast and not a crash;
   an id past the cap runs off the end of the block's region into the next
   block's, same result.  This reads the ids straight out of the source.

3. ``test_the_python_side_mirrors_the_kernels_workspace_geometry`` -- the
   launcher allocates the workspace and the kernel indexes it.  If the two
   disagree about slot count, lane width or per-slot extent, the kernel
   writes past the allocation.  Each warp needs its own region even when
   several warps share a block, or columns alias silently.

4. ``test_no_column_array_lives_on_the_stack`` -- since the 2026-10-07 WRF
   port every column array, all 66 of them, is a workspace slot.  The old
   kernel kept two on the stack because moving them broke an FMA fusion the
   output depended on; the port compiles with ``--fmad=false`` (it is graded
   against gfortran, which contracts nothing), so that reason is gone and
   so are they.

5. ``test_the_workspace_is_free_of_residue`` (GPU) -- the workspace is
   allocated with ``cp.empty`` and reused across tiles, exactly as the local
   frame was reused across launches.  That is only sound if no column array
   is ever read before it is written.  This fills the workspace with two
   different garbage patterns and asserts the outputs are bit-identical.
   (WRF's own SNaN-initialised build shows the same of the Fortran:
   tools/kf_wrf461_oracle.)

6. ``test_the_tiled_launch_matches_the_single_launch`` (GPU) -- the columns
   go in tiles, offset by ``col0``.  A tile boundary that dropped or
   double-counted a column, or a workspace not re-initialised between tiles,
   shows up here and nowhere else.

The batches are the WRF oracle's own nz = 50 fixture columns (40 deep, 23
shallow, 33 without convection), interleaved so neighbouring lanes take
different branches.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from woof.core.kf import (                                  # noqa: E402
    KFWS_SLOTS, KF_TILE_BLOCKS_PER_SM, _TPB, _WS_LANES, kf_workspace_floats,
)

_CU = os.path.join(_ROOT, "woof", "core", "kernels", "kf.cu")
_FIXTURE = Path(_ROOT) / "tests" / "data" / "oracles" / "kf"

#: The CUDA default per-thread stack.  A frame at or under this reserves
#: nothing, because the driver's backing store is
#: ``(frame - default stack) x SMs x threads/SM`` floored at zero.
DEFAULT_STACK_BYTES = 1024

_INPUTS = ("u", "v", "temperature", "qv", "pressure", "rho", "exner", "dz",
           "w0avg")
_COLUMN_FIELDS = {"temperature": "t", "pressure": "p", "exner": "pi"}


def _source() -> str:
    with open(_CU, encoding="utf-8") as fh:
        return fh.read()


def _define(name: str) -> int:
    m = re.search(rf"^#define {name} (\d+)$", _source(), re.M)
    assert m, f"kf.cu has no #define {name}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# source-only gates
# ---------------------------------------------------------------------------
def test_the_workspace_slot_map_fits_the_declared_region():
    ids = [int(m.group(1)) for m in re.finditer(
        r"KFWS_AT(?:_I)?\(kfws, (\d+), nz\)", _source())]
    assert ids, "kf_column declares no workspace column arrays"
    assert len(ids) == len(set(ids)), (
        "kf_column reuses a workspace slot id: two column arrays would "
        f"alias.  ids={sorted(ids)}")
    assert max(ids) < KFWS_SLOTS, (
        f"kf_column uses slot {max(ids)} of a {KFWS_SLOTS}-slot region: it "
        "would run into the next block's arrays")
    assert len(ids) == KFWS_SLOTS, (
        f"{len(ids)} slots are declared but KFWS_SLOTS is {KFWS_SLOTS}; a "
        "gap wastes the workspace and a shortfall over-runs it")


def test_the_python_side_mirrors_the_kernels_workspace_geometry():
    assert _define("KFWS_SLOTS") == KFWS_SLOTS
    assert _define("KFWS_LANES") == _WS_LANES == 32
    assert _TPB % _WS_LANES == 0 and _TPB <= 256
    assert "blockDim.x / KFWS_LANES" in _source()
    assert "threadIdx.x / KFWS_LANES" in _source()
    assert "threadIdx.x % KFWS_LANES" in _source()
    # One whole block of columns, both sides.
    assert kf_workspace_floats(49, _TPB) == KFWS_SLOTS * 49 * _TPB
    # A partial block still allocates a whole block's region.
    assert kf_workspace_floats(49, 1) == kf_workspace_floats(49, _TPB)
    # The extent is the RUNTIME nz: KF_KMAX must not appear in the size.
    assert (kf_workspace_floats(98, _TPB)
            == 2 * kf_workspace_floats(49, _TPB))
    assert KF_TILE_BLOCKS_PER_SM >= 1


def test_no_column_array_lives_on_the_stack():
    src = _source()
    body = src[src.index("void kf_column("):]
    stack_arrays = re.findall(r"^\s+(?:float|int) ([a-z_0-9]+)\[", body,
                              re.M)
    assert stack_arrays == [], (
        f"kf_column declares stack arrays {stack_arrays}: the frame goes "
        "back under the driver's resident-thread pricing")


def test_kf_kmax_is_only_a_refusal_ceiling():
    """``KF_KMAX`` sizes nothing; it is the nz refusal and nothing else."""
    from woof.core.kf import _KMAX, VERTICAL_LEVEL_BOUNDS

    src = _source()
    assert VERTICAL_LEVEL_BOUNDS == (8, _KMAX)
    assert _define("KF_KMAX") == _KMAX
    uses = [ln.strip() for ln in src.split("\n")
            if re.search(r"(?<![A-Z_])KF_KMAX(?![A-Z_])", ln)
            and not ln.lstrip().startswith(("*", "//", "/*", "#ifndef",
                                            "#define", "#endif"))]
    assert uses == ["if (nz < 8 || nz > KF_KMAX) return;"], uses


# ---------------------------------------------------------------------------
# device gates
# ---------------------------------------------------------------------------
def _frame_bytes(fn) -> int:
    """CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES of a module Function or a
    RawKernel wrapper (``.kernel``)."""
    from cupy_backends.cuda.api import driver

    handle = getattr(fn, "kernel", fn)
    return int(driver.funcGetAttribute(3, handle.ptr))


@pytest.mark.gpu
def test_the_kf_frame_stays_under_the_default_stack():
    pytest.importorskip("cupy")
    from woof.core import kf

    frame = _frame_bytes(kf._kf_function("kf_column"))
    assert frame <= DEFAULT_STACK_BYTES, (
        f"kf_column compiles a {frame} B per-thread local frame on this "
        "platform.  Over the 1,024 B default stack the driver reserves "
        "(frame - 1024) x SMs x threads/SM of device memory at first launch "
        "and never returns it -- 840.0 MiB on a 70-SM RTX 5070 Ti at the "
        "9,216 B frame the workspace removed.  A column array is back on "
        "the stack.")


def _kf_batch(copies: int):
    """The WRF oracle's nz = 50 fixture columns, repeated ``copies`` times
    over, as (nz, 1, copies) float32 arrays keyed by launch_kf's names."""
    sys.path.insert(0, _ROOT)
    from tools.kf_wrf461_oracle import kf_columns

    cols = kf_columns.read_columns(_FIXTURE / "columns-structured.bin")
    meta = json.loads((_FIXTURE / "columns-structured.json").read_text(
        encoding="utf-8"))
    base = [c for c, m in zip(cols, meta) if m["nz"] == 50]
    pick = [base[i % len(base)] for i in range(copies)]
    return {name: np.ascontiguousarray(np.stack(
        [c[_COLUMN_FIELDS.get(name, name)] for c in pick], axis=1)[:, None],
        dtype=np.float32) for name in _INPUTS}


def _fresh(nz, ny, nx):
    import cupy as cp

    out = {n: cp.zeros((nz, ny, nx), dtype=cp.float32) for n in (
        "rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rqrcuten",
        "rqscuten", "updraft_mass_flux", "downdraft_mass_flux",
        "cldfra_dp", "cldfra_sh", "qc_kf", "qi_kf")}
    out.update({n: cp.zeros((ny, nx), dtype=cp.float32) for n in (
        "raincv", "pratec", "nca_seconds", "cutop", "cubot", "shall",
        "cape_before", "cape_after", "timec")})
    return out


def _launch(kernel, dev, out, nz, ny, nx, ws, col0, span, phase):
    """kf_column with launch_kf's argument list, against ``out``."""
    from woof.core.kf import _device_table, load_kf_table
    from woof.core.state import DTYPE

    table = load_kf_table()
    kernel(((span + _TPB - 1) // _TPB,), (_TPB,), (
        *(dev[name] for name in _INPUTS), *_device_table(),
        out["rthcuten"], out["rqvcuten"], out["rqccuten"], out["rqicuten"],
        out["rqrcuten"], out["rqscuten"],
        out["raincv"], out["pratec"], out["nca_seconds"],
        out["cutop"], out["cubot"], out["shall"],
        out["cape_before"], out["cape_after"], out["timec"],
        out["updraft_mass_flux"], out["downdraft_mass_flux"],
        out["cldfra_dp"], out["cldfra_sh"], out["qc_kf"], out["qi_kf"], ws,
        DTYPE(table.pressure_top), DTYPE(table.pressure_reciprocal),
        DTYPE(table.thetae_reciprocal), DTYPE(12000.0), DTYPE(60.0),
        DTYPE(5.0), np.int32(phase), np.int32(1), np.int32(1),
        np.int32(nz), np.int32(ny), np.int32(nx), np.int32(col0)))


def _bits(out):
    import cupy as cp

    return {n: np.ascontiguousarray(cp.asnumpy(v)).view(np.uint32).copy()
            for n, v in out.items()}


def _differing(a, b):
    return {n: int((a[n].ravel() != b[n].ravel()).sum())
            for n in sorted(a) if (a[n].ravel() != b[n].ravel()).any()}


def _convected(bits) -> int:
    shall = bits["shall"].view(np.float32)
    return int((shall < 1.5).sum())


@pytest.mark.gpu
def test_the_workspace_is_free_of_residue():
    cp = pytest.importorskip("cupy")
    from woof.core import kf

    host = _kf_batch(192)
    nz, ny, nx = host["temperature"].shape
    ncol = ny * nx
    dev = {n: cp.asarray(v) for n, v in host.items()}
    kernel = kf._kf_function("kf_column")
    words = kf_workspace_floats(nz, ncol)

    runs = []
    for fill in (np.float32(0.0), np.float32(-7.0e30)):
        ws = cp.full(words, fill, dtype=cp.float32)
        per_phase = []
        for phase in range(4):
            out = _fresh(nz, ny, nx)
            _launch(kernel, dev, out, nz, ny, nx, ws, 0, ncol, phase)
            per_phase.append(_bits(out))
        cp.cuda.Stream.null.synchronize()
        runs.append(per_phase)
    assert _convected(runs[0][3]) > 0, (
        "no column convected, so this compared two fields of zeros")
    for phase in range(4):
        bad = _differing(runs[0][phase], runs[1][phase])
        assert not bad, (
            f"phase {phase}: {bad} words move when the workspace starts "
            "from different residue, so some column array is read before "
            "it is written and the result depends on the previous tile")


@pytest.mark.gpu
def test_the_tiled_launch_matches_the_single_launch():
    cp = pytest.importorskip("cupy")
    from woof.core import kf

    host = _kf_batch(8 * _TPB + 8)
    nz, ny, nx = host["temperature"].shape
    ncol = ny * nx
    dev = {n: cp.asarray(v) for n, v in host.items()}
    kernel = kf._kf_function("kf_column")

    for phase in range(4):
        whole = _fresh(nz, ny, nx)
        ws = cp.empty(kf_workspace_floats(nz, ncol), dtype=cp.float32)
        _launch(kernel, dev, whole, nz, ny, nx, ws, 0, ncol, phase)
        # A tile that does NOT divide the column count, so the last one is
        # partial: a col0 that walked by the tile instead of by the span
        # would skip columns here.
        tile = 3 * _TPB
        tiled = _fresh(nz, ny, nx)
        ws2 = cp.empty(kf_workspace_floats(nz, tile), dtype=cp.float32)
        for col0 in range(0, ncol, tile):
            _launch(kernel, dev, tiled, nz, ny, nx, ws2, col0,
                    min(tile, ncol - col0), phase)
        cp.cuda.Stream.null.synchronize()
        a, b = _bits(whole), _bits(tiled)
        assert _convected(a) > 0
        assert ncol % tile != 0
        bad = _differing(a, b)
        assert not bad, f"phase {phase}: tiling moved {bad}"


@pytest.mark.gpu
def test_the_shipped_launcher_tiles_and_frees_its_workspace():
    """``launch_kf`` is the door, so the tiling has to work THROUGH it.

    Its tile comes from the live card's SM count, so a domain wider than one
    tile is what proves the loop runs more than once -- and the workspace it
    allocates must be gone when it returns, or a per-call leak of hundreds
    of MiB accumulates over a forecast.
    """
    cp = pytest.importorskip("cupy")
    from woof.core import kf
    from woof.core.kf import KFPhaseMode, kf_tile_columns, launch_kf

    host = _kf_batch(2 * _TPB)
    nz, ny, nx = host["temperature"].shape
    dev = {n: cp.asarray(v) for n, v in host.items()}
    tile = kf_tile_columns(kf._kf_function("kf_column"), ny * nx)
    assert tile >= _TPB

    before = cp.get_default_memory_pool().used_bytes()
    out = launch_kf(*(dev[name] for name in _INPUTS), dx=12000.0, dt=60.0,
                    cudt_minutes=5.0,
                    phase_mode=KFPhaseMode.SEPARATE_ICE_SNOW)
    cp.cuda.Stream.null.synchronize()
    assert int(cp.asnumpy(out["triggered"]).sum()) > 0
    held = cp.get_default_memory_pool().used_bytes() - before
    outputs = sum(int(v.nbytes) for k, v in out.items() if k != "rainc")
    workspace = kf_workspace_floats(nz, min(tile, ny * nx)) * 4
    # Not `<= outputs`: CuPy's pool rounds every block up, so the output
    # arrays carry a few KiB of slack.  The claim is that the WORKSPACE is
    # gone, and it is three orders of magnitude bigger than that slack.
    assert held < outputs + workspace // 2, (
        f"launch_kf still holds {held - outputs} B beyond its outputs "
        f"against a {workspace} B workspace; the workspace was not "
        "released")
