"""rrfs/nssl-fast: the fast NSSL route never touches the strict text, and its smaller column arena holds the audited
column-phase stack (CPU only)."""
import collections
import hashlib
import re

import pytest

from woof.hex.mpasphys.rrfs import mp_nssl2m as m
from woof.hex.mpasphys.rrfs import nssl_fast



def _strict():
    return m.cuda_source(shared_math_source="", numerics="strict")


def test_fast_off_is_the_strict_text(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_NSSL_FAST", "0")
    assert m.cuda_source(shared_math_source="", numerics="fast") == _strict()


def test_strict_text_ignores_fast_knobs(monkeypatch):
    before = hashlib.sha256(_strict().encode()).hexdigest()
    monkeypatch.setenv("WOOF_HEX_NSSL_FAST", "1")
    monkeypatch.setenv("WOOF_HEX_NSSL_FAST_STAGES", "ops,maths,float,shape")
    assert hashlib.sha256(_strict().encode()).hexdigest() == before
    assert nssl_fast.column_stride(59) < 65536 + 10240 * 59


def test_step_path_has_no_binary64(monkeypatch):
    from woof.hex.mpasphys.rrfs import nssl_math
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST_STAGES", raising=False)
    fast = m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    for routine in ("f_sediment1d", "f_setvtz", "f_nucond", "f_nssl_2mom_gs", "f_calcnfromq", "f_radardd02"):
        start = re.search(r"__device__ (?:__noinline__ |__forceinline__ )?\w+ " + routine + r"\(", fast).start()
        end = nssl_fast._block_end(fast, start)
        assert not re.search(r"\bdouble\b", fast[start:end]), routine
    assert "nssl_fast_tables" in fast


def _column_bound(nz):
    """workspace_bound.py's largest root-to-leaf stack of the driver, without the collection routine."""
    alloc, calls, fn = [], collections.defaultdict(set), None
    for line in m.MODULE_SOURCE.splitlines():
        if line.startswith("__device__"):
            match = re.search(r"\bf_\w+", line)
            fn = match[0] if match else None
            continue
        if fn:
            calls[fn].update(re.findall(r"\b(f_\w+)\(", line))
        if fn and "nssl_array<" in line and "(arena" in line:
            typ, lo, hi = re.search(r"nssl_array<(\w+),\s*\d+>\(arena, \{(.*?)\}, \{(.*?)\}\)", line).groups()
            alloc.append((fn, lo.split(","), hi.split(",")))
    ctx = dict(nx=1, ny=1, nz=nz, its=1, ite=1, kts=1, kte=nz, na=30, nxtra=1, nproc=1, nor=3, norz=3, ng1=1,
               ngs=min(500, nz), ngscnt=min(500, nz), ng=12, ndiam=10, lc=3, lr=4, li=5, ls=6, lh=7, lv=2,
               lhab=9, lqmx=30)
    by = collections.Counter()
    for fn, lo, hi in alloc:
        count = 1
        for a, b in zip(lo, hi):
            count *= eval(b, {"__builtins__": {}}, ctx) - eval(a, {"__builtins__": {}}, ctx) + 1
        by[fn] += 7 + count * 8

    def live(f):
        return by[f] + max((live(c) for c in calls[f] if c != "f_nssl_2mom_gs"), default=0)
    return live("f_nssl_2mom_driver")


@pytest.mark.parametrize("nz", [1, 2, 30, 55, 59, 60, 65, 100, 128, 200])
def test_fast_column_stride_holds_the_column_phase_stack(nz, monkeypatch):
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST_STAGES", raising=False)
    bound = _column_bound(nz)
    assert bound <= nssl_fast.COLUMN_BOUND_INTERCEPT + nssl_fast.COLUMN_BOUND_SLOPE * nz
    assert nssl_fast.column_stride(nz) * 3 // 4 >= bound


def test_collection_work_arrays_are_registers(monkeypatch):
    from woof.hex.mpasphys.rrfs import nssl_math
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST_STAGES", raising=False)
    fast = m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    start = fast.index("void f_nssl_2mom_level(")
    level = fast[start:nssl_fast._block_end(fast, start)]
    assert level.count("NsslScalar1<") >= 700
    assert "nssl_array<float, 1>(arena, {1}, {ngs})" in level  # the arrays handed whole to f_setvtz stay arrays
    monkeypatch.setenv("WOOF_HEX_NSSL_FAST_STAGES", "ops,maths,float,shape")
    assert "NsslScalar1<" not in m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")


def test_column_collection_maps_one_thread_per_column(monkeypatch):
    from woof.hex.mpasphys.rrfs import nssl_math
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST_STAGES", raising=False)
    fast = m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    assert nssl_fast._COLLECT_COLUMN in fast and nssl_fast._COLLECT_POINT not in fast
    assert "for (int selected_level=1; selected_level<=nz; ++selected_level)" in fast
    assert nssl_fast.collect_columns()


def test_split_sedimentation_is_fast_only_and_keeps_the_column_fallback(monkeypatch):
    from woof.hex.mpasphys.rrfs import nssl_fastsed, nssl_math
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST_STAGES", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_SEDFALL", raising=False)
    strict = hashlib.sha256(_strict().encode()).hexdigest()
    fast = m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    for kernel in ("nssl_sediment(", "nssl_sedpre(", "nssl_sedfall(", "nssl_sedlanes(", "nssl_fast_sed_bind("):
        assert re.search(r'extern "C" __global__ void (__launch_bounds__\(\d+, \d+\) )?' + re.escape(kernel), fast), kernel
    assert "f_fast_sed_export(" in fast and "f_ziegfall1d_lvl(" in fast and "f_calcnfromz1d_lvl(" in fast
    for routine in ("f_fast_sed_point", "f_fast_sed_lanes", "f_ziegfall1d_lvl", "f_calczgr1d_lvl", "f_calcnfromz1d_lvl"):
        start = re.search(r"__device__ __forceinline__ void " + routine + r"\(", fast).start()
        assert not re.search(r"\bdouble\b", fast[start:nssl_fast._block_end(fast, start)]), routine
    monkeypatch.setenv("WOOF_HEX_NSSL_SEDFALL", "column")
    column = m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    assert "f_fast_sed_fall_column(dtp, nz" in column and column != fast
    assert hashlib.sha256(_strict().encode()).hexdigest() == strict
    monkeypatch.setenv("WOOF_HEX_NSSL_FAST_STAGES", "ops,maths,float,shape,regs,column")
    assert "nssl_sedlanes" not in m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    with pytest.raises(ValueError):
        nssl_fastsed.apply("f_fast_sed_lanes", "")
    monkeypatch.setenv("WOOF_HEX_NSSL_SEDFALL", "levels")
    with pytest.raises(ValueError):
        nssl_fastsed.fall_mode()


def test_launch_bounds_cap_the_column_phases(monkeypatch):
    from woof.hex.mpasphys.rrfs import nssl_math
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST_STAGES", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_BOUNDS", raising=False)
    fast = m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    for kernel in ("setup", "sedpre", "condense", "finish"):
        assert f'extern "C" __global__ void __launch_bounds__(32, 16) nssl_{kernel}(' in fast, kernel
    assert '__launch_bounds__(32, 16) nssl_collect(' not in fast
    monkeypatch.setenv("WOOF_HEX_NSSL_BOUNDS", "none")
    assert "__launch_bounds__" not in m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")


def _point_scratch(fast, routines, nz=128, lhab=8):
    """Bytes the per-point routines allocate from their thread-local arena (ngs = 1), each allocation 8-byte aligned."""
    total = 0
    ctx = dict(nx=1, ny=1, nz=nz, nor=0, norz=0, ng1=1, ngs=1, ngscnt=1, lc=3, lr=4, li=5, lv=2, ls=6, lh=7, lhl=8,
               lhab=lhab, lqmx=30, itile=1, nxi=1)
    for routine in routines:
        start = re.search(r"__device__ __forceinline__ void " + routine + r"\(", fast).start()
        text = fast[start:nssl_fast._block_end(fast, start)]
        for typ, lo, hi in re.findall(r"nssl_array<(\w+), \d>\((?:arena|la), \{(.*?)\}, \{(.*?)\}\)", text):
            count = 1
            for a, b in zip(lo.split(","), hi.split(",")):
                count *= eval(b, {"__builtins__": {}}, ctx) - eval(a, {"__builtins__": {}}, ctx) + 1
            total += -(-count * {"float": 4, "int": 4, "bool": 1, "double": 8}[typ] // 8) * 8
    return total


def test_point_condensation_and_scratch_bounds(monkeypatch):
    from woof.hex.mpasphys.rrfs import nssl_fastcond, nssl_fastsed, nssl_math
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_FAST_STAGES", raising=False)
    monkeypatch.delenv("WOOF_HEX_NSSL_SEDFALL", raising=False)
    fast = m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    for kernel in ("nssl_condense(", "nssl_condpre(", "nssl_condpoint(", "nssl_fast_cond_bind("):
        assert kernel in fast, kernel
    assert "f_nucond_lvl(int selected_level," in fast and "f_nucond_pre(" in fast
    # thread-local scratch holds the per-point routines' allocations, up to 128 levels and 16 species
    # (sized to the bound: thread-local bytes are reserved for every resident thread of the card)
    assert _point_scratch(fast, ("f_nucond_lvl", "f_qvexcess"), lhab=18) <= nssl_fastcond.SCRATCH
    assert _point_scratch(fast, ("f_fast_sed_point", "f_ziegfall1d_lvl", "f_setvtz"), lhab=18) <= nssl_fastsed.SCRATCH
    monkeypatch.setenv("WOOF_HEX_NSSL_FAST_STAGES", "ops,maths,float,shape,regs,column,sed,bounds")
    assert "nssl_condpoint" not in m.cuda_source(shared_math_source=nssl_math.source(), numerics="fast")
    with pytest.raises(ValueError):
        nssl_fastcond.apply("f_fast_cond_column", "")
