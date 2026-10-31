"""The RRFS MYNN-EDMF lane planner splits only loops whose iterations are independent (CPU tier).

The word-for-word proof on real inputs is the module replay in tools/rrfs_mynnedmf_replay (c2efafcf module vs the
lane-group module, fast and strict, three captured hrrrv5 calls) and the whole-run comparisons. These checks pin
the planner's rules on small Fortran, each naming the breakage it prevents: a wrongly split loop computes a
different word (a recurrence read before its predecessor is written, a carried scalar read from the wrong lane).
"""
import sys
from pathlib import Path

HERE = Path(__file__).with_name("rrfs_mynnedmf")
sys.path.insert(0, str(HERE))

import translate_module as tm  # noqa: E402
import lane_plan as lp  # noqa: E402

SOURCE = """
module lane_plan_case
contains
  subroutine mynnedmf_driver(n, a, b, c, x)
    integer, intent(in) :: n
    real, dimension(n), intent(inout) :: a, b, c
    real, intent(out) :: x
    integer :: k
    real :: t, s
    real, dimension(n) :: w1, w2
    do k=1,n
       t = b(k)*2.
       a(k) = t + c(k)
    enddo
    x = t
    do k=2,n
       a(k) = a(k-1) + b(k)
    enddo
    s = 0.
    do k=1,n
       s = s + a(k)
       b(k) = s
    enddo
    do k=1,n
       if (a(k) > 1.) exit
       c(k) = 0.
    enddo
    do k=1,n
       if (a(k) > 0.) then
          t = 1.
       else
          t = 2.
       endif
       c(k) = t
    enddo
    do k=1,n
       if (a(k) > 0.) t = a(k)
       c(k) = 1.
    enddo
    x = t
    do k=1,n-1
       c(k) = a(k+1)
       a(k) = 0.
    enddo
    w1(1:n) = c(1:n)
    w2(1:n) = w2(1:n)*2.
  end subroutine mynnedmf_driver
end module lane_plan_case
"""

LINES = {line.strip(): number for number, line in enumerate(SOURCE.splitlines(), 1)}


def plan(tmp_path):
    path = tmp_path / "lane_plan_case.F90"
    path.write_text(SOURCE)
    globals_, routines = tm.parse_modules([path])
    routine = routines["mynnedmf_driver"]
    emitter = tm.Emitter(globals_, routines, routine)
    lane_safe = lp.lane_safe_routines(globals_, routines)
    alias = lp.alias_classes(globals_, routines, "mynnedmf_driver")
    result = lp.plan_routine(emitter, routine, lane_safe, alias)
    split = sorted(routine.body[i][0] for i in result.loops)
    assigns = sorted(routine.body[i][0] for i in result.elementwise)
    privates = {routine.body[i][0]: [n for n, _ in p.privates] for i, p in result.loops.items()}
    return split, assigns, privates, routine, emitter, result


def loop_lines(text):
    return [n for n, line in enumerate(SOURCE.splitlines(), 1) if line.strip() == text]


def test_split_and_serial_loops(tmp_path):
    split, assigns, privates, *_ = plan(tmp_path)
    loops = loop_lines("do k=1,n")
    first, reduction, exit_loop, ifelse, conditional = loops
    recurrence = loop_lines("do k=2,n")[0]
    shifted = loop_lines("do k=1,n-1")[0]
    # independent iterations, scalar assigned at the top of each: split, t broadcast from the last iteration
    assert first in split and privates[first] == ["t"]
    # scalar assigned in both branches: split, broadcast
    assert ifelse in split and privates[ifelse] == ["t"]
    # Breakage: a(k) = a(k-1) + ... split across lanes reads a(k-1) before another lane writes it.
    assert recurrence not in split
    # Breakage: s = s + a(k) split across lanes sums on each lane separately.
    assert reduction not in split
    # Breakage: EXIT on one lane would not stop the others.
    assert exit_loop not in split
    # Breakage: t assigned on some iterations only and read after the loop: the last writer is not the last lane.
    assert conditional not in split
    # Breakage: c(k) = a(k+1); a(k) = 0 split across lanes reads a(k+1) after its lane zeroed it.
    assert shifted not in split


def test_array_assignments(tmp_path):
    split, assigns, *_ = plan(tmp_path)
    assert LINES["w1(1:n) = c(1:n)"] in assigns
    # Breakage: an array assignment that reads the array it writes is left in its serial element order.
    assert LINES["w2(1:n) = w2(1:n)*2."] not in assigns


def test_emitted_split_loop_restores_serial_exit_state(tmp_path):
    *_, routine, emitter, result = plan(tmp_path)
    emitter.lane_plan = result
    code = emitter.routine_code()
    # the index takes its serial exit value and the private its last iteration's value, from the lane that ran it
    assert "k = mynn_lo0 + mynn_n0*mynn_st0;" in code
    assert "t = mynn_lane_bcast(t, mynn_src0);" in code
    assert "for (int mynn_it0 = lane_sub; mynn_it0 < mynn_n0; mynn_it0 += lane_width)" in code
    assert code.count("mynn_lane_sync();") >= 2 * len(result.loops) + 2 * len(result.elementwise)


def test_generated_module_carries_the_lane_plan():
    # Breakage: a regenerated backend without the plan would fall back to one thread per column (81 warps on a
    # 170-SM card for x1.2562).
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf
    import generate_backend
    assert generate_backend.KERNEL_TILE_LINE in edmf.MODULE_CUDA
    assert edmf.MODULE_CUDA.count("mynn_lane_sync(); for (int mynn_it") > 150
    assert "def lanes_per_column(" in Path(edmf.__file__).read_text()
    # one kernel per compiled lane width, the generator's and the launcher's lists the same
    assert tuple(generate_backend.LANE_WIDTHS) == tuple(edmf.LANE_WIDTHS) and 1 in edmf.LANE_WIDTHS
    for width in edmf.LANE_WIDTHS:
        qualifier = 'MYNN_L4_REGS ' if width == 4 else ''
        assert f'extern "C" __global__ {qualifier}void rrfs_mynnedmf_run_l{width}(' in edmf.MODULE_CUDA


def test_lane_width_is_a_compile_time_constant():
    # Breakage: a run-time lane width made every automatic array's stride a run-time value; the kernel frame grew
    # from 2,640 B to 4,544 B (fast, sm_120), over the 4 KiB at which device_memory_release gives the local-memory
    # store back after every PBL phase (two device syncs a step).
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf
    assert "template<int MYNN_LANES> struct MynnColumn {" in edmf.MODULE_CUDA
    assert "static constexpr int lane_width = MYNN_LANES;" in edmf.MODULE_CUDA
    assert "mynn_array<float, 1, mynn_tile>(arena," in edmf.MODULE_CUDA
    assert "result.stride[d]=count*TILE;" in edmf.ARRAY_SUPPORT


def test_arena_bound_uses_nested_frames_not_all_siblings(monkeypatch):
    # Both children reuse the same cursor. Summing siblings creates needless
    # batches; taking only the largest individual frame corrupts parent data.
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf
    array = lambda n: {'kind': 'float', 'dimensions': [str(n)]}
    monkeypatch.setattr(edmf, 'WORKSPACE_FRAMES', {
        'mynnedmf_driver': {'arrays': [array(3)], 'calls': ['a', 'b']},
        'a': {'arrays': [array(5)], 'calls': ['c']},
        'b': {'arrays': [array(8)], 'calls': []},
        'c': {'arrays': [array(3)], 'calls': []},
    })
    args = dict(kts=1, kte=59, kme=60, nchem=0, ndvel=0)
    assert edmf.workspace_bytes(args) == 16 + max(24 + 16, 32)


def test_generated_frames_cover_every_translated_call_and_allocation():
    # A missing expression-function edge or scratch array would underallocate
    # the arena. Check the generated C++ itself, not just the Fortran metadata.
    import re
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf
    text = edmf.MODULE_CUDA.split('template<int MYNN_LANES> __device__')[0]
    bodies = re.split(r'__device__ __\w+__ \w+ f_(\w+)\(', text)
    for name, body in zip(bodies[1::2], bodies[2::2]):
        frame = edmf.WORKSPACE_FRAMES[name]
        calls = set(re.findall(r'\bf_(\w+)\(', body))
        assert calls <= set(frame['calls'])
        assert body.count(' = mynn_array<') == len(frame['arrays'])


def test_arena_bound_keeps_optional_chemistry():
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf
    args = dict(kts=1, kte=59, kme=60, nchem=0, ndvel=0)
    assert edmf.workspace_bytes(args) == 71528
    assert edmf.workspace_bytes({**args, 'nchem': 10, 'ndvel': 10}) > 71528


def test_dispatch_preserves_small_shape_and_strict_throughput(monkeypatch):
    # Occupancy alone rejects the measured fast W4 winner on a large tile.
    # Applying that fast choice to strict regresses its uncapped kernel.
    from types import ModuleType, SimpleNamespace
    from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf
    cupy, cuda = ModuleType('cupy'), ModuleType('cupy.cuda')
    cupy.cuda = cuda
    cuda.Device = lambda: SimpleNamespace(attributes={'MultiProcessorCount': 170})
    cuda.driver = SimpleNamespace(occupancyMaxActiveBlocksPerMultiprocessor=lambda *args: 8)
    monkeypatch.setitem(sys.modules, 'cupy', cupy)
    monkeypatch.setitem(sys.modules, 'cupy.cuda', cuda)
    monkeypatch.delenv('WOOF_HEX_MYNNEDMF_LANES', raising=False)
    kernels = {w: SimpleNamespace(ptr=w) for w in edmf.LANE_WIDTHS}
    for mode, large_width in [('fast', 4), ('strict', 1)]:
        monkeypatch.setattr(edmf._hex_numerics, 'mode', lambda: mode)
        assert edmf.lanes_per_column(kernels, 2562) == 16
        assert edmf.lanes_per_column(kernels, 30016) == large_width
    monkeypatch.setenv('WOOF_HEX_MYNNEDMF_LANES', '4')
    assert edmf.lanes_per_column(kernels, 30016) == 4
