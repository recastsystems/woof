"""RRFS MYNN-EDMF and fork scalar transport keep their GPU execution shape (CPU tier).

The word-for-word proof on real inputs is the module replay in tools/rrfs_mynnedmf_replay (fast and strict: 0 of
8,974,686 words differ on each of three captured hrrrv5 calls) and the whole-run comparison. These checks keep the
shape from sliding back, each naming the breakage it prevents.
"""
from pathlib import Path

from woof.hex import cuda_transport_rrfs as transport
from woof.hex.mpasphys.rrfs import bl_mynnedmf_c942418 as edmf

# Lane groups (rrfs/speed2-mynn-edmf): a warp holds 32/lane_width columns, one tile slot each.
TILE_KERNEL_LINE = ("MynnArena arena{workspace+(unsigned long long)(column_index-column_index%tile_columns)"
                    "*workspace_stride,0,tile_columns,column_index%tile_columns};")


def test_strict_image_is_optimised_and_still_strict():
    # Breakage: the strict image compiled at ptxas -O0 ran the PBL at 242 ms of a 322 ms strict hrrrv5 step.
    options = edmf.CUDA_OPTIONS
    assert not any("-O0" in option for option in options)
    for flag in ("--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"):
        assert flag in options


def test_image_compiles_on_all_cores():
    # Breakage: the three lane-width kernels compiled on one core (fast 70-126 s, strict several minutes) on a box's
    # first hrrrv5 run without a kernel pack.
    assert "--split-compile=0" in edmf.CUDA_OPTIONS
    assert "--ptxas-options=--split-compile=0" in edmf.CUDA_OPTIONS


def test_operation_wrappers_are_inline():
    # Breakage: every one of the module's ~5,000 binary32 operations was a non-inlined device call
    # (kernel 25 ms per call on x1.2562, fast).
    for name in ("r6_neg", "r6_nan", "r6_add", "r6_sub", "r6_mul", "r6_div"):
        assert f"__noinline__ float {name}(" not in edmf.ARRAY_SUPPORT
        assert f"__forceinline__ float {name}(" in edmf.ARRAY_SUPPORT


def test_fast_operations_cannot_contract():
    # Breakage: an inlined plain operator may be contracted into an FMA under --fmad=true, changing fast words
    # (the isolated calls rounded every operation on its own).
    fast = edmf.ARRAY_SUPPORT.split("#if defined(HEX_NUMERICS_FAST) && HEX_NUMERICS_FAST", 1)[1].split("#else", 1)[0]
    for op, intrinsic in (("ADD", "__fadd_rn"), ("SUB", "__fsub_rn"), ("MUL", "__fmul_rn"), ("DIV", "__fdiv_rn")):
        assert f"#define MYNN_R6_{op}(a,b) {intrinsic}(a,b)" in fast


def test_arena_is_warp_tiled_in_module_and_generator():
    # Breakage: a per-thread arena (110 KB apart per column) makes every automatic-array access 32 separate lines,
    # and a regenerated backend would silently bring it back.
    assert edmf.MODULE_CUDA.count(TILE_KERNEL_LINE) == 1
    generator = Path(__file__).with_name("rrfs_mynnedmf") / "generate_backend.py"
    assert TILE_KERNEL_LINE in generator.read_text()
    assert "result.stride[d]=count*TILE;" in edmf.ARRAY_SUPPORT


def test_fork_scalar_transport_threads_cover_levels():
    # Breakage: one thread per cell/edge looping every level ran 2,562 threads (21 blocks) for 1.5 ms per launch.
    source = Path(transport.__file__).read_text()
    assert "(((nl*ne+127)//128,),(128,)" in source
    assert "(((nl*nc+127)//128,),(128,)" in source


def test_entry_routines_are_inline_so_the_frame_stays_under_4_kib():
    # Breakage: with the entry routines called (not inlined), their ~100-array by-value parameter blocks made the
    # kernel's per-thread stack 4,432 B (fast) / 4,688 B (strict), over the 4 KiB at which device_memory_release
    # gives the local-memory store back after every PBL phase: two device syncs every step (hrrrv5 x1.2562: 30
    # resets, 29 regrows in 30 steps), which erased the kernel's speedup from the step time.
    for name in ("f_mynnedmf_driver", "f_mynnedmf"):
        assert f"__device__ __forceinline__ void {name}(" in edmf.MODULE_CUDA
        assert f"__device__ __noinline__ void {name}(" not in edmf.MODULE_CUDA
    generator = (Path(__file__).with_name("rrfs_mynnedmf") / "generate_backend.py").read_text()
    assert "inline_entry_routines(lower_gnu_operand_order(text+kernel))" in generator
