"""RUC arena rows are read from the pinned CUDA source (ruc_lsm_8c9f1a8.arena_words), not WORK_WORDS per cell.

The breakage these guard: every RUC launch reserved 16,384 float32 words per cell for every cell at once (2.6 GB on
x1.40962, 10.7 GB on x1.163842, where one 32 GB card ran out of memory), and each cell rebuilt one 5,001-word saturation
table that depends on constants alone.
"""
from woof.hex.mpasphys.rrfs import ruc_lsm_8c9f1a8 as ruc


class _Shape:
    def __init__(self, *shape):
        self.shape, self.ndim = shape, len(shape)


def test_bound_read_from_the_pinned_source():
    assert ruc._arena_plan() is not None
    assert ruc.arena_words("land_ruc_land", dict(nsl=9, nlcat=21, nscat=16)) == 623
    assert ruc.arena_words("land_ruc_land", dict(nsl=9, nlcat=50, nscat=30)) == 753
    assert ruc.arena_words("ice_ruc_ice", dict(nsl=9)) == 190
    assert ruc.arena_words("init_rucinit", dict(nsoillevels=9)) == 9
    assert ruc.arena_words("diags_sfcdiags_ruclsm", {}) == 2


def test_unknown_extents_keep_the_full_workspace():
    assert ruc.arena_rows("land_ruc_land", {}) == (ruc.WORK_WORDS, None)


def test_rows_take_the_larger_of_shape_and_scalar_extents():
    arguments = {}
    for field in ruc.ARGUMENTS["land_ruc_land"]:
        real = [d for d in field["dim"] if d not in ruc.ARENA_CELL_DIMS]
        size = {"1:nsl": 9, "1:nlcat": 21, "1:nscat": 16}.get(real[0]) if len(real) == 1 else None
        if size:
            arguments[field["name"]] = _Shape(size, 100)
    assert ruc.arena_rows("land_ruc_land", arguments) == (623, 623)
    assert ruc.arena_rows("land_ruc_land", arguments, {"nlcat": 50, "nscat": 30}) == (753, 753)


def test_one_shared_saturation_table():
    assert "arena.alloc<float>(1, 5001)" not in ruc.CUDA_SOURCE
    assert ruc.CUDA_SOURCE.count("FArray<float> tbq{ruc_tbq_table, 1, 1, 1, 1, 5001, 5001, 5001, 1, 1}") == 2
    assert "extern \"C\" __global__ void ruc_tbq_init()" in ruc.CUDA_SOURCE


def test_conus3km_rank_fits_the_int32_arena_offsets():
    # Largest conus3km rank of 4 (hex K3 partition): 585,440 owned cells.  Before the shared table, 5,754 words.
    assert 585_440 * ruc.arena_words("land_ruc_land", dict(nsl=9, nlcat=50, nscat=30)) < 2**31
