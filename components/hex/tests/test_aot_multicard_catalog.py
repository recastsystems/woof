"""The CUDA 13 recipe catalog holds the kernels a multi-card run asks CuPy for."""
import json


def test_catalog_holds_the_multi_card_cupy_kernels():
    """THE BREAKAGE: a fresh box's multi-card run compiled CuPy's own kernels the shipped pack lacked, on each
    rank, at its first step (woof hex 0.4.9, VRFS 8x RTX 5090, 2026-10-28: ft-r700-2c/8c, pc-after-8c, the RRFS
    hrrrv5 runs at 2-8 cards): the int64 2-D add, take, scatter_update and halo_copy_words_2d of the same-host
    exchange, and CUB's bool any() reduction, whose block CuPy sizes from the reduced length, so each mesh and card
    count can ask for another source (block 256 at r700 2/8 cards, block 64 at x1.40962 4/8 cards and on one card at
    r300; a shorter array takes the plain cupy_any kernel). The CUDA 13 catalog holds them, and the any() pair at
    every CUB block size CuPy chose for n = 200, 400, 800 and 4000 (64, 128, 256, 512; n = 100 took cupy_any)."""
    import re
    from pathlib import Path
    import woof.hex
    rows = json.loads((Path(woof.hex.__file__).parent / "data" / "compiler-recipes.json").read_text())
    blocks = {"cupy_cub_any_pass1": set(), "cupy_cub_any_pass2": set()}
    for entry in rows:
        source = entry["source"]
        for name in blocks:
            if re.search(r"__global__\s+void\s+" + name + r"\b", source) and "typedef bool type_in0_raw;" in source:
                blocks[name].update(int(b) for b in re.findall(r"#define BLOCK_SIZE\s+(\d+)", source))
    assert blocks == {name: {64, 128, 256, 512} for name in blocks}
    for kernel in ("cupy_any", "cupy_take", "cupy_scatter_update", "halo_copy_words_2d"):
        assert any(re.search(r"__global__\s+void\s+" + kernel + r"\b", entry["source"]) for entry in rows), kernel
    assert any("cupy_add__int64_int64_int64(const CArray<long long, 2, 0, 1>" in entry["source"] for entry in rows)
