"""G4 (MPAS-ROUTE-SPEED-2026-10-24): fast-numerics MYNN column scheduling and resident scratch.

The breakage these guard: a generator change to the MYNN entry loop silently dropping the dynamic
schedule (or leaving a half-rewritten loop that skips or repeats columns), and the strict image
picking up the fast-only schedule (strict is the image the hl-3 bitwise gate proves).
"""
import pytest

from woof.hex.mpasphys import bl_mynn_v842 as m


def test_dynamic_loop_replaces_the_static_stride_once():
    source = m._dynamic_columns(m.MYNN_MODULE_CUDA)
    assert source.startswith("__device__ unsigned int g_mynn_next_column;\n")
    assert m._STATIC_COLUMNS not in source
    assert source.count(m._DYNAMIC_COLUMNS) == 1
    assert source.endswith(m._DYNAMIC_END)
    # Every warp claims its next 32 columns from the counter, after its static first chunk.
    assert "atomicAdd(&g_mynn_next_column, 32u)" in source
    assert "mynn_base = mynn_next + (int)(blockDim.x*gridDim.x);" in source
    # The numerical body between the loop head and the error store is untouched.
    body = m.MYNN_MODULE_CUDA.split(m._STATIC_COLUMNS, 1)[1][:-len(m._STATIC_END)]
    assert body in source


def test_changed_entry_loop_is_refused():
    with pytest.raises(ValueError, match="dynamic column scheduling needs review"):
        m._dynamic_columns(m.MYNN_MODULE_CUDA.replace("thread += blockDim.x*gridDim.x", "thread += 1"))


@pytest.mark.parametrize("mode,dynamic", [("fast", True), ("strict", False)])
def test_schedule_follows_numerics(monkeypatch, mode, dynamic):
    cp = pytest.importorskip("cupy")
    monkeypatch.setattr(m._hex_numerics, "is_strict", lambda: mode == "strict")
    module = m.MynnCudaModule(shared_math_source="")
    assert module._dynamic is dynamic
    assert ("g_mynn_next_column" in module._source) is dynamic
    assert cp is not None


def test_scratch_is_kept_between_calls(monkeypatch):
    monkeypatch.setenv('WOOF_HEX_MEMBER_MEMORY', 'single')
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001
        pytest.skip("no CUDA device")
    module = m.MynnCudaModule(shared_math_source="")
    first_ws, first_err, first_batch = module._storage(1000, 4096)
    second_ws, second_err, second_batch = module._storage(1000, 4096)
    assert second_ws is first_ws and second_batch == first_batch
    # Error flags stay per call: a caller may still hold the previous step's flags.
    assert second_err is not first_err
    third_ws, _, _ = module._storage(2000, 4096)
    assert third_ws is not first_ws
