"""CPU native SW dispatch regression. Device operations are faked, not parity."""
from types import SimpleNamespace
import sys
import numpy as np
import pytest

@pytest.fixture
def harness(monkeypatch):
    from woof.hex.mpasphys import rrtmg_sw_v842 as native
    from woof import nvrtc_ptx_cache
    import importlib
    kernel_manifest = importlib.import_module("woof.certify.kernel_manifest")
    calls, records, waits = [], [], []
    class Module:
        def load(self, image): self.image = image
        def get_function(self, name): return self, name
    class Event:
        def __init__(self, **kwargs): self.recorded = False
        def record(self): self.recorded = True
    def stream(ptr):
        return SimpleNamespace(ptr=ptr, wait_event=lambda event: waits.append((ptr, event)))
    active = [stream(11)]
    cp = SimpleNamespace(asarray=np.asarray, cuda=SimpleNamespace(
        function=SimpleNamespace(Module=Module), Event=Event,
        get_current_stream=lambda: active[0]))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    failure = [False]
    def compile_source(*args):
        calls.append(args)
        if failure[0]: raise RuntimeError("injected native shape compile failure")
        return "fake-ptx", {}
    monkeypatch.setattr(nvrtc_ptx_cache, "compile_using_nvrtc", compile_source)
    monkeypatch.setattr(kernel_manifest, "record_module",
                        lambda key, **kwargs: records.append((key, kwargs)))
    tables = SimpleNamespace(ngb=[1, 2])
    monkeypatch.setattr(native, "_pack_cuda_tables",
                        lambda tab: (np.zeros(2, dtype=np.float32), "#define NATIVE_TABLE_SENTINEL 7\n"))
    monkeypatch.setattr(native.MPASShortwave, "_laysolfr_band_spec", lambda self: [1, 2])
    initialized = []
    def raw_layout(self, raw):
        self.tab = tables
        return np.zeros(2, dtype=np.float32), "#define NATIVE_TABLE_SENTINEL 7\n", ["plan"]
    monkeypatch.setattr(native.MPASShortwave, "_raw_table_layout", raw_layout)
    monkeypatch.setattr(native.MPASShortwave, "_initialize_raw",
                        lambda self, raw, plans: initialized.append((raw, plans)))
    return SimpleNamespace(native=native, tables=tables, calls=calls, records=records,
                           waits=waits, active=active, stream=stream, failure=failure,
                           initialized=initialized)

@pytest.mark.parametrize("mode", ["strict", "fast"])
@pytest.mark.parametrize("route", ["tables", "raw"])
def test_native_constructor_shape_and_stream_contract(harness, monkeypatch, mode, route):
    h = harness
    monkeypatch.setattr(h.native._hex_numerics, "_mode", mode)
    monkeypatch.setattr(h.native._hex_numerics, "_bound", False)
    kwargs = dict(tables=h.tables) if route == "tables" else dict(raw_coefficients={"raw": 1})
    sw = h.native.MPASShortwave(**kwargs)
    assert bool(h.initialized) == (route == "raw")
    original_source, original_options, _, original_name = h.calls[0]
    assert original_source.startswith("#define NATIVE_TABLE_SENTINEL 7\n")
    assert original_source.endswith(h.native.CUDA_SOURCE)
    assert original_name == "mpas_rrtmg_sw_v842.cu"
    assert "--prec-div=true" in original_options and "--prec-sqrt=true" in original_options
    assert ("--fmad=false" if mode == "strict" else "--fmad=true") in original_options
    assert ("--ftz=false" if mode == "strict" else "--ftz=true") in original_options
    assert sw._tables_ready.recorded
    generic = sw._k("rsw_spcvmc_gpt_b")
    first = sw._k("rsw_spcvmc_gpt_b", shape=(55, 1, 513))
    assert first != generic and len(h.calls) == 2
    assert sw._k("rsw_spcvmc_gpt_b", shape=(55, 1, 513)) == first
    assert len(h.calls) == 2 and len(h.waits) == 1
    specialized, options, arch, name = h.calls[1]
    assert (options, arch, name) == (original_options, None, original_name)
    # Undo just the constants and ABI parameter renames: all native code stays.
    restored = specialized.replace("gpuwm_runtime_ncol", "ncol").replace("gpuwm_runtime_nlayers", "nlayers")
    assert "constexpr int ncol = 513;" in specialized
    assert "constexpr int nlayers = 55;" in specialized
    restored = restored.replace("\n    constexpr int ncol = 513;\n    constexpr int nlayers = 55;\n", "")
    assert restored == original_source
    assert h.records[1][0] == "woof.hex.mpasphys.rrtmg_sw_v842:rrtmg_sw.shape.55.1.513"
    assert h.records[1][1]["source"] == specialized
    assert h.records[1][1]["options"] == original_options
    h.active[0] = h.stream(22)
    assert sw._k("rsw_spcvmc_gpt_b", shape=(55, 1, 513)) == first
    assert len(h.waits) == 2 and sw._table_streams[22] is h.active[0]
    assert sw._k("rsw_spcvmc_gpt_b", shape=(54, 1, 257)) != first
    assert len(h.calls) == 3 and len(h.waits) == 2
    assert sw._k("rsw_spcvmc_gpt_b") == generic
    h.failure[0] = True
    with pytest.raises(RuntimeError, match="injected native shape compile failure"):
        sw._k("rsw_spcvmc_gpt_b", shape=(53, 1, 257))
    assert (53, 1, 257) not in sw._shape_modules
    h.failure[0] = False
    sw._k("rsw_spcvmc_gpt_b", shape=(53, 1, 257))
    assert (53, 1, 257) in sw._shape_modules

@pytest.mark.parametrize("limit", [0, 15569256448])
def test_sunlit_batch_prices_solver_and_member_pool_limit(limit, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_MEMBER_MEMORY", "shared-card")
    """A roomy card must not overrun a smaller per-member allocator budget."""
    from woof.hex.mpasphys.rrtmg_sw_v842 import MPASShortwave
    from woof.core.rrtmg_sw import sw_batched_vram_bytes
    pool = SimpleNamespace(free_bytes=lambda: 0, get_limit=lambda: limit,
                           used_bytes=lambda: 13 * 1024**3)
    engine = object.__new__(MPASShortwave)
    engine.cp = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        memGetInfo=lambda: (16 * 1024**3, 32 * 1024**3))),
        get_default_memory_pool=lambda: pool)
    width = engine.sunlit_column_chunk(167254, 56, aerosol=True, column_chunk=256)
    wrapper = width * 4 * (8 * 56 * 112 + 15 * 56 + 2 * 57 + 3 * 14 * 56 + 16 * 57)
    headroom = limit - pool.used_bytes() if limit else 16 * 1024**3
    assert wrapper + sw_batched_vram_bytes(width, 56, mcica_layout="column") <= headroom // 2
    assert width % 256 == 0
    if limit:
        assert width < 2048
