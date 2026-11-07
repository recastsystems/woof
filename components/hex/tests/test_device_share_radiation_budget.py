"""Same-host ranks on one card share the radiation wrappers' per-call memory budgets and TEMPO's table blob.

THE BREAKAGE THIS PREVENTS (hex 0.4.14, woof.hex.device_share): eight ranks on one card each took the whole
shortwave sunlit-batch cap and the whole longwave wrapper budget at the same radiation step, so the step's
transient was eight times the one-card route's.  x1.40962 hrrrv5 at 8 rank threads on a 32 GB RTX 5090
peaked at 25.8 GiB pool used on radiation steps (15.6 GiB between them) and ran out of memory on a radiation
step between +9 h and +12 h, in 3 of 3 120 h runs.  The default executor runs one process per rank, which
cannot see the others, so the launcher must tell each rank process how many ranks share its card.
"""
from types import SimpleNamespace

import pytest

from woof.hex import device_share

G = 1 << 30


class _Runtime:
    def __init__(self, free):
        self.free = free

    def memGetInfo(self):
        return self.free, 32 * G

    def getDevice(self):
        return 0


class _Device:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def use(self):
        return None


def _cp(free):
    pool = SimpleNamespace(free_bytes=lambda: 0, get_limit=lambda: 0, used_bytes=lambda: 0)
    cuda = SimpleNamespace(runtime=_Runtime(free), Device=_Device)
    return SimpleNamespace(cuda=cuda, get_default_memory_pool=lambda: pool)


def _sunlit_width(nd, ranks, free=30 * G, nl=56, *, processes=False):
    from woof.hex.mpasphys.rrtmg_sw_v842 import MPASShortwave
    engine = SimpleNamespace(cp=_cp(free), SUNLIT_CHUNK_QUANTUM=MPASShortwave.SUNLIT_CHUNK_QUANTUM,
                             SUNLIT_CHUNK_CAP=MPASShortwave.SUNLIT_CHUNK_CAP,
                             batch_column_chunk=lambda layers: 1)
    engine._member_sunlit_column_chunk = (
        lambda *a, **k: MPASShortwave._member_sunlit_column_chunk(engine, *a, **k))
    if ranks > 1 and processes:
        with device_share.rank_process_card(0, ranks):
            return MPASShortwave.sunlit_column_chunk(engine, nd, nl, True, column_chunk=1)
    held = device_share.register([0] * ranks) if ranks > 1 else ()
    try:
        return MPASShortwave.sunlit_column_chunk(engine, nd, nl, True, column_chunk=1)
    finally:
        device_share.unregister(held)


def test_shortwave_batches_of_eight_ranks_sum_to_one_callers_cap(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_MEMBER_MEMORY", "single")
    one = _sunlit_width(40962, 1)
    eight = _sunlit_width(5120, 8)
    assert one == 4096
    assert eight == 512
    assert 8 * eight <= one


def test_compact_shortwave_is_divided_too(monkeypatch):
    # The compact policy (the default) caps one caller at 1,024 columns; eight ranks get the 256-column quantum.
    pytest.importorskip("woof.core.rrtmg_sw")
    monkeypatch.setenv("WOOF_HEX_MEMBER_MEMORY", "compact")
    assert _sunlit_width(40962, 1) == 1024
    assert _sunlit_width(5120, 8) == 256
    assert _sunlit_width(5120, 8, processes=True) == 256
    assert _sunlit_width(5120, 2, processes=True) == 512


def test_rank_processes_are_counted_like_rank_threads(monkeypatch):
    # The default executor: one process per rank sees only itself, so in-process registration reads 1.
    monkeypatch.setenv("WOOF_HEX_MEMBER_MEMORY", "single")
    assert _sunlit_width(5120, 8, processes=True) == 512
    with device_share.rank_process_card(0, 8):
        assert device_share.ranks_on_device(0) == 8
        assert device_share.ranks_on_device(1) == 1
    assert device_share.ranks_on_device(0) == 1


def test_one_card_route_shortwave_width_unchanged(monkeypatch):
    # Nothing registered: the one-card route keeps its batch widths (and so every launch shape).
    monkeypatch.setenv("WOOF_HEX_MEMBER_MEMORY", "single")
    assert device_share.ranks_on_device(0) == 1
    assert _sunlit_width(20000, 1) == 4096
    assert _sunlit_width(1000, 1) == 1000
    with device_share.rank_process_card(0, 1):
        assert _sunlit_width(20000, 1) == 4096


@pytest.mark.parametrize("policy", ["single", "compact"])
def test_longwave_wrapper_budget_is_shared(monkeypatch, policy):
    cupy = pytest.importorskip("cupy")
    pytest.importorskip("woof.core.rrtmg_lw")
    from woof.hex.mpasphys import rrtmg_lw_v842 as lw
    monkeypatch.setenv("WOOF_HEX_MEMBER_MEMORY", policy)
    monkeypatch.setattr(cupy.cuda.runtime, "memGetInfo", lambda: (30 * G, 32 * G))
    monkeypatch.setattr(cupy.cuda.runtime, "getDevice", lambda: 0)
    pool = SimpleNamespace(free_bytes=lambda: 0, get_limit=lambda: 0, used_bytes=lambda: 0)
    monkeypatch.setattr(cupy, "get_default_memory_pool", lambda: pool)
    layers = 57
    per_column = (4 * lw._LW_WRAPPER_COLUMN_WORDS_PER_LAYER * layers
                  + (lw._legacy.lw_batched_vram_bytes(256, layers) + 255) // 256)
    one = lw._lw_wrapper_chunk(10 ** 6, layers)
    held = device_share.register([0] * 8)
    try:
        eight = lw._lw_wrapper_chunk(10 ** 6, layers)
    finally:
        device_share.unregister(held)
    with device_share.rank_process_card(0, 8):
        assert lw._lw_wrapper_chunk(10 ** 6, layers) == eight
    assert eight == 256 or 8 * eight * per_column <= max(lw._LW_WRAPPER_BUDGET_BYTES, one * per_column)
    assert eight < one
    assert lw._lw_wrapper_chunk(10 ** 6, layers) == one


def test_samehost_ranks_register_their_card_until_closed():
    from woof.hex.partition_samehost_v841 import SameHostRanks
    group = SameHostRanks((0, 0, 0, 1), SimpleNamespace(), cp=_cp(30 * G), streams=[None] * 4)
    try:
        assert device_share.ranks_on_device(0) == 3
        assert device_share.ranks_on_device(1) == 1
    finally:
        group.close()
    assert device_share.ranks_on_device(0) == 1
    group.close()  # idempotent
    assert device_share.ranks_on_device(0) == 1


def test_rank_process_launcher_tells_every_rank_its_card_count():
    # The process executor's request carries each rank's card count and the IPC slot directory.
    import inspect
    from woof.hex.mpasphys import process_ranks_v842 as pr
    source = inspect.getsource(pr.run_process_ranks)
    assert "card_ranks=card_ranks" in source and "card_share_dir=" in source
    assert "rank_process_card(" in inspect.getsource(pr._run_rank)


def test_tempo_table_blob_is_held_once_per_card():
    import inspect
    from woof.hex.mpasphys.rrfs import tempo_v303
    assert "shared_readonly(" in inspect.getsource(tempo_v303._bind_native_views)


def test_read_only_objects_are_built_once_per_card_while_ranks_share_it():
    # THE BREAKAGE (woof.hex.device_share): every rank session built TEMPO's const table blob,
    # 8 identical copies on one card.
    import threading
    calls = []

    def build():
        calls.append(1)
        return object()
    assert device_share.shared_readonly("k", build, device=0) is not device_share.shared_readonly("k", build, device=0)
    assert len(calls) == 2  # one rank on the card: built per call, nothing stored
    calls.clear()
    held = device_share.register([0] * 8)
    try:
        got = []
        threads = [threading.Thread(target=lambda: got.append(device_share.shared_readonly("k", build, device=0)))
                   for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(calls) == 1 and len({id(g) for g in got}) == 1
        assert device_share.shared_readonly("other", build, device=0) is not got[0]
    finally:
        device_share.unregister(held)
    calls.clear()
    device_share.shared_readonly("k", build, device=0)
    assert len(calls) == 1  # the store went with the last rank
    calls.clear()
    with device_share.rank_process_card(0, 1, share_dir="/nonexistent"):
        device_share.shared_readonly("k", build, device=0)
    assert len(calls) == 1  # one rank process on the card: built, no IPC slot


_CHILD = r"""
import sys, cupy as cp
from woof.hex import device_share
share = sys.argv[1]
with device_share.rank_process_card(0, 2, share):
    a = device_share.shared_readonly(("blob", 1), lambda: cp.arange(1 << 20, dtype=cp.float32), cp=cp)
    print(int(bool(device_share._imported)), float(a[12345]), float(a.sum()), flush=True)
    # Ranks hold their tables until the run ends: wait for the other rank before leaving.
    import os, pathlib, time
    root = pathlib.Path(share); (root / f"done.{os.getpid()}").write_text("1")
    while len(list(root.glob("done.*"))) < 2:
        time.sleep(0.05)
"""


@pytest.mark.gpu
def test_rank_processes_map_one_table_per_card(tmp_path):
    # Two rank processes on one card: the second maps the first's export (CUDA IPC) instead of building a copy.
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA device")
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    share = tmp_path / "card-share"
    first = subprocess.Popen([sys.executable, "-c", _CHILD, str(share)], stdout=subprocess.PIPE, text=True, env=env)
    second = subprocess.Popen([sys.executable, "-c", _CHILD, str(share)], stdout=subprocess.PIPE, text=True, env=env)
    outs = [p.communicate(timeout=300)[0].split() for p in (first, second)]
    assert first.returncode == 0 and second.returncode == 0
    expected = float(cupy.arange(1 << 20, dtype=cupy.float32).sum())
    for out in outs:
        assert out[1] == "12345.0" and float(out[2]) == expected
    assert sorted(int(out[0]) for out in outs) == [0, 1]  # one built and exported it, the other mapped it
    # one export (builder + its record) and one importer; all slots cleaned at exit
    assert not [p for p in share.iterdir() if ".user" in p.name or p.suffix in (".builder", ".json")]


# --- admission: co-tenant ranks are priced from the measured row (INTAKE 2026-11-06T00:10Z) ---------------------
_R8_CELLS = [5927, 5935, 5952, 5947, 6004, 5882, 5931, 5913]   # x1.40962 seed-841 part.8, halos included


def _card():
    from woof.hex.device_admission import CardProfile
    return CardProfile(name="RTX 5090", multiprocessors=170, max_threads_per_sm=1536)


def test_eight_cotenant_ranks_fit_one_32_gb_card_from_the_measured_row():
    # Before: 105,905,390,889 bytes (eight whole members, each extrapolated below the curve), refused on any 32 GB card.
    from woof.hex.mpasphys.memory_admission import MPAS_COTENANT_RANK_ROWS, rank_requirements
    basis = {}
    need = rank_requirements(_R8_CELLS, [0] * 8, {0: _card()}, 59, basis=basis)[0]
    row = MPAS_COTENANT_RANK_ROWS[0]
    measured = row.card_peak_mib * (1 << 20)
    assert measured < need < 34_000_000_000            # above the measured peak, inside an idle 32 GB card
    assert need < 105_905_390_889 / 3
    fixed, per_column = row.terms_mib()
    assert abs(8 * fixed + per_column * sum(_R8_CELLS) - row.card_peak_mib) < 1.0   # the row reproduces itself
    assert abs(fixed + per_column * row.single_cells - row.single_peak_mib) < 1.0
    assert basis[0]["ranks"] == 8 and basis[0]["method"].startswith("co-tenant")


def test_oversubscribed_cards_still_refuse_with_the_real_number():
    from woof.hex.mpasphys.memory_admission import _pricing_words, rank_requirements
    basis = {}
    # 24 ranks of the same mesh on one card, and 8 ranks of the 60 km mesh on one card.
    need24 = rank_requirements([2_200] * 24, [0] * 24, {0: _card()}, 59, basis=basis)[0]
    assert need24 > 50_000_000_000
    words = _pricing_words(basis[0])
    assert "24 co-tenant ranks" in words and "MiB per rank" in words
    need60 = rank_requirements([22_500] * 8, [0] * 8, {0: _card()}, 59)[0]
    assert need60 > 60_000_000_000


def test_one_rank_per_card_keeps_the_one_member_price():
    from woof.hex.device_admission import MPAS_RANK_CARD_ROWS, model_for_card, route_required_free_bytes
    from woof.hex.mpasphys.memory_admission import rank_requirements
    model = model_for_card(_card(), "global", levels=59)
    one = route_required_free_bytes(40_962, model, route="mpas")
    assert rank_requirements([40_962], [0], {0: _card()}, 59)[0] == one
    # One rank alone on each of 8 cards (59 levels: no measured rank row): the one-member price of its columns,
    # raised by the largest measured rank-to-member ratio, never lowered.
    spread = rank_requirements(_R8_CELLS, list(range(8)), dict.fromkeys(range(8), _card()), 59)
    for d, c in enumerate(_R8_CELLS):
        member = route_required_free_bytes(c, model, route="mpas")
        assert member <= spread[d] <= 1.3 * member
