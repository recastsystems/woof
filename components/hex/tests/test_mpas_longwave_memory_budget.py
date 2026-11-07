"""Longwave admission prices the solver that shares each wrapper's lifetime."""
import sys
from types import SimpleNamespace

import pytest

from woof.hex.mpasphys import rrtmg_lw_v842 as lw


@pytest.mark.parametrize('layers', [56, 87, 128])
@pytest.mark.parametrize('limit', [0, 12 << 30])
def test_complete_batch_fits_budget_and_member_headroom(monkeypatch, layers, limit):
    pool = SimpleNamespace(free_bytes=lambda: 2 << 30, get_limit=lambda: limit,
                           used_bytes=lambda: 8 << 30)
    device = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(
        memGetInfo=lambda: (16 << 30, 32 << 30))), get_default_memory_pool=lambda: pool)
    monkeypatch.setitem(sys.modules, 'cupy', device)
    width = lw._lw_wrapper_chunk(72966, layers)
    available = min(18 << 30, limit - (8 << 30)) if limit else 18 << 30
    budget = min(lw._LW_WRAPPER_BUDGET_BYTES, available // 8)
    wrapper = width * 4 * lw._LW_WRAPPER_COLUMN_WORDS_PER_LAYER * layers
    assert wrapper + lw._legacy.lw_batched_vram_bytes(width, layers) <= budget
    assert width % 256 == 0
    assert lw._lw_wrapper_chunk(73, layers) == 73
