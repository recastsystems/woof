"""Radiation chunk width: priced by the engine, never on small cards."""

import pytest

from woof.hex import radiation_chunk_v841 as rc

GIB = 2**30


def _price(width):
    # Affine like legacy_radiation_vram_bytes: constants + per-column workspace.
    return 300 * 2**20 + width * 900_000


def test_large_card_widens_within_budget_on_quantum():
    width = rc.choose_width(ncol=327_680, nz=55, p_top=3000.0, total_bytes=96 * GIB,
                            price=_price, default_bytes=_price(2304))
    assert width is not None and width % rc.QUANTUM == 0
    assert width > 2816
    assert _price(width) <= rc.RADIATION_CHUNK_BUDGET_FRACTION * 96 * GIB
    assert _price(width + rc.QUANTUM) > rc.RADIATION_CHUNK_BUDGET_FRACTION * 96 * GIB


def test_small_card_keeps_engine_width():
    assert rc.choose_width(ncol=327_680, nz=55, p_top=3000.0, total_bytes=32 * GIB,
                           price=_price, default_bytes=_price(2048)) is None


def test_width_never_exceeds_the_rank_columns():
    width = rc.choose_width(ncol=1000, nz=55, p_top=3000.0, total_bytes=96 * GIB,
                            price=_price, default_bytes=_price(256))
    assert width == 1024


def test_default_over_budget_keeps_engine_width():
    assert rc.choose_width(ncol=327_680, nz=55, p_top=3000.0, total_bytes=96 * GIB,
                           price=_price, default_bytes=10**12) is None


class RRTMGLegacyRadiation:
    def __init__(self):
        self.column_chunk = None
        self.p_top = 3000.0


class _Seam:
    def __init__(self, adapter):
        self._driver = type("D", (), {"radiation_callable": adapter})()
        self._identity = {"n_columns": 327_680, "n_levels": 55, "p_top_pa": 3000.0}


def test_engine_setting_and_explicit_width(monkeypatch):
    adapter = RRTMGLegacyRadiation()
    monkeypatch.setenv("WOOF_HEX_RADIATION_CHUNK", "engine")
    receipt = rc.widen_radiation_chunk(_Seam(adapter), cp=None)
    assert adapter.column_chunk is None and receipt["applied"] is False
    monkeypatch.setenv("WOOF_HEX_RADIATION_CHUNK", "4096")
    receipt = rc.widen_radiation_chunk(_Seam(adapter), cp=None)
    assert adapter.column_chunk == 4096 and receipt["applied"] is True
    monkeypatch.setenv("WOOF_HEX_RADIATION_CHUNK", "zero")
    with pytest.raises(ValueError, match="ambiguous radiation receipt"):
        rc.widen_radiation_chunk(_Seam(RRTMGLegacyRadiation()), cp=None)


def test_other_radiation_schemes_are_untouched(monkeypatch):
    monkeypatch.delenv("WOOF_HEX_RADIATION_CHUNK", raising=False)
    other = type("RRTMGPRadiation", (), {"column_chunk": None})()
    receipt = rc.widen_radiation_chunk(_Seam(other), cp=None)
    assert other.column_chunk is None and receipt["applied"] is False


@pytest.mark.parametrize("mode,budget,expected", [
    ("pool", "9000", 9000 * 2**20), ("pool", "200000", 96 * GIB),
    ("exclusive", "9000", 96 * GIB), ("pool", "0", 96 * GIB),
])
def test_widening_uses_pool_reservation(monkeypatch, mode, budget, expected):
    monkeypatch.setenv("CARD_QUEUE_MODE", mode)
    monkeypatch.setenv("CARD_QUEUE_VRAM_MIB", budget)
    assert rc.chunk_capacity(96 * GIB) == expected


def test_small_pool_share_keeps_engine_width_on_large_card(monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setenv("CARD_QUEUE_MODE", "pool")
    monkeypatch.setenv("CARD_QUEUE_VRAM_MIB", "9000")
    monkeypatch.setenv("WOOF_HEX_RADIATION_CHUNK", "auto")
    monkeypatch.setitem(sys.modules, "woof.core.rrtmg_legacy", SimpleNamespace(
        legacy_radiation_vram_bytes=lambda **kwargs: _price(kwargs.get("column_chunk", 2816))))
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda: SimpleNamespace(mem_info=(90 * GIB, 96 * GIB))))
    adapter = RRTMGLegacyRadiation()
    result = rc.widen_radiation_chunk(_Seam(adapter), cp=cp)
    assert result["applied"] is False and adapter.column_chunk is None
    assert result["chunk_capacity_bytes"] == 9000 * 2**20
