"""YSU launches take every column up to the workspace byte budget (not a fixed 128-column tile)."""
import pytest

from woof.hex.mpasphys import ysu_v842 as ysu


def test_small_mesh_runs_in_one_launch():
    assert ysu.workspace_tile(55, 2562) >= 2562


def test_large_mesh_is_capped_by_the_budget_in_whole_warps(monkeypatch):
    monkeypatch.delenv("WOOF_HEX_YSU_WORKSPACE_MIB", raising=False)
    tile = ysu.workspace_tile(55, 149322)
    assert tile % 32 == 0 and 128 < tile < 149322
    assert tile * ysu.WORKSPACE_ARRAYS * 57 * 4 <= ysu.WORKSPACE_BUDGET_MIB << 20


def test_override(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_YSU_WORKSPACE_MIB", "8")
    assert ysu.workspace_tile(55, 149322) == ((8 << 20) // (ysu.WORKSPACE_ARRAYS * 57 * 4)) // 32 * 32
    monkeypatch.setenv("WOOF_HEX_YSU_WORKSPACE_MIB", "0")
    with pytest.raises(ValueError):
        ysu.workspace_tile(55, 10)
