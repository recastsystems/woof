"""Correct-mode bindings reach both generated WSM6 column implementations."""
import pytest

from woof.hex.mpasphys import wsm6_v842 as wsm6


@pytest.mark.parametrize("mode", ["fast", "strict"])
@pytest.mark.parametrize("cooperative", ["0", "1"])
def test_corrections_precede_source_cloning(monkeypatch, mode, cooperative):
    monkeypatch.setenv("WOOF_HEX_WSM6_COOP", cooperative)
    source = wsm6.cuda_source("// source binding test", max_levels=55,
                             corrections=("WB3", "WB5"), numerics=mode)
    copies = 2 if cooperative == "1" else 1
    for old, new, _ in wsm6.CORRECT_MODE_SOURCE["WB5"]:
        assert old not in source
        assert source.count(new) == copies
    old, new, _ = wsm6.CORRECT_MODE_SOURCE["WB3"][0]
    assert old not in source
    assert source.count(new) == 1


def test_changed_original_source_still_refuses(monkeypatch):
    monkeypatch.setenv("WOOF_HEX_WSM6_COOP", "1")
    old = wsm6.CORRECT_MODE_SOURCE["WB5"][0][0]
    monkeypatch.setattr(wsm6, "CUDA_MODULES", wsm6.CUDA_MODULES.replace(old, "// changed source"))
    with pytest.raises(RuntimeError, match="WB5 occurs 0 times, not 1"):
        wsm6.cuda_source("// source binding test", max_levels=55,
                         corrections=("WB5",), numerics="fast")
