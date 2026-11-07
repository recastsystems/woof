"""Strict numerics and other P3 variants must never select fast MP52 math."""
from types import SimpleNamespace

import pytest

from woof import numerics
from woof.core import p3_device as pd


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("n_cat", [1, 2])
def test_default_math_module_respects_numerics(monkeypatch, strict, n_cat):
    class ModuleSelected(Exception):
        pass

    selected = []

    def select_module(options):
        selected.append(options)
        raise ModuleSelected

    monkeypatch.setattr(numerics, "is_strict", lambda: strict)
    monkeypatch.setattr(pd, "p3_module", select_module)
    with pytest.raises(ModuleSelected):
        pd.run_p3_device({}, {}, {}, dt=12., it=300, n_cat=n_cat,
                         workspace=SimpleNamespace(ncol=1, nk=1),
                         tables=SimpleNamespace(n_cat=n_cat))
    assert ("-DP3_FAST_2CAT=1" in selected[0]) == (n_cat == 2 and not strict)
    assert ("-DP3_NATIVE_GNU239=1" in selected[0]) == (n_cat == 2 and strict)
