"""Per-card module caches hold every card's entries (no per-step reloads on many cards).

THE BREAKAGE THIS PREVENTS: ``thompson_v842._module`` was ``lru_cache(maxsize=8)`` keyed by card.  Correct mode
asks for two corrections variants per card, so on 8 cards the 16 keys cycled through 8 slots: half the ranks
rebuilt two Thompson modules every step and the others waited for them at the halo (8x RTX 5090, 286,530-cell
zoom globe, convection_permitting, 2026-10-28: 1.279 s/step on 8 cards against 0.248 s with the cache fixed).
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import types

import woof.hex

_CARD_ARGUMENTS = {"device", "device_id", "card", "devices", "device_ordinal"}


def test_thompson_module_compiles_once_per_card_and_variant(monkeypatch):
    from woof.hex.mpasphys import thompson_v842

    compiles = []

    class _Device:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    stub = types.ModuleType("cupy")
    stub.cuda = types.SimpleNamespace(Device=_Device)
    monkeypatch.setitem(sys.modules, "cupy", stub)
    monkeypatch.setattr(thompson_v842, "cuda_source", lambda **options: repr(sorted(options.items())))
    monkeypatch.setattr(thompson_v842, "compile_module", lambda source: compiles.append(source) or object())
    thompson_v842._module.cache_clear()
    try:
        for _step in range(3):
            for device in range(8):
                # The two variants correct mode asks for on every card (no corrections, and WB3).
                thompson_v842._module(device, "maths", 55)
                thompson_v842._module(device, "maths", 55, ("WB3",))
        assert len(compiles) == 16
    finally:
        thompson_v842._module.cache_clear()


def test_no_bounded_lru_cache_is_keyed_by_card():
    """Any ``lru_cache(maxsize=<int>)`` on a function taking a card argument thrashes once cards x variants pass it."""
    root = Path(woof.hex.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
            if not names & _CARD_ARGUMENTS:
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                target = decorator.func
                label = getattr(target, "attr", getattr(target, "id", ""))
                if label != "lru_cache":
                    continue
                for keyword in decorator.keywords:
                    if (keyword.arg == "maxsize" and isinstance(keyword.value, ast.Constant)
                            and isinstance(keyword.value.value, int)):
                        offenders.append(f"{path.relative_to(root)}:{node.lineno} {node.name} maxsize={keyword.value.value}")
                if decorator.args and isinstance(decorator.args[0], ast.Constant) and isinstance(decorator.args[0].value, int):
                    offenders.append(f"{path.relative_to(root)}:{node.lineno} {node.name} maxsize={decorator.args[0].value}")
    assert not offenders, ("bounded per-card module caches thrash on many cards (thompson_v842._module, 8 cards, "
                           "2026-10-28): " + "; ".join(offenders))
