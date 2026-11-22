"""CUDA modules are held per card.

THE BREAKAGE THIS PREVENTS (2026-10-23, woof hex 0.4.3 and 0.4.4): every
two-card same-host forecast stopped at step 1 with CUDA_ERROR_INVALID_HANDLE in
the sea-ice step (arwen_seaice._run_noahmp).  ``seaice_v842._module`` was an
lru_cache on the soil-layer count alone, so the first rank loaded it on card 0
and the second rank was handed card 0's module.  A CUDA module belongs to the
card it was loaded on.

The CPU member reads the source: a cached function that loads a
``function.Module`` must take the card among its arguments.  The GPU member
loads the sea-ice module on card 0 and launches it on card 1.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / ".." / ".." / "woof" / "hex"
CARD_ARGS = {"device", "device_id", "dev", "card"}


def _cached(node):
    return any("cache" in ast.unparse(d) for d in node.decorator_list)


def _loads_module(node):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and ast.unparse(sub.func).endswith("function.Module"):
            return True
    return False


def test_every_cached_module_loader_is_keyed_by_card():
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "function.Module" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and _cached(node) and _loads_module(node):
                args = {a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
                if not args & CARD_ARGS:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} {node.name}")
    assert not offenders, "cached CUDA module loaders without a card key: " + ", ".join(offenders)


@pytest.mark.gpu
def test_seaice_module_launches_on_the_second_card():
    cp = pytest.importorskip("cupy")
    try:
        cards = cp.cuda.runtime.getDeviceCount()
    except Exception:  # noqa: BLE001 - no driver is no card
        cards = 0
    if cards < 2 or os.environ.get("HEX_TWO_CARD_GPU") != "1":
        pytest.skip("needs two CUDA cards to itself and HEX_TWO_CARD_GPU=1")
    import numpy as np
    from woof.hex.mpasphys import seaice_v842 as si

    with cp.cuda.Device(0):
        si._module(4)
    for card in (0, 1):
        with cp.cuda.Device(card):
            last = cp.full(1, 3, dtype=cp.int32)
            state = cp.zeros(2, dtype=cp.int32)
            si._module(4).get_function("seaice_module_state")(
                (1,), (1,), (last, state, np.int32(1), np.int32(1)))
            cp.cuda.Device(card).synchronize()
