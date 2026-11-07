"""A rank of a split mesh checks the columns it owns; its outermost halo ring is not its to refuse.

THE BREAKAGE THIS PREVENTS (hex 1 km, 300 km core, Oklahoma, 399,875 cells, four cards, GFS 2026-10-31 12Z,
run_37b3a1671fb8a034, 5 of 5): every bad word of rank 3 at the refused step sat in ring 3 of its halo, columns it does
not own and whose words its neighbours' are about to replace, while one card ran the same mesh clean. The forecast
stopped at step 1 on "CUDA validation flag".
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from woof.hex.cuda_driver import CudaDryDycoreDriver as Driver

SOURCE = Path(__import__("woof.hex.cuda_driver").hex.cuda_driver.__file__).parent


def _driver(exchanger):
    return SimpleNamespace(
        halo_exchanger_v841=exchanger,
        cp=np,
        _rank_owned_extents_v841=lambda: Driver._rank_owned_extents_v841(SimpleNamespace(halo_exchanger_v841=exchanger)),
    )


def test_one_card_keeps_its_own_word():
    flag = np.zeros((1,), dtype=np.int32)
    assert Driver._column_flag_v841(_driver(None), flag) is flag


def test_a_class_executor_with_no_ownership_tables_keeps_its_own_word():
    flag = np.zeros((1,), dtype=np.int32)
    assert Driver._column_flag_v841(_driver(SimpleNamespace()), flag) is flag


def test_a_rank_sends_the_local_extent_kernels_to_a_scratch_word():
    flag = np.zeros((1,), dtype=np.int32)
    tables = SimpleNamespace(n_owned_cells=99_979, n_owned_edges=300_000)
    driver = _driver(SimpleNamespace(tables=tables))
    assert driver._rank_owned_extents_v841() == (99_979, 300_000)
    scratch = Driver._column_flag_v841(driver, flag)
    assert scratch is not flag and scratch.shape == (1,) and scratch.dtype == np.int32
    assert Driver._column_flag_v841(driver, flag) is scratch


def _calls(path: Path, method: str, name: str) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno
        for parent in ast.walk(tree)
        if isinstance(parent, ast.FunctionDef) and parent.name == method
        for node in ast.walk(parent)
        if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == name
    ]


def test_the_step_routes_every_whole_extent_state_test_through_the_rank_aware_helpers():
    for module in ("cuda_driver.py", "cuda_driver_rrfs.py"):
        path = SOURCE / module
        for method in ("_advance_dynamics_subcycle_v841", "_step_device_v841"):
            assert not _calls(path, method, "validate_recovered_state_cuda_v841"), (module, method)
            assert not _calls(path, method, "accumulate_finite_array_cuda_v841"), (module, method)
        assert _calls(path, "_advance_dynamics_subcycle_v841", "_validate_recovered_stage_v841"), module
