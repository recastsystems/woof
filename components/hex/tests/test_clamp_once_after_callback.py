# SPDX-License-Identifier: Apache-2.0
"""Negative scalars are clamped once a step, before microphysics, as native does.

Native MPAS clamps negative mixing ratios once, at mpas_atm_time_integration.F:2818,
before microphysics. In hex a post-transport physics callback runs that clamp (and
then TEMPO) itself. THE BREAKAGE THIS PREVENTS: cuda_driver.py clamped the regional
scalars a second time after the callback, so after TEMPO (model audit D8, 15 Nov:
the recording ran the clamp at event 8136, after tempo_to at 8121). A clamp there
can zero words native keeps and adds one kernel a step. cuda_driver_rrfs.py already
skipped it. Every regional clamp call in both drivers must be guarded by the callback.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / ".." / ".." / "woof" / "hex"


def _guarded_clamps(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    rows = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "clamp_negative_scalars"):
            continue
        owner = node.func.value
        if not (isinstance(owner, ast.Attribute) and owner.attr == "regional_v841"
                and isinstance(owner.value, ast.Name) and owner.value.id == "self"):
            continue
        guarded, up = False, parents.get(node)
        while up is not None and not isinstance(up, ast.FunctionDef):
            if isinstance(up, ast.If) and "physics_post_transport_callback is None" in ast.unparse(up.test):
                guarded = True
            up = parents.get(up)
        rows.append((node.lineno, guarded))
    return rows


@pytest.mark.parametrize("name", ["cuda_driver.py", "cuda_driver_rrfs.py"])
def test_regional_clamp_skipped_when_a_callback_ran(name):
    rows = _guarded_clamps(SRC / name)
    assert rows, f"{name}: no self.regional_v841.clamp_negative_scalars call found"
    bad = [line for line, ok in rows if not ok]
    assert not bad, (f"{name}: lines {bad} clamp the regional scalars without "
                     "'physics_post_transport_callback is None': a second clamp after TEMPO")


def test_the_check_sees_an_unguarded_clamp(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("class D:\n def step(self, physics_post_transport_callback=None):\n"
                 "  if self.config.config_moist_physics:\n"
                 "   self.regional_v841.clamp_negative_scalars(s)\n"
                 "  if physics_post_transport_callback is None:\n"
                 "   self.regional_v841.clamp_negative_scalars(s)\n", encoding="utf-8")
    assert _guarded_clamps(p) == [(4, False), (6, True)]
