"""The KIM statistics builder: the engine's rw_mpas_static when the wheel's bundled one predates the recipe.

THE BREAKAGE THIS GUARDS (hex 0.5.0 on gpuwm 3.0.0, 2026-11-10): the wheel bundles an rw_mpas_static without
--gwd-recipe and the engine ladder reads it first, so the opt-in KIM drag stopped before every forecast although
gpuwm 3.0's own rw_mpas_static carries --gwd-recipe kim2026.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from woof.hex import engines
from woof.hex import kim_gwd_static as kim


def _exe(path: Path, help_text: str) -> Path:
    path.write_text("#!/bin/sh\necho '" + help_text + "'\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.skipif(os.name != "posix", reason="shell stubs")
def test_the_engine_builder_is_used_when_the_bundled_one_lacks_the_recipe(tmp_path, monkeypatch) -> None:
    old = _exe(tmp_path / "bundled_rw_mpas_static", "--grid --out")
    new = _exe(tmp_path / "engine_rw_mpas_static", "--grid --out --gwd-recipe v842|kim2026 --gwd-only")
    monkeypatch.setattr(engines, "resolve", lambda spec, explicit=None: old)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (new,))
    seen = []
    monkeypatch.setattr(kim.subprocess, "run", _recording_run(seen))
    with pytest.raises(kim.KimGwdStaticError, match="exited 9"):
        kim.build(tmp_path / "grid.nc", tmp_path / "out.nc")
    assert seen and seen[-1][0] == str(new) and "--gwd-recipe" in seen[-1]


@pytest.mark.skipif(os.name != "posix", reason="shell stubs")
def test_no_builder_with_the_recipe_is_still_refused_by_name(tmp_path, monkeypatch) -> None:
    old = _exe(tmp_path / "bundled_rw_mpas_static", "--grid --out")
    monkeypatch.setattr(engines, "resolve", lambda spec, explicit=None: old)
    monkeypatch.setattr(engines, "gpuwm_candidates", lambda spec: (old,))
    with pytest.raises(kim.KimGwdStaticError, match="has no --gwd-recipe"):
        kim.build(tmp_path / "grid.nc", tmp_path / "out.nc")


def _recording_run(seen):
    real = kim.subprocess.run

    def run(argv, *args, **kwargs):
        if "--help" in argv:
            return real(argv, *args, **kwargs)
        seen.append(list(map(str, argv)))
        return type("R", (), {"returncode": 9, "stdout": "", "stderr": "stub"})()

    return run
