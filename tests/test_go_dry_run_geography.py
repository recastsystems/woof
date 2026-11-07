"""``woof go --dry-run`` refuses an unusable WPS_GEOG tree, as the launch does.

The breakage: the dry run claims to validate the route, and both of its
branches (the prepared stage composer the GFS route takes, and the
registered launch the HRRR and ICON routes take) printed the plan and
exited 0 for ``--geog-root`` naming a directory that does not exist, while
the real launch of the same command refused with ``geography_refusal``
("the staged WPS_GEOG tree is not usable") before its fetch.  Measured on
the 2.8.8 candidate on test host 1 with a GFS config: dry run rc 0, launch
refused.
"""
from __future__ import annotations

import pytest

from woof import go_cli
from woof.cli import main

#: The real check, taken before tests/conftest.py's autouse pin answers
#: the default root as staged, for the default-root negative control.
REAL_GEOGRAPHY_REFUSAL = go_cli.geography_refusal

_POINT = {"gfs": "--point=39,-98", "hrrr": "--point=39,-98",
          "icon-eu": "--point=50.1,8.7"}


def _emit(tmp_path, source):
    config = tmp_path / f"{source} area.toml"
    assert main(["domain", _POINT[source], "--card", "16gb", "--ladder", "12",
                 "--source", source, "--cycle", "2026-08-18T06",
                 "--hours", "3", "--out", str(config)]) == 0
    return config


def _staged(root):
    # The shared stand-in stages each container archive child by child
    # (soilgrids/texture_top carries the default soil categories).
    from _staged_geog import staged_geog_tree

    return staged_geog_tree(root / "WPS_GEOG")


@pytest.mark.parametrize("source", ["gfs", "hrrr", "icon-eu"])
def test_the_dry_run_refuses_a_geog_root_that_does_not_exist(
        tmp_path, capsys, source):
    config = _emit(tmp_path, source)
    capsys.readouterr()
    missing = tmp_path / "never-fetched" / "WPS_GEOG"
    rc = main(["go", str(config), "--dry-run", "--geog-root", str(missing),
               "--outdir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert rc == 2, captured.out + captured.err
    # Either refusal names the missing tree and the fetch remedy. A source
    # whose dry run validates the selected static fields first (2.9.0,
    # box/hrrr-retime caf6501f5: the HRRR and ICON-EU recipes) names the
    # exact indexes it lacks, which says more than the generic tree line.
    assert ("the staged WPS_GEOG tree is not usable" in captured.err
            or "require missing geography indexes in" in captured.err), captured.err
    assert str(missing) in captured.err
    assert "woof fetch-geog" in captured.err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("source", ["gfs", "hrrr", "icon-eu"])
def test_the_dry_run_passes_a_staged_geog_root(tmp_path, capsys, source):
    """Non-vacuity: the same command over a tree shaped like a real one."""
    config = _emit(tmp_path, source)
    geog = _staged(tmp_path / "geog")
    rc = main(["go", str(config), "--dry-run", "--geog-root", str(geog),
               "--outdir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "WPS_GEOG tree is not usable" not in captured.err


def test_the_dry_run_refuses_an_absent_default_root(
        tmp_path, capsys, monkeypatch):
    """The default root is asked too; the suite's pin is put back here."""
    config = _emit(tmp_path, "gfs")
    # Domain creation now checks geography too. Restore the real check
    # after creating the input so this test reaches the dry-run refusal.
    monkeypatch.setattr(go_cli, "geography_refusal", REAL_GEOGRAPHY_REFUSAL)
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    capsys.readouterr()
    rc = main(["go", str(config), "--dry-run",
               "--outdir", str(tmp_path / "out")])
    captured = capsys.readouterr()
    assert rc == 2, captured.out + captured.err
    assert "the staged WPS_GEOG tree is not usable" in captured.err
    assert str(tmp_path / "case-data" / "WPS_GEOG") in captured.err
