# SPDX-License-Identifier: Apache-2.0
"""``--init`` on an install without the recorder refuses before anything is fetched.

THE BREAKAGE (a fresh install of the 0.6.0 wheel with the quick-start extra, 2026-11-11): ``--init`` on a card of the
right class downloaded the row and the plan (942 MB, 2 requests) and only then stopped in the recorder on "CUPTI is
not installed". The recorder (CUPTI) comes with the ``ad`` extra; whether it is there needs no asset to know.
"""
import argparse

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets, rows
from woof.hex.ad.record import env as record_env


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def requests(monkeypatch, tmp_path):
    """A card that fits (class 12.0, 97 GB), a runner, an empty row cache; every download request is counted."""
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    made = []

    def asset_root(cache=None):
        made.append("asset_root")
        raise AssertionError("the asset root was resolved (a first use downloads ~940 MB)")

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: made.append("_fetch"))
    for name in ("WOOF_HEX_AD_ASSETS", rows.IMAGE_BUILDER_ENV, d.RECORDER_ENV, "HEXAD_CUDA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(rows.ROWS_ENV, str(tmp_path / "rows"))
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 500000)   # a host that fits the 120 km rows
    (tmp_path / "init.nc").write_bytes(b"")
    return made


def _toolkit(tmp_path, cupti: bool):
    """A CUDA toolkit folder as the wheels lay it out, with or without CUPTI."""
    root = tmp_path / "nvidia" / "cu13"
    (root / "lib").mkdir(parents=True, exist_ok=True)
    (root / "include").mkdir(exist_ok=True)
    if cupti:
        (root / "lib" / "libcupti.so.13").write_bytes(b"")
        (root / "include" / "cupti.h").write_bytes(b"")
    return root


@pytest.mark.parametrize("door", ["tangent", "adjoint-test", "sensitivity", "param-grad"])
def test_init_without_the_recorder_makes_no_request(requests, tmp_path, monkeypatch, capsys, door):
    monkeypatch.setattr(record_env, "cuda_root", lambda: _toolkit(tmp_path, cupti=False))
    args = _parse([door, "--init", str(tmp_path / "init.nc"), "--out", str(tmp_path / "out")])
    with pytest.raises(d.AdDoorError, match=r"--init needs the recorder, and this install cannot run it: CUPTI is "
                                            r"not installed .*recast-woof\[ad\].*Nothing was downloaded"):
        d.Context(args, args.command)
    assert requests == []                            # 0 requests
    assert not (tmp_path / "out").exists()           # and nothing written
    assert "DEFECT" not in capsys.readouterr().err   # the door speaks, not the recorder


def test_init_without_any_cuda_toolkit_makes_no_request(requests, tmp_path, monkeypatch):
    monkeypatch.setattr(record_env, "purelib", lambda: tmp_path / "empty")
    monkeypatch.setattr(record_env.sys, "path", [])
    args = _parse(["tangent", "--init", str(tmp_path / "init.nc"), "--out", str(tmp_path / "out")])
    with pytest.raises(d.AdDoorError, match="--init needs the recorder.*no CUDA toolkit wheel.*Nothing was downloaded"):
        d.Context(args, "tangent")
    assert requests == []


def test_init_and_plan_together_make_no_request(requests, tmp_path):
    args = _parse(["tangent", "--init", str(tmp_path / "init.nc"), "--plan", str(tmp_path / "plan"),
                   "--out", str(tmp_path / "out")])
    with pytest.raises(d.AdDoorError, match="--plan and --init are alternatives"):
        d.Context(args, "tangent")
    assert requests == []


def test_with_the_recorder_the_door_goes_on_to_the_asset_root(requests, tmp_path, monkeypatch):
    """The refusal is the missing recorder and nothing else: with CUPTI there, the next thing asked is the assets."""
    monkeypatch.setattr(record_env, "cuda_root", lambda: _toolkit(tmp_path, cupti=True))
    args = _parse(["tangent", "--init", str(tmp_path / "init.nc"), "--out", str(tmp_path / "out")])
    with pytest.raises(AssertionError, match="the asset root was resolved"):
        d.Context(args, "tangent")
    assert requests == ["asset_root"]


def test_a_recorder_named_by_command_is_not_judged(requests, tmp_path, monkeypatch):
    monkeypatch.setattr(record_env, "cuda_root", lambda: _toolkit(tmp_path, cupti=False))
    monkeypatch.setenv(d.RECORDER_ENV, "my-recorder --out {out}")
    args = _parse(["tangent", "--init", str(tmp_path / "init.nc"), "--out", str(tmp_path / "out")])
    with pytest.raises(AssertionError, match="the asset root was resolved"):
        d.Context(args, "tangent")
    assert requests == ["asset_root"]


def test_the_quick_start_without_init_is_not_asked_for_a_recorder(requests, tmp_path, monkeypatch):
    monkeypatch.setattr(record_env, "cuda_root", lambda: _toolkit(tmp_path, cupti=False))
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)   # an install that holds the plan's libraries
    args = _parse(["tangent", "--out", str(tmp_path / "out")])
    with pytest.raises(AssertionError, match="the asset root was resolved"):
        d.Context(args, "tangent")
    assert requests == ["asset_root"]
