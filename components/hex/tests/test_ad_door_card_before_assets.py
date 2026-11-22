# SPDX-License-Identifier: Apache-2.0
"""A differentiable-engine door checks its card before it resolves the asset root: a door the window admission
would refuse (no readable card, a card smaller than the window, the memory not free) is refused before the first-use
download of the row and plan (about 940 MB). The default assets name the 0.6.0 asset release (host-only)."""
import argparse
import json

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def no_fetch(monkeypatch, tmp_path):
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    fetched = []

    def asset_root(cache=None):
        fetched.append("asset_root")
        raise AssertionError("the asset root was resolved (a first use downloads ~940 MB)")

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(assets, "_fetch", lambda *a, **k: fetched.append("_fetch"))
    monkeypatch.delenv("WOOF_HEX_AD_ASSETS", raising=False)
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 500000)   # a host that fits the 120 km rows
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)   # an install that holds the plan's libraries
    return fetched


def test_an_unreadable_card_is_refused_before_any_download(no_fetch, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "compute_capability", lambda card: None)
    with pytest.raises(d.AdDoorError, match="cannot read card 7 .*before it fetches"):
        d.Context(_parse(["tangent", "--card", "7", "--out", str(tmp_path / "o")]), "tangent")
    assert no_fetch == [] and not (tmp_path / "o").exists()


def test_a_card_without_the_memory_is_refused_before_any_download(no_fetch, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)   # a card the 120 km row fits, held by another job
    monkeypatch.setattr(d, "free_mib", lambda card: 1000)
    with pytest.raises(d.AdDoorError, match="MiB free; this row's window needs"):
        d.Context(_parse(["adjoint-test", "--out", str(tmp_path / "o")]), "adjoint-test")
    monkeypatch.setattr(d, "total_mib", lambda card: 32607)
    with pytest.raises(d.AdDoorError, match="more than the whole card"):          # --wait-card cannot help here
        d.Context(_parse(["sensitivity", "--wait-card", "60", "--need-mib", "40000",
                          "--out", str(tmp_path / "o")]), "sensitivity")
    assert no_fetch == []


def test_a_card_that_fits_goes_on_to_the_assets(no_fetch, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: 97887)
    monkeypatch.setattr(d, "free_mib", lambda card: 97000)
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(["tangent", "--out", str(tmp_path / "o")]), "tangent")
    assert no_fetch == ["asset_root"]


def test_default_assets_name_the_0_5_1_asset_release():
    pins = json.loads(assets.PINS.read_text())
    assert pins["release_assets"] == {"repository": "recastsystems/gpuwm-hex", "tag": "ad-assets-v0.6.0"}
    url = assets.asset_url("ad-row-hrrrv5-x1.2562-v1.tar.gz", pins)
    assert url == ("https://github.com/recastsystems/gpuwm-hex/releases/download/ad-assets-v0.6.0/"
                   "ad-row-hrrrv5-x1.2562-v1.tar.gz")


def test_a_card_that_is_not_there_has_no_compute_capability(monkeypatch):
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")   # card 7 named, so the card may be read

    class Done:
        returncode, stdout = 6, "No devices were found\n"

    monkeypatch.setattr(d.subprocess, "run", lambda *a, **k: Done())
    assert d.compute_capability("7") is None
    Done.returncode, Done.stdout = 0, "12.0\n"
    assert d.compute_capability("0") == "12.0"
