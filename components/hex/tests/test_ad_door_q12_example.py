# SPDX-License-Identifier: Apache-2.0
"""The 12 km cut of the storm box: the adjoint example a 32 GB card runs, named first in the README.

THE BREAKAGE (the dist47 checker's fresh install, 2026-11-15): no home card and no 48 GB card could run any shipped
adjoint example (the 120 km rows need 73,156 and 80,000 MiB free, the 3 km storm box 96,000 MiB), so nothing a user
types had run on a card they own. The 12 km cut had run the release's own adjoint-test on an RTX 5090 (32 GB) through
a stand-in of the door's context; it was not a row of the table and its files were not a download.
"""
import argparse
import re
from pathlib import Path

import pytest

from woof.hex import ad_doors as d
from woof.hex.ad import assets

Q12 = "hrrrv5-q12.96.2890.n40.60w91.30"
STORM = "hrrrv5-q3.96.38278.n40.60w91.30"
RTX5090_MIB = 32607   # nvidia-smi memory.total of an RTX 5090


def _parse(argv):
    parser = argparse.ArgumentParser()
    d.add_ad_parsers(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.fixture
def card32(monkeypatch, tmp_path):
    """An RTX 5090 with nothing else on it, a host and a disk that fit, and an asset root that must not be resolved."""
    for name in d.NO_LOCAL_GPU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("WOOF_HEX_AD_ASSETS", raising=False)
    monkeypatch.setenv("WOOF_HEX_AD_SESSIONS", str(tmp_path / "sessions"))
    fetched = []

    def asset_root(cache=None, want=None):
        fetched.append(sorted(assets.wanted(assets.load_pins(), want)))
        raise AssertionError("the asset root was resolved")

    monkeypatch.setattr(assets, "asset_root", asset_root)
    monkeypatch.setattr(d, "resolve_runner", lambda explicit: tmp_path / "runner")
    monkeypatch.setattr(d, "compute_capability", lambda card: "12.0")
    monkeypatch.setattr(d, "total_mib", lambda card: RTX5090_MIB)
    monkeypatch.setattr(d, "free_mib", lambda card: RTX5090_MIB - 520)   # the driver's own share
    monkeypatch.setattr(d, "host_available_mib", lambda meminfo=None: 60_000)
    monkeypatch.setattr(d, "_free_disk_mib", lambda folder: 200_000)
    monkeypatch.setattr(d, "refuse_plan_libraries_missing", lambda *a, **k: None)
    return fetched


def test_the_12km_example_runs_on_a_32gb_card_and_fetches_only_its_own_files(card32, tmp_path, capsys):
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(["adjoint-test", "--row", Q12, "--steps", "15", "--out", str(tmp_path / "o")]),
                  "adjoint-test")
    [fetched] = card32
    # the default boundary-layer route's five images (one pinned file) come with it: adjoint-test checks that route
    assert fetched == ["ad-aux-bl-closure-sm120-v1.tar.gz", "ad-kset-storm-bs5-lean2-astra2-dyn6-v1.tar.gz",
                       "ad-mesh-q12.96.2890.n40.60w91.30-p1-v1.tar.gz",
                       "ad-plan-hrrrv5-q12.96.2890.n40.60w91.30-20240521T15-p1-v1.tar.gz", "ad-row-hrrrv5-any-v6.tar.gz"]
    # the exact route of the held form fetches only the row's own files
    with pytest.raises(AssertionError, match="asset root was resolved"):
        d.Context(_parse(["adjoint-test", "--row", Q12, "--steps", "15", "--boundary-layer", "taped",
                          "--out", str(tmp_path / "t")]), "adjoint-test")
    assert card32[1] == [f for f in fetched if f != "ad-aux-bl-closure-sm120-v1.tar.gz"]
    # the same card refuses the 3 km storm box and the 120 km example before any download, as before
    for argv in (["--row", STORM, "--steps", "3"], ["--steps", "3"]):
        with pytest.raises(d.AdDoorError, match="more than the whole card"):
            d.Context(_parse(["adjoint-test", *argv, "--out", str(tmp_path / "p")]), "adjoint-test")
    assert len(card32) == 2


def test_the_rows_words_are_the_measured_ones_and_its_needs_hold_the_measured_peaks():
    table = d.load_table()
    row, storm = table["rows"][Q12], table["rows"][STORM]
    # 59 levels: the recorded plan's (sensitivity.nc of the shipped example, nVertLevels 59); the row said 55 until
    # a review of 0.6.0 read 59 (2026-11-15), and --level 56 to 59 were refused
    assert (row["cells"], row["edges"], row["levels"], row["dt_s"]) == (2890, 8834, 59, 60.0)
    assert row["example_plan_steps"] == row["default_steps"] == 15
    assert row["example_plan_start"] == "2024-05-21T15:00:00"
    # the 3 km row's derivative files, on the cut's own mesh
    for key in ("kernels", "cubins", "bundle", "schemes", "tempo_cubin", "tempo_block_cubin", "tempo_tlf_cubin"):
        assert row["assets"][key] == storm["assets"][key], key
    assert row["assets"]["grid"] == "meshes/q12.96.2890.n40.60w91.30.grid.nc"
    peak = int(re.search(r"card peak ([\d,]+) MiB", row["need_free_mib_source"]).group(1).replace(",", ""))
    host = int(re.search(r"host peak ([\d,]+) MiB", row["need_host_source"]).group(1).replace(",", ""))
    assert peak < row["need_free_mib"] <= RTX5090_MIB - 520 and host < row["need_host_mib"]
    assert "param-grad" not in row["doors"] and not row.get("gradient_line")


def test_its_two_downloads_are_pinned_and_hold_its_mesh_and_plan():
    pins = assets.load_pins()["assets"]
    row = d.load_table()["rows"][Q12]
    mesh = pins["ad-mesh-q12.96.2890.n40.60w91.30-p1-v1.tar.gz"]
    plan = pins["ad-plan-hrrrv5-q12.96.2890.n40.60w91.30-20240521T15-p1-v1.tar.gz"]
    assert (mesh["kind"], mesh["holds"]) == ("ad-mesh", [row["assets"]["grid"]])
    assert (plan["kind"], plan["holds"]) == ("ad-plan", [row["example_plan"] + "/"])
    assert "UGWP" in plan["what"] and len(mesh["sha256"]) == len(plan["sha256"]) == 64


def test_the_readme_names_the_12km_example_first_with_its_card():
    readme = ((Path(d.__file__).resolve().parents[2] / "components" / "hex") / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("## The differentiable engine"):]
    first = re.search(r"```bash\n(.*?)```", section, re.S).group(1)
    assert f"--row {Q12}" in first and "--steps 15" in first
    assert "RTX 5090" in section[:section.index("```bash")]
