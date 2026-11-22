"""A kernel-pack row names no folder of the machine that built it.

THE BREAKAGE THIS PREVENTS (0.6.2's check): three published packs carried the builder's home folder in the
raw-route rows' compile evidence (``-I/home/<builder>/.../site-packages/cupy/_core/include``), and a run receipt
copies that evidence. The builder now keeps the flags without include folders, as the NVRTC capture does.
"""
from __future__ import annotations

import json

from woof.hex import assets
from woof.hex.cuda_backend.aot_build import without_include_paths

import pytest


def test_reused_row_loses_include_folders_and_counts_them():
    row = {"module_key": "nvrtc.raw.x", "effective_compile": {"status": "resolved", "observations": [
        {"source_sha256": "s", "effective_flags": ["--std=c++17", "-I/home/someone/venv/site-packages/cupy/_core/include",
                                                   "--fmad=true", "--gpu-architecture=sm_120"],
         "include_path_count": 0, "compiled_image": {"sha256": "c"}}]}}
    cleaned = without_include_paths(row)
    observation = cleaned["effective_compile"]["observations"][0]
    assert observation["effective_flags"] == ["--std=c++17", "--fmad=true", "--gpu-architecture=sm_120"]
    assert observation["include_path_count"] == 1
    assert "/home/" not in json.dumps(cleaned)
    assert without_include_paths(cleaned) == cleaned


@pytest.mark.parametrize("name", ["kernel-pack-cuda12.json", "kernel-pack-cuda13.0.json", "kernel-pack-cuda13.2.json",
                                  "kernel-pack-cuda13.4.59.json", "kernel-pack-cuda13.4.json", "kernel-pack-cuda13.json"])
def test_shipped_pack_rows_name_no_home_folder(name):
    pin = assets.load_pins()["assets"][name]
    path = assets.checkout_asset(name) or assets.asset_path(name, pin)
    if not path.is_file():
        pytest.skip("optional release-asset check needs staged or private checkout packs")
    assets.verify(path, pin)
    document = json.loads(path.read_text())
    for entry in document["entries"]:
        evidence = json.dumps(entry["arguments"].get("effective_compile"))
        assert "/home/" not in evidence and "/Users/" not in evidence, entry["arguments"]["module_key"]
