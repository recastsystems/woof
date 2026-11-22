# SPDX-License-Identifier: Apache-2.0
"""woof.hex.ad.assets: the operator-row files are pinned, fetched once from an offline directory or the release,
verified before unpacking, and reused (host-only; no network)."""
import hashlib
import io
import json
import tarfile

import pytest

from woof.hex.ad import assets


def _archive(path, files):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "kind": "ad-row"}


def _pins(tmp_path, monkeypatch, rows):
    pins = tmp_path / "ad-assets.json"
    pins.write_text(json.dumps({"schema": "gpuwm-hex.on-demand-assets/v1", "assets": rows,
                                "release_assets": {"repository": "example/hex", "tag": "ad-assets-test"}}))
    monkeypatch.setattr(assets, "PINS", pins)
    monkeypatch.setenv("WOOF_HEX_CACHE_ROOT", str(tmp_path / "cache"))


def test_shipped_pins_are_valid_and_name_the_row_and_plan():
    pins = assets.load_pins()
    kinds = sorted(r["kind"] for r in pins["assets"].values())
    # the 120 km example (mesh, plan in two parts), the 3 km storm example (mesh, plan in two parts, its start),
    # the storm box's 6 km gradient mesh, rows; the storm row's drag image, the two images of its tape-free
    # boundary layer (one file) and its kernel set; from dist44 the radiation step's two rewritten images, the
    # cloud scheme's gated derivative images and the default route's kernel set
    # and the three images of the storm row's guarded boundary-layer route (one file); from dist48 the 12 km
    # example (its mesh and its plan in one part); from 0.6.0 the 3 km row's number cloud images (one file),
    # the default route's 3 km set with three double-arithmetic dynamics derivative images (the exact route keeps
    # the single-arithmetic set) and (F10-full) the full-physics rows' floored boundary-layer images (one file)
    assert kinds == ["ad-aux", "ad-aux", "ad-aux", "ad-aux", "ad-aux", "ad-aux", "ad-aux", "ad-case", "ad-gradient-mesh",
                     "ad-kset", "ad-kset", "ad-kset", "ad-mesh", "ad-mesh", "ad-mesh", "ad-plan", "ad-plan", "ad-plan",
                     "ad-plan", "ad-plan",
                     "ad-row"]
    assert pins["release_assets"]["tag"].startswith("ad-assets-")


def test_fetch_from_directory_unpacks_once_and_reuses(tmp_path, monkeypatch):
    src = tmp_path / "from"
    src.mkdir()
    row = _archive(src / "ad-row-x-v1.tar.gz", {"rows/x/params.json": b"{}"})
    plan = _archive(src / "ad-plan-x-v1.tar.gz", {"plans/p/plan.json": b"{}"})
    _pins(tmp_path, monkeypatch, {"ad-row-x-v1.tar.gz": row, "ad-plan-x-v1.tar.gz": plan})
    monkeypatch.setenv("WOOF_HEX_AD_ASSETS_FROM", str(src))
    root = assets.asset_root()
    assert (root / "rows/x/params.json").is_file() and (root / "plans/p/plan.json").is_file()
    for f in src.iterdir():
        f.unlink()   # the second call needs no source at all
    assert assets.asset_root() == root


def test_a_file_that_differs_from_its_pin_is_refused_and_not_unpacked(tmp_path, monkeypatch):
    src = tmp_path / "from"
    src.mkdir()
    row = _archive(src / "ad-row-x-v1.tar.gz", {"rows/x/params.json": b"{}"})
    row["sha256"] = "0" * 64
    _pins(tmp_path, monkeypatch, {"ad-row-x-v1.tar.gz": row})
    monkeypatch.setenv("WOOF_HEX_AD_ASSETS_FROM", str(src))
    with pytest.raises(assets.AdAssetError):
        assets.asset_root()
    assert not (tmp_path / "cache" / "ad").exists() or not any((tmp_path / "cache" / "ad").glob("*/rows"))


def test_mirror_url_overrides_the_release(monkeypatch):
    pins = {"release_assets": {"repository": "example/hex", "tag": "ad-assets-test"}}
    assert assets.asset_url("a.tar.gz", pins) == "https://github.com/example/hex/releases/download/ad-assets-test/a.tar.gz"
    monkeypatch.setenv("WOOF_HEX_AD_ASSETS_URL", "https://mirror.example/hex/")
    assert assets.asset_url("a.tar.gz", pins) == "https://mirror.example/hex/a.tar.gz"
