"""The observation-grid lane, finally reached from a door.

``rw_obsgrid`` renders five products from a ``gpuwm-obs.radar-grid.v1``
observation volume and has existed, wrapped and handshaken, with no
caller at all: the two DA render tools printed it as a HINT and drew the
same fields in matplotlib.  These cases stub the binary, so they run with
no build.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from woof import rustwx_lanes


def test_the_lane_resolves_its_engine_through_the_shared_contract(monkeypatch):
    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin", lambda: None)
    with pytest.raises(RuntimeError) as excinfo:
        rustwx_lanes.resolve_obsgrid_engine("rust")
    message = str(excinfo.value)
    assert rustwx_lanes.OBSGRID_NAME in message
    assert rustwx_lanes.CARGO_BUILD_HINT.split()[0] in message
    with pytest.raises(RuntimeError):
        rustwx_lanes.resolve_obsgrid_engine("auto")
    assert rustwx_lanes.resolve_obsgrid_engine("matplotlib") == (
        "matplotlib", "requested")


def _stage_engine(monkeypatch, tmp_path, written):
    monkeypatch.setattr(rustwx_lanes, "find_obsgrid_bin",
                        lambda: tmp_path / "rw_obsgrid")
    monkeypatch.setattr(rustwx_lanes, "probe_obsgrid_bin",
                        lambda path: (True, "--abi matches the contract"))

    def fake_run(engine, obs, *, out_dir, **kwargs):
        paths = []
        for slug in rustwx_lanes.OBSGRID_PRODUCTS:
            path = Path(out_dir) / f"{slug}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes.fromhex("89504e470d0a1a0a"))
            paths.append(path)
        written.extend(paths)
        return paths, [], [], []

    monkeypatch.setattr(rustwx_lanes, "run_obsgrid_renderer", fake_run)

