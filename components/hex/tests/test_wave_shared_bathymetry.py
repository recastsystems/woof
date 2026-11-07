"""Runs on one host download the wave bathymetry once and share it; the preparation reads that one file.

THE BREAKAGE THIS CHECKS: every run's preparation downloaded its own 491 MB ETOPO copy; two runs started seconds
apart were rate-limited by the server (HTTP 429) and the second stopped ("bathymetry acquisition incomplete").
A stand-in for the Rust stage door records downloads; nothing is fetched.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import wave_prepared  # noqa: E402


def test_concurrent_runs_download_the_bathymetry_once_and_prepare_from_it(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_HEX_WAVES_INPUT_CACHE", str(tmp_path / "cache"))
    downloads, inside, peak, prepared = [], [0], [0], []
    guard = threading.Lock()

    def invoke(request, *, operation):
        document = json.loads(Path(request).read_text())
        assert operation == "stage" and document["operation"] == "stage"
        with guard:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        time.sleep(0.05)
        row = document["sources"][0]
        out = Path(document["output"])
        with open(out / row["path"], "wb") as handle:   # a sparse stand-in of the catalog's byte count
            handle.truncate(int(row["expected_bytes"]))
        with open(out / row["path"], "rb") as handle:    # the door's per-file checksum receipt
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        (out / (row["path"] + ".receipt.json")).write_text(json.dumps({"sha256": digest}))
        (out / "ACQUIRED.json").write_text(json.dumps({"status": "complete"}))
        downloads.append(document["output"])
        with guard:
            inside[0] -= 1
        return out

    data = types.ModuleType("woof.ocean.waves.data")
    data.invoke = invoke
    import woof.ocean.waves.data as real_data
    data.__path__ = real_data.__path__        # importlib.resources reads public_sources.v1.json from here
    data.__spec__ = real_data.__spec__
    monkeypatch.setitem(sys.modules, "woof.ocean.waves.data", data)

    def prepare_mesh(**kwargs):
        prepared.append(kwargs["options"]["shared_inputs"]["bathymetry"])
        raise RuntimeError("stop after preparation")   # the rest of ensure reads real prepared files

    prepared_module = types.ModuleType("woof.ocean.waves.prepared")
    prepared_module.prepare_mesh = prepare_mesh
    monkeypatch.setitem(sys.modules, "woof.ocean.waves.prepared", prepared_module)

    def run(index):
        try:
            wave_prepared.ensure(mesh=None, grid_path=None, start_time=None, end_time=None,
                                 output_root=tmp_path / f"run{index}", options={})
        except RuntimeError:
            pass

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(run, range(3)))
    assert len(downloads) == 1 and peak[0] == 1        # old code: no shared bathymetry at all
    assert len(prepared) == 3 and len(set(prepared)) == 1
    assert Path(prepared[0]).is_file() and str(tmp_path / "cache") in prepared[0]


def test_a_damaged_cached_bathymetry_is_fetched_again_not_reused(tmp_path, monkeypatch):
    """THE BREAKAGE THIS CHECKS: a cached copy was reused on its byte count alone, so a damaged file of the right
    size went into every later preparation. Reuse now needs the sha256 of the stage door's receipt."""
    monkeypatch.setenv("WOOF_HEX_WAVES_INPUT_CACHE", str(tmp_path / "cache"))
    import woof.ocean.waves.data as real_data
    from importlib.resources import files
    row = json.loads(files("woof.ocean.waves.data").joinpath("public_sources.v1.json").read_text())["bathymetry"]
    downloads = []

    def invoke(request, *, operation):
        out = Path(json.loads(Path(request).read_text())["output"])
        with open(out / row["path"], "wb") as handle:
            handle.truncate(int(row["expected_bytes"]))
        with open(out / row["path"], "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        (out / (row["path"] + ".receipt.json")).write_text(json.dumps({"sha256": digest}))
        (out / "ACQUIRED.json").write_text(json.dumps({"status": "complete"}))
        downloads.append(out)
        return out

    data = types.ModuleType("woof.ocean.waves.data")
    data.invoke = invoke
    data.__path__, data.__spec__ = real_data.__path__, real_data.__spec__
    monkeypatch.setitem(sys.modules, "woof.ocean.waves.data", data)
    first = Path(wave_prepared.shared_bathymetry({})["shared_inputs"]["bathymetry"])
    assert len(downloads) == 1
    assert wave_prepared.shared_bathymetry({})["shared_inputs"]["bathymetry"] == str(first) and len(downloads) == 1
    with open(first, "r+b") as handle:     # same size, other bytes
        handle.seek(1000)
        handle.write(b"damaged")
    assert Path(wave_prepared.shared_bathymetry({})["shared_inputs"]["bathymetry"]) == first
    assert len(downloads) == 2                # old code: reused the damaged copy
    with open(first, "rb") as handle:
        handle.seek(1000)
        assert handle.read(7) == bytes(7)
