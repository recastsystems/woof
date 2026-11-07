"""Same-host ranks prepare the wave inputs one at a time, so later ranks reuse the first rank's preparation.

THE BREAKAGE THIS CHECKS: with ``--waves on`` and 2 same-host rank processes, both ranks ran the shared preparation
at once into one directory and the second stopped the run ("bathymetry acquisition incomplete"). A stand-in
preparation records how many callers are inside it at once; nothing is downloaded.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import wave_prepared  # noqa: E402


def test_concurrent_ensure_calls_prepare_one_at_a_time(tmp_path, monkeypatch):
    inside, peak, calls = [0], [0], []
    guard = threading.Lock()

    def prepare_mesh(**kwargs):
        with guard:
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
        time.sleep(0.05)
        calls.append(kwargs["output_root"])
        with guard:
            inside[0] -= 1
        raise RuntimeError("stop after preparation")   # the rest of ensure reads real prepared files

    module = types.ModuleType("woof.ocean.waves.prepared")
    module.prepare_mesh = prepare_mesh
    monkeypatch.setitem(sys.modules, "woof.ocean.waves.prepared", module)
    monkeypatch.setattr(wave_prepared, "shared_bathymetry", lambda options: options)   # no download here

    def one(_):
        try:
            wave_prepared.ensure(mesh=None, grid_path=None, start_time=None, end_time=None,
                                 output_root=tmp_path / "out", options={})
        except RuntimeError:
            pass

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(one, range(4)))
    assert len(calls) == 4
    assert peak[0] == 1   # old code: all 4 inside at once
