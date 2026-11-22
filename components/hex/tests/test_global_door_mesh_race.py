"""Two global runs started at once on an empty cache both fetch the pinned mesh archive.

THE BREAKAGE THIS PREVENTS (hex 0.5.0 both-routes matrix, 2026-11-09): the engine's fetcher never replaces existing
bytes, so the run that finished second stopped in its mesh stage ("could not publish x1.40962.tar.gz without replacing
existing bytes: File exists (os error 17)") although the pinned archive was in place.
"""
import hashlib
import sys
import threading
import time
import types

import pytest

from woof.hex import global_door


class _Status:
    def __init__(self):
        self.notes = []

    def stage(self, *args, **kwargs): pass
    def skip(self, *args, **kwargs): pass
    def step(self, *args, **kwargs): pass
    def done(self, *args, **kwargs): pass

    def note(self, text):
        self.notes.append(text)


def _engine(monkeypatch, on_asset, calls):
    class GeogFetchError(RuntimeError):
        pass

    def rw_asset(action, **kwargs):
        calls.append(action)
        if action == "asset":
            return on_asset(GeogFetchError, **kwargs)
        raise GeogFetchError("stop after the fetch")     # the unpack is not what this test is about

    monkeypatch.setitem(sys.modules, "woof.table_assets", types.SimpleNamespace(_rw_asset=rw_asset))
    monkeypatch.setitem(sys.modules, "woof.geog_assets", types.SimpleNamespace(GeogFetchError=GeogFetchError))
    monkeypatch.setattr(global_door.shutil, "which", lambda name: "/usr/bin/" + name)


def _value(tmp_path, payload):
    return {"mesh": {"id": "x1.test", "url": "https://example.invalid/x1.test.tar.gz", "grid_member": "x1.test.grid.nc",
                     "archive_sha256": hashlib.sha256(payload).hexdigest(), "grid_sha256": "0" * 64,
                     "archive_bytes": len(payload)},
            "paths": {"grid": str(tmp_path / "x1.test.grid.nc")}}


def test_an_archive_another_run_published_first_is_used(tmp_path, monkeypatch):
    payload, calls, status = b"the pinned archive", [], _Status()
    value = _value(tmp_path, payload)

    def lost_the_race(error, *, output, **_):
        output.write_bytes(payload)                      # the other run's publish
        raise error(f"rw_fetch asset: could not publish {output} without replacing existing bytes: File exists")

    _engine(monkeypatch, lost_the_race, calls)
    with pytest.raises(global_door.GlobalDoorError, match="stop after the fetch"):
        global_door.stage_mesh(value, status)
    assert calls == ["asset", "unpack"]                  # the stage went on past the fetch
    assert any("fetched by another run" in note for note in status.notes)


def test_a_failed_fetch_with_other_bytes_in_place_still_stops(tmp_path, monkeypatch):
    payload, calls = b"the pinned archive", []
    value = _value(tmp_path, payload)

    def other_bytes(error, *, output, **_):
        output.write_bytes(b"something else")
        raise error(f"rw_fetch asset: could not publish {output} without replacing existing bytes: File exists")

    _engine(monkeypatch, other_bytes, calls)
    with pytest.raises(global_door.GlobalDoorError, match="without replacing existing bytes"):
        global_door.stage_mesh(value, _Status())
    assert calls == ["asset"]


def test_two_runs_started_together_on_an_empty_cache_both_get_the_grid(tmp_path, monkeypatch):
    """The exact rerun of the matrix case (2026-11-09T22:27Z): with the lost publish forgiven, the second run stopped
    on FileNotFoundError because the first had unpacked the grid and removed the archive under it."""
    payload, grid_bytes, calls = b"the pinned archive", b"the pinned grid", []
    value = _value(tmp_path, payload)
    value["mesh"]["grid_sha256"] = hashlib.sha256(grid_bytes).hexdigest()

    class GeogFetchError(RuntimeError):
        pass

    def rw_asset(action, **kwargs):
        calls.append(action)
        if action == "asset":
            time.sleep(0.2)                              # both runs are inside their fetch at the same time
            if kwargs["output"].exists():
                raise GeogFetchError(f"rw_fetch asset: could not publish {kwargs['output']} without replacing "
                                     f"existing bytes: File exists")
            kwargs["output"].write_bytes(payload)
        else:
            time.sleep(0.2)
            kwargs["output"].mkdir(parents=True)
            (kwargs["output"] / kwargs["datasets"]).write_bytes(grid_bytes)

    monkeypatch.setitem(sys.modules, "woof.table_assets", types.SimpleNamespace(_rw_asset=rw_asset))
    monkeypatch.setitem(sys.modules, "woof.geog_assets", types.SimpleNamespace(GeogFetchError=GeogFetchError))
    monkeypatch.setattr(global_door.shutil, "which", lambda name: "/usr/bin/" + name)
    results, errors = [], []

    def run():
        try:
            results.append(global_door.stage_mesh(value, _Status()))
        except BaseException as error:                   # noqa: BLE001 - the test reports whatever stopped a run
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    assert sorted(result["cached"] for result in results) == [False, True]
    assert calls == ["asset", "unpack"]                  # one fetch, one unpack; the other run used the grid
    assert (tmp_path / "x1.test.grid.nc").read_bytes() == grid_bytes
