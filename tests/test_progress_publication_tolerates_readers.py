"""A progress heartbeat must never kill the forecast it reports on.

MEASURED, 2026-08-17, Windows 11, RTX 3080, gpuwm 2.4.1 wheel: a
`woof-prepared-tree-forecast` 12-3 km run died at outer step 76 of 120
with ``PermissionError: [WinError 5] Access is denied`` inside
``progress_callback`` -> ``_atomic_json`` -> ``os.replace``, because a
reader held ``evidence/progress.json`` open at the instant the runner
republished it.  On Windows a plain ``open()`` for READ denies rename
over the file (no FILE_SHARE_DELETE), so the documented act of watching
the documented progress file -- FIRST-LIGHT.md: "Watch progress at
``<outdir>/evidence/progress.json``" -- can abort a healthy CUDA run.
``woof go``'s own stopwatch heartbeat reads the same file on a 20 s
poll, so the product races itself without any user involved.

The supervisor already ships the doctrine (`woof.supervisor`):
bounded-backoff retry for every publication, and quarantine-the-temp
for heartbeats -- "a stale heartbeat is safer than terminating a healthy
CUDA worker".  These tests hold the two prepared runners to it.
"""

from __future__ import annotations

import json
import os

import pytest

from woof import prepared_domain_tree_forecast as tree_runner
from woof import prepared_single_domain_forecast as single_runner
from woof import supervisor


@pytest.fixture()
def fast_backoff(monkeypatch):
    """Shrink the supervisor's 0.50 s retry ladder for test wall time."""
    monkeypatch.setattr(
        supervisor, "_REPLACE_BACKOFF_SECONDS", (0.0, 0.0, 0.0))


def _denying_replace(monkeypatch, *, failures: int):
    """os.replace that raises PermissionError ``failures`` times.

    Denies only replaces ONTO a publication target, the way a Windows
    reader does -- it holds the destination, not the quarantine
    directory -- so a heartbeat's quarantine move still lands.
    """
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(source, destination):
        if ".quarantine" in str(destination):
            return real_replace(source, destination)
        calls["n"] += 1
        if calls["n"] <= failures:
            raise PermissionError(13, "Access is denied", str(destination))
        return real_replace(source, destination)

    monkeypatch.setattr(supervisor.os, "replace", flaky)
    return calls


@pytest.mark.parametrize("runner", [tree_runner, single_runner],
                         ids=["tree", "single"])
def test_atomic_json_retries_a_transient_reader(
        tmp_path, monkeypatch, fast_backoff, runner):
    """A transient WinError-5 on the destination is retried, not fatal."""
    calls = _denying_replace(monkeypatch, failures=2)
    target = tmp_path / "progress.json"
    runner._atomic_json(target, {"status": "RUNNING", "outer_step": 7})
    assert calls["n"] >= 3
    assert json.loads(target.read_text(encoding="utf-8"))["outer_step"] == 7


@pytest.mark.parametrize("runner", [tree_runner, single_runner],
                         ids=["tree", "single"])
def test_heartbeat_survives_a_persistent_reader(
        tmp_path, monkeypatch, fast_backoff, runner):
    """A persistently-denied heartbeat is skipped, never raised.

    The old progress content stays readable, the in-flight temporary is
    quarantined out of the publication directory, and the caller -- the
    forecast loop -- continues.
    """
    target = tmp_path / "progress.json"
    target.write_text('{"status": "RUNNING", "outer_step": 6}\n',
                      encoding="utf-8")
    _denying_replace(monkeypatch, failures=10_000)
    runner._atomic_json(target, {"status": "RUNNING", "outer_step": 7},
                        heartbeat=True)
    assert json.loads(target.read_text(encoding="utf-8"))["outer_step"] == 6
    leftovers = [p for p in tmp_path.iterdir()
                 if p.name not in ("progress.json", ".quarantine")]
    assert leftovers == [], (
        "the denied temporary must not linger beside the publication")


@pytest.mark.parametrize("runner", [tree_runner, single_runner],
                         ids=["tree", "single"])
def test_receipts_stay_fail_loud(
        tmp_path, monkeypatch, fast_backoff, runner):
    """Durable receipts keep raising after the bounded retry."""
    _denying_replace(monkeypatch, failures=10_000)
    with pytest.raises(PermissionError):
        runner._atomic_json(tmp_path / "run-receipt.json", {"status": "PASS"})


@pytest.mark.parametrize("runner", [tree_runner, single_runner],
                         ids=["tree", "single"])
def test_every_progress_publication_declares_heartbeat(runner):
    """Every progress.json publication in both runners says heartbeat.

    Inspect the complete calls: publication sites are closures buried in
    the drivers and constructing a live run needs a GPU. The invariant
    reads off the source -- an ``_atomic_json`` call whose destination
    is the progress file must carry ``heartbeat=True``, or a watcher
    holding that file open kills the run it is watching.
    """
    import ast
    import inspect

    sites = 0
    for call in ast.walk(ast.parse(inspect.getsource(runner))):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "_atomic_json" and call.args):
            continue
        if "progress" not in ast.unparse(call.args[0]):
            continue
        sites += 1
        assert any(keyword.arg == "heartbeat"
                   and isinstance(keyword.value, ast.Constant)
                   and keyword.value.value is True for keyword in call.keywords), (
            "a progress publication without heartbeat=True can kill the "
            f"run it reports on: {ast.unparse(call)[:160]!r}")
    assert sites > 0, "no progress publication sites found"
