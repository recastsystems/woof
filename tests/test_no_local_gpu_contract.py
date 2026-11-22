"""GPUWM_NO_LOCAL_GPU means NO device read, everywhere, from one definition.

The 2.5.0 upgrader walk set ``GPUWM_NO_LOCAL_GPU=1`` for every step and
still watched the ``woof go`` memory gate report "the card has 8.88 GiB
free right now": the gate's probe subprocess read the local device
because nothing on that path consulted the variable.  Whatever a user
believes the switch governs, a gate that measures the card anyway makes
the belief false in the one place it costs VRAM on a card the user said
not to touch.

The contract this file pins:

* :mod:`woof.local_gpu` is THE definition -- one truthiness rule, one
  docstring stating the scope (no device open, no context, no
  memGetInfo; it does not claim cupy is uninstalled);
* the device memory probe honors it BEFORE spawning anything, and the
  reason it returns names the variable;
* the go memory gate under the variable prices the phases, refuses
  nothing on a card it was told not to read, and its verdict says why
  there are no measured numbers;
* no product module reads the raw environment variable beside the one
  definition, so the scope cannot fork again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from woof import local_gpu
from woof.core import preflight

REPO = Path(__file__).resolve().parents[1]


def test_the_definition_reads_the_documented_truthiness(monkeypatch):
    monkeypatch.delenv(local_gpu.NO_LOCAL_GPU_ENV, raising=False)
    assert local_gpu.no_local_gpu() is False
    monkeypatch.setenv(local_gpu.NO_LOCAL_GPU_ENV, "0")
    assert local_gpu.no_local_gpu() is False
    monkeypatch.setenv(local_gpu.NO_LOCAL_GPU_ENV, "")
    assert local_gpu.no_local_gpu() is False
    monkeypatch.setenv(local_gpu.NO_LOCAL_GPU_ENV, "1")
    assert local_gpu.no_local_gpu() is True


def test_the_probe_honors_the_variable_before_spawning(monkeypatch):
    """No subprocess, no numbers, and the reason names the variable."""

    monkeypatch.setenv(local_gpu.NO_LOCAL_GPU_ENV, "1")

    def _boom(*_args, **_kwargs):
        raise AssertionError(
            "the probe spawned a device reader under GPUWM_NO_LOCAL_GPU")

    assert preflight.device_memory_probe_subprocess(run=_boom) is None
    reason = preflight.device_memory_probe_reason(run=_boom)
    assert reason is not None and "GPUWM_NO_LOCAL_GPU" in reason


def test_the_mynn_width_probe_honors_the_variable(monkeypatch):
    """The MYNN column-width probe opened the card under the variable
    (measured 2026-11-12: a 16 GB node's 7.27 GB free, read with it set)."""

    import sys
    import types

    from woof.core import mynn_pbl_scratch

    asked = []

    def _record(*args, **_kwargs):
        # The probe swallows any exception, so a raise would pass unseen.
        asked.append(args)
        raise RuntimeError("no device in this test")

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(
        runtime=types.SimpleNamespace(getDevice=_record, memGetInfo=_record,
                                      getDeviceProperties=_record),
        Device=_record))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setenv(local_gpu.NO_LOCAL_GPU_ENV, "1")
    assert mynn_pbl_scratch.probe_mynn_card() is None
    assert asked == [], "the MYNN width probe asked CuPy under GPUWM_NO_LOCAL_GPU"


def test_the_probe_still_measures_when_the_variable_is_unset(monkeypatch):
    """The measured-thresholds rule survives: unset means measure."""

    import json

    monkeypatch.delenv(local_gpu.NO_LOCAL_GPU_ENV, raising=False)

    class _Completed:
        returncode = 0
        stdout = json.dumps({"free_bytes": 1024, "total_bytes": 2048})
        stderr = ""

    payload = preflight.device_memory_probe_subprocess(
        run=lambda *a, **k: _Completed())
    assert payload is not None and payload["free_bytes"] == 1024


def test_the_go_gate_under_the_variable_gates_declared_and_says_so(
        monkeypatch, tmp_path):
    """The gate prices the config, refuses nothing on the unread card,
    and its verdict carries the variable's name."""

    from woof import go_cli
    from woof.cli import main as cli_main

    config = tmp_path / "gated.toml"
    assert cli_main([
        "domain", "--point", "35.3,-97.5", "--source", "gfs",
        "--ladder", "12", "--card", "24gb",
        "--cycle", "2026-07-29T18", "--hours", "6",
        "--physics-profile", "morrison-mp10-ysu-mm5-noah-kf-rte-rrtmgp-v1",
        "--out", str(config)]) == 0

    monkeypatch.setenv(local_gpu.NO_LOCAL_GPU_ENV, "1")
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "out")
    gate = go_cli.memory_gate(plan)
    assert gate["refuse"] is False
    assert gate["free_bytes"] is None
    assert "GPUWM_NO_LOCAL_GPU" in gate["verdict"]


def test_the_variable_has_one_definition_in_the_product():
    """No second raw read may appear beside :mod:`woof.local_gpu`.

    Scope forks are exactly how the go gate came to read the device
    under the variable while doctor did not: three hand-rolled
    ``os.environ.get`` reads agreed by luck.  The census below fails
    the moment a fourth appears.
    """

    pattern = re.compile(
        r"""environ(?:\.get)?\s*[(\[]\s*['"]GPUWM_NO_LOCAL_GPU""")
    offenders = []
    for path in (REPO / "woof").rglob("*.py"):
        if path.name == "local_gpu.py":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], (
        "raw GPUWM_NO_LOCAL_GPU reads outside woof/local_gpu.py: "
        f"{offenders}; import woof.local_gpu.no_local_gpu instead")
