"""A forecast that refuses, runs out of memory or is killed never exits 0: not through the console door, and not
through the child process the composing doors (global, regional, country) run it in.

THE BREAKAGE THIS HOLDS SHUT (bitwise ladder, 2026-11-06): lanes counted a forecast that stopped with
``ForecastDoorRefusal`` and one that ran out of device memory as finished, because the run's recorded exit status
said 0.  The door was right that time (the wrappers recorded ``$?`` after a command substitution had reset it), but
a door that swallowed a refusal or an out-of-memory stop would do the same to every pool, lane and composing door,
which all act on the exit status alone.  Each case runs the real door code in a fresh interpreter with only the
forecast body replaced, and requires a non-zero status and, for a refusal, its sentence on stderr.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("woof.output_formats")  # the forecast door's parser needs the engine (test_cli_without_engine)

SRC = Path(__file__).resolve().parents[1] / "src"

CASES = {
    "finished": 0,
    "refused": 2,       # the door's named refusal, relayed by woof.hex.cli
    "out-of-memory": 1,  # an uncaught MemoryError (cupy's OutOfMemoryError is one) ends the interpreter with 1
}
if hasattr(os, "kill") and sys.platform != "win32":
    CASES["killed"] = None  # SIGKILL, as the kernel's out-of-memory killer sends it


@pytest.fixture
def door_env(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(textwrap.dedent("""
        import os
        mode = os.environ.get("HEX_EXIT_TEST")
        if mode:
            import woof.hex.forecast_door as door
            def body(arguments):
                print("forecast started", flush=True)
                if mode == "refused":
                    raise door.ForecastDoorRefusal("the exit-status test refuses this run part way through")
                if mode == "out-of-memory":
                    raise MemoryError("the exit-status test ran out of memory")
                if mode == "killed":
                    import signal
                    os.kill(os.getpid(), signal.SIGKILL)
                return 0
            door.run_forecast = body
    """))
    path = os.pathsep.join([str(tmp_path), str(SRC), os.environ.get("PYTHONPATH", "")])
    return lambda mode: {**os.environ, "PYTHONPATH": path, "HEX_EXIT_TEST": mode, "PYTHONDONTWRITEBYTECODE": "1"}


def _expected(mode: str, status: int) -> None:
    want = CASES[mode]
    if want is None:
        assert status in (-9, 137), status
    else:
        assert status == want, status


@pytest.mark.parametrize("mode", sorted(CASES))
def test_the_console_door_exit_status_carries_the_verdict(door_env, mode):
    script = "import sys; from woof.hex.cli import entry; sys.argv = ['woof hex', 'forecast']; raise SystemExit(entry())"
    done = subprocess.run([sys.executable, "-c", script], env=door_env(mode), capture_output=True, text=True,
                          timeout=300)
    assert "forecast started" in done.stdout, done.stderr[-2000:]
    _expected(mode, done.returncode)
    if mode == "refused":
        assert "woof hex: the exit-status test refuses this run part way through" in done.stderr


@pytest.mark.parametrize("mode", sorted(CASES))
def test_a_composing_door_sees_the_forecast_child_fail(door_env, mode):
    """``cli.run_stage`` is how global, regional and country run their forecast stage; they stop on non-zero."""
    script = "from woof.hex import cli; print('stage status', cli.run_stage(['forecast']))"
    done = subprocess.run([sys.executable, "-c", script], env=door_env(mode), capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    _expected(mode, int(done.stdout.strip().splitlines()[-1].split()[-1]))
