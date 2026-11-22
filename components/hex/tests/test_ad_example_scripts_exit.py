"""THE BREAKAGE (0.6.0 train, turn 12): run-32gb.sh and run-96gb.sh printed each command's exit status and then ended 0,
so a run in which every command was refused (no card visible) read as a success to any caller that checks the status.
Each script now ends with the highest exit status of its commands."""
import os
import stat
import subprocess
from pathlib import Path

import pytest

EX = Path(__file__).resolve().parents[1] / "examples" / "differentiable"


def _stand_in(tmp_path, rc):
    b = tmp_path / "bin"; b.mkdir(exist_ok=True)
    for name, code in (("woof", rc), ("python", 0)):
        f = b / name; f.write_text(f"#!/bin/bash\nexit {code}\n"); f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return {**os.environ, "PATH": f"{b}:{os.environ['PATH']}"}


@pytest.mark.parametrize("script", ["run-32gb.sh", "run-96gb.sh"])
@pytest.mark.parametrize("rc", [0, 1, 2])
def test_the_script_ends_with_its_worst_command_status(tmp_path, script, rc):
    env = _stand_in(tmp_path, rc)
    p = subprocess.run(["bash", str(EX / script), str(tmp_path / "out")], env=env, capture_output=True, text=True)
    assert p.returncode == rc, p.stdout + p.stderr
    assert f"exit {rc}" in p.stdout
