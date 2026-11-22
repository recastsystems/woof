"""`woof hex --help` and each door's --help build where the POSIX-only modules are missing (Windows).

THE BREAKAGE THIS PREVENTS (0.6.1 on Windows): `--help` builds every door, and two derivative modules imported
`resource` and `fcntl` at the top, so `woof hex --help` (and a folded `woof hex --help`) stopped with
`ModuleNotFoundError: No module named 'resource'` before printing anything.

THE BREAKAGE THE ENCODING ARM PREVENTS (0.6.2's check, Windows 11): with output piped or redirected, Windows
gives Python the ANSI code page (cp1252), and ten doors' --help stopped with UnicodeEncodeError on the arrow in
their help text. The probe runs with that stream (PYTHONIOENCODING=cp1252), which reproduces it on any host.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import sys
for name in ("resource", "fcntl", "pwd", "grp", "termios"):
    sys.modules[name] = None          # what an interpreter without them raises on import
sys.argv = ["woof hex", *sys.argv[1:]]
from woof.hex import cli
cli.entry()
"""


@pytest.mark.parametrize("door", [[], ["adjoint-test"], ["sensitivity"], ["tangent"], ["param-grad"],
                                  ["render"], ["forecast"], ["global"], ["regional"], ["country"], ["mesh-plan"],
                                  ["pair"], ["products"], ["s2s"], ["run-plan"], ["global-cycle"]])
def test_help_builds_without_posix_only_modules(door):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
           "PYTHONIOENCODING": "cp1252"}
    done = subprocess.run([sys.executable, "-c", _PROBE, *door, "--help"], capture_output=True, encoding="cp1252",
                          timeout=300, env=env, cwd=str(ROOT))
    assert done.returncode == 0, done.stderr[-2000:]
    assert "usage: woof hex" in done.stdout


def test_the_session_lock_names_what_it_needs_without_fcntl(monkeypatch):
    sys.path.insert(0, str(ROOT / "src"))
    from woof.hex import ad_session
    monkeypatch.setattr(ad_session, "fcntl", None)
    with pytest.raises(RuntimeError, match="fcntl.flock"):
        ad_session._posix_lock()
