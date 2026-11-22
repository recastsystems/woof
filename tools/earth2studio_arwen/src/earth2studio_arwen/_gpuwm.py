# SPDX-License-Identifier: Apache-2.0
"""How this package reaches woof.

woof (WOOF-ArWen) and woof hex carry their own CUDA stack, so they usually live
in a different Python environment from Earth2Studio. This package never
imports woof: it runs the ``woof`` command of the environment named by
``WOOF_PYTHON`` (that environment's interpreter), or the ``woof`` found on
PATH. Everything model-specific (variables, levels, profile) is read from
woof's own tables through that interpreter, so this package names no model.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ENV = "WOOF_PYTHON"


def gpuwm_python() -> str:
    """The interpreter of the environment that has woof installed."""
    named = os.environ.get(ENV)
    if named:
        if not Path(named).exists():
            raise FileNotFoundError(f"{ENV}={named} does not exist")
        return named
    exe = shutil.which("woof")
    if exe:
        first = Path(exe).read_text(errors="ignore").splitlines()[:1]
        if first and first[0].startswith("#!"):
            return first[0][2:].strip().split()[0]
    raise FileNotFoundError(
        f"woof not found: set {ENV} to the python of the environment where "
        "`pip install recast-woof` was run, or put its `woof` on PATH")


def gpuwm_cmd() -> list[str]:
    return [gpuwm_python(), "-m", "woof"]


def run(args: list[str], *, cwd: Path, log: Path, env: dict | None = None) -> float:
    """Run one woof command, log it, return wall seconds; raise with the log tail on failure."""
    cmd = gpuwm_cmd() + [str(a) for a in args]
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=full_env)
    seconds = time.perf_counter() - t0
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"# exit {proc.returncode} after {seconds:.1f} s\n")
    if proc.returncode != 0:
        tail = "".join(Path(log).read_text(encoding="utf-8", errors="replace").splitlines(True)[-40:])
        raise RuntimeError(f"woof {args[0]} failed (exit {proc.returncode}); log {log}:\n{tail}")
    return seconds


def earth2_table() -> dict:
    """woof's Earth-2 table (woof/authorities/earth2-models.v1.json), read by woof's interpreter."""
    code = ("import json, woof.earth2_run as e; "
            "print(json.dumps(e.model_table()))")
    out = subprocess.run([gpuwm_python(), "-c", code], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def profile_variables(profile: str) -> list[str]:
    """Earth2Studio variable ids a packaged earth2 profile reads (levels x families + surface)."""
    row = earth2_table()["profiles"][profile]
    names = [f"{fam}{lev}" for fam in row["level_families"] for lev in row["levels_hpa"]]
    names += list(row["surface"])
    return names


def profile_boundary_hours(profile: str) -> int:
    """The boundary spacing the profile's mapping takes (hours)."""
    code = ("import json, sys; from importlib import resources; "
            "p = resources.files('woof') / 'authorities' / ('rw-wps-' + sys.argv[1].removesuffix('-v1') + '.mapping.json'); "
            "print(json.loads(p.read_text())['target']['boundary_interval_seconds'])")
    out = subprocess.run([gpuwm_python(), "-c", code, profile], check=True, capture_output=True, text=True)
    return int(out.stdout.strip()) // 3600


def default_geog_root() -> str:
    """The static-geography tree woof itself defaults to ($GPUWM_CASE_DATA_ROOT/WPS_GEOG)."""
    code = "from woof.geog_assets import default_geog_root; print(default_geog_root())"
    out = subprocess.run([gpuwm_python(), "-c", code], check=True, capture_output=True, text=True)
    return out.stdout.strip()


def python_version() -> str:
    return sys.version.split()[0]
