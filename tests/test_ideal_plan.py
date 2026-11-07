"""CPU artifact checks for source recipes and truthful ideal admission status."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from woof.ideal_plan import boundary_topology, case_catalog, inspect_namelist


NATIVE = """&time_control run_seconds=4, history_interval_s=1, /
&domains max_dom=1, e_we=13, e_sn=11, e_vert=7,
 dx=50., dy=50., ztop=2000., time_step=0,
 time_step_fract_num=1, time_step_fract_den=3, /
&dynamics damp_opt=0, /
&bdy_control periodic_x=.true., periodic_y=.true., /
"""


def write(tmp_path, text=NATIVE):
    path = tmp_path / "namelist.input"
    path.write_text(text, encoding="utf-8")
    return path


def test_build_time_initializer_requires_explicit_provenance_when_renamed(tmp_path):
    with pytest.raises(ValueError, match="build-time ideal case is unresolved"):
        inspect_namelist(write(tmp_path))
    directory = tmp_path / "em_heldsuarez"
    directory.mkdir()
    assert inspect_namelist(write(directory))["case"] == "em_heldsuarez"


def test_boundary_switches_preserve_unsupplied_child_defaults_and_conflicts():
    doc = {"bdy_control": {"periodic_x": [True], "open_xs": [True],
                           "symmetric_ys": [True], "open_ye": [True]}}
    root = boundary_topology(doc)
    assert root["x"]["conflict"]
    assert root["y"]["sides"] == {"s": ["symmetric"], "e": ["open"]}
    assert boundary_topology(doc, 1)["x"]["periodic"] is False
    with pytest.raises(ValueError, match="must be logical"):
        boundary_topology({"bdy_control": {"periodic_x": [1]}})


@pytest.mark.parametrize("before,after,message", [
    ("e_we=13", "e_we=1", "mass grid is empty"),
    ("dx=50.", "dx=-1.", "finite and positive"),
    ("time_step_fract_den=3", "time_step_fract_den=0", "integer >= 1"),
    ("time_step_fract_num=1", "time_step_fract_num=0", "fixed timestep is zero"),
])
def test_invalid_grid_and_clock_have_concrete_refusals(tmp_path, before, after, message):
    text = NATIVE.replace(before, after)
    with pytest.raises(ValueError, match=message):
        inspect_namelist(write(tmp_path, text),case="em_heldsuarez")


def test_public_cli_reports_refused_import_with_cupy_blocked(tmp_path):
    path = write(tmp_path, NATIVE)
    # Run the actual public entry point. A module finder fails if inspection
    # ever attempts to import CuPy; this does not replace any engine function.
    script = """import importlib.abc,runpy,sys
class NoGPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='cupy' or fullname.startswith('cupy.'):
   raise RuntimeError('inspection attempted GPU import')
sys.meta_path.insert(0,NoGPU())
sys.argv=['woof','ideal-plan',sys.argv[1],'--case','em_heldsuarez','--json']
runpy.run_module('woof',run_name='__main__')
"""
    env = dict(os.environ, GPUWM_NO_LOCAL_GPU="1", CUDA_VISIBLE_DEVICES="-1",
               PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-c", script, str(path)],
                         cwd=Path(__file__).resolve().parents[1], env=env,
                         capture_output=True, text=True, timeout=40)
    assert run.returncode == 2, run.stderr
    result = json.loads(run.stdout)
    assert not result["imports"] and not result["runs"]
    assert result["case"] == "em_heldsuarez"
    assert "inspection attempted GPU import" not in run.stderr


def test_packaged_catalog_is_independent_between_calls():
    cat = case_catalog()
    assert len(cat["cases"]) == 1
    assert {r["selector"] for r in cat["cases"] if r["selector"]} == set()
    assert len({r["name"] for r in cat["cases"]}) == 1
    assert len(cat["source_files"]) == 1
    assert all(len(r["sha256"]) == 64 for r in cat["source_files"])
    cat["cases"].clear()
    assert len(case_catalog()["cases"]) == 1
