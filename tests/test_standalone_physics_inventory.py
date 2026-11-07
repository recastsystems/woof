"""CPU preparation needs the same field-inventory metadata as state pricing."""
import os
from pathlib import Path
import subprocess
import sys


def test_cpu_wheel_imports_physics_inventory_without_cuda(tmp_path):
    from tools.build_rw_wps_release import _stage_rw_wps_python_project
    staged = tmp_path / "standalone"
    _stage_rw_wps_python_project(staged)
    script = r"""
from importlib.abc import MetaPathFinder
from pathlib import Path
import sys
class NoCuda(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cupy" or fullname.startswith("cupy."):
            raise ModuleNotFoundError("CPU inventory must not load CUDA")
sys.meta_path.insert(0, NoCuda())
from woof.config import RunConfig
from woof.core import physics_inventory
assert Path(physics_inventory.__file__).resolve().parent == Path.cwd() / "woof/core"
assert physics_inventory.physics_driver_required(RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=10000., dt=1., run_seconds=2., bl_pbl_physics=17))
assert "cupy" not in sys.modules
"""
    env = dict(os.environ, PYTHONPATH=str(staged))
    # The CPU-wheel import does not request CUDA compiler shims.
    env.pop("GPUWM_WRF_EXACT", None)
    result = subprocess.run([sys.executable, "-c", script], cwd=staged, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
