"""The GFS PBL CUDA kernel against WRF v4.6.1's own words (card tests).

The CPU half -- fixture pins, the constants header and the NumPy reference
-- is tests/test_gfs_pbl_wrf461_parity.py, which also documents the
oracle records these tests replay.  Here the production launcher
(woof.core.gfs_pbl.gfs_pbl_step) is graded word for word on every call of
every record under the battery's strict arithmetic and, in its own process,
under GPUWM_WRF_EXACT=1. A one-bit change
planted in one kernel constant must be caught, and the workspace chunking
must be invisible.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from test_gfs_pbl_wrf461_parity import (ORACLE, ROOT, _fixture_paths,
                                        compare, oracle_io)

cp = pytest.importorskip("cupy")


def test_the_kernel_reproduces_every_oracle_word_strict_battery_arithmetic(tmp_path):
    report = compare.grade(_fixture_paths(tmp_path), "gpu")
    assert report["words"] == 468864
    assert report["differing_words"] == 0, report["fields"]


def test_the_kernel_reproduces_every_oracle_word_strict_arithmetic(tmp_path):
    """GPUWM_WRF_EXACT=1 is read at import, so the strict arm is its own
    process running the same launcher and the same kernel source."""
    paths = [str(p) for p in _fixture_paths(tmp_path)]
    receipt = tmp_path / "strict.json"
    env = dict(os.environ, GPUWM_WRF_EXACT="1")
    run = subprocess.run(
        [sys.executable, str(ORACLE / "compare.py"), "--impl", "gpu",
         "--receipt", str(receipt), *paths],
        env=env, capture_output=True, text=True, cwd=str(ROOT))
    report = json.loads(receipt.read_text("utf-8"))
    assert report["words"] == 468864, run.stdout + run.stderr
    assert report["differing_words"] == 0, report["fields"]
    assert run.returncode == 0


def test_a_one_bit_change_in_one_kernel_constant_is_caught(tmp_path):
    """Flip the last mantissa bit of GOCP (g/cp, the dry adiabatic lapse
    rate MONINP adds to every heat flux) in the compiled source: the grade
    must see it."""
    from woof.core import gfs_pbl
    from woof import nvrtc_ptx_cache as compiler
    from woof.core.kernels import module_options
    source = gfs_pbl.module_source()
    line = next(l for l in source.splitlines()
                if l.startswith("#define GFS_GOCP "))
    word = float.fromhex(line.split("(")[1].split(")")[0])
    bits = np.float64(word).view(np.uint64) ^ np.uint64(1)
    planted = line.replace(line.split("(")[1].split(")")[0],
                           float(bits.view(np.float64)).hex())
    assert planted != line
    ptx, _ = compiler.compile_using_nvrtc(
        source.replace(line, planted), module_options("gfs_pbl"), None,
        "gfs_pbl_planted.cu")
    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    paths = _fixture_paths(tmp_path)[2:3]
    caught = compare.grade(paths, "gpu", module=module)
    assert caught["differing_words"] > 0
    clean = compare.grade(paths, "gpu")
    assert clean["differing_words"] == 0


def test_a_chunked_launch_equals_a_single_one(tmp_path):
    """The workspace chunking is invisible: a budget forcing 72 one-column
    chunks gives the same words as one launch."""
    from woof.core.gfs_pbl import gfs_pbl_step
    rec = oracle_io.read_oracle(_fixture_paths(tmp_path)[2])
    call = rec["steps"][0]["inputs"]
    nz, ncol = call["u"].shape
    names = {"u": "u", "v": "v", "t": "t", "qv": "qv", "qc": "qc",
             "qi": "qi", "p": "p", "exner": "pi", "dz": "dz8w", "z": "z"}

    def run(budget):
        cols = {k: cp.asarray(call[v].reshape(nz, 1, ncol))
                for k, v in names.items()}
        sfc = {k: cp.asarray(call[k].reshape(1, ncol)) for k in
               ("psfc", "psim", "psih", "hfx", "qfx", "gz1oz0", "br",
                "wspd")}
        out = gfs_pbl_step(cols, sfc, dt=rec["dt"], flqi=rec["flqi"],
                           budget_bytes=budget)
        return {k: cp.asnumpy(out[k]) for k in
                ("rublten", "rthblten", "rqvblten", "rqiblten", "pblh",
                 "kpbl", "ust")}

    one = run(1 << 30)
    many = run(1)
    for name in one:
        assert one[name].tobytes() == many[name].tobytes(), name


@pytest.mark.parametrize("bldt", [0.0, 1.0])
def test_gfs_restart_continuation_preserves_every_word(tmp_path, bldt):
    from woof.core.dycore import run_steps
    from woof.io import restart
    from test_restart import _physics_state, _assert_restart_equal

    def build():
        return _physics_state(cp, bl_pbl_physics=3, sf_sfclay_physics=3,
                              bldt=bldt, cu_physics=0)

    full, cfg, driver = build()
    run_steps(full, cfg, 40)
    reference = restart.write_restart(tmp_path / "full.npz", full, cfg)
    assert driver.call_counts["ysu"] > 0  # shared PBL cadence counter
    assert bool(cp.all(cp.isfinite(driver.fields["pblh"])))
    assert bool(cp.any(driver.fields["pblh"] > 0))
    assert bool(cp.all((driver.fields["kpbl"] >= 1) & (driver.fields["kpbl"] <= cfg.nz)))

    first, cfg, _ = build()
    run_steps(first, cfg, 20)
    checkpoint = restart.write_restart(tmp_path / "mid.npz", first, cfg)
    resumed, cfg, _ = build()
    info = restart.restore_restart(checkpoint, resumed, cfg)
    assert info.elapsed_seconds == 200.0
    run_steps(resumed, cfg, 20)
    actual = restart.write_restart(tmp_path / "resumed.npz", resumed, cfg)
    _assert_restart_equal(actual, reference)


def test_fused_validation_keeps_each_field_failure_and_deferred_drain():
    """A bad column must still name its field and fail at the ledger drain."""
    from woof.core.gfs_pbl import (GFS_PBL_COLUMN_OUTPUTS,
                                    validate_gfs_pbl_outputs)
    from woof.core.health_ledger import HealthLedger, deferring

    names = GFS_PBL_COLUMN_OUTPUTS + ("ust", "pblh")
    for bad_name in names:
        outputs = {name: cp.zeros((4, 1, 3) if name in
                                 GFS_PBL_COLUMN_OUTPUTS else (1, 3), cp.float32)
                   for name in names}
        outputs[bad_name].reshape(-1)[-1] = cp.nan
        assert validate_gfs_pbl_outputs(outputs) == bad_name
        ledger = HealthLedger(label="gfs-finite-regression")
        with deferring(ledger):
            assert validate_gfs_pbl_outputs(outputs) is None
        assert ledger.drains == 0
        with pytest.raises(FloatingPointError, match="non-finite " + bad_name):
            ledger.drain()
        # The failed verdict was consumed and cleared before it raised.
        ledger.drain()
