"""UW shallow cumulus (shcu_physics=2) against its WRF v4.6.1 column oracle.

Fixtures: tools/uwshcu_wrf461_oracle (build.sh writes BUILD/fixtures);
point UWSHCU_ORACLE_DIR at that directory. The grade requires exact output
words, including signed zeros, in both modes. These generated soundings
do not satisfy the real-input forecast capture requirement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from woof.verify.uwshcu_oracle import (
    CORE_FULL_OUTPUTS, CORE_MASS_OUTPUTS, CORE_SURFACE_OUTPUTS,
    STEP_FULL_OUTPUTS, STEP_MASS_OUTPUTS, STEP_SURFACE_OUTPUTS, core_arrays,
    driver_dims, grade, load, step_arrays)

GRIDS = ("g40", "g56", "g33")
def _fixture(stem):
    directory = os.environ.get("UWSHCU_ORACLE_DIR")
    if not directory:
        pytest.skip("UWSHCU_ORACLE_DIR is unset")
    path = Path(directory) / stem
    if not Path(str(path) + ".manifest").is_file():
        pytest.skip(f"missing oracle fixture {path}")
    return load(path)


def _core_inputs(a):
    return {n: a[n] for n in ("ps0", "zs0", "p0", "z0", "dp0", "u0", "v0",
                              "qv0", "ql0", "qi0", "t0", "trl", "tri", "tke",
                              "pblh", "cush")}


def _grade_core(out, a):
    report = {}
    for name in CORE_FULL_OUTPUTS + CORE_MASS_OUTPUTS + ("trtl", "trti"):
        report[name] = grade(out[name].get(), a["out/" + name])
    for name in CORE_SURFACE_OUTPUTS:
        report[name] = grade(out[name].get(), a["out/" + name])
    return report


@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
def test_core_matches_oracle(grid):
    """compute_uwshcu_inv alone on binary64 inputs: every output."""
    from woof.core.uwshcu import uwshcu_core
    a = core_arrays(_fixture(f"core-{grid}"))
    # the constituents whose tendencies never leave the scheme are zero
    assert not np.any(a["out/trten123"])
    out = uwshcu_core(_core_inputs(a), dt=a["dt"])
    report = _grade_core(out, a)
    print(grid, json.dumps(report))
    for name, (bad, ulp, _) in report.items():
        assert bad == 0, f"{grid} {name}: {bad} words, max {ulp} ULP"
    active = int(np.count_nonzero(a["out/cush"] != -1.0))
    assert active >= 10, f"{grid}: only {active} convecting columns"


@pytest.mark.gpu
@pytest.mark.parametrize("grid", GRIDS)
def test_driver_matches_oracle(grid):
    """camuwshcu_driver, every step, every WRF-facing output."""
    import cupy as cp
    from woof.core.uwshcu import uwshcu_step
    fx = _fixture(f"cases-{grid}")
    ncol, nk, nsteps = driver_dims(fx)
    worst = {}
    for step in range(1, nsteps + 1):
        a = step_arrays(fx, step)
        mass = lambda n: cp.asarray(np.ascontiguousarray(a[n].T)[:, None, :])
        sfc = lambda n: cp.asarray(np.ascontiguousarray(a[n])[None, :])
        inputs = {n: mass(n) for n in ("u", "v", "t", "qv", "qc", "qi", "qnc",
                                        "qni", "p", "z", "pi", "dlf_in")}
        inputs.update({n: mass(n) for n in ("p8w", "z_at_w", "tke_pbl",
                                             "cmfmc_in")})
        inputs.update({"pblh": sfc("pblh"), "ht": sfc("ht"),
                       "cutop_in": sfc("cutop_in"), "cubot_in": sfc("cubot_in"),
                       "rliq_in": sfc("rliq_in")})
        carried = {"cush": sfc("cush_in")}
        out = uwshcu_step(inputs, carried, dt=a["dt"], has_qnc=a["has_qnc"],
                          has_qni=a["has_qni"])
        out["cush"] = carried["cush"]
        for name in STEP_MASS_OUTPUTS + STEP_FULL_OUTPUTS:
            got = out[name].get()[:, 0, :].T
            bad, ulp, sz = grade(got, a[name])
            worst[name] = max(worst.get(name, (0, 0, 0)), (bad, ulp, sz))
            assert bad == 0, \
                f"{grid} step {step} {name}: {bad} words, max {ulp} ULP"
        for name in STEP_SURFACE_OUTPUTS:
            got = out[name].get()[0]
            bad, ulp, sz = grade(got, a[name])
            worst[name] = max(worst.get(name, (0, 0, 0)), (bad, ulp, sz))
            assert bad == 0, \
                f"{grid} step {step} {name}: {bad} words, max {ulp} ULP"
    print(grid, json.dumps(worst))


def test_saturation_table_matches_oracle():
    """The UW_ESTBL upload is the oracle host's own gestbl table."""
    from woof.core.uwpbl_constants import ESTBL
    fx = _fixture("cases-g40")
    table = np.asarray(ESTBL, dtype=np.float64)
    bad, ulp, _ = grade(table, fx["const/estbl"])
    assert bad == 0, f"estbl: {bad} words differ, max {ulp} ULP"


@pytest.mark.gpu
def test_planted_one_bit_change_is_caught():
    """Flip the last bit of one folded constant in the port; the grade fails.

    UWSC_TWO_25066 is (2._r8/2.5066_r8), the cloud-base vertical velocity
    factor of every convecting column (module_shcu_camuwshcu.F:2192).
    """
    import cupy as cp
    from woof.core.kernels import module_source
    from woof.core.uwshcu import uwshcu_core
    from woof.core.uwpbl_constants import ESTBL
    src = module_source("uwshcu")
    good = "0x1.9885813c5beadp-1"
    assert src.count(good) == 1
    mod = cp.RawModule(code=src.replace(good, "0x1.9885813c5beacp-1"),
                       options=("-std=c++17",))
    table = np.asarray(ESTBL, dtype=np.float64)
    cp.ndarray(table.shape, dtype=cp.float64,
               memptr=mod.get_global("UW_ESTBL")).set(table)
    a = core_arrays(_fixture("core-g40"))
    report = _grade_core(uwshcu_core(_core_inputs(a), dt=a["dt"], module=mod), a)
    caught = {n: r for n, r in report.items() if r[0]}
    print(json.dumps(caught))
    assert "umf" in caught and "qvten" in caught


def test_case_coverage():
    """At least 64 columns per the brief, across every regime family."""
    directory = os.environ.get("UWSHCU_ORACLE_DIR")
    if not directory:
        pytest.skip("UWSHCU_ORACLE_DIR is unset")
    index = json.loads((Path(directory) / "cases-index.json").read_text())
    columns = [c for g in index.values() for c in g["columns"]]
    assert len(columns) >= 64
    families = {c.split("-")[0] for c in columns}
    assert {"trade", "continental", "mixedphase", "dry", "highterrain",
            "capped", "overshoot", "edge"} <= families
    assert all(g["nsteps"] >= 3 for g in index.values())
