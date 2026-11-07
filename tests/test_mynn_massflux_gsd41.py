"""GSD v4.1 plume population, trigger, ascent and flux oracle controls."""
from pathlib import Path
import csv

import numpy as np
import pytest

from test_mynn_gsd41 import requires_gpu

ORACLE = Path(__file__).resolve().parents[1] / "tests/data/oracles/mynn/dmp-mf-gsd41.csv"


def _cases():
    with ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 17 * 30
    return {name: [row for row in rows if row["case"] == name]
            for name in dict.fromkeys(row["case"] for row in rows)}


@requires_gpu
def test_gsd41_dmp_massflux_matches_unmodified_fork():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_dmp_mf_cuda
    from test_mynn_pbl import _dmp_mf_inputs, _dmp_mf_expected

    for case, rows in _cases().items():
        values = _dmp_mf_inputs(rows)
        values["sgm"] = np.array([[np.float32(row["sgm"]) for row in rows]])
        actual = mynn_dmp_mf_cuda(values, bl_mynn_version="gsd_41")
        wanted = _dmp_mf_expected(rows)
        # maxwidth is diagnostic only and is not returned by fork DMP_mf.
        del wanted["maxwidth"]
        for name, expected in wanted.items():
            got = cp.asnumpy(getattr(actual, name)).reshape(expected.shape)
            if name == "ktop":
                np.testing.assert_array_equal(got, expected, err_msg=case)
            else:
                np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32),
                                              err_msg=f"{case}: {name}")


def test_fork_dmp_fixture_exercises_trigger_and_tapers():
    cases = _cases()
    for name in ("stable_off", "resolved_w", "vapour_trigger"):
        assert int(cases[name][0]["nup2"]) == 0
        assert np.float32(cases[name][0]["maxmf"]) == 0
    for name in ("land_cumulus", "water_cumulus", "high_wind_thin",
                 "w_taper", "downdraft_taper", "tiny_population"):
        assert int(cases[name][0]["nup2"]) > 0, name
        assert abs(np.float32(cases[name][0]["maxmf"])) > 0, name
    stopped = cases["all_plumes_stop"][0]
    assert int(stopped["nup2"]) > 0
    assert int(stopped["ktop"]) == 0
    assert np.float32(stopped["maxmf"]) == 0


@requires_gpu
def test_gsd41_plumes_ignore_wind_speed_taper_and_external_fltv():
    import cupy as cp
    from woof.core.mynn_pbl_gpu import mynn_dmp_mf_cuda
    from test_mynn_pbl import _dmp_mf_inputs

    rows = _cases()["land_cumulus"]
    values = _dmp_mf_inputs(rows)
    values["sgm"] = np.zeros_like(values["th"])
    reference = mynn_dmp_mf_cuda(values, bl_mynn_version="gsd_41")
    before = {name: cp.asnumpy(getattr(reference, name)).copy()
              for name in ("edmf_a", "edmf_w", "s_aw", "s_awthl", "maxmf")}
    values["u"] += 30.0
    values["fltv"][:] = -7.0
    changed = mynn_dmp_mf_cuda(values, bl_mynn_version="gsd_41")
    for name, expected in before.items():
        np.testing.assert_array_equal(cp.asnumpy(getattr(changed, name)), expected)


@requires_gpu
def test_gsd41_transport_matches_clear_fork_columns():
    import cupy as cp
    from woof.core.fp32_ulp import fp32_ulp_distance
    from woof.core.mynn_pbl import MYNN_TENDENCIES_LAYER_INPUTS, MYNN_TENDENCIES_INTERFACE_INPUTS
    from woof.core.mynn_pbl_gpu import mynn_tendencies_default_cuda, _TENDENCIES_LAYER_SCRATCH
    from woof.core.mynn_pbl_scratch import MynnPblScratch, SLOT_TENDENCY_WORK
    path = ORACLE.with_name("transport-gsd41.csv")
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8 * 12
    fields = {key: np.array([np.float32(row[key]) for row in rows]).reshape(8, 12)
              for key in rows[0] if key not in ("case", "k")}
    values = {name: fields.get(name, np.zeros((8, 12), np.float32))
              for name in MYNN_TENDENCIES_LAYER_INPUTS}
    for name in MYNN_TENDENCIES_INTERFACE_INPUTS:
        values[name] = (np.concatenate((fields[name][:, :1], fields[name + "_next"]), axis=1)
                        if name in fields else np.zeros((8, 13), np.float32))
    for name in ("delt", "ust", "flt", "flqv", "wspd"):
        values[name] = fields[name][:, 0]
    values.update(psfc=np.full(8, 95500.0, np.float32), uoce=np.zeros(8, np.float32),
                  voce=np.zeros(8, np.float32), flqc=np.zeros(8, np.float32))
    values.update(qc=np.zeros((8, 12), np.float32), qi=np.zeros((8, 12), np.float32))
    scratch = MynnPblScratch.standalone(8, 12)
    actual = mynn_tendencies_default_cuda(values, scratch=scratch, bl_mynn_version="gsd_41")
    for name in ("du", "dv", "dth"):
        np.testing.assert_array_equal(cp.asnumpy(getattr(actual, name)), fields[name], err_msg=name)
    work = scratch.group(SLOT_TENDENCY_WORK, _TENDENCIES_LAYER_SCRATCH, (8, 12))
    # Compare the internal solve with the inverse-converted source output
    # (a FP32 round trip); the direct P21 output is checked below.
    assert int(fp32_ulp_distance(cp.asnumpy(work["sqv2"]), fields["sqv_solved"]).max()) <= 2
    np.testing.assert_array_equal(cp.asnumpy(actual.dqv).view(np.uint32), fields["dqv"].view(np.uint32))


def test_fork_workspace_prices_ten_plume_classes_without_changing_default():
    from woof.core.mynn_pbl_scratch import (
        mynn_pbl_scratch_shapes, SLOT_PLUME_WORK, SLOT_PLUME_SCRATCH)
    before = mynn_pbl_scratch_shapes(32, 50)
    after = mynn_pbl_scratch_shapes(32, 50, bl_mynn_version="gsd_41")
    assert before[SLOT_PLUME_WORK] == (64 * 32 * 51,)
    assert after[SLOT_PLUME_WORK] == (80 * 32 * 51,)
    assert after[SLOT_PLUME_SCRATCH] == (13 * 32 * 50,)
    for slot in before.keys() - {SLOT_PLUME_WORK, SLOT_PLUME_SCRATCH}:
        assert after[slot] == before[slot]
    # The generation's own working sets exist only under the key.
    assert set(after) - set(before) == {
        "mynn_pbl_gsd41_condensation_work", "mynn_pbl_gsd41_thvl",
        "mynn_pbl_gsd41_diagnostics"}
    # Five native driver diagnostics per column (6d79b6e9c5d).
    assert after["mynn_pbl_gsd41_diagnostics"] == (5 * 32,)
    assert after["mynn_pbl_gsd41_condensation_work"] == (5 * 32 * 50,)
    assert after["mynn_pbl_gsd41_thvl"] == (32 * 50,)
