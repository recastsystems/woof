"""The ISHMAEL comparison gate must reject incomplete evidence."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("ishmael_compare", Path(__file__).resolve().parents[1] / "tools/ishmael_wrf461_oracle/compare.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

def batch(nsteps=2, dt=20.0):
    fields = {n: np.ones((2, 64), np.float32) for n in cmp.OUTPUT_FIELDS}
    fields.update({n: np.ones(64, np.float32) for n in cmp.SURFACE_FIELDS})
    return {"ncol": 64, "nz": 2, "nsteps": nsteps, "dt": dt,
            "steps": [{n: a.copy() for n, a in fields.items()} for _ in range(nsteps)]}

def test_identical_complete_history_passes():
    report = cmp.compare(batch(), batch())
    assert report["total_words"] == 0 and report["passed"]
    assert report["compared_words"] == 2 * 64 * (2 * 35 + 4)

def test_matching_nonfinite_words_need_a_native_undefined_witness():
    got, want = batch(), batch()
    got["steps"][0]["th"][0, 0] = np.nan
    want["steps"][0]["th"][0, 0] = np.nan
    report = cmp.compare(got, want)
    assert report["total_words"] == 0
    assert report["nonfinite_words"] == 1 and not report["passed"]

def test_truncated_history_is_rejected():
    with pytest.raises(ValueError, match="step|history"):
        cmp.compare(batch(1), batch(2))

def test_different_timestep_is_rejected():
    with pytest.raises(ValueError, match="timestep|dt"):
        cmp.compare(batch(dt=30), batch(dt=20))

def test_one_bit_habit_change_is_caught():
    got, want = batch(), batch()
    got["steps"][1]["ai2"].view(np.uint32)[1, 37] ^= np.uint32(1)
    report = cmp.compare(got, want)
    assert report["total_words"] == 1
    assert report["differences"][0]["field"] == "ai2"
    assert report["differences"][0]["max_ulp"] == 1


def test_habit_rate_mutation_flips_exactly_one_bit():
    import re
    source = (Path(__file__).resolve().parents[1] / "tools/ishmael_wrf461_oracle/run_gpu.py").read_text()
    literal = re.findall(r'"(2\.324[0-9]+)f, 2\.08818f"', source)[-1]
    word = int(np.float32(literal).view(np.uint32))
    pristine = int(np.float32(2.32423).view(np.uint32))
    assert (word ^ pristine).bit_count() == 1
