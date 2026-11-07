"""Acceptance failures that the variant referee must never hide."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare


def fields(value=1.0):
    return {n: np.full((2, 3), value, np.float32) for n in compare.FIELDS}


def test_identical_finite_fields_pass():
    assert compare.compare_mode(fields(), fields(), "strict")["passed"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("arm", ["strict", "default"])
def test_nonfinite_outputs_fail_even_when_identical(value, arm):
    assert not compare.compare_mode(fields(value), fields(value), arm)["passed"]


def test_one_bit_variant_moment_is_caught():
    wrf, gpu = fields(), fields()
    gpu["qzh"].view(np.uint32)[1, 2] ^= np.uint32(1)
    result = compare.compare_mode(wrf, gpu, "strict")
    assert not result["passed"]
    assert result["fields"]["qzh"]["max_ulp"] == 1
    assert result["fields"]["qzh"]["cells_differing"] == 1


def test_signed_zero_is_not_bitwise_identity():
    assert not compare.compare_mode(fields(0.0), fields(-0.0), "strict")["passed"]


def test_unknown_arm_refuses():
    with pytest.raises(ValueError, match="arm"):
        compare.compare_mode(fields(), fields(), "strcit")


def test_shape_mismatch_refuses():
    gpu = fields()
    gpu["qzg"] = np.ones((3, 2), np.float32)
    with pytest.raises(ValueError, match="shape"):
        compare.compare_mode(fields(), gpu, "strict")


def test_double_precision_artifact_is_not_silently_rounded():
    gpu = fields()
    gpu["qzh"] = gpu["qzh"].astype(np.float64)
    with pytest.raises(TypeError, match="float32"):
        compare.compare_mode(fields(), gpu, "strict")


def test_empty_field_refuses():
    with pytest.raises(ValueError, match="empty field"):
        compare.compare_mode({n: np.empty(0, np.float32) for n in compare.FIELDS},
                             {n: np.empty(0, np.float32) for n in compare.FIELDS}, "strict")


def test_empty_directory_fails(tmp_path, monkeypatch):
    oracle, gpu = tmp_path / "oracle", tmp_path / "gpu"
    oracle.mkdir()
    gpu.mkdir()
    monkeypatch.setattr(sys, "argv", ["compare.py", str(oracle), str(gpu),
                                    str(tmp_path / "report.json")])
    assert compare.main() != 0


def test_missing_mode_fails(tmp_path, monkeypatch):
    oracle, gpu = tmp_path / "oracle", tmp_path / "gpu"
    oracle.mkdir()
    gpu.mkdir()
    data = fields()
    np.savez(oracle / "mp18-default.npz", **{f"wrf_{n}": a for n, a in data.items()})
    np.savez(gpu / "mp18-default.npz", **{f"gpu_{n}": a for n, a in data.items()})
    monkeypatch.setattr(sys, "argv", ["compare.py", str(oracle), str(gpu),
                                    str(tmp_path / "report.json")])
    assert compare.main() != 0
