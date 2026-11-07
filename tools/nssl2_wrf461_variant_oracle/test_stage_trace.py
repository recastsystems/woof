"""The stage referee refuses partial or corrupt traces."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_trace import read_trace, TRACE_FIELDS, PHASES


def fixture(path, mutate=None):
    rows = []
    for phase in PHASES:
        for ix in (1, 2):
            for k in (1, 2, 3):
                rows.append(f"{phase},2,1,1,{ix},{k}," + ",".join(["3F800000"] * len(TRACE_FIELDS)))
    if mutate:
        rows = mutate(rows)
    path.write_text("\n".join(rows) + "\n")


def test_complete_trace_retains_all_float_images(tmp_path):
    path = tmp_path / "trace.csv"
    fixture(path)
    out = read_trace(path, nx=2, nz=3, nsteps=1, it0=2)
    assert set(out) == set(PHASES)
    assert all(a.dtype == np.float32 and a.shape == (1, 17, 2, 3) for a in out.values())
    assert all(np.all(a.view(np.uint32) == 0x3F800000) for a in out.values())


@pytest.mark.parametrize("mutate", [
    lambda rows: rows[:-1],
    lambda rows: rows + rows[:1],
    lambda rows: [rows[0].replace("sediment", "unknown")] + rows[1:],
    lambda rows: [rows[0].replace(",2,1,1,1,1,", ",2,1,1,0,1,")] + rows[1:],
    lambda rows: [rows[0].replace("3F800000", "7FC00000", 1)] + rows[1:],
    lambda rows: [rows[0].replace("3F800000", "invalid", 1)] + rows[1:],
    lambda rows: [rows[0].replace(",2,1,1,", ",2,2,1,")] + rows[1:],
])
def test_corrupt_trace_refuses(tmp_path, mutate):
    path = tmp_path / "trace.csv"
    fixture(path, mutate)
    with pytest.raises(ValueError):
        read_trace(path, nx=2, nz=3, nsteps=1, it0=2)
