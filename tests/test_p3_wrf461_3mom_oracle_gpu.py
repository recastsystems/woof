"""WRF 4.6.1 mp53 full-field gate, including a planted closure-bit defect."""
import json
from pathlib import Path
import pytest
from conftest import requires_gpu
from tools.p3_wrf461_oracle import make_columns, run_woof
from tools.p3_wrf461_oracle.hash_gate import verify

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/data/oracles/p3-3mom/full-fields.json").read_text())

@pytest.fixture(scope="module")
def columns(tmp_path_factory):
    path = tmp_path_factory.mktemp("p3-three-moment")
    make_columns.main(str(path), variant=53)
    return path, json.loads((path / "index.json").read_text())

def run(columns, monkeypatch, output, arm, plant=False):
    path,index = columns
    argv = ["run_woof.py", str(path), str(output), "--arms", arm]
    if plant: argv.append("--plant-mu-bit")
    monkeypatch.setattr("sys.argv", argv)
    assert run_woof.main() == 0
    return verify(index, REFERENCE, output / arm)

@requires_gpu
@pytest.mark.parametrize("arm", ["unfused","fused","levels","sedlevels"])
def test_all_fields_and_frames_match_wrfs_columns(columns,monkeypatch,tmp_path,arm):
    result = run(columns,monkeypatch,tmp_path,arm)
    assert result["columns"] >= 64
    assert result["computed_words"] == 2907020
    assert result["pass"], result["failures"][:10]

@requires_gpu
def test_planted_mu_coefficient_bit_fails_same_gate(columns,monkeypatch,tmp_path):
    healthy = run(columns,monkeypatch,tmp_path / "healthy","sedlevels")
    assert healthy["pass"]
    planted = run(columns,monkeypatch,tmp_path / "planted","sedlevels",True)
    assert not planted["pass"]
    assert any(row.get("field") == "qzi" for row in planted["failures"])
