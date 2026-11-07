"""The P3 oracle command must fail when proof is missing or differs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "tools/p3_wrf461_oracle/compare.py"


def fixture(tmp_path):
    fix, ref, cand = (tmp_path / s for s in ("fixtures", "ref", "cand"))
    for p in (fix, ref, cand):
        p.mkdir()
    case = dict(name="interaction", ncol=1, nk=1, nsteps=1)
    (fix / "index.json").write_text(json.dumps(dict(cases=[case], out3d=["qi1", "qi2"])))
    words = np.ones(14, dtype="<f4")  # two frames: two volume + five surface fields
    words.tofile(ref / "interaction.ref")
    words.tofile(cand / "interaction.ref")
    return fix, ref, cand


def run(paths):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, str(COMPARE), *map(str, paths)],
                          env=env, capture_output=True, text=True)


def test_exact_comparison_succeeds(tmp_path):
    assert run(fixture(tmp_path)).returncode == 0


def test_one_bit_change_fails(tmp_path):
    paths = fixture(tmp_path)
    p = paths[2] / "interaction.ref"
    words = np.fromfile(p, dtype="<u4")
    words[8] ^= np.uint32(1)  # frame 1, category 2
    words.tofile(p)
    result = run(paths)
    assert result.returncode != 0, result.stdout
    assert '"claim_differing": 1' in result.stdout


def test_missing_candidate_fails(tmp_path):
    paths = fixture(tmp_path)
    (paths[2] / "interaction.ref").unlink()
    result = run(paths)
    assert result.returncode != 0, result.stdout


def test_empty_manifest_fails(tmp_path):
    paths = fixture(tmp_path)
    (paths[0] / "index.json").write_text(json.dumps(dict(cases=[])))
    result = run(paths)
    assert result.returncode != 0, result.stdout
