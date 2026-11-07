"""Replay real native WRF ACM2/PX inputs through the shipped strict kernel."""
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys

import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.gpu
def test_real_native_acm2_px_capture_is_bitwise(tmp_path, monkeypatch):
    from woof.numerics import is_strict
    if not is_strict():
        pytest.skip("native word replay requires GPUWM_NUMERICS=strict")
    root = Path(__file__).resolve().parents[1]
    fixture = root / "tests/data/oracles/acm2/native-px"
    manifest = json.loads((fixture / "manifest.json").read_text())
    for row in manifest["files"]:
        compressed = (fixture / row["file"]).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == row["gzip_sha256"]
        raw = gzip.decompress(compressed)
        assert len(raw) == row["raw_bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
        (tmp_path / row["file"].removesuffix(".gz")).write_bytes(raw)
    receipt = tmp_path / "replay.json"
    tool = root / "tools/acm2_wrf461_oracle/replay_native.py"
    monkeypatch.setattr(sys, "argv", [str(tool), str(tmp_path),
                                    "--receipt", str(receipt)])
    runpy.run_path(str(tool), run_name="__main__")
    result = json.loads(receipt.read_text())
    assert result["words"] == manifest["word_count"] == 32157
    assert result["columns"] == 81
    assert result["different"] == 0
