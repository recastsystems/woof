import hashlib
import json
from pathlib import Path

import pytest

from woof.table_assets import read_binary_table

ROOT = Path(__file__).resolve().parents[1] / "woof/data/sfgfs"


def test_packaged_table_retains_every_pinned_wrf_word():
    data = read_binary_table(ROOT / "fpvs_table.bin")
    assert len(data) == 7503 * 8
    assert hashlib.sha256(data).hexdigest() == "bef4a54b636c57f74951560d3f5f0a65e12337cbedeae1ac17be1376f66e2235"
    assert data == (ROOT.parents[2] / "tests/data/oracles/sfgfs/fpvs_table.bin").read_bytes()


def test_changed_table_word_is_refused_before_device_upload(tmp_path):
    (tmp_path / "MANIFEST.json").write_bytes((ROOT / "MANIFEST.json").read_bytes())
    data = bytearray((ROOT / "fpvs_table.bin").read_bytes())
    data[0] ^= 1
    (tmp_path / "fpvs_table.bin").write_bytes(data)
    with pytest.raises(ValueError, match="surface fluxes"):
        read_binary_table(tmp_path / "fpvs_table.bin")
