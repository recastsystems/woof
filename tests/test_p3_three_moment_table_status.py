"""The public inspector and offline repair must see the selected ice table."""
from argparse import Namespace
import shutil

import pytest

from woof import tables
from woof.core.p3_tables import TABLE_1_3MOM_ASSET, packaged_p3_table_root
from woof.table_assets import fetch_tables_main


def test_three_moment_inspection_and_offline_repair(tmp_path, monkeypatch):
    root = tmp_path / "tables"
    monkeypatch.setenv("WOOF_TABLE_ROOT", str(root))
    missing = tables.status({"mp_physics": 53})
    assert len(missing.missing) == 1
    assert missing.missing[0].path.name == TABLE_1_3MOM_ASSET.packaged_filename
    with pytest.raises(tables.MissingTableAssets, match="--schemes p3-3mom"):
        tables.ensure({"mp_physics": 53}, fetch=False)
    source = tmp_path / "source"
    source.mkdir()
    shutil.copyfile(packaged_p3_table_root() / TABLE_1_3MOM_ASSET.packaged_filename,
                    source / TABLE_1_3MOM_ASSET.packaged_filename)
    args = Namespace(schemes=["p3-3mom"], from_dir=str(source), replace_damaged=False)
    assert fetch_tables_main(args) == 0
    repaired = tables.ensure({"mp_physics": 53}, fetch=False)
    assert repaired.ready and repaired.present == missing.needed
    # A parseable copy of the wrong ice table cannot pass the public pin.
    path = repaired.present[0].path
    damaged = bytearray(path.read_bytes())
    damaged[-50] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(tables.CorruptTableAssets, match="--schemes p3-3mom"):
        tables.ensure({"mp_physics": 53}, fetch=False)
    assert fetch_tables_main(args) == 2
    assert path.read_bytes() == damaged
    args.replace_damaged = True
    assert fetch_tables_main(args) == 0
    assert tables.ensure({"mp_physics": 53}, fetch=False).ready


def test_two_moment_requirement_and_packed_precedence_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOF_P3_TABLE_ROOT", str(tmp_path))
    before = tables.status({"mp_physics": 50}).needed
    assert len(before) == 1 and before[0].path.name.endswith("2momI")
    # rw-p3 prefers xz whenever present, even beside a damaged raw table.
    packed = tmp_path / TABLE_1_3MOM_ASSET.packaged_filename
    shutil.copyfile(packaged_p3_table_root() / packed.name, packed)
    raw = tmp_path / TABLE_1_3MOM_ASSET.filename
    raw.write_text("stale text")
    assert tables.ensure({"mp_physics": 53}, fetch=False).ready
    assert tables.status({"mp_physics": 50}).needed == before
