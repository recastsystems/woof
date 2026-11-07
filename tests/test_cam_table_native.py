"""Real WRF table parity and a planted table-byte failure through the native CLI."""
import json
import os
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]
TOOL=ROOT/'tools/cam_tables/target/release/rw_cam_tables'


def oracle_root():
    root=os.environ.get('WOOF_CAM_ORACLE')
    if not root:
        pytest.skip('requires the real WRF column oracle build')
    return Path(root)


def test_native_tables_match_every_wrf_word(tmp_path):
    oracle=oracle_root()
    table_root=os.environ['WOOF_CAM_TABLE_ROOT']
    subprocess.run([TOOL,table_root,tmp_path],check=True)
    assert (tmp_path/'tables.tsv').read_bytes()==(oracle/'tables.tsv').read_bytes()
    assert (tmp_path/'tables.bin').read_bytes()==(oracle/'tables.bin').read_bytes()


def test_one_bit_table_change_is_refused(tmp_path):
    oracle_root()
    table_root=Path(os.environ['WOOF_CAM_TABLE_ROOT'])
    (tmp_path/'CAM_ABS_DATA').symlink_to(table_root/'CAM_ABS_DATA')
    values=bytearray((table_root/'CAM_AEROPT_DATA').read_bytes())
    values[15]^=1
    (tmp_path/'CAM_AEROPT_DATA').write_bytes(values)
    run=subprocess.run([TOOL,tmp_path,tmp_path/'decoded'],text=True,capture_output=True)
    assert run.returncode==1
    assert 'CAM_AEROPT_DATA: SHA-256' in run.stderr
    assert 'differs from pinned WRF table' in run.stderr
    assert not (tmp_path/'decoded').exists()


def test_native_reader_and_table_path_use_same_pins():
    from woof.table_assets import CAM3_TABLE_ASSETS
    metadata=json.loads((ROOT/'woof/data/cam3/table-assets.json').read_text())
    assert [(x.filename,x.bytes,x.sha256) for x in CAM3_TABLE_ASSETS]==[
        (x['filename'],x['size_bytes'],x['sha256']) for x in metadata['assets']]
