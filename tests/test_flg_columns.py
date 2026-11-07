from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import numpy as np
import pytest
from woof.core.flg import load_tables, TABLE_ROOT, column_primitives

FIXTURE=Path(__file__).parent/'fixtures'/'flg'/'primitives.npz'
CHECK_PATH=Path(__file__).resolve().parents[1]/'tools'/'flg_wrf461_oracle'/'check.py'
spec=importlib.util.spec_from_file_location('flg_word_check',CHECK_PATH)
check=importlib.util.module_from_spec(spec); spec.loader.exec_module(check)

def test_pinned_tables_and_shapes():
    tables=load_tables()
    assert len(tables)==198
    assert tables['band_new.coeco2_4_new'].shape==(3,11,20)
    assert tables['module_ozone.pres'].shape==(75,5)
    assert tables['planck.nd'].dtype==np.float32

def test_coefficient_bit_mutation_refused(tmp_path):
    shutil.copytree(TABLE_ROOT,tmp_path/'tables')
    root=tmp_path/'tables'; metadata=json.loads((root/'manifest.json').read_text())
    data=bytearray((root/'tables.bin').read_bytes())
    data[metadata['tables']['band_new.hk_1_new']['offset']+4]^=1
    (root/'tables.bin').write_bytes(data)
    with pytest.raises(ValueError,match='changed k-distribution or optics'):
        load_tables(root)

def test_table_manifest_mutation_refused(tmp_path):
    shutil.copytree(TABLE_ROOT,tmp_path/'tables'); root=tmp_path/'tables'
    m=json.loads((root/'manifest.json').read_text()); m['tables']['module_ozone.pres']['shape']=[5,75]
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='manifest differs'):
        load_tables(root)

@pytest.mark.parametrize('value',['7fc00000','7f800000','ff800000'])
def test_word_checker_refuses_nonfinite(tmp_path,value):
    p=tmp_path/'words.csv'; p.write_text('case,field,index,bits\n1,OLR,1,'+value+'\n')
    with pytest.raises(ValueError,match='nonfinite'):
        check.load_words(p)

def test_word_checker_refuses_duplicate(tmp_path):
    p=tmp_path/'words.csv'; p.write_text('case,field,index,bits\n1,OLR,1,00000000\n1,OLR,1,00000000\n')
    with pytest.raises(ValueError,match='duplicate'):
        check.load_words(p)

def test_word_checker_orders_negative_words():
    assert check.ordered(0xbf800000)+1==check.ordered(0xbf7fffff)
    assert check.ordered(0x80000000)+1==check.ordered(0x00000000)

def test_fixture_provenance():
    receipt=json.loads(FIXTURE.with_suffix('.json').read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()==receipt['fixture_sha256']
    assert receipt['columns']==64
    with np.load(FIXTURE) as f:
        for field,row in receipt['fields'].items():
            assert hashlib.sha256(f[field].tobytes()).hexdigest()==row['sha256']
            assert np.all(np.isfinite(f[field]))



def test_cloud_fixture_provenance():
    path=FIXTURE.with_name('cloud_optics.npz'); receipt=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['fixture_sha256']
    assert receipt['columns']==64 and len(receipt['routines'])==11
    with np.load(path) as f:
        for field,row in receipt['fields'].items():
            assert hashlib.sha256(f[field].tobytes()).hexdigest()==row['sha256']
            assert np.all(np.isfinite(f[field]))






def test_compiler_zero_initialized_ice6_delta_table_is_present():
    tables=load_tables()
    assert tables['ice6.dps'].shape==(4,6)
    np.testing.assert_array_equal(tables['ice6.dps'].view(np.uint32),np.zeros((4,6),dtype=np.uint32))


def test_gas_helper_fixture_provenance():
    path=FIXTURE.with_name('gas_helpers.npz'); receipt=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['fixture_sha256']
    assert receipt['columns']==64
    with np.load(path) as f:
        for field,row in receipt['fields'].items():
            assert hashlib.sha256(f[field].tobytes()).hexdigest()==row['sha256']
            assert np.all(np.isfinite(f[field]))


def test_full_gas_driver_fixture_provenance_and_point_coverage():
    path=FIXTURE.with_name('gas_driver.npz'); receipt=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['fixture_sha256']
    assert receipt['columns']==64 and receipt['old_points']==121 and receipt['new_points']==1053
    from woof.core.flg import gas_points
    with np.load(path) as f:
        np.testing.assert_array_equal(gas_points(),f['points'])
        for field,row in receipt['fields'].items():
            assert hashlib.sha256(f[field].tobytes()).hexdigest()==row['sha256']
            assert np.all(np.isfinite(f[field]))


def test_stream_fixture_provenance():
    path=FIXTURE.with_name('streams.npz'); receipt=json.loads(path.with_suffix('.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['fixture_sha256']
    assert receipt['columns']==64 and len(receipt['routines'])==4
    with np.load(path) as f:
        for field,row in receipt['fields'].items():
            assert hashlib.sha256(f[field].tobytes()).hexdigest()==row['sha256']
            assert np.all(np.isfinite(f[field]))
