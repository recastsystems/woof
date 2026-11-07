"""The oracle gate must reject a changed word, missing file or missing case."""
import json
import numpy as np
import pytest
from tools.p3_wrf461_oracle.hash_gate import field_hashes, verify
from woof.config import P3_MP_PHYSICS


@pytest.fixture
def columns(tmp_path):
    case = dict(name='column', ncol=2, nk=3, nsteps=2, sha256='input')
    index = dict(out3d=['qzi', 'nc'], out2d=['rainnc'], cases=[case])
    words = np.arange(42, dtype='<f4')
    path = tmp_path / 'column.ref'
    words.tofile(path)
    reference = dict(cases={'column': dict(field_hashes(index, case, tmp_path), input_sha256='input')})
    return tmp_path, path, words, index, reference


def test_zero_ulp_gate_and_qzi_bit_sensitivity(columns):
    root,path,words,index,reference = columns
    assert verify(index,reference,root)['pass']
    words.view(np.uint32)[14] ^= 1
    words.tofile(path)
    result = verify(index,reference,root)
    assert not result['pass']
    assert result['failures'][0]['field'] == 'qzi'
    assert result['failures'][0]['step'] == 1


def test_missing_and_truncated_outputs_are_failures(columns):
    root,path,words,index,reference = columns
    path.write_bytes(b'')
    assert not verify(index,reference,root)['pass']
    path.unlink()
    assert not verify(index,reference,root)['pass']


def test_empty_or_missing_reference_cannot_pass(columns):
    root,path,words,index,reference = columns
    assert not verify(dict(index,cases=[]),reference,root)['pass']
    assert not verify(index,dict(cases={}),root)['pass']
    reference['cases']['column']['input_sha256'] = 'changed'
    assert not verify(index,reference,root)['pass']


@pytest.mark.parametrize("selector", sorted(P3_MP_PHYSICS))
def test_shared_p3_backend_remains_restart_bound(selector):
    from woof.core.model import _drop_off_scheme_run_fields
    run = dict(mp_physics=selector, p3_backend="fused", mp28_aerosol_source="climatology")
    _drop_off_scheme_run_fields(run)
    assert run["p3_backend"] == "fused"
    assert "mp28_aerosol_source" not in run


@pytest.mark.parametrize("selector", sorted(P3_MP_PHYSICS))
def test_p3_rows_name_the_full_transported_state(selector):
    from woof.config import p3_variant
    from woof.core.moist import p3_extra_species_present
    row = p3_variant(selector)
    assert row.extra_species == ({50: (),51: ("nc",),52: ("nc","qi2","ni2","qir2","qib2"),53: ("nc","qzi")}[selector])
    class State:
        pass
    state = State()
    for field in row.extra_species: setattr(state,field,object())
    assert p3_extra_species_present(state) == row.extra_species


def test_three_moment_packaged_tables_match_fortran_reads():
    import hashlib
    from woof.core import p3_tables as tables
    itab,coll = tables.load_lookup_table_1(tables.p3_table_root(),True)
    assert itab.shape == (11,5,4,50,15)
    assert coll.shape == (11,5,4,50,30,2)
    assert hashlib.sha256(itab.astype("<f4").tobytes()).hexdigest() == tables.FORTRAN_ITAB3_SHA256
    assert hashlib.sha256(coll.astype("<f4").tobytes()).hexdigest() == tables.FORTRAN_ITABCOLL3_SHA256


def test_three_moment_sedimentation_workspace_is_rebuilt():
    from woof.io.restart import classify_scratch_slot
    from woof.core.p3_device import SEDW_SLOTS_3MOM
    for name in SEDW_SLOTS_3MOM:
        assert classify_scratch_slot("p3_sed_"+name) == "rebuild"


def test_qzi_has_complete_stock_input_and_boundary_metadata():
    from woof.ingest.wrfinput import WRFINPUT_DIMENSIONS
    assert WRFINPUT_DIMENSIONS["QZI"] == WRFINPUT_DIMENSIONS["QICE"]
    from woof.wrf_direct import _load_contract,_physics_contract_bundle
    result=_physics_contract_bundle(_load_contract(),53)
    inputs={row["name"]:row for row in result["wrfinput"]["variables"]}
    boundaries={row["name"]:row for row in result["wrfbdy"]["variables"]}
    assert inputs["QZI"]["attributes"]["units"] == "m(3) kg(-1)"
    assert inputs["QZI"]["dimensions"] == inputs["QICE"]["dimensions"]
    assert "QNCLOUD" in inputs
    assert not {"QSNOW","QGRAUP"} & set(inputs)
    suffixes={name.removeprefix("QCLOUD") for name in boundaries if name.startswith("QCLOUD_B")}
    assert len(suffixes)==8
    for suffix in suffixes:
        assert "QZI"+suffix in boundaries
        assert boundaries["QZI"+suffix]["dimensions"] == boundaries["QCLOUD"+suffix]["dimensions"]
    for selector in sorted(set(P3_MP_PHYSICS) - {53}):
        names={row["name"] for row in _physics_contract_bundle(_load_contract(),selector)["wrfinput"]["variables"]}
        assert "QZI" not in names


def test_three_moment_rust_rain_tables_match_native_words():
    from pathlib import Path
    from woof.core.p3_rust import init_three_moment
    from woof.core.p3_tables import p3_table_root
    raw = np.fromfile(Path(__file__).parent / "data/oracles/p3/wrf461_rain_tables.bin", dtype="<f4")
    for i, actual in enumerate(init_three_moment(str(p3_table_root()))[2:]):
        expected = raw[i * 3000:(i + 1) * 3000].reshape(300, 10, order="F")
        assert np.array_equal(actual.view(np.uint32), np.ascontiguousarray(expected).view(np.uint32))
