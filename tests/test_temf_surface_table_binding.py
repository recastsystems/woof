import pytest
from woof.core import temf_constants
from woof.core.kernels import module_source

def test_surface_source_binds_the_verified_pair_table_before_use():
    source=module_source("temf_sfclay")
    table=temf_constants.cuda_table()
    assert source.count(table)==1
    assert source.index("#define TC_TEMIN")<source.index("const real TEmin = TC_TEMIN")
    assert temf_constants.table_sha256()==temf_constants.TEMF_CONSTANTS_SHA256

def test_surface_compilation_rejects_a_changed_pair_table_hash(monkeypatch):
    monkeypatch.setattr(temf_constants,"TEMF_CONSTANTS_SHA256","0"*64)
    with pytest.raises(ValueError,match="TEMF constant table hash mismatch"):
        module_source("temf_sfclay")
