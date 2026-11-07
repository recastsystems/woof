"""MP40 archived-parent identity must preserve its native selectors."""
import pytest
from woof.offline_child import (
    OfflineChildContractError, bind_parent_physics_from_wrf_namelist,
    _transported_source_fields,
)

@pytest.mark.parametrize('rimed', [0, 1])
def test_mp40_offline_binding_keeps_rimed_selection(tmp_path, rimed):
    path = tmp_path / 'namelist.input'
    path.write_text(f'&physics\n mp_physics=40, morr_rimed_ice={rimed}, aercu_opt=0,\n/\n')
    binding = bind_parent_physics_from_wrf_namelist(path)
    assert (binding.mp_physics, binding.morr_rimed_ice) == (40, rimed)
    assert set(_transported_source_fields(40)) == {
        'qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'nc', 'nr', 'ni', 'ns', 'ng'}

@pytest.mark.parametrize('coupling', [1, 2])
def test_mp40_offline_parent_cannot_hide_positive_coupling(tmp_path, coupling):
    path = tmp_path / 'namelist.input'
    path.write_text(f'&physics\n mp_physics=40, cu_physics=11, aercu_opt={coupling},\n/\n')
    with pytest.raises(OfflineChildContractError, match='CU11/aerosol coupling identity'):
        bind_parent_physics_from_wrf_namelist(path)
