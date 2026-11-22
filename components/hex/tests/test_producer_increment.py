from types import SimpleNamespace
import pytest
from woof.hex.producer_increment import increment_layout, native_scalar_names

MASS=('qv','qc','qr','qi','qs','qg')

@pytest.mark.parametrize('pbl,extra',[('bl_mynn',('ni',)),('bl_ysu',())])
def test_actual_mpas_wsm6_registry_owns_number_inventory(pbl,extra):
    layout=SimpleNamespace(fields=[{'pool':'state','name':'scalars','components':(*MASS,*extra)}])
    result=increment_layout(route='mpas',microphysics='mp_wsm6',pbl=pbl,
                            native_scalars=native_scalar_names(layout))
    assert result['scalars']==list((*MASS,*extra))
    assert result['cell_interface_level']==['w']
    assert result['edge_mass_level']==['u']
    assert bool(result['sgs_native_fields'])==bool(extra)

def test_actual_arwen_wsm6_does_not_adopt_schema_number_placeholders():
    result=increment_layout(route='arwen',microphysics='mp_wsm6',pbl='mynnedmf')
    assert result['scalars']==list(MASS)
    assert result['sgs_native_fields']==['cldfrac_bl','qc_bl','qi_bl']

def test_thompson_numbers_are_producer_row_controls():
    result=increment_layout(route='arwen',microphysics='mp_thompson',pbl='ysu')
    assert result['scalars']==list((*MASS,'ni','nr'))
    assert result['sgs_native_fields']==[]

def test_missing_registry_and_unknown_authority_refuse():
    with pytest.raises(ValueError,match='Registry'):
        increment_layout(route='mpas',microphysics='mp_tempo',pbl='bl_mynn')
    with pytest.raises(ValueError,match='active Registry'):
        native_scalar_names(SimpleNamespace(fields=[]))
    with pytest.raises(ValueError,match='no native SGS'):
        increment_layout(route='arwen',microphysics='mp_wsm6',pbl='unknown')
