from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from woof.hex.cycle.land_state import LAND_FIELDS, LAND_BY_NAME, require_land, validate_land


def test_every_noah_output_has_a_contract_row():
    from woof.hex.mpasphys.noah_v842 import _MPAS_OUTPUT_FIELDS
    assert set(_MPAS_OUTPUT_FIELDS) <= set(LAND_BY_NAME)


def test_default_history_publishes_land_memory(tmp_path):
    from woof.hex.mpasphys.forecast_v842 import write_default_streams
    root=ET.parse(Path(__file__).resolve().parents[1]/'../../woof/hex/data/mpasphys-framework/src/core_atmosphere/Registry.xml').getroot()
    fields=[dict(v.attrib) for v in root.findall('./var_struct/var')]
    layout=SimpleNamespace(root=root,fields=fields)
    write_default_streams(layout,tmp_path/'streams.atmosphere',history_minutes=60)
    members=(tmp_path/'stream_list.atmosphere.output').read_text().splitlines()
    assert {f.name for f in LAND_FIELDS} <= set(members)


def test_pool_input_request_reads_marked_diagnostic_memory():
    from woof.hex.mpasphys.pool_v842 import RegistryLayout
    fields=[dict(pool='diag_physics',name=n,kind='real',dimensions=['nCells'],shape=[4],components=[],initial='0',required=True)
            for n in ('canwat','snotime')]
    layout=SimpleNamespace(root=ET.fromstring('<registry><streams/></registry>'),fields=fields,
                           land_state_fields=('canwat','snotime'))
    request=RegistryLayout.request(layout,[])
    assert all(f['read_input'] and f['required'] for f in request['fields'])


def test_no_soil_field_can_be_missing():
    complete={f.name for f in LAND_FIELDS}
    for name in (f.name for f in LAND_FIELDS if f.required):
        with pytest.raises(ValueError,match='lacks '+name):
            require_land({'smois'},complete-{name})


def test_nan_and_shape_errors_cannot_broadcast():
    fields={'smois':np.ones((3,4))*.2,'sh2o':np.ones((3,1))*.1,'tslb':np.ones((3,4))*280}
    with pytest.raises(ValueError,match='identical'):validate_land(fields,{})


def test_optional_surface_memory_is_refused_until_supported():
    from woof.hex.cycle.land_state import admit_warm_configuration, UNSUPPORTED_WARM_SURFACE
    admit_warm_configuration({})
    for option in UNSUPPORTED_WARM_SURFACE:
        with pytest.raises(ValueError,match='surface-service state'):
            admit_warm_configuration({option:True})


def test_warm_initialization_restores_carried_arrays_after_driver():
    from woof.hex.cycle.land_state import initialize_preserving_land
    arrays = {'smois': np.array([[.2, .3]], dtype='f4'),
              'canwat': np.array([.01], dtype='f4'),
              'uncarried': np.array([1.], dtype='f4')}
    original = {n: a.copy() for n, a in arrays.items()}
    def initialize(session):
        for value in arrays.values():
            value.fill(0)
    session = SimpleNamespace(
        layout=SimpleNamespace(land_state_fields=('smois', 'canwat'),
            fields=[{'pool': 'p', 'name': n} for n in arrays]),
        driver=SimpleNamespace(initialize=initialize),
        field=lambda pool, name: arrays[name],
        put=lambda pool, name, value: arrays[name].__setitem__(Ellipsis, value))
    initialize_preserving_land(session)
    for name in ('smois', 'canwat'):
        np.testing.assert_array_equal(arrays[name], original[name])
    assert arrays['uncarried'][0] == 0
