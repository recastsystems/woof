"""Original precision at the native mesh producer/consumer boundary."""
import json
import numpy as np
import pytest
from woof.hex import native_geometry as ng
from test_native_geometry import native


@pytest.mark.parametrize('byteorder', ['<', '>'])
@pytest.mark.parametrize('precision', ['f4', 'f8'])
def test_original_precision_roundtrip_preserves_typed_words(native, byteorder, precision):
    arrays, request, valid = native
    original = {n: a.astype(byteorder + ('i4' if n.startswith('index') else precision))
                for n, a in arrays.items()}
    if precision == 'f8':
        original['latCell'][0] += 1.e-10  # Deliberately disappears in native RKIND.
    before = {n: a.tobytes() for n, a in original.items()}
    pin = ng.source_identity(original)
    document = json.loads(request.read_text())
    document['source_identity'] = pin
    request.write_text(json.dumps(document))
    record = ng.seal(arrays, request, valid, 3)
    assert ng.validate(arrays, record, request, valid) == record
    assert ng.validate_template(arrays, record, original) == pin
    assert record['source_identity'] == pin
    assert before == {n: a.tobytes() for n, a in original.items()}
    for n in arrays:
        assert pin['fields'][n]['original'] == ng.fingerprint(original[n])
        assert pin['fields'][n]['native'] == ng.fingerprint(arrays[n])


@pytest.mark.parametrize('change', ['sub-rkind', 'dtype', 'order', 'signed-zero', 'native', 'missing', 'missing-native'])
def test_template_refuses_changed_original_or_native_words(native, change):
    arrays, _, _ = native
    original = {n: a.copy() for n, a in arrays.items()}
    original['latCell'] = original['latCell'].astype('f8')
    original['latCell'][0] = -0.0
    arrays['latCell'][0] = -0.0
    record = {'source_identity': ng.source_identity(original)}
    if change == 'sub-rkind':
        original['latCell'][1] = np.nextafter(original['latCell'][1], 1.)
    elif change == 'dtype':
        original['latCell'] = original['latCell'].astype('f4')
    elif change == 'order':
        original['indexToCellID'] = original['indexToCellID'][::-1]
    elif change == 'signed-zero':
        original['latCell'][0] = 0.0
    elif change == 'native':
        arrays['latCell'][1] = np.nextafter(arrays['latCell'][1], np.float32(1.))
    elif change == 'missing-native':
        del arrays['latCell']
    else:
        del original['lonEdge']
    with pytest.raises(ValueError, match='original typed mesh'):
        ng.validate_template(arrays, record, original)


@pytest.mark.parametrize('bad', ['float-ids', 'wide-ids', 'unsigned-ids', 'duplicate-ids',
                                'zero-id', 'nan', 'overflow', 'masked', 'rank', 'extent'])
def test_source_identity_refuses_unsupported_native_source(native, bad):
    arrays, _, _ = native
    if bad == 'float-ids': arrays['indexToCellID'] = arrays['indexToCellID'].astype('f4')
    elif bad == 'wide-ids': arrays['indexToCellID'] = arrays['indexToCellID'].astype('i8')
    elif bad == 'unsigned-ids': arrays['indexToCellID'] = arrays['indexToCellID'].astype('u4')
    elif bad == 'duplicate-ids': arrays['indexToCellID'][0] = arrays['indexToCellID'][1]
    elif bad == 'zero-id': arrays['indexToCellID'][0] = 0
    elif bad == 'nan': arrays['latCell'][0] = np.nan
    elif bad == 'overflow': arrays['latCell'] = np.full(3, 1.e300, 'f8')
    elif bad == 'masked': arrays['latCell'] = np.ma.array(arrays['latCell'], mask=[1, 0, 0])
    elif bad == 'rank': arrays['latCell'] = arrays['latCell'][None, :]
    else: arrays['latCell'] = arrays['latCell'][:-1]
    with pytest.raises(ValueError, match='original typed mesh'):
        ng.source_identity(arrays)


@pytest.mark.parametrize('bad', ['null', 'schema', 'field', 'native', 'original-dtype', 'original-shape', 'hash'])
def test_seal_refuses_invalid_original_precision_declaration(native, bad):
    arrays, request, valid = native
    pin = ng.source_identity(arrays)
    if bad == 'null': pin = None
    elif bad == 'schema': pin['schema'] = 'unknown'
    elif bad == 'field': del pin['fields']['lonEdge']
    elif bad == 'native': pin['fields']['latCell']['native']['sha256'] = '0' * 64
    elif bad == 'original-dtype': pin['fields']['latCell']['original']['dtype'] = '<i4'
    elif bad == 'original-shape': pin['fields']['latCell']['original']['shape'] = [4]
    else: pin['fields']['latCell']['original']['sha256'] = 'not a hash'
    document = json.loads(request.read_text())
    document['source_identity'] = pin
    request.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='original typed mesh'):
        ng.seal(arrays, request, valid, 3)


def test_rebinding_original_pin_cannot_reuse_previous_seal(native):
    arrays, request, valid = native
    original = {n: a.copy() for n, a in arrays.items()}
    original['latCell'] = original['latCell'].astype('f8')
    document = json.loads(request.read_text())
    document['source_identity'] = ng.source_identity(original)
    request.write_text(json.dumps(document))
    record = ng.seal(arrays, request, valid, 3)
    original['latCell'][0] += 1.e-10
    document['source_identity'] = ng.source_identity(original)
    request.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='binding differs'):
        ng.validate(arrays, record, request, valid)
