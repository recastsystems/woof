"""CPU binding witness against the installed fork Registry, not a CUDA replay."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from woof.hex.mpasphys import Backend, SlotCall, StepFlags
from woof.hex.mpasphys import thompson_v842 as shared
from woof.hex.mpasphys.device_pools_v842 import NativeDevicePoolViews
from woof.hex.mpasphys.pool_v842 import RegistryLayout
from woof.hex.mpasphys.rrfs import thompson
from woof.hex.mpasphys.rrfs.driver_rrfs import resolve_namelist
from woof.hex.mpasphys.rrfs.framework_rrfs import FrameworkPhysicsRRFS, registry_source


def test_thompson_binds_full_fork_column_without_v842_cutoff():
    source = registry_source()
    root = ET.parse(source / 'src/core_atmosphere/Registry.xml').getroot()
    dimensions = {d.get('name'): 1 for d in root.findall('./dims/dim')
                  if not d.get('definition')}
    dimensions.update(nCells=4, nEdges=6, nVertices=4, nVertLevels=3,
                      maxEdges=3, maxEdges2=6, vertexDegree=3, nlcat=20, nscat=16)
    text = "&physics\nconfig_physics_suite='rrfs_convection_permitting'\n/\n"
    config = resolve_namelist(text, default_backend='hex-cuda')
    layout = RegistryLayout(source, config, text, dimensions)
    assert not any(r['pool_name'] == 'mp_top_level' for r in layout.fields)
    with FrameworkPhysicsRRFS(layout=layout) as native:
        for row in layout.fields:
            if row['kind'] == 'character':
                continue
            dtype = np.float32 if row['kind'] == 'real' else np.int32
            for level in range(1, row['nlevels'] + 1):
                native.put_field(row['pool'], row['pool_name'],
                                 np.full(row['shape'], level, dtype),
                                 level=level, nlevels=row['nlevels'])
        for name, value in dict(ims=1, ime=4, jms=1, jme=1, kms=1, kme=4).items():
            native.put_variable(name, np.array(value, np.int32))
        native.call('allocate_microphysics')
        provider = NativeDevicePoolViews(SimpleNamespace(native=native, layout=layout), np)
        provider.level = 2
        provider.resident = True
        # Binding uses the certificate only; no lookup or maths are executed.
        provider.thompson_v842_resources = dict(
            tables={'_file_sha256': dict(shared.TABLE_FILE_SHA256)},
            device_math_source='binding-only; no CUDA execution')
        call = SlotCall('microphysics', shared.SCHEME, Backend.HEX_CUDA, 1, 0., 10.,
                        StepFlags(diagnostics=True), None, {'config_bucket_rainnc': 500.})
        # The shared default still requires the v842 Registry cutoff.
        with pytest.raises(KeyError, match='tend_physics.mp_top_level'):
            shared.bind_native_pools(provider, call)
        packet = thompson.bind_native_pools(provider, call)['thompson_v842']
        assert packet['mp_top_level'] == packet['model']['qv'].shape[0] == 3
        assert packet['model']['qv'].shape == (3, 4)
        np.testing.assert_array_equal(packet['model']['qv'], np.full((3, 4), 2, np.float32))
        assert packet['fields']['w'].shape == (3, 4)
        assert packet['diagnostics']['refl10cm'].shape == (3, 4)
