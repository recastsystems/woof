"""SSiB restart carries every native state field and rebuilds device records."""
from datetime import datetime, timezone
import numpy as np
import pytest


@pytest.mark.gpu
def test_checkpoint_continuation_and_member_storage_are_independent(tmp_path):
    import cupy as cp
    from woof.core import ssib
    from woof.core.physics import initialize_physics
    from woof.core.state import DomainState
    from woof.io.restart import write_restart, restore_restart
    from test_ssib_admission import cfg

    config = cfg()

    def prepare():
        state = DomainState(config)
        initialize_physics(state, config,
            radiation_start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            radiation_latitude=35., radiation_longitude=-100.,
            landuse_dataset='SSIB', ivgtyp=7,
            soil_temperature=278., soil_moisture=.25, tsk=278.,
            snow=50., snow_depth=.2)
        fields = state.physics.fields
        for name, value in [('swdown', 400.), ('glw', 300.), ('coszen', .6),
                            ('hfx_sea', 0.), ('qfx_sea', 0.), ('lh_sea', 0.),
                            ('tsk_sea', 278.)]:
            fields[name].fill(value)
        return state

    atmosphere = {name: cp.full((config.nz, config.ny, config.nx), value, cp.float32)
        for name, value in [('dz', 50), ('u', 5), ('v', 2), ('qv', .004),
                            ('temperature', 278), ('pressure', 90000)]}
    original = prepare()
    original.physics._run_column_land_surface(atmosphere, config, 1)
    original.elapsed_seconds = config.dt
    path = write_restart(tmp_path/'ssib.npz', original, config)
    resumed = prepare()
    restore_restart(path, resumed, config)
    assert resumed.physics.column_surface_params.records is None
    for name in ssib.SSIB_REGISTRY_STATE:
        a, b = original.physics.fields[name], resumed.physics.fields[name]
        assert a.data.ptr != b.data.ptr
        np.testing.assert_array_equal(a.get().view(np.uint32), b.get().view(np.uint32))
    for state in (original, resumed):
        state.physics._run_column_land_surface(atmosphere, config, 2)
    contract = original.physics.column_surface_params.contract
    for name in ssib.STATE:
        binding = contract['state_bindings'][name]
        key, _, index = binding.partition(':')
        a, b = original.physics.fields[key], resumed.physics.fields[key]
        if index:
            a, b = a[int(index)], b[int(index)]
        np.testing.assert_array_equal(a.get().view(np.uint32), b.get().view(np.uint32),
                                      err_msg=name)
    resumed.physics.fields['ssib_ect'].fill(77.)
    assert not bool(cp.all(original.physics.fields['ssib_ect'] == 77.))
