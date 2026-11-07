"""Regressions exposed by the MP52 real restart and nested checkpoint runs."""
from types import SimpleNamespace

import numpy as np

from tilestream import checkpoint, output, physics_inventory


def test_two_category_output_scatters_canonical_ice_diagnostics():
    slots = {name: np.full((2, 3, 4), i + 1, np.float32)
             for i, name in enumerate(("p3_vmi", "p3_di", "p3_rhopo",
                                       "p3_vmi2", "p3_di2", "p3_rhopo2"))}
    state = SimpleNamespace(qi2=np.ones((2, 3, 4), np.float32),
                            existing_scratch=slots.get, physics=None)
    members = output.diagnostic_members(state)
    names = ("v_ice", "d_ice", "rho_ice", "v_ice2", "d_ice2", "rho_ice2")
    assert set(members) == {"diag/" + name for name in names}
    for name, slot in zip(names, slots):
        assert members["diag/" + name] is slots[slot]
    state.qi2 = None
    assert output.diagnostic_members(state) == {}


def test_solar_split_provenance_survives_the_streamed_checkpoint_boundary():
    driver = SimpleNamespace(call_counts={"radiation": 1},
                             ysu_nan_guard_fires=0, microphysics_updates=2,
                             _swdd_from_scheme=True)
    state = SimpleNamespace(elapsed_seconds=12.0, physics=driver)
    values = physics_inventory.carrier_scalars(state)
    assert values["swdd_from_scheme"] is True
    header = {"elapsed_seconds": 12.0,
              "driver": {"call_counts": values["call_counts"],
                         "ysu_nan_guard_fires": 0,
                         "microphysics_updates": 2,
                         "swdd_from_scheme": True}}
    restored = checkpoint.restart_scalars(header)
    driver._swdd_from_scheme = False
    physics_inventory.set_carrier_scalars(state, restored)
    assert driver._swdd_from_scheme is True
