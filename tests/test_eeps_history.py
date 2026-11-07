"""EEPS history follows native Registry publication and member slicing."""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.io.history_layout import live_state_history_fields
from woof.io.wrf_output_schema import EEPS_PBL_OUTPUT_FIELDS


@pytest.mark.parametrize("runner", ["_run_eeps", "_run_ysu", None])
def test_history_uses_the_pbl_owned_pair_only_when_eeps_is_routed(runner):
    owned = {name: np.arange(24, dtype=np.float32).reshape(3, 2, 4)
             for name in ("pek_pbl", "pep_pbl")}
    state = SimpleNamespace(
        pek_adv=np.ones((3, 2, 4), np.float32),
        pep_adv=np.ones((3, 2, 4), np.float32),
        physics=SimpleNamespace(fields=owned,
                                scheme_dispatch={"bl_pbl_physics": runner}))
    history = live_state_history_fields(state)
    for name, schema in EEPS_PBL_OUTPUT_FIELDS.items():
        assert schema.wrf_history and schema.stagger == ""
        if runner == "_run_eeps":
            assert history[schema.netcdf_name] is owned[name]
        else:
            assert schema.netcdf_name not in history
    assert "PEK_ADV" not in history and "PEP_ADV" not in history


def test_member_history_borrows_the_correct_eeps_stripe(monkeypatch):
    import woof.ensemble.batch_dycore as dycore
    from woof.ensemble.native_forecast import member_output_view

    monkeypatch.setattr(dycore, "member_domain_view",
                        lambda batch, member: SimpleNamespace())
    nz, ny, nx, members = 3, 2, 4, 3
    pair = {name: np.arange(nz * ny * nx * members, dtype=np.float32)
                  .reshape(nz, members * ny, nx)
            for name in ("pek_pbl", "pep_pbl")}
    driver = SimpleNamespace(fields=pair, scheme_dispatch={"bl_pbl_physics": "_run_eeps"},
                             output_fields=lambda: {}, surface_enabled=True)
    owners = SimpleNamespace(batch=SimpleNamespace(members=members,
                              cfg=SimpleNamespace(ny=ny, nx=nx)),
                              physics=SimpleNamespace(driver=driver))
    history = live_state_history_fields(member_output_view(owners, 1))
    for name, array in pair.items():
        actual = history[EEPS_PBL_OUTPUT_FIELDS[name].netcdf_name]
        assert actual.shape == (nz, ny, nx)
        assert actual.tobytes() == array[:, ny:2 * ny].tobytes()
        assert np.shares_memory(actual, array)
