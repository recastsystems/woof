"""QNSE metadata follows the same two selectors at every public door."""
import pytest
from woof.config import RunConfig,validate_run_config,qnse_mfshconv_active

def config(**updates):
    values=dict(nx=8,ny=6,nz=40,dx=3000.,dy=3000.,ztop=16000.,dt=6.,
        run_seconds=60.,time_step_sound=4,moist=True,mp_physics=6,
        sf_sfclay_physics=4,sf_surface_physics=2,bl_pbl_physics=4)
    values.update(updates)
    return RunConfig(**values)

@pytest.mark.parametrize("mf",[0,1])
def test_matched_pair_is_admitted(mf):
    cfg=validate_run_config(config(mfshconv=mf))
    assert qnse_mfshconv_active(cfg)==bool(mf)

@pytest.mark.parametrize("values",[{"sf_sfclay_physics":1},{"bl_pbl_physics":1}])
def test_half_pair_names_the_missing_boundary_state(values):
    with pytest.raises(ValueError,match="requires"):
        validate_run_config(config(**values))

def test_mass_flux_is_inert_beside_another_pbl():
    cfg=config(bl_pbl_physics=1,sf_sfclay_physics=1,mfshconv=1)
    assert cfg.mfshconv == 0
    assert not qnse_mfshconv_active(cfg)

def test_native_import_and_inventory_cover_the_same_selector():
    from woof.namelist_import import _BL_MAP,_SFCLAY_ALLOWED
    from woof.core.physics_inventory import PHYSICS_SLOT_DISPATCH
    from woof.core.preflight import physics_array_shapes
    assert _BL_MAP[4]==(4,"QNSE","QNSE") and 4 in _SFCLAY_ALLOWED
    assert PHYSICS_SLOT_DISPATCH["bl_pbl_physics"][4]=="_run_qnse_pbl"
    shapes=physics_array_shapes(config())
    for name in ("tke_myj","el_myj","massflux_edkf","rc_mf","akhs","akms"):
        assert "fields/"+name in shapes,name


def test_mass_flux_parameter_is_registry_metadata():
    from woof.physics_registry import physics_registry
    row=physics_registry()["parameters"]["mfshconv"]
    assert row["enum"]==[0,1] and row["default"]==1
    assert row["read_when"]=={"bl_pbl_physics":4}


def test_surface_table_hash_is_published_in_the_manifest():
    from woof.physics_registry import physics_registry
    from woof.core.myjsfc_tables import QNSE_PSI_TABLES_SHA256
    row=physics_registry()["components"]["surface_layer"]["options"]["qnse"]["extensions"]["table_manifest"]
    assert row["sha256"]==QNSE_PSI_TABLES_SHA256
    assert row["bytes"]==4*10001*4


def test_inactive_mass_flux_keeps_earlier_document_identity():
    from woof.checkpoint_identity import drop_capability_campaign_keys
    values = {"bl_pbl_physics": 1, "mfshconv": 0}
    drop_capability_campaign_keys(values)
    assert values == {"bl_pbl_physics": 1}
    active = {"bl_pbl_physics": 4, "mfshconv": 0}
    drop_capability_campaign_keys(active)
    assert active["mfshconv"] == 0
