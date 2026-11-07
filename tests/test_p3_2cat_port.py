"""Two-category P3 must preserve its Registry package through the run door."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from woof.config import RunConfig, validate_run_config


def cfg(**kw):
    return RunConfig(nx=8, ny=8, nz=10, dx=2000., dy=2000., ztop=10000.,
                     dt=10., run_seconds=20., moist=True, mp_physics=52, **kw)


def test_bare_config_admits_two_category_p3():
    value=cfg()
    assert validate_run_config(value) is value
    assert value.p3_backend == "cuda"


def test_two_category_reference_refuses_missing_physics():
    with pytest.raises(ValueError, match="category-2 collection, merge and sedimentation"):
        validate_run_config(cfg(p3_backend="reference"))


def test_state_carries_second_category_and_droplet_number():
    from woof.core.state import DomainState
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS
    from woof.io.restart import STATE_REBUILT_ATTRS
    from woof.io.history_layout import live_state_history_fields
    state=DomainState(cfg(),array_module=np)
    for field in ("qi", "ni", "nr", "qir", "qib", "nc", "qi2", "ni2", "qir2", "qib2"):
        assert getattr(state,field).shape == (10,8,8)
        assert not np.any(getattr(state,field))
        assert getattr(state,field+"0").shape == (10,8,8)
        assert field in STATE_SERIALIZED_ATTRS
        assert field+"0" in STATE_REBUILT_ATTRS
    assert not hasattr(state,"qs") and not hasattr(state,"qg")
    assert {"QICE2","QNICE2","QIR2","QIB2","QNCLOUD"} <= set(live_state_history_fields(state))


def test_table_two_matches_fortran_init_and_package_hash():
    from woof.core import p3_tables as tables
    a,b=tables.load_lookup_table_2(tables.p3_table_root())
    assert a.shape == b.shape == (5,4,25,5,4,25)
    assert a.dtype == b.dtype == np.float32
    assert hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest() == tables.FORTRAN_ITABCOLLI_SHA256
    asset=tables.TABLE_2_ASSET
    path=Path(tables.p3_table_root())/asset.filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == asset.sha256


def test_restart_identity_names_both_tables_and_count():
    from woof.io.restart import _p3_setup_identity
    old=_p3_setup_identity()
    new=_p3_setup_identity(2)
    assert "n_cat" not in old
    assert new["n_cat"] == 2
    assert len(old["assets"]) == 1 and len(new["assets"]) == 2
    assert new["assets"][0] == old["assets"][0]


def test_registry_packages_new_option_and_no_longer_refuses_it():
    from woof.physics_registry import physics_registry
    from woof.config import _P3_UNPORTED_VARIANTS, MP_PHYSICS_ACCEPTED
    from woof.wrf_physics_inventory import _INVENTORIES
    assert 52 in MP_PHYSICS_ACCEPTED and 52 not in _P3_UNPORTED_VARIANTS
    row=physics_registry()["components"]["microphysics"]["options"]["p3-mp52"]
    assert row["implemented"] and row["selectors"] == {"mp_physics":52}
    names={f.netcdf_name for f in _INVENTORIES[52].wrfinput_fields}
    assert {"QICE2","QNICE2","QIR2","QIB2","QNCLOUD"} <= names


def test_backend_scope_keeps_both_p3_variants_and_drops_other_schemes():
    from woof.core.model import scope_microphysics_run_fields
    for mp in (6,50,52):
        run=dict(mp_physics=mp,p3_backend="fused")
        scope_microphysics_run_fields(run)
        assert ("p3_backend" in run) == (mp in (50,52))


def test_public_namelist_selector_preserves_two_category_identity():
    from woof.namelist_import import _MP_MAP
    assert _MP_MAP[52] == (52,"P3 two-category","P3 two-category")


def test_second_category_wrfinput_geometry_and_state_names_are_closed():
    from woof.ingest.wrfinput import WRFINPUT_DIMENSIONS, active_moisture_map
    mapping=active_moisture_map(cfg())
    for name,field in (("QICE2","qi2"),("QNICE2","ni2"),("QIR2","qir2"),("QIB2","qib2")):
        assert WRFINPUT_DIMENSIONS[name] == ("bottom_top","south_north","west_east")
        assert mapping[name] == field


def test_category_diagnostics_are_present_from_frame_zero_and_priced():
    from woof.core.state import DomainState
    from woof.io.history_layout import live_state_history_fields, produced_history_shapes
    state=DomainState(cfg(),array_module=np)
    names={"v_ice","d_ice","rho_ice","v_ice2","d_ice2","rho_ice2"}
    fields=live_state_history_fields(state)
    assert names <= set(fields)
    assert names <= set(produced_history_shapes(cfg()))
    for name in names:
        assert fields[name].shape == (10,8,8)
        assert not np.any(fields[name])


def test_reference_entry_refuses_two_categories_before_touching_device_state():
    from types import SimpleNamespace
    from woof.core.p3 import apply_reference
    # Direct callers can bypass config validation. A device-like qv must
    # never make the one-category reference accept an MP52 state.
    state = SimpleNamespace(qv=SimpleNamespace(get=lambda: None))
    with pytest.raises(ValueError, match="category-2 sedimentation"):
        apply_reference(state, cfg(), 10.0)


def test_table_repair_inventory_includes_category_two_without_changing_mp50():
    from dataclasses import replace
    from woof.tables import _files
    from woof.core.p3_tables import TABLE_1_2MOM_ASSET, TABLE_2_ASSET

    names = lambda config: {asset.path.name for asset in _files("p3", config)}
    assert names(replace(cfg(), mp_physics=50)) == {TABLE_1_2MOM_ASSET.filename}
    expected = {TABLE_1_2MOM_ASSET.filename, TABLE_2_ASSET.filename}
    assert names(cfg()) == expected
    assert names(None) == expected


def test_analysis_reflectivity_dispatches_to_p3_for_both_categories(monkeypatch):
    import sys
    from types import SimpleNamespace
    from woof.core import p3_device, state_reflectivity

    state = object()
    result = object()
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    def p3_reflectivity(value):
        assert value is state
        return result
    monkeypatch.setattr(p3_device, "reflectivity", p3_reflectivity)
    assert state_reflectivity.diagnose(state, cfg()) is result


def test_kf_uses_the_native_no_snow_phase_for_two_category_p3():
    from woof.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics

    assert kf_phase_mode_for_microphysics(52) is KFPhaseMode.NO_SEPARATE_SNOW
