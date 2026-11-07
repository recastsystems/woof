"""Selector, ingestion and state metadata for the shared Ferrier variant."""
from types import SimpleNamespace
import numpy as np
import pytest
from woof import config
from woof.core import fer_hires
from woof.physics_registry import component_options, implemented_selector_values

def test_selector_15_is_a_separate_variant_of_the_shared_kernel():
    opts = component_options("microphysics")
    ids = implemented_selector_values("microphysics")
    a, b = opts[ids[5]], opts[ids[15]]
    assert a["asset_requirements"] == b["asset_requirements"]
    assert fer_hires.variant_for(5) == "state_fractions"
    assert fer_hires.variant_for(15) == "advected_rime"
    assert fer_hires.package_for(15)["transported"] == ["qi", "qrimef"]
    assert fer_hires.package_for(15)["state_fields"] == ["qi", "qrimef"]
    assert 15 in config.MP_PHYSICS_ACCEPTED
    assert fer_hires.package_for(15)["no_fmad"] is False
    assert not fer_hires.package_for(5).get("no_fmad", False)

def test_wrfinput_requires_the_advected_scalar_and_no_snow():
    from woof.ingest.wrfinput import active_moisture_inventory, MOISTURE_MAP
    required, allowed = active_moisture_inventory(SimpleNamespace(moist=True, mp_physics=15))
    assert {"QICE", "QRIMEF"} <= required
    assert "QSNOW" not in allowed and "QGRAUP" not in allowed
    assert "CWM" not in allowed
    assert MOISTURE_MAP["QRIMEF"] == "qrimef"

def test_cold_start_and_restart_keep_qrimef():
    from woof.ingest.microphysics_cold_start import source_absent_microphysics
    from woof.state_serialization_contract import STATE_SERIALIZED_ATTRS
    assert source_absent_microphysics(SimpleNamespace(mp_physics=15)) == (("QRIMEF",), {"qrimef": 0.0})
    assert "qrimef" in STATE_SERIALIZED_ATTRS

def test_radiation_has_the_ferrier_variant_row():
    from woof.physics_registry import consumer_rows_by_selector
    rows = consumer_rows_by_selector("microphysics", "cloud_optics")
    assert rows[15] == rows[5]

@pytest.mark.gpu
def test_state_transports_rime_without_adding_water_mass():
    from woof.core.state import DomainState
    from woof.core.moist import extra_moist_species, WRF_MOIST_ARRAY_SPECIES
    from woof.core.device_inventory import state_array_shapes
    cfg = config.RunConfig(nx=8, ny=6, nz=4, dx=12000., dy=12000.,
                           ztop=10000., dt=1., run_seconds=2., moist=True, mp_physics=15)
    state = DomainState(cfg, array_module=np)
    assert extra_moist_species(state) == ("qi", "qrimef")
    assert state.qrimef.shape == (4, 6, 8)
    assert hasattr(state, "qrimef0")
    assert getattr(state, "qs", None) is None
    assert getattr(state, "f_rimef", None) is None
    assert "qrimef" not in WRF_MOIST_ARRAY_SPECIES
    shapes = state_array_shapes(cfg)
    assert shapes["qrimef"] == state.qrimef.shape
    assert shapes["qrimef0"] == state.qrimef0.shape
    from woof.io.restart import state_manifest, classify_state_attr
    assert classify_state_attr("_scheme_transport_fields") == "infra"
    assert classify_state_attr("qrimef") == "serialize"
    assert classify_state_attr("qrimef0") == "rebuild"
    assert "state/qrimef" in state_manifest(state)


@pytest.mark.gpu
def test_shared_unflushed_loader_preserves_subnormal_results():
    import cupy as cp
    from woof.core.kernels import load_module_no_ftz
    a = np.array([1.e-30, 1.e-39], dtype=np.float32)
    b = np.array([1.e-10, .5], dtype=np.float32)
    out = cp.empty(2, dtype=cp.uint32)
    load_module_no_ftz("ftz_probe").get_function("ftz_probe_mul_op")(
        (1,), (32,), (cp.asarray(a.view(np.uint32)),
                     cp.asarray(b.view(np.uint32)), out, np.int32(2)))
    assert np.array_equal(out.get(), (a*b).view(np.uint32))
    assert np.all(out.get() > 0)


@pytest.mark.gpu
def test_zero_out_all_uses_qrimef_as_the_first_scalar():
    import cupy as cp
    from woof.core.state import DomainState
    from woof.core.microphysics import microphysics_zero_out
    cfg = config.RunConfig(nx=8, ny=6, nz=4, dx=12000., dy=12000.,
                           ztop=10000., dt=1., run_seconds=2., moist=True,
                           mp_physics=15, mp_zero_out=1, mp_zero_out_all=1,
                           mp_zero_out_thresh=1.e-8)
    state = DomainState(cfg)
    state.qi[...] = np.float32(1.e-10)
    state.qrimef[...] = np.float32(1.e-10)
    microphysics_zero_out(state, cfg)
    assert bool(cp.all(state.qi == 0))
    assert bool(cp.all(state.qrimef == np.float32(1.e-10)))


@pytest.mark.parametrize("selector", [0, 1, 5, 6, 15])
def test_device_shortwave_guard_call_keeps_supported_selectors(selector):
    from woof.core.rrtmg_legacy_prep import _mp_guard
    _mp_guard(selector, allowed_extra=(85,))


def test_device_shortwave_guard_still_names_the_hwrf_breakage():
    from woof.core.rrtmg_legacy_prep import _mp_guard
    with pytest.raises(NotImplementedError, match="HWRF/NMM-only"):
        _mp_guard(85, allowed_extra=(85,))



def test_stock_export_has_qrimef_metadata():
    from woof.wrf_direct import _PACKAGE_FIELD_METADATA
    from woof.wrf_physics_inventory import _INVENTORIES
    assert _PACKAGE_FIELD_METADATA["QRIMEF"] == ("rime factor * qi", "kg kg-1")
    row = _INVENTORIES[15]
    assert [f.netcdf_name for f in row.wrfinput_fields] == [
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QRIMEF"]
