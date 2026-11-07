"""The LSM execution census follows its checked layout as ice evolves."""
from types import SimpleNamespace

import pytest

from woof.hex.cuda_arwen_physics_v841 import (
    PersistentTwoPhaseCudaPhysicsBackendV841 as Backend,
    _GLACIER_CUDA_PROVENANCE,
)


def backend(*, prognostic=True):
    initial = dict(sflx_land_columns=1, open_water_columns=1,
                   sea_ice_columns=2, glacier_columns=0)
    executed = dict(land=1, water=1, sea_ice=1, glacier=1)
    driver = SimpleNamespace(
        fields={"ice_volume": object()} if prognostic else {},
        noahmp_params=SimpleNamespace(_surface_layout_cache=(None, None,
                                                           {"census": executed})),
    )
    return SimpleNamespace(
        _constructor=SimpleNamespace(expected_surface_classification=lambda: initial),
        _seam=SimpleNamespace(_driver=driver,
            last_noahmp_census={**executed, "glacier_path": _GLACIER_CUDA_PROVENANCE}),
    )


def test_ice_threshold_crossing_uses_pre_lsm_layout_without_array_reads():
    b = backend()
    assert dict(Backend._validate_noahmp_census(b, require=True)) == b._seam.last_noahmp_census


@pytest.mark.parametrize("key,value", [("sea_ice", 2), ("glacier_path", "host")])
def test_executed_layout_and_cuda_provenance_still_checked(key, value):
    b = backend()
    b._seam.last_noahmp_census[key] = value
    with pytest.raises(ValueError, match="census/provenance"):
        Backend._validate_noahmp_census(b, require=True)


def test_uncoupled_surface_also_follows_executed_ice_routing():
    b = backend(prognostic=False)
    assert dict(Backend._validate_noahmp_census(b, require=True)) == b._seam.last_noahmp_census


def test_active_ice_requires_execution_layout():
    b = backend()
    b._seam._driver.noahmp_params._surface_layout_cache = None
    with pytest.raises(ValueError, match="census/provenance"):
        Backend._validate_noahmp_census(b, require=True)


def test_service_keeps_executed_counts_through_an_analysis_alarm():
    b = backend()
    b._surface_updates = SimpleNamespace(
        classification=b._constructor.expected_surface_classification(),
        physics_classification=dict(sflx_land_columns=1, open_water_columns=1,
                                    sea_ice_columns=1, glacier_columns=1))
    b._seam._driver.noahmp_params._surface_layout_cache = None
    assert dict(Backend._validate_noahmp_census(b, require=True)) == b._seam.last_noahmp_census


def test_routing_receipt_allows_category_changes_but_rejects_count_or_provenance_drift():
    from woof.hex.drivers import run_cuda_v841_full_physics_x4 as runner
    initial = dict(runner.EXPECTED_SURFACE_CLASSIFICATION)
    routed = dict(runner.EXPECTED_NOAHMP_CENSUS)
    routed["sea_ice"] -= 1
    routed["water"] += 1
    payload = dict(surface_classification=initial, noahmp_routing_census=routed,
                   last_noahmp_census=dict(routed))
    assert runner.require_arwen_v2_surface_execution(payload, executed=True, label="threshold")["last_noahmp_census"] == routed
    wrong = dict(routed, water=routed["water"] + 1)
    with pytest.raises(ValueError, match="routing census"):
        runner.require_arwen_v2_surface_execution(dict(payload, noahmp_routing_census=wrong), executed=True, label="missing column")
    with pytest.raises(ValueError, match="census/provenance changed"):
        runner.require_arwen_v2_surface_execution(dict(payload, last_noahmp_census=wrong), executed=True, label="wrong execution")


def test_a_suite_frame_after_a_move_is_judged_by_the_layout_noahmp_ran():
    """run_dbe08aff17eae2c7: a Thompson roster member's snapshot has no routing census, so the frame was judged by the
    new mesh's cold ice (3,349 sea-ice columns) against the carried ice Noah-MP really ran (3,492)."""
    from woof.hex.drivers.samehost_forecast_v841 import _expected_noahmp_census
    cold = dict(sflx_land_columns=1, open_water_columns=4, sea_ice_columns=2, glacier_columns=0)
    ran = dict(land=1, water=3, sea_ice=3, glacier=0)
    assert _expected_noahmp_census(None, SimpleNamespace(noahmp_routing_census=lambda: ran), cold) == ran
    assert _expected_noahmp_census(None, SimpleNamespace(), cold) == dict(land=1, water=4, sea_ice=2, glacier=0)
