"""Depth-node soil sources through the declarative mapped contract.

RUC-family models (HRRR, RAP) publish soil as point samples at depths --
including both the 0 m surface and the 3 m column bottom -- not as layer
means.  The contract declares that geometry as ``source_nodes`` with the
``linear_node_samples`` remap, and the executor reuses WRF's sorted linear
node interpolation, which is exactly what the certified native HRRR route
runs on its fixed node table.  These tests pin the two against each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from woof.ingest import soil as soil_module
from woof.ingest.soil import (
    HRRR_SOIL_NODE_DEPTHS_M, NOAH_LAYER_MIDPOINTS_M, _interp_nodes,
    _remap_declared_soil,
)
from woof.ingest.soil_contract import (
    soil_node_depths, soil_source_sample_count, validate_soil_layer_contract,
)


def _node_selector(name: str, depth: float) -> dict[str, object]:
    parameter = 18 if name == "soil_temperature" else 192
    return {
        "format": "grib2", "discipline": 2, "category": 3 if name == "soil_temperature" else 0,
        "parameter": parameter, "center": 7, "subcenter": 0,
        "master_table_version": 2, "local_table_version": 1,
        "level_type": 106, "level_value": depth,
        "second_level_type": 106, "second_level_value": depth,
    }


def _node_contract(depths=tuple(float(d) for d in HRRR_SOIL_NODE_DEPTHS_M)):
    return {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_nodes": [
            {
                "depth": depth,
                "selectors": {
                    "soil_temperature": _node_selector(
                        "soil_temperature", depth),
                    "volumetric_soil_moisture": _node_selector(
                        "volumetric_soil_moisture", depth),
                },
            }
            for depth in depths
        ],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1},
            {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0},
            {"top": 1.0, "bottom": 2.0},
        ],
        "remap": {
            "kind": "linear_node_samples",
            "source_value_location": "level_node",
            "target_value_location": "layer_midpoint",
        },
        "missing": {
            "land": "reject",
            "ocean": {
                "stage": "after_horizontal_interpolation",
                "temperature": "skin_temperature",
                "moisture": 1.0,
            },
        },
    }


def test_node_contract_validates_and_reports_its_geometry():
    contract = validate_soil_layer_contract(_node_contract())
    assert soil_source_sample_count(contract) == 9
    assert soil_node_depths(contract) == tuple(
        float(d) for d in HRRR_SOIL_NODE_DEPTHS_M)


@pytest.mark.parametrize("depths", [(0.01, 0.04, 0.1, 0.3, 0.6, 1., 1.6, 3.),
                                    (0.0, 0.1, 0.4, 1.0, 2.0), (0.05, 0.25, 0.7, 1.5)])
def test_node_contract_requires_actual_noah_target_coverage(depths):
    contract = validate_soil_layer_contract(_node_contract(depths=depths))
    values = (280 + np.asarray(depths) * 3)[:, None, None]
    temperature, _ = _remap_declared_soil(values, values/1000, contract,
                                         tsk=np.array([[100.]]), deep=np.array([[400.]]))
    np.testing.assert_allclose(temperature[:, 0, 0], 280 + NOAH_LAYER_MIDPOINTS_M*3, rtol=2.5e-7, atol=0)  # REAL arithmetic, as real.exe


@pytest.mark.parametrize("depths", [(0.1, 0.4, 1., 3.), (0., 0.1, 0.4, 1.)])
def test_node_contract_refuses_uncovered_target_midpoints(depths):
    with pytest.raises(ValueError, match="do not cover Noah target"):
        validate_soil_layer_contract(_node_contract(depths=depths))


@pytest.mark.parametrize("count", [6, 9])
def test_noah_coverage_does_not_invent_ruc_surface_node(count):
    from woof.ingest.ruc_soil import remap_soil_to_ruc_levels
    depths = (0.01, 0.04, 0.1, 0.3, 0.6, 1., 1.6, 3.)
    validate_soil_layer_contract(_node_contract(depths=depths))
    with pytest.raises(ValueError, match="RUC level 1.*outside the source"):
        remap_soil_to_ruc_levels(
            source_temperature=np.full((8,1,1),280), source_moisture=np.full((8,1,1),.3),
            source_levels_cm=np.array([1,4,10,30,60,100,160,300]), source_geometry="levels",
            skin_temperature=np.array([[280.]]),deep_temperature=np.array([[280.]]),
            landmask=np.array([[1.]]), num_soil_layers=count)


def test_node_contract_refuses_a_layer_remap_kind():
    contract = _node_contract()
    contract["remap"] = {
        "kind": "conservative_layer_means",
        "source_value_location": "layer_mean",
        "target_value_location": "layer_mean",
        "coverage": "require_complete",
    }
    with pytest.raises(ValueError, match="source_nodes require"):
        validate_soil_layer_contract(contract)


def test_node_contract_refuses_both_geometries_at_once():
    contract = _node_contract()
    contract["source_layers"] = [
        {"top": 0.0, "bottom": 0.1, "selectors": {
            "soil_temperature": _node_selector("soil_temperature", 0.0),
            "volumetric_soil_moisture": _node_selector(
                "volumetric_soil_moisture", 0.0),
        }},
    ]
    with pytest.raises(ValueError, match="exactly one source"):
        validate_soil_layer_contract(contract)


def test_node_remap_matches_the_native_hrrr_node_interpolation():
    """The declared executor IS the native route's arithmetic."""

    contract = validate_soil_layer_contract(_node_contract())
    rng = np.random.default_rng(7)
    temperature = 280.0 + rng.random((9, 4, 5))
    moisture = 0.2 + 0.5 * rng.random((9, 4, 5))
    tsk = np.full((4, 5), 300.0)
    deep = np.full((4, 5), 285.0)
    soil_t, soil_m = _remap_declared_soil(
        temperature, moisture, contract, tsk=tsk, deep=deep)
    native_t = _interp_nodes(
        temperature, HRRR_SOIL_NODE_DEPTHS_M, NOAH_LAYER_MIDPOINTS_M)
    native_m = _interp_nodes(
        moisture, HRRR_SOIL_NODE_DEPTHS_M, NOAH_LAYER_MIDPOINTS_M)
    np.testing.assert_array_equal(soil_t, native_t)
    np.testing.assert_array_equal(soil_m, native_m)
    # The anchors were never consulted: the endpoints are source samples.
    hot_tsk = np.full((4, 5), 400.0)
    soil_t_again, _ = _remap_declared_soil(
        temperature, moisture, contract, tsk=hot_tsk, deep=deep)
    np.testing.assert_array_equal(soil_t, soil_t_again)


def test_mapped_route_puts_every_moisture_overshoot_on_the_range(capsys):
    """WPS sixteen_pt genuinely overshoots on sharp gradients.

    MEASURED on the first real HRRR wrfprs preparation: one land cell of
    22,482 at -0.016 volumetric moisture; on the 2026-09-27 06Z HRRR a 1 km
    grid over a reservoir's dry bank reached -0.055, past the 0.05 margin
    that used to separate a repair from a refusal.  Every finite value
    outside 0..1 goes on the range, loudly and default-on; a land cell with
    no value at all still refuses, by name.
    """

    from woof.ingest.soil import preprocess_noah_soil
    from woof.ingest.soil_contract import (
        MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE)

    contract = _node_contract()
    surface = {
        "LANDSEA": np.asarray([[1.0, 0.0], [1.0, 1.0]]),
        "SKINTEMP": np.asarray([[291.0, 285.0], [289.0, 288.0]]),
        "TMN": np.asarray([[286.0, 280.0], [285.0, 284.0]]),
    }
    temperature = np.broadcast_to(
        np.linspace(290.0, 284.0, 9)[:, None, None], (9, 2, 2)).copy()
    moisture = np.full((9, 2, 2), 0.25)
    moisture[0, 0, 0] = -0.016          # the measured overshoot shape
    fields = {
        **surface,
        MAPPED_SOIL_TEMPERATURE: temperature,
        MAPPED_SOIL_MOISTURE: moisture,
    }
    state = preprocess_noah_soil(
        fields, soil_type=np.full((2, 2), 6),
        soil_layer_contract=contract)
    assert float(np.min(state.soil_moisture)) >= 0.0
    captured = capsys.readouterr()
    assert "1 land value(s) outside 0..1" in captured.err

    beyond = moisture.copy()
    beyond[0, 0, 0] = -0.10             # past the retired 0.05 margin
    beyond[3, 1, 1] = 1.2
    fields[MAPPED_SOIL_MOISTURE] = beyond
    state = preprocess_noah_soil(
        fields, soil_type=np.full((2, 2), 6),
        soil_layer_contract=contract)
    assert float(np.min(state.soil_moisture)) >= 0.0
    assert float(np.max(state.soil_moisture)) <= 1.0
    captured = capsys.readouterr()
    assert "2 land value(s) outside 0..1" in captured.err

    missing = moisture.copy()
    missing[0, 0, 0] = np.nan
    fields[MAPPED_SOIL_MOISTURE] = missing
    with pytest.raises(ValueError, match="carries no value on 1 land"):
        preprocess_noah_soil(
            fields, soil_type=np.full((2, 2), 6),
            soil_layer_contract=contract)


def test_moisture_past_the_operators_reach_refuses_on_land_only():
    """A value no interpolation of a 0..1 field makes is not repaired.

    The overlapping parabola carries a 0..1 field no further than
    -0.2969..1.2969 (9/32 of the donors' range past each end, the donors
    widened by packing roundoff), and every other operator is a weighted
    mean.  A met_em file reaches the initializer with no masked mapping in
    front of it, so this is where a fill value or percent read as a
    fraction is caught, on either land-surface arm.  A water column is
    set to 1.0 whatever it held and is not judged.
    """

    from woof.ingest.horiz import parabolic_reach
    from woof.ingest.ruc_soil import _source_soil_profiles
    from woof.ingest.soil import preprocess_noah_soil
    from woof.ingest.soil_contract import (
        MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE)

    lowest, highest = parabolic_reach(0.0, 1.0)
    assert lowest == pytest.approx(-0.296875, abs=1.0e-5)
    assert highest == pytest.approx(1.296875, abs=1.0e-5)
    contract = _node_contract()
    landsea = np.asarray([[1.0, 0.0], [1.0, 1.0]])
    surface = {
        "LANDSEA": landsea,
        "SKINTEMP": np.asarray([[291.0, 285.0], [289.0, 288.0]]),
        "TMN": np.asarray([[286.0, 280.0], [285.0, 284.0]]),
    }
    temperature = np.broadcast_to(
        np.linspace(290.0, 284.0, 9)[:, None, None], (9, 2, 2)).copy()

    def prepare(moisture):
        fields = {**surface, MAPPED_SOIL_TEMPERATURE: temperature,
                  MAPPED_SOIL_MOISTURE: moisture}
        state = preprocess_noah_soil(
            fields, soil_type=np.full((2, 2), 6),
            soil_layer_contract=contract)
        ruc = _source_soil_profiles(fields, contract, land=landsea >= 0.5)
        return state, ruc

    moisture = np.full((9, 2, 2), 0.25)
    # The operator's own reach at both ends, on land; a metgrid fill on the
    # water column.
    moisture[0, 0, 0] = -0.2968
    moisture[3, 1, 1] = 1.2968
    moisture[:, 0, 1] = -1.0e30
    state, ruc = prepare(moisture)
    assert float(np.min(state.soil_moisture)) >= 0.0
    assert float(np.max(state.soil_moisture)) <= 1.0
    assert ruc[1][0, 0, 0] == 0.0 and ruc[1][3, 1, 1] == 1.0

    for value in (-0.2970, 30.0, -1.0e30):
        beyond = np.full((9, 2, 2), 0.25)
        beyond[2, 1, 0] = value
        with pytest.raises(ValueError, match=(
                r"mapped soil moisture: 1 land value\(s\) outside "
                r"-0.2969..1.2969")) as refusal:
            prepare(beyond)
        assert "a fill value or soil moisture in another unit" in str(
            refusal.value)
        with pytest.raises(ValueError, match=(
                r"mapped soil moisture \(RUC .*\): 1 land value\(s\) "
                r"outside -0.2969..1.2969")):
            _source_soil_profiles(
                {MAPPED_SOIL_TEMPERATURE: temperature,
                 MAPPED_SOIL_MOISTURE: beyond}, contract,
                land=landsea >= 0.5)


def test_layer_contracts_are_unchanged():
    """The historical layer path validates and executes as before."""

    contract = {
        "temperature_field": "soil_temperature",
        "moisture_field": "volumetric_soil_moisture",
        "depth_units": "m",
        "source_layers": [
            {"top": top, "bottom": bottom, "selectors": {
                "soil_temperature": {
                    "format": "grib2", "discipline": 0, "category": 0,
                    "parameter": 0, "level_type": 106, "level_value": top,
                    "second_level_type": 106, "second_level_value": bottom,
                },
                "volumetric_soil_moisture": {
                    "format": "grib2", "discipline": 2, "category": 0,
                    "parameter": 192, "center": 7, "subcenter": 2,
                    "master_table_version": 2, "local_table_version": 1,
                    "level_type": 106, "level_value": top,
                    "second_level_type": 106, "second_level_value": bottom,
                },
            }}
            for top, bottom in ((0.0, 0.1), (0.1, 0.4), (0.4, 1.0), (1.0, 2.0))
        ],
        "target_layers": [
            {"top": 0.0, "bottom": 0.1},
            {"top": 0.1, "bottom": 0.4},
            {"top": 0.4, "bottom": 1.0},
            {"top": 1.0, "bottom": 2.0},
        ],
        "remap": {
            "kind": "conservative_layer_means",
            "source_value_location": "layer_mean",
            "target_value_location": "layer_mean",
            "coverage": "require_complete",
        },
        "missing": {
            "land": "reject",
            "ocean": {
                "stage": "after_horizontal_interpolation",
                "temperature": "skin_temperature",
                "moisture": 1.0,
            },
        },
    }
    validated = validate_soil_layer_contract(contract)
    assert soil_source_sample_count(validated) == 4
    temperature = np.full((4, 2, 2), 290.0)
    moisture = np.full((4, 2, 2), 0.3)
    soil_t, soil_m = _remap_declared_soil(
        temperature, moisture, validated,
        tsk=np.full((2, 2), 300.0), deep=np.full((2, 2), 285.0))
    np.testing.assert_array_equal(soil_t, temperature)
    np.testing.assert_array_equal(soil_m, moisture)


def test_fractional_seaice_selects_real_soil_column_and_preserves_default_bytes():
    from woof.ingest.ruc_soil import preprocess_land_surface_soil
    from woof.ingest.soil_contract import MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE
    shape=(1,3)
    fields={"SKINTEMP":np.full(shape,268.),"LANDSEA":np.zeros(shape),
            "SEAICE":np.array([[.01,.2,.7]]), "SNOW":np.zeros(shape),"SNOWH":np.zeros(shape),
            MAPPED_SOIL_TEMPERATURE:np.full((9,*shape),280.),
            MAPPED_SOIL_MOISTURE:np.full((9,*shape),.3)}
    kw=dict(sf_surface_physics=2,soil_type=np.full(shape,14),
            deep_soil_temperature=np.full(shape,280.),soil_layer_contract=_node_contract(),
            landmask=np.zeros(shape),water_temperature_policy="wrf_compat")
    old=preprocess_land_surface_soil(fields,**kw)
    binary=preprocess_land_surface_soil(fields,fractional_seaice=False,**kw)
    fractional=preprocess_land_surface_soil(fields,fractional_seaice=True,**kw)
    for name in ("tsk","soil_temperature","soil_moisture","liquid_moisture","landmask","xice"):
        np.testing.assert_array_equal(getattr(old,name),getattr(binary,name))
    np.testing.assert_array_equal(binary.xice,[[0,0,1]])
    np.testing.assert_array_equal(fractional.xice,[[0,.2,.7]])
    np.testing.assert_array_equal(binary.landmask,[[0,0,1]])
    np.testing.assert_array_equal(fractional.landmask,[[0,1,1]])
    depths=(np.arange(4)+.5)*.75
    expected=((3-depths)*268+depths*271.4)/3
    np.testing.assert_allclose(fractional.soil_temperature[:,0,1],expected,rtol=2.5e-7,atol=0)  # REAL arithmetic, as real.exe
