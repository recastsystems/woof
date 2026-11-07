from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from woof.core import nssl2_production_coordinator as coordinator


_FIELD_NAMES = (
    "qv",
    "qc",
    "qr",
    "qi",
    "qs",
    "qg",
    "qh",
    "qndrop",
    "qnr",
    "qni",
    "qns",
    "qng",
    "qnh",
    "qnn",
    "qvolg",
    "qvolh",
)
_CONCENTRATION_FIELDS = _FIELD_NAMES[7:]


class _Workspace:
    def __init__(self, registry, density):
        self.shape = registry.qv.shape
        self.category_surface_export = np.zeros((5, *self.shape[1:]), dtype=np.float32)
        self._fields = {}
        for name in _FIELD_NAMES:
            value = getattr(registry, name).copy()
            if name in _CONCENTRATION_FIELDS:
                value *= density
            self._fields[name] = value

    def field(self, name):
        return self._fields[name]


def _registry():
    fields = {
        name: np.full((2, 1, 1), index + 1, dtype=np.float32)
        for index, name in enumerate(_FIELD_NAMES)
    }
    return coordinator.NSSL2RegistryFields(**fields)


def _precipitation():
    fields = {
        name: np.full((1, 1), index, dtype=np.float32)
        for index, name in enumerate(
            (
                "rainnc",
                "rainncv",
                "snownc",
                "snowncv",
                "graupelnc",
                "graupelncv",
                "hailnc",
                "hailncv",
                "sr",
            )
        )
    }
    return coordinator.NSSL2PrecipitationFields(**fields)


def _snapshot(registry):
    return {name: getattr(registry, name).copy() for name in _FIELD_NAMES}


def _assert_registry_equal(registry, expected):
    for name in _FIELD_NAMES:
        np.testing.assert_array_equal(getattr(registry, name), expected[name])


def _assert_identity_sequence(actual, expected):
    assert len(actual) == len(expected)
    assert all(left is right for left, right in zip(actual, expected))


def test_production_order_keeps_one_concentration_workspace_until_scatter(monkeypatch):
    registry = _registry()
    precipitation = _precipitation()
    before = _snapshot(registry)
    density = np.full((2, 1, 1), 2.0, dtype=np.float32)
    dz = np.full((2, 1, 1), 100.0, dtype=np.float32)
    temperature = object()
    reflectivity = object()
    radius_outputs = (object(), object(), object())
    workspace = _Workspace(registry, density)
    events = []

    def assert_pre_scatter():
        _assert_registry_equal(registry, before)

    def gather(*args, **kwargs):
        events.append("gather")
        assert_pre_scatter()
        _assert_identity_sequence(args[:2], (density, dz))
        _assert_identity_sequence(args[2:18], registry.as_tuple())
        assert args[18] == 12.5
        assert kwargs == {
            "temperature_k": temperature,
            "first_step": True,
            "cu_used": True,
            "qrcuten": "rain-rate",
            "qscuten": "snow-rate",
            "qicuten": "ice-rate",
            "qccuten": "cloud-rate",
            # The resolved variant reaches the gather, which owns WRF's
            # CCN load; the default lane predicts CCN.
            "predicted_ccn": True,
            "nssl_cccn": 0.5e9,
        "shapes": (0.0, 1.0),
        }
        return workspace

    def reduce_precip(actual_workspace, *outputs, air_density, dt_s):
        events.append("precip")
        assert actual_workspace is workspace
        assert air_density is density
        _assert_identity_sequence(outputs, precipitation.as_tuple())
        assert_pre_scatter()

    def fused_gs(actual_workspace):
        events.append("fused-gs")
        assert actual_workspace is workspace
        assert_pre_scatter()
        np.testing.assert_array_equal(workspace.field("qnr"), before["qnr"] * density)
        workspace.field("qnr")[...] = 111.0

    def nucond(actual_workspace):
        events.append("nucond")
        assert actual_workspace is workspace
        assert_pre_scatter()
        np.testing.assert_array_equal(workspace.field("qnr"), 111.0)
        workspace.field("qv")[...] = 222.0

    def qv_excess(actual_workspace):
        events.append("qv-excess")
        assert actual_workspace is workspace
        assert_pre_scatter()
        np.testing.assert_array_equal(workspace.field("qv"), 222.0)

    radar_names = (
        "qr",
        "qi",
        "qs",
        "qg",
        "qh",
        "qnr",
        "qni",
        "qns",
        "qng",
        "qnh",
        "qvolg",
        "qvolh",
    )

    def radar(*args, **kwargs):
        events.append("radar")
        assert_pre_scatter()
        expected = (
            density,
            temperature,
            *(workspace.field(name) for name in radar_names),
            reflectivity,
        )
        _assert_identity_sequence(args, expected)
        assert kwargs == {
            "output_due": True,
            "shapes": (0.0, 1.0),
            "concentration_space": True,
            "validate_values": False,
        }

    def effective_radius(*args, **kwargs):
        events.append("radii")
        assert_pre_scatter()
        expected = (
            density,
            workspace.field("qc"),
            workspace.field("qndrop"),
            workspace.field("qi"),
            workspace.field("qni"),
            workspace.field("qs"),
            workspace.field("qns"),
            *radius_outputs,
        )
        _assert_identity_sequence(args, expected)
        assert kwargs == {"validate_values": False}

    def scatter(actual_workspace, actual_density, *outputs,
                predicted_ccn=True):
        events.append("scatter")
        assert actual_workspace is workspace
        assert actual_density is density
        # The resolved variant reaches the scatter, so the default lane
        # must be seen asking for the CCN store explicitly.
        assert predicted_ccn is True
        _assert_identity_sequence(outputs, registry.as_tuple())
        assert_pre_scatter()
        for index, output in enumerate(outputs, start=1):
            output[...] = -index

    def finish(actual_workspace):
        events.append("finish")
        assert actual_workspace is workspace
        for index, name in enumerate(_FIELD_NAMES, start=1):
            np.testing.assert_array_equal(getattr(registry, name), -index)

    monkeypatch.setattr(coordinator, "gather_initialize_and_sediment", gather)
    monkeypatch.setattr(coordinator, "reduce_nssl2_precipitation", reduce_precip)
    monkeypatch.setattr(coordinator, "launch_radardd02", radar)
    monkeypatch.setattr(
        coordinator, "launch_effective_radius_concentration", effective_radius
    )
    monkeypatch.setattr(coordinator, "scatter_nssl2_driver_workspace", scatter)

    hooks = coordinator.NSSL2ProductionHooks(
        fused_gs=fused_gs,
        nucond=nucond,
        qv_excess=qv_excess,
        moist_physics_finish=finish,
    )
    result = coordinator.run_nssl2_production_step(
        density,
        dz,
        registry,
        precipitation,
        hooks,
        12.5,
        first_step=True,
        cu_used=True,
        qrcuten="rain-rate",
        qscuten="snow-rate",
        qicuten="ice-rate",
        qccuten="cloud-rate",
        output_due=True,
        temperature_k=temperature,
        refl_10cm=reflectivity,
        radiation_due=True,
        re_cloud_m=radius_outputs[0],
        re_ice_m=radius_outputs[1],
        re_snow_m=radius_outputs[2],
        validate_values=False,
    )

    assert result is workspace
    assert events == [
        "gather",
        "precip",
        "fused-gs",
        "nucond",
        "qv-excess",
        "radar",
        "radii",
        "scatter",
        "finish",
    ]
    assert events.count("gather") == 1
    assert events.count("scatter") == 1


def test_not_due_is_strict_noop_and_combined_nucond_qvexcess_is_explicit(monkeypatch):
    registry = _registry()
    before = _snapshot(registry)
    precipitation = _precipitation()
    density = np.ones((2, 1, 1), dtype=np.float32)
    dz = np.ones_like(density)
    workspace = _Workspace(registry, density)
    events = []

    monkeypatch.setattr(
        coordinator,
        "gather_initialize_and_sediment",
        lambda *args, **kwargs: events.append("gather") or workspace,
    )
    monkeypatch.setattr(
        coordinator, "reduce_nssl2_precipitation", lambda *args, **kwargs: events.append("precip")
    )
    monkeypatch.setattr(
        coordinator,
        "launch_radardd02",
        lambda *args, **kwargs: pytest.fail("radar ran when not due"),
    )
    monkeypatch.setattr(
        coordinator,
        "launch_effective_radius_concentration",
        lambda *args, **kwargs: pytest.fail("radii ran when not due"),
    )
    monkeypatch.setattr(
        coordinator,
        "scatter_nssl2_driver_workspace",
        lambda *args, **kwargs: events.append("scatter"),
    )

    hooks = coordinator.NSSL2ProductionHooks(
        fused_gs=lambda ws: events.append("fused-gs"),
        nucond_qvexcess=lambda ws: events.append("nucond-qvexcess"),
        moist_physics_finish=lambda ws: events.append("finish"),
    )
    coordinator.run_nssl2_production_step(
        density,
        dz,
        registry,
        precipitation,
        hooks,
        1.0,
        temperature_k=np.full_like(density, 260.0),
        output_due=False,
        radiation_due=False,
    )

    assert events == [
        "gather",
        "precip",
        "fused-gs",
        "nucond-qvexcess",
        "scatter",
        "finish",
    ]
    _assert_registry_equal(registry, before)


@pytest.mark.parametrize(
    ("hooks", "message"),
    [
        (
            coordinator.NSSL2ProductionHooks(
                nucond=lambda ws: None,
                qv_excess=lambda ws: None,
                moist_physics_finish=lambda ws: None,
            ),
            "fused_gs",
        ),
        (
            coordinator.NSSL2ProductionHooks(
                fused_gs=lambda ws: None,
                qv_excess=lambda ws: None,
                moist_physics_finish=lambda ws: None,
            ),
            "nucond",
        ),
        (
            coordinator.NSSL2ProductionHooks(
                fused_gs=lambda ws: None,
                nucond=lambda ws: None,
                moist_physics_finish=lambda ws: None,
            ),
            "qv_excess",
        ),
        (
            coordinator.NSSL2ProductionHooks(
                fused_gs=lambda ws: None, nucond_qvexcess=lambda ws: None
            ),
            "moist_physics_finish",
        ),
    ],
)
def test_missing_required_hook_fails_before_gather(monkeypatch, hooks, message):
    gathered = []
    monkeypatch.setattr(
        coordinator,
        "gather_initialize_and_sediment",
        lambda *args, **kwargs: gathered.append(True),
    )

    with pytest.raises(coordinator.NSSL2ProductionConfigurationError, match=message):
        coordinator.run_nssl2_production_step(
            object(), object(), _registry(), _precipitation(), hooks, 1.0
        )
    assert gathered == []


def test_ambiguous_or_noncallable_hooks_fail_before_gather(monkeypatch):
    gathered = []
    monkeypatch.setattr(
        coordinator,
        "gather_initialize_and_sediment",
        lambda *args, **kwargs: gathered.append(True),
    )

    def callback(workspace):
        return None

    ambiguous = coordinator.NSSL2ProductionHooks(
        fused_gs=callback,
        nucond=callback,
        qv_excess=callback,
        nucond_qvexcess=callback,
        moist_physics_finish=callback,
    )
    with pytest.raises(coordinator.NSSL2ProductionConfigurationError, match="not both"):
        coordinator.run_nssl2_production_step(
            object(), object(), _registry(), _precipitation(), ambiguous, 1.0
        )

    noncallable = coordinator.NSSL2ProductionHooks(
        fused_gs=callback,
        nucond=callback,
        qv_excess=42,
        moist_physics_finish=callback,
    )
    with pytest.raises(TypeError, match="qv_excess"):
        coordinator.run_nssl2_production_step(
            object(), object(), _registry(), _precipitation(), noncallable, 1.0
        )
    assert gathered == []


@pytest.mark.parametrize(
    "due_kwargs",
    [
        {"output_due": True},
        {"output_due": True, "temperature_k": object()},
        {"radiation_due": True},
        {"radiation_due": True, "re_cloud_m": object(), "re_ice_m": object()},
    ],
)
def test_due_products_are_required_before_gather(monkeypatch, due_kwargs):
    gathered = []
    monkeypatch.setattr(
        coordinator,
        "gather_initialize_and_sediment",
        lambda *args, **kwargs: gathered.append(True),
    )

    def callback(workspace):
        return None

    hooks = coordinator.NSSL2ProductionHooks(
        fused_gs=callback,
        nucond_qvexcess=callback,
        moist_physics_finish=callback,
    )

    with pytest.raises(coordinator.NSSL2ProductionConfigurationError):
        coordinator.run_nssl2_production_step(
            object(), object(), _registry(), _precipitation(), hooks, 1.0, **due_kwargs
        )
    assert gathered == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("output_due", np.bool_(False)),
        ("radiation_due", 0),
        ("validate_values", None),
    ],
)
def test_gate_types_are_fail_closed_before_gather(monkeypatch, name, value):
    gathered = []
    monkeypatch.setattr(
        coordinator,
        "gather_initialize_and_sediment",
        lambda *args, **kwargs: gathered.append(True),
    )

    def callback(workspace):
        return None

    hooks = coordinator.NSSL2ProductionHooks(
        fused_gs=callback,
        nucond_qvexcess=callback,
        moist_physics_finish=callback,
    )

    with pytest.raises(TypeError, match=name):
        coordinator.run_nssl2_production_step(
            object(),
            object(),
            _registry(),
            _precipitation(),
            hooks,
            1.0,
            **{name: value},
        )
    assert gathered == []


def test_stage_failure_never_scatter_partial_workspace(monkeypatch):
    registry = _registry()
    before = _snapshot(registry)
    density = np.ones((2, 1, 1), dtype=np.float32)
    workspace = _Workspace(registry, density)
    scattered = []

    monkeypatch.setattr(
        coordinator, "gather_initialize_and_sediment", lambda *args, **kwargs: workspace
    )
    monkeypatch.setattr(coordinator, "reduce_nssl2_precipitation", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        coordinator,
        "scatter_nssl2_driver_workspace",
        lambda *args, **kwargs: scattered.append(True),
    )

    def failed_qv_excess(workspace):
        raise RuntimeError("qv-excess failed")

    hooks = coordinator.NSSL2ProductionHooks(
        fused_gs=lambda ws: None,
        nucond=lambda ws: None,
        qv_excess=failed_qv_excess,
        moist_physics_finish=lambda ws: None,
    )
    with pytest.raises(RuntimeError, match="qv-excess failed"):
        coordinator.run_nssl2_production_step(
            density, np.ones_like(density), registry, _precipitation(), hooks,
            1.0, temperature_k=np.full_like(density, 260.0)
        )

    assert scattered == []
    _assert_registry_equal(registry, before)


def test_coordinator_has_one_static_gather_and_scatter_call_site():
    tree = ast.parse(inspect.getsource(coordinator.run_nssl2_production_step))
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("gather_initialize_and_sediment") == 1
    assert calls.count("reduce_nssl2_precipitation") == 1
    assert calls.count("scatter_nssl2_driver_workspace") == 1
