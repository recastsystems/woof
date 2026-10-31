"""Strict scalar residuals keep a floor in every non-forced cell."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.core import health
from woof.core.state import DomainState
from woof.wrf_exact import ENABLED
from test_health import _GPUState, _inventory_config


@pytest.mark.parametrize("specified,nested,width", [
    (True, False, 1), (False, True, 2), (False, False, 2)])
def test_geometry_uses_actual_boundary_configuration(specified, nested, width):
    from woof.core.acoustic import _spec_zone
    from woof.io.restart import classify_state_attr
    cfg = replace(_inventory_config(), specified=specified, nested=nested,
                  spec_zone=width)
    state = DomainState(cfg, array_module=np)
    assert state._health_spec_zone == _spec_zone(cfg)
    assert classify_state_attr("_health_spec_zone") == "infra"
    rule = next(f.rule for f in health.collect_state_fields(state) if f.name == "nr")
    assert rule.rim_width == (_spec_zone(cfg) if ENABLED else 0)


@pytest.mark.parametrize("name", ["nc", "nr", "ni", "ns", "ng", "qndrop",
    "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg", "qvolh"])
@pytest.mark.parametrize("width", [0, 1, 2])
def test_numpy_and_native_rim_floors(name, width):
    values = np.zeros((2, 7, 9), np.float32)
    for j, i in [(0, 4), (6, 4), (3, 0), (3, 8), (1, 4), (3, 4)]:
        values.fill(0)
        values[1, j, i] = -1.0e-7
        fields = health.collect_state_fields(SimpleNamespace(
            **{name: values}, _health_spec_zone=width))
        distance = min(j, 6-j, i, 8-i)
        before = values.tobytes()
        for scan in (health._validate_fields_numpy, health.validate_fields_cpu):
            report = scan(fields)
            assert report.ok == (ENABLED and distance < width)
            assert values.tobytes() == before
            if not report.ok:
                assert report.first_bad_index == (1, j, i)
        for bad in (np.nan, np.inf, fields[0].rule.upper * 2):
            values[1, j, i] = bad
            assert not health._validate_fields_numpy(fields).ok
            assert not health.validate_fields_cpu(fields).ok


def test_missing_geometry_keeps_floor_and_strided_arrays_keep_shape():
    values = np.zeros((2, 7, 9), np.float32)
    values[1, 0, 4] = -1.0e-7
    assert not health.validate_state_cpu(SimpleNamespace(nr=values)).ok
    assert not health.validate_fields_cpu({"nr": np.array([-1.0e-7], np.float32)}).ok
    fields = health.collect_state_fields(SimpleNamespace(
        nr=values.transpose(0, 2, 1), _health_spec_zone=1))
    assert health.validate_fields_cpu(fields).ok == ENABLED


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["nr", "ni", "qvolg"])
@pytest.mark.parametrize("width", [0, 1, 2])
def test_cuda_rim_floor_finiteness_and_ceiling(name, width):
    import cupy as cp
    state = _GPUState(cp, name, 0.0)
    state._health_spec_zone = width
    values = cp.zeros((3, 7, 9), cp.float32)
    setattr(state, name, values)
    validator = health.StateHealthValidator(state)
    for j, i in [(0, 4), (6, 4), (3, 0), (3, 8), (1, 4), (3, 4)]:
        values.fill(0)
        values[2, j, i] = -7.607106878081259e-23
        before = cp.asnumpy(values).tobytes()
        report = validator.validate()
        assert report.ok == (ENABLED and min(j, 6-j, i, 8-i) < width)
        assert cp.asnumpy(values).tobytes() == before
        if not report.ok:
            assert report.first_bad_index == (2, j, i)
        for bad in (np.nan, np.inf, 2*health.rule_for_field(name).upper):
            values[2, j, i] = bad
            assert not validator.validate().ok


def test_member_gate_prices_private_rim_metadata():
    from woof.ensemble.batch_health import state_health_memory_plan
    plan = state_health_memory_plan()
    rows = {r['name']: r for r in plan.inventory(3)}
    assert rows['health:validate:rim']['shape'] == (3, health.MAX_HEALTH_FIELDS, 3)
    assert rows['health:validate:rim']['ownership'] == 'member'


@pytest.mark.gpu
@pytest.mark.parametrize('members', [1, 3])
def test_member_gate_uses_each_fields_logical_rim(members):
    import cupy as cp
    from woof.ensemble.batch_health import PreparedBatchStateHealth
    from woof.ensemble.batch_diagnostics import diagnostics_specs
    from woof.ensemble.batch_state import BatchedDomainState
    from test_ensemble_batch_bigstep_gpu import _inputs
    inputs = _inputs(members, terrain=True, mapped=True)
    batch = BatchedDomainState.from_prepared(inputs, array_module=cp,
                                            available_bytes=1 << 28,
                                            extra_specs=diagnostics_specs(inputs[0].cfg))
    values = cp.zeros((2, members * 7, 9), cp.float32)
    validator = PreparedBatchStateHealth(batch, available_bytes=1 << 26)
    validator.field_provider = lambda m: health.collect_state_fields(SimpleNamespace(
        ni=values[:, m*7:(m+1)*7], _health_spec_zone=1), backend='gpu')
    for j, value in [(0, -7.607106878081259e-23), (3, -1.0e-7),
                     (0, np.nan), (0, np.inf), (0, 2.0e15)]:
        values.fill(0)
        values[1, (members-1)*7+j, 4] = value
        reports = validator.validate()
        assert all(r.ok for r in reports[:-1])
        assert reports[-1].ok == (ENABLED and j == 0 and value < 0)
        if not reports[-1].ok:
            assert reports[-1].first_bad_index == (1, j, 4)


@pytest.mark.parametrize("name", ["nwfa", "nifa"])
@pytest.mark.parametrize("strict", [False, True])
def test_strict_aerosol_numbers_keep_wrfs_interior_residual(monkeypatch,
                                                            name, strict):
    """The two-domain strict UAE mp=28 case (2026-10-28): stock WRF 4.6.1
    writes QNIFA -3.70e-6 at an interior top-layer point at +1 h, and strict
    WOOF stopped at nifa(43, 94, 47) = -7.236842e-6, edge distance 5,
    under the rim-only floor.  Strict mode admits that word; the default
    route keeps its floor, and both keep finiteness and the ceiling."""
    monkeypatch.setattr(health, "WRF_EXACT", strict)
    values = np.full((44, 100, 120), 1343.0, np.float32)
    values[43, 94, 47] = np.float32(-7.236842066049576e-06)
    fields = (health.HealthField(name, values, health.rule_for_field(name)),)
    for scan in (health._validate_fields_numpy, health.validate_fields_cpu):
        report = scan(fields)
        assert report.ok == strict
        if not strict:
            assert report.first_bad_index == (43, 94, 47)
    for bad in (np.nan, np.inf, 2.0e15):
        values[43, 94, 47] = bad
        for scan in (health._validate_fields_numpy, health.validate_fields_cpu):
            assert not scan(fields).ok
