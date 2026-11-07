"""CPU regressions for host views, forecast kwargs and native tail publication."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest

from woof.hex.cuda_dualrun import PreparedCudaInputs
from woof.hex.driver import DrySavedDiagnostics, TerrainMetrics
from woof.hex.drivers import samehost_forecast_v841 as adapter
from woof.hex.drivers.samehost_identity_v841 import assemble_owned_arrays
from woof.hex.mesh import Mesh
from woof.hex.partition_assets_v841 import PartitionLayout
from woof.hex.partition_executor_v841 import ExecutorError
from woof.hex.partition_local_mesh_v841 import slice_prepared_host
from woof.hex.state import PrognosticState
from woof.hex.transport import AdvectionCoefficients


def _layouts():
    layouts = []
    for rank in (0, 1):
        indexes = {}
        owned = {}
        for kind, total in (("cell", 12), ("edge", 24), ("vertex", 36)):
            half = total // 2
            indexes[kind] = np.concatenate((np.arange(rank * half, (rank + 1) * half, dtype=np.int32), np.array([((1 - rank) * half), ((2 - rank) * half) - 1], dtype=np.int32)))
            owned[kind] = half
        layouts.append(PartitionLayout(
            partition=rank, n_owned_cells=owned["cell"], n_owned_edges=owned["edge"], n_owned_vertices=owned["vertex"],
            cell_l2g=indexes["cell"], edge_l2g=indexes["edge"], vertex_l2g=indexes["vertex"],
            cell_ring=np.zeros(indexes["cell"].size, dtype=np.int16),
            cell_row_complete=np.ones(indexes["cell"].size, dtype=bool),
            edge_row_complete=np.ones(indexes["edge"].size, dtype=bool),
            vertex_row_complete=np.ones(indexes["vertex"].size, dtype=bool),
            _global_cells=12, _global_edges=24, _global_vertices=36,
        ))
    return layouts


def test_slotted_prepared_view_slices_init_false_fields_without_mutating_source():
    layout = _layouts()[0]
    mesh = Mesh(arrays={"areaCell": np.arange(12, dtype=np.float32)},
        dimensions={"nCells": 12, "nEdges": 24, "nVertices": 36}, variable_dimensions={"areaCell": ("nCells",)}, attrs={})
    state = PrognosticState(
        rho=np.arange(24, dtype=np.float32).reshape(2, 12),
        rho_theta=np.arange(24, dtype=np.float32).reshape(2, 12),
        rho_u=np.arange(48, dtype=np.float32).reshape(2, 24),
        rho_w=np.arange(36, dtype=np.float32).reshape(3, 12),
        scalars=np.arange(144, dtype=np.float32).reshape(6, 2, 12),
    )
    saved = DrySavedDiagnostics(**{
        name: np.arange((3 if name == "vertical_velocity" else 2) * (24 if name == "normal_velocity" else 12), dtype=np.float32).reshape(3 if name == "vertical_velocity" else 2, 24 if name == "normal_velocity" else 12)
        for name in DrySavedDiagnostics.__slots__
    })
    prepared = PreparedCudaInputs(profile="test", target="test", preparation_method="byte-copy", mesh=mesh,
        state=state, vertical=SimpleNamespace(zgrid=np.arange(36, dtype=np.float32).reshape(3, 12)),
        reference=SimpleNamespace(rho_base=np.arange(24, dtype=np.float32).reshape(2, 12)),
        saved_diagnostics=saved,
        terrain_metrics=TerrainMetrics(zb_cell=np.arange(72, dtype=np.float32).reshape(3, 12, 2), zb3_cell=np.zeros((3, 12, 2), dtype=np.float32)),
        input_bytes={"grid": {"sha256": "test"}},
    )
    coefficients = AdvectionCoefficients(
        adv_coefs=np.arange(72, dtype=np.float32).reshape(24, 3),
        adv_coefs_3rd=np.arange(72, dtype=np.float32).reshape(24, 3),
        n_adv_cells_for_edge=np.full(24, 3, dtype=np.int32),
        adv_cells_for_edge=np.tile(np.array([0, 11, 12], dtype=np.int32), (24, 1)),
        horizontal_order=3,
    )
    fingerprint = {"authority": "global", "n_cells": 12}
    object.__setattr__(prepared, "advection_coefficients", coefficients)
    object.__setattr__(prepared, "expected_execution_fingerprint", fingerprint)
    assert not hasattr(prepared, "__dict__")
    original_rho = state.rho.tobytes()
    original_stencil = coefficients.adv_cells_for_edge.tobytes()
    view = adapter._prepared_attribute_view(prepared)
    assert view.advection_coefficients is coefficients
    assert view.expected_execution_fingerprint is fingerprint
    local = slice_prepared_host(view, layout)
    assert local.state.rho.tobytes() == state.rho[..., layout.cell_l2g].tobytes()
    assert local.advection_coefficients.adv_coefs.tobytes() == coefficients.adv_coefs[layout.edge_l2g].tobytes()
    assert local.advection_coefficients.adv_cells_for_edge[0].tolist() == [0, 7, 8]
    assert local.terrain_metrics.zb_cell.tobytes() == prepared.terrain_metrics.zb_cell[:, layout.cell_l2g, :].tobytes()
    assert local.mesh.partition_source_mesh is mesh
    local.state.rho[0, 0] = 42
    local.advection_coefficients.adv_cells_for_edge[0, 0] = 7
    assert prepared.mesh is mesh and prepared.state is state
    assert prepared.advection_coefficients is coefficients
    assert prepared.expected_execution_fingerprint is fingerprint
    assert state.rho.tobytes() == original_rho
    assert coefficients.adv_cells_for_edge.tobytes() == original_stencil


def _forecast_kwargs(tmp_path):
    return {
        "host": {"config": SimpleNamespace(config_local_timestep=False)},
        "schedule": {}, "cache_root": tmp_path / "cache", "output_root": tmp_path / "output",
        "arwen_checkout": tmp_path / "engine", "source_receipt": {}, "authority_receipt": {},
        "fingerprint_every": 9, "stop_on_refusal": False, "grid_path": None,
        "park_physics_tier": False, "required_free_bytes": None,
        "local_timestep_classing_path": None,
    }


def test_none_forecast_option_forwards_through_adapter_to_normal_preflight(tmp_path, monkeypatch):
    from woof.hex.drivers import samehost_physics_scope_v841 as scope
    monkeypatch.setattr(scope, "scoped_physics_resources", lambda: pytest.fail("preflight reached physics allocation"))
    original = lambda **kwargs: pytest.fail("same-host selection did not replace the forecast")
    forecast = SimpleNamespace(proof=SimpleNamespace(), execute_forecast=original)
    with adapter.installed(forecast, devices=(0, 0), layouts_path=Path("unused.npz"), part_path=Path("unused.part.2")):
        # A missing grid is the ordinary named preflight refusal.  Reaching
        # it proves the optional e6 argument was accepted and forwarded.
        with pytest.raises(ValueError, match="layouts need the grid path"):
            forecast.execute_forecast(**_forecast_kwargs(tmp_path))
    assert forecast.execute_forecast is original


def test_door_iau_and_balance_keywords_reach_the_rank_executor(tmp_path, monkeypatch):
    # Breakage it prevents: the door forwards iau=None and balance_series=False
    # on every run since W1 (918a3774); the rank executor did not take them, so
    # every same-host forecast stopped with "TypeError: _execute_samehost() got
    # an unexpected keyword argument 'iau'" before its ranks were built.
    import inspect
    from contextlib import contextmanager
    from woof.hex.drivers import samehost_physics_scope_v841 as scope

    class Reached(Exception):
        pass

    signature = inspect.signature(adapter._execute_samehost)

    def executor(forecast, **kwargs):
        signature.bind(forecast, **kwargs)   # TypeError on any keyword it does not take
        raise Reached

    @contextmanager
    def no_resources():
        yield SimpleNamespace(receipt=lambda: {})

    monkeypatch.setattr(adapter, "_execute_samehost", executor)
    monkeypatch.setattr(scope, "scoped_physics_resources", no_resources)
    forecast = SimpleNamespace(proof=SimpleNamespace(), execute_forecast=lambda **kwargs: None)
    kwargs = _forecast_kwargs(tmp_path)
    kwargs.update(grid_path=tmp_path / "grid.nc", iau=None, balance_series=False,
                  tc_climate=None)
    with adapter.installed(forecast, devices=(0, 0), layouts_path=Path("unused.npz"), part_path=Path("unused.part.2")):
        with pytest.raises(Reached):
            forecast.execute_forecast(**kwargs)
    # IAU reaches the rank executor now (each rank admits the increment on its own cells and edges);
    # the balance series is still refused by name, the ranks having no sampler.
    kwargs.update(iau={"route": "native"})
    with adapter.installed(forecast, devices=(0, 0), layouts_path=Path("unused.npz"), part_path=Path("unused.part.2")):
        with pytest.raises(Reached):
            forecast.execute_forecast(**kwargs)
    kwargs.update(iau=None, balance_series=True)
    with adapter.installed(forecast, devices=(0, 0), layouts_path=Path("unused.npz"), part_path=Path("unused.part.2")):
        with pytest.raises(ValueError, match="no balance-series sampler"):
            forecast.execute_forecast(**kwargs)


def test_enabled_tc_climate_refuses_before_physics_allocation(tmp_path, monkeypatch):
    from woof.hex.drivers import samehost_physics_scope_v841 as scope
    monkeypatch.setattr(scope, "scoped_physics_resources",
                        lambda: pytest.fail("unsupported TC diagnostics reached physics allocation"))
    forecast = SimpleNamespace(proof=SimpleNamespace(), execute_forecast=lambda **kwargs: None)
    kwargs = _forecast_kwargs(tmp_path)
    kwargs.update(grid_path=tmp_path / "grid.nc", tc_climate=tmp_path / "tc.toml")
    with adapter.installed(forecast, devices=(0, 0), layouts_path=Path("unused.npz"), part_path=Path("unused.part.2")):
        with pytest.raises(ValueError, match="TC|tc-climate"):
            forecast.execute_forecast(**kwargs)


def test_explicit_lts_classing_refuses_before_any_cuda_or_file_contact(tmp_path, monkeypatch):
    from woof.hex import partition_assets_v841 as assets
    from woof.hex.drivers import samehost_physics_scope_v841 as scope
    monkeypatch.setattr(scope, "scoped_physics_resources", lambda: pytest.fail("preflight reached physics allocation"))
    monkeypatch.setattr(assets, "load_layouts_npz", lambda *args, **kwargs: pytest.fail("refusal reached asset I/O"))
    forecast = SimpleNamespace(proof=SimpleNamespace(), execute_forecast=lambda **kwargs: None)
    kwargs = _forecast_kwargs(tmp_path)
    kwargs["local_timestep_classing_path"] = Path("unsupported-rates.txt")
    with adapter.installed(forecast, devices=(0, 0), layouts_path=Path("unused.npz"), part_path=Path("unused.part.2")):
        with pytest.raises(ValueError, match="native step schedule"):
            forecast.execute_forecast(**kwargs)


def _boundary_record():
    return {
        "atmosphere": {"model_time_seconds": 60., "state": {"fields": {"rho": {"sha256": "a"}}},
                       "saved_diagnostics": {"fields": {"theta_m": {"sha256": "b"}}}},
        "backend": {"arrays": {"soil": {"sha256": "c"}},
                    "scalars": {"identity/n_columns": 12,
                                "identity/constructor_identity_sha256": "whole",
                                "seam/scalars/step_index": 1}},
    }


def test_reference_gate_refuses_execution_metadata_despite_identical_array_bytes():
    from woof.hex.drivers.samehost_checks_v841 import compare_boundary_records

    reference = _boundary_record()
    candidate = deepcopy(reference)
    candidate["backend"]["scalars"]["seam/scalars/step_index"] = 2
    # Array equality alone cannot qualify a boundary at another backend step.
    verdict = compare_boundary_records(candidate, reference)
    assert verdict["state_invariant"]
    assert verdict["unexpected_metadata_diffs"]
    with pytest.raises(RuntimeError, match=r"step 1:.*metadata=.*seam/scalars/step_index"):
        adapter._checked_boundary_comparison(candidate, reference, 1)


def test_reference_gate_preserves_only_declared_constructor_metadata_divergences():
    reference = _boundary_record()
    candidate = deepcopy(reference)
    candidate["backend"]["scalars"]["identity/n_columns"] = 8
    candidate["backend"]["scalars"]["identity/constructor_identity_sha256"] = "rank0"
    verdict = adapter._checked_boundary_comparison(candidate, reference, 1)
    assert verdict["state_invariant"]
    assert verdict["unexpected_metadata_diffs"] == []
    assert len(verdict["metadata_diffs"]) == 2


def _native_blocks(kind):
    layouts = _layouts()
    count = layouts[0]._global_edges if kind == "edge" else layouts[0]._global_cells
    truth = np.arange(2 * count, dtype=np.float32).reshape(2, count)
    pad = np.array([0x80000000, 0], dtype=np.uint32).reshape(2, 1).view(np.float32)
    mappings = []
    for layout in layouts:
        local = truth[..., getattr(layout, f"{kind}_l2g")].copy()
        local[..., getattr(layout, f"n_owned_{'edges' if kind == 'edge' else 'cells'}"):] = 42
        mappings.append({"field": np.concatenate((local, pad), axis=-1)})
    return layouts, mappings, np.concatenate((truth, pad), axis=-1)


@pytest.mark.parametrize("kind", ["cell", "edge"])
def test_native_tail_join_keeps_dtype_signed_zero_and_all_owned_bytes(kind):
    layouts, mappings, expected = _native_blocks(kind)
    result = assemble_owned_arrays(layouts, mappings, kinds={"field": kind}, native_pads=("field",))["field"]
    assert result.dtype == expected.dtype == np.dtype(np.float32)
    assert result.shape == expected.shape
    assert result.tobytes() == expected.tobytes()
    assert result.view(np.uint32)[0, -1] == 0x80000000


@pytest.mark.parametrize("mutation, reason", [("sign", "columns differ bytewise"), ("dtype", "columns differ bytewise"), ("missing", "garbage extent differs")])
def test_native_tail_join_refuses_different_tail_bits_dtype_and_extents(mutation, reason):
    layouts, mappings, _ = _native_blocks("edge")
    if mutation == "sign":
        mappings[1]["field"].view(np.uint32)[0, -1] = 0
    elif mutation == "dtype":
        mappings[1]["field"] = mappings[1]["field"].astype(np.float64)
    else:
        mappings[1]["field"] = mappings[1]["field"][..., :-1]
    with pytest.raises(ExecutorError, match=reason):
        assemble_owned_arrays(layouts, mappings, kinds={"field": "edge"}, native_pads=("field",))


def test_native_tail_join_requires_explicit_entity_kind():
    layouts, mappings, _ = _native_blocks("edge")
    with pytest.raises(ExecutorError, match="explicit entity kind"):
        assemble_owned_arrays(layouts, mappings, native_pads=("field",))


def test_existing_writer_omits_padded_normal_u_but_publishes_unpadded_control(tmp_path, monkeypatch):
    from woof.hex.drivers import run_cuda_v841_full_physics_x4 as proof
    for name, count in (("N_CELLS", 12), ("N_EDGES", 24), ("N_LEVELS", 2), ("N_INTERFACES", 3), ("N_SOIL_LEVELS", 4)):
        monkeypatch.setattr(proof, name, count)
    layouts, mappings, expected = _native_blocks("edge")
    for mapping in mappings:
        mapping["normal_u"] = mapping.pop("field")
    joined = assemble_owned_arrays(layouts, mappings, kinds={"normal_u": "edge"}, native_pads=("normal_u",))["normal_u"]
    assert joined.tobytes() == expected.tobytes()
    static = {name: np.arange(count, dtype=np.int32 if name.startswith("index") else np.float32)
        for name, count in (("indexToCellID", 12), ("latCell", 12), ("lonCell", 12), ("ter", 12), ("indexToEdgeID", 24), ("latEdge", 24), ("lonEdge", 24))}
    rho = np.zeros((2, 12), dtype=np.float32)
    padded_path = tmp_path / "padded.nc"
    proof.write_snapshot_netcdf(padded_path, {"arrays": {"normal_u": joined, "rho": rho}}, static)
    with Dataset(padded_path) as dataset:
        assert "rho" in dataset.variables
        assert "normal_u" not in dataset.variables
    unpadded_path = tmp_path / "unpadded.nc"
    proof.write_snapshot_netcdf(unpadded_path, {"arrays": {"normal_u": joined[..., :-1], "rho": rho}}, static)
    with Dataset(unpadded_path) as dataset:
        assert dataset.variables["normal_u"].dimensions == ("Time", "nEdges", "nVertLevels")
        assert np.asarray(dataset.variables["normal_u"][0]).T.tobytes() == joined[..., :-1].tobytes()


def test_regional_scalar_boundary_adjust_exchanges_first():
    """Public conus12km N2/N4/N8 differed from N1 at step 1 until the stage scalars were exchanged
    before the relaxation filter read neighbour cells; one wrapper, exchange then the original."""
    from woof.hex.drivers.samehost_forecast_v841 import _exchange_before_scalar_boundary_adjust

    calls = []

    class Regional:
        def bdy_adjust_scalars(self, scalars, rk_step):
            calls.append(("adjust", scalars, rk_step))

    class Exchanger:
        def round_transport(self, scalars):
            calls.append(("D", scalars))

    regional, exchanger = Regional(), Exchanger()
    _exchange_before_scalar_boundary_adjust(regional, exchanger)
    _exchange_before_scalar_boundary_adjust(regional, exchanger)
    regional.bdy_adjust_scalars("q", 2)
    assert calls == [("D", "q"), ("adjust", "q", 2)]


def test_one_card_samehost_forwards_history_session_to_actual_executor(tmp_path, monkeypatch):
    import inspect
    from contextlib import contextmanager
    from woof.hex.drivers import samehost_physics_scope_v841 as scope
    signature = inspect.signature(adapter._execute_samehost)
    callback = lambda event: None
    session = lambda: None
    seen = []
    class Reached(Exception):
        pass
    def executor(forecast, **kwargs):
        signature.bind(forecast, **kwargs)
        seen.append(kwargs)
        raise Reached
    @contextmanager
    def resources():
        yield SimpleNamespace(receipt=lambda: {})
    monkeypatch.setattr(scope, 'scoped_physics_resources', resources)
    monkeypatch.setattr(adapter, '_execute_samehost', executor)
    forecast = SimpleNamespace(proof=SimpleNamespace(), execute_forecast=lambda **kw: None)
    for consumer in ({'on_history_ready': callback}, {'history_product_session': session}):
        kwargs = _forecast_kwargs(tmp_path)
        kwargs.update(grid_path=tmp_path / 'grid.nc', history_product_max_pending=2,
                      required_product_ids=('map/F003',), **consumer)
        with adapter.installed(forecast, devices=(0,), layouts_path=Path('layouts.npz'), part_path=Path('part.1')):
            with pytest.raises(Reached):
                forecast.execute_forecast(**kwargs)
        assert all(seen[-1][key] is value for key, value in consumer.items())
        assert seen[-1]['required_product_ids'] == ('map/F003',)
