"""CPU contracts for Phase-5 child initialization ordering and N1 tooling."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import threading

import numpy as np
import pytest

import woof.ingest.nest_init as ni
from woof.case_data import SourceOrography
from woof.core.grid import BaseState
from woof.experiment import VerticalConfig
from woof.ingest.grib import Era5Snapshot
from woof.ingest.hrrr import HrrrNativeSnapshot
from woof.ingest.horiz import HorizontalSnapshot
from woof.static.lambert import LambertGrid
from woof.verify.cases.wk82 import wk82_sounding, wk82_theta
from woof.verify.npref import np_sint


class _DeviceSintPayload:
    """Device-shaped CPU witness, distinct from a NumPy preparation array."""
    def __init__(self, value):
        self.value = np.ascontiguousarray(value)

    def __getattr__(self, name):
        return getattr(self.value, name)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)

    def __getitem__(self, index):
        return self.value[index]

    def __setitem__(self, index, value):
        self.value[index] = np.asarray(value)


@pytest.mark.parametrize('stagger', ['', 'x', 'y'])
@pytest.mark.parametrize('dimensions', [2, 3])
def test_full_device_reconstruction_stages_only_exact_host_donors(
        monkeypatch, stagger, dimensions):
    import sys
    from woof.core.nest_interp import register_nest, window_registration
    reg = register_nest(nri=3, nrj=3, i_parent_start=7, j_parent_start=8,
                        child_nx=9, child_ny=12, parent_nx=64, parent_ny=64,
                        stagger=stagger, wrapper='interp')
    shape = (reg.nyp, reg.nxp) if dimensions == 2 else (2, reg.nyp, reg.nxp)
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    uploads, seen = [], []
    def upload(value):
        assert value.flags.c_contiguous
        uploads.append(value.copy())
        return _DeviceSintPayload(value)
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(asarray=upload))
    def interpolate(payload, cropped, *, preparation):
        assert preparation is True
        assert isinstance(payload, _DeviceSintPayload)
        seen.append(cropped)
        return np_sint(payload.value, cropped, dtype=np.float64).astype(np.float32)
    monkeypatch.setattr(ni, 'sint', interpolate)
    observed = ni._reconstruction_sint(source, reg, device_windows=True)
    cropped, donor = window_registration(reg, (slice(0, reg.nyc), slice(0, reg.nxc)))
    assert len(uploads) == len(seen) == 1
    np.testing.assert_array_equal(uploads[0], source[(...,)+donor])
    assert uploads[0].nbytes < source.nbytes
    assert (seen[0].nyc, seen[0].nxc, seen[0].ystag, seen[0].xstag) == (
        reg.nyc, reg.nxc, reg.ystag, reg.xstag)
    np.testing.assert_array_equal(observed,
        np_sint(source, reg, dtype=np.float64).astype(np.float32))


@pytest.mark.parametrize('backend', ['numpy-preparation', 'resident-device'])
def test_original_reconstruction_backend_keeps_original_registration(monkeypatch, backend):
    source = np.zeros((2, 12, 12), np.float32)
    if backend == 'resident-device':
        source = _DeviceSintPayload(source)
    registration, result = object(), object()
    seen = []
    monkeypatch.setattr(ni, 'sint', lambda value, reg, **kwargs:
                        seen.append((value, reg, kwargs)) or result)
    assert ni._reconstruction_sint(source, registration,
        device_windows=backend == 'resident-device') is result
    assert len(seen) == 1 and seen[0][0] is source and seen[0][1] is registration
    assert seen[0][2] == {"preparation": True}


def test_pending_child_inputs_overlap_and_collect_in_parent_order(monkeypatch):
    domains = tuple(SimpleNamespace(grid_id=grid_id) for grid_id in (2, 3, 4))
    grids = {domain.grid_id: object() for domain in domains}
    barrier = threading.Barrier(len(domains))
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def prepare(domain, grid, catalog, source_orography,
                preprocess_backend, preprocess_workers, cpu_bridge):
        nonlocal active, peak_active
        assert grid is grids[domain.grid_id]
        assert catalog == "catalog"
        assert source_orography == "orography"
        assert preprocess_backend == "cpu"
        assert preprocess_workers == 1
        assert cpu_bridge is None
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        barrier.wait(timeout=5.0)
        with lock:
            active -= 1
        # Reverse completion order without relying on process scheduling.
        threading.Event().wait((4 - domain.grid_id) * 0.001)
        return SimpleNamespace(domain=domain)

    monkeypatch.setattr(ni, "_prepare_child_input_on_grid", prepare)
    # A three-thread host: the budget is the host's, split over the slots.
    monkeypatch.setattr("woof.ingest.cpu_backend.automatic_workers", lambda: 3)
    pending = ni.PendingChildInputs(
        domains, grids, "catalog", "orography", workers=3)
    result = pending.result()

    assert peak_active == 3
    assert [item.domain.grid_id for item in result] == [2, 3, 4]
    assert pending._closed
    assert pending.worker_budget == 3
    assert pending.allocated_workers == 3


def test_nested_input_catalog_binds_ordered_source_to_static_catalog():
    snapshots = tuple(HrrrNativeSnapshot(
        valid_time=datetime(2026, 7, 20, hour), forecast_hour=hour,
        i_start=0, j_start=0, ny=1, nx=1, fields={})
        for hour in (0, 1))
    static_catalog = object()
    catalog = ni.NestedInputCatalog(
        snapshots=snapshots, static_catalog=static_catalog,
        inventory=("LANDSEA",))

    assert catalog.valid_times == tuple(
        datetime(2026, 7, 20, hour) for hour in (0, 1))
    assert catalog.static_catalog is static_catalog
    with pytest.raises(ValueError, match="unique increasing"):
        ni.NestedInputCatalog(
            snapshots=(snapshots[0], snapshots[0]),
            static_catalog=static_catalog)


def test_nested_input_catalog_preserves_era5_soilgeo_units_through_mapping(
        monkeypatch):
    valid_time = datetime(1974, 4, 3, 12)
    source = Era5Snapshot(
        valid_time=valid_time, levels_hpa=np.asarray([1000.0, 500.0]),
        latitude=np.asarray([30.0, 31.0]),
        longitude=np.asarray([-100.0, -99.0]),
        fields={"SOILGEO": np.full((2, 2), 981.0)})
    raw = SimpleNamespace(
        snapshots=(source,), inventory=("SOILGEO",), files=(),
        units={"SOILGEO": "m2 s-2"},
        provenance={"product": "ERA5"})
    static_catalog = object()
    catalog = ni.NestedInputCatalog.from_source_catalog(raw, static_catalog)
    child = SimpleNamespace(
        grid_id=2, start_time=valid_time,
        run=SimpleNamespace(moist=True, terrain_opt=1))
    grid = object()
    # HGT_M: since c1227a11a the child door checks the terrain slope before
    # building a coordinate; a flat plane needs no pass.
    static = {
        "LANDMASK": np.ones((2, 2)),
        "LU_INDEX": np.ones((2, 2)),
        "HGT_M": np.zeros((2, 2)),
    }
    backend = SimpleNamespace(receipt=lambda: {"backend": "cpu"})
    monkeypatch.setattr(
        "woof.ingest.preprocess_backend.resolve_preprocess_backend",
        lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(
        ni, "build_static_for_domain",
        lambda actual_grid, actual_catalog, grid_id: static
        if (actual_grid is grid and actual_catalog is static_catalog
            and grid_id == 2)
        else pytest.fail("wrong static binding"))
    monkeypatch.setattr(
        ni, "geog_selection_from_catalog",
        lambda *_args: SimpleNamespace(
            landuse_global_attrs=lambda: {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}))
    monkeypatch.setattr(
        ni, "interpolate_lake_skin_temperature",
        lambda *_args: np.full((2, 2), np.nan))
    horizontal = SimpleNamespace(fields={})

    def map_era5(snapshot, actual_grid, **options):
        assert snapshot is source
        assert actual_grid is grid
        bound = options["source_orography_catalog"]
        assert bound is catalog
        assert bound.units["SOILGEO"] == "m2 s-2"
        assert bound.provenance["product"] == "ERA5"
        return horizontal

    monkeypatch.setattr(ni, "interpolate_era5_to_lambert", map_era5)
    prepared = ni._prepare_child_input_on_grid(
        child, grid, catalog, preprocess_backend="cpu")

    assert prepared.horizontal is horizontal
    assert prepared.preprocess_receipt["source_adapter"] == \
        "regular-grid-native-state-v1"


def test_pending_child_inputs_carries_the_cause_and_its_remedy(monkeypatch):
    """A worker that reads only the last exception line keys its retry on
    the refusal's remedy; the wrapper used to drop both."""

    domain = SimpleNamespace(grid_id=2)
    refusal = ("no surface-matched donor within 8 cells.\n"
               "Raising surface_fallback_radius_cells to 24 lets it reach "
               "the coast")

    def fail(*_args):
        raise ValueError(refusal)

    monkeypatch.setattr(ni, "_prepare_child_input_on_grid", fail)
    pending = ni.PendingChildInputs(
        (domain,), {2: object()}, object(), None, workers=1)
    with pytest.raises(RuntimeError) as caught:
        pending.result()
    message = str(caught.value)
    assert message.startswith("d02 independent input preparation failed: ")
    assert "Raising surface_fallback_radius_cells to 24" in message
    assert caught.value.__cause__.args == (refusal,)


def test_one_child_takes_the_hosts_threads_not_the_job_slots(monkeypatch):
    """An HRRR 3-1 tree's one 1 km nest ran on 8 of 48 cores (F-CHILD-INIT-1KM)."""
    seen = []
    monkeypatch.setattr("woof.ingest.cpu_backend.automatic_workers", lambda: 48)
    monkeypatch.setattr(
        ni, "_prepare_child_input_on_grid",
        lambda domain, *args: seen.append(args[4]) or SimpleNamespace(domain=domain))
    domain = SimpleNamespace(grid_id=2)
    pending = ni.PendingChildInputs((domain,), {2: object()}, object(), None,
                                    workers=8)
    pending.result()
    assert seen == [48]
    assert dict(pending.preprocess_workers_by_domain) == {2: 48}
    # Two children prepared at once split the host's threads between them.
    two = ni.PendingChildInputs(
        (SimpleNamespace(grid_id=2), SimpleNamespace(grid_id=3)),
        {2: object(), 3: object()}, object(), None, workers=2)
    two.close()
    assert dict(two.preprocess_workers_by_domain) == {2: 24, 3: 24}


def test_blend_frame_capture_is_the_whole_capture_on_the_frame():
    """The frame-only SINT is the full SINT to the bit wherever the blend reads it."""
    from woof.core.nest_interp import blend_terrain, register_nest, sint

    rng = np.random.default_rng(7)
    nxp, nyp, nx, ny, nz = 40, 34, 61, 52, 4
    parent = SimpleNamespace(
        ht=rng.uniform(0, 3000, (nyp, nxp)).astype(np.float32),
        mub2d=rng.uniform(6e4, 9.5e4, (nyp, nxp)).astype(np.float32),
        phb=rng.uniform(0, 2e5, (nz, nyp, nxp)).astype(np.float32))
    child_dc = SimpleNamespace(
        parent_grid_ratio=3, i_parent_start=5, j_parent_start=6,
        run=SimpleNamespace(nx=nx, ny=ny))
    parent_node = SimpleNamespace(
        cfg=SimpleNamespace(run=SimpleNamespace(nx=nxp, ny=nyp)), state=parent)
    full = ni._capture_parent_blend_fields(child_dc, parent_node)
    framed = ni._capture_parent_blend_frame(child_dc, parent_node, frame=10)
    reg = register_nest(nri=3, nrj=3, i_parent_start=5, j_parent_start=6,
                        child_nx=nx, child_ny=ny, parent_nx=nxp, parent_ny=nyp)
    np.testing.assert_array_equal(full[2], sint(parent.phb, reg))
    for whole, frame in zip(full, framed):
        assert frame.shape == whole.shape and frame.dtype == whole.dtype
        ring = np.ones(whole.shape[-2:], dtype=bool)
        ring[10:-10, 10:-10] = False
        np.testing.assert_array_equal(frame[..., ring], whole[..., ring])
        fine = rng.uniform(0, 1, whole.shape)
        a, b = fine.copy(), fine.copy()
        blend_terrain(whole.astype(np.float64), a, spec_bdy_width=5, blend_width=5)
        blend_terrain(frame.astype(np.float64), b, spec_bdy_width=5, blend_width=5)
        assert a.tobytes() == b.tobytes()
    # A frame that covers the child captures it whole.
    assert ni._blend_frame_windows(20, 61, 10) is None


def test_level_slabs_write_the_one_call_bytes(monkeypatch):
    from woof.core.nest_interp import blend_terrain

    monkeypatch.setattr("woof.ingest.cpu_backend.automatic_workers", lambda: 7)
    monkeypatch.setattr(ni, "_LEVEL_SLAB_MIN_CELLS", 1)
    rng = np.random.default_rng(3)
    coarse, fine = rng.uniform(0, 1, (2, 23, 40, 50))
    one = fine.copy()
    blend_terrain(coarse, one, spec_bdy_width=5, blend_width=5)
    sliced = fine.copy()
    ni._on_level_slabs(23, lambda k0, k1: blend_terrain(
        coarse[k0:k1], sliced[k0:k1], spec_bdy_width=5, blend_width=5),
        cells=sliced.size)
    assert one.tobytes() == sliced.tobytes()


def test_adjust_tempqv_on_level_slabs_writes_the_one_call_bytes(monkeypatch):
    from woof.core.nest_interp import adjust_tempqv

    monkeypatch.setattr("woof.ingest.cpu_backend.automatic_workers", lambda: 4)
    monkeypatch.setattr(ni, "_LEVEL_SLAB_MIN_CELLS", 1)
    rng = np.random.default_rng(5)
    nz, ny, nx = 9, 6, 7
    mub = rng.uniform(8e4, 9e4, (ny, nx)).astype(np.float32)
    save = (mub + rng.uniform(-300, 300, (ny, nx))).astype(np.float32)
    c3 = np.linspace(1.0, 0.0, nz).astype(np.float32)
    c4 = np.linspace(0.0, 4000.0, nz).astype(np.float32)
    th = rng.uniform(-10, 40, (nz, ny, nx)).astype(np.float32)
    pp = rng.uniform(-50, 50, (nz, ny, nx)).astype(np.float32)
    qv = rng.uniform(1e-5, 1.5e-2, (nz, ny, nx)).astype(np.float32)
    th1, qv1 = th.copy(), qv.copy()
    adjust_tempqv(mub, save, c3, c4, 1e4, th1, pp, qv1)
    th2, qv2 = th.copy(), qv.copy()
    ni._on_level_slabs(nz, lambda k0, k1: adjust_tempqv(
        mub, save, c3[k0:k1], c4[k0:k1], 1e4, th2[k0:k1], pp[k0:k1],
        qv2[k0:k1]), cells=th2.size)
    assert th1.tobytes() == th2.tobytes() and qv1.tobytes() == qv2.tobytes()


def test_pending_child_inputs_empty_hierarchy_allocates_no_workers():
    pending = ni.PendingChildInputs((), {}, object(), None, workers=8)
    assert pending.result() == ()
    assert pending.workers == 0
    assert dict(pending.preprocess_workers_by_domain) == {}
    assert pending.allocated_workers == 0
    assert pending._closed


def test_pending_child_inputs_validates_domain_grid_inventory_before_launch():
    duplicate = SimpleNamespace(grid_id=2)
    with pytest.raises(ValueError, match="duplicate"):
        ni.PendingChildInputs(
            (duplicate, duplicate), {2: object()}, object(), None, workers=2)
    with pytest.raises(ValueError, match=r"d03"):
        ni.PendingChildInputs(
            (SimpleNamespace(grid_id=3),), {}, object(), None, workers=1)


def _prepared_for_barrier(grid_id, parent_id):
    return ni.PreparedChildInput(
        domain=SimpleNamespace(grid_id=grid_id, parent_id=parent_id),
        grid=object(), static_fields={}, horizontal=object(),
        declared_orography=None, lake_mask=np.zeros((1, 1), dtype=bool),
        lake_skin_temperature=np.full((1, 1), np.nan),
        preprocess_backend=object(), preprocess_receipt={"backend": "test"},
        preparation_seconds=0.0)


def test_finalize_prepared_child_chain_is_parent_before_child(monkeypatch):
    calls = []

    def finalize(prepared, parent, vertical, **options):
        calls.append((prepared.domain.grid_id, parent.cfg.grid_id, options))
        return SimpleNamespace(grid=prepared.grid, state=object())

    monkeypatch.setattr(ni, "finalize_prepared_child", finalize)
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), grid=object(), state=object())
    prepared = tuple(
        _prepared_for_barrier(grid_id, grid_id - 1)
        for grid_id in (2, 3, 4))
    results = ni.finalize_prepared_child_chain(
        prepared, root, object(), scratch_arena="arena",
        dycore_state_workspace="workspace", sfcp_to_sfcp=False,
        soil_layer_contract="soil-contract")

    assert len(results) == 3
    assert [(child, parent) for child, parent, _ in calls] == [
        (2, 1), (3, 2), (4, 3)]
    assert all(options == {
        "scratch_arena": "arena",
        "dycore_state_workspace": "workspace",
        "state_backend": "cuda",
        "sfcp_to_sfcp": False,
        "soil_layer_contract": "soil-contract",
    } for _child, _parent, options in calls)


def test_finalize_prepared_child_chain_rejects_missing_or_duplicate_parent(
        monkeypatch):
    monkeypatch.setattr(
        ni, "finalize_prepared_child",
        lambda prepared, *_args, **_kwargs: SimpleNamespace(
            grid=prepared.grid, state=object()))
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), grid=object(), state=object())
    with pytest.raises(ValueError, match=r"d03.*before parent d02"):
        ni.finalize_prepared_child_chain(
            (_prepared_for_barrier(3, 2),), root, object())
    with pytest.raises(ValueError, match=r"repeats d01"):
        ni.finalize_prepared_child_chain(
            (_prepared_for_barrier(1, 1),), root, object())


def test_initialize_child_chain_parallel_composes_prepare_and_barrier(
        monkeypatch):
    prepared = tuple(
        _prepared_for_barrier(grid_id, grid_id - 1)
        for grid_id in (2, 3, 4))
    root = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1), grid=object(), state=object())
    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1), *(
            item.domain for item in prepared)),
        vertical=object())
    events = []

    class Pending:
        def __enter__(self):
            events.append("enter")
            return self

        def iter_parent_order(self):
            events.append("iter")
            yield from prepared

        def __exit__(self, *_args):
            events.append("exit")

    def start(actual_exp, catalog, source_orography, **options):
        assert actual_exp is exp
        assert catalog == "catalog"
        assert source_orography == "orography"
        assert options == {
            "workers": 8, "preprocess_backend": "cpu",
            "cpu_bridge": "bridge"}
        events.append("start")
        return Pending()

    expected = (object(), object(), object())

    def finalize(actual, actual_root, vertical, **options):
        events.append("finalize")
        assert tuple(actual) == prepared
        assert actual_root is root
        assert vertical is exp.vertical
        assert options == {
            "scratch_arena": "arena",
            "dycore_state_workspace": "workspace",
            "state_backend": "cuda",
            "sfcp_to_sfcp": False,
            "soil_layer_contract": "soil-contract"}
        return expected

    monkeypatch.setattr(ni, "start_child_input_preparations", start)
    monkeypatch.setattr(ni, "finalize_prepared_child_chain", finalize)
    result = ni.initialize_child_chain_parallel(
        exp, root, "catalog", "orography", workers=8,
        preprocess_backend="cpu", cpu_bridge="bridge",
        scratch_arena="arena", dycore_state_workspace="workspace",
        sfcp_to_sfcp=False, soil_layer_contract="soil-contract")

    assert result is expected
    assert events == ["start", "enter", "finalize", "iter", "exit"]


def test_initialize_child_chain_parallel_rejects_wrong_root_before_launch(
        monkeypatch):
    exp = SimpleNamespace(
        domains=(SimpleNamespace(grid_id=1),), vertical=object())
    root = SimpleNamespace(cfg=SimpleNamespace(grid_id=2))
    monkeypatch.setattr(
        ni, "start_child_input_preparations",
        lambda *_args, **_kwargs: pytest.fail("preparation must not launch"))
    with pytest.raises(ValueError, match="root node"):
        ni.initialize_child_chain_parallel(exp, root, object())


@pytest.mark.requires_capability("masked_stencil_bridge")
def test_prepare_child_input_dispatches_hrrr_on_own_static_landmask(
        monkeypatch):
    valid_time = datetime(2026, 7, 20)
    source = HrrrNativeSnapshot(
        valid_time=valid_time, forecast_hour=0,
        i_start=0, j_start=0, ny=4, nx=5, fields={})
    static_catalog = object()
    catalog = SimpleNamespace(
        snapshots=(source,), valid_times=(valid_time,), inventory=(), files=(),
        static_catalog=static_catalog)
    child = SimpleNamespace(
        grid_id=2, start_time=valid_time,
        run=SimpleNamespace(moist=True, terrain_opt=1))
    grid = object()
    landmask = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    # HGT_M: since c1227a11a the child door checks the terrain slope before
    # building a coordinate; a flat plane needs no pass.
    static = {
        "LANDMASK": landmask,
        "LU_INDEX": np.asarray([[21, 1], [16, 21]]),
        "HGT_M": np.zeros((2, 2)),
    }
    horizontal = HorizontalSnapshot(
        valid_time=valid_time, levels_hpa=np.asarray([1000.0]), fields={
            "SKINTEMP": np.asarray([[281.0, 282.0], [283.0, 284.0]])})

    class Backend:
        @staticmethod
        def receipt():
            return {"backend": "cpu", "workers": 2}

    backend = Backend()
    monkeypatch.setattr(
        "woof.ingest.preprocess_backend.resolve_preprocess_backend",
        lambda *_args, **_kwargs: backend)
    def build_static(actual_grid, actual_catalog, grid_id):
        assert actual_grid is grid
        assert actual_catalog is static_catalog
        assert grid_id == 2
        return static

    monkeypatch.setattr(ni, "build_static_for_domain", build_static)
    monkeypatch.setattr(
        ni, "geog_selection_from_catalog",
        lambda actual_catalog, grid_id: (
            SimpleNamespace(landuse_global_attrs=lambda: {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13})
            if actual_catalog is static_catalog and grid_id == 2
            else pytest.fail("wrong static catalog binding")))
    observed = {}

    def map_hrrr(snapshot, actual_grid, **options):
        assert snapshot is source
        assert actual_grid is grid
        np.testing.assert_array_equal(options["target_landmask"], landmask)
        assert options["backend"] is backend
        options["soil_mapping_report"]["status"] = "PASS"
        observed["mapped"] = True
        return horizontal

    monkeypatch.setattr(ni, "interpolate_hrrr_to_lambert", map_hrrr)
    monkeypatch.setattr(
        ni, "interpolate_era5_to_lambert",
        lambda *_args, **_kwargs: pytest.fail("ERA5 mapper must not run"))
    monkeypatch.setattr(
        ni, "interpolate_lake_skin_temperature",
        lambda *_args, **_kwargs: pytest.fail("regular lake mapper must not run"))

    prepared = ni._prepare_child_input_on_grid(
        child, grid, catalog, preprocess_backend="cpu",
        preprocess_workers=2)

    assert observed == {"mapped": True}
    np.testing.assert_array_equal(
        prepared.lake_skin_temperature,
        np.asarray([[281.0, np.nan], [np.nan, 284.0]]))
    assert prepared.declared_orography is None
    assert prepared.preprocess_receipt["source_adapter"] == \
        "hrrr-native-state-v1"
    assert prepared.preprocess_receipt["soil_mapping"] == {"status": "PASS"}


def _hrrr_child_case(monkeypatch, static, *, catalog_extra=None):
    """The HRRR child door of the test above, around one static build."""
    valid_time = datetime(2026, 7, 20)
    source = HrrrNativeSnapshot(
        valid_time=valid_time, forecast_hour=0,
        i_start=0, j_start=0, ny=4, nx=5, fields={})
    static_catalog = object()
    catalog = SimpleNamespace(
        snapshots=(source,), valid_times=(valid_time,), inventory=(), files=(),
        static_catalog=static_catalog, **(catalog_extra or {}))
    child = SimpleNamespace(
        grid_id=2, start_time=valid_time,
        run=SimpleNamespace(moist=True, terrain_opt=1))
    horizontal = HorizontalSnapshot(
        valid_time=valid_time, levels_hpa=np.asarray([1000.0]), fields={
            "SKINTEMP": np.asarray([[281.0, 282.0], [283.0, 284.0]])})

    class Backend:
        @staticmethod
        def receipt():
            return {"backend": "cpu", "workers": 2}

    monkeypatch.setattr(
        "woof.ingest.preprocess_backend.resolve_preprocess_backend",
        lambda *_args, **_kwargs: Backend())
    monkeypatch.setattr(ni, "build_static_for_domain",
                        lambda grid, catalog, grid_id: dict(static))
    monkeypatch.setattr(
        ni, "geog_selection_from_catalog",
        lambda actual_catalog, grid_id: SimpleNamespace(
            landuse_global_attrs=lambda: {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
                                          "ISWATER": 17, "ISLAKE": 21,
                                          "ISICE": 15, "ISURBAN": 13}))

    def map_hrrr(snapshot, actual_grid, **options):
        options["soil_mapping_report"]["status"] = "PASS"
        return horizontal

    monkeypatch.setattr(ni, "interpolate_hrrr_to_lambert", map_hrrr)
    return child, catalog


def test_a_corridor_child_is_born_on_its_corridor_crop(monkeypatch):
    """Plan call 5 (2.9.0): a child whose preparation sealed a statics
    corridor initializes on that corridor's crop at its declared placement,
    not on a footprint build, so its terrain is the ground its moves crop
    from; the footprint's own slope pass does not run on it."""
    from woof.static.corridor import CorridorBirth
    footprint = {"LANDMASK": np.asarray([[1.0, 1.0], [0.0, 1.0]]),
                 "LU_INDEX": np.asarray([[21.0, 1.0], [16.0, 21.0]]),
                 "HGT_M": np.zeros((2, 2))}
    crop = {name: np.array(value, copy=True) for name, value in footprint.items()}
    crop["HGT_M"] = np.asarray([[120.0, 130.0], [125.0, 128.0]])
    crop["EXTRA_CORRIDOR_FIELD"] = np.ones((2, 2))
    birth = CorridorBirth(grid_id=2, fields=crop, cache_sha256="0" * 64)
    monkeypatch.setattr(
        "woof.static.terrain_autosmooth.prepare_fields",
        lambda *a, **k: pytest.fail("a corridor child is not re-bounded"))
    child, catalog = _hrrr_child_case(
        monkeypatch, footprint, catalog_extra={"statics_corridor_births": {2: birth}})

    prepared = ni._prepare_child_input_on_grid(
        child, object(), catalog, preprocess_backend="cpu", preprocess_workers=2)

    assert set(prepared.static_fields) == set(footprint)
    for name in footprint:
        np.testing.assert_array_equal(prepared.static_fields[name], crop[name])


def test_a_corridor_crop_missing_a_footprint_field_is_refused_by_name(monkeypatch):
    from woof.static.corridor import CorridorBirth, CorridorRefusal
    footprint = {"LANDMASK": np.ones((2, 2)), "LU_INDEX": np.ones((2, 2)),
                 "HGT_M": np.zeros((2, 2))}
    birth = CorridorBirth(grid_id=2, cache_sha256="0" * 64,
                          fields={"LANDMASK": np.ones((2, 2)), "HGT_M": np.zeros((2, 2))})
    child, catalog = _hrrr_child_case(
        monkeypatch, footprint, catalog_extra={"statics_corridor_births": {2: birth}})
    with pytest.raises(CorridorRefusal, match="LU_INDEX"):
        ni._prepare_child_input_on_grid(
            child, object(), catalog, preprocess_backend="cpu", preprocess_workers=2)


@pytest.mark.parametrize("backend", ("cuda", "auto", object()))
def test_pending_child_inputs_rejects_unsafe_parallel_backend(backend):
    domains = tuple(SimpleNamespace(grid_id=grid_id) for grid_id in (2, 3))
    with pytest.raises(ValueError, match="explicit CPU"):
        ni.PendingChildInputs(
            domains, {domain.grid_id: object() for domain in domains},
            object(), None, workers=2, preprocess_backend=backend)


@pytest.mark.parametrize("workers", (0, 33, True, 1.5))
def test_pending_child_inputs_rejects_unbounded_or_invalid_workers(workers):
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        ni.PendingChildInputs((), {}, object(), None, workers=workers)


def _child_binding_case(monkeypatch, child_id, soil_case=None,
                        catalog_mapping=None, mapping_seen=None):
    """Review anchor: REAL -> SINT -> blend all 3 -> adjust -> rederive."""
    events = []
    # These ordering/soil tests replace interpolation; constructing its CUDA
    # backend must not make this CPU contract depend on an installed CuPy.
    backend = SimpleNamespace(receipt=lambda: {"backend": "test-ordering"})
    monkeypatch.setattr(
        "woof.ingest.preprocess_backend.resolve_preprocess_backend",
        lambda *_args, **_kwargs: backend)

    class State:
        def __init__(self):
            self.ht = np.full((4, 5), 10.0, np.float32)
            self.mub2d = np.full((4, 5), 90000.0, np.float32)
            self.phb = np.full((3, 4, 5), 100.0, np.float32)

    state = State()
    # The child's own float64 base before the blend: the fine operand of
    # the float64 terrain blend (nest_init._blend_terrain_triple).
    own_base = SimpleNamespace(
        terrain_z=np.full((4, 5), 12.25), mub=np.full((4, 5), 91000.125),
        phb=np.full((3, 4, 5), 101.0625))
    real = SimpleNamespace(state=state, base=own_base)
    soil = object()
    static = {
        "HGT_M": np.full((4, 5), 10.0),
        "SCT_DOM": np.ones((4, 5)),
        "TMN": np.full((4, 5), 280.0),
        "LU_INDEX": np.vstack((np.array([[21.0, 1.0, 1.0, 1.0, 1.0]]),
                                np.ones((3, 5)))),
    }
    horizontal = SimpleNamespace(fields={})
    cfg = SimpleNamespace(
        nx=5, ny=4, nz=2, dx=1.0, dy=1.0, moist=True,
        terrain_opt=1, spec_bdy_width=5, hypsometric_opt=2,
        # The land-surface selector the soil seam routes on.  A cfg without
        # it no longer reaches an ingest at all, which is the point of the
        # seam: the geometry a run initialises on is a decision, not a
        # default.
        sf_surface_physics=2, num_soil_layers=4)
    if soil_case is not None:
        from test_ruc_soil_wiring import _SOURCES
        source, cfg.sf_surface_physics, cfg.num_soil_layers = soil_case
        horizontal.fields = _SOURCES[source](4, 5)
        static["SCT_DOM"][:] = 6.0
        static["LU_INDEX"][:] = 1.0
    # A parsed [[domain]] always carries a start_time -- it defaults to
    # [experiment].start_time rather than to None -- and the snapshot the
    # child initialises on is now chosen by it, so the stub carries one too.
    child_start = datetime(2026, 7, 20, 6)
    child_dc = SimpleNamespace(
        grid_id=child_id, parent_id=child_id - 1, run=cfg, i_parent_start=3,
        j_parent_start=3, parent_grid_ratio=3, start_time=child_start)
    parent_node = SimpleNamespace(cfg=SimpleNamespace(grid_id=child_id - 1))
    catalog = SimpleNamespace(inventory=("SOILGEO",), valid_times=(object(),),
                              snapshots=(object(),), files=(),
                              water_temperature_policy="wrf_compat",
                              **(catalog_mapping or {}))
    vertical = VerticalConfig(eta_levels=(1.0, 0.5, 0.0), p_top=10000.0,
                              hybrid_opt=2, etac=0.2)

    child_grid = object()
    monkeypatch.setattr(ni, "_child_grid", lambda *_: child_grid)
    monkeypatch.setattr(
        ni, "build_static_for_domain", lambda *_: static)
    def initial_snapshot(_catalog, valid_time):
        # The delayed-child work made this lookup start_time-driven; a stub
        # that swallowed the argument would let a regression to "first valid
        # time" pass unnoticed.
        assert valid_time == child_start
        events.append("snapshot")
        return object()

    monkeypatch.setattr(ni, "_initial_snapshot", initial_snapshot)
    monkeypatch.setattr(
        ni, "geog_selection_from_catalog",
        lambda *_: SimpleNamespace(
            landuse_global_attrs=lambda: {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15, "ISURBAN": 13}))
    def lake_skin(_source, _grid, lake_mask):  # pragma: no cover
        raise AssertionError(
            "regular-grid lane must not run the lake skin override: "
            "metgrid's masked=both SKINTEMP chain already yields "
            "water-source skin at lakes")

    monkeypatch.setattr(ni, "interpolate_lake_skin_temperature", lake_skin)
    def interpolate(*_args, **kwargs):
        assert kwargs["source_orography_catalog"] is catalog
        assert kwargs["backend"] is backend
        if mapping_seen is not None:
            mapping_seen.append(kwargs)
        events.append("era5-own-grid")
        return horizontal

    monkeypatch.setattr(ni, "interpolate_era5_to_lambert", interpolate)
    shared_coord = object()
    monkeypatch.setattr(
        ni, "_shared_vertical_coord", lambda *_: shared_coord)
    def initialize_real(*_args, **kwargs):
        assert kwargs["source_orography"] is None
        events.append("real-unblended")
        return real

    monkeypatch.setattr(ni, "initialize_real", initialize_real)
    monkeypatch.setattr(
        ni, "_set_map_fields", lambda *_: events.append("map"))
    monkeypatch.setattr(
        ni, "update_diagnostics", lambda *_, **kw: events.append("preblend-eos"))
    def preprocess(*args, **kwargs):
        nonlocal soil
        assert kwargs["lake_mask"] is None
        assert kwargs["lake_skin_temperature"] is None
        assert kwargs["soil_layer_contract"] == (
            "soil-contract" if soil_case is None else None)
        events.append("soil-fine")
        if soil_case is not None:
            from woof.ingest.ruc_soil import preprocess_land_surface_soil
            soil = preprocess_land_surface_soil(*args, **kwargs)
        return soil

    monkeypatch.setattr(ni, "preprocess_land_surface_soil", preprocess)
    captures = (np.zeros_like(state.ht), np.zeros_like(state.mub2d),
                np.zeros_like(state.phb))
    monkeypatch.setattr(
        ni, "_capture_parent_blend_fields",
        lambda *_: events.append("sint-ht-mub-phb") or captures)

    # Each field is blended in float64: the parent capture against the
    # child's own float64 base value, identified here by that value.
    fields = {12.25: ("ht", 0), 91000.125: ("mub", 1), 101.0625: ("phb", 2)}

    def blend(parent, target, **_kwargs):
        name, index = fields[float(target.flat[0])]
        assert target.dtype == np.float64
        np.testing.assert_array_equal(parent, captures[index])
        events.append("blend-" + name)

    monkeypatch.setattr(ni, "blend_terrain", blend)
    base = object()
    def adjust_then_rederive(_state, _cfg, _coord, _save_mub, ht_fine, *,
                             blended, preparation=False):
        assert preparation
        assert ht_fine is static["HGT_M"]
        # The re-derivation reads the float64 blend, and the FP32 state
        # already holds its rounding.
        assert [value.dtype for value in blended] == [np.float64] * 3
        for value, stored in zip(blended, (state.ht, state.mub2d,
                                           state.phb)):
            np.testing.assert_array_equal(stored, value.astype(np.float32))
        events.append("adjust-then-rederive-eos-press-adj")
        return base

    monkeypatch.setattr(ni, "_adjust_and_rederive", adjust_then_rederive)
    updated = object()
    monkeypatch.setattr(ni, "_updated_real_result", lambda *_: updated)

    result = ni.initialize_child(
        child_dc, parent_node, catalog, vertical,
        soil_layer_contract="soil-contract" if soil_case is None else None)

    assert events == [
        "snapshot", "era5-own-grid", "real-unblended", "map",
        "preblend-eos", "soil-fine", "sint-ht-mub-phb", "blend-ht",
        "blend-mub", "blend-phb", "adjust-then-rederive-eos-press-adj",
    ]
    assert result.soil is soil
    assert result.static_fields is static
    assert result.real is updated
    assert result.coord is shared_coord
    return result, horizontal.fields, static


@pytest.mark.parametrize("child_id", (2, 3, 4))
def test_initialize_child_binding_order_and_soil_never_readjusted(monkeypatch, child_id):
    _child_binding_case(monkeypatch, child_id)


def test_a_child_maps_with_the_root_conventions_its_catalog_carries(monkeypatch):
    """A GFS tree's child: WPS fix_gfs_rh on RH2 too, lakes on the WPS search."""
    seen = []
    _child_binding_case(
        monkeypatch, 2, catalog_mapping={
            "relative_humidity_convention": "gfs_mixed",
            "lake_whole_source_search": True},
        mapping_seen=seen)
    (kwargs,) = seen
    assert kwargs["relative_humidity_convention"] == "gfs_mixed"
    lakes = kwargs["inland_water_search"]
    # The stub's land use classes one cell as ISLAKE (21).
    assert lakes.dtype == np.bool_ and lakes.shape == (4, 5)
    assert lakes.sum() == 1 and lakes[0, 0]


def test_a_child_catalog_without_them_keeps_the_regular_grid_defaults(monkeypatch):
    seen = []
    _child_binding_case(monkeypatch, 2, mapping_seen=seen)
    (kwargs,) = seen
    assert kwargs["relative_humidity_convention"] == "era5_mixed"
    assert kwargs["inland_water_search"] is None
    import dataclasses
    defaults = {field.name: field.default
                for field in dataclasses.fields(ni.NestedInputCatalog)}
    assert defaults["relative_humidity_convention"] == "era5_mixed"
    assert defaults["lake_whole_source_search"] is False


@pytest.mark.parametrize("source", ("era5", "hrrr"))
@pytest.mark.parametrize("scheme,layers", ((2, 4), (3, 6), (3, 9)))
def test_child_soil_uses_configured_geometry_and_matches_direct_provider(
        monkeypatch, source, scheme, layers):
    """Six-layer children used to get nine layers and fail forecast loading."""
    from woof.ingest.ruc_soil import preprocess_ruc_soil
    from woof.ingest.soil import preprocess_noah_soil

    # The child seam first maps a foreign soil model's moisture onto the run's
    # land model (woof/ingest/land_state.py: ERA5's HTESSEL into RUC or Noah,
    # HRRR's RUC into Noah), with the child's own soil classes; after that it
    # is the direct provider itself. Record the seam's converted fields.
    from woof.ingest import land_state
    converted_seen = []
    convert = land_state.convert_source_soil_moisture

    def recording(*args, **kwargs):
        converted, receipt = convert(*args, **kwargs)
        converted_seen.append(converted)
        return converted, receipt
    monkeypatch.setattr(land_state, "convert_source_soil_moisture", recording)
    result, fields, static = _child_binding_case(
        monkeypatch, 2, soil_case=(source, scheme, layers))
    provider = preprocess_ruc_soil if scheme == 3 else preprocess_noah_soil
    assert len(converted_seen) == 1
    converted = converted_seen[0]
    expected = provider(
        converted, soil_type=static["SCT_DOM"], deep_soil_temperature=static["TMN"],
        water_temperature_policy="wrf_compat",
        **({"num_soil_layers": layers} if scheme == 3 else {}))
    for name in ("soil_temperature", "soil_moisture", "liquid_moisture"):
        actual = getattr(result.soil, name)
        assert actual.shape == (layers, 4, 5), (name, actual.shape)
        np.testing.assert_array_equal(actual, getattr(expected, name))


def test_child_soil_rejects_unrepresentable_configured_geometry(monkeypatch):
    with pytest.raises(ValueError, match="num_soil_layers"):
        _child_binding_case(monkeypatch, 2, soil_case=("era5", 3, 4))


def test_as_like_moves_cuda_style_parent_operand_to_cpu_child():
    class DeviceOperand:
        def __init__(self, value):
            self.value = value
            self.readbacks = 0

        def get(self):
            self.readbacks += 1
            return self.value

    source = DeviceOperand(
        np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::-1])
    template = np.empty((3, 4), dtype=np.float32)
    result = ni._as_like(source, template)
    assert source.readbacks == 1
    assert isinstance(result, np.ndarray)
    assert result.dtype == template.dtype
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, source.value)


def _write_source_orography(path: Path, domain_id: int) -> np.ndarray:
    import netCDF4

    expected = np.full((domain_id + 1, domain_id + 2), domain_id, np.float32)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", expected.shape[0])
        dataset.createDimension("west_east", expected.shape[1])
        dataset.createVariable(
            "SOILHGT", "f4", ("Time", "south_north", "west_east"))[0] = expected
    return expected.astype(np.float64)


def test_domain_tagged_catalog_resolves_each_child_orography_artifact(tmp_path):
    expected = {}
    files = []
    for domain_id in (2, 3, 4):
        path = tmp_path / f"met_em.d{domain_id:02d}.nc"
        expected[domain_id] = _write_source_orography(path, domain_id)
        files.append(SimpleNamespace(
            role="source_orography", path=path,
            provenance=f"variable=SOILHGT;domain=d{domain_id:02d}"))
    catalog = SimpleNamespace(files=tuple(files))

    for domain_id in (2, 3, 4):
        np.testing.assert_array_equal(
            ni._declared_source_orography(catalog, domain_id),
            expected[domain_id])


def test_d01_only_declaration_fails_at_d02_initialization_with_domain_named(
        tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        nx=5, ny=4, nz=2, moist=True, terrain_opt=1,
        hypsometric_opt=2)
    child_dc = SimpleNamespace(grid_id=2, parent_id=1, run=cfg,
                               start_time=datetime(2026, 7, 20, 6))
    parent_node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1))
    catalog = SimpleNamespace(
        inventory=(), files=(), valid_times=(object(),), snapshots=(object(),))
    vertical = VerticalConfig(eta_levels=(1.0, 0.5, 0.0), p_top=10000.0,
                              hybrid_opt=2, etac=0.2)
    declaration = SourceOrography(tmp_path / "d01-only.nc", "SOILHGT")
    monkeypatch.setattr(ni, "_child_grid", lambda *_: object())
    monkeypatch.setattr(ni, "build_static_for_domain", lambda *_: {})
    monkeypatch.setattr(ni, "_initial_snapshot", lambda *_: object())
    # This refusal precedes interpolation and needs no device/backend setup.
    monkeypatch.setattr(
        "woof.ingest.preprocess_backend.resolve_preprocess_backend",
        lambda *_args, **_kwargs: object())

    with pytest.raises(ValueError, match=r"d02"):
        ni.initialize_child(
            child_dc, parent_node, catalog, vertical, declaration)


@pytest.mark.parametrize("prepared", [False, True])
def test_press_adj_mu_matches_wrf_expression_after_eos(monkeypatch, prepared):
    """start_em.F:878-892: diagnose once, correct MU_2, seed MU_1."""
    events = []

    class State:
        def __init__(self):
            self.mub2d = np.full((1, 2), 90000.0, np.float32)
            self.c3h = np.array([0.75, 0.25], np.float32)
            self.c4h = np.zeros(2, np.float32)
            self.p_top = np.float32(10000.0)
            self.pb = np.ones((2, 1, 2), np.float32)
            self.p = self.pb.copy()
            self.qv = np.zeros_like(self.pb)
            self.thb = np.zeros_like(self.pb)
            self.thp = np.zeros_like(self.pb)
            self.ht = np.array([[120.0, 80.0]], np.float32)
            self.mup = np.array([[10.0, -4.0]], np.float32)
            self.mup0 = np.full((1, 2), -999.0, np.float32)
            self.al = np.zeros_like(self.pb)
            self.alt = np.zeros_like(self.pb)
            self.alb = np.zeros_like(self.pb)

        def total_theta(self):
            return np.full_like(self.thp, 300.0)

        def load_base(self, _coord, base):
            events.append("load-base")
            self.thb[...] = base.thb

    state = State()
    base = SimpleNamespace(thb=np.zeros_like(state.thb))
    monkeypatch.setattr(
        ni, "adjust_tempqv",
        lambda *_args, **_kwargs: events.append("adjust-tempqv"))
    monkeypatch.setattr(
        ni, "_base_from_blended",
        lambda *_args, **_kwargs: events.append("rederive-base") or base)

    def diagnose(_state, _hypsometric_opt, *, preparation=False):
        assert preparation == prepared
        events.append("eos")
        _state.al[0, 0] = np.array([0.02, 0.03], np.float32)
        _state.alt[0, 0] = np.array([0.80, 0.81], np.float32)
        _state.alb[0, 0] = np.array([0.78, 0.78], np.float32)

    monkeypatch.setattr(ni, "update_diagnostics", diagnose)
    ht_fine = np.array([[100.0, 120.0]], np.float32)

    ni._adjust_and_rederive(
        state, SimpleNamespace(hypsometric_opt=2), object(),
        state.mub2d.copy(), ht_fine, blended=object(),
        **({"preparation": True} if prepared else {}))

    expected = np.array([
        10.0 + 0.02 / (0.80 * 0.78) * 9.81 * (120.0 - 100.0),
        -4.0 + 0.03 / (0.81 * 0.78) * 9.81 * (80.0 - 120.0),
    ], np.float32)
    np.testing.assert_allclose(state.mup[0], expected, rtol=2e-6, atol=0.0)
    np.testing.assert_array_equal(state.mup0, state.mup)
    assert events == ["adjust-tempqv", "rederive-base", "load-base", "eos"]


def _flat_parent_state(nz=3, ny=12, nx=12):
    znw = np.linspace(1.0, 0.0, nz + 1)
    znu = 0.5 * (znw[:-1] + znw[1:])
    dnw = np.diff(znw)
    dn = np.zeros(nz)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    rdn = np.zeros(nz)
    rdn[1:] = 1.0 / dn[1:]
    fnp = np.zeros(nz)
    fnm = np.zeros(nz)
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    p_top = 10000.0
    mub = 90000.0
    pb = znu * mub + p_top
    height = np.array([500.0, 3500.0, 8500.0])
    thb = wk82_theta(height)
    alb = 287.0 * thb * (pb / 100000.0) ** (287.0 / 1004.0) / pb
    phb = np.array([0.0, 30000.0, 75000.0, 140000.0])
    qv = wk82_sounding(height)[1][:, None, None] * np.ones((nz, ny, nx))
    x = np.arange(nx, dtype=np.float32)[None, None, :]
    y = np.arange(ny, dtype=np.float32)[None, :, None]
    x_u = np.arange(nx + 1, dtype=np.float32)[None, None, :]
    y_v = np.arange(ny + 1, dtype=np.float32)[None, :, None]
    thp = np.broadcast_to(0.01 * x + 0.02 * y, (nz, ny, nx)).copy()
    return SimpleNamespace(
        znw=znw, znu=znu, dnw=dnw, rdnw=1.0 / dnw,
        dn=dn, rdn=rdn, fnp=fnp, fnm=fnm,
        c1f=np.ones(nz + 1), c2f=np.zeros(nz + 1), c3f=znw,
        c4f=np.zeros(nz + 1), c1h=np.ones(nz), c2h=np.zeros(nz),
        c3h=znu, c4h=np.zeros(nz),
        mub=mub, mub2d=np.full((ny, nx), mub, np.float32),
        p_top=p_top, pb=pb, alb=alb, thb=thb, phb=phb,
        ht=np.zeros((ny, nx), np.float32),
        u=np.broadcast_to(x_u, (nz, ny, nx + 1)).copy(),
        v=np.broadcast_to(y_v, (nz, ny + 1, nx)).copy(),
        w=np.zeros((nz + 1, ny, nx), np.float32),
        thp=thp.astype(np.float32),
        php=np.zeros((nz + 1, ny, nx), np.float32),
        mup=np.zeros((ny, nx), np.float32),
        qv=qv.astype(np.float32), qc=np.zeros_like(qv, np.float32),
        qr=np.zeros_like(qv, np.float32), h_diabatic=np.zeros_like(qv,
                                                                   np.float32))


class _CpuState:
    """Minimal NumPy DomainState twin for the WK82 parent-only fixture."""

    def __init__(self, cfg):
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        self.u = np.zeros((nz, ny, nx + 1), np.float32)
        self.v = np.zeros((nz, ny + 1, nx), np.float32)
        self.w = np.zeros((nz + 1, ny, nx), np.float32)
        self.thp = np.zeros((nz, ny, nx), np.float32)
        self.php = np.zeros((nz + 1, ny, nx), np.float32)
        self.mup = np.zeros((ny, nx), np.float32)
        self.qv = np.zeros((nz, ny, nx), np.float32)
        self.qc = np.zeros_like(self.qv)
        self.qr = np.zeros_like(self.qv)
        self.h_diabatic = np.zeros_like(self.qv)
        for name, source in (("u0", self.u), ("v0", self.v),
                             ("w0", self.w), ("thp0", self.thp),
                             ("php0", self.php), ("mup0", self.mup),
                             ("qv0", self.qv), ("qc0", self.qc),
                             ("qr0", self.qr)):
            setattr(self, name, np.zeros_like(source))

    def load_base(self, coord, base: BaseState):
        for name in ("znw", "znu", "dnw", "rdnw", "dn", "rdn", "fnp",
                     "fnm", "c1f", "c2f", "c3f", "c4f", "c1h", "c2h",
                     "c3h", "c4h"):
            setattr(self, name, np.array(getattr(coord, name), copy=True))
        for name in ("pb", "alb", "thb", "phb"):
            setattr(self, name, np.array(getattr(base, name), copy=True))
        self.mub = float(base.mub)
        self.mub2d = np.full(self.mup.shape, self.mub, np.float32)
        self.ht = np.zeros(self.mup.shape, np.float32)
        self.p_top = float(base.p_top)

    def set_map_coriolis(self, *_args, **_kwargs):
        return None


def test_parent_only_wk82_child_fixture_is_cpu_testable(monkeypatch):
    """input_from_file=F keeps the full-parent SINT fill for WK82 nests."""
    parent_state = _flat_parent_state()
    parent_run = SimpleNamespace(nx=12, ny=12, nz=3)
    parent_grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=13, e_sn=13)
    parent_node = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, run=parent_run),
        state=parent_state, grid=parent_grid)
    child_run = SimpleNamespace(
        nx=4, ny=4, nz=3, dx=1000.0, dy=1000.0,
        terrain_opt=0, hypsometric_opt=1)
    child_dc = SimpleNamespace(
        grid_id=2, parent_id=1, i_parent_start=4, j_parent_start=4,
        parent_grid_ratio=1, run=child_run)

    monkeypatch.setattr(ni, "DomainState", _CpuState)
    monkeypatch.setattr(
        ni, "sint",
        lambda field, reg, *, preparation: np_sint(field, reg, dtype=np.float64).astype(
            np.float32))
    diagnosed = []
    monkeypatch.setattr(
        ni, "update_diagnostics", lambda *_: diagnosed.append(True))

    result = ni.parent_only_init(child_dc, parent_node)
    reg = ni._mass_registration(child_dc, parent_node)
    expected_qv = np_sint(parent_state.qv, reg, dtype=np.float64).astype(
        np.float32)
    np.testing.assert_array_equal(result.state.qv, expected_qv)
    np.testing.assert_array_equal(result.state.qv0, expected_qv)
    assert np.isfinite(result.state.thp).all()
    assert float(result.state.qv.max()) > 0.0  # the analytic WK82 moisture
    assert diagnosed == [True]
    assert result.real is result.soil is result.static_fields is None


def test_device_child_from_host_parent_retains_full_child_geometry(monkeypatch):
    import sys
    parent_state = _flat_parent_state()
    parent_run = SimpleNamespace(nx=12, ny=12, nz=3)
    parent_grid = LambertGrid(ref_lat=35., ref_lon=-97., truelat1=30., truelat2=60.,
        stand_lon=-97., dx=1000., dy=1000., e_we=13, e_sn=13)
    parent_node = SimpleNamespace(cfg=SimpleNamespace(grid_id=1, run=parent_run),
                                  state=parent_state, grid=parent_grid)
    child_run = SimpleNamespace(nx=4, ny=4, nz=3, dx=1000., dy=1000.,
                                terrain_opt=0, hypsometric_opt=1)
    child_dc = SimpleNamespace(grid_id=2, parent_id=1, i_parent_start=4,
                               j_parent_start=4, parent_grid_ratio=1, run=child_run)
    built, registrations = [], []
    class DeviceState(_CpuState):
        def __init__(self, cfg):
            built.append(cfg)
            super().__init__(cfg)
            for name, value in list(vars(self).items()):
                if isinstance(value, np.ndarray):
                    setattr(self, name, _DeviceSintPayload(value))
    monkeypatch.setattr(ni, 'DomainState', DeviceState)
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(asarray=_DeviceSintPayload))
    def interpolate(payload, reg, *, preparation):
        assert preparation is True
        assert isinstance(payload, _DeviceSintPayload)
        registrations.append(reg)
        return _DeviceSintPayload(np_sint(payload.value, reg, dtype=np.float64).astype(np.float32))
    monkeypatch.setattr(ni, 'sint', interpolate)
    monkeypatch.setattr(ni, 'update_diagnostics', lambda *args: None)
    result = ni.parent_only_init(child_dc, parent_node, clamp_undershoot=False)
    assert built == [child_run] and built[0] is child_run
    assert result.state.u.shape == (3, 4, 5) and result.state.v.shape == (3, 5, 4)
    assert (result.grid.e_we, result.grid.e_sn) == (5, 5)
    assert registrations and all(reg.nyp < 12 for reg in registrations)
    mass_reg = ni._mass_registration(child_dc, parent_node)
    expected = np_sint(parent_state.qv, mass_reg, dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(result.state.qv.value, expected)
    np.testing.assert_array_equal(result.state.qv0.value, expected)


def test_mixed_parent_only_init_reuses_flat_force_slot_and_resets_heating(
        monkeypatch):
    """MP8->MP18 cold-start translation cannot poison later FORCE shape."""
    parent_state = _flat_parent_state()
    parent_state.h_diabatic.fill(np.float32(77.0))
    for index, name in enumerate(("qi", "qs", "qg", "nr", "ni"), 1):
        setattr(parent_state, name, np.full(
            parent_state.qv.shape, np.float32(index), dtype=np.float32))
    parent_run = SimpleNamespace(
        nx=12, ny=12, nz=3, mp_physics=8, moist=True, moist_cq=True)
    parent_grid = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=1000.0, dy=1000.0, e_we=13, e_sn=13)
    parent_node = SimpleNamespace(
        cfg=SimpleNamespace(grid_id=1, run=parent_run),
        state=parent_state, grid=parent_grid)
    child_run = SimpleNamespace(
        nx=4, ny=4, nz=3, dx=1000.0, dy=1000.0,
        terrain_opt=0, hypsometric_opt=1, mp_physics=18,
        moist=True, moist_cq=True, spec_bdy_width=5,
        spec_zone=1, relax_zone=4,
        nest_microphysics_transition="mp8-to-mp18-mass-diagnosed-v1")
    child_dc = SimpleNamespace(
        grid_id=2, parent_id=1, i_parent_start=4, j_parent_start=4,
        parent_grid_ratio=1, run=child_run)

    class MixedCpuState(_CpuState):
        def __init__(self, cfg):
            super().__init__(cfg)
            for name in ("qi", "qs", "qg", "qh", "qndrop", "qnr",
                         "qni", "qns", "qng", "qnh", "qnn", "qvolg",
                         "qvolh"):
                value = np.zeros_like(self.qv)
                setattr(self, name, value)
                setattr(self, name + "0", np.zeros_like(value))
            self._scratch = {}

        def scratch(self, shape, slot, dtype=None):
            shape = tuple(shape)
            if slot not in self._scratch:
                self._scratch[slot] = np.zeros(shape, np.float32)
            value = self._scratch[slot]
            assert value.shape == shape
            return value

    monkeypatch.setattr(ni, "DomainState", MixedCpuState)

    def cpu_sint(field, reg, *, out=None, preparation=True):
        assert preparation is True
        result = np_sint(field, reg, dtype=np.float64).astype(np.float32)
        if out is not None:
            out[...] = result
            return out
        return result

    field_value = {
        name: np.float32(index + 1)
        for index, name in enumerate((
            "qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop",
            "qnr", "qni", "qns", "qng", "qnh", "qnn", "qvolg",
            "qvolh"))
    }

    def translate(_contract, _state, name, *, out, coupled):
        assert coupled is False
        out.fill(field_value[name])
        return out

    monkeypatch.setattr(ni, "sint", cpu_sint)
    monkeypatch.setattr(
        ni, "launch_microphysics_edge_parent_field", translate)
    monkeypatch.setattr(ni, "update_diagnostics", lambda *_: None)

    result = ni.parent_only_init(child_dc, parent_node)
    state = result.state
    assert state._microphysics_transition_init_count == 1
    expected_capacity = max(
        3 * 12 * 12, 3 * 12 * 13, 3 * 13 * 12, 4 * 12 * 12)
    assert state._scratch["nest_parent_field"].shape == (expected_capacity,)
    np.testing.assert_array_equal(state.h_diabatic, 0.0)
    for name in field_value:
        target = getattr(state, name)
        seed = getattr(state, name + "0", None)
        assert np.all(target == field_value[name])
        if seed is not None:
            np.testing.assert_array_equal(seed, target)


def test_blend_zone_mask_matches_wrf_ten_row_frame():
    mask = ni.blend_zone_mask((24, 26), spec_bdy_width=5, blend_width=5)
    assert mask[0].all() and mask[:, 0].all()
    assert mask[9, 12] and mask[12, 9]
    assert not mask[10, 12] and not mask[12, 10]


def _write_reference(path: Path, hgt, mub=None):
    import netCDF4

    path.parent.mkdir(parents=True, exist_ok=True)
    ny, nx = hgt.shape
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", ny)
        dataset.createDimension("west_east", nx)
        hgt_var = dataset.createVariable(
            "HGT" if mub is not None else "HGT_M", "f4",
            ("Time", "south_north", "west_east"))
        hgt_var[0] = hgt
        if mub is not None:
            dataset.createVariable(
                "MUB", "f4",
                ("Time", "south_north", "west_east"))[0] = mub


def test_n1_static_comparison_entry_point(tmp_path):
    bundle = tmp_path / "bundle"
    produced = {}
    hgt = np.arange(24 * 24, dtype=np.float32).reshape(24, 24)
    mub = np.full((24, 24), 90000.0, np.float32)
    for domain_id in (2, 3, 4):
        label = f"d{domain_id:02d}"
        _write_reference(
            bundle / "wrfout_reference" /
            f"wrfout_{label}_1974-04-03_13_15_00", hgt, mub)
        _write_reference(bundle / "geo_em" / f"geo_em.{label}.nc", hgt)
        path = tmp_path / f"{label}.npz"
        np.savez(path, HGT=hgt, MUB=mub, HGT_UNBLENDED=hgt)
        produced[domain_id] = path

    result = ni.compare_n1_static(produced, bundle)
    assert result.passed
    assert set(result.metrics) == {
        f"d{domain_id:02d}_{field}_blend_max_abs_diff_{unit}"
        for domain_id in (2, 3, 4)
        for field, unit in (("hgt", "m"), ("mub", "pa"))
    }
    assert all(value == 0.0 for value in result.metrics.values())
    assert all(value == 0.0
               for value in result.fine_hgt_max_abs_diff_m.values())


def _n1_bundle(root: Path, frames: dict[int, str]) -> tuple[Path, dict]:
    """Bundle + produced NPZs whose reference frames are named by caller."""
    bundle = root / "bundle"
    produced = {}
    hgt = np.arange(24 * 24, dtype=np.float32).reshape(24, 24)
    mub = np.full((24, 24), 90000.0, np.float32)
    for domain_id, frame in frames.items():
        label = f"d{domain_id:02d}"
        _write_reference(bundle / "wrfout_reference" / frame, hgt, mub)
        _write_reference(bundle / "geo_em" / f"geo_em.{label}.nc", hgt)
        path = root / f"{label}.npz"
        np.savez(path, HGT=hgt, MUB=mub, HGT_UNBLENDED=hgt)
        produced[domain_id] = path
    return bundle, produced


def test_n1_reference_frame_is_case_data_not_a_literal(tmp_path):
    """The comparator carries no campaign valid time.

    A bundle whose frames are stamped with a DIFFERENT valid time than the
    one this module used to hard-code compares fine, both by declaration
    and by discovery.  That is the negative control: with the literal in
    place, every assertion below raised FileNotFoundError."""
    frames = {domain_id: f"wrfout_d{domain_id:02d}_2031-07-19_04_30_00"
              for domain_id in (2, 3, 4)}
    bundle, produced = _n1_bundle(tmp_path / "declared", frames)
    declared = ni.compare_n1_static(
        produced, bundle,
        reference_frames={f"d{k:02d}": v for k, v in frames.items()})
    assert declared.passed

    # Same frames, keyed by domain id rather than dNN label.
    by_id = ni.compare_n1_static(produced, bundle, reference_frames=frames)
    assert by_id.metrics == declared.metrics

    # No declaration: the single frame per domain is discovered.
    discovered = ni.compare_n1_static(produced, bundle)
    assert discovered.metrics == declared.metrics


def test_n1_reference_frame_discovery_refuses_to_guess(tmp_path):
    """Ambiguity and absence are hard errors naming the candidates, never
    a silent pick of whichever frame sorts first."""
    frames = {domain_id: f"wrfout_d{domain_id:02d}_2031-07-19_04_30_00"
              for domain_id in (2, 3, 4)}
    bundle, produced = _n1_bundle(tmp_path / "ambiguous", frames)
    hgt = np.arange(24 * 24, dtype=np.float32).reshape(24, 24)
    _write_reference(bundle / "wrfout_reference" /
                     "wrfout_d02_2031-07-19_05_30_00", hgt,
                     np.full((24, 24), 90000.0, np.float32))
    with pytest.raises(ValueError, match="exactly one reference frame"):
        ni.compare_n1_static(produced, bundle)
    # Declaring the frame resolves the ambiguity without widening anything.
    assert ni.compare_n1_static(
        produced, bundle, reference_frames={"d02": frames[2]}).passed

    empty = tmp_path / "empty" / "bundle"
    (empty / "wrfout_reference").mkdir(parents=True)
    for domain_id in (2, 3, 4):
        _write_reference(empty / "geo_em" / f"geo_em.d{domain_id:02d}.nc",
                         hgt)
    with pytest.raises(ValueError, match="found 0"):
        ni.compare_n1_static(produced, empty)


def test_n1_cli_accepts_a_declared_reference_frame(tmp_path, capsys):
    import json as _json

    frames = {domain_id: f"wrfout_d{domain_id:02d}_2031-07-19_04_30_00"
              for domain_id in (2, 3, 4)}
    bundle, produced = _n1_bundle(tmp_path / "cli", frames)
    argv = ["--bundle", str(bundle)]
    for domain_id in (2, 3, 4):
        argv += [f"--d{domain_id:02d}", str(produced[domain_id])]
        argv += ["--reference-frame", f"d{domain_id:02d}={frames[domain_id]}"]
    assert ni.main(argv) == 0
    assert _json.loads(capsys.readouterr().out)["passed"] is True

    with pytest.raises(SystemExit):
        ni.main([*argv, "--reference-frame", "malformed"])


# ---------------------------------------------------------------------
# a child result carries every receipt its own initialize_real produced
# ---------------------------------------------------------------------


def _real_result_for_receipt_carry():
    """A ``RealInitResult`` with every receipt field filled distinctly."""
    from dataclasses import fields as dataclass_fields

    from woof.ingest.real import RealInitResult

    array = np.zeros((2, 2), dtype=np.float64)
    state = SimpleNamespace(
        p=np.zeros((2, 2, 2), dtype=np.float64),
        phb=np.zeros((3, 2, 2), dtype=np.float64),
        php=np.zeros((3, 2, 2), dtype=np.float64),
        alt=np.zeros((2, 2, 2), dtype=np.float64),
        c3h=np.array([0.5, 1.0]), c4h=np.array([10.0, 0.0]),
        p_top=5000.0, total_mu=lambda: np.full((2, 2), 90000.0),
    )
    receipts = {
        "hydrometeor_initialization": {"source": "native"},
        "aerosol_initialization": {"aerosol_source": "wif-climatology"},
        "surface_moisture_floor": {"cells": 3},
        "initial_perturbation": {"bubbles": 1},
    }
    # Every mapping-shaped receipt on the dataclass, not only the four
    # that exist today: a floor or receipt added later must show up here
    # filled, so the carry assertion below is not testing an absence.
    for field in dataclass_fields(RealInitResult):
        if field.name not in receipts and field.name.endswith(
                ("_initialization", "_floor", "_perturbation")):
            receipts[field.name] = {"filled": field.name}
    return RealInitResult(
        state=state, coord=object(), base=object(),
        surface_pressure=array, surface_qv=array, dry_mass=array,
        dry_pressure=array, total_pressure=array,
        total_geopotential=array, total_specific_volume=array,
        integrated_moisture_pressure=array, hypsometric_opt=2,
        **receipts), receipts


def test_a_child_result_carries_every_receipt_field_its_parent_carried():
    """THE AEROSOL RECEIPT THAT VANISHED AT THE NEST BOUNDARY.

    ``_updated_real_result`` spelled out the constructor by hand and
    spelled one argument short, so a child's ``aerosol_initialization``
    came back as the dataclass default -- an EMPTY receipt, which every
    reader downstream reads as "no aerosol dataset was used".  The set of
    carried fields is computed from ``dataclasses.fields`` here for the
    same reason it is computed there: the next receipt added to
    ``RealInitResult`` is covered without anyone editing this cell.
    """

    from dataclasses import fields as dataclass_fields

    original, receipts = _real_result_for_receipt_carry()
    updated = ni._updated_real_result(original, base="rederived-base")

    assert updated.base == "rederived-base"
    carried = [field.name for field in dataclass_fields(original)
               if field.name not in ni._REDERIVED_CHILD_FIELDS]
    for name in carried:
        assert getattr(updated, name) is getattr(original, name), name
    # And the receipts specifically, by value, so an empty default can
    # never pass as a carried receipt.
    for name, value in receipts.items():
        assert getattr(updated, name) == value, name
    assert updated.aerosol_initialization == {
        "aerosol_source": "wif-climatology"}


def test_a_child_result_that_drops_a_receipt_is_refused_by_name():
    """The guard, exercised: the next hand-spelled omission is a refusal
    naming the field, not a silently empty receipt in a proof."""

    from dataclasses import replace

    original, _ = _real_result_for_receipt_carry()
    lost = replace(original, aerosol_initialization={})
    with pytest.raises(ValueError, match="aerosol_initialization"):
        ni._refuse_dropped_child_receipts(original, lost)
