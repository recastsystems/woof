"""A65: every CUDA preparation is priced and decided before it allocates.

The defect: a CUDA preparation of 1792x1024x55 on a 24 GB card decoded its
inputs, built its statics and then stopped about two minutes in with a raw
CuPy out-of-memory inside ``DomainState.__init__``.  The engine had a device
price for the phase and no door asked it.  These tests hold:

* the price never undercounts a measured preparation peak;
* on every route CUDA keeps its arithmetic while memory is occupied,
  refuses impossible capacity before allocation, and a card with room
  still prepares on the card;
* every door decides before its first device allocation, and defaults to
  ``auto``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.config import RunConfig
from woof.core import device_probe
from woof.ingest import preparation_price as pp
from woof.ingest import preprocess_backend as backend
from woof.ingest.bounded_cuda import BoundedCudaPreprocessBackend

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30


@pytest.fixture(autouse=True)
def _stand_in_cards_are_not_the_local_device(monkeypatch):
    """Every card in this file is a stand-in with a stated free memory.

    The CPU battery legs set GPUWM_NO_LOCAL_GPU, and since 083ac7cd0
    (lane/sw-excess) the backend selection refuses CUDA under it before any
    probe.  That refusal is right for the local device and has its own
    tests (tests/test_preprocess_no_local_gpu.py); here it would hide the
    pricing decisions this file exists to hold, none of which opens a
    device.  The process-wide CUDA visibility ban stays in force.
    """
    monkeypatch.setattr("woof.local_gpu.no_local_gpu", lambda: False)


def _cfg(nx, ny, nz, **overrides):
    """The measured runs' physics (mp=8, terrain, km_opt=4, YSU, Noah)."""
    values = dict(dx=3000.0, dy=3000.0, ztop=20000.0, dt=15.0,
                  run_seconds=21600.0, terrain_opt=1, moist=True,
                  mp_physics=8, km_opt=4, bl_pbl_physics=1,
                  nwp_diagnostics=1, specified=True, spec_bdy_width=5,
                  sf_sfclay_physics=91, sf_surface_physics=2)
    values.update(overrides)
    return RunConfig(nx=nx, ny=ny, nz=nz, **values)


def _measured_domains(row):
    shapes = row["domains"]
    domains = [_cfg(*shapes[0])]
    for shape in shapes[1:]:
        domains.append(_cfg(*shape, specified=False, nested=True))
    return domains


# ---------------------------------------------------------------- the price

def test_state_term_is_the_exact_constructor_inventory():
    """84 arrays, 21,171,152,780 bytes at the 3 km CONUS shape (the byte
    total both surveys reached: a traced constructor and the inventory; 2.9.11
    added the stored MF_VX_INV plane msfv_inv, 4 x 1025 x 1792 = 7,347,200 B,
    paper/conus-suite-exact; 83 and 21,163,805,580 before)."""
    reference = RunConfig(nx=1792, ny=1024, nz=55, dx=3000.0, dy=3000.0,
                          ztop=20000.0, dt=15.0, run_seconds=3600.0,
                          terrain_opt=1, moist=True, mp_physics=8,
                          specified=True)
    from woof.core.preflight import state_array_shapes
    assert len(state_array_shapes(reference)) == 84
    assert pp.state_bytes(reference) == 21_171_152_780


@pytest.mark.parametrize("row", pp.MEASURED_PREPARATION_PEAKS,
                         ids=[row["case"] for row in pp.MEASURED_PREPARATION_PEAKS])
def test_the_price_never_undercounts_a_measured_peak(row):
    """Linux context (the measured box), the measured source inventory."""
    price = pp.price_preparation(
        "mapped", _measured_domains(row), pp.MEASURED_SOURCE_INVENTORY,
        boundary_intervals=pp.MEASURED_BOUNDARY_INTERVALS, platform="linux")
    card = row["card_gb"] * 1e9
    assert price.need_bytes >= card, (row["case"], price.need_bytes / 1e9)
    # Calibrated, not padded: within a quarter of the measured peak.
    assert price.need_bytes <= 1.25 * card, (row["case"], price.need_bytes / 1e9)
    # The pool term alone (price minus context) covers what the pool held.
    pooled = price.need_bytes - price.terms["cuda_context"]
    assert pooled >= row["reserved_gb"] * 1e9


def _route_price(row):
    """The price of a measured route row, from the shape it was run at."""
    if row["route"] == "downscale-child":
        nx, ny, nz = row["child"]
        child = _cfg(nx, ny, nz, dx=row["child_dx"], dy=row["child_dx"],
                     specified=False, nested=True)
        pnx, pny, pnz = row["parent"]
        return pp.price_downscale_interpolation(
            parent_nx=pnx, parent_ny=pny, parent_nz=pnz,
            parent_fields=row["parent_fields"], child_cfg=child)
    physics = {key: row[key] for key in ("mp_physics",) if key in row}
    dx = row.get("dx", 3000.0)
    domains = [_cfg(*shape, dx=dx, dy=dx, **physics)
               for shape in row["domains"]]
    if "inventory" in row:
        levels, fields, planes = row["inventory"]
        inventory = pp.SourceInventory(levels=levels, level_fields=fields,
                                       surface_planes=planes)
    else:
        inventory = pp.NOMINAL_SOURCE_INVENTORIES[row["route"]]
    return pp.price_preparation(
        row["route"], domains, inventory,
        boundary_intervals=row.get("boundary_intervals", 0),
        boundary_workers=row.get("boundary_workers", 0))


@pytest.mark.parametrize("row", pp.MEASURED_ROUTE_PEAKS,
                         ids=[row["route"] for row in pp.MEASURED_ROUTE_PEAKS])
def test_each_measured_route_is_priced_at_or_above_its_card_peak(row):
    """A101: the routes the mapped calibration was carried to unmeasured,
    each run once on a card.  The price sits at or above the preparation's
    own card peak, and without the CUDA context at or above what the pool
    reserved; it never falls below what the door decided on in the run.
    The experiment route reserved 1.4% past its pooled price at 1.20
    headroom, which its row's 1.25 covers."""
    assert row["route"] in pp.PREPARATION_ROUTES
    price = _route_price(row)
    assert price.need_bytes == row["priced_bytes"], row["case"]
    assert price.need_bytes >= row["predicted_bytes"]
    assert price.need_bytes >= row["card_gb"] * 1e9, (
        row["case"], price.need_bytes / 1e9)
    if row["reserved_gb"] is not None:
        pooled = price.need_bytes - price.terms["cuda_context"]
        assert pooled >= row["reserved_gb"] * 1e9, (row["case"], pooled / 1e9)
    route = pp.PREPARATION_ROUTES[row["route"]]
    if route.pool_headroom is not None:
        assert f"pool headroom x{route.pool_headroom:g}" in price.basis


def test_the_measured_live_setup_sits_inside_the_residual():
    """The fitted terms: live non-state memory at each measured peak is
    1.005 to 1.006 of the itemized analysis and vertical setup, under the
    1.10 the price carries; the pool reserved 1.10 to 1.18 of live, under
    its 1.20."""
    for row in pp.MEASURED_PREPARATION_PEAKS[:2]:
        (cfg,) = _measured_domains(row)
        inventory = pp.MEASURED_SOURCE_INVENTORY
        itemized = (pp.analysis_bytes(cfg, inventory)
                    + pp.vertical_setup_bytes(cfg, inventory))
        live_setup = row["live_gb"] * 1e9 - pp.state_bytes(cfg)
        assert 1.0 <= live_setup / itemized <= pp.SETUP_RESIDUAL
        assert row["reserved_gb"] / row["live_gb"] <= pp.PREPARATION_POOL_HEADROOM


def test_the_24_gb_failure_is_priced_over_a_24_gb_card_and_6_km_under_it():
    gfs = pp.SourceInventory(levels=23, level_fields=5, surface_planes=19)
    big = pp.price_preparation("gfs", [_cfg(1792, 1024, 55)], gfs,
                               platform="linux")
    small = pp.price_preparation("gfs", [_cfg(896, 512, 59)], gfs,
                                 platform="linux")
    # The failure had 22.92 GiB reserved with 1.95 GiB of the state still
    # to allocate: the price must sit above 24.87 GiB, and above the card.
    assert big.need_bytes > 24.87 * GIB
    assert big.need_bytes > 24 * 10 ** 9
    assert small.need_bytes < 20 * GIB


def test_the_ingest_estimate_retires_the_fractional_transient():
    """``estimate_ingest`` (sizing, check, go) reads the same itemized setup:
    43.51 GiB at 1792x1024x55 GFS under the 0.65 fraction, now within a
    tenth of the door's own price."""
    from woof.core import preflight
    from woof.experiment import experiment_from_run_config
    from datetime import datetime

    exp = experiment_from_run_config(_cfg(1792, 1024, 55),
                                     datetime(2026, 9, 27, 21))
    estimate = preflight.estimate_ingest(exp, source="gfs",
                                         forcing_interval_seconds=10800.0)
    assert not hasattr(preflight, "INGEST_TRANSIENT_PER_TIME_FRACTION")
    assert estimate.headroom == pp.PREPARATION_POOL_HEADROOM
    assert estimate.transient_bytes == estimate.setup_bytes > 0
    assert estimate.peak_envelope_bytes < 43.51 * GIB


def test_the_ingest_estimate_takes_its_route_rows_headroom(tmp_path):
    """The run route's door prices its pool at its row's 1.25 (a 12 km
    ERA5 run reserved 1.217 times its itemized arrays), while
    ``estimate_ingest`` priced the same preparation at 1.20.  Named by its
    route it takes that row's headroom, and ``woof check`` names the run
    route for a config whose forcing is its ``[case_data]``."""
    from datetime import datetime

    from woof.core import preflight
    from woof.experiment import experiment_from_run_config

    exp = experiment_from_run_config(
        _cfg(500, 400, 49, dx=12000.0, dy=12000.0),
        datetime(2026, 9, 27, 21))

    def estimate(route):
        return preflight.estimate_ingest(
            exp, source="era5", forcing_interval_seconds=21600.0,
            route=route)

    default, run = estimate(None), estimate("experiment")
    assert default.headroom == pp.PREPARATION_POOL_HEADROOM
    assert run.headroom == pp.PREPARATION_ROUTES["experiment"].headroom
    assert run.headroom == 1.25
    assert run.subtotal_bytes == default.subtotal_bytes
    assert run.alloc_estimate_bytes > default.alloc_estimate_bytes
    assert estimate("gfs").headroom == pp.PREPARATION_POOL_HEADROOM
    with pytest.raises(ValueError, match="no preparation route 'nowhere'"):
        estimate("nowhere")

    case = tmp_path / "case.toml"
    case.write_text('[experiment]\nname = "c"\n\n[case_data]\n'
                    'path = "data"\n', encoding="utf-8")
    fetched = tmp_path / "fetched.toml"
    fetched.write_text('[experiment]\nname = "f"\n\n[fetch]\n'
                       'source = "gfs"\n', encoding="utf-8")
    assert preflight.config_preparation_route(case) == "experiment"
    assert preflight.config_preparation_route(fetched) is None


def test_the_downscale_price_takes_its_route_rows_headroom(monkeypatch):
    """``price_downscale_interpolation`` read the module default headroom
    directly, so a headroom set on the ``downscale-child`` row would have
    been dropped; it reads the row, as ``price_preparation`` does."""
    from dataclasses import replace
    from types import MappingProxyType

    child = _cfg(450, 450, 49, dx=1000.0, dy=1000.0, specified=False,
                 nested=True)

    def price():
        return pp.price_downscale_interpolation(
            parent_nx=408, parent_ny=420, parent_nz=49, parent_fields=16,
            child_cfg=child)

    before = price()
    rows = dict(pp.PREPARATION_ROUTES)
    rows["downscale-child"] = replace(rows["downscale-child"],
                                      pool_headroom=1.5)
    monkeypatch.setattr(pp, "PREPARATION_ROUTES", MappingProxyType(rows))
    after = price()
    live = sum(before.terms[key] for key in
               ("parent_fields", "child_fields", "setup_residual"))
    assert after.terms["pool_headroom"] > before.terms["pool_headroom"]
    assert after.terms["pool_headroom"] == -(-live // 2)
    assert "pool headroom x1.5" in after.basis


def test_source_inventory_counts_levels_fields_and_planes():
    shapes = {"TT": (39, 10, 12), "UU": (39, 10, 13), "VV": (39, 11, 12),
              "QC": (39, 10, 12), "PSFC": (10, 12), "SOILT": (9, 10, 12)}
    inventory = pp.SourceInventory.from_shapes(shapes)
    assert (inventory.levels, inventory.level_fields,
            inventory.surface_planes) == (39, 4, 10)
    assert inventory.mass_outputs == 3


def test_era5_prices_its_whole_source_grid_humidity_conversion():
    """A small target from a global source: the FP64 conversion on the
    SOURCE grid binds, which no target-shaped term can see (F13)."""
    inventory = pp.SourceInventory(levels=37, level_fields=5,
                                   surface_planes=23,
                                   source_points=1440 * 721,
                                   fp64_humidity_transform=True)
    price = pp.price_preparation("era5", [_cfg(100, 100, 40)], inventory,
                                 platform="linux")
    assert price.phase == "source transform"
    assert price.terms["source_transform"] == 72 * 37 * 1440 * 721


def test_native_hrrr_prices_its_boundary_workers():
    """The kept f00 state beside spawned workers, each with a context (F12)."""
    inventory = pp.NOMINAL_SOURCE_INVENTORIES["hrrr-native"]
    cfg = _cfg(1799, 1059, 50)
    one = pp.price_preparation("hrrr-native", [cfg], inventory,
                               boundary_workers=0, platform="linux")
    many = pp.price_preparation("hrrr-native", [cfg], inventory,
                                boundary_workers=32, platform="linux")
    assert many.need_bytes > one.need_bytes
    assert many.phase == "boundary workers"
    assert many.terms["worker_slots"] > 32 * pp.context_bytes(platform="linux")


def test_downscale_prices_the_parent_extent_not_the_child_alone():
    """A small child of a large parent: the parent's fields bind (F08)."""
    child = _cfg(138, 138, 20, specified=False, nested=True)
    price = pp.price_downscale_interpolation(
        parent_nx=4096, parent_ny=4096, parent_nz=137, parent_fields=14,
        child_cfg=child, platform="linux")
    assert price.terms["parent_fields"] > 40 * GIB
    assert price.need_bytes > price.terms["parent_fields"]


def test_forcing_price_reads_one_snapshot_not_every_time():
    """A lazy forcing sequence is priced from its first snapshot and its
    length; iterating it would pack every valid time at once (the mapped
    route's 17 GB host regression at 1792x1024x55)."""

    class LazyForcing:
        def __init__(self, count):
            self.count, self.read = count, []

        def __len__(self):
            return self.count

        def __getitem__(self, index):
            if isinstance(index, slice):
                raise AssertionError("the price sliced the forcing")
            self.read.append(index)
            fields = {"TT": SimpleNamespace(shape=(40, 1059, 1799)),
                      "UU": SimpleNamespace(shape=(40, 1059, 1799)),
                      "VV": SimpleNamespace(shape=(40, 1059, 1799)),
                      "PSFC": SimpleNamespace(shape=(1059, 1799))}
            return SimpleNamespace(fields=fields)

        def __iter__(self):
            raise AssertionError("the price iterated every forcing time")

    forcing = LazyForcing(7)
    exp = SimpleNamespace(domains=[SimpleNamespace(run=_cfg(896, 512, 59))])
    price = pp.price_forcing_preparation("mapped", exp, forcing)
    assert forcing.read == [0]
    assert price.need_bytes > 0
    one_shot = pp.price_forcing_preparation(
        "mapped", exp, (forcing[0] for _ in range(7)))
    assert one_shot.need_bytes == price.need_bytes


# ------------------------------------------------- one stand-in card, every route

def _route_prices():
    c3 = _cfg(1792, 1024, 55)
    gfs = pp.SourceInventory(levels=23, level_fields=5, surface_planes=19)
    era5 = pp.SourceInventory(levels=37, level_fields=5, surface_planes=23,
                              source_points=1440 * 721,
                              fp64_humidity_transform=True)
    child = _cfg(900, 900, 55, specified=False, nested=True)
    return {
        "gfs": pp.price_preparation("gfs", [c3], gfs),
        "gfs tree": pp.price_preparation("gfs", [c3, child], gfs,
                                         boundary_intervals=4),
        "era5": pp.price_preparation("era5", [c3], era5),
        "mapped": pp.price_preparation("mapped", [c3],
                                       pp.MEASURED_SOURCE_INVENTORY),
        "met_em": pp.price_preparation("met_em", [c3, child], gfs,
                                       boundary_intervals=4),
        "hrrr-native": pp.price_preparation(
            "hrrr-native", [c3], pp.NOMINAL_SOURCE_INVENTORIES["hrrr-native"],
            boundary_workers=2),
        "experiment": pp.price_preparation("experiment", [c3], gfs,
                                           boundary_intervals=4),
        "experiment host store": pp.price_preparation(
            "experiment-host-store", [c3], gfs, boundary_intervals=4),
        "downscale child": pp.price_downscale_interpolation(
            parent_nx=1792, parent_ny=1024, parent_nz=55, parent_fields=14,
            child_cfg=child),
    }


ROUTES = tuple(_route_prices())
SMALL_CARD = {"free_bytes": 3 * GIB, "total_bytes": 24 * GIB,
              "utilization_gpu_percent": 0}
ROOMY_CARD = {"free_bytes": 900 * GIB, "total_bytes": 960 * GIB,
              "utilization_gpu_percent": 0}


@pytest.fixture
def card(monkeypatch):
    """A certified stand-in CUDA backend whose allocations are recorded,
    and a stand-in card whose reading the test sets."""
    allocations = []

    def _allocate(*args, **kwargs):
        allocations.append((args, kwargs))
        raise AssertionError("the stand-in card was allocated on")

    runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDevice=lambda: 0,
                              runtimeGetVersion=lambda: 13020)
    module = SimpleNamespace(__version__="14.2.0", zeros=_allocate,
                             empty=_allocate, asarray=_allocate,
                             cuda=SimpleNamespace(runtime=runtime))

    def cuda_backend():
        return SimpleNamespace(name="cuda", array_module=module)

    # A class, not a factory function: preprocessing_math_scope asks
    # isinstance(backend, ParallelCpuPreprocessBackend) at every wrapped
    # preparation door, and a function there is a TypeError.
    class cpu_backend(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(name="cpu", workers=kwargs.get("workers"))

    reading = {}
    monkeypatch.setattr("woof.ingest.preparation_workers.host_available_bytes", lambda: 1024 * GIB)
    monkeypatch.setattr(backend, "CudaPreprocessBackend", cuda_backend)
    monkeypatch.setattr(backend, "ParallelCpuPreprocessBackend", cpu_backend)
    monkeypatch.setattr(backend, "_gpu_runtime_installed", lambda: True)
    monkeypatch.setattr(backend, "_ANNOUNCED_AUTO_REASONS", set())
    monkeypatch.delenv("WOOF_NATIVE_DISTRIBUTION_MANIFEST", raising=False)
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess",
                        lambda **_: dict(reading))
    return SimpleNamespace(reading=reading, allocations=allocations)


@pytest.mark.parametrize("route", ROUTES)
def test_auto_keeps_cuda_waiting_for_memory_or_refuses_impossible_capacity(
        route, card, capsys, monkeypatch):
    import time
    card.reading.update(SMALL_CARD)
    price = _route_prices()[route]
    waits = []
    def release(seconds):
        waits.append(seconds)
        card.reading.update(ROOMY_CARD)
    monkeypatch.setattr(time, "sleep", release)
    if route != "mapped" and price.need_bytes > SMALL_CARD["total_bytes"]:
        with pytest.raises(backend.PreparationDeviceRefused, match="out-of-memory"):
            backend.resolve_preprocess_backend("auto", price=price)
        assert waits == []
    else:
        chosen = backend.resolve_preprocess_backend("auto", price=price)
        assert chosen.name == "cuda"
        assert chosen.selection["device_fit"]["fits"] is True
        if route == "mapped":
            assert isinstance(chosen, BoundedCudaPreprocessBackend)
            assert chosen.selection["chunking"]["retained_arrays"] == "host"
        else:
            assert waits == [1.0]
            assert "waiting for" in capsys.readouterr().err
    assert card.allocations == []


@pytest.mark.parametrize("route", ROUTES)
def test_explicit_cuda_is_refused_by_name_before_anything_is_allocated(
        route, card):
    card.reading.update(SMALL_CARD)
    price = _route_prices()[route]
    # Occupied memory waits. Refusal is for insufficient total capacity,
    # including a mapped build whose smallest bounded batch cannot fit.
    card.reading.update(free_bytes=128 * 1024**2, total_bytes=128 * 1024**2)
    with pytest.raises(backend.PreparationDeviceRefused) as refused:
        backend.resolve_preprocess_backend("cuda", price=price)
    message = str(refused.value)
    assert "refused before anything was allocated" in message
    assert "--preprocess-backend cpu" in message
    assert "out-of-memory" in message and price.stage in message
    assert price.summary() in message
    assert isinstance(refused.value, MemoryError)
    assert card.allocations == []


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_a_card_with_room_still_prepares_on_the_card(route, requested, card):
    card.reading.update(ROOMY_CARD)
    price = _route_prices()[route]
    chosen = backend.resolve_preprocess_backend(requested, price=price)
    assert chosen.name == "cuda"
    if route == "mapped":
        assert isinstance(chosen, BoundedCudaPreprocessBackend)
    assert chosen.selection["device_fit"]["fits"] is True
    assert card.allocations == []


def test_an_unread_card_keeps_the_choice_and_says_so(card, monkeypatch):
    monkeypatch.setattr(device_probe, "device_memory_probe_subprocess",
                        lambda **_: None)
    price = _route_prices()["gfs"]
    chosen = backend.resolve_preprocess_backend("cuda", price=price)
    assert chosen.name == "cuda"
    assert chosen.selection["device_fit"]["fits"] is None


def test_the_name_road_takes_the_same_decision(card, capsys, monkeypatch):
    """The offline child passes its backend by name."""
    price = _route_prices()["downscale child"]
    card.reading.update(SMALL_CARD)
    import time
    monkeypatch.setattr(time, "sleep", lambda _: card.reading.update(ROOMY_CARD))
    name, selection = backend.decide_preparation_device("auto", price)
    assert name == "cuda" and selection["device_fit"]["fits"] is True
    assert "waiting for" in capsys.readouterr().err
    card.reading.update(free_bytes=GIB, total_bytes=GIB)
    with pytest.raises(backend.PreparationDeviceRefused):
        backend.decide_preparation_device("cuda", price)
    assert backend.decide_preparation_device("cpu", price) == ("cpu", None)
    card.reading.clear()
    card.reading.update(ROOMY_CARD)
    assert backend.decide_preparation_device("cuda", price)[0] == "cuda"


def test_native_hrrr_budgeted_backend_takes_the_price(card):
    from tools import hrrr_single_domain_benchmark as benchmark

    card.reading.update(SMALL_CARD)
    args = SimpleNamespace(preprocess_backend="auto", preprocess_workers=4,
                           cpu_preprocess_bridge=None)
    price = benchmark.native_preparation_price(
        _cfg(1799, 1059, 50), forcing_times=7, prepare_workers=2)
    with pytest.raises(backend.PreparationDeviceRefused):
        benchmark._budgeted_preprocess_backend(args, price=price)
    args.preprocess_backend = "cuda"
    with pytest.raises(backend.PreparationDeviceRefused):
        benchmark._budgeted_preprocess_backend(args, price=price)


# ------------------------------------------------------ doors: order and default

#: Each door, the function that prepares, the call that decides, and the
#: calls that make the first device allocation.
DOORS = (
    ("woof/gfs_direct.py", "prepare_gfs_wrf", "admit_preparation",
     ("build_forcing_time", "interpolate_era5_to_lambert", "initialize_real")),
    ("woof/era5_direct.py", "prepare_era5_wrf", "admit_preparation",
     ("build_forcing_time", "interpolate_era5_to_lambert", "initialize_real")),
    ("woof/mapped_direct.py", "prepare_mapped_wrf", "price_forcing_preparation",
     ("build_forcing_time", "interpolate_era5_to_lambert", "initialize_real")),
    ("woof/metem_forecast.py", "prepare_metem_run", "price_preparation",
     ("initialize_real",)),
    ("woof/runtime.py", "prepare_real_case", "price_preparation",
     ("interpolate_era5_to_lambert", "initialize_real")),
    ("tools/hrrr_single_domain_benchmark.py", "run", "native_preparation_price",
     ("_initialize_state", "_initialize_boundary_sides")),
    ("woof/offline_child_run.py", "_run", "decide_preparation_device",
     ("interpolate_parent_initial_state", "build_offline_lateral_boundaries")),
)


def _call_lines(function, names):
    lines = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            target = node.func
            name = getattr(target, "id", None) or getattr(target, "attr", None)
            if name in names:
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("path,function,decision,device", DOORS,
                         ids=[door[1] for door in DOORS])
def test_every_door_decides_before_its_first_device_allocation(
        path, function, decision, device):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == function)
    decided = _call_lines(body, {decision})
    allocated = _call_lines(body, set(device))
    assert decided, f"{path}:{function} never prices its preparation"
    assert allocated, f"{path}:{function} lost its device calls"
    assert min(decided) < min(allocated), (path, decided, allocated)


def test_every_card_door_defaults_to_auto():
    from woof import offline_child_run, twentycrv3_wrf
    from woof.ingest.case_store import CaseStoreRequest
    from woof.preprocess_policy import preprocess_backend_choice

    assert twentycrv3_wrf._parser().get_default("preprocess_backend") == "auto"
    assert offline_child_run._parser().get_default("preprocess_backend") == "auto"
    assert CaseStoreRequest(Path("x")).backend == "auto"
    assert preprocess_backend_choice(source="met_em") == ("auto", None)
    assert preprocess_backend_choice(source="hrrr-prs") == ("auto", None)
    assert inspect.signature(
        twentycrv3_wrf.prepare_20crv3_wrf).parameters[
            "preprocess_backend"].default == "auto"
    # The library entry points, not only their mains (A104).
    from woof import era5_direct, gfs_direct, mapped_direct
    for function in (gfs_direct.prepare_gfs_wrf, era5_direct.prepare_era5_wrf,
                     mapped_direct.prepare_mapped_wrf):
        assert inspect.signature(function).parameters[
            "preprocess_backend"].default == "auto", function.__name__
    for path in ("tools/hrrr_single_domain_benchmark.py", "woof/downscale.py"):
        text = (ROOT / path).read_text(encoding="utf-8")
        start = text.index('"--preprocess-backend"')
        flag = text[start:text.index(")", text.index("default=", start))]
        assert 'default="auto"' in flag, path
        assert '"auto"' in flag.split("default=")[0], path


def test_every_route_row_names_its_door_and_stage():
    for name, row in pp.PREPARATION_ROUTES.items():
        assert row.name == name and row.door and row.stage.startswith("while ")


# ------------------------------------------------------------------ woof go

_STREAMED_ERA5 = """\
[experiment]
name = "prep-fit"
start_time = 2026-09-26T06:00:00
run_seconds = 21600.0
restart_interval_s = 0.0

[fetch]
source = "era5"
cycle = "2026-09-26T06"
hours = 6
cadence = 6

[shared]
nz = 49
ztop = 20000.0
moist = true
moist_cq = true
mp_physics = 10
ra_lw_physics = 4
ra_sw_physics = 4
sf_sfclay_physics = 91
sf_surface_physics = 2
bl_pbl_physics = 1
nwp_diagnostics = 1

[tiles]
mode = "on"
store = "host"

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 700
ny = 700
time_step = 15
dx = 3000.0
history_interval_s = 3600.0
"""


def test_go_does_not_admit_using_a_load_dependent_cpu_fallback(
        tmp_path, monkeypatch):
    """A streamed forecast that fits the card beside a preparation that does
    not: ``woof go`` used to refuse the WHOLE run on the preparation's card
    price, although the door (backend auto) prepares it on the CPU.  The
    gate now prices that road, and weighs the CPU preparation's host term."""
    from woof import go_cli
    from woof.core import preflight, streaming

    config = tmp_path / "prep-fit.toml"
    config.write_text(_STREAMED_ERA5, encoding="utf-8")
    monkeypatch.setattr(streaming, "_host_total_bytes", lambda: 2048 * GIB)
    # The host's AVAILABLE bytes too: the gate weighs the CPU preparation against them, and on a 16 GB runner
    # busy with its own test workers the real reading fell under the preparation's 9.2 GiB (2.9.2 R7b route B).
    monkeypatch.setattr(preflight, "host_available_bytes", lambda: 2048 * GIB)
    exp = preflight._load_experiment_any(config)
    # The gate's own machine, built from the stand-in card's reading: no
    # process here stands up a CUDA context.
    free = 8 * GIB
    probe = {"free_bytes": free, "total_bytes": free + GIB,
             "name": "stand-in card", "local_memory_bytes_per_thread": 0}
    on_card = preflight.estimate_phases(
        exp, source="era5", forcing_interval_seconds=21600.0,
        machine=go_cli._planner_machine(probe))
    assert on_card.preprocess_backend == "auto"
    assert on_card.streamed is not None
    assert on_card.forecast_envelope_bytes < free < on_card.ingest_envelope_bytes

    monkeypatch.setattr(preflight, "device_memory_probe_subprocess",
                        lambda *a, **k: dict(probe))
    verdict = go_cli.memory_gate({"config": str(config), "source": "era5",
                                  "cadence": 6})
    assert verdict["refuse"] is True, verdict["verdict"]
    assert verdict["phases"].preprocess_backend == "auto"
    assert "may prepare on the CPU" not in verdict["verdict"]
    assert verdict["preparation_on_cpu"] is None


def test_a_door_that_resolved_before_its_decode_reads_the_card_again(card):
    """GFS and ERA5 resolve ``auto`` before their host decode and weigh the
    price after it, minutes later on a card other work shares (a chained
    preparation beside a forecast).  The decision reads the free memory at
    the decision, not the reading ``auto`` took before the decode: reusing
    that one kept CUDA on a card that had since filled."""
    card.reading.update(ROOMY_CARD)
    chosen = backend.resolve_preprocess_backend("auto")
    assert chosen.name == "cuda"
    card.reading.clear()
    card.reading.update(SMALL_CARD)
    for route in ("gfs", "era5"):
        with pytest.raises(backend.PreparationDeviceRefused, match="3.0 GiB free"):
            backend.admit_preparation(chosen, _route_prices()[route])
        assert chosen.name == "cuda", route
    assert card.allocations == []


def test_a_preparation_already_on_the_cpu_is_never_priced(card):
    """The GFS, ERA5 and mapped doors hand the price over as a callable, so
    a backend already on the CPU never runs it: pricing is only ever for a
    decision about the card, and a CPU preparation has none to take.  A
    CUDA answer still prices, once."""
    calls = []

    def price():
        calls.append("priced")
        return _route_prices()["gfs"]

    chosen = backend.resolve_preprocess_backend("cpu", price=price)
    assert chosen.name == "cpu" and calls == []
    assert backend.admit_preparation(chosen, price) is chosen and calls == []
    assert backend.decide_preparation_device("cpu", price) == ("cpu", None)
    assert calls == []
    card.reading.update(SMALL_CARD)
    with pytest.raises(backend.PreparationDeviceRefused):
        backend.resolve_preprocess_backend("auto", price=price)
    assert calls == ["priced"]
    card.reading.clear()
    card.reading.update(ROOMY_CARD)
    chosen = backend.resolve_preprocess_backend("cuda", price=price)
    assert chosen.name == "cuda" and calls == ["priced", "priced"]
    assert card.allocations == []


def _door_argv(door, tmp_path):
    common = ["--wps-namelist", str(tmp_path / "namelist.wps"),
              "--experiment-config", str(tmp_path / "case.toml"),
              "--input-manifest", str(tmp_path / "inputs.json"),
              "--input-manifest-sha256", "0" * 64,
              "--output-root", str(tmp_path / "out"),
              "--preprocess-backend", "cuda"]
    if door == "gfs":
        return ["--series", "series.tsv", "--cycle", "2026-07-29_06:00:00",
                "--bridge", "bridge.exe", *common]
    if door == "era5":
        return ["--grib", "era5.grib", "--vtable", "Vtable.ERA5",
                "--bridge", "bridge.exe", *common]
    return ["--source-format", "netcdf", "--mapping", "mapping.json",
            "--composition", "composition.json",
            "--input", str(tmp_path / "one.nc"),
            "--geog-root", str(tmp_path / "geog"), *common]


@pytest.mark.parametrize("door,label", [
    ("gfs", "rw-wps --source gfs: "),
    ("era5", "rw-wps --source era5: "),
    ("mapped", "rw-wps: "),
])
def test_an_explicit_cuda_refusal_reaches_the_user_as_a_sentence(
        door, label, monkeypatch, tmp_path, capsys):
    """An explicit ``--preprocess-backend cuda`` the card cannot hold is
    refused with its remedy line.  The GFS, ERA5 and mapped door mains let
    PreparationDeviceRefused out as a traceback, which buried that line
    under a stack; they now answer it the way ``woof`` does: the message
    on stderr and exit 2."""
    import importlib

    module = importlib.import_module(f"woof.{door}_direct")
    price = _route_prices()[door]
    message = backend.preparation_refusal_message(
        price, SMALL_CARD["free_bytes"], SMALL_CARD["total_bytes"])

    def refuse(**_kwargs):
        raise backend.PreparationDeviceRefused(message)

    monkeypatch.setattr(module, f"prepare_{door}_wrf", refuse)
    if door == "mapped":
        monkeypatch.setattr(module, "load_mapping",
                            lambda path: {"format": "netcdf"})
    code = module.main(_door_argv(door, tmp_path))
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith(label + message.splitlines()[0]), err
    assert "--preprocess-backend cpu" in err
    assert "Traceback" not in err


# ---------------------------------------------- the floor, before the decode

#: Each door, its experiment load and its host decode (A98).
FLOOR_DOORS = (
    ("woof/gfs_direct.py", "prepare_gfs_wrf", "load_experiment",
     "_load_bridge_snapshots"),
    ("woof/era5_direct.py", "prepare_era5_wrf", "load_era5_adapter_config",
     "cached_era5_snapshots"),
    ("woof/mapped_direct.py", "prepare_mapped_wrf", "load_experiment",
     "decode_composed_source"),
)


@pytest.mark.parametrize("route", ["gfs", "era5", "mapped"])
def test_the_floor_never_prices_above_the_decoded_preparation(route):
    """A98: the floor is a lower bound, so it never refuses a preparation
    the decoded price admits on the same card, and it still carries the
    domains' own state."""
    domains = [_cfg(1792, 1024, 55),
               _cfg(900, 900, 55, specified=False, nested=True)]
    exp = SimpleNamespace(domains=[SimpleNamespace(run=cfg) for cfg in domains])
    floor = pp.price_preparation_floor(route, exp)
    for inventory in (pp.FLOOR_SOURCE_INVENTORY, pp.MEASURED_SOURCE_INVENTORY,
                      pp.SourceInventory(levels=37, level_fields=5,
                                         surface_planes=23,
                                         source_points=1440 * 721,
                                         fp64_humidity_transform=True)):
        for intervals in (0, 6):
            decoded = pp.price_preparation(route, domains, inventory,
                                           boundary_intervals=intervals)
            assert floor.need_bytes <= decoded.need_bytes, (inventory,
                                                            intervals)
    assert floor.terms["model_state"] >= pp.state_bytes(domains[0])
    assert floor.basis == pp.PREPARATION_FLOOR_BASIS


@pytest.mark.parametrize("path,function,load,decode", FLOOR_DOORS,
                         ids=[door[1] for door in FLOOR_DOORS])
def test_every_source_door_weighs_the_floor_before_its_decode(
        path, function, load, decode):
    """A98: an explicit cuda too big for the card was refused only after
    the host decode and the statics (501 s at 1792x1024x55 on a 24 GB
    card).  Each door prices the floor after its experiment load and
    before its decode."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == function)
    floor = _call_lines(body, {"price_preparation_floor"})
    loaded = _call_lines(body, {load})
    decoded = _call_lines(body, {decode})
    assert floor, f"{path}:{function} never weighs the floor"
    assert decoded, f"{path}:{function} lost its decode"
    assert max(loaded) < min(floor) < min(decoded), (loaded, floor, decoded)


_FLOOR_CASE = _STREAMED_ERA5.split("[tiles]")[0] + """\
[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
nx = 1792
ny = 1024
time_step = 15
dx = 3000.0
history_interval_s = 3600.0
"""


def test_an_explicit_cuda_gfs_preparation_is_refused_before_the_decoder_runs(
        card, monkeypatch, tmp_path):
    """A98, driven through the GFS door on a stand-in card with 3 GiB
    free: the refusal comes from the floor, and the decoder is never
    invoked and the card never allocated."""
    from woof import gfs_direct

    decoded = []

    def decoder(*args, **kwargs):
        decoded.append(args)
        raise AssertionError("the decoder ran before the card was weighed")

    class _StandInCuda:
        name = "cuda"
        array_module = SimpleNamespace(cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDevice=lambda: 0)))

        def __init__(self):
            self.selection = {"requested": "cuda", "backend": "cuda",
                              "reason": backend.NAMED_BY_CALLER}

        def receipt(self):
            return {"backend": "cuda", "selection": dict(self.selection)}

    card.reading.update(SMALL_CARD)
    monkeypatch.setattr(gfs_direct, "_load_bridge_snapshots", decoder)
    monkeypatch.setattr(gfs_direct, "resolve_preprocess_backend",
                        lambda *args, **kwargs: _StandInCuda())
    monkeypatch.setattr(gfs_direct, "_implementation_sha256", lambda: {})
    monkeypatch.setattr(gfs_direct, "_git_source_identity", lambda: {})
    monkeypatch.setattr(
        gfs_direct, "_verify_input_manifest",
        lambda *args, **kwargs: {
            "source": {"model": "GFS", "product": "pgrb2.0p25",
                       "cycle": "2026-09-26T06:00:00Z"},
            "files": {"bridge": {"sha256": "0" * 64},
                      "experiment_config": {"sha256": "0" * 64}}})
    for name in ("gfs.t06z.pgrb2.0p25.f000", "gfs.t06z.pgrb2.0p25.f006",
                 "namelist.wps", "inputs.json", "geog.keep"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "series.tsv").write_text(
        "0\tgfs.t06z.pgrb2.0p25.f000\n6\tgfs.t06z.pgrb2.0p25.f006\n",
        encoding="utf-8")
    bridge = tmp_path / "bridge"
    bridge.write_bytes(b"")
    bridge.chmod(0o755)
    config = tmp_path / "case.toml"
    config.write_text(_FLOOR_CASE, encoding="utf-8")

    with pytest.raises(backend.PreparationDeviceRefused) as refused:
        gfs_direct.prepare_gfs_wrf(
            series=tmp_path / "series.tsv", cycle="2026-09-26_06:00:00",
            bridge=bridge, wps_namelist=tmp_path / "namelist.wps",
            static_input=None, static_receipt=None,
            experiment_config=config, input_manifest=tmp_path / "inputs.json",
            input_manifest_sha256="0" * 64,
            output_root=tmp_path / "out", preprocess_backend="cuda",
            geog_root=tmp_path)
    assert decoded == []
    assert card.allocations == []
    assert "refused before anything was allocated" in str(refused.value)


def test_device_real_columns_cover_the_measured_pool_peak():
    # sm_120, NVRTC 13.4.92, three init-1 capture replays with allocation
    # hooks: 12,371,798,528 B live and 13,114,399,232 B reserved each time.
    cfg = _cfg(800, 600, 50)
    inventory = pp.SourceInventory(39, 11, 29, device_real_columns=True)
    price = pp.price_preparation("mapped", [cfg], inventory)
    measured = 13_114_399_232 + price.terms["cuda_context"]
    assert measured <= price.need_bytes <= 1.25 * measured
    assert price.terms["real_columns"] > 0
    host_store = pp.price_preparation("experiment-host-store", [cfg], inventory)
    assert "real_columns" not in host_store.terms


def test_device_real_inventory_uses_the_dispatch_fields():
    shapes = {"PRES": (39, 2, 3), "SPFH": (39, 2, 3), "Q2": (2, 3)}
    assert pp.SourceInventory.from_shapes(shapes).device_real_columns
    decoded = dict(zip(("air_pressure", "specific_humidity", "specific_humidity_2m"), shapes.values()))
    assert pp.SourceInventory.from_shapes(decoded).device_real_columns
    del shapes["Q2"]
    assert not pp.SourceInventory.from_shapes(shapes).device_real_columns


def test_mp28_prices_host_columns_and_lazy_temperature_uploads():
    from dataclasses import replace
    cfg = replace(_cfg(800, 600, 50), mp_physics=28)
    inventory = pp.SourceInventory(1, 1, 1, device_real_columns=True)
    terms = pp._build(cfg, inventory)
    cells = cfg.nx * cfg.ny * cfg.nz
    closure = (12 + 16) * cells + 112 * min(cells, 1048576)
    assert "real_columns" not in terms
    assert terms["setup_residual"] >= closure
