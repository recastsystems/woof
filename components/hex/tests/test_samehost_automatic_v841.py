"""Capacity rows and default public door dispatch, without GPU allocation."""
from types import SimpleNamespace
import sys
import pytest
from test_forecast_door import _proven_architecture, _reference_card, _satisfied_seam_pin
from woof.hex.device_admission import CARD_COUNT_ROWS, select_card_count
from woof.hex.drivers import samehost_forecast_v841 as samehost


@pytest.mark.parametrize("visible,expected", [(1, 1), (2, 2), (4, 4), (7, 4), (8, 8)])
@pytest.mark.parametrize("memory", [16, 32, 96])
def test_large_mesh_selects_available_row_count(visible, expected, memory):
    policy = select_card_count(157859, memory * 1024**3, visible)
    assert policy["selected_cards"] == expected
    row = next(row for row in CARD_COUNT_ROWS if row.key == policy["row"])
    assert expected in row.counts
    assert row.min_card_bytes <= policy["card_memory_bytes"] <= row.max_card_bytes


def test_small_mesh_keeps_single_card_without_cuda(monkeypatch):
    monkeypatch.setattr(samehost, "visible_card_memory", lambda: pytest.fail("small row queried CUDA"))
    assert samehost.automatic_options(SimpleNamespace(), SimpleNamespace(cells=41145)) is None


@pytest.mark.parametrize("cells", [131072, 294799, 655362, 835586])
@pytest.mark.parametrize("mode", ["shadow", "active"])
def test_fullstep_lts_defaults_to_one_card_before_capacity_census(monkeypatch, cells, mode):
    from woof.hex import time_ledger
    events = []
    monkeypatch.setattr(time_ledger, "emit", lambda *args, **kwargs: events.append(kwargs))
    monkeypatch.setattr(samehost, "visible_card_memory",
                        lambda: pytest.fail("LTS must not select extra visible cards"))
    request = SimpleNamespace(cells=cells, fullstep_local_timestep=True, fullstep_lts_mode=mode)
    assert samehost.automatic_options(SimpleNamespace(samehost_devices=None), request) is None
    assert events[-1]["attrs"]["card_selection"]["selected_cards"] == 1


def test_dycore_lts_also_keeps_one_card(monkeypatch):
    monkeypatch.setattr(samehost, "visible_card_memory",
                        lambda: pytest.fail("dycore LTS has no partitioned executor"))
    assert samehost.automatic_options(SimpleNamespace(samehost_devices=None),
                                     SimpleNamespace(cells=655362, local_timestep=True)) is None


def test_medium_mesh_uses_memory_tier():
    assert select_card_count(100000, 16 * 1024**3, 4)["selected_cards"] == 4
    assert select_card_count(100000, 96 * 1024**3, 4)["selected_cards"] == 1


def test_capacity_rows_cover_boundaries_without_overlap():
    for cells in (1, 65535, 65536, 131071, 131072, 2**63 - 1):
        for memory in (0, 24 * 1024**3 - 1, 24 * 1024**3, 48 * 1024**3 - 1, 48 * 1024**3):
            assert select_card_count(cells, memory, 4)["selected_cards"] in (1, 2, 4)


def test_automatic_layout_builder_ships_in_package():
    from woof.hex.drivers.samehost_layouts_v841 import build_assets
    assert callable(build_assets)


def test_new_work_directory_reuses_mesh_partition(tmp_path, monkeypatch):
    from pathlib import Path
    from test_build_samehost_layouts_v841 import _grid
    from woof.hex.drivers import samehost_layouts_v841 as builder

    monkeypatch.delenv("WOOF_HEX_CACHE_ROOT", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "user-cache"))
    grid = _grid(tmp_path)
    args = SimpleNamespace()
    first = samehost.built_layout_arguments(
        args, SimpleNamespace(grid=grid, scratch=tmp_path / "cycle-a" / "scratch"), 1)
    original = Path(first.samehost_part).read_bytes(), Path(first.samehost_layouts).read_bytes()
    monkeypatch.setattr(builder, "_canonical_graph", lambda *a: pytest.fail("partition rebuilt for new cycle"))
    second = samehost.built_layout_arguments(
        args, SimpleNamespace(grid=grid, scratch=tmp_path / "cycle-b" / "scratch"), 1)
    assert second.samehost_part == first.samehost_part
    assert second.samehost_layouts == first.samehost_layouts
    assert (Path(second.samehost_part).read_bytes(), Path(second.samehost_layouts).read_bytes()) == original
    assert first.samehost_layouts.is_relative_to(tmp_path / "user-cache" / "woof hex" / "samehost-layouts")
    assert not (grid.parent / "samehost-layouts").exists()


def test_concurrent_cycles_build_mesh_partition_once(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import time
    from test_build_samehost_layouts_v841 import _grid
    from woof.hex.drivers import samehost_layouts_v841 as builder

    monkeypatch.delenv("WOOF_HEX_CACHE_ROOT", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "user-cache"))
    grid = _grid(tmp_path)
    original = builder._canonical_graph
    walks = []
    start = Barrier(2)

    def walk(mesh):
        walks.append(1)
        time.sleep(.05)  # Let the other cold caller attempt to acquire the cache.
        return original(mesh)

    monkeypatch.setattr(builder, "_canonical_graph", walk)

    def cycle(index):
        start.wait(timeout=5)
        return samehost.built_layout_arguments(SimpleNamespace(), SimpleNamespace(
            grid=grid, scratch=tmp_path / str(index) / "scratch"), 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(cycle, (0, 1)))
    assert walks == [1]
    assert a.samehost_part == b.samehost_part
    assert a.samehost_layouts == b.samehost_layouts


@pytest.mark.parametrize("visible", [2, 4])
def test_automatic_options_builds_and_verifies_real_layout_assets(tmp_path, monkeypatch, visible):
    from woof.hex import device_admission as capacity
    from test_build_samehost_layouts_v841 import _grid
    from woof.hex.partition_assets_v841 import sha256_file
    from woof.hex.drivers import samehost_layouts_v841 as builder
    import subprocess

    grid = _grid(tmp_path)
    row = capacity.CardCountRow("fixture-capacity", 1, 100, 0, 2**63 - 1, (4, 2, 1))
    monkeypatch.setattr(capacity, "CARD_COUNT_ROWS", (row,))
    monkeypatch.setattr(samehost, "visible_card_memory", lambda: (96 * 1024**3,) * visible)
    monkeypatch.setenv("WOOF_HEX_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/fixture/gpmetis")
    def metis(command, **kwargs):
        from pathlib import Path
        Path(f"{command[-2]}.part.{visible}").write_text(
            "".join(f"{cell * visible // 12}\n" for cell in range(12)))
        return subprocess.CompletedProcess(command, 0, "fixture METIS partition\n", "")
    monkeypatch.setattr(builder.subprocess, "run", metis)
    arguments = SimpleNamespace(samehost_transport="auto", samehost_reference=None,
                                samehost_fingerprint_every=0)
    request = SimpleNamespace(cells=12, grid=grid, scratch=tmp_path / "scratch")
    before = sha256_file(grid)
    options = samehost.automatic_options(arguments, request)
    assert options["devices"] == tuple(range(visible))
    assert len(options["layouts"]) == visible
    assert options["card_selection"]["selected_cards"] == visible
    assert options["fingerprint_every"] == 0
    assert sha256_file(grid) == before
    assert samehost.automatic_options(arguments, request)["layouts_path"] == options["layouts_path"]


def test_resident_admission_names_cells_device_and_out_of_memory(monkeypatch):
    from contextlib import nullcontext
    cp = SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda device: nullcontext(),
        runtime=SimpleNamespace(getDeviceCount=lambda: 1, memGetInfo=lambda: (1, 96 * 1024**3),
            cudaDevAttrMaxThreadsPerMultiProcessor=39, deviceGetAttribute=lambda attribute, device: 1536,
            getDeviceProperties=lambda device: {"major": 12, "minor": 0, "name": b"capacity fixture",
                                                "multiProcessorCount": 170})))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.delenv("WOOF_HEX_NO_LOCAL_GPU", raising=False)
    door = SimpleNamespace(_SuppliedRow=type("Supplied", (), {}), request_configuration=lambda request: "global")
    request = SimpleNamespace(cells=157859, mesh="capacity-fixture", model=None, headroom_bytes=None)
    options = {"devices": (0,), "layouts": [SimpleNamespace(n_local_cells=80000)]}
    with pytest.raises(ValueError, match=r"157,859 mesh cells on device 0.*prevents out of memory"):
        samehost.admit_request(request, {}, options, door)


def test_resident_admission_prices_the_measured_thread_capacity(monkeypatch):
    from contextlib import nullcontext
    from woof.hex import forecast_door as door
    cp = SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda device: nullcontext(),
        runtime=SimpleNamespace(getDeviceCount=lambda: 1,
            memGetInfo=lambda: (96 * 1024**3, 96 * 1024**3),
            cudaDevAttrMaxThreadsPerMultiProcessor=39,
            deviceGetAttribute=lambda attribute, device: 2048,
            getDeviceProperties=lambda device: {"major": 12, "minor": 0, "name": b"capacity fixture",
                "multiProcessorCount": 188})))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.delenv("WOOF_HEX_NO_LOCAL_GPU", raising=False)
    request = SimpleNamespace(cells=40962, mesh="capacity-fixture", model=None,
                              headroom_bytes=None, regional=False)
    monkeypatch.setattr(door, "request_configuration", lambda request: "global")
    options = {"devices": (0,), "layouts": [SimpleNamespace(n_local_cells=40962)]}
    verdict = samehost.admit_request(request, {}, options, door)
    assert verdict.admitted
    assert verdict.model.card.max_threads_per_sm == 2048
    record = options["admission_records"][0]
    assert record["model"]["card"]["max_threads_per_sm"] == 2048
    assert record["required_free_bytes"] == verdict.required_bytes
    assert verdict.as_dict()["footprint_model"] == record["model"]


@pytest.mark.parametrize('stochastic', [None, '{"sppt":true}', '{"skeb":true,"degree":24}'])
def test_single_rank_stochastic_array_floor_is_enforced_before_construction(monkeypatch, stochastic):
    from contextlib import nullcontext
    from woof.hex import forecast_door as door
    from woof.hex.device_admission import CardProfile, model_for_card, samehost_rank_model
    # Device metadata is synthetic. This test never opens a CUDA context.
    card = CardProfile('capacity fixture', 170)
    model = samehost_rank_model(model_for_card(card, 'global'))
    arrays = door.stochastic_array_ledger(40962, stochastic)
    required = int(round(model.predict_bytes(40962)))+sum(arrays.values())
    available = [required]
    cp = SimpleNamespace(cuda=SimpleNamespace(
        Device=lambda device: nullcontext(),
        runtime=SimpleNamespace(getDeviceCount=lambda: 1,
            memGetInfo=lambda: (available[0], 96*1024**3),
            # Admission now reads resident threads per SM (merged capacity
            # model); 1536 is the CardProfile default this fixture prices.
            cudaDevAttrMaxThreadsPerMultiProcessor=39,
            deviceGetAttribute=lambda attribute, device: 1536,
            getDeviceProperties=lambda device: {'major':12, 'minor':0, 'name':b'capacity fixture',
                                                'multiProcessorCount':170})))
    monkeypatch.setitem(sys.modules, 'cupy', cp)
    monkeypatch.delenv('GPUWM_NO_LOCAL_GPU', raising=False)
    monkeypatch.delenv('WOOF_HEX_NO_LOCAL_GPU', raising=False)
    request = SimpleNamespace(cells=40962, mesh='capacity-fixture', model=None,
                              headroom_bytes=0, stochastic=stochastic, lbc_dir=None)
    options = {'devices':(0,), 'layouts':[SimpleNamespace(n_local_cells=40962)]}
    verdict = samehost.admit_request(request, {}, options, door)
    assert verdict.admitted and verdict.required_bytes == required
    assert verdict.as_dict()['stochastic_arrays'] == arrays
    assert options['memory_floor_by_device'] == {0:required}
    assert options['admission_records'][0]['stochastic_array_bytes'] == sum(arrays.values())
    available[0] -= 1
    with pytest.raises(ValueError, match='prevents out of memory'):
        samehost.admit_request(request, {}, options, door)


@pytest.mark.parametrize("cells,memory,cards", [(0, 1, 1), (1, -1, 1), (1, 1, 0)])
def test_invalid_allocation_names_the_breakage(cells, memory, cards):
    with pytest.raises(ValueError, match="nonempty mesh and card allocation"):
        select_card_count(cells, memory, cards)


def test_process_admission_refuses_the_budget_that_failed_at_hour_17(monkeypatch):
    from contextlib import nullcontext
    from woof.hex import forecast_door as door
    from woof.hex.device_admission import MIB
    from woof.hex.drivers import samehost_rank_group_v841 as ranks
    monkeypatch.setattr(ranks, "executor_choice", lambda requested, devices: (requested, "requested"))
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda device: nullcontext(),
        runtime=SimpleNamespace(getDeviceCount=lambda: 1,
            memGetInfo=lambda: (24_000*MIB, 32_000*MIB),
            cudaDevAttrMaxThreadsPerMultiProcessor=39,
            deviceGetAttribute=lambda attribute, device: 1536,
            getDeviceProperties=lambda device: {"major":12, "minor":0,
                "name":b"capacity fixture", "multiProcessorCount":170})))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.delenv("WOOF_HEX_NO_LOCAL_GPU", raising=False)
    monkeypatch.setattr(door, "request_configuration", lambda request: "global")
    request = SimpleNamespace(cells=40962, mesh="capacity-fixture", model=None,
                              headroom_bytes=None, regional=False)
    options = dict(devices=(0,0), layouts=[SimpleNamespace(n_local_cells=n) for n in (21445,21449)],
                   rank_executor="threads")
    assert samehost.admit_request(request, {}, options, door).admitted
    options["rank_executor"] = "processes"
    with pytest.raises(ValueError, match="prevents out of memory"):
        samehost.admit_request(request, {}, options, door)


def test_one_visible_large_mesh_records_automatic_choice(monkeypatch):
    from woof.hex import time_ledger
    events = []
    monkeypatch.setattr(time_ledger, "emit", lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setattr(samehost, "visible_card_memory", lambda: (96 * 1024**3,))
    assert samehost.automatic_options(SimpleNamespace(), SimpleNamespace(cells=157859)) is None
    assert events[-1][1]["attrs"]["card_selection"]["selected_cards"] == 1


def test_default_door_installs_automatic_adapter(tmp_path, monkeypatch):
    # Reuse the existing full successful public-door fixture, changing only
    # selection to the default path. It asserts the final receipt and exit.
    from test_forecast_door import test_samehost_two_device_run_finishes_the_door_receipt as check
    from woof.hex import forecast_door as door
    original = door.run_forecast
    def ordinary(arguments):
        arguments.samehost_devices = None
        monkeypatch.setattr(samehost, "automatic_options", samehost.request_options)
        return original(arguments)
    monkeypatch.setattr(door, "run_forecast", ordinary)
    check(tmp_path, monkeypatch)


def test_rank_ledger_keeps_physical_card_and_rank(tmp_path, monkeypatch):
    events = []
    module = SimpleNamespace(emit=lambda *args, **kwargs: events.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "woof.timeline.ledger", module)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,5,7")
    monkeypatch.delenv("GPU_MUTEX_CARDS", raising=False)
    monkeypatch.delenv("WOOF_TIME_LEDGER", raising=False)
    samehost.rank_event(tmp_path, "forecast.integrate", "begin", 2, 2, attrs={"step": 1})
    assert events[0][1]["card"] == 5
    assert events[0][1]["attrs"] == {"rank": 2, "visible_device": 2, "span_id": "rank2", "step": 1}


def test_rank_uuid_mask_uses_mutex_physical_card_indices(tmp_path, monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "woof.timeline.ledger", SimpleNamespace(
        emit=lambda *args, **kwargs: events.append(kwargs)))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    monkeypatch.setenv("GPU_MUTEX_CARDS", "3 7")
    samehost.rank_event(tmp_path, "forecast.integrate", "end", 1, 1)
    assert events[0]["card"] == 7


def test_rank_interval_ends_after_own_stream_completion():
    from test_samehost_rank_workers_v841 import _FakeCUDA, _Transport
    from woof.hex.partition_samehost_v841 import SameHostRanks
    cp = _FakeCUDA()
    observations = []
    with SameHostRanks((0, 1), _Transport(), cp=cp) as ranks:
        def work(rank):
            return rank
        work.after_synchronize = lambda rank: observations.append((rank, ranks.streams[rank].synchronize_calls))
        assert ranks.call(work) == [0, 1]
    assert sorted(observations) == [(0, 1), (1, 1)]


def test_failed_rank_observer_runs_after_own_stream_drain():
    from test_samehost_rank_workers_v841 import _FakeCUDA, _Transport
    from woof.hex.partition_samehost_v841 import SameHostRanks
    cp = _FakeCUDA()
    observations = []
    with SameHostRanks((0,), _Transport(), cp=cp) as ranks:
        def work(rank):
            raise ValueError("rank fixture failed")
        work.after_synchronize = lambda rank: observations.append((rank, ranks.streams[rank].synchronize_calls, ranks.failed))
        with pytest.raises(ValueError, match="rank fixture failed"):
            ranks.call(work)
    assert observations == [(0, 1, True)]


def test_output_intervals_close_on_original_writer_failure(monkeypatch):
    from woof.hex import time_ledger
    events = []
    monkeypatch.setattr(time_ledger, "emit", lambda *args, **kwargs: events.append((args, kwargs)))
    with pytest.raises(OSError, match="original writer failure"):
        with samehost.output_span("forecast.output.history", 60):
            raise OSError("original writer failure")
    assert [args for args, _ in events] == [("forecast.output.history", "begin"), ("forecast.output.history", "end")]
    assert all(kwargs["attrs"]["step"] == 60 for _, kwargs in events)


def test_explicit_cards_without_layouts_build_them_from_the_grid(tmp_path, monkeypatch):
    # A moving region crosses many member meshes: --samehost-devices alone partitions each one from its own grid.
    from test_build_samehost_layouts_v841 import _grid
    from woof.hex.drivers import samehost_layouts_v841 as builder
    import subprocess

    grid = _grid(tmp_path)
    monkeypatch.setenv("WOOF_HEX_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(builder.shutil, "which", lambda name: "/fixture/gpmetis")

    def metis(command, **kwargs):
        from pathlib import Path
        Path(f"{command[-2]}.part.2").write_text("".join(f"{cell * 2 // 12}\n" for cell in range(12)))
        return subprocess.CompletedProcess(command, 0, "fixture METIS partition\n", "")

    monkeypatch.setattr(builder.subprocess, "run", metis)
    arguments = SimpleNamespace(samehost_devices="0,1", samehost_layouts=None, samehost_part=None,
                                samehost_transport="auto", samehost_reference=None, samehost_fingerprint_every=0)
    request = SimpleNamespace(cells=12, grid=grid, scratch=tmp_path / "scratch")
    options = samehost.request_options(arguments, request)
    assert options["devices"] == (0, 1) and len(options["layouts"]) == 2
    assert str(options["layouts_path"]).startswith(str(tmp_path / "cache" / "samehost-layouts"))
    with pytest.raises(ValueError, match="need both --samehost-layouts and --samehost-part"):
        samehost.request_options(SimpleNamespace(**dict(vars(arguments), samehost_part=options["part_path"])),
                                 request)
