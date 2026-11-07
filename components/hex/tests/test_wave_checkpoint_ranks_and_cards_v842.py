"""Waves with same-host ranks across a checkpoint, and with local time stepping on several cards.

THE BREAKAGES THESE CHECK (hex 0.4.14):
1. A same-host checkpoint dropped the wave state (rank_carrier kept no 'waves'; the proxy session had no wave
   binding), so a same-host resume with --waves on refused ("wave mode differs between checkpoint and resumed
   forecast"), and ``DistributedMoving`` had no ``offset_seconds``, so a resumed run's wave clock started at 0.
2. ``for_mesh`` keyed one service per (output root, card), so the MPAS route's local time stepping on several cards
   (one process, a block of every rate class per card) built one whole-mesh service per card, each fed only its
   own card's cells.
No card is needed: NumPy stands in for CuPy, with a per-thread current card.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import pickle
import sys
import threading
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex import waves  # noqa: E402
from woof.hex.mpasphys import moving_mpi_v842 as mm  # noqa: E402

WAVES = dict(schema="gpuwm-hex.waves/v1", start_epoch=1680264000.0, clock=3600.0,
             solver=dict(arrays=dict(action=np.ones((4, 3, 3), np.float32))))
CARRIER = dict(physics=dict(fields=["a"], manager={}, manager_elapsed="3600"), waves=WAVES, mesh="source-mesh")


def test_rank_carrier_keeps_a_wave_marker_not_the_state():
    kept = mm.rank_carrier(CARRIER)
    assert kept["waves"]["start_epoch"] == WAVES["start_epoch"]   # old code: no 'waves' at all
    assert "solver" not in kept["waves"] and mm.is_wave_marker(kept["waves"])
    assert pickle.loads(pickle.dumps(mm.RankZeroOnly(WAVES))).value is None   # never sent to a rank process


def _moving(rank):
    route = SimpleNamespace(carrier=mm.rank_carrier(CARRIER), shock=True, control=None, frame_seconds=0.,
                            offset_seconds=3600., fixed_mesh=True, receipt=lambda: {})
    prepared = dict(route=route, control=None,
                    wave_state=mm.RankZeroOnly(dict(state=WAVES, weights=None, source_mesh="source-mesh")))
    comm = SimpleNamespace(Get_rank=lambda: rank)
    return mm.DistributedMoving(comm, prepared, [None, None], execution=None)


def test_rank0_resumes_the_full_wave_state_and_every_rank_its_clock_offset():
    zero, one = _moving(0), _moving(1)
    assert zero.carrier["waves"] is WAVES and zero.carrier["mesh"] == "source-mesh"
    assert mm.is_wave_marker(one.carrier["waves"])
    assert zero.offset_seconds == one.offset_seconds == 3600.   # old code: no attribute (runtime read 0)


class _Device:
    current = threading.local()

    def __init__(self, index=None):
        self.index = index

    @property
    def id(self):
        return getattr(_Device.current, "id", 0) if self.index is None else self.index

    def __enter__(self):
        stack = getattr(_Device.current, "stack", [])
        stack.append(getattr(_Device.current, "id", 0))
        _Device.current.stack, _Device.current.id = stack, self.index
        return self

    def __exit__(self, *exc):
        _Device.current.id = _Device.current.stack.pop()

    def use(self):
        _Device.current.id = self.index

    syncs = []

    def synchronize(self):
        _Device.syncs.append(self.index)


class _FakeCupy:
    """NumPy with a per-thread current card (cuda.Device)."""
    cuda = SimpleNamespace(Device=_Device)
    ndarray = np.ndarray

    def __getattr__(self, name):
        return getattr(np, name)


def test_lts_blocks_on_two_cards_bind_one_service(monkeypatch, tmp_path):
    cp = _FakeCupy()
    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setattr(waves, "configuration", lambda: {"enabled": True})
    monkeypatch.setattr(waves, "_SERVICES", {})
    import woof.hex.wave_prepared as wp
    monkeypatch.setattr(wp, "ensure", lambda **kw: SimpleNamespace())
    built = []

    class Service:
        def __init__(self, mesh, prepared, options, **kw):
            self.device, self.n, self.start_epoch, self.clock = cp.cuda.Device().id, 4, kw["start_epoch"], 0.
            self.bound = []
            built.append(self)

        def bind(self, **kw):
            self.bound.append((cp.cuda.Device().id, list(kw["global_cells"])))
            return SimpleNamespace()

        on_card = staticmethod(nullcontext)

        def output(self, clock):
            return None

    monkeypatch.setattr(waves, "HexWaveService", Service)
    mesh = SimpleNamespace(dimensions={"nCells": 4}, static_path="static.nc")
    owners = {0: np.array([0, 1]), 1: np.array([2, 3])}

    def block(card):
        with cp.cuda.Device(card):
            execution = SimpleNamespace(wave_input=lambda: (mesh, owners[card], 2), wave_service_card=0)
            waves.attach_mpas(SimpleNamespace(), mesh, SimpleNamespace(grid=None), start="2023-03-31_12:00:00",
                              dt=60., steps=3, out=tmp_path, execution=execution)

    threads = [threading.Thread(target=block, args=(card,)) for card in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(built) == 1 and built[0].device == 0   # old code: one service per card
    assert sorted(cells for _, cells in built[0].bound) == [[0, 1], [2, 3]]


def test_a_block_on_another_card_hands_its_words_on_the_service_card(monkeypatch):
    cp = _FakeCupy()
    service = object.__new__(waves.HexWaveService)
    service.cp, service.device, service.n = cp, 0, 4
    service.options = dict(coupling_interval_seconds=1.e9, roughness_feedback=True)
    service.clock, service.end_seconds, service.watermarks, service.bindings = 0., 1.e9, {}, []
    service.buffers = {k: np.zeros(4, np.float32) for k in ("u10", "v10", "psfc", "ustar", "tau_x", "tau_y",
                                                            "rho_air")}
    service.wall, service._checkpoint_cache = dict(forcing=0.), None
    service.z0 = np.arange(4, dtype=np.float32)
    ran_on = []

    def kernel(grid, block, args):
        ran_on.append(cp.cuda.Device().id)
        local, cells, u = args[0], args[1], args[2]
        service.buffers["u10"][cells] = u[local]

    service.forcing_kernel = kernel
    _Device.syncs.clear()
    with cp.cuda.Device(1):
        binding = service.bind(local=np.arange(2), global_cells=np.array([2, 3]),
                               all_global_cells=np.array([2, 3]), label="mpas")
        fields = {name: np.full(2, 7., np.float32) for name in ("u10", "v10", "psfc", "ust", "rho_air")}
        service.handoff(binding, fields, 60.)
        z0 = binding.roughness()
    assert binding.device == 1 and ran_on == [0]   # old code: no card of its own, the kernel ran on card 1
    assert service.buffers["u10"].tolist() == [0., 0., 7., 7.]
    assert z0.tolist() == [2., 3.]
    assert 0 in _Device.syncs   # the service card is synchronised around another card's work


def test_a_resume_in_its_own_directory_restores_the_same_prepared_words():
    """3. A wave checkpoint named its preparation by the manifest's raw bytes, which hold the run's own output path
    and wall times, so every resume into a new output directory refused ("prepared inputs ... differs")."""
    from woof.hex import wave_prepared as wp
    base = dict(schema="gpuwm.waves.prepared/v1", arrays=dict(depth=dict(path="depth.bin", sha256="aa")),
                grid=dict(ncell=4), initial_valid_time=0)
    straight = dict(base, request=dict(output="/runs/straight/waves-prepared/x"), stages=dict(wall_seconds=1.9),
                    parallelism=dict(threads=96))
    resume = dict(base, request=dict(output="/runs/resume/waves-prepared/x"), stages=dict(wall_seconds=0.8),
                  parallelism=dict(threads=96))
    other = dict(resume, arrays=dict(depth=dict(path="depth.bin", sha256="bb")))
    assert wp.content_sha256(straight) == wp.content_sha256(resume)
    assert wp.content_sha256(other) != wp.content_sha256(resume)

    service = object.__new__(waves.HexWaveService)
    service.options, service.start_epoch = {}, 0.
    service.prepared = SimpleNamespace(sha256="raw-resume", content_sha256=wp.content_sha256(resume),
                                       manifest={})
    calls = []
    service.solver = SimpleNamespace(restore_state=lambda payload: calls.append("restored"),
                                     checkpoint_arrays=lambda: {})
    state = dict(schema="gpuwm-hex.waves/v1", options={}, prepared_sha256="raw-straight",
                 prepared_content_sha256=wp.content_sha256(straight), start_epoch=0., station_mapping_sha256=None,
                 solver=dict(metadata={}, arrays={}))
    try:
        service.restore(state)
    except ValueError as error:   # old code refused here, before the solver was touched
        assert "prepared inputs" not in str(error), error
    except Exception:
        pass   # the stand-in stops after the identity check; the check is what this test is about
    assert calls == ["restored"]


def test_wave_maps_render_under_a_symlinked_output_path(tmp_path):
    """4. The wave map render handed NativeIO an unresolved folder; under a symlinked output path every map failed
    ("prepared arrays must be .bin files inside the prepared folder") after the forecast finished."""
    from woof.hex import wave_products
    real = tmp_path / "disk"
    (real / "waves").mkdir(parents=True)
    link = tmp_path / "scratch"
    link.symlink_to(real)
    work = wave_products.render_workdir(link)
    array = work / "grid" / "hs.bin"
    array.parent.mkdir()
    array.write_bytes(b"\0" * 4)
    root = array.parent
    assert array.resolve().is_relative_to(root)   # NativeIO.load's own check
