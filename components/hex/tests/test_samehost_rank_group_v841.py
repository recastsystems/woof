"""The ArWen-physics route's rank executor: one process per card by default, rank threads on request.

THE BREAKAGE (woof.hex.drivers.samehost_rank_group_v841): the ArWen-physics route ran every same-host rank as a
thread of the forecast process, so on a Python with the GIL its multi-card runs serialised their host work (only a
stderr line said so).  These CPU checks pin the executor choice, the module-state hand-over a rank process relies
on, and the limited-area exchange hook every rank needs; the words themselves are proven on cards (threads against
processes at the same N).
"""
from __future__ import annotations

import sys
import types
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from woof.hex.drivers import samehost_rank_group_v841 as group


def test_auto_runs_one_process_per_card_where_the_host_supports_it(monkeypatch):
    import woof.hex.rank_processes as rank_processes

    monkeypatch.delenv("WOOF_HEX_SAMEHOST_RANKS", raising=False)
    monkeypatch.setattr(rank_processes, "supported", lambda: (True, "linux"))
    assert group.executor_choice(None, (0, 1)) == ("processes", "default (one process per card)")
    assert group.executor_choice("auto", (0, 1, 2, 3))[0] == "processes"
    assert group.executor_choice("threads", (0, 1)) == ("threads", "requested")
    monkeypatch.setenv("WOOF_HEX_SAMEHOST_RANKS", "threads")
    assert group.executor_choice(None, (0, 1)) == ("threads", "requested")


def test_one_rank_and_unsupported_hosts_keep_threads_and_say_why(monkeypatch, capsys):
    import woof.hex.rank_processes as rank_processes

    monkeypatch.delenv("WOOF_HEX_SAMEHOST_RANKS", raising=False)
    monkeypatch.setattr(rank_processes, "supported", lambda: (False, "no memfd here"))
    assert group.executor_choice(None, (0,)) == ("threads", "one rank")
    assert group.executor_choice(None, (0, 1)) == ("threads", "no memfd here")
    assert "no memfd here" in capsys.readouterr().err
    with pytest.raises(ValueError, match="not available here: no memfd here"):
        group.executor_choice("processes", (0, 1))
    with pytest.raises(ValueError, match="unknown same-host rank executor"):
        group.executor_choice("mpi", (0, 1))


def test_door_option_reaches_the_arwen_route_options():
    from woof.hex.drivers import samehost_forecast_v841 as samehost
    import inspect

    assert "rank_executor" in inspect.signature(samehost.installed).parameters
    assert "door_replay" in inspect.signature(samehost.installed).parameters
    assert "rank_executor" in inspect.signature(samehost._execute_samehost).parameters
    source = inspect.getsource(samehost.request_options)
    assert '"rank_executor": getattr(arguments, "samehost_ranks", None)' in source


def _fake_rank(rank):
    adjusted = []

    class Regional:
        def bdy_adjust_scalars(self, scalars, rk_step):
            adjusted.append((rank, rk_step))

    driver = SimpleNamespace(regional_v841=Regional())
    ctx = group.RankContext(rank=rank, device=rank, card=rank, proof=None, forecast=None, host={}, schedule={},
                            output_root=".", layouts=[None, None, None], tables=[object(), object(), object()],
                            stream=None, physics_scope=None, cp=None)
    ctx.stack = {"driver": driver, "interior_edge_mask": np.ones(4, bool)}
    ctx.transport = SimpleNamespace(abort=lambda reason: None)
    return ctx


def test_every_limited_area_rank_exchanges_scalars_before_its_boundary_adjust():
    # THE DEFECT: the hook sat after the exchanger loop inside the IAU block, so it reached the last rank only and
    # only with IAU; every other limited-area rank issued fewer D rounds than the regional round law requires.
    contexts = [_fake_rank(rank) for rank in range(3)]
    threads = group.ThreadRankGroup(contexts, (0, 1, 2), cp=None, physics_scope=None, streams=None)
    threads.local_call("setup_exchange", physics_rounds=("D", "A"), regional=True)
    for ctx in contexts:
        adjust = ctx.stack["driver"].regional_v841.bdy_adjust_scalars
        assert getattr(adjust, "_samehost_exchange_first", False), f"rank {ctx.rank} has no scalar exchange"
        assert ctx.stack["driver"].halo_exchanger_v841 is ctx.exchanger
        assert ctx.exchanger.interior_edge_mask is ctx.stack["interior_edge_mask"]


def test_global_ranks_keep_their_boundary_adjust_untouched():
    contexts = [_fake_rank(rank) for rank in range(2)]
    threads = group.ThreadRankGroup(contexts, (0, 1), cp=None, physics_scope=None, streams=None)
    threads.local_call("setup_exchange", physics_rounds=("D", "A"), regional=False)
    for ctx in contexts:
        assert not getattr(ctx.stack["driver"].regional_v841.bdy_adjust_scalars, "_samehost_exchange_first", False)


def test_rank_process_takes_the_forecast_process_module_constants(monkeypatch):
    name = "woof.hex._rank_group_probe"
    parent = types.ModuleType(name)
    parent.N_SOIL_LEVELS = 9
    parent.PIN = MappingProxyType({"shape": (5, 3), "sha": "abc"})
    parent.LEVELS = np.arange(4, dtype=np.float32)
    parent._PREFLIGHTED = {0, 1}          # a private cache: never handed over
    parent.KERNEL = lambda: None          # not data
    monkeypatch.setitem(sys.modules, name, parent)
    constants = group._module_constants()[name]
    assert set(constants) == {"N_SOIL_LEVELS", "PIN", "LEVELS"}

    child = types.ModuleType(name)
    child.N_SOIL_LEVELS = 4
    child.PIN = MappingProxyType({"shape": (5, 3), "sha": "abc"})
    child.LEVELS = np.zeros(4, dtype=np.float32)
    child._PREFLIGHTED = set()
    monkeypatch.setitem(sys.modules, name, child)
    record = group._apply_module_constants({name: constants})
    assert child.N_SOIL_LEVELS == 9
    assert np.array_equal(child.LEVELS, np.arange(4, dtype=np.float32))
    assert record["constants_set"] == 2          # the equal pin is left as it is
    assert child._PREFLIGHTED == set()


def test_module_constants_survive_the_rank_link_pickle():
    from woof.hex.rank_processes import _dumps
    import pickle

    value = {"hexcore.x": {"PIN": MappingProxyType({"a": MappingProxyType({"b": 1})}), "N": 3}}
    buffers = []
    data = _dumps(value, buffers)
    back = pickle.loads(data, buffers=buffers)
    assert isinstance(back["hexcore.x"]["PIN"], MappingProxyType)
    assert back["hexcore.x"]["PIN"]["a"]["b"] == 1


def test_a_rank_process_frame_reads_like_a_pending_snapshot():
    arrays = {"theta": np.ones(3, np.float32)}
    assert group._Ready(arrays).result() is arrays


def test_rank0_failure_reaps_its_waiting_child_before_releasing_halo_flags():
    """A real hour-17 OOM waited forever for a child already waiting on rank 0's next GPU halo round."""
    import socket
    import subprocess
    import threading
    from woof.hex.rank_processes import Link, RankProcesses

    ranks = RankProcesses((0, 0), {0: [1], 1: [0]}, module="unused", timeout=60, job="unused")
    ranks.rank0_grace = None
    local, remote = socket.socketpair()
    ranks.step_links[1] = Link(local, timeout=60, name="step-0-1")
    child = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"],
                             pass_fds=(remote.fileno(),))
    ranks.processes[1] = child
    remote.close()
    unrelated = subprocess.Popen([sys.executable, "-c", "import signal; signal.pause()"])
    executor = group.ProcessRankGroup.__new__(group.ProcessRankGroup)
    executor.group, executor.size = ranks, 2
    executor.failed, executor.failure = False, None
    executor.sync_seconds, executor.sync_calls = [0., 0.], [0, 0]
    released, errors = [], []
    executor.ctx = SimpleNamespace(transport=SimpleNamespace(
        abort=lambda reason: released.append((reason, child.poll(), unrelated.poll()))))
    original = MemoryError("rank-zero allocation failed")

    def fail(*args):
        raise original

    executor._local = fail

    def call():
        try:
            executor.call("advance")
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=call, daemon=True)
    try:
        thread.start()
        thread.join(5)
        assert not thread.is_alive(), "rank zero waited for its child instead of stopping it"
        assert errors == [original]
        assert released == [("rank 0: MemoryError: rank-zero allocation failed", -9, None)]
        assert ranks.failure == released[0][0]
        assert ranks._rank0_responded.is_set()
        assert executor.failed
    finally:
        ranks.kill()
        child.wait(5)
        ranks.close()
        unrelated.kill()
        unrelated.wait(5)
        thread.join(5)


def test_iau_increment_is_formed_once_and_any_rank_set_built_from_it(monkeypatch):
    import woof.hex.iau_native as iau_native

    built = []

    class Forcing:
        def __init__(self, data, layouts, **kwargs):
            built.append((data, tuple(layouts), kwargs))

    monkeypatch.setattr(iau_native, "NativeIauForcing", Forcing)
    source = {"route": "native", "window": 3600.0, "window_start": 0.0, "mode": "exact",
              "scalar_names": ("qv",), "data": {"u": 1}, "receipt": {"route": "native"}, "host_seconds": 1.5}
    forcing, receipt = iau_native.iau_forcing_from_source(source, layouts=["rank3"], cp=None, devices=[0],
                                                          cell_widths=[10])
    assert built == [({"u": 1}, ("rank3",), {"window_seconds": 3600.0, "mode": "exact", "scalar_names": ("qv",),
                                             "cp": None, "window_start": 0.0})]
    assert receipt["route"] == "native" and receipt["prepare_seconds"] >= 1.5
    assert source["receipt"] == {"route": "native"}   # each rank's receipt is its own copy
