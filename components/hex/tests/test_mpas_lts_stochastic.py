"""U-H5: SPPT/SKEB with active local time stepping on the MPAS-physics route.

A WOOF storm roster runs SPPT/SKEB members on a refined moving mesh, and the moving controller always asks for
active full-step LTS (``moving_run.segment_command``).  The MPAS-route class executor refused stochastic forcing,
so every storm roster failed on the default route (run_392edb793fea0966 shape).  The executor now drives the
WOOF-ArWen route's class-aware generator (``LtsStochasticPhysics``): one process per member, drawn per macro by the
coordinator, each class forcing its own sources through ``LtsClassStochastic``.
"""
from argparse import Namespace
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from woof.hex.mpasphys.lts_v842 import LtsClassExecution, _Board, admission_refusal, stochastic_config
from woof.hex.stochastic import StochasticConfig, member_number, member_options

LAYOUT = SimpleNamespace(dimensions=dict(nCells=10, nCellsSolve=10), configs={})
CONFIG = SimpleNamespace(table_key="mpasphys_v842")


def _member_forcing():
    # The roster's own member options (moving_members.moving_roster), as the forecast door receives them.
    return json.dumps(member_options(member_number("member-001"), seed=53, source_members=1, ensemble_size=2))


def _args(**values):
    base = dict(fullstep_local_timestep=True, fullstep_lts_mode="active", samehost_devices=None,
                stochastic=_member_forcing(), physics_namelist=None)
    base.update(values)
    return Namespace(**base)


def test_storm_roster_member_forcing_is_admitted_on_one_card() -> None:
    assert stochastic_config(_args()).enabled
    assert admission_refusal(_args(), LAYOUT, CONFIG) is None
    assert admission_refusal(_args(samehost_devices="0"), LAYOUT, CONFIG) is None
    assert admission_refusal(_args(stochastic="off"), LAYOUT, CONFIG) is None


def test_several_cards_stay_refused_as_on_the_arwen_route() -> None:
    refusal = admission_refusal(_args(samehost_devices="0,1"), LAYOUT, CONFIG)
    assert refusal is not None and "several cards" in refusal
    # Without forcing, several cards run the classes cut into blocks (hex 0.4.6).
    assert admission_refusal(_args(samehost_devices="0,1", stochastic=None), LAYOUT, CONFIG) is None


class _Process:
    def __init__(self):
        self.restored = []

    def restore(self, value):
        self.restored.append(value)

    def snapshot(self, *, device=False):
        return dict(base=dict(time=0.0, draw=1))


class _Shared:
    def __init__(self, config):
        self.config = config
        self.process = _Process()
        self.applied = []

    def apply(self, index, held, **context):
        self.applied.append((index, context["time_seconds"], context["dt"]))
        return held

    def receipt(self):
        return dict(schema="gpuwm-hex.lts-stochastic/v1")


@pytest.fixture
def fake_cupy(monkeypatch):
    module = ModuleType("cupy")
    monkeypatch.setitem(sys.modules, "cupy", module)
    return module


def test_every_class_forces_through_the_member_process(fake_cupy) -> None:
    import threading
    config = StochasticConfig.parse(_member_forcing())
    shared = _Shared(config)
    lock = threading.Lock()
    board = _Board(2)
    classes = []
    for index, rate in enumerate((2, 1)):
        execution = LtsClassExecution(board, index, rate, 2, [SimpleNamespace()] * 2, None, None)
        execution.stochastic_shared, execution.stochastic_lock = shared, lock
        classes.append(execution.stochastic_physics(config))
    for index, view in enumerate(classes):
        assert view.apply("held", state=None, rho_edge=None, normal_velocity=None, time_seconds=900.0 * index,
                          dt=450.0, names=("qv",)) == "held"
    assert shared.applied == [(0, 0.0, 450.0), (1, 900.0, 450.0)]
    # A resumed moving segment: every class session restores the carried pattern; it is applied once.
    carried = dict(base=dict(time=1800.0, draw=5))
    for view in classes:
        view.process.restore(carried)
    assert shared.process.restored == [carried]
    with pytest.raises(ValueError, match="different stochastic patterns"):
        classes[1].process.restore(dict(base=dict(time=1800.0, draw=6)))
    # A class that resolved other options than the member's process is refused.
    other = StochasticConfig.parse(dict(json.loads(_member_forcing()), skeb=False))
    with pytest.raises(ValueError, match="needs the member process"):
        LtsClassExecution(board, 0, 2, 2, [SimpleNamespace()] * 2, None, None).stochastic_physics(config)
    execution = LtsClassExecution(board, 0, 2, 2, [SimpleNamespace()] * 2, None, None)
    execution.stochastic_shared, execution.stochastic_lock = shared, lock
    with pytest.raises(ValueError, match="another stochastic configuration"):
        execution.stochastic_physics(other)

