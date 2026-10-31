"""Local time stepping on the MPAS suite route: full-step LTS runs (active and shadow); what is not carried is
refused by name instead of ignored (HEX-PHYSICS-PARITY run group 2 #8)."""
from argparse import Namespace
from types import SimpleNamespace

from woof.hex.forecast_door import mpas_suite_lts_refusal
from woof.hex.mpasphys.lts_v842 import admission_refusal, lts_requested, shadow_requested


def _args(**values):
    base = dict(fullstep_local_timestep=False, local_timestep=False, fullstep_lts_mode=None,
                fullstep_lts_max_rate=None, fullstep_lts_buffer_rings=None, samehost_devices=None)
    base.update(values)
    return Namespace(**base)


def test_full_step_lts_is_admitted_on_the_mpas_suite_route() -> None:
    for mode in (None, "shadow", "active"):
        assert mpas_suite_lts_refusal(_args(fullstep_local_timestep=True, fullstep_lts_mode=mode), environ={}) is None
    assert lts_requested(_args(fullstep_local_timestep=True, fullstep_lts_mode="active"))
    assert not lts_requested(_args(fullstep_local_timestep=True))
    assert shadow_requested(_args(fullstep_local_timestep=True))
    assert mpas_suite_lts_refusal(_args(), environ={}) is None


def test_what_is_not_carried_is_refused_by_name() -> None:
    message = mpas_suite_lts_refusal(_args(local_timestep=True), environ={})
    assert message is not None and message.startswith("--local-timestep is the acoustic rate ladder")
    message = mpas_suite_lts_refusal(_args(fullstep_lts_mode="active"), environ={})
    assert message is not None and "LTS-only knobs need --fullstep-local-timestep" in message
    # Active LTS on several cards runs the classes cut into blocks (hex 0.4.6); the shadow stays on one device.
    assert mpas_suite_lts_refusal(_args(fullstep_local_timestep=True, fullstep_lts_mode="active",
                                        samehost_devices="0,1"), environ={}) is None
    assert admission_refusal(_args(fullstep_local_timestep=True, fullstep_lts_mode="active", samehost_devices="0,1"),
                             SimpleNamespace(dimensions=dict(nCells=10, nCellsSolve=10), configs={}),
                             SimpleNamespace(table_key="mpasphys_v842")) is None
    message = mpas_suite_lts_refusal(_args(fullstep_local_timestep=True, samehost_devices="0,1"), environ={})
    assert message is not None and "shadow runs on one device" in message
    # A moving segment carries the controller's own LTS flags; the moving-route admission judges it.
    assert mpas_suite_lts_refusal(_args(fullstep_local_timestep=True, local_timestep=True),
                                  environ={"WOOF_HEX_MOVING_CONTROL": "/x/control.json"}) is None


def test_class_executor_admission_names_each_gap() -> None:
    layout = SimpleNamespace(dimensions=dict(nCells=10, nCellsSolve=10), configs={})
    config = SimpleNamespace(table_key="mpasphys_v842")
    assert admission_refusal(_args(fullstep_local_timestep=True, fullstep_lts_mode="active"), layout, config) is None
    # --restart-from runs on the classes (a checkpoint seals every class and the coordinator's interface).
    assert admission_refusal(_args(restart_from="x.pkl"), layout, config) is None
    for name, value in (("da_analysis", "a.nc"), ("iau_window_seconds", 3600.0), ("balance_series", True),
                        ("chemistry_config", "c.json"), ("lbc_dir", "lbc")):
        refusal = admission_refusal(_args(**{name: value}), layout, config)
        assert refusal is not None, name
    assert "config_sst_update" in admission_refusal(_args(), SimpleNamespace(
        dimensions=dict(nCells=10, nCellsSolve=10), configs=dict(config_sst_update=True)), config)
    assert "limited-area" in admission_refusal(_args(), SimpleNamespace(
        dimensions=dict(nCells=12, nCellsSolve=10), configs={}), config)


def test_correct_mode_class_executor_is_admitted() -> None:
    """F-LTS-F000 (the f000 first call from every class) and F-LTS-RACE (the physics edge projection read past
    its tendency arrays on the classes' outermost halo ring) are fixed, so the refusal that cited them is gone:
    correct mode, the door default, runs the class executor as exact mode does."""
    layout = SimpleNamespace(dimensions=dict(nCells=10, nCellsSolve=10), configs={})
    active = _args(fullstep_local_timestep=True, fullstep_lts_mode="active")
    for mode in ("correct", "exact"):
        assert admission_refusal(active, layout, SimpleNamespace(table_key="mpasphys_v842", mode=mode)) is None
