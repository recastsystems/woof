"""The learned correction gets the forecast route's step.

THE BREAKAGE THIS PREVENTS (0.6.2, 20 Nov 2026, a 3 h hrrrv5 run on x1.40962 from an installed wheel): with
WOOF_HEX_LEARNED_CORRECTION set, every run on the MPAS-physics route stopped at its first step with "the step length
is unknown (pass --dt)": the slot's state there carries no step and the door's --dt is unset; and ``--dt auto``
stopped at install on float('auto').  No card and no torch needed: the network is built at the first step.
"""
from __future__ import annotations

import inspect
from fractions import Fraction

import pytest

from woof.hex.ad import external_tendency
from woof.hex.ad import learned_correction as lc


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "net.pt"
    path.write_bytes(b"not read before the first step")
    monkeypatch.setenv(lc.WORD, str(path))
    yield path
    external_tendency.uninstall()


@pytest.mark.parametrize("selected", [None, "auto", "row", ""])
def test_a_door_that_names_no_step_installs_and_takes_the_routes(checkpoint, tmp_path, selected):
    callback = lc.install_from_env(tmp_path / "init.nc", selected)
    assert callback is not None and callback.dt_seconds is None
    assert lc.bind_step(Fraction(720)) is callback
    assert callback.dt_seconds == 720.0


def test_a_step_the_door_named_is_kept(checkpoint, tmp_path):
    callback = lc.install_from_env(tmp_path / "init.nc", "300")
    lc.bind_step(Fraction(720))
    assert callback.dt_seconds == 300.0


def test_bind_step_leaves_another_callback_alone(tmp_path):
    external_tendency.install(lambda k, state: {})
    try:
        assert lc.bind_step(600) is None
    finally:
        external_tendency.uninstall()


def test_the_mpas_route_binds_its_resolved_step():
    from woof.hex.mpasphys import forecast_v842
    source = inspect.getsource(forecast_v842)
    resolved = source.index("raise ValueError('native physics needs a positive config_dt')")
    assert "_learned_correction.bind_step(dt)" in source[resolved:resolved + 600]


def test_relative_paths_are_read_where_the_door_was_started(checkpoint, tmp_path, monkeypatch):
    """The route enters the run's native working folder before the first step: a relative --init and checkpoint are
    taken against the folder the door was started in (0.6.2: "No such file or directory")."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "init.nc").write_bytes(b"")
    callback = lc.LearnedCorrection("net.pt", "init.nc")
    elsewhere = tmp_path / "run"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert callback.init_path == (tmp_path / "init.nc").resolve() and callback.init_path.is_file()
    assert callback.checkpoint == (tmp_path / "net.pt").resolve() and callback.checkpoint.is_file()
