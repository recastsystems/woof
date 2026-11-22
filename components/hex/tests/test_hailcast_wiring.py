"""woof.hex.hailcast's wiring into the MPAS-physics output diagnostics, on the CPU (the card work is stubbed).

THE BREAKAGE THIS GUARDS: the hail fields must reach the diagnostics stream of every MPAS-physics run with a 3 km zone
(19 Nov 2026: WOOF-ArWen's HAILCAST in hex), restart their window only after a whole-hour frame has taken
them, and stay off on a coarse mesh and under WOOF_HEX_HAILCAST=0.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex import hailcast


class _Stub:
    def __init__(self, cp, **kw):
        self.kw = kw
        self.n_fine = int(np.asarray(kw["fine"]).sum())
        self.n_out = kw["n_out"]
        self.error = None
        self.updates = []
        self.resets = 0
        self.psd = kw["psd"]

    def update(self, arrays, *, seconds, dt):
        self.updates.append((sorted(arrays), seconds, dt))

    def fields(self):
        return {"HAILCAST_DIAM_MAX": (np.zeros(self.n_out, np.float32), ("nCells",), "mm")}

    def reset_window(self):
        self.resets += 1

    def fail(self, where, error):
        raise AssertionError(f"{where}: {error}")

    def receipt(self):
        return {"fine_cells": self.n_fine, "psd": self.psd}

    def carry(self):
        return {hailcast.CARRY + "dur": np.full(3, 7.0)}

    def resume(self, words):
        self.resumed = sorted(words)
        return ["dur"]


class _Pool(dict):
    def __contains__(self, name):
        return dict.__contains__(self, name)


def _diagnostics(spacing_m, *, tempo=True, scheme="mp_tempo"):
    n, nl = len(spacing_m), 4
    mesh = _Pool(cellsOnCell=np.full((6, n), n + 1), nEdgesOnCell=np.full(n, 6),
                 edgesOnCell=np.tile(np.arange(n)[None, :] + 1, (6, 1)), dcEdge=np.asarray(spacing_m, np.float64))
    physics = _Pool(max_hail_diameter_sfc=1, max_hail_diameter_column=1) if tempo else _Pool()
    layout = SimpleNamespace(dimensions=dict(nCellsSolve=n, nCells=n, nVertLevels=nl),
                             indices={("state", "index_qv"): 1, ("state", "index_qg"): 2},
                             configs={"config_microp_scheme": scheme})
    fields = {"scalars": np.zeros((2, nl, n), np.float32)}

    def authority(pool, name):
        return fields.get(name, np.zeros((nl + 1, n), np.float32))

    calls = []
    d = SimpleNamespace(cp=SimpleNamespace(asnumpy=np.asarray), session=SimpleNamespace(layout=layout),
                        pools={"mesh": mesh, "diag_physics": physics}, authority=authority,
                        dycore=SimpleNamespace(config=SimpleNamespace(config_dt=20.0)),
                        _update_convective=lambda: calls.append("convective"),
                        after_snapshot=lambda: calls.append("snapshot"), when=None, accumulates=False)
    return d, calls


def _attach(monkeypatch, spacing, **kw):
    monkeypatch.setattr(hailcast, "Hailcast", _Stub)
    monkeypatch.setattr(hailcast, "check_shared", lambda: None)
    d, calls = _diagnostics(spacing, **kw)
    output = SimpleNamespace(streams={})
    row = hailcast.attach_mpas(d, output)
    return d, calls, output, row


def test_arms_on_the_fine_zone_and_writes_through_the_diagnostics_stream(monkeypatch):
    monkeypatch.delenv(hailcast.ENV, raising=False)
    d, calls, output, row = _attach(monkeypatch, [3000., 3000., 15000.])
    assert row["enabled"] and row["fine_cells"] == 2 and row["psd"] == "tempo"
    assert d.accumulates
    d.when = datetime(2024, 5, 21, 19, 0, 20)
    d._update_convective()
    names, seconds, dt = d.hailcast.updates[-1]
    assert calls == ["convective"] and dt == 20.0
    assert seconds == (datetime(2024, 5, 21, 19, 0, 20) - datetime(1970, 1, 1)).total_seconds()
    assert {"theta_m", "exner", "pressure_base", "pressure_p", "rho_zz", "zz", "zgrid", "w", "qv", "qg",
            "max_hail_diameter_sfc", "max_hail_diameter_column"} <= set(names)
    # A 15-minute frame keeps the hour's window; the whole-hour frame restarts it after the snapshot.
    d.when = datetime(2024, 5, 21, 19, 15)
    assert "HAILCAST_DIAM_MAX" in output.extra_diagnostics()
    d.after_snapshot()
    assert d.hailcast.resets == 0
    d.when = datetime(2024, 5, 21, 20, 0)
    output.extra_diagnostics()
    d.after_snapshot()
    assert d.hailcast.resets == 1 and calls[-1] == "snapshot"


def test_thompson_route_uses_gpuwm_hail_psd(monkeypatch):
    monkeypatch.delenv(hailcast.ENV, raising=False)
    d, calls, output, row = _attach(monkeypatch, [3000.], tempo=False, scheme="mp_thompson")
    assert row["psd"] == "arwen" and d.hailcast.kw["psd_row"]["thompson"] == 1


def test_coarse_mesh_and_off_word_build_nothing(monkeypatch):
    monkeypatch.delenv(hailcast.ENV, raising=False)
    d, calls, output, row = _attach(monkeypatch, [15000., 30000.])
    assert not row["enabled"] and not hasattr(output, "extra_diagnostics")
    monkeypatch.setenv(hailcast.ENV, "on")
    d, calls, output, row = _attach(monkeypatch, [15000., 30000.])
    assert row["enabled"] and row["fine_cells"] == 2
    monkeypatch.setenv(hailcast.ENV, "0")
    d, calls, output, row = _attach(monkeypatch, [3000.])
    assert row is None


class _Comm:
    """Two ranks' collectives played one rank at a time: rank 1 calls first, rank 0 last."""

    def __init__(self, others=()):
        self.others, self.sent = list(others), {}

    def allgather(self, value):
        return [value, *self.others]

    def gather(self, value, root=0):
        self.sent[len(self.sent)] = value
        return None if len(self.sent) == 1 else [self.sent[1], self.sent[0]]


def test_frames_gather_every_ranks_owned_cells_to_rank_0():
    """Rank 0 writes the global frame from each rank's owned words (only the cells holding hail travel)."""
    comm = _Comm()
    # 6 global cells; rank 1 owns global 3, 4, 5 (local 0-2) and holds 2 as halo; rank 0 owns 0, 1, 2.
    one = {"HAILCAST_DIAM_MAX": (np.array([0., 31., 0., 99.], np.float32), ("nCells",), "mm"),
           "HAIL_MAXK1": (np.array([0., 0.02, 0.01, 0.5], np.float32), ("nCells",), "m")}
    assert hailcast.gather_to_rank0(comm, 1, [3, 4, 5, 2], 3, 6, one) == {}
    zero = {"HAILCAST_DIAM_MAX": (np.array([12., 0., 0., 55.], np.float32), ("nCells",), "mm"),
            "HAIL_MAXK1": (np.array([0.004, 0., 0., 0.3], np.float32), ("nCells",), "m")}
    got = hailcast.gather_to_rank0(comm, 0, [0, 1, 2, 3], 3, 6, zero)
    np.testing.assert_array_equal(got["HAILCAST_DIAM_MAX"][0], [12., 0., 0., 0., 31., 0.])
    np.testing.assert_allclose(got["HAIL_MAXK1"][0], [0.004, 0., 0., 0., 0.02, 0.01])
    assert got["HAIL_MAXK1"][1:] == (("nCells",), "m")
    assert comm.sent[0][2].tolist() == [4, 5] and comm.sent[1][2].tolist() == [0]   # zeros and halo words never travel


def test_a_rank_with_no_words_writes_none_of_the_frame():
    comm = _Comm()
    assert hailcast.gather_to_rank0(comm, 1, [3, 4, 5], 3, 6, {}) == {}
    words = {"HAILCAST_DIAM_MAX": (np.ones(3, np.float32), ("nCells",), "mm")}
    assert hailcast.gather_to_rank0(comm, 0, [0, 1, 2], 3, 6, words) == {}


def _rank_halo(comm, owned):
    table = SimpleNamespace(n_local_cells=owned + 2)
    execution = SimpleNamespace(comm=comm, rank=0, tables=[table], partition=SimpleNamespace(
        n_owned_cells=owned, cell_l2g=np.arange(owned + 2)), global_layout=SimpleNamespace(dimensions={"nCells": 9}))
    return SimpleNamespace(execution=execution, exchange=lambda label, fields: None)


def test_multi_rank_arms_every_rank_or_none(monkeypatch, capsys):
    """THE BREAKAGE THIS GUARDS: one rank armed and another not would wait for each other's halo exchange and
    frame gather; the ranks agree before arming."""
    monkeypatch.delenv(hailcast.ENV, raising=False)
    monkeypatch.setattr(hailcast, "Hailcast", _Stub)
    monkeypatch.setattr(hailcast, "check_shared", lambda: None)
    d, _ = _diagnostics([3000., 3000., 3000.])
    other_failed = (False, False, {"enabled": False, "error": "RuntimeError: no kernel"})
    row = hailcast.attach_mpas(d, SimpleNamespace(streams={}), halo=_rank_halo(_Comm([other_failed]), 3))
    assert row["enabled"] is False and "no kernel" in row["error"]
    assert "not armed on any rank" in capsys.readouterr().err
    d, _ = _diagnostics([3000., 3000., 3000.])
    output = SimpleNamespace(streams={})
    row = hailcast.attach_mpas(d, output, halo=_rank_halo(_Comm([(True, False, None)]), 3))
    assert row["enabled"] and d.hailcast.kw["n_ext"] == 5 and d.hailcast.kw["halo"] is not None
    assert "on rank 0 (2 halo cells exchanged" in capsys.readouterr().err
    # A rank with no fine cell still arms when another has them (it has nothing to count, but takes part).
    d, _ = _diagnostics([15000., 15000., 15000.])
    row = hailcast.attach_mpas(d, SimpleNamespace(streams={}), halo=_rank_halo(_Comm([(True, True, None)]), 3))
    assert row["enabled"] and row["fine_cells"] == 0


def test_the_seal_carries_the_count_and_a_resume_takes_it(monkeypatch, capsys):
    """THE BREAKAGE THIS FIXES: a run resumed from a seal started its updraft count at zero (no stone for 15
    minutes); hailcast's words now ride the convective accumulators' continuation, and only its own words reach it."""
    monkeypatch.delenv(hailcast.ENV, raising=False)
    monkeypatch.setattr(hailcast, "Hailcast", _Stub)
    monkeypatch.setattr(hailcast, "check_shared", lambda: None)
    d, _ = _diagnostics([3000., 3000., 3000.])
    resumed = []
    d.accumulator_state = lambda: {"updraft_helicity_max": np.ones(3)}
    d.resume_accumulators = lambda words: resumed.append(sorted(words))
    hailcast.attach_mpas(d, SimpleNamespace(streams={}))
    words = d.accumulator_state()
    assert sorted(words) == [hailcast.CARRY + "dur", "updraft_helicity_max"]
    d.resume_accumulators(words)
    assert resumed == [["updraft_helicity_max"]] and d.hailcast.resumed == [hailcast.CARRY + "dur"]
    assert "hailcast: resumed from the seal (dur)" in capsys.readouterr().err
