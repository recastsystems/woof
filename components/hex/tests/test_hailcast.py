"""HAILCAST on hex (woof.hex.hailcast): WOOF-ArWen's shared stone model, matched to WRF's and HRRR's Fortran.

THE BREAKAGE THIS GUARDS: hex wrote no HAILCAST hail size at all (19 Nov 2026: WOOF-ArWen had ported
HAILCAST and tested it against HRRR's pinned version); WOOF-ArWen's GPU HAILCAST matched WRF 4.6.1's and operational HRRR
v4.1.21's Fortran on real storm columns (woof fix/hailcast).  These tests hold hex to the same Fortran on the same
94 columns (tests/data/oracles/hailcast/real_columns.npz, copied from woof rel/3.0.4 unchanged), first through
woof's kernel as hex loads it, then through hex's own MPAS-state route into that kernel.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from woof.hex import hailcast

ORACLE = Path(__file__).parent / "data/oracles/hailcast/real_columns.npz"


def test_words_and_defaults(monkeypatch):
    monkeypatch.delenv(hailcast.ENV, raising=False)
    monkeypatch.delenv(hailcast.ENV_VERSION, raising=False)
    assert hailcast.mode() == "auto" and hailcast.version() == "wrf_39_noaa"
    for word, want in (("0", "off"), ("off", "off"), ("1", "on"), ("auto", "auto")):
        monkeypatch.setenv(hailcast.ENV, word)
        assert hailcast.mode() == want
    monkeypatch.setenv(hailcast.ENV_VERSION, "wrf_461")
    assert hailcast.version() == "wrf_461"
    monkeypatch.setenv(hailcast.ENV_VERSION, "wrf_99")
    with pytest.raises(ValueError, match="wrf_39_noaa"):
        hailcast.version()


def test_installed_gpuwm_carries_the_proven_stone_model():
    hailcast.check_shared()
    assert hailcast.shared_sources() == hailcast.SHARED_SHA256


def test_oracle_is_gpuwms_bytes():
    # woof rel/3.0.4 tests/data/oracles/hailcast/real_columns.npz
    assert hashlib.sha256(ORACLE.read_bytes()).hexdigest()[:16] == "84084cec4cb5f80c"


def test_fine_cells_by_edge_spacing():
    dc = np.array([3000., 3000., 3000., 12000., 12000., 12000.])
    eoc = np.array([[0, 1, 2, -1], [0, 1, 2, -1], [3, 4, 5, -1]])
    spacing = hailcast.cell_spacing(dc, eoc, np.array([3, 2, 3]), 3)
    np.testing.assert_allclose(spacing, [3000., 3000., 12000.])
    assert list(spacing <= hailcast.FINE_M) == [True, True, False]


def test_window_resets_on_whole_hours_only():
    assert hailcast._whole_hour(datetime(2024, 5, 21, 20, 0, 0))
    assert not hailcast._whole_hour(datetime(2024, 5, 21, 20, 15, 0))
    assert not hailcast._whole_hour(None)


def _fixture():
    fx = np.load(ORACLE)
    prof = fx["profiles"].astype(np.float32)        # (11, column, level)
    return fx, prof, prof.shape[1], prof.shape[2]


@pytest.mark.parametrize("version", ["wrf_461", "wrf_39_noaa"])
def test_shared_kernel_matches_fortran_on_real_storm_columns(version):
    """woof's own column kernel, loaded the way hex loads it, against each generation's gfortran build."""
    import cupy as cp
    fx, prof, n, nz = _fixture()
    profiles = [cp.asarray(np.ascontiguousarray(prof[v].T)) for v in range(11)]
    outputs = [cp.zeros(n, dtype=cp.float32) for _ in range(9)]
    hailcast.gpuwm_kernel(hailcast.KERNELS[version])(
        ((n + 31) // 32,), (32,), (*profiles, cp.asarray(fx["ht"]), cp.asarray(fx["wdur"]),
                                    *outputs, np.int32(n), np.int32(1), np.int32(nz)))
    actual = np.stack([o.get() for o in outputs[:5]], 1)
    np.testing.assert_allclose(actual, fx[version], rtol=.005, atol=.1)
    assert np.array_equal(actual.max(1) > 0, fx[version].max(1) > 0)


@pytest.mark.parametrize("version", ["wrf_461", "wrf_39_noaa"])
def test_hex_route_matches_fortran_on_real_storm_columns(version):
    """The same columns as MPAS state (theta_m, exner, pressure_base/_p, rho_zz, zz, zgrid, scalars, w faces)
    through hex's gather into the shared kernel: the five embryo stones against the Fortran's."""
    import cupy as cp
    fx, prof, n, nz = _fixture()
    t, z, p, rho, qv, qi, qc, qr, qs, qg, w = (prof[v].T.copy() for v in range(11))   # (level, column)
    rvord = np.float32(np.float32(461.6) / np.float32(287.0))
    ht = np.asarray(fx["ht"], np.float32)
    zgrid = np.zeros((nz + 1, n), np.float32)
    zgrid[0] = ht
    for k in range(nz):
        zgrid[k + 1] = np.float32(2) * z[k] - zgrid[k]
    wfaces = np.zeros((nz + 1, n), np.float32)
    wfaces[:nz] = w
    arrays = {"theta_m": t * (np.float32(1) + rvord * qv), "exner": np.ones_like(t), "pressure_base": p,
              "pressure_p": np.zeros_like(p), "rho_zz": rho / (np.float32(1) + qv), "zz": np.ones_like(t),
              "zgrid": zgrid, "w": wfaces, "qv": qv, "qi": qi, "qc": qc, "qr": qr, "qs": qs, "qg": qg}
    arrays = {k: cp.asarray(np.ascontiguousarray(v, dtype=np.float32)) for k, v in arrays.items()}
    hail = hailcast.Hailcast(cp, n_cells=n, n_levels=nz, cells_on_cell=np.zeros((n, 1), np.int64),
                             n_edges_on_cell=np.zeros(n, np.int32), fine=np.ones(n, bool), stone_model=version)
    hail.dur[...] = cp.asarray(np.asarray(fx["wdur"], np.float32))
    hail._integrate(arrays, stones=True)
    assert hail.error is None
    window = hail.window.get()
    actual = window[3:8].T                              # HAILCAST_DHAIL1..5
    want = fx[version]
    np.testing.assert_allclose(actual, want, rtol=.005, atol=.1)
    assert np.array_equal(actual.max(1) > 0, want.max(1) > 0)
    np.testing.assert_allclose(window[0], want.max(1), rtol=.005, atol=.1)   # HAILCAST_DIAM_MAX
    landed = (want.max(1) > 0).sum()
    assert (landed == n) if version == "wrf_39_noaa" else (0 < landed < n // 2)


def test_duration_counts_a_persistent_neighbour_updraft():
    """WRF's rule on the hexagonal mesh: a cell's count grows while it or a neighbour held 10 m/s at the last
    step, and the stone model runs once the count passes 900 s, every haildt of model time."""
    import cupy as cp
    n, nl = 7, 3
    # cell 0 in the middle, cells 1-6 around it; each outer cell sees cell 0 and two outer neighbours.
    coc = np.full((n, 6), -1, np.int64)
    coc[0] = [1, 2, 3, 4, 5, 6]
    for c in range(1, 7):
        coc[c, :3] = [0, 1 + c % 6, 1 + (c - 2) % 6]
    nedges = np.array([6, 3, 3, 3, 3, 3, 3], np.int32)
    # As the pools hand it over: the native (maxEdges, nCells) table, transposed (a Fortran-ordered view).
    native = np.ascontiguousarray(coc.T)
    hail = hailcast.Hailcast(cp, n_cells=n, n_levels=nl, cells_on_cell=native.T, n_edges_on_cell=nedges,
                             fine=np.ones(n, bool), stone_model="wrf_39_noaa")
    assert hail.coc.flags.c_contiguous and np.array_equal(hail.coc.get(), np.where(coc >= 0, coc, -1))
    w = np.zeros((nl + 1, n), np.float32)
    w[1, 0] = 12.0                                         # one strong updraft in the middle cell
    t = np.full((nl, n), 250., np.float32)
    arrays = {"w": cp.asarray(w), "theta_m": cp.asarray(t), "exner": cp.ones((nl, n), cp.float32),
              "pressure_base": cp.full((nl, n), 5.e4, cp.float32), "pressure_p": cp.zeros((nl, n), cp.float32),
              "rho_zz": cp.ones((nl, n), cp.float32), "zz": cp.ones((nl, n), cp.float32),
              "zgrid": cp.asarray(np.repeat(np.arange(nl + 1, dtype=np.float32)[:, None] * 1000., n, 1))}
    dt, start = 20.0, 1716292800.0                         # 2024-05-21 12Z
    for step in range(1, 51):
        hail.update(arrays, seconds=start + step * dt, dt=dt)
    assert hail.error is None
    dur = hail.dur.get()
    # From step 2 every cell's neighbourhood held the updraft at the step before: one count, as WRF's 3x3.
    assert np.all(dur == 50 * dt)
    assert hail.integrations >= 1 and hail.columns >= n     # counts past 900 s reached the stone model (960 s)
    w[1, 0] = 0.0
    arrays["w"] = cp.asarray(w)
    hail.update(arrays, seconds=start + 51 * dt, dt=dt)    # the middle's own mask ends; the old mask still counts
    assert np.all(hail.dur.get() == 51 * dt)
    hail.update(arrays, seconds=start + 52 * dt, dt=dt)
    assert np.all(hail.dur.get() == 0)


def _torus(ny, nx):
    """A periodic triangular lattice: every cell has six neighbours, as an interior hexagon of the MPAS mesh."""
    n = ny * nx
    coc = np.zeros((n, 6), np.int64)
    for y in range(ny):
        for x in range(nx):
            shift = y % 2
            coc[y * nx + x] = [y * nx + (x + 1) % nx, y * nx + (x - 1) % nx,
                               ((y + 1) % ny) * nx + (x + shift - 1) % nx, ((y + 1) % ny) * nx + (x + shift) % nx,
                               ((y - 1) % ny) * nx + (x + shift - 1) % nx, ((y - 1) % ny) * nx + (x + shift) % nx]
    return coc


def _updrafts(rng, n, nl, steps):
    """Strong updrafts that wander and persist: each step some cells start one, and most keep it."""
    w = np.zeros((steps, nl + 1, n), np.float32)
    on = rng.random(n) < 0.08
    for s in range(steps):
        on = (on & (rng.random(n) < 0.9)) | (rng.random(n) < 0.02)
        w[s, 1] = np.where(on, 14.0, 2.0)
    return w


def test_two_ranks_count_as_one_rank_across_the_partition_edge():
    """THE BREAKAGE THIS GUARDS (multi-rank arming, 19 Nov 2026): a rank's cells at a partition edge read
    their neighbours' previous mask and count from halo cells; with the halo exchange they equal the one-rank run's
    in every owned cell at every step, and without it they would not."""
    import cupy as cp
    ny, nx, nl, steps, dt, start = 12, 10, 2, 120, 20.0, 1716292800.0
    n = ny * nx
    coc = _torus(ny, nx)
    w = _updrafts(np.random.default_rng(7), n, nl, steps)
    one = hailcast.Hailcast(cp, n_cells=n, n_levels=nl, cells_on_cell=coc, n_edges_on_cell=np.full(n, 6),
                            fine=np.zeros(n, bool))
    # Two ranks: rows 0-5 and rows 6-11; each local order is owned cells first, then its halo (the neighbours of
    # owned cells another rank owns), as the partitions lay them out.
    ranks = []
    for part in (np.arange(0, n // 2), np.arange(n // 2, n)):
        owned = set(part.tolist())
        halo = sorted({int(j) for c in part for j in coc[c] if int(j) not in owned})
        l2g = np.concatenate([part, np.array(halo, np.int64)])
        g2l = {int(g): i for i, g in enumerate(l2g)}
        local = np.array([[g2l[int(j)] for j in coc[c]] for c in part], np.int64)
        ranks.append(dict(l2g=l2g, owned=len(part), coc=local))
    snapshot = {}

    class Halo:
        def __init__(self, me):
            self.me = me

        def exchange(self, label, fields):
            r = ranks[self.me]
            for name, array, kind in fields:
                assert kind == "cell" and array.shape[-1] == len(r["l2g"])
                array[r["owned"]:] = cp.asarray(snapshot[name][r["l2g"][r["owned"]:]])

    for i, r in enumerate(ranks):
        r["hail"] = hailcast.Hailcast(cp, n_cells=r["owned"], n_levels=nl, cells_on_cell=r["coc"],
                                      n_edges_on_cell=np.full(r["owned"], 6), fine=np.zeros(r["owned"], bool),
                                      n_ext=len(r["l2g"]), halo=Halo(i))
    edge_cells_counted = 0
    for s in range(steps):
        # The owners' words at the end of the last step (what a real exchange sends: every rank enters the round
        # before any rank's step changes them).
        for key in ("mask", "dur"):
            words = np.zeros(n, np.float32)
            for r in ranks:
                words[r["l2g"][:r["owned"]]] = getattr(r["hail"], key)[:r["owned"]].get()
            snapshot["hailcast/" + key] = words
        seconds = start + (s + 1) * dt
        one.update({"w": cp.asarray(w[s])}, seconds=seconds, dt=dt)
        for r in ranks:
            r["hail"].update({"w": cp.asarray(np.ascontiguousarray(w[s][:, r["l2g"][:r["owned"]]]))},
                             seconds=seconds, dt=dt)
        whole = one.dur.get()
        for r in ranks:
            assert r["hail"].error is None
            np.testing.assert_array_equal(r["hail"].dur[:r["owned"]].get(), whole[r["l2g"][:r["owned"]]])
        edge = [c for c in ranks[0]["l2g"][:ranks[0]["owned"]] if any(j >= n // 2 for j in coc[c])]
        edge_cells_counted += int((whole[edge] > 0).sum())
    assert one.dur.get().max() > hailcast.DURATION_S and edge_cells_counted > 0
    assert all(r["hail"].exchanges == steps for r in ranks)


def test_a_run_resumed_from_the_seal_folds_on_as_the_straight_run():
    """THE BREAKAGE THIS FIXES: a run resumed from a seal began its updraft count at zero and made no stone for 15
    minutes.  Straight 90 steps against 45 steps, carry(), a fresh Hailcast resume(), 45 more: every word equal."""
    import cupy as cp
    ny, nx, nl, steps, dt, start = 8, 8, 2, 90, 20.0, 1716292800.0
    n = ny * nx
    coc = _torus(ny, nx)
    rng = np.random.default_rng(11)
    w = _updrafts(rng, n, nl, steps)
    sfc = rng.random((steps, n)).astype(np.float32) * 30.0
    col = sfc + rng.random((steps, n)).astype(np.float32) * 20.0

    def fresh():
        return hailcast.Hailcast(cp, n_cells=n, n_levels=nl, cells_on_cell=coc, n_edges_on_cell=np.full(n, 6),
                                 fine=np.zeros(n, bool), psd="tempo")

    def step(hail, s):
        hail.update({"w": cp.asarray(w[s]), "max_hail_diameter_sfc": cp.asarray(sfc[s]),
                     "max_hail_diameter_column": cp.asarray(col[s])}, seconds=start + (s + 1) * dt, dt=dt)

    straight, first = fresh(), fresh()
    for s in range(steps):
        step(straight, s)
    for s in range(steps // 2):
        step(first, s)
    first.window[0, :5] = cp.asarray(np.arange(5, dtype=np.float32) + 20)   # stones of the hour so far ride too
    sealed = first.carry()
    assert {hailcast.CARRY + k for k in ("mask", "dur", "window", "runmax", "last", "psd")} == set(sealed)
    resumed = fresh()
    # A one-card seal's words may carry the native dummy cell after the global cells (moving_mpi_v842).
    padded = {k: np.concatenate([v, np.zeros((*v.shape[:-1], 1), v.dtype)], -1) for k, v in sealed.items()}
    assert set(resumed.resume(padded)) == {"mask", "dur", "window", "runmax", "last", "psd"}
    assert resumed.last == first.last
    for s in range(steps // 2, steps):
        step(resumed, s)
    np.testing.assert_array_equal(resumed.dur.get(), straight.dur.get())
    np.testing.assert_array_equal(resumed.mask.get(), straight.mask.get())
    np.testing.assert_array_equal(resumed.psd_window.get(), straight.psd_window.get())
    np.testing.assert_array_equal(resumed.window[0, :5].get(), np.arange(5) + 20)
    assert straight.dur.get().max() > hailcast.DURATION_S
    # What the defect did: a resume with nothing carried starts the count again.
    zero = fresh()
    for s in range(steps // 2, steps):
        step(zero, s)
    assert not np.array_equal(zero.dur.get(), straight.dur.get())
