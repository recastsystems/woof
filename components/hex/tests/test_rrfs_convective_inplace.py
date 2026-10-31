"""RRFS convective diagnostics: in-place pool layout, split launches and the per-event will-be-written set.

THE BREAKAGES THESE PREVENT: (1) the compute kernel splits a large mesh into consecutive launches; an offset error
would leave columns past the first launch unwritten or written from the wrong cell. (2) The in-place driver writes
each output row through a device pointer table; a wrong row would put CAPE into CIN's allocation. (3) The
will-be-written query answers one output event from one name set; it must give the loop's answer for every name
and see a new event's alarms.
"""
import numpy as np
import pytest

from woof.hex.mpasdiag.utils import DiagnosticsUtils, Direction, StreamView


def _views(due):
    return (
        StreamView('history', Direction.OUTPUT, True, due, {'cape': True, 'cin': True, 'theta': True}),
        StreamView('diag', Direction.OUTPUT, True, due, {'srh_0_1km': True, 'w_velocity_max': False}),
        StreamView('input', Direction.INPUT, True, True, {'lcl': True}),
        StreamView('restart', Direction.INPUT_OUTPUT, True, not due, {'lfc': True}),
        StreamView('empty', Direction.OUTPUT, False, due, {'uzonal_1km': True}),
    )


NAMES = ('cape', 'cin', 'theta', 'srh_0_1km', 'w_velocity_max', 'lcl', 'lfc', 'uzonal_1km', 'nothing', 'cape ')


@pytest.mark.parametrize('due', (True, False))
def test_will_be_written_event_set_matches_the_loop(due):
    held, fresh = DiagnosticsUtils(), DiagnosticsUtils()
    views = _views(due)
    held.init(lambda: views)                  # one tuple per event: answered from one name set
    fresh.init(lambda: iter(_views(due)))     # any other provider: the MPAS loop
    for name in NAMES:
        assert held.field_will_be_written(name) == fresh.field_will_be_written(name), name


def test_will_be_written_sees_the_next_event():
    utils, state = DiagnosticsUtils(), {'views': _views(True)}
    utils.init(lambda: state['views'])
    assert utils.field_will_be_written('cape') and not utils.field_will_be_written('lfc')
    state['views'] = _views(False)
    assert not utils.field_will_be_written('cape') and utils.field_will_be_written('lfc')


def _columns(cp, n_alloc, nl, seed=7):
    """Moist, sheared columns in the pools' (level, cell) layout, padded past nCellsSolve."""
    rng = np.random.default_rng(seed)
    terrain = rng.uniform(0.0, 1500.0, n_alloc)
    zgrid = terrain[None, :] + np.linspace(0.0, 20000.0, nl + 1)[:, None]
    zmid = 0.5 * (zgrid[1:] + zgrid[:-1])
    p = 100000.0 * np.exp(-zmid / 8000.0) * rng.uniform(0.98, 1.02, n_alloc)[None, :]
    pb = 0.9 * p
    exner = (p / 100000.0) ** (287.0 / 1004.5)
    qv = 0.016 * np.exp(-zmid / 2500.0) * rng.uniform(0.3, 1.1, n_alloc)[None, :]
    theta = 298.0 + 3.5e-3 * zmid + rng.normal(0.0, 0.5, zmid.shape)
    rvord = np.float32(461.6) / np.float32(287.0)
    theta_m = theta * (1.0 + rvord * qv)
    u = 5.0 + 2.0e-3 * zmid + rng.normal(0.0, 1.0, zmid.shape)
    v = -2.0 + 1.0e-3 * zmid + rng.normal(0.0, 1.0, zmid.shape)
    f = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float32))
    return {'zgrid': f(zgrid), 'uReconstructZonal': f(u), 'uReconstructMeridional': f(v), 'theta_m': f(theta_m),
            'qv': f(qv), 'exner': f(exner), 'pressure_base': f(pb), 'pressure_p': f(p - pb)}


def test_compute_in_place_rows_and_split_launches_keep_every_word():
    cp = pytest.importorskip('cupy')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip('no CUDA device')
    from woof.hex.mpasdiag import convective_rrfs as m
    n_alloc, n_solve, nl = 300, 290, 24
    arrays = _columns(cp, n_alloc, nl)
    start = np.random.default_rng(3).normal(0.0, 1.0, (len(m.FIELDS), n_alloc)).astype(np.float32)
    names = m.FIELDS[:14]

    stacked = cp.asarray(start)
    m.ConvectiveDiagnostics(stacked, n_solve, nl, m.ACCUMULATORS).compute(arrays, names)

    rows = [cp.asarray(start[i]) for i in range(len(m.FIELDS))]
    m.ConvectiveDiagnostics(rows, n_solve, nl, m.ACCUMULATORS).compute(arrays, names)

    split_rows = [cp.asarray(start[i]) for i in range(len(m.FIELDS))]
    split = m.ConvectiveDiagnostics(split_rows, n_solve, nl, m.ACCUMULATORS)
    split.COMPUTE_WORK_BYTES = 4 * (17 * nl + 3 + 2 * split.CAPE_SUBSTEPS) * 64   # 64 columns per launch: 5 launches
    split.compute(arrays, names)

    one = cp.asnumpy(stacked).view(np.uint32)
    assert np.array_equal(cp.asnumpy(cp.stack(rows)).view(np.uint32), one)
    assert np.array_equal(cp.asnumpy(cp.stack(split_rows)).view(np.uint32), one)
    # Only solve cells change; halo words and the accumulators keep their entry words.
    assert np.array_equal(one[:, n_solve:], start[:, n_solve:].view(np.uint32))
    assert np.array_equal(one[14:], start[14:].view(np.uint32))
    # The moist columns are unstable, so the ascent ran: CAPE was formed in a good share of the solve cells.
    assert (cp.asnumpy(stacked[0, :n_solve]) > 0.0).sum() > n_solve // 4
