"""Same-host ranks resumed from a seal hand HAILCAST its carried words, as one card does.

THE BREAKAGE THIS PREVENTS (0.6.2, 20 Nov 2026): from the 0.6.2 wheel on a clean RTX PRO 6000 box, the 3 km
Iowa storm box (HRRR 2024-05-21 15Z, geometry 2, 51,671 cells) on two ranks sealed at 1 h and resumed to 2 h: every
history word equal to the run that never stopped, but 1,246 of the 17Z diagnostics words differed, all eight HAILCAST
fields over 156 cells, with the direct per-rank restore on and off. The seal carried HAILCAST's updraft count, mask
and hour's maxima ('hailcast:*' among the output accumulators), but DistributedMoving.restore_continuation passed on
only the words that have a diagnostics buffer of their own, so each rank's HAILCAST started its count at zero again
and printed no "resumed from the seal" line.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tests"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from test_mpasphys_mpi_v842 import Comm, partitions  # noqa: E402


def test_rank_resume_hands_hailcast_its_owned_words():
    from woof.hex.mpasphys import moving_mpi_v842 as moving
    from woof.hex.mpasphys.checkpoint_v842 import CheckpointRoute

    parts = partitions()
    sums = np.arange(5, dtype=np.float32) + 10           # one card's layout: 4 cells + the native dummy
    dur = np.arange(5, dtype=np.float32) * 60.0          # HAILCAST's updraft count (s), per cell
    window = np.arange(40, dtype=np.float32).reshape(8, 5)
    carried = {'prec_acc_nc': sums, 'hailcast:dur': dur, 'hailcast:window': window}
    carrier = dict(physics=dict(fields={}, manager=dict(alarms={}), manager_elapsed='3600',
                                continuation=dict(scheme_flags={}, output_accumulators=carried, clock_start={})))
    kept = moving.rank_carrier(carrier)
    # The assembled (not direct) path projects every carried word onto each rank's owned cells.
    restored = moving.project_restored_words(dict(restored={}, window=None, surface_service=None, accumulators=carried),
        SimpleNamespace(fields=[], dimensions=dict(nCells=4, nEdges=4, nVertices=4)), parts)
    for part, words in zip(parts, restored):
        owned = part.cell_l2g[:int(part.n_owned_cells)]
        np.testing.assert_array_equal(words['accumulators']['hailcast:dur'], dur[owned])
        np.testing.assert_array_equal(words['accumulators']['hailcast:window'], window[:, owned])
    worker = CheckpointRoute(carrier=kept, offset_seconds=3600.0, resume_continues=True)
    rank = moving.DistributedMoving.__new__(moving.DistributedMoving)
    rank.comm, rank.partitions, rank.route = Comm(), parts, worker
    rank._accumulators = restored[0]['accumulators']
    session = SimpleNamespace(driver=SimpleNamespace(manager=SimpleNamespace(alarms={})),
                              native=SimpleNamespace(scheme_state={}))
    resumed = {}
    diagnostics = SimpleNamespace(buffers={('diag', 'prec_acc_nc'): np.full(5, -1, np.float32)},
                                  cp=SimpleNamespace(asnumpy=np.asarray), resume_accumulators=resumed.update)
    rank.restore_continuation(session=session, diagnostics=diagnostics, dt=20.0)
    owned = parts[0].cell_l2g[:int(parts[0].n_owned_cells)]
    assert sorted(resumed) == ['hailcast:dur', 'hailcast:window', 'prec_acc_nc']
    np.testing.assert_array_equal(resumed['hailcast:dur'], dur[owned])
    np.testing.assert_array_equal(resumed['hailcast:window'], window[:, owned])
