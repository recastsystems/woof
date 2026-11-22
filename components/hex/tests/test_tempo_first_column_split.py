"""TEMPO v3.0.3's first-call column is the one-process run's column on every partition.

THE BREAKAGE THIS PREVENTS: TEMPO v3.0.3 initialises graupel number and volume (graupel_init) and checks the
aerosols on the first column a process hands it, once (Fortran SAVE first_call_main).  Every same-host rank used
its own local column 0, so the 120 km hrrrv5 forecast on 2 and 4 pieces differed from the one-piece forecast after
step 1 in exactly the other ranks' first cells (global cells 3; 2, 3, 5) and then over the globe
(split-120-forward, 2026-11-13).  An N-card gradient was then the gradient of a different forecast.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys.rrfs import tempo_v303


def _provider(ids, owned):
    native = SimpleNamespace(get_field=lambda pool, name: np.asarray(ids, dtype=np.int32)
                             if (pool, name) == ("mesh", "indexToCellID") else None)
    layout = SimpleNamespace(dimensions={"nCells": len(ids), "nCellsSolve": owned})
    return SimpleNamespace(native=native, session=SimpleNamespace(layout=layout))


def _split(part, halo):
    """Local id lists as partition_assets_v841 orders them: owned ascending global id, then halo cells."""
    ranks = []
    for r in range(int(part.max()) + 1):
        owned = np.flatnonzero(part == r)
        extra = np.array(sorted(set(halo[r]) - set(owned.tolist())), dtype=np.int64)
        ranks.append((np.concatenate([owned, extra]) + 1, owned.size))
    return ranks


def test_one_process_initialises_its_column_zero():
    assert tempo_v303.first_call_column_here(_provider(np.arange(1, 11), 10))


def test_one_process_regional_mesh_keeps_column_zero_whatever_its_ids():
    # No halo: the process is the whole run, so its first column is the run's first column.
    assert tempo_v303.first_call_column_here(_provider(np.arange(501, 511), 10))


@pytest.mark.parametrize("ranks", [2, 4])
def test_exactly_one_rank_initialises_and_it_holds_file_cell_zero(ranks):
    # The 4-piece cut of the 120 km run put local cell 0 of ranks 0..3 on global cells 0, 5, 3, 2.
    part = np.array([0, 0, 3, 2, 2, 1, 3, 0, 1, 1, 3, 2] * 3)[:36] % ranks
    n = part.size
    halo = {r: [(int(np.flatnonzero(part == r)[0]) + k) % n for k in (1, 2)] + [0] for r in range(ranks)}
    holders = []
    for r, (ids, owned) in enumerate(_split(part, halo)):
        assert ids[0] != 1 or r == part[0]
        if tempo_v303.first_call_column_here(_provider(ids, owned)):
            holders.append(r)
    # A halo copy of file cell 0 (id 1 past the owned prefix) does not count: only its owner initialises.
    assert holders == [int(part[0])]
