"""Extra history fields (the 0-1/0-2/0-3/2-5 km UH maxima) on the LTS class executor are written on the global mesh.

THE BREAKAGE THIS GUARDS (WOOF canary on woof hex 0.4.12, 2026-11-05 19:01Z, run_db17f6b76d9c7a74): every storm- and
setup-following run stopped in segment 1 with "updraft_helicity_max_01: nCells=75192 differs from authority 98108".
A moving member rides the class executor (lts_v842.moving_admitted); each class's UH layers hold its local cells and
class 0 handed its own local words to the global history writer.

Two tests:

* the gather itself, on the host (CPU tier): two classes, owned cells first, class 0 writes every class's owned words
  at their global cells and the other class gets nothing to write;
* the real door (gpu, slow): a two-segment storm-following run of a prepared case with the UH layers on.  It stops in
  segment 1 on 0.4.12 and must write ``updraft_helicity_max_01/02/03`` sized to every segment's own member mesh.
  WOOF_HEX_UH_MOVING_CASE names the case JSON and WOOF_HEX_UH_MOVING_CASEDIR its staged and prepared folder
  (``woof hex moving stage`` / ``prepare``); WOOF_HEX_UH_MOVING_GEOG a WPS_GEOG root.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from woof.hex.mpasphys import lts_v842


def _execution(board, index, partitions, cells):
    return SimpleNamespace(board=board, index=index, partitions=partitions,
                           global_layout=SimpleNamespace(dimensions={'nCells': cells}))


def test_class_extras_are_scattered_to_their_global_cells_and_class_0_writes_them():
    cells = 7
    # Class 0 owns global cells 0, 2, 4, 6 and keeps a halo of 1, 3; class 1 owns 1, 3, 5 and keeps a halo of 0.
    partitions = [SimpleNamespace(n_owned_cells=4, cell_l2g=np.array([0, 2, 4, 6, 1, 3])),
                  SimpleNamespace(n_owned_cells=3, cell_l2g=np.array([1, 3, 5, 0]))]
    board = lts_v842._Board(2)
    local = {0: np.array([10, 12, 14, 16, -1, -1], dtype=np.float32),
             1: np.array([11, 13, 15, -1], dtype=np.float32)}
    results = {}

    def one(index):
        extras = {'updraft_helicity_max_01': (local[index], ('nCells',), 'm2 s-2')}
        results[index] = lts_v842.gather_extra_history(_execution(board, index, partitions, cells),
                                                       ('2026-10-24_01:00:00', 'extra-history'), extras)

    other = threading.Thread(target=one, args=(1,))
    other.start()
    one(0)
    other.join()
    assert results[1] == {}
    values, dims, units = results[0]['updraft_helicity_max_01']
    assert dims == ('nCells',) and units == 'm2 s-2'
    assert values.shape == (cells,)
    np.testing.assert_array_equal(values, np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.float32))
    assert board.deposits == {}


def test_a_class_without_its_owned_words_is_refused_by_name():
    partitions = [SimpleNamespace(n_owned_cells=2, cell_l2g=np.array([0, 1])),
                  SimpleNamespace(n_owned_cells=2, cell_l2g=np.array([2, 3]))]
    board = lts_v842._Board(2)
    board.deposits[('t', 'extra-history')] = {1: {'x': (np.zeros(1, np.float32), ('nCells',), 'm')}}
    with pytest.raises(ValueError, match='class 1 holds 1 words for 2 owned cells'):
        lts_v842.gather_extra_history(_execution(board, 0, partitions, 4), ('t', 'extra-history'),
                                      {'x': (np.zeros(2, np.float32), ('nCells',), 'm')})


CASE = os.environ.get("WOOF_HEX_UH_MOVING_CASE")
CASEDIR = os.environ.get("WOOF_HEX_UH_MOVING_CASEDIR")
GEOG = os.environ.get("WOOF_HEX_UH_MOVING_GEOG")


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not (CASE and CASEDIR), reason="WOOF_HEX_UH_MOVING_CASE and WOOF_HEX_UH_MOVING_CASEDIR name a "
                                                     "staged and prepared moving case")
def test_a_two_segment_storm_following_run_writes_the_uh_layers_on_every_segment(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    out = tmp_path / "run"
    command = [sys.executable, "-m", "woof.hex.cli", "moving", "run", "--case", CASE, "--casedir", CASEDIR,
               "--out", str(out), "--hours", os.environ.get("WOOF_HEX_UH_MOVING_HOURS", "6"),
               "--members", str(tmp_path / "members")]
    if GEOG:
        command += ["--geog", GEOG]
    result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "WOOF_HEX_UH_LAYERS": "1"})
    assert result.returncode == 0, (result.stdout[-2000:], result.stderr[-4000:])
    segments = sorted(p for p in out.glob("segment-*") if p.is_dir())
    assert len(segments) >= 2, f"the run never moved: {[p.name for p in segments]}"
    for segment in segments:
        histories = sorted(segment.glob("fc/history.*.nc"))
        assert histories, f"{segment.name} wrote no history"
        for path in histories:
            with netCDF4.Dataset(path) as data:
                cells = len(data.dimensions["nCells"])
                for name in ("updraft_helicity_max_01", "updraft_helicity_max_02", "updraft_helicity_max_03"):
                    assert name in data.variables, f"{segment.name}/{path.name} lacks {name}"
                    values = np.asarray(data.variables[name][...])
                    assert values.shape[-1] == cells
                    assert np.all(np.isfinite(values)) and values.min() >= 0.0
