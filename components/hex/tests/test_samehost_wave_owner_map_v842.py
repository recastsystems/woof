"""Same-host ranks hand the wave service the global mesh their partition was cut from, on both executors.

THE BREAKAGE THIS CHECKS: ``SameHostRankExecution.wave_input`` read ``wave_global_mesh``, which only the thread
executor set, so ``--waves on`` with same-host rank processes (the Linux default) stopped with AttributeError
when the wave service attached. No card or mesh files are needed: the rank state is set by hand.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from woof.hex.mpasphys.process_ranks_v842 import ProcessRankExecution  # noqa: E402
from woof.hex.mpasphys.samehost_v842 import SameHostRankExecution  # noqa: E402


def _rank(cls, source):
    execution = object.__new__(cls)   # only the rank's inputs and partition are read
    local = SimpleNamespace(arrays={})
    if source is not None:
        local.partition_source_mesh = source
    execution.inputs = (local,)
    execution.partition = SimpleNamespace(cell_l2g=np.array([4, 0, 2], dtype=np.int64), n_owned_cells=2)
    return execution


@pytest.mark.parametrize("cls", [ProcessRankExecution, SameHostRankExecution])
def test_wave_owner_map_is_the_partition_source_mesh(cls):
    source = SimpleNamespace(dimensions={"nCells": 5})
    mesh, ids, owned = _rank(cls, source).wave_input()
    assert mesh is source
    assert ids.tolist() == [4, 0, 2] and owned == 2


def test_a_rank_without_its_source_mesh_is_refused_by_name():
    with pytest.raises(ValueError, match="global mesh it was cut from"):
        _rank(ProcessRankExecution, None).wave_input()
