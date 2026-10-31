"""Fork-only core pool initialization; arithmetic and traversal live in Rust.

MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1481-1584 at 1d6e370e.
The second ring is consumed by fork convection, not by numerical dycore kernels.
"""
import os
import shutil
import struct
import subprocess
import numpy as np


# MPAS-Model-rrfs/src/core_atmosphere/mpas_atm_core.F:1481-1584 at 1d6e370e.
def initialize_second_ring_rrfs(mesh, *, cell_solve_count=None, bridge=None):
    """Return native traversal order in the host mesh's zero-based convention."""
    arrays = getattr(mesh, 'arrays', {})
    counts = getattr(mesh, 'nEdgesOnCell', arrays.get('nEdgesOnCell'))
    neighbors = getattr(mesh, 'cellsOnCell', arrays.get('cellsOnCell'))
    width2 = getattr(mesh, 'maxEdges2', arrays.get('maxEdges2'))
    if width2 is None:
        width2 = getattr(mesh, 'edgesOnEdge', arrays.get('edgesOnEdge')).shape[1]
    cells, width = neighbors.shape
    solved = cells if cell_solve_count is None else int(cell_solve_count)
    entry_counts = getattr(mesh, 'nCellsOnCellsOnCell', arrays.get('nCellsOnCellsOnCell'))
    entry_neighbors = getattr(mesh, 'cellsOnCellsOnCell', arrays.get('cellsOnCellsOnCell'))
    if (entry_counts is None) != (entry_neighbors is None):
        raise ValueError('native second-ring entry requires both count and neighbor pools')
    if entry_counts is not None and (entry_counts.shape != (cells,) or
                                    entry_neighbors.shape != (cells, int(width2))):
        raise ValueError('native second-ring entry pool extent mismatch')
    real_cells = cells
    counts, neighbors = np.asarray(counts), np.asarray(neighbors)
    if np.any(neighbors < 0):
        # A limited-area cull: its boundary cells store 0 ("no neighbour", -1 here), which native's block creator
        # maps to the garbage cell nCells+1, whose nEdgesOnCell is 0 (mpas_block_creator.F:1490-1496).  Native's
        # traversal (mpas_atm_core.F:1521-1548) then takes the garbage cell into a boundary cell's group like any
        # neighbour.  The bridge gets that garbage row explicitly (0-based index nCells, zero neighbours), so its
        # output names it as native does (nCells, i.e. native nCells+1).  THE BREAKAGE THIS FIXES: every
        # limited-area RRFS run stopped in core init (rw_mpas_rrfs_connectivity exit 1 on the -1 neighbours).
        garbage = cells
        counts = np.append(counts, 0)
        neighbors = np.vstack((np.where(neighbors < 0, garbage, neighbors), np.full((1, width), garbage)))
        if entry_counts is not None:
            entry_counts = np.append(np.asarray(entry_counts), 0)
            entry_neighbors = np.asarray(entry_neighbors)
            entry_neighbors = np.vstack((np.where(entry_neighbors < 0, garbage, entry_neighbors),
                                         np.full((1, int(width2)), garbage)))
        cells += 1
    from .engines import bundled_program
    executable = (bridge or os.environ.get('RW_MPAS_RRFS_CONNECTIVITY') or bundled_program('rw_mpas_rrfs_connectivity')
                  or shutil.which('rw_mpas_rrfs_connectivity'))
    if executable is None:
        raise FileNotFoundError('rw_mpas_rrfs_connectivity Rust bridge is not installed')
    values = (counts, neighbors)
    if entry_counts is not None:
        values += (entry_counts, entry_neighbors)
    payload = struct.pack('<8sQQQQ', b'RRFSN2V2' if entry_counts is not None else b'RRFSN2V1',
                          cells, solved, width, int(width2))
    for value in values:
        payload += np.ascontiguousarray(value, dtype='<i4').tobytes()
    output = subprocess.run([str(executable)], input=payload, capture_output=True, check=True).stdout
    if len(output) != (cells + cells*int(width2))*4:
        raise ValueError('connectivity bridge returned an invalid word count')
    words = np.frombuffer(output, dtype='<i4').copy()
    return words[:real_cells], words[cells:].reshape(cells, int(width2))[:real_cells]
