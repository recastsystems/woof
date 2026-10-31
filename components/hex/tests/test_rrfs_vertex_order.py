"""Boundary vertex ordering regression for the native first-valid-neighbour rule."""
from types import SimpleNamespace
import numpy as np
from woof.hex.mpasphys.rrfs.output_diagnostics_rrfs import native_rank_vertex_order


def test_outside_first_slot_does_not_reverse_owned_boundary_vertices():
    # Cut file order is 1,2,3; native's cell traversal first sees 2,1,3.
    # Vertex 2 has outside neighbour nCells+1 in slot 1, but owns slot 2.
    fields = {
        'verticesOnCell': np.array([[2, 3], [1, 1], [3, 2]]),
        'nEdgesOnCell': np.array([3, 3]),
        'cellsOnVertex': np.array([[1, 3, 2], [2, 1, 1], [3, 2, 3]]),
        'indexToCellID': np.array([1, 2]),
    }
    session = SimpleNamespace(
        native=SimpleNamespace(get_field=lambda pool, name: fields[name]),
        layout=SimpleNamespace(dimensions={'nCellsSolve': 2, 'nCells': 2, 'nVertices': 3}),
        device_pools=SimpleNamespace(cp=np),
    )
    np.testing.assert_array_equal(native_rank_vertex_order(session), [1, 0, 2])
