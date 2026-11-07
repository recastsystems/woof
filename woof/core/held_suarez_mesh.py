"""Resident mesh-edge binding for WRF Held–Suarez momentum damping.

The column seam owns thermal relaxation. The mesh caller owns momentum at
every RK stage. This operator has no prognostic state; restarts rebind the
mesh and base pressure, then read the restored stage pressure and momentum.
"""
import numpy as np


class HeldSuarezMeshMomentum:
    schema = "held-suarez-edge-v1"

    def __init__(self):
        self.bound = False

    def bind(self, cells_on_edge, pressure_base, *, solved_edges=None):
        import cupy as cp
        from woof.core.kernels import load_module_no_ftz
        if self.bound:
            raise RuntimeError("Held–Suarez mesh binding already owns a mesh")
        for name, value, dtype in (("cells_on_edge", cells_on_edge, cp.int32),
                                   ("pressure_base", pressure_base, cp.float32)):
            if not isinstance(value, cp.ndarray) or value.dtype != dtype:
                raise TypeError(f"{name} must be a resident {dtype} array")
            if value.ndim != 2 or not value.flags.c_contiguous:
                raise ValueError(f"{name} must be a packed matrix")
        if cells_on_edge.shape[1] != 2:
            raise ValueError("each mesh edge requires two adjacent cells")
        nz, nc = pressure_base.shape
        ne = cells_on_edge.shape[0]
        solved = ne if solved_edges is None else solved_edges
        if type(solved) is not int or not 0 <= solved <= ne:
            raise ValueError("solved edge count lies outside the mesh")
        # The mesh uploader validates topology indices before this binding.
        self.cells, self.pb = cells_on_edge, pressure_base
        self.shape, self.pressure_shape = (nz, ne), (nz, nc)
        self.args = tuple(np.int32(v) for v in (nz, nc, ne, solved))
        self.kernel = load_module_no_ftz("held_suarez_mesh", no_fmad=True).get_function(
            "held_suarez_edge_momentum")
        self.bound = True

    def apply(self, tendency, momentum, pressure_perturbation):
        if not self.bound:
            raise RuntimeError("Held–Suarez RK-stage edge momentum damping is not bound")
        import cupy as cp
        for name, value, shape in (("tendency", tendency, self.shape),
                                    ("momentum", momentum, self.shape),
                                    ("pressure", pressure_perturbation, self.pressure_shape)):
            if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f"{name} must be a packed resident float32 {shape}")
        self.kernel(((momentum.size+255)//256,), (256,),
                    (*self.args, self.cells, pressure_perturbation, self.pb,
                     momentum, tendency))
