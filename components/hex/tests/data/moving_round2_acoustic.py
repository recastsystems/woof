"""Scalar reference from d620bf0c; same repository Apache-2.0 license."""

def convert_w_tendency_to_omega(
    mesh: object,
    w_tendency: FloatArray,
    u_tendency: FloatArray,
    *,
    fzm: FloatArray,
    fzp: FloatArray,
    zz: FloatArray,
    zb_cell: FloatArray,
    zb3_cell: FloatArray,
    boundary_mask_cell: NDArray[np.integer[Any]] | None = None,
    relaxation_zone: int = 0,
) -> FloatArray:
    """Transcribe ``atm_set_smlstep_pert_variables_work`` lines 2088-2110."""
    out = np.asarray(w_tendency).copy()
    u_tendency = np.asarray(u_tendency)
    nlev, nedges = u_tendency.shape
    ncells = _mesh_array(mesh, "areaCell").size
    if out.shape != (nlev + 1, ncells):
        raise ValueError(f"w_tendency shape {out.shape} != ({nlev + 1}, {ncells})")
    counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    signs = edge_signs_on_cells(mesh).astype(out.dtype, copy=False)
    if boundary_mask_cell is None:
        boundary_mask_cell = np.zeros(ncells, dtype=np.int32)
    for cell in range(ncells):
        if boundary_mask_cell[cell] > relaxation_zone:
            continue
        for slot in range(int(counts[cell])):
            edge = int(edges[cell, slot])
            if edge < 0 or edge >= nedges:
                continue
            for level in range(1, nlev):
                flux = signs[cell, slot] * (
                    fzm[level] * u_tendency[level, edge]
                    + fzp[level] * u_tendency[level - 1, edge]
                )
                upwind_sign = np.copysign(out.dtype.type(1.0), u_tendency[level, edge])
                out[level, cell] -= (
                    zb_cell[level, cell, slot] + upwind_sign * zb3_cell[level, cell, slot]
                ) * flux
        for level in range(1, nlev):
            out[level, cell] *= fzm[level] * zz[level, cell] + fzp[level] * zz[level - 1, cell]
    return out
