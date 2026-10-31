"""Scalar reference from d620bf0c; same repository Apache-2.0 license."""

def recover_velocities(
    *,
    rho_zz: object,
    ru: object,
    rw: object,
    zz: object,
    fzm: object,
    fzp: object,
    cf1: float,
    cf2: float,
    cf3: float,
    coupling: TerrainCoupling,
    array_layout: ArrayLayout = "logical",
    time_index: int = 0,
) -> RecoveredVelocity:
    """Recover exact public ``u``/``w`` from coupled ``ru``/``rw`` fields."""

    rho_l, ru_l, rw_l, zz_l, fzm_l, fzp_l = _coupled_recovery_inputs(
        rho_zz=rho_zz,
        ru=ru,
        rw=rw,
        zz=zz,
        fzm=fzm,
        fzp=fzp,
        coupling=coupling,
        array_layout=array_layout,
        time_index=time_index,
    )
    dtype = rho_l.dtype
    if coupling.n_vert_levels < 3:
        raise ValueError("bottom cf1/cf2/cf3 recovery requires at least three levels")
    if not all(np.isfinite(value) for value in (cf1, cf2, cf3)):
        raise ValueError("cf1, cf2 and cf3 must be finite")
    cf1_l, cf2_l, cf3_l = (
        dtype.type(cf1),
        dtype.type(cf2),
        dtype.type(cf3),
    )
    two = dtype.type(2.0)
    one = dtype.type(1.0)
    nlev, ncells = rho_l.shape
    nedges = coupling.n_edges

    u = np.empty((nlev, nedges), dtype=dtype)
    for edge in range(nedges):
        cell1 = int(coupling.cells_on_edge[edge, 0])
        cell2 = int(coupling.cells_on_edge[edge, 1])
        for level in range(nlev):
            denominator = rho_l[level, cell1] + rho_l[level, cell2]
            if denominator == dtype.type(0.0):
                raise FloatingPointError(
                    f"zero edge-density denominator at level {level}, edge {edge}"
                )
            u[level, edge] = two * ru_l[level, edge] / denominator

    w = np.zeros((nlev + 1, ncells), dtype=dtype)
    for cell in range(ncells):
        for level in range(1, nlev):
            interp_zz = (
                fzm_l[level] * zz_l[level, cell]
                + fzp_l[level] * zz_l[level - 1, cell]
            )
            if interp_zz == dtype.type(0.0):
                raise FloatingPointError(
                    f"zero interface zz denominator at level {level}, cell {cell}"
                )
            w[level, cell] = rw_l[level, cell] / interp_zz

    for cell in range(ncells):
        for slot in range(int(coupling.n_edges_on_cell[cell])):
            edge = int(coupling.edges_on_cell[cell, slot])
            cell_sign = coupling.edges_on_cell_sign[cell, slot]
            flux = (
                cf1_l * ru_l[0, edge]
                + cf2_l * ru_l[1, edge]
                + cf3_l * ru_l[2, edge]
            )
            w[0, cell] = w[0, cell] + cell_sign * (
                coupling.zb_cell[0, cell, slot]
                + np.copysign(one, flux) * coupling.zb3_cell[0, cell, slot]
            ) * flux
            for level in range(1, nlev):
                flux = (
                    fzm_l[level] * ru_l[level, edge]
                    + fzp_l[level] * ru_l[level - 1, edge]
                )
                w[level, cell] = w[level, cell] + cell_sign * (
                    coupling.zb_cell[level, cell, slot]
                    + np.copysign(one, flux) * coupling.zb3_cell[level, cell, slot]
                ) * flux

        bottom_density = (
            cf1_l * rho_l[0, cell]
            + cf2_l * rho_l[1, cell]
            + cf3_l * rho_l[2, cell]
        )
        if bottom_density == dtype.type(0.0):
            raise FloatingPointError(f"zero bottom density denominator at cell {cell}")
        w[0, cell] = w[0, cell] / bottom_density
        for level in range(1, nlev):
            interp_rho = (
                fzm_l[level] * rho_l[level, cell]
                + fzp_l[level] * rho_l[level - 1, cell]
            )
            if interp_rho == dtype.type(0.0):
                raise FloatingPointError(
                    f"zero interface density denominator at level {level}, cell {cell}"
                )
            w[level, cell] = w[level, cell] / interp_rho

    return RecoveredVelocity(u=u, w=w)
