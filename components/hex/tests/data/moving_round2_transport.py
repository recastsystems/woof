"""Scalar reference from d620bf0c; same repository Apache-2.0 license."""

def _atmosphere_horizontal_edge_values(
    tracers: FloatArray,
    velocity: FloatArray,
    dv_edge: FloatArray,
    cells_on_edge: NDArray[np.int64],
    adv: FloatArray,
    adv3: FloatArray,
    n_adv: NDArray[np.int64],
    adv_cells: NDArray[np.int64],
    order: int,
    coefficient: float,
    monotonic_source_flux: bool = False,
    config_apply_lbcs: bool = False,
    bdy_mask_edge: NDArray[np.int64] | None = None,
) -> FloatArray:
    """Scalar value times edge length, before multiplication by ``ruAvg``."""

    n_tracers, n_levels, _ = tracers.shape
    n_edges = cells_on_edge.shape[0]
    dtype = tracers.dtype
    result = np.zeros((n_tracers, n_levels, n_edges), dtype=dtype)
    half = dtype.type(0.5)
    coef = dtype.type(coefficient if order == 3 else 0.0)
    for edge in range(n_edges):
        if bdy_mask_edge is not None and order != 2 and not monotonic_source_flux:
            # The monotonic path computes the complete high-order stencil at
            # every edge and applies its regional downgrade later at the
            # flux-array stage (F:5536-5546); only the non-monotonic path
            # downgrades here (F:4764-4842).
            mode = _regional_edge_flux_mode(
                config_apply_lbcs, bdy_mask_edge, edge
            )
            if mode == "skip":
                continue
            if mode == "upwind":
                # atm_advance_scalars_work F:4824-4841: first-order upwind
                # edge values at the outermost relaxation edges; uhAvg is
                # applied by the cell loop exactly as for the full stencil.
                cell1 = int(cells_on_edge[edge, 0])
                cell2 = int(cells_on_edge[edge, 1])
                for level in range(n_levels):
                    u_direction = np.copysign(
                        dtype.type(0.5), velocity[level, edge]
                    )
                    u_positive = dv_edge[edge] * abs(
                        u_direction + dtype.type(0.5)
                    )
                    u_negative = dv_edge[edge] * abs(
                        u_direction - dtype.type(0.5)
                    )
                    for scalar in range(n_tracers):
                        value = (
                            u_positive * tracers[scalar, level, cell1]
                            + u_negative * tracers[scalar, level, cell2]
                        )
                        if monotonic_source_flux:
                            value = velocity[level, edge] * value
                        result[scalar, level, edge] = value
                continue
        if order == 2:
            cell1, cell2 = cells_on_edge[edge]
            for level in range(n_levels):
                for scalar in range(n_tracers):
                    result[scalar, level, edge] = dv_edge[edge] * half * (
                        tracers[scalar, level, cell1] + tracers[scalar, level, cell2]
                    )
                    if monotonic_source_flux:
                        result[scalar, level, edge] = (
                            velocity[level, edge]
                            * result[scalar, level, edge]
                        )
            continue
        if monotonic_source_flux:
            count = int(n_adv[edge])
            if count == 10:
                # v8.4.1 retains a literal ten-term sum for hexagonal edges,
                # then multiplies the completed edge value by uhAvg.
                for level in range(n_levels):
                    sign = _fortran_sign_one(velocity[level, edge], dtype)
                    for scalar in range(n_tracers):
                        cell = int(adv_cells[edge, 0])
                        value = (
                            adv[edge, 0] + coef * sign * adv3[edge, 0]
                        ) * tracers[scalar, level, cell]
                        for slot in range(1, 10):
                            cell = int(adv_cells[edge, slot])
                            value = value + (
                                adv[edge, slot]
                                + coef * sign * adv3[edge, slot]
                            ) * tracers[scalar, level, cell]
                        result[scalar, level, edge] = (
                            velocity[level, edge] * value
                        )
            else:
                # The generic source associates uhAvg with each stencil
                # weight before multiplying by the scalar and accumulating.
                for slot in range(count):
                    cell = int(adv_cells[edge, slot])
                    if cell < 0 or cell >= tracers.shape[2]:
                        raise ValueError(
                            "active adv_cells_for_edge entry is out of range"
                        )
                    for level in range(n_levels):
                        scalar_weight = velocity[level, edge] * (
                            adv[edge, slot]
                            + coef
                            * _fortran_sign_one(velocity[level, edge], dtype)
                            * adv3[edge, slot]
                        )
                        for scalar in range(n_tracers):
                            result[scalar, level, edge] = (
                                result[scalar, level, edge]
                                + scalar_weight * tracers[scalar, level, cell]
                            )
            continue
        # The generic branch at time_integration.F:3192-3203 has stencil slot
        # outside the level/scalar loops; retain that accumulation order.
        for slot in range(int(n_adv[edge])):
            cell = int(adv_cells[edge, slot])
            if cell < 0 or cell >= tracers.shape[2]:
                raise ValueError("active adv_cells_for_edge entry is out of range")
            for level in range(n_levels):
                weight = adv[edge, slot] + coef * _fortran_sign_one(
                    velocity[level, edge], dtype
                ) * adv3[edge, slot]
                for scalar in range(n_tracers):
                    result[scalar, level, edge] = (
                        result[scalar, level, edge]
                        + weight * tracers[scalar, level, cell]
                    )
    return result
