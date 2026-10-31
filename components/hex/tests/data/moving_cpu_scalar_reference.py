"""Frozen scalar operators from woof hex 4e84bed6, for word-for-word replay."""

def _rho_theta_tendency(
    mesh: object,
    state: PrognosticState,
    saved: PrognosticState,
    vertical: VerticalGrid,
    coefficients: AdvectionCoefficients,
    *,
    rk_step: int,
    third_order_coefficient: float,
    theta_m: FloatArray | None = None,
    theta_m_saved: FloatArray | None = None,
    inv_area_cell: FloatArray | None = None,
    skip_cells: NDArray[np.bool_] | None = None,
) -> FloatArray:
    """Frozen dry ``tend_theta`` loops at lines 5254-5346.

    ``skip_cells`` is the regional specified-zone set (bdyMaskCell >
    nRelaxZone): native computes these cells' tendencies through
    garbage-element gathers and then unconditionally overwrites them with
    driving tendencies (atm_bdy_adjust_dynamics_speczone_tend), so the
    skipped lanes are provably dead; skipping them keeps every gather on
    this path sentinel-free.
    """

    theta = (
        state.rho_theta / state.rho
        if theta_m is None
        else np.asarray(theta_m, dtype=state.rho.dtype)
    )
    theta_saved = (
        saved.rho_theta / saved.rho
        if theta_m_saved is None
        else np.asarray(theta_m_saved, dtype=saved.rho.dtype)
    )
    if theta.shape != state.rho.shape or theta_saved.shape != saved.rho.shape:
        raise ValueError("theta_m overrides must match their state shapes")
    nlev, ncells = theta.shape
    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    area = _mesh_array(mesh, "areaCell").astype(theta.dtype, copy=False)
    dv = _mesh_array(mesh, "dvEdge").astype(theta.dtype, copy=False)
    adv = np.asarray(coefficients.adv_coefs, dtype=theta.dtype)
    adv3 = np.asarray(coefficients.adv_coefs_3rd, dtype=theta.dtype)
    adv_cells = np.asarray(coefficients.adv_cells_for_edge, dtype=np.int64)
    n_adv = np.asarray(coefficients.n_adv_cells_for_edge, dtype=np.int64)
    result = np.zeros((nlev, ncells), dtype=theta.dtype)
    source_inverse = None
    if inv_area_cell is not None:
        source_inverse = np.asarray(inv_area_cell)
        if source_inverse.dtype != theta.dtype:
            raise TypeError(
                f"inv_area_cell dtype {source_inverse.dtype} != RKIND {theta.dtype}"
            )
        if source_inverse.shape != area.shape:
            raise ValueError("inv_area_cell must have one value per cell")
        if not np.all(np.isfinite(source_inverse)):
            raise ValueError("inv_area_cell must be finite")

    if source_inverse is None:
        for edge, (cell0, cell1) in enumerate(cells_on_edge):
            count = int(n_adv[edge])
            stencil = adv_cells[edge, :count]
            sign_velocity = np.copysign(theta.dtype.type(1.0), state.rho_u[:, edge])
            weights = adv[edge, :count][None, :] + (
                theta.dtype.type(third_order_coefficient)
                * sign_velocity[:, None]
                * adv3[edge, :count][None, :]
            )
            edge_theta = np.sum(weights * theta[:, stencil], axis=1)
            flux = state.rho_u[:, edge] * edge_theta
            result[:, cell0] -= flux
            result[:, cell1] += flux
            if rk_step > 1:
                correction = (
                    dv[edge]
                    * (saved.rho_u[:, edge] - state.rho_u[:, edge])
                    * theta.dtype.type(0.5)
                    * (theta_saved[:, cell0] + theta_saved[:, cell1])
                )
                result[:, cell0] -= correction
                result[:, cell1] += correction
        result /= area[None, :]
    else:
        counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
        edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
        signs = edge_signs_on_cells(mesh).astype(theta.dtype, copy=False)
        coefficient = theta.dtype.type(third_order_coefficient)
        one = theta.dtype.type(1.0)
        for cell in range(ncells):
            if skip_cells is not None and skip_cells[cell]:
                continue
            for slot in range(int(counts[cell])):
                edge = int(edges[cell, slot])
                edge_theta = np.zeros(nlev, dtype=theta.dtype)
                sign_velocity = np.copysign(one, state.rho_u[:, edge])
                for stencil_slot in range(int(n_adv[edge])):
                    adv_cell = int(adv_cells[edge, stencil_slot])
                    scalar_weight = (
                        adv[edge, stencil_slot]
                        + coefficient
                        * sign_velocity
                        * adv3[edge, stencil_slot]
                    )
                    edge_theta = (
                        edge_theta + scalar_weight * theta[:, adv_cell]
                    )
                result[:, cell] = result[:, cell] - (
                    signs[cell, slot] * state.rho_u[:, edge] * edge_theta
                )
        if rk_step > 1:
            half = theta.dtype.type(0.5)
            for cell in range(ncells):
                if skip_cells is not None and skip_cells[cell]:
                    continue
                for slot in range(int(counts[cell])):
                    edge = int(edges[cell, slot])
                    cell0, cell1 = cells_on_edge[edge]
                    correction = (
                        signs[cell, slot]
                        * dv[edge]
                        * (saved.rho_u[:, edge] - state.rho_u[:, edge])
                        * half
                        * (theta_saved[:, cell1] + theta_saved[:, cell0])
                    )
                    result[:, cell] = result[:, cell] - correction
        for cell in range(ncells):
            result[:, cell] = result[:, cell] * source_inverse[cell]

    vertical_flux = np.zeros((nlev + 1, ncells), dtype=theta.dtype)
    if nlev >= 2:
        level = 1
        vertical_flux[level] = state.rho_w[level] * (
            vertical.fzm[level] * theta[level] + vertical.fzp[level] * theta[level - 1]
        )
        vertical_flux[level] += (saved.rho_w[level] - state.rho_w[level]) * (
            vertical.fzm[level] * theta_saved[level]
            + vertical.fzp[level] * theta_saved[level - 1]
        )
        for level in range(2, nlev - 1):
            vertical_flux[level] = flux3(
                theta[level - 2],
                theta[level - 1],
                theta[level],
                theta[level + 1],
                state.rho_w[level],
                third_order_coefficient,
            )
            vertical_flux[level] += (saved.rho_w[level] - state.rho_w[level]) * (
                vertical.fzm[level] * theta_saved[level]
                + vertical.fzp[level] * theta_saved[level - 1]
            )
        level = nlev - 1
        if level > 1:
            vertical_flux[level] = saved.rho_w[level] * (
                vertical.fzm[level] * theta[level]
                + vertical.fzp[level] * theta[level - 1]
            )
    result -= np.asarray(vertical.rdzw, dtype=theta.dtype)[:, None] * np.diff(vertical_flux, axis=0)
    return result

def _vertical_momentum_transport(
    mesh: object,
    state: PrognosticState,
    vertical: VerticalGrid,
    coefficients: AdvectionCoefficients,
    terrain_metrics: TerrainMetrics | None = None,
    *,
    third_order_coefficient: float,
    vertical_velocity: FloatArray | None = None,
    inv_area_cell: FloatArray | None = None,
    skip_cells: "NDArray[np.bool_] | None" = None,
) -> FloatArray:
    """Dry w horizontal/vertical transport at frozen lines 5017-5168.

    ``skip_cells`` is the regional specified-zone set; those cells' w
    tendencies are unconditionally zeroed by
    atm_bdy_adjust_dynamics_speczone_tend, so skipping keeps their
    advection stencils (which reach garbage elements) from being gathered.
    """

    nlev, ncells = state.rho.shape
    w = (
        _interface_velocity(
            state,
            vertical,
            mesh,
            terrain_metrics,
            third_order_coefficient=third_order_coefficient,
        )
        if vertical_velocity is None
        else np.asarray(vertical_velocity, dtype=state.rho.dtype)
    )
    if w.shape != (nlev + 1, ncells):
        raise ValueError("vertical_velocity override has the wrong shape")
    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    area = _mesh_array(mesh, "areaCell").astype(state.rho.dtype, copy=False)
    adv = np.asarray(coefficients.adv_coefs, dtype=state.rho.dtype)
    adv3 = np.asarray(coefficients.adv_coefs_3rd, dtype=state.rho.dtype)
    adv_cells = np.asarray(coefficients.adv_cells_for_edge, dtype=np.int64)
    n_adv = np.asarray(coefficients.n_adv_cells_for_edge, dtype=np.int64)
    result = np.zeros_like(state.rho_w)
    source_inverse = None
    if inv_area_cell is not None:
        source_inverse = np.asarray(inv_area_cell)
        if source_inverse.dtype != state.rho.dtype:
            raise TypeError(
                f"inv_area_cell dtype {source_inverse.dtype} != RKIND {state.rho.dtype}"
            )
        if source_inverse.shape != area.shape:
            raise ValueError("inv_area_cell must have one value per cell")
        if not np.all(np.isfinite(source_inverse)):
            raise ValueError("inv_area_cell must be finite")
    if source_inverse is None:
        for edge, (cell0, cell1) in enumerate(cells_on_edge):
            count = int(n_adv[edge])
            stencil = adv_cells[edge, :count]
            for level in range(1, nlev):
                ru_interface = (
                    vertical.fzm[level] * state.rho_u[level, edge]
                    + vertical.fzp[level] * state.rho_u[level - 1, edge]
                )
                weights = adv[edge, :count] + (
                    state.rho.dtype.type(third_order_coefficient)
                    * np.copysign(state.rho.dtype.type(1.0), ru_interface)
                    * adv3[edge, :count]
                )
                edge_w = np.sum(weights * w[level, stencil])
                flux = ru_interface * edge_w
                result[level, cell0] -= flux
                result[level, cell1] += flux
        result /= area[None, :]
    else:
        counts = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
        edges = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
        signs = edge_signs_on_cells(mesh).astype(state.rho.dtype, copy=False)
        coefficient = state.rho.dtype.type(third_order_coefficient)
        one = state.rho.dtype.type(1.0)
        for cell in range(ncells):
            if skip_cells is not None and skip_cells[cell]:
                continue
            for slot in range(int(counts[cell])):
                edge = int(edges[cell, slot])
                for level in range(1, nlev):
                    ru_interface = (
                        vertical.fzm[level] * state.rho_u[level, edge]
                        + vertical.fzp[level] * state.rho_u[level - 1, edge]
                    )
                    edge_w = state.rho.dtype.type(0.0)
                    for stencil_slot in range(int(n_adv[edge])):
                        adv_cell = int(adv_cells[edge, stencil_slot])
                        scalar_weight = (
                            adv[edge, stencil_slot]
                            + coefficient
                            * np.copysign(one, ru_interface)
                            * adv3[edge, stencil_slot]
                        )
                        edge_w = (
                            edge_w
                            + scalar_weight * w[level, adv_cell]
                        )
                    result[level, cell] = result[level, cell] - (
                        signs[cell, slot] * ru_interface * edge_w
                    )
        for cell in range(ncells):
            result[:, cell] = result[:, cell] * source_inverse[cell]

    vertical_flux = np.zeros_like(state.rho_w)
    if nlev >= 2:
        level = 1
        vertical_flux[level] = (
            state.rho.dtype.type(0.25)
            * (state.rho_w[level] + state.rho_w[level - 1])
            * (w[level] + w[level - 1])
        )
        for level in range(2, nlev - 1):
            vertical_flux[level] = flux3(
                w[level - 2],
                w[level - 1],
                w[level],
                w[level + 1],
                state.rho.dtype.type(0.5)
                * (state.rho_w[level] + state.rho_w[level - 1]),
                1.0,
            )
        level = nlev - 1
        if level > 1:
            vertical_flux[level] = (
                state.rho.dtype.type(0.25)
                * (state.rho_w[level] + state.rho_w[level - 1])
                * (w[level] + w[level - 1])
            )
    for level in range(1, nlev):
        result[level] -= vertical.rdzu[level] * (
            vertical_flux[level + 1] - vertical_flux[level]
        )
    result[0] = 0.0
    result[-1] = 0.0
    return result

def vector_invariant_momentum_tendency_v841(
    mesh: object,
    *,
    normal_velocity: FloatArray,
    rho_edge: FloatArray,
    pv_edge: FloatArray,
    kinetic_energy: FloatArray,
    horizontal_divergence: FloatArray,
    reference_wind: V841ReferenceWindProfiles,
    initial_tendency: FloatArray | None = None,
) -> FloatArray:
    """Add the released unconditional perturbation-Coriolis subtraction.

    The second source-order neighbor loop deliberately uses the neighbor
    edge's angle and the target edge's Coriolis parameter.
    """

    u = np.asarray(normal_velocity)
    if u.ndim != 2 or u.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("normal_velocity must be a float32/float64 (level, edge) array")
    if np.shape(rho_edge) != u.shape or np.shape(pv_edge) != u.shape:
        raise ValueError("rho_edge and pv_edge must match velocity")
    reference_wind.validate(u.shape[0], u.dtype)
    cells = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    edges_on_edge = _mesh_array(mesh, "edgesOnEdge").astype(np.int64, copy=False)
    counts = _mesh_array(mesh, "nEdgesOnEdge").astype(np.int64, copy=False)
    weights = _mesh_array(mesh, "weightsOnEdge").astype(u.dtype, copy=False)
    angle_edge = _mesh_array(mesh, "angleEdge").astype(u.dtype, copy=False)
    f_edge = _mesh_array(mesh, "fEdge").astype(u.dtype, copy=False)
    inv_dc = precomputed_mesh_inverse_v841(mesh, "dcEdge", u.dtype)
    u_init = np.asarray(reference_wind.u_init)
    v_init = np.asarray(reference_wind.v_init)
    half = u.dtype.type(0.5)
    result = np.zeros_like(u)
    if initial_tendency is not None and (np.shape(initial_tendency) != u.shape or np.asarray(initial_tendency).dtype != u.dtype):
        raise ValueError("initial_tendency must match velocity shape and dtype")
    for edge, (cell0, cell1) in enumerate(cells):
        q = np.zeros(u.shape[0], dtype=u.dtype)
        for slot in range(int(counts[edge])):
            neighbor_edge = int(edges_on_edge[edge, slot])
            work_pv = half * (pv_edge[:, edge] + pv_edge[:, neighbor_edge])
            q = q + weights[edge, slot] * u[:, neighbor_edge] * work_pv
        for slot in range(int(counts[edge])):
            neighbor_edge = int(edges_on_edge[edge, slot])
            for level in range(u.shape[0]):
                reference_u = (
                    u_init[level] * np.cos(angle_edge[neighbor_edge])
                    - v_init[level] * np.sin(angle_edge[neighbor_edge])
                )
                q[level] = q[level] - (
                    weights[edge, slot] * reference_u * f_edge[edge]
                )
        horizontal = rho_edge[:, edge] * (
            q
            - (kinetic_energy[:, cell1] - kinetic_energy[:, cell0])
            * inv_dc[edge]
        )
        # MPAS adds the vertical transport before subtracting divergence.
        if initial_tendency is not None:
            horizontal = initial_tendency[:, edge] + horizontal
        result[:, edge] = horizontal - u[:, edge] * half * (
            horizontal_divergence[:, cell0] + horizontal_divergence[:, cell1]
        )
    return result
