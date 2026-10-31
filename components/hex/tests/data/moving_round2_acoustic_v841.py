"""Scalar reference from d620bf0c; same repository Apache-2.0 license."""

def solve_vertical_system_v841(
    rhs: FloatArray,
    coefficients: VerticalImplicitCoefficientsV841,
    *,
    dts: float,
) -> FloatArray:
    """Apply the released v8.4.1 dts-squared Thomas sweep."""

    result = np.asarray(rhs).copy()
    nlev, ncells = coefficients.a_tri.shape
    if result.shape != (nlev + 1, ncells):
        raise ValueError(f"rhs shape {result.shape} != ({nlev + 1}, {ncells})")
    if result.dtype != coefficients.a_tri.dtype:
        raise TypeError("rhs and coefficient dtypes differ")
    dts_r = result.dtype.type(dts)
    dts2 = dts_r**2
    for cell in range(ncells):
        for level in range(1, nlev):
            result[level, cell] = (
                result[level, cell]
                - dts2 * coefficients.a_tri[level, cell] * result[level - 1, cell]
            ) * coefficients.alpha_tri[level, cell]
        for level in range(nlev - 1, -1, -1):
            result[level, cell] = result[level, cell] - (
                coefficients.gamma_tri[level, cell] * result[level + 1, cell]
            )
    return result

def advance_acoustic_step_v841(
    mesh: object,
    state: AcousticStepState,
    forcing: AcousticStepForcing,
    coefficients: VerticalImplicitCoefficientsV841,
    *,
    dts: float,
    small_step: int,
    offcentering: AcousticOffcenteringV841,
    fzm: FloatArray,
    fzp: FloatArray,
    rdzw: FloatArray,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
    specified_zone_edge: FloatArray | None = None,
    specified_zone_cell: FloatArray | None = None,
) -> AcousticStepState:
    """Advance one closed-mesh v8.4.1 acoustic substep."""

    if small_step < 1:
        raise ConfigurationRefusal(
            "small_step", small_step, "MPAS acoustic substeps are numbered from one", "small_step=1"
        )
    out = state.copy()
    nlev, nedges = out.ru_p.shape
    ncells = out.rho_pp.shape[1]
    if out.rw_p.shape != (nlev + 1, ncells):
        raise ValueError("rw_p must have shape (nVertLevels+1, nCells)")
    dtype = out.rho_pp.dtype
    if any(
        np.asarray(getattr(out, name)).dtype != dtype
        for name in out.__slots__
    ):
        raise TypeError("all acoustic state arrays must share one dtype")
    etp, etm, ewp, ewm = _validate_offcentering(
        offcentering, nlev=nlev, dtype=dtype
    )
    # The regional specified-zone masks (mpas_atm_setup_bdy_masks): REAL
    # arrays, 1.0 on rings 6-7, 0.0 elsewhere.  A nonzero mask activates the
    # transcribed limited-area acoustic branches below
    # (atm_advance_acoustic_step_work F:3909, F:3992-4103).
    spec_edge = None
    spec_cell = None
    if specified_zone_edge is not None:
        spec_edge = np.asarray(specified_zone_edge)
        if spec_edge.shape != (nedges,):
            raise ValueError("specified_zone_edge must have one value per edge")
        if spec_edge.dtype != dtype:
            spec_edge = spec_edge.astype(dtype)
    if specified_zone_cell is not None:
        spec_cell = np.asarray(specified_zone_cell)
        if spec_cell.shape != (ncells,):
            raise ValueError("specified_zone_cell must have one value per cell")
        if spec_cell.dtype != dtype:
            spec_cell = spec_cell.astype(dtype)
    if (spec_edge is None) != (spec_cell is None):
        raise ConfigurationRefusal(
            "specified_zone_cell",
            None,
            "the regional acoustic branch needs both masks or neither",
            "specified_zone_edge and specified_zone_cell together",
        )

    cells_on_edge = _mesh_array(mesh, "cellsOnEdge").astype(np.int64, copy=False)
    edges_on_cell = _mesh_array(mesh, "edgesOnCell").astype(np.int64, copy=False)
    n_edges_on_cell = _mesh_array(mesh, "nEdgesOnCell").astype(np.int64, copy=False)
    inv_dc_edge = np.reciprocal(
        _mesh_array(mesh, "dcEdge").astype(dtype, copy=False)
    )
    dv_edge = _mesh_array(mesh, "dvEdge").astype(dtype, copy=False)
    inv_area_cell = np.reciprocal(
        _mesh_array(mesh, "areaCell").astype(dtype, copy=False)
    )
    signs = edge_signs_on_cells(mesh).astype(dtype, copy=False)
    dts_r = dtype.type(dts)
    half = dtype.type(0.5)
    one = dtype.type(1.0)
    gravity_r = dtype.type(gravity)
    rcv = dtype.type(rgas) / (dtype.type(cp) - dtype.type(rgas))
    c2 = dtype.type(cp) * rcv
    fzm_a = np.asarray(fzm, dtype=dtype)
    fzp_a = np.asarray(fzp, dtype=dtype)
    rdzw_a = np.asarray(rdzw, dtype=dtype)

    if small_step != 1:
        for edge in range(nedges):
            cell0, cell1 = cells_on_edge[edge]
            if cell0 < 0 or cell1 < 0:
                if spec_edge is None or spec_edge[edge] != one:
                    raise ConfigurationRefusal(
                        "regional_boundary_state",
                        f"edge {edge}",
                        "the acoustic pressure gradient needs both edge cells",
                        "explicit lateral boundary cells before this substep",
                    )
                # Ring-7 one-cell edge: specZoneMaskEdge is 1.0, so native
                # multiplies its garbage-gathered pgrad by exactly zero
                # (F:3909) and the update is the driving tendency alone.
                # ru/u at these edges are then overwritten from driving
                # state after recover, so the lane is dead regardless.
                out.ru_p[:, edge] = out.ru_p[:, edge] + dts_r * (
                    forcing.tend_ru[:, edge]
                )
                out.ru_avg[:, edge] = out.ru_avg[:, edge] + out.ru_p[:, edge]
                continue
            pgrad = (
                (out.rtheta_pp[:, cell1] - out.rtheta_pp[:, cell0])
                * inv_dc_edge[edge]
            ) / (half * (forcing.zz[:, cell1] + forcing.zz[:, cell0]))
            pgrad = (
                forcing.cqu[:, edge]
                * half
                * c2
                * (forcing.exner[:, cell0] + forcing.exner[:, cell1])
                * pgrad
            )
            pgrad = pgrad + (
                half
                * forcing.zxu[:, edge]
                * gravity_r
                * (out.rho_pp[:, cell0] + out.rho_pp[:, cell1])
            )
            if spec_edge is None:
                out.ru_p[:, edge] = out.ru_p[:, edge] + dts_r * (
                    forcing.tend_ru[:, edge] - pgrad
                )
            else:
                # F:3909: ru_p += dts*(tend_ru - (1-specZoneMaskEdge)*pgrad);
                # a specified-zone edge integrates its driving tendency with
                # the pressure gradient masked off.
                out.ru_p[:, edge] = out.ru_p[:, edge] + dts_r * (
                    forcing.tend_ru[:, edge]
                    - (one - spec_edge[edge]) * pgrad
                )
            out.ru_avg[:, edge] = out.ru_avg[:, edge] + out.ru_p[:, edge]
    else:
        out.ru_p[:] = dts_r * forcing.tend_ru
        out.ru_avg[:] = out.ru_p

    if small_step == 1:
        out.rtheta_pp_old.fill(dtype.type(0.0))
    else:
        out.rtheta_pp_old[:] = out.rtheta_pp

    for cell in range(ncells):
        if small_step == 1:
            out.ww_avg[:, cell] = dtype.type(0.0)
            out.rho_pp[:, cell] = dtype.type(0.0)
            out.rtheta_pp[:, cell] = dtype.type(0.0)
            out.rw_p[:, cell] = dtype.type(0.0)

        if spec_cell is not None and spec_cell[cell] != dtype.type(0.0):
            # F:4093-4103, the specified zone in regional_MPAS: no rs/ts
            # fluxes, no vertically implicit solve: the driving tendencies
            # integrate forward over rows k=1..nVertLevels only, and wwAvg
            # accumulates ewp-weighted rw_p over the same rows (including
            # row 1, unlike the interior branch).
            out.rho_pp[:, cell] = out.rho_pp[:, cell] + dts_r * forcing.tend_rho[:, cell]
            out.rtheta_pp[:, cell] = (
                out.rtheta_pp[:, cell] + dts_r * forcing.tend_rt[:, cell]
            )
            out.rw_p[:nlev, cell] = (
                out.rw_p[:nlev, cell] + dts_r * forcing.tend_rw[:nlev, cell]
            )
            out.ww_avg[:nlev, cell] = out.ww_avg[:nlev, cell] + (
                ewp[:nlev] * out.rw_p[:nlev, cell]
            )
            continue

        rs = np.zeros(nlev, dtype=dtype)
        ts = np.zeros(nlev, dtype=dtype)
        for slot in range(int(n_edges_on_cell[cell])):
            edge = int(edges_on_cell[cell, slot])
            cell0, cell1 = cells_on_edge[edge]
            flux = (
                signs[cell, slot]
                * dts_r
                * dv_edge[edge]
                * out.ru_p[:, edge]
                * inv_area_cell[cell]
            )
            rs = rs - flux
            ts = ts - flux * half * (
                forcing.theta_m[:, cell1] + forcing.theta_m[:, cell0]
            )

        rs = (
            out.rho_pp[:, cell]
            + dts_r * forcing.tend_rho[:, cell]
            + rs
            - dts_r
            * coefficients.cofrz
            * (
                ewm[1:] * out.rw_p[1:, cell]
                - ewm[:-1] * out.rw_p[:-1, cell]
            )
        )
        ts = (
            out.rtheta_pp[:, cell]
            + dts_r * forcing.tend_rt[:, cell]
            + ts
            - dts_r
            * rdzw_a
            * (
                ewm[1:] * coefficients.coftz[1:, cell] * out.rw_p[1:, cell]
                - ewm[:-1] * coefficients.coftz[:-1, cell] * out.rw_p[:-1, cell]
            )
        )
        out.ww_avg[1:nlev, cell] = out.ww_avg[1:nlev, cell] + (
            ewm[1:nlev] * out.rw_p[1:nlev, cell]
        )
        for level in range(1, nlev):
            out.rw_p[level, cell] = out.rw_p[level, cell] + dts_r * (
                forcing.tend_rw[level, cell]
                - coefficients.cofwz[level, cell]
                * (
                    (
                        etp[level] * forcing.zz[level, cell] * ts[level]
                        - etp[level - 1]
                        * forcing.zz[level - 1, cell]
                        * ts[level - 1]
                    )
                    + (
                        etm[level]
                        * forcing.zz[level, cell]
                        * out.rtheta_pp[level, cell]
                        - etm[level - 1]
                        * forcing.zz[level - 1, cell]
                        * out.rtheta_pp[level - 1, cell]
                    )
                )
                - coefficients.cofwr[level, cell]
                * (
                    (
                        etp[level] * rs[level]
                        + etp[level - 1] * rs[level - 1]
                    )
                    + (
                        etm[level] * out.rho_pp[level, cell]
                        + etm[level - 1] * out.rho_pp[level - 1, cell]
                    )
                )
                + coefficients.cofwt[level, cell]
                * (
                    etp[level] * ts[level]
                    + etm[level] * out.rtheta_pp[level, cell]
                )
                + coefficients.cofwt[level - 1, cell]
                * (
                    etp[level - 1] * ts[level - 1]
                    + etm[level - 1] * out.rtheta_pp[level - 1, cell]
                )
            )

        solved = solve_vertical_system_v841(
            out.rw_p[:, cell : cell + 1],
            _coefficients_for_cell(coefficients, cell),
            dts=dts,
        )
        out.rw_p[:, cell] = solved[:, 0]
        for level in range(1, nlev):
            delta_saved = forcing.rw_save[level, cell] - forcing.rw[level, cell]
            damping = forcing.dss[level, cell]
            zz_at_interface = (
                fzm_a[level] * forcing.zz[level, cell]
                + fzp_a[level] * forcing.zz[level - 1, cell]
            )
            rho_at_interface = (
                fzm_a[level] * forcing.rho_zz[level, cell]
                + fzp_a[level] * forcing.rho_zz[level - 1, cell]
            )
            out.rw_p[level, cell] = (
                out.rw_p[level, cell]
                + delta_saved
                - dts_r * damping * zz_at_interface * rho_at_interface * forcing.w[level, cell]
            ) / (one + dts_r * damping) - delta_saved
        out.ww_avg[1:nlev, cell] = out.ww_avg[1:nlev, cell] + (
            ewp[1:nlev] * out.rw_p[1:nlev, cell]
        )
        out.rho_pp[:, cell] = rs - dts_r * coefficients.cofrz * (
            ewp[1:] * out.rw_p[1:, cell]
            - ewp[:-1] * out.rw_p[:-1, cell]
        )
        out.rtheta_pp[:, cell] = ts - dts_r * rdzw_a * (
            ewp[1:] * coefficients.coftz[1:, cell] * out.rw_p[1:, cell]
            - ewp[:-1] * coefficients.coftz[:-1, cell] * out.rw_p[:-1, cell]
        )
    return out
