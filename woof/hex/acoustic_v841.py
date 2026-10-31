"""MPAS-A v8.4.1 vertically implicit acoustic integration.

This additive module keeps the frozen v8.2.3 implementation byte-stable while
transcribing the released v8.4.1 ``etp/etm/ewp/ewm`` equations from
``mpas_atm_time_integration.F:3366-3507,3783-4109``.  The public driver only
routes an explicitly selected v8.4.1 configuration here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .acoustic import AcousticStepForcing, AcousticStepState, edge_signs_on_cells
from .errors import ConfigurationRefusal
from .offcentering_v841 import AcousticOffcenteringV841

FloatArray = NDArray[np.floating[Any]]


def _mesh_array(mesh: object, name: str) -> NDArray[Any]:
    try:
        return np.asarray(getattr(mesh, name))
    except AttributeError:
        arrays = getattr(mesh, "arrays", None)
        if arrays is None or name not in arrays:
            raise AttributeError(f"mesh has no MPAS field {name!r}") from None
        return np.asarray(arrays[name])


@dataclass(frozen=True, slots=True)
class VerticalImplicitCoefficientsV841:
    """Unscaled v8.4.1 acoustic coefficients and Thomas factors."""

    cofwr: FloatArray
    cofwz: FloatArray
    coftz: FloatArray
    cofwt: FloatArray
    cofrz: FloatArray
    a_tri: FloatArray
    b_tri: FloatArray
    c_tri: FloatArray
    alpha_tri: FloatArray
    gamma_tri: FloatArray


def _require_shape(reference: FloatArray, **arrays: FloatArray) -> None:
    for name, value in arrays.items():
        if np.shape(value) != np.shape(reference):
            raise ValueError(f"{name} shape {np.shape(value)} != {np.shape(reference)}")


def _validate_offcentering(
    profile: AcousticOffcenteringV841,
    *,
    nlev: int,
    dtype: np.dtype[Any],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    expected = {
        "etp": (nlev,),
        "etm": (nlev,),
        "ewp": (nlev + 1,),
        "ewm": (nlev + 1,),
    }
    values: dict[str, FloatArray] = {}
    for name, shape in expected.items():
        value = np.asarray(getattr(profile, name))
        if value.shape != shape:
            raise ValueError(f"{name} shape {value.shape} != {shape}")
        if value.dtype != dtype:
            raise TypeError(f"{name} dtype {value.dtype} != coefficient dtype {dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")
        values[name] = value
    return values["etp"], values["etm"], values["ewp"], values["ewm"]


def compute_vertical_implicit_coefficients_v841(
    *,
    dts: float,
    offcentering: AcousticOffcenteringV841,
    zz: FloatArray,
    cqw: FloatArray,
    exner: FloatArray,
    theta: FloatArray,
    rho_base: FloatArray,
    rho_theta_base: FloatArray,
    exner_base: FloatArray,
    rho_theta_perturbation: FloatArray,
    qtot: FloatArray,
    rdzw: FloatArray,
    fzm: FloatArray,
    fzp: FloatArray,
    rdzu: FloatArray,
    gravity: float = 9.80616,
    rgas: float = 287.0,
    cp: float = 1004.5,
) -> VerticalImplicitCoefficientsV841:
    """Transcribe v8.4.1 vertical coefficients in released source order."""

    zz = np.asarray(zz)
    if zz.ndim != 2 or zz.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("zz must be a float32/float64 (nVertLevels, nCells) array")
    dtype = zz.dtype
    arrays = {
        "cqw": np.asarray(cqw),
        "exner": np.asarray(exner),
        "theta": np.asarray(theta),
        "rho_base": np.asarray(rho_base),
        "rho_theta_base": np.asarray(rho_theta_base),
        "exner_base": np.asarray(exner_base),
        "rho_theta_perturbation": np.asarray(rho_theta_perturbation),
        "qtot": np.asarray(qtot),
    }
    _require_shape(zz, **arrays)
    for name, value in arrays.items():
        if value.dtype != dtype:
            raise TypeError(f"{name} dtype {value.dtype} != zz dtype {dtype}")
    nlev, ncells = zz.shape
    metrics: dict[str, FloatArray] = {}
    for name, metric in {"rdzw": rdzw, "fzm": fzm, "fzp": fzp, "rdzu": rdzu}.items():
        value = np.asarray(metric)
        if value.shape != (nlev,):
            raise ValueError(f"{name} shape {value.shape} != ({nlev},)")
        if value.dtype != dtype:
            raise TypeError(f"{name} dtype {value.dtype} != zz dtype {dtype}")
        metrics[name] = value
    if not np.isfinite(dts) or dts <= 0.0:
        raise ConfigurationRefusal("dts", dts, "the acoustic step must be positive", "dts>0")
    if cp == rgas:
        raise ValueError("cp-rgas must be nonzero")
    if np.any(dtype.type(1.0) + arrays["qtot"] == dtype.type(0.0)):
        raise ValueError("1+qtot is zero")
    etp, _etm, ewp, _ewm = _validate_offcentering(
        offcentering, nlev=nlev, dtype=dtype
    )

    rdzw_a = metrics["rdzw"]
    fzm_a = metrics["fzm"]
    fzp_a = metrics["fzp"]
    rdzu_a = metrics["rdzu"]
    half = dtype.type(0.5)
    one = dtype.type(1.0)
    gravity_r = dtype.type(gravity)
    rcv = dtype.type(rgas) / (dtype.type(cp) - dtype.type(rgas))
    c2 = dtype.type(cp) * rcv
    dts_r = dtype.type(dts)
    dts2 = dts_r**2

    cofrz = rdzw_a.copy()
    cofwr = np.zeros_like(zz)
    cofwz = np.zeros_like(zz)
    coftz = np.zeros((nlev + 1, ncells), dtype=dtype)
    cofwt = np.zeros_like(zz)
    for level in range(1, nlev):
        interp_zz = fzm_a[level] * zz[level] + fzp_a[level] * zz[level - 1]
        cofwr[level] = half * gravity_r * interp_zz
        cofwz[level] = (
            c2
            * interp_zz
            * rdzu_a[level]
            * arrays["cqw"][level]
            * (
                fzm_a[level] * arrays["exner"][level]
                + fzp_a[level] * arrays["exner"][level - 1]
            )
        )
        coftz[level] = (
            fzm_a[level] * arrays["theta"][level]
            + fzp_a[level] * arrays["theta"][level - 1]
        )
    cofwt[:] = (
        half
        * rcv
        * zz
        * gravity_r
        * arrays["rho_base"]
        / (one + arrays["qtot"])
        * arrays["exner"]
        / (
            (arrays["rho_theta_base"] + arrays["rho_theta_perturbation"])
            * arrays["exner_base"]
        )
    )

    a_tri = np.zeros_like(zz)
    b_tri = np.zeros_like(zz)
    c_tri = np.zeros_like(zz)
    alpha_tri = np.zeros_like(zz)
    gamma_tri = np.zeros_like(zz)
    b_tri[0] = one
    for level in range(1, nlev):
        a_tri[level] = (
            -cofwz[level]
            * coftz[level - 1]
            * rdzw_a[level - 1]
            * zz[level - 1]
            + cofwr[level] * cofrz[level - 1]
            - cofwt[level - 1] * coftz[level - 1] * rdzw_a[level - 1]
        )
        a_tri[level] = a_tri[level] * etp[level - 1] * ewp[level - 1]

        b_tri[level] = (
            cofwz[level]
            * coftz[level]
            * (
                etp[level] * rdzw_a[level] * zz[level]
                + etp[level - 1] * rdzw_a[level - 1] * zz[level - 1]
            )
            - coftz[level]
            * (
                etp[level] * cofwt[level] * rdzw_a[level]
                - etp[level - 1] * cofwt[level - 1] * rdzw_a[level - 1]
            )
            + cofwr[level]
            * (
                etp[level] * cofrz[level]
                - etp[level - 1] * cofrz[level - 1]
            )
        )
        b_tri[level] = b_tri[level] * ewp[level]

        c_tri[level] = (
            -cofwz[level]
            * coftz[level + 1]
            * rdzw_a[level]
            * zz[level]
            - cofwr[level] * cofrz[level]
            + cofwt[level] * coftz[level + 1] * rdzw_a[level]
        )
        c_tri[level] = c_tri[level] * etp[level] * ewp[level + 1]
    if nlev:
        c_tri[nlev - 1] = dtype.type(0.0)

    for level in range(1, nlev):
        denominator = one + dts2 * (
            b_tri[level] - a_tri[level] * gamma_tri[level - 1]
        )
        if np.any(denominator == dtype.type(0.0)):
            raise FloatingPointError(f"singular vertical acoustic system at level {level}")
        alpha_tri[level] = one / denominator
        gamma_tri[level] = dts2 * c_tri[level] * alpha_tri[level]

    return VerticalImplicitCoefficientsV841(
        cofwr=cofwr,
        cofwz=cofwz,
        coftz=coftz,
        cofwt=cofwt,
        cofrz=cofrz,
        a_tri=a_tri,
        b_tri=b_tri,
        c_tri=c_tri,
        alpha_tri=alpha_tri,
        gamma_tri=gamma_tri,
    )


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
    for level in range(1, nlev):
        result[level, :] = (
            result[level, :]
            - dts2 * coefficients.a_tri[level, :] * result[level - 1, :]
        ) * coefficients.alpha_tri[level, :]
    for level in range(nlev - 1, -1, -1):
        result[level, :] = result[level, :] - (
            coefficients.gamma_tri[level, :] * result[level + 1, :]
        )
    return result


def _coefficients_for_cell(
    coefficients: VerticalImplicitCoefficientsV841, cell: int
) -> VerticalImplicitCoefficientsV841:
    values: dict[str, FloatArray] = {}
    for name in coefficients.__slots__:
        value = np.asarray(getattr(coefficients, name))
        values[name] = value if value.ndim == 1 else value[:, cell : cell + 1]
    return VerticalImplicitCoefficientsV841(**values)


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
        valid = np.all(cells_on_edge >= 0, axis=1)
        boundary = np.flatnonzero(~valid)
        if boundary.size:
            invalid = boundary if spec_edge is None else boundary[spec_edge[boundary] != one]
            if invalid.size:
                raise ConfigurationRefusal(
                    "regional_boundary_state", f"edge {invalid[0]}",
                    "the acoustic pressure gradient needs both edge cells",
                    "explicit lateral boundary cells before this substep")
            out.ru_p[:, boundary] = out.ru_p[:, boundary] + dts_r * forcing.tend_ru[:, boundary]
            out.ru_avg[:, boundary] = out.ru_avg[:, boundary] + out.ru_p[:, boundary]
        edge = np.flatnonzero(valid)
        cell0, cell1 = cells_on_edge[edge].T
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

    if small_step == 1:
        out.ww_avg.fill(dtype.type(0.0))
        out.rho_pp.fill(dtype.type(0.0))
        out.rtheta_pp.fill(dtype.type(0.0))
        out.rw_p.fill(dtype.type(0.0))
    if spec_cell is not None:
        specified = np.flatnonzero(spec_cell != dtype.type(0.0))
        out.rho_pp[:, specified] = out.rho_pp[:, specified] + dts_r * forcing.tend_rho[:, specified]
        out.rtheta_pp[:, specified] = out.rtheta_pp[:, specified] + dts_r * forcing.tend_rt[:, specified]
        out.rw_p[:nlev, specified] = out.rw_p[:nlev, specified] + dts_r * forcing.tend_rw[:nlev, specified]
        out.ww_avg[:nlev, specified] = out.ww_avg[:nlev, specified] + ewp[:nlev, None] * out.rw_p[:nlev, specified]
    cell = np.arange(ncells) if spec_cell is None else np.flatnonzero(spec_cell == dtype.type(0.0))
    rs = np.zeros((nlev, len(cell)), dtype=dtype)
    ts = np.zeros_like(rs)
    # No reduction reassociation: each column consumes its original slot order.
    for slot in range(edges_on_cell.shape[1]):
        lanes = np.flatnonzero(n_edges_on_cell[cell] > slot)
        selected = cell[lanes]
        edge = edges_on_cell[selected, slot]
        cell0, cell1 = cells_on_edge[edge].T
        flux = signs[selected, slot] * dts_r * dv_edge[edge] * out.ru_p[:, edge] * inv_area_cell[selected]
        rs[:, lanes] = rs[:, lanes] - flux
        ts[:, lanes] = ts[:, lanes] - flux * half * (forcing.theta_m[:, cell1] + forcing.theta_m[:, cell0])
    rs = (
        out.rho_pp[:, cell]
        + dts_r * forcing.tend_rho[:, cell]
        + rs
        - dts_r
        * coefficients.cofrz[:, None]
        * (
            ewm[1:, None] * out.rw_p[1:, cell]
            - ewm[:-1, None] * out.rw_p[:-1, cell]
        )
    )
    ts = (
        out.rtheta_pp[:, cell]
        + dts_r * forcing.tend_rt[:, cell]
        + ts
        - dts_r
        * rdzw_a[:, None]
        * (
            ewm[1:, None] * coefficients.coftz[1:, cell] * out.rw_p[1:, cell]
            - ewm[:-1, None] * coefficients.coftz[:-1, cell] * out.rw_p[:-1, cell]
        )
    )
    out.ww_avg[1:nlev, cell] = out.ww_avg[1:nlev, cell] + (
        ewm[1:nlev, None] * out.rw_p[1:nlev, cell]
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
        out.rw_p[:, cell],
        VerticalImplicitCoefficientsV841(**{name: (value if value.ndim == 1 else value[:, cell])
            for name in coefficients.__slots__ for value in (np.asarray(getattr(coefficients, name)),)}),
        dts=dts,
    )
    out.rw_p[:, cell] = solved
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
        ewp[1:nlev, None] * out.rw_p[1:nlev, cell]
    )
    out.rho_pp[:, cell] = rs - dts_r * coefficients.cofrz[:, None] * (
        ewp[1:, None] * out.rw_p[1:, cell]
        - ewp[:-1, None] * out.rw_p[:-1, cell]
    )
    out.rtheta_pp[:, cell] = ts - dts_r * rdzw_a[:, None] * (
        ewp[1:, None] * coefficients.coftz[1:, cell] * out.rw_p[1:, cell]
        - ewp[:-1, None] * coefficients.coftz[:-1, cell] * out.rw_p[:-1, cell]
    )
    return out


__all__ = [
    "VerticalImplicitCoefficientsV841",
    "advance_acoustic_step_v841",
    "compute_vertical_implicit_coefficients_v841",
    "solve_vertical_system_v841",
]
