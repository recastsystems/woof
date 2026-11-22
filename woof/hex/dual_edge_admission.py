"""Dual-edge (Voronoi) geometry admission for registered MPAS meshes.

WHAT THIS PREVENTS, MEASURED.  A generated mesh whose Delaunay carries
pentagon-heptagon dislocation pairs has near-cocircular quads: two Voronoi
vertices nearly coincide while their two cells stay a full spacing apart, so
``dvEdge`` collapses while ``dcEdge`` does not.  The TRiSK operators divide by
``dvEdge``.  The first of them to run in a v8.4.1 outer step is the
potential-vorticity gradient in ``pv_apvm_v841_f32``
(``src/hexcore/cuda_horizontal_v841.py:215-217``)::

    grad_pv_tangential = (pv_vertex[v1] - pv_vertex[v0]) * inv_dv_edge[edge]

so a tangential gradient across such an edge is amplified by ``dcEdge/dvEdge``
relative to a normal gradient across the same cell pair.

Measured on the registered generated mesh ``v15.150.38857`` (an RTX 5070 Ti, RTX
5070 Ti, 2026-08-24): edge 19786 carries ``dvEdge`` 6.514 m against
``dcEdge`` 38,657 m, ratio 1.685e-04, amplification 5,935x.  A per-launch
non-finite probe over the first outer step puts every runaway magnitude at
that one edge -- ``grad_pv_tangential`` grows 3.5e4x there, then
``vector_momentum``, then ``acoustic_ru`` to 1.1e6 -- and the first
non-finite value in the whole run is ``exner`` at cell 6461, an immediate
neighbour of both cells of that edge, produced by ``powf`` of a negative
mass-weighted potential temperature (``cuda_driver.py:1216-1221``).  The
outer step is then refused with a single four-byte validation flag that
names nothing.

THE FLOOR IS THE RATIO, NOT THE LENGTH.  Amplification is ``dc/dv``, which is
scale-free, so a finer mesh is not penalised for being fine.  The published
family measures 0.394477 (``x1.40962``) and 0.033650 (``x4.163842``); the
generator refuses its own emissions below 0.02 for the same reason.  The
same 0.02 is used here when APVM is active. With APVM disabled the cited
gradient kernel is not launched, so the ratio is recorded and positive
geometry remains required. This does not qualify a forecast on its own. An absolute
minimum ``dvEdge`` floor is deliberately NOT applied: the frozen native
``x4.163842`` measures 1,170 m and integrates, so an absolute floor would
refuse the correctness anchor itself.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "DualEdgeAdmissionError",
    "DualEdgePolicy",
    "DualEdgeAdmission",
    "admit_dual_edges",
]


class DualEdgeAdmissionError(RuntimeError):
    """The mesh's dual edges cannot carry the TRiSK operators safely."""


@dataclass(frozen=True, slots=True)
class DualEdgePolicy:
    schema: str = "gpuwm-hex.dual-edge-admission/v1"
    minimum_dv_over_dc: float = 0.02
    apvm_upwinding: float = 0.5
    description: str = (
        "TRiSK tangential terms divide by dvEdge; the amplification of a "
        "tangential gradient over a normal one is dcEdge/dvEdge. The floor sits "
        "between the published family's roughest measured reading (0.033650 on "
        "x4.163842) and the measured generated-mesh dislocation class, and is "
        "the same floor the mesh generator refuses its own emissions below"
    )

    def validate(self) -> None:
        if not math.isfinite(self.apvm_upwinding) or self.apvm_upwinding < 0.0:
            raise DualEdgeAdmissionError("APVM coefficient must be finite and nonnegative")
        if (
            not math.isfinite(self.minimum_dv_over_dc)
            or not 0.0 < self.minimum_dv_over_dc < 1.0
        ):
            raise DualEdgeAdmissionError(
                "dual-edge policy minimum_dv_over_dc must lie in (0, 1)"
            )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {**asdict(self), "ratio_floor_active": self.apvm_upwinding > 0.0}


@dataclass(frozen=True, slots=True)
class DualEdgeAdmission:
    count: int
    minimum_ratio: float
    minimum_ratio_edge: int
    minimum_ratio_dv_edge_m: float
    minimum_ratio_dc_edge_m: float
    amplification: float
    percentile_0_01_ratio: float
    percentile_0_1_ratio: float
    percentile_1_ratio: float
    median_ratio: float
    minimum_dv_edge_m: float
    edges_below_floor: int
    policy: DualEdgePolicy

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy"] = self.policy.as_dict()
        return payload


def _physical(name: str, values: object, count: int | None) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise DualEdgeAdmissionError(
            f"{name} must be a non-empty one-dimensional nEdges array, got {raw.shape}"
        )
    if raw.dtype.kind != "f":
        raise DualEdgeAdmissionError(
            f"{name} must be floating point physical metres, got {raw.dtype}"
        )
    if count is not None and raw.size != count:
        raise DualEdgeAdmissionError(
            f"dvEdge and dcEdge disagree on nEdges: {raw.size} != {count}"
        )
    return np.asarray(raw, dtype=np.float64)


def admit_dual_edges(
    dv_edge: object,
    dc_edge: object,
    *,
    policy: DualEdgePolicy | None = None,
    cells_on_edge: object | None = None,
    cells_on_edge_base: int = 1,
    mesh_name: str | None = None,
) -> DualEdgeAdmission:
    """Admit or refuse a mesh's dual edges before any device allocation.

    ``dcEdge`` is validated as strictly positive because it divides; ``dvEdge``
    is allowed to be zero on input only so the refusal can report the real
    value rather than a filtered one.
    """

    selected = policy or DualEdgePolicy()
    selected.validate()

    dc = _physical("dcEdge", dc_edge, None)
    dv = _physical("dvEdge", dv_edge, dc.size)
    if not np.all(np.isfinite(dc)) or not np.all(dc > 0.0):
        bad = np.flatnonzero(~np.isfinite(dc) | (dc <= 0.0))
        raise DualEdgeAdmissionError(
            f"dcEdge carries {bad.size} non-finite or non-positive lengths "
            f"(first edges {bad[:8].tolist()}); the mesh geometry is corrupt, not merely rough"
        )
    if not np.all(np.isfinite(dv)) or np.any(dv < 0.0):
        bad = np.flatnonzero(~np.isfinite(dv) | (dv < 0.0))
        raise DualEdgeAdmissionError(
            f"dvEdge carries {bad.size} non-finite or negative lengths "
            f"(first edges {bad[:8].tolist()}); the mesh geometry is corrupt, not merely rough"
        )

    ratio = dv / dc
    worst = int(np.argmin(ratio))
    minimum = float(ratio[worst])
    below = int(np.count_nonzero(ratio < selected.minimum_dv_over_dc))
    amplification = float("inf") if minimum <= 0.0 else float(1.0 / minimum)
    quantiles = np.percentile(ratio, [0.01, 0.1, 1.0, 50.0], method="linear")

    admission = DualEdgeAdmission(
        count=int(ratio.size),
        minimum_ratio=minimum,
        minimum_ratio_edge=worst,
        minimum_ratio_dv_edge_m=float(dv[worst]),
        minimum_ratio_dc_edge_m=float(dc[worst]),
        amplification=amplification,
        percentile_0_01_ratio=float(quantiles[0]),
        percentile_0_1_ratio=float(quantiles[1]),
        percentile_1_ratio=float(quantiles[2]),
        median_ratio=float(quantiles[3]),
        minimum_dv_edge_m=float(dv.min()),
        edges_below_floor=below,
        policy=selected,
    )
    if selected.apvm_upwinding == 0.0:
        # The measured runaway mechanism is the APVM tangential gradient.
        # cuda_horizontal_v841.py launches pv_apvm_v841_f32 only for a
        # positive coefficient. UCAR requires APVM=0 for its dense refined
        # meshes. Keep positivity for the other inverse-geometry consumers;
        # record roughness rather than refusing an operator that is not run.
        if minimum <= 0.0:
            raise DualEdgeAdmissionError(
                f"dual edge {worst} has zero dvEdge; disabling APVM does not "
                "make zero inverse-geometry denominators valid")
        return admission
    if minimum >= selected.minimum_dv_over_dc:
        return admission

    cells = ""
    if cells_on_edge is not None:
        pair = np.asarray(cells_on_edge)
        if pair.ndim == 2 and pair.shape[0] == ratio.size and pair.shape[1] == 2:
            # The base is stated because both live in this tree: a file's
            # cellsOnEdge is one-based, the port's authority representation is
            # zero-based, and two receipts for the SAME edge would otherwise
            # print cell numbers that differ by one with nothing saying why.
            base = "one-based" if int(cells_on_edge_base) == 1 else "zero-based"
            cells = (
                f" joining cells {pair[worst, 0]} and {pair[worst, 1]} ({base})"
            )
    label = f"mesh {mesh_name!r}: " if mesh_name else ""
    raise DualEdgeAdmissionError(
        f"{label}dual-edge geometry refused before CUDA allocation. "
        f"Edge {worst}{cells} has dvEdge={admission.minimum_ratio_dv_edge_m:.6g} m "
        f"against dcEdge={admission.minimum_ratio_dc_edge_m:.6g} m, "
        f"dvEdge/dcEdge={minimum:.6g}, below the admitted floor "
        f"{selected.minimum_dv_over_dc:.6g}; {below} of {ratio.size} edges are below it. "
        "THE BREAKAGE THIS PREVENTS: the TRiSK tangential terms divide by dvEdge -- "
        "pv_apvm_v841_f32 forms grad_pv_tangential = (pv_vertex[v1]-pv_vertex[v0])/dvEdge "
        "(src/hexcore/cuda_horizontal_v841.py:215-217) -- so a tangential gradient "
        f"across this edge is amplified {amplification:.4g}x over a normal one. "
        "Measured on this geometry: every runaway magnitude in the first outer step "
        "sits on the worst such edge, and the first non-finite value is exner at a "
        "neighbouring cell, from powf of a negative mass-weighted potential temperature "
        "(cuda_driver.py:1216-1221). The run then dies inside step 0 on a single "
        "validation flag that names nothing. "
        "REMEDY: regenerate the mesh. A uniform request seeds from the icosahedral "
        "Goldberg subdivision, which has no dislocations to form near-cocircular quads "
        "around and measures dvEdge/dcEdge >= 0.39. The published family measures "
        "0.394477 (x1.40962) and 0.033650 (x4.163842). The timestep is not the lever "
        "here and will not be reduced to compensate."
    )
