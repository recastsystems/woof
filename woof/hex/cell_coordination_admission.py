"""Cell-coordination admission for registered MPAS meshes.

WHAT THIS PREVENTS, MEASURED (an RTX 5090, 2026-08-26; evidence in
``evidence/graded-blowup-20260826/``).  ``v16.66.195630`` -- 195,630 cells,
16.5 km core inside a 66 km background, the finest graded mesh this program
has generated -- carries exactly ONE cell with four edges, cell 195615 at
33.74N 117.65W.  It is a cell the generator's count-changing defect surgery
inserted: it sits fourteen from the end of the cell array, and the mesh's own
emission receipt records the coordination histogram ``{4: 1, 5: 1028,
6: 193584, 7: 1016, 8: 1}``.

At its declared 100 s the forecast died at step 23 of 36.  Re-run at 75 s --
28% of Courant margin against 3.5% -- it died at step 31 of 48.  Those are the
same MODEL time, 2,300 s and 2,325 s, not the same step count:

    dt      died at                     model time   theta_m max at 1,800 s
    100 s   step 23 of 36                  2,300 s   981.61 K
     75 s   step 31 of 48                  2,325 s   981.95 K
     20 s   step 495 of 540 (|w| 281 m/s)  9,900 s   965.49 K

The timestep sets only how long it takes.  No timestep tested removes it.

In all three arms the model's theta maximum sits on cell 195615, at the top
model level, and the growth above the initial state there is 197.4 / 197.7 /
181.3 K -- the same anomaly, converged in the timestep -- while every global
minimum stays flat to four figures (``theta_m`` min 231.26 -> 231.15,
``exner`` min 0.2357 -> 0.2361, ``rho`` min 0.010747 -> 0.010770).  The
failure is one cell and its four neighbours in a field of 195,630 that is
otherwise healthy.

WHY IT IS NOT THE TIMESTEP, AND NOT THE DUAL EDGES.  ``v20.80.151649`` -- the
same generator, the same campaign, the same init pipeline -- completed a 6 h
forecast (180/180 steps, no refusal) at 120 s, which is 94.9% of ITS OWN
126.44 s Courant limit, against this mesh's 96.5%.  Its dual-edge
amplification is 24.34x, WORSE than this mesh's 24.03x.  The one property it
does not share is the coordination defect: its histogram is ``{5: 1029,
6: 149603, 7: 1017}`` and no cell has fewer than five edges.  Both meshes'
tightest-Courant edge and worst dual edge are thousands of kilometres from
the failure; on ``v16.66.195630`` cell 195615's own tightest edge admits
145.4 s, half again its declared timestep.

THE FLOOR IS ONE-SIDED, ON PURPOSE.  Only coordination BELOW five is refused.
The same mesh carries one 8-coordinated cell (168727) and it was measured NOT
to misbehave -- it is nowhere in the top forty cells by theta growth in any
arm -- so refusing high coordination would be a gate with no measured
breakage behind it, which is not a gate.  Coordination 7 is ordinary in this
family: ``v20.80.151649`` carries 1,017 of them and forecasts fine.

A SMALLER TIMESTEP IS NOT THE REMEDY AND IS NOT OFFERED AS ONE.  At 20 s the
mesh finishes a ONE-HOUR forecast -- 180 of 180 steps, twice, byte identical
frame for frame -- and that is the whole of what it buys.  Run to three
hours, the same cell's vertical velocity leaves the ground it was sitting on:
4.83 m/s at 07:00Z, **35.04 m/s at 07:30Z**, and 281 m/s at step 495, where
the step-health gate refuses it for exceeding the 200 m/s divergence limit.
It is still one cell -- one or two cells in the whole mesh are above 20 m/s
at that moment, and cell 195615 owns the maximum -- and the standing theta
error there never leaves, reading 181, 193, 175, 107 and 135 K across the
five committed frames.  So the timestep buys 2 h 45 m instead of 38 minutes.
It does not buy a forecast.

REMEDY: regenerate the mesh WITH A FIXED GENERATOR.  Regenerating with the
generator that made this mesh was a coin flip, and the measurement said so:
two of the three graded meshes reachable on 2026-08-26 carried a
4-coordinated cell -- ``v16.66.195630`` (cell 195615, fourteen from the end)
and the registered 15 km row ``v15.60.224210`` (cell 224206, three from the
end) -- while ``v20.80.151649`` did not, and both defects sat at the end of
the cell array where count-changing surgery appends.

**THAT GENERATOR-SIDE FOLLOW-UP LANDED THE SAME DAY** (woof
`the meshgen coordination work`, 2026-08-26, evidence
``tree/evidence/meshgen-coordination-20260826/``).  The cause was the
insertion operator placing its new generator on the quad's own circumcentre,
where its Delaunay ring is exactly the four quad cells -- measured at 18 of 18
and 13 of 13 insertions -- and a local polish that then PINNED the cell it had
just damaged.  ``rw-mpas`` surgery now reads coordination as half of its own
repair test and refuses below five, and ``validate`` refuses to emit such a
mesh at all.  All three spec rows regenerate clean: ``v16.66.195629``
(registered here) replaces this row, ``v15.60.224210`` keeps its cell count
with the defect repaired, and ``v20.80.151649`` regenerates BIT-IDENTICAL in
geometry, so its completed forecast still describes what the generator emits.

This gate is NOT retired by that fix and must not be.  Its breakage is a mesh
that ALREADY EXISTS: ``v16.66.195630`` is still a registered row, still on
disk, and any older or foreign grid file can carry the same defect.  What the
gate guarantees is that such a mesh cannot reach a card.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "CellCoordinationAdmissionError",
    "CellCoordinationPolicy",
    "CellCoordinationAdmission",
    "admit_cell_coordination",
]


class CellCoordinationAdmissionError(RuntimeError):
    """The mesh carries a cell whose edge count the dycore was not shown to survive."""


@dataclass(frozen=True, slots=True)
class CellCoordinationPolicy:
    schema: str = "gpuwm-hex.cell-coordination-admission/v1"
    minimum_edges_on_cell: int = 5
    description: str = (
        "A Voronoi cell with fewer than five edges is not produced by the "
        "icosahedral Goldberg seeding this family is built on; it can only be "
        "created by count-changing defect surgery. The one measured instance "
        "(v16.66.195630 cell 195615) carries a 197 K standing theta error at "
        "the model top that is converged in the timestep and terminates the "
        "forecast at 100 s and at 75 s. The floor is one-sided: high "
        "coordination is admitted, because the 8-coordinated cell in the same "
        "mesh was measured not to misbehave"
    )

    def validate(self) -> None:
        if int(self.minimum_edges_on_cell) < 3:
            raise CellCoordinationAdmissionError(
                "cell-coordination policy minimum_edges_on_cell must be at "
                f"least 3; got {self.minimum_edges_on_cell}"
            )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CellCoordinationAdmission:
    count: int
    minimum_edges_on_cell: int
    maximum_edges_on_cell: int
    histogram: dict[int, int]
    coordination_defect: int
    cells_below_floor: int
    first_cells_below_floor: tuple[int, ...]
    policy: CellCoordinationPolicy

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["histogram"] = {str(k): int(v) for k, v in self.histogram.items()}
        payload["first_cells_below_floor"] = list(self.first_cells_below_floor)
        payload["policy"] = self.policy.as_dict()
        return payload


def admit_cell_coordination(
    n_edges_on_cell: object,
    *,
    policy: CellCoordinationPolicy | None = None,
    mesh_name: str | None = None,
) -> CellCoordinationAdmission:
    """Admit or refuse a mesh's cell coordination before any device allocation.

    The histogram is recorded for EVERY mesh, admitted or not, so a mesh that
    passes today is still characterised in its own receipt.
    """

    selected = policy or CellCoordinationPolicy()
    selected.validate()

    raw = np.asarray(n_edges_on_cell)
    if raw.ndim != 1 or raw.size == 0:
        raise CellCoordinationAdmissionError(
            "nEdgesOnCell must be a non-empty one-dimensional nCells array, "
            f"got {raw.shape}"
        )
    counts = np.asarray(raw, dtype=np.int64)
    if np.any(counts < 0):
        bad = np.flatnonzero(counts < 0)
        raise CellCoordinationAdmissionError(
            f"nEdgesOnCell carries {bad.size} negative entries "
            f"(first cells {bad[:8].tolist()}); the mesh topology is corrupt"
        )

    values, occurrences = np.unique(counts, return_counts=True)
    histogram = {int(k): int(v) for k, v in zip(values, occurrences)}
    floor = int(selected.minimum_edges_on_cell)
    below = np.flatnonzero(counts < floor)

    admission = CellCoordinationAdmission(
        count=int(counts.size),
        minimum_edges_on_cell=int(counts.min()),
        maximum_edges_on_cell=int(counts.max()),
        histogram=histogram,
        # sum over cells of (6 - n); 12 for any closed Voronoi sphere
        coordination_defect=int(sum((6 - k) * v for k, v in histogram.items())),
        cells_below_floor=int(below.size),
        first_cells_below_floor=tuple(int(c) for c in below[:8]),
        policy=selected,
    )
    if below.size == 0:
        return admission

    worst = int(below[0])
    label = f"mesh {mesh_name!r}: " if mesh_name else ""
    raise CellCoordinationAdmissionError(
        f"{label}cell coordination refused before CUDA allocation. "
        f"Cell {worst} has {int(counts[worst])} edges, below the admitted floor "
        f"{floor}; {below.size} of {counts.size} cells are below it "
        f"(histogram {histogram}). "
        "THE BREAKAGE THIS PREVENTS, MEASURED (an RTX 5090, 2026-08-26, "
        "evidence/graded-blowup-20260826/): v16.66.195630 carries exactly one "
        "4-coordinated cell, 195615, inserted by the generator's "
        "count-changing defect surgery. At its declared 100 s the forecast "
        "died at step 23 of 36; at 75 s -- 28% of Courant margin against "
        "3.5% -- it died at step 31 of 48, the SAME model time (2,300 s and "
        "2,325 s), not the same step count. In every arm the model's theta "
        "maximum and its |w| maximum both sit on that one cell, at the top "
        "model level, and theta there grows 197.4 K (100 s), 197.7 K (75 s) "
        "and 181.3 K (20 s) above its initial value by 1,800 s while every "
        "global minimum stays flat to four figures. "
        "IT IS NOT THE TIMESTEP AND NOT THE DUAL EDGES: v20.80.151649, same "
        "generator and same campaign, completed 6 h at 94.9% of its own "
        "Courant limit with a worse dual-edge amplification (24.34x against "
        "24.03x) and carries no cell below coordination 5. "
        "A SMALLER TIMESTEP IS NOT THE REMEDY: at 20 s this mesh finishes one "
        "hour (180/180 steps, twice, byte identical) still carrying a 181 K "
        "error at that cell, and then the same cell's vertical velocity goes "
        "4.83 m/s at 07:00Z to 35.04 m/s at 07:30Z to 281 m/s at step 495 of "
        "540, where the step-health gate refuses it. The timestep buys 2 h "
        "45 m instead of 38 minutes; it does not buy a forecast. "
        "REMEDY: regenerate the mesh WITH A FIXED GENERATOR. Regenerating with "
        "the generator that made it was a coin flip: two of the three graded "
        "meshes reachable on 2026-08-26 carried such a cell, both at the end "
        "of the cell array where count-changing surgery appends. THAT "
        "GENERATOR-SIDE FOLLOW-UP LANDED THE SAME DAY (woof "
        "the meshgen coordination work, evidence/meshgen-coordination-20260826/): "
        "the insertion operator placed its new generator on the quad's own "
        "circumcentre, where its Delaunay ring is exactly the four quad cells "
        "-- 18 of 18 and 13 of 13 insertions measured -- and the local polish "
        "then pinned the cell it had just damaged. rw-mpas surgery now refuses "
        "its own emission below five. Every spec row regenerates clean; this "
        "row's replacement is v16.66.195629. This gate is not retired by that "
        "fix, because the mesh it refuses already exists: it guarantees such a "
        "mesh cannot reach a card. "
        "The floor is one-sided by measurement: the 8-coordinated cell in the "
        "same mesh was measured not to misbehave and is admitted."
    )
