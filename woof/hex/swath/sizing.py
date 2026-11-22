"""Pricing a placement through the generator itself, before anything is built.

``rw_mpas_mesh --dry-run`` sizes a resolution spec on the CPU, writes
nothing and needs no card.  That is the whole reason this module exists:
a plan that does not fit should be refused in the seconds before a cycle
starts, not discovered when a forecast runs out of device memory forty
minutes in.  It is also the difference between a cell count that came from
the generator's own sizing integral and one this layer computed with
sphere-area arithmetic and hoped about.

WHAT THIS DOES NOT QUOTE, AND WHY.  The dry-run receipt carries a
``region_attainment`` block with an ``attained_spacing_km`` per region.
For CAP regions it is right.  For POLYGON regions -- which is every swath
-- it is not: measured 2026-08-26 against the staged ``rw_mpas_mesh``
0.1.0, a 4-degree polygon reports its deepest interior point 19,683 km
inside itself and therefore reports its request exactly met at every size,
where a cap covering comparable ground correctly reports 6.36 km attained
against a 4.00 km request.  The cause is in
``rw-mpas/src/mesh/density.rs::polygon_contains``: a closed ring divides a
SPHERE into two discs and the winding number is +/-2*pi in both, so a test
that accepts on ``abs(winding) > pi`` calls the complement interior too.
The emitted MESHES are unaffected -- the density field integrates
correctly, which is measured -- so this is a reporting defect and swaths
are safe to generate.  But the ruling is that a receipt quotes ATTAINED
spacing and never requested, and a number that is structurally always the
request cannot satisfy it.

So this module quotes attainment from an INSCRIBED CAP PROBE instead: the
same spec, the same background, the same transition, with the swath region
replaced by a cap centred on the swath's own widest point at that point's
half-width.  That cap fits inside the swath, so the spacing the field
reaches inside it is an upper bound on -- never better than -- the spacing
the swath reaches, and it is produced by the same binary on the path that
binary gets right.  It is labelled ``inscribed_cap_probe`` everywhere it
appears.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from ..engines import MESH, EngineRefusal, resolve
from .errors import SwathCapacityRefusal, SwathRefusal

#: Printed wherever a polygon region's own attainment would otherwise be
#: quoted.  Names the defect so a reader can find the measurement.
POLYGON_ATTAINMENT_UNAVAILABLE = (
    "not quoted: the generator's polygon region_attainment reports every "
    "polygon's request exactly met (measured 2026-08-26; see "
    "evidence/swath-following-20260826/RECEIPT.md). Attainment for this swath "
    "is measured by an inscribed cap probe instead"
)


@dataclass(frozen=True)
class SizingResult:
    """What the generator says one spec costs, plus what it will deliver.

    TWO CELL COUNTS, AND THEY ARE NOT THE SAME NUMBER.  ``parent_cells`` is
    the whole graded GLOBAL mesh the spec describes -- background everywhere
    plus the refined swath -- and it is what the generator's own sizing
    integral returns.  ``swath_cells`` is what the limited-area cull leaves
    behind, which is the only one that ever reaches a card.  Comparing the
    first against a regional ceiling refuses every legitimate placement: a
    75 km background alone is 104,714 cells before a swath exists.
    """

    parent_cells: float
    swath_cells: float
    footprint_mib: float | None
    card: str | None
    steepest_gradient_percent_per_cell: float | None
    attained_spacing_km: float | None
    attained_basis: str
    #: What the SPEC ASKED FOR at this region, in km, before anything moved it.
    requested_spacing_km: float | None
    #: What the generator will actually BUILD there, in km.  The graded ladder
    #: refines by midpoint insertion, which halves a spacing exactly, so a
    #: refined core can only land on ``background / 2^k``; `rw-mpas`
    #: ``mesh::ladder_snap`` moves a request onto the nearest such rung, always
    #: FINER, and records the move.  Quoting only the request would hide it,
    #: and quoting only the attainment would leave a reader unable to tell a
    #: ramp that under-delivers from a snap that over-delivers.
    delivered_spacing_km: float | None
    #: The generator's own record of the move, verbatim from its receipt.
    ladder_snap: Mapping[str, Any] | None
    receipt: Mapping[str, Any]
    probe_receipt: Mapping[str, Any] | None

    def as_row(self) -> Mapping[str, Any]:
        return {
            "parent_cells": round(self.parent_cells, 1),
            "parent_basis": "generator_dry_run",
            "parent_footprint_mib": self.footprint_mib,
            "card": self.card,
            "steepest_gradient_percent_per_cell": self.steepest_gradient_percent_per_cell,
            "swath_cells": round(self.swath_cells, 1),
            "swath_basis": "area_integral_at_attained_spacing",
            "attained_spacing_km": (
                None if self.attained_spacing_km is None
                else round(self.attained_spacing_km, 4)
            ),
            "attained_basis": self.attained_basis,
            # NOT rounded, unlike the measured figures above: these two are
            # exact spec values, and 75/32 is 2.34375 km rather than 2.3438.
            "requested_spacing_km": self.requested_spacing_km,
            "delivered_spacing_km": self.delivered_spacing_km,
            "ladder_snap": self.ladder_snap,
            "polygon_attainment": POLYGON_ATTAINMENT_UNAVAILABLE,
        }


def resolve_engine(explicit: str | Path | None = None) -> Path:
    """``rw_mpas_mesh``, through the one ladder every door in this package uses."""

    try:
        return resolve(MESH, Path(explicit) if explicit is not None else None)
    except EngineRefusal as error:
        raise SwathRefusal(str(error)) from error


def _run(engine: Path, arguments: Sequence[str]) -> Mapping[str, Any]:
    completed = subprocess.run(
        [str(engine), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SwathRefusal(
            f"{engine.name} exited {completed.returncode} while sizing a swath "
            f"spec.\nargv: {' '.join(arguments)}\nstderr: {completed.stderr.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SwathRefusal(
            f"{engine.name} --dry-run did not print a JSON receipt: {error}. "
            f"First 400 characters of stdout: {completed.stdout[:400]!r}"
        ) from error


def _sized(
    spec: Mapping[str, Any],
    exe: Path,
    card: str | None,
    vram_gib: float | None,
    generator_words: Sequence[str] = (),
) -> Mapping[str, Any]:
    """One ``--dry-run`` invocation. No gates: this is the raw instrument.

    ``generator_words`` are the generator's own words for a geometry other than its default
    (``--geometry 2``): the count is then the one that geometry builds.
    """

    with tempfile.TemporaryDirectory(prefix="gpuwm-hex-swath-") as scratch:
        spec_path = Path(scratch) / "spec.json"
        spec_path.write_text(
            json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8", newline="\n"
        )
        arguments = ["--spec", str(spec_path), "--dry-run"]
        if card is not None:
            arguments.extend(["--card", card])
        if vram_gib is not None:
            arguments.extend(["--vram-gib", str(vram_gib)])
        arguments.extend(str(word) for word in generator_words)
        return _run(exe, arguments)


def dry_run(
    spec: Mapping[str, Any],
    *,
    engine: str | Path | None = None,
    card: str | None = None,
    vram_gib: float | None = None,
    generator_words: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Size one resolution spec through the real generator. Writes nothing.

    THE GATES ARE APPLIED HERE, and that is the point of this function
    rather than :func:`_sized`.  ``rw_mpas_mesh --dry-run`` prints a sizing
    receipt and exits 0 for specs its own build refuses seconds or minutes
    later -- the defect class ``docs/LANE-BRIEFING.md`` names as "a dry-run
    path that does not apply the real gate".  Measured 2026-08-28: a
    0.75/6/75 km design sized clean and the build refused it 87 ms later on
    the transition-band gate; a 0.75/3/15/75 km design sized clean, cleared
    that gate, spent 1,251 s deriving 217,621 cells and was refused on the
    200 m shortest-dual-edge floor with no grid written.  :mod:`woof.hex.mesh_spec_gates` applies the
    first (exact, from the receipt's own gradient) and reports the second
    with the numbers that decide it, under ``gates_applied_by_hexcore``.
    """

    from ..mesh_spec_gates import gates_from_receipt

    exe = resolve_engine(engine)
    receipt = _sized(spec, exe, card, vram_gib, generator_words)
    if generator_words:
        # A geometry other than the default: its dry run is the count under the field it builds,
        # and it carries none of geometry 1's ladder figures for the gates below to read.  Geometry
        # 2 gates itself at the build (it writes no mesh with an obtuse triangle).
        return {**receipt, "gates_applied_by_hexcore": None}

    def measure(candidate: Mapping[str, Any]) -> float:
        probe = _sized(candidate, exe, card, vram_gib)
        return float(probe["steepest_requested_gradient_percent_per_cell"]) / 100.0

    gates = gates_from_receipt(spec, receipt, measure=measure)
    return {**receipt, "gates_applied_by_hexcore": gates}


def _cap_probe_spec(
    spec: Mapping[str, Any], region_index: int, centre: tuple[float, float], radius_km: float
) -> Mapping[str, Any]:
    probe = json.loads(json.dumps(spec))
    region = probe["regions"][region_index]
    region["shape"] = {
        "kind": "cap",
        "center_deg": [centre[0], centre[1]],
        "radius_km": radius_km,
    }
    probe["name"] = "inscribed-cap-probe"
    return probe


def size_swath_spec(
    spec: Mapping[str, Any],
    *,
    region_index: int,
    probe_centre: tuple[float, float],
    probe_radius_km: float,
    ring: Sequence[tuple[float, float]],
    engine: str | Path | None = None,
    card: str | None = None,
    vram_gib: float | None = None,
) -> SizingResult:
    """Price a spec and measure what its swath region will actually deliver."""

    from .geometry import predicted_cells_in

    receipt = dry_run(spec, engine=engine, card=card, vram_gib=vram_gib)
    probe = dry_run(
        _cap_probe_spec(spec, region_index, probe_centre, probe_radius_km),
        engine=engine,
        card=card,
        vram_gib=vram_gib,
    )
    attainment = probe.get("region_attainment") or []
    attained = (
        float(attainment[region_index]["attained_spacing_km"])
        if region_index < len(attainment)
        else None
    )
    requested = float(spec["regions"][region_index]["spacing_km"])
    # The generator SNAPS every region onto the power-of-two ladder it can
    # actually build, before it seeds anything, and the dry-run record carries
    # the move.  `attained` is measured against the SNAPPED request, so the
    # bare request is no longer the number to compare it with -- reading it as
    # one is how a receipt comes to say a swath under-delivers when the
    # generator moved it FINER.
    snap = probe.get("ladder_snap") or receipt.get("ladder_snap")
    delivered = requested
    if isinstance(snap, Mapping):
        rows = snap.get("regions") or []
        if region_index < len(rows):
            row = rows[region_index]
            if row.get("delivered_spacing_km") is not None:
                delivered = float(row["delivered_spacing_km"])
    # The cull leaves the cells INSIDE the ring.  Their spacing is the
    # attained figure at the deepest point and coarser everywhere nearer the
    # boundary, so the integral at the attained spacing is an UPPER bound on
    # what the card must hold -- and a tighter one than the same integral at
    # the requested spacing, which is all a plan that never measured
    # attainment could use.
    swath_cells = predicted_cells_in(
        list(ring), spacing_km=attained if attained else delivered
    )
    return SizingResult(
        parent_cells=float(receipt.get("predicted_cells", 0.0)),
        swath_cells=swath_cells,
        footprint_mib=(
            None if receipt.get("footprint_mib") is None
            else float(receipt["footprint_mib"])
        ),
        card=receipt.get("card"),
        steepest_gradient_percent_per_cell=(
            None if receipt.get("steepest_requested_gradient_percent_per_cell") is None
            else float(receipt["steepest_requested_gradient_percent_per_cell"])
        ),
        attained_spacing_km=attained,
        attained_basis="inscribed_cap_probe",
        requested_spacing_km=requested,
        delivered_spacing_km=delivered,
        ladder_snap=receipt.get("ladder_snap"),
        receipt=receipt,
        probe_receipt=probe,
    )


def refuse_over_ceiling(cells: float, ceiling: float, slot_id: str) -> None:
    """Refuse a CULLED swath above the regional cell ceiling.

    Applied to ``swath_cells``, never to ``parent_cells``: the graded
    parent is generated on the CPU and culled before anything reaches a
    device, so a 142,708-cell parent carrying a 15,000-cell swath is a
    normal, admissible placement.
    """

    if cells > ceiling:
        raise SwathCapacityRefusal(
            f"swath {slot_id} predicts {cells:,.0f} cells against a declared "
            f"maximum_cells_per_swath of {ceiling:,.0f}. The regional ceiling is "
            "what a card admits with its boundary pool; a swath above it does not "
            "fail at generation, it fails during the forecast that consumes it, "
            "after the mesh, the statics and the boundaries have all been paid for. "
            "Reduce swath.lead_hours, reduce the half-width, or coarsen spacing_km"
        )


__all__ = [
    "POLYGON_ATTAINMENT_UNAVAILABLE",
    "SizingResult",
    "dry_run",
    "refuse_over_ceiling",
    "resolve_engine",
    "size_swath_spec",
]
