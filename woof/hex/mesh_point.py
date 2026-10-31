"""``woof hex mesh-plan --point``: a fine core at an arbitrary point.

WHAT THIS DOOR DOES.  Every registered sub-kilometre row this program
holds sits at 35.0 N 97.0 W, because a person typed its spec and its row.
This door takes a POINT, a fine spacing and a core radius, and does what
that person did, in order and with receipts:

1. writes the resolution spec -- the registered ``v0.9.120.110533`` recipe
   generalised: nested caps on one global background, each rung halving the
   spacing, each ramp a fixed multiple of its own spacing, each cap sitting
   a fixed number of ramps outside the next finer one -- centred on the
   point;
2. prices it through the generator's own ``--dry-run`` with the gates its
   build applies (:mod:`woof.hex.swath.sizing`), predicts what the limited-
   area CULL will cost, and prices THAT against the named card on the one
   admission surface (:mod:`woof.hex.device_admission`, the limited-area
   row), with the margin that surface names;
3. with ``--generate``: builds the parent grid (``rw_mpas_mesh``), cuts
   the limited-area cull at the shipped pad, builds the geography
   (``rw_mpas_static``) on a HALO -- a wider cut of the parent -- and cuts
   it down to the cull, measures the admissions (dual edge, cell
   coordination, Courant at the declared timestep), registers the pair as
   RUNTIME ROWS (:mod:`woof.hex.mesh_rows`, a file beside the mesh, never a
   checkout edit); and, with ``--vertical-spec``, mints the native-free
   vertical artifact on the halo and culls it so the child can be
   initialised from a regional meteorological source.  ``--parent-static``
   builds the geography and the vertical on the whole parent instead, the
   way this door always did; the cull's files are the same bytes either
   way (:func:`generate_point` says which and how it is held).

WHY THE LADDER IS DATA.  The spec is a pure function of six numbers
(:func:`ladder_spec`); nothing here branches on a place.  Moving the core
is moving the centre of every cap.  The transition and ring factors are
the registered row's own (18 and 3) and are arguments, not constants a
reader has to trust.

WHAT IS PRICED, AND WHAT IS NOT.  The generator prices the whole GRADED
GLOBAL mesh; only the cull reaches a card.  The plan therefore quotes both
and admits on the second, exactly as the swath layer does
(:mod:`woof.hex.swath.sizing` states why).  The cull prediction is an area
integral at the generator's own attained spacing -- a bound, labelled
``basis: area_integral`` -- and the cull receipt replaces it with the count.

THE INITIAL CONDITION ROAD THIS DOOR PREPARES.  ``woof hex init`` refuses
a limited-area grid in its native-free mode: the closed-sphere vertical
authority does not invent exterior state (``woof.hex.vertical``).  A global
init needs global meteorology, and a regional source (HRRR) covers only
its own domain.  The road that satisfies both is: mint the vertical on the
GLOBAL parent (grid + static + spec, no meteorology), cull the artifact
with the same region as the grid and static, and hand the culled artifact
to ``woof hex init --capsule/--reference`` with the regional intermediate.
The cull moves no cell centre, so the child's vertical IS the parent's,
cell for cell.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

from .errors import MpasPortError

#: The registered 937.5 m row's recipe, restated as the six numbers it is.
DEFAULT_BACKGROUND_KM = 120.0
#: Each rung's ramp is this many times its own spacing (the registered row:
#: "each ramp eighteen times its own target").
DEFAULT_TRANSITION_FACTOR = 18.0
#: Each cap sits this many ramps of the next finer rung outside it (the
#: registered row: 100 + 3 x 16.875 = 150.625 km, and so on up).
DEFAULT_RING_FACTOR = 3.0
#: The measured domain-size knee, and the shipped cull pad
#: (``evidence/nest-ratio-20260827/``: every field improves monotonically
#: with cut width and the knee is at 1.35x the fine core).
DEFAULT_CULL_PAD_SCALE = 1.35
#: Column count every measured row and every anchor in this tree runs at.
DEFAULT_LEVELS = 55
#: How far beyond the cut the halo route builds geography and the vertical
#: grid (:func:`generate_point`).  The static usually needs a few cells of it
#: (its neighbour stencils, and the albedo carried across cells with no pixel
#: of their own): on a 937.5 m, 100 km cut a 10 km halo already gave every
#: static variable bit for bit.  Where land has no valid pixel of a field
#: for hundreds of kilometres, the value is carried that far, and
#: :func:`check_halo_static` holds the static to its rim on every build.
#: The vertical grid needs far more, because
#: its coordinate surfaces are smoothed about 3,000 times.  MEASURED on 937.5
#: m, 100 km cuts at six places from 19.7 N to 64.8 N, the antimeridian
#: included (``evidence/halo-route-20261001``): the narrowest halo past which
#: the cut's vertical grid matched the whole sphere bit for bit at every
#: wider halo tried was 600 to 1,000 km, and a narrower one that happened to
#: match was followed by a wider one that did not.  1,600 km is that worst
#: case with 60 % to spare. :func:`check_halo_vertical` compares with the
#: whole parent's authority on every build; the width measurements alone
#: do not prove identity for another terrain or vertical specification.
DEFAULT_HALO_KM = 1600.0
#: Neighbour rings a cell's static values reach besides a value carried to
#: it (rw-mpas ``static_builder``): a pixel goes to its nearest cell, so a
#: cell beside the halo's rim collects pixels that belong to a cell beyond
#: it, and that cell's values reach its neighbours through the GWD
#: isolated-point smooth and the reconstruction and ``deriv_two`` stencils,
#: which read a cell's own ring and the next.  Three rings bound all of it;
#: :func:`check_halo_static` adds the rings the static builder says it
#: carried values across.
STATIC_STENCIL_RINGS = 3
#: The earth radius rw_mpas_static scales a unit-sphere grid by.
MPAS_EARTH_RADIUS_M = 6_371_229.0
#: The generator's own hexagon area factor (rw-mpas density.rs).
HEXAGON_AREA_FACTOR = math.sqrt(3.0) / 2.0
EARTH_RADIUS_KM = 6371.229

#: ``--card`` spellings, mapped onto the two tables that price a mesh: the
#: admission surface's card profile (``woof.hex.device_admission``) and the
#: generator's own measured part key (``rw_mpas_mesh --card``), plus the
#: nameplate memory a plan compares against when no card is in the room.
#: A card is a row here; a card with no generator row is priced by the
#: admission surface alone and the receipt says so.
CARD_ALIASES: Mapping[str, tuple[str, str | None, int]] = {
    "96gb": ("96gib-188sm", None, 97_887),
    "96gib": ("96gib-188sm", None, 97_887),
    "rtx-pro-6000": ("96gib-188sm", None, 97_887),
    "96gib-188sm": ("96gib-188sm", None, 97_887),
    "32gb": ("32gib-170sm", "rtx-5090", 32_607),
    "32gib": ("32gib-170sm", "rtx-5090", 32_607),
    "rtx-5090": ("32gib-170sm", "rtx-5090", 32_607),
    "32gib-170sm": ("32gib-170sm", "rtx-5090", 32_607),
    "24gb": ("24gib-128sm", None, 24_576),
    "24gib": ("24gib-128sm", None, 24_576),
    "rtx-4090": ("24gib-128sm", None, 24_576),
    "24gib-128sm": ("24gib-128sm", None, 24_576),
    "16gb": ("16gib-70sm", "rtx-5070-ti", 16_303),
    "16gib": ("16gib-70sm", "rtx-5070-ti", 16_303),
    "rtx-5070-ti": ("16gib-70sm", "rtx-5070-ti", 16_303),
    "16gib-70sm": ("16gib-70sm", "rtx-5070-ti", 16_303),
    "10gb": ("10gib-68sm", None, 10_240),
    "10gib": ("10gib-68sm", None, 10_240),
    "rtx-3080": ("10gib-68sm", None, 10_240),
    "10gib-68sm": ("10gib-68sm", None, 10_240),
}


class PointPlanRefusal(MpasPortError):
    """A point plan is refused, and the message names what would break."""


# ---------------------------------------------------------------------------
# the spec
# ---------------------------------------------------------------------------
def parse_point(text: str) -> tuple[float, float]:
    parts = [piece.strip() for piece in str(text).split(",")]
    if len(parts) != 2:
        raise PointPlanRefusal(
            f"--point {text!r} is not LAT,LON; a fine core needs one centre "
            f"and nothing here guesses one"
        )
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError as error:
        raise PointPlanRefusal(f"--point {text!r} is not two numbers: {error}") from error
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 360.0:
        raise PointPlanRefusal(
            f"--point {text!r} is outside the sphere (latitude -90..90, "
            f"longitude -180..360)"
        )
    if lon > 180.0:
        lon -= 360.0
    return lat, lon


def ladder_rungs(background_km: float, fine_dx_m: float) -> list[float]:
    """The halving ladder from the background down to the fine spacing, km.

    Refuses a fine spacing that is not on the ladder, naming the rungs on
    either side: the generator snaps a request onto the nearest rung
    ALWAYS FINER (rw-mpas ``mesh::ladder_snap``), so a 900 m request would
    silently build 468.75 m and cost four times the cells the card was
    priced for.  Choosing the rung is the caller's decision.
    """

    if not background_km > 0.0:
        raise PointPlanRefusal(f"--background-km {background_km} is not positive")
    fine_km = float(fine_dx_m) / 1000.0
    if not fine_km > 0.0:
        raise PointPlanRefusal(f"--fine-dx-m {fine_dx_m} is not positive")
    if fine_km >= background_km:
        raise PointPlanRefusal(
            f"--fine-dx-m {fine_dx_m:g} m is not finer than the "
            f"{background_km:g} km background, so there is nothing to refine"
        )
    ratio = background_km / fine_km
    k = round(math.log2(ratio))
    if k < 1 or abs(background_km / (2.0 ** k) - fine_km) > 1e-9 * background_km:
        below = background_km / (2.0 ** math.floor(math.log2(ratio)))
        above = background_km / (2.0 ** math.ceil(math.log2(ratio)))
        raise PointPlanRefusal(
            f"--fine-dx-m {fine_dx_m:g} m is not a rung of the "
            f"{background_km:g} km halving ladder.  The graded generator "
            f"refines by midpoint insertion, so a refined core can only land "
            f"on background / 2^k, and rw_mpas_mesh snaps a request onto the "
            f"nearest such rung ALWAYS FINER -- {fine_dx_m:g} m would build "
            f"{above * 1000.0:g} m and cost about {(below / above) ** 2:.0f}x "
            f"the cells this plan would price.  The rungs on either side are "
            f"{below * 1000.0:g} m and {above * 1000.0:g} m; pass one of them, "
            f"or change --background-km"
        )
    return [background_km / (2.0 ** level) for level in range(1, k + 1)]


def ladder_spec(
    point: tuple[float, float],
    *,
    fine_dx_m: float,
    radius_km: float,
    background_km: float = DEFAULT_BACKGROUND_KM,
    transition_factor: float = DEFAULT_TRANSITION_FACTOR,
    ring_factor: float = DEFAULT_RING_FACTOR,
    name: str | None = None,
) -> dict[str, Any]:
    """The nested-cap resolution spec ``rw_mpas_mesh --spec`` reads unchanged.

    The registered ``v0.9.120.110533`` spec is ``ladder_spec((35.0, -97.0),
    fine_dx_m=937.5, radius_km=100.0)`` to the digit (tested).
    """

    if not radius_km > 0.0:
        raise PointPlanRefusal(f"--radius-km {radius_km} is not positive")
    if transition_factor <= 0.0 or ring_factor <= 0.0:
        raise PointPlanRefusal(
            "the transition and ring factors must be positive; a zero ramp "
            "is a step the surgery-locality gate refuses"
        )
    rungs = ladder_rungs(background_km, fine_dx_m)
    lat, lon = point
    regions: list[dict[str, Any]] = []
    radius = float(radius_km)
    for spacing in reversed(rungs):
        transition = transition_factor * spacing
        regions.append(
            {
                "shape": {"kind": "cap", "center_deg": [lat, lon], "radius_km": radius},
                "spacing_km": spacing,
                "transition_km": transition,
            }
        )
        radius = radius + ring_factor * transition
    regions.reverse()
    label = name or (
        f"point {lat:.4f}N {lon:.4f}E {fine_dx_m:g}m core {radius_km:g}km"
    )
    return {"name": label, "background_km": float(background_km), "regions": regions}


def cull_region(point: tuple[float, float], radius_km: float, pad_scale: float) -> dict[str, Any]:
    """The cap ``rw_mpas_mesh --cull-parent --region`` is handed."""

    if pad_scale < 1.0:
        raise PointPlanRefusal(
            f"--cull-pad-scale {pad_scale} is below 1.0: a cut inside the fine "
            f"core hands the boundary rings fine cells the parent's ramp "
            f"never reaches, which is the seam the pad exists to move outward"
        )
    lat, lon = point
    return {
        "kind": "cap",
        "center_deg": [lat, lon],
        "radius_km": float(radius_km) * float(pad_scale),
    }


def row_token(point: tuple[float, float]) -> str:
    lat, lon = point
    ns = "n" if lat >= 0 else "s"
    ew = "e" if lon >= 0 else "w"
    return f"{ns}{abs(lat):.2f}{ew}{abs(lon):.2f}"


def parent_row_name(point: tuple[float, float], fine_dx_m: float, background_km: float, cells: int) -> str:
    return f"p{fine_dx_m / 1000.0:g}.{background_km:g}.{cells}.{row_token(point)}"


def cull_row_name(point: tuple[float, float], fine_dx_m: float, background_km: float, cells: int) -> str:
    return f"q{fine_dx_m / 1000.0:g}.{background_km:g}.{cells}.{row_token(point)}"


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------
def cap_area_km2(radius_km: float) -> float:
    return 2.0 * math.pi * EARTH_RADIUS_KM ** 2 * (1.0 - math.cos(radius_km / EARTH_RADIUS_KM))


def spacing_profile(spec: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    """``(cap radius, spacing, transition)`` per cap region, innermost first.

    The generator's density puts a region's spacing INSIDE ``radius -
    transition`` and ramps it to the next coarser rung across the last
    ``transition`` of the cap (measured against the registered
    ``r0.9.120.40520``: "the whole 937.5 m core plus the first 25 km of the
    1.875 km ramp" is 40,520 cells, and this profile predicts it to 2 %).
    """

    rows: list[tuple[float, float, float]] = []
    for region in spec.get("regions", ()):
        shape = region.get("shape") or {}
        if shape.get("kind") != "cap":
            continue
        spacing = float(region["spacing_km"])
        transition = float(
            region.get("transition_km")
            if region.get("transition_km") is not None
            else float(region.get("transition_cells", 0.0)) * spacing
        )
        rows.append((float(shape["radius_km"]), spacing, transition))
    rows.sort()
    return rows


def spacing_at(profile: Sequence[tuple[float, float, float]], r_km: float,
               background_km: float) -> float:
    """The spacing the ladder asks for at ``r_km`` from the centre."""

    for index, (radius, spacing, transition) in enumerate(profile):
        if r_km > radius:
            continue
        flat_edge = radius - transition
        if r_km <= flat_edge:
            return spacing
        coarser = profile[index + 1][1] if index + 1 < len(profile) else background_km
        return spacing + (coarser - spacing) * (r_km - flat_edge) / max(transition, 1e-9)
    return background_km


def predicted_cull_cells(
    *,
    spec: Mapping[str, Any],
    radius_km: float,
    pad_scale: float,
    attained_fine_km: float | None = None,
    boundary_rings: int = 7,
    steps: int = 600,
) -> dict[str, Any]:
    """Cells the cull holds, integrated over the ladder's own spacing profile.

    ``basis: area_integral``.  The innermost rung's spacing is replaced by
    the generator's ATTAINED spacing when the plan measured one, so the
    count is a bound at what the mesh delivers rather than at the request.
    The seven boundary rings are integrated too: the disc is dilated by
    their width at the cut's own spacing.
    """

    profile = spacing_profile(spec)
    if not profile:
        raise PointPlanRefusal("the spec carries no cap region to price a cull of")
    background = float(spec.get("background_km", DEFAULT_BACKGROUND_KM))
    if attained_fine_km is not None:
        radius0, _, transition0 = profile[0]
        profile[0] = (radius0, float(attained_fine_km), transition0)
    cut = float(radius_km) * float(pad_scale)
    halo = boundary_rings * spacing_at(profile, cut, background)
    outer = cut + halo
    cells = 0.0
    core_cells = 0.0
    edges = [outer * i / steps for i in range(steps + 1)]
    for inner_r, outer_r in zip(edges, edges[1:]):
        mid = 0.5 * (inner_r + outer_r)
        area = cap_area_km2(outer_r) - cap_area_km2(inner_r)
        count = area / (HEXAGON_AREA_FACTOR * spacing_at(profile, mid, background) ** 2)
        cells += count
        if outer_r <= radius_km + 1e-9:
            core_cells += count
    return {
        "basis": "area_integral",
        "predicted_cells": round(cells, 1),
        "core_cells": round(core_cells, 1),
        "core_radius_km": float(radius_km),
        "fine_flat_radius_km": profile[0][0] - profile[0][2],
        "cut_radius_km": cut,
        "spacing_at_cut_km": spacing_at(profile, cut, background),
        "boundary_rings": boundary_rings,
        "halo_km": halo,
    }


def resolve_card(key: str | None) -> dict[str, Any] | None:
    """A ``--card`` spelling resolved onto the two pricing tables."""

    if key is None:
        return None
    normalized = str(key).strip().lower()
    if normalized not in CARD_ALIASES:
        raise PointPlanRefusal(
            f"--card {key!r} is not a card this plan can price.  Known "
            f"spellings: {', '.join(sorted(CARD_ALIASES))}.  A card is a row "
            f"in woof.hex.mesh_point.CARD_ALIASES joined to a measured row of "
            f"woof.hex.device_admission; a card nobody measured is refused "
            f"rather than given another card's number"
        )
    admission_key, mesh_key, nameplate_mib = CARD_ALIASES[normalized]
    return {
        "requested": key,
        "admission_card": admission_key,
        "generator_card": mesh_key,
        "nameplate_mib": nameplate_mib,
    }


def device_verdict(
    cells: float, card: Mapping[str, Any], *, budget_mib: float | None = None
) -> dict[str, Any]:
    """The limited-area admission row applied to the predicted cull."""

    from . import device_admission

    key = str(card["admission_card"])
    if key == "24gib-128sm":
        # The CUDA properties of the 4090, not a borrowed 5090 footprint.
        # model_for_card marks the card-scaled estimate as derived. The
        # forecast still measures actual free device memory at launch.
        profile = device_admission.CardProfile("RTX 4090 (24 GiB, sm_89)", 128, 1536)
    else:
        profile = device_admission.KNOWN_CARDS[key]
    model = device_admission.model_for_card(profile, configuration="limited-area")
    required = model.required_bytes(int(math.ceil(cells)))
    predicted = model.predict_bytes(int(math.ceil(cells)))
    margin = model.margin_bytes()
    budget_bytes = int(
        (budget_mib if budget_mib is not None else float(card["nameplate_mib"])) * device_admission.MIB
    )
    fits = int(model.max_cells(budget_bytes))
    return {
        "row": device_admission.row_key("limited-area", profile),
        "row_measured": bool(model.measured),
        "card": profile.as_dict(),
        "cells_priced": int(math.ceil(cells)),
        "predicted_mib": round(predicted / device_admission.MIB, 1),
        "margin_mib": round(margin / device_admission.MIB, 1),
        "required_free_mib": round(required / device_admission.MIB, 1),
        "budget_mib": round(budget_bytes / device_admission.MIB, 1),
        "budget_basis": (
            "--vram-gib" if budget_mib is not None
            else "nameplate; the forecast door measures FREE memory at launch and "
                 "admits on that"
        ),
        "fits": required <= budget_bytes,
        "cells_that_fit": fits,
        "short_by_mib": round(max(0.0, (required - budget_bytes) / device_admission.MIB), 1),
    }


def max_radius_for_cells(
    cells: int, request: "PointRequest", *, attained_fine_km: float | None = None
) -> float:
    """The largest core radius whose predicted cull fits ``cells``."""

    low, high = 0.0, 4000.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        spec = ladder_spec(
            request.point, fine_dx_m=request.fine_dx_m, radius_km=mid,
            background_km=request.background_km,
            transition_factor=request.transition_factor, ring_factor=request.ring_factor,
        )
        predicted = predicted_cull_cells(
            spec=spec, radius_km=mid, pad_scale=request.cull_pad_scale,
            attained_fine_km=attained_fine_km,
        )["predicted_cells"]
        if predicted <= cells:
            low = mid
        else:
            high = mid
    return round(low, 1)


def choose_timestep(minimum_dc_edge_m: float, *, fine_dx_m: float) -> dict[str, Any]:
    """The timestep a generated row declares: MPAS-A's rule, made stable.

    RETIRED 2026-10-10: this chose the largest ANCHORED timestep under the
    old 0.90 x 125 m/s Courant policy and refused a mesh no anchor fitted
    ("no anchored timestep fits this mesh").  It now takes MPAS-A's 6 s per
    km of the fine spacing, rounded down to divide the 1800 s radiation
    interval (woof.hex.mpas_defaults.default_dt_seconds), and steps down
    through the whole-second divisors of that interval until the physical
    stability check (woof.hex.timestep_admission) admits it on this mesh's
    real min(dcEdge).  Whether the result matches a registered anchor is
    recorded, not required.
    """

    import numpy as np

    from . import convection_admission, dt_admission, mpas_defaults
    from .timestep_admission import (
        TimestepAdmissionError, admit_timestep, edge_length_authority,
    )

    authority = edge_length_authority(np.asarray([float(minimum_dc_edge_m)]))
    cadence = mpas_defaults.DEFAULT_RADIATION_SECONDS
    first = mpas_defaults.default_dt_seconds(float(fine_dx_m), radiation_seconds=cadence)
    sub_3km = float(fine_dx_m) < convection_admission.CONVECTION_OFF_BELOW_M
    scheme = None if sub_3km else "gf"
    candidates = [first["dt_seconds"]] + [
        float(seconds) for seconds in range(int(first["dt_seconds"]) - 1, 0, -1)
        if cadence % seconds == 0.0
    ]
    for dt in candidates:
        try:
            admitted = admit_timestep(dt, authority)
        except TimestepAdmissionError:
            continue
        return {
            "dt_seconds": float(dt),
            "rule": first["rule"],
            "courant_limit_seconds": admitted.maximum_admitted_dt_seconds,
            "margin": admitted.maximum_admitted_dt_seconds / float(dt),
            "cumulus_scheme": scheme,
            "anchor": dt_admission.anchor_record(
                dt, radiation_seconds=cadence, surface_pbl_seconds=None,
                cumulus_seconds=None if scheme is None else dt, cumulus_scheme=scheme,
            ),
        }
    raise PointPlanRefusal(
        f"no whole-second timestep that divides the {cadence:g} s radiation "
        f"interval is stable on this mesh (min dcEdge {minimum_dc_edge_m:.1f} "
        f"m): even 1 s exceeds the advective or acoustic Courant limit "
        f"(woof.hex.timestep_admission).  The mesh's smallest edge is "
        f"degenerate; regenerate it or coarsen --fine-dx-m"
    )


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PointRequest:
    point: tuple[float, float]
    fine_dx_m: float
    radius_km: float
    background_km: float
    transition_factor: float
    ring_factor: float
    cull_pad_scale: float
    card: Mapping[str, Any] | None
    vram_gib: float | None
    mesh_exe: Path | None
    name: str | None
    #: A drawn region (a polygon row, :mod:`woof.hex.mesh_shape`) instead of
    #: the cap at ``point``.  ``point`` is then its centroid and
    #: ``radius_km`` its equal-area radius, both descriptive.
    shape: Mapping[str, Any] | None = None

    @property
    def fine_km(self) -> float:
        return self.fine_dx_m / 1000.0

    @property
    def transition_km(self) -> float:
        return self.transition_factor * self.fine_km


def drawn_shape(arguments: argparse.Namespace) -> dict[str, Any] | None:
    """The region ``--shape FILE`` or ``--ellipse ...`` draws, or None."""

    from . import mesh_shape

    shape_file = getattr(arguments, "shape", None)
    ellipse = getattr(arguments, "ellipse", None)
    if shape_file is not None and ellipse is not None:
        raise PointPlanRefusal("--shape and --ellipse both draw the core; give one")
    try:
        if shape_file is not None:
            return mesh_shape.load_shape(Path(shape_file))
        if ellipse is not None:
            return mesh_shape.parse_ellipse(ellipse)
    except mesh_shape.ShapeRefusal as error:
        raise PointPlanRefusal(str(error)) from error
    return None


def request_from_arguments(arguments: argparse.Namespace) -> PointRequest:
    from . import mesh_shape

    shape = drawn_shape(arguments)
    if shape is not None and getattr(arguments, "point", None) is not None:
        raise PointPlanRefusal(
            "--point and a drawn core (--shape/--ellipse) were both given; the drawn "
            "region's centre is its own, so name one")
    if shape is not None and shape["kind"] == "cap":
        # A drawn circle IS the circle door: same spec, same bytes.
        point = (float(shape["center_deg"][0]), float(shape["center_deg"][1]))
        radius = float(shape["radius_km"])
        shape = None
    elif shape is not None:
        point = mesh_shape.centroid(shape)
        radius = mesh_shape.equal_area_radius_km(shape)
    else:
        if getattr(arguments, "point", None) is None:
            raise PointPlanRefusal(
                "no fine core was named: pass --point LAT,LON (with --radius-km), "
                "--shape FILE or --ellipse LAT,LON,SEMI_MAJOR_KM,SEMI_MINOR_KM[,AZIMUTH_DEG]")
        point = parse_point(arguments.point)
        radius = float(arguments.radius_km)
    return PointRequest(
        point=point,
        fine_dx_m=float(arguments.fine_dx_m),
        radius_km=radius,
        background_km=float(arguments.background_km),
        transition_factor=float(arguments.transition_factor),
        ring_factor=float(arguments.ring_factor),
        cull_pad_scale=float(arguments.cull_pad_scale),
        card=resolve_card(arguments.card),
        vram_gib=arguments.vram_gib,
        mesh_exe=arguments.mesh_exe,
        name=getattr(arguments, "name", None),
        shape=shape,
    )


def request_spec(request: PointRequest) -> dict[str, Any]:
    """The resolution spec a request builds: the cap ladder at the point, or
    the same ladder around a drawn region (:mod:`woof.hex.mesh_shape`)."""

    if request.shape is None:
        return ladder_spec(
            request.point,
            fine_dx_m=request.fine_dx_m,
            radius_km=request.radius_km,
            background_km=request.background_km,
            transition_factor=request.transition_factor,
            ring_factor=request.ring_factor,
            name=request.name,
        )
    from . import mesh_shape

    lat, lon = request.point
    return mesh_shape.shape_ladder_spec(
        request.shape,
        rungs_km=ladder_rungs(request.background_km, request.fine_dx_m),
        background_km=request.background_km,
        transition_factor=request.transition_factor,
        ring_factor=request.ring_factor,
        name=request.name or (
            f"shape {lat:.4f}N {lon:.4f}E {request.fine_dx_m:g}m core "
            f"{mesh_shape.area_km2(request.shape):,.0f} km2, outline grown "
            f"{request.transition_km:g} km"),
    )


def request_cull_region(request: PointRequest) -> dict[str, Any]:
    if request.shape is None:
        return cull_region(request.point, request.radius_km, request.cull_pad_scale)
    from . import mesh_shape

    try:
        return mesh_shape.shape_cull_region(
            request.shape, grown_km=request.transition_km, pad_scale=request.cull_pad_scale)
    except mesh_shape.ShapeRefusal as error:
        raise PointPlanRefusal(str(error)) from error


def request_halo_region(request: PointRequest, halo_km: float) -> dict[str, Any]:
    if request.shape is None:
        return halo_region(request.point, request.radius_km, request.cull_pad_scale, halo_km)
    from . import mesh_shape

    try:
        return mesh_shape.shape_halo_region(
            request.shape, grown_km=request.transition_km, pad_scale=request.cull_pad_scale,
            halo_km=halo_km)
    except mesh_shape.ShapeRefusal as error:
        raise PointPlanRefusal(str(error)) from error


def plan_point(request: PointRequest) -> dict[str, Any]:
    """Price the point: the spec, the parent, the cull, the card."""

    from .swath.sizing import dry_run

    spec = request_spec(request)
    region = request_cull_region(request)
    generator_card = request.card["generator_card"] if request.card else None
    sizing = dry_run(
        spec,
        engine=request.mesh_exe,
        card=generator_card,
        vram_gib=request.vram_gib if generator_card else None,
    )
    attainment = sizing.get("region_attainment") or []
    attained_km = (
        float(attainment[-1]["attained_spacing_km"])
        if attainment and attainment[-1].get("attained_spacing_km")
        else request.fine_km
    )
    if request.shape is None:
        cull = predicted_cull_cells(
            spec=spec,
            radius_km=request.radius_km,
            pad_scale=request.cull_pad_scale,
            attained_fine_km=attained_km,
        )
        cull["attainment_basis"] = (
            "generator region_attainment on the innermost cap (cap regions are "
            "the shape the generator's attainment reports correctly; see "
            "woof.hex.swath.sizing)"
        )
    else:
        from . import mesh_shape

        cull = mesh_shape.predicted_shape_cull_cells(
            spec=spec, drawn=request.shape, cull=region, attained_fine_km=attained_km)
        cull["core_radius_km"] = request.radius_km
        cull["cut_radius_km"] = None
        cull["outline_grown_km"] = request.transition_km
        cull["attainment_basis"] = "generator region_attainment on the innermost region"
    cull["attained_fine_spacing_km"] = attained_km
    verdict = None
    if request.card is not None:
        verdict = device_verdict(
            cull["predicted_cells"], request.card, budget_mib=(
                None if request.vram_gib is None else float(request.vram_gib) * 1024.0
            ),
        )
        # The largest CIRCLE the card holds; a drawn region has no radius to
        # solve for, so it is priced by the cells it needs alone.
        verdict["max_core_radius_km_on_this_card"] = (
            max_radius_for_cells(int(verdict["cells_that_fit"]), request,
                                 attained_fine_km=attained_km)
            if request.shape is None else None)
    return {
        "schema": "gpuwm-hex.point-plan/v1",
        "point_deg": list(request.point),
        "fine_dx_m": request.fine_dx_m,
        "core_radius_km": request.radius_km,
        "background_km": request.background_km,
        "transition_factor": request.transition_factor,
        "ring_factor": request.ring_factor,
        "ladder_km": ladder_rungs(request.background_km, request.fine_dx_m),
        "spec": spec,
        "cull_region": region,
        "cull_pad_scale": request.cull_pad_scale,
        "parent": {
            "predicted_cells": sizing.get("predicted_cells"),
            "basis": "generator_dry_run",
            "footprint_mib": sizing.get("footprint_mib"),
            "card": sizing.get("card"),
            "steepest_gradient_percent_per_cell": sizing.get(
                "steepest_requested_gradient_percent_per_cell"),
            "ladder_snap": sizing.get("ladder_snap"),
            "gates_applied_by_hexcore": sizing.get("gates_applied_by_hexcore"),
        },
        "cull": cull,
        "device": verdict,
        "card": request.card,
        "drawn_shape": None if request.shape is None else dict(request.shape),
    }


def _report(plan: Mapping[str, Any]) -> list[str]:
    lat, lon = plan["point_deg"]
    parent = plan["parent"]
    cull = plan["cull"]
    gates = parent.get("gates_applied_by_hexcore") or {}
    band = gates.get("transition_band") or {}
    edge = gates.get("short_dual_edge_floor") or {}
    lines = [
        f"mesh-plan --point: {plan['spec']['name']}",
        f"  centre           {lat:.4f} N {lon:.4f} E",
        f"  ladder           {' > '.join(f'{r:g}' for r in plan['ladder_km'])} km "
        f"on a {plan['background_km']:g} km background "
        f"({len(plan['ladder_km'])} rungs, ramp {plan['transition_factor']:g}x, "
        f"ring {plan['ring_factor']:g}x)",
        f"  parent (global)  {parent['predicted_cells']:,.0f} cells (the generator's "
        f"own sizing integral)",
    ]
    if plan.get("drawn_shape"):
        lines += [
            f"  core             drawn, {len(plan['drawn_shape']['vertices_deg'])} vertices, "
            f"{cull['core_area_km2']:,.0f} km2 (equal-area radius {plan['core_radius_km']:.1f} km), "
            f"outline grown {cull['outline_grown_km']:g} km so the drawn area is fine; "
            f"{cull['attained_fine_spacing_km'] * 1000.0:.1f} m attained -> "
            f"{cull['core_cells']:,.0f} cells",
            f"  cull             the fine region grown by the pad ({plan['cull_pad_scale']:g}) -> "
            f"{cull['predicted_cells']:,.0f} cells predicted, {cull['boundary_rings']} boundary "
            f"rings ({cull['basis']})",
        ]
    else:
        lines += [
            f"  core             {plan['core_radius_km']:g} km at "
            f"{cull['attained_fine_spacing_km'] * 1000.0:.1f} m attained -> "
            f"{cull['core_cells']:,.0f} cells",
            f"  cull             cut at {cull['cut_radius_km']:g} km (pad "
            f"{plan['cull_pad_scale']:g}) -> {cull['predicted_cells']:,.0f} cells "
            f"predicted, {cull['boundary_rings']} boundary rings ({cull['basis']})",
        ]
    device = plan.get("device")
    if device is None:
        lines.append(
            "  footprint        not priced: no --card, and the fixed term is a "
            "property of the part"
        )
    else:
        lines.extend([
            f"  footprint        {device['predicted_mib']:,.1f} MiB predicted + "
            f"{device['margin_mib']:,.1f} MiB margin = {device['required_free_mib']:,.1f} "
            f"MiB free needed on row {device['row']} "
            f"({'measured' if device['row_measured'] else 'DERIVED'})",
            f"  budget           {device['budget_mib']:,.1f} MiB ({device['budget_basis']})",
            f"  verdict          {'FITS' if device['fits'] else 'DOES NOT FIT'}"
            + ("" if device["fits"] else f", short by {device['short_by_mib']:,.1f} MiB")
            + f"; this card holds {device['cells_that_fit']:,} cells"
            + ("" if device["max_core_radius_km_on_this_card"] is None else
               f", a core of up to {device['max_core_radius_km_on_this_card']:g} km at this "
               f"spacing and pad"),
        ])
    lines.extend([
        "",
        "  GATES THE SPEC DECIDES, applied here:",
        f"    transition band   PASS  "
        f"{band.get('steepest_gradient_percent_per_cell', float('nan')):.4f} %/cell -> "
        f"{band.get('band_cells', float('nan')):.2f} cells (floor "
        f"{band.get('band_cells_floor', float('nan')):.0f}, ceiling "
        f"{band.get('gradient_percent_per_cell_ceiling', float('nan')):.4f} %/cell)",
        "  GATES ONLY A BUILD DECIDES:",
        f"    shortest dual edge  {str(edge.get('verdict', 'unknown')).upper()}  "
        f"limit {edge.get('limit_m', float('nan')):.0f} m ({edge.get('gate', 'unknown gate')})",
    ])
    return lines


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@contextmanager
def _place_stage(stage: str, *, attrs: Mapping[str, Any] | None = None):
    """Emit C1 boundaries without changing preparation or its failure path."""
    def emit(phase: str) -> None:
        try:
            from woof.timeline.ledger import emit as append

            append(stage, phase, product="hex", attrs=dict(attrs or {}))
        except BaseException:
            # Timing must never prevent a mesh build or replace its error.
            pass

    emit("begin")
    try:
        yield
    finally:
        emit("end")


def _run(argv: Sequence[str], *, what: str, log: Path | None = None) -> subprocess.CompletedProcess:
    started = time.perf_counter()
    stage = {"rw_mpas_mesh": "prep.place.mesh", "rw_mpas_static": "prep.place.static"}[what]
    from .prep_store import run_or_fetch

    with _place_stage(stage, attrs={"engine": what, "terrain_only": "--terrain-only" in argv}):
        # Same bytes from the per-mesh store when it holds this exact request
        # (woof.hex.prep_store); otherwise the engine runs exactly as before.
        outcome = run_or_fetch(
            argv, runner=lambda a: subprocess.run(a, capture_output=True, text=True))
        completed = outcome.get("completed") or subprocess.CompletedProcess(
            [str(item) for item in argv], 0,
            stdout=f"prep-store: {outcome['kind']} {outcome['key']} from {outcome['store']} "
                   f"(SHA-256 verified) in {outcome['seconds']:.1f} s\n", stderr="")
    elapsed = time.perf_counter() - started
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f"$ {' '.join(str(item) for item in argv)}\n[{elapsed:.1f} s, exit "
            f"{completed.returncode}]\n--- stdout ---\n{completed.stdout}\n--- stderr ---\n"
            f"{completed.stderr}\n",
            encoding="utf-8",
        )
    if completed.returncode != 0:
        from .engines import refusal_text

        raise PointPlanRefusal(
            f"{what} exited {completed.returncode}"
            + (f"; its log is {log}" if log is not None else "")
            + f".  It said:\n{refusal_text(completed.stderr, completed.stdout)}"
        )
    completed.elapsed = elapsed  # type: ignore[attr-defined]
    return completed


def resolve_geog(explicit: Path | None) -> Path:
    """The WPS_GEOG root the static reads, or a refusal naming the ladder."""

    if explicit is not None:
        root = Path(explicit).expanduser()
        if not root.is_dir():
            raise PointPlanRefusal(
                f"--geog {root} is not a directory.  The static half of the "
                f"pair reads terrain, land use, soil, green-ness and albedo "
                f"from a WPS_GEOG archive, and the registry refuses a grid "
                f"with no matching static"
            )
        return root
    try:
        from woof import rustwx_static
    except ImportError:
        rustwx_static = None  # type: ignore[assignment]
    if rustwx_static is not None:
        found = rustwx_static.default_geog_root()
        if found is not None:
            return Path(found)
        ladder = ", ".join(str(item) for item in rustwx_static.geog_root_candidates())
    else:
        ladder = "$GPUWM_WPS_GEOG (woof is not importable here, so its ladder cannot be read)"
    raise PointPlanRefusal(
        f"no WPS_GEOG archive was found; pass --geog, or stage one with "
        f"`woof fetch-geog --datasets mesh`.  Looked at: {ladder}"
    )


def static_equivalent_lengths(grid: Path) -> tuple[Any, Any]:
    """``dvEdge`` and ``dcEdge`` exactly as ``rw_mpas_static`` stores them.

    The static builder reads a generated grid's unit-sphere lengths, scales
    them by MPAS's earth radius in float64 and stores them as float32
    (rw-mpas ``static_builder``: a grid whose first 64 centres average under
    ten units is a unit-sphere grid).  The same two operations here give the
    same float32 bits, so a parent with no static is admitted on the numbers
    its static would have carried.
    """

    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        xyz = [np.asarray(dataset.variables[name][:64], dtype=np.float64)
               for name in ("xCell", "yCell", "zCell")]
        dv = np.asarray(dataset.variables["dvEdge"][:], dtype=np.float64)
        dc = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
    sample = float(np.mean(np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)))
    if sample < 10.0:
        dv = dv * MPAS_EARTH_RADIUS_M
        dc = dc * MPAS_EARTH_RADIUS_M
    return dv.astype(np.float32), dc.astype(np.float32)


def _admission(label: str, mesh: Any, dv_edge: Any, dc_edge: Any, *, fine_dx_m: float,
               log=print) -> dict[str, Any]:
    """Dual edges, cell coordination and Courant, on one mesh's lengths."""

    import numpy as np

    from .cell_coordination_admission import (
        CellCoordinationAdmissionError, admit_cell_coordination,
    )
    from .dual_edge_admission import DualEdgeAdmissionError, admit_dual_edges
    from .timestep_admission import (
        CourantPolicy, TimestepAdmissionError, admit_timestep, edge_length_authority,
    )

    try:
        dual = admit_dual_edges(
            dv_edge, dc_edge, cells_on_edge=mesh.cellsOnEdge,
            cells_on_edge_base=0, mesh_name=label,
        )
    except DualEdgeAdmissionError as error:
        raise PointPlanRefusal(
            f"the generated pair is refused on its own dual edges and is NOT "
            f"registered: {error}"
        ) from error
    try:
        coordination = admit_cell_coordination(
            np.asarray(mesh.arrays["nEdgesOnCell"]), mesh_name=label,
        )
    except CellCoordinationAdmissionError as error:
        raise PointPlanRefusal(
            f"the generated pair is refused on its cell coordination and is NOT "
            f"registered: {error}"
        ) from error
    authority = edge_length_authority(np.asarray(dc_edge, dtype=np.float64))
    chosen = choose_timestep(authority.minimum_m, fine_dx_m=fine_dx_m)
    try:
        timestep = admit_timestep(chosen["dt_seconds"], authority, policy=CourantPolicy())
    except TimestepAdmissionError as error:
        raise PointPlanRefusal(str(error)) from error
    log(
        f"ADMIT {label}: dual edges {dual.as_dict().get('minimum_ratio', '?')}, "
        f"coordination {coordination.as_dict().get('histogram', '?')}, "
        f"min dcEdge {authority.minimum_m:.3f} m, dt {chosen['dt_seconds']:g} s "
        f"(Courant limit {chosen['courant_limit_seconds']:.3f} s, "
        f"{chosen['margin']:.3f}x)"
    )
    return {
        "dual_edge_admission": dual.as_dict(),
        "cell_coordination_admission": coordination.as_dict(),
        "timestep_admission": timestep.as_dict(),
        "timestep_choice": chosen,
        "n_cells": int(mesh.dimensions["nCells"]),
        "n_edges": int(mesh.dimensions["nEdges"]),
        "regional": bool(mesh.is_regional),
    }


def admit_pair(grid: Path, static: Path, *, fine_dx_m: float, log=print) -> dict[str, Any]:
    """The door's own admission pass over a pair it just built."""

    from .mesh import Mesh

    mesh = Mesh.from_netcdf(grid, static)
    return _admission(grid.name, mesh, mesh.dvEdge, mesh.dcEdge, fine_dx_m=fine_dx_m, log=log)


def admit_parent_grid(grid: Path, *, fine_dx_m: float, log=print) -> dict[str, Any]:
    """The same admission pass over a parent that has no static.

    The lengths are the ones its static would have stored
    (:func:`static_equivalent_lengths`), so every number, the timestep
    included, is the one :func:`admit_pair` measures on the parent pair.
    """

    from .mesh import Mesh

    mesh = Mesh.from_netcdf(grid)
    dv_edge, dc_edge = static_equivalent_lengths(grid)
    return _admission(grid.name, mesh, dv_edge, dc_edge, fine_dx_m=fine_dx_m, log=log)


def halo_region(point: tuple[float, float], radius_km: float, pad_scale: float,
                halo_km: float) -> dict[str, Any]:
    """The working halo the halo route builds geography and the vertical on."""

    if not halo_km > 0.0:
        raise PointPlanRefusal(
            f"--halo-km {halo_km} is not positive; the halo route needs room "
            f"outside the cut for the cut's own stencils to read real cells"
        )
    region = cull_region(point, radius_km, pad_scale)
    region["radius_km"] = float(region["radius_km"]) + float(halo_km)
    return region


def _write_region(path: Path, region: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(dict(region), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")
    return path


def _supports_flag(engine: Path, flag: str) -> bool:
    """Ask the executable before selecting a route that requires its API."""
    completed = subprocess.run([str(engine), "--help"], capture_output=True, text=True)
    return flag in (completed.stdout + completed.stderr)


def _element_index(sub: Path, sup: Path, kind: str) -> Any:
    """Where each ``kind`` element of ``sub`` sits in ``sup``, by its centre.

    Both are cuts of one parent, and a cut copies coordinates byte for byte,
    so an element is the element with the same three coordinates.
    """

    import numpy as np
    from netCDF4 import Dataset

    def keys(path: Path) -> Any:
        with Dataset(str(path)) as dataset:
            dataset.set_auto_maskandscale(False)
            return np.ascontiguousarray(np.stack(
                [np.asarray(dataset.variables[f"{axis}{kind}"][:], dtype=np.float64)
                 for axis in "xyz"], axis=1))

    small, large = keys(sub), keys(sup)
    where = {row.tobytes(): index for index, row in enumerate(large)}
    found = np.array([where.get(row.tobytes(), -1) for row in small], dtype=np.int64)
    if np.any(found < 0):
        raise PointPlanRefusal(
            f"{int(np.sum(found < 0))} {kind.lower()}s of {sub.name} are not in "
            f"{sup.name}; the cut does not lie inside its halo"
        )
    return found


def prove_regional_cut(
    *,
    mesh_exe: Path,
    work_grid: Path,
    region: Path,
    direct_cut: Path,
    direct_graph: Path,
    scratch_stem: Path,
) -> str | None:
    """Cut the region out of the halo grid and demand the direct cut's bytes.

    The halo route cuts its static and vertical grid out of the HALO, not out
    of the parent; that cut is only right if it writes exactly what cutting
    the parent writes.  The grid is cut both ways here, in about a second,
    before anything is built on the halo.  ``None`` when the bytes agree; else
    why the halo route is not taken -- an engine that refuses a culled parent
    (one built before it learned to cut one), or a cut that differs.
    """

    probe = Path(f"{scratch_stem}.grid.nc")
    probe_graph = Path(f"{scratch_stem}.graph.info")
    receipt = Path(f"{probe}.cull-receipt.json")
    for path in (probe, probe_graph, receipt):
        path.unlink(missing_ok=True)
    completed = subprocess.run(
        [str(mesh_exe), "--cull-parent", str(work_grid), "--region", str(region),
         "--out", str(probe), "--receipt", str(receipt), "--graph", str(probe_graph)],
        capture_output=True, text=True,
    )
    try:
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip().splitlines()
            return (f"the mesh engine cannot cut a region out of a culled file "
                    f"({tail[-1] if tail else 'exit ' + str(completed.returncode)})")
        same = (probe.read_bytes() == direct_cut.read_bytes()
                and probe_graph.read_bytes() == direct_graph.read_bytes())
        if not same:
            return ("cutting the region out of the halo wrote different bytes than "
                    "cutting it out of the parent")
        return None
    finally:
        for path in (probe, probe_graph, receipt):
            path.unlink(missing_ok=True)


#: Vertical fields the halo check compares, and the element each indexes
#: on its last axis.
_HALO_CHECKED = (
    ("hx", "Cell"), ("zgrid", "Cell"), ("zz", "Cell"), ("dss", "Cell"),
    ("zxu", "Edge"), ("zb", "Edge"), ("zb3", "Edge"),
)


def _vertical_fields(vertical: Any, edge: Any) -> dict[str, Any]:
    return {
        "hx": vertical.hx, "zgrid": vertical.zgrid, "zz": vertical.zz,
        "dss": vertical.dss, "zxu": vertical.zxu, "zb": edge.zb, "zb3": edge.zb3,
    }


def check_halo_vertical(
    *,
    work_grid: Path,
    work_static: Path,
    parent_grid: Path,
    parent_static: Path,
    vertical_spec: Path,
    cut_grid: Path,
    parent_built: Any = None,
    log=print,
) -> dict[str, Any]:
    """Compare every terrain-dependent cut value with the whole parent.

    THE BREAKAGE THIS PREVENTS.  The vertical grid smooths its coordinate
    surfaces thousands of times, so a cut cell's value depends on terrain far
    outside the cut, and a halo too narrow for a place's terrain moves the
    cut's vertical coordinate without any refusal.  Measured on a 937.5 m,
    100 km cut of 41,802 cells: a 400 km halo moved ``hx`` in 11,298 of
    2,340,912 values (up to 1.1 mm) and a 700 km halo in 5 (up to 0.03 mm),
    while 600 and 800 km matched the whole sphere bit for bit. Two halos can
    agree and still share an error caused by terrain outside both. The
    reference therefore uses the parent's exact terrain and operator metrics,
    emitted by the Rust static builder without reading the other geography.
    A disagreement widens the halo, then takes the whole-parent route.
    """

    import numpy as np

    from .mesh import Mesh
    from .vertical_spec import VerticalSpec, build_vertical_from_spec

    spec = VerticalSpec.from_file(vertical_spec)
    started = time.perf_counter()
    with _place_stage("prep.place.vertical", attrs={"operation": "whole-parent-check"}):
        built = build_vertical_from_spec(Mesh.from_netcdf(work_grid, work_static), spec,
                                         regional_halo=True)
        if parent_built is None:
            parent_built = build_vertical_from_spec(Mesh.from_netcdf(parent_grid, parent_static), spec)
    wide_fields, parent_fields = _vertical_fields(*built), _vertical_fields(*parent_built)
    index = {}
    for kind in ("Cell", "Edge"):
        index[kind] = (_element_index(cut_grid, work_grid, kind),
                       _element_index(cut_grid, parent_grid, kind))
    differing: dict[str, int] = {}
    compared = 0
    for name, kind in _HALO_CHECKED:
        wide_index, parent_index = index[kind]
        a = np.ascontiguousarray(np.take(wide_fields[name], wide_index, axis=-1))
        b = np.ascontiguousarray(np.take(parent_fields[name], parent_index, axis=-1))
        compared += int(a.size)
        # A VALUE moved when any of its bytes did; counting bytes overstated it
        # up to eightfold in the receipt.
        moved = int(np.count_nonzero(np.any(
            a.view(np.uint8).reshape(a.size, -1) != b.view(np.uint8).reshape(b.size, -1),
            axis=1)))
        if moved:
            differing[name] = moved
    seconds = time.perf_counter() - started
    agreed = not differing
    log(f"HALO CHECK {'AGREED' if agreed else 'DIFFERED'}: {compared:,} cut values of "
        f"the vertical grid against the whole parent"
        + ("" if agreed else f", differing in {differing}") + f" ({seconds:.1f} s)")
    return {"agreed": agreed, "values_compared": compared, "differing": differing,
            "seconds": round(seconds, 2), "authority": "whole-parent",
            "built": built, "parent_built": parent_built}


def cut_to_rim_hops(halo_grid: Path, cut_grid: Path) -> int:
    """Fewest neighbour steps from a cell of the cut to the halo's rim.

    The rim is every halo cell with a neighbour the halo cut away (a
    ``cellsOnCell`` slot in use that names no cell of the file).  A cut cell
    on the rim is 0 steps from it.
    """

    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(halo_grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        neighbours = np.asarray(dataset.variables["cellsOnCell"][:], dtype=np.int64) - 1
        counts = np.asarray(dataset.variables["nEdgesOnCell"][:], dtype=np.int64)
    n_cells = int(counts.size)
    used = np.arange(neighbours.shape[1])[None, :] < counts[:, None]
    real = used & (neighbours >= 0) & (neighbours < n_cells)
    distance = np.full(n_cells, -1, dtype=np.int64)
    frontier = np.flatnonzero(np.any(used & ~real, axis=1))
    if frontier.size == 0:
        raise PointPlanRefusal(
            f"{halo_grid.name} has no rim: no cell of it lost a neighbour, so it "
            f"is not a cut of the parent"
        )
    in_cut = np.zeros(n_cells, dtype=bool)
    in_cut[_element_index(cut_grid, halo_grid, "Cell")] = True
    distance[frontier] = 0
    steps = 0
    while not np.any(in_cut[frontier]):
        steps += 1
        reached = np.unique(neighbours[frontier][real[frontier]])
        frontier = reached[distance[reached] < 0]
        if frontier.size == 0:
            raise PointPlanRefusal(
                f"no cell of {cut_grid.name} is connected to the rim of "
                f"{halo_grid.name}; the cut does not lie inside its halo"
            )
        distance[frontier] = steps
    return steps


#: One 30-arc-second geography pixel along a meridian, in metres, on MPAS's sphere.
THIRTY_SECOND_PIXEL_M = MPAS_EARTH_RADIUS_M * math.pi / (180.0 * 120.0)


def thirty_second_supersample(grid: Path) -> int:
    """The 30-arc-second sampling factor that gives every cell of ``grid`` a pixel.

    THE BREAKAGE THIS PREVENTS.  ``rw_mpas_static`` averages the 30" pixels
    (terrain, soil, vegetation, texture) whose sample point lands in each cell,
    at one sample per pixel unless ``--supersample-30s N`` asks for N x N.  A
    cell narrower than a pixel (about 927 m) can own no sample at all, and its
    terrain is then 0/0: a 750 m core mesh came out with 4,258 non-finite
    ``ter`` cells and the door stopped at "terrain contains non-finite values".
    Native MPAS has the same knob (``config_30s_supersample_factor``) and the
    same failure when it is left at 1 on a sub-kilometre mesh.

    Every Voronoi cell holds the disc of radius dcEdge/2 about its centre, and a
    regular grid of spacing s puts a point within s/sqrt(2) of anywhere, so
    N > pixel * sqrt(2) / min(dcEdge) is enough.  Meshes whose shortest dcEdge
    is 1,310 m or more get N = 1, the builder's default, so their statics (and
    every native comparison made on them) are unchanged.
    """

    _dv, dc = static_equivalent_lengths(grid)
    positive = dc[dc > 0]
    if positive.size == 0:
        return 1
    ratio = THIRTY_SECOND_PIXEL_M * math.sqrt(2.0) / float(positive.min())
    return 1 if ratio < 1.0 else int(math.floor(ratio)) + 1


def _shortest_dc_edge(grid: Path) -> float:
    """The shortest positive ``dcEdge`` a grid file stores, as stored."""

    import numpy as np
    from netCDF4 import Dataset

    with Dataset(str(grid)) as dataset:
        dataset.set_auto_maskandscale(False)
        dc = np.asarray(dataset.variables["dcEdge"][:], dtype=np.float64)
    dc = dc[np.isfinite(dc) & (dc > 0.0)]
    return float(dc.min()) if dc.size else math.nan


def check_halo_static(
    *,
    halo_grid: Path,
    parent_grid: Path,
    static_receipt: Path,
    cut_grid: Path,
    log=print,
) -> dict[str, Any]:
    """Hold the static built on the halo to the cut, before anything is cut.

    THE BREAKAGE THIS PREVENTS.  Almost every static value is the cell's own
    pixels and its first rings, but a cell of a land field with no valid
    pixel of its own takes its neighbours' value, carried ring by ring until
    one has some (rw-mpas ``spread_to_unsampled``).  Where land holds no valid
    pixel for hundreds of kilometres -- the Antarctic sea-ice margin's albedo
    is fill from 55 S to 65 S -- a cut cell's value comes from that far, and
    a halo whose rim is nearer gives it a different value with nothing in
    the file to say so.  The static builder's receipt says, per dataset, how
    many rings it carried values across (``spread_rounds``); the static is
    the parent's inside the cut when the cut is more rings inside the halo's
    rim than that plus :data:`STATIC_STENCIL_RINGS`.  The categorical
    sampling rate is set by the mesh's shortest spacing, so the halo must
    carry the parent's shortest ``dcEdge`` too.  A receipt that does not say
    (an ``rw_mpas_static`` from before it counted) proves nothing, and the
    door does not use the halo.
    """

    started = time.perf_counter()
    result: dict[str, Any] = {"agreed": False}
    try:
        receipt = json.loads(Path(static_receipt).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        result["why"] = f"the halo static's receipt {Path(static_receipt).name} is unreadable ({error})"
        return result
    datasets = receipt.get("datasets") if isinstance(receipt, dict) else None
    if not isinstance(datasets, list) or not datasets:
        result["why"] = "the halo static's receipt names no datasets"
        return result
    rounds: dict[str, int] = {}
    for row in datasets:
        if (not isinstance(row, dict) or type(row.get("spread_rounds")) is not int
                or row["spread_rounds"] < 0):
            result["why"] = (
                "the static builder does not say how far it carried values to "
                "cells with no pixel of their own (an rw_mpas_static older than "
                "its spread_rounds receipt), so the halo cannot be shown to hold"
            )
            return result
        field = str(row.get("field"))
        rounds[field] = max(rounds.get(field, 0), int(row["spread_rounds"]))
    required_fields = {"ter", "landuse", "soilcat_top", "greenfrac", "albedo12m",
                       "snoalb", "soiltemp", "soilcomp", "soilcl1", "soilcl2", "soilcl3", "soilcl4"}
    missing = required_fields - rounds.keys()
    if missing:
        result["why"] = f"the halo static's receipt omits dataset fields {sorted(missing)}"
        return result
    from .mesh_rows import sha256_file
    stamped = receipt.get("stamped_provenance")
    if not isinstance(stamped, dict) or stamped.get("grid_sha256") != sha256_file(halo_grid):
        result["why"] = "the halo static's receipt does not identify the halo grid's bytes"
        return result
    result["spread_rounds"] = rounds
    halo_dc, parent_dc = _shortest_dc_edge(halo_grid), _shortest_dc_edge(parent_grid)
    result["shortest_dc_edge"] = {"halo": halo_dc, "parent": parent_dc}
    if halo_dc != parent_dc:
        result["why"] = (
            f"the halo's shortest dcEdge {halo_dc!r} is not the parent's "
            f"{parent_dc!r}, and the static builder samples land use at a rate "
            f"set by the mesh's shortest spacing"
        )
        return result
    try:
        hops = cut_to_rim_hops(halo_grid, cut_grid)
    except PointPlanRefusal as error:
        result["why"] = str(error)
        return result
    reach = max(rounds.values(), default=0) + STATIC_STENCIL_RINGS
    result.update({
        "cut_to_rim_rings": hops,
        "reach_rings": reach,
        "agreed": hops > reach,
        "seconds": round(time.perf_counter() - started, 2),
    })
    if not result["agreed"]:
        carried = max(rounds, key=rounds.get) if rounds else "?"
        result["why"] = (
            f"{carried} was carried {rounds.get(carried, 0)} rings to a cell with "
            f"no pixel of its own, and the cut is only {hops} rings inside the "
            f"halo's rim"
        )
    log(f"HALO STATIC {'HOLDS' if result['agreed'] else 'NOT SHOWN'}: values carried at most "
        f"{max(rounds.values(), default=0)} rings (+{STATIC_STENCIL_RINGS} stencil), the cut "
        f"{hops} rings inside the halo's rim")
    return result


class StaticBuilderRefusal(MpasPortError):
    """The static builder wrote a static that does not hold what it declares.

    Deliberately NOT a :class:`PointPlanRefusal`: the halo route reads those
    as a reason to widen the halo or to take the parent's geography, and a
    builder that mislabels one static mislabels every one, so falling back
    would rebuild the same defect at greater cost and then refuse anyway.
    """


#: The coordinate arrays a static carries beside its grid's, and whether
#: each is a Cartesian component (scaled by the sphere radius) or an angle.
_STATIC_COORDINATES = tuple(
    (f"{axis}{kind}", axis in "xyz") for kind in ("Cell", "Edge", "Vertex")
    for axis in ("x", "y", "z", "lat", "lon"))

#: The representation whose promise this door can check bit for bit.
_BINARY64 = "binary64_earth_centred"


def static_coordinate_errors(grid: Path, static: Path) -> list[str]:
    """Does a static built on ``grid`` hold the coordinates it declares?

    THE BREAKAGE THIS PREVENTS (2026-10-15).  woof
    2.9.0's ``rw_mpas_static``, and every build of woof integrate/hex-mpas
    before 46c8a71513e, reads every grid through binary32.  The static of a
    GENERATED grid then declares ``binary64_earth_centred`` and holds binary32
    positions up to 0.44 m off its grid, and this door refused it twice, both
    times blaming the mesh: the vertical check ("dcEdge disagrees with
    spherical cell-center arc length") and ``rw_mpas_mesh --cull-parent``
    ("provenance grid latCell differs").  This names the builder instead,
    right after it wrote the file.

    The comparison is the one ``rw_mpas_mesh --cull-parent
    --cull-provenance-grid`` makes when it binds a static's cut to its grid:
    a binary64 static holds the grid's Cartesian coordinates times the ratio
    of the two files' ``sphere_radius``, in binary64, and the grid's own
    latitudes and longitudes, bit for bit.  A static that declares nothing
    or binary32 -- every native one -- is not judged: native init_atmosphere
    rounds its coordinates by design and the i2 replay proves that rounding.
    """

    import numpy as np
    from netCDF4 import Dataset

    from .mesh import COORDINATE_REPRESENTATION_ATTR

    with Dataset(str(static)) as stat, Dataset(str(grid)) as mesh:
        declared = stat.__dict__.get(COORDINATE_REPRESENTATION_ATTR)
        if declared is None or str(declared).strip() != _BINARY64:
            return []
        stat.set_auto_maskandscale(False)
        mesh.set_auto_maskandscale(False)
        static_radius = float(np.asarray(stat.__dict__["sphere_radius"]).ravel()[0])
        scale = static_radius / float(np.asarray(mesh.__dict__["sphere_radius"]).ravel()[0])
        moved = compared = 0
        worst_m = 0.0
        problems: list[str] = []
        for kind in ("Cell", "Edge", "Vertex"):
            squared = None
            for name, cartesian in _STATIC_COORDINATES:
                if not name.endswith(kind):
                    continue
                stored = stat.variables[name]
                if stored.dtype != np.float64:
                    problems.append(f"{name} is stored as {stored.dtype}")
                    continue
                held = np.asarray(stored[:], dtype=np.float64)
                want = np.asarray(mesh.variables[name][:], dtype=np.float64)
                if held.shape != want.shape:
                    problems.append(f"{name} has shape {held.shape}, its grid's {want.shape}")
                    continue
                if cartesian:
                    want = want * scale
                    error = (held - want) ** 2
                    squared = error if squared is None else squared + error
                compared += int(held.size)
                moved += int(np.count_nonzero(held.view(np.uint64) != want.view(np.uint64)))
            if squared is not None and squared.size:
                worst_m = max(worst_m, float(np.sqrt(np.max(squared))))
    if not moved and not problems:
        return []
    found = "; ".join(problems) if problems else (
        f"{moved:,} of {compared:,} coordinate values are not {grid.name}'s own "
        f"(its binary64 positions times sphere_radius {scale:.17g}, and its angles, bit "
        f"for bit): positions are up to {worst_m:.3g} m off the grid")
    return [
        f"{static.name} declares {_BINARY64}, but {found}.  A static builder that reads "
        f"its grid through binary32 wrote it -- woof 2.9.0's rw_mpas_static does, and so "
        f"does every build of woof integrate/hex-mpas before 46c8a71513e -- and every "
        f"later step here refuses the pair: the vertical check (\"dcEdge disagrees with "
        f"spherical cell-center arc length\") and rw_mpas_mesh --cull-parent (\"provenance "
        f"grid latCell differs\")"
    ]


def generate_point(
    request: PointRequest,
    plan: Mapping[str, Any],
    *,
    out_dir: Path,
    geog: Path | None,
    vertical_spec: Path | None,
    clobber: bool,
    static_exe: Path | None = None,
    n_levels: int = DEFAULT_LEVELS,
    log=print,
    parent_static: bool = False,
    halo_km: float = DEFAULT_HALO_KM,
) -> dict[str, Any]:
    """Build, admit, register and cull the pair the plan priced.

    TWO ROUTES, ONE RESULT.  ``parent_static=True`` is the route this door
    always took: the static (global geography) and the vertical grid are
    built on the whole parent and cut down.  The default HALO route builds
    them on a halo, a wider cut of the parent ``halo_km`` beyond the cut,
    and cuts that down: the cut's grid, static and vertical grid carry the
    same values bit for bit. :func:`check_halo_static` bounds the static's
    dependency reach and :func:`check_halo_vertical` compares with the
    whole parent's exact terrain-based authority on every build, before
    anything is registered or cut; a halo that fails either is doubled once,
    and then the parent route is taken whole.

    What the halo route does NOT build is the parent's static, so the parent
    row is its cull's lineage, described by its grid alone
    (``generated-parent-grid``), and the parent cannot be run.  A caller that
    needs a runnable parent asks for ``parent_static``.
    """

    from . import mesh_rows
    from .cull_door import carry_lineage, cull_one
    from .engines import MESH, STATIC, EngineRefusal, resolve

    # The vertical builder consumes this declaration, so its layer count
    # must also own the parent and inherited cull registrations. A fallback
    # count here made valid non-default vertical artifacts fail at startup.
    if vertical_spec is not None:
        from .vertical_spec import VerticalSpec
        n_levels = VerticalSpec.from_file(vertical_spec).n_vert_levels

    try:
        mesh_exe = resolve(MESH, request.mesh_exe)
        static_bin = resolve(STATIC, static_exe)
    except EngineRefusal as error:
        raise PointPlanRefusal(str(error)) from error
    geog_root = resolve_geog(geog)
    out_dir = Path(out_dir).expanduser().absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    stem = f"point-{row_token(request.point)}-{request.fine_dx_m:g}m-r{request.radius_km:g}km"
    if request.shape is not None:
        import hashlib

        digest = hashlib.sha256(json.dumps(request.shape, sort_keys=True).encode()).hexdigest()[:8]
        stem = f"shape-{row_token(request.point)}-{request.fine_dx_m:g}m-{digest}"
    if request.name:
        stem = request.name
    spec_path = out_dir / f"{stem}.spec.json"
    grid = out_dir / f"{stem}.grid.nc"
    static = out_dir / f"{stem}.static.nc"
    grid_receipt = out_dir / f"{stem}.grid.receipt.json"
    static_receipt = out_dir / f"{stem}.static.receipt.json"
    rows_path = out_dir / mesh_rows.MESH_ROWS_FILENAME
    for existing in (grid, static, rows_path):
        if existing.exists() and not clobber:
            raise PointPlanRefusal(
                f"{existing} exists; pass --clobber to replace it, or a fresh "
                f"--out-dir.  A generated pair that silently overwrote another "
                f"would leave a row file naming bytes that moved"
            )
    spec_path.write_text(
        json.dumps(plan["spec"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    legs: list[dict[str, Any]] = []

    argv: list[Any] = [mesh_exe, "--spec", spec_path, "--out", grid, "--receipt", grid_receipt]
    generator_card = request.card["generator_card"] if request.card else None
    if generator_card:
        argv += ["--card", generator_card]
        if request.vram_gib is not None:
            argv += ["--vram-gib", str(request.vram_gib)]
    if clobber:
        argv.append("--clobber")
    log(f"GENERATE {grid.name} ...")
    done = _run(argv, what="rw_mpas_mesh", log=logs / "mesh.log")
    legs.append({"leg": "mesh", "seconds": round(done.elapsed, 2)})
    log(f"GENERATE {grid.name} written in {done.elapsed:.1f} s")

    def build_static(on_grid: Path, out: Path, receipt: Path, leg: str,
                     *, terrain_only: bool = False) -> None:
        # One content-keyed static per host (woof.hex.static_cache) when a hex
        # cache is configured, shared with every other consumer of the same
        # mesh, builder, geography and options; the build itself still goes
        # through _run, so the per-mesh store (woof.hex.prep_store) and the
        # prep.place.static timeline stage work exactly as before.  With no
        # cache configured the static is built in place, as it always was.
        from .static_cache import build_static as cached_static, door_cache_dir

        options = ["--nominal-dx-m", repr(float(request.fine_dx_m))]
        if terrain_only:
            options.append("--terrain-only")
        supersample = thirty_second_supersample(on_grid)
        if supersample > 1:
            options += ["--supersample-30s", str(supersample)]
            log(f"STATIC {out.name}: 30-second geography sampled {supersample}x{supersample} "
                f"per pixel (the shortest cell is narrower than one pixel)")
        log(f"STATIC {out.name} ...")
        cache = door_cache_dir()
        if cache is None:
            argv = [static_bin, "--grid", on_grid, "--out", out, "--geog", geog_root,
                    "--nominal-dx-m", repr(float(request.fine_dx_m)), "--receipt", receipt]
            if clobber:
                argv.append("--clobber")
            if terrain_only:
                argv.append("--terrain-only")
            if supersample > 1:
                argv += ["--supersample-30s", str(supersample)]
            done = _run(argv, what="rw_mpas_static", log=logs / f"{leg}.log")
            legs.append({"leg": leg, "seconds": round(done.elapsed, 2)})
            log(f"STATIC {out.name} written in {done.elapsed:.1f} s")
        else:
            result = cached_static(
                engine=static_bin, grid=on_grid, geog=geog_root, out=out, receipt=receipt,
                cache=cache, options=options, clobber=clobber,
                runner=lambda command, _log: _run(command, what="rw_mpas_static", log=logs / f"{leg}.log"))
            legs.append({"leg": leg, "seconds": round(result["seconds"], 2),
                         "static_cache": {"hit": result["hit"], "key": result["key"], "cache": result["cache"]}})
            log(f"STATIC {out.name} written in {result['seconds']:.1f} s "
                f"(static cache {'hit' if result['hit'] else 'built'}, key {result['key'][:12]})")
        # The static is checked against its grid before anything reads it
        # (static_coordinate_errors says what this prevents).
        mislabelled = static_coordinate_errors(on_grid, out)
        if mislabelled:
            raise StaticBuilderRefusal(
                f"{mislabelled[0]}.  The builder was {static_bin}.  Use the rw_mpas_static "
                f"this wheel bundles (read first unless --static-exe or "
                f"$WOOF_HEX_RW_MPAS_STATIC names another), or one built from woof "
                f"46c8a71513e or newer")

    def cut(parent_file: Path, region: Path, out: Path, *, graph: Path | None = None,
            provenance_grid: Path | None = None) -> dict[str, Any]:
        row = cull_one(mesh_exe, parent_file, region, out, graph=graph, clobber=clobber,
                       provenance_grid=provenance_grid)
        row["lineage"] = carry_lineage(parent_file, out, drives_boundaries=False)
        return row

    region_path = _write_region(out_dir / f"{stem}.cull-region.json", plan["cull_region"])
    cull_grid = out_dir / f"{stem}-cull.grid.nc"
    cull_static = out_dir / f"{stem}-cull.static.nc"
    work_grid = out_dir / f"{stem}-halo.grid.nc"
    work_static = out_dir / f"{stem}-halo.static.nc"
    work_static_receipt = out_dir / f"{stem}-halo.static.receipt.json"
    halo: dict[str, Any] | None = None
    working: list[Path] = []

    def build_halo(width_km: float) -> dict[str, Any]:
        region = request_halo_region(request, width_km)
        path = _write_region(out_dir / f"{stem}.halo-region.json", region)
        started = time.perf_counter()
        for stale in (work_grid, work_static, work_static_receipt):
            stale.unlink(missing_ok=True)
        row = cut(grid, path, work_grid)
        legs.append({"leg": "halo", "seconds": round(time.perf_counter() - started, 2)})
        for path_ in (work_grid, work_static):
            if path_ not in working:
                working.append(path_)
        cells = (row.get("receipt") or {}).get("region_cells")
        log(f"HALO {work_grid.name}: {width_km:g} km beyond the cut, {cells} cells")
        return {"halo_km": float(width_km), "region": region, "region_file": str(path),
                "cells": cells, "grid": str(work_grid), "static": str(work_static),
                "static_receipt": str(work_static_receipt)}

    # The cull of the grid comes first: the halo route proves the engine on it.
    started = time.perf_counter()
    cuts: dict[str, Any] = {}
    graph = out_dir / f"{stem}-cull.graph.info"
    cuts["grid"] = cull_one(mesh_exe, grid, region_path, cull_grid, graph=graph, clobber=clobber)
    cull_seconds = time.perf_counter() - started

    # THE ROUTE IS DECIDED BEFORE ANYTHING IS REGISTERED OR CUT.  Every check
    # the halo route makes runs first, so the cut's static, the receipt its
    # row names and the vertical grid all come from the one static that held:
    # the halo's that passed every check, or the parent's.  Until 2026-10-01
    # the cut's static was cut from the first halo and its row written before
    # the vertical check ran, so a doubled halo replaced the static receipt
    # that row named, and the fall back to the parent's geography built the
    # vertical grid on the parent but kept the static cut from the halo.
    admission: dict[str, Any] | None = None
    prebuilt = None
    parent_built = None
    terrain_reference = out_dir / f"{stem}-parent-terrain.static.nc"
    terrain_reference_receipt = out_dir / f"{stem}-parent-terrain.static.receipt.json"
    route_note: str | None = None
    if not parent_static:
        missing = [(engine.name, flag) for engine, flag in (
            (mesh_exe, "--cull-provenance-grid"), (static_bin, "--terrain-only"))
            if not _supports_flag(engine, flag)]
        if missing:
            route_note = (f"engine APIs {missing} are unavailable; the halo route cannot "
                          "prove the parent's vertical authority and emit canonical cut provenance")
            log(f"HALO not used: {route_note}; building the parent's own geography")
            parent_static = True
    if not parent_static:
        # The parent's admission needs no geography (admit_parent_grid), so a
        # refused parent stops before any is built.
        admission = admit_parent_grid(grid, fine_dx_m=request.fine_dx_m, log=log)
        if admission["regional"]:
            raise PointPlanRefusal(
                f"{grid} carries a boundary zone; the generator was asked for a "
                f"global mesh and this is not one"
            )
        static_checks: list[dict[str, Any]] = []
        vertical_checks: list[dict[str, Any]] = []
        for attempt, width in enumerate((halo_km, 2.0 * halo_km)):
            halo = build_halo(width)
            halo["static_checks"], halo["checks"] = static_checks, vertical_checks
            route_note = prove_regional_cut(
                mesh_exe=mesh_exe, work_grid=work_grid, region=region_path, direct_cut=cull_grid,
                direct_graph=graph, scratch_stem=out_dir / f"{stem}-halo-probe",
            )
            if route_note is not None:
                break
            try:
                build_static(work_grid, work_static, work_static_receipt,
                             "static" if attempt == 0 else "static-wider")
            except PointPlanRefusal as error:
                route_note = f"the static builder refused the {width:g} km halo ({error})"
                log(f"HALO {route_note}")
                continue
            static_check = check_halo_static(
                halo_grid=work_grid, parent_grid=grid, static_receipt=work_static_receipt,
                cut_grid=cull_grid, log=log,
            )
            static_check["halo_km"] = float(width)
            static_checks.append(static_check)
            if not static_check["agreed"]:
                route_note = static_check["why"]
                continue
            if vertical_spec is None:
                route_note = None
                break
            if parent_built is None:
                try:
                    build_static(grid, terrain_reference, terrain_reference_receipt,
                                 "parent-terrain", terrain_only=True)
                except PointPlanRefusal as error:
                    route_note = f"the static builder cannot emit the parent's terrain authority ({error})"
                    break
                working.append(terrain_reference)
            check = check_halo_vertical(
                work_grid=work_grid, work_static=work_static,
                parent_grid=grid, parent_static=terrain_reference,
                vertical_spec=Path(vertical_spec), cut_grid=cull_grid,
                parent_built=parent_built, log=log,
            )
            built = check.pop("built")
            parent_built = check.pop("parent_built")
            check["halo_km"] = float(width)
            check["parent_terrain_receipt"] = str(terrain_reference_receipt)
            vertical_checks.append(check)
            if check["agreed"]:
                prebuilt, route_note = built, None
                break
            route_note = "the cut's vertical grid differs from the whole parent's authority"
        if route_note is not None:
            log(f"HALO not used: {route_note}; building the parent's own geography")
            parent_static = True
            assert halo is not None
            halo["not_used"] = route_note
            if static_checks or vertical_checks:
                halo["fell_back_to_parent_static"] = True
            prebuilt = None
    # Only now: carry_lineage stamps the cut's file, and prove_regional_cut
    # compares every halo's cut against the cut's bytes as the engine wrote them.
    cuts["grid"]["lineage"] = carry_lineage(grid, cull_grid, drives_boundaries=False)
    if parent_static:
        build_static(grid, static, static_receipt, "static")
        admission = admit_pair(grid, static, fine_dx_m=request.fine_dx_m, log=log)
    assert admission is not None
    if admission["regional"]:
        raise PointPlanRefusal(
            f"{grid} carries a boundary zone; the generator was asked for a "
            f"global mesh and this is not one"
        )
    parent = mesh_rows.describe_generated(
        name=parent_row_name(request.point, request.fine_dx_m, request.background_km,
                             admission["n_cells"]),
        grid=grid, static=static if parent_static else None, spec_path=spec_path,
        generator_receipt=grid_receipt,
        static_receipt=static_receipt if parent_static else None,
        point_deg=request.point, fine_dx_m=request.fine_dx_m,
        core_radius_km=request.radius_km, background_km=request.background_km,
        dt_seconds=admission["timestep_choice"]["dt_seconds"],
        n_levels=n_levels, admission=admission,
    )
    mesh_rows.write_rows(rows_path, [parent])
    log(f"ROW {parent.name} -> {rows_path}")

    # The cull's static: cut from the static the route decided on (the
    # parent's, or the halo's that held), by the same region row as the grid.
    started = time.perf_counter()
    cuts["static"] = cut(static if parent_static else work_static, region_path, cull_static,
                         provenance_grid=cull_grid)
    legs.append({"leg": "cull", "seconds": round(cull_seconds + time.perf_counter() - started, 2)})
    cull_admission = admit_pair(cull_grid, cull_static, fine_dx_m=request.fine_dx_m, log=log)
    if not cull_admission["regional"]:
        raise PointPlanRefusal(
            f"{cull_grid} carries no boundary zone after the cull; the "
            f"region {region_path} did not cut a limited-area mesh"
        )
    lbc_dir = out_dir / f"{stem}-cull.lbc"
    cull_row = mesh_rows.describe_cull(
        name=cull_row_name(request.point, request.fine_dx_m, request.background_km,
                           cull_admission["n_cells"]),
        parent=parent, grid=cull_grid, static=cull_static,
        cull_receipt=cull_grid.with_suffix(cull_grid.suffix + ".cull-receipt.json"),
        cull_region=plan["cull_region"], cull_pad_scale=request.cull_pad_scale,
        lbc_source=(
            f"{lbc_dir} (rw_mpas_lbc on the wps-intermediate route from the "
            f"same source that initialises the cull; the forecast door refuses "
            f"an absent or empty --lbc-dir)"
        ),
        admission=cull_admission,
        dt_seconds=parent.dt_seconds,
        static_receipt=None if parent_static else work_static_receipt,
    )
    mesh_rows.append_row(rows_path, cull_row)
    log(f"ROW {cull_row.name} -> {rows_path} ({cull_row.n_cells:,} cells, "
        f"zone {cull_row.boundary_zone_width}, bdyMask {cull_row.bdy_mask_sha256[:16]}...)")

    vertical: dict[str, Any] | None = None
    if vertical_spec is not None:
        from .vertical_spec import materialize_vertical_artifact

        started = time.perf_counter()
        regional_halo = not parent_static
        source_grid, source_static = (work_grid, work_static) if regional_halo else (grid, static)
        artifact_name = (f"{stem}-halo.vertical.nc" if regional_halo else f"{stem}.vertical.nc")
        artifact = out_dir / artifact_name
        log(f"VERTICAL {artifact.name} from {Path(vertical_spec).name} on the "
            f"{'halo' if regional_halo else 'parent'} ...")
        with _place_stage("prep.place.vertical", attrs={"operation": "artifact", "regional_halo": regional_halo}):
            payload = materialize_vertical_artifact(
                grid=source_grid, static=source_static, spec_path=vertical_spec, output=artifact,
                receipt_path=artifact.with_name(artifact.name + ".receipt.json"),
                regional_halo=regional_halo, prebuilt=prebuilt,
            )
        mint_seconds = time.perf_counter() - started
        cull_artifact = out_dir / f"{stem}-cull.vertical.nc"
        cuts["vertical"] = cut(artifact, region_path, cull_artifact, provenance_grid=cull_grid)
        if regional_halo:
            working.append(artifact)
            working.extend(out_dir.glob("derived-geometry-*.npz"))
        legs.append({"leg": "vertical", "seconds": round(time.perf_counter() - started, 2)})
        vertical = {
            "spec": str(vertical_spec),
            "parent_artifact": None if regional_halo else str(artifact),
            "halo_artifact": str(artifact) if regional_halo else None,
            "parent_receipt": payload.get("receipt") if isinstance(payload, dict) else None,
            "cull_artifact": str(cull_artifact),
            "mint_seconds": round(mint_seconds, 2),
            "n_vert_levels": int(payload["invariants"].get("n_vert_levels", n_levels))
            if isinstance(payload, dict) and isinstance(payload.get("invariants"), dict)
            else n_levels,
        }
        log(f"VERTICAL {cull_artifact.name} culled; init the cull with "
            f"--capsule/--reference {cull_artifact.name}")

    # The halo's files are working files: the cut carries everything the run
    # reads, and their receipts stay beside it.
    for path in working:
        Path(path).unlink(missing_ok=True)
    if halo is not None:
        halo["kept"] = "receipts only; the halo grid, static and vertical grid are deleted after the cut"

    receipt = {
        "schema": "gpuwm-hex.point-generate/v1",
        "route": "parent-static" if parent_static else "halo",
        "route_note": route_note,
        "plan": dict(plan),
        "engines": {"rw_mpas_mesh": str(mesh_exe), "rw_mpas_static": str(static_bin)},
        "geog": str(geog_root),
        "spec": str(spec_path),
        "parent": {
            "row": parent.name, "grid": str(grid),
            "static": str(static) if parent_static else None,
            "n_cells": parent.n_cells, "n_edges": parent.n_edges,
            "dt_seconds": parent.dt_seconds,
            "generator_receipt": str(grid_receipt),
            "static_receipt": str(static_receipt) if parent_static else None,
            "admission": admission,
        },
        "halo": halo,
        "cull": {
            "row": cull_row.name, "grid": str(cull_grid), "static": str(cull_static),
            "n_cells": cull_row.n_cells, "n_edges": cull_row.n_edges,
            "boundary_zone_width": cull_row.boundary_zone_width,
            "bdy_mask_sha256": cull_row.bdy_mask_sha256,
            "region": str(region_path), "files": cuts, "admission": cull_admission,
            "predicted_cells": plan["cull"]["predicted_cells"],
            "lbc_dir": str(lbc_dir),
        },
        "vertical": vertical,
        "rows_file": str(rows_path),
        "legs": legs,
        "next": [
            "woof hex intermediate --source hrrr --from-plan "
            f"{out_dir / (stem + '.point-generate.json')} --grib-dir <fetched HRRR> "
            "--cycle <YYYY-MM-DDTHH> --hours 0-3 --out-dir <MET>",
            f"woof hex init --met <MET/FILE:YYYY-MM-DD_HH> --static {cull_static} "
            + (f"--capsule {vertical['cull_artifact']} --reference {vertical['cull_artifact']} "
               if vertical else "--capsule <culled vertical> --reference <culled vertical> ")
            + f"--out {out_dir / (stem + '-cull.init.nc')} --start-time ... "
            "--nfglevels 51 --nfgsoillevels 4 --extrap-airtemp constant --use-spechumd yes "
            "--theta-adv-order 3 --coef-3rd-order 0.25 --virtual-factor reproduce-fortran "
            "--deep-soil-moisture reproduce-fortran --landuse-table MODIFIED_IGBP_MODIS_NOAH "
            "--frac-seaice yes --tsk-seaice-threshold 100.0 --oned-underflow preserve",
            f"woof hex lbc --grid {out_dir / (stem + '-cull.init.nc')} --met-dir <MET> "
            f"--out-dir {lbc_dir} --start-time ... --stop-time ... --nfglevels 51 "
            "--extrap-airtemp constant --use-spechumd yes",
            f"WOOF_HEX_MESH_ROWS={rows_path} woof hex forecast --mesh {cull_row.name} "
            f"--grid {cull_grid} --static {cull_static} --init {out_dir / (stem + '-cull.init.nc')} "
            f"--lbc-dir {lbc_dir} --hours 3 --start-time ... --out <RUN>",
        ],
    }
    receipt_path = out_dir / f"{stem}.point-generate.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    log(f"RECEIPT {receipt_path}")
    return receipt


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------
def add_point_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("a fine core at a point (instead of --spec)")
    group.add_argument(
        "--point", default=None, metavar="LAT,LON",
        help="centre of the fine core; the spec is the registered 937.5 m "
             "ladder recipe moved to this point")
    group.add_argument(
        "--shape", type=Path, default=None, metavar="FILE",
        help="a drawn fine core instead of --point/--radius-km: a Shape row JSON "
             "(kind polygon with vertices_deg [[lat, lon], ...], lat_lon_box or cap) or "
             "GeoJSON (Polygon, MultiPolygon, Feature, FeatureCollection; the largest "
             "outer ring).  The outline is grown by one ramp of the fine rung so the "
             "whole drawn area gets the fine spacing")
    group.add_argument(
        "--ellipse", default=None, metavar="LAT,LON,SEMI_MAJOR_KM,SEMI_MINOR_KM[,AZIMUTH_DEG]",
        help="a drawn elliptical fine core; the azimuth is the major axis's bearing "
             "clockwise from north (default 0)")
    group.add_argument(
        "--fine-dx-m", type=float, default=937.5, metavar="M",
        help="fine spacing in metres; must be a rung of the halving ladder "
             "under --background-km (default 937.5)")
    group.add_argument(
        "--radius-km", type=float, default=100.0, metavar="KM",
        help="radius of the fine core (default 100)")
    group.add_argument(
        "--background-km", type=float, default=DEFAULT_BACKGROUND_KM, metavar="KM",
        help=f"global background spacing (default {DEFAULT_BACKGROUND_KM:g})")
    group.add_argument(
        "--transition-factor", type=float, default=DEFAULT_TRANSITION_FACTOR, metavar="X",
        help=f"each rung's ramp as a multiple of its own spacing (default {DEFAULT_TRANSITION_FACTOR:g})")
    group.add_argument(
        "--ring-factor", type=float, default=DEFAULT_RING_FACTOR, metavar="X",
        help=f"how many ramps of the next finer rung a cap sits outside it (default {DEFAULT_RING_FACTOR:g})")
    group.add_argument(
        "--cull-pad-scale", type=float, default=DEFAULT_CULL_PAD_SCALE, metavar="X",
        help=f"the limited-area cut, as a multiple of the core radius (default {DEFAULT_CULL_PAD_SCALE:g}, the measured knee)")
    group.add_argument(
        "--name", default=None, metavar="TEXT",
        help="file-name stem and spec name (default: derived from the point)")
    group.add_argument(
        "--generate", action="store_true",
        help="build the pair, admit it, register it as a runtime row beside it, and cull it")
    group.add_argument(
        "--out-dir", type=Path, default=None, metavar="DIR",
        help="where --generate writes (required with --generate)")
    group.add_argument(
        "--geog", type=Path, default=None, metavar="DIR",
        help="WPS_GEOG root for the static (default: woof's fetch-geog ladder)")
    group.add_argument(
        "--vertical-spec", type=Path, default=None, metavar="JSON",
        help="with --generate: mint the parent's native-free vertical artifact "
             "from this gpuwm-hex.vertical-spec/v1 declaration and cull it, so "
             "the cull can be initialised from a regional source")
    group.add_argument("--static-exe", type=Path, default=None, metavar="FILE")
    group.add_argument(
        "--parent-static", action="store_true",
        help="with --generate: build the geography and the vertical grid on the "
             "whole parent, so the parent row is runnable too (minutes of global "
             "geography); by default they are built on a halo around the cut and "
             "the cut's files are the same bytes")
    group.add_argument(
        "--halo-km", type=float, default=DEFAULT_HALO_KM, metavar="KM",
        help=f"how far beyond the cut the halo reaches (default {DEFAULT_HALO_KM:g})")
    group.add_argument("--clobber", action="store_true")


def run_point(arguments: argparse.Namespace) -> int:
    request = request_from_arguments(arguments)
    plan = plan_point(request)
    if arguments.json:
        print(json.dumps(plan, indent=2, sort_keys=True, default=str))
    else:
        print("\n".join(_report(plan)))
    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8", newline="\n",
        )
    if not arguments.generate:
        return 0
    if arguments.out_dir is None:
        raise PointPlanRefusal(
            "--generate needs --out-dir: the pair, the row file and the cull "
            "land beside each other and nothing here guesses where"
        )
    device = plan.get("device")
    if device is not None and not device["fits"]:
        raise PointPlanRefusal(
            f"refused before anything is built: the predicted cull "
            f"({plan['cull']['predicted_cells']:,.0f} cells) needs "
            f"{device['required_free_mib']:,.1f} MiB free on "
            f"{request.card['admission_card']} and the budget is "
            f"{device['budget_mib']:,.1f} MiB, short by {device['short_by_mib']:,.1f} "
            f"MiB.  This card holds a core of up to "
            f"{device['max_core_radius_km_on_this_card']:g} km at this spacing "
            f"and pad; reduce --radius-km, coarsen --fine-dx-m, or name a bigger card"
        )
    generate_point(
        request, plan,
        out_dir=arguments.out_dir, geog=arguments.geog,
        vertical_spec=arguments.vertical_spec, clobber=bool(arguments.clobber),
        static_exe=arguments.static_exe,
        parent_static=bool(getattr(arguments, "parent_static", False)),
        halo_km=float(getattr(arguments, "halo_km", DEFAULT_HALO_KM)),
    )
    return 0


__all__ = [
    "CARD_ALIASES",
    "DEFAULT_BACKGROUND_KM",
    "DEFAULT_CULL_PAD_SCALE",
    "DEFAULT_HALO_KM",
    "STATIC_STENCIL_RINGS",
    "DEFAULT_RING_FACTOR",
    "DEFAULT_TRANSITION_FACTOR",
    "PointPlanRefusal",
    "StaticBuilderRefusal",
    "PointRequest",
    "add_point_arguments",
    "admit_pair",
    "admit_parent_grid",
    "check_halo_static",
    "check_halo_vertical",
    "choose_timestep",
    "cull_region",
    "cull_row_name",
    "cut_to_rim_hops",
    "device_verdict",
    "generate_point",
    "halo_region",
    "ladder_rungs",
    "ladder_spec",
    "max_radius_for_cells",
    "parent_row_name",
    "parse_point",
    "plan_point",
    "drawn_shape",
    "request_cull_region",
    "request_halo_region",
    "request_spec",
    "predicted_cull_cells",
    "prove_regional_cut",
    "spacing_at",
    "spacing_profile",
    "resolve_card",
    "resolve_geog",
    "run_point",
    "static_coordinate_errors",
    "static_equivalent_lengths",
]
