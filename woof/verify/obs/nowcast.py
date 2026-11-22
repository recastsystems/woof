"""The nowcast score: how close an hour of forecast came to the radar.

A local DA window ends with a short forecast.  Until now its receipt counted
the observations the analysis accepted and said nothing at all about the
forecast, so the only way to answer "was it any good" was to look at the
pictures.  This module answers it with a number, using the instrument the
tree already has: the masked neighbourhood FSS of
:mod:`woof.verify.obs.fss`, after Roberts and Lean (2008), against the MRMS
composite.

It is the same engine the free-forecast battery drives and deliberately not
the same registration.  The battery scores hours 2 to 18 of a free forecast
at four thresholds and four neighbourhoods; a nowcast lives inside the first
hour, where the leads are minutes, the useful neighbourhoods are small, and
50 dBZ at a 99 km box is a number nobody will read.  So the pins are their
own, they are hashed the same way, and the digest travels in the receipt.

**The persistence baseline is not optional here, and that is the point.**
In the first hour radar persistence -- the last scan, carried forward
unchanged -- is a genuinely strong forecast, and a model FSS published
beside nothing is unreadable: 0.6 is excellent against a persistence of 0.4
and a failure against a persistence of 0.7.  So every scored lead carries
the persistence score computed under the identical thresholds, boxes and
masks, and the difference model minus persistence for the primary scalar.
The persistence field is an OBSERVATION carried forward, not a model run, so
this is a baseline and not a reference-model control.

**Model and persistence are scored on identical cells.**  The shared
validity is the intersection of the lead scan's validity, the analysis
scan's validity and the model's finite mask, and both scores use it.  Score
them on different masks and the difference between them stops being skill
and starts being coverage.

**A lead that cannot be scored is never a zero.**  A lead whose valid time
has not arrived, or whose scan the archive has not published yet, is
``pending`` with a reason; a lead whose matching window holds no frame, or
none above the coverage floor, is ``missing-obs``; a lead the archive could
not be asked about at all -- no network -- is ``unavailable``, with the
error.  Each of those is a state a reader can act on.  A zero is a claim the
forecast put no echo where the radar saw one, and none of these is that.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from woof.verify.chaos_envelope import canonical_hash
from woof.verify.obs import battery, fss, regrid
from woof.verify.obs.contracts import (
    ModelGrid, ObservedFractionBelowFloor, format_valid_time,
    parse_valid_time,
)

#: The receipt a local DA window writes.  Versioned in the runtime's own
#: ``arwen.local-da-*.v1`` family because that is the document family a
#: consumer of a local DA case already reads.
NOWCAST_SCORE_SCHEMA = "arwen.local-da-nowcast-score.v1"

#: The pins, hashed before a score is looked at, in the registration
#: discipline :mod:`woof.verify.obs.registration` sets for the battery.
NOWCAST_REGISTRATION_SCHEMA = "arwen.local-da-nowcast-registration.v1"

#: Leads after the analysis, in minutes.  A forecast scores whichever of
#: these it reaches: a 1800 s forecast scores 15 and 30, a 3600 s forecast
#: all four.  Leads the forecast does not reach are not in the receipt at
#: all, because a lead that was never run is not a lead that went unscored.
DEFAULT_LEAD_MINUTES: tuple[int, ...] = (15, 30, 45, 60)

#: Reflectivity thresholds (dBZ).  50 dBZ is dropped from the battery's set:
#: inside one hour on a 192 km domain the 50 dBZ population is often a
#: handful of cells, and an FSS over a handful of cells is noise with a
#: decimal point.
DEFAULT_THRESHOLDS_DBZ: tuple[float, ...] = (20.0, 30.0, 40.0)

#: Neighbourhood half-widths (cells): 9 km and 27 km boxes at dx 3 km.
DEFAULT_HALF_WIDTHS: tuple[int, ...] = (1, 4)

#: The primary scalar: FSS at 30 dBZ in the 9 km box.  One number per lead.
DEFAULT_PRIMARY_THRESHOLD_DBZ = 30.0
DEFAULT_PRIMARY_HALF_WIDTH = 1

#: Matching window around a lead's valid time.  MRMS composites arrive every
#: two minutes, so plus or minus 4 minutes always holds a frame when the feed
#: is healthy and never reaches into a different weather situation.
DEFAULT_MATCH_TOLERANCE_SECONDS = 240

#: Coverage floor for frame selection, the battery's registered value.
DEFAULT_FRAME_COVERAGE_FLOOR = 0.9

#: Remap pins.  ``nearest`` because composite reflectivity is a maximum and
#: averaging it would be a physical statement nobody made.
DEFAULT_REGRID_METHOD = "nearest"
DEFAULT_REGRID_MAX_DISTANCE_M = 4500.0

#: Interior rim beyond the specified and relaxation rows.  The MASK is the
#: battery's -- domain minus the specified and relaxation rows minus a
#: physical rim, intersected with the shared validity, scored under the zero
#: boundary -- and the rim's VALUE is this registration's own, because the
#: battery's 45 km was sized for an 18-hour free forecast on a continental
#: domain and a nowcast runs for an hour on a couple of hundred kilometres.
#:
#: MEASURED, on the real 64 x 64 three-window case of 2026-09-10 12:00Z at
#: 3 km: a 45 km rim leaves 576 of 4096 cells, and at every one of the six
#: scored valid times the observed 30 dBZ coverage of that remnant is
#: exactly zero while the full interior carries 2 to 5 percent.  A rim that
#: deletes all of the weather does not make a conservative score, it makes a
#: score of a quiet box.  Three cells leaves 2304 cells and keeps the storms.
DEFAULT_INTERIOR_RIM_M = 9000.0
DEFAULT_BOUNDARY_WIDTH_CELLS = 5

#: How long after a lead's valid time a missing scan is still read as "the
#: archive has not published it yet" rather than "this lead has no
#: observation".  MRMS composites appear within a couple of minutes; an hour
#: is generous on purpose, because calling a late frame missing is the one
#: error that turns into a permanent hole in a receipt.
DEFAULT_PUBLICATION_GRACE_SECONDS = 3600

#: Lead states.  Exactly one of these is on every lead row.
SCORED = "scored"
PENDING = "pending"
MISSING_OBS = "missing-obs"
UNAVAILABLE = "unavailable"
LEAD_STATUSES = (SCORED, PENDING, MISSING_OBS, UNAVAILABLE)


class ObservationsUnreachable(RuntimeError):
    """The archive could not be asked: no network, or the door refused.

    Distinct from "there is no frame" on purpose.  An offline box must record
    that it was offline; recording it as a missing observation would put a
    permanent hole in a receipt for a temporary condition.
    """


def nowcast_parameters(
        *, lead_minutes: Sequence[int] = DEFAULT_LEAD_MINUTES,
        thresholds_dbz: Sequence[float] = DEFAULT_THRESHOLDS_DBZ,
        half_widths: Sequence[int] = DEFAULT_HALF_WIDTHS,
        primary_threshold_dbz: float = DEFAULT_PRIMARY_THRESHOLD_DBZ,
        primary_half_width: int = DEFAULT_PRIMARY_HALF_WIDTH,
        match_tolerance_seconds: int = DEFAULT_MATCH_TOLERANCE_SECONDS,
        frame_coverage_floor: float = DEFAULT_FRAME_COVERAGE_FLOOR,
        regrid_method: str = DEFAULT_REGRID_METHOD,
        regrid_max_distance_m: float = DEFAULT_REGRID_MAX_DISTANCE_M,
        interior_rim_m: float = DEFAULT_INTERIOR_RIM_M,
        boundary_width_cells: int = DEFAULT_BOUNDARY_WIDTH_CELLS,
        neighborhood_boundary: str = fss.ZERO_BOUNDARY,
        publication_grace_seconds: int = DEFAULT_PUBLICATION_GRACE_SECONDS,
        ) -> dict[str, object]:
    """Every choice the score makes, frozen before a score is looked at."""
    leads = tuple(int(value) for value in lead_minutes)
    thresholds = tuple(float(value) for value in thresholds_dbz)
    widths = tuple(int(value) for value in half_widths)
    if not leads or sorted(leads) != list(leads) or len(set(leads)) != len(leads):
        raise ValueError("nowcast leads must be ascending, unique and non-empty")
    if any(value <= 0 for value in leads):
        raise ValueError("a nowcast lead is a positive number of minutes")
    if not thresholds or not widths:
        raise ValueError("nowcast scoring needs thresholds and neighbourhoods")
    if float(primary_threshold_dbz) not in thresholds:
        raise ValueError("the primary threshold must be one of the scored thresholds")
    if int(primary_half_width) not in widths:
        raise ValueError("the primary neighbourhood must be one of the scored ones")
    if any(value < 0 for value in widths):
        raise ValueError("neighbourhood half-widths must be non-negative")
    if int(match_tolerance_seconds) <= 0:
        raise ValueError("the matching window is a positive number of seconds")
    if not 0.0 <= float(frame_coverage_floor) <= 1.0:
        raise ValueError("the frame coverage floor is a fraction in [0, 1]")
    if neighborhood_boundary not in fss.BOUNDARIES:
        raise ValueError(f"unknown neighbourhood boundary {neighborhood_boundary!r}")
    if int(boundary_width_cells) < 0 or float(interior_rim_m) < 0.0:
        raise ValueError("the interior mask widths must be non-negative")
    if int(publication_grace_seconds) < 0:
        raise ValueError("the publication grace is a non-negative duration")
    return {
        "lead_minutes": list(leads),
        "lead_policy": (
            "a forecast scores whichever registered leads it reaches; a lead "
            "the forecast never ran is absent from the receipt rather than "
            "recorded unscored"),
        "model_field": (
            "column maximum over k of REFL_10CM, the forecast scheme's own "
            "reflectivity operator"),
        "observed_quantity": "composite_reflectivity",
        "thresholds_dbz": list(thresholds),
        "neighborhood_half_widths_cells": list(widths),
        "primary_threshold_dbz": float(primary_threshold_dbz),
        "primary_half_width_cells": int(primary_half_width),
        "primary_scalar": (
            "FSS at the primary threshold and primary neighbourhood, per lead"),
        "obs_match_tolerance_seconds": int(match_tolerance_seconds),
        "frame_coverage_floor": float(frame_coverage_floor),
        "frame_selection": (
            "inside the matching tolerance, the scored frame is the nearest "
            "frame whose observed fraction meets the coverage floor (ties to "
            "the earlier frame); the tolerance never widens"),
        "regrid_method": str(regrid_method),
        "regrid_max_distance_m": float(regrid_max_distance_m),
        "interior_rim_m": float(interior_rim_m),
        "boundary_width_cells": int(boundary_width_cells),
        "interior_mask": (
            "the battery's mask: the domain minus the specified and "
            "relaxation rows minus the physical rim, intersected with the "
            "shared validity of the lead scan, the analysis scan and the "
            "model's finite cells"),
        "neighborhood_boundary": str(neighborhood_boundary),
        "publication_grace_seconds": int(publication_grace_seconds),
        "fss_definition": (
            "Roberts and Lean 2008 with a shared validity mask: neighbourhood "
            "fraction = boxcar(event AND valid)/boxcar(valid); cells with an "
            "empty neighbourhood are dropped; FSS = 1 when both fraction "
            "fields are identically zero over the scored region"),
        "fss_useful_definition": "0.5 + f_obs/2",
        "persistence_baseline": (
            "the MRMS scan nearest the analysis time, carried forward "
            "unchanged to the lead's valid time and scored against the lead's "
            "scan under the same thresholds, neighbourhoods and masks; the "
            "carried field is an observation, not a model run"),
        "persistence_mask_policy": (
            "model and persistence score the identical cells: the lead scan's "
            "validity, the analysis scan's validity and the model's finite "
            "mask intersected, so their difference is skill and not coverage"),
        "difference_definition": (
            "model primary scalar minus persistence primary scalar at the "
            "same lead; positive means the forecast beat the last radar scan"),
        "lead_statuses": list(LEAD_STATUSES),
        "missing_policy": (
            "a lead whose valid time has not arrived, or whose scan the "
            "archive has not published inside the publication grace, is "
            "pending with a reason; a lead whose matching window holds no "
            "frame at or above the coverage floor is missing-obs; a lead the "
            "archive could not be asked about is unavailable with the error. "
            "None of the three is ever recorded as a score of zero"),
    }


def make_nowcast_registration(*, evaluating_tree: Mapping[str, object],
                              parameters: Mapping[str, object] | None = None,
                              ) -> dict[str, object]:
    """Freeze the pins, hash them, and stamp the tree that will evaluate."""
    pins = dict(nowcast_parameters() if parameters is None else parameters)
    tree = dict(evaluating_tree)
    for key in ("package", "version"):
        if not str(tree.get(key, "")).strip():
            raise ValueError(
                f"the evaluating tree record needs {key}; a score whose "
                f"evaluator cannot be named is a score nobody can reproduce")
    registration = {
        "schema": NOWCAST_REGISTRATION_SCHEMA,
        "evaluating_tree": tree,
        "parameters": pins,
    }
    registration["registration_sha256"] = canonical_hash(pins)
    return registration


def validate_nowcast_registration(registration: Mapping[str, object]
                                  ) -> dict[str, object]:
    """Refuse a registration whose pins no longer hash to its own digest."""
    value = dict(registration)
    missing = ({"schema", "evaluating_tree", "parameters",
                "registration_sha256"} - set(value))
    if missing:
        raise ValueError(f"the nowcast registration is missing {sorted(missing)}")
    if value["schema"] != NOWCAST_REGISTRATION_SCHEMA:
        raise ValueError("nowcast registration schema mismatch")
    parameters = value["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("nowcast registration parameters must be an object")
    if canonical_hash(parameters) != value["registration_sha256"]:
        raise ValueError("the nowcast registration hash does not match its pins")
    return value


def evaluating_tree_record() -> dict[str, object]:
    """What is about to do the scoring, named so a number can be reproduced."""
    from woof import __version__
    from woof.supervisor import git_commit

    return {"package": "woof", "version": str(__version__),
            "commit": git_commit(),
            "commit_scope": ("HEAD of the checkout enclosing the imported "
                             "package, which is not a statement about an "
                             "installed wheel's own bytes")}


def lead_valid_times(analysis_time: str, lead_minutes: Sequence[int]
                     ) -> tuple[str, ...]:
    """Seam timestamps for a set of nowcast leads."""
    start = parse_valid_time(analysis_time)
    return tuple(format_valid_time(start + timedelta(minutes=int(minutes)))
                 for minutes in lead_minutes)


def scored_region(shape: tuple[int, int], registration: Mapping[str, object],
                  dx_m: float) -> np.ndarray:
    """The interior a nowcast scores, from the registered mask widths."""
    parameters = registration["parameters"]
    return battery.interior_mask(
        shape,
        boundary_width_cells=int(parameters["boundary_width_cells"]),
        rim_m=float(parameters["interior_rim_m"]), dx_m=float(dx_m))


@dataclass(frozen=True)
class _Resolved:
    """One observed frame, already remapped onto the model grid."""

    values: np.ndarray
    valid: np.ndarray
    record: dict[str, object]


def _scan_record(observed, requested: str, detail=None) -> dict[str, object]:
    """What the receipt says about the one frame a lead was scored against.

    ``detail`` is an optional hook on the observation source: the archive
    object a frame came from is the source's knowledge, not the seam's, and a
    reader chasing a number wants the bucket key beside the digest rather
    than in a separate block they have to join by hand.
    """
    provenance = observed.provenance
    offset = (parse_valid_time(observed.valid_time)
              - parse_valid_time(requested)).total_seconds()
    record = {
        "valid_time": observed.valid_time,
        "requested_valid_time": requested,
        "offset_seconds": float(offset),
        "sha256": str(provenance.sha256).lower(),
        "uri": provenance.uri,
        "source": provenance.source,
        "product": provenance.product,
        "fetched_at": provenance.fetched_at,
        "is_stub": bool(provenance.is_stub),
    }
    if detail is not None:
        record.update(detail(observed.valid_time))
    return record


def _absent_scan_state(valid_time: str, *, now: datetime,
                       grace_seconds: int) -> tuple[str, str]:
    """Is an absent scan not-yet-published, or genuinely absent?"""
    valid = parse_valid_time(valid_time).replace(tzinfo=timezone.utc)
    if now < valid:
        return PENDING, (
            f"the lead's valid time {valid_time} has not arrived; this lead "
            f"is scored when it does")
    age = (now - valid).total_seconds()
    if age <= float(grace_seconds):
        return PENDING, (
            f"no MRMS composite is published for {valid_time} yet "
            f"({age:.0f} s after the valid time, inside the "
            f"{grace_seconds} s publication grace)")
    return MISSING_OBS, ""


def score_nowcast(*, registration: Mapping[str, object], analysis_time: str,
                  model, observations, grid: ModelGrid,
                  region: np.ndarray | None = None,
                  now: datetime | None = None) -> dict[str, object]:
    """Score every registered lead this forecast reaches, against MRMS.

    ``model`` answers ``lead_minutes()`` with the leads it holds a frame for
    and ``composite(minutes)`` with that frame's composite reflectivity.
    ``observations`` answers ``field(valid_time)`` with the seam's
    :class:`~woof.verify.obs.contracts.ObsGridField`, raising
    :class:`LookupError` when the matching window holds no frame,
    :class:`~woof.verify.obs.contracts.ObservedFractionBelowFloor` when
    none of them clears the coverage floor, and
    :class:`ObservationsUnreachable` when the archive could not be asked.

    Nothing here writes a file and nothing here raises for a lead: an
    unscoreable lead is a row that says why.
    """
    validate_nowcast_registration(registration)
    parameters = registration["parameters"]
    thresholds = [float(value) for value in parameters["thresholds_dbz"]]
    half_widths = [int(value) for value in
                   parameters["neighborhood_half_widths_cells"]]
    primary_key = (float(parameters["primary_threshold_dbz"]),
                   int(parameters["primary_half_width_cells"]))
    boundary = str(parameters["neighborhood_boundary"])
    grace = int(parameters["publication_grace_seconds"])
    moment = (datetime.now(timezone.utc) if now is None
              else now.astimezone(timezone.utc))
    parse_valid_time(analysis_time)

    if region is None:
        region = scored_region(grid.shape, registration, grid.dx_m)
    region = np.asarray(region, dtype=bool)
    if region.shape != grid.shape:
        raise ValueError("the scored region must match the model grid")

    state: dict[str, object] = {"plan": None, "source_shape": None}
    scan_detail = getattr(observations, "scan_detail", None)

    def resolve(requested: str) -> _Resolved:
        observed = observations.field(requested)
        if observed.quantity != parameters["observed_quantity"]:
            raise ValueError(
                f"the observation source serves {observed.quantity!r}, the "
                f"registration scores {parameters['observed_quantity']!r}")
        plan = state["plan"]
        if plan is None:
            plan = regrid.build_plan(
                source_latitude=observed.latitude,
                source_longitude=observed.longitude,
                destination_latitude=grid.latitude,
                destination_longitude=grid.longitude,
                method=str(parameters["regrid_method"]),
                max_distance_m=float(parameters["regrid_max_distance_m"]))
            state["plan"] = plan
            state["source_shape"] = tuple(observed.values.shape)
        elif tuple(observed.values.shape) != tuple(state["source_shape"]):
            raise ValueError(
                "the observation grid changed between leads of one window; "
                "one remap plan must serve every lead or the leads are not "
                "comparable")
        values, valid = regrid.apply_plan(plan, observed.values, observed.valid)
        return _Resolved(values=values, valid=valid,
                         record=_scan_record(observed, requested, scan_detail))

    analysis_scan: _Resolved | None = None
    analysis_problem: dict[str, object] | None = None
    try:
        analysis_scan = resolve(analysis_time)
    except ObservedFractionBelowFloor as outage:
        analysis_problem = {
            "status": MISSING_OBS, "reason": str(outage),
            "minimum_observed_fraction": outage.minimum_observed_fraction,
            "candidate_frames": outage.candidates}
    except ObservationsUnreachable as error:
        analysis_problem = {"status": UNAVAILABLE, "reason": str(error)}
    except LookupError as error:
        status, reason = _absent_scan_state(analysis_time, now=moment,
                                            grace_seconds=grace)
        analysis_problem = {"status": status, "reason": reason or str(error)}

    available = {int(value) for value in model.lead_minutes()}
    registered = [int(value) for value in parameters["lead_minutes"]]
    requested_times = lead_valid_times(analysis_time, registered)
    rows: list[dict[str, object]] = []
    for minutes, requested in zip(registered, requested_times):
        if minutes not in available:
            continue
        row: dict[str, object] = {"lead_minutes": int(minutes),
                                  "valid_time": requested}
        try:
            lead_scan = resolve(requested)
        except ObservedFractionBelowFloor as outage:
            row.update(status=MISSING_OBS, reason=str(outage),
                       minimum_observed_fraction=outage.minimum_observed_fraction,
                       candidate_frames=outage.candidates)
            rows.append(row)
            continue
        except ObservationsUnreachable as error:
            row.update(status=UNAVAILABLE, reason=str(error))
            rows.append(row)
            continue
        except LookupError as error:
            status, reason = _absent_scan_state(requested, now=moment,
                                                grace_seconds=grace)
            row.update(status=status, reason=reason or str(error))
            rows.append(row)
            continue

        forecast = np.asarray(model.composite(int(minutes)), dtype=np.float64)
        if forecast.shape != grid.shape:
            raise ValueError(
                f"the {minutes} minute model composite is {forecast.shape}, "
                f"the scored grid is {grid.shape}")
        finite = np.isfinite(forecast)
        masks = [lead_scan.valid, finite]
        if analysis_scan is not None:
            masks.append(analysis_scan.valid)
        valid = fss.shared_validity(*masks)
        interior_cells = int(np.count_nonzero(region))
        coverage = (float(np.count_nonzero(lead_scan.valid & region)
                          / interior_cells) if interior_cells else 0.0)
        if not np.any(valid & region):
            row.update(status=MISSING_OBS,
                       reason=("no cell of the scored interior is valid in the "
                               "lead scan, the analysis scan and the model at "
                               "once; there is nothing to compare"),
                       observed_coverage_fraction=coverage,
                       scan=lead_scan.record)
            rows.append(row)
            continue

        matrix = fss.fss_matrix(
            forecast, lead_scan.values, valid=valid, thresholds=thresholds,
            half_widths=half_widths, score_mask=region, boundary=boundary)
        primary = matrix[primary_key]
        row.update(
            status=SCORED, scan=lead_scan.record,
            observed_coverage_fraction=coverage,
            interior_valid_fraction=(
                float(np.count_nonzero(valid & region) / interior_cells)
                if interior_cells else 0.0),
            fss=fss.matrix_records(matrix, dx_m=grid.dx_m),
            primary_fss=float(primary.fss),
            primary_fss_useful=float(primary.fss_useful),
            primary_observed_base_rate=float(primary.observed_base_rate),
            primary_model_base_rate=float(primary.model_base_rate),
            scored_cells=int(primary.scored_cells),
            regrid=state["plan"].record())

        if analysis_scan is None:
            row["persistence"] = dict(analysis_problem or {
                "status": UNAVAILABLE,
                "reason": "the analysis-time scan was not resolved"})
            row["difference_primary"] = None
        else:
            carried = fss.fss_matrix(
                analysis_scan.values, lead_scan.values, valid=valid,
                thresholds=thresholds, half_widths=half_widths,
                score_mask=region, boundary=boundary)
            carried_primary = carried[primary_key]
            row["persistence"] = {
                "status": SCORED,
                "basis": str(parameters["persistence_baseline"]),
                "scan": analysis_scan.record,
                "fss": fss.matrix_records(carried, dx_m=grid.dx_m),
                "primary_fss": float(carried_primary.fss),
                "primary_fss_useful": float(carried_primary.fss_useful),
            }
            row["difference_primary"] = float(primary.fss - carried_primary.fss)
        rows.append(row)

    scored = [row for row in rows if row["status"] == SCORED]
    return {
        "schema": NOWCAST_SCORE_SCHEMA,
        "analysis_time": analysis_time,
        "scored_utc": format_valid_time(moment.replace(tzinfo=None)),
        "evaluating_tree": dict(registration["evaluating_tree"]),
        "registration_sha256": registration["registration_sha256"],
        "registration": {
            "schema": registration["schema"],
            "parameters": dict(parameters),
        },
        "primary": {
            "threshold_dbz": primary_key[0],
            "half_width_cells": primary_key[1],
            "box_length_m": fss.box_length_m(primary_key[1], grid.dx_m),
            "statistic": (
                f"FSS at {primary_key[0]:g} dBZ in the "
                f"{fss.box_length_m(primary_key[1], grid.dx_m) / 1000.0:g} km "
                f"neighbourhood"),
        },
        "grid": {
            "shape": [int(grid.shape[0]), int(grid.shape[1])],
            "dx_m": float(grid.dx_m),
            "scored_interior_cells": int(np.count_nonzero(region)),
            "interior_rim_m": float(parameters["interior_rim_m"]),
            "boundary_width_cells": int(parameters["boundary_width_cells"]),
        },
        "model": dict(model.record()),
        "observations": dict(observations.record()),
        "analysis_scan": (analysis_scan.record if analysis_scan is not None
                          else dict(analysis_problem or {})),
        "leads": rows,
        "leads_scored": [int(row["lead_minutes"]) for row in scored],
        "leads_pending": [int(row["lead_minutes"]) for row in rows
                          if row["status"] == PENDING],
        "leads_missing_obs": [int(row["lead_minutes"]) for row in rows
                              if row["status"] == MISSING_OBS],
        "leads_unavailable": [int(row["lead_minutes"]) for row in rows
                              if row["status"] == UNAVAILABLE],
        "primary_by_lead": {str(int(row["lead_minutes"])): row["primary_fss"]
                            for row in scored},
        "persistence_primary_by_lead": {
            str(int(row["lead_minutes"])): row["persistence"]["primary_fss"]
            for row in scored if row["persistence"].get("status") == SCORED},
        "difference_by_lead": {
            str(int(row["lead_minutes"])): row["difference_primary"]
            for row in scored if row["difference_primary"] is not None},
    }


def lead_summary(document: Mapping[str, object]) -> list[dict[str, object]]:
    """The compact per-lead rows a status document and a run receipt carry.

    Three numbers and a state per lead: the model's primary scalar, the
    persistence baseline at the same lead, and the difference.  A reader who
    sees only this must still be able to tell a good forecast from one that
    lost to the last radar scan, which is why the baseline is in the compact
    form and not only in the full receipt.

    The two base rates travel with them for the same reason.  A box with no
    observed echo at the primary threshold scores FSS 1 against a model that
    also drew nothing, and its persistence baseline scores 1 as well, so the
    difference is 0 and the compact row reads exactly like a forecast that
    put the storms in the right place.  ``primary_observed_base_rate`` of
    0.0 is the fact that separates them, and dropping it here is what made
    the first fresh run on this build publish a perfect score for clear air
    on the status document, the execution report and the completion record
    at once.
    """
    rows: list[dict[str, object]] = []
    for lead in document.get("leads", ()):
        row = {
            "lead_minutes": int(lead["lead_minutes"]),
            "valid_time": lead["valid_time"],
            "status": lead["status"],
            "primary_fss": lead.get("primary_fss"),
            "persistence_primary_fss": (
                (lead.get("persistence") or {}).get("primary_fss")),
            "difference_primary": lead.get("difference_primary"),
            "primary_observed_base_rate": lead.get(
                "primary_observed_base_rate"),
            "primary_model_base_rate": lead.get("primary_model_base_rate"),
        }
        if lead["status"] != SCORED:
            row["reason"] = lead.get("reason", "")
        rows.append(row)
    return rows


class CachedCompositeSource:
    """The MRMS composite, fetched on demand into the case's own cache.

    Satisfies the seam a scorer reads: ``field(valid_time)`` answers with an
    :class:`~woof.verify.obs.contracts.ObsGridField` or raises the three
    distinguishable failures the nowcast registration records separately.
    """

    def __init__(self, cache, *, match_seconds: int,
                 coverage_floor: float) -> None:
        self.cache = cache
        self.match_seconds = int(match_seconds)
        self.coverage_floor = float(coverage_floor)
        self._source = None
        self._packs: tuple[str, ...] = ()
        self._details: dict[str, dict[str, object]] = {}
        self._walk_problem: dict[str, str] = {}

    def _rebuilt(self):
        from woof.obs.sources import MrmsCompositeSource

        packs = tuple(str(path) for path in self.cache.pack_paths())
        if not packs:
            raise LookupError(
                "the case's observation cache holds no composite packs")
        if self._source is None or packs != self._packs:
            self._source = MrmsCompositeSource(
                [Path(value) for value in packs], self.cache.geometry_path,
                match_seconds=self.match_seconds,
                minimum_observed_fraction=self.coverage_floor)
            self._packs = packs
        return self._source

    def field(self, valid_time: str):
        from woof.obs.mrms_fetch import MrmsArchiveUnavailable
        from woof.verify.obs.contracts import ObservedFractionBelowFloor

        walked: list[object] = []
        try:
            frame = self.cache.ensure(valid_time)
        except MrmsArchiveUnavailable as error:
            raise ObservationsUnreachable(str(error)) from error
        if frame.observed_fraction < self.coverage_floor:
            # The registered selection walks outward inside the same window
            # for a better-covered frame; it can only walk over frames that
            # are on disk.
            try:
                walked = list(self.cache.ensure_window(valid_time))
            except MrmsArchiveUnavailable as error:
                # An offline run refuses the listing outright, and so does a
                # box that cannot reach the archive.  The nearest frame came
                # back a line ago and is on disk: score it rather than throw
                # the lead away, and say on the lead that the outward walk
                # did not run.  The alternative recorded `unavailable` for a
                # scan the case already held.
                self._walk_problem[frame.valid_time] = (
                    "the poorly covered nearest scan was scored because the "
                    f"frames around it could not be listed: {error}")
        # Every frame the walk pulled is recorded, not only the one ensure()
        # picked: the registered selection can score any of them, and a lead
        # scored on a neighbour used to lose the archive identity of the
        # object it was actually scored from.
        for candidate in [frame, *walked]:
            self._details[candidate.valid_time] = {
                "archive_bucket": candidate.bucket,
                "archive_key": candidate.key,
                "archive_object_uri": candidate.object_uri,
                "archive_object_sha256": candidate.object_sha256,
                "pack_path": candidate.pack_path,
                "pack_sha256": candidate.pack_sha256,
                "observed_fraction": candidate.observed_fraction,
            }
        try:
            return self._rebuilt().field(valid_time)
        except ObservedFractionBelowFloor:
            raise
        except MrmsArchiveUnavailable as error:  # pragma: no cover - defensive
            raise ObservationsUnreachable(str(error)) from error

    def scan_detail(self, valid_time: str) -> dict[str, object]:
        detail = dict(self._details.get(str(valid_time), {}))
        problem = self._walk_problem.get(str(valid_time))
        if problem:
            detail["coverage_walk"] = problem
        return detail

    def record(self) -> dict[str, object]:
        return self.cache.record()


__all__ = [
    "CachedCompositeSource",
    "DEFAULT_BOUNDARY_WIDTH_CELLS", "DEFAULT_FRAME_COVERAGE_FLOOR",
    "DEFAULT_HALF_WIDTHS", "DEFAULT_INTERIOR_RIM_M", "DEFAULT_LEAD_MINUTES",
    "DEFAULT_MATCH_TOLERANCE_SECONDS", "DEFAULT_PRIMARY_HALF_WIDTH",
    "DEFAULT_PRIMARY_THRESHOLD_DBZ", "DEFAULT_PUBLICATION_GRACE_SECONDS",
    "DEFAULT_REGRID_MAX_DISTANCE_M", "DEFAULT_REGRID_METHOD",
    "DEFAULT_THRESHOLDS_DBZ", "LEAD_STATUSES", "MISSING_OBS",
    "NOWCAST_REGISTRATION_SCHEMA", "NOWCAST_SCORE_SCHEMA",
    "ObservationsUnreachable", "PENDING", "SCORED", "UNAVAILABLE",
    "evaluating_tree_record", "lead_summary", "lead_valid_times",
    "make_nowcast_registration", "nowcast_parameters", "score_nowcast",
    "scored_region", "validate_nowcast_registration",
]
