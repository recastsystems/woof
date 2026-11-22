"""The two documents that decide what is a threat and what beats what.

NOTHING IN THIS PACKAGE BRANCHES ON A PHENOMENON.  A tropical cyclone and
a four-ingredient fire-weather region reach the same eight functions; what
differs between them is a row in ``threat-metrics.v3.json``.  That is
the arbitrary acceptance test applied one level up from a driving
model: if adding "detect derechos" or "detect polar lows" needs a
function, this layer has failed and must be redesigned, not patched.

A COMPOUND THREAT IS ALSO A ROW, which is what v2 added.  A field row's
operands come either from the file (``source_variables``) or from OTHER
FIELD ROWS (``inputs``), so a derived quantity -- a bulk shear, a vapour
pressure deficit, a moisture transport -- is a composition of rows.  The
``threshold_margin`` kind turns any field into a dimensionless distance
past a threshold and ``extremum_of`` takes the weakest of several, so
"hot AND dry AND windy over dry fuel" is five field rows and one metric
row and reaches the SAME ``area_threshold_exceedance`` detector that
finds a blob of reflectivity.

Every vocabulary below is CLOSED and refused at load, by name, with the
breakage stated.  An open vocabulary is how a per-phenomenon branch
re-enters wearing JSON's clothes: a document that may carry
``"kind": "tropical_cyclone_special"`` will eventually carry it, and the
code that honours it is the bandaid.  A document that cannot express it
never gets the chance.

Both documents ship inside the wheel (``woof/hex/data/swath/``) and both
can be replaced from the command line.  A user adding a threat row does
not edit source, which is what makes this reachable after
``pip install woof hex``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import SwathDocumentError

#: Where the shipped defaults live inside the installed package.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "swath"
DEFAULT_METRICS = DATA_DIR / "threat-metrics.v3.json"
DEFAULT_POLICY = DATA_DIR / "placement-policy.v2.json"

METRICS_SCHEMA = "gpuwm-hex.threat-metrics.v3"
POLICY_SCHEMA = "gpuwm-hex.placement-policy.v2"

#: The schema strings this build replaced, so a document written for the
#: previous one is refused with the MOVE named rather than with a bare
#: mismatch.  v1 had no ``rank.intensity_reference`` on a metric row, so
#: every row's intensity was divided by one policy-wide scale: pascals and
#: dBZ were summed on one axis and differed by about 470x from units
#: alone.  Measured twice on real global forecasts; see
#: ``evidence/swath-metrics-20260826/``.
SUPERSEDED_SCHEMAS = {
    "gpuwm-hex.threat-metrics.v2": (
        "a v2 swath row carries no 'cull_pad_scale', so this build would have "
        "to guess how far past the swath ring to take the limited-area cut. "
        "The implicit v2 answer was 'at the ring', which is measured to hand a "
        "71 km parent state directly to cells the fine core's own size -- "
        "about 7.6:1 at the boundary interface, where cutting 1.35x wider "
        "measures 5.7:1 and 1.70x measures 3.4:1 for no extra forecast "
        "(evidence/nest-ratio-20260827/). It has no default precisely because "
        "that answer should be chosen rather than inherited. Add "
        "'cull_pad_scale': <at least 1.0> to every metric row's 'swath' block "
        "-- 1.0 reproduces v2 exactly -- and set the schema to " + "gpuwm-hex.threat-metrics.v3"
    ),
    "gpuwm-hex.threat-metrics.v1": (
        "a v1 metric row carries no 'rank.intensity_reference', so this build "
        "would have no field-units-to-significance conversion for it and would "
        "fall back to ranking pascals against dBZ on one axis. Add "
        "'rank': {'intensity_reference': <field units>} to every metric row and "
        "set the schema to " + METRICS_SCHEMA
    ),
    "gpuwm-hex.placement-policy.v1": (
        "a v1 placement policy carries no 'budget.maximum_per_threat_class', and "
        "its 'intensity' rank term's scale was in the units of ONE field. Set "
        "the intensity term's scale to 1.0 (the units now come off the metric "
        "row), declare 'maximum_per_threat_class', and set the schema to "
        + POLICY_SCHEMA
    ),
}

# ---------------------------------------------------------------------------
# closed vocabularies
# ---------------------------------------------------------------------------
#: How a per-cell scalar is derived.
#:
#: THE FIRST FOUR ARE LEAVES: they read a published history variable and
#: reduce its raw shape to one value per cell.  EVERYTHING ELSE IS AN
#: OPERATOR over per-cell scalars, and an operator may take its operands
#: either from the file (``source_variables``) or from OTHER FIELD ROWS
#: (``inputs``).  That one property is what makes a derived or compound
#: quantity a row instead of a function: integrated transport, a bulk
#: shear, a vapour-pressure deficit and a four-ingredient fire-weather
#: conjunction are all built by composing these, and none of them appears
#: anywhere in this package by name.
#:
#: ``threshold_margin`` is the load-bearing one.  It converts a field into
#: a DIMENSIONLESS distance past a threshold, which is what lets
#: ``extremum_of`` with ``minimum`` express "all of these conditions hold
#: here" over quantities measured in kelvin, per cent, metres per second
#: and cubic metres per cubic metre at the same time.
DERIVATION_KINDS = (
    # leaves: read the file
    "direct",
    "level_slice",
    "vertical_extremum",
    "time_rate",
    # operators: read scalars, from the file or from other rows
    "vector_magnitude",
    "sea_level_reduction",
    "linear_combination",
    "product",
    "ratio",
    "threshold_margin",
    "extremum_of",
    "saturation_vapour_pressure",
    "vapour_pressure",
)

#: Which kinds read the file's raw shapes (and so must name history
#: variables), and how many operands each takes.  ``None`` as the maximum
#: means variadic.  A kind absent from this table cannot be loaded, which
#: is the second lock on the vocabulary: adding a member to
#: ``DERIVATION_KINDS`` without an arity row refuses at load rather than
#: at derive time, three stages downstream of the document that caused it.
DERIVATION_ARITY: Mapping[str, tuple[int, int | None, str]] = MappingProxyType({
    "direct": (1, 1, "variables"),
    "level_slice": (1, 1, "variables"),
    "vertical_extremum": (1, 1, "variables"),
    "time_rate": (1, 1, "variables"),
    "vector_magnitude": (2, 2, "either"),
    "sea_level_reduction": (3, 3, "either"),
    "linear_combination": (1, None, "either"),
    "product": (2, None, "either"),
    "ratio": (2, 2, "either"),
    "threshold_margin": (1, 1, "either"),
    "extremum_of": (2, None, "either"),
    "saturation_vapour_pressure": (1, 1, "either"),
    "vapour_pressure": (2, 2, "either"),
})

#: Why a kind's arity is what it is, for the kinds whose operands are
#: ORDERED and whose order cannot be recovered from the names.  A refusal
#: that only counts is a refusal that tells an author to add an operand
#: without telling them which one goes where.
ARITY_HINTS: Mapping[str, str] = MappingProxyType({
    "sea_level_reduction": (
        "They are ordered [pressure, height, temperature]: the reduction is "
        "p * exp(g z / (R Tbar)) and there is no way to guess which published "
        "name is which"
    ),
    "vapour_pressure": (
        "They are ordered [mixing_ratio, pressure]: e = w p / (eps + w), and "
        "the two are not interchangeable by seven orders of magnitude"
    ),
    "ratio": (
        "They are ordered [numerator, denominator]. A ratio taken the wrong "
        "way round is still a finite number at every cell, so nothing "
        "downstream would notice"
    ),
    "vector_magnitude": (
        "Two components of one vector; the magnitude does not care which is "
        "which, but the count does"
    ),
})

#: Where a threat definition applies at all.  A region is part of the
#: DEFINITION, not a filter bolted on: "severe convection" as the term is
#: operationally defined (hail, damaging wind, tornado criteria) is a
#: North American construct, and a row carrying those thresholds fires
#: continuously over the tropical oceans where the construct means nothing.
#: The shapes are the ones the mesh grammar already reads, spelled the
#: same way, so a region and a cull region are the same kind of object.
REGION_KINDS = ("global", "bounding_box", "cap")

#: How features are found in that scalar.  ``extremum_ball`` finds POINTS
#: (a cyclone centre); ``area_threshold_exceedance`` finds REGIONS (a
#: convective area).  Both return the same feature record, which is what
#: lets everything downstream stay phenomenon-blind.
DETECTOR_KINDS = ("extremum_ball", "area_threshold_exceedance")

#: How a confirming field is reduced over a ball around a candidate.
AGGREGATION_KINDS = ("ball_maximum", "ball_minimum", "ball_mean")

COMPARISONS = ("at_least", "at_most")
EXTREMA = ("minimum", "maximum")

#: When a swath ignites.  ``time_of_first_exceedance`` is what makes a
#: delayed start a COLUMN rather than a code path: the hour comes from the
#: coarse forecast's own field, not from an operator.
START_POLICY_KINDS = ("cycle_start", "time_of_first_exceedance")

#: The terms a rank score may be built from.  Each is a pure function of a
#: track record; none may read the clock, the filesystem, or a case name.
RANK_TERM_KINDS = (
    "metric_extremum",
    "track_frames",
    "track_displacement_km",
    "feature_area_km2",
)

#: Keys a tiebreak may order on.  The last element of a tiebreak must be
#: one that is unique across candidates or the order is not total.
TIEBREAK_KEYS = ("rank_score", "threat_class", "metric_id", "feature_id")

#: Tiebreak keys that are unique per candidate within one cycle.
TOTAL_ORDER_KEYS = ("feature_id",)


def _refuse_unknown(raw: Mapping[str, Any], known: Sequence[str], what: str) -> None:
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise SwathDocumentError(
            f"{what} carries unknown key(s) {unknown}; known keys are "
            f"{sorted(known)}. An unknown key is silently ignored by a permissive "
            "loader, so a misspelled threshold would run the whole cycle at the "
            "default and place swaths nobody asked for. A new option needs a "
            "schema version, not a tolerated key"
        )


def _refuse_unknown_member(value: Any, vocabulary: Sequence[str], what: str) -> str:
    if value not in vocabulary:
        raise SwathDocumentError(
            f"{what} is {value!r}, which is not in the closed vocabulary "
            f"{list(vocabulary)}. This vocabulary is closed on purpose: an open "
            "one is how a per-phenomenon code path re-enters as data. Adding a "
            "member means implementing it here and versioning the schema"
        )
    return str(value)


def _positive(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SwathDocumentError(f"{what} must be a number, not {value!r}") from error
    if not number > 0.0 or number != number or number in (float("inf"),):
        raise SwathDocumentError(
            f"{what} is {value!r}; it must be a finite positive number, because "
            "every use of it is a distance, a duration or a spacing and none of "
            "those has a meaning at zero or below"
        )
    return number


# ---------------------------------------------------------------------------
# metric rows
# ---------------------------------------------------------------------------
def _operand_list(raw: Any, what: str, row_id: str) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str) or not raw:
        raise SwathDocumentError(
            f"field row {row_id!r}: {what!r} must be a non-empty list of names. It "
            "is a LIST even for one name because the publication manifest is derived "
            "from it, and a manifest that has to guess whether a string is one name "
            "or many is a manifest that will one day publish nothing"
        )
    return tuple(str(name) for name in raw)


@dataclass(frozen=True)
class FieldRow:
    """One per-cell scalar: a leaf read from the file, or an operator over
    other field rows.

    ``source_variables`` names history variables; ``inputs`` names OTHER
    FIELD ROWS.  A row uses exactly one of the two, and which one is legal
    is decided by :data:`DERIVATION_ARITY`, never by the row's id.  A leaf
    kind reads the file's raw shapes and so must name variables; every
    operator kind takes per-cell scalars and does not care where they came
    from, which is the property that makes "integrated transport",
    "bulk shear" and "hot AND dry AND windy" compositions of rows rather
    than functions in this package.
    """

    id: str
    derivation_kind: str
    source_variables: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    level_index: int | None = None
    extremum: str | None = None
    lapse_rate_k_per_m: float | None = None
    coefficients: tuple[float, ...] = ()
    offset: float = 0.0
    coefficient: float = 1.0
    denominator_floor: float | None = None
    comparison: str | None = None
    threshold: float | None = None
    scale: float | None = None
    units: str = ""
    description: str = ""

    @property
    def operands(self) -> tuple[str, ...]:
        """The names this row reads, whichever side they came from."""

        return self.inputs if self.inputs else self.source_variables

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FieldRow":
        _refuse_unknown(
            raw,
            ("id", "source_variables", "inputs", "derivation", "units", "description"),
            "a threat-metrics field row",
        )
        row_id = str(raw.get("id", ""))
        if not row_id:
            raise SwathDocumentError("a field row has no 'id'")
        derivation = raw.get("derivation")
        if not isinstance(derivation, Mapping):
            raise SwathDocumentError(
                f"field row {row_id!r} has no 'derivation' object; a field "
                "without a derivation cannot be computed from a history file"
            )
        _refuse_unknown(
            derivation,
            (
                "kind", "level_index", "extremum", "lapse_rate_k_per_m",
                "coefficients", "offset", "coefficient", "denominator_floor",
                "comparison", "threshold", "scale",
            ),
            "a field derivation",
        )
        kind = _refuse_unknown_member(
            derivation.get("kind"), DERIVATION_KINDS, "derivation.kind"
        )
        minimum, maximum, side = DERIVATION_ARITY[kind]

        has_variables = raw.get("source_variables") is not None
        has_inputs = raw.get("inputs") is not None
        if has_variables and has_inputs:
            raise SwathDocumentError(
                f"field row {row_id!r} names BOTH 'source_variables' and 'inputs'. A "
                "row reads its operands from one side or the other: the file, or "
                "other rows. Naming both leaves the order of the combined operand "
                "list undefined, and a derivation whose operand ORDER is undefined "
                "computes a different quantity depending on how the loader felt"
            )
        if not has_variables and not has_inputs:
            raise SwathDocumentError(
                f"field row {row_id!r} names neither 'source_variables' nor "
                f"'inputs', so its derivation {kind!r} has nothing to operate on"
            )
        if has_inputs and side == "variables":
            raise SwathDocumentError(
                f"field row {row_id!r}: derivation {kind!r} reads the history file's "
                "own array shapes, so its operands must be history VARIABLE names in "
                "'source_variables', not other field rows in 'inputs'. Only the "
                f"leaf kinds "
                f"{sorted(k for k, v in DERIVATION_ARITY.items() if v[2] == 'variables')} "
                "touch raw shapes; every other kind takes per-cell scalars and "
                "accepts either side"
            )
        operands = _operand_list(
            raw.get("inputs") if has_inputs else raw.get("source_variables"),
            "inputs" if has_inputs else "source_variables",
            row_id,
        )
        if len(operands) < minimum or (maximum is not None and len(operands) > maximum):
            wanted = (
                f"exactly {minimum}" if minimum == maximum
                else f"at least {minimum}" if maximum is None
                else f"between {minimum} and {maximum}"
            )
            raise SwathDocumentError(
                f"field row {row_id!r}: derivation {kind!r} takes {wanted} "
                f"source variables, not {len(operands)}."
                + (" " + ARITY_HINTS[kind] if kind in ARITY_HINTS else "")
            )

        if kind == "level_slice" and derivation.get("level_index") is None:
            raise SwathDocumentError(
                f"field row {row_id!r}: derivation 'level_slice' needs "
                "'level_index'; without it the field is a whole 3-D array and the "
                "detector has no scalar to search"
            )
        if kind in ("vertical_extremum", "extremum_of"):
            _refuse_unknown_member(
                derivation.get("extremum"), EXTREMA, "derivation.extremum"
            )
        if kind == "sea_level_reduction":
            # Ordered [pressure, height, temperature]: the reduction is
            # p * exp(g z / (R Tbar)) and there is no way to guess which
            # published name is which.
            pass
        coefficients: tuple[float, ...] = ()
        if kind == "linear_combination":
            declared = derivation.get("coefficients")
            if declared is None:
                raise SwathDocumentError(
                    f"field row {row_id!r}: derivation 'linear_combination' needs "
                    "'coefficients', one per operand. A default of all ones would "
                    "turn every difference this row was written to express into a "
                    "sum, silently, and a dewpoint depression computed as a sum is "
                    "not wrong by a little"
                )
            if not isinstance(declared, Sequence) or isinstance(declared, str):
                raise SwathDocumentError(
                    f"field row {row_id!r}: 'coefficients' must be a list of numbers"
                )
            if len(declared) != len(operands):
                raise SwathDocumentError(
                    f"field row {row_id!r}: {len(declared)} coefficient(s) against "
                    f"{len(operands)} operand(s). They are matched BY POSITION, so a "
                    "mismatched list would either drop a term or pair a coefficient "
                    "with the wrong field"
                )
            coefficients = tuple(float(value) for value in declared)
        if kind == "threshold_margin":
            _refuse_unknown_member(
                derivation.get("comparison"), COMPARISONS, "derivation.comparison"
            )
            if derivation.get("threshold") is None:
                raise SwathDocumentError(
                    f"field row {row_id!r}: derivation 'threshold_margin' needs a "
                    "'threshold' in the operand's own units; it is the value the "
                    "margin is measured from"
                )
            _positive(
                derivation.get("scale"),
                f"field row {row_id!r} derivation.scale",
            )
        if kind == "ratio":
            _positive(
                derivation.get("denominator_floor"),
                f"field row {row_id!r} derivation.denominator_floor",
            )
        return cls(
            id=row_id,
            derivation_kind=kind,
            source_variables=() if has_inputs else operands,
            inputs=operands if has_inputs else (),
            level_index=(
                None if derivation.get("level_index") is None
                else int(derivation["level_index"])
            ),
            extremum=(
                None if derivation.get("extremum") is None
                else str(derivation["extremum"])
            ),
            lapse_rate_k_per_m=(
                None if derivation.get("lapse_rate_k_per_m") is None
                else _positive(
                    derivation["lapse_rate_k_per_m"], "derivation.lapse_rate_k_per_m"
                )
            ),
            coefficients=coefficients,
            offset=float(derivation.get("offset", 0.0)),
            coefficient=float(derivation.get("coefficient", 1.0)),
            denominator_floor=(
                None if derivation.get("denominator_floor") is None
                else float(derivation["denominator_floor"])
            ),
            comparison=(
                None if derivation.get("comparison") is None
                else str(derivation["comparison"])
            ),
            threshold=(
                None if derivation.get("threshold") is None
                else float(derivation["threshold"])
            ),
            scale=(
                None if derivation.get("scale") is None
                else float(derivation["scale"])
            ),
            units=str(raw.get("units", "")),
            description=str(raw.get("description", "")),
        )


@dataclass(frozen=True)
class Detector:
    kind: str
    extremum: str | None
    comparison: str | None
    threshold: float
    search_radius_km: float | None
    minimum_separation_km: float | None
    minimum_area_km2: float | None
    maximum_area_km2: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], metric_id: str) -> "Detector":
        _refuse_unknown(
            raw,
            (
                "kind",
                "extremum",
                "comparison",
                "threshold",
                "search_radius_km",
                "minimum_separation_km",
                "minimum_area_km2",
                "maximum_area_km2",
            ),
            f"metric {metric_id!r} detector",
        )
        kind = _refuse_unknown_member(raw.get("kind"), DETECTOR_KINDS, "detector.kind")
        if "threshold" not in raw:
            raise SwathDocumentError(
                f"metric {metric_id!r}: the detector has no 'threshold'. Without one "
                "every local extremum on the mesh is a feature, so a 40,962-cell "
                "global forecast yields thousands of candidates and the ranking "
                "decides the cycle by arithmetic noise"
            )
        if kind == "extremum_ball":
            _refuse_unknown_member(raw.get("extremum"), EXTREMA, "detector.extremum")
            _positive(raw.get("search_radius_km"), f"metric {metric_id!r} search_radius_km")
            _positive(
                raw.get("minimum_separation_km"),
                f"metric {metric_id!r} minimum_separation_km",
            )
        else:
            _refuse_unknown_member(
                raw.get("comparison"), COMPARISONS, "detector.comparison"
            )
            floor = _positive(
                raw.get("minimum_area_km2"), f"metric {metric_id!r} minimum_area_km2"
            )
            # THE BREAKAGE A CEILING PREVENTS, measured on a real 24 h
            # global forecast: the moisture-transport row's connected
            # regions reached 11,433,855 km^2 -- one region covering two
            # per cent of the planet.  A region that size has a centroid in
            # the middle of an ocean basin, a track that is the wander of
            # that centroid, and a swath that could cover three per cent of
            # it.  It is a climate belt, not a placeable feature, and
            # nothing downstream can tell the difference: it scores, it
            # ranks, it wins slots, and the frames it produces look like a
            # grid placed on nothing in particular.
            ceiling = _positive(
                raw.get("maximum_area_km2"), f"metric {metric_id!r} maximum_area_km2"
            )
            if ceiling <= floor:
                raise SwathDocumentError(
                    f"metric {metric_id!r}: maximum_area_km2={ceiling} is not above "
                    f"minimum_area_km2={floor}, so no region can ever be both large "
                    "enough to keep and small enough to place"
                )
        return cls(
            kind=kind,
            extremum=None if raw.get("extremum") is None else str(raw["extremum"]),
            comparison=None if raw.get("comparison") is None else str(raw["comparison"]),
            threshold=float(raw["threshold"]),
            search_radius_km=(
                None if raw.get("search_radius_km") is None
                else float(raw["search_radius_km"])
            ),
            minimum_separation_km=(
                None if raw.get("minimum_separation_km") is None
                else float(raw["minimum_separation_km"])
            ),
            minimum_area_km2=(
                None if raw.get("minimum_area_km2") is None
                else float(raw["minimum_area_km2"])
            ),
            maximum_area_km2=(
                None if raw.get("maximum_area_km2") is None
                else float(raw["maximum_area_km2"])
            ),
        )


@dataclass(frozen=True)
class ConfirmRow:
    field: str
    aggregation_kind: str
    radius_km: float
    comparison: str
    value: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], metric_id: str) -> "ConfirmRow":
        _refuse_unknown(
            raw,
            ("field", "aggregation", "comparison", "value"),
            f"metric {metric_id!r} confirm_with row",
        )
        aggregation = raw.get("aggregation")
        if not isinstance(aggregation, Mapping):
            raise SwathDocumentError(
                f"metric {metric_id!r}: a confirm_with row needs an 'aggregation' "
                "object saying how the confirming field is reduced near the candidate"
            )
        _refuse_unknown(aggregation, ("kind", "radius_km"), "a confirm aggregation")
        return cls(
            field=str(raw["field"]),
            aggregation_kind=_refuse_unknown_member(
                aggregation.get("kind"), AGGREGATION_KINDS, "aggregation.kind"
            ),
            radius_km=_positive(
                aggregation.get("radius_km"), f"metric {metric_id!r} aggregation.radius_km"
            ),
            comparison=_refuse_unknown_member(
                raw.get("comparison"), COMPARISONS, "confirm_with comparison"
            ),
            value=float(raw["value"]),
        )


@dataclass(frozen=True)
class TrackRow:
    maximum_speed_km_per_hour: float
    minimum_frames: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], metric_id: str) -> "TrackRow":
        _refuse_unknown(
            raw,
            ("maximum_speed_km_per_hour", "minimum_frames"),
            f"metric {metric_id!r} track row",
        )
        frames = int(raw.get("minimum_frames", 1))
        if frames < 1:
            raise SwathDocumentError(
                f"metric {metric_id!r}: minimum_frames={frames}; a track needs at "
                "least one frame to exist at all"
            )
        return cls(
            maximum_speed_km_per_hour=_positive(
                raw.get("maximum_speed_km_per_hour"),
                f"metric {metric_id!r} maximum_speed_km_per_hour",
            ),
            minimum_frames=frames,
        )


def _cull_pad_scale(value: Any, metric_id: str) -> float:
    """Refuse a cull pad that would cut into the science region.

    THE BREAKAGE THIS PREVENTS: a pad below 1.0 takes the limited-area cut
    INSIDE the swath the placement layer sized, so the seven driven rings --
    which the culler grows OUTWARD from the requested polygon -- land on
    ground the row asked to be forecast.  Measured on exactly that mistake
    (2026-08-27, evidence/nest-ratio-20260827/): at 0.45 the interior tracked
    the no-boundary run at r = 0.443 on vertical velocity where the same
    configuration cut at 1.0 reached 0.620, and the cut ran through the middle
    of the cyclone it was placed for.  A pad is for keeping MORE of the
    parent's ramp, never less of the swath.
    """

    if value is None:
        raise SwathDocumentError(
            f"metric {metric_id!r} carries no 'cull_pad_scale'. It has no "
            "default because the value that used to be implicit -- cutting "
            "exactly at the swath ring, 1.0 -- is the one measured to hand the "
            "coarse parent's state straight to the fine core, and a silent "
            "default would keep every row there without anybody choosing it. "
            "Declare it: 1.0 reproduces the previous behaviour exactly"
        )
    scale = _positive(value, f"metric {metric_id!r} cull_pad_scale")
    if scale < 1.0:
        raise SwathDocumentError(
            f"metric {metric_id!r}: cull_pad_scale={scale} is below 1.0, so the "
            "limited-area cut would fall INSIDE the swath this row sized. The "
            "culler grows its seven driven rings outward from the requested "
            "polygon, so those rings would overwrite ground the row asked to "
            "have forecast. A pad keeps more of the parent's resolution ramp; "
            "it is not a way to shrink a swath -- shrink half_width_km for that"
        )
    return scale


@dataclass(frozen=True)
class SwathRow:
    """The half-width profile and the mesh request one metric asks for."""

    half_width_km: float
    flare_km_per_hour: float
    maximum_half_width_km: float
    spacing_km: float
    transition_cells: float
    lead_hours: float
    path_step_km: float
    cap_points: int
    #: How far past the swath ring the limited-area CULL is taken, as a
    #: geodesic similarity factor about the ring's own centroid.  1.0 cuts at
    #: the ring, which is what shipped before this column existed.
    #:
    #: THIS IS THE LADDER, AND IT IS A NUMBER IN A ROW.  The parent is a
    #: variable-resolution mesh that ramps from the swath's own spacing out to
    #: the background, so the atmosphere between the fine core and the cut is
    #: already resolved at intermediate resolutions -- MPAS's own form of an
    #: intermediate level, a region of one mesh rather than another forecast.
    #: Cutting at the ring throws all of it away and lands the coarse parent's
    #: state directly on cells the fine core's size.  Cutting wider keeps it,
    #: and the driven rings move out to where the mesh is coarser, so the
    #: interface ratio falls with no extra integration.
    #:
    #: MEASURED (2026-08-27, an RTX 5090, evidence/nest-ratio-20260827/):
    #: on one placed swath, over one shared patch of 2,937 cells, with the
    #: coarse parent's finest spacing at 71.0 km --
    #:
    #:   pad 0.45 -> ring cells  5.03-5.49 km, about 13.7:1 at the interface
    #:   pad 1.00 -> ring cells  8.99-9.65 km, about  7.6:1  (what shipped)
    #:   pad 1.35 -> ring cells 10.86-14.30 km, about 5.7:1
    #:   pad 1.70 -> ring cells 15.89-25.27 km, about 3.4:1
    #:
    #: The count of intermediate levels is therefore not a discrete rung list
    #: at all: it is how much of the parent's own ramp a row chooses to keep,
    #: and adding a differently-rungged ladder is one number in one row with
    #: no source edit anywhere.  ``tools/prove_cull_pad_is_data.py`` asserts
    #: exactly that, with ``git status --porcelain`` unchanged.
    cull_pad_scale: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], metric_id: str) -> "SwathRow":
        known = tuple(field.name for field in fields(cls))
        _refuse_unknown(raw, known, f"metric {metric_id!r} swath row")
        half_width = _positive(raw.get("half_width_km"), f"metric {metric_id!r} half_width_km")
        maximum = _positive(
            raw.get("maximum_half_width_km"), f"metric {metric_id!r} maximum_half_width_km"
        )
        if maximum < half_width:
            raise SwathDocumentError(
                f"metric {metric_id!r}: maximum_half_width_km={maximum} is below "
                f"half_width_km={half_width}, so the swath would be clamped narrower "
                "than its own base width at hour zero and the flare would run backwards"
            )
        flare = float(raw.get("flare_km_per_hour", 0.0))
        if flare < 0.0:
            raise SwathDocumentError(
                f"metric {metric_id!r}: flare_km_per_hour={flare} is negative. The "
                "flare exists because track uncertainty GROWS with lead; a swath "
                "that narrows downstream is narrower than the error it must contain"
            )
        return cls(
            half_width_km=half_width,
            flare_km_per_hour=flare,
            maximum_half_width_km=maximum,
            spacing_km=_positive(raw.get("spacing_km"), f"metric {metric_id!r} spacing_km"),
            transition_cells=_positive(
                raw.get("transition_cells"), f"metric {metric_id!r} transition_cells"
            ),
            lead_hours=_positive(raw.get("lead_hours"), f"metric {metric_id!r} lead_hours"),
            path_step_km=_positive(
                raw.get("path_step_km", 100.0), f"metric {metric_id!r} path_step_km"
            ),
            cap_points=int(raw.get("cap_points", 16)),
            cull_pad_scale=_cull_pad_scale(raw.get("cull_pad_scale"), metric_id),
        )


@dataclass(frozen=True)
class StartPolicy:
    kind: str
    lead_margin_hours: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], metric_id: str) -> "StartPolicy":
        _refuse_unknown(
            raw, ("kind", "lead_margin_hours"), f"metric {metric_id!r} start_policy"
        )
        kind = _refuse_unknown_member(
            raw.get("kind"), START_POLICY_KINDS, "start_policy.kind"
        )
        margin = float(raw.get("lead_margin_hours", 0.0))
        if margin < 0.0:
            raise SwathDocumentError(
                f"metric {metric_id!r}: lead_margin_hours={margin} is negative, which "
                "would ignite the swath AFTER the exceedance it exists to resolve"
            )
        return cls(kind=kind, lead_margin_hours=margin)


@dataclass(frozen=True)
class Region:
    """Where a threat DEFINITION applies, in the mesh grammar's own shapes.

    A ``global`` region is the absence of one.  A ``bounding_box`` whose
    west edge is east of its east edge wraps the antimeridian, which is
    the only way to write a Pacific box at all.
    """

    kind: str
    south_deg: float = -90.0
    north_deg: float = 90.0
    west_deg: float = -180.0
    east_deg: float = 180.0
    center_deg: tuple[float, float] = (0.0, 0.0)
    radius_km: float = 0.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], metric_id: str) -> "Region":
        _refuse_unknown(
            raw,
            ("kind", "south_deg", "north_deg", "west_deg", "east_deg",
             "center_deg", "radius_km"),
            f"metric {metric_id!r} region",
        )
        kind = _refuse_unknown_member(raw.get("kind"), REGION_KINDS, "region.kind")
        if kind == "global":
            return cls(kind=kind)
        if kind == "bounding_box":
            south = float(raw.get("south_deg", -90.0))
            north = float(raw.get("north_deg", 90.0))
            if not -90.0 <= south < north <= 90.0:
                raise SwathDocumentError(
                    f"metric {metric_id!r}: region south_deg={south} north_deg="
                    f"{north} is not an ordered latitude span inside [-90, 90]. A "
                    "reversed span selects nothing, and a row whose region selects "
                    "nothing detects nothing while reporting a quiet world"
                )
            return cls(
                kind=kind, south_deg=south, north_deg=north,
                west_deg=float(raw.get("west_deg", -180.0)),
                east_deg=float(raw.get("east_deg", 180.0)),
            )
        centre = raw.get("center_deg")
        if not isinstance(centre, Sequence) or isinstance(centre, str) or len(centre) != 2:
            raise SwathDocumentError(
                f"metric {metric_id!r}: region kind 'cap' needs "
                "'center_deg': [latitude, longitude]"
            )
        return cls(
            kind=kind,
            center_deg=(float(centre[0]), float(centre[1])),
            radius_km=_positive(raw.get("radius_km"), f"metric {metric_id!r} radius_km"),
        )

    def contains(self, latitude_deg: float, longitude_deg: float) -> bool:
        if self.kind == "global":
            return True
        if self.kind == "bounding_box":
            if not self.south_deg <= latitude_deg <= self.north_deg:
                return False
            west = ((self.west_deg + 180.0) % 360.0) - 180.0
            east = ((self.east_deg + 180.0) % 360.0) - 180.0
            longitude = ((longitude_deg + 180.0) % 360.0) - 180.0
            if west <= east:
                return west <= longitude <= east
            return longitude >= west or longitude <= east
        from .geometry import great_circle_km  # noqa: PLC0415 - cycle at import time

        return great_circle_km(
            latitude_deg, longitude_deg, self.center_deg[0], self.center_deg[1]
        ) <= self.radius_km

    def as_row(self) -> Mapping[str, Any]:
        if self.kind == "global":
            return {"kind": "global"}
        if self.kind == "bounding_box":
            return {
                "kind": "bounding_box",
                "south_deg": self.south_deg, "north_deg": self.north_deg,
                "west_deg": self.west_deg, "east_deg": self.east_deg,
            }
        return {
            "kind": "cap",
            "center_deg": [self.center_deg[0], self.center_deg[1]],
            "radius_km": self.radius_km,
        }


@dataclass(frozen=True)
class RankRow:
    """What one unit of ranked intensity IS, in this row's own field units.

    THE DEFECT THIS FIXES, measured twice on real global forecasts.  Until
    this column existed, the placement policy divided every candidate's
    ``metric_extremum`` by one shared ``scale``, so a sea-level-pressure
    anomaly in pascals (about 5,000 for a deep low) and a reflectivity
    anomaly in decibels (about 11 for a strong convective core) were added
    to the same score.  They differ by about 470x from UNITS ALONE.  On a
    24 h global forecast that held 41 cyclone tracks and 217
    deep-convection tracks, the twenty-seven highest-ranked candidates of
    205 were all cyclones and a budget of four never reached the
    convection row -- twice, in two independent lanes, on two different
    runs of the same cycle.

    ``intensity_reference`` is the anomaly past the detector's own
    threshold that this row calls one unit of significance.  A row
    declares it because a row is the only thing that knows what its field
    measures.  After the division the term is DIMENSIONLESS, so two rows
    on two fields are on one axis for the first time.
    """

    intensity_reference: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], metric_id: str) -> "RankRow":
        _refuse_unknown(raw, ("intensity_reference",), f"metric {metric_id!r} rank row")
        return cls(
            intensity_reference=_positive(
                raw.get("intensity_reference"),
                f"metric {metric_id!r} rank.intensity_reference",
            )
        )


@dataclass(frozen=True)
class MetricRow:
    id: str
    threat_class: str
    field: str
    enabled: bool
    detector: Detector
    confirm_with: tuple[ConfirmRow, ...]
    track: TrackRow
    swath: SwathRow
    start_policy: StartPolicy
    rank: RankRow
    region: Region

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "MetricRow":
        _refuse_unknown(
            raw,
            (
                "id",
                "threat_class",
                "field",
                "enabled",
                "detector",
                "confirm_with",
                "track",
                "swath",
                "start_policy",
                "rank",
                "region",
                "description",
            ),
            "a threat-metrics metric row",
        )
        metric_id = str(raw.get("id", ""))
        if not metric_id:
            raise SwathDocumentError("a metric row has no 'id'")
        for required in ("threat_class", "field", "detector", "track", "swath"):
            if required not in raw:
                raise SwathDocumentError(
                    f"metric {metric_id!r} has no {required!r}; every metric row "
                    "carries all five so that no downstream stage has to ask which "
                    "kind of threat it is holding"
                )
        if "rank" not in raw:
            raise SwathDocumentError(
                f"metric {metric_id!r} has no 'rank' block, so it declares no "
                "'intensity_reference' and there is no way to convert its field's "
                "units into a ranked intensity. Without one, this row's anomaly "
                "would be summed against every other row's on a single axis -- "
                "pascals against decibels -- and the class with the larger numbers "
                "would sweep the whole budget. That is measured, twice: see "
                "hexcore.swath.registry.RankRow"
            )
        return cls(
            id=metric_id,
            threat_class=str(raw["threat_class"]),
            field=str(raw["field"]),
            enabled=bool(raw.get("enabled", True)),
            detector=Detector.from_mapping(raw["detector"], metric_id),
            confirm_with=tuple(
                ConfirmRow.from_mapping(row, metric_id)
                for row in raw.get("confirm_with", ())
            ),
            track=TrackRow.from_mapping(raw["track"], metric_id),
            swath=SwathRow.from_mapping(raw["swath"], metric_id),
            start_policy=StartPolicy.from_mapping(
                raw.get("start_policy", {"kind": "cycle_start"}), metric_id
            ),
            rank=RankRow.from_mapping(raw["rank"], metric_id),
            region=Region.from_mapping(raw.get("region", {"kind": "global"}), metric_id),
        )


@dataclass(frozen=True)
class MetricRegistry:
    schema: str
    source_path: Path | None
    sha256: str
    field_rows: Mapping[str, FieldRow]
    metric_rows: Mapping[str, MetricRow]

    @property
    def armed(self) -> tuple[MetricRow, ...]:
        return tuple(row for row in self.metric_rows.values() if row.enabled)

    def leaf_variables(self, field_id: str, *, whose: str) -> tuple[str, ...]:
        """Every history variable one field row needs, transitively.

        A field row may be an operator over other rows, so this walks the
        ``inputs`` graph down to the leaves.  The walk carries the path it
        took, so a row that reaches itself -- directly, or around a ring
        of four -- is refused with the RING PRINTED rather than recursing
        until the interpreter runs out of stack, which is a crash whose
        traceback names none of the rows that caused it.
        """

        out: set[str] = set()
        seen: set[str] = set()

        def walk(name: str, path: tuple[str, ...]) -> None:
            row = self.field_rows.get(name)
            if row is None:
                raise SwathDocumentError(
                    f"{whose} names field {name!r}, which no field row defines. "
                    f"Known fields: {sorted(self.field_rows)}. A metric pointing at "
                    "a field nobody derives would detect nothing and report no "
                    "threats, which reads as a quiet world"
                    + (f" (reached through {' -> '.join(path)})" if path else "")
                )
            if name in path:
                ring = " -> ".join((*path[path.index(name):], name))
                raise SwathDocumentError(
                    f"field row {name!r} depends on itself through {ring}. A cycle "
                    "in the field graph has no value at any cell: deriving it would "
                    "recurse until the interpreter gave up, and the traceback names "
                    "none of the rows that caused it"
                )
            if name in seen:
                return
            if row.inputs:
                for child in row.inputs:
                    walk(child, (*path, name))
            else:
                out.update(row.source_variables)
            seen.add(name)

        walk(field_id, ())
        return tuple(sorted(out))

    def publication_manifest(self) -> tuple[str, ...]:
        """Every history variable the armed rows need, derived.

        This is the whole point of the operand columns: a new metric row
        extends what the coarse run must publish WITHOUT a second edit
        anywhere, and it does so THROUGH the rows it composes.  A test
        asserts that by adding a compound row to a fixture and reading
        this function.
        """

        needed: set[str] = set()
        for metric in self.armed:
            for field_id in (metric.field, *(row.field for row in metric.confirm_with)):
                needed.update(
                    self.leaf_variables(field_id, whose=f"metric {metric.id!r}")
                )
        return tuple(sorted(needed))

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, source_path: Path | None, sha256: str
    ) -> "MetricRegistry":
        _refuse_unknown(raw, ("schema", "fields", "metrics", "notes"), "threat-metrics")
        schema = str(raw.get("schema", ""))
        if schema != METRICS_SCHEMA:
            raise SwathDocumentError(
                f"threat-metrics declares schema {schema!r}; this build reads "
                f"{METRICS_SCHEMA!r}. A document read under the wrong schema is read "
                "with the wrong defaults, and the placement it produces is a plan "
                "nobody wrote."
                + (
                    " What moved: " + SUPERSEDED_SCHEMAS[schema]
                    if schema in SUPERSEDED_SCHEMAS else ""
                )
            )
        field_rows = {}
        for entry in raw.get("fields", ()):
            row = FieldRow.from_mapping(entry)
            if row.id in field_rows:
                raise SwathDocumentError(f"duplicate field row id {row.id!r}")
            field_rows[row.id] = row
        metric_rows = {}
        for entry in raw.get("metrics", ()):
            row = MetricRow.from_mapping(entry)
            if row.id in metric_rows:
                raise SwathDocumentError(f"duplicate metric row id {row.id!r}")
            metric_rows[row.id] = row
        if not metric_rows:
            raise SwathDocumentError(
                "threat-metrics declares no metric rows, so no cycle could ever "
                "place a swath. An empty document is refused rather than run, "
                "because a cycle that publishes an empty globe looks identical to "
                "a quiet day"
            )
        registry = cls(
            schema=schema,
            source_path=source_path,
            sha256=sha256,
            field_rows=MappingProxyType(field_rows),
            metric_rows=MappingProxyType(metric_rows),
        )
        # EVERY field row, not only the armed ones.  A dangling input or a
        # ring inside a row that today's document does not arm is a trap
        # laid for whoever arms it next, and it costs nothing to refuse it
        # here rather than three cycles later on a card.
        for field_id in field_rows:
            registry.leaf_variables(field_id, whose=f"field row {field_id!r}")
        registry.publication_manifest()
        return registry


# ---------------------------------------------------------------------------
# placement policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RankTerm:
    id: str
    kind: str
    weight: float
    scale: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "RankTerm":
        _refuse_unknown(raw, ("id", "kind", "weight", "scale"), "a rank term")
        return cls(
            id=str(raw["id"]),
            kind=_refuse_unknown_member(raw.get("kind"), RANK_TERM_KINDS, "rank_terms kind"),
            weight=float(raw.get("weight", 1.0)),
            scale=_positive(raw.get("scale", 1.0), f"rank term {raw.get('id')!r} scale"),
        )


@dataclass(frozen=True)
class Hysteresis:
    """The rule that stops a swath trading places with its runner-up.

    THE BREAKAGE THIS PREVENTS: two candidates whose rank scores differ by
    less than the arithmetic that produced them trade the last admitted
    slot every cycle.  Each trade discards a fine domain that already ran
    and mints a fresh mesh, fresh statics, fresh boundaries and a fresh
    cold start on ground the discarded swath already covered, and the
    animation for that slot cuts between two different storms.  On the
    shipped cascade ladder one L3 slot is 56.5 min of GPU plus 9.0 min of
    init, so a single avoidable trade is over an hour of a six-hour cycle.
    The measurement that earns these numbers is in
    ``evidence/swath-following-20260826/``: the same placement sequence run
    with the rule armed and disarmed, with the churn counted both ways.
    """

    promotion_margin: float
    minimum_dwell_cycles: int
    continuation_radius_km: float
    regenerate_centroid_km: float
    regenerate_overlap_below: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Hysteresis":
        known = tuple(field.name for field in fields(cls))
        _refuse_unknown(raw, known, "the hysteresis block")
        margin = float(raw.get("promotion_margin", 0.0))
        if margin < 0.0:
            raise SwathDocumentError(
                f"promotion_margin={margin} is negative, which would demand a "
                "challenger score WORSE than the incumbent to take its slot"
            )
        overlap = float(raw.get("regenerate_overlap_below", 0.0))
        if not 0.0 <= overlap <= 1.0:
            raise SwathDocumentError(
                f"regenerate_overlap_below={overlap} is not a fraction in [0, 1]"
            )
        dwell = int(raw.get("minimum_dwell_cycles", 0))
        if dwell < 0:
            raise SwathDocumentError(f"minimum_dwell_cycles={dwell} is negative")
        return cls(
            promotion_margin=margin,
            minimum_dwell_cycles=dwell,
            continuation_radius_km=_positive(
                raw.get("continuation_radius_km"), "continuation_radius_km"
            ),
            regenerate_centroid_km=_positive(
                raw.get("regenerate_centroid_km"), "regenerate_centroid_km"
            ),
            regenerate_overlap_below=overlap,
        )


@dataclass(frozen=True)
class Budget:
    maximum_swaths: int
    maximum_cells_per_swath: float
    background_km: float
    minimum_separation_km: float
    maximum_per_threat_class: int

    #: THE BREAKAGE ``maximum_per_threat_class`` PREVENTS, measured on a
    #: real 24 h global forecast: the detector formed 258 tracks -- 41
    #: cyclone, 217 deep-convection -- and the twenty-seven highest-ranked
    #: candidates of 205 were all cyclones, so a budget of four placed four
    #: cyclones and none of the 217 convective regions the same forecast
    #: held.  Commensurable units (see :class:`RankRow`) put the two classes
    #: on one axis; they do not make the atmosphere interleave them.  A
    #: machine that can only ever resolve the one thing its strongest class
    #: is doing is a cyclone tracker with a threat table attached, and the
    #: cycles it spends are cycles the other threat did not get.
    #:
    #: It applies to incumbents too, deliberately.  A cap that exempted
    #: protected slots could never take effect on a cycle that already
    #: holds a full class, which is the only cycle it matters on.

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Budget":
        known = tuple(field.name for field in fields(cls))
        _refuse_unknown(raw, known, "the budget block")
        maximum = int(raw.get("maximum_swaths", 0))
        if maximum < 1:
            raise SwathDocumentError(
                f"maximum_swaths={maximum}; a cycle that may place no swath is a "
                "coarse-only cycle and should be declared by disabling the metric "
                "rows, not by a budget of zero that silently drops every candidate"
            )
        per_class = int(raw.get("maximum_per_threat_class", maximum))
        if per_class < 1:
            raise SwathDocumentError(
                f"maximum_per_threat_class={per_class}; a cap of zero admits no "
                "candidate of any class, so the cycle places nothing while every "
                "row reports features. Leave the key out to mean 'no cap' "
                f"(it then takes maximum_swaths, {maximum})"
            )
        return cls(
            maximum_swaths=maximum,
            maximum_cells_per_swath=_positive(
                raw.get("maximum_cells_per_swath"), "maximum_cells_per_swath"
            ),
            background_km=_positive(raw.get("background_km"), "background_km"),
            minimum_separation_km=_positive(
                raw.get("minimum_separation_km"), "minimum_separation_km"
            ),
            maximum_per_threat_class=per_class,
        )


@dataclass(frozen=True)
class PlacementPolicy:
    schema: str
    source_path: Path | None
    sha256: str
    rank_terms: tuple[RankTerm, ...]
    tiebreak: tuple[str, ...]
    hysteresis: Hysteresis
    budget: Budget

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, source_path: Path | None, sha256: str
    ) -> "PlacementPolicy":
        _refuse_unknown(
            raw,
            ("schema", "rank_terms", "tiebreak", "hysteresis", "budget", "notes"),
            "placement-policy",
        )
        schema = str(raw.get("schema", ""))
        if schema != POLICY_SCHEMA:
            raise SwathDocumentError(
                f"placement-policy declares schema {schema!r}; this build reads "
                f"{POLICY_SCHEMA!r}."
                + (
                    " What moved: " + SUPERSEDED_SCHEMAS[schema]
                    if schema in SUPERSEDED_SCHEMAS else ""
                )
            )
        terms = tuple(RankTerm.from_mapping(entry) for entry in raw.get("rank_terms", ()))
        if not terms:
            raise SwathDocumentError(
                "placement-policy declares no rank terms, so every candidate scores "
                "zero and the admitted set is decided by the tiebreak alone -- which "
                "means the machine is not ranking threats, it is sorting identifiers"
            )
        tiebreak = tuple(
            _refuse_unknown_member(key, TIEBREAK_KEYS, "a tiebreak key")
            for key in raw.get("tiebreak", ())
        )
        if not tiebreak or tiebreak[-1] not in TOTAL_ORDER_KEYS:
            raise SwathDocumentError(
                f"tiebreak {list(tiebreak)} does not end in one of "
                f"{list(TOTAL_ORDER_KEYS)}, so it is not a TOTAL order. Two "
                "candidates that compare equal on every key would then be admitted "
                "in whatever order the detector happened to emit them, and the same "
                "history file could produce two different plans on two machines"
            )
        return cls(
            schema=schema,
            source_path=source_path,
            sha256=sha256,
            rank_terms=terms,
            tiebreak=tiebreak,
            hysteresis=Hysteresis.from_mapping(raw.get("hysteresis", {})),
            budget=Budget.from_mapping(raw.get("budget", {})),
        )


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _read_json(path: Path, what: str) -> tuple[Mapping[str, Any], str]:
    try:
        raw_bytes = Path(path).read_bytes()
    except OSError as error:
        raise SwathDocumentError(
            f"cannot read the {what} document at {path}: {error}"
        ) from error
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SwathDocumentError(
            f"the {what} document at {path} is not valid UTF-8 JSON: {error}"
        ) from error
    if not isinstance(parsed, Mapping):
        raise SwathDocumentError(f"the {what} document at {path} is not a JSON object")
    return parsed, digest


def load_metrics(path: str | Path | None = None) -> MetricRegistry:
    """The armed threat rows: the shipped document, or one the user names."""

    target = Path(path).expanduser() if path is not None else DEFAULT_METRICS
    raw, digest = _read_json(target, "threat-metrics")
    return MetricRegistry.from_mapping(raw, source_path=target, sha256=digest)


def load_policy(path: str | Path | None = None) -> PlacementPolicy:
    """The ranking, hysteresis and budget: shipped, or one the user names."""

    target = Path(path).expanduser() if path is not None else DEFAULT_POLICY
    raw, digest = _read_json(target, "placement-policy")
    return PlacementPolicy.from_mapping(raw, source_path=target, sha256=digest)


__all__ = [
    "AGGREGATION_KINDS",
    "COMPARISONS",
    "DEFAULT_METRICS",
    "DEFAULT_POLICY",
    "DERIVATION_ARITY",
    "DERIVATION_KINDS",
    "DETECTOR_KINDS",
    "EXTREMA",
    "METRICS_SCHEMA",
    "POLICY_SCHEMA",
    "RANK_TERM_KINDS",
    "REGION_KINDS",
    "START_POLICY_KINDS",
    "SUPERSEDED_SCHEMAS",
    "TIEBREAK_KEYS",
    "Budget",
    "ConfirmRow",
    "Detector",
    "FieldRow",
    "Hysteresis",
    "MetricRegistry",
    "MetricRow",
    "PlacementPolicy",
    "RankRow",
    "RankTerm",
    "Region",
    "StartPolicy",
    "SwathRow",
    "TrackRow",
    "load_metrics",
    "load_policy",
]
