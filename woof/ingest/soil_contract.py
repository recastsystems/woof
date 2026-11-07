"""Declarative soil-depth and remapping contract for mapped RW-WPS sources.

The mapped-source path must not infer soil meaning from a product name or an
array alias.  This module validates the complete source geometry, target Noah
geometry, numerical remap, and land/ocean missing-data policy used by both
composition validation and the WRF-real initializer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from woof.core.noah import NUM_SOIL_LAYERS as _NOAH_NUM_SOIL_LAYERS


MAPPED_SOIL_TEMPERATURE = "RW_SOIL_TEMPERATURE"
MAPPED_SOIL_MOISTURE = "RW_SOIL_MOISTURE"

#: The four-layer Noah/Noah-MP soil target this contract remaps INTO.  It is
#: deliberately a single hardcoded geometry rather than a per-scheme table:
#: RUC's nine levels are a different discretization with a different
#: value-location convention, and mapping a source profile onto them requires
#: RUC's own level table and remap policy, not a widened bound check here.
#: A RUC ingest lane must add its target explicitly and prove its remap
#: against the WRF-real initialization -- fabricating one by relaxing this
#: comparison would silently produce a plausible, unvalidated soil column.
NOAH_LAYER_BOUNDS_M = (
    (0.0, 0.1),
    (0.1, 0.4),
    (0.4, 1.0),
    (1.0, 2.0),
)

#: The count this contract targets, taken from the scheme that owns it rather
#: than counted into a message as the word "four".  The ingest layer holds no
#: soil-layer constant of its own: it holds one geometry, Noah's, and says
#: whose it is.  A drift between this table and Noah's own
#: ``NUM_SOIL_LAYERS`` -- the reviewable failure mode, since these bounds are
#: the accumulation of ``init_soil_depth_2``'s dzs -- is refused at import.
NOAH_TARGET_SOIL_LAYERS = len(NOAH_LAYER_BOUNDS_M)
if NOAH_TARGET_SOIL_LAYERS != _NOAH_NUM_SOIL_LAYERS:
    raise AssertionError(
        f"NOAH_LAYER_BOUNDS_M has {NOAH_TARGET_SOIL_LAYERS} layers but "
        f"woof.core.noah.NUM_SOIL_LAYERS is {_NOAH_NUM_SOIL_LAYERS}; the "
        "mapped soil target and the scheme's own geometry have drifted")

#: The RUC LSM soil target, added EXPLICITLY as the Noah comparison below
#: demands: RUC's nine LEVEL depths in metres -- value-at-a-node, not mean
#: over a slab.  Its remap policies are a TABLE, :data:`RUC_REMAP_POLICIES`,
#: one row per source geometry the contract language can declare; the
#: RUC-family producers (HRRR, RAP, RRFS) publish at exactly these depths
#: and take the identity row, and every other geometry takes the row that
#: matches it.  See :func:`ruc_soil_remap_policy`.  The authority for these
#: numbers is the scheme's own oracle-validated table,
#: ``woof/ingest/ruc_soil.py:RUC_LEVEL_DEPTHS_M[9]``; the import-time
#: drift check mirrors the Noah one above.
RUC_TARGET_LEVEL_DEPTHS_M: tuple[float, ...] = (
    0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00)

from woof.ingest.ruc_soil import RUC_LEVEL_DEPTHS_M as _RUC_LEVEL_DEPTHS_M

if RUC_TARGET_LEVEL_DEPTHS_M != _RUC_LEVEL_DEPTHS_M[9]:
    raise AssertionError(
        f"RUC_TARGET_LEVEL_DEPTHS_M {RUC_TARGET_LEVEL_DEPTHS_M!r} differs "
        f"from woof.ingest.ruc_soil.RUC_LEVEL_DEPTHS_M[9] "
        f"{_RUC_LEVEL_DEPTHS_M[9]!r}; the mapped RUC target and the "
        "scheme's own level table have drifted")

_LINEAR_REMAP = "linear_point_samples"
_CONSERVATIVE_REMAP = "conservative_layer_means"
_NODE_REMAP = "linear_node_samples"
_CANONICAL_SOIL_FIELDS = (
    "soil_temperature",
    "volumetric_soil_moisture",
)


def _object(
    value: object,
    label: str,
    *,
    allowed: set[str],
    required: set[str],
) -> dict[str, object]:
    # Runtime composition bundles deliberately freeze their validated JSON
    # objects with ``MappingProxyType``.  Accept any Mapping here and make a
    # plain local copy before applying the same closed-key validation.  The
    # previous concrete-dict check meant a contract could pass composition
    # loading and then fail solely because the immutable runtime copy reached
    # Noah preprocessing.
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    result = dict(value)
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise ValueError(f"{label} has unknown key(s): {unknown}")
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"{label} is missing required key(s): {missing}")
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _selector(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be one non-empty direct selector")
    source_format = value.get("format")
    allowed = {
        "grib1": {
            "format", "parameter", "table_version", "center", "level_type",
            "level_value",
        },
        "grib2": {
            "format", "discipline", "category", "parameter", "level_type",
            "level_value", "second_level_type", "second_level_value", "member",
            "center", "subcenter", "master_table_version",
            "local_table_version",
        },
        # `layer_*` addresses ONE slice of a producer's own layer dimension,
        # for the sources that publish an N-layer soil quantity as one
        # variable rather than as N variables.  See
        # `woof.mapped_source._validate_layer_slice`.
        "netcdf": {
            "format", "name", "standard_name", "attributes",
            "layer_dimension", "layer_value", "layer_units",
        },
    }
    if source_format not in allowed:
        raise ValueError(f"{label} has an unsupported selector format")
    unknown = sorted(set(value) - allowed[source_format])
    if unknown:
        raise ValueError(
            f"{label} has keys incompatible with {source_format}: {unknown}"
        )
    required = {
        "grib1": {"parameter"},
        "grib2": {"discipline", "category", "parameter"},
        "netcdf": set(),
    }[source_format]
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{label} is missing selector key(s): {missing}")
    if any(value.get(key) is None for key in required):
        raise ValueError(f"{label} required selector identifiers must not be null")
    if source_format == "netcdf":
        # `name` may be an ordered list of accepted spellings of the SAME
        # variable, so a soil binding survives a producer's rename without
        # the depth-to-selector pairing being rewritten.
        configured = value.get("name")
        spellings = (
            [] if configured is None
            else [configured] if isinstance(configured, str)
            else configured if isinstance(configured, list)
            else None
        )
        if spellings is None or any(
            not isinstance(name, str) or not name for name in spellings
        ):
            raise ValueError(f"{label} NetCDF names must be non-empty strings")
        if len(set(spellings)) != len(spellings):
            raise ValueError(f"{label} repeats an accepted spelling")
        standard = value.get("standard_name")
        if standard is not None and (not isinstance(standard, str) or not standard):
            raise ValueError(f"{label} NetCDF names must be non-empty strings")
        if not spellings and not standard:
            raise ValueError(f"{label} needs name and/or standard_name")
        from woof.mapped_source import (_validate_layer_slice,
                                         _validate_selector_attributes)

        _validate_selector_attributes(value, label)
        _validate_layer_slice(value, label)
        return value
    integer_keys = (
        "parameter", "table_version", "center", "level_type",
        "second_level_type", "discipline", "category", "member",
        "subcenter", "master_table_version", "local_table_version",
    )
    for key in integer_keys:
        item = value.get(key)
        if item is None:
            continue
        maximum = (
            65535
            if source_format == "grib2" and key in {"center", "subcenter"}
            else 255
        )
        if isinstance(item, bool) or not isinstance(item, int) \
                or not 0 <= item <= maximum:
            raise ValueError(
                f"{label}.{key} must be an integer in [0, {maximum}]"
            )
    for key in ("level_value", "second_level_value"):
        if value.get(key) is not None:
            _number(value[key], f"{label}.{key}")
    second_keys = {
        key for key in ("second_level_type", "second_level_value")
        if value.get(key) is not None
    }
    if source_format != "grib2" and second_keys:
        raise ValueError(f"{label} second fixed surfaces require GRIB2")
    if second_keys and second_keys != {"second_level_type", "second_level_value"}:
        raise ValueError(
            f"{label} second_level_type and second_level_value are an atomic pair"
        )
    identifier_keys = (
        ("parameter", "level_type", "table_version", "center")
        if source_format == "grib1"
        else ("discipline", "category", "parameter", "second_level_type")
    )
    undefined = sorted(key for key in identifier_keys if value.get(key) == 255)
    if undefined:
        raise ValueError(
            f"{label} uses missing/undefined identifier code 255 for {undefined}"
        )
    if source_format == "grib2" and value.get("level_type") == 255 \
            and any(value.get(key) is not None for key in (
                "level_value", "second_level_type", "second_level_value",
            )):
        raise ValueError(
            f"{label} uses level_type=255 with fixed-surface metadata"
        )
    if source_format == "grib2" and any(
        isinstance(value.get(key), int) and 192 <= value[key] <= 254
        for key in (
            "discipline", "category", "parameter", "level_type",
            "second_level_type",
        )
    ):
        authority = {
            "center", "subcenter", "master_table_version", "local_table_version"
        }
        missing_authority = sorted(authority - set(value))
        if missing_authority:
            raise ValueError(
                f"{label} local-use selector is missing Section 1 authority "
                f"{missing_authority}"
            )
        if value["local_table_version"] == 255:
            raise ValueError(
                f"{label} local-use selector cannot use local_table_version=255"
            )
    return value


def _selector_semantics(value: Mapping[str, object]) -> dict[str, object]:
    """Match the typed Rust selector's absent/explicit-null semantics."""

    return {key: item for key, item in value.items() if item is not None}


def _layer_bounds(
    value: object,
    label: str,
    *,
    bind_selectors: bool = False,
) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty ordered layer list")
    result: list[tuple[float, float]] = []
    selector_formats: set[str] = set()
    previous_top = -math.inf
    previous_bottom: float | None = None
    for index, raw in enumerate(value):
        layer_keys = {"top", "bottom"}
        if bind_selectors:
            layer_keys.add("selectors")
        layer = _object(
            raw,
            f"{label}[{index}]",
            allowed=layer_keys,
            required=layer_keys,
        )
        if bind_selectors:
            bindings = _object(
                layer["selectors"],
                f"{label}[{index}].selectors",
                allowed=set(_CANONICAL_SOIL_FIELDS),
                required=set(_CANONICAL_SOIL_FIELDS),
            )
            for field_name in _CANONICAL_SOIL_FIELDS:
                selector = _selector(
                    bindings[field_name],
                    f"{label}[{index}].selectors.{field_name}",
                )
                selector_formats.add(str(selector["format"]))
        top = _number(layer["top"], f"{label}[{index}].top")
        bottom = _number(layer["bottom"], f"{label}[{index}].bottom")
        if top < 0.0 or bottom <= top:
            raise ValueError(
                f"{label}[{index}] must have non-negative, positive-thickness "
                "depths ordered top-to-bottom"
            )
        if top < previous_top:
            raise ValueError(f"{label} is not ordered shallow-to-deep")
        if previous_bottom is not None:
            if top < previous_bottom:
                raise ValueError(
                    f"{label} layers {index - 1} and {index} overlap"
                )
            if top > previous_bottom:
                raise ValueError(
                    f"{label} has a gap between layers {index - 1} and {index}"
                )
        result.append((top, bottom))
        previous_top = top
        previous_bottom = bottom
    if bind_selectors and len(selector_formats) != 1:
        raise ValueError(
            f"{label} selectors must all use one common source format"
        )
    return tuple(result)


def _node_depths(
    value: object,
    label: str,
    *,
    bind_selectors: bool = False,
    selector_fields: tuple[str, ...] = _CANONICAL_SOIL_FIELDS,
) -> tuple[float, ...]:
    """Validate an ordered list of soil depth NODES (point samples).

    A node source (RUC-family models: HRRR, RAP publish TSOIL/SOILW at
    depths, not layer means) is a different geometry from a layer source
    and is declared as one: each row carries the sample ``depth`` in
    metres and, on the source side, the selector bound to that exact
    depth.  Requiring the 0 m and 3 m endpoints is what makes the linear
    remap anchor-free -- the surface and deep values are the source's own
    samples, not synthetic TSK/TMN brackets.
    """

    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{label} must be an ordered list of at least two nodes")
    depths: list[float] = []
    selector_formats: set[str] = set()
    for index, raw in enumerate(value):
        node_keys = {"depth"}
        if bind_selectors:
            node_keys.add("selectors")
        node = _object(
            raw,
            f"{label}[{index}]",
            allowed=node_keys,
            required=node_keys,
        )
        if bind_selectors:
            bindings = _object(
                node["selectors"],
                f"{label}[{index}].selectors",
                allowed=set(selector_fields),
                required=set(selector_fields),
            )
            for field_name in selector_fields:
                selector = _selector(
                    bindings[field_name],
                    f"{label}[{index}].selectors.{field_name}",
                )
                selector_formats.add(str(selector["format"]))
        depth = _number(node["depth"], f"{label}[{index}].depth")
        if depth < 0.0:
            raise ValueError(f"{label}[{index}].depth must be non-negative")
        if depths and depth <= depths[-1]:
            raise ValueError(f"{label} is not strictly ordered shallow-to-deep")
        depths.append(depth)
    if bind_selectors and len(selector_formats) != 1:
        raise ValueError(
            f"{label} selectors must all use one common source format"
        )
    if len(depths) < 2:
        raise ValueError(f"{label} needs at least two distinct nodes for interpolation")
    return tuple(depths)


def _validate_mapping_node_fields(
    contract: Mapping[str, object],
    node_depths: tuple[float, ...],
    mapping: Mapping[str, object],
) -> None:
    """The node twin of :func:`_validate_mapping_fields`."""

    fields = mapping.get("fields")
    target = mapping.get("target")
    if not isinstance(fields, dict) or not isinstance(target, dict):
        raise ValueError("soil contract requires a validated mapped-source mapping")
    expected = (
        (str(contract["temperature_field"]), "K"),
        (str(contract["moisture_field"]), "m3 m-3"),
    )
    for name, units in expected:
        field = fields.get(name)
        if not isinstance(field, dict):
            raise ValueError(f"soil contract field {name!r} is not mapped")
        if tuple(field.get("target_axes", ())) != ("soil", "y", "x") \
                or field.get("location") != "soil" \
                or field.get("staggering", "none") != "none" \
                or not isinstance(field.get("units"), dict) \
                or field["units"].get("target") != units:
            raise ValueError(
                f"soil contract field {name!r} must be unstaggered "
                f"soil/y/x in {units}"
            )
        missing = field.get("missing")
        if not isinstance(missing, dict) \
                or missing.get("kind") not in {
                    "reject", "preserve_mask", "landmask_water",
                }:
            raise ValueError(
                f"soil contract field {name!r} must reject missing values or "
                "preserve an ocean mask for declared repair"
            )
        selectors = field.get("selectors", ())
        if not isinstance(selectors, list) \
                or len(selectors) != len(node_depths):
            raise ValueError(
                f"soil contract requires exactly {len(node_depths)} "
                f"ordered direct selectors for {name!r}"
            )
        declared_selectors = tuple(
            node["selectors"][name] for node in contract["source_nodes"]
        )
        for index, (selector, declared) in enumerate(
            zip(selectors, declared_selectors)
        ):
            if _selector_semantics(selector) != _selector_semantics(declared):
                raise ValueError(
                    f"{name} selector {index} differs from the selector "
                    "bound to its declared soil depth"
                )
        # A GRIB2 node is a degenerate type-106 layer whose two fixed
        # surfaces carry the SAME depth (HRRR's "0.3-0.3 m below ground").
        # Binding both surfaces to the contract's node depth is what makes
        # a reordered producer a refusal instead of a silent level swap.
        if mapping.get("format") == "grib2":
            for index, (selector, depth) in enumerate(zip(selectors, node_depths)):
                if not isinstance(selector, dict) \
                        or selector.get("level_type") != 106 \
                        or selector.get("second_level_type") != 106 \
                        or "level_value" not in selector \
                        or "second_level_value" not in selector:
                    raise ValueError(
                        f"{name} selector {index} must declare a GRIB2 "
                        "depth-below-land node (type 106, both surfaces)"
                    )
                observed = (
                    _number(selector["level_value"], f"{name} selector {index} depth"),
                    _number(
                        selector["second_level_value"],
                        f"{name} selector {index} second depth",
                    ),
                )
                if observed != (depth, depth):
                    raise ValueError(
                        f"{name} selector {index} depth {observed!r} differs "
                        f"from soil contract node {depth!r}"
                    )

    soil_count = target.get("soil_layer_count")
    if isinstance(soil_count, bool) or not isinstance(soil_count, int) \
            or soil_count != len(node_depths):
        raise ValueError(
            "mapping target.soil_layer_count differs from the declarative "
            "source soil-node count"
        )


def _validate_mapping_mixed_node_fields(
    contract: Mapping[str, object],
    node_depths: tuple[float, ...],
    provenance_layers: tuple[tuple[float, float], ...],
    mapping: Mapping[str, object],
) -> None:
    """Node-ladder temperature beside layer-mass moisture, both proven.

    The DWD/TERRA family publishes soil temperature as point samples on
    the ladder's depths and soil water as COLUMN-INTEGRATED mass over
    layers whose exact midpoints are those same interior depths.  The
    contract stays a single node geometry; what this validator adds is
    the moisture side's provenance proof: the mapping must derive the
    canonical volumetric moisture from a declared layer-mass field
    through exactly the two closed-catalog operations
    (``volumetric_soil_moisture_from_layer_mass`` over the declared
    bounds, then ``soil_surface_node_from_shallowest``), with the
    layer-mass selectors bound layer-for-layer.  Anything else claiming
    the name is refused -- a moisture column must never be fabricated
    from an unproven chain.
    """

    fields = mapping.get("fields")
    target = mapping.get("target")
    if not isinstance(fields, dict) or not isinstance(target, dict):
        raise ValueError("soil contract requires a validated mapped-source mapping")
    temperature_name = str(contract["temperature_field"])
    moisture_name = str(contract["moisture_field"])
    provenance = contract["moisture_provenance"]
    layer_mass_name = str(provenance["layer_mass_field"])
    volumetric_name = str(provenance["volumetric_field"])

    # -- temperature: the node half, unchanged from the single-geometry rule.
    field = fields.get(temperature_name)
    if not isinstance(field, dict):
        raise ValueError(f"soil contract field {temperature_name!r} is not mapped")
    if tuple(field.get("target_axes", ())) != ("soil", "y", "x") \
            or field.get("location") != "soil" \
            or field.get("staggering", "none") != "none" \
            or not isinstance(field.get("units"), dict) \
            or field["units"].get("target") != "K":
        raise ValueError(
            f"soil contract field {temperature_name!r} must be unstaggered "
            "soil/y/x in K"
        )
    missing = field.get("missing")
    if not isinstance(missing, dict) or missing.get("kind") not in {
        "reject", "preserve_mask", "landmask_water",
    }:
        raise ValueError(
            f"soil contract field {temperature_name!r} must reject missing "
            "values or carry a declared ocean-missing mask for repair"
        )
    selectors = field.get("selectors", ())
    if not isinstance(selectors, list) or len(selectors) != len(node_depths):
        raise ValueError(
            f"soil contract requires exactly {len(node_depths)} ordered "
            f"direct selectors for {temperature_name!r}"
        )
    declared_selectors = tuple(
        node["selectors"][temperature_name]
        for node in contract["source_nodes"]
    )
    for index, (selector, declared) in enumerate(
        zip(selectors, declared_selectors)
    ):
        if _selector_semantics(selector) != _selector_semantics(declared):
            raise ValueError(
                f"{temperature_name} selector {index} differs from the "
                "selector bound to its declared soil depth"
            )
    if mapping.get("format") == "grib2":
        for index, (selector, depth) in enumerate(zip(selectors, node_depths)):
            # A GRIB2 node is EITHER a degenerate type-106 layer whose two
            # surfaces carry the same depth (the HRRR/RUC encoding) or a
            # single type-106 surface with the second surface genuinely
            # missing (the DWD/TERRA encoding, octet 255); both pin the
            # sample to the contract's depth.
            has_second = "second_level_type" in selector \
                if isinstance(selector, dict) else False
            if not isinstance(selector, dict) \
                    or selector.get("level_type") != 106 \
                    or "level_value" not in selector \
                    or (has_second and (
                        selector.get("second_level_type") != 106
                        or "second_level_value" not in selector)):
                raise ValueError(
                    f"{temperature_name} selector {index} must declare a "
                    "GRIB2 depth-below-land node (type 106, at the sample "
                    "depth, with the second surface either matching or "
                    "absent)"
                )
            observed = _number(
                selector["level_value"],
                f"{temperature_name} selector {index} depth",
            )
            second_observed = (
                _number(
                    selector["second_level_value"],
                    f"{temperature_name} selector {index} second depth",
                )
                if has_second else observed
            )
            if (observed, second_observed) != (depth, depth):
                raise ValueError(
                    f"{temperature_name} selector {index} depth "
                    f"{(observed, second_observed)!r} differs from soil "
                    f"contract node {depth!r}"
                )

    # -- moisture: the derived half.  Chain proven link by link.
    derivations = {
        str(item.get("name")): item for item in mapping.get("derivations", [])
        if isinstance(item, dict)
    }
    moisture = fields.get(moisture_name)
    if not isinstance(moisture, dict):
        raise ValueError(f"soil contract field {moisture_name!r} is not mapped")
    if tuple(moisture.get("target_axes", ())) != ("soil", "y", "x") \
            or moisture.get("location") != "soil" \
            or moisture.get("staggering", "none") != "none" \
            or not isinstance(moisture.get("units"), dict) \
            or moisture["units"].get("target") != "m3 m-3":
        raise ValueError(
            f"soil contract field {moisture_name!r} must be unstaggered "
            "soil/y/x in m3 m-3"
        )
    surface_step = derivations.get(str(moisture.get("derivation")))
    if not isinstance(surface_step, dict) \
            or surface_step.get("operation") != "soil_surface_node_from_shallowest" \
            or str(surface_step.get("source")) != volumetric_name:
        raise ValueError(
            f"{moisture_name} must derive by soil_surface_node_from_"
            f"shallowest over {volumetric_name!r}, exactly as the "
            "moisture provenance declares"
        )
    volumetric = fields.get(volumetric_name)
    if not isinstance(volumetric, dict):
        raise ValueError(
            f"moisture provenance field {volumetric_name!r} is not mapped"
        )
    mass_step = derivations.get(str(volumetric.get("derivation")))
    if not isinstance(mass_step, dict) \
            or mass_step.get("operation") != "volumetric_soil_moisture_from_layer_mass" \
            or str(mass_step.get("layer_mass")) != layer_mass_name:
        raise ValueError(
            f"{volumetric_name} must derive by volumetric_soil_moisture_"
            f"from_layer_mass over {layer_mass_name!r}, exactly as the "
            "moisture provenance declares"
        )
    declared_bounds = tuple(
        (float(pair[0]), float(pair[1]))
        for pair in mass_step.get("layer_bounds_m", ())
    )
    if declared_bounds != provenance_layers:
        raise ValueError(
            "the layer-mass derivation's layer_bounds_m differ from the "
            "moisture provenance source_layers; the conversion and the "
            "geometry must be the same declaration"
        )
    layer_mass = fields.get(layer_mass_name)
    if not isinstance(layer_mass, dict):
        raise ValueError(
            f"moisture provenance field {layer_mass_name!r} is not mapped"
        )
    if tuple(layer_mass.get("target_axes", ())) != ("soil", "y", "x") \
            or layer_mass.get("location") != "soil" \
            or layer_mass.get("staggering", "none") != "none" \
            or not isinstance(layer_mass.get("units"), dict) \
            or layer_mass["units"].get("target") != "kg m-2":
        raise ValueError(
            f"moisture provenance field {layer_mass_name!r} must be "
            "unstaggered soil/y/x in kg m-2"
        )
    mass_missing = layer_mass.get("missing")
    if not isinstance(mass_missing, dict) or mass_missing.get("kind") not in {
        "reject", "preserve_mask", "landmask_water",
    }:
        raise ValueError(
            f"moisture provenance field {layer_mass_name!r} must reject "
            "missing values or carry a declared ocean-missing mask"
        )
    mass_selectors = layer_mass.get("selectors", ())
    if not isinstance(mass_selectors, list) \
            or len(mass_selectors) != len(provenance_layers):
        raise ValueError(
            f"moisture provenance requires exactly {len(provenance_layers)} "
            f"ordered direct selectors for {layer_mass_name!r}"
        )
    if mapping.get("format") == "grib2":
        for index, (selector, (top, bottom)) in enumerate(
            zip(mass_selectors, provenance_layers)
        ):
            if not isinstance(selector, dict) \
                    or selector.get("level_type") != 106 \
                    or selector.get("second_level_type") != 106 \
                    or "level_value" not in selector \
                    or "second_level_value" not in selector:
                raise ValueError(
                    f"{layer_mass_name} selector {index} must declare a "
                    "GRIB2 depth-below-land layer (type 106, both surfaces)"
                )
            observed = (
                _number(
                    selector["level_value"],
                    f"{layer_mass_name} selector {index} top",
                ),
                _number(
                    selector["second_level_value"],
                    f"{layer_mass_name} selector {index} bottom",
                ),
            )
            if observed != (top, bottom):
                raise ValueError(
                    f"{layer_mass_name} selector {index} layer {observed!r} "
                    f"differs from declared bounds {(top, bottom)!r}"
                )

    soil_count = target.get("soil_layer_count")
    if isinstance(soil_count, bool) or not isinstance(soil_count, int) \
            or soil_count != len(node_depths):
        raise ValueError(
            "mapping target.soil_layer_count differs from the declarative "
            "source soil-node count"
        )


def _validate_mapping_fields(
    contract: Mapping[str, object],
    source_bounds: tuple[tuple[float, float], ...],
    mapping: Mapping[str, object],
) -> None:
    from woof.mapped_source import layer_slice_depth_metres

    fields = mapping.get("fields")
    target = mapping.get("target")
    if not isinstance(fields, dict) or not isinstance(target, dict):
        raise ValueError("soil contract requires a validated mapped-source mapping")
    expected = (
        (str(contract["temperature_field"]), "K"),
        (str(contract["moisture_field"]), "m3 m-3"),
    )
    for name, units in expected:
        field = fields.get(name)
        if not isinstance(field, dict):
            raise ValueError(f"soil contract field {name!r} is not mapped")
        if tuple(field.get("target_axes", ())) != ("soil", "y", "x") \
                or field.get("location") != "soil" \
                or field.get("staggering", "none") != "none" \
                or not isinstance(field.get("units"), dict) \
                or field["units"].get("target") != units:
            raise ValueError(
                f"soil contract field {name!r} must be unstaggered "
                f"soil/y/x in {units}"
            )
        missing = field.get("missing")
        if not isinstance(missing, dict) \
                or missing.get("kind") not in {
                    "reject", "preserve_mask", "landmask_water",
                }:
            raise ValueError(
                f"soil contract field {name!r} must reject missing values or "
                "preserve an ocean mask for declared repair"
            )
        selectors = field.get("selectors", ())
        if not isinstance(selectors, list) \
                or len(selectors) != len(source_bounds):
            raise ValueError(
                f"soil contract requires exactly {len(source_bounds)} "
                f"ordered direct selectors for {name!r}"
            )
        stack_axis = field.get("selector_stack_axis")
        expected_stack_axis = "soil" if mapping.get("format") == "netcdf" else None
        if stack_axis != expected_stack_axis:
            raise ValueError(
                f"soil contract field {name!r} must use selector_stack_axis "
                f"{expected_stack_axis!r} for {mapping.get('format')!r}"
            )
        declared_selectors = tuple(
            layer["selectors"][name] for layer in contract["source_layers"]
        )
        for index, (selector, declared) in enumerate(
            zip(selectors, declared_selectors)
        ):
            if _selector_semantics(selector) != _selector_semantics(declared):
                raise ValueError(
                    f"{name} selector {index} differs from the selector "
                    "bound to its declared soil depth"
                )

        # WMO GRIB2 type 106 is a layer below land in metres.  When the
        # mapping exposes those bounds, bind every positional selector to the
        # declarative source layer instead of trusting list order alone.
        #
        # A composition may instead DECLARE that its producer addresses
        # layers by ordinal on a producer-owned fixed-surface type
        # (``selector_depth_binding``, e.g. code table 4.5 type 151 with
        # scaled layer indices, which is how ECMWF products encode soil).
        # The declared index pair then binds each positional selector to
        # its declared depth, so a producer that reorders or renumbers its
        # layers is still a refusal, never a silent level swap.
        binding = contract.get("selector_depth_binding")
        if mapping.get("format") == "grib2" and binding is not None:
            expected_type = binding["level_type"]
            first_index = binding["first_index"]
            for index, selector in enumerate(selectors):
                expected_pair = (first_index + index, first_index + index + 1)
                if not isinstance(selector, dict) \
                        or selector.get("level_type") != expected_type \
                        or selector.get("second_level_type") != expected_type \
                        or "level_value" not in selector \
                        or "second_level_value" not in selector:
                    raise ValueError(
                        f"{name} selector {index} must declare the bound "
                        f"index-addressed GRIB2 layer (type {expected_type}, "
                        "both surfaces)"
                    )
                observed = (
                    _number(selector["level_value"], f"{name} selector {index} index"),
                    _number(
                        selector["second_level_value"],
                        f"{name} selector {index} second index",
                    ),
                )
                if observed != tuple(float(value) for value in expected_pair):
                    raise ValueError(
                        f"{name} selector {index} index pair {observed!r} "
                        f"differs from the declared ordinal layer "
                        f"{expected_pair!r}"
                    )
        elif mapping.get("format") == "grib2":
            for index, (selector, bounds) in enumerate(zip(selectors, source_bounds)):
                if not isinstance(selector, dict) \
                        or selector.get("level_type") != 106 \
                        or selector.get("second_level_type") != 106 \
                        or "level_value" not in selector \
                        or "second_level_value" not in selector:
                    raise ValueError(
                        f"{name} selector {index} must declare a bounded GRIB2 "
                        "depth-below-land layer (type 106)"
                    )
                observed = (
                    _number(selector["level_value"], f"{name} selector {index} top"),
                    _number(
                        selector["second_level_value"],
                        f"{name} selector {index} bottom",
                    ),
                )
                if observed != bounds:
                    raise ValueError(
                        f"{name} selector {index} depth {observed!r} differs "
                        f"from soil contract layer {bounds!r}"
                    )

        # The NetCDF twin of the GRIB2 depth binding above.  A producer that
        # publishes its soil column as ONE variable with its own layer
        # dimension is addressed by VALUE on that dimension, and the value
        # has to be the depth the contract says the selector stands for --
        # otherwise the ordered list alone decides which layer is which, and
        # a producer that reorders its layers swaps two soil levels with no
        # error anywhere.  Either every selector carries the slice or none
        # does: a half-bound stack is the ambiguity this refuses.
        if mapping.get("format") == "netcdf":
            sliced = [
                index for index, selector in enumerate(selectors)
                if isinstance(selector, dict) and "layer_value" in selector
            ]
            if sliced and len(sliced) != len(selectors):
                raise ValueError(
                    f"{name} binds {len(sliced)} of {len(selectors)} soil "
                    "selectors to a layer value; bind all of them or none"
                )
            for index, (selector, bounds) in enumerate(
                zip(selectors, source_bounds)
            ):
                if not sliced:
                    break
                depth = layer_slice_depth_metres(selector)
                if depth is None or not math.isclose(
                    depth, bounds[0], rel_tol=0.0, abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"{name} selector {index} addresses layer depth "
                        f"{depth!r} m, which is not the soil contract layer "
                        f"top {bounds[0]!r} m"
                    )

    soil_count = target.get("soil_layer_count")
    if isinstance(soil_count, bool) or not isinstance(soil_count, int) \
            or soil_count != len(source_bounds):
        raise ValueError(
            "mapping target.soil_layer_count differs from the declarative "
            "source soil-layer count"
        )


def validate_soil_layer_contract(
    value: object,
    *,
    mapping: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate and return one fail-closed mapped soil contract.

    Only metres are currently accepted because neither classic Vtables nor
    the native decoder ABI supply a general soil-depth unit conversion.  The
    target is the exact Noah state consumed by woof/WRF-real, whose layer
    count is :data:`NOAH_TARGET_SOIL_LAYERS`.  This function takes no
    ``RunConfig`` and does not need one: it does not decide how many layers a
    run has -- :func:`woof.config.soil_layer_count` does -- it only refuses
    a target that is not the one geometry this contract has a proven remap
    for.  A nine-layer run reaching a Noah-shaped ingest is caught by shape
    at ``woof/core/physics.py:_as_soil``, which cannot broadcast four layers
    onto nine.
    """

    contract = _object(
        value,
        "composition.soil_layers",
        allowed={
            "temperature_field", "moisture_field", "depth_units",
            "source_layers", "source_nodes", "target_layers", "remap",
            "missing", "moisture_provenance", "selector_depth_binding",
            "deep_donor", "deep_donor_layers",
        },
        required={
            "temperature_field", "moisture_field", "depth_units",
            "target_layers", "remap", "missing",
        },
    )
    if contract["temperature_field"] != "soil_temperature" \
            or contract["moisture_field"] != "volumetric_soil_moisture":
        raise ValueError("composition must bind the canonical soil fields")
    if contract["depth_units"] != "m":
        raise ValueError(
            "composition soil depth_units must be 'm'; implicit depth-unit "
            "conversion is forbidden"
        )

    has_layers = contract.get("source_layers") is not None
    has_nodes = contract.get("source_nodes") is not None
    if has_layers == has_nodes:
        raise ValueError(
            "composition.soil_layers must declare exactly one source "
            "geometry: source_layers (layer means/midpoints) or "
            "source_nodes (depth point samples)"
        )
    moisture_provenance = contract.get("moisture_provenance")
    provenance_layers: tuple[tuple[float, float], ...] = ()
    if moisture_provenance is not None:
        if not has_nodes:
            raise ValueError(
                "composition.soil_layers.moisture_provenance describes "
                "layer-mass moisture joining a NODE ladder; a layer-geometry "
                "contract declares its moisture in source_layers directly"
            )
        provenance = _object(
            moisture_provenance,
            "composition.soil_layers.moisture_provenance",
            allowed={
                "kind", "layer_mass_field", "volumetric_field",
                "source_layers", "surface_policy",
            },
            required={
                "kind", "layer_mass_field", "volumetric_field",
                "source_layers", "surface_policy",
            },
        )
        if provenance["kind"] != "derived_layer_mass":
            raise ValueError(
                "moisture_provenance.kind must be 'derived_layer_mass'"
            )
        if provenance["surface_policy"] != "replicate_shallowest":
            raise ValueError(
                "moisture_provenance.surface_policy must be "
                "'replicate_shallowest' (WRF's own moisture endpoint "
                "convention, module_soil_pre.F)"
            )
        for key in ("layer_mass_field", "volumetric_field"):
            if not isinstance(provenance[key], str) or not provenance[key]:
                raise ValueError(
                    f"moisture_provenance.{key} must be a non-empty field name"
                )
        provenance_layers = _layer_bounds(
            provenance["source_layers"],
            "moisture provenance source_layers",
        )
        if provenance_layers[0][0] != 0.0:
            raise ValueError(
                "moisture provenance source_layers must begin at the "
                "0 m surface"
            )
    binding = contract.get("selector_depth_binding")
    if binding is not None:
        if not has_layers:
            raise ValueError(
                "selector_depth_binding describes index-addressed "
                "source_layers; a node source carries its depths in its "
                "own selectors"
            )
        binding = _object(
            binding,
            "composition.soil_layers.selector_depth_binding",
            allowed={"kind", "level_type", "first_index"},
            required={"kind", "level_type", "first_index"},
        )
        if binding["kind"] != "indexed_fixed_surfaces":
            raise ValueError(
                "selector_depth_binding.kind must be "
                "'indexed_fixed_surfaces': the only declared alternative "
                "to metre-valued type-106 layers is a producer that "
                "numbers its layers on its own vertical coordinate"
            )
        level_type = binding["level_type"]
        if isinstance(level_type, bool) or not isinstance(level_type, int) \
                or not 0 <= level_type <= 254:
            raise ValueError(
                "selector_depth_binding.level_type must be a GRIB2 fixed "
                "surface code in 0..254"
            )
        if level_type == 106:
            raise ValueError(
                "selector_depth_binding.level_type 106 is refused: WMO "
                "type 106 fixed surfaces carry metre depths, and declaring "
                "them ordinals would launder a depth mismatch the default "
                "metre-valued cross-check exists to refuse"
            )
        first_index = binding["first_index"]
        if isinstance(first_index, bool) or not isinstance(first_index, int) \
                or first_index < 0:
            raise ValueError(
                "selector_depth_binding.first_index must be a non-negative "
                "integer"
            )
        contract["selector_depth_binding"] = binding
    source: tuple[tuple[float, float], ...] = ()
    node_depths: tuple[float, ...] = ()
    if has_nodes:
        node_depths = _node_depths(
            contract["source_nodes"],
            "soil source_nodes",
            bind_selectors=True,
            selector_fields=(
                (str(contract["temperature_field"]),)
                if moisture_provenance is not None
                else _CANONICAL_SOIL_FIELDS
            ),
        )
        if moisture_provenance is not None:
            midpoints = tuple(
                (top + bottom) / 2.0 for top, bottom in provenance_layers
            )
            if len(node_depths) != len(midpoints) + 1 \
                    or node_depths[1:] != midpoints:
                raise ValueError(
                    "moisture provenance layers do not join the node "
                    "ladder: every interior node depth must be the exact "
                    "midpoint of one declared layer, with the 0 m surface "
                    "node supplied by the replicate_shallowest policy; "
                    f"nodes {node_depths!r} vs layer midpoints {midpoints!r}"
                )
    else:
        source = _layer_bounds(
            contract["source_layers"],
            "soil source_layers",
            bind_selectors=True,
        )
        if source[0][0] != 0.0:
            raise ValueError("soil source_layers must begin at the 0 m surface")
    if contract.get("deep_donor") is not None:
        # The layers below the source's own come from a declared donor
        # analysis (woof/ingest/deep_soil_donor.py).  The source's own
        # layers are still checked against its mapping below; the donor's
        # are checked here, against those.
        if not has_layers:
            raise ValueError(
                "a deep_donor extends a layer source below its published "
                "layers; a node source declares its own ladder")
        from woof.ingest.deep_soil_donor import validate_deep_donor
        contract["deep_donor"] = validate_deep_donor(contract["deep_donor"], source)
    donor_layers = contract.get("deep_donor_layers")
    if donor_layers is not None:
        # Only the stacked contract deep_soil_donor builds carries this.
        donor_layers = _object(
            donor_layers, "composition.soil_layers.deep_donor_layers",
            allowed={"first_index", "source_id", "soil_model"},
            required={"first_index", "source_id", "soil_model"},
        )
        first = donor_layers["first_index"]
        if contract.get("deep_donor") is not None or not has_layers                 or isinstance(first, bool) or not isinstance(first, int)                 or not 1 <= first < len(source):
            raise ValueError(
                "deep_donor_layers names the donor's layers inside a stacked "
                "layer contract; it needs at least one source layer above them")
    target = _layer_bounds(contract["target_layers"], "soil target_layers")
    if target != NOAH_LAYER_BOUNDS_M:
        raise ValueError(
            f"soil target_layers differ from the selected "
            f"{NOAH_TARGET_SOIL_LAYERS}-layer Noah contract "
            f"{NOAH_LAYER_BOUNDS_M!r}; this contract has exactly one target, "
            "and RUC's levels are a different discretization with a "
            "different value-location convention rather than a longer list "
            "of the same thing.  Admitting RUC here means adding its target "
            "and proving its remap against the WRF-real initialization; "
            "widening this comparison would fabricate a soil column"
        )

    if has_nodes:
        target_depths = tuple((top + bottom) / 2.0 for top, bottom in target)
        if node_depths[0] > target_depths[0] or node_depths[-1] < target_depths[-1]:
            raise ValueError(
                f"soil source nodes {node_depths[0]}..{node_depths[-1]} m do not cover "
                f"Noah target midpoints {target_depths[0]}..{target_depths[-1]} m; "
                "the node remap has no extrapolation policy")

    remap = _object(
        contract["remap"],
        "composition.soil_layers.remap",
        allowed={
            "kind", "source_value_location", "target_value_location",
            "top_anchor", "bottom_anchor", "coverage",
        },
        required={"kind", "source_value_location", "target_value_location"},
    )
    kind = remap["kind"]
    if kind == _NODE_REMAP:
        if not has_nodes:
            raise ValueError(
                f"soil remap kind {_NODE_REMAP!r} requires source_nodes; "
                "a layer source declares its own remap kind"
            )
        if set(remap) != {
            "kind", "source_value_location", "target_value_location",
        }:
            raise ValueError(
                "linear node remap takes no anchors: the source must "
                "cover the actual target samples"
            )
        if remap["source_value_location"] != "level_node" \
                or remap["target_value_location"] != "layer_midpoint":
            raise ValueError(
                "linear node remap requires level-node source values and "
                "layer-midpoint target values (WRF init_soil_3_real's "
                "FLAG_SOIL_LEVELS geometry)"
            )
    elif has_nodes:
        raise ValueError(
            f"source_nodes require soil remap kind {_NODE_REMAP!r}; "
            f"{kind!r} describes a layer source"
        )
    elif kind == _LINEAR_REMAP:
        if set(remap) != {
            "kind", "source_value_location", "target_value_location",
            "top_anchor", "bottom_anchor",
        }:
            raise ValueError(
                "linear soil remap requires exactly top_anchor and bottom_anchor"
            )
        # Any sample location the one sampler has a rule for is admitted;
        # the refusal is for a location nothing can sample, and it names
        # the rows that exist.
        linear_sample_rule(remap["source_value_location"])
        if remap["target_value_location"] != "layer_midpoint":
            raise ValueError(
                "linear soil remap requires layer-midpoint target values"
            )
        anchors = []
        expected_anchors = (
            ("top_anchor", 0.0, "skin_temperature", "repeat_shallowest"),
            ("bottom_anchor", 3.0, "deep_soil_temperature", "repeat_deepest"),
        )
        for name, expected_depth, expected_temperature, expected_moisture \
                in expected_anchors:
            anchor = _object(
                remap[name],
                f"composition.soil_layers.remap.{name}",
                allowed={"depth", "temperature", "moisture"},
                required={"depth", "temperature", "moisture"},
            )
            depth = _number(anchor["depth"], f"soil remap {name}.depth")
            if (depth, anchor["temperature"], anchor["moisture"]) != (
                expected_depth, expected_temperature, expected_moisture,
            ):
                raise ValueError(
                    f"soil remap {name} does not match the WRF-real/Noah "
                    "surface/deep boundary contract"
                )
            anchors.append(depth)
        # The depths that are actually interpolated, from the one function
        # the remap itself calls.  Validating the layer bottoms here while
        # the remap sampled somewhere else is what let a colliding contract
        # through.
        plan = linear_sample_plan(contract)
        source_points = plan.depths
        labels = plan.labels
        collisions = [
            index
            for index, (earlier, later) in enumerate(
                zip(source_points, source_points[1:]))
            if later <= earlier
        ]
        if collisions:
            index = collisions[0]
            raise ValueError(
                f"linear soil remap sample depths {source_points!r} are not "
                f"strictly ordered: {labels[index]} samples "
                f"{source_points[index]} m and {labels[index + 1]} samples "
                f"{source_points[index + 1]} m under "
                f"source_value_location={remap['source_value_location']!r}. "
                "Two samples at one depth give the interpolation a zero "
                "denominator, so the remap would divide by zero or shift the "
                "whole column; separate the layers, or declare the location "
                "their values are really stated at"
            )
        target_points = tuple((top + bottom) / 2.0 for top, bottom in target)
        if target_points[0] < source_points[0] or target_points[-1] > source_points[-1]:
            raise ValueError("linear soil remap does not cover every target midpoint")
    elif kind == _CONSERVATIVE_REMAP:
        if set(remap) != {
            "kind", "source_value_location", "target_value_location", "coverage",
        }:
            raise ValueError(
                "conservative soil remap requires exactly one coverage policy"
            )
        if remap["source_value_location"] != "layer_mean" \
                or remap["target_value_location"] != "layer_mean" \
                or remap["coverage"] != "require_complete":
            raise ValueError(
                "conservative soil remap requires layer means and complete coverage"
            )
        if source[0][0] > target[0][0] or source[-1][1] < target[-1][1]:
            raise ValueError(
                "conservative soil source layers do not completely cover target layers"
            )
    else:
        raise ValueError(f"unsupported declarative soil remap kind {kind!r}")

    missing = _object(
        contract["missing"],
        "composition.soil_layers.missing",
        allowed={"land", "ocean"},
        required={"land", "ocean"},
    )
    ocean = _object(
        missing["ocean"],
        "composition.soil_layers.missing.ocean",
        allowed={"stage", "temperature", "moisture"},
        required={"stage", "temperature", "moisture"},
    )
    land_policy = missing["land"]
    if isinstance(land_policy, Mapping):
        # A declared, BOUNDED land repair for producers whose soil tiling
        # disagrees with their land-cover field on a handful of coastal
        # cells (ECMWF's fractional land mask against its soil tile
        # mask): the nearest fully defined soil column inside the stated
        # radius answers.  A land cell beyond it stays missing, exactly as
        # under "reject": it is not a donor to the masked horizontal
        # mapping, which fills the target land near it from the source
        # land around it (woof/mapped_source.py:
        # _admit_source_land_soil_gaps).
        land = _object(
            land_policy,
            "composition.soil_layers.missing.land",
            allowed={"kind", "maximum_cells"},
            required={"kind", "maximum_cells"},
        )
        if land["kind"] != "nearest_soil_column_within_cells":
            raise ValueError(
                "soil land repair kind must be "
                "'nearest_soil_column_within_cells'"
            )
        radius = land["maximum_cells"]
        if isinstance(radius, bool) or not isinstance(radius, int) \
                or not 1 <= radius <= 8:
            raise ValueError(
                "soil land repair maximum_cells must be an integer in "
                "[1, 8]: a wider search would silently paint whole "
                "regions from one column"
            )
    elif land_policy != "reject":
        raise ValueError(
            "soil missing land policy must be 'reject' or a bounded "
            "nearest_soil_column_within_cells repair object"
        )
    if ocean["stage"] != "after_horizontal_interpolation" \
            or ocean["temperature"] != "skin_temperature" \
            or _number(ocean["moisture"], "soil ocean moisture repair") != 1.0:
        raise ValueError(
            "soil missing policy must repair ocean temperature from skin "
            "temperature with moisture 1.0 after horizontal interpolation"
        )

    if mapping is not None:
        if has_nodes and moisture_provenance is not None:
            _validate_mapping_mixed_node_fields(
                contract, node_depths, provenance_layers, mapping,
            )
        elif has_nodes:
            _validate_mapping_node_fields(contract, node_depths, mapping)
        else:
            _validate_mapping_fields(contract, source, mapping)
    return contract


def soil_layer_bounds(
    contract: Mapping[str, object], key: str,
) -> tuple[tuple[float, float], ...]:
    """Return already-validated source or target bounds as float tuples."""

    if key not in {"source_layers", "target_layers"}:
        raise ValueError("soil layer key must be source_layers or target_layers")
    return _layer_bounds(
        contract[key],
        f"soil {key}",
        bind_selectors=key == "source_layers",
    )


def soil_node_depths(contract: Mapping[str, object]) -> tuple[float, ...]:
    """Return an already-validated node contract's sample depths."""

    selector_fields = _CANONICAL_SOIL_FIELDS
    if contract.get("moisture_provenance") is not None:
        selector_fields = (str(contract["temperature_field"]),)
    return _node_depths(
        contract["source_nodes"], "soil source_nodes", bind_selectors=True,
        selector_fields=selector_fields,
    )


#: The RUC admission table.  One row per SOURCE GEOMETRY the contract
#: language can declare, and the row is the whole policy: which arm of
#: ``init_soil_3_real`` runs, how the source's own sample depths are formed
#: from the declaration, and the integer scale those depths are carried in.
#: Adding a future model is choosing a row, not writing an arm -- which is
#: the point: ``rw-wps-icon-eu-regular-grib2`` and
#: ``rw-wps-ecmwf-open-data-oper-grib2`` reach RUC through this table with
#: no ICON code and no ECMWF code anywhere.
#:
#: ``depth_scale_per_m`` is the integer unit the sample depths are carried
#: in, and it is part of the policy rather than a module constant because
#: WRF's own encoding cannot express every producer's ladder.  WRF carries
#: soil level depths as INTEGER CENTIMETRES -- ``char2int2``
#: (``share/module_optional_input.F:1949-1954``) forms them by integer
#: division and ``init_soil_3_real:1994`` divides the INTEGER by 100 -- so
#: 100 is the scale every metgrid-shaped source uses and the scale the
#: oracle rows are measured in.  ICON's shallowest interior node is
#: 0.005 m; at scale 100 it rounds onto the 0.0 m surface node and two
#: distinct samples collide at one depth.  That is a limit of metgrid's
#: FILE FORMAT, not of the soil column, so the policy declares the finer
#: scale instead of refusing the source.  The arithmetic is the same
#: ``REAL(level)/scale`` in the same float32 expression order either way;
#: what changes is only which producers can say what they publish.
RUC_REMAP_POLICIES: dict[str, dict[str, object]] = {
    "node_point_samples": {
        "arm": "flag_soil_levels",
        "geometry": "levels",
        "declaration": "source_nodes",
        "authority": "share/module_soil_pre.F:init_soil_3_real:1950-2000",
    },
    "layer_midpoint_samples": {
        "arm": "flag_soil_layers",
        "geometry": "layers",
        "declaration": "source_layers",
        "authority": (
            "share/module_soil_pre.F:init_soil_3_real:1899-1948, sampled at "
            "share/module_optional_input.F:1949-1954 char2int2 midpoints"),
    },
}

#: The scales a policy may declare, coarsest first.  A source is carried at
#: the FIRST scale that represents every one of its declared depths exactly
#: and distinctly, so every source WRF itself can express keeps WRF's own
#: centimetres and stays bit-identical to the oracle rows.
_RUC_DEPTH_SCALES = (100, 1000)


def _scaled_depths(depths: tuple[float, ...]) -> tuple[tuple[int, ...], int]:
    """Return the source depths as distinct integers, and their scale.

    ``100`` -- WRF's own integer centimetres -- is tried first and is what
    every metgrid-shaped source gets, so this function changes no existing
    number.  A finer scale is used only when the coarser one would MERGE
    two declared depths or MOVE one, both of which would be inventing a
    sample depth the producer never published.
    """

    for scale in _RUC_DEPTH_SCALES:
        scaled = tuple(int(round(depth * scale)) for depth in depths)
        exact = all(
            abs(value / scale - depth) <= 1e-12
            for value, depth in zip(scaled, depths))
        distinct = all(
            later > earlier for earlier, later in zip(scaled, scaled[1:]))
        if exact and distinct:
            return scaled, scale
    raise ValueError(
        f"soil sample depths {depths!r} are not representable as distinct "
        f"integers at any declared scale {_RUC_DEPTH_SCALES!r}; a sample "
        "depth may not be rounded onto another one, because that would hand "
        "init_soil_3_real two source values at one depth and a zero "
        "interpolation denominator")


def ruc_soil_remap_policy(value: object) -> dict[str, object]:
    """Return the DECLARED RUC remap policy for one mapped soil contract.

    This is the RUC target the Noah comparison in
    :func:`validate_soil_layer_contract` requires be ADDED rather than
    faked by widening that check.  It is a TABLE lookup on the geometry the
    contract already declares -- :data:`RUC_REMAP_POLICIES` -- and it
    dispatches into the two arms of WRF's own ``init_soil_3_real``, the arms
    :func:`woof.ingest.ruc_soil.remap_soil_to_ruc_levels` already
    reproduces and ``woof/data/ruc/oracle/soil_ingest.csv`` already
    measures:

    ``source_nodes``  -> ``flag_soil_levels``
        The producer's own point samples are the interpolation samples,
        with no synthetic anchors. The RUC evaluator checks these source
        samples against its requested six- or nine-level target; an
        uncovered level is refused before interpolation.  This admits the
        RUC-family ladders (HRRR/RAP/RRFS publish TSOIL/SOILW at exactly
        RUC's nine depths, where the remap is the identity) AND any other
        published ladder, because WRF's arm INTERPOLATES -- the identity is
        a property of those three sources, not a requirement of the code.

    ``source_layers`` -> ``flag_soil_layers``
        Each declared layer is sampled at the INTEGER midpoint of its
        bounds, exactly as ``char2int2`` forms the level a metgrid
        ``ST<top><bottom>`` name encodes: a 0-7 cm layer is sampled at
        3 cm, not 3.5 and not 7.  The profile is then anchored at 0 m and
        3 m by WRF's own ``:1899-1948`` anchors.  This is the arm the
        ERA5/GFS NATIVE routes have always run; the only thing missing was
        letting a MAPPED contract declare it.

    ``moisture_provenance`` -- a moisture the composition derives from a
    published layer-mass field rather than reading volumetrically -- is NOT
    a refusal here.  It is a property of how the mapped route builds
    ``RW_SOIL_MOISTURE``, resolved before this function is reached and
    resolved identically for Noah, which has always accepted it.  Refusing
    RUC for it would be refusing a source because it publishes soil water a
    different way, which is not a physical impossibility.

    Two things ARE refused, each naming the concrete breakage:

    * a layer source with fewer than two published layers -- WRF's 0 m
      moisture anchor (``:1938-1943``) is a linear extrapolation off the
      TOP TWO layers, so with one layer WRF reads ``sm_input(3)``
      uninitialised.  There is no WRF number to reproduce, and one 0-10 cm
      slab does not contain the 3 m of column ``LSMRUC`` integrates.
    * a ladder two of whose depths collapse onto one integer at every
      declared scale -- see :func:`_scaled_depths`.

    Returns the policy row extended with ``contract`` (the fully validated
    contract) and ``sample_depths``/``depth_scale_per_m`` (the integer
    sample depths the remap runs on).
    """

    contract = validate_soil_layer_contract(value)
    if contract.get("deep_donor") is not None:
        # A declared deep-soil donor supplies the layers below the source's
        # own (woof/ingest/deep_soil_donor.py), so RUC sees the stacked
        # column: the one-slab refusal below names a source with nothing
        # under its slab, which this source no longer is.
        from woof.ingest.deep_soil_donor import effective_soil_layer_contract

        contract = validate_soil_layer_contract(
            effective_soil_layer_contract(contract))
    if contract.get("source_nodes") is not None:
        scaled, scale = _scaled_depths(soil_node_depths(contract))
        return dict(
            RUC_REMAP_POLICIES["node_point_samples"],
            policy="node_point_samples",
            contract=contract,
            sample_depths=scaled,
            depth_scale_per_m=scale,
        )

    bounds = soil_layer_bounds(contract, "source_layers")
    if len(bounds) < 2:
        raise ValueError(
            f"mapped soil declares {len(bounds)} source layer(s) {bounds!r}; "
            "the RUC layer arm needs at least two.  "
            "init_soil_3_real:1938-1943 builds the 0 m moisture anchor by "
            "extrapolating off the top TWO layers and with one layer reads "
            "sm_input(3) uninitialised, so there is no WRF behaviour to "
            "reproduce -- and a single shallow slab does not carry the 3 m "
            "of soil column LSMRUC integrates")
    # ``_layer_bounds`` has already proved the layers are contiguous and
    # ordered, so the edge ladder is top[0] followed by every bottom, and
    # layer ``i`` spans ``ladder[i] .. ladder[i + 1]``.
    ladder = (bounds[0][0],) + tuple(bottom for _, bottom in bounds)
    scaled_edges, scale = _scaled_depths(ladder)
    # char2int2 / :1339-1352 -- the midpoint is formed by INTEGER division
    # of the two integer edges, so it is the file format's own midpoint and
    # never a half unit.
    midpoints = tuple(
        (scaled_edges[index] + scaled_edges[index + 1]) // 2
        for index in range(len(bounds)))
    if any(later <= earlier for earlier, later in zip(midpoints, midpoints[1:])):
        raise ValueError(
            f"mapped soil source_layers {bounds!r} have integer midpoints "
            f"{midpoints!r} that are not strictly increasing; "
            "init_soil_3_real interpolates between successive source levels "
            "and two layers sampled at one depth give it a zero denominator")
    return dict(
        RUC_REMAP_POLICIES["layer_midpoint_samples"],
        policy="layer_midpoint_samples",
        contract=contract,
        sample_depths=midpoints,
        depth_scale_per_m=scale,
    )


def validate_ruc_soil_node_source(value: object) -> dict[str, object]:
    """The node arm of :func:`ruc_soil_remap_policy`, kept by name.

    Callers that specifically want the node geometry -- and the tests that
    pin what the identity policy meant -- keep this entry point.  It is now
    a thin restriction of the table, not the whole RUC admission.
    """

    policy = ruc_soil_remap_policy(value)
    if policy["policy"] != "node_point_samples":
        raise ValueError(
            "mapped soil source is a LAYER geometry; use "
            "ruc_soil_remap_policy, whose layer_midpoint_samples row runs "
            "init_soil_3_real's flag_soil_layers arm")
    return policy["contract"]


def soil_source_sample_count(contract: Mapping[str, object]) -> int:
    """How many soil samples (layers or nodes) the source declares."""

    if contract.get("source_nodes") is not None:
        return len(contract["source_nodes"])
    return len(contract["source_layers"])


#: Where a layer-form soil source's values actually sit, one row per
#: ``source_value_location`` the ``linear_point_samples`` remap can sample.
#: Adding a dataset whose values sit somewhere else is a row here plus a
#: line in a composition document, never a new interpolation arm.
#:
#: Declaring the wrong row silently shifts the whole column, and most
#: layer-form sources genuinely are WRF-convention midpoint sources: WRF
#: places layer-form soil values at the INTEGER-centimetre layer midpoints
#: (module_optional_input.F:char2int2, (top+bottom)/2 in whole cm).  Declare
#: the location the dataset's own documentation states, not the one that
#: looks tidiest.
@dataclass(frozen=True)
class _SampleLocation:
    """One row of the sampler's table.

    ``states_declared_face`` says whether the depth this row returns is a
    boundary the source layer itself declares (a face), or a location
    DERIVED from those bounds.  A face landing exactly on a bracketing
    anchor is the source stating that boundary's value itself; a derived
    location landing there is a coincidence of the derivation, and the two
    want opposite answers (see :func:`linear_sample_plan`).
    """

    basis: str
    sample: Callable[[float, float], float]
    states_declared_face: bool


_LINEAR_SAMPLE_LOCATIONS: Mapping[str, _SampleLocation] = {
    "wrf_integer_cm_layer_midpoint": _SampleLocation(
        basis="WRF's own layer-form convention: the integer-centimetre "
              "layer midpoint, (top+bottom)/2 rounded in whole cm "
              "(module_optional_input.F:char2int2)",
        sample=lambda top, bottom: (
            (int(round(top * 100.0)) + int(round(bottom * 100.0))) // 2
        ) / 100.0,
        states_declared_face=False,
    ),
    "layer_bottom": _SampleLocation(
        basis="the layer's bottom face, for a source whose values are "
              "stated at the deeper boundary of each layer",
        sample=lambda _top, bottom: float(bottom),
        states_declared_face=True,
    ),
    "layer_top": _SampleLocation(
        basis="the layer's top face, for a source whose values are stated "
              "at the shallower boundary of each layer",
        sample=lambda top, _bottom: float(top),
        states_declared_face=True,
    ),
}


def linear_sample_rule(location: object) -> _SampleLocation:
    """The sampling rule for one declared ``source_value_location``.

    Refuses a location nothing here can sample, naming it and listing the
    rows that exist.
    """

    key = location if isinstance(location, str) else None
    rule = _LINEAR_SAMPLE_LOCATIONS.get(key) if key is not None else None
    if rule is None:
        known = ", ".join(
            f"{name!r} ({row.basis})"
            for name, row in _LINEAR_SAMPLE_LOCATIONS.items()
        )
        raise ValueError(
            f"soil remap source_value_location={location!r} has no sampling "
            f"rule, so this contract does not say where its values sit. "
            f"linear_point_samples samples a layer source at: {known}. "
            "Declare the one this dataset's values are stated at, or add its "
            "row to _LINEAR_SAMPLE_LOCATIONS"
        )
    return rule


@dataclass(frozen=True)
class LinearSamplePlan:
    """The nodes a ``linear_point_samples`` remap interpolates between.

    ``depths`` are those nodes in order and ``labels`` names each one for a
    refusal.  ``top_anchor``/``bottom_anchor`` say whether the surface and
    deep anchor values are nodes at all, which is what the remap needs to
    stack the same values the validator checked.
    """

    depths: tuple[float, ...]
    labels: tuple[str, ...]
    top_anchor: bool
    bottom_anchor: bool


def _same_depth(left: float, right: float) -> bool:
    """One nanometre apart is one depth; declared bounds are never closer."""

    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def linear_sample_plan(
    contract: Mapping[str, object],
) -> LinearSamplePlan:
    """The nodes a ``linear_point_samples`` source's values actually sit at.

    ONE function, both doors: :func:`validate_soil_layer_contract` checks
    ordering and coverage on these nodes, and
    ``woof.ingest.soil._remap_declared_soil`` interpolates on them and
    stacks its values to match.  They were two separate expressions before,
    so a contract whose sample depths collided (two layers inside one
    centimetre, or a first layer whose integer midpoint rounds onto the 0 m
    skin anchor) passed validation on its layer bottoms and then divided by
    zero between two identical nodes.

    The nodes are bracketed by the contract's own top and bottom anchor
    depths, because those are the surface and deep boundary values the
    interpolation runs between -- UNLESS the source states its own value at
    that exact depth.  A source declaring ``layer_top`` whose shallowest
    layer starts at 0.0 m states the surface value itself; so does a
    ``layer_bottom`` source whose deepest layer ends at the deep anchor.
    The anchor there would be a second value at one depth, which is why
    that shape used to be refused as a collision with no way out.  The
    source's own value wins and the redundant anchor is not a node.

    This applies only to a row that samples a face the source DECLARES
    (``_SampleLocation.states_declared_face``).  A derived location that
    lands on an anchor -- an integer-centimetre midpoint rounding down onto
    0 m -- is not the source stating a surface value, it is the derivation
    collapsing, and that stays a refusal.
    """

    remap = contract.get("remap")
    if not isinstance(remap, Mapping):
        raise TypeError("composition.soil_layers.remap must be an object")
    rule = linear_sample_rule(remap.get("source_value_location"))
    anchors: list[float] = []
    for name in ("top_anchor", "bottom_anchor"):
        anchor = remap.get(name)
        if not isinstance(anchor, Mapping) or "depth" not in anchor:
            raise ValueError(
                f"linear soil remap requires a {name} carrying a depth: it is "
                "the boundary value the interpolation is bracketed by"
            )
        anchors.append(_number(anchor["depth"], f"soil remap {name}.depth"))
    source = soil_layer_bounds(contract, "source_layers")
    samples = tuple(rule.sample(top, bottom) for top, bottom in source)
    stated = rule.states_declared_face and bool(samples)
    top_anchor = not (stated and _same_depth(samples[0], anchors[0]))
    bottom_anchor = not (stated and _same_depth(samples[-1], anchors[1]))
    depths = (
        *((anchors[0],) if top_anchor else ()),
        *samples,
        *((anchors[1],) if bottom_anchor else ()),
    )
    labels = (
        *(("the top anchor",) if top_anchor else ()),
        *(f"source layer {index + 1}" for index in range(len(samples))),
        *(("the bottom anchor",) if bottom_anchor else ()),
    )
    return LinearSamplePlan(
        depths=depths, labels=labels,
        top_anchor=top_anchor, bottom_anchor=bottom_anchor,
    )


def linear_source_sample_depths(
    contract: Mapping[str, object],
) -> tuple[float, ...]:
    """Just the node depths of :func:`linear_sample_plan`."""

    return linear_sample_plan(contract).depths


def conservative_overlap_weights(
    source: tuple[tuple[float, float], ...],
    target: tuple[tuple[float, float], ...],
) -> np.ndarray:
    """Return target-by-source overlap weights for layer-mean remapping."""

    weights = np.zeros((len(target), len(source)), dtype=np.float64)
    for target_index, (target_top, target_bottom) in enumerate(target):
        thickness = target_bottom - target_top
        for source_index, (source_top, source_bottom) in enumerate(source):
            overlap = max(
                0.0,
                min(target_bottom, source_bottom) - max(target_top, source_top),
            )
            weights[target_index, source_index] = overlap / thickness
        if not math.isclose(
            float(weights[target_index].sum()), 1.0, rel_tol=0.0, abs_tol=2e-15,
        ):
            raise ValueError(
                f"conservative soil remap target layer {target_index} lacks "
                "complete source coverage"
            )
    return weights


__all__ = [
    "MAPPED_SOIL_MOISTURE", "MAPPED_SOIL_TEMPERATURE",
    "NOAH_LAYER_BOUNDS_M", "NOAH_TARGET_SOIL_LAYERS",
    "RUC_TARGET_LEVEL_DEPTHS_M",
    "conservative_overlap_weights",
    "LinearSamplePlan", "linear_sample_plan",
    "linear_sample_rule", "linear_source_sample_depths",
    "soil_layer_bounds", "soil_node_depths", "soil_source_sample_count",
    "RUC_REMAP_POLICIES", "ruc_soil_remap_policy",
    "validate_ruc_soil_node_source", "validate_soil_layer_contract",
]
