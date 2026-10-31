"""Hash-bound composition of mapped products on compatible donor grids.

``rw-wps.mapping.v1`` describes one source format and the canonical field
semantics, but a meteorological archive may split those fields across product
files with different geographic extents.  This module implements the first
strict composition slice: an invariant GRIB terrain product whose regular
latitude/longitude grid contains the primary forcing grid as an exact
coordinate subset.

The join is intentionally narrow.  It does not fabricate terrain, treat WRF
target ``HGT_M`` as source terrain, or silently interpolate between donor
grids.  Mapping, composition, data, provenance, and native decoder bytes are
verified before and after decoding.  The terrain selector and unit conversion
remain owned by the sealed mapping document.  The same contract is used for
GRIB1, GRIB2, and NetCDF. Soil source geometry, target geometry, remapping,
and ocean repair are declarative and contain no source-family dispatch.

Cross-source composition extends the same contract across PACKAGED SOURCES:
a composition may declare that a canonical field's values come from a
DIFFERENT source's decode (soil from a physical analysis under an
AI-forecast atmosphere, and in general any declared field borrowed across
products).  The grammar is per-field source bindings in
``field_sources``: each binding names the contributing source id, pins that
source's OWN sealed mapping document by SHA-256, and declares a
grid-alignment and a cycle/time-alignment rule.  The primary mapping
declares each borrowed field as ``provider: "composition_bound"`` so the
gap is a declaration, never an absence.  A binding declares either exact
coordinate subsetting or native bilinear alignment with nearest categories
and land/sea-aware surface state.  Member alignment and vertical merging
remain explicit contracts.  The
contributing mapping document is deliberately NOT a row of the input
manifest: the composition itself pins it by hash, and the manifest pins the
composition, so the donor's table version is bound transitively while the
manifest schema stays v1.  Provenance receipts name every contributing
source with its hashes.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Sequence as _ABCSequence
from dataclasses import dataclass, field as dataclass_field, replace as _dataclass_replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


_SHARED_COMPOSITION = ContextVar("shared_composition", default=None)


@contextmanager
def shared_composition_source(source_key, bank_root):
    """Bind a roster-owned bank in this worker without changing process env.

    The roster removes the bank after its last preparation. The context is
    local to this thread/task; process workers must receive the same two
    arguments. Native decoding remains entirely in the Rust engine.
    """
    token = _SHARED_COMPOSITION.set((str(source_key),
        None if bank_root is None else Path(bank_root).resolve()))
    try:
        yield
    finally:
        _SHARED_COMPOSITION.reset(token)


def _shared_composition_key(arguments, source_key):
    """Bind exact decode operands as well as the planner's source partition."""
    before = arguments["before"]
    def bound(path):
        # Source references carry these paths. Reusing a different spelling
        # would change the frame/proof even when the source words matched.
        return {"path": str(path), "sha256": before[str(path)]}

    identity = {
        "schema": "ensemble-composition-v1", "source_key": source_key,
        "mapping": before[str(arguments["mapping_path"])],
        "composition": before[str(arguments["composition_path"])],
        # Both manifests were fully validated before this seam. Relative
        # paths depend on their location, so bind their verified role values
        # below instead of their serialized location-dependent digest.
        "primary": [bound(path) for path in arguments["primary"]],
        "supplements": {role: [bound(path) for path in paths]
                        for role, paths in arguments["supplements"].items()},
        "provenance": {role: bound(path) for role, path in arguments["provenance"].items()},
        "contributing": {role: bound(path) for role, path in arguments["contributing"].items()},
        "decoders": {role: bound(path) for role, path in arguments["decoders"].items()},
        "engine": _snapshot_authority(arguments["engine"]).sha256,
        "implementation": _snapshot_authority(Path(__file__)).sha256,
        "member": arguments["member"], "member_identity": arguments["member_identity"],
        "grids": [grid.definition() for grid in arguments["atmospheric_grids"]],
        "lead_batch": arguments["lead_batch"],
        "memory_budget_bytes": arguments["memory_budget_bytes"],
        # A fallback the engine substitutes for an unpublished field changes
        # the composed frames, so two members that differ only here must not
        # share one bank entry.  Bound only when present, so a composition
        # without fallbacks keeps its key.
        **({"unpublished_fallbacks": dict(sorted(arguments["unpublished_fallbacks"].items()))}
           if arguments.get("unpublished_fallbacks") else {}),
    }
    return _canonical_sha256(identity)


def _shared_engine_composition(arguments, binding):
    """One writer per source; later members open their own lazy frame reader."""
    import shutil

    from woof.remote_artifact_cache import Lease
    from woof.timeline.ledger import emit

    source_key, bank_root = binding
    key = _shared_composition_key(arguments, source_key)
    bank_root.mkdir(parents=True, exist_ok=True)
    directory = bank_root / key
    with Lease(bank_root / f"{key}.lock", timeout=3600) as lock:
        if lock.file is None:
            raise TimeoutError("shared source composition writer did not release its lease within 3600 s; refusing concurrent writes to the same frames")
        marker = directory / "READY.json"
        hit = marker.is_file()
        if not hit:
            # Only this bank entry, created by this roster, can be unfinished.
            # No ready entry is silently repaired or demoted to a warning.
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir()
        attrs = {"source_key": source_key, "composition_key": key, "cache_hit": hit}
        if hit:
            ready = json.loads(marker.read_text(encoding="utf-8"))
            if ready.get("key") != key or ready.get("schema") != "ensemble-composition-v1":
                raise ValueError("shared composition marker differs from its source binding; refusing wrong-source frames")
            for name, digest in ready["documents"].items():
                if name not in ("frames.json", "composition.json") or _snapshot_authority(
                        directory / "composed" / name).sha256 != digest:
                    raise ValueError("shared composition metadata changed after seal; refusing unbound frame values")
            if set(ready["documents"]) != {"frames.json", "composition.json"}:
                raise ValueError("shared composition seal lacks frame evidence; refusing unbound frame values")
            emit("prep.compose", "mark", product="ensemble", attrs=attrs)
        else:
            # Rust compose performs both decode and composition. These spans
            # overlap deliberately; timeline unions them rather than adding.
            emit("prep.decode", "begin", product="ensemble", attrs=attrs)
            emit("prep.compose", "begin", product="ensemble", attrs=attrs)
        try:
            bundle = _compose_through_engine(**{
                **arguments, "_shared_directory": directory / "composed", "_shared_hit": hit})
            if not hit:
                pending = directory / "READY.pending"
                pending.write_text(json.dumps({"schema": "ensemble-composition-v1", "key": key,
                    "decode_memory": bundle.native_decode_memory,
                    "documents": {name: _snapshot_authority(directory / "composed" / name).sha256
                                  for name in ("frames.json", "composition.json")}},
                    sort_keys=True) + "\n", encoding="utf-8")
                pending.replace(marker)
            return bundle
        finally:
            if not hit:
                attrs["complete"] = marker.is_file()
                emit("prep.compose", "end", product="ensemble", attrs=attrs)
                emit("prep.decode", "end", product="ensemble", attrs=attrs)

from woof.authority_digests import map_parallel
from woof.mapped_source import (
    MappedSourceFrame,
    NothingMatched,
    _DecodedCollection,
    _DirectValue,
    _AuthoritySnapshot,
    _array_sha256,
    _decode_netcdf,
    _decode_grib,
    _load_json_document,
    _load_json_bytes,
    _mapped_engine_choice,
    _materialize_frames,
    _object,
    _require_authority_snapshot,
    _snapshot_authority,
    _string,
    completed_field_summary,
    load_mapping,
    mapped_frames_to_regular_snapshots,
    warn_completed_fields,
)
from woof.mapped_engine_bridge import ENGINE_RUST as _ENGINE_RUST
from woof.ingest.grib import Era5Snapshot
from woof.ingest.source_coverage import ScratchDiskRefusal, scratch_disk_refusal
from woof.ingest.soil_contract import (
    MAPPED_SOIL_MOISTURE,
    MAPPED_SOIL_TEMPERATURE,
    validate_soil_layer_contract,
)


COMPOSITION_SCHEMA = "gpuwm-mapped-composition-v2"
#: An explicit "no runnable composition exists yet" declaration.  An
#: atmosphere-only source (complete 3-D state, zero land surface) ships
#: this document in its composition role: loading it always refuses, and
#: the refusal names the state the source does not publish, so no
#: initialization can quietly be built on invented surfaces.  The
#: cross-source composition that will supply that state consumes the
#: same declaration when it arrives.
PENDING_COMPOSITION_SCHEMA = "gpuwm-cross-source-composition-pending-v1"
INPUT_MANIFEST_SCHEMA = "gpuwm-mapped-composition-inputs-v1"
RECEIPT_SCHEMA = "gpuwm-mapped-composition-receipt-v1"
BINDING_RECEIPT_SCHEMA = "gpuwm-cross-source-binding-v1"
_EXTERNAL_FIELD = "terrain_height"
_SOURCE_FORMATS = {"grib1", "grib2", "netcdf"}
_SOIL_PAIR = ("soil_temperature", "volumetric_soil_moisture")

#: The DECLARED clocks a cross-source binding can run on.
#:
#: ``valid_time_exact`` requires the contributing source at every primary
#: valid time.  ``cycle_invariant_broadcast`` is for borrowed statics the
#: donor publishes once per cycle: proven byte-invariant across every
#: supplied donor time, then carried to every primary time of the one
#: primary cycle.  ``source_cycle_analysis_broadcast`` is the hybrid
#: alignment: the donor supplies exactly ONE analysis record whose valid
#: time must BE the primary's source cycle, and that initialization state
#: is carried to every primary lead -- soil from a physical analysis under
#: an AI forecast is deliberately frozen at the shared analysis, and the
#: receipt names every carried time so nothing is silently frozen.
#: ``source_cycle_window_start_broadcast`` takes the donor's ONE record
#: valid at the primary's first valid time, from the primary's own cycle,
#: and carries it: a forecast started at its cycle gets the same analysis
#: record, a time-lagged member started at lead L gets its own run's state
#: at L instead of an analysis three to seventeen hours stale.
_BOUND_TIME_ALIGNMENTS = frozenset({
    "valid_time_exact", "cycle_invariant_broadcast",
    "source_cycle_analysis_broadcast", "source_cycle_analysis_select_broadcast",
    "source_cycle_window_start_broadcast",
})

#: Borrowed fields support exact contiguous/strided subsets and native
#: horizontal regridding. Continuous fields use bilinear weights; masks use nearest.
_BOUND_GRID_ALIGNMENTS = frozenset({
    "exact_coordinate_subset", "exact_coordinate_strided_subset", "nearest_coordinate",
    "bilinear_mask_aware_nearest",
})

_BINDING_KEYS = {
    "source_id", "mapping_role", "mapping_sha256", "data_role",
    "provenance_role", "fields", "grid_alignment", "time_alignment", "vertical_merge",
}
#: Optional binding keys.  ``companion`` declares the contributing source a
#: DECLARED COMPANION of the primary: a run of the same grid and cycle
#: whose column extends the primary's vertical ladder above the primary's
#: top (a member-only ensemble subset completed by its deterministic run).
_BINDING_OPTIONAL_KEYS = {"companion"}
_COMPANION_KEYS = {"vertical_extension", "fields"}
#: The one declared way a companion extends the primary ladder: the
#: companion's levels strictly above the primary's top are stacked on top
#: of the primary column, and every primary level stays the primary's.
_COMPANION_VERTICAL_EXTENSIONS = frozenset({"above_primary_top"})
#: The vertical coordinate whose "above" this composition can name:
#: pressure, where above is lower pressure.  A hybrid ladder would need its
#: A/B coefficients spliced as well, which is not declared here.
_COMPANION_VERTICAL_KINDS = frozenset({"pressure"})


def _digest(value: object, label: str) -> str:
    text = _string(value, label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return text


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def load_composition(
    path: str | Path,
    mapping_path: str | Path,
    *,
    _raw: object | None = None,
    _mapping_raw: object | None = None,
) -> dict[str, object]:
    """Validate one composition contract against exact mapping bytes."""

    path = Path(path).resolve()
    mapping_path = Path(mapping_path).resolve()
    raw = _load_json_document(path, "composition") if _raw is None else _raw
    if isinstance(raw, dict) and raw.get("schema") == PENDING_COMPOSITION_SCHEMA:
        raise ValueError(_pending_composition_refusal(raw, path))
    contract = _object(
        raw, "composition",
        allowed={
            "schema", "name", "mapping_binding", "soil_layers", "supplements",
            "field_sources",
        },
        required={
            "schema", "name", "mapping_binding", "soil_layers", "supplements",
        },
    )
    if contract["schema"] != COMPOSITION_SCHEMA:
        raise ValueError(f"unsupported composition schema {contract['schema']!r}")
    _string(contract["name"], "composition.name")
    if contract["mapping_binding"] != "input_manifest_sha256":
        raise ValueError("composition mapping_binding must be input_manifest_sha256")
    mapping = load_mapping(mapping_path, _raw=_mapping_raw)

    declared_bound = {
        name for name, field in mapping["fields"].items()
        if field.get("provider") == "composition_bound"
    }
    bindings = _validate_field_sources(
        contract.get("field_sources", {}), mapping, declared_bound,
    )
    covered = {
        name: binding_name
        for binding_name, binding in bindings.items()
        for name in binding["fields"]
    }
    unbound = sorted(declared_bound - set(covered))
    if unbound:
        raise ValueError(
            f"composition_bound field(s) {unbound} have no contributing "
            "source; every declared gap must be bound in field_sources"
        )
    soil_bindings = {
        covered[name] for name in _SOIL_PAIR if name in covered
    }
    if len(soil_bindings) > 1 or (
        soil_bindings and any(name not in covered for name in _SOIL_PAIR)
    ):
        raise ValueError(
            "the canonical soil pair must ride one contributing source: "
            "a soil-depth contract validates against a single donor "
            "mapping's layer geometry"
        )
    validate_soil_layer_contract(
        contract["soil_layers"],
        # When soil is borrowed, the layer geometry belongs to the donor
        # mapping, which decode re-validates once its bytes are pinned.
        mapping=None if soil_bindings else mapping,
    )

    supplements = _object(
        contract["supplements"], "composition.supplements",
        allowed={_EXTERNAL_FIELD}, required=set(),
    )
    terrain_supplement = supplements.get(_EXTERNAL_FIELD)
    if terrain_supplement is not None and _EXTERNAL_FIELD in covered:
        raise ValueError(
            "terrain has two providers: the declared supplement and a "
            "contributing source binding"
        )
    if terrain_supplement is None and _EXTERNAL_FIELD not in covered:
        raise ValueError(
            "terrain_height has no contributing source and no supplement; "
            "the composition must provide terrain through exactly one"
        )
    if terrain_supplement is not None:
        _validate_terrain_supplement(terrain_supplement, mapping)
    return contract


def _validate_terrain_supplement(
    value: object, mapping: Mapping[str, object],
) -> None:
    spec = _object(
        value,
        f"composition.supplements.{_EXTERNAL_FIELD}",
        allowed={
            "data_role", "provenance_role", "format", "field",
            "selector_authority", "grid_alignment", "time_alignment",
            "require_invariant_across_time",
        },
        required={
            "data_role", "provenance_role", "format", "field",
            "selector_authority", "grid_alignment", "time_alignment",
            "require_invariant_across_time",
        },
    )
    for key in (
        "data_role", "provenance_role", "format", "field",
        "selector_authority", "grid_alignment", "time_alignment",
    ):
        _string(spec[key], f"terrain supplement.{key}")
    if spec["field"] != _EXTERNAL_FIELD:
        raise ValueError("terrain supplement must provide terrain_height")
    if spec["format"] != mapping["format"] \
            or spec["format"] not in _SOURCE_FORMATS:
        raise ValueError("terrain supplement format must equal the mapping format")
    if spec["selector_authority"] != "mapping_field_exact":
        raise ValueError("terrain selector authority must be mapping_field_exact")
    if spec["grid_alignment"] not in {"exact_coordinate_subset", "exact_coordinate_strided_subset"}:
        raise ValueError("terrain grid alignment must be exact_coordinate_subset or exact_coordinate_strided_subset")
    if spec["time_alignment"] not in _TERRAIN_TIME_ALIGNMENTS:
        raise ValueError(
            "terrain time alignment must be one of "
            f"{sorted(_TERRAIN_TIME_ALIGNMENTS)}"
        )
    if spec["require_invariant_across_time"] is not True:
        raise ValueError("terrain supplement must be invariant across all supplied times")
    field = mapping["fields"].get(_EXTERNAL_FIELD)
    if field is None or field.get("derivation") is not None \
            or not field.get("selectors"):
        raise ValueError("mapping terrain_height must have a direct selector")
    materialized_source_axes = tuple(
        axis for axis in field["source_axes"] if axis not in {"time", "member"}
    )
    if materialized_source_axes != ("y", "x") \
            or tuple(field["target_axes"]) != ("y", "x") \
            or field["location"] != "surface" \
            or field.get("staggering", "none") != "none" \
            or field["units"]["target"] != "m" \
            or field["missing"]["kind"] != "reject":
        raise ValueError(
            "mapping terrain_height must be finite unstaggered surface metres"
        )


def _validate_vertical_merge(value, label):
    spec = _object(value, f"{label}.vertical_merge", allowed={
        "method", "pressure_band_hpa", "primary_pressure_scale", "donor_pressure_scale",
        "fields", "max_relative_humidity", "pressure_field"}, required={
        "method", "pressure_band_hpa", "primary_pressure_scale", "donor_pressure_scale",
        "fields", "max_relative_humidity"})
    if spec.get("pressure_field") not in (None, "air_pressure"):
        raise ValueError(f"{label}: pressure_field must name air_pressure")
    if spec["method"] != "cosine_pressure_distance_log_anomaly":
        raise ValueError(f"{label}: unsupported vertical transition method")
    band = spec["pressure_band_hpa"]
    if (not isinstance(band, list) or len(band) != 2 or
            not all(isinstance(x, (int, float)) and np.isfinite(x) for x in band) or
            not 0 <= band[0] < band[1]):
        raise ValueError(f"{label}: pressure_band_hpa must be increasing nonnegative distances")
    for key in ("primary_pressure_scale", "donor_pressure_scale"):
        if spec[key] not in (1, 100):
            raise ValueError(f"{label}: pressure scales must convert Pa or hPa to Pa")
    if not 1 <= spec["max_relative_humidity"] <= 1.05:
        raise ValueError(f"{label}: max_relative_humidity must be in [1, 1.05]")
    if not isinstance(spec["fields"], dict) or not spec["fields"]:
        raise ValueError(f"{label}: vertical merge needs field rules")
    for name, rule in spec["fields"].items():
        _object(rule, f"{label}.{name}", allowed={"primary_field", "representation"},
                required={"primary_field", "representation"})
        _string(rule["primary_field"], f"{label}.{name}.primary_field")
        if (rule["representation"] not in ("canonical", "relative_humidity_percent") or
                (rule["representation"] == "relative_humidity_percent" and name != "specific_humidity")):
            raise ValueError(f"{label}.{name}: unsupported representation")


def _validate_field_sources(
    value: object,
    mapping: Mapping[str, object],
    declared_bound: set[str],
) -> dict[str, dict[str, object]]:
    """Validate the per-field source bindings of a cross-source composition."""

    if not isinstance(value, dict):
        raise TypeError("composition.field_sources must be an object")
    bindings: dict[str, dict[str, object]] = {}
    covered: dict[str, str] = {}
    roles: dict[str, str] = {}
    companions: list[str] = []
    for binding_name, raw_binding in value.items():
        _string(binding_name, "field_sources binding name")
        label = f"composition.field_sources.{binding_name}"
        binding = _object(
            raw_binding, label, allowed=_BINDING_KEYS | _BINDING_OPTIONAL_KEYS,
            required=_BINDING_KEYS - {"vertical_merge"},
        )
        _string(binding["source_id"], f"{label}.source_id")
        _digest(binding["mapping_sha256"], f"{label}.mapping_sha256")
        for key in ("mapping_role", "data_role", "provenance_role"):
            role = _string(binding[key], f"{label}.{key}")
            previous = roles.get(role)
            if previous is not None:
                raise ValueError(
                    f"{label}.{key} reuses role {role!r} already claimed by "
                    f"{previous}; every binding role must be unique"
                )
            roles[role] = f"{label}.{key}"
        if binding["grid_alignment"] not in _BOUND_GRID_ALIGNMENTS:
            raise ValueError(
                f"{label}.grid_alignment must be one of "
                f"{sorted(_BOUND_GRID_ALIGNMENTS)}"
            )
        if binding["time_alignment"] not in _BOUND_TIME_ALIGNMENTS:
            raise ValueError(
                f"{label}.time_alignment must be one of "
                f"{sorted(_BOUND_TIME_ALIGNMENTS)}"
            )
        merge = binding.get("vertical_merge")
        if merge is not None:
            _validate_vertical_merge(merge, label)
            vertical = mapping["coordinates"]["vertical"]
            column_pressure = merge.get("pressure_field") == "air_pressure"
            if column_pressure:
                if vertical["kind"] != "model_level" or vertical["units"] != "1":
                    raise ValueError(f"{label}: column pressure merge requires model_level indices")
                expected = 1
                if merge["donor_pressure_scale"] != 1:
                    raise ValueError(f"{label}: column pressure must be in Pa")
            else:
                if vertical["kind"] != "pressure":
                    raise ValueError(f"{label}: vertical merge requires a pressure coordinate")
                expected = {"Pa": 1, "hPa": 100}.get(vertical["units"])
            if merge["primary_pressure_scale"] != expected:
                raise ValueError(f"{label}: primary pressure scale disagrees with coordinate units")
        if merge is not None and "companion" in binding:
            raise ValueError(f"{label}: companion and vertical_merge both rewrite the column ladder")
        companion = None
        if "companion" in binding:
            companion = _validate_companion(
                binding["companion"], mapping, label=label,
                time_alignment=binding["time_alignment"],
            )
            companions.append(binding_name)
            if len(companions) > 1:
                raise ValueError(
                    f"bindings {companions[0]!r} and {binding_name!r} both "
                    "declare a companion; two companions extending one "
                    "primary ladder would leave the levels above its top "
                    "with two sources"
                )
        fields = binding["fields"]
        # A declared companion may lend nothing but the levels above the
        # primary's top; every other binding exists to lend fields.
        if not isinstance(fields, list) or (not fields and companion is None):
            raise ValueError(f"{label}.fields must be a non-empty list")
        names = [_string(name, f"{label}.fields[]") for name in fields]
        if merge:
            if not set(merge["fields"]).issubset(names):
                raise ValueError(f"{label}: every vertical merge output must be a bound field")
            for name, rule in merge["fields"].items():
                declared = mapping["fields"].get(rule["primary_field"])
                if declared is not None and declared["target_axes"] != ["vertical", "y", "x"]:
                    raise ValueError(f"{label}.{name}: primary column must have vertical,y,x axes")
        if len(set(names)) != len(names):
            raise ValueError(f"{label}.fields repeats a field")
        for name in names:
            declared = mapping["fields"].get(name)
            if declared is None:
                raise ValueError(f"{label} binds unknown field {name!r}")
            if name not in declared_bound and name not in (merge or {}).get("fields", {}):
                raise ValueError(
                    f"field {name!r} has two providers: the primary mapping "
                    f"and contributing source binding {binding_name!r}"
                )
            if name in covered:
                raise ValueError(
                    f"field {name!r} is bound by more than one contributing "
                    f"source ({covered[name]!r} and {binding_name!r})"
                )
            covered[name] = binding_name
        bindings[binding_name] = dict(binding, fields=list(names))
        if companion is not None:
            bindings[binding_name]["companion"] = companion
    return bindings


def _companion_vertical_fields(mapping: Mapping[str, object]) -> list[str]:
    """The primary's own directly decoded fields that stack on its ladder."""

    return sorted(
        name for name, field in mapping["fields"].items()
        if field.get("provider") != "composition_bound"
        and field.get("derivation") is None
        and field.get("selectors")
        and "vertical" in tuple(field.get("source_axes", ()))
    )


def _validate_companion(
    value: object,
    mapping: Mapping[str, object],
    *,
    label: str,
    time_alignment: object,
) -> dict[str, object]:
    """Validate one binding's ``companion`` declaration.

    A declared companion is the only contributing source whose values may
    share a column with the primary's: its levels above the primary's top
    are stacked onto the primary's own fields.  Every refusal names what a
    looser rule would let through.
    """

    spec = _object(
        value, f"{label}.companion",
        allowed=_COMPANION_KEYS, required=_COMPANION_KEYS,
    )
    if spec["vertical_extension"] not in _COMPANION_VERTICAL_EXTENSIONS:
        raise ValueError(
            f"{label}.companion.vertical_extension must be one of "
            f"{sorted(_COMPANION_VERTICAL_EXTENSIONS)}"
        )
    if time_alignment != "valid_time_exact":
        raise ValueError(
            f"{label} declares a companion on the {time_alignment!r} clock; "
            "a companion extends every valid time's column with its own "
            "valid time, and a broadcast would freeze the top of a moving "
            "column, so it requires valid_time_exact"
        )
    vertical = mapping["coordinates"]["vertical"]
    if str(vertical.get("kind")) not in _COMPANION_VERTICAL_KINDS:
        raise ValueError(
            f"{label} declares a companion over a {vertical.get('kind')!r} "
            "ladder; only a pressure ladder says which levels lie above "
            "its top, and a hybrid ladder would need its coefficients "
            "spliced as well"
        )
    fields = spec["fields"]
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"{label}.companion.fields must be a non-empty list")
    names = [_string(name, f"{label}.companion.fields[]") for name in fields]
    if len(set(names)) != len(names):
        raise ValueError(f"{label}.companion.fields repeats a field")
    stacked = _companion_vertical_fields(mapping)
    for name in names:
        if name not in stacked:
            raise ValueError(
                f"{label}.companion extends {name!r}, which the primary "
                "mapping does not decode directly on its vertical ladder; "
                "a companion extends the primary's own columns and lends "
                "nothing else through this declaration"
            )
    short = [name for name in stacked if name not in names]
    if short:
        raise ValueError(
            f"{label}.companion leaves {short} on the primary's ladder "
            "alone: those fields would stop at the primary's top while the "
            "extended fields reach above it, so one frame would carry "
            "columns of two heights"
        )
    return {"vertical_extension": str(spec["vertical_extension"]),
            "fields": list(names)}


def _pending_composition_refusal(raw: Mapping[str, object], path: Path) -> str:
    """The named refusal a pending composition declaration always earns.

    The message is composed from the declaration's own table data: which
    canonical state the source does not publish, why, and where it must
    come from.  A malformed declaration refuses on its own account.
    """

    declaration = _object(
        raw, f"pending composition {path.name}",
        allowed={"schema", "name", "pending"},
        required={"schema", "name", "pending"},
    )
    name = _string(declaration["name"], "pending composition.name")
    pending = _object(
        declaration["pending"], "pending composition.pending",
        allowed={"missing_canonical_state", "reason", "supply_route"},
        required={"missing_canonical_state", "reason", "supply_route"},
    )
    missing = pending["missing_canonical_state"]
    if not isinstance(missing, list) or not missing:
        raise ValueError(
            f"{path.name} pending.missing_canonical_state must be a "
            "non-empty list of canonical field names"
        )
    names = ", ".join(
        _string(entry, "pending.missing_canonical_state entry")
        for entry in missing
    )
    reason = _string(pending["reason"], "pending composition.reason")
    route = _string(pending["supply_route"], "pending composition.supply_route")
    return (
        f"composition {name!r} is an explicit PENDING declaration, not a "
        f"runnable contract: the source publishes no {names}. {reason}. "
        f"Initializing from this source alone would invent those fields, "
        f"so it is refused; {route}."
    )


def _manifest_file(
    raw: object,
    label: str,
    manifest_path: Path,
    actual_path: Path,
    snapshot: _AuthoritySnapshot | None = None,
) -> None:
    actual_path = actual_path.resolve()
    snapshot = snapshot or _snapshot_authority(actual_path)
    if snapshot.path != actual_path:
        raise ValueError(f"{label} snapshot path differs from requested file")
    row = _object(
        raw, label,
        allowed={"path", "bytes", "sha256"},
        required={"path", "bytes", "sha256"},
    )
    declared = Path(_string(row["path"], f"{label}.path"))
    if not declared.is_absolute():
        declared = manifest_path.parent / declared
    if declared.resolve() != actual_path.resolve():
        raise ValueError(f"{label} path differs from the requested file")
    if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) \
            or row["bytes"] < 0:
        raise ValueError(f"{label}.bytes must be a nonnegative integer")
    if row["bytes"] != snapshot.size:
        raise ValueError(f"{label} byte count differs from the requested file")
    if _digest(row["sha256"], f"{label}.sha256") != snapshot.sha256:
        raise ValueError(f"{label} SHA differs from the requested file")


def _manifest_file_inventory(
    raw: object,
    label: str,
    manifest_path: Path,
    actual_paths: Path | Sequence[Path],
    snapshots: Mapping[Path, _AuthoritySnapshot] | None = None,
) -> None:
    paths = (actual_paths,) if isinstance(actual_paths, Path) \
        else tuple(actual_paths)
    if not paths:
        raise ValueError(f"{label} requested inventory is empty")
    if len(paths) == 1 and isinstance(raw, dict):
        _manifest_file(
            raw,
            label,
            manifest_path,
            paths[0],
            None if snapshots is None else snapshots[paths[0].resolve()],
        )
        return
    if not isinstance(raw, list) or len(raw) != len(paths):
        raise ValueError(f"{label} file inventory differs from the request")
    for index, (row, path) in enumerate(zip(raw, paths)):
        _manifest_file(
            row,
            f"{label}[{index}]",
            manifest_path,
            path,
            None if snapshots is None else snapshots[path.resolve()],
        )


def _verify_manifest(
    path: Path,
    expected_sha256: str,
    *,
    mapping_path: Path,
    composition_path: Path,
    primary_files: Sequence[Path],
    supplement_files: Mapping[str, Path | Sequence[Path]],
    provenance_files: Mapping[str, Path],
    decoder_files: Mapping[str, Path],
    _snapshots: Mapping[Path, _AuthoritySnapshot] | None = None,
    _recheck_snapshots: bool = True,
) -> dict[str, object]:
    path = path.resolve()
    supplied = {
        key.resolve(): value for key, value in (_snapshots or {}).items()
    }
    used: dict[Path, _AuthoritySnapshot] = {}

    def snapshot(actual: Path, *, retain_bytes: bool = False) -> _AuthoritySnapshot:
        actual = actual.resolve()
        value = supplied.get(actual)
        if value is None:
            value = _snapshot_authority(actual, retain_bytes=retain_bytes)
        if value.path != actual or (retain_bytes and value.data is None):
            raise ValueError(f"authority snapshot is incomplete for {actual}")
        used[actual] = value
        return value

    manifest_snapshot = snapshot(path, retain_bytes=True)
    observed = manifest_snapshot.sha256
    if observed != _digest(expected_sha256, "input manifest SHA"):
        raise ValueError("composition input manifest SHA mismatch")
    manifest = _object(
        _load_json_bytes(
            manifest_snapshot.data,
            "composition input manifest",
            path,
        ),
        "composition input manifest",
        allowed={
            "schema", "mapping_sha256", "composition_sha256",
            "primary_files", "supplements", "provenance", "decoders",
            "member", "member_identity",
        },
        required={
            "schema", "mapping_sha256", "composition_sha256",
            "primary_files", "supplements", "provenance", "decoders",
        },
    )
    if manifest["schema"] != INPUT_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported composition input schema {manifest['schema']!r}")
    # An EXPLICIT ensemble-member binding, for archives whose product
    # octets carry none: the caller's verified authority named the
    # member, the manifest seals it, and the composition stamps it onto
    # every canonical frame.  Atomic pair -- a member with no identity
    # policy is a number with no provenance.
    if ("member" in manifest) != ("member_identity" in manifest):
        raise ValueError(
            "manifest member and member_identity are an atomic pair"
        )
    if "member" in manifest:
        _string(manifest["member"], "manifest.member")
        _string(manifest["member_identity"], "manifest.member_identity")
    mapping_snapshot = snapshot(mapping_path)
    composition_snapshot = snapshot(composition_path)
    if _digest(manifest["mapping_sha256"], "manifest.mapping_sha256") \
            != mapping_snapshot.sha256:
        raise ValueError("manifest mapping SHA differs from mapping bytes")
    if _digest(manifest["composition_sha256"], "manifest.composition_sha256") \
            != composition_snapshot.sha256:
        raise ValueError("manifest composition SHA differs from composition bytes")
    primary = manifest["primary_files"]
    if not isinstance(primary, list) or len(primary) != len(primary_files):
        raise ValueError("manifest primary file inventory differs from the request")
    for index, (row, source) in enumerate(zip(primary, primary_files)):
        _manifest_file(
            row,
            f"manifest.primary_files[{index}]",
            path,
            source,
            snapshot(source),
        )
    for section, actual in (
        ("supplements", supplement_files), ("provenance", provenance_files),
        ("decoders", decoder_files),
    ):
        rows = _object(
            manifest[section], f"manifest.{section}",
            allowed=set(actual), required=set(actual),
        )
        for role, file_path in actual.items():
            if section == "supplements":
                _manifest_file_inventory(
                    rows[role],
                    f"manifest.{section}.{role}",
                    path,
                    file_path,
                    snapshots={
                        item.resolve(): snapshot(item)
                        for item in (
                            (file_path,)
                            if isinstance(file_path, Path)
                            else tuple(file_path)
                        )
                    },
                )
            else:
                _manifest_file(
                    rows[role],
                    f"manifest.{section}.{role}",
                    path,
                    file_path,
                    snapshot(file_path),
                )
    if _recheck_snapshots:
        for value in used.values():
            _require_authority_snapshot(value)
    return manifest


def _exact_subset_indices(
    larger: np.ndarray,
    smaller: np.ndarray,
    label: str,
    *,
    cyclic_degrees: bool = False,
    allow_strided: bool = False,
) -> np.ndarray:
    larger = np.asarray(larger, dtype=np.float64)
    smaller = np.asarray(smaller, dtype=np.float64)
    if larger.ndim != 1 or smaller.ndim != 1:
        raise ValueError(f"{label} coordinates must be 1-D")
    larger_comparison = np.mod(larger, 360.0) if cyclic_degrees else larger
    smaller_comparison = np.mod(smaller, 360.0) if cyclic_degrees else smaller
    indices = []
    for value in smaller_comparison:
        matches = np.flatnonzero(larger_comparison == value)
        if matches.size != 1:
            raise ValueError(
                f"primary {label} coordinate {value!r} has {matches.size} exact "
                "matches in the terrain grid"
            )
        indices.append(int(matches[0]))
    result = np.asarray(indices, dtype=np.int64)
    if result.size > 1:
        differences = np.diff(result)
        contiguous = bool(np.all(differences == 1) or np.all(differences == -1))
        if (not contiguous and cyclic_degrees
                and result.size <= larger.size
                and _closed_longitude_circle(larger)):
            # A donor axis that closes the circle has its last and first
            # cells as neighbours, so a primary crossing that seam (a
            # European domain across 0 deg on a 0..360 donor, a Pacific
            # one across 180 on a -180..180 donor) is still a run of
            # adjacent cells.  At most one full turn: a longer request
            # would repeat a cell.
            wrapped = np.mod(differences, larger.size)
            contiguous = bool(np.all(wrapped == 1)
                              or np.all(wrapped == larger.size - 1))
        if not contiguous and allow_strided:
            contiguous = _regular_index_stride(
                result, larger.size,
                cyclic_degrees and _closed_longitude_circle(larger),
            ) is not None
        if not contiguous:
            kind = "regular strided" if allow_strided else "contiguous"
            raise ValueError(
                f"primary {label} is not a {kind} terrain-grid subset"
            )
    return result


def _regular_index_stride(indices: np.ndarray, size: int, cyclic: bool) -> int | None:
    """Exact regular selection, with no repeated cells or second circle."""
    if indices.size < 2:
        return 0
    differences = np.diff(indices)
    step = int(differences[0])
    if step != 0 and np.all(differences == step):
        return step
    if cyclic:
        wrapped = np.mod(differences, size)
        step = int(wrapped[0])
        if step > size // 2:
            step -= size
        if (step != 0 and np.all(wrapped == step % size)
                and abs(step) * (indices.size - 1) < size):
            return step
    return None


def _closed_longitude_circle(longitude: np.ndarray) -> bool:
    """Whether a donor longitude axis goes once round the globe, evenly.

    Its cells, taken in index order and wrapping from the last back to the
    first, must step the same way round the circle, one turn in all, with
    no step as long as 1.5 of the shortest: a regional axis fails on its
    wrap step (the gap it does not cover) and an axis missing a cell fails
    on the step that skips it.  Coordinates are still matched EXACTLY by
    the caller; this only decides whether the index sequence may wrap, so
    the 1.5 bound picks no cell and tolerates only the rounding of a
    decoded axis such as 0.15 deg.
    """
    folded = np.mod(np.asarray(longitude, dtype=np.float64), 360.0)
    if folded.size < 2:
        return False
    following = np.roll(folded, -1)
    for steps in (np.mod(following - folded, 360.0),
                  np.mod(folded - following, 360.0)):
        shortest = float(steps.min())
        if (shortest > 0.0 and float(steps.max()) < 1.5 * shortest
                and round(float(steps.sum()) / 360.0) == 1):
            return True
    return False


def _index_direction(indices: np.ndarray, size: int) -> int:
    """+1 or -1 for the way an index run steps, wrapped or not; 0 for one cell.

    An unwrapped run keeps the sign of last minus first, the value every
    receipt already carries; only a run across the seam needs the wrap.
    """
    if indices.size < 2:
        return 0
    differences = np.diff(indices)
    if np.all(differences == 1) or np.all(differences == -1):
        return int(np.sign(indices[-1] - indices[0]))
    return 1 if int(np.mod(differences[0], size)) == 1 else -1


#: The two DECLARED shapes a terrain supplement's clock can take.
#:
#: ``valid_time_exact`` requires a terrain frame at every primary valid
#: time -- the default, and the only accurate choice for a producer that
#: writes terrain into every step (HRRR's in-band terrain, 20CRv3's
#: per-time supplement).  ``cycle_invariant_broadcast`` is for the
#: producers that write their static fields only into the analysis frame
#: of a cycle (ECMWF open data, DWD's time-invariant objects, GDPS's
#: analysis statics): the terrain decodes wherever the supplement
#: supplies it, must STILL be invariant across every supplied frame, and
#: the one invariant field is carried to every primary valid time of the
#: ONE source cycle, with the carried times named in the receipt.  It is
#: the composition-side twin of the mapping grammar's
#: ``time_binding: cycle_invariant``.
_TERRAIN_TIME_ALIGNMENTS = frozenset({
    "valid_time_exact", "cycle_invariant_broadcast",
})


def _compose_terrain(
    primary: _DecodedCollection,
    terrain: _DecodedCollection,
    *,
    time_alignment: str = "valid_time_exact",
    grid_alignment: str = "exact_coordinate_subset",
) -> tuple[_DecodedCollection, dict[str, object]]:
    if time_alignment not in _TERRAIN_TIME_ALIGNMENTS:
        raise ValueError(
            "terrain supplement time_alignment must be one of "
            f"{sorted(_TERRAIN_TIME_ALIGNMENTS)}, got {time_alignment!r}"
        )
    if grid_alignment != "exact_coordinate_subset":
        raise ValueError("declared strided terrain selection requires the native Rust engine")
    primary_terrain = [key for key in primary.direct if key[2] == _EXTERNAL_FIELD]
    if primary_terrain:
        raise ValueError(
            "terrain has two providers: the primary source and the declared supplement"
        )
    terrain_inventory = {key[2] for key in terrain.direct}
    if terrain_inventory != {_EXTERNAL_FIELD}:
        raise ValueError(
            f"terrain supplement decoded unexpected fields {sorted(terrain_inventory)}"
        )
    latitude_indices = _exact_subset_indices(
        terrain.latitude, primary.latitude, "latitude"
    )
    longitude_indices = _exact_subset_indices(
        terrain.longitude, primary.longitude, "longitude", cyclic_degrees=True
    )
    terrain_items = sorted(
        terrain.direct.items(), key=lambda item: (item[0][0], str(item[0][1]))
    )
    if not terrain_items:
        raise ValueError("terrain supplement decoded no terrain messages")
    full_reference = terrain_items[0][1].values
    # equal_nan: a missing cell is data too, and NaN != NaN made one
    # unchanged record with a missing cell differ from itself.  A cell
    # that turns missing, or a finite value that moves, still differs.
    if any(not np.array_equal(full_reference, value.values, equal_nan=True)
           for _key, value in terrain_items[1:]):
        raise ValueError("terrain supplement changes across supplied valid times")
    primary_keys = tuple(sorted(
        primary.source_cycles, key=lambda item: (item[0], str(item[1]))
    ))
    terrain_by_time = {
        (time, member): value
        for (time, member, _field), value in terrain_items
    }
    missing_times = [key for key in primary_keys if key not in terrain_by_time]
    if missing_times and time_alignment == "valid_time_exact":
        raise ValueError(
            f"terrain supplement lacks exact primary valid time(s) {missing_times}"
        )
    if time_alignment != "valid_time_exact":
        # The producer publishes terrain at the analysis step only; the
        # supplement's one proven-invariant record answers every primary
        # valid time of the one source cycle.  One broadcast belongs to
        # one cycle: mixed primary cycles refuse rather than share an
        # invariant record no cycle proved for itself.
        primary_cycles = {
            str(cycle) for cycle in primary.source_cycles.values()
        }
        if len(primary_cycles) != 1:
            raise ValueError(
                "cycle-invariant terrain broadcast cannot span mixed "
                f"primary source cycles {sorted(primary_cycles)}"
            )
    # The invariance check above proved every supplied terrain frame is
    # equal, missing cells included, so under the broadcast alignment the
    # ONE invariant field
    # -- already the only field there is -- stands in at the primary valid
    # times the producer did not re-publish it for.  The carrier frame's
    # own metadata travels with it, and the receipt names each carried
    # time so a reader can see what was broadcast rather than matched.
    carrier = terrain_items[0][1]
    direct = dict(primary.direct)
    subset_reference = None
    for valid_time, member in primary_keys:
        supplied = terrain_by_time.get((valid_time, member), carrier)
        values = supplied.values[np.ix_(latitude_indices, longitude_indices)]
        if subset_reference is None:
            subset_reference = values
        elif not np.array_equal(subset_reference, values, equal_nan=True):
            raise ValueError("terrain subset changes across primary valid times")
        direct[(valid_time, member, _EXTERNAL_FIELD)] = _DirectValue(
            name=_EXTERNAL_FIELD,
            valid_time=valid_time,
            member=member,
            source_cycle=supplied.source_cycle,
            axes=supplied.axes,
            values=values,
            # The subset's own count, as the bound-field join below has
            # always taken it: the supplement's whole count also counted
            # missing cells outside the primary window, which the
            # canonical validator then refused as a count that did not
            # match the array.
            missing_count=int(np.isnan(values).sum()),
            references=supplied.references,
        )
    assert subset_reference is not None
    receipt = {
        "schema": "gpuwm-mapped-exact-subset-binding-v1",
        "status": "PASS",
        "field": _EXTERNAL_FIELD,
        "primary_shape": [int(primary.latitude.size), int(primary.longitude.size)],
        "supplement_shape": [int(terrain.latitude.size), int(terrain.longitude.size)],
        "latitude_index_range": [int(latitude_indices[0]), int(latitude_indices[-1])],
        "longitude_index_range": [int(longitude_indices[0]), int(longitude_indices[-1])],
        "latitude_sha256": _array_sha256(primary.latitude),
        "longitude_sha256": _array_sha256(primary.longitude),
        "terrain_full_sha256": _array_sha256(full_reference),
        "terrain_subset_sha256": _array_sha256(subset_reference),
        "supplement_valid_times": [key[0].isoformat() for key, _value in terrain_items],
        "matched_primary_valid_times": [key[0].isoformat() for key in primary_keys],
        # Recorded only when the composition declared the broadcast
        # alignment, so every receipt written before these keys existed is
        # byte-identical to what the same preparation writes today; under
        # the broadcast the receipt names the alignment AND each carried
        # time, so a reader can see what was broadcast rather than matched.
        **(
            {
                "time_alignment": time_alignment,
                "broadcast_primary_valid_times": [
                    key[0].isoformat() for key in missing_times],
            }
            if time_alignment != "valid_time_exact" else {}
        ),
        "invariant_across_all_supplement_times": True,
        "latitude_index_direction": int(np.sign(latitude_indices[-1] - latitude_indices[0])),
        "longitude_index_direction": _index_direction(
            longitude_indices, int(terrain.longitude.size)),
        "coordinate_match": "exact_equivalent_contiguous_subset",
        "longitude_equivalence": "modulo_360_exact",
    }
    return _DecodedCollection(
        latitude=primary.latitude,
        longitude=primary.longitude,
        vertical_values=primary.vertical_values,
        direct=MappingProxyType(direct),
        source_cycles=primary.source_cycles,
        grid_fingerprint=primary.grid_fingerprint,
        # The composed frame keeps the PRIMARY's vertical identity, its
        # hybrid coefficient ladder included: the borrow injects surface
        # and soil planes, never a vertical coordinate.
        hybrid_a=primary.hybrid_a,
        hybrid_b=primary.hybrid_b,
    ), receipt


def _bind_manifest_member(
    collection: _DecodedCollection,
    member: str,
) -> _DecodedCollection:
    """Stamp the manifest's EXPLICIT member onto every composed record.

    For archives whose product octets carry no ensemble identity: the
    caller's verified authority named the member, the sealed input
    manifest declares it, and this replaces ABSENT product-defined
    membership only.  Bytes that already encode a member contradict the
    declaration and refuse rather than being overwritten -- overwriting
    would let one archive's frames claim another member's identity.
    """

    encoded = {value.member for value in collection.direct.values()}
    if encoded != {None}:
        raise ValueError(
            f"the composition input manifest binds explicit member "
            f"{member!r}, but the decoded records already encode "
            f"product-defined member(s) "
            f"{sorted(str(value) for value in encoded - {None})}; an "
            "explicit member binding replaces ABSENT product membership "
            "only, so stamping it here would overwrite a producer-declared "
            "identity"
        )
    direct = {}
    for (valid_time, _old_member, field_name), value in collection.direct.items():
        direct[(valid_time, member, field_name)] = _dataclass_replace(
            value, member=member)
    cycles = {
        (valid_time, member): cycle
        for (valid_time, _old_member), cycle in collection.source_cycles.items()
    }
    return _DecodedCollection(
        latitude=collection.latitude,
        longitude=collection.longitude,
        vertical_values=collection.vertical_values,
        direct=MappingProxyType(direct),
        source_cycles=MappingProxyType(cycles),
        grid_fingerprint=collection.grid_fingerprint,
        hybrid_a=collection.hybrid_a,
        hybrid_b=collection.hybrid_b,
    )


def _binding_subset_indices(
    primary: _DecodedCollection,
    donor: _DecodedCollection,
    binding_name: str,
    grid_alignment: str = "exact_coordinate_subset",
) -> tuple[np.ndarray, np.ndarray]:
    if grid_alignment == "nearest_coordinate":
        # The Rust engine's own rule (join.rs binding_subset_indices), so the
        # reference composes what the engine composes: each primary
        # coordinate takes the donor's nearest published cell, longitude
        # cyclic, ties to the first published cell (argmin's first minimum
        # is min_by's).  Explicit nearest-cell sampling, as for categorical
        # surface analyses (a sea-ice analysis under an IFS atmosphere).
        def nearest(source, target, cyclic: bool) -> np.ndarray:
            source = np.asarray(source, dtype=np.float64)
            target = np.asarray(target, dtype=np.float64)
            if source.size == 0 or not np.isfinite(source).all()                     or not np.isfinite(target).all():
                raise ValueError("nearest-coordinate binding requires finite nonempty axes")
            indices = np.empty(target.size, dtype=np.intp)
            for position, value in enumerate(target):
                distance = np.abs(source - value)
                if cyclic:
                    distance = np.mod(distance, 360.0)
                    distance = np.minimum(distance, 360.0 - distance)
                indices[position] = int(np.argmin(distance))
            return indices

        return (nearest(donor.latitude, primary.latitude, False),
                nearest(donor.longitude, primary.longitude, True))
    allow_strided = grid_alignment == "exact_coordinate_strided_subset"
    try:
        latitude = _exact_subset_indices(
            donor.latitude, primary.latitude, "latitude",
            allow_strided=allow_strided,
        )
        longitude = _exact_subset_indices(
            donor.longitude, primary.longitude, "longitude",
            cyclic_degrees=True,
            allow_strided=allow_strided,
        )
    except ValueError as error:
        raise ValueError(
            f"contributing source binding {binding_name!r} is cross-grid "
            f"({error}); borrowing a field across grids requires a "
            "horizontal regrid capability this composition does not "
            "declare, so it refuses rather than interpolate"
        ) from error
    return latitude, longitude


def _take_subset(
    values: np.ndarray,
    axes: Sequence[str],
    latitude_indices: np.ndarray,
    longitude_indices: np.ndarray,
) -> np.ndarray:
    result = np.asarray(values)
    result = result.take(latitude_indices, axis=tuple(axes).index("y"))
    result = result.take(longitude_indices, axis=tuple(axes).index("x"))
    return result


def _compose_bound_fields(
    primary: _DecodedCollection,
    donor: _DecodedCollection,
    *,
    binding_name: str,
    binding: Mapping[str, object],
) -> tuple[_DecodedCollection, dict[str, object]]:
    """Inject one contributing source's bound fields into the primary decode.

    The donor collection is a partitioned decode of the contributing
    source's OWN mapping, so units, missing policy, and layer stacking are
    already the donor table's.  This join only aligns grids (exact
    coordinate subset or refuse naming regrid), aligns clocks by the
    binding's declared rule, refuses double provision, and keeps the
    donor's source cycle inside every injected value's provenance.
    """

    fields = tuple(str(name) for name in binding["fields"])
    companion = binding.get("companion")
    extended = (
        tuple(str(name) for name in companion["fields"])
        if companion is not None else ()
    )
    alignment = str(binding["time_alignment"])
    if alignment not in _BOUND_TIME_ALIGNMENTS:
        raise ValueError(
            f"binding time_alignment must be one of "
            f"{sorted(_BOUND_TIME_ALIGNMENTS)}, got {alignment!r}"
        )
    donor_members = {member for _time, member in donor.source_cycles}
    if len(donor_members) != 1:
        raise ValueError(
            f"contributing source binding {binding_name!r} decoded "
            f"{len(donor_members)} members; cross-source borrowing has no "
            "member-alignment capability and requires a single-member donor"
        )
    donor_member = next(iter(donor_members))
    primary_members = sorted(
        {member for _time, member in primary.source_cycles},
        key=lambda value: (value is not None, str(value)),
    )
    # Two product-declared members that differ are two trajectories.  A
    # donor that declares no member (a deterministic run or an analysis)
    # lends land state under any member, as the hybrids always have; a
    # member lending to ANOTHER member is refused unless it is the
    # declared companion, whose receipt names both.
    other = [member for member in primary_members
             if member is not None and member != donor_member]
    if companion is None and donor_member is not None and other:
        raise ValueError(
            f"contributing source binding {binding_name!r} decoded member "
            f"{donor_member!r} under primary member {other[0]!r}; borrowing "
            "another member's fields would mix two trajectories in one "
            "frame, which only a declared companion may do"
        )
    donor_inventory = {key[2] for key in donor.direct}
    if donor_inventory != set(fields) | set(extended):
        raise ValueError(
            f"contributing source binding {binding_name!r} decoded fields "
            f"{sorted(donor_inventory)} instead of the bound "
            f"{sorted(set(fields) | set(extended))}"
        )
    provided_twice = sorted(
        {key[2] for key in primary.direct if key[2] in set(fields)}
    )
    if provided_twice:
        raise ValueError(
            f"field {provided_twice[0]!r} has two providers: the primary "
            f"decode and contributing source binding {binding_name!r}"
        )
    if any("vertical" in value.axes for value in donor.direct.values()
           if value.name not in extended) \
            and not np.array_equal(donor.vertical_values, primary.vertical_values):
        raise ValueError(
            f"contributing source binding {binding_name!r} borrows a "
            "vertical-bearing field on a different vertical ladder; "
            "cross-ladder borrowing requires a vertical interpolation "
            "capability this composition does not declare"
        )
    latitude_indices, longitude_indices = _binding_subset_indices(
        primary, donor, binding_name, str(binding["grid_alignment"]),
    )
    primary_keys = tuple(sorted(
        primary.source_cycles, key=lambda item: (item[0], str(item[1]))
    ))
    primary_cycles = sorted({
        cycle for cycle in primary.source_cycles.values()
    })
    by_field: dict[str, dict[datetime, _DirectValue]] = {}
    for (time_value, _member, name), value in donor.direct.items():
        by_field.setdefault(name, {})[time_value] = value

    direct = dict(primary.direct)
    matched_times: set[datetime] = set()
    broadcast_times: set[datetime] = set()
    subset_hashes: dict[str, str] = {}
    for name in fields:
        supplied = by_field[name]
        if alignment == "valid_time_exact":
            missing = [
                time for time, _member in primary_keys if time not in supplied
            ]
            if missing:
                raise ValueError(
                    f"contributing source binding {binding_name!r} lacks "
                    f"{name!r} at primary valid time(s) {missing}"
                )
            carrier = None
        elif alignment == "source_cycle_analysis_select_broadcast":
            raise ValueError("cycle analysis selection requires the Rust mapped engine")
        elif alignment == "source_cycle_window_start_broadcast":
            if len(primary_cycles) != 1:
                raise ValueError(
                    "source-cycle window-start broadcast cannot span mixed "
                    f"primary source cycles {primary_cycles}"
                )
            first = min(time for time, _member in primary_keys)
            if len(supplied) != 1:
                raise ValueError(
                    f"contributing source binding {binding_name!r} supplies "
                    f"{name!r} at {len(supplied)} valid times; the window-start "
                    "broadcast requires exactly one record, valid at the run's start"
                )
            valid_time, carrier = next(iter(supplied.items()))
            # The run's start is no later than this call's first primary
            # time (a remaining-forcing pass starts one lead in) and no
            # earlier than the primary cycle.
            if (not primary_cycles[0] <= valid_time <= first
                    or carrier.source_cycle != primary_cycles[0]):
                raise ValueError(
                    f"contributing source binding {binding_name!r} supplies "
                    f"{name!r} valid at {valid_time.isoformat()} from cycle "
                    f"{carrier.source_cycle.isoformat()}; the window-start "
                    f"broadcast needs the primary cycle "
                    f"{primary_cycles[0].isoformat()}'s own record valid "
                    f"between that cycle and the first primary time "
                    f"{first.isoformat()}"
                )
        elif alignment == "source_cycle_analysis_broadcast":
            if len(primary_cycles) != 1:
                raise ValueError(
                    "source-cycle analysis broadcast cannot span mixed "
                    f"primary source cycles {primary_cycles}"
                )
            if len(supplied) != 1:
                raise ValueError(
                    f"contributing source binding {binding_name!r} supplies "
                    f"{name!r} at {len(supplied)} valid times; the analysis "
                    "broadcast requires exactly one analysis record"
                )
            analysis_time = next(iter(supplied))
            if analysis_time != primary_cycles[0]:
                raise ValueError(
                    f"contributing source binding {binding_name!r} supplies "
                    f"{name!r} at {analysis_time.isoformat()}, which is not "
                    f"the primary source cycle "
                    f"{primary_cycles[0].isoformat()}; a hybrid borrows its "
                    "initialization state from the SAME cycle's analysis"
                )
            carrier = supplied[analysis_time]
        else:  # cycle_invariant_broadcast
            if len(primary_cycles) != 1:
                raise ValueError(
                    "cycle-invariant broadcast cannot span mixed primary "
                    f"source cycles {primary_cycles}"
                )
            ordered = [supplied[time] for time in sorted(supplied)]
            reference = ordered[0]
            if any(not np.array_equal(reference.values, value.values,
                                      equal_nan=True)
                   for value in ordered[1:]):
                raise ValueError(
                    f"contributing source binding {binding_name!r} field "
                    f"{name!r} changes across supplied valid times; the "
                    "cycle-invariant broadcast is for proven statics only"
                )
            carrier = reference
        subset_reference = None
        for valid_time, member in primary_keys:
            value = supplied.get(valid_time)
            if value is None:
                value = carrier
                broadcast_times.add(valid_time)
            else:
                matched_times.add(valid_time)
            values = _take_subset(
                value.values, value.axes, latitude_indices, longitude_indices,
            )
            if subset_reference is None:
                subset_reference = values
                subset_hashes[name] = _array_sha256(values)
            direct[(valid_time, member, name)] = _DirectValue(
                name=name,
                valid_time=valid_time,
                member=member,
                source_cycle=value.source_cycle,
                axes=value.axes,
                values=values,
                missing_count=int(np.isnan(values).sum()),
                references=value.references,
            )
    donor_cycles = sorted({
        value.source_cycle for value in donor.direct.values()
    })
    receipt = {
        "schema": BINDING_RECEIPT_SCHEMA,
        "status": "PASS",
        "binding": binding_name,
        "source_id": str(binding["source_id"]),
        "fields": sorted(fields),
        "grid_alignment": str(binding["grid_alignment"]),
        "coordinate_match": (
            "nearest_published_cell"
            if binding["grid_alignment"] == "nearest_coordinate"
            else "exact_equivalent_regular_strided_subset"
            if binding["grid_alignment"] == "exact_coordinate_strided_subset"
            else "exact_equivalent_contiguous_subset"
        ),
        "longitude_equivalence": "modulo_360_exact",
        "primary_shape": [int(primary.latitude.size), int(primary.longitude.size)],
        "donor_shape": [int(donor.latitude.size), int(donor.longitude.size)],
        "latitude_index_range": [
            int(latitude_indices[0]), int(latitude_indices[-1])],
        "longitude_index_range": [
            int(longitude_indices[0]), int(longitude_indices[-1])],
        "time_alignment": alignment,
        "donor_member": donor_member,
        "donor_source_cycles": [cycle.isoformat() for cycle in donor_cycles],
        "donor_valid_times": sorted(
            {key[0].isoformat() for key in donor.direct}
        ),
        "matched_primary_valid_times": sorted(
            time.isoformat() for time in matched_times
        ),
        "broadcast_primary_valid_times": sorted(
            time.isoformat() for time in broadcast_times
        ),
        "field_subset_sha256": subset_hashes,
    }
    if binding["grid_alignment"] == "exact_coordinate_strided_subset":
        receipt.update({
            "latitude_indices": latitude_indices.tolist(),
            "longitude_indices": longitude_indices.tolist(),
            "latitude_index_stride": _regular_index_stride(
                latitude_indices, donor.latitude.size, False),
            "longitude_index_stride": _regular_index_stride(
                longitude_indices, donor.longitude.size,
                _closed_longitude_circle(donor.longitude)),
        })
    vertical_values = primary.vertical_values
    if companion is not None:
        vertical_values, receipt["companion"] = _extend_above_top(
            primary, donor, direct,
            binding_name=binding_name,
            fields=extended,
            extension=str(companion["vertical_extension"]),
            primary_keys=primary_keys,
            primary_members=primary_members,
            donor_member=donor_member,
            latitude_indices=latitude_indices,
            longitude_indices=longitude_indices,
        )
    return _DecodedCollection(
        latitude=primary.latitude,
        longitude=primary.longitude,
        vertical_values=vertical_values,
        direct=MappingProxyType(direct),
        source_cycles=primary.source_cycles,
        grid_fingerprint=primary.grid_fingerprint,
        # The composed frame keeps the PRIMARY's vertical identity, its
        # hybrid coefficient ladder included: the borrow injects surface
        # and soil planes, never a vertical coordinate.  A declared
        # companion is the one exception, and only on a pressure ladder:
        # it stacks its levels above the primary's top.
        hybrid_a=primary.hybrid_a,
        hybrid_b=primary.hybrid_b,
    ), receipt


def companion_ladder(
    primary_levels: np.ndarray,
    companion_levels: np.ndarray,
    *,
    binding_name: str,
) -> tuple[np.ndarray, list[int]]:
    """The composed ladder and the companion level indices above the top.

    On a pressure ladder "above the top" is every companion level whose
    pressure is lower than the primary's lowest; they are stacked in the
    primary's own order (top first for an ascending ladder), and every
    primary level stays the primary's.  Refuses when there is nothing above
    the top, because the receipt would then claim an extension that did
    not happen.
    """

    primary_levels = np.asarray(primary_levels, dtype=np.float64)
    companion_levels = np.asarray(companion_levels, dtype=np.float64)
    if primary_levels.size == 0:
        raise ValueError(
            f"contributing source binding {binding_name!r} declares a "
            "companion, but the primary decode carries no vertical ladder "
            "to extend"
        )
    top = float(primary_levels.min())
    above = [index for index, level in enumerate(companion_levels)
             if float(level) < top]
    if not above:
        raise ValueError(
            f"contributing source binding {binding_name!r} declares a "
            "companion that extends the ladder above the primary's top "
            f"({top!r}), but the companion carries no level above it"
        )
    ascending = primary_levels.size < 2 \
        or float(primary_levels[0]) < float(primary_levels[-1])
    above.sort(key=lambda index: float(companion_levels[index]),
               reverse=not ascending)
    stacked = companion_levels[above]
    composed = (np.concatenate([stacked, primary_levels]) if ascending
                else np.concatenate([primary_levels, stacked]))
    return composed, above


def _extend_above_top(
    primary: _DecodedCollection,
    donor: _DecodedCollection,
    direct: dict,
    *,
    binding_name: str,
    fields: Sequence[str],
    extension: str,
    primary_keys,
    primary_members,
    donor_member,
    latitude_indices: np.ndarray,
    longitude_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Stack a declared companion's levels above the primary's top.

    ``direct`` (the composed field table, already a copy) is updated in
    place: each extended field at each primary valid time becomes the
    companion's levels above the primary's top, cut to the primary grid,
    joined to the primary's own column along the vertical axis.  The
    returned receipt names exactly which levels of which field came from
    which source, and both members, so the mix is never silent.
    """

    if extension not in _COMPANION_VERTICAL_EXTENSIONS:
        raise ValueError(
            f"companion vertical_extension must be one of "
            f"{sorted(_COMPANION_VERTICAL_EXTENSIONS)}, got {extension!r}"
        )
    composed, above = companion_ladder(
        primary.vertical_values, donor.vertical_values,
        binding_name=binding_name,
    )
    ascending = primary.vertical_values.size < 2 or float(
        primary.vertical_values[0]) < float(primary.vertical_values[-1])
    stacked_short = sorted({
        key[2] for key, value in primary.direct.items()
        if "vertical" in value.axes and key[2] not in set(fields)
    })
    if stacked_short:
        raise ValueError(
            f"contributing source binding {binding_name!r} extends "
            f"{sorted(fields)} above the primary's top, but the primary also "
            f"decoded {stacked_short} on its ladder; those columns would stop "
            "at the primary's top"
        )
    above_array = np.asarray(above, dtype=np.intp)
    slab_hashes: dict[str, str] = {}
    for name in fields:
        supplied = {
            key[0]: value for key, value in donor.direct.items()
            if key[2] == name
        }
        for valid_time, member in primary_keys:
            own = primary.direct.get((valid_time, member, name))
            if own is None:
                raise ValueError(
                    f"contributing source binding {binding_name!r} extends "
                    f"{name!r}, which the primary decode does not carry at "
                    f"{valid_time.isoformat()}"
                )
            value = supplied.get(valid_time)
            if value is None:
                raise ValueError(
                    f"contributing source binding {binding_name!r} lacks "
                    f"{name!r} at primary valid time(s) [{valid_time!r}]"
                )
            cycle = primary.source_cycles[(valid_time, member)]
            if value.source_cycle != cycle:
                raise ValueError(
                    f"contributing source binding {binding_name!r} is a "
                    f"companion from cycle {value.source_cycle.isoformat()} "
                    f"under primary cycle {cycle.isoformat()}; a companion "
                    "from another cycle would put a different forecast on "
                    "top of this one's column"
                )
            if tuple(value.axes) != tuple(own.axes):
                raise ValueError(
                    f"contributing source binding {binding_name!r} field "
                    f"{name!r} has axes {list(value.axes)} where the primary "
                    f"has {list(own.axes)}; a column is joined on one axis "
                    "order"
                )
            axis = tuple(own.axes).index("vertical")
            slab = _take_subset(
                value.values, value.axes, latitude_indices, longitude_indices,
            ).take(above_array, axis=axis)
            if name not in slab_hashes:
                slab_hashes[name] = _array_sha256(slab)
            parts = [slab, own.values] if ascending else [own.values, slab]
            values = np.concatenate(parts, axis=axis)
            direct[(valid_time, member, name)] = _DirectValue(
                name=name,
                valid_time=valid_time,
                member=member,
                source_cycle=own.source_cycle,
                axes=own.axes,
                values=values,
                missing_count=int(np.isnan(values).sum()),
                references=tuple(own.references) + tuple(value.references),
            )
    real_members = [member for member in primary_members if member is not None]
    receipt = {
        "vertical_extension": extension,
        "fields": sorted(fields),
        "primary_member": real_members[0] if len(real_members) == 1 else None,
        "companion_member": donor_member,
        "primary_levels": [float(level) for level in primary.vertical_values],
        "companion_levels": [float(level) for level in donor.vertical_values[above_array]],
        "composed_levels": [float(level) for level in composed],
        "companion_slab_sha256": slab_hashes,
    }
    return composed, receipt


@dataclass(frozen=True)
class MappedSourceBundle:
    frames: tuple[MappedSourceFrame, ...]
    mapping_path: Path
    mapping_sha256: str
    composition_path: Path
    composition_sha256: str
    input_manifest_path: Path
    input_manifest_sha256: str
    decoder_paths: Mapping[str, Path]
    decoder_sha256: Mapping[str, str]
    terrain_data_paths: tuple[Path, ...]
    terrain_data_sha256: tuple[str, ...]
    terrain_provenance_path: Path
    terrain_provenance_sha256: str
    soil_layer_contract: Mapping[str, object]
    alignment_receipt: Mapping[str, object]
    #: One record per cross-source binding: contributing source id, its own
    #: mapping authority hash, its data and provenance hashes, and the
    #: grid/time alignment receipt.  Empty for single-source compositions.
    contributing_sources: tuple[Mapping[str, object], ...] = dataclass_field(
        default=(),
    )
    #: Primary records read through ``mapping.record_aliases``, counted
    #: per field they answer; ``None`` when none were.
    record_aliases: Mapping[str, int] | None = None
    #: Runtime allocation telemetry, excluded from composition/cache identity.
    native_decode_memory: Mapping[str, object] | None = dataclass_field(default=None, compare=False)
    _native_decode_control: dict | None = dataclass_field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        # A streamed frameset is NOT tupled here: `tuple()` would read
        # every valid time's arrays to hold them, which is the residency
        # this route exists to avoid.  It answers the same two questions
        # from its own document instead -- which fields a frame carries,
        # and each field's declared digest -- and the digest is the one
        # the frame is checked against when it is actually read, so a
        # terrain array that does not match its declaration refuses
        # before the preparation publishes anything.
        frames = self.frames
        summary = _frame_summary(frames)
        if not len(frames) or any(
                _EXTERNAL_FIELD not in names for _time, names, _digest in summary):
            raise ValueError("composed source bundle lacks terrain in a canonical frame")
        terrain_hashes = {digest for _time, _names, digest in summary}
        if len(terrain_hashes) != 1:
            raise ValueError("composed canonical terrain changes across forcing times")
        if not isinstance(frames, _ABCSequence):
            frames = tuple(frames)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(
            self, "alignment_receipt", MappingProxyType(dict(self.alignment_receipt))
        )
        object.__setattr__(
            self, "contributing_sources", tuple(
                MappingProxyType(dict(record))
                for record in self.contributing_sources
            )
        )
        object.__setattr__(
            self, "soil_layer_contract", MappingProxyType(dict(self.soil_layer_contract))
        )
        decoder_paths = MappingProxyType({
            str(role): Path(path).resolve()
            for role, path in self.decoder_paths.items()
        })
        decoder_sha256 = MappingProxyType({
            str(role): str(digest) for role, digest in self.decoder_sha256.items()
        })
        if set(decoder_paths) != set(decoder_sha256):
            raise ValueError("decoder path and SHA inventories differ")
        object.__setattr__(self, "decoder_paths", decoder_paths)
        object.__setattr__(self, "decoder_sha256", decoder_sha256)
        terrain_paths = tuple(Path(path).resolve() for path in self.terrain_data_paths)
        terrain_sha = tuple(str(digest) for digest in self.terrain_data_sha256)
        if not terrain_paths or len(terrain_paths) != len(terrain_sha):
            raise ValueError("terrain data path and SHA inventories differ")
        object.__setattr__(self, "terrain_data_paths", terrain_paths)
        object.__setattr__(self, "terrain_data_sha256", terrain_sha)

    def regular_snapshots(self):
        """Pack the WRF-real ABI with explicit soil and zero policies.

        ONE valid time at a time.  Packing them all up front allocated
        five zero-filled hydrometeor arrays PER TIME the size of the
        source pressure field -- on a 3 km CONUS source with a 45-level
        ladder that is 3.2 GiB per valid time of nothing but zeros,
        held from the first time to the last, beside the frames.  The
        route that consumes these takes one snapshot, interpolates it,
        keeps its perimeter and drops it (see
        ``start_last_forcing_order``), so nothing ever needed two.

        The returned object indexes, iterates and slices like the tuple
        it replaces, and each snapshot it yields is element-for-element
        the one the all-at-once packing produced.
        """

        return _RegularSnapshots(self)

    def close(self) -> None:
        """Release the decode scratch these frames stream from, if any."""

        release = getattr(self.frames, "close", None)
        if release is not None:
            release()

    def future_decode_host_bytes(self) -> int | None:
        control = self._native_decode_control
        if control is None or not control.get("pending"):
            return 0
        price = self.native_decode_memory
        if not isinstance(price, Mapping) or not price.get("budget_supported"):
            return None
        peak = price.get("one_time_peak_bytes")
        return peak if isinstance(peak, int) and not isinstance(peak, bool) and peak > 0 else None

    def limit_future_decode(self, budget_bytes: int) -> None:
        """Cap an initially uncapped reader's later whole-grid fallback."""
        if self._native_decode_control is not None:
            if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 1:
                raise ValueError("future decode budget must be positive integer bytes")
            self._native_decode_control["budget_bytes"] = budget_bytes


def _frame_summary(frames) -> tuple[tuple[object, tuple[str, ...], str], ...]:
    """``(valid_time, field names, terrain digest)`` per frame, cheaply.

    A streamed frameset answers all three from its own document without
    reading an array; a materialized tuple of frames answers them from
    the frames it already holds.  One shape, so the invariants above
    read the same either way.
    """

    digest = getattr(frames, "field_digest", None)
    if digest is None:
        return tuple(
            (frame.valid_time, tuple(frame.fields),
             _array_sha256(frame.fields[_EXTERNAL_FIELD].values)
             if _EXTERNAL_FIELD in frame.fields else "")
            for frame in frames)
    times = frames.valid_times
    summary = []
    for index in range(len(frames)):
        names = frames.field_names(index)
        summary.append((
            times[index], names,
            digest(index, _EXTERNAL_FIELD) if _EXTERNAL_FIELD in names else ""))
    return tuple(summary)


class _RegularSnapshots(_ABCSequence):
    """The regular-source ABI of a composed bundle, one time at a time.

    Named breakage: materializing every valid time's snapshot at once
    made a preparation's host memory scale with the number of forcing
    times -- both the frames the snapshots read and the five zero
    hydrometeor arrays each snapshot allocates.  Measured on real 3 km
    CONUS bytes, that is what took a seven-time preparation past 100 GiB
    and killed it on smaller boxes.

    Indexing is one-deep cached, so a caller that reads the same index
    twice in a row pays once; asking for a different index drops the
    previous snapshot BEFORE reading the next, so the peak is one.

    Once the target grids are known (:meth:`for_grids`), a whole-globe
    source ring is re-cut opposite them, in the snapshots AND in the
    geometry :meth:`snapshot_metadata` states, by the rule the GFS route
    applies (:func:`woof.ingest.horiz.orient_global_source_longitudes`).
    Named breakage: without it the ring kept the -180/180 cut its decode
    stores, so every domain touching the antimeridian was refused as
    "outside the source grid" by every mapped global source while the
    GFS route built the same domains.  A ring whose cut is already clear
    of every stencil is left untouched, so off-seam domains are unchanged.
    """

    def __init__(self, bundle: "MappedSourceBundle",
                 order: tuple[int, ...] | None = None, *, target_grids=()):
        self._bundle = bundle
        self._target_grids = tuple(target_grids)
        self._order = (
            tuple(range(len(bundle.frames))) if order is None else tuple(order))
        land_policy = bundle.soil_layer_contract["missing"]["land"]
        self._soil_land_repair = (
            land_policy if isinstance(land_policy, Mapping) else None)
        self._cached_position: int | None = None
        self._cached_snapshot = None
        self._target_longitudes: tuple[np.ndarray, ...] | None = None
        self._ring_cuts: dict[tuple[bytes, bytes], object] = {}

    def _ring_cut(self, latitude, longitude, projection):
        """The re-cut this sequence's targets need on ``longitude``, or ``None``."""

        if not self._target_grids or projection is not None:
            return None
        from woof.ingest.horiz import global_source_cut

        key = (np.ascontiguousarray(latitude, dtype=np.float64).tobytes(),
               np.ascontiguousarray(longitude, dtype=np.float64).tobytes())
        if key not in self._ring_cuts:
            if self._target_longitudes is None:
                self._target_longitudes = tuple(
                    pair[1] for grid in self._target_grids
                    for pair in (grid.latlon_mass(), grid.latlon_u(),
                                 grid.latlon_v()))
            self._ring_cuts[key] = global_source_cut(
                latitude, longitude, *self._target_longitudes,
                target_latitudes=tuple(
                    pair[0] for grid in self._target_grids
                    for pair in (grid.latlon_mass(), grid.latlon_u(),
                                 grid.latlon_v())))
        return self._ring_cuts[key]

    def __len__(self) -> int:
        return len(self._order)

    @property
    def valid_times(self) -> tuple[object, ...]:
        """Every snapshot's valid time, without packing one."""

        frames = self._bundle.frames
        times = getattr(frames, "valid_times", None)
        if times is None:
            times = tuple(frame.valid_time for frame in frames)
        return tuple(times[index] for index in self._order)

    def sorted_by_valid_time(self) -> "_RegularSnapshots":
        """The same snapshots in valid-time order."""

        times = self.valid_times
        order = sorted(range(len(self._order)), key=lambda index: times[index])
        return _RegularSnapshots(
            self._bundle, tuple(self._order[index] for index in order),
            target_grids=self._target_grids)

    def for_grids(self, grids):
        """Keep full donor fields and pack only exact atmospheric support."""
        return _RegularSnapshots(self._bundle, self._order, target_grids=grids)

    def select(self, positions):
        """The snapshots at ``positions`` of this sequence, in that order."""
        return _RegularSnapshots(
            self._bundle, tuple(self._order[int(position)] for position in positions),
            target_grids=self._target_grids)

    def snapshot_metadata(self, position: int):
        """Describe the fixed snapshot ABI without copying its field arrays."""
        from woof.ingest.source_metadata import SourceSnapshotMetadata
        from woof.source_frame import validate_source_frame

        index = self._order[int(position)]
        frames = self._bundle.frames
        coordinates = getattr(frames, "coordinates", None)
        if coordinates is None:
            frame = frames[index]
            latitude, longitude = frame.latitude, frame.longitude
            header = frame.header
        else:
            latitude, longitude = coordinates(index)
            header = frames.header(index)
        validate_source_frame(header)
        grid = header.grid
        projection = (None if grid.projection == "regular_latitude_longitude"
                      else {"family": grid.projection,
                            "parameters": dict(grid.parameters)})
        cut = self._ring_cut(latitude, longitude, projection)
        if cut is not None:
            from woof.ingest.horiz import polar_donor_latitude
            latitude, _, _ = polar_donor_latitude(latitude, cut)
            longitude = cut.longitude
        # _pack always constructs Era5Snapshot, for every mapped input format.
        # Full field validation still happens in that constructor and frameset.
        return SourceSnapshotMetadata(
            Era5Snapshot, latitude, longitude, projection)

    def source_pressure_hpa(self, position: int):
        """The level ladder of one snapshot, read from pressure alone.

        The vertical-coverage check needs the source column's top and
        nothing else.  Reading the whole valid time for it would be
        twenty arrays where one answers, so a streamed frameset is asked
        for the pressure field on its own.
        """

        index = self._order[int(position)]
        frames = self._bundle.frames
        pressure_levels = getattr(frames, "pressure_levels_hpa", None)
        if self._target_grids and pressure_levels is not None:
            return pressure_levels(index)
        field = getattr(frames, "field", None)
        if field is None:
            pressure = frames[index].fields["air_pressure"].values
        else:
            pressure = field(index, "air_pressure").values
        return np.median(np.asarray(pressure), axis=(1, 2)) / 100.0

    def prefetch_through(self, positions) -> None:
        """Ask a batching frameset to decode every time ``positions`` name at once."""

        hook = getattr(self._bundle.frames, "prefetch_through", None)
        indices = [self._order[int(position)] for position in positions]
        if hook is not None and indices:
            hook(max(indices))

    def _leading_positions(self, reader_name: str) -> int:
        reader = getattr(self._bundle.frames, reader_name, None)
        if reader is None:
            return len(self)
        ready = int(reader())
        count = 0
        while count < len(self) and self._order[count] < ready:
            count += 1
        return count

    def decoded_positions(self) -> int:
        """How many leading snapshots read with no decode and no wait (all, unless as posted)."""

        return self._leading_positions("decoded_through")

    def posted_positions(self) -> int:
        """How many leading snapshots the source has posted now (all, unless as posted).

        A live window's later leads are not posted yet; an archived window's
        are all posted, whether or not the fetch has moved them yet.
        """

        return self._leading_positions("posted_through")

    def surface_field_values(self, position: int, legacy: str):
        """One surface field of one snapshot, as :meth:`_pack` packs it, or ``None``.

        ``_pack`` builds a whole valid time -- every field read, verified and
        joined -- and the daily-mean inland air temperature needs one 2 m
        field from each.  On the 18 h HRRR CONUS start that was 19 whole
        packs, 30 s on one core in front of the start time's build.  A field
        the frameset stores under its canonical name is read and verified
        alone here and converted exactly as the pack converts it
        (``np.asarray(values, dtype=np.float64)``).  A latitude-longitude
        source may be re-cut around the globe by the pack, so it, and any
        frameset without single-field reads, answers ``None`` and the
        caller packs the snapshot as before.
        """

        from woof.mapped_source import REGULAR_SNAPSHOT_LEGACY_NAMES

        canonical = next((name for name, value in REGULAR_SNAPSHOT_LEGACY_NAMES.items()
                          if value == legacy), None)
        frames = self._bundle.frames
        field = getattr(frames, "field", None)
        header = getattr(frames, "header", None)
        if canonical is None or field is None or header is None:
            return None
        index = self._order[int(position)]
        if header(index).grid.projection == "regular_latitude_longitude":
            return None
        try:
            values = field(index, canonical).values
        except KeyError:
            return None
        return np.asarray(values, dtype=np.float64)

    def _pack(self, index: int, *, full=False):
        fieldwise = getattr(self._bundle.frames, "fieldwise_frame", None)
        frame = (fieldwise(index) if fieldwise is not None else
                 self._bundle.frames[index])
        options = {}
        if self._target_grids and not full:
            from woof.ingest.atmospheric_window import atmospheric_window_for_grids
            from woof.ingest.source_metadata import SourceSnapshotMetadata
            source_grid = frame.header.grid
            projection = (None if source_grid.projection == "regular_latitude_longitude"
                          else {"family": source_grid.projection,
                                "parameters": dict(source_grid.parameters)})
            window = atmospheric_window_for_grids(SourceSnapshotMetadata(
                Era5Snapshot, frame.latitude, frame.longitude, projection), self._target_grids)
            if window is not None:
                window_provider = getattr(frame, "with_atmospheric_window", None)
                if window_provider is not None:
                    frame = window_provider(window)
                # The fallback retains the source, not this sequence's
                # one-snapshot cache. Capturing self would create a cycle
                # that kept the last atmosphere alive until cyclic GC.
                fallback = _RegularSnapshots(self._bundle)
                options = {"atmospheric_window": window,
                           "full_snapshot_factory": lambda: fallback._pack(index, full=True)}
        snapshot = mapped_frames_to_regular_snapshots(
            (frame,), soil_land_repair=self._soil_land_repair,
            initialize_absent_hydrometeors=True, **options)[0]
        if "SOURCE_OROGRAPHY" not in snapshot.fields:
            raise ValueError(
                "canonical terrain did not reach the regular-source ABI")
        if MAPPED_SOIL_TEMPERATURE not in snapshot.fields \
                or MAPPED_SOIL_MOISTURE not in snapshot.fields:
            raise ValueError("canonical mapped soil arrays are absent")
        cut = self._ring_cut(snapshot.latitude, snapshot.longitude, snapshot.projection)
        if cut is not None:
            from woof.ingest.horiz import recut_global_ring

            snapshot = recut_global_ring(snapshot, cut)
        # Materialized providers may still cache a complete canonical frame.
        # The snapshot owns its copies; that cache has no packing consumer.
        release = getattr(self._bundle.frames, "release_frame", None)
        if release is not None:
            release(frame)
        return snapshot

    def pack_uncached(self, position):
        """The snapshot ``self[position]`` returns, packed without the one-deep cache.

        For builds that run side by side (the forcing-time workers): each
        call packs its own copy from the frames, whose field reads open their
        own handles, so concurrent calls share no cache. Element for element
        the snapshot indexing returns; the peak is one snapshot per caller.
        """
        value = int(position)
        if value < 0:
            value += len(self._order)
        if not 0 <= value < len(self._order):
            raise IndexError(position)
        return self._pack(self._order[value])

    def __getitem__(self, position):
        if isinstance(position, slice):
            return _RegularSnapshots(
                self._bundle,
                tuple(self._order[index]
                      for index in range(*position.indices(len(self._order)))),
                target_grids=self._target_grids)
        value = int(position)
        if value < 0:
            value += len(self._order)
        if not 0 <= value < len(self._order):
            raise IndexError(position)
        if self._cached_position == value:
            return self._cached_snapshot
        self._cached_position = None
        self._cached_snapshot = None
        snapshot = self._pack(self._order[value])
        self._cached_position = value
        self._cached_snapshot = snapshot
        return snapshot


def _decoder_inventory(
    source_format: str | Sequence[str],
    *,
    grib1_bridge: str | Path | None,
    grib2_inventory: str | Path | None,
    grib2_dump: str | Path | None,
    engine: str | Path | None = None,
) -> dict[str, Path]:
    """Exact decoder roles for one source format, or a composed set of them.

    A cross-source composition may decode two formats in one preparation
    (a GRIB2 primary borrowing from a GRIB1 archive); the expected role set
    is then the union of each format's roles.

    On the Rust engine the inventory is ONE role -- the engine itself --
    whatever formats the composition spans, because the engine decodes
    GRIB1, GRIB2 and NetCDF in process and no per-format subprocess tool
    runs.  The manifest that seals a preparation binds the binary that
    actually read the bytes; a manifest sealed against the subprocess
    tools therefore refuses on this route rather than being replayed
    under a decoder that never ran (:func:`_verify_manifest` compares
    the role sets, and :func:`_engine_manifest_refusal` explains it).
    """

    if engine is not None:
        from woof.mapped_engine_bridge import ENGINE_NAME

        provided = {
            role: Path(path).resolve()
            for role, path in {
                "grib1_bridge": grib1_bridge,
                "grib2_inventory": grib2_inventory,
                "grib2_dump": grib2_dump,
            }.items()
            if path is not None
        }
        if provided:
            raise ValueError(
                f"the {ENGINE_NAME} route decodes in process; it cannot "
                f"also run the subprocess tools {sorted(provided)}")
        return {ENGINE_NAME: Path(engine).resolve()}

    formats = (
        (source_format,) if isinstance(source_format, str)
        else tuple(source_format)
    )
    provided = {
        role: Path(path).resolve()
        for role, path in {
            "grib1_bridge": grib1_bridge,
            "grib2_inventory": grib2_inventory,
            "grib2_dump": grib2_dump,
        }.items()
        if path is not None
    }
    per_format = {
        "grib1": {"grib1_bridge"},
        "grib2": {"grib2_inventory", "grib2_dump"},
        "netcdf": set(),
    }
    expected: set[str] = set()
    for name in formats:
        expected |= per_format[name]
    label = "+".join(sorted(set(formats)))
    if set(provided) != expected:
        raise ValueError(
            f"{label} decoder inventory differs from the contract; "
            f"missing={sorted(expected - set(provided))}, "
            f"extra={sorted(set(provided) - expected)}"
        )
    return provided


def _partition_mapping(
    mapping: Mapping[str, object], *, terrain_only: bool,
) -> dict[str, object]:
    """Return a decoder-only mapping for one side of the composition.

    The sealed, fully validated mapping remains the sole semantic authority.
    Partitioning only prevents a decoder from demanding a field that the
    composition contract explicitly assigns to the other product.
    """

    partition = copy.deepcopy(dict(mapping))
    fields = partition["fields"]
    assert isinstance(fields, dict)
    partition["fields"] = {
        name: field for name, field in fields.items()
        if (name == _EXTERNAL_FIELD) == terrain_only
        and field.get("provider") != "composition_bound"
    }
    if terrain_only and set(partition["fields"]) != {_EXTERNAL_FIELD}:
        raise ValueError("mapping lacks the direct terrain field")
    if not terrain_only and not partition["fields"]:
        raise ValueError("mapping has no primary fields after terrain partitioning")
    return partition


def _partition_contributing(
    donor_mapping: Mapping[str, object],
    field_names: Sequence[str],
    *,
    binding_name: str,
) -> dict[str, object]:
    """A decoder-only view of one contributing source's bound fields.

    The donor's sealed, fully validated mapping stays the sole semantic
    authority for the borrowed octets; partitioning only keeps the donor
    from demanding fields the composition never asked it for.  A borrowed
    field must be DIRECTLY selected in the donor: a derived donor field
    would need its dependency fields decoded too, and this partitioned
    decode deliberately does not carry them.
    """

    partition = copy.deepcopy(dict(donor_mapping))
    fields = partition["fields"]
    assert isinstance(fields, dict)
    kept: dict[str, object] = {}
    for name in field_names:
        field = fields.get(name)
        if field is None:
            raise ValueError(
                f"contributing source binding {binding_name!r} borrows "
                f"{name!r} but the contributing mapping does not map it"
            )
        if not field.get("selectors"):
            raise ValueError(
                f"contributing source binding {binding_name!r} borrows "
                f"{name!r}, which the contributing mapping does not provide "
                "with direct selectors; a derived or externally bound donor "
                "field is not decodable in a partitioned donor decode"
            )
        kept[name] = field
    partition["fields"] = kept
    return partition


def _decode_partition(
    mapping: Mapping[str, object],
    files: Sequence[Path],
    decoders: Mapping[str, Path],
) -> _DecodedCollection:
    if mapping["format"] == "netcdf":
        return _decode_netcdf(mapping, files)
    return _decode_grib(
        mapping,
        files,
        grib1_bridge=decoders.get("grib1_bridge"),
        grib2_inventory=decoders.get("grib2_inventory"),
        grib2_dump=decoders.get("grib2_dump"),
    )


def _path_inventory(
    value: str | Path | Sequence[str | Path], label: str,
) -> tuple[Path, ...]:
    raw = (value,) if isinstance(value, (str, Path)) else tuple(value)
    paths = tuple(Path(path).resolve() for path in raw)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"{label} file inventory must be non-empty and unique")
    return paths


_COMPOSE_SCRATCH_ENV = "WOOF_COMPOSE_SCRATCH"


def _compose_scratch_base(destination: Path | None) -> Path | None:
    """The directory the engine's compose scratch is created IN.

    The engine stages the WHOLE composed frame stream in the scratch --
    f64 frames, tens of GB for the largest registered sources -- so the
    placement is a correctness matter: the system temp is RAM-backed
    tmpfs on common Linux boxes (Ubuntu with ``/tmp`` on tmpfs and a
    user quota; Fedora mounts ``/tmp`` at half of RAM by default), and a
    bare default prep of the biggest registered source died there,
    deterministically, with "cannot write the frame stream: Disk quota
    exceeded (os error 122)".

    Resolution order:

    1. ``WOOF_COMPOSE_SCRATCH`` -- an explicit scratch directory.
       Refused by name when it does not exist, because silently falling
       back would put the multi-GB stream on exactly the filesystem the
       caller set the variable to steer away from.
    2. ``destination`` -- the output directory a caller that owns a real
       preparation threads down (:func:`woof.mapped_direct.prepare_mapped_wrf`).
       The scratch is created in the destination's PARENT -- the same
       disk-backed filesystem the output lands on anyway, and the same
       placement the route's own atomic staging uses -- while the
       destination itself stays untouched so its create-only refusal
       keeps meaning something.
    3. ``None`` -- the system temp, for callers with no real output
       directory (in-memory recompose paths), whose streams are sized
       to what they already hold in memory.

    The ``decode`` and ``inspect`` routes in :mod:`woof.mapped_source`
    resolve through the same function with no destination, so the ONE
    variable covers every temporary the engine route writes; a
    0.25-degree analysis decode once died on the tmpfs quota with the
    variable set, because only the compose scratch honored it.
    """

    import os

    override = os.environ.get(_COMPOSE_SCRATCH_ENV)
    if override:
        base = Path(override)
        if not base.is_dir():
            raise NotADirectoryError(
                f"{_COMPOSE_SCRATCH_ENV}={override} does not name an "
                "existing directory.  The engine stages the whole composed "
                "frame stream there -- tens of GB for the largest sources "
                "-- and silently falling back to the system temp would put "
                "that stream on exactly the filesystem this variable was "
                "set to avoid")
        return base
    if destination is None:
        return None
    destination = Path(destination).resolve()
    base = destination.parent
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise NotADirectoryError(
            f"cannot create the parent directory of the preparation "
            f"output {destination}: {error}.  The compose scratch stages "
            f"the multi-GB frame stream in {base} so it shares the "
            "output's disk-backed filesystem; choose an output root whose "
            "parent is writable") from error
    return base


def _unpublished_binding_record(binding_name, binding, fallbacks, *,
                                mapping_path, donor_files, provenance_path,
                                before):
    """The receipt of a binding whose files publish none of its fields.

    Twin of the engine's ``compose.rs unpublished_record``: ``None`` when
    any bound field has no fallback in ``fallbacks`` (or the binding
    supplies terrain, which has its own ``when_absent``), so the caller
    refuses as it always did.
    """

    names = [str(name) for name in binding["fields"]]
    if (not fallbacks or _EXTERNAL_FIELD in names
            or any(name not in fallbacks for name in names)):
        return None
    return {
        "binding": binding_name,
        "source_id": str(binding["source_id"]),
        "mapping": {"path": str(mapping_path),
                    "sha256": before[str(mapping_path)]},
        "data": [{"path": str(path), "sha256": before[str(path)]}
                 for path in donor_files],
        "provenance": {"path": str(provenance_path),
                       "sha256": before[str(provenance_path)]},
        "fields": sorted(names),
        "alignment": {
            "status": "UNPUBLISHED",
            "reason": "the supplied files publish no record for these fields",
            "fallback": {name: fallbacks[name] for name in names},
        },
    }


def _compose_through_engine(
    *,
    engine: Path,
    mapping_path: Path,
    composition_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    primary: Sequence[Path],
    supplements: Mapping[str, Sequence[Path]],
    provenance: Mapping[str, Path],
    contributing: Mapping[str, Path],
    contract: Mapping[str, object],
    bindings: Mapping[str, Mapping[str, object]],
    terrain_spec: Mapping[str, object] | None,
    decoders: Mapping[str, Path],
    snapshots: Mapping[Path, _AuthoritySnapshot],
    before: Mapping[str, str],
    member: str | None = None,
    member_identity: str | None = None,
    scratch_destination: Path | None = None,
    atmospheric_grids=(),
    workers: int | None = None,
    lead_batch: bool = False,
    memory_budget_bytes: int | None = None,
    unpublished_fallbacks: Mapping[str, str] | None = None,
    _shared_directory: Path | None = None,
    _shared_hit: bool = False,
) -> MappedSourceBundle:
    """Compose on the Rust engine, keeping every policy check on this side.

    Everything above this call has already run: manifest verification,
    the role inventory, the primary-versus-donor field agreement, the
    authority hash window.  What crosses the seam is the byte work the
    design's Ã‚Â§1.1 names -- partition decode, terrain composition, bound
    fields, frame materialization -- and what comes back is a frameset
    plus the two evidence products only that work can produce.

    The soil-layer contract is validated against the DONOR table here
    exactly as the Python engine does it, because it is a check on
    sealed documents rather than on bytes and must not become the
    engine's to remember.

    ``scratch_destination`` places the engine's scratch -- which holds
    the whole composed frame stream -- on the same filesystem as the
    preparation output; see :func:`_compose_scratch_base` for the
    resolution order and the tmpfs quota failure it prevents.

    The scratch OUTLIVES this call on purpose.  The returned bundle
    reads one valid time at a time out of the frame stream instead of
    holding every time in host memory, so the bytes have to still be
    there while the preparation walks them; the frameset owns the
    scratch handle and deletes the tree when it is closed or collected.
    A failure before the bundle exists deletes it here.
    """

    if _shared_directory is None and (binding := _SHARED_COMPOSITION.get()) is not None and binding[1] is not None:
        arguments = dict(locals())
        arguments.pop("binding")
        return _shared_engine_composition(arguments, binding)

    import os
    import tempfile

    from woof import mapped_engine_bridge

    scratch_base = _compose_scratch_base(scratch_destination)
    work = None if _shared_directory is not None else tempfile.TemporaryDirectory(
        prefix="gpuwm-mapped-compose-",
        dir=None if scratch_base is None else os.fspath(scratch_base),
    )
    try:
        directory = _shared_directory if work is None else Path(work.name) / "composed"
        try:
            if _shared_hit:
                ready = json.loads((directory.parent / "READY.json").read_text(encoding="utf-8"))
                execution = {"decode_memory": ready["decode_memory"]}
            else:
                from woof.timeline.ledger import emit
                measured = _SHARED_COMPOSITION.get()
                measured = measured if measured is not None and measured[1] is None else None
                attrs = {"source_key": measured[0], "cache_hit": False} if measured else None
                if measured:
                    emit("prep.decode", "begin", product="ensemble", attrs=attrs)
                    emit("prep.compose", "begin", product="ensemble", attrs=attrs)
                try:
                    execution = mapped_engine_bridge.run_engine(
                        "compose", mapping=mapping_path, files=primary, output=directory,
                        composition=composition_path, supplements=supplements, provenance=provenance,
                        contributing_mappings=contributing, input_manifest=manifest_path,
                        input_manifest_sha256=manifest_sha256, engine=engine,
                        atmospheric_grids=atmospheric_grids, threads=workers, lead_batch=lead_batch,
                        **({} if memory_budget_bytes is None else {"memory_budget_bytes": memory_budget_bytes}),
                        **({"unpublished_fallbacks": unpublished_fallbacks}
                           if unpublished_fallbacks else {}),
                    )
                finally:
                    if measured:
                        emit("prep.compose", "end", product="ensemble", attrs=attrs)
                        emit("prep.decode", "end", product="ensemble", attrs=attrs)
        except ScratchDiskRefusal as refusal:
            raise scratch_disk_refusal(refusal, scratch_base) from refusal
        evidence = mapped_engine_bridge.read_composition_evidence(directory)
        for record in evidence["contributing_sources"]:
            regrid = record.get("alignment", {}).get("horizontal_regrid")
            if regrid:
                print(f"prep donor regrid: {record['binding']} {regrid['method']} "
                      f"{regrid['source_shape']} -> {regrid['target_shape']}, "
                      f"{regrid['seconds']:.3f} s, {regrid['threads']} threads",
                      flush=True)
        decode_control = {"budget_bytes": memory_budget_bytes,
                          "pending": bool(atmospheric_grids)}
        def full_fallback():
            # Re-read the same sealed source through the unchanged full writer
            # only if a later consumer needs donors beyond the proven window.
            # The new bundle revalidates all source/composition identities.
            full = _compose_through_engine(
                engine=engine, mapping_path=mapping_path, composition_path=composition_path,
                manifest_path=manifest_path, manifest_sha256=manifest_sha256,
                primary=primary, supplements=supplements, provenance=provenance,
                contributing=contributing, contract=contract, bindings=bindings,
                terrain_spec=terrain_spec, decoders=decoders, snapshots=snapshots,
                before=before, member=member, member_identity=member_identity,
                scratch_destination=scratch_destination, workers=workers,
                lead_batch=lead_batch,
                memory_budget_bytes=decode_control["budget_bytes"],
                unpublished_fallbacks=unpublished_fallbacks).frames
            decode_control["pending"] = False
            return full
        frames = mapped_engine_bridge.open_frameset(
            directory, retain=work,
            **({"full_fallback": full_fallback} if atmospheric_grids else {}))
        bundle = _composed_bundle_from_frames(
            frames=frames,
            evidence=evidence,
            engine_name=mapped_engine_bridge.ENGINE_NAME,
            mapping_path=mapping_path,
            composition_path=composition_path,
            manifest_path=manifest_path,
            contributing=contributing,
            contract=contract,
            bindings=bindings,
            terrain_spec=terrain_spec,
            decoders=decoders,
            snapshots=snapshots,
            before=before,
            member=member,
            member_identity=member_identity,
            supplements=supplements,
            provenance=provenance,
        )
        return _dataclass_replace(bundle, native_decode_memory=(
            execution.get("decode_memory") if isinstance(execution, Mapping) else None),
            _native_decode_control=decode_control)
    except BaseException:
        if work is not None:
            work.cleanup()
        raise


def _composed_bundle_from_frames(
    *,
    frames,
    evidence,
    engine_name: str,
    mapping_path: Path,
    composition_path: Path,
    manifest_path: Path,
    contributing: Mapping[str, Path],
    contract: Mapping[str, object],
    bindings: Mapping[str, Mapping[str, object]],
    terrain_spec: Mapping[str, object] | None,
    decoders: Mapping[str, Path],
    snapshots: Mapping[Path, _AuthoritySnapshot],
    before: Mapping[str, str],
    member: str | None,
    member_identity: str | None,
    supplements: Mapping[str, Sequence[Path]],
    provenance: Mapping[str, Path],
) -> MappedSourceBundle:
    """Seal an engine-composed frameset into a bundle.

    Every check here reads the frameset's DOCUMENT -- the per-frame
    scalars the engine wrote beside the arrays -- so sealing a
    preparation does not have to read a single valid time's numbers.
    The arrays are checked against the same document when each frame is
    actually read, which is the only place they can be wrong.
    """

    mapped_engine_bridge_name = engine_name
    for digest in frames.mapping_sha256s:
        if digest != before[str(mapping_path)]:
            raise RuntimeError(
                f"{mapped_engine_bridge_name} composed against "
                f"mapping bytes hashing to {digest}; this "
                f"call verified {mapping_path} at "
                f"{before[str(mapping_path)]}, so the bundle would be "
                "bound to a document nobody checked")
    if member is not None:
        # The manifest bound an explicit ensemble member; an engine that
        # returned frames without it decoded under a contract this call
        # did not seal, for an ENSEMBLE source.
        stamped = set(frames.members)
        if stamped != {member}:
            raise RuntimeError(
                f"{mapped_engine_bridge_name} returned frames with "
                f"member(s) {sorted(str(value) for value in stamped)}; the "
                f"sealed input manifest binds explicit member {member!r}, "
                "so the ensemble identity did not survive the engine")
        recorded_alignment = dict(evidence["alignment_receipt"])
        if (recorded_alignment.get("member") != member
                or recorded_alignment.get("member_identity")
                != member_identity):
            raise RuntimeError(
                f"{mapped_engine_bridge_name} composed an alignment "
                "receipt without the manifest's explicit member binding, "
                "so the preparation's sealed receipt would lose the "
                "ensemble identity")

    contributing_records = tuple(
        dict(record) for record in evidence["contributing_sources"]
    )
    recorded = {str(record["binding"]) for record in contributing_records}
    if recorded != set(bindings):
        raise RuntimeError(
            f"{mapped_engine_bridge_name} composed "
            f"{sorted(recorded)} cross-source bindings; the composition "
            f"declares {sorted(bindings)}, so a declared borrow either "
            "did not happen or was invented")

    soil_donor_mapping: Mapping[str, object] | None = None
    terrain_binding: Mapping[str, object] | None = None
    for binding_name in sorted(bindings):
        binding = bindings[binding_name]
        bound_names = [str(name) for name in binding["fields"]]
        if _EXTERNAL_FIELD in bound_names:
            terrain_binding = binding
        if any(name in bound_names for name in _SOIL_PAIR):
            donor_path = contributing[str(binding["mapping_role"])]
            soil_donor_mapping = load_mapping(donor_path)
    if soil_donor_mapping is not None:
        validate_soil_layer_contract(
            contract["soil_layers"], mapping=soil_donor_mapping,
        )

    map_parallel(_require_authority_snapshot, snapshots.values())

    if terrain_spec is not None:
        terrain_paths = tuple(supplements[str(terrain_spec["data_role"])])
        terrain_provenance_path = provenance[
            str(terrain_spec["provenance_role"])]
    else:
        if terrain_binding is None:
            raise ValueError(
                "composed source has neither a terrain supplement nor a "
                "terrain field binding")
        terrain_paths = tuple(supplements[str(terrain_binding["data_role"])])
        terrain_provenance_path = provenance[
            str(terrain_binding["provenance_role"])]
    warn_completed_fields(completed_field_summary(frames), subject="the source")
    from woof.mapped_source import warn_unpublished_bindings
    warn_unpublished_bindings(contributing_records, subject="the source")
    return MappedSourceBundle(
        frames=frames,
        mapping_path=mapping_path,
        mapping_sha256=before[str(mapping_path)],
        composition_path=composition_path,
        composition_sha256=before[str(composition_path)],
        input_manifest_path=manifest_path,
        input_manifest_sha256=before[str(manifest_path)],
        decoder_paths=dict(decoders),
        decoder_sha256={
            role: before[str(path)] for role, path in decoders.items()
        },
        terrain_data_paths=terrain_paths,
        terrain_data_sha256=tuple(
            before[str(path)] for path in terrain_paths
        ),
        terrain_provenance_path=terrain_provenance_path,
        terrain_provenance_sha256=before[str(terrain_provenance_path)],
        soil_layer_contract=contract["soil_layers"],
        alignment_receipt=dict(evidence["alignment_receipt"]),
        contributing_sources=contributing_records,
        record_aliases=(
            {str(name): int(count)
             for name, count in evidence["record_aliases"].items()}
            if evidence.get("record_aliases") else None),
    )


def decode_composed_source(
    composition_path: str | Path,
    mapping_path: str | Path,
    primary_files: Sequence[str | Path],
    supplement_files: Mapping[
        str, str | Path | Sequence[str | Path]
    ],
    provenance_files: Mapping[str, str | Path],
    *,
    input_manifest: str | Path,
    input_manifest_sha256: str,
    contributing_mappings: Mapping[str, str | Path] | None = None,
    grib1_bridge: str | Path | None = None,
    grib2_inventory: str | Path | None = None,
    grib2_dump: str | Path | None = None,
    scratch_destination: str | Path | None = None,
    atmospheric_grids=(),
    workers: int | None = None,
    lead_batch: bool = False,
    memory_budget_bytes: int | None = None,
) -> MappedSourceBundle:
    """Decode a complete mapped source with scientifically sourced terrain.

    A cross-source composition additionally supplies each contributing
    source's own sealed mapping document under its declared
    ``mapping_role`` via ``contributing_mappings``; the observed bytes must
    hash to the SHA-256 the composition pins for that binding.  The
    contributing data files ride ``supplement_files`` and the contributing
    provenance documents ride ``provenance_files``, each under the
    binding's declared role, so the input manifest binds every byte it
    always bound; the contributing mapping itself is pinned by the
    composition (which the manifest pins), not by a manifest row.

    ``scratch_destination`` is the output directory of the preparation
    this decode feeds, when the caller has one: the engine's compose
    scratch -- the whole composed frame stream, tens of GB for the
    largest sources -- is created beside it instead of in the system
    temp, which is RAM-backed tmpfs on common Linux boxes and killed the
    bare default prep of the biggest registered source with a disk
    quota error.  ``WOOF_COMPOSE_SCRATCH`` overrides it; omitted
    entirely, the system temp remains the documented fallback for
    callers with no real output directory.

    ``workers`` is the most threads the Rust engine decodes with
    (``--preprocess-workers``); ``None`` lets it use every core it may
    run on.  Either way a GRIB2 decode narrows to what its first valid
    time fits beside in free memory, before that time is decoded, with
    as many valid times in flight as memory allows.

    ``lead_batch`` decodes some of a window's leads (an as-posted
    preparation's batch, which may be one lead): the window's series rules,
    at least two times on one uniform cadence that the target's boundary
    interval accepts, are the caller's to hold over every batch's times
    together, and every other check is the same.
    """

    composition_path = Path(composition_path).resolve()
    mapping_path = Path(mapping_path).resolve()
    manifest_path = Path(input_manifest).resolve()
    primary = tuple(Path(path).resolve() for path in primary_files)
    supplements = {
        role: _path_inventory(path, f"supplement {role!r}")
        for role, path in supplement_files.items()
    }
    provenance = {role: Path(path).resolve() for role, path in provenance_files.items()}
    contributing = {
        str(role): Path(path).resolve()
        for role, path in (contributing_mappings or {}).items()
    }
    if not primary or len(set(primary)) != len(primary):
        raise ValueError("primary source inventory must be non-empty and unique")
    for path in (
        composition_path, mapping_path, manifest_path,
        *primary,
        *(path for paths in supplements.values() for path in paths),
        *provenance.values(),
        *contributing.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    mapping_snapshot = _snapshot_authority(mapping_path, retain_bytes=True)
    composition_snapshot = _snapshot_authority(
        composition_path,
        retain_bytes=True,
    )
    manifest_snapshot = _snapshot_authority(manifest_path, retain_bytes=True)
    mapping_raw = _load_json_bytes(
        mapping_snapshot.data,
        "mapping",
        mapping_path,
    )
    composition_raw = _load_json_bytes(
        composition_snapshot.data,
        "composition",
        composition_path,
    )
    mapping = load_mapping(mapping_path, _raw=mapping_raw)
    contract = load_composition(
        composition_path,
        mapping_path,
        _raw=composition_raw,
        _mapping_raw=mapping_raw,
    )
    bindings = {
        str(name): binding
        for name, binding in (contract.get("field_sources") or {}).items()
    }
    terrain_spec = contract["supplements"].get(_EXTERNAL_FIELD)

    expected_mapping_roles = {
        str(binding["mapping_role"]) for binding in bindings.values()
    }
    if set(contributing) != expected_mapping_roles:
        raise ValueError(
            "contributing mapping role inventory differs from the "
            f"composition: expected {sorted(expected_mapping_roles)}, "
            f"got {sorted(contributing)}"
        )
    donor_mappings: dict[str, Mapping[str, object]] = {}
    donor_paths: dict[str, Path] = {}
    contributing_snapshots: dict[Path, _AuthoritySnapshot] = {}
    for binding_name in sorted(bindings):
        binding = bindings[binding_name]
        donor_path = contributing[str(binding["mapping_role"])]
        snapshot = _snapshot_authority(donor_path, retain_bytes=True)
        pinned = _digest(
            binding["mapping_sha256"],
            f"field_sources.{binding_name}.mapping_sha256",
        )
        if snapshot.sha256 != pinned:
            raise ValueError(
                f"contributing source binding {binding_name!r} authority "
                f"hash mismatch: the composition pins {pinned}, the supplied "
                f"mapping bytes hash to {snapshot.sha256}"
            )
        donor_raw = _load_json_bytes(
            snapshot.data, "contributing mapping", donor_path,
        )
        donor_mapping = load_mapping(donor_path, _raw=donor_raw)
        if str(donor_mapping["name"]) != str(binding["source_id"]):
            raise ValueError(
                f"contributing source binding {binding_name!r} names "
                f"source_id {binding['source_id']!r} but the pinned mapping "
                f"is {donor_mapping['name']!r}"
            )
        companion = binding.get("companion")
        extended = [str(name) for name in (companion or {}).get("fields", ())]
        if companion is not None:
            ours = mapping["coordinates"]["vertical"]
            theirs = donor_mapping["coordinates"]["vertical"]
            for aspect in ("kind", "units"):
                if ours.get(aspect) != theirs.get(aspect):
                    raise ValueError(
                        f"contributing source binding {binding_name!r} is a "
                        f"companion whose vertical {aspect} is "
                        f"{theirs.get(aspect)!r} where the primary's is "
                        f"{ours.get(aspect)!r}; levels on two coordinates "
                        "cannot be stacked into one column"
                    )
        for name in [*binding["fields"], *extended]:
            name = str(name)
            declared = mapping["fields"][name]
            donor_field = donor_mapping["fields"].get(name)
            if donor_field is None:
                continue  # _partition_contributing refuses with the full story
            for aspect, ours, theirs in (
                ("units.target", declared["units"]["target"],
                 donor_field["units"]["target"]),
                ("target_axes", tuple(declared["target_axes"]),
                 tuple(donor_field["target_axes"])),
                ("location", declared["location"], donor_field["location"]),
                ("staggering", declared.get("staggering", "none"),
                 donor_field.get("staggering", "none")),
                ("missing.kind", declared["missing"]["kind"],
                 donor_field["missing"]["kind"]),
            ):
                if aspect == "missing.kind" and name in binding.get("vertical_merge", {}).get("fields", {}):
                    continue
                if ours != theirs:
                    raise ValueError(
                        f"field {name!r} {aspect} disagrees between the "
                        f"primary mapping ({ours!r}) and contributing source "
                        f"{binding['source_id']!r} ({theirs!r})"
                    )
        donor_mappings[binding_name] = donor_mapping
        donor_paths[binding_name] = donor_path
        contributing_snapshots[donor_path] = snapshot

    formats = {str(mapping["format"])} | {
        str(donor_mapping["format"])
        for donor_mapping in donor_mappings.values()
    }
    engine_binary = None
    if _mapped_engine_choice(
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
        subcommand="compose",
        source_format=str(mapping["format"]),
    ) == _ENGINE_RUST:
        from woof.mapped_engine_bridge import require_engine

        engine_binary = require_engine()
    decoders = _decoder_inventory(
        sorted(formats),
        grib1_bridge=grib1_bridge,
        grib2_inventory=grib2_inventory,
        grib2_dump=grib2_dump,
        engine=engine_binary,
    )
    for path in decoders.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_data_roles: set[str] = set()
    expected_provenance_roles: set[str] = set()
    if terrain_spec is not None:
        terrain_data_role = str(terrain_spec["data_role"])
        terrain_provenance_role = str(terrain_spec["provenance_role"])
        expected_data_roles.add(terrain_data_role)
        expected_provenance_roles.add(terrain_provenance_role)
    for binding in bindings.values():
        expected_data_roles.add(str(binding["data_role"]))
        expected_provenance_roles.add(str(binding["provenance_role"]))
    if set(supplements) != expected_data_roles \
            or set(provenance) != expected_provenance_roles:
        raise ValueError("composition role inventory differs from the contract")

    authorities = (
        composition_path, mapping_path, manifest_path,
        *decoders.values(), *primary,
        *(path for paths in supplements.values() for path in paths),
        *provenance.values(),
        *contributing.values(),
    )
    snapshots = {
        composition_path: composition_snapshot,
        mapping_path: mapping_snapshot,
        manifest_path: manifest_snapshot,
        **contributing_snapshots,
    }
    # Every input file at once across the cores (woof.authority_digests): the
    # serial loop read ~22 GB of HRRR GRIB on one core before the decode.
    missing = [path for path in dict.fromkeys(authorities) if path not in snapshots]
    for path, snapshot in zip(missing, map_parallel(_snapshot_authority, missing)):
        snapshots[path] = snapshot
    manifest_document = _verify_manifest(
        manifest_path, input_manifest_sha256,
        mapping_path=mapping_path, composition_path=composition_path,
        primary_files=primary, supplement_files=supplements,
        provenance_files=provenance, decoder_files=decoders,
        _snapshots=snapshots,
        _recheck_snapshots=False,
    )
    declared_member = manifest_document.get("member")
    declared_member_identity = manifest_document.get("member_identity")
    for snapshot in (
        mapping_snapshot,
        composition_snapshot,
        manifest_snapshot,
        *contributing_snapshots.values(),
    ):
        _require_authority_snapshot(snapshot)
    before = {str(path): snapshots[path].sha256 for path in authorities}
    if any(binding.get("vertical_merge") for binding in bindings.values()) and engine_binary is None:
        raise ValueError("vertical_merge requires the native mapped engine for column splicing")
    if any(binding.get("grid_alignment") == "bilinear_mask_aware_nearest"
           for binding in bindings.values()) and engine_binary is None:
        raise ValueError("horizontal donor regrid requires the native mapped engine")
    if memory_budget_bytes is not None and engine_binary is None:
        from woof.ingest.memory_refusal import InitializationMemoryRefused

        raise InitializationMemoryRefused(
            "the live preparation's future decode requires the native host-memory budget; "
            "the Python decoder cannot enforce that reservation")
    from woof.runtime_surface_fetch import composition_unpublished_fallbacks
    unpublished_fallbacks = composition_unpublished_fallbacks(bindings)
    if engine_binary is not None:
        return _compose_through_engine(
            unpublished_fallbacks=unpublished_fallbacks,
            engine=engine_binary,
            mapping_path=mapping_path,
            composition_path=composition_path,
            manifest_path=manifest_path,
            manifest_sha256=input_manifest_sha256,
            primary=primary,
            supplements=supplements,
            provenance=provenance,
            contributing=contributing,
            contract=contract,
            bindings=bindings,
            terrain_spec=terrain_spec,
            decoders=decoders,
            snapshots=snapshots,
            before=before,
            member=declared_member,
            member_identity=declared_member_identity,
            scratch_destination=(
                None if scratch_destination is None
                else Path(scratch_destination)),
            atmospheric_grids=atmospheric_grids,
            workers=workers,
            lead_batch=lead_batch,
            memory_budget_bytes=memory_budget_bytes,
        )
    combined = _decode_partition(
        _partition_mapping(mapping, terrain_only=False), primary, decoders,
    )
    record_aliases = combined.record_aliases
    terrain_supplement_receipt = None
    if terrain_spec is not None:
        derived_terrain = None
        try:
            terrain_collection = _decode_partition(
                _partition_mapping(mapping, terrain_only=True),
                supplements[terrain_data_role], decoders,
            )
        except NothingMatched:
            # The files carry no terrain record at all.  A mapping that
            # declares how terrain is derived has it derived from the
            # primary's own first valid time; without that declaration
            # the miss is refused by name, as it always was.
            operation = mapping["fields"][_EXTERNAL_FIELD].get("when_absent")
            if operation is None:
                raise
            terrain_collection, derived_terrain = _derive_absent_terrain(
                mapping, operation, combined)
        combined, terrain_supplement_receipt = _compose_terrain(
            combined, terrain_collection,
            time_alignment=str(terrain_spec["time_alignment"]),
            grid_alignment=str(terrain_spec["grid_alignment"]),
        )
        if derived_terrain is not None:
            # Recorded only when terrain was derived, so every receipt of
            # a source that carries its terrain is unchanged.
            terrain_supplement_receipt = {
                **terrain_supplement_receipt, "derived": derived_terrain,
            }

    contributing_records: list[dict[str, object]] = []
    terrain_binding: Mapping[str, object] | None = None
    terrain_binding_receipt: dict[str, object] | None = None
    soil_donor_mapping: Mapping[str, object] | None = None
    unpublished: set[str] = set()
    for binding_name in sorted(bindings):
        binding = bindings[binding_name]
        bound_names = [str(name) for name in binding["fields"]]
        extended_names = [
            str(name)
            for name in (binding.get("companion") or {}).get("fields", ())
        ]
        donor_files = supplements[str(binding["data_role"])]
        try:
            donor_collection = _decode_partition(
                _partition_contributing(
                    donor_mappings[binding_name], bound_names + extended_names,
                    binding_name=binding_name,
                ),
                donor_files, decoders,
            )
        except NothingMatched:
            if binding.get("companion") is not None:
                # Land fallbacks cannot supply the absent upper column.
                raise
            # Twin of the engine's unpublished binding (compose.rs
            # unpublished_record): files that publish no record for any
            # bound field, every one of which has a field-keyed fallback.
            record = _unpublished_binding_record(
                binding_name, binding, unpublished_fallbacks,
                mapping_path=donor_paths[binding_name],
                donor_files=donor_files,
                provenance_path=provenance[str(binding["provenance_role"])],
                before=before)
            if record is None:
                raise
            contributing_records.append(record)
            unpublished.update(bound_names)
            continue
        combined, receipt = _compose_bound_fields(
            combined, donor_collection,
            binding_name=binding_name, binding=binding,
        )
        provenance_path = provenance[str(binding["provenance_role"])]
        contributing_records.append({
            "binding": binding_name,
            "source_id": str(binding["source_id"]),
            "mapping": {
                "path": str(donor_paths[binding_name]),
                "sha256": before[str(donor_paths[binding_name])],
            },
            "data": [
                {"path": str(path), "sha256": before[str(path)]}
                for path in donor_files
            ],
            "provenance": {
                "path": str(provenance_path),
                "sha256": before[str(provenance_path)],
            },
            "fields": sorted(bound_names),
            "alignment": receipt,
        })
        if _EXTERNAL_FIELD in bound_names:
            terrain_binding = binding
            terrain_binding_receipt = receipt
        if any(name in bound_names for name in _SOIL_PAIR):
            soil_donor_mapping = donor_mappings[binding_name]
    if soil_donor_mapping is not None:
        # The soil geometry the composition declares must be the DONOR
        # table's, now that the donor's sealed bytes are pinned and loaded.
        validate_soil_layer_contract(
            contract["soil_layers"], mapping=soil_donor_mapping,
        )

    # The manifest's explicit member binding lands LAST, after every
    # join: the composition operates on the identity the bytes carry
    # (donor alignment refuses member-bearing donors on its own terms),
    # and the stamp then reaches every canonical frame at once.
    if declared_member is not None:
        combined = _bind_manifest_member(combined, declared_member)

    # Materialize against the union view: the primary mapping's own fields
    # plus, for every borrowed field, the contributing mapping's sealed
    # field spec -- the decode authority for those octets.
    union = copy.deepcopy(dict(mapping))
    union_fields = dict(union["fields"])
    for binding_name in sorted(bindings):
        donor_mapping = donor_mappings[binding_name]
        for name in bindings[binding_name]["fields"]:
            if str(name) in unpublished:
                continue
            spec = copy.deepcopy(dict(donor_mapping["fields"][str(name)]))
            # The primary binds a borrowed field in order to publish it,
            # so the donor's own ``dependency_only`` does not ride along:
            # it would hold the field the composition exists to supply
            # off the composed frame.
            spec.pop("dependency_only", None)
            union_fields[str(name)] = spec
    for name in unpublished:
        # Off the composed frame: the start reads the recorded fallback.
        union_fields.pop(name, None)
    union["fields"] = union_fields
    if unpublished:
        target = dict(union["target"])
        target["required_fields"] = [
            item for item in target["required_fields"]
            if str(item["name"]) not in unpublished]
        union["target"] = target

    input_hashes = {
        str(path): before[str(path)]
        for path in (
            *primary, *(path for paths in supplements.values() for path in paths)
        )
    }
    frames = _materialize_frames(
        union, combined, mapping_sha256=before[str(mapping_path)],
        input_sha256=input_hashes, lead_batch=lead_batch,
    )
    map_parallel(_require_authority_snapshot, snapshots.values())
    if terrain_spec is not None:
        terrain_paths = supplements[terrain_data_role]
        terrain_provenance_path = provenance[terrain_provenance_role]
        terrain_receipt = terrain_supplement_receipt
    else:
        assert terrain_binding is not None and terrain_binding_receipt is not None
        terrain_paths = supplements[str(terrain_binding["data_role"])]
        terrain_provenance_path = provenance[
            str(terrain_binding["provenance_role"])
        ]
        terrain_receipt = terrain_binding_receipt
    if declared_member is not None:
        # The sealed alignment receipt keeps the ensemble identity the
        # manifest bound, beside the alignment it describes.  Recorded
        # only under an explicit binding, so every receipt written
        # before the binding existed is byte-identical today.
        terrain_receipt = {
            **terrain_receipt,
            "member": declared_member,
            "member_identity": declared_member_identity,
        }
    warn_completed_fields(completed_field_summary(frames), subject="the source")
    from woof.mapped_source import warn_unpublished_bindings
    warn_unpublished_bindings(contributing_records, subject="the source")
    return MappedSourceBundle(
        frames=frames,
        mapping_path=mapping_path,
        mapping_sha256=before[str(mapping_path)],
        composition_path=composition_path,
        composition_sha256=before[str(composition_path)],
        input_manifest_path=manifest_path,
        input_manifest_sha256=before[str(manifest_path)],
        decoder_paths=decoders,
        decoder_sha256={
            role: before[str(path)] for role, path in decoders.items()
        },
        terrain_data_paths=terrain_paths,
        terrain_data_sha256=tuple(
            before[str(path)] for path in terrain_paths
        ),
        terrain_provenance_path=terrain_provenance_path,
        terrain_provenance_sha256=before[str(terrain_provenance_path)],
        soil_layer_contract=contract["soil_layers"],
        alignment_receipt=terrain_receipt,
        contributing_sources=tuple(contributing_records),
        record_aliases=record_aliases,
    )


def _receipt_times(bundle: MappedSourceBundle):
    frames = bundle.frames
    times = getattr(frames, "valid_times", None)
    if times is None:
        return tuple(frame.valid_time for frame in frames)
    return times


def _receipt_frames(bundle: MappedSourceBundle) -> list[dict[str, object]]:
    frames = bundle.frames
    digest = getattr(frames, "field_digest", None)
    if digest is None:
        return [{
            "header_sha256": _canonical_sha256(frame.header.to_dict()),
            "terrain_sha256": _array_sha256(frame.fields[_EXTERNAL_FIELD].values),
            "field_count": len(frame.fields),
        } for frame in frames]
    return [{
        "header_sha256": _canonical_sha256(frames.header(index).to_dict()),
        "terrain_sha256": digest(index, _EXTERNAL_FIELD),
        "field_count": frames.field_count(index),
    } for index in range(len(frames))]


def mapped_composition_receipt(bundle: MappedSourceBundle) -> dict[str, object]:
    """Return immutable evidence for the complete canonical materialization."""

    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "CANONICAL_FRAMES_COMPLETE_NOT_STOCK_WRF_CERTIFIED",
        "mapping": {
            "path": str(bundle.mapping_path), "sha256": bundle.mapping_sha256,
        },
        "composition": {
            "path": str(bundle.composition_path),
            "sha256": bundle.composition_sha256,
        },
        "input_manifest": {
            "path": str(bundle.input_manifest_path),
            "sha256": bundle.input_manifest_sha256,
        },
        "decoders": {
            role: {
                "path": str(path),
                "sha256": bundle.decoder_sha256[role],
            }
            for role, path in bundle.decoder_paths.items()
        },
        "terrain_products": [{
            "path": str(path),
            "sha256": digest,
        } for path, digest in zip(
            bundle.terrain_data_paths, bundle.terrain_data_sha256,
        )],
        "terrain_provenance": {
            "provenance_path": str(bundle.terrain_provenance_path),
            "provenance_sha256": bundle.terrain_provenance_sha256,
        },
        "alignment": dict(bundle.alignment_receipt),
        # Present only for cross-source compositions, so every receipt a
        # single-source preparation wrote before this key existed stays
        # byte-identical to what the same preparation writes today.  Each
        # entry names one contributing source: its id, its OWN sealed
        # mapping authority hash, its data and provenance hashes, and the
        # grid/time alignment receipt for the borrowed fields.
        **(
            {
                "contributing_sources": [
                    {
                        "binding": str(record["binding"]),
                        "source_id": str(record["source_id"]),
                        "mapping": dict(record["mapping"]),
                        "data": [dict(row) for row in record["data"]],
                        "provenance": dict(record["provenance"]),
                        "fields": list(record["fields"]),
                        "alignment": dict(record["alignment"]),
                    }
                    for record in bundle.contributing_sources
                ],
            }
            if bundle.contributing_sources else {}
        ),
        "soil_layers": dict(bundle.soil_layer_contract),
        # Present only when the primary's files spelled records the way
        # an earlier publication of the product did, so every other
        # receipt is unchanged.
        **(
            {"record_aliases": {
                str(name): int(count)
                for name, count in sorted(bundle.record_aliases.items())
            }}
            if bundle.record_aliases else {}
        ),
        # Present only when a frame derived values the source did not
        # carry, so a receipt of a source that carried every value is
        # byte-identical to what it was before this key existed.  Each
        # entry counts the derived values against the field's total.
        **(
            {"completed_fields": completed}
            if (completed := completed_field_summary(bundle.frames)) else {}
        ),
        "frame_count": len(bundle.frames),
        # Read from the frameset's own document where there is one, so
        # sealing a receipt does not pull every valid time's arrays back
        # into host memory.  The numbers are the same either way: the
        # terrain digest recorded here is exactly the digest each frame
        # is checked against when it is read, so a preparation that
        # publishes this receipt read arrays that match it.
        "valid_times": [value.isoformat() for value in _receipt_times(bundle)],
        "frames": _receipt_frames(bundle),
    }
    payload["receipt_content_sha256"] = _canonical_sha256(payload)
    return payload


#: The receipt keys that say where a file sat in this run's folders.  Each
#: sits beside the sha256 of the same file, which is what the receipt
#: binds; the location is this run's, not the source's.
_RECEIPT_PATH_KEYS = frozenset({"path", "provenance_path"})


def _without_paths(value):
    if isinstance(value, Mapping):
        return {key: _without_paths(item) for key, item in value.items()
                if key not in _RECEIPT_PATH_KEYS}
    if isinstance(value, (list, tuple)):
        return [_without_paths(item) for item in value]
    return value


def composition_receipt_identity(receipt: Mapping[str, object]) -> dict:
    """What a prepared cache's identity binds of a composition receipt.

    The receipt less every file's location (its ``path`` and
    ``provenance_path`` keys) and less its own content digest: each file
    stays bound by the sha256 recorded beside it, and the frames, times
    and alignment are kept whole.  The proof keeps the whole receipt.

    The breakage this prevents (A151): the receipt records where this run
    found its input manifest and data, so two preparations of the same
    bytes into different run folders bound different cache identities and
    content digests although every array was byte-equal.
    """

    body = {key: value for key, value in receipt.items()
            if key != "receipt_content_sha256"}
    return _without_paths(body)


def composition_receipt_identity_sha256(receipt: Mapping[str, object]) -> str:
    """The digest a prepared cache binds a composition receipt by (A151)."""

    return _canonical_sha256(composition_receipt_identity(receipt))


def composition_receipt_binding_matches(bound, receipt) -> bool:
    """Whether a cache's bound receipt digest names this proof's receipt.

    A cache written since A151 binds :func:`composition_receipt_identity_sha256`;
    one written before bound the whole receipt's ``receipt_content_sha256``
    and still restores.  Anything else names another composition.
    """

    if not isinstance(receipt, Mapping) or not isinstance(bound, str):
        return False
    return bound in (composition_receipt_identity_sha256(receipt),
                     receipt.get("receipt_content_sha256"))


def _derive_absent_terrain(
    mapping: Mapping[str, object],
    operation: Mapping[str, object],
    combined,
):
    """Terrain for a source whose terrain files carry none.

    Derived once, from the primary's first valid time, and broadcast by
    the terrain join exactly like a published analysis terrain.  Returns
    the one-field collection the join reads and the receipt entry.  The
    mapped engine's ``compose::derive_absent_terrain`` is the same step.
    """

    from woof.mapped_source import (
        _height_at_surface_pressure, _WHEN_ABSENT_OPERATIONS,
    )

    kind = str(operation["operation"])
    labels = _WHEN_ABSENT_OPERATIONS[kind]
    key = sorted(combined.source_cycles, key=lambda item: (item[0], str(item[1])))[0]
    values = []
    names = []
    for position, label in enumerate(labels):
        name = str(operation[label])
        value = combined.direct.get((key[0], key[1], name))
        if value is None:
            raise ValueError(
                f"terrain_height from surface pressure needs {name!r}, which "
                f"the primary files do not carry at {key[0].isoformat()}")
        axes = ("vertical", "y", "x") if position < 3 else ("y", "x")
        if tuple(value.axes) != axes:
            raise ValueError(
                f"terrain_height from surface pressure needs {name!r} on "
                f"{axes} axes; got {tuple(value.axes)}")
        values.append(value)
        names.append(name)
    factor = {"Pa": 1.0, "hPa": 100.0}[
        str(mapping["coordinates"]["vertical"]["units"])]
    levels = [float(level) * factor for level in combined.vertical_values]
    height, counts = _height_at_surface_pressure(
        levels, *(value.values for value in values))
    source_cycle = values[3].source_cycle
    terrain = _DirectValue(
        name=_EXTERNAL_FIELD, valid_time=key[0], member=key[1],
        source_cycle=source_cycle, axes=("y", "x"), values=height,
        missing_count=0,
        references=tuple(
            reference for value in values for reference in value.references),
    )
    collection = _DecodedCollection(
        latitude=combined.latitude, longitude=combined.longitude,
        vertical_values=combined.vertical_values,
        direct={(key[0], key[1], _EXTERNAL_FIELD): terrain},
        source_cycles={key: source_cycle},
        grid_fingerprint=combined.grid_fingerprint,
    )
    receipt = {
        "operation": kind,
        "why": "the terrain files carry no terrain_height record",
        "valid_time": key[0].isoformat(),
        "fields": names,
        **counts,
    }
    return collection, receipt


def decoded_vertical_ladder(
    bundle: MappedSourceBundle, mapping: Mapping[str, object],
) -> dict[str, object] | None:
    """The declared levels a decode's files did not carry, or ``None``.

    A publication on one of the mapping's ``vertical.era_ladders`` is
    decoded on that ladder rather than refused, so the column the
    preparation builds has fewer levels than ``vertical.levels`` names.
    This is the receipt entry that says so: the declared ladder, the one
    the frames carry, and the levels between them.  ``None`` when the
    frames carry every declared level, which keeps a full-ladder
    preparation's receipt as it was.
    """

    vertical = dict(mapping.get("coordinates") or {}).get("vertical") or {}
    if not vertical.get("era_ladders"):
        # Only a declared era ladder lets a decode carry fewer levels than
        # it declares; any other shortfall was refused by the decoder.
        return None
    declared = [float(value) for value in vertical.get("levels", ())]
    frames = bundle.frames
    if not declared or not len(frames):
        return None
    reader = getattr(frames, "header", None)
    header = reader(0) if callable(reader) else frames[0].header
    atmosphere = header.vertical_coordinates.get("atmosphere")
    if atmosphere is None:
        return None
    decoded = [float(value) for value in atmosphere.level_values]
    carried = set(decoded)
    absent = [level for level in declared if level not in carried]
    if not absent:
        return None
    return {
        "units": str(vertical["units"]),
        "declared_levels": declared,
        "decoded_levels": decoded,
        "absent_levels": absent,
        "absent_level_count": len(absent),
    }


# ---------------------------------------------------------------------------
# As posted: one composition decoded lead batch by lead batch (DESIGN A136
# 2.4 item 3).  Each batch is one ``decode_composed_source`` over the leads
# posted so far and not yet decoded, bound by its own sub-manifest; the
# seal reads every batch as one frameset and writes the whole receipt.
# ---------------------------------------------------------------------------


class PostedCompositionRefusal(ValueError):
    """A composition this window cannot decode lead batch by lead batch."""


class PostedFrames(_ABCSequence):
    """The frames of a window's lead batches, read as one frameset.

    ``valid_times`` are the window's planned times; ``locate(index)``
    returns the batch frames that hold time ``index`` and its position
    there, decoding the batch (and waiting for its leads) first when it has
    not been.  Every reader a composed bundle and its regular snapshots use
    is answered from the batch's own frameset, one valid time at a time.
    """

    def __init__(self, valid_times, locate, *, prefetch=None, decoded=None,
                 posted=None):
        self._valid_times = tuple(valid_times)
        self._locate = locate
        self._prefetch = prefetch
        self._decoded = decoded
        self._posted = posted
        self._parts: list = []

    def prefetch_through(self, index) -> None:
        """Have every time through ``index`` decoded in one batch, if the source batches."""

        if self._prefetch is not None:
            self._prefetch(int(index))

    def decoded_through(self) -> int:
        """How many leading times are decoded now: read with no decode and no wait."""

        return len(self) if self._decoded is None else int(self._decoded())

    def posted_through(self) -> int:
        """How many leading times the source has posted now (its posting schedule).

        Not how many this fetch has moved: reading one may wait for its lead.
        """

        return len(self) if self._posted is None else int(self._posted())

    def __len__(self) -> int:
        return len(self._valid_times)

    @property
    def valid_times(self) -> tuple:
        return self._valid_times

    def add_part(self, frames) -> None:
        self._parts.append(frames)

    @property
    def parts(self) -> tuple:
        return tuple(self._parts)

    @property
    def members(self) -> tuple:
        return tuple(member for part in self._parts
                     for member in getattr(part, "members", ()))

    @property
    def mapping_sha256s(self) -> tuple:
        return tuple(digest for part in self._parts
                     for digest in getattr(part, "mapping_sha256s", ()))

    def _part(self, index):
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError(index)
        return self._locate(position)

    def field_names(self, index) -> tuple:
        frames, local = self._part(index)
        reader = getattr(frames, "field_names", None)
        return (reader(local) if reader is not None
                else tuple(frames[local].fields))

    def field_digest(self, index, name) -> str:
        frames, local = self._part(index)
        reader = getattr(frames, "field_digest", None)
        if reader is not None:
            return reader(local, name)
        return _array_sha256(frames[local].fields[name].values)

    def field_count(self, index) -> int:
        frames, local = self._part(index)
        reader = getattr(frames, "field_count", None)
        return (reader(local) if reader is not None
                else len(frames[local].fields))

    def header(self, index):
        frames, local = self._part(index)
        reader = getattr(frames, "header", None)
        return reader(local) if callable(reader) else frames[local].header

    def coordinates(self, index):
        frames, local = self._part(index)
        reader = getattr(frames, "coordinates", None)
        if reader is not None:
            return reader(local)
        frame = frames[local]
        return frame.latitude, frame.longitude

    def pressure_levels_hpa(self, index):
        frames, local = self._part(index)
        reader = getattr(frames, "pressure_levels_hpa", None)
        if reader is not None:
            return reader(local)
        pressure = frames[local].fields["air_pressure"].values
        return np.median(np.asarray(pressure), axis=(1, 2)) / 100.0

    def field(self, index, name):
        frames, local = self._part(index)
        reader = getattr(frames, "field", None)
        return (reader(local, name) if reader is not None
                else frames[local].fields[name])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position]
                    for position in range(*index.indices(len(self)))]
        frames, local = self._part(index)
        return frames[local]

    def fieldwise_frame(self, index):
        frames, local = self._part(index)
        reader = getattr(frames, "fieldwise_frame", None)
        return reader(local) if reader is not None else frames[local]

    def release_frame(self, frame) -> None:
        for part in self._parts:
            release = getattr(part, "release_frame", None)
            if release is not None:
                release(frame)

    def close(self) -> None:
        for part in self._parts:
            close = getattr(part, "close", None)
            if close is not None:
                close()


#: Alignment receipt keys that list valid times or cycles.  A batch lists
#: its own; the whole decode lists every batch's in order.  A list every
#: batch carries whole (a supplement or donor posted once) is kept once.
_BATCH_TIME_LIST_KEYS = frozenset({
    "matched_primary_valid_times", "broadcast_primary_valid_times",
    "supplement_valid_times", "donor_valid_times", "donor_source_cycles",
})
#: Alignment receipt keys the whole decode takes from its first valid time
#: (``plan_bound_fields`` hashes the first valid time's subset only).
_FIRST_TIME_KEYS = frozenset({"field_subset_sha256"})


def merge_batch_alignment(receipts: Sequence[Mapping[str, object]], *,
                          label: str) -> dict[str, object]:
    """The alignment receipt one decode of every batch's times would write.

    A key every batch agrees on is kept; a time list is the batches' lists
    in batch order (each batch composed only its own times), and must then
    name each time once, in order; a key the receipt takes from the first
    valid time is the first batch's.  Anything else that differs is refused
    by name, because the one-shot receipt could not be written from it.
    """

    first = dict(receipts[0])
    for receipt in receipts[1:]:
        if set(receipt) != set(first):
            raise PostedCompositionRefusal(
                f"the lead batches' {label} receipts carry different keys "
                f"({sorted(set(receipt) ^ set(first))}); one decode of the "
                "window would write one receipt")
    merged: dict[str, object] = {}
    for key in first:
        values = [receipt[key] for receipt in receipts]
        if all(value == values[0] for value in values):
            merged[key] = values[0]
        elif key in _FIRST_TIME_KEYS:
            merged[key] = values[0]
        elif key in _BATCH_TIME_LIST_KEYS and all(
                isinstance(value, list) for value in values):
            joined = [item for value in values for item in value]
            if joined != sorted(set(joined)):
                raise PostedCompositionRefusal(
                    f"the lead batches' {label} {key} overlap or run out of "
                    f"order ({joined}); each batch composes its own times")
            merged[key] = joined
        else:
            raise PostedCompositionRefusal(
                f"the lead batches' {label} receipts disagree on {key!r}, "
                "which one decode of the window writes once; this source "
                "cannot be decoded lead batch by lead batch")
    return merged


def without_earlier_batch_times(bundle: MappedSourceBundle,
                                earlier: MappedSourceBundle
                                ) -> MappedSourceBundle:
    """A lead batch's bundle without the times an earlier batch listed.

    A batch that decoded the window's first lead again, because that lead's
    file alone carries the cycle's step-0 statics
    (``mapped_direct._PostedMappedSource._needs_first_lead``), lists that
    lead's time in its alignment receipts' time lists.  One decode of the
    window lists each time once, so every item the first batch's same list
    holds is left out here, and :func:`merge_batch_alignment` joins the
    lists as it joins any batch's own.  The frames are the batch's own; the
    re-decoded time is simply never located.
    """

    from dataclasses import replace

    def trim(receipt, base):
        trimmed = dict(receipt)
        for key in _BATCH_TIME_LIST_KEYS:
            mine, theirs = trimmed.get(key), base.get(key)
            if isinstance(mine, list) and isinstance(theirs, list):
                trimmed[key] = [item for item in mine if item not in theirs]
        return trimmed

    if len(bundle.contributing_sources) != len(earlier.contributing_sources):
        raise PostedCompositionRefusal(
            "the lead batches decoded different contributing sources")
    return replace(
        bundle,
        alignment_receipt=trim(bundle.alignment_receipt,
                               earlier.alignment_receipt),
        contributing_sources=tuple(
            {**dict(record),
             "alignment": trim(record["alignment"], base["alignment"])}
            for record, base in zip(bundle.contributing_sources,
                                    earlier.contributing_sources)))


def _identical(values, what: str):
    first = values[0]
    for value in values[1:]:
        if value != first:
            raise PostedCompositionRefusal(
                f"the lead batches decoded under different {what}")
    return first


def _role_holding(supplement_files, paths) -> str | None:
    wanted = {str(Path(path).resolve()) for path in paths}
    for role, inventory in supplement_files.items():
        if wanted and wanted <= {str(Path(path).resolve())
                                 for path in inventory}:
            return role
    return None


def posted_composition_bundle(
    batches: Sequence[MappedSourceBundle], frames: PostedFrames, *,
    input_manifest_path: Path, input_manifest_sha256: str,
    supplement_files: Mapping[str, Sequence[Path]],
    shared_primary: bool,
) -> MappedSourceBundle:
    """The bundle one decode of the whole window would have returned.

    ``batches`` are the lead batches' bundles in order and ``frames``
    reads all of their frames as one frameset.  What every batch decoded
    the same (the mapping, composition, decoders, terrain provenance and
    soil contract) is kept, and refused if a batch differs; the input
    manifest is the window's sealed one; the terrain products and each
    contributing source's data are the window's whole inventory, in its
    order, with the digests the batches read; the alignment receipts are
    merged by :func:`merge_batch_alignment`.  ``shared_primary`` says a
    primary object (a cycle-invariant field) was decoded by every batch, in
    which case record alias counts cannot be summed and are refused.
    """

    digests: dict[str, str] = {}
    for batch in batches:
        for path, digest in zip(batch.terrain_data_paths,
                                batch.terrain_data_sha256):
            digests[str(path)] = str(digest)
        for record in batch.contributing_sources:
            for row in record["data"]:
                digests[str(row["path"])] = str(row["sha256"])

    def whole(role: str) -> tuple[tuple[Path, ...], tuple[str, ...]]:
        paths = tuple(Path(path).resolve() for path in supplement_files[role])
        missing = [str(path) for path in paths if str(path) not in digests]
        if missing:
            raise PostedCompositionRefusal(
                f"supplement role {role!r} names {missing}, which no lead "
                "batch decoded")
        return paths, tuple(digests[str(path)] for path in paths)

    first = batches[0]
    terrain_role = _role_holding(supplement_files, first.terrain_data_paths)
    if terrain_role is None:
        raise PostedCompositionRefusal(
            "no supplement role holds the terrain the first lead batch "
            "decoded")
    terrain_paths, terrain_sha = whole(terrain_role)
    contributing = []
    for index, record in enumerate(first.contributing_sources):
        records = [batch.contributing_sources[index] for batch in batches]
        for key in ("binding", "source_id", "mapping", "provenance", "fields"):
            _identical([item[key] for item in records],
                       f"{record['binding']} {key}")
        role = _role_holding(supplement_files,
                             [row["path"] for row in record["data"]])
        if role is None:
            raise PostedCompositionRefusal(
                f"no supplement role holds contributing source "
                f"{record['binding']}'s data")
        paths, shas = whole(role)
        contributing.append({
            **dict(record),
            "data": [{"path": str(path), "sha256": digest}
                     for path, digest in zip(paths, shas)],
            "alignment": merge_batch_alignment(
                [item["alignment"] for item in records],
                label=f"{record['binding']} binding"),
        })
    aliases = [dict(batch.record_aliases or {}) for batch in batches]
    record_aliases = None
    if any(aliases):
        if shared_primary:
            raise PostedCompositionRefusal(
                "the primary's files spell records the way an earlier "
                "publication did, and every lead batch decoded the same "
                "cycle-invariant objects, so the whole window's alias count "
                "cannot be summed from the batches")
        record_aliases = {}
        for counts in aliases:
            for name, count in counts.items():
                record_aliases[name] = record_aliases.get(name, 0) + int(count)
    alignment = merge_batch_alignment(
        [batch.alignment_receipt for batch in batches], label="terrain")
    return MappedSourceBundle(
        frames=frames,
        mapping_path=_identical([b.mapping_path for b in batches], "mappings"),
        mapping_sha256=_identical([b.mapping_sha256 for b in batches],
                                  "mappings"),
        composition_path=_identical([b.composition_path for b in batches],
                                    "compositions"),
        composition_sha256=_identical([b.composition_sha256 for b in batches],
                                      "compositions"),
        input_manifest_path=Path(input_manifest_path),
        input_manifest_sha256=str(input_manifest_sha256),
        decoder_paths=_identical([dict(b.decoder_paths) for b in batches],
                                 "decoders"),
        decoder_sha256=_identical([dict(b.decoder_sha256) for b in batches],
                                  "decoders"),
        terrain_data_paths=terrain_paths,
        terrain_data_sha256=terrain_sha,
        terrain_provenance_path=_identical(
            [b.terrain_provenance_path for b in batches], "terrain provenance"),
        terrain_provenance_sha256=_identical(
            [b.terrain_provenance_sha256 for b in batches],
            "terrain provenance"),
        soil_layer_contract=_identical(
            [dict(b.soil_layer_contract) for b in batches], "soil contracts"),
        alignment_receipt=alignment,
        contributing_sources=tuple(contributing),
        record_aliases=record_aliases,
    )


__all__ = [
    "COMPOSITION_SCHEMA", "INPUT_MANIFEST_SCHEMA", "RECEIPT_SCHEMA",
    "MappedSourceBundle", "PostedCompositionRefusal", "PostedFrames",
    "composition_receipt_binding_matches", "merge_batch_alignment",
    "posted_composition_bundle", "without_earlier_batch_times",
    "composition_receipt_identity", "composition_receipt_identity_sha256",
    "decode_composed_source", "decoded_vertical_ladder",
    "load_composition", "mapped_composition_receipt",
]
