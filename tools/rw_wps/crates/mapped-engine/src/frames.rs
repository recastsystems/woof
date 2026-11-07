//! Canonical frames and the frameset the seam writes.
//!
//! Port of `mapped_source._materialize_frames` and `_frame_header`, plus
//! the `gpuwm-mapped-frameset-v1` writer.  The frameset is the ABI: its
//! schema name is the compiled ABI marker, so anything that changes the
//! shape below changes the marker (design doc §3.4).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

use chrono::NaiveDateTime;
use serde_json::{json, Map, Value};

use crate::array;
use crate::assemble::DecodedCollection;
use crate::derive::{
    complete_geopotential_height, completed_fields, derivation_dependencies, derivation_shape_operand,
    evaluate_derivation, CanonicalField, CanonicalFields, HYPSOMETRIC_OPERANDS,
};
use crate::model::{FieldSpec, Mapping, GRID_FAMILY_LAMBERT, GRID_FAMILY_ROTATED, PROJECTED_AXIS_UNIT_M};
use crate::refusal::{frame_invalid, mapping_invalid, Result};

/// One completed field as a canonical field; `None` while an operand of
/// its completion is absent.
fn complete_field(
    field: &FieldSpec<'_>,
    seed: Option<&CanonicalField>,
    operands: &impl CanonicalFields,
) -> Result<Option<CanonicalField>> {
    let name = field.name.as_str();
    let units = field.units_target()?;
    let completed = match name {
        "geopotential_height" => complete_geopotential_height(seed, operands, name, units)?,
        other => unreachable!("{other} has no completion"),
    };
    let Some((values, axes, source_references)) = completed else {
        return Ok(None);
    };
    let missing_count = array::count_nan(&values);
    Ok(Some(CanonicalField {
        name: name.to_owned(),
        units: units.to_owned(),
        axes,
        location: field.location()?.to_owned(),
        staggering: field.staggering().to_owned(),
        values,
        missing_count,
        source_references,
    }))
}

/// `gpuwm.source_frame.SCHEMA`.
pub const SOURCE_FRAME_SCHEMA: &str = "gpuwm-canonical-source-frame-v1";

/// One materialized frame: `mapped_source.MappedSourceFrame`.
pub struct MappedSourceFrame {
    pub valid_time: NaiveDateTime,
    pub member: Option<String>,
    pub source_cycle: NaiveDateTime,
    pub latitude: Vec<f64>,
    pub longitude: Vec<f64>,
    pub vertical_kind: String,
    pub vertical_units: String,
    pub vertical_values: Vec<f64>,
    /// Fields in the mapping's declared order.
    pub fields: Vec<CanonicalField>,
    pub header: Value,
}

/// Python's `datetime.isoformat()` on a naive datetime.
pub fn naive_isoformat(value: NaiveDateTime) -> String {
    value.format("%Y-%m-%dT%H:%M:%S").to_string()
}

/// Python's `datetime.replace(tzinfo=utc).isoformat()`.
fn utc_isoformat(value: NaiveDateTime) -> String {
    format!("{}+00:00", naive_isoformat(value))
}

/// The frame keys a frameset will carry, plus the mapping-level facts
/// every frame is built against.
///
/// Split out of `materialize_frames` so a frameset can be MATERIALIZED
/// AND WRITTEN one valid time at a time.  Named breakage, measured on
/// real RRFS bytes (3 km CONUS, 45 pressure levels): building every
/// frame before writing any of them held the decoded collection and a
/// full second copy of every valid time's arrays at once, so the engine
/// peaked at 9.0 GiB per forcing time and a seven-time compose needed
/// 67 GiB of host memory.  Nothing about the frameset needs two frames
/// resident: the stream is written sequentially, and each frame's
/// digests cover only its own arrays.
pub struct FramePlan {
    /// Valid time and member, in the order the frameset writes them.
    pub keys: Vec<(NaiveDateTime, Option<String>)>,
    /// The fields a frame publishes, in the mapping's declared order:
    /// every declared field but those marked `dependency_only`, which are
    /// decoded for a derivation to read and then released unwritten.
    pub declared_names: Vec<String>,
    /// The names a frame must carry to initialize WRF.
    pub required_names: BTreeSet<String>,
}

/// Why the target contract refuses a uniform series `observed` seconds
/// apart, or `None` -- `source_authorities.boundary_interval_refusal`.
///
/// `boundary_interval_seconds` is the publisher's own spacing.  A target
/// that also declares `accept_boundary_interval_multiples` takes any whole
/// multiple of it: each valid time of such a series is one the publisher
/// wrote, and the mapping grammar has no time-window statistic whose value
/// would depend on the spacing.  Without the key the declared spacing is
/// the only one taken.
pub fn boundary_interval_refusal(target: &crate::node::Node, observed: i64) -> Option<String> {
    let declared = target
        .field("boundary_interval_seconds")
        .and_then(crate::node::Node::as_i64);
    let multiples = target
        .field("accept_boundary_interval_multiples")
        .and_then(crate::node::Node::as_bool)
        .unwrap_or(false);
    match declared {
        Some(spacing) if multiples && spacing > 0 => {
            if observed > 0 && observed % spacing == 0 {
                None
            } else {
                Some(format!(
                    "mapped cadence {observed} seconds is not a whole multiple of \
                     the {spacing} seconds the target contract declares"
                ))
            }
        }
        Some(spacing) if spacing == observed => None,
        _ => Some(format!(
            "mapped cadence {observed} seconds differs from target contract {declared:?}"
        )),
    }
}

/// Every check a frameset can make before reading a valid time's arrays.
///
/// Takes the SERIES CLOCK rather than a decoded collection: every check
/// below reads `source_cycles` and the mapping, and a streamed decode
/// knows its whole clock from the inventory octets before it has
/// assembled one valid time's arrays.
pub fn plan_frames(
    mapping: &Mapping,
    source_cycles: &BTreeMap<(NaiveDateTime, Option<String>), NaiveDateTime>,
) -> Result<FramePlan> {
    plan_frames_within(mapping, source_cycles, false)
}

/// [`plan_frames`] for one lead batch of a window when `lead_batch` is
/// set: a batch may hold one time, and its cadence is the window's, so
/// the two lateral-boundary series rules (at least two times, one
/// uniform cadence matching the target's boundary interval) are left to
/// the caller, which holds them over every batch's times together.  Named
/// breakage: an as-posted preparation decodes each lead as it posts, and
/// a one-lead batch was refused as a one-time forcing series.
pub fn plan_frames_within(
    mapping: &Mapping,
    source_cycles: &BTreeMap<(NaiveDateTime, Option<String>), NaiveDateTime>,
    lead_batch: bool,
) -> Result<FramePlan> {
    // A field marked `dependency_only` is an input to a derivation and
    // nothing else, so the frame stream does not carry it.  Named
    // breakage: ICON-D2's six raw mass fractions were written beside the
    // mixing ratios rebased from them, 390 of 1,217 layers per valid time
    // and about 16 GB of the 50 GB a 48 h 1 km run stages, read by nothing.
    let mut declared_names: Vec<String> = Vec::new();
    let mut unbound: Vec<String> = Vec::new();
    for field in mapping.fields()? {
        if field.provider() == Some("composition_bound") {
            unbound.push(field.name.clone());
        }
        if !field.dependency_only()? {
            declared_names.push(field.name.clone());
        }
    }
    if !unbound.is_empty() {
        unbound.sort();
        let unbound = crate::refusal::python_list_repr(&unbound);
        return Err(frame_invalid(format!(
            "fields {unbound} are composition_bound: this mapping cannot \
             materialize alone, because those values live in another packaged \
             source's decode; a cross-source composition must bind each of \
             them to a contributing source"
        )));
    }
    let keys: Vec<&(NaiveDateTime, Option<String>)> = source_cycles.keys().collect();
    if keys.is_empty() {
        return Err(frame_invalid("mapped source has no valid times"));
    }
    let members: BTreeSet<&Option<String>> = keys.iter().map(|(_time, member)| member).collect();
    if members.len() != 1 {
        return Err(frame_invalid(
            "mapped WRF initialization requires exactly one member",
        ));
    }
    let required_names: BTreeSet<String> = mapping.required_field_names()?.into_iter().collect();
    for name in &required_names {
        if mapping.field(name)?.optional()? {
            return Err(mapping_invalid(format!("required field {name} cannot be optional")));
        }
    }
    // A required field held off the stream would be written nowhere while
    // every check that it was derived still passed, so the reader would
    // meet a frame without it.
    let mut withheld: Vec<&String> = required_names
        .iter()
        .filter(|name| !declared_names.contains(name))
        .filter(|name| mapping.field(name).and_then(|field| field.dependency_only()).unwrap_or(false))
        .collect();
    if !withheld.is_empty() {
        withheld.sort();
        return Err(mapping_invalid(format!(
            "fields {} are required by the target and marked dependency_only; \
             a frame must publish every required field",
            crate::refusal::python_list_repr(&withheld)
        )));
    }
    let times: Vec<NaiveDateTime> = keys.iter().map(|(time, _member)| *time).collect();
    let mut sorted = times.clone();
    sorted.sort();
    sorted.dedup();
    if sorted != times {
        return Err(frame_invalid(
            "mapped forcing times are not unique and increasing",
        ));
    }
    let target = mapping.target()?;
    if !lead_batch
        && target
            .field("require_lateral_boundaries")
            .and_then(crate::node::Node::as_bool)
            .unwrap_or(false)
    {
        if times.len() < 2 {
            // The owned class, not frame_invalid: the Python engine
            // raises ForcingSeriesRefusal for this exact condition, and
            // the refusal contract maps class -> exception 1:1.
            return Err(crate::refusal::forcing_series(
                "mapped lateral-boundary forcing requires at least two times",
            ));
        }
        let deltas: BTreeSet<i64> = times
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).num_seconds())
            .collect();
        if deltas.len() != 1 || *deltas.iter().next().expect("one delta") <= 0 {
            return Err(frame_invalid(
                "mapped forcing cadence must be positive and uniform",
            ));
        }
        let observed = *deltas.iter().next().expect("one delta");
        if let Some(message) = boundary_interval_refusal(target, observed) {
            return Err(frame_invalid(message));
        }
    }
    Ok(FramePlan {
        keys: keys.into_iter().cloned().collect(),
        declared_names,
        required_names,
    })
}

/// ONE valid time's canonical frame -- `mapped_source._materialize_frames`
/// for a single key.
pub fn materialize_frame(
    mapping: &Mapping,
    collection: &DecodedCollection,
    plan: &FramePlan,
    valid_time: NaiveDateTime,
    member: &Option<String>,
) -> Result<MappedSourceFrame> {
    let declared_names = &plan.declared_names;
    let required_names = &plan.required_names;
    let mut available: BTreeMap<String, CanonicalField> = BTreeMap::new();
    for ((time, member_value, field_name), direct) in &collection.direct {
        if *time != valid_time || member_value != member {
            continue;
        }
        let field = mapping.field(field_name)?;
        available.insert(
            field_name.clone(),
            CanonicalField {
                name: field_name.clone(),
                units: field.units_target()?.to_owned(),
                axes: direct.axes.clone(),
                location: field.location()?.to_owned(),
                staggering: field.staggering().to_owned(),
                values: direct.values.clone(),
                missing_count: direct.missing_count,
                source_references: direct.references.clone(),
            }
            .validated()?,
        );
    }
    let mut pending: BTreeSet<String> = BTreeSet::new();
    for field in mapping.fields()? {
        if field.derivation().is_some() {
            pending.insert(field.name.clone());
        }
    }
    // A field the frame completes, which the source published at only
    // some of its values or at none, is held back from every derivation
    // until it is whole; `seeds` keeps the values the source did publish.
    let completed_names = completed_fields(mapping.vertical_kind()?, required_names);
    let mut seeds: BTreeMap<&str, Option<CanonicalField>> = BTreeMap::new();
    for name in &completed_names {
        if pending.contains(*name)
            || available.get(*name).is_some_and(|field| field.missing_count == 0)
        {
            continue;
        }
        seeds.insert(*name, available.remove(*name));
    }
    let waits_on = |pending: &BTreeSet<String>| {
        HYPSOMETRIC_OPERANDS.iter().any(|(operand, _)| pending.contains(*operand))
    };
    while !pending.is_empty() || !seeds.is_empty() {
        let mut progress = false;
        for name in pending.clone() {
            let field = mapping.field(&name)?;
            let derivation_name = field.derivation().expect("pending entries are derived");
            let operation = mapping.derivation(derivation_name).ok_or_else(|| {
                mapping_invalid(format!(
                    "field {name} names unknown derivation '{derivation_name}'"
                ))
            })?;
            let Some((values, axes, references)) = evaluate_derivation(
                operation,
                &available,
                collection,
                &field,
                &name,
                mapping.vertical()?,
            )?
            else {
                continue;
            };
            let missing_count = array::count_nan(&values);
            available.insert(
                name.clone(),
                CanonicalField {
                    name: name.clone(),
                    units: field.units_target()?.to_owned(),
                    axes,
                    location: field.location()?.to_owned(),
                    staggering: field.staggering().to_owned(),
                    values,
                    missing_count,
                    source_references: references,
                }
                .validated()?,
            );
            pending.remove(&name);
            progress = true;
        }
        // A completion reads the final value of each operand, so it waits
        // while a derivation still has one to produce.
        for name in &completed_names {
            if !seeds.contains_key(*name) || waits_on(&pending) {
                continue;
            }
            let field = mapping.field(name)?;
            if let Some(completed) = complete_field(&field, seeds[*name].as_ref(), &available)? {
                available.insert((*name).to_owned(), completed.validated()?);
                seeds.remove(*name);
                progress = true;
            }
        }
        if !progress {
            // A completed field whose own derivation cannot run is
            // derived hydrostatically instead, and the loop goes on for
            // anything that reads it.
            for name in &completed_names {
                if !pending.contains(*name) || waits_on(&pending) {
                    continue;
                }
                let field = mapping.field(name)?;
                if let Some(completed) = complete_field(&field, None, &available)? {
                    available.insert((*name).to_owned(), completed.validated()?);
                    pending.remove(*name);
                    progress = true;
                }
            }
        }
        if !progress {
            if !pending.is_empty() {
                let stuck: Vec<String> = pending.iter().cloned().collect();
                return Err(frame_invalid(format!(
                    "derived fields have missing dependencies or a cycle: {}",
                    stuck.join(", ")
                )));
            }
            // What is left is a completion whose operands this source
            // does not carry: the frame lacks that field, and says so by
            // name below.
            break;
        }
    }
    // The frame states its fields in the mapping's own declared order:
    // assembly order is the PRODUCER's record layout, and a broadcast
    // invariant lands after the per-time records, so two frames with
    // identical field SETS could otherwise disagree about sequence.
    // MOVED out of `available`, not cloned out of it.  `available`
    // is this valid time's own map and nothing below reads it
    // again, so cloning made a second full copy of every array in
    // the frame -- on a 3 km CONUS source with a 45-level ladder
    // that is 4.2 GiB per valid time of duplicate, live at the
    // moment the next valid time starts assembling.  Same fields,
    // same order, same bytes.
    let ordered: Vec<CanonicalField> = declared_names
        .iter()
        .filter_map(|name| available.remove(name))
        .collect();
    let present: BTreeSet<&str> = ordered.iter().map(|field| field.name.as_str()).collect();
    let missing: Vec<&String> = required_names
        .iter()
        .filter(|name| !present.contains(name.as_str()))
        .collect();
    if !missing.is_empty() {
        let missing = crate::refusal::python_list_repr(&missing);
        return Err(frame_invalid(format!(
            "mapped frame at {valid_time} lacks required fields {missing}"
        )));
    }
    for field in &ordered {
        let finite_required = required_names.contains(&field.name)
            && field.name != "soil_temperature"
            && field.name != "volumetric_soil_moisture";
        if finite_required && field.values.iter().any(|value| !value.is_finite())
            && mapping.field(&field.name)?.missing_kind()? != "preserve_mask"
        {
            return Err(frame_invalid(format!(
                "required mapped field {} is not finite at {valid_time}",
                field.name
            )));
        }
    }
    if let Some(soil_count) = mapping.soil_layer_count()?.filter(|count| *count > 0) {
        for name in ["soil_temperature", "volumetric_soil_moisture"] {
            let field = ordered
                .iter()
                .find(|field| field.name == name)
                .ok_or_else(|| {
                    frame_invalid(format!("mapped frame at {valid_time} lacks {name}"))
                })?;
            let axis = field
                .axes
                .iter()
                .position(|axis| axis == "soil")
                .ok_or_else(|| frame_invalid(format!("{name} has no soil axis")))?;
            let observed = field.values.shape()[axis] as i64;
            if observed != soil_count {
                return Err(frame_invalid(format!(
                    "{name} has {observed} layers, target declares {soil_count}"
                )));
            }
        }
    }
    let source_cycle = collection.source_cycles[&(valid_time, member.clone())];
    validate_frame_axes(collection, &ordered)?;
    let header = frame_header(
        mapping,
        valid_time,
        source_cycle,
        collection,
        &ordered,
    )?;
    require_wrf_initial_state(&header)?;
    Ok(MappedSourceFrame {
        valid_time,
        member: member.clone(),
        source_cycle,
        latitude: collection.latitude.clone(),
        longitude: collection.longitude.clone(),
        vertical_kind: mapping
            .vertical()?
            .get("kind")
            .and_then(crate::node::Node::as_str)
            .unwrap_or_default()
            .to_owned(),
        vertical_units: mapping
            .vertical()?
            .get("units")
            .and_then(crate::node::Node::as_str)
            .unwrap_or_default()
            .to_owned(),
        vertical_values: collection.vertical_values.clone(),
        fields: ordered,
        header,
    })
}

/// One field inventory across the whole series, from the names alone.
///
/// Collected as each frame is written rather than from a held set of
/// frames: the names are a few hundred bytes per valid time, and a
/// frameset that fails this is scratch that is deleted whole, so no
/// caller ever sees a partially written stream.
pub fn require_one_inventory(inventories: &BTreeSet<Vec<String>>) -> Result<()> {
    if inventories.len() != 1 {
        return Err(frame_invalid(
            "mapped field inventory changes between valid times",
        ));
    }
    Ok(())
}

/// The whole series at once -- `inspect` and the goldens want this.
///
/// A frameset WRITE does not: see `write_frameset`, which pulls one
/// decoded time and materializes its canonical fields on demand.
pub fn materialize_frames(
    mapping: &Mapping,
    collection: &DecodedCollection,
) -> Result<Vec<MappedSourceFrame>> {
    let plan = plan_frames(mapping, &collection.source_cycles)?;
    let mut frames = Vec::with_capacity(plan.keys.len());
    let mut inventories: BTreeSet<Vec<String>> = BTreeSet::new();
    for (valid_time, member) in &plan.keys {
        let frame = materialize_frame(mapping, collection, &plan, *valid_time, member)?;
        inventories.insert(frame.fields.iter().map(|f| f.name.clone()).collect());
        frames.push(frame);
    }
    require_one_inventory(&inventories)?;
    Ok(frames)
}

/// `MappedSourceFrame.__post_init__`'s coordinate and grid-sharing checks.
fn validate_frame_axes(collection: &DecodedCollection, fields: &[CanonicalField]) -> Result<()> {
    for (name, axis) in [
        ("latitude", &collection.latitude),
        ("longitude", &collection.longitude),
        ("vertical", &collection.vertical_values),
    ] {
        if axis.is_empty() || axis.iter().any(|value| !value.is_finite()) {
            return Err(frame_invalid(format!(
                "mapped {name} coordinate must be finite non-empty 1-D"
            )));
        }
    }
    if collection.latitude.len() < 2 || collection.longitude.len() < 2 {
        return Err(frame_invalid("mapped horizontal grid must be at least 2x2"));
    }
    for (name, axis) in [
        ("latitude", &collection.latitude),
        ("longitude", &collection.longitude),
    ] {
        let differences: Vec<f64> = axis.windows(2).map(|pair| pair[1] - pair[0]).collect();
        let increasing = differences.iter().all(|value| *value > 0.0);
        let decreasing = differences.iter().all(|value| *value < 0.0);
        if !increasing && !decreasing {
            return Err(frame_invalid(format!(
                "mapped {name} coordinate is not strictly monotonic"
            )));
        }
        let first = differences[0];
        // `np.allclose(difference, difference[0], rtol=1e-10, atol=1e-12)`.
        if differences
            .iter()
            .any(|value| (value - first).abs() > 1e-12 + 1e-10 * first.abs())
        {
            return Err(frame_invalid(format!(
                "mapped {name} coordinate is not a regular axis; curvilinear \
                 export is not enabled"
            )));
        }
    }
    for field in fields {
        let shape: BTreeMap<&str, usize> = field
            .axes
            .iter()
            .map(String::as_str)
            .zip(field.values.shape().iter().copied())
            .collect();
        if shape.get("y") != Some(&collection.latitude.len())
            || shape.get("x") != Some(&collection.longitude.len())
        {
            return Err(frame_invalid(format!(
                "{} does not share the source horizontal grid",
                field.name
            )));
        }
        if let Some(levels) = shape.get("half_level")
            && *levels != collection.vertical_values.len() + 1 {
                return Err(frame_invalid(format!("{} does not bound the vertical coordinate", field.name)));
            }
        if let Some(levels) = shape.get("vertical")
            && *levels != collection.vertical_values.len() {
                return Err(frame_invalid(format!(
                    "{} does not share the vertical coordinate",
                    field.name
                )));
            }
    }
    Ok(())
}

/// One canonical descriptor, shared by the whole-frame oracle and the
/// field writer. Its digest always covers the full original field.
fn field_descriptor(field: &CanonicalField, time: &Value, digest: &str) -> Value {
    json!({
        "canonical_name": field.name,
        "units": field.units,
        "dimensions": field.axes,
        "grid_location": field.location,
        "vertical_coordinate": if field.axes.iter().any(|axis| axis == "vertical") {
            Value::String("atmosphere".to_owned())
        } else if field.axes.iter().any(|axis| axis == "soil") {
            Value::String("soil".to_owned())
        } else {
            Value::Null
        },
        "time": time,
        "data_reference": format!("sha256:{digest}"),
        "dtype": "<f8",
        "shape": field.values.shape(),
        "missing_value_policy": if field.missing_count > 0 {
            "explicit_missing"
        } else {
            "reject_nonfinite"
        },
        "source_field": field.source_references.join(";"),
    })
}

/// `mapped_source._frame_header`.
fn frame_header(
    mapping: &Mapping,
    valid_time: NaiveDateTime,
    source_cycle: NaiveDateTime,
    collection: &DecodedCollection,
    fields: &[CanonicalField],
) -> Result<Value> {
    let vertical = mapping.vertical()?;
    let vertical_kind = vertical
        .get("kind")
        .and_then(crate::node::Node::as_str)
        .unwrap_or_default();
    let coordinate = match vertical_kind {
        "hybrid_sigma_pressure" => "hybrid",
        "embedded_levels" => "model_level",
        other => other,
    };
    let mut vertical_coordinates = Map::new();
    vertical_coordinates.insert(
        "atmosphere".to_owned(),
        json!({
            "coordinate": coordinate,
            "level_count": collection.vertical_values.len(),
            "level_values": collection.vertical_values,
            // The resolved hybrid ladder (empty on every other vertical
            // kind), the tuples both engines used to hard-code empty,
            // which validate_source_frame rightly refused.
            "a_coefficients": collection.hybrid_a,
            "b_coefficients": collection.hybrid_b,
            "positive": vertical
                .field("positive")
                .and_then(crate::node::Node::as_str)
                .unwrap_or("down"),
            "units": vertical
                .get("units")
                .and_then(crate::node::Node::as_str)
                .unwrap_or_default(),
        }),
    );
    if let Some(soil) = fields.iter().find(|field| field.axes.iter().any(|axis| axis == "soil")) {
        let axis = soil
            .axes
            .iter()
            .position(|axis| axis == "soil")
            .expect("soil axis proven present");
        vertical_coordinates.insert(
            "soil".to_owned(),
            json!({
                "coordinate": "soil_depth",
                "level_count": soil.values.shape()[axis],
                "level_values": Vec::<f64>::new(),
                "a_coefficients": Vec::<f64>::new(),
                "b_coefficients": Vec::<f64>::new(),
                "positive": "down",
                "units": "index",
            }),
        );
    }
    let time = json!({
        "reference_time": utc_isoformat(source_cycle),
        "valid_time": utc_isoformat(valid_time),
        "lead_seconds": (valid_time - source_cycle).num_seconds(),
        "statistic": "instantaneous",
        "interval_start": Value::Null,
        "interval_end": Value::Null,
        "accumulation_reset": Value::Null,
    });
    // One descriptor per field, hashed CONCURRENTLY.  sha256 over a
    // field's bytes is the whole cost here and the fields are
    // independent; the indexed collect puts field i's descriptor at
    // index i whatever order the threads finished in, so the header --
    // and therefore its digest -- is the serial engine's.
    let field_descriptors: Vec<Value> = crate::threads::install(|| {
        use rayon::prelude::*;
        fields.par_iter().map(|field| {
            let flat = array::contiguous(&field.values);
            field_descriptor(field, &time, &crate::digest::array_sha256(field.values.shape(), &flat))
        }).collect()
    });
    let declaration = mapping.grid_declaration()?;
    let grid = source_grid(&declaration, &collection.latitude, &collection.longitude);
    let policies = mapping
        .target()?
        .field("initialization_policies")
        .map(crate::node::Node::to_value)
        .unwrap_or_else(|| Value::Object(Map::new()));
    Ok(json!({
        "source_id": mapping.name()?,
        "source_cycle": utc_isoformat(source_cycle),
        "grid": grid,
        "vertical_coordinates": Value::Object(vertical_coordinates),
        "fields": field_descriptors,
        "initialization_policies": policies,
        "schema": SOURCE_FRAME_SCHEMA,
    }))
}

/// The frame header's grid document for the declared source grid over the
/// canonical axes: the same document whether it is written into a frame or
/// sent ahead of the decode with an atmospheric window request.
pub fn source_grid(
    declaration: &crate::model::GridDeclaration,
    latitude: &[f64],
    longitude: &[f64],
) -> Value {
    if declaration.family == GRID_FAMILY_LAMBERT {
        let parameters = declaration
            .parameters
            .as_ref()
            .expect("lambert declarations carry parameters");
        let mut grid = json!({
            "projection": GRID_FAMILY_LAMBERT,
            "nx": longitude.len(),
            "ny": latitude.len(),
            "earth_shape": format!("grib_shape_of_earth:{}", parameters.shape_of_earth),
            "scan_order": "+x,+y",
            // Grid-relative sources were rotated to the earth basis at decode
            // time; the frame states what its arrays ARE, not what the
            // producer published.
            "wind_basis": "earth_relative",
            "parameters": {
                "axis_unit_m": PROJECTED_AXIS_UNIT_M,
                "dx_m": parameters.dx_m,
                "dy_m": parameters.dy_m,
                "earth_radius_m": parameters.earth_radius_m,
                "lat1": parameters.lat1,
                "latin1": parameters.latin1,
                "latin2": parameters.latin2,
                "lon1": parameters.lon1,
                "lov": parameters.lov,
                "nx": parameters.nx,
                "ny": parameters.ny,
                "shape_of_earth": parameters.shape_of_earth,
                "source_wind_basis": declaration.wind_basis,
            },
        });
        if let Some(pairing) = declaration.same_grid_pairing.as_ref() {
            grid["parameters"]["same_grid_pairing"] = json!(pairing);
        }
        grid
    } else if let Some(parameters) = declaration.rotated.as_ref() {
        // The rotated lattice states its own GRIB2 numbers plus the
        // plane spacings every projected consumer reads (dx_m, dy_m,
        // earth_radius_m, axis_unit_m), so pairing, soil meshes and
        // coverage refusals need no family-specific reader.
        json!({
            "projection": GRID_FAMILY_ROTATED,
            "nx": longitude.len(),
            "ny": latitude.len(),
            "earth_shape": format!("grib_shape_of_earth:{}", parameters.shape_of_earth),
            "scan_order": "+x,+y",
            "wind_basis": "earth_relative",
            "parameters": {
                "axis_unit_m": PROJECTED_AXIS_UNIT_M,
                "dlat_deg": parameters.dlat_deg,
                "dlon_deg": parameters.dlon_deg,
                "dx_m": parameters.dx_m(),
                "dy_m": parameters.dy_m(),
                "earth_radius_m": parameters.earth_radius_m,
                "lat1": parameters.lat1,
                "lon1": parameters.lon1,
                "nx": parameters.nx,
                "ny": parameters.ny,
                "rotation_angle": parameters.rotation_angle,
                "shape_of_earth": parameters.shape_of_earth,
                "source_wind_basis": declaration.wind_basis,
                "south_pole_lat": parameters.south_pole_lat,
                "south_pole_lon": parameters.south_pole_lon,
            },
        })
    } else {
        regular_latlon_grid(latitude, longitude)
    }
}

/// The frame header's grid document for a regular latitude/longitude
/// source: the same document whether it is written into a frame or sent
/// ahead of the decode with an atmospheric window request.
pub fn regular_latlon_grid(latitude: &[f64], longitude: &[f64]) -> Value {
    json!({
        "projection": "regular_latitude_longitude",
        "nx": longitude.len(),
        "ny": latitude.len(),
        "earth_shape": "source_metadata_bound",
        "scan_order": format!(
            "{},{}",
            if longitude[longitude.len() - 1] > longitude[0] { "+x" } else { "-x" },
            if latitude[latitude.len() - 1] > latitude[0] { "+y" } else { "-y" },
        ),
        "wind_basis": "earth_relative",
        "parameters": {
            "latitude_first": latitude[0],
            "latitude_last": latitude[latitude.len() - 1],
            "longitude_first": longitude[0],
            "longitude_last": longitude[longitude.len() - 1],
        },
    })
}

/// Values per pass when encoding the frame stream: 1 MiB of f64.
const STREAM_CHUNK: usize = 128 * 1024;

/// `gpuwm.source_frame.REQUIRED_3D_FIELDS`.
const REQUIRED_3D_FIELDS: [&str; 5] = [
    "air_temperature",
    "eastward_wind",
    "geopotential_height",
    "northward_wind",
    "specific_humidity",
];

/// `gpuwm.source_frame.REQUIRED_SURFACE_FIELDS`.
const REQUIRED_SURFACE_FIELDS: [&str; 10] = [
    "air_temperature_2m",
    "eastward_wind_10m",
    "land_fraction",
    "northward_wind_10m",
    "skin_temperature",
    "soil_temperature",
    "specific_humidity_2m",
    "surface_pressure",
    "terrain_height",
    "volumetric_soil_moisture",
];

/// `gpuwm.source_frame.POLICY_CONTROLLED_FIELDS`.
const POLICY_CONTROLLED_FIELDS: [&str; 9] = [
    "cloud_ice_mixing_ratio",
    "cloud_water_mixing_ratio",
    "graupel_or_hail_mixing_ratio",
    "rain_water_mixing_ratio",
    "sea_ice_fraction",
    "snow_depth",
    "snow_mixing_ratio",
    "snow_water_equivalent",
    "vertical_velocity",
];

/// `gpuwm.source_frame.validate_source_frame_header`'s WRF-initial-state
/// block, run on the header this engine just built.
///
/// The Python engine validates the header at construction, so a frame it
/// cannot initialize WRF from is refused THERE, before the next frame is
/// even reached.  Without this the engine ran on and refused a LATER
/// frame for a different reason, which is not the same answer: on a real
/// staged analysis whose f00 carries the 3-D state and whose f06 carries
/// only surface records, Python refused f00 naming the absent surface
/// fields and this engine refused f06 naming the absent 3-D fields.  Both
/// refuse, so nothing wrong was ever produced, but a caller reading the
/// message would be sent to the wrong end of the source.
fn require_wrf_initial_state(header: &Value) -> Result<()> {
    let names: BTreeSet<&str> = header
        .get("fields")
        .and_then(Value::as_array)
        .map(|fields| {
            fields
                .iter()
                .filter_map(|field| field.get("canonical_name").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();
    let mut missing: BTreeSet<&str> = REQUIRED_3D_FIELDS
        .iter()
        .chain(REQUIRED_SURFACE_FIELDS.iter())
        .copied()
        .filter(|name| !names.contains(name))
        .collect();
    // Pressure may be stated as a field or implied by a hybrid vertical
    // coordinate; one of the two must be there.
    let hybrid = header
        .get("vertical_coordinates")
        .and_then(Value::as_object)
        .map(|coordinates| {
            coordinates.values().any(|vertical| {
                vertical.get("coordinate").and_then(Value::as_str) == Some("hybrid")
            })
        })
        .unwrap_or(false);
    if !names.contains("air_pressure") && !hybrid {
        missing.insert("air_pressure_or_hybrid_coordinate");
    }
    if !missing.is_empty() {
        let missing: Vec<&str> = missing.into_iter().collect();
        return Err(frame_invalid(format!(
            "source frame is incomplete for WRF initialization: {}",
            missing.join(", ")
        )));
    }
    let policies: BTreeSet<&str> = header
        .get("initialization_policies")
        .and_then(Value::as_object)
        .map(|policies| policies.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let missing_policies: Vec<&str> = POLICY_CONTROLLED_FIELDS
        .iter()
        .copied()
        .filter(|name| !names.contains(name) && !policies.contains(name))
        .collect();
    if !missing_policies.is_empty() {
        return Err(frame_invalid(format!(
            "missing explicit initialization policy for absent fields: {}",
            missing_policies.join(", ")
        )));
    }
    Ok(())
}

/// One axis as `mapped_engine_bridge._axis_document`.
///
/// The numbers ride the JSON, and the sha256 of their little-endian
/// `<f8` bytes rides beside them, so a reader can tell a grid that was
/// re-parsed exactly from one that a JSON round trip moved by an ulp.
/// A bare array could not: the values would still look like an axis.
pub(crate) fn axis_document(values: &[f64]) -> Value {
    let mut payload = Vec::with_capacity(values.len() * 8);
    for value in values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    json!({
        "values": values,
        "count": values.len(),
        "sha256": crate::digest::bytes_sha256(&payload),
    })
}

/// The series facts a frameset states once, above its frames.
///
/// A streamed decode knows all of them before it has assembled a single
/// valid time: the clock comes from the inventory octets and the grid
/// fingerprint from the first slice.
#[derive(Debug, Clone)]
pub struct SeriesSummary {
    pub source_cycles: BTreeMap<(NaiveDateTime, Option<String>), NaiveDateTime>,
    pub grid_fingerprint: String,
    /// The series is one lead batch of a window (`compose --lead-batch`):
    /// the window's series rules are held by the caller across batches.
    pub lead_batch: bool,
}

/// A field's input versions. Recording versions preserves the original
/// fixpoint evaluator even when a direct field seeds a derived field of
/// the same name; the later value must not become its own dependency.
struct FieldStep {
    name: String,
    derivation: Option<String>,
    /// Derived hydrostatically where the source leaves the field out;
    /// a dependency named like the step itself is the source's own
    /// partial field.
    completion: bool,
    dependencies: Vec<(String, usize)>,
}

/// Writer-only ownership of one decoded time. Direct arrays move out of
/// the decoder collection on first use. Derived arrays live only until
/// their last derivation or publication consumer; the writer never asks
/// for a second complete canonical frame alongside the source arrays.
struct FieldMaterializer<'a> {
    mapping: &'a Mapping,
    collection: DecodedCollection,
    key: (NaiveDateTime, Option<String>),
    steps: Vec<FieldStep>,
    output: Vec<usize>,
    remaining: Vec<usize>,
    available: BTreeMap<usize, CanonicalField>,
}

impl<'a> FieldMaterializer<'a> {
    fn new(
        mapping: &'a Mapping,
        collection: DecodedCollection,
        plan: &FramePlan,
        key: &(NaiveDateTime, Option<String>),
    ) -> Result<Self> {
        let mut steps = Vec::new();
        let mut versions = BTreeMap::new();
        // Validate every direct input, including one subsequently
        // replaced by a derivation. Publication never hides a bad input.
        // Each input is a full scan of its values, independent of every
        // other, so they are checked concurrently and drained in the
        // collection's order: the refusal is the first one in that order.
        let inputs: Vec<(&String, &crate::assemble::DirectValue)> = collection.direct.iter()
            .filter(|((time, member, _), _)| *time == key.0 && *member == key.1)
            .map(|((_, _, name), direct)| (name, direct))
            .collect();
        let checks: Vec<Result<()>> = crate::threads::install(|| {
            use rayon::prelude::*;
            inputs.par_iter().map(|(name, direct)| {
                let field = mapping.field(name)?;
                field.units_target()?;
                field.location()?;
                CanonicalField::validate_values(name, &direct.axes, &direct.values, direct.missing_count)
            }).collect()
        });
        crate::threads::in_order(checks)?;
        for (name, _direct) in inputs {
            versions.insert(name.clone(), steps.len());
            steps.push(FieldStep { name: name.clone(), derivation: None, completion: false,
                dependencies: Vec::new() });
        }
        let mut pending: BTreeSet<String> = mapping.fields()?.into_iter()
            .filter(|field| field.derivation().is_some()).map(|field| field.name.clone()).collect();
        // A field the frame completes, which the source published at only
        // some of its values or at none, is held back from every
        // derivation until it is whole; `seeds` keeps the step of the
        // values the source did publish.
        let completed_names = completed_fields(mapping.vertical_kind()?, &plan.required_names);
        let mut seeds: BTreeMap<&str, Option<usize>> = BTreeMap::new();
        for name in &completed_names {
            if pending.contains(*name) { continue; }
            let partial = collection.direct.get(&(key.0, key.1.clone(), (*name).to_owned()))
                .map(|direct| direct.missing_count > 0);
            if partial == Some(false) { continue; }
            seeds.insert(*name, versions.remove(*name));
        }
        // The completion reads whole columns of five operands together.
        // A field decoded over the atmospheric window alone has other
        // columns than a full-grid operand, so pairing them would read
        // one column's temperature with another column's pressure.  The
        // decode stream reads any frame that completes a field whole
        // (`DecodeStream::completes_over_whole_columns`) or never decodes
        // the operands over the window (`decode_window_candidates`), so
        // reaching this is an engine defect, and it is refused rather
        // than published.
        if !seeds.is_empty() {
            let full = [collection.latitude.len(), collection.longitude.len()];
            let windowed: Vec<&str> = HYPSOMETRIC_OPERANDS.iter().map(|(name, _)| *name)
                .chain(seeds.keys().copied())
                .filter(|name| collection.direct.get(&(key.0, key.1.clone(), (*name).to_owned()))
                    .is_some_and(|direct| {
                        let shape = direct.values.shape();
                        shape.len() >= 2 && shape[shape.len() - 2..] != full
                    }))
                .collect();
            if !windowed.is_empty() {
                return Err(frame_invalid(format!(
                    "the frame at {} completes {} hydrostatically, which reads whole \
                     columns, but {} was decoded over the atmospheric window alone, \
                     so its columns are not the surface pressure's; the decode \
                     should have read this frame whole (an engine defect)",
                    key.0,
                    seeds.keys().copied().collect::<Vec<_>>().join(", "),
                    windowed.join(", "))));
            }
        }
        // A completion step reads the operands' current versions and, when
        // the source published part of the field, that partial field.
        let plan_completion = |name: &str, seed: Option<usize>, versions: &mut BTreeMap<String, usize>,
                               steps: &mut Vec<FieldStep>| -> bool {
            if HYPSOMETRIC_OPERANDS.iter().any(|(operand, _)| !versions.contains_key(*operand)) {
                return false;
            }
            let mut dependencies: Vec<(String, usize)> = Vec::new();
            if let Some(seed) = seed {
                dependencies.push((name.to_owned(), seed));
            }
            for (operand, _) in HYPSOMETRIC_OPERANDS {
                dependencies.push((operand.to_owned(), versions[operand]));
            }
            versions.insert(name.to_owned(), steps.len());
            steps.push(FieldStep { name: name.to_owned(), derivation: None, completion: true,
                dependencies });
            true
        };
        let waits_on = |pending: &BTreeSet<String>| {
            HYPSOMETRIC_OPERANDS.iter().any(|(operand, _)| pending.contains(*operand))
        };
        while !pending.is_empty() || !seeds.is_empty() {
            let mut progress = false;
            for name in pending.clone() {
                let field = mapping.field(&name)?;
                let derivation = field.derivation().expect("pending entries are derived");
                let operation = mapping.derivation(derivation).ok_or_else(|| mapping_invalid(format!(
                    "field {name} names unknown derivation '{derivation}'")))?;
                let Some(dependencies) = derivation_dependencies(operation, mapping.vertical()?, &name)?
                    else { continue; };
                if dependencies.iter().any(|name| !versions.contains_key(name)) { continue; }
                let dependencies = dependencies.into_iter().map(|name| {
                    let version = versions[&name];
                    (name, version)
                }).collect();
                versions.insert(name.clone(), steps.len());
                steps.push(FieldStep { name: name.clone(), derivation: Some(derivation.to_owned()),
                    completion: false, dependencies });
                pending.remove(&name);
                progress = true;
            }
            // A completion reads the final version of each operand, so it
            // waits while a derivation still has one to produce.
            for name in &completed_names {
                let Some(seed) = seeds.get(*name).copied() else { continue; };
                if !waits_on(&pending) && plan_completion(name, seed, &mut versions, &mut steps) {
                    seeds.remove(*name);
                    progress = true;
                }
            }
            if !progress {
                // A completed field whose own derivation cannot run is
                // derived hydrostatically instead.
                for name in &completed_names {
                    if pending.contains(*name) && !waits_on(&pending)
                        && plan_completion(name, None, &mut versions, &mut steps) {
                        pending.remove(*name);
                        progress = true;
                    }
                }
            }
            if !progress {
                if !pending.is_empty() {
                    return Err(frame_invalid(format!(
                        "derived fields have missing dependencies or a cycle: {}",
                        pending.iter().cloned().collect::<Vec<_>>().join(", "))));
                }
                // A completion whose operands this source does not carry:
                // the frame lacks that field, and says so by name below.
                break;
            }
        }
        let missing: Vec<&String> = plan.required_names.iter()
            .filter(|name| !versions.contains_key(*name)).collect();
        if !missing.is_empty() {
            return Err(frame_invalid(format!(
                "mapped frame at {} lacks required fields {}", key.0,
                crate::refusal::python_list_repr(&missing))));
        }
        let output: Vec<usize> = plan.declared_names.iter()
            .filter_map(|name| versions.get(name).copied()).collect();
        let mut remaining = vec![0; steps.len()];
        for step in &steps {
            for (_, dependency) in &step.dependencies { remaining[*dependency] += 1; }
        }
        for id in &output { remaining[*id] += 1; }
        Ok(Self { mapping, collection, key: key.clone(), steps, output, remaining,
                  available: BTreeMap::new() })
    }

    fn names(&self) -> Vec<String> {
        self.output.iter().map(|id| self.steps[*id].name.clone()).collect()
    }

    fn materialize(&mut self, id: usize) -> Result<()> {
        if self.available.contains_key(&id) { return Ok(()); }
        let name = self.steps[id].name.clone();
        let dependencies = self.steps[id].dependencies.clone();
        for (_, dependency) in &dependencies { self.materialize(*dependency)?; }
        let field = self.mapping.field(&name)?;
        let result = if self.steps[id].completion {
            let operands: BTreeMap<String, &CanonicalField> = dependencies.iter()
                .map(|(name, dependency)| (name.clone(), &self.available[dependency])).collect();
            let seed = operands.get(&name).copied();
            complete_field(&field, seed, &operands)?
                .ok_or_else(|| frame_invalid(format!("planned completion of {name} lost its operands")))?
                .validated()?
        } else if let Some(derivation) = &self.steps[id].derivation {
            let operands: BTreeMap<String, &CanonicalField> = dependencies.iter()
                .map(|(name, dependency)| (name.clone(), &self.available[dependency])).collect();
            let operation = self.mapping.derivation(derivation).expect("planned derivation exists");
            let (values, axes, source_references) = evaluate_derivation(
                operation, &operands, &self.collection, &field, &name, self.mapping.vertical()?)?
                .ok_or_else(|| frame_invalid(format!("planned derivation {name} lost its dependencies")))?;
            let missing_count = array::count_nan(&values);
            CanonicalField { name: name.clone(), units: field.units_target()?.to_owned(),
                axes, location: field.location()?.to_owned(), staggering: field.staggering().to_owned(),
                values, missing_count, source_references }.validated()?
        } else {
            let direct = self.collection.direct.remove(&(self.key.0, self.key.1.clone(), name.clone()))
                .expect("planned direct field is moved exactly once");
            // Its complete values were validated before planning.
            CanonicalField { name: name.clone(), units: field.units_target()?.to_owned(),
                axes: direct.axes, location: field.location()?.to_owned(),
                staggering: field.staggering().to_owned(), values: direct.values,
                missing_count: direct.missing_count, source_references: direct.references }
        };
        self.available.insert(id, result);
        for (_, dependency) in dependencies { self.consume(dependency); }
        Ok(())
    }

    fn consume(&mut self, id: usize) {
        self.remaining[id] -= 1;
        if self.remaining[id] == 0 { self.available.remove(&id); }
    }

    /// The values this frame publishes, counted before one is derived.
    ///
    /// A direct field carries its decoded shape.  A derived field keeps
    /// the shape of the operand its derivation keeps
    /// (`derivation_shape_operand`), one soil layer taller for a surface
    /// node, laid out on its own target axes; a derivation built on the
    /// frame's ladder, and a completed field, take their declared target
    /// axes over this frame's own grid, ladder and soil column.  Every
    /// step follows the steps it reads, so one pass in step order sizes
    /// them all.  A windowed field keeps its levels over the window's rows
    /// and columns, as the writer crops it.  `None` when a field names an
    /// axis this cannot size; the write itself still refuses a full disk
    /// then, it only cannot refuse before the first byte.
    fn planned_values(&self, window: Option<&crate::window::Window>) -> Result<Option<u64>> {
        let collection = &self.collection;
        let direct = |name: &str| {
            collection.direct.get(&(self.key.0, self.key.1.clone(), name.to_owned()))
        };
        let soil = collection.direct.iter()
            .filter(|((time, member, _), _)| *time == self.key.0 && *member == self.key.1)
            .find_map(|(_, value)| value.axes.iter().position(|axis| axis == "soil")
                .map(|axis| value.values.shape()[axis]));
        let soil = match soil {
            Some(layers) => Some(layers),
            None => self.mapping.soil_layer_count()?.and_then(|count| usize::try_from(count).ok()),
        };
        let on_grid = |axes: &[String]| -> Option<Vec<usize>> {
            axes.iter().map(|axis| match axis.as_str() {
                "y" => Some(collection.latitude.len()),
                "x" => Some(collection.longitude.len()),
                "vertical" => Some(collection.vertical_values.len()),
                "soil" => soil,
                _ => None,
            }).collect()
        };
        let mut shapes: Vec<Option<(Vec<String>, Vec<usize>)>> = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            let target = self.mapping.field(&step.name)?.target_axes()?;
            let planned = match (&step.derivation, direct(&step.name)) {
                (None, Some(value)) => Some((value.axes.clone(), value.values.shape().to_vec())),
                (Some(derivation), _) => {
                    let operation = self.mapping.derivation(derivation).ok_or_else(|| mapping_invalid(
                        format!("field {} names unknown derivation '{derivation}'", step.name)))?;
                    match derivation_shape_operand(operation) {
                        None => on_grid(&target).map(|shape| (target, shape)),
                        Some((label, surface_node)) => {
                            let operand = operation.get(label).and_then(crate::node::Node::as_str)
                                .and_then(|name| step.dependencies.iter().find(|(dependency, _)| dependency == name))
                                .and_then(|(_, id)| shapes[*id].clone());
                            operand.and_then(|(axes, mut shape)| {
                                if surface_node {
                                    shape[axes.iter().position(|axis| axis == "soil")?] += 1;
                                }
                                // The result keeps its operand's axes, then is
                                // laid out on the field's own target axes.
                                let laid_out = target.iter()
                                    .map(|axis| axes.iter().position(|name| name == axis).map(|at| shape[at]))
                                    .collect::<Option<Vec<usize>>>()?;
                                Some((target, laid_out))
                            })
                        }
                    }
                }
                (None, None) => on_grid(&target).map(|shape| (target, shape)),
            };
            shapes.push(planned);
        }
        let mut total: u64 = 0;
        for id in &self.output {
            let step = &self.steps[*id];
            let Some((_, shape)) = &shapes[*id] else { return Ok(None); };
            let values: usize = match window.filter(|w| w.fields.contains(&step.name)) {
                Some(w) if shape.len() == 3 => w.shape(shape[0]).iter().product(),
                _ => shape.iter().product(),
            };
            total = total.saturating_add(values as u64);
        }
        Ok(Some(total))
    }
}

/// Where the frame stream stands when a write fails: what it needs in
/// all, what had gone out, and what the disk holding it has left.
fn stream_detail(directory: &std::path::Path, planned: Option<u64>, written: u64,
                 available: &dyn Fn(&std::path::Path) -> Option<u64>) -> String {
    use crate::refusal::bytes_and_gib;
    let mut detail = match planned {
        Some(needed) => format!("the stream needs {} in all and {} had been written",
            bytes_and_gib(needed), bytes_and_gib(written)),
        None => format!("{} of the stream had been written", bytes_and_gib(written)),
    };
    if let Some(free) = available(directory) {
        detail.push_str(&format!(", and the disk that holds {} has {} free",
            directory.display(), bytes_and_gib(free)));
    }
    detail
}

/// Write `frames.json` + `frames.f64` into `directory`, PULLING one valid
/// time at a time from `slice`.
///
/// The writer never sees the whole series.  `slice` is handed one
/// `(valid_time, member)` key at a time, in the order the frameset writes
/// them, and returns a collection carrying THAT key alone; each field is
/// materialized, written, and released after its final consumer. The
/// decoded time is dropped before the next key is asked for. Named
/// breakage, measured on real RRFS bytes (3 km
/// CONUS, 45 pressure levels): a writer handed the assembled series held
/// every valid time's arrays at once, so a seven-time preparation needed
/// about 65 GiB of host memory and was killed by the OOM reaper on
/// anything smaller.  Nothing about the frameset needs two valid times
/// resident -- the stream is written sequentially and each frame's
/// digests cover only its own arrays.
pub fn write_frameset(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    slice: impl FnMut(&(NaiveDateTime, Option<String>)) -> Result<DecodedCollection> + Send,
) -> Result<Value> {
    write_frameset_with_window(directory, mapping, series, input_sha256, false, slice)
}

pub fn write_frameset_with_window(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    request_windows: bool,
    slice: impl FnMut(&(NaiveDateTime, Option<String>)) -> Result<DecodedCollection> + Send,
) -> Result<Value> {
    write_frameset_with_decode_window(directory, mapping, series, input_sha256, request_windows,
        None, slice)
}

/// The writer, when the decode was granted its atmospheric window before
/// the first record was read (`DecodeStream::decode_window`): every frame
/// publishes that window instead of asking again, and a field the decode
/// already produced over the window is published as it was decoded.
pub fn write_frameset_with_decode_window(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    request_windows: bool,
    decode_window: Option<crate::window::Window>,
    slice: impl FnMut(&(NaiveDateTime, Option<String>)) -> Result<DecodedCollection> + Send,
) -> Result<Value> {
    write_frameset_within(directory, mapping, series, input_sha256, request_windows,
        decode_window, &crate::space::available_bytes, slice)
}

/// [`write_frameset_with_window`] with `lanes` valid times in flight.
///
/// `slice` is called with each valid time's POSITION in the frameset as
/// well as its key, from several threads at once, so it must be able to
/// decode any valid time on its own (`DecodeStream::slice_detached`).
/// Everything a reader can observe is the one-lane writer's: frames are
/// written in frameset order, each atmospheric-window request goes to
/// the parent in frameset order, and the refusal reported is the first
/// one in frameset order -- a valid time that fails is only reported
/// once every earlier valid time has been written, exactly where the
/// one-lane writer would have raised it.
///
/// Named breakage: a 49-time 3 km CONUS compose ran one valid time at a
/// time, about 15 s each, on 2.3 cores of work whatever the box had,
/// because most of a valid time's wall is serial (assembly, the field
/// digests, the frame write); that made the source decode the longest
/// stage of a rented GPU box's preparation.
#[allow(clippy::too_many_arguments)]
pub fn write_frameset_lanes(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    request_windows: bool,
    decode_window: Option<crate::window::Window>,
    lanes: usize,
    slice: impl Fn(usize, &(NaiveDateTime, Option<String>)) -> Result<DecodedCollection> + Sync,
) -> Result<Value> {
    write_frameset_core(directory, mapping, series, input_sha256, request_windows,
        decode_window.as_ref(), &crate::space::available_bytes, lanes, &slice)
}

/// The writer, with the free-space reading it admits the stream against.
///
/// The stream's size is known exactly once the first valid time's field
/// plan and its atmospheric window are: every frame publishes the same
/// inventory on the same grid, so the first frame's values times the
/// number of valid times is the whole stream.  That is checked against
/// `available` before the first byte goes out.  Named breakage: a 48 h
/// global source composed for fourteen minutes and then stopped on a full
/// disk, because nothing asked whether the disk could hold its stream.
#[allow(clippy::too_many_arguments)]
fn write_frameset_within(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    request_windows: bool,
    decode_window: Option<crate::window::Window>,
    available: &(dyn Fn(&std::path::Path) -> Option<u64> + Sync),
    slice: impl FnMut(&(NaiveDateTime, Option<String>)) -> Result<DecodedCollection> + Send,
) -> Result<Value> {
    write_frameset_within_overlap(directory, mapping, series, input_sha256, request_windows,
        decode_window, available, false, slice)
}

/// [`write_frameset_within`], optionally writing each frame's bytes while
/// the next valid time decodes (`overlap_write`, which the caller grants
/// only when one more decoded frame fits in memory:
/// [`crate::threads::overlap_write`]).
#[allow(clippy::too_many_arguments)]
fn write_frameset_within_overlap(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    request_windows: bool,
    decode_window: Option<crate::window::Window>,
    available: &(dyn Fn(&std::path::Path) -> Option<u64> + Sync),
    overlap_write: bool,
    slice: impl FnMut(&(NaiveDateTime, Option<String>)) -> Result<DecodedCollection> + Send,
) -> Result<Value> {
    // One lane calls `slice` from this thread, in frameset order; the
    // lock only lets the one-lane and many-lane writers share one body.
    let slice = std::sync::Mutex::new(slice);
    write_frameset_core_with(directory, mapping, series, input_sha256, request_windows,
        decode_window.as_ref(), available, 1, overlap_write,
        &|_index, key| (slice.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))(key))
}

/// The one-valid-time writer with the stream written behind the next
/// valid time's decode when `overlap_write` (see
/// [`write_frameset_within_overlap`]).
#[allow(clippy::too_many_arguments)]
pub fn write_frameset_one_lane(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    request_windows: bool,
    decode_window: Option<crate::window::Window>,
    overlap_write: bool,
    slice: impl FnMut(&(NaiveDateTime, Option<String>)) -> Result<DecodedCollection> + Send,
) -> Result<Value> {
    write_frameset_within_overlap(directory, mapping, series, input_sha256, request_windows,
        decode_window, &crate::space::available_bytes, overlap_write, slice)
}

type FrameKey = (NaiveDateTime, Option<String>);

/// One valid time between its decode and its field work: the frame's
/// header and field plan, and the inventory it publishes.
struct PlannedFrame<'a> {
    index: usize,
    source_cycle: NaiveDateTime,
    header: Value,
    fields: FieldMaterializer<'a>,
    names: Vec<String>,
    soil_count: Option<i64>,
}

/// One published field, digested, windowed and ready to be streamed.
struct PreparedField {
    name: String,
    units: String,
    axes: Vec<String>,
    location: String,
    staggering: String,
    shape: Vec<usize>,
    values: Vec<f64>,
    sha256: String,
    missing_count: usize,
    source_references: Vec<String>,
    original: Option<Value>,
}

/// What one field's work produces, in the order the frame publishes it.
struct FieldWork {
    descriptor: Value,
    soil_levels: Option<usize>,
    pressure_levels: Option<Vec<f64>>,
    prepared: PreparedField,
}

/// One valid time ready to be written: every field that precedes its
/// first refusal, and that refusal, so the write reproduces the one-lane
/// writer's order (fields before the failing one go out, then it fails).
struct PreparedFrame {
    index: usize,
    source_cycle: NaiveDateTime,
    header: Value,
    window: Option<crate::window::Window>,
    pressure_levels: Option<Vec<f64>>,
    latitude: Value,
    longitude: Value,
    vertical_values: Value,
    fields: Vec<PreparedField>,
    failure: Option<crate::refusal::Refusal>,
}

/// A decoded valid time's header and field plan -- everything the frame
/// knows before its window is asked for.
fn plan_frame<'a>(
    mapping: &'a Mapping,
    plan: &FramePlan,
    index: usize,
    collection: DecodedCollection,
) -> Result<PlannedFrame<'a>> {
    let key = &plan.keys[index];
    let (valid_time, _member) = key;
    validate_frame_axes(&collection, &[])?;
    let source_cycle = collection.source_cycles[key];
    let header = frame_header(mapping, *valid_time, source_cycle, &collection, &[])?;
    // ONE valid time, with canonical fields pulled on demand and
    // released after their final consumer. Source decoder residency
    // is separate; this removes the duplicate complete frame.
    let fields = FieldMaterializer::new(mapping, collection, plan, key)?;
    let names = fields.names();
    let soil_count = mapping.soil_layer_count()?.filter(|count| *count > 0);
    if soil_count.is_some() {
        for name in ["soil_temperature", "volumetric_soil_moisture"] {
            if !names.iter().any(|field| field == name) {
                return Err(frame_invalid(format!("mapped frame at {valid_time} lacks {name}")));
            }
        }
    }
    Ok(PlannedFrame { index, source_cycle, header, fields, names, soil_count })
}

/// One field's checks, digests and window: pure work on a materialized
/// field, so a frame's fields run it concurrently.
#[allow(clippy::too_many_arguments)]
fn field_work(
    fields: &FieldMaterializer<'_>,
    field: CanonicalField,
    plan: &FramePlan,
    soil_count: Option<i64>,
    window: Option<&crate::window::Window>,
    decode_window: Option<&crate::window::Window>,
    time: &Value,
    valid_time: NaiveDateTime,
) -> Result<FieldWork> {
    let finite_required = plan.required_names.contains(&field.name)
        && field.name != "soil_temperature" && field.name != "volumetric_soil_moisture";
    if finite_required && field.values.iter().any(|value| !value.is_finite())
        && fields.mapping.field(&field.name)?.missing_kind()? != "preserve_mask"
    {
        return Err(frame_invalid(format!(
            "required mapped field {} is not finite at {valid_time}", field.name)));
    }
    if let Some(soil_count) = soil_count
        && ["soil_temperature", "volumetric_soil_moisture"].contains(&field.name.as_str()) {
            let axis = field.axes.iter().position(|axis| axis == "soil")
                .ok_or_else(|| frame_invalid(format!("{} has no soil axis", field.name)))?;
            let observed = field.values.shape()[axis] as i64;
            if observed != soil_count {
                return Err(frame_invalid(format!(
                    "{} has {observed} layers, target declares {soil_count}", field.name)));
            }
        }
    // A field the decode produced over the granted window alone: its y/x
    // are the window's and every value it has was checked above, so it
    // is published as decoded.
    let decoded = decode_window.filter(|w| w.holds_decoded(&field));
    if let Some(w) = decoded {
        w.validate_decoded(&field, fields.collection.vertical_values.len())?;
    } else {
        validate_frame_axes(&fields.collection, std::slice::from_ref(&field))?;
        if let Some(window) = window { window.validate_field(&field)?; }
    }
    let original_shape = field.values.shape().to_vec();
    let original_flat = array::contiguous(&field.values);
    // Hash once over the complete source. The same digest enters
    // the original canonical header and the field descriptor.
    let field_sha256 = crate::digest::array_sha256(field.values.shape(), &original_flat);
    let descriptor = field_descriptor(&field, time, &field_sha256);
    let soil_levels = field.axes.iter().position(|axis| axis == "soil")
        .map(|axis| field.values.shape()[axis]);
    let retained = window.filter(|w| w.fields.contains(&field.name));
    let shape = retained.map_or_else(|| field.values.shape().to_vec(),
        |w| w.shape(field.values.shape()[0]).to_vec());
    let mut pressure_levels = None;
    if decoded.is_some() && field.name == "air_pressure" {
        // Its whole planes were medianed as they were unpacked
        // (`assemble::DecodedCollection::original_pressure_hpa`).
        pressure_levels = Some(fields.collection.original_pressure_hpa.clone().ok_or_else(|| {
            frame_invalid("air_pressure was decoded over the atmospheric window                            without its whole-plane pressure ladder")
        })?);
    }
    let values: Vec<f64> = if let Some(w) = retained.filter(|_| decoded.is_none()) {
        if field.name == "air_pressure" {
            pressure_levels = Some(crate::window::pressure_levels(
                &original_flat, w.source_shape[0] * w.source_shape[1])?);
        }
        w.crop(&original_flat, shape[0])
    } else {
        match original_flat {
            std::borrow::Cow::Owned(values) => values,
            std::borrow::Cow::Borrowed(_) => array::into_contiguous(field.values),
        }
    };
    let sha256 = if retained.is_some() && decoded.is_none() {
        crate::digest::array_sha256(&shape, &values)
    } else { field_sha256.clone() };
    let missing_count = if retained.is_some() {
        values.iter().filter(|v| v.is_nan()).count()
    } else { field.missing_count };
    let original = if let Some(w) = decoded {
        // Nothing outside the window was decoded, so there is no
        // whole-field digest or missing count to state.
        Some(json!({
            "shape": [original_shape[0], w.source_shape[0], w.source_shape[1]],
            "sha256": Value::Null,
            "missing_count": Value::Null,
            "validation": crate::window::DECODED_VALIDATION,
        }))
    } else {
        retained.map(|_| json!({
            "shape": original_shape, "sha256": field_sha256,
            "missing_count": field.missing_count,
            "validation": "complete-canonical-field-before-window-v1",
        }))
    };
    Ok(FieldWork {
        descriptor,
        soil_levels,
        pressure_levels,
        prepared: PreparedField {
            name: field.name.clone(),
            units: field.units.clone(),
            axes: field.axes.clone(),
            location: field.location.clone(),
            staggering: field.staggering.clone(),
            shape,
            values,
            sha256,
            missing_count,
            source_references: field.source_references.clone(),
            original,
        },
    })
}

/// Materialize, check, digest and window every field one valid time
/// publishes.
///
/// Fields are MATERIALIZED in publication order (a derivation reads the
/// fields before it), then checked and digested concurrently -- the
/// digest over a complete 3-D field is the dominant cost and the fields
/// are independent -- and drained in publication order, so the descriptor
/// list, the soil coordinate and the first refusal are the one-lane
/// writer's.
fn prepare_frame(plan: &FramePlan, planned: PlannedFrame<'_>,
                 window: Option<crate::window::Window>,
                 decode_window: Option<&crate::window::Window>) -> PreparedFrame {
    let PlannedFrame { index, source_cycle, mut header, mut fields, soil_count, .. } = planned;
    let valid_time = plan.keys[index].0;
    let time = json!({
        "reference_time": utc_isoformat(source_cycle), "valid_time": utc_isoformat(valid_time),
        "lead_seconds": (valid_time - source_cycle).num_seconds(), "statistic": "instantaneous",
        "interval_start": Value::Null, "interval_end": Value::Null, "accumulation_reset": Value::Null,
    });
    let mut failure = None;
    let mut ready = Vec::with_capacity(fields.output.len());
    for id in fields.output.clone() {
        if let Err(refusal) = fields.materialize(id) {
            failure = Some(refusal);
            break;
        }
        ready.push(id);
    }
    // All derivations have finished reading their inputs. Take each
    // publication field in order so its worker can hand the allocation
    // straight to the stream, rather than clone every complete array.
    let ready: Vec<CanonicalField> = ready.into_iter()
        .map(|id| fields.available.remove(&id).expect("materialized output exists"))
        .collect();
    let works: Vec<Result<FieldWork>> = {
        let fields = &fields;
        let (time, window) = (&time, window.as_ref());
        crate::threads::install(|| {
            use rayon::prelude::*;
            ready.into_par_iter()
                .map(|field| field_work(fields, field, plan, soil_count, window, decode_window, time,
                    valid_time))
                .collect()
        })
    };
    let mut descriptors = Vec::with_capacity(works.len());
    let mut prepared = Vec::with_capacity(works.len());
    let mut pressure_levels = None;
    for work in works {
        match work {
            Ok(work) => {
                descriptors.push(work.descriptor);
                if header["vertical_coordinates"].get("soil").is_none()
                    && let Some(levels) = work.soil_levels {
                        header["vertical_coordinates"]["soil"] = json!({
                            "coordinate": "soil_depth", "level_count": levels,
                            "level_values": Vec::<f64>::new(), "a_coefficients": Vec::<f64>::new(),
                            "b_coefficients": Vec::<f64>::new(), "positive": "down", "units": "index",
                        });
                    }
                if work.pressure_levels.is_some() {
                    pressure_levels = work.pressure_levels;
                }
                prepared.push(work.prepared);
            }
            Err(refusal) => {
                // An earlier field's refusal is the one the one-lane
                // writer reached first, before any later materialization.
                failure = Some(refusal);
                break;
            }
        }
    }
    if failure.is_none() {
        header["fields"] = Value::Array(descriptors);
        if let Err(refusal) = require_wrf_initial_state(&header) {
            failure = Some(refusal);
        }
    }
    PreparedFrame {
        index,
        source_cycle,
        header,
        window,
        pressure_levels,
        latitude: axis_document(&fields.collection.latitude),
        longitude: axis_document(&fields.collection.longitude),
        vertical_values: axis_document(&fields.collection.vertical_values),
        fields: prepared,
        failure,
    }
}

/// The ordered half of the writer: the window requests, the stream, its
/// digest and the frame documents.  Only the thread that owns the
/// frameset touches it, and only in frameset order.
struct Committer<'w> {
    directory: &'w std::path::Path,
    mapping: &'w Mapping,
    series: &'w SeriesSummary,
    input_sha256: &'w BTreeMap<String, String>,
    request_windows: bool,
    /// The window granted before decoding (`DecodeStream::decode_window`):
    /// published by every frame instead of asking the parent again.
    decode_window: Option<&'w crate::window::Window>,
    available: &'w (dyn Fn(&std::path::Path) -> Option<u64> + Sync),
    keys: &'w [FrameKey],
    count: usize,
    stream_path: std::path::PathBuf,
    // The frame stream's bytes, and the whole-stream digest, go out on
    // their own thread (`StreamWriter`), in frameset order.
    stream: StreamWriter,
    offset: u64,
    // What the whole stream needs, once the first frame has said; kept
    // so a write that still meets a full disk can say how far it got.
    planned: Option<u64>,
    frame_documents: Vec<Value>,
    windowed: bool,
    inventories: BTreeSet<Vec<String>>,
}

impl Committer<'_> {
    fn stream_failure(&self, error: std::io::Error, written: u64) -> crate::refusal::Refusal {
        let detail = crate::refusal::out_of_space(&error)
            .then(|| stream_detail(self.directory, self.planned, written, self.available));
        crate::refusal::write_error(
            &format!("write the frame stream {}", self.stream_path.display()), &error, detail)
    }

    /// The frame's atmospheric window, asked of the parent IN FRAMESET
    /// ORDER, and -- on the first frame -- the whole stream's admission
    /// against the free space, before its first byte.
    fn window_for(&mut self, frame: &PlannedFrame<'_>) -> Result<Option<crate::window::Window>> {
        use crate::refusal::bytes_and_gib;
        let window = if let Some(granted) = self.decode_window {
            // Granted before the first record was decoded, so it is not
            // asked for again; every frame publishes it over the
            // canonical atmospheric fields it carries.
            let valid_time = self.keys[frame.index].0;
            if let Some(name) = granted.fields.iter().find(|name| !frame.names.contains(*name)) {
                return Err(frame_invalid(format!(
                    "the atmospheric window granted before decoding names {name}, which the \
                     frame at {valid_time} does not carry")));
            }
            if granted.source_shape != [frame.fields.collection.latitude.len(),
                                        frame.fields.collection.longitude.len()] {
                return Err(frame_invalid(format!(
                    "the atmospheric window granted before decoding was taken on another \
                     source grid than the frame at {valid_time}")));
            }
            Some(granted.for_inventory(&frame.names))
        } else if self.request_windows {
            crate::window::request_for_source(&frame.fields.collection.latitude,
                &frame.fields.collection.longitude, &frame.header["grid"], &frame.names,
                frame.index, &self.series.grid_fingerprint)?
        } else { None };
        self.windowed |= window.is_some();
        if frame.index == 0
            && let Some(values) = frame.fields.planned_values(window.as_ref())? {
                let frame_bytes = values.saturating_mul(8);
                let needed = frame_bytes.saturating_mul(self.count as u64);
                self.planned = Some(needed);
                if let Some(free) = (self.available)(self.directory).filter(|free| needed > *free) {
                    // The folder and both numbers make the first sentence,
                    // so a front end that shows one sentence shows them.
                    return Err(crate::refusal::disk_full(format!(
                        "the frame stream needs {} in {} and the disk that holds \
                         that folder has {} free.  It is {} valid times of {} each, \
                         refused before its first byte rather than written until \
                         the disk fills",
                        bytes_and_gib(needed), self.directory.display(), bytes_and_gib(free),
                        self.count, bytes_and_gib(frame_bytes))));
                }
            }
        self.inventories.insert(frame.names.clone());
        Ok(window)
    }

    /// Hand one prepared frame's fields to the stream writer and document
    /// it.  The bytes go out on the writer's thread, in frameset order;
    /// [`Self::drain`] waits for them.  A frame that carries a refusal is
    /// drained here, so a write that fails ahead of that refusal is the
    /// one reported, as when the bytes went out on this thread.
    fn commit(&mut self, frame: PreparedFrame) -> Result<()> {
        let (valid_time, member) = &self.keys[frame.index];
        let mut field_documents = Vec::with_capacity(frame.fields.len());
        let mut payload = Vec::with_capacity(frame.fields.len());
        for field in frame.fields {
            let length = (field.values.len() * 8) as u64;
            payload.push(field.values);
            let mut field_document = json!({
                "name": field.name,
                "units": field.units,
                "axes": field.axes,
                "location": field.location,
                "staggering": field.staggering,
                "shape": field.shape,
                "dtype": "<f8",
                "offset": self.offset,
                "length": length,
                "sha256": field.sha256,
                "missing_count": field.missing_count,
                "source_references": field.source_references,
            });
            if let Some(original) = field.original {
                field_document["original"] = original;
            }
            field_documents.push(field_document);
            self.offset += length;
        }
        self.stream.submit(payload);
        if let Some(refusal) = frame.failure {
            self.drain()?;
            return Err(refusal);
        }
        // Every MappedSourceFrame scalar rides the frame, not the
        // document: the reader rebuilds one dataclass per entry and
        // re-runs its validators, and a frame that had to borrow its
        // mapping digest, its input digests or its grid fingerprint
        // from an enclosing object could be replayed under a different
        // decode's provenance without anything noticing.
        let mut frame_document = json!({
            "valid_time": naive_isoformat(*valid_time),
            "member": member,
            "source_cycle": naive_isoformat(frame.source_cycle),
            "latitude": frame.latitude,
            "longitude": frame.longitude,
            "vertical_kind": self.mapping.vertical()?.get("kind").and_then(crate::node::Node::as_str).unwrap_or_default(),
            "vertical_units": self.mapping.vertical()?.get("units").and_then(crate::node::Node::as_str).unwrap_or_default(),
            "vertical_values": frame.vertical_values,
            "grid_fingerprint": self.series.grid_fingerprint,
            "mapping_sha256": self.mapping.sha256,
            "input_sha256": self.input_sha256,
            "fields": field_documents,
            "header": frame.header,
        });
        if let Some(w) = frame.window {
            frame_document["atmospheric_window"] = w.document();
            if let Some(levels) = frame.pressure_levels {
                frame_document["original_pressure_hpa"] = axis_document(&levels);
            }
        }
        self.frame_documents.push(frame_document);
        Ok(())
    }
}

impl Committer<'_> {
    /// Wait until every frame handed to the writer is written, and report
    /// the first write that failed as the serial writer reported it.
    fn drain(&mut self) -> Result<()> {
        match self.stream.drain() {
            Ok(()) => Ok(()),
            Err((error, written)) => Err(self.stream_failure(error, written)),
        }
    }
}

/// The frame stream's writer: its own thread, fed frames in frameset
/// order, writing each field's little-endian bytes at their offsets from
/// a few threads while one more folds the same bytes, in order, into the
/// whole-stream SHA-256 the reader re-computes before it trusts a single
/// array.  Accumulated as the bytes go out rather than by re-reading the
/// file: a real frameset is multi-gigabyte.
///
/// Both threads encode their own FIXED chunk at a time.  One field of a
/// 0.25-degree analysis is ~700 MB as f64; encoding a whole field into a
/// second Vec first would double the peak of every decode this seam
/// exists to make cheaper -- the same reason the reader streams its hash
/// instead of reading the stream whole.  The field's OWN digest was taken
/// from the full original values.
///
/// Named breakage: the encode, the whole-stream digest and the write ran
/// one after another on the thread that orders the frameset, so a 3 km
/// CONUS native-level compose spent about 6 s of every valid time's
/// 13 GB on one core while the pool and the cards waited.
struct StreamWriter {
    jobs: Option<std::sync::mpsc::Sender<Vec<Vec<f64>>>>,
    done: std::sync::mpsc::Receiver<std::result::Result<(), (std::io::Error, u64)>>,
    pending: usize,
    failed: Option<(std::io::Error, u64)>,
    broken: bool,
    thread: Option<std::thread::JoinHandle<(std::fs::File, sha2::Sha256)>>,
}

/// Threads writing one field's chunks at their offsets.  Page-cache
/// writes from one thread top out near 2 GB/s on the measured box, below
/// the whole-stream digest's rate, so a few share each field.
const STREAM_WRITERS: usize = 4;

impl StreamWriter {
    fn new(file: std::fs::File) -> Self {
        let (jobs, inbox) = std::sync::mpsc::channel::<Vec<Vec<f64>>>();
        let (report, done) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("mapped-engine-stream".into())
            .spawn(move || {
                let stream = file;
                let mut digest = <sha2::Sha256 as sha2::Digest>::new();
                let mut offset: u64 = 0;
                let mut broken = false;
                for fields in inbox {
                    let mut outcome = Ok(());
                    if !broken {
                        for values in &fields {
                            if let Err(failure) = stream_field(&stream, &mut digest, values, offset) {
                                outcome = Err(failure);
                                broken = true;
                                break;
                            }
                            offset += (values.len() * 8) as u64;
                        }
                    }
                    drop(fields);
                    if report.send(outcome).is_err() { break; }
                }
                (stream, digest)
            })
            .expect("spawn the frame stream writer");
        StreamWriter { jobs: Some(jobs), done, pending: 0, failed: None, broken: false, thread: Some(thread) }
    }

    /// A writer with no thread, left behind when the real one is closed.
    fn closed() -> Self {
        let (_jobs, done) = std::sync::mpsc::channel();
        StreamWriter { jobs: None, done, pending: 0, failed: None, broken: false, thread: None }
    }

    fn submit(&mut self, fields: Vec<Vec<f64>>) {
        if let Some(jobs) = &self.jobs
            && jobs.send(fields).is_ok() {
                self.pending += 1;
            }
    }

    fn drain(&mut self) -> std::result::Result<(), (std::io::Error, u64)> {
        while self.pending > 0 {
            self.pending -= 1;
            match self.done.recv() {
                Ok(Ok(())) => {}
                Ok(Err(failure)) => { self.failed.get_or_insert(failure); }
                Err(_) => {
                    self.failed.get_or_insert((std::io::Error::other("the frame stream writer stopped"), 0));
                    self.pending = 0;
                }
            }
        }
        // The failed write's own error goes back the first time, so its
        // OS code still classifies a full disk; the stream is broken from
        // then on and every later wait says so.
        match self.failed.take() {
            Some(failure) => {
                self.broken = true;
                Err(failure)
            }
            None if self.broken => Err((std::io::Error::other(
                "the frame stream writer stopped after a failed write"), 0)),
            None => Ok(()),
        }
    }

    /// Every frame written: the buffered stream and its digest back.
    fn close(mut self) -> std::result::Result<(std::fs::File, sha2::Sha256), (std::io::Error, u64)> {
        self.drain()?;
        self.jobs = None;
        let thread = self.thread.take().expect("the writer is joined once");
        match thread.join() {
            Ok(parts) => Ok(parts),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
}

impl Drop for StreamWriter {
    /// A writer dropped on a refusal path still finishes and joins its
    /// thread, so nothing writes into the scratch folder after the
    /// engine has reported.
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// One field's bytes into the stream and the digest, side by side; the
/// error carries the stream offset the serial writer reported: the field's
/// start plus every chunk before the FIRST chunk (in stream order) whose
/// write failed.
fn stream_field(
    stream: &std::fs::File,
    digest: &mut sha2::Sha256,
    values: &[f64],
    offset: u64,
) -> std::result::Result<(), (std::io::Error, u64)> {
    fn encode(chunk: &mut Vec<u8>, values: &[f64]) {
        chunk.clear();
        for value in values {
            chunk.extend_from_slice(&value.to_le_bytes());
        }
    }
    let chunks: Vec<&[f64]> = values.chunks(STREAM_CHUNK).collect();
    let chunk_bytes = (STREAM_CHUNK * 8) as u64;
    std::thread::scope(|scope| {
        let hasher = {
            let chunks = &chunks;
            scope.spawn(move || {
                let mut buffer: Vec<u8> = Vec::with_capacity(STREAM_CHUNK * 8);
                for values in chunks {
                    encode(&mut buffer, values);
                    sha2::Digest::update(digest, &buffer);
                }
            })
        };
        let writers: Vec<_> = (0..STREAM_WRITERS.min(chunks.len().max(1))).map(|first| {
            let chunks = &chunks;
            scope.spawn(move || -> Option<(usize, std::io::Error)> {
                let mut buffer: Vec<u8> = Vec::with_capacity(STREAM_CHUNK * 8);
                let mut index = first;
                while index < chunks.len() {
                    encode(&mut buffer, chunks[index]);
                    let at = offset + index as u64 * chunk_bytes;
                    if let Err(error) = write_all_at(stream, &buffer, at) {
                        return Some((index, error));
                    }
                    index += STREAM_WRITERS;
                }
                None
            })
        }).collect();
        let mut failed: Option<(usize, std::io::Error)> = None;
        for writer in writers {
            match writer.join() {
                Ok(Some((index, error))) => {
                    if failed.as_ref().is_none_or(|(first, _)| index < *first) {
                        failed = Some((index, error));
                    }
                }
                Ok(None) => {}
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
        if let Err(payload) = hasher.join() {
            std::panic::resume_unwind(payload);
        }
        match failed {
            Some((index, error)) => Err((error, offset + index as u64 * chunk_bytes)),
            None => Ok(()),
        }
    })
}

/// `buffer` written whole at byte `at` of `file`, from any thread.
#[cfg(unix)]
fn write_all_at(file: &std::fs::File, buffer: &[u8], at: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buffer, at)
}

/// `buffer` written whole at byte `at` of `file`, from any thread.
#[cfg(windows)]
fn write_all_at(file: &std::fs::File, mut buffer: &[u8], mut at: u64) -> std::io::Result<()> {
    while !buffer.is_empty() {
        match std::os::windows::fs::FileExt::seek_write(file, buffer, at) {
            Ok(0) => return Err(std::io::Error::new(std::io::ErrorKind::WriteZero,
                "failed to write whole buffer")),
            Ok(written) => {
                buffer = &buffer[written..];
                at += written as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// What a pipeline job hands back to the thread that writes the stream.
enum Stage<'a> {
    Planned(usize, Result<PlannedFrame<'a>>),
    Prepared(PreparedFrame),
    /// A job panicked.  Carried back so the ordered thread re-raises it:
    /// left in the job, it would never report and the ordered thread
    /// would wait for it forever.
    Panicked(Box<dyn std::any::Any + Send>),
}

// Per-invocation test control for arrival order. It is absent from
// production builds and cannot affect the decoded fields or stream bytes.
//
// Called on the ordered thread with each frame's index as its prepared
// result arrives.  Returning true withholds that result, still counted
// as in flight, until it is the only work in flight; the writer then
// takes it as if it had just arrived.  That is a slow frame without a
// slow job.  The control used to block inside the frame's pool job
// instead, which deadlocks: a pool worker waiting on the stolen half of
// a decode's parallel step runs queued jobs meanwhile, so the blocked
// job could sit on top of the very decode it was waiting for (public CI
// 37981479846, native windows-2025; 8 of 100 runs of the whole test
// binary on four cores of a development machine).
#[cfg(test)]
type PipelinePreparedHook = std::sync::Arc<dyn Fn(usize) -> bool + Send + Sync>;
#[cfg(test)]
std::thread_local! {
    static PIPELINE_PREPARED_HOOK: std::cell::RefCell<Option<PipelinePreparedHook>> =
        std::cell::RefCell::new(None);
}

/// The one writer body, for one lane or many, writing each frame before
/// the next valid time is decoded.
#[allow(clippy::too_many_arguments)]
fn write_frameset_core(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    request_windows: bool,
    decode_window: Option<&crate::window::Window>,
    available: &(dyn Fn(&std::path::Path) -> Option<u64> + Sync),
    lanes: usize,
    slice: &(dyn Fn(usize, &FrameKey) -> Result<DecodedCollection> + Sync),
) -> Result<Value> {
    write_frameset_core_with(directory, mapping, series, input_sha256, request_windows,
        decode_window, available, lanes, false, slice)
}

/// [`write_frameset_core`]; with one lane and `overlap_write`, a frame's
/// bytes go out while the next valid time decodes.
#[allow(clippy::too_many_arguments)]
fn write_frameset_core_with(
    directory: &std::path::Path,
    mapping: &Mapping,
    series: &SeriesSummary,
    input_sha256: &BTreeMap<String, String>,
    request_windows: bool,
    decode_window: Option<&crate::window::Window>,
    available: &(dyn Fn(&std::path::Path) -> Option<u64> + Sync),
    lanes: usize,
    overlap_write: bool,
    slice: &(dyn Fn(usize, &FrameKey) -> Result<DecodedCollection> + Sync),
) -> Result<Value> {
    use crate::refusal::write_error;
    let plan = plan_frames_within(mapping, &series.source_cycles, series.lead_batch)?;
    std::fs::create_dir_all(directory).map_err(|error| {
        write_error(&format!("create the output directory {}", directory.display()), &error, None)
    })?;
    let stream_path = directory.join("frames.f64");
    let file = std::fs::File::create(&stream_path).map_err(|error| {
        write_error(&format!("create the frame stream {}", stream_path.display()), &error, None)
    })?;
    let mut committer = Committer {
        directory,
        mapping,
        series,
        input_sha256,
        request_windows,
        decode_window,
        available,
        keys: &plan.keys,
        count: plan.keys.len(),
        stream_path,
        stream: StreamWriter::new(file),
        offset: 0,
        planned: None,
        frame_documents: Vec::with_capacity(plan.keys.len()),
        windowed: false,
        inventories: BTreeSet::new(),
    };
    let lanes = lanes.min(plan.keys.len()).max(1);
    // Set when the ordered thread stops early, so a job the scope still
    // has queued does not decode a valid time nobody will write.
    let abandoned = std::sync::atomic::AtomicBool::new(false);
    let pipelined = if lanes > 1 {
        crate::threads::in_place_scope(|scope| {
            pipeline(scope, mapping, &plan, lanes, slice, decode_window, &abandoned, &mut committer)
        })
    } else { None };
    match pipelined {
        Some(outcome) => outcome?,
        None => {
            for index in 0..plan.keys.len() {
                // With `overlap_write`, the previous frame's bytes are
                // still going out while this one decodes and plans; they
                // are drained before anything else happens in frameset
                // order (the window request, the next write), and a write
                // that failed is reported ahead of this frame's refusal,
                // where the serial writer would have stopped.
                let planned = slice(index, &plan.keys[index])
                    .and_then(|collection| plan_frame(mapping, &plan, index, collection));
                committer.drain()?;
                let planned = planned?;
                let window = committer.window_for(&planned)?;
                let prepared = prepare_frame(&plan, planned, window, decode_window);
                committer.commit(prepared)?;
                if !overlap_write {
                    committer.drain()?;
                }
            }
        }
    }
    committer.finish()
}

/// `lanes` valid times in flight, written in order.
///
/// Jobs on the pool decode a valid time and plan its frame (stage one),
/// or do its field work once its window is known (stage two).  THIS
/// thread does everything ordered: it asks each frame's window of the
/// parent in frameset order, writes each prepared frame in frameset
/// order, and admits a new valid time only while fewer than `lanes` hold
/// a decoded valid time (decoding, waiting for their window, or in field
/// work) -- which is what bounds memory.  A prepared frame waiting for an
/// earlier one to be written holds only its published arrays, so up to
/// half as many again may wait that way; without that allowance one slow
/// valid time at the head stalled every lane behind it.
///
/// A refusal at frame `i` (its decode, its plan or its window) is held
/// until every frame before `i` has been written, then returned; nothing
/// after `i` is started once it is known, and a later frame's refusal can
/// never be reported ahead of an earlier frame's.
#[allow(clippy::too_many_arguments)]
fn pipeline<'s>(
    scope: &rayon::Scope<'s>,
    mapping: &'s Mapping,
    plan: &'s FramePlan,
    lanes: usize,
    slice: &'s (dyn Fn(usize, &FrameKey) -> Result<DecodedCollection> + Sync),
    decode_window: Option<&'s crate::window::Window>,
    abandoned: &'s std::sync::atomic::AtomicBool,
    committer: &mut Committer<'_>,
) -> Result<()> {
    use std::sync::atomic::Ordering;
    let (sender, receiver) = std::sync::mpsc::channel::<Stage<'s>>();
    let count = plan.keys.len();
    let (mut started, mut windowed, mut committed, mut outstanding) = (0usize, 0usize, 0usize, 0usize);
    // Frames holding a decoded valid time, and how many may be between
    // their decode and their write in all.
    let mut decoded = 0usize;
    let depth = lanes + (lanes / 2).max(1);
    let mut planned: BTreeMap<usize, Result<PlannedFrame<'s>>> = BTreeMap::new();
    let mut prepared: BTreeMap<usize, PreparedFrame> = BTreeMap::new();
    let mut halted: Option<(usize, crate::refusal::Refusal)> = None;
    #[cfg(test)]
    let prepared_hook = PIPELINE_PREPARED_HOOK.with(|hook| hook.borrow().clone());
    #[cfg(test)]
    let mut withheld: Option<Stage<'s>> = None;
    let outcome = loop {
        while halted.is_none() && started < count && decoded < lanes && started - committed < depth {
            let (index, sender) = (started, sender.clone());
            scope.spawn(move |_| {
                if abandoned.load(Ordering::Relaxed) { return; }
                let stage = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Stage::Planned(index, slice(index, &plan.keys[index])
                        .and_then(|collection| plan_frame(mapping, plan, index, collection)))
                })).unwrap_or_else(Stage::Panicked);
                let _ = sender.send(stage);
            });
            started += 1;
            outstanding += 1;
            decoded += 1;
        }
        let mut written = Ok(());
        while let Some(frame) = prepared.remove(&committed) {
            written = committer.commit(frame).and_then(|()| committer.drain());
            if written.is_err() { break; }
            committed += 1;
        }
        if written.is_err() { break written; }
        if committed == count { break Ok(()); }
        if halted.as_ref().is_some_and(|(index, _)| *index == committed) {
            break Err(halted.take().expect("just matched").1);
        }
        if outstanding == 0 {
            // A complete prepared batch can free admission only after
            // the start loop saw a full depth. Re-enter that loop before
            // declaring a stall when there is now another frame to admit.
            if halted.is_none() && started < count && decoded < lanes && started - committed < depth {
                continue;
            }
            break Err(frame_invalid(format!(
                "the frame writer stalled at valid time {committed} of {count} with no work in flight")));
        }
        #[cfg(test)]
        let released = if outstanding == 1 { withheld.take() } else { None };
        #[cfg(not(test))]
        let released = None;
        #[cfg(test)]
        let arriving = released.is_none();
        let stage = match released {
            Some(stage) => stage,
            None => match receiver.recv() {
                Ok(stage) => stage,
                Err(_) => break Err(frame_invalid("the frame writer lost its workers")),
            },
        };
        #[cfg(test)]
        let stage = match (&prepared_hook, stage) {
            (Some(hook), Stage::Prepared(frame)) if arriving && hook(frame.index) && withheld.is_none() => {
                withheld = Some(Stage::Prepared(frame));
                continue;
            }
            (_, stage) => stage,
        };
        outstanding -= 1;
        match stage {
            Stage::Planned(index, outcome) => {
                if outcome.is_err() { decoded -= 1; }
                planned.insert(index, outcome);
            }
            Stage::Prepared(frame) => {
                decoded -= 1;
                prepared.insert(frame.index, frame);
            }
            Stage::Panicked(payload) => {
                // The one-lane writer would have panicked on this thread;
                // so does this one, once nothing else will be started.
                abandoned.store(true, Ordering::Relaxed);
                std::panic::resume_unwind(payload);
            }
        }
        while halted.is_none() {
            let Some(outcome) = planned.remove(&windowed) else { break };
            let index = windowed;
            windowed += 1;
            match outcome.and_then(|frame| committer.window_for(&frame).map(|window| (frame, window))) {
                Ok((frame, window)) => {
                    let sender = sender.clone();
                    scope.spawn(move |_| {
                        if abandoned.load(Ordering::Relaxed) { return; }
                        let stage = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            Stage::Prepared(prepare_frame(plan, frame, window, decode_window))
                        })).unwrap_or_else(Stage::Panicked);
                        let _ = sender.send(stage);
                    });
                    outstanding += 1;
                }
                Err(refusal) => {
                    halted = Some((index, refusal));
                    // Frames after the refusal are never written; their
                    // decoded valid times are released now.
                    planned.clear();
                }
            }
        }
    };
    abandoned.store(true, Ordering::Relaxed);
    drop(receiver);
    outcome
}

impl Committer<'_> {
    /// Checked once the whole series has been written, from the names
    /// alone.  The frameset is scratch until the caller reads it back,
    /// so a series that fails here is deleted whole; nothing partial is
    /// ever handed on.
    fn finish(mut self) -> Result<Value> {
        use crate::refusal::write_error;
        let writer = std::mem::replace(&mut self.stream, StreamWriter::closed());
        let (mut stream, stream_digest) = match writer.close() {
            Ok(parts) => parts,
            Err((error, written)) => return Err(self.stream_failure(error, written)),
        };
        require_one_inventory(&self.inventories)?;
        if let Err(error) = stream.flush() {
            return Err(self.stream_failure(error, self.offset));
        }
        let document = json!({
            "schema": if self.windowed { crate::window::FRAMESET_SCHEMA } else { crate::FRAMESET_SCHEMA },
            "engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION},
            // One object, not a bare name beside a loose byte count: the
            // reader verifies path, size and whole-stream digest together
            // before it maps a byte, and two spellings of the same number
            // are how a stream and its manifest drift apart.
            "stream": {
                "path": "frames.f64",
                "dtype": "<f8",
                "bytes": self.offset,
                "sha256": crate::digest::hex_digest(stream_digest),
            },
            "mapping_sha256": self.mapping.sha256,
            "mapping_path": self.mapping.path,
            "input_sha256": self.input_sha256,
            "grid_fingerprint": self.series.grid_fingerprint,
            "frames": self.frame_documents,
        });
        let manifest_path = self.directory.join("frames.json");
        std::fs::write(&manifest_path, serde_json::to_vec_pretty(&document).unwrap_or_default())
            .map_err(|error| {
                write_error(&format!("write the frameset manifest {}", manifest_path.display()),
                    &error, None)
            })?;
        Ok(document)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_grid_pairing_reaches_the_frame_only_when_declared() {
        let (_, collection, plan) = chain_fixture(false);
        for selected in [false, true] {
            let mut document = json!({
                "name": "projected-pairing-test",
                "coordinates": {"vertical": {"kind": "pressure", "units": "Pa"}},
                "fields": {}, "target": {},
                "grid": {
                    "family": "lambert_conformal",
                    "wind_basis": "grid_relative_with_rotation",
                    "parameters": {
                        "latin1": 50.0, "latin2": 50.0, "lov": 253.0,
                        "lat1": 1.0, "lon1": 214.5,
                        "dx_m": 32463.0, "dy_m": 32463.0,
                        "nx": 2, "ny": 2,
                        "earth_radius_m": 6371229.0, "shape_of_earth": 6
                    }
                }
            });
            if selected {
                document["grid"]["same_grid_pairing"] = json!("identity");
            }
            let payload = serde_json::to_vec(&document).unwrap();
            let mapping = Mapping {
                doc: crate::node::Node::parse(&payload).unwrap(),
                sha256: crate::digest::bytes_sha256(&payload),
                path: "<projected-pairing-test>".to_owned(),
            };
            let header = frame_header(&mapping, plan.keys[0].0, plan.keys[0].0, &collection, &[]).unwrap();
            let parameters = &header["grid"]["parameters"];
            if selected {
                assert_eq!(parameters["same_grid_pairing"], json!("identity"));
            } else {
                assert!(parameters.get("same_grid_pairing").is_none());
                assert!(!serde_json::to_string(&header).unwrap().contains("same_grid_pairing"));
            }
        }
    }

    fn chain_fixture(self_seed: bool) -> (Mapping, DecodedCollection, FramePlan) {
        let source = json!({"units": {"source": "1", "target": "1"}, "location": "mass",
            "source_axes": ["y", "x"], "target_axes": ["y", "x"]});
        let mut middle = source.clone();
        middle["derivation"] = json!("double");
        middle["units"]["scale"] = json!(2.0);
        let mut output = source.clone();
        output["derivation"] = json!("triple");
        output["units"]["scale"] = json!(3.0);
        let fields = if self_seed { json!({"source": middle}) }
            else { json!({"source": source, "middle": middle, "output": output}) };
        let document = json!({"name": "field-stream-test",
            "coordinates": {"vertical": {"kind": "pressure", "units": "Pa"}},
            "target": {}, "fields": fields, "derivations": [
                {"name": "double", "operation": "copy", "source": "source"},
                {"name": "triple", "operation": "copy", "source": "middle"}]});
        let payload = serde_json::to_vec(&document).unwrap();
        let mapping = Mapping { doc: crate::node::Node::parse(&payload).unwrap(),
            sha256: crate::digest::bytes_sha256(&payload), path: "<field-stream-test>".to_owned() };
        let valid_time = NaiveDateTime::parse_from_str("2026-08-17 06:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let direct = crate::assemble::DirectValue { name: "source".to_owned(), valid_time,
            member: None, source_cycle: valid_time, axes: vec!["y".to_owned(), "x".to_owned()],
            values: ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1., 2., 3., 4.]).unwrap(),
            missing_count: 0, references: vec!["@source".to_owned()] };
        let collection = DecodedCollection { latitude: vec![1., 2.], longitude: vec![3., 4.],
            vertical_values: vec![100000.], direct: BTreeMap::from([((valid_time, None, "source".to_owned()), direct)]),
            source_cycles: BTreeMap::from([((valid_time, None), valid_time)]),
            grid_fingerprint: "grid".to_owned(), hybrid_a: Vec::new(), hybrid_b: Vec::new(), original_pressure_hpa: None };
        let plan = FramePlan { keys: vec![(valid_time, None)],
            declared_names: if self_seed { vec!["source".to_owned()] }
                else { ["source", "middle", "output"].map(str::to_owned).to_vec() },
            required_names: BTreeSet::new() };
        (mapping, collection, plan)
    }

    #[test]
    fn field_writer_moves_direct_arrays_and_releases_last_consumers() {
        let (mapping, collection, plan) = chain_fixture(false);
        let pointer = collection.direct.values().next().unwrap().values.as_ptr();
        let mut fields = FieldMaterializer::new(&mapping, collection, &plan, &plan.keys[0]).unwrap();
        let ids = fields.output.clone();
        fields.materialize(ids[0]).unwrap();
        assert_eq!(fields.available[&ids[0]].values.as_ptr(), pointer);
        assert!(fields.collection.direct.is_empty());
        fields.consume(ids[0]);
        assert!(fields.available.contains_key(&ids[0]), "the derivation still needs its source");
        fields.materialize(ids[1]).unwrap();
        assert!(!fields.available.contains_key(&ids[0]));
        fields.consume(ids[1]);
        fields.materialize(ids[2]).unwrap();
        assert!(!fields.available.contains_key(&ids[1]));
        assert_eq!(array::contiguous(&fields.available[&ids[2]].values).as_ref(), &[6., 12., 18., 24.]);
        fields.consume(ids[2]);
        assert!(fields.available.is_empty());
        assert!(fields.remaining.iter().all(|uses| *uses == 0));
    }

    #[test]
    fn frame_preparation_moves_outputs_after_their_derivations() {
        let (mapping, collection, plan) = chain_fixture(false);
        let pointer = collection.direct.values().next().unwrap().values.as_ptr();
        let planned = plan_frame(&mapping, &plan, 0, collection).unwrap();
        let prepared = prepare_frame(&plan, planned, None, None);
        assert!(prepared.failure.as_ref().unwrap().message.contains(
            "source frame is incomplete for WRF initialization"));
        assert_eq!(prepared.fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
                   ["source", "middle", "output"]);
        assert_eq!(prepared.fields[0].values.as_ptr(), pointer);
        assert_eq!(prepared.fields[0].values, [1., 2., 3., 4.]);
        assert_eq!(prepared.fields[1].values, [2., 4., 6., 8.]);
        assert_eq!(prepared.fields[2].values, [6., 12., 18., 24.]);
    }

    /// The chain fixture with `source` marked `dependency_only`, `required`
    /// naming the target's required fields.
    fn dependency_only_fixture(required: &[&str]) -> (Mapping, DecodedCollection) {
        let (mapping, collection, _) = chain_fixture(false);
        let mut document: Value = mapping.doc.to_value();
        document["fields"]["source"]["dependency_only"] = json!(true);
        document["target"] = json!({"require_lateral_boundaries": false,
            "required_fields": required.iter().map(|name| json!({"name": name})).collect::<Vec<_>>()});
        let payload = serde_json::to_vec(&document).unwrap();
        let mapping = Mapping { doc: crate::node::Node::parse(&payload).unwrap(),
            sha256: crate::digest::bytes_sha256(&payload), path: "<dependency-only-test>".to_owned() };
        (mapping, collection)
    }

    #[test]
    fn a_dependency_only_input_feeds_its_derivation_and_is_not_written() {
        let (mapping, collection) = dependency_only_fixture(&["output"]);
        let plan = plan_frames(&mapping, &collection.source_cycles).unwrap();
        assert_eq!(plan.declared_names, ["middle", "output"].map(str::to_owned).to_vec());
        let mut fields = FieldMaterializer::new(&mapping, collection, &plan, &plan.keys[0]).unwrap();
        assert_eq!(fields.names(), ["middle", "output"].map(str::to_owned).to_vec());
        // Priced as written: the disk admission reads this figure.
        assert_eq!(fields.planned_values(None).unwrap(), Some(8));
        let mut written = Vec::new();
        for id in fields.output.clone() {
            fields.materialize(id).unwrap();
            written.push(array::contiguous(&fields.available[&id].values).into_owned());
            fields.consume(id);
        }
        assert_eq!(written, vec![vec![2., 4., 6., 8.], vec![6., 12., 18., 24.]]);
        assert!(fields.collection.direct.is_empty(), "the input was read");
        assert!(fields.available.is_empty(), "and released after its last reader");
    }

    #[test]
    fn a_required_field_cannot_be_held_off_the_frame() {
        let (mapping, collection) = dependency_only_fixture(&["source", "output"]);
        let refusal = plan_frames(&mapping, &collection.source_cycles).err().unwrap();
        assert_eq!(refusal.message,
            "fields ['source'] are required by the target and marked dependency_only; \
             a frame must publish every required field");
    }

    #[test]
    fn a_direct_seed_keeps_its_version_when_derived_under_the_same_name() {
        let (mapping, collection, plan) = chain_fixture(true);
        let mut fields = FieldMaterializer::new(&mapping, collection, &plan, &plan.keys[0]).unwrap();
        let id = fields.output[0];
        fields.materialize(id).unwrap();
        assert_eq!(array::contiguous(&fields.available[&id].values).as_ref(), &[2., 4., 6., 8.]);
        assert_eq!(fields.available.len(), 1);
        fields.consume(id);
        assert!(fields.available.is_empty());
    }

    /// A soil column with nine temperature layers and eight water-mass
    /// layers, the water turned volumetric (eight) and then given a
    /// surface node above its shallowest layer (nine), on a 2 x 3 grid.
    fn layered_soil_fixture() -> (Mapping, DecodedCollection, FramePlan) {
        let soil = json!({"units": {"source": "1", "target": "1"}, "location": "soil",
            "source_axes": ["soil", "y", "x"], "target_axes": ["soil", "y", "x"]});
        let mut volumetric = soil.clone();
        volumetric["derivation"] = json!("volumetric-from-layer-mass");
        let mut surface_node = soil.clone();
        surface_node["derivation"] = json!("volumetric-with-surface-node");
        let bounds: Vec<Value> = (0..8).map(|layer| json!([layer as f64, layer as f64 + 1.0])).collect();
        let document = json!({"coordinates": {"vertical": {"kind": "pressure", "units": "Pa"}},
            "fields": {"soil_temperature": soil, "soil_water_column": soil,
                "soil_water_column_volumetric": volumetric, "volumetric_soil_moisture": surface_node},
            "derivations": [
                {"name": "volumetric-from-layer-mass", "operation": "volumetric_soil_moisture_from_layer_mass",
                 "layer_mass": "soil_water_column", "layer_bounds_m": bounds, "water_density_kg_m3": 1000.0},
                {"name": "volumetric-with-surface-node", "operation": "soil_surface_node_from_shallowest",
                 "source": "soil_water_column_volumetric"}]});
        let payload = serde_json::to_vec(&document).unwrap();
        let mapping = Mapping { doc: crate::node::Node::parse(&payload).unwrap(),
            sha256: crate::digest::bytes_sha256(&payload), path: "<layered-soil-test>".to_owned() };
        let valid_time = NaiveDateTime::parse_from_str("2026-08-17 06:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let column = |name: &str, layers: usize, value: f64| {
            ((valid_time, None, name.to_owned()), crate::assemble::DirectValue { name: name.to_owned(),
                valid_time, member: None, source_cycle: valid_time,
                axes: ["soil", "y", "x"].map(str::to_owned).to_vec(),
                values: ndarray::ArrayD::from_elem(ndarray::IxDyn(&[layers, 2, 3]), value),
                missing_count: 0, references: vec![format!("@{name}")] })
        };
        let collection = DecodedCollection { latitude: vec![1., 2.], longitude: vec![3., 4., 5.],
            vertical_values: vec![100000.], direct: BTreeMap::from([
                column("soil_temperature", 9, 280.0), column("soil_water_column", 8, 50.0)]),
            source_cycles: BTreeMap::from([((valid_time, None), valid_time)]),
            grid_fingerprint: "grid".to_owned(), hybrid_a: Vec::new(), hybrid_b: Vec::new(), original_pressure_hpa: None };
        let plan = FramePlan { keys: vec![(valid_time, None)],
            declared_names: ["soil_temperature", "soil_water_column", "soil_water_column_volumetric",
                "volumetric_soil_moisture"].map(str::to_owned).to_vec(),
            required_names: BTreeSet::new() };
        (mapping, collection, plan)
    }

    #[test]
    fn the_planned_stream_is_the_stream_written_when_soil_columns_differ_in_depth() {
        // The disk admission multiplies this figure by the valid times, so
        // a plan one soil layer taller than what is written refuses a
        // stream that fits.
        let (mapping, collection, plan) = layered_soil_fixture();
        let mut fields = FieldMaterializer::new(&mapping, collection, &plan, &plan.keys[0]).unwrap();
        let planned = fields.planned_values(None).unwrap().expect("every axis here is sizable");
        let mut written = 0u64;
        let mut layers = BTreeMap::new();
        for id in fields.output.clone() {
            fields.materialize(id).unwrap();
            let field = &fields.available[&id];
            written += field.values.len() as u64;
            layers.insert(field.name.clone(), field.values.shape()[0]);
            fields.consume(id);
        }
        assert_eq!(layers, BTreeMap::from([("soil_temperature".to_owned(), 9),
            ("soil_water_column".to_owned(), 8), ("soil_water_column_volumetric".to_owned(), 8),
            ("volumetric_soil_moisture".to_owned(), 9)]));
        assert_eq!(written, (9 + 8 + 8 + 9) * 6);
        assert_eq!(planned, written);
    }

    #[test]
    fn replaced_direct_inputs_are_validated_and_unseeded_cycles_refuse() {
        let (mapping, mut collection, plan) = chain_fixture(true);
        collection.direct.values_mut().next().unwrap().values[[0, 0]] = f64::INFINITY;
        let refusal = FieldMaterializer::new(&mapping, collection, &plan, &plan.keys[0]).err().unwrap();
        assert!(refusal.message.contains("contains infinity"));
        let (mapping, mut collection, plan) = chain_fixture(true);
        collection.direct.clear();
        let refusal = FieldMaterializer::new(&mapping, collection, &plan, &plan.keys[0]).err().unwrap();
        assert!(refusal.message.contains("missing dependencies or a cycle"));
    }

    #[test]
    fn field_writer_matches_the_complete_canonical_oracle_on_owned_netcdf_bytes() {
        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let golden_root = crate_root.join("tests/goldens");
        let golden: Value = serde_json::from_slice(&std::fs::read(golden_root.join("netcdf-pressure-level.json")).unwrap()).unwrap();
        let repository = crate_root.ancestors().nth(4).unwrap();
        let mapping_path = repository.join(golden["mapping"].as_str().unwrap());
        let mapping = Mapping::load(&mapping_path.display().to_string()).unwrap();
        let inputs: Vec<String> = golden["input_names"].as_array().unwrap().iter()
            .map(|name| golden_root.join(name.as_str().unwrap()).display().to_string()).collect();
        let collection = crate::engine::decode_collection(&mapping, &inputs, &mut |_| {}).unwrap();
        let oracle = materialize_frames(&mapping, &collection).unwrap();
        let series = SeriesSummary { source_cycles: collection.source_cycles.clone(),
            grid_fingerprint: collection.grid_fingerprint.clone(), lead_batch: false };
        let scratch = std::env::temp_dir().join(format!("gpuwm-field-writer-oracle-{}", std::process::id()));
        let mut remaining = collection;
        let document = write_frameset(&scratch, &mapping, &series, &crate::engine::input_digests(&inputs).unwrap(),
            |key| Ok(crate::engine::carve_valid_time(&mut remaining, key))).unwrap();
        let mut expected = Vec::new();
        for (index, frame) in oracle.iter().enumerate() {
            assert_eq!(document["frames"][index]["header"], frame.header);
            for field in &frame.fields {
                for value in array::contiguous(&field.values).iter() { expected.extend_from_slice(&value.to_le_bytes()); }
            }
        }
        assert_eq!(std::fs::read(scratch.join("frames.f64")).unwrap(), expected);
        std::fs::remove_file(scratch.join("frames.f64")).unwrap();
        std::fs::remove_file(scratch.join("frames.json")).unwrap();
        std::fs::remove_dir(scratch).unwrap();
    }

    fn cadence_plan(target: Value, minutes: &[i64]) -> Result<FramePlan> {
        let document = json!({"coordinates": {"vertical": {"kind": "pressure", "units": "Pa"}},
            "fields": {}, "target": target});
        let payload = serde_json::to_vec(&document).unwrap();
        let mapping = Mapping { doc: crate::node::Node::parse(&payload).unwrap(),
            sha256: crate::digest::bytes_sha256(&payload), path: "<cadence-test>".to_owned() };
        let cycle = NaiveDateTime::parse_from_str("2023-03-31 18:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let source_cycles = minutes.iter()
            .map(|minute| ((cycle + chrono::Duration::minutes(*minute), None), cycle))
            .collect();
        plan_frames(&mapping, &source_cycles)
    }

    #[test]
    fn a_target_declaring_multiples_takes_every_whole_multiple_of_its_spacing() {
        let exact = json!({"require_lateral_boundaries": true, "required_fields": [],
            "boundary_interval_seconds": 3600});
        let mut multiples = exact.clone();
        multiples["accept_boundary_interval_multiples"] = json!(true);
        let hourly = [0, 60, 120, 180];
        let three_hourly = [0, 180, 360, 540];
        let refusal = cadence_plan(exact.clone(), &three_hourly).err().unwrap();
        assert_eq!(refusal.message,
            "mapped cadence 10800 seconds differs from target contract Some(3600)");
        assert!(cadence_plan(exact, &hourly).is_ok());
        assert!(cadence_plan(multiples.clone(), &three_hourly).is_ok());
        assert!(cadence_plan(multiples.clone(), &hourly).is_ok());
        let refusal = cadence_plan(multiples, &[0, 90, 180]).err().unwrap();
        assert_eq!(refusal.message,
            "mapped cadence 5400 seconds is not a whole multiple of the 3600 seconds \
             the target contract declares");
    }

    /// The owned NetCDF golden, decoded whole, with the inputs it came from.
    fn netcdf_golden() -> (Mapping, DecodedCollection, Vec<String>) {
        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let golden_root = crate_root.join("tests/goldens");
        let golden: Value = serde_json::from_slice(&std::fs::read(golden_root.join("netcdf-pressure-level.json")).unwrap()).unwrap();
        let repository = crate_root.ancestors().nth(4).unwrap();
        let mapping = Mapping::load(&repository.join(golden["mapping"].as_str().unwrap()).display().to_string()).unwrap();
        let inputs: Vec<String> = golden["input_names"].as_array().unwrap().iter()
            .map(|name| golden_root.join(name.as_str().unwrap()).display().to_string()).collect();
        let collection = crate::engine::decode_collection(&mapping, &inputs, &mut |_| {}).unwrap();
        (mapping, collection, inputs)
    }

    #[test]
    fn required_analyzed_numbers_keep_masks_until_horizontal_repair() {
        // A required field must be present, but preserve_mask explicitly
        // delegates its finite-neighbor repair to the next stage. Rejecting
        // its NaNs here prevents that stage from reading legitimate input.
        for policy in ["preserve_mask", "reject"] {
            let (mapping, mut collection, _) = netcdf_golden();
            let mut document = mapping.doc.to_value();
            let name = "water_friendly_aerosol_number";
            let mut field = document["fields"]["air_temperature"].clone();
            field["units"] = json!({"source": "kg-1", "target": "kg-1"});
            field["missing"] = json!({"kind": policy});
            document["fields"][name] = field;
            document["target"]["required_fields"].as_array_mut().unwrap().push(
                json!({"name": name, "axes": ["vertical", "y", "x"],
                       "location": "mass", "target_units": "kg-1"}));
            let payload = serde_json::to_vec(&document).unwrap();
            let mapping = Mapping { doc: crate::node::Node::parse(&payload).unwrap(),
                sha256: crate::digest::bytes_sha256(&payload), path: "<masked-required>".to_owned() };
            let originals: Vec<_> = collection.direct.iter()
                .filter(|((_, _, field), _)| field == "air_temperature")
                .map(|(key, value)| (key.clone(), value.clone())).collect();
            for ((time, member, _), mut value) in originals {
                value.name = name.to_owned();
                *value.values.iter_mut().next().unwrap() = f64::NAN;
                value.missing_count = 1;
                collection.direct.insert((time, member, name.to_owned()), value);
            }
            let materialized = materialize_frames(&mapping, &collection);
            assert_eq!(materialized.is_ok(), policy == "preserve_mask");
            let plan = plan_frames(&mapping, &collection.source_cycles).unwrap();
            let mut fields = FieldMaterializer::new(&mapping, collection, &plan, &plan.keys[0]).unwrap();
            for id in fields.output.clone() {
                fields.materialize(id).unwrap();
                if fields.available[&id].name == name {
                    let field = fields.available[&id].clone();
                    let result = field_work(&fields, field, &plan, None, None, None,
                                            &json!({}), plan.keys[0].0);
                    assert_eq!(result.is_ok(), policy == "preserve_mask");
                }
            }
        }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("gpuwm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    /// Write the golden through `write_frameset_within` with `available`
    /// as the disk's free space; the result and how many valid times the
    /// writer pulled.
    fn write_golden(directory: &std::path::Path,
                    available: &(dyn Fn(&std::path::Path) -> Option<u64> + Sync))
        -> (Result<Value>, usize) {
        let (mapping, collection, inputs) = netcdf_golden();
        let series = SeriesSummary { source_cycles: collection.source_cycles.clone(),
            grid_fingerprint: collection.grid_fingerprint.clone(), lead_batch: false };
        let mut remaining = collection;
        let mut pulled = 0usize;
        let result = write_frameset_within(directory, &mapping, &series,
            &crate::engine::input_digests(&inputs).unwrap(), false, None, available, |key| {
                pulled += 1;
                Ok(crate::engine::carve_valid_time(&mut remaining, key))
            });
        (result, pulled)
    }

    type PositionedGolden = (Mapping, SeriesSummary, BTreeMap<String, String>,
                             Vec<std::sync::Mutex<Option<DecodedCollection>>>);

    /// The golden, carved into one collection per valid time, handed out
    /// by position to a many-lane writer.
    fn golden_by_position() -> PositionedGolden {
        let (mapping, collection, inputs) = netcdf_golden();
        let series = SeriesSummary { source_cycles: collection.source_cycles.clone(),
            grid_fingerprint: collection.grid_fingerprint.clone(), lead_batch: false };
        let keys: Vec<_> = collection.source_cycles.keys().cloned().collect();
        let mut remaining = collection;
        let slices = keys.iter()
            .map(|key| std::sync::Mutex::new(Some(crate::engine::carve_valid_time(&mut remaining, key))))
            .collect();
        (mapping, series, crate::engine::input_digests(&inputs).unwrap(), slices)
    }

    #[test]
    fn many_lanes_write_the_one_lane_frameset_byte_for_byte() {
        let one = scratch("lanes-one");
        let (document, frames) = write_golden(&one, &|_| None);
        document.unwrap();
        assert!(frames >= 2, "the golden is a series, so lanes overlap");
        for lanes in [2, 3, frames + 4] {
            let (mapping, series, digests, slices) = golden_by_position();
            let many = scratch(&format!("lanes-{lanes}"));
            write_frameset_core(&many, &mapping, &series, &digests, false, None, &|_| None, lanes,
                &|index, key| {
                    let slice = slices[index].lock().unwrap().take().expect("each valid time is pulled once");
                    assert!(slice.source_cycles.contains_key(key), "position {index} carries its own key");
                    Ok(slice)
                }).unwrap();
            for name in ["frames.json", "frames.f64"] {
                assert_eq!(std::fs::read(one.join(name)).unwrap(), std::fs::read(many.join(name)).unwrap(),
                    "{name} differs at {lanes} lanes");
            }
            std::fs::remove_dir_all(&many).unwrap();
        }
        std::fs::remove_dir_all(&one).unwrap();
    }

    #[test]
    fn a_full_prepared_batch_reopens_admission_after_its_last_commit() {
        use std::sync::{Arc, Mutex};
        let (mapping, mut collection, _) = netcdf_golden();
        let first = collection.source_cycles.keys().next().unwrap().clone();
        let template = crate::engine::carve_valid_time(&mut collection, &first);
        let cycle = *template.source_cycles.values().next().unwrap();
        let slices: Vec<_> = (0..13).map(|hour| {
            let time = cycle + chrono::Duration::hours(hour * 6);
            let mut collection = template.clone();
            collection.source_cycles = BTreeMap::from([((time, None), cycle)]);
            collection.direct = collection.direct.into_iter().map(|((_, member, name), mut value)| {
                value.valid_time = time;
                ((time, member, name), value)
            }).collect();
            Mutex::new(Some(collection))
        }).collect();
        let series = SeriesSummary {
            source_cycles: slices.iter().flat_map(|slot| {
                slot.lock().unwrap().as_ref().unwrap().source_cycles.clone()
            }).collect(),
            grid_fingerprint: template.grid_fingerprint.clone(), lead_batch: false,
        };
        let one = scratch("full-batch-one");
        let many = scratch("full-batch-many");
        let serial = |index: usize, _: &FrameKey| Ok(slices[index].lock().unwrap().as_ref().unwrap().clone());
        write_frameset_core(&one, &mapping, &series, &BTreeMap::new(), false,
            None, &|_| None, 1, &serial).unwrap();

        // Four lanes allow six decoded/prepared frames in all. Withhold
        // frame zero's prepared result until it is the only work in
        // flight, so it arrives after frames one to five (four and five
        // admitted while it was out) and the writer must commit a full
        // batch with no work left in flight.  There are seven further
        // frames to admit after that batch.  The writer does the
        // withholding on its own thread; no pool job waits on another.
        let arrivals = Arc::new(Mutex::new(Vec::new()));
        let seen = arrivals.clone();
        let hook: PipelinePreparedHook = Arc::new(move |index| {
            seen.lock().unwrap().push(index);
            index == 0
        });
        PIPELINE_PREPARED_HOOK.with(|slot| *slot.borrow_mut() = Some(hook));
        let outcome = write_frameset_core(&many, &mapping, &series, &BTreeMap::new(), false,
            None, &|_| None, 4, &|index, _| {
                Ok(slices[index].lock().unwrap().take().expect("each frame is pulled once"))
            });
        PIPELINE_PREPARED_HOOK.with(|slot| *slot.borrow_mut() = None);
        outcome.unwrap();
        let mut arrived = arrivals.lock().unwrap().clone();
        let batch: BTreeSet<usize> = arrived[..6].iter().copied().collect();
        assert_eq!(batch, (0..6).collect(), "frames after the batch were admitted before it committed");
        arrived.sort_unstable();
        assert_eq!(arrived, (0..13).collect::<Vec<_>>(), "every frame's prepared result arrived once");
        for name in ["frames.json", "frames.f64"] {
            assert_eq!(std::fs::read(one.join(name)).unwrap(),
                std::fs::read(many.join(name)).unwrap(), "{name} differs from serial output");
        }
        std::fs::remove_dir_all(one).unwrap();
        std::fs::remove_dir_all(many).unwrap();
    }

    #[test]
    fn many_lanes_report_the_first_refusal_in_frameset_order_after_writing_what_precedes_it() {
        // A later valid time's refusal produced FIRST must not win: the
        // one-lane writer would have stopped at the earlier one.
        let (mapping, series, digests, _slices) = golden_by_position();
        let lanes = series.source_cycles.len();
        assert!(lanes >= 2, "the golden needs two valid times");
        let early = scratch("lanes-refusal-order");
        let refusal = write_frameset_core(&early, &mapping, &series, &digests, false, None, &|_| None, lanes,
            &|index, _key| {
                if index == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(200));
                }
                Err(frame_invalid(format!("valid time {index} refused")))
            }).unwrap_err();
        assert_eq!(refusal.message, "valid time 0 refused");
        assert_eq!(std::fs::metadata(early.join("frames.f64")).unwrap().len(), 0);
        assert!(!early.join("frames.json").exists());
        std::fs::remove_dir_all(&early).unwrap();

        // What precedes the refusal is written whole, as the one-lane
        // writer writes it, and nothing after it.
        let one = scratch("lanes-refusal-one");
        let reference = {
            let (mapping, collection, inputs) = netcdf_golden();
            let series = SeriesSummary { source_cycles: collection.source_cycles.clone(),
                grid_fingerprint: collection.grid_fingerprint.clone(), lead_batch: false };
            let mut remaining = collection;
            let mut pulled = 0usize;
            write_frameset_within(&one, &mapping, &series,
                &crate::engine::input_digests(&inputs).unwrap(), false, None, &|_| None, |key| {
                    pulled += 1;
                    if pulled == 2 { return Err(frame_invalid("valid time 1 refused")); }
                    Ok(crate::engine::carve_valid_time(&mut remaining, key))
                })
        };
        assert_eq!(reference.unwrap_err().message, "valid time 1 refused");
        let (mapping, series, digests, slices) = golden_by_position();
        let many = scratch("lanes-refusal-many");
        let refusal = write_frameset_core(&many, &mapping, &series, &digests, false, None, &|_| None, lanes,
            &|index, _key| {
                if index == 1 { return Err(frame_invalid("valid time 1 refused")); }
                Ok(slices[index].lock().unwrap().take().expect("pulled once"))
            }).unwrap_err();
        assert_eq!(refusal.message, "valid time 1 refused");
        let written = std::fs::read(many.join("frames.f64")).unwrap();
        assert!(!written.is_empty());
        assert_eq!(std::fs::read(one.join("frames.f64")).unwrap(), written);
        assert!(!many.join("frames.json").exists());
        std::fs::remove_dir_all(&one).unwrap();
        std::fs::remove_dir_all(&many).unwrap();
    }

    #[test]
    fn a_job_that_panics_panics_the_writer_instead_of_hanging_it() {
        let (mapping, series, digests, slices) = golden_by_position();
        let lanes = series.source_cycles.len();
        let directory = scratch("lanes-panic");
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            write_frameset_core(&directory, &mapping, &series, &digests, false, None, &|_| None, lanes,
                &|index, _key| {
                    if index == 1 { panic!("valid time 1 panicked"); }
                    Ok(slices[index].lock().unwrap().take().expect("pulled once"))
                })
        }));
        assert!(outcome.is_err(), "the panic reached the caller");
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_stream_the_disk_cannot_hold_is_refused_before_its_first_byte_with_its_size() {
        let written = scratch("stream-admission-fits");
        let (document, frames) = write_golden(&written, &|_| None);
        let bytes = document.unwrap()["stream"]["bytes"].as_u64().unwrap();
        assert!(frames >= 2, "the golden is a series, so the plan multiplies");
        std::fs::remove_dir_all(&written).unwrap();

        // One byte short: refused as a full disk, naming the folder, the
        // exact size the stream would have had and the space there was,
        // after pulling only the first valid time and writing nothing.
        let short = scratch("stream-admission-short");
        let (refused, pulled) = write_golden(&short, &|_| Some(bytes - 1));
        let refusal = refused.err().unwrap();
        assert_eq!(refusal.class, crate::refusal::class::DISK_FULL);
        assert!(refusal.message.contains(&short.display().to_string()), "{}", refusal.message);
        assert!(refusal.message.contains(&format!("needs {bytes} bytes")), "{}", refusal.message);
        assert!(refusal.message.contains(&format!("has {} bytes", bytes - 1)), "{}", refusal.message);
        assert!(refusal.message.contains(&format!("{frames} valid times")), "{}", refusal.message);
        assert_eq!(pulled, 1);
        assert_eq!(std::fs::metadata(short.join("frames.f64")).unwrap().len(), 0);
        assert!(!short.join("frames.json").exists());
        std::fs::remove_dir_all(&short).unwrap();

        // Exactly enough is enough: the plan is the stream, not a margin.
        let exact = scratch("stream-admission-exact");
        let (document, _) = write_golden(&exact, &|_| Some(bytes));
        assert_eq!(document.unwrap()["stream"]["bytes"].as_u64(), Some(bytes));
        std::fs::remove_dir_all(&exact).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_write_that_meets_a_full_disk_is_a_disk_refusal_not_a_missing_input() {
        // `/dev/full` answers every write with ENOSPC, so a stream path
        // linked to it is a disk that fills at the first byte.
        if !std::path::Path::new("/dev/full").exists() {
            return;
        }
        let directory = scratch("stream-enospc");
        std::fs::create_dir_all(&directory).unwrap();
        std::os::unix::fs::symlink("/dev/full", directory.join("frames.f64")).unwrap();
        let (refused, _) = write_golden(&directory, &|_| Some(1 << 40));
        let refusal = refused.err().unwrap();
        assert_eq!(refusal.class, crate::refusal::class::DISK_FULL, "{}", refusal.message);
        assert!(refusal.message.starts_with(&format!(
            "cannot write the frame stream {}: ", directory.join("frames.f64").display())),
            "{}", refusal.message);
        assert!(refusal.message.contains("os error 28"), "{}", refusal.message);
        assert!(refusal.message.contains("the stream needs "), "{}", refusal.message);
        assert!(!refusal.remedy.contains("input list"), "{}", refusal.remedy);
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_directory_that_cannot_be_created_is_a_write_refusal_naming_it() {
        let blocker = scratch("stream-blocked");
        std::fs::write(&blocker, b"a file where the folder should be").unwrap();
        let directory = blocker.join("frames");
        let (refused, pulled) = write_golden(&directory, &|_| None);
        let refusal = refused.err().unwrap();
        assert_eq!(refusal.class, crate::refusal::class::WRITE_FAILED, "{}", refusal.message);
        assert!(refusal.message.starts_with(&format!(
            "cannot create the output directory {}: ", directory.display())), "{}", refusal.message);
        assert_eq!(pulled, 0);
        std::fs::remove_file(&blocker).unwrap();
    }

    #[test]
    fn naive_and_utc_isoformats_match_pythons_two_spellings() {
        let value =
            NaiveDateTime::parse_from_str("2026-08-17 06:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(naive_isoformat(value), "2026-08-17T06:00:00");
        assert_eq!(utc_isoformat(value), "2026-08-17T06:00:00+00:00");
    }
}
