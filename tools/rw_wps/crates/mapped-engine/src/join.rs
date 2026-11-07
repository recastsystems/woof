//! The composition JOINS: the coordinate-subset solve, terrain
//! composition and the cross-source bound-field borrow.
//!
//! Port of `gpuwm.mapped_composition._exact_subset_indices`,
//! `_compose_terrain`, `_binding_subset_indices`, `_take_subset` and
//! `_compose_bound_fields`.  These four are the numeric heart of
//! `compose`: they decide WHICH CELLS of a donor grid are borrowed and
//! under which clock, so an error here is silently offset data rather
//! than a refusal.
//!
//! Three transcription hazards are called out where they occur, because
//! each one reads as a harmless simplification:
//!
//! * the coordinate comparison is EXACT float equality, never a
//!   tolerance: an epsilon would accept grids the Python engine refuses;
//! * `np.mod(x, 360.0)` is floor-modulo, which is `f64::rem_euclid`, not
//!   `%`;
//! * both joins count the missing cells of the SUBSET they take, never
//!   the whole donor's: a donor's missing cells need not fall inside the
//!   borrowed window, and the canonical validator checks the count
//!   against the cropped array.  (The terrain path used to inherit the
//!   supplement's whole count; that equals the subset's on every
//!   composition that passed validation, so no accepted number moved.)
//! * two invariant records are EQUAL when their finite values are equal
//!   and their missing cells are the same cells: `NaN != NaN`, so plain
//!   array equality called one unchanged static different from itself.
//!
//! **What the compose goldens do and do not grade here.**  Measured
//! across every staged composed source: each one's supplement or donor
//! grid EQUALS its primary grid, so the index window solved below spans
//! the whole donor axis on all of them.  The goldens therefore grade
//! this solve in its degenerate full-window form only.  The strict
//! subset -- an interior window, a reversed axis, a donor spelling its
//! longitudes in [-180, 180) against a primary in [0, 360), a
//! non-contiguous match -- is graded by the unit tests at the bottom of
//! this file and by nothing else.  Do not read a green battery as
//! coverage of the window arithmetic; a source with a genuinely larger
//! donor grid would be the first golden that does.

use std::collections::{BTreeMap, BTreeSet};

use chrono::NaiveDateTime;
use ndarray::{ArrayD, Axis};
use serde_json::{json, Map, Value};

use crate::assemble::{DecodedCollection, DirectValue, TimeKey};
use crate::compose::{Binding, EXTERNAL_FIELD};
use crate::frames::naive_isoformat;
use crate::refusal::{frame_invalid, python_float_repr, python_list_repr, Result};

/// `gpuwm-mapped-exact-subset-binding-v1`.
pub const SUBSET_RECEIPT_SCHEMA: &str = "gpuwm-mapped-exact-subset-binding-v1";
/// `mapped_composition.BINDING_RECEIPT_SCHEMA`.
pub const BINDING_RECEIPT_SCHEMA: &str = "gpuwm-cross-source-binding-v1";

/// The two DECLARED shapes a terrain supplement's clock can take
/// (`mapped_composition._TERRAIN_TIME_ALIGNMENTS`).
pub const TERRAIN_TIME_ALIGNMENTS: [&str; 2] = ["valid_time_exact", "cycle_invariant_broadcast"];

/// The DECLARED clocks a cross-source binding can run on
/// (`mapped_composition._BOUND_TIME_ALIGNMENTS`).
pub const BOUND_TIME_ALIGNMENTS: [&str; 5] = [
    "cycle_invariant_broadcast",
    "source_cycle_analysis_broadcast",
    "source_cycle_analysis_select_broadcast",
    "source_cycle_window_start_broadcast",
    "valid_time_exact",
];

/// `datetime.__str__`: `2026-08-17 06:00:00`, the spelling `str(cycle)`
/// puts in the mixed-cycle refusals.
fn naive_str(value: NaiveDateTime) -> String {
    value.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// `repr(datetime.datetime(...))`, as a refusal that interpolates a list
/// of valid times spells them.
fn naive_repr(value: NaiveDateTime) -> String {
    use chrono::{Datelike, Timelike};

    let mut parts = vec![
        value.year().to_string(),
        value.month().to_string(),
        value.day().to_string(),
        value.hour().to_string(),
        value.minute().to_string(),
    ];
    // Python drops trailing zero second/microsecond components, and every
    // valid time in a mapped source is whole-second.
    if value.second() != 0 {
        parts.push(value.second().to_string());
    }
    format!("datetime.datetime({})", parts.join(", "))
}

/// `repr` of a `(valid_time, member)` key, as `_compose_terrain`'s
/// "lacks exact primary valid time(s)" refusal interpolates it.
fn time_key_repr(key: &TimeKey) -> String {
    let member = match &key.1 {
        Some(value) => crate::refusal::python_repr(value),
        None => "None".to_owned(),
    };
    format!("({}, {member})", naive_repr(key.0))
}

fn time_key_list_repr(keys: &[TimeKey]) -> String {
    let rendered: Vec<String> = keys.iter().map(time_key_repr).collect();
    format!("[{}]", rendered.join(", "))
}

/// `mapped_composition._exact_subset_indices`.
///
/// Every coordinate of `smaller` must have EXACTLY ONE exact match in
/// `larger`, and the matches must be contiguous and monotone.  The
/// comparison is `==` on float64 and nothing else: this is the function
/// that decides which cells of the donor are borrowed, and a tolerance
/// would accept a grid that is merely close, taking the wrong cells and
/// reporting a PASS receipt for them.
pub fn exact_subset_indices(
    larger: &[f64],
    smaller: &[f64],
    label: &str,
    cyclic_degrees: bool,
) -> Result<Vec<usize>> {
    exact_subset_indices_with_stride(larger, smaller, label, cyclic_degrees, false)
}

fn exact_subset_indices_with_stride(
    larger: &[f64],
    smaller: &[f64],
    label: &str,
    cyclic_degrees: bool,
    allow_strided: bool,
) -> Result<Vec<usize>> {
    // `np.mod(x, 360.0)`: floor-modulo, whose result takes the divisor's
    // sign.  Rust's `%` is truncating and would leave -0.25 where numpy
    // leaves 359.75, so a donor spelling longitudes in [-180, 180) would
    // match nothing against a primary spelling them in [0, 360).
    let fold = |values: &[f64]| -> Vec<f64> {
        if cyclic_degrees {
            values.iter().map(|value| value.rem_euclid(360.0)).collect()
        } else {
            values.to_vec()
        }
    };
    let larger_comparison = fold(larger);
    let smaller_comparison = fold(smaller);
    let mut indices: Vec<usize> = Vec::with_capacity(smaller_comparison.len());
    for value in &smaller_comparison {
        let matches: Vec<usize> = larger_comparison
            .iter()
            .enumerate()
            .filter(|(_, candidate)| *candidate == value)
            .map(|(index, _)| index)
            .collect();
        if matches.len() != 1 {
            return Err(frame_invalid(format!(
                "primary {label} coordinate np.float64({}) has {} exact matches \
                 in the terrain grid",
                python_float_repr(*value),
                matches.len()
            )));
        }
        indices.push(matches[0]);
    }
    if indices.len() > 1 {
        let ascending = indices.windows(2).all(|pair| pair[1] == pair[0] + 1);
        let descending = indices.windows(2).all(|pair| pair[0] == pair[1] + 1);
        let mut contiguous = ascending || descending;
        if !contiguous
            && cyclic_degrees
            && indices.len() <= larger.len()
            && closed_longitude_circle(larger)
        {
            // A donor axis that closes the circle has its last and first
            // cells as neighbours, so a primary crossing that seam is
            // still a run of adjacent cells.  At most one full turn: a
            // longer request would repeat a cell.
            let size = larger.len();
            let wrapped: Vec<usize> = indices
                .windows(2)
                .map(|pair| (pair[1] + size - pair[0]) % size)
                .collect();
            contiguous = wrapped.iter().all(|step| *step == 1)
                || wrapped.iter().all(|step| *step == size - 1);
        }
        if !contiguous && allow_strided {
            contiguous = regular_index_stride(
                &indices, larger.len(),
                cyclic_degrees && closed_longitude_circle(larger),
            ).is_some();
        }
        if !contiguous {
            let kind = if allow_strided { "regular strided" } else { "contiguous" };
            return Err(frame_invalid(format!(
                "primary {label} is not a {kind} terrain-grid subset"
            )));
        }
    }
    Ok(indices)
}

/// Exact regular selection, with no repeated cells or second circle.
fn regular_index_stride(indices: &[usize], size: usize, cyclic: bool) -> Option<i64> {
    if indices.len() < 2 {
        return Some(0);
    }
    let differences: Vec<i64> = indices.windows(2)
        .map(|pair| pair[1] as i64 - pair[0] as i64).collect();
    let mut step = differences[0];
    if step != 0 && differences.iter().all(|value| *value == step) {
        return Some(step);
    }
    if cyclic {
        let size = size as i64;
        let wrapped: Vec<i64> = differences.iter()
            .map(|value| value.rem_euclid(size)).collect();
        step = wrapped[0];
        if step > size / 2 { step -= size; }
        if step != 0 && wrapped.iter().all(|value| *value == step.rem_euclid(size))
            && step.abs() * ((indices.len() - 1) as i64) < size
        {
            return Some(step);
        }
    }
    None
}

/// `mapped_composition._closed_longitude_circle`: whether a donor
/// longitude axis goes once round the globe, evenly.
///
/// Its cells, taken in index order and wrapping from the last back to the
/// first, must step the same way round the circle, one turn in all, with
/// no step as long as 1.5 of the shortest: a regional axis fails on its
/// wrap step and an axis missing a cell fails on the step that skips it.
/// Coordinates are still matched EXACTLY above; this only decides whether
/// the index sequence may wrap, so the bound picks no cell.
fn closed_longitude_circle(longitude: &[f64]) -> bool {
    let folded: Vec<f64> = longitude.iter().map(|value| value.rem_euclid(360.0)).collect();
    let size = folded.len();
    if size < 2 {
        return false;
    }
    for backward in [false, true] {
        let steps: Vec<f64> = (0..size)
            .map(|index| {
                let (here, next) = (folded[index], folded[(index + 1) % size]);
                if backward {
                    (here - next).rem_euclid(360.0)
                } else {
                    (next - here).rem_euclid(360.0)
                }
            })
            .collect();
        let shortest = steps.iter().copied().fold(f64::INFINITY, f64::min);
        let longest = steps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let turns = (steps.iter().sum::<f64>() / 360.0).round();
        if shortest > 0.0 && longest < 1.5 * shortest && turns == 1.0 {
            return true;
        }
    }
    false
}

/// `int(np.sign(last - first))` over the solved index range.
fn index_direction(indices: &[usize]) -> i64 {
    let first = indices[0] as i64;
    let last = indices[indices.len() - 1] as i64;
    (last - first).signum()
}

/// `mapped_composition._index_direction`: +1 or -1 for the way a
/// longitude run steps, across the seam or not.  An unwrapped run keeps
/// [`index_direction`], the value every receipt already carries.
fn wrapped_index_direction(indices: &[usize], size: usize) -> i64 {
    if indices.len() < 2 {
        return 0;
    }
    let ascending = indices.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let descending = indices.windows(2).all(|pair| pair[0] == pair[1] + 1);
    if ascending || descending {
        return index_direction(indices);
    }
    if (indices[1] + size - indices[0]) % size == 1 {
        1
    } else {
        -1
    }
}

/// Equality for records that must be invariant: same shape, equal finite
/// values, and missing cells in the same places
/// (`np.array_equal(..., equal_nan=True)`).
fn invariant_values_equal(left: &ArrayD<f64>, right: &ArrayD<f64>) -> bool {
    left.shape() == right.shape()
        && left
            .iter()
            .zip(right.iter())
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
}

/// `values[np.ix_(latitude_indices, longitude_indices)]` on a 2-D field.
fn take_grid(values: &ArrayD<f64>, latitude: &[usize], longitude: &[usize]) -> ArrayD<f64> {
    values
        .select(Axis(0), latitude)
        .select(Axis(1), longitude)
}

/// `mapped_composition._take_subset`: take along the axes NAMED `y` and
/// `x`, wherever the field's own axis order puts them.
fn take_subset(
    values: &ArrayD<f64>,
    axes: &[String],
    latitude: &[usize],
    longitude: &[usize],
) -> Result<ArrayD<f64>> {
    let position = |name: &str| -> Result<usize> {
        axes.iter().position(|axis| axis == name).ok_or_else(|| {
            frame_invalid(format!(
                "a borrowed field declares axes {} with no '{name}' axis to \
                 subset along",
                python_list_repr(axes)
            ))
        })
    };
    let result = values.select(Axis(position("y")?), latitude);
    Ok(result.select(Axis(position("x")?), longitude))
}

/// The refusal [`take_subset`] makes, without taking the subset: the
/// field must name a `y` axis, then an `x` axis.
fn check_subset_axes(axes: &[String]) -> Result<()> {
    for name in ["y", "x"] {
        if !axes.iter().any(|axis| axis == name) {
            return Err(frame_invalid(format!(
                "a borrowed field declares axes {} with no '{name}' axis to \
                 subset along",
                python_list_repr(axes)
            )));
        }
    }
    Ok(())
}

fn array_digest(values: &ArrayD<f64>) -> String {
    crate::digest::array_sha256(values.shape(), &crate::array::contiguous(values))
}

fn axis_digest(values: &[f64]) -> String {
    crate::digest::array_sha256(&[values.len()], values)
}

/// Everything a join reads about the primary decode that is NOT one of
/// its arrays: the grid, the vertical ladder, the series clock and the
/// field inventory.
///
/// Split out so a composition can be RESOLVED -- every refusal made,
/// every receipt written -- against a primary that is being decoded one
/// valid time at a time.  Both joins below turned out to read the
/// primary's arrays nowhere: they only insert into it.
pub struct PrimaryHeader<'a> {
    pub latitude: &'a [f64],
    pub longitude: &'a [f64],
    pub vertical_values: &'a [f64],
    pub source_cycles: &'a BTreeMap<TimeKey, NaiveDateTime>,
    pub direct_names: &'a BTreeSet<String>,
}

impl PrimaryHeader<'_> {
    /// The `(valid_time, member)` keys, in Python's order.
    fn keys(&self) -> Vec<TimeKey> {
        self.source_cycles.keys().cloned().collect()
    }
}

/// What one valid time inherits from the terrain supplement.
struct TerrainEntry {
    source_cycle: NaiveDateTime,
    axes: Vec<String>,
    missing_count: usize,
    references: Vec<String>,
}

/// The terrain join, resolved before a primary valid time is read.
///
/// One array, not one per valid time: the join refuses a terrain subset
/// that changes across the series, so the series it accepts has exactly
/// one array to insert.
pub struct TerrainPlan {
    values: ArrayD<f64>,
    entries: BTreeMap<TimeKey, TerrainEntry>,
}

/// Insert this valid time's terrain into the slice being composed.
pub fn apply_terrain(
    plan: &TerrainPlan,
    slice: &mut DecodedCollection,
    key: &TimeKey,
) -> Result<()> {
    let entry = plan.entries.get(key).ok_or_else(|| {
        frame_invalid(format!(
            "terrain composition has no binding for {}",
            naive_isoformat(key.0)
        ))
    })?;
    slice.direct.insert(
        (key.0, key.1.clone(), EXTERNAL_FIELD.to_owned()),
        DirectValue {
            name: EXTERNAL_FIELD.to_owned(),
            valid_time: key.0,
            member: key.1.clone(),
            source_cycle: entry.source_cycle,
            axes: entry.axes.clone(),
            values: plan.values.clone(),
            // Counted on this subset by `plan_terrain`: the array the
            // canonical validator checks the count against.
            missing_count: entry.missing_count,
            references: entry.references.clone(),
        },
    );
    Ok(())
}

/// `mapped_composition._compose_terrain`, resolved against the header.
pub fn plan_terrain(
    primary: &PrimaryHeader,
    terrain: &DecodedCollection,
    time_alignment: &str,
    grid_alignment: &str,
) -> Result<(TerrainPlan, Value)> {
    if !TERRAIN_TIME_ALIGNMENTS.contains(&time_alignment) {
        let allowed: Vec<&str> = {
            let mut names = TERRAIN_TIME_ALIGNMENTS.to_vec();
            names.sort_unstable();
            names
        };
        return Err(frame_invalid(format!(
            "terrain supplement time_alignment must be one of {}, got {}",
            python_list_repr(&allowed),
            crate::refusal::python_repr(time_alignment)
        )));
    }
    if primary.direct_names.iter().any(|name| name == EXTERNAL_FIELD) {
        return Err(frame_invalid(
            "terrain has two providers: the primary source and the declared supplement",
        ));
    }
    let inventory: BTreeSet<&str> = terrain
        .direct
        .keys()
        .map(|(_time, _member, name)| name.as_str())
        .collect();
    if inventory.iter().copied().ne([EXTERNAL_FIELD]) {
        let names: Vec<&str> = inventory.into_iter().collect();
        return Err(frame_invalid(format!(
            "terrain supplement decoded unexpected fields {}",
            python_list_repr(&names)
        )));
    }
    let allow_strided = grid_alignment == "exact_coordinate_strided_subset";
    let latitude_indices = exact_subset_indices_with_stride(
        &terrain.latitude, primary.latitude, "latitude", false, allow_strided,
    )?;
    let longitude_indices = exact_subset_indices_with_stride(
        &terrain.longitude, primary.longitude, "longitude", true, allow_strided,
    )?;
    // `sorted(terrain.direct.items(), key=(valid_time, str(member)))`.
    // The map is already keyed `(valid_time, member, field)` and the
    // inventory above proved there is exactly one field name, so the map
    // order IS that sort.
    let terrain_items: Vec<(&NaiveDateTime, &Option<String>, &DirectValue)> = terrain
        .direct
        .iter()
        .map(|((time, member, _field), value)| (time, member, value))
        .collect();
    if terrain_items.is_empty() {
        return Err(frame_invalid("terrain supplement decoded no terrain messages"));
    }
    let full_reference = &terrain_items[0].2.values;
    if terrain_items[1..]
        .iter()
        .any(|(_time, _member, value)| !invariant_values_equal(&value.values, full_reference))
    {
        return Err(frame_invalid(
            "terrain supplement changes across supplied valid times",
        ));
    }
    let keys = primary.keys();
    let terrain_by_time: BTreeMap<TimeKey, &DirectValue> = terrain_items
        .iter()
        .map(|(time, member, value)| ((**time, (*member).clone()), *value))
        .collect();
    let missing_times: Vec<TimeKey> = keys
        .iter()
        .filter(|key| !terrain_by_time.contains_key(key))
        .cloned()
        .collect();
    if !missing_times.is_empty() && time_alignment == "valid_time_exact" {
        return Err(frame_invalid(format!(
            "terrain supplement lacks exact primary valid time(s) {}",
            time_key_list_repr(&missing_times)
        )));
    }
    if time_alignment != "valid_time_exact" {
        // The producer publishes terrain at the analysis step only; the
        // supplement's one proven-invariant record answers every primary
        // valid time of the one source cycle.  One broadcast belongs to
        // one cycle: mixed primary cycles refuse rather than share an
        // invariant record no cycle proved for itself.
        let cycles: BTreeSet<String> = primary.source_cycles.values().copied().map(naive_str).collect();
        if cycles.len() != 1 {
            let names: Vec<String> = cycles.into_iter().collect();
            return Err(frame_invalid(format!(
                "cycle-invariant terrain broadcast cannot span mixed primary \
                 source cycles {}",
                python_list_repr(&names)
            )));
        }
    }
    let carrier = terrain_items[0].2;
    // ONE subset is kept, not one per valid time: the equality check
    // below is what makes that sound, and it is the check the join
    // already made.
    let mut subset_reference: Option<ArrayD<f64>> = None;
    let mut entries: BTreeMap<TimeKey, TerrainEntry> = BTreeMap::new();
    for key in &keys {
        let supplied = terrain_by_time.get(key).copied().unwrap_or(carrier);
        let values = take_grid(&supplied.values, &latitude_indices, &longitude_indices);
        // The subset's own count, not the supplement's: missing donor
        // cells outside the primary window are not in this array.
        let missing_count = crate::array::count_nan(&values);
        match &subset_reference {
            None => subset_reference = Some(values),
            Some(reference) => {
                if !invariant_values_equal(reference, &values) {
                    return Err(frame_invalid(
                        "terrain subset changes across primary valid times",
                    ));
                }
            }
        }
        entries.insert(
            key.clone(),
            TerrainEntry {
                source_cycle: supplied.source_cycle,
                axes: supplied.axes.clone(),
                missing_count,
                references: supplied.references.clone(),
            },
        );
    }
    // A primary with no valid times would leave the receipt with no
    // subset to hash.  `materialize_frames` refuses that collection a
    // step later; refusing here keeps the composition from writing a
    // receipt describing a binding it never performed.
    let Some(subset_reference) = subset_reference else {
        return Err(frame_invalid(
            "terrain composition has no primary valid time to bind to",
        ));
    };
    let mut receipt = Map::new();
    receipt.insert("schema".into(), json!(SUBSET_RECEIPT_SCHEMA));
    receipt.insert("status".into(), json!("PASS"));
    receipt.insert("field".into(), json!(EXTERNAL_FIELD));
    receipt.insert(
        "primary_shape".into(),
        json!([primary.latitude.len(), primary.longitude.len()]),
    );
    receipt.insert(
        "supplement_shape".into(),
        json!([terrain.latitude.len(), terrain.longitude.len()]),
    );
    receipt.insert(
        "latitude_index_range".into(),
        json!([
            latitude_indices[0],
            latitude_indices[latitude_indices.len() - 1]
        ]),
    );
    receipt.insert(
        "longitude_index_range".into(),
        json!([
            longitude_indices[0],
            longitude_indices[longitude_indices.len() - 1]
        ]),
    );
    receipt.insert("latitude_sha256".into(), json!(axis_digest(primary.latitude)));
    receipt.insert(
        "longitude_sha256".into(),
        json!(axis_digest(primary.longitude)),
    );
    receipt.insert(
        "terrain_full_sha256".into(),
        json!(array_digest(full_reference)),
    );
    receipt.insert(
        "terrain_subset_sha256".into(),
        json!(array_digest(&subset_reference)),
    );
    receipt.insert(
        "supplement_valid_times".into(),
        json!(terrain_items
            .iter()
            .map(|(time, _member, _value)| naive_isoformat(**time))
            .collect::<Vec<String>>()),
    );
    receipt.insert(
        "matched_primary_valid_times".into(),
        json!(keys
            .iter()
            .map(|key| naive_isoformat(key.0))
            .collect::<Vec<String>>()),
    );
    if time_alignment != "valid_time_exact" {
        // Recorded ONLY under the broadcast, so every receipt written
        // before these two keys existed is byte-identical to what the
        // same preparation writes today.
        receipt.insert("time_alignment".into(), json!(time_alignment));
        receipt.insert(
            "broadcast_primary_valid_times".into(),
            json!(missing_times
                .iter()
                .map(|key| naive_isoformat(key.0))
                .collect::<Vec<String>>()),
        );
    }
    receipt.insert("invariant_across_all_supplement_times".into(), json!(true));
    receipt.insert(
        "latitude_index_direction".into(),
        json!(index_direction(&latitude_indices)),
    );
    receipt.insert(
        "longitude_index_direction".into(),
        json!(wrapped_index_direction(&longitude_indices, terrain.longitude.len())),
    );
    receipt.insert(
        "coordinate_match".into(),
        json!("exact_equivalent_contiguous_subset"),
    );
    receipt.insert("longitude_equivalence".into(), json!("modulo_360_exact"));
    Ok((
        // The composed frame keeps the PRIMARY's vertical identity, its
        // hybrid coefficient ladder included: the slice `apply_terrain`
        // inserts into IS the primary's own decode of that valid time,
        // so the ladder rides the slice and the plan holds none of it.
        TerrainPlan {
            values: subset_reference,
            entries,
        },
        Value::Object(receipt),
    ))
}

/// `mapped_composition._binding_subset_indices`: the same solve, with the
/// refusal that names the missing regrid capability.
fn binding_subset_indices(
    primary: &PrimaryHeader,
    donor: &DecodedCollection,
    binding_name: &str,
    grid_alignment: &str,
) -> Result<(Vec<usize>, Vec<usize>)> {
    if grid_alignment == "nearest_coordinate" {
        // Explicit nearest-cell sampling for categorical surface analyses.
        // Longitude is cyclic; ties resolve to the first published cell.
        let nearest = |source: &[f64], target: &[f64], cyclic: bool| -> Result<Vec<usize>> {
            if source.is_empty() || source.iter().chain(target).any(|v| !v.is_finite()) {
                return Err(frame_invalid("nearest-coordinate binding requires finite nonempty axes"));
            }
            target.iter().map(|v| {
                let distance = |x: f64| {
                    let d = (x-v).abs(); if cyclic { let d=d.rem_euclid(360.0); d.min(360.0-d) } else { d }
                };
                source.iter().enumerate().min_by(|a,b| distance(*a.1).total_cmp(&distance(*b.1)))
                    .map(|(i,_)| i).ok_or_else(|| frame_invalid("empty donor axis"))
            }).collect()
        };
        return Ok((nearest(&donor.latitude, primary.latitude, false)?,
                   nearest(&donor.longitude, primary.longitude, true)?));
    }
    let allow_strided = grid_alignment == "exact_coordinate_strided_subset";
    let wrap = |error: crate::refusal::Refusal| -> crate::refusal::Refusal {
        frame_invalid(format!(
            "contributing source binding {} is cross-grid ({}); borrowing a \
             field across grids requires a horizontal regrid capability this \
             composition does not declare, so it refuses rather than \
             interpolate",
            crate::refusal::python_repr(binding_name),
            error.message
        ))
    };
    let latitude = exact_subset_indices_with_stride(
        &donor.latitude, primary.latitude, "latitude", false, allow_strided)
        .map_err(wrap)?;
    let longitude = exact_subset_indices_with_stride(
        &donor.longitude, primary.longitude, "longitude", true, allow_strided).map_err(wrap)?;
    Ok((latitude, longitude))
}

/// One contributing-source borrow, resolved before a primary valid time
/// is read.
///
/// The donor RECORD each primary valid time borrows is named here; the
/// subset itself is taken when that valid time is composed, so the plan
/// costs a key per field per time rather than an array.
pub struct BindingPlan {
    latitude_indices: Vec<usize>,
    longitude_indices: Vec<usize>,
    chosen: BTreeMap<(String, NaiveDateTime), crate::assemble::DirectKey>,
    vertical_merge: Option<crate::vertical::VerticalMerge>,
    binding_name: String,
    companion: Option<CompanionPlan>,
}

impl BindingPlan {
    /// The ladder every slice carries once this binding is applied: the
    /// companion's composed ladder, or `None` when the binding lends no
    /// levels.
    pub fn composed_ladder(&self) -> Option<&[f64]> {
        self.companion.as_ref().map(|plan| plan.composed.as_slice())
    }

    /// Bytes one valid time gains from the companion's levels, counted
    /// twice (the slab and the joined column it is copied into).
    pub fn extension_bytes(&self) -> u64 {
        self.companion.as_ref().map_or(0, |plan| plan.extension_bytes)
    }
}

/// A declared companion's ladder extension, resolved before a primary
/// valid time is read.
struct CompanionPlan {
    binding: String,
    fields: Vec<String>,
    primary: Vec<f64>,
    composed: Vec<f64>,
    above: Vec<usize>,
    ascending: bool,
    chosen: BTreeMap<(String, NaiveDateTime), crate::assemble::DirectKey>,
    extension_bytes: u64,
}

/// `mapped_composition.companion_ladder`: the composed ladder, the
/// companion level indices above the primary's top in stacking order, and
/// whether the primary ladder runs top first.
pub fn companion_ladder(
    primary: &[f64],
    companion: &[f64],
    binding_name: &str,
) -> Result<(Vec<f64>, Vec<usize>, bool)> {
    let quoted = crate::refusal::python_repr(binding_name);
    if primary.is_empty() {
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} declares a companion, but \
             the primary decode carries no vertical ladder to extend"
        )));
    }
    let top = primary.iter().copied().fold(f64::INFINITY, f64::min);
    let mut above: Vec<usize> = companion
        .iter()
        .enumerate()
        .filter(|(_, level)| **level < top)
        .map(|(index, _)| index)
        .collect();
    if above.is_empty() {
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} declares a companion that \
             extends the ladder above the primary's top ({}), but the \
             companion carries no level above it",
            python_float_repr(top)
        )));
    }
    let ascending = primary.len() < 2 || primary[0] < primary[primary.len() - 1];
    // Python's stable `sort(key=..., reverse=...)`: ties keep their order.
    above.sort_by(|left, right| {
        let order = companion[*left].total_cmp(&companion[*right]);
        if ascending { order } else { order.reverse() }
    });
    let stacked: Vec<f64> = above.iter().map(|index| companion[*index]).collect();
    let composed: Vec<f64> = if ascending {
        stacked.iter().chain(primary.iter()).copied().collect()
    } else {
        primary.iter().chain(stacked.iter()).copied().collect()
    };
    Ok((composed, above, ascending))
}

/// Stack the companion's levels above the primary's top, at one valid time.
fn apply_companion(
    plan: &BindingPlan,
    companion: &CompanionPlan,
    donor: &DecodedCollection,
    slice: &mut DecodedCollection,
    key: &TimeKey,
    decode_window: Option<&crate::window::Window>,
) -> Result<()> {
    let quoted = crate::refusal::python_repr(&companion.binding);
    if slice.vertical_values != companion.primary {
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} was resolved against a \
             primary ladder this valid time does not carry; the composed \
             column would join levels it was never planned for"
        )));
    }
    let short: BTreeSet<&str> = slice
        .direct
        .iter()
        .filter(|((time, member, name), value)| {
            *time == key.0
                && *member == key.1
                && value.axes.iter().any(|axis| axis == "vertical")
                && !companion.fields.iter().any(|field| field == name)
        })
        .map(|((_, _, name), _)| name.as_str())
        .collect();
    if !short.is_empty() {
        let mut fields = companion.fields.clone();
        fields.sort();
        let short: Vec<&str> = short.into_iter().collect();
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} extends {} above the \
             primary's top, but the primary also decoded {} on its ladder; \
             those columns would stop at the primary's top",
            python_list_repr(&fields),
            python_list_repr(&short)
        )));
    }
    for name in &companion.fields {
        let entry = (key.0, key.1.clone(), name.clone());
        let Some(own) = slice.direct.get(&entry) else {
            return Err(frame_invalid(format!(
                "contributing source binding {quoted} extends {}, which the \
                 primary decode does not carry at {}",
                crate::refusal::python_repr(name),
                naive_isoformat(key.0)
            )));
        };
        let donor_key = companion.chosen.get(&(name.clone(), key.0)).ok_or_else(|| {
            frame_invalid(format!(
                "contributing source binding {quoted} lacks {} at primary \
                 valid time(s) [{}]",
                crate::refusal::python_repr(name),
                naive_repr(key.0)
            ))
        })?;
        let value = donor.direct.get(donor_key).ok_or_else(|| {
            frame_invalid(format!(
                "the contributing source no longer carries {} at {}",
                crate::refusal::python_repr(name),
                naive_isoformat(donor_key.0)
            ))
        })?;
        if value.axes != own.axes {
            return Err(frame_invalid(format!(
                "contributing source binding {quoted} field {} has axes {} \
                 where the primary has {}; a column is joined on one axis \
                 order",
                crate::refusal::python_repr(name),
                python_list_repr(&value.axes),
                python_list_repr(&own.axes)
            )));
        }
        let axis = own
            .axes
            .iter()
            .position(|axis| axis == "vertical")
            .expect("the primary field was planned on its vertical axis");
        // A primary atmospheric field may already have been decoded over
        // the granted window while borrowed surface fields stay whole. Cut
        // the donor in that same source index space before stacking it. A
        // hydrostatic-completion fallback can decode the primary whole even
        // when a window was granted, so its actual shape remains authoritative.
        let y_axis = own.axes.iter().position(|axis| axis == "y").unwrap();
        let x_axis = own.axes.iter().position(|axis| axis == "x").unwrap();
        let full_shape = [plan.latitude_indices.len(), plan.longitude_indices.len()];
        let own_shape = [own.values.shape()[y_axis], own.values.shape()[x_axis]];
        let (latitude, longitude) = match decode_window {
            Some(window) if own_shape != full_shape
                && window.fields.contains(name)
                && window.source_shape == full_shape
                && own_shape == [window.rows[1] - window.rows[0],
                                 window.columns[1] - window.columns[0]] =>
            {
                (&plan.latitude_indices[window.rows[0]..window.rows[1]],
                 &plan.longitude_indices[window.columns[0]..window.columns[1]])
            }
            _ => (plan.latitude_indices.as_slice(), plan.longitude_indices.as_slice()),
        };
        let slab = take_subset(&value.values, &value.axes, latitude, longitude)?
            .select(Axis(axis), &companion.above);
        let joined = if companion.ascending {
            ndarray::concatenate(Axis(axis), &[slab.view(), own.values.view()])
        } else {
            ndarray::concatenate(Axis(axis), &[own.values.view(), slab.view()])
        }
        .map_err(|error| {
            frame_invalid(format!(
                "contributing source binding {quoted} cannot join {} to the \
                 primary column: donor {:?}, primary {:?}: {error}",
                crate::refusal::python_repr(name), slab.shape(), own.values.shape()
            ))
        })?;
        let missing_count = crate::array::contiguous(&joined)
            .iter()
            .filter(|item| item.is_nan())
            .count();
        let mut references = own.references.clone();
        references.extend(value.references.iter().cloned());
        let joined_value = DirectValue {
            name: name.clone(),
            valid_time: key.0,
            member: key.1.clone(),
            source_cycle: own.source_cycle,
            axes: own.axes.clone(),
            values: joined,
            missing_count,
            references,
        };
        slice.direct.insert(entry, joined_value);
    }
    slice.vertical_values = companion.composed.clone();
    Ok(())
}

/// Insert this valid time's borrowed fields into the slice being composed.
pub fn apply_bound_fields(
    plan: &BindingPlan,
    donor: &DecodedCollection,
    slice: &mut DecodedCollection,
    key: &TimeKey,
    decode_window: Option<&crate::window::Window>,
) -> Result<Option<Value>> {
    // Each borrowed field is its own subset of its own donor array, so
    // they are cut on every core and inserted in the plan's order; the
    // refusal is the first one in that order.  Named breakage: the
    // borrowed soil and vegetation fields were cut one after another,
    // about 0.6 s of every 3 km CONUS valid time on one core.
    let chosen: Vec<(&String, &crate::assemble::DirectKey)> = plan.chosen.iter()
        .filter(|((_name, time), _donor_key)| *time == key.0)
        .map(|((name, _time), donor_key)| (name, donor_key))
        .collect();
    let cut: Vec<Result<DirectValue>> = crate::threads::install(|| {
        use rayon::prelude::*;
        chosen.par_iter().map(|(name, donor_key)| {
            let value = donor.direct.get(*donor_key).ok_or_else(|| {
                frame_invalid(format!(
                    "the contributing source no longer carries {} at {}",
                    crate::refusal::python_repr(name),
                    naive_isoformat(donor_key.0)
                ))
            })?;
            let values = take_subset(
                &value.values,
                &value.axes,
                &plan.latitude_indices,
                &plan.longitude_indices,
            )?;
            let missing_count = crate::array::contiguous(&values)
                .iter()
                .filter(|item| item.is_nan())
                .count();
            Ok(DirectValue {
                name: (*name).clone(),
                valid_time: key.0,
                member: key.1.clone(),
                source_cycle: value.source_cycle,
                axes: value.axes.clone(),
                values,
                // Counted on the subset: a donor grid's missing cells
                // need not fall inside the borrowed window.
                missing_count,
                references: value.references.clone(),
            })
        }).collect()
    });
    let cut = crate::threads::in_order(cut)?;
    let receipt = if let Some(merge) = &plan.vertical_merge {
        Some(merge.apply(slice, &cut, &donor.vertical_values, key, &plan.binding_name)?)
    } else { None };
    for ((name, _donor_key), value) in chosen.iter().zip(cut) {
        if !plan.vertical_merge.as_ref().is_some_and(|merge| merge.fields.contains_key(*name)) {
            slice.direct.insert((key.0, key.1.clone(), (*name).clone()), value);
        }
    }
    if let Some(companion) = &plan.companion {
        apply_companion(plan, companion, donor, slice, key, decode_window)?;
    }
    Ok(receipt)
}

/// `mapped_composition._compose_bound_fields`, resolved against the header.
pub fn plan_bound_fields(
    primary: &PrimaryHeader,
    donor: &DecodedCollection,
    binding: &Binding,
) -> Result<(BindingPlan, Value)> {
    let binding_name = binding.name.as_str();
    let quoted = crate::refusal::python_repr(binding_name);
    let alignment = binding.time_alignment.as_str();
    if !BOUND_TIME_ALIGNMENTS.contains(&alignment) {
        return Err(frame_invalid(format!(
            "binding time_alignment must be one of {}, got {}",
            python_list_repr(&BOUND_TIME_ALIGNMENTS),
            crate::refusal::python_repr(alignment)
        )));
    }
    let donor_members: BTreeSet<Option<String>> = donor
        .source_cycles
        .keys()
        .map(|(_time, member)| member.clone())
        .collect();
    if donor_members.len() != 1 {
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} decoded {} members; \
             cross-source borrowing has no member-alignment capability and \
             requires a single-member donor",
            donor_members.len()
        )));
    }
    let donor_member = donor_members.into_iter().next().expect("one donor member");
    let extended: Vec<String> = binding
        .companion
        .as_ref()
        .map_or_else(Vec::new, |companion| companion.fields.clone());
    // Python's `sorted(..., key=(is not None, str))`: an absent member
    // first, then the declared ones by name.
    let primary_members: Vec<Option<String>> = primary
        .source_cycles
        .keys()
        .map(|(_time, member)| member.clone())
        .collect::<BTreeSet<Option<String>>>()
        .into_iter()
        .collect();
    // Two product-declared members that differ are two trajectories.  A
    // donor that declares no member lends land state under any member, as
    // the hybrids always have; a member lending to ANOTHER member is
    // refused unless it is the declared companion, whose receipt names
    // both (`mapped_composition._compose_bound_fields`).
    let other: Vec<&String> = primary_members
        .iter()
        .flatten()
        .filter(|member| Some(*member) != donor_member.as_ref())
        .collect();
    if binding.companion.is_none() && donor_member.is_some() && !other.is_empty() {
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} decoded member {} under \
             primary member {}; borrowing another member's fields would mix \
             two trajectories in one frame, which only a declared companion \
             may do",
            crate::refusal::python_repr(donor_member.as_deref().unwrap_or_default()),
            crate::refusal::python_repr(other[0])
        )));
    }
    let donor_inventory: BTreeSet<&str> = donor
        .direct
        .keys()
        .map(|(_time, _member, name)| name.as_str())
        .collect();
    let bound: BTreeSet<&str> = binding
        .fields
        .iter()
        .chain(extended.iter())
        .map(String::as_str)
        .collect();
    if donor_inventory != bound {
        let decoded: Vec<&str> = donor_inventory.into_iter().collect();
        let wanted: Vec<&str> = bound.into_iter().collect();
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} decoded fields {} instead of \
             the bound {}",
            python_list_repr(&decoded),
            python_list_repr(&wanted)
        )));
    }
    let provided_twice: BTreeSet<&str> = primary
        .direct_names
        .iter()
        .map(String::as_str)
        .filter(|name| binding.fields.iter().any(|field| field == name)
            && !binding.vertical_merge.as_ref().is_some_and(|merge| merge.fields.contains_key(*name)))
        .collect();
    if let Some(name) = provided_twice.into_iter().next() {
        return Err(frame_invalid(format!(
            "field {} has two providers: the primary decode and contributing \
             source binding {quoted}",
            crate::refusal::python_repr(name)
        )));
    }
    if donor
        .direct
        .values()
        .filter(|value| !extended.iter().any(|name| *name == value.name))
        .any(|value| value.axes.iter().any(|axis| axis == "vertical"))
        && donor.vertical_values != primary.vertical_values
        && binding.vertical_merge.is_none()
    {
        // The primary's declared ladder, read from the header.
        return Err(frame_invalid(format!(
            "contributing source binding {quoted} borrows a vertical-bearing \
             field on a different vertical ladder; cross-ladder borrowing \
             requires a vertical interpolation capability this composition \
             does not declare"
        )));
    }
    let (latitude_indices, longitude_indices) = binding_subset_indices(
        primary, donor, binding_name, &binding.grid_alignment)?;
    let keys = primary.keys();
    let primary_cycles: Vec<NaiveDateTime> = primary
        .source_cycles
        .values()
        .copied()
        .collect::<BTreeSet<NaiveDateTime>>()
        .into_iter()
        .collect();
    let mut by_field: BTreeMap<&str, BTreeMap<NaiveDateTime, (&DirectValue, &crate::assemble::DirectKey)>> =
        BTreeMap::new();
    for (entry, value) in &donor.direct {
        by_field
            .entry(entry.2.as_str())
            .or_default()
            .insert(entry.0, (value, entry));
    }

    let mut chosen: BTreeMap<(String, NaiveDateTime), crate::assemble::DirectKey> = BTreeMap::new();
    let mut matched_times: BTreeSet<NaiveDateTime> = BTreeSet::new();
    let mut broadcast_times: BTreeSet<NaiveDateTime> = BTreeSet::new();
    let mut subset_hashes: BTreeMap<&str, String> = BTreeMap::new();
    // The first valid time's subset of each field, digested once the plan
    // is resolved: the subset and its digest are whole-array work per
    // field, so they run on every core.  Every refusal they could make
    // (`check_subset_axes`) is still made in place, in the plan's order.
    // Named breakage: the borrowed soil, terrain and vegetation fields
    // were cut and hashed one after another on one core, about 1 s of a
    // 3 km CONUS compose's start with the pool idle.
    let mut deferred_hashes: Vec<(&str, &DirectValue)> = Vec::new();
    for name in &binding.fields {
        let supplied = &by_field[name.as_str()];
        let carrier: Option<(&DirectValue, &crate::assemble::DirectKey)> = match alignment {
            "valid_time_exact" => {
                let missing: Vec<NaiveDateTime> = keys
                    .iter()
                    .map(|key| key.0)
                    .filter(|time| !supplied.contains_key(time))
                    .collect();
                if !missing.is_empty() {
                    let rendered: Vec<String> = missing.iter().copied().map(naive_repr).collect();
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} lacks {} at \
                         primary valid time(s) [{}]",
                        crate::refusal::python_repr(name),
                        rendered.join(", ")
                    )));
                }
                None
            }
            "source_cycle_analysis_select_broadcast" => {
                if primary_cycles.len() != 1 { return Err(mixed_cycles("source-cycle analysis selection", &primary_cycles)); }
                let entry = supplied.get(&primary_cycles[0]).ok_or_else(|| frame_invalid(
                    format!("binding {quoted} lacks the primary cycle's exact analysis")))?;
                if entry.0.source_cycle != primary_cycles[0] {
                    return Err(frame_invalid("selected surface donor is a forecast rather than a cycle analysis"));
                }
                Some(*entry)
            }
            // The donor's ONE record from the primary's own cycle, valid at
            // the run's start, carried to every later time. The run's start
            // is no later than the first primary time of this call (a
            // remaining-forcing pass starts one lead in) and no earlier than
            // the primary cycle. For a forecast that starts at its cycle this
            // is the record the analysis broadcast takes. Breakage it
            // prevents: a time-lagged member (HRRR 07Z started at f03) took
            // its soil from the 07Z analysis, three to seventeen hours stale,
            // so the top RUC nodes started at another hour's temperature.
            "source_cycle_window_start_broadcast" => {
                if primary_cycles.len() != 1 {
                    return Err(mixed_cycles(
                        "source-cycle window-start broadcast",
                        &primary_cycles,
                    ));
                }
                let first = keys.iter().map(|key| key.0).min().ok_or_else(|| {
                    frame_invalid("a window-start broadcast needs a primary valid time")
                })?;
                if supplied.len() != 1 {
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} supplies {} at {} valid \
                         times; the window-start broadcast requires exactly one \
                         record, valid at the run's start",
                        crate::refusal::python_repr(name),
                        supplied.len()
                    )));
                }
                let (valid_time, entry) = supplied.iter().next().expect("one record");
                if *valid_time < primary_cycles[0]
                    || *valid_time > first
                    || entry.0.source_cycle != primary_cycles[0]
                {
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} supplies {} valid at {} \
                         from cycle {}; the window-start broadcast needs the primary \
                         cycle {}'s own record valid between that cycle and the \
                         first primary time {}",
                        crate::refusal::python_repr(name),
                        naive_isoformat(*valid_time),
                        naive_isoformat(entry.0.source_cycle),
                        naive_isoformat(primary_cycles[0]),
                        naive_isoformat(first)
                    )));
                }
                Some(*entry)
            }
            "source_cycle_analysis_broadcast" => {
                if primary_cycles.len() != 1 {
                    return Err(mixed_cycles(
                        "source-cycle analysis broadcast",
                        &primary_cycles,
                    ));
                }
                if supplied.len() != 1 {
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} supplies {} at {} \
                         valid times; the analysis broadcast requires exactly \
                         one analysis record",
                        crate::refusal::python_repr(name),
                        supplied.len()
                    )));
                }
                let (analysis_time, entry) = supplied.iter().next().expect("one record");
                if *analysis_time != primary_cycles[0] {
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} supplies {} at {}, \
                         which is not the primary source cycle {}; a hybrid \
                         borrows its initialization state from the SAME cycle's \
                         analysis",
                        crate::refusal::python_repr(name),
                        naive_isoformat(*analysis_time),
                        naive_isoformat(primary_cycles[0])
                    )));
                }
                Some(*entry)
            }
            _ => {
                if primary_cycles.len() != 1 {
                    return Err(mixed_cycles("cycle-invariant broadcast", &primary_cycles));
                }
                let ordered: Vec<(&DirectValue, &crate::assemble::DirectKey)> =
                    supplied.values().copied().collect();
                let reference = ordered[0];
                if ordered[1..]
                    .iter()
                    .any(|(value, _entry)| {
                        !invariant_values_equal(&value.values, &reference.0.values)
                    })
                {
                    return Err(frame_invalid(format!(
                        "contributing source binding {quoted} field {} changes \
                         across supplied valid times; the cycle-invariant \
                         broadcast is for proven statics only",
                        crate::refusal::python_repr(name)
                    )));
                }
                Some(reference)
            }
        };
        let mut hashed = false;
        for key in &keys {
            let exact = if alignment == "source_cycle_analysis_select_broadcast" {
                // Never consume a future analysis from a multi-time object.
                None
            } else { supplied.get(&key.0) };
            let (value, entry) = match exact {
                Some(entry) => {
                    matched_times.insert(key.0);
                    *entry
                }
                None => {
                    broadcast_times.insert(key.0);
                    // Unreachable through the clock rules above -- the
                    // exact alignment refuses every missing time before
                    // this loop, and both broadcasts set a carrier -- and
                    // written as a refusal rather than a panic anyway: an
                    // engine that aborts leaves no refusal object on
                    // stderr, so the caller gets "exited without a class"
                    // instead of a sentence naming what was missing.
                    let Some(carrier) = carrier else {
                        return Err(frame_invalid(format!(
                            "contributing source binding {quoted} has no \
                             record for {} at {} and no carrier to broadcast; \
                             the {alignment} clock rule left the borrow \
                             unresolved",
                            crate::refusal::python_repr(name),
                            naive_isoformat(key.0)
                        )));
                    };
                    carrier
                }
            };
            // The receipt hashes the FIRST valid time's subset, exactly
            // as it always did; the rest are taken when their valid time
            // is composed, so the plan holds keys rather than arrays.
            if !hashed {
                check_subset_axes(&value.axes)?;
                deferred_hashes.push((name.as_str(), value));
                hashed = true;
            }
            chosen.insert((name.clone(), key.0), entry.clone());
        }
    }
    let digests: Vec<Result<String>> = crate::threads::install(|| {
        use rayon::prelude::*;
        deferred_hashes.par_iter().map(|(_name, value)| {
            take_subset(&value.values, &value.axes, &latitude_indices, &longitude_indices)
                .map(|values| array_digest(&values))
        }).collect()
    });
    for ((name, _value), digest) in deferred_hashes.iter().zip(crate::threads::in_order(digests)?) {
        subset_hashes.insert(name, digest);
    }
    let donor_cycles: BTreeSet<NaiveDateTime> = donor
        .direct
        .values()
        .map(|value| value.source_cycle)
        .collect();
    let mut fields = binding.fields.clone();
    fields.sort();
    let companion = match &binding.companion {
        Some(declared) => Some(plan_companion(
            primary,
            donor,
            binding_name,
            declared,
            &primary_members,
            donor_member.as_deref(),
            &keys,
            &latitude_indices,
            &longitude_indices,
        )?),
        None => None,
    };
    let mut receipt = json!({
        "schema": BINDING_RECEIPT_SCHEMA,
        "status": "PASS",
        "binding": binding_name,
        "source_id": binding.source_id,
        "fields": fields,
        "grid_alignment": binding.grid_alignment,
        "coordinate_match": if binding.grid_alignment == "nearest_coordinate" {
            "nearest_published_cell"
        } else if binding.grid_alignment == "exact_coordinate_strided_subset" {
            "exact_equivalent_regular_strided_subset"
        } else { "exact_equivalent_contiguous_subset" },
        "longitude_equivalence": "modulo_360_exact",
        "primary_shape": [primary.latitude.len(), primary.longitude.len()],
        "donor_shape": [donor.latitude.len(), donor.longitude.len()],
        "latitude_index_range": [
            latitude_indices[0],
            latitude_indices[latitude_indices.len() - 1],
        ],
        "longitude_index_range": [
            longitude_indices[0],
            longitude_indices[longitude_indices.len() - 1],
        ],
        "time_alignment": alignment,
        "donor_member": donor_member,
        "donor_source_cycles": donor_cycles
            .iter()
            .copied()
            .map(naive_isoformat)
            .collect::<Vec<String>>(),
        "donor_valid_times": donor
            .direct
            .keys()
            .map(|(time, _member, _name)| naive_isoformat(*time))
            .collect::<BTreeSet<String>>(),
        "matched_primary_valid_times": matched_times
            .iter()
            .copied()
            .map(naive_isoformat)
            .collect::<Vec<String>>(),
        "broadcast_primary_valid_times": broadcast_times
            .iter()
            .copied()
            .map(naive_isoformat)
            .collect::<Vec<String>>(),
        "field_subset_sha256": subset_hashes,
    });
    if binding.grid_alignment == "exact_coordinate_strided_subset" {
        receipt["latitude_indices"] = json!(latitude_indices);
        receipt["longitude_indices"] = json!(longitude_indices);
        receipt["latitude_index_stride"] = json!(regular_index_stride(
            &latitude_indices, donor.latitude.len(), false));
        receipt["longitude_index_stride"] = json!(regular_index_stride(
            &longitude_indices, donor.longitude.len(), closed_longitude_circle(&donor.longitude)));
    }
    if let (Some((_, companion_receipt)), Some(object)) = (&companion, receipt.as_object_mut()) {
        // Present only when a companion is declared, so every binding
        // receipt written before companions existed is unchanged.
        object.insert("companion".to_owned(), companion_receipt.clone());
    }
    Ok((
        // The composed frame keeps the PRIMARY's vertical identity, its
        // hybrid coefficient ladder included: the slice
        // `apply_bound_fields` inserts into IS the primary's own decode
        // of that valid time, so the ladder rides the slice and the plan
        // holds none of it.
        BindingPlan {
            latitude_indices,
            longitude_indices,
            chosen,
            vertical_merge: binding.vertical_merge.clone(),
            binding_name: binding.name.clone(),
            companion: companion.map(|(plan, _receipt)| plan),
        },
        receipt,
    ))
}

/// The companion half of `mapped_composition._extend_above_top`, resolved
/// against the header: which donor record each primary valid time stacks,
/// the composed ladder, and the receipt naming exactly which levels of
/// which field came from which source.
#[allow(clippy::too_many_arguments)]
fn plan_companion(
    primary: &PrimaryHeader,
    donor: &DecodedCollection,
    binding_name: &str,
    declared: &crate::compose::Companion,
    primary_members: &[Option<String>],
    donor_member: Option<&str>,
    keys: &[TimeKey],
    latitude_indices: &[usize],
    longitude_indices: &[usize],
) -> Result<(CompanionPlan, Value)> {
    let quoted = crate::refusal::python_repr(binding_name);
    if !crate::compose::COMPANION_VERTICAL_EXTENSIONS.contains(&declared.vertical_extension.as_str()) {
        return Err(frame_invalid(format!(
            "companion vertical_extension must be one of {}, got {}",
            python_list_repr(&crate::compose::COMPANION_VERTICAL_EXTENSIONS),
            crate::refusal::python_repr(&declared.vertical_extension)
        )));
    }
    let (composed, above, ascending) =
        companion_ladder(primary.vertical_values, &donor.vertical_values, binding_name)?;
    let mut chosen: BTreeMap<(String, NaiveDateTime), crate::assemble::DirectKey> = BTreeMap::new();
    let mut slab_hashes: BTreeMap<String, String> = BTreeMap::new();
    let mut extension_bytes: u64 = 0;
    let cells = (primary.latitude.len() as u64) * (primary.longitude.len() as u64);
    for name in &declared.fields {
        let supplied: BTreeMap<NaiveDateTime, (&crate::assemble::DirectKey, &DirectValue)> = donor
            .direct
            .iter()
            .filter(|(entry, _)| entry.2 == *name)
            .map(|(entry, value)| (entry.0, (entry, value)))
            .collect();
        for key in keys {
            let Some((entry, value)) = supplied.get(&key.0) else {
                return Err(frame_invalid(format!(
                    "contributing source binding {quoted} lacks {} at primary \
                     valid time(s) [{}]",
                    crate::refusal::python_repr(name),
                    naive_repr(key.0)
                )));
            };
            let cycle = primary.source_cycles[key];
            if value.source_cycle != cycle {
                return Err(frame_invalid(format!(
                    "contributing source binding {quoted} is a companion from \
                     cycle {} under primary cycle {}; a companion from another \
                     cycle would put a different forecast on top of this one's \
                     column",
                    naive_isoformat(value.source_cycle),
                    naive_isoformat(cycle)
                )));
            }
            if !slab_hashes.contains_key(name) {
                let axis = value
                    .axes
                    .iter()
                    .position(|axis| axis == "vertical")
                    .ok_or_else(|| {
                        frame_invalid(format!(
                            "contributing source binding {quoted} field {} has \
                             no vertical axis to extend",
                            crate::refusal::python_repr(name)
                        ))
                    })?;
                let slab = take_subset(&value.values, &value.axes, latitude_indices, longitude_indices)?
                    .select(Axis(axis), &above);
                slab_hashes.insert(name.clone(), array_digest(&slab));
            }
            chosen.insert((name.clone(), key.0), (*entry).clone());
        }
        extension_bytes = extension_bytes
            .saturating_add(cells.saturating_mul(above.len() as u64).saturating_mul(16));
    }
    let real: Vec<&String> = primary_members.iter().flatten().collect();
    let mut fields = declared.fields.clone();
    fields.sort();
    let receipt = json!({
        "vertical_extension": declared.vertical_extension,
        "fields": fields,
        "primary_member": if real.len() == 1 { Some(real[0].clone()) } else { None },
        "companion_member": donor_member,
        "primary_levels": primary.vertical_values,
        "companion_levels": above.iter().map(|index| donor.vertical_values[*index]).collect::<Vec<f64>>(),
        "composed_levels": composed,
        "companion_slab_sha256": slab_hashes,
    });
    Ok((
        CompanionPlan {
            binding: binding_name.to_owned(),
            fields: declared.fields.clone(),
            primary: primary.vertical_values.to_vec(),
            composed,
            above,
            ascending,
            chosen,
            extension_bytes,
        },
        receipt,
    ))
}

fn mixed_cycles(label: &str, cycles: &[NaiveDateTime]) -> crate::refusal::Refusal {
    let rendered: Vec<String> = cycles.iter().copied().map(naive_repr).collect();
    frame_invalid(format!(
        "{label} cannot span mixed primary source cycles [{}]",
        rendered.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declared_stride_matches_the_whole_prep_coordinate_regression() {
        let latitude: Vec<f64> = (0..47).map(|i| 32.75 - i as f64 * 0.25).collect();
        let longitude: Vec<f64> = (0..49).map(|i| -93.0 + i as f64 * 0.25).collect();
        let primary_lat: Vec<f64> = (0..7).map(|i| 31.5 - i as f64 * 1.5).collect();
        let primary_lon: Vec<f64> = (0..7).map(|i| 268.5 + i as f64 * 1.5).collect();
        assert!(exact_subset_indices(&latitude, &primary_lat, "latitude", false).is_err());
        assert_eq!(exact_subset_indices_with_stride(
            &latitude, &primary_lat, "latitude", false, true).unwrap(),
            vec![5, 11, 17, 23, 29, 35, 41]);
        assert_eq!(exact_subset_indices_with_stride(
            &longitude, &primary_lon, "longitude", true, true).unwrap(),
            vec![6, 12, 18, 24, 30, 36, 42]);
    }

    #[test]
    fn declared_stride_wraps_once_and_never_interpolates() {
        let donor: Vec<f64> = (0..12).map(|i| i as f64 * 30.0).collect();
        for (primary, expected, stride) in [
            (vec![300.0, 0.0, 60.0], vec![10, 0, 2], 2),
            (vec![60.0, 0.0, 300.0], vec![2, 0, 10], -2),
            (vec![240.0, 180.0, 120.0], vec![8, 6, 4], -2),
        ] {
            let indices = exact_subset_indices_with_stride(
                &donor, &primary, "longitude", true, true).unwrap();
            assert_eq!(indices, expected);
            assert_eq!(regular_index_stride(&indices, donor.len(), true), Some(stride));
        }
        for primary in [vec![0.0, 60.0, 150.0], vec![0.0, 120.0, 240.0, 0.0],
            vec![0.0, 150.0, 300.0, 90.0], vec![0.0, 60.00000000001]] {
            assert!(exact_subset_indices_with_stride(
                &donor, &primary, "longitude", true, true).is_err());
        }
        assert!(exact_subset_indices_with_stride(
            &[0.0, 360.0], &[0.0], "longitude", true, true).is_err());
    }

    #[test]
    fn an_exact_subset_solves_to_its_index_window() {
        let larger: Vec<f64> = (0..10).map(|index| index as f64 * 0.25).collect();
        let smaller = vec![0.5, 0.75, 1.0];
        assert_eq!(
            exact_subset_indices(&larger, &smaller, "latitude", false).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn a_near_miss_is_a_refusal_rather_than_a_tolerance() {
        // The named breakage: 0.7000000000000001 is the f64 next to 0.7,
        // and a tolerance would borrow the 0.7 cell for it and record a
        // PASS receipt saying the grids matched exactly.
        let larger = vec![0.5, 0.7, 0.9];
        let refusal =
            exact_subset_indices(&larger, &[0.700_000_000_000_000_1], "latitude", false).unwrap_err();
        assert!(refusal.message.contains("0 exact matches"), "{refusal}");
    }

    #[test]
    fn the_refusal_spells_the_coordinate_the_way_python_repr_does() {
        let refusal = exact_subset_indices(&[1.0], &[45.0], "latitude", false).unwrap_err();
        assert_eq!(
            refusal.message,
            "primary latitude coordinate np.float64(45.0) has 0 exact matches \
             in the terrain grid"
        );
    }

    #[test]
    fn longitudes_are_compared_modulo_360_by_floor_not_truncation() {
        // A donor spelling longitudes in [-180, 180) and a primary
        // spelling them in [0, 360) name the same meridian.  Rust's `%`
        // would leave -0.25 where numpy's floor-modulo leaves 359.75, so
        // this row would find no match at all.
        let donor = vec![-0.5, -0.25, 0.0];
        let primary = vec![359.75];
        assert_eq!(
            exact_subset_indices(&donor, &primary, "longitude", true).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn a_scattered_subset_refuses_as_non_contiguous() {
        let larger = vec![0.0, 1.0, 2.0, 3.0];
        let refusal = exact_subset_indices(&larger, &[0.0, 2.0], "latitude", false).unwrap_err();
        assert!(refusal.message.contains("not a contiguous"), "{refusal}");
    }

    #[test]
    fn a_reversed_subset_is_contiguous_and_records_its_direction() {
        let larger = vec![0.0, 1.0, 2.0, 3.0];
        let indices = exact_subset_indices(&larger, &[2.0, 1.0], "latitude", false).unwrap();
        assert_eq!(indices, vec![2, 1]);
        assert_eq!(index_direction(&indices), -1);
    }

    fn at(text: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
    }

    fn field(time: NaiveDateTime, name: &str, values: ArrayD<f64>) -> DirectValue {
        DirectValue {
            name: name.to_owned(),
            valid_time: time,
            member: None,
            source_cycle: time,
            axes: vec!["y".into(), "x".into()],
            missing_count: crate::array::count_nan(&values),
            values,
            references: vec!["fixture".into()],
        }
    }

    fn collection(
        latitude: &[f64],
        longitude: &[f64],
        records: Vec<DirectValue>,
    ) -> DecodedCollection {
        let mut direct = BTreeMap::new();
        let mut source_cycles = BTreeMap::new();
        for record in records {
            source_cycles.insert((record.valid_time, None), record.source_cycle);
            direct.insert((record.valid_time, None, record.name.clone()), record);
        }
        DecodedCollection {
            latitude: latitude.to_vec(),
            longitude: longitude.to_vec(),
            vertical_values: vec![],
            direct,
            source_cycles,
            grid_fingerprint: "fixture".into(),
            hybrid_a: vec![],
            hybrid_b: vec![],
            original_pressure_hpa: None,
        }
    }

    fn grid(rows: usize, columns: usize) -> ArrayD<f64> {
        ndarray::Array::from_shape_vec(
            (rows, columns),
            (0..rows * columns).map(|value| value as f64).collect(),
        )
        .unwrap()
        .into_dyn()
    }

    #[test]
    fn a_primary_crossing_a_global_seam_takes_its_exact_cells() {
        // The donor's last and first longitudes are neighbours on a
        // whole-globe axis, so these runs are contiguous although their
        // indices wrap; each was refused as "not a contiguous
        // terrain-grid subset".
        let east: Vec<f64> = (0..1440).map(|index| index as f64 * 0.25).collect();
        let pacific: Vec<f64> = (0..1440).map(|index| -180.0 + index as f64 * 0.25).collect();
        let west: Vec<f64> = (0..1440).map(|index| 359.75 - index as f64 * 0.25).collect();
        for (donor, primary, expected) in [
            (&east, vec![-0.5, -0.25, 0.0, 0.25], vec![1438, 1439, 0, 1]),
            (&east, vec![0.25, 0.0, -0.25, -0.5], vec![1, 0, 1439, 1438]),
            (&pacific, vec![179.5, 179.75, 180.0, 180.25], vec![1438, 1439, 0, 1]),
            (&west, vec![0.25, 0.0, -0.25, -0.5], vec![1438, 1439, 0, 1]),
        ] {
            assert_eq!(
                exact_subset_indices(donor, &primary, "longitude", true).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn a_rounded_decoded_global_axis_still_closes_its_seam() {
        // 0.15 deg decoded as micro-degrees over 1e6: not exactly even.
        let donor: Vec<f64> = (0..2400).map(|index| (index * 150_000) as f64 / 1.0e6).collect();
        let primary = vec![donor[2398], donor[2399], donor[0], donor[1]];
        assert_eq!(
            exact_subset_indices(&donor, &primary, "longitude", true).unwrap(),
            vec![2398, 2399, 0, 1]
        );
    }

    #[test]
    fn the_seam_wraps_only_a_closed_axis_and_one_turn() {
        let quarter = [0.0, 90.0, 180.0, 270.0];
        for (donor, primary, cyclic) in [
            (&[0.0, 1.0, 2.0][..], &[2.0, 0.0][..], true),
            (&quarter[..], &[270.0, 90.0][..], true),
            (&quarter[..], &[270.0, 0.0, 90.0][..], false),
            (&[0.0, 90.0, 180.0][..], &[180.0, 0.0][..], true),
            (&[0.0, 90.0, 180.0, 270.0, 360.0][..], &[0.0][..], true),
            (&quarter[..], &[270.0, 0.000_001][..], true),
            (&quarter[..], &[270.0, 0.0, 90.0, 180.0, 270.0][..], true),
        ] {
            assert!(
                exact_subset_indices(donor, primary, "longitude", cyclic).is_err(),
                "{donor:?} {primary:?} {cyclic}"
            );
        }
    }

    #[test]
    fn terrain_across_the_seam_gathers_and_records_its_direction() {
        let time = at("2026-09-27 00:00:00");
        let donor_longitude: Vec<f64> = (0..1440).map(|index| index as f64 * 0.25).collect();
        let terrain = collection(
            &[10.0, 11.0],
            &donor_longitude,
            vec![field(time, EXTERNAL_FIELD, grid(2, 1440))],
        );
        for (primary_longitude, indices, direction) in [
            (vec![-0.5, -0.25, 0.0, 0.25], [1438_usize, 1439, 0, 1], 1),
            (vec![0.25, 0.0, -0.25, -0.5], [1, 0, 1439, 1438], -1),
        ] {
            let cycles = BTreeMap::from([((time, None), time)]);
            let names = BTreeSet::from(["surface_pressure".to_owned()]);
            let header = PrimaryHeader {
                latitude: &[10.0, 11.0],
                longitude: &primary_longitude,
                vertical_values: &[],
                source_cycles: &cycles,
                direct_names: &names,
            };
            let (plan, receipt) = plan_terrain(&header, &terrain, "valid_time_exact", "exact_coordinate_subset").unwrap();
            for row in 0..2 {
                for (column, index) in indices.iter().enumerate() {
                    assert_eq!(plan.values[[row, column]], (row * 1440 + index) as f64);
                }
            }
            assert_eq!(receipt["longitude_index_direction"], json!(direction));
            assert_eq!(receipt["longitude_index_range"], json!([indices[0], indices[3]]));
        }
    }

    #[test]
    fn an_unwrapped_run_keeps_its_receipt_direction() {
        assert_eq!(wrapped_index_direction(&[5, 4, 3], 7), -1);
        assert_eq!(wrapped_index_direction(&[1, 0], 2), -1);
        assert_eq!(wrapped_index_direction(&[0, 1], 2), 1);
        assert_eq!(wrapped_index_direction(&[3], 7), 0);
    }

    fn with_missing(values: &ArrayD<f64>, cell: [usize; 2]) -> ArrayD<f64> {
        let mut copy = values.clone();
        copy[[cell[0], cell[1]]] = f64::NAN;
        copy
    }

    #[test]
    fn missing_donor_cells_outside_the_subset_are_not_a_change() {
        // An unchanged static with a missing cell outside the primary
        // window: several identical records were refused (NaN != NaN) and
        // one broadcast record kept the whole donor's count, which the
        // canonical validator refused against the cropped array.
        let time = at("2026-09-27 00:00:00");
        let later = at("2026-09-27 06:00:00");
        let cycles = BTreeMap::from([((time, None), time), ((later, None), time)]);
        let names = BTreeSet::from(["surface_pressure".to_owned()]);
        let header = PrimaryHeader {
            latitude: &[10.0, 11.0],
            longitude: &[1.0, 2.0],
            vertical_values: &[],
            source_cycles: &cycles,
            direct_names: &names,
        };
        let values = with_missing(&grid(2, 3), [0, 0]);
        for one_record in [true, false] {
            let times = if one_record { vec![time] } else { vec![time, later] };
            let records = times
                .into_iter()
                .map(|valid| field(valid, EXTERNAL_FIELD, values.clone()))
                .collect();
            let terrain = collection(&[10.0, 11.0], &[0.0, 1.0, 2.0], records);
            let alignment = if one_record { "cycle_invariant_broadcast" } else { "valid_time_exact" };
            let (plan, _receipt) = plan_terrain(&header, &terrain, alignment, "exact_coordinate_subset").unwrap();
            assert_eq!(plan.values.iter().copied().collect::<Vec<f64>>(), vec![1.0, 2.0, 4.0, 5.0]);
            for entry in plan.entries.values() {
                assert_eq!(entry.missing_count, 0);
                crate::derive::CanonicalField::validate_values(
                    EXTERNAL_FIELD, &entry.axes, &plan.values, entry.missing_count,
                )
                .unwrap();
            }
        }
        // A missing cell INSIDE the window is counted there.
        let inside = with_missing(&grid(2, 3), [1, 2]);
        let terrain = collection(
            &[10.0, 11.0],
            &[0.0, 1.0, 2.0],
            vec![field(time, EXTERNAL_FIELD, inside.clone()), field(later, EXTERNAL_FIELD, inside)],
        );
        let (plan, _receipt) = plan_terrain(&header, &terrain, "valid_time_exact", "exact_coordinate_subset").unwrap();
        assert!(plan.entries.values().all(|entry| entry.missing_count == 1));
        // A static that changes beside a stable missing cell still refuses,
        // and so does a cell that turns missing.
        for changed in [grid(2, 3), with_missing(&with_missing(&grid(2, 3), [0, 0]), [1, 1])] {
            let mut moved = changed;
            if !moved[[1, 1]].is_nan() {
                moved[[0, 0]] = f64::NAN;
                moved[[1, 1]] += 1.0;
            }
            let terrain = collection(
                &[10.0, 11.0],
                &[0.0, 1.0, 2.0],
                vec![field(time, EXTERNAL_FIELD, values.clone()), field(later, EXTERNAL_FIELD, moved)],
            );
            let refusal = plan_terrain(&header, &terrain, "valid_time_exact", "exact_coordinate_subset").err().unwrap();
            assert!(refusal.message.contains("changes across supplied valid times"), "{refusal}");
        }
    }

    fn binding(alignment: &str) -> Binding {
        Binding {
            name: "donor".into(),
            source_id: "donor-fixture".into(),
            mapping_role: "donor_mapping".into(),
            mapping_sha256: "0".repeat(64),
            data_role: "donor_data".into(),
            provenance_role: "donor_provenance".into(),
            fields: vec!["land_fraction".into()],
            grid_alignment: "exact_coordinate_subset".into(),
            time_alignment: alignment.into(),
            vertical_merge: None,
            companion: None,
        }
    }

    #[test]
    fn cycle_analysis_selection_never_borrows_a_future_surface_analysis() {
        let cycle = at("2019-02-01 00:00:00");
        let first = at("2019-02-01 03:00:00");
        let future = at("2019-02-01 06:00:00");
        let cycles = BTreeMap::from([((first, None), cycle), ((future, None), cycle)]);
        let names = BTreeSet::from(["surface_pressure".to_owned()]);
        let header = PrimaryHeader {
            latitude: &[10.0, 11.0], longitude: &[0.0, 1.0],
            vertical_values: &[], source_cycles: &cycles, direct_names: &names,
        };
        let current = grid(2, 2);
        let later = current.mapv(|v| v + 100.0);
        let donor = collection(&[10.0, 11.0], &[0.0, 1.0], vec![
            field(cycle, "land_fraction", current.clone()),
            field(future, "land_fraction", later.clone()),
        ]);
        let rule = binding("source_cycle_analysis_select_broadcast");
        let (plan, _) = plan_bound_fields(&header, &donor, &rule).unwrap();
        for valid in [first, future] {
            let mut slice = collection(&[10.0, 11.0], &[0.0, 1.0], vec![]);
            apply_bound_fields(&plan, &donor, &mut slice, &(valid, None), None).unwrap();
            assert_eq!(slice.direct[&(valid, None, "land_fraction".to_owned())].values, current);
        }
        let missing = collection(&[10.0, 11.0], &[0.0, 1.0], vec![field(future, "land_fraction", later)]);
        assert!(plan_bound_fields(&header, &missing, &rule).is_err());
        let mut forecast = field(cycle, "land_fraction", current);
        forecast.source_cycle = cycle - chrono::Duration::hours(6);
        let wrong_cycle = collection(&[10.0, 11.0], &[0.0, 1.0], vec![forecast]);
        assert!(plan_bound_fields(&header, &wrong_cycle, &rule).is_err());
    }

    #[test]
    fn a_bound_field_crosses_the_seam_and_keeps_a_stable_missing_cell() {
        let time = at("2026-08-17 00:00:00");
        let later = at("2026-08-17 03:00:00");
        let cycles = BTreeMap::from([((time, None), time)]);
        let names = BTreeSet::from(["surface_pressure".to_owned()]);
        let header = PrimaryHeader {
            latitude: &[10.0, 11.0],
            longitude: &[-90.0, 0.0, 90.0],
            vertical_values: &[],
            source_cycles: &cycles,
            direct_names: &names,
        };
        let values = grid(2, 4);
        let donor = collection(
            &[10.0, 11.0],
            &[0.0, 90.0, 180.0, 270.0],
            vec![field(time, "land_fraction", values.clone())],
        );
        let (plan, receipt) = plan_bound_fields(&header, &donor, &binding("valid_time_exact")).unwrap();
        assert_eq!(receipt["longitude_index_range"], json!([3, 1]));
        let mut slice = collection(&[10.0, 11.0], &[-90.0, 0.0, 90.0], vec![]);
        apply_bound_fields(&plan, &donor, &mut slice, &(time, None), None).unwrap();
        let taken = &slice.direct[&(time, None, "land_fraction".to_owned())].values;
        assert_eq!(taken.iter().copied().collect::<Vec<f64>>(), vec![3.0, 0.0, 1.0, 7.0, 4.0, 5.0]);

        // Two identical statics with a missing cell are invariant.
        let marked = with_missing(&values, [0, 2]);
        let donor = collection(
            &[10.0, 11.0],
            &[0.0, 90.0, 180.0, 270.0],
            vec![field(time, "land_fraction", marked.clone()), field(later, "land_fraction", marked)],
        );
        let (plan, _receipt) =
            plan_bound_fields(&header, &donor, &binding("cycle_invariant_broadcast")).unwrap();
        let mut slice = collection(&[10.0, 11.0], &[-90.0, 0.0, 90.0], vec![]);
        apply_bound_fields(&plan, &donor, &mut slice, &(time, None), None).unwrap();
        assert_eq!(slice.direct[&(time, None, "land_fraction".to_owned())].missing_count, 0);
    }

    /// A three-level column on a 2x3 grid whose level `k` holds `base + k`.
    fn column(time: NaiveDateTime, name: &str, member: Option<&str>, levels: usize, base: f64) -> DirectValue {
        let values = ndarray::Array::from_shape_fn((levels, 2, 3), |(level, _, _)| base + level as f64)
            .into_dyn();
        DirectValue {
            name: name.to_owned(),
            valid_time: time,
            member: member.map(str::to_owned),
            source_cycle: time,
            axes: vec!["vertical".into(), "y".into(), "x".into()],
            missing_count: 0,
            values,
            references: vec![format!("{name}-{base}")],
        }
    }

    fn companion_binding() -> Binding {
        Binding {
            companion: Some(crate::compose::Companion {
                vertical_extension: "above_primary_top".into(),
                fields: vec!["air_temperature".into()],
            }),
            fields: vec![],
            ..binding("valid_time_exact")
        }
    }

    #[test]
    fn the_companion_ladder_stacks_only_levels_above_the_top_in_the_primary_order() {
        let donor = [200.0, 5000.0, 20000.0, 25000.0, 50000.0, 100000.0];
        let (composed, above, ascending) =
            companion_ladder(&[25000.0, 50000.0, 100000.0], &donor, "c").unwrap();
        assert_eq!(composed, vec![200.0, 5000.0, 20000.0, 25000.0, 50000.0, 100000.0]);
        assert_eq!(above, vec![0, 1, 2]);
        assert!(ascending);
        let (composed, above, ascending) =
            companion_ladder(&[100000.0, 50000.0, 25000.0], &donor, "c").unwrap();
        assert_eq!(composed, vec![100000.0, 50000.0, 25000.0, 20000.0, 5000.0, 200.0]);
        assert_eq!(above, vec![2, 1, 0]);
        assert!(!ascending);
        // The Python engine's sentence, so both refuse the same bytes alike.
        let refusal = companion_ladder(&[200.0, 1000.0], &[500.0, 1000.0], "c").unwrap_err();
        assert_eq!(
            refusal.message,
            "contributing source binding 'c' declares a companion that extends \
             the ladder above the primary's top (200.0), but the companion \
             carries no level above it"
        );
    }

    #[test]
    fn a_declared_companion_joins_its_levels_onto_the_primary_column() {
        let time = at("2026-10-05 12:00:00");
        let member = Some("1".to_owned());
        let cycles = BTreeMap::from([((time, member.clone()), time)]);
        let names = BTreeSet::from(["air_temperature".to_owned()]);
        let primary_levels = [25000.0, 50000.0, 100000.0];
        let header = PrimaryHeader {
            latitude: &[10.0, 11.0],
            longitude: &[0.0, 1.0, 2.0],
            vertical_values: &primary_levels,
            source_cycles: &cycles,
            direct_names: &names,
        };
        let mut donor = collection(&[10.0, 11.0], &[0.0, 1.0, 2.0], vec![]);
        donor.vertical_values = vec![200.0, 5000.0, 25000.0, 50000.0, 100000.0];
        donor.direct.insert(
            (time, None, "air_temperature".into()),
            column(time, "air_temperature", None, 5, 500.0),
        );
        donor.source_cycles.insert((time, None), time);
        let (plan, receipt) = plan_bound_fields(&header, &donor, &companion_binding()).unwrap();
        assert_eq!(plan.composed_ladder().unwrap(), &[200.0, 5000.0, 25000.0, 50000.0, 100000.0]);
        let extension = &receipt["companion"];
        assert_eq!(extension["companion_levels"], json!([200.0, 5000.0]));
        assert_eq!(extension["primary_levels"], json!([25000.0, 50000.0, 100000.0]));
        assert_eq!(extension["primary_member"], json!("1"));
        assert_eq!(extension["companion_member"], Value::Null);
        assert_eq!(extension["fields"], json!(["air_temperature"]));

        let mut slice = collection(&[10.0, 11.0], &[0.0, 1.0, 2.0], vec![]);
        slice.vertical_values = primary_levels.to_vec();
        slice.direct.insert(
            (time, member.clone(), "air_temperature".into()),
            column(time, "air_temperature", Some("1"), 3, 100.0),
        );
        apply_bound_fields(&plan, &donor, &mut slice, &(time, member.clone()), None).unwrap();
        assert_eq!(slice.vertical_values, vec![200.0, 5000.0, 25000.0, 50000.0, 100000.0]);
        let joined = &slice.direct[&(time, member.clone(), "air_temperature".to_owned())];
        let profile: Vec<f64> = (0..5).map(|level| joined.values[[level, 0, 0]]).collect();
        assert_eq!(profile, vec![500.0, 501.0, 100.0, 101.0, 102.0]);
        assert_eq!(joined.member.as_deref(), Some("1"));
        assert_eq!(joined.references, vec!["air_temperature-100".to_owned(), "air_temperature-500".to_owned()]);
    }

    #[test]
    fn companion_join_uses_the_primary_decode_window_and_keeps_whole_fallbacks() {
        let time = at("2026-10-05 12:00:00");
        let cycles = BTreeMap::from([((time, None), time)]);
        let names = BTreeSet::from(["air_temperature".to_owned()]);
        let header = PrimaryHeader {
            latitude: &[10.0, 11.0], longitude: &[0.0, 1.0, 2.0],
            vertical_values: &[25000.0, 50000.0, 100000.0],
            source_cycles: &cycles, direct_names: &names,
        };
        let mut donor = collection(header.latitude, header.longitude, vec![]);
        donor.vertical_values = vec![200.0, 5000.0, 25000.0, 50000.0, 100000.0];
        let mut upper = column(time, "air_temperature", None, 5, 500.0);
        upper.values = ndarray::Array::from_shape_fn((5, 2, 3), |(z, y, x)|
            500.0 + (100 * z + 10 * y + x) as f64).into_dyn();
        donor.direct.insert((time, None, "air_temperature".into()), upper);
        donor.source_cycles.insert((time, None), time);
        let (plan, _) = plan_bound_fields(&header, &donor, &companion_binding()).unwrap();
        let window = crate::window::Window {
            rows: [1, 2], columns: [1, 3], source_shape: [2, 3], fields: names.clone(),
        };
        for cropped in [true, false] {
            let mut slice = collection(header.latitude, header.longitude, vec![]);
            slice.vertical_values = header.vertical_values.to_vec();
            let mut own = column(time, "air_temperature", None, 3, 100.0);
            if cropped {
                own.values = take_subset(&own.values, &own.axes, &[1], &[1, 2]).unwrap();
            }
            let original = own.values.clone();
            slice.direct.insert((time, None, "air_temperature".into()), own);
            apply_bound_fields(&plan, &donor, &mut slice, &(time, None), Some(&window)).unwrap();
            let joined = &slice.direct[&(time, None, "air_temperature".to_owned())].values;
            let (ny, nx, y0, x0) = if cropped { (1, 2, 1, 1) } else { (2, 3, 0, 0) };
            assert_eq!(joined.shape(), &[5, ny, nx]);
            for y in 0..ny { for x in 0..nx {
                for z in 0..2 {
                    assert_eq!(joined[[z, y, x]], 500.0 + (100 * z + 10 * (y + y0) + x + x0) as f64);
                }
                for z in 0..3 { assert_eq!(joined[[z + 2, y, x]], original[[z, y, x]]); }
            }}
        }
    }

    #[test]
    fn only_a_declared_companion_may_lend_another_members_values() {
        let time = at("2026-10-05 12:00:00");
        let cycles = BTreeMap::from([((time, Some("1".to_owned())), time)]);
        let names = BTreeSet::new();
        let header = PrimaryHeader {
            latitude: &[10.0, 11.0],
            longitude: &[0.0, 1.0, 2.0],
            vertical_values: &[],
            source_cycles: &cycles,
            direct_names: &names,
        };
        let mut record = field(time, "land_fraction", grid(2, 3));
        record.member = Some("2".into());
        let mut donor = collection(&[10.0, 11.0], &[0.0, 1.0, 2.0], vec![]);
        donor.direct.insert((time, Some("2".into()), "land_fraction".into()), record);
        donor.source_cycles.insert((time, Some("2".into())), time);
        let refusal = plan_bound_fields(&header, &donor, &binding("valid_time_exact")).err().unwrap();
        assert_eq!(
            refusal.message,
            "contributing source binding 'donor' decoded member '2' under primary \
             member '1'; borrowing another member's fields would mix two \
             trajectories in one frame, which only a declared companion may do"
        );
    }

    #[test]
    fn a_datetime_list_is_spelled_the_way_python_repr_spells_it() {
        let value =
            NaiveDateTime::parse_from_str("2026-08-17 06:00:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert_eq!(naive_repr(value), "datetime.datetime(2026, 8, 17, 6, 0)");
        assert_eq!(
            time_key_list_repr(&[(value, None)]),
            "[(datetime.datetime(2026, 8, 17, 6, 0), None)]"
        );
    }
}

/// Express a Lambert donor's axes in the primary window's origin before
/// the existing exact subset join. Projected axes are distances, so comparing
/// two local zero origins without this registration would borrow wrong cells.
pub fn register_projected_subset(
    primary: &crate::model::Mapping, donor: &crate::model::Mapping,
    collection: &mut DecodedCollection,
) -> Result<()> {
    let p = primary.grid_declaration()?;
    let d = donor.grid_declaration()?;
    if !p.is_projected() && !d.is_projected() { return Ok(()); }
    if p.is_rotated() || d.is_rotated() {
        // A rotated donor joins only a primary on the identical declared
        // lattice: the axes are then the same distances from the same
        // origin and the exact subset join needs no registration.
        if p.is_rotated() && d.is_rotated() && p.rotated == d.rotated {
            return Ok(());
        }
        return Err(frame_invalid(
            "projected composition needs both sources on one declared rotated lattice"));
    }
    let (Some(p), Some(d)) = (p.parameters, d.parameters) else {
        return Err(frame_invalid("projected composition needs both sources on a declared Lambert grid"));
    };
    if p.latin1 != d.latin1 || p.latin2 != d.latin2
        || crate::lambert::wrap180(p.lov - d.lov) != 0.0
        || p.dx_m != d.dx_m || p.dy_m != d.dy_m || p.earth_radius_m != d.earth_radius_m {
        return Err(frame_invalid("projected composition needs matching Lambert projection and spacing"));
    }
    let grid = crate::lambert::LambertGrid::from_declaration(&d);
    let (x, y) = grid.latlon_to_ij(p.lat1, p.lon1);
    let x = x - 1.0;
    let y = (y - 1.0) * d.dx_m / d.dy_m;
    if (x - x.round()).abs() > 0.1 || (y - y.round()).abs() > 0.1
        || x.round() < 0.0 || y.round() < 0.0
        || x.round() + p.nx as f64 > d.nx as f64
        || y.round() + p.ny as f64 > d.ny as f64 {
        return Err(frame_invalid("primary Lambert window is not a cell-aligned subset of donor grid"));
    }
    collection.latitude = (0..d.ny).map(|index| (index as f64-y.round()) * d.dy_m / crate::model::PROJECTED_AXIS_UNIT_M).collect();
    collection.longitude = (0..d.nx).map(|index| (index as f64-x.round()) * d.dx_m / crate::model::PROJECTED_AXIS_UNIT_M).collect();
    Ok(())
}

#[cfg(test)]
mod projected_registration_tests {
    use super::*;
    use crate::{model::Mapping, node::Node};

    fn mapping() -> Mapping {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/projected.mapping.json");
        Mapping::load(path.to_str().unwrap()).unwrap()
    }

    #[test]
    fn a_projected_donor_joins_the_interior_cells_and_refuses_half_cell_offsets() {
        let donor = mapping();
        let parameters = donor.grid_declaration().unwrap().parameters.unwrap();
        let grid = crate::lambert::LambertGrid::from_declaration(&parameters);
        let mut primary = mapping();
        let mut doc = primary.doc.to_value();
        let (lat, lon) = grid.ij_to_latlon(2.0, 2.0);
        doc["grid"]["parameters"]["lat1"] = json!(lat);
        doc["grid"]["parameters"]["lon1"] = json!(lon);
        doc["grid"]["parameters"]["nx"] = json!(2);
        doc["grid"]["parameters"]["ny"] = json!(2);
        primary.doc = Node::parse(&serde_json::to_vec(&doc).unwrap()).unwrap();
        let mut collection = DecodedCollection {
            latitude: vec![0.0, 0.03, 0.06], longitude: vec![0.0, 0.03, 0.06, 0.09],
            vertical_values: vec![], direct: BTreeMap::new(), source_cycles: BTreeMap::new(),
            grid_fingerprint: "fixture".into(), hybrid_a: vec![], hybrid_b: vec![],
            original_pressure_hpa: None,
        };
        register_projected_subset(&primary, &donor, &mut collection).unwrap();
        assert_eq!(exact_subset_indices(&collection.latitude, &[0.0, 0.03], "y", false).unwrap(), vec![1, 2]);
        assert_eq!(exact_subset_indices(&collection.longitude, &[0.0, 0.03], "x", false).unwrap(), vec![1, 2]);
        let (lat, lon) = grid.ij_to_latlon(2.5, 2.0);
        doc["grid"]["parameters"]["lat1"] = json!(lat);
        doc["grid"]["parameters"]["lon1"] = json!(lon);
        primary.doc = Node::parse(&serde_json::to_vec(&doc).unwrap()).unwrap();
        assert!(register_projected_subset(&primary, &donor, &mut collection).unwrap_err()
            .to_string().contains("cell-aligned subset"));
    }
}
