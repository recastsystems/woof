//! Record assembly: matched GRIB records to a canonical decoded collection.
//!
//! Port of `mapped_source._assemble_grib`, `_broadcast_invariant_fields`,
//! `_apply_landmask_water_missing` and `_rotate_grid_relative_winds`.  The
//! order of the four steps is the Python order and is not incidental:
//! invariants broadcast BEFORE the landmask keys off `land_fraction` (an
//! invariant on most sources), and the wind rotation runs LAST so it sees
//! every component the frame will carry.

use std::collections::{BTreeMap, BTreeSet};

use chrono::NaiveDateTime;
use ndarray::ArrayD;

use crate::array;
use crate::grib::{declared_vertical_admits, selector_matches, GribRecord};
use crate::model::{Mapping, ROTATED_WIND_PAIRS};
use crate::node::Node;
use crate::refusal::{frame_invalid, selector_unmatched, Result};

/// A (valid_time, member, field) address into the decoded collection.
pub type DirectKey = (NaiveDateTime, Option<String>, String);
/// A (valid_time, member) address.
pub type TimeKey = (NaiveDateTime, Option<String>);

/// `mapped_source._DirectValue`.
#[derive(Debug, Clone)]
pub struct DirectValue {
    pub name: String,
    pub valid_time: NaiveDateTime,
    pub member: Option<String>,
    pub source_cycle: NaiveDateTime,
    pub axes: Vec<String>,
    pub values: ArrayD<f64>,
    pub missing_count: usize,
    pub references: Vec<String>,
}

/// `mapped_source._DecodedCollection`.
#[derive(Debug, Clone)]
pub struct DecodedCollection {
    pub latitude: Vec<f64>,
    pub longitude: Vec<f64>,
    pub vertical_values: Vec<f64>,
    pub direct: BTreeMap<DirectKey, DirectValue>,
    pub source_cycles: BTreeMap<TimeKey, NaiveDateTime>,
    pub grid_fingerprint: String,
    /// Resolved hybrid A (Pa) / B coefficients for a
    /// hybrid_sigma_pressure vertical: N+1 half-level interfaces or N
    /// full-level values, top of the atmosphere first.  Empty on every
    /// other vertical kind (`_DecodedCollection.hybrid_a/hybrid_b`).
    pub hybrid_a: Vec<f64>,
    pub hybrid_b: Vec<f64>,
    /// The full-plane pressure ladder (hPa, one value per vertical level,
    /// `window::pressure_levels`) of an `air_pressure` decoded over the
    /// atmospheric window, taken from each whole plane before its crop
    /// (`GribRecord::plane_ladder_hpa`).  `None` whenever `air_pressure`
    /// is held whole, which states its own ladder.  One valid time's slice.
    pub original_pressure_hpa: Option<Vec<f64>>,
}

/// `mapped_source._HYBRID_LITERAL_ABS_TOL` / `_REL_TOL`: pv rides IEEE
/// f32 and literals are authored from the provider's published table of
/// the same numbers, so print rounding separates them by well under
/// 1e-3 Pa in A and 1e-6 in B, while a wrong-model ladder moves
/// adjacent coefficients by tens of Pa.
const HYBRID_LITERAL_ABS_TOL: f64 = 1e-3;
const HYBRID_LITERAL_REL_TOL: f64 = 1e-6;

/// `mapped_source._resolve_hybrid_coefficients`.
fn resolve_hybrid_coefficients(
    mapping: &Mapping,
    nlevels: usize,
    record_pv: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let literals = mapping.hybrid_literals()?;
    if !record_pv.is_empty() {
        if !record_pv.len().is_multiple_of(2) {
            return Err(frame_invalid(format!(
                "pv coordinate list length {} is not an even A+B split",
                record_pv.len()
            )));
        }
        let half = record_pv.len() / 2;
        if half != nlevels + 1 && half != nlevels {
            return Err(frame_invalid(format!(
                "hybrid coefficient count mismatch: the pv coordinate \
                 octets carry {half} A and {half} B coefficients; \
                 {nlevels} levels accept {} (half-level interfaces) or \
                 {nlevels} (full levels)",
                nlevels + 1
            )));
        }
        let a_values = &record_pv[..half];
        let b_values = &record_pv[half..];
        if a_values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(frame_invalid(
                "pv A coefficients must be finite and non-negative (Pa)",
            ));
        }
        if b_values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
        {
            return Err(frame_invalid(
                "pv B coefficients must be finite within [0, 1]",
            ));
        }
        if let Some((literal_a, literal_b)) = literals {
            for (label, declared, observed) in [
                ("hybrid_a", &literal_a, a_values),
                ("hybrid_b", &literal_b, b_values),
            ] {
                if declared.len() != observed.len() {
                    return Err(frame_invalid(format!(
                        "inline vertical.{label} declares {} coefficients \
                         but the source's pv octets carry {}; the literals \
                         disagree with the bytes",
                        declared.len(),
                        observed.len()
                    )));
                }
                for (index, (lit, pv)) in
                    declared.iter().zip(observed.iter()).enumerate()
                {
                    let tolerance =
                        HYBRID_LITERAL_ABS_TOL + HYBRID_LITERAL_REL_TOL * pv.abs();
                    if (lit - pv).abs() > tolerance {
                        return Err(frame_invalid(format!(
                            "inline vertical.{label} literals disagree with \
                             the source's pv octets at index {index}: \
                             literal {lit} vs pv {pv}"
                        )));
                    }
                }
            }
        }
        return Ok((a_values.to_vec(), b_values.to_vec()));
    }
    if let Some((literal_a, literal_b)) = literals {
        let count = literal_a.len();
        if count != nlevels + 1 && count != nlevels {
            return Err(frame_invalid(format!(
                "hybrid coefficient count mismatch: vertical.hybrid_a \
                 declares {count} coefficients; {nlevels} levels accept \
                 {} (half-level interfaces) or {nlevels} (full levels)",
                nlevels + 1
            )));
        }
        return Ok((literal_a, literal_b));
    }
    Err(frame_invalid(
        "hybrid_sigma_pressure source supplies no A/B coefficients: the \
         selected records carry no pv coordinate octets and the mapping \
         declares no inline vertical.hybrid_a/hybrid_b literals",
    ))
}

/// The literal-only resolution the NetCDF route uses (its bytes carry
/// no pv channel).
pub fn resolve_hybrid_literals_only(
    mapping: &Mapping,
    nlevels: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    resolve_hybrid_coefficients(mapping, nlevels, &[])
}

/// `mapped_source._assemble_grib`.
pub fn assemble_grib(mapping: &Mapping, records: &[GribRecord]) -> Result<DecodedCollection> {
    assemble_grib_within(mapping, records, None)
}

/// [`assemble_grib`] for records of which some were decoded over the
/// atmospheric window `(rows, columns)` of the canonical grid
/// (`DecodeStream::decode_window`): their grid-relative winds are rotated
/// with the rotation of those cells.
pub fn assemble_grib_within(
    mapping: &Mapping,
    records: &[GribRecord],
    window: Option<([usize; 2], [usize; 2])>,
) -> Result<DecodedCollection> {
    if records.is_empty() {
        return Err(selector_unmatched("no mapped GRIB records were decoded"));
    }
    let source_format = mapping.format()?.to_owned();
    let declared_levels = mapping.declared_levels()?;
    let interface_levels = mapping.interface_levels()?;
    let field_names = mapping.field_names()?;

    // matched[(valid_time, member, field)] -> record positions, in decode order.
    let mut matched: BTreeMap<DirectKey, Vec<usize>> = BTreeMap::new();
    for name in &field_names {
        let field = mapping.field(name)?;
        if field.derivation().is_some() {
            continue;
        }
        for (position, record) in records.iter().enumerate() {
            let identity = record.identity();
            let hit = field
                .selectors()
                .iter()
                .any(|selector| selector_matches(selector, &identity, &source_format));
            if hit && declared_vertical_admits(&declared_levels, &interface_levels, &field, record.level_value)? {
                matched
                    .entry((record.valid_time, record.member.clone(), name.clone()))
                    .or_default()
                    .push(position);
            }
        }
    }
    if matched.is_empty() {
        return Err(selector_unmatched(
            "no GRIB messages match the mapping selectors",
        ));
    }


    // Some regular-grid publications pad a rotated regional footprint with
    // missing values. The intersection of selected field masks supplies the footprint. Keep the
    // largest fully finite rectangle; all other field validations still apply.
    let horizontal = mapping.doc.get("coordinates").and_then(|v|v.get("horizontal")).unwrap_or(&Node::Null);
    if horizontal.get("normalization").and_then(Node::as_str)==Some("largest_common_finite_rectangle") {
        if window.is_some() {return Err(frame_invalid("finite-rectangle normalization precedes atmospheric windowing"));}
        let mut positions:Vec<usize>=matched.values().flatten().copied().collect();
        positions.sort_unstable();positions.dedup();
        let first=&records[positions[0]];let ny=first.latitude.len();let nx=first.longitude.len();
        if positions.iter().any(|i|records[*i].values.shape()!=[ny,nx] || records[*i].latitude!=first.latitude || records[*i].longitude!=first.longitude) {
            return Err(frame_invalid("finite rectangle requires one grid per decode batch; decode moving grids independently"));
        }
        let masks:Vec<_>=positions.iter().map(|i|records[*i].values.as_slice().ok_or_else(||frame_invalid("anchor must be contiguous"))).collect::<Result<_>>()?;
        let mut heights=vec![0usize;nx];let mut best=(0usize,0usize,0usize,0usize,0usize);
        for y in 0..ny {
            for x in 0..nx {heights[x]=if masks.iter().all(|m|m[y*nx+x].is_finite()){heights[x]+1}else{0};}
            let mut stack:Vec<usize>=Vec::new();
            for x in 0..=nx {
                let h=if x==nx{0}else{heights[x]};
                while stack.last().is_some_and(|i|heights[*i]>h) {
                    let k=stack.pop().unwrap();let left=stack.last().map_or(0,|i|i+1);let area=heights[k]*(x-left);
                    if area>best.0 {best=(area,y+1-heights[k],y+1,left,x);}
                }
                stack.push(x);
            }
        }
        let (_,y0,y1,x0,x1)=best;
        if y1-y0<3 || x1-x0<3 {return Err(frame_invalid("finite rectangle is smaller than three cells"));}
        if y0!=0 || y1!=ny || x0!=0 || x1!=nx {
            use rayon::prelude::*;
            let normalized=crate::threads::install(||records.par_iter().map(|r| {
                if r.latitude!=first.latitude || r.longitude!=first.longitude || r.values.shape()!=[ny,nx] {
                    return Err(frame_invalid("finite rectangle requires all records on the anchor grid"));
                }
                let mut out=r.clone();out.latitude=r.latitude[y0..y1].to_vec();out.longitude=r.longitude[x0..x1].to_vec();
                out.values=r.values.slice(ndarray::s![y0..y1,x0..x1]).to_owned().into_dyn();
                out.grid_fingerprint=format!("{}:finite-rectangle:{y0}:{y1}:{x0}:{x1}",r.grid_fingerprint);
                Ok(out)
            }).collect::<Vec<Result<GribRecord>>>());
            return assemble_grib(mapping,&crate::threads::in_order(normalized)?);
        }
    }

    let selected: Vec<usize> = matched.values().flatten().copied().collect();
    let fingerprints: BTreeSet<&str> = selected
        .iter()
        .map(|position| records[*position].grid_fingerprint.as_str())
        .collect();
    if fingerprints.len() != 1 && mapping.doc.get("coordinates")
        .and_then(|n| n.get("horizontal")).and_then(|n| n.get("normalization"))
        .and_then(|n| n.as_str()) == Some("coarsest_exact_subset") {
        // A declared nested-grid publication retains every level on the
        // coarsest published lattice. No interpolated coordinates or values.
        use rayon::prelude::*;
        let coarse = selected.iter().map(|i| &records[*i])
            .min_by_key(|r| r.latitude.len() * r.longitude.len()).unwrap();
        let mut unique = selected.clone(); unique.sort_unstable(); unique.dedup();
        let normalized = crate::threads::install(|| unique.par_iter().map(|i| {
            let record = &records[*i];
            let axis = |source: &[f64], target: &[f64], longitude: bool| -> Result<Vec<usize>> {
                target.iter().map(|v| {
                    let indices: Vec<usize> = source.iter().enumerate().filter_map(|(j, x)| {
                        let equal = if longitude { x.rem_euclid(360.0) == v.rem_euclid(360.0) } else { x == v };
                        equal.then_some(j)
                    }).collect();
                    if indices.len() != 1 { return Err(frame_invalid("mixed-grid normalization requires one exact source cell per target coordinate")); }
                    Ok(indices[0])
                }).collect()
            };
            let y = axis(&record.latitude, &coarse.latitude, false)?;
            let x = axis(&record.longitude, &coarse.longitude, true)?;
            if record.values.ndim() != 2 { return Err(frame_invalid("mixed-grid GRIB record must have two horizontal dimensions")); }
            let mut out = record.clone();
            out.values = record.values.select(ndarray::Axis(0), &y).select(ndarray::Axis(1), &x);
            out.latitude = coarse.latitude.clone(); out.longitude = coarse.longitude.clone();
            out.grid_fingerprint = coarse.grid_fingerprint.clone();
            Ok(out)
        }).collect::<Vec<Result<GribRecord>>>());
        let normalized = crate::threads::in_order(normalized)?;
        return assemble_grib(mapping, &normalized);
    }
    if fingerprints.len() != 1 {
        return Err(frame_invalid(
            "selected GRIB fields do not share one source grid",
        ));
    }
    let grid_fingerprint = (*fingerprints.iter().next().expect("one fingerprint")).to_owned();
    let latitude = records[selected[0]].latitude.clone();
    let longitude = records[selected[0]].longitude.clone();
    if selected.iter().any(|position| {
        records[*position].latitude != latitude || records[*position].longitude != longitude
    }) {
        return Err(frame_invalid("selected GRIB coordinate axes differ"));
    }

    let members: BTreeSet<Option<String>> = selected
        .iter()
        .map(|position| records[*position].member.clone())
        .collect();
    if members.len() != 1 {
        return Err(frame_invalid(
            "mapped WRF initialization requires exactly one GRIB member",
        ));
    }

    let mut processes: BTreeMap<TimeKey, BTreeSet<(i64, i64)>> = BTreeMap::new();
    for position in &selected {
        let record = &records[*position];
        if let Some(identity) = record.process_identity {
            processes
                .entry((record.valid_time, record.member.clone()))
                .or_default()
                .insert(identity);
        }
    }
    let mixed: Vec<&TimeKey> = processes
        .iter()
        .filter(|(_key, identities)| identities.len() > 1)
        .map(|(key, _)| key)
        .collect();
    if !mixed.is_empty() {
        return Err(frame_invalid(format!(
            "selected GRIB2 fields mix generating-process identities within \
             a valid time: {mixed:?}"
        )));
    }

    let mut unsupported: Vec<(usize, Vec<i64>)> = Vec::new();
    for position in &selected {
        let record = &records[*position];
        let supported = if source_format == "grib1" {
            record.time_semantics.first() == Some(&0) && record.time_semantics.get(2) == Some(&0)
        } else {
            matches!(record.time_semantics.first(), Some(0) | Some(1))
        };
        if !supported {
            unsupported.push((record.index, record.time_semantics.clone()));
        }
    }
    if !unsupported.is_empty() {
        unsupported.truncate(8);
        return Err(frame_invalid(format!(
            "selected GRIB fields use interval/derived time semantics that \
             rw-wps.mapping.v1 cannot bind: {unsupported:?}"
        )));
    }

    // Fields a pressure-level frame completes where the source leaves
    // them out: their levels may be a subset of the ladder.
    let required: BTreeSet<String> = mapping
        .required_field_names()
        .map(|names| names.into_iter().collect())
        .unwrap_or_default();
    let completed: BTreeSet<&str> =
        crate::derive::completed_fields(mapping.vertical_kind()?, &required)
            .into_iter()
            .collect();

    let vertical_values = if !declared_levels.is_empty() {
        let ladder = carried_ladder(mapping, &declared_levels, &matched, records)?;
        // Records on a declared level the chosen era ladder omits are
        // not stacked: a publication that carries that level for some
        // fields and not others is read on the ladder all of them share.
        if ladder.len() != declared_levels.len() {
            let kept: BTreeSet<u64> = ladder.iter().map(|level| level.to_bits()).collect();
            let omitted: BTreeSet<u64> = declared_levels
                .iter()
                .map(|level| level.to_bits())
                .filter(|bits| !kept.contains(bits))
                .collect();
            for ((_time, _member, name), group) in matched.iter_mut() {
                if !stacks_vertically(mapping, name)? {
                    continue;
                }
                group.retain(|position| {
                    !omitted.contains(&records[*position].level_value.to_bits())
                });
            }
        }
        ladder
    } else {
        let mut level_sets: BTreeSet<Vec<u64>> = BTreeSet::new();
        let mut chosen: Option<Vec<f64>> = None;
        // The ladder is read off the fields the source must publish whole;
        // a completed field states it only when nothing else does.
        let ladder_from_completed = matched.keys().all(|(_time, _member, name)| {
            completed.contains(name.as_str())
                || !mapping
                    .field(name)
                    .and_then(|field| field.target_axes())
                    .map(|axes| axes.iter().any(|axis| axis == "vertical"))
                    .unwrap_or(false)
        });
        for ((_time, _member, name), group) in &matched {
            if !mapping
                .field(name)?
                .target_axes()?
                .iter()
                .any(|axis| axis == "vertical")
            {
                continue;
            }
            if completed.contains(name.as_str()) && !ladder_from_completed {
                continue;
            }
            let mut levels: Vec<f64> = group
                .iter()
                .map(|position| records[*position].level_value)
                .collect();
            levels.sort_by(f64::total_cmp);
            levels.dedup();
            level_sets.insert(levels.iter().map(|value| value.to_bits()).collect());
            chosen = Some(levels);
        }
        if level_sets.len() > 1 {
            return Err(frame_invalid(
                "GRIB atmospheric fields do not share one complete vertical inventory",
            ));
        }
        // A surface-only contribution carries no atmospheric ladder.
        // Full atmospheric completeness remains a composition/target check.
        chosen.unwrap_or_default()
    };

    let (hybrid_a, hybrid_b) = if mapping.vertical_kind()? == "hybrid_sigma_pressure" {
        // Every selected vertical-bearing record states the whole ladder
        // in its pv octets; one source has one ladder, so disagreement
        // is a mixed-source input, not a choice to make.
        let mut pv_lists: BTreeSet<Vec<u64>> = BTreeSet::new();
        let mut chosen_pv: Vec<f64> = Vec::new();
        for ((_time, _member, name), group) in &matched {
            if !mapping
                .field(name)?
                .target_axes()?
                .iter()
                .any(|axis| axis == "vertical")
            {
                continue;
            }
            for position in group {
                let pv = &records[*position].coordinate_values;
                pv_lists.insert(pv.iter().map(|value| value.to_bits()).collect());
                if !pv.is_empty() {
                    chosen_pv = pv.clone();
                }
            }
        }
        let nonempty = pv_lists.iter().filter(|list| !list.is_empty()).count();
        if nonempty > 1 || (nonempty == 1 && pv_lists.len() > 1) {
            return Err(frame_invalid(
                "selected GRIB records do not share one pv coordinate \
                 list; hybrid A/B coefficients must be identical across \
                 the source",
            ));
        }
        resolve_hybrid_coefficients(mapping, vertical_values.len(), &chosen_pv)?
    } else {
        (Vec::new(), Vec::new())
    };

    // ONE TASK PER MAPPED FIELD.  Below this point every field is
    // independent work over disjoint slices of the same immutable
    // records: stack the group, resolve missing cells, convert units,
    // transpose to the target axes, count the NaNs.  The tasks land in
    // pre-assigned slots (`crate::threads`) and are drained in the
    // BTreeMap's own key order below, so the collection, the refusal
    // and the cycle cross-check are the serial engine's.  Nothing new
    // is allocated by doing it this way: each task's stacked array is
    // the array that ends up in the collection either way.
    let entries: Vec<(&DirectKey, &Vec<usize>)> = matched.iter().collect();
    let slots: Vec<Result<(DirectValue, NaiveDateTime)>> = crate::threads::install(|| {
        use rayon::prelude::*;
        entries
            .par_iter()
            .map(|(key, group)| {
                assemble_one_field(
                    mapping,
                    records,
                    &source_format,
                    &vertical_values,
                    &completed,
                    key,
                    group,
                )
            })
            .collect()
    });
    let mut direct: BTreeMap<DirectKey, DirectValue> = BTreeMap::new();
    let mut cycles: BTreeMap<TimeKey, NaiveDateTime> = BTreeMap::new();
    for ((key, _group), slot) in entries.iter().zip(slots) {
        let (value, cycle) = slot?;
        let (valid_time, member, _field_name) = *key;
        let cycle_key = (*valid_time, member.clone());
        if let Some(existing) = cycles.get(&cycle_key)
            && *existing != cycle {
                return Err(frame_invalid(format!(
                    "mapped GRIB fields mix source cycles at {valid_time}"
                )));
            }
        cycles.insert(cycle_key, cycle);
        direct.insert((*key).clone(), value);
    }

    let (direct, cycles) = broadcast_invariant_fields(mapping, direct, cycles)?;
    let direct = apply_landmask_water_missing(mapping, direct)?;
    let declaration = mapping.grid_declaration()?;
    let direct = if declaration.rotates_winds() {
        rotate_grid_relative_winds(direct, &declaration, window)?
    } else {
        direct
    };

    // The whole-plane ladder of an air_pressure decoded over the window,
    // in the vertical order its planes were stacked in: the frame states
    // it where the full decode states the medians of its whole planes.
    let mut original_pressure_hpa: Option<Vec<f64>> = None;
    for ((_time, _member, name), group) in &matched {
        if name != "air_pressure" {
            continue;
        }
        let ladders: Vec<Option<f64>> =
            group.iter().map(|position| records[*position].plane_ladder_hpa).collect();
        if ladders.iter().all(Option::is_none) {
            continue;
        }
        if original_pressure_hpa.is_some() || ladders.iter().any(Option::is_none) {
            return Err(frame_invalid(
                "air_pressure decoded over the atmospheric window must state a \
                 whole-plane ladder value for every record of one valid time",
            ));
        }
        let mut by_level: BTreeMap<u64, f64> = BTreeMap::new();
        for (position, ladder) in group.iter().zip(&ladders) {
            let level = records[*position].level_value;
            if by_level.insert(level.to_bits(), ladder.expect("checked above")).is_some() {
                return Err(frame_invalid(format!(
                    "duplicate air_pressure GRIB level {level} decoded over the window"
                )));
            }
        }
        let ladder: Option<Vec<f64>> = vertical_values
            .iter()
            .map(|level| by_level.get(&level.to_bits()).copied())
            .collect();
        original_pressure_hpa = Some(ladder.ok_or_else(|| {
            frame_invalid(
                "air_pressure decoded over the atmospheric window does not state \
                 a whole-plane ladder value on every level of the frame",
            )
        })?);
    }

    Ok(DecodedCollection {
        latitude,
        longitude,
        vertical_values,
        direct,
        source_cycles: cycles,
        grid_fingerprint,
        hybrid_a,
        hybrid_b,
        original_pressure_hpa,
    })
}

/// Whether a mapped field's records stack on the vertical axis.
fn stacks_vertically(mapping: &Mapping, name: &str) -> Result<bool> {
    Ok(mapping
        .field(name)?
        .source_axes()?
        .iter()
        .any(|axis| axis == "vertical"))
}

/// `mapped_source._carried_ladder`: the declared ladder a decode stacks.
///
/// A publisher adds or drops levels between releases of one product,
/// so a file from the other side of the change lacks a level
/// `vertical.levels` names.  `vertical.era_ladders` lists the whole
/// ladders such publications carry; the decode stacks the largest
/// declared ladder every vertical field's records carry in full.  With
/// no era ladders, or none that fits, the declared ladder comes back and
/// the coverage check names what is missing from it.
fn carried_ladder(
    mapping: &Mapping,
    declared: &[f64],
    matched: &BTreeMap<DirectKey, Vec<usize>>,
    records: &[GribRecord],
) -> Result<Vec<f64>> {
    let eras = mapping.era_ladders()?;
    if eras.is_empty() {
        return Ok(declared.to_vec());
    }
    let mut carried: Vec<BTreeSet<u64>> = Vec::new();
    for ((_time, _member, name), group) in matched {
        if !stacks_vertically(mapping, name)? {
            continue;
        }
        let mut levels: BTreeSet<u64> = group
            .iter()
            .map(|position| records[*position].level_value.to_bits())
            .collect();
        // A field declared on part of the ladder (`published_levels`)
        // holds the levels it is zero on as well as the ones it carries.
        levels.extend(unpublished_levels(mapping, name, declared)?);
        carried.push(levels);
    }
    Ok(choose_ladder(declared, &eras, &carried))
}

/// The declared levels (as bits) a `published_levels` field is not
/// published on; empty for a field that declares no subset.
fn unpublished_levels(mapping: &Mapping, name: &str, declared: &[f64]) -> Result<BTreeSet<u64>> {
    let Some(published) = mapping.field(name)?.published_levels()? else {
        return Ok(BTreeSet::new());
    };
    let published: BTreeSet<u64> = published.iter().map(|level| level.to_bits()).collect();
    Ok(declared
        .iter()
        .map(|level| level.to_bits())
        .filter(|bits| !published.contains(bits))
        .collect())
}

/// The largest of `declared` and the era ladders that every set in
/// `carried` holds, in the declared order; `declared` when none does.
fn choose_ladder(declared: &[f64], eras: &[Vec<f64>], carried: &[BTreeSet<u64>]) -> Vec<f64> {
    if carried.is_empty() {
        return declared.to_vec();
    }
    let fits = |ladder: &[f64]| {
        carried
            .iter()
            .all(|levels| ladder.iter().all(|level| levels.contains(&level.to_bits())))
    };
    if fits(declared) {
        return declared.to_vec();
    }
    let mut best: Option<Vec<f64>> = None;
    for era in eras {
        let members: BTreeSet<u64> = era.iter().map(|level| level.to_bits()).collect();
        let ladder: Vec<f64> = declared
            .iter()
            .copied()
            .filter(|level| members.contains(&level.to_bits()))
            .collect();
        if fits(&ladder) && best.as_ref().is_none_or(|chosen| ladder.len() > chosen.len()) {
            best = Some(ladder);
        }
    }
    best.unwrap_or_else(|| declared.to_vec())
}

/// One mapped field, assembled from the records its selectors matched.
///
/// The body of `assemble_grib`'s per-field loop, lifted out so it can
/// run as one task.  It returns the field and the source cycle its
/// records agree on; the caller cross-checks that cycle against the
/// other fields of the same valid time, in order, because that check is
/// about the collection rather than about one field.
fn assemble_one_field(
    mapping: &Mapping,
    records: &[GribRecord],
    source_format: &str,
    vertical_values: &[f64],
    completed: &BTreeSet<&str>,
    key: &DirectKey,
    group: &[usize],
) -> Result<(DirectValue, NaiveDateTime)> {
    {
        let (valid_time, member, field_name) = key;
        let field = mapping.field(field_name)?;
        let source_axes = field.source_axes()?;
        if source_axes.iter().any(|axis| axis == "time" || axis == "member") {
            return Err(frame_invalid(format!(
                "GRIB embedded time/member metadata must not also appear in \
                 {field_name}.source_axes"
            )));
        }
        let selectors = field.selectors();
        let policy = field.missing_kind()?;
        // `repeated_record: "last"`: a record the publisher writes twice at
        // one level and valid time is read from its later message (HRRRv1
        // wrfprs repeats HGT at 1000 hPa and at the surface; only the later
        // 1000 hPa record survives into HRRRv2 and current files).
        let repeated_last = field.repeated_record_last()?;
        let stacking_axis = if source_axes.iter().any(|axis| axis == "vertical") {
            Some("vertical")
        } else if source_axes.iter().any(|axis| axis == "half_level") {
            Some("half_level")
        } else if source_axes.iter().any(|axis| axis == "soil") {
            Some("soil")
        } else {
            None
        };
        // Each record answers the FIRST of the field's selectors it
        // matches: a field lists its selectors in the order it prefers
        // them, so a record of a later selector stands in only where no
        // earlier one was published.  With one selector every record has
        // rank 0 and nothing here changes.
        let rank = |position: usize| -> usize {
            let identity = records[position].identity();
            selectors
                .iter()
                .position(|selector| selector_matches(selector, &identity, source_format))
                .unwrap_or(0)
        };
        // The record's values in the field's source units: a selector may
        // declare the factor that takes its record there.
        let scaled = |position: usize, rank: usize| -> ArrayD<f64> {
            let values = &records[position].values;
            match selectors
                .get(rank)
                .and_then(|selector| selector.field("scale"))
                .and_then(Node::as_f64)
            {
                Some(scale) => values.mapv(|value| value * scale),
                None => values.clone(),
            }
        };
        // The records the field is built from.
        let mut used: Vec<usize> = Vec::with_capacity(group.len());
        // Levels of the ladder the source did not publish for a field the
        // frame completes; they are NaN planes until then.
        let mut absent_levels = 0usize;

        let mut values = match stacking_axis {
            None => {
                let ranks: Vec<usize> = group.iter().map(|position| rank(*position)).collect();
                let best = ranks.iter().copied().min().unwrap_or(0);
                let mut chosen: Vec<usize> = group
                    .iter()
                    .zip(&ranks)
                    .filter(|(_position, rank)| **rank == best)
                    .map(|(position, _rank)| *position)
                    .collect();
                if chosen.len() > 1 && repeated_last {
                    let later = *chosen.iter().max().expect("non-empty");
                    chosen = vec![later];
                }
                if chosen.len() != 1 {
                    return Err(frame_invalid(format!(
                        "duplicate GRIB messages for scalar field {field_name} at {valid_time}"
                    )));
                }
                used.push(chosen[0]);
                scaled(chosen[0], best)
            }
            Some(axis_name @ ("vertical" | "half_level")) => {
                let interfaces = mapping.interface_levels()?;
                let vertical_values = if axis_name == "half_level" { interfaces.as_slice() } else { vertical_values };
                let ranks: Vec<usize> = group.iter().map(|position| rank(*position)).collect();
                let mut best: BTreeMap<u64, usize> = BTreeMap::new();
                for (position, rank) in group.iter().zip(&ranks) {
                    let entry = best
                        .entry(records[*position].level_value.to_bits())
                        .or_insert(*rank);
                    *entry = (*entry).min(*rank);
                }
                let mut by_level: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
                for (position, rank) in group.iter().zip(&ranks) {
                    let level = records[*position].level_value;
                    if best[&level.to_bits()] != *rank {
                        continue;
                    }
                    let earlier = by_level.insert(level.to_bits(), (*position, *rank));
                    if let Some((earlier_position, earlier_rank)) = earlier {
                        if repeated_last {
                            // The later message stands, whichever order the
                            // group lists them in.
                            if earlier_position > *position {
                                by_level.insert(level.to_bits(), (earlier_position, earlier_rank));
                            }
                            continue;
                        }
                        return Err(frame_invalid(format!(
                            "duplicate {field_name} GRIB level {level} at {valid_time}"
                        )));
                    }
                }
                // A level the field is not published on (`published_levels`)
                // is zero when the file leaves it out; a published level the
                // file leaves out is still missing.  A record on such a
                // level is read like any other.
                let published: Option<BTreeSet<u64>> = if axis_name == "vertical" {
                    field
                        .published_levels()?
                        .map(|levels| levels.iter().map(|level| level.to_bits()).collect())
                } else {
                    None
                };
                let upper_boundary = field.raw.field("published_levels")
                    .and_then(|n| n.get("absent")).and_then(|n| n.as_str())
                    == Some("nearest_upper_boundary");
                let boundary = if upper_boundary {
                    if field_name != "specific_humidity" || mapping.vertical_kind()? != "pressure" {
                        return Err(frame_invalid("upper boundary extension requires pressure-level specific_humidity"));
                    }
                    let levels = field.published_levels()?.unwrap();
                    let top = levels.iter().copied().min_by(f64::total_cmp).unwrap();
                    if vertical_values.iter().any(|v| !levels.contains(v) && *v >= top) {
                        return Err(frame_invalid("upper boundary extension cannot fill interior or lower levels"));
                    }
                    Some(*by_level.get(&top.to_bits()).ok_or_else(|| frame_invalid("published upper boundary level is absent"))?)
                } else { None };
                let zero_levels: BTreeSet<u64> = match &published {
                    Some(published) => vertical_values
                        .iter()
                        .map(|level| level.to_bits())
                        .filter(|bits| !by_level.contains_key(bits) && !published.contains(bits))
                        .collect(),
                    None => BTreeSet::new(),
                };
                let missing: Vec<f64> = vertical_values
                    .iter()
                    .copied()
                    .filter(|level| {
                        !by_level.contains_key(&level.to_bits())
                            && !zero_levels.contains(&level.to_bits())
                    })
                    .collect();
                let declared: BTreeSet<u64> =
                    vertical_values.iter().map(|value| value.to_bits()).collect();
                let extra: Vec<f64> = by_level
                    .keys()
                    .filter(|bits| !declared.contains(bits))
                    .map(|bits| f64::from_bits(*bits))
                    .collect();
                // A field the frame completes may leave levels out; it is
                // derived there from the frame's own state.  Only under the
                // reject policy, whose NaN can mean nothing but "absent".
                let completes = completed.contains(field_name.as_str()) && policy == "reject";
                if (!missing.is_empty() && !completes) || !extra.is_empty() {
                    let missing = crate::refusal::python_float_list_repr(&missing);
                    let extra = crate::refusal::python_float_list_repr(&extra);
                    return Err(selector_unmatched(format!(
                        "{field_name} vertical coverage mismatch; missing={missing}, \
                         extra={extra}"
                    )));
                }
                absent_levels = missing.len();
                let plane: Vec<usize> = by_level
                    .values()
                    .next()
                    .map(|(position, _rank)| records[*position].values.shape().to_vec())
                    .expect("a matched field has at least one level");
                let mut ordered: Vec<ArrayD<f64>> = Vec::with_capacity(vertical_values.len());
                for level in vertical_values {
                    match by_level.get(&level.to_bits()) {
                        Some((position, rank)) => {
                            let level_values = scaled(*position, *rank);
                            if absent_levels > 0
                                && level_values.iter().any(|value| !value.is_finite())
                            {
                                return Err(frame_invalid(format!(
                                    "{field_name} contains missing/non-finite GRIB data"
                                )));
                            }
                            used.push(*position);
                            ordered.push(level_values);
                        }
                        None if zero_levels.contains(&level.to_bits()) => {
                            if let Some((position, rank)) = boundary {
                                let values=scaled(position,rank);
                                if values.iter().any(|v| !v.is_finite()) {
                                    return Err(frame_invalid("upper boundary field contains non-finite data"));
                                }
                                used.push(position);
                                ordered.push(values);
                            } else { ordered.push(ArrayD::from_elem(plane.clone(), 0.0)); }
                        }
                        None => ordered.push(ArrayD::from_elem(plane.clone(), f64::NAN)),
                    }
                }
                let axis = source_axes
                    .iter()
                    .position(|item| item == axis_name)
                    .expect("vertical axis proven present");
                array::stack(&ordered, axis, field_name)?
            }
            Some("soil") => {
                if mapping.soil_layer_count()?.is_none() {
                    return Err(frame_invalid(
                        "soil_layer_count is required to stack GRIB soil fields",
                    ));
                }
                if selectors.is_empty() {
                    return Err(frame_invalid(format!(
                        "{field_name} soil stacking requires ordered GRIB selectors"
                    )));
                }
                if group.len() != selectors.len() {
                    return Err(frame_invalid(format!(
                        "{field_name} has {} GRIB soil records; the mapping declares \
                         {} ordered soil selectors",
                        group.len(),
                        selectors.len()
                    )));
                }
                let mut by_selector: BTreeMap<usize, usize> = BTreeMap::new();
                for position in group {
                    let identity = records[*position].identity();
                    let hits: Vec<usize> = selectors
                        .iter()
                        .enumerate()
                        .filter(|(_index, selector)| {
                            selector_matches(selector, &identity, source_format)
                        })
                        .map(|(index, _)| index)
                        .collect();
                    if hits.len() != 1 {
                        return Err(frame_invalid(format!(
                            "{field_name} GRIB soil record {} matches {} selectors; \
                             expected exactly one",
                            records[*position].index,
                            hits.len()
                        )));
                    }
                    if by_selector.insert(hits[0], *position).is_some() {
                        return Err(frame_invalid(format!(
                            "duplicate {field_name} GRIB soil selector {} at {valid_time}",
                            hits[0]
                        )));
                    }
                }
                let absent: Vec<usize> = (0..selectors.len())
                    .filter(|index| !by_selector.contains_key(index))
                    .collect();
                if !absent.is_empty() {
                    return Err(selector_unmatched(format!(
                        "{field_name} is missing GRIB soil selectors {absent:?}"
                    )));
                }
                used.extend(group.iter().copied());
                let ordered: Vec<ArrayD<f64>> = (0..selectors.len())
                    .map(|index| scaled(by_selector[&index], index))
                    .collect();
                let axis = source_axes
                    .iter()
                    .position(|item| item == "soil")
                    .expect("soil axis proven present");
                array::stack(&ordered, axis, field_name)?
            }
            Some(other) => unreachable!("unexpected stacking axis {other}"),
        };

        // The missing cells are answered in ONE pass over the field.
        // The mask this replaces was a `Vec<bool>` as long as the field
        // -- 63 million bytes for one 3-km pressure-level variable --
        // materialized only to be walked once and dropped.  A cell that
        // is replaced is finite afterwards, so a single in-place pass
        // decides and rewrites exactly the cells the mask selected.
        // A field with absent levels had each published level checked as
        // it was stacked, and its NaN planes are exactly the absent ones.
        if policy == "reject" {
            if absent_levels == 0 && values.iter().any(|value| !value.is_finite()) {
                return Err(frame_invalid(format!(
                    "{field_name} contains missing/non-finite GRIB data"
                )));
            }
        } else {
            let replacement = if policy == "value" {
                field.missing_value()?
            } else {
                f64::NAN
            };
            for value in values.iter_mut() {
                if !value.is_finite() {
                    *value = replacement;
                }
            }
        }
        let values = array::unit_transform(
            values,
            field.unit_scale(),
            field.unit_offset(),
            field_name,
        )?;
        let target_axes = field.target_axes()?;
        let values = array::transpose_to_target(values, &source_axes, &target_axes, field_name)?;

        let mut ordered_group = used;
        ordered_group.sort_by_key(|position| records[*position].index);
        let references: Vec<String> = ordered_group
            .iter()
            .map(|position| format!("{}:{}", records[*position].source, records[*position].index))
            .collect();
        let group_cycles: BTreeSet<NaiveDateTime> = ordered_group
            .iter()
            .map(|position| records[*position].reference_time)
            .collect();
        if group_cycles.len() != 1 {
            return Err(frame_invalid(format!(
                "{field_name} GRIB records mix source cycles at {valid_time}"
            )));
        }
        let cycle = *group_cycles.iter().next().expect("one cycle");
        let missing_count = array::count_nan(&values);
        Ok((
            DirectValue {
                name: field_name.clone(),
                valid_time: *valid_time,
                member: member.clone(),
                source_cycle: cycle,
                axes: target_axes,
                values,
                missing_count,
                references,
            },
            cycle,
        ))
    }
}

/// `mapped_source._broadcast_invariant_fields`.
pub fn broadcast_invariant_fields(
    mapping: &Mapping,
    direct: BTreeMap<DirectKey, DirectValue>,
    cycles: BTreeMap<TimeKey, NaiveDateTime>,
) -> Result<(BTreeMap<DirectKey, DirectValue>, BTreeMap<TimeKey, NaiveDateTime>)> {
    let mut invariant: BTreeSet<String> = BTreeSet::new();
    for field in mapping.fields()? {
        if field.time_binding() == Some("cycle_invariant") {
            invariant.insert(field.name.clone());
        }
    }
    if invariant.is_empty() {
        return Ok((direct, cycles));
    }
    let dependent_keys: BTreeSet<TimeKey> = direct
        .keys()
        .filter(|(_time, _member, name)| !invariant.contains(name))
        .map(|(time, member, _name)| (*time, member.clone()))
        .collect();
    if dependent_keys.is_empty() {
        return Err(frame_invalid(
            "every decoded mapped field is declared time-invariant; there is \
             no time-dependent state to define the forcing axis",
        ));
    }
    let dependent_cycles: BTreeSet<NaiveDateTime> = dependent_keys
        .iter()
        .filter_map(|key| cycles.get(key).copied())
        .collect();
    if dependent_cycles.len() > 1 {
        return Err(frame_invalid(format!(
            "cycle-invariant fields cannot broadcast across mixed source \
             cycles {dependent_cycles:?}; one broadcast belongs to one cycle"
        )));
    }
    // The time-dependent fields are MOVED into the result, not copied
    // into it.  Rebuilding this map by cloning every value duplicated
    // the whole decoded frameset -- 8 GB on a 3-km CONUS source -- to
    // produce a map whose only difference is which keys it holds.  The
    // invariant entries are lifted out instead, in the same BTreeMap key
    // order the filter walked, and the broadcast below reads them from
    // there.
    let lifted: Vec<DirectKey> = direct
        .keys()
        .filter(|(_time, _member, name)| invariant.contains(name))
        .cloned()
        .collect();
    let mut result = direct;
    let mut invariant_values: Vec<(DirectKey, DirectValue)> = Vec::with_capacity(lifted.len());
    for key in lifted {
        if let Some(value) = result.remove(&key) {
            invariant_values.push((key, value));
        }
    }
    let kept_cycles: BTreeMap<TimeKey, NaiveDateTime> = cycles
        .iter()
        .filter(|(key, _)| dependent_keys.contains(key))
        .map(|(key, value)| (key.clone(), *value))
        .collect();
    for name in &invariant {
        let instances: Vec<&DirectValue> = invariant_values
            .iter()
            .filter(|((_time, _member, field), _value)| field == name)
            .map(|(_key, value)| value)
            .collect();
        let Some(reference) = instances.first() else {
            continue;
        };
        for instance in instances.iter().skip(1) {
            if instance.values != reference.values {
                return Err(frame_invalid(format!(
                    "cycle-invariant field {name} changes across its supplied \
                     valid times; the declaration promises one array per source cycle"
                )));
            }
        }
        for (valid_time, member) in &dependent_keys {
            result.insert(
                (*valid_time, member.clone(), name.clone()),
                DirectValue {
                    name: name.clone(),
                    valid_time: *valid_time,
                    member: member.clone(),
                    source_cycle: kept_cycles[&(*valid_time, member.clone())],
                    axes: reference.axes.clone(),
                    values: reference.values.clone(),
                    missing_count: reference.missing_count,
                    references: reference.references.clone(),
                },
            );
        }
    }
    Ok((result, kept_cycles))
}

/// `mapped_source._apply_landmask_water_missing`.
pub fn apply_landmask_water_missing(
    mapping: &Mapping,
    direct: BTreeMap<DirectKey, DirectValue>,
) -> Result<BTreeMap<DirectKey, DirectValue>> {
    let mut masked_names: Vec<String> = Vec::new();
    for field in mapping.fields()? {
        if ["landmask_water", "landmask_zero_water"].contains(&field.missing_kind()?) {
            masked_names.push(field.name.clone());
        }
    }
    if masked_names.is_empty() {
        return Ok(direct);
    }
    masked_names.sort();
    // The replacements are computed first and applied to the map this
    // was HANDED, rather than to a full copy of it: only the masked
    // fields change, and cloning the whole decoded frameset to overwrite
    // a handful of its entries duplicated every array in it.
    let mut replacements: Vec<(DirectKey, DirectValue)> = Vec::new();
    let time_keys: BTreeSet<TimeKey> = direct
        .keys()
        .map(|(time, member, _name)| (*time, member.clone()))
        .collect();
    for (valid_time, member) in time_keys {
        let present: Vec<&String> = masked_names
            .iter()
            .filter(|name| {
                direct.contains_key(&(valid_time, member.clone(), (*name).clone()))
            })
            .collect();
        if present.is_empty() {
            continue;
        }
        let land = direct
            .get(&(valid_time, member.clone(), "land_fraction".to_owned()))
            .ok_or_else(|| {
                frame_invalid(format!(
                    "landmask_water fields {present:?} at {valid_time} have no \
                     decoded land_fraction to key water cells from"
                ))
            })?;
        if land.values.ndim() != 2 || land.values.iter().any(|value| !value.is_finite()) {
            return Err(frame_invalid(
                "land_fraction must be a finite 2-D field to serve as the \
                 landmask_water mask",
            ));
        }
        let water: Vec<bool> = array::contiguous(&land.values)
            .iter()
            .map(|value| *value < 0.5)
            .collect();
        for name in present {
            let key = (valid_time, member.clone(), name.clone());
            let value = &direct[&key];
            let zero_water = mapping.field(name)?.missing_kind()? == "landmask_zero_water";
            let mut masked = value.values.as_standard_layout().to_owned();
            let cells = water.len();
            let total = masked.len();
            for (position, cell) in masked.iter_mut().enumerate() {
                if water[position % cells] {
                    *cell = if zero_water { 0.0 } else { f64::NAN };
                } else if zero_water && !cell.is_finite() {
                    return Err(frame_invalid(format!("{name} has missing land values under landmask_zero_water")));
                }
            }
            debug_assert_eq!(total % cells, 0, "a masked field is whole horizontal planes");
            let missing_count = array::count_nan(&masked);
            replacements.push((
                key,
                DirectValue {
                    values: masked,
                    missing_count,
                    references: if zero_water {
                        let mut references = value.references.clone();
                        references.push("missing_policy:landmask_zero_water:land_fraction_threshold_0.5".into());
                        references
                    } else { value.references.clone() },
                    ..value.clone()
                },
            ));
        }
    }
    let mut result = direct;
    for (key, value) in replacements {
        result.insert(key, value);
    }
    Ok(result)
}

/// `mapped_source._rotate_grid_relative_winds`.
pub fn rotate_grid_relative_winds(
    direct: BTreeMap<DirectKey, DirectValue>,
    declaration: &crate::model::GridDeclaration,
    window: Option<([usize; 2], [usize; 2])>,
) -> Result<BTreeMap<DirectKey, DirectValue>> {
    let (ny, nx) = declaration
        .projected_shape()
        .expect("a rotating declaration carries parameters");
    // The family's own rotation field, over the whole grid or a window of
    // it: the Lambert cone angle, or the rotated lattice's bearing.
    let rotation_within = |rows: [usize; 2], columns: [usize; 2]| -> (Vec<f64>, Vec<f64>) {
        if let Some(parameters) = &declaration.parameters {
            crate::lambert::declared_grid_rotation_within(parameters, rows, columns)
        } else {
            let parameters = declaration
                .rotated
                .as_ref()
                .expect("a rotating declaration carries parameters");
            crate::rotated::declared_grid_rotation_within(parameters, rows, columns)
        }
    };
    // Each rotation is computed on first need: the whole grid's for a pair
    // decoded whole, the window's cells for a pair decoded over the window.
    let mut whole: Option<(Vec<f64>, Vec<f64>)> = None;
    let mut windowed: Option<(Vec<f64>, Vec<f64>)> = None;
    // Same reasoning as the landmask: the rotated components are
    // computed first and written into the map this was handed.  Only
    // the wind pairs change, and copying every other field to rewrite
    // two of them duplicated the whole decoded frameset.
    let mut rotated: Vec<(DirectKey, DirectValue)> = Vec::new();
    let time_keys: BTreeSet<TimeKey> = direct
        .keys()
        .map(|(time, member, _name)| (*time, member.clone()))
        .collect();
    for (valid_time, member) in time_keys {
        for (u_name, v_name) in ROTATED_WIND_PAIRS {
            let u_key = (valid_time, member.clone(), u_name.to_owned());
            let v_key = (valid_time, member.clone(), v_name.to_owned());
            let has_u = direct.contains_key(&u_key);
            let has_v = direct.contains_key(&v_key);
            if !has_u && !has_v {
                continue;
            }
            if !has_u || !has_v {
                let absent = if has_u { v_name } else { u_name };
                return Err(frame_invalid(format!(
                    "grid-relative wind rotation at {valid_time} needs both \
                     components of ({u_name}, {v_name}); {absent} is not mapped"
                )));
            }
            let u = &direct[&u_key];
            let v = &direct[&v_key];
            if u.axes != v.axes || u.values.shape() != v.values.shape() {
                return Err(frame_invalid(format!(
                    "({u_name}, {v_name}) at {valid_time} disagree in axes/shape; \
                     rotation needs one shared grid"
                )));
            }
            let rank = u.axes.len();
            if rank < 2 || u.axes[rank - 2] != "y" || u.axes[rank - 1] != "x" {
                return Err(frame_invalid(format!(
                    "({u_name}, {v_name}) must end in y/x axes to rotate"
                )));
            }
            let shape = u.values.shape();
            let plane_shape = [shape[rank - 2], shape[rank - 1]];
            let (sina, cosa) = if plane_shape == [ny, nx] {
                &*whole.get_or_insert_with(|| rotation_within([0, ny], [0, nx]))
            } else {
                match window {
                    Some((rows, columns))
                        if plane_shape == [rows[1] - rows[0], columns[1] - columns[0]] =>
                    {
                        &*windowed.get_or_insert_with(|| rotation_within(rows, columns))
                    }
                    _ => {
                        return Err(frame_invalid(format!(
                            "({u_name}, {v_name}) at {valid_time} do not share the \
                             declared grid shape"
                        )));
                    }
                }
            };
            let cells = sina.len();
            let u_values = array::contiguous(&u.values);
            let v_values = array::contiguous(&v.values);
            // Cell by cell, each from its own two inputs and its plane's
            // rotation, so the cells are computed on every core and
            // collected in position order: the same arithmetic in the
            // same order per cell, the same bytes.  Named breakage: the
            // two loops ran on one core, about 1.3 s of every 3 km CONUS
            // native-level valid time, and `..u.clone()` copied each wind
            // component whole only to replace its values.
            let (earth_u, earth_v): (Vec<f64>, Vec<f64>) = crate::threads::install(|| {
                use rayon::prelude::*;
                let earth_u = (0..u_values.len()).into_par_iter().map(|position| {
                    let plane = position % cells;
                    u_values[position] * cosa[plane] - v_values[position] * sina[plane]
                }).collect();
                let earth_v = (0..v_values.len()).into_par_iter().map(|position| {
                    let plane = position % cells;
                    v_values[position] * cosa[plane] + u_values[position] * sina[plane]
                }).collect();
                (earth_u, earth_v)
            });
            let earth_u = ArrayD::from_shape_vec(ndarray::IxDyn(u.values.shape()), earth_u)
                .map_err(|error| frame_invalid(format!("rotated {u_name}: {error}")))?;
            let earth_v = ArrayD::from_shape_vec(ndarray::IxDyn(v.values.shape()), earth_v)
                .map_err(|error| frame_invalid(format!("rotated {v_name}: {error}")))?;
            rotated.push((
                u_key,
                DirectValue {
                    name: u.name.clone(),
                    valid_time: u.valid_time,
                    member: u.member.clone(),
                    source_cycle: u.source_cycle,
                    axes: u.axes.clone(),
                    values: earth_u,
                    missing_count: u.missing_count,
                    references: u.references.clone(),
                },
            ));
            rotated.push((
                v_key,
                DirectValue {
                    name: v.name.clone(),
                    valid_time: v.valid_time,
                    member: v.member.clone(),
                    source_cycle: v.source_cycle,
                    axes: v.axes.clone(),
                    values: earth_v,
                    missing_count: v.missing_count,
                    references: v.references.clone(),
                },
            ));
        }
    }
    let mut result = direct;
    for (key, value) in rotated {
        result.insert(key, value);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hybrid_mapping(literals: Option<(&str, &str)>) -> Mapping {
        let vertical = match literals {
            Some((a, b)) => format!(
                r#""kind": "hybrid_sigma_pressure", "units": "1",
                   "surface_pressure_field": "surface_pressure",
                   "hybrid_a": {a}, "hybrid_b": {b}"#
            ),
            None => r#""kind": "hybrid_sigma_pressure", "units": "1",
                       "surface_pressure_field": "surface_pressure""#
                .to_owned(),
        };
        let text = format!(
            r#"{{"schema": "rw-wps.mapping.v1", "name": "t", "format": "grib2",
                "coordinates": {{"vertical": {{{vertical}}}}}, "fields": {{}}}}"#
        );
        let bytes = text.as_bytes().to_vec();
        Mapping {
            sha256: crate::digest::bytes_sha256(&bytes),
            doc: crate::node::Node::parse(&bytes).unwrap(),
            path: "<test>".to_owned(),
        }
    }

    const PV: [f64; 8] = [0.0, 6000.0, 4000.0, 0.0, 0.0, 0.24, 0.7, 1.0];

    #[test]
    fn pv_octets_resolve_into_an_a_then_b_split() {
        let mapping = hybrid_mapping(None);
        let (a, b) = resolve_hybrid_coefficients(&mapping, 3, &PV).unwrap();
        assert_eq!(a, vec![0.0, 6000.0, 4000.0, 0.0]);
        assert_eq!(b, vec![0.0, 0.24, 0.7, 1.0]);
    }

    #[test]
    fn literals_supply_the_ladder_when_the_bytes_carry_no_pv() {
        let mapping = hybrid_mapping(Some((
            "[0.0, 6000.0, 4000.0, 0.0]",
            "[0.0, 0.24, 0.7, 1.0]",
        )));
        let (a, b) = resolve_hybrid_coefficients(&mapping, 3, &[]).unwrap();
        assert_eq!(a, vec![0.0, 6000.0, 4000.0, 0.0]);
        assert_eq!(b, vec![0.0, 0.24, 0.7, 1.0]);
    }

    #[test]
    fn literals_disagreeing_with_pv_refuse_naming_the_index() {
        let mapping = hybrid_mapping(Some((
            "[0.0, 6100.0, 4000.0, 0.0]",
            "[0.0, 0.24, 0.7, 1.0]",
        )));
        let refusal = resolve_hybrid_coefficients(&mapping, 3, &PV).unwrap_err();
        assert!(refusal.message.contains("index 1"), "{}", refusal.message);
    }

    #[test]
    fn no_pv_and_no_literals_refuses_naming_both_channels() {
        let mapping = hybrid_mapping(None);
        let refusal = resolve_hybrid_coefficients(&mapping, 3, &[]).unwrap_err();
        assert!(refusal.message.contains("pv"), "{}", refusal.message);
        assert!(refusal.message.contains("hybrid_a"), "{}", refusal.message);
    }

    /// Two stacked pressure fields on a declared 1000/5000/10000 Pa
    /// ladder, with `era_ladders` spliced in verbatim when given.
    fn pressure_mapping(era_ladders: &str) -> Mapping {
        let field = |category: i64, parameter: i64, units: &str| {
            format!(
                r#"{{"selectors": [{{"format": "grib2", "discipline": 0,
                     "category": {category}, "parameter": {parameter},
                     "level_type": 100}}],
                   "units": {{"source": "{units}", "target": "{units}"}},
                   "source_axes": ["vertical", "y", "x"],
                   "target_axes": ["vertical", "y", "x"],
                   "location": "mass", "staggering": "none",
                   "missing": {{"kind": "reject"}}}}"#
            )
        };
        let text = format!(
            r#"{{"schema": "rw-wps.mapping.v1", "name": "t", "format": "grib2",
                "coordinates": {{"vertical": {{"kind": "pressure", "units": "Pa",
                    "levels": [1000, 5000, 10000]{era_ladders}}}}},
                "fields": {{"air_temperature": {},
                            "eastward_wind": {}}}}}"#,
            field(0, 0, "K"),
            field(2, 2, "m s-1"),
        );
        let bytes = text.as_bytes().to_vec();
        Mapping {
            sha256: crate::digest::bytes_sha256(&bytes),
            doc: crate::node::Node::parse(&bytes).unwrap(),
            path: "<test>".to_owned(),
        }
    }

    fn pressure_record(index: usize, category: i64, parameter: i64, level: f64) -> GribRecord {
        let time = chrono::NaiveDate::from_ymd_opt(2024, 5, 17)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        GribRecord {
            source: "<test>".to_owned(),
            index,
            reference_time: time,
            valid_time: time,
            member: None,
            parameter,
            level_type: 100,
            level_value: level,
            table_version: None,
            center: None,
            subcenter: None,
            master_table_version: None,
            local_table_version: None,
            discipline: Some(0),
            category: Some(category),
            second_level_type: None,
            second_level_value: None,
            process_identity: None,
            time_semantics: vec![0],
            coordinate_values: Vec::new(),
            values: ArrayD::from_elem(ndarray::IxDyn(&[2, 3]), level),
            latitude: vec![10.0, 11.0],
            longitude: vec![20.0, 21.0, 22.0],
            grid_fingerprint: "grid".to_owned(),
            plane_ladder_hpa: None,
        }
    }

    /// Temperature records on `temperature` levels, wind on `wind`.
    fn pressure_records(temperature: &[f64], wind: &[f64]) -> Vec<GribRecord> {
        let mut records = Vec::new();
        for level in temperature {
            records.push(pressure_record(records.len(), 0, 0, *level));
        }
        for level in wind {
            records.push(pressure_record(records.len(), 2, 2, *level));
        }
        records
    }


    #[test]
    fn mixed_grids_require_explicit_exact_coarsest_subset() {
        let mapping = pressure_mapping("");
        let mut records = pressure_records(&[1000.0,5000.0,10000.0], &[1000.0,5000.0,10000.0]);
        for record in records.iter_mut().filter(|r| r.level_value != 1000.0) {
            record.latitude = vec![10.0,10.5,11.0];
            record.longitude = vec![20.0,20.5,21.0,21.5,22.0];
            record.values = ndarray::Array::from_shape_fn((3,5), |(y,x)| (10*y+x) as f64).into_dyn();
            record.grid_fingerprint = "fine".into();
        }
        assert!(assemble_grib(&mapping, &records).is_err());
        let mut doc = mapping.doc.to_value();
        doc["coordinates"]["horizontal"] = serde_json::json!({"normalization":"coarsest_exact_subset"});
        let mut mapping = mapping;
        mapping.doc = crate::node::Node::parse(&serde_json::to_vec(&doc).unwrap()).unwrap();
        let assembled = assemble_grib(&mapping, &records).unwrap();
        assert_eq!(assembled.latitude, vec![10.0,11.0]);
        assert_eq!(assembled.longitude, vec![20.0,21.0,22.0]);
        let temperature = assembled.direct.iter().find(|((_,_,name),_)| name=="air_temperature").unwrap().1;
        assert_eq!(temperature.values[[1,1,2]],24.0);
        records[1].longitude[4] = 22.1;
        assert!(assemble_grib(&mapping, &records).is_err());
    }

    #[test]
    fn a_surface_contribution_has_no_invented_vertical_ladder() {
        let mut mapping = pressure_mapping("");
        let mut doc = mapping.doc.to_value();
        doc["coordinates"]["vertical"]["levels"] = serde_json::json!([]);
        let mut field=doc["fields"]["air_temperature"].clone();
        field["source_axes"]=serde_json::json!(["y","x"]);
        field["target_axes"]=serde_json::json!(["y","x"]);
        doc["fields"]=serde_json::json!({"land_fraction":field});
        mapping.doc = crate::node::Node::parse(&serde_json::to_vec(&doc).unwrap()).unwrap();
        let records=vec![pressure_record(0,0,0,0.0)];
        let assembled=assemble_grib(&mapping,&records).unwrap();
        assert!(assembled.vertical_values.is_empty());
        assert_eq!(assembled.direct.len(),1);
    }

    const ERA: &str = r#", "era_ladders": [[5000, 10000]]"#;

    #[test]
    fn a_publication_without_the_top_level_decodes_on_its_era_ladder() {
        let mapping = pressure_mapping(ERA);
        let records = pressure_records(&[5000.0, 10000.0], &[5000.0, 10000.0]);
        let collection = assemble_grib(&mapping, &records).unwrap();
        assert_eq!(collection.vertical_values, vec![5000.0, 10000.0]);
        let temperature = collection
            .direct
            .values()
            .find(|value| value.name == "air_temperature")
            .unwrap();
        assert_eq!(temperature.values.shape(), &[2, 2, 3]);
        assert_eq!(temperature.values[ndarray::IxDyn(&[0, 0, 0])], 5000.0);
        assert_eq!(temperature.values[ndarray::IxDyn(&[1, 0, 0])], 10000.0);
    }

    #[test]
    fn a_publication_with_every_declared_level_keeps_the_declared_ladder() {
        let mapping = pressure_mapping(ERA);
        let all = [1000.0, 5000.0, 10000.0];
        let collection = assemble_grib(&mapping, &pressure_records(&all, &all)).unwrap();
        assert_eq!(collection.vertical_values, all.to_vec());
    }

    #[test]
    fn a_level_only_some_fields_carry_is_left_out_of_every_field() {
        let mapping = pressure_mapping(ERA);
        let records = pressure_records(&[1000.0, 5000.0, 10000.0], &[5000.0, 10000.0]);
        let collection = assemble_grib(&mapping, &records).unwrap();
        assert_eq!(collection.vertical_values, vec![5000.0, 10000.0]);
        let temperature = collection
            .direct
            .values()
            .find(|value| value.name == "air_temperature")
            .unwrap();
        assert_eq!(temperature.references, vec!["<test>:1", "<test>:2"]);
    }

    #[test]
    fn a_level_no_declared_ladder_omits_still_refuses_by_name() {
        let mapping = pressure_mapping(ERA);
        let records = pressure_records(&[1000.0, 10000.0], &[1000.0, 10000.0]);
        let refusal = assemble_grib(&mapping, &records).unwrap_err();
        assert_eq!(
            refusal.message,
            "air_temperature vertical coverage mismatch; missing=[5000.0], extra=[]"
        );
    }

    #[test]
    fn without_era_ladders_the_absent_level_refuses_as_it_always_did() {
        let mapping = pressure_mapping("");
        let records = pressure_records(&[5000.0, 10000.0], &[5000.0, 10000.0]);
        let refusal = assemble_grib(&mapping, &records).unwrap_err();
        assert_eq!(
            refusal.message,
            "air_temperature vertical coverage mismatch; missing=[1000.0], extra=[]"
        );
    }

    #[test]
    fn the_largest_era_ladder_carried_in_full_wins() {
        let declared = [1000.0, 5000.0, 10000.0, 20000.0];
        let eras = vec![vec![10000.0, 20000.0], vec![5000.0, 10000.0, 20000.0]];
        let carried: Vec<BTreeSet<u64>> = vec![
            [5000.0f64, 10000.0, 20000.0]
                .iter()
                .map(|level| level.to_bits())
                .collect(),
        ];
        assert_eq!(
            choose_ladder(&declared, &eras, &carried),
            vec![5000.0, 10000.0, 20000.0]
        );
    }

    /// Temperature on the 1000/5000/10000 Pa ladder plus cloud water
    /// (0.1.22) declared published on 5000 and 10000 Pa only.
    fn partly_published_mapping(era_ladders: &str) -> Mapping {
        let text = format!(
            r#"{{"schema": "rw-wps.mapping.v1", "name": "t", "format": "grib2",
                "coordinates": {{"vertical": {{"kind": "pressure", "units": "Pa",
                    "levels": [1000, 5000, 10000]{era_ladders}}}}},
                "fields": {{
                  "air_temperature": {{"selectors": [{{"format": "grib2",
                     "discipline": 0, "category": 0, "parameter": 0,
                     "level_type": 100}}],
                   "units": {{"source": "K", "target": "K"}},
                   "source_axes": ["vertical", "y", "x"],
                   "target_axes": ["vertical", "y", "x"],
                   "location": "mass", "staggering": "none",
                   "missing": {{"kind": "reject"}}}},
                  "cloud_water_mixing_ratio": {{"selectors": [{{"format": "grib2",
                     "discipline": 0, "category": 1, "parameter": 22,
                     "level_type": 100}}],
                   "units": {{"source": "kg kg-1", "target": "kg kg-1"}},
                   "source_axes": ["vertical", "y", "x"],
                   "target_axes": ["vertical", "y", "x"],
                   "location": "mass", "staggering": "none",
                   "missing": {{"kind": "reject"}},
                   "published_levels": {{"levels": [5000, 10000], "absent": "zero"}}}}}}}}"#
        );
        let bytes = text.as_bytes().to_vec();
        Mapping {
            sha256: crate::digest::bytes_sha256(&bytes),
            doc: crate::node::Node::parse(&bytes).unwrap(),
            path: "<test>".to_owned(),
        }
    }

    fn partly_published_records(temperature: &[f64], cloud: &[f64]) -> Vec<GribRecord> {
        let mut records = Vec::new();
        for level in temperature {
            records.push(pressure_record(records.len(), 0, 0, *level));
        }
        for level in cloud {
            records.push(pressure_record(records.len(), 1, 22, *level));
        }
        records
    }

    fn field_named<'c>(collection: &'c DecodedCollection, name: &str) -> &'c DirectValue {
        collection
            .direct
            .values()
            .find(|value| value.name == name)
            .unwrap()
    }

    #[test]
    fn upper_humidity_boundary_extends_only_above_the_published_column() {
        let mut mapping = partly_published_mapping("");
        let text = serde_json::to_string(&mapping.doc.to_value()).unwrap()
            .replace("cloud_water_mixing_ratio", "specific_humidity")
            .replace("\"zero\"", "\"nearest_upper_boundary\"");
        mapping.doc = crate::node::Node::parse(text.as_bytes()).unwrap();
        let all = [1000.0, 5000.0, 10000.0];
        let records = partly_published_records(&all, &[5000.0, 10000.0]);
        let collection = assemble_grib(&mapping, &records).unwrap();
        let humidity = field_named(&collection, "specific_humidity");
        assert!(humidity.values.index_axis(ndarray::Axis(0), 0).iter().all(|v| *v == 5000.0));
        assert_eq!(humidity.values[ndarray::IxDyn(&[2, 0, 0])], 10000.0);
        assert!(assemble_grib(&mapping, &partly_published_records(&all, &[10000.0])).is_err());
        let bad = text.replace("[5000,10000]", "[1000,10000]");
        mapping.doc = crate::node::Node::parse(bad.as_bytes()).unwrap();
        assert!(assemble_grib(&mapping, &partly_published_records(&all, &[1000.0, 10000.0])).is_err());
    }

    #[test]
    fn a_field_published_on_part_of_the_ladder_is_zero_on_the_rest() {
        let mapping = partly_published_mapping("");
        let all = [1000.0, 5000.0, 10000.0];
        let records = partly_published_records(&all, &[5000.0, 10000.0]);
        let collection = assemble_grib(&mapping, &records).unwrap();
        // The other fields keep the whole declared ladder.
        assert_eq!(collection.vertical_values, all.to_vec());
        assert_eq!(field_named(&collection, "air_temperature").values.shape(), &[3, 2, 3]);
        let cloud = field_named(&collection, "cloud_water_mixing_ratio");
        assert_eq!(cloud.values.shape(), &[3, 2, 3]);
        assert!(cloud.values.index_axis(ndarray::Axis(0), 0).iter().all(|value| *value == 0.0));
        assert_eq!(cloud.values[ndarray::IxDyn(&[1, 0, 0])], 5000.0);
        assert_eq!(cloud.values[ndarray::IxDyn(&[2, 1, 2])], 10000.0);
        assert_eq!(cloud.missing_count, 0);
        assert_eq!(cloud.references, vec!["<test>:3", "<test>:4"]);
    }

    #[test]
    fn a_published_level_the_file_leaves_out_still_refuses_by_name() {
        let mapping = partly_published_mapping("");
        let all = [1000.0, 5000.0, 10000.0];
        let records = partly_published_records(&all, &[10000.0]);
        let refusal = assemble_grib(&mapping, &records).unwrap_err();
        assert_eq!(
            refusal.message,
            "cloud_water_mixing_ratio vertical coverage mismatch; missing=[5000.0], extra=[]"
        );
    }

    #[test]
    fn a_record_on_an_unpublished_level_is_read() {
        let mapping = partly_published_mapping("");
        let all = [1000.0, 5000.0, 10000.0];
        let collection = assemble_grib(&mapping, &partly_published_records(&all, &all)).unwrap();
        let cloud = field_named(&collection, "cloud_water_mixing_ratio");
        assert_eq!(cloud.values[ndarray::IxDyn(&[0, 0, 0])], 1000.0);
    }

    #[test]
    fn a_partly_published_field_does_not_shorten_the_era_ladder_choice() {
        // Temperature carries every declared level, so the declared
        // ladder fits; the cloud field's unpublished 1000 Pa does not
        // push the decode onto the era ladder that omits it.
        let mapping = partly_published_mapping(r#", "era_ladders": [[5000, 10000]]"#);
        let all = [1000.0, 5000.0, 10000.0];
        let collection =
            assemble_grib(&mapping, &partly_published_records(&all, &[5000.0, 10000.0])).unwrap();
        assert_eq!(collection.vertical_values, all.to_vec());
        // A publication without 1000 Pa anywhere still takes the era ladder.
        let shorter = assemble_grib(
            &mapping,
            &partly_published_records(&[5000.0, 10000.0], &[5000.0, 10000.0]),
        )
        .unwrap();
        assert_eq!(shorter.vertical_values, vec![5000.0, 10000.0]);
    }

    #[test]
    fn a_published_levels_declaration_other_than_zero_refuses() {
        let text = r#"{"schema": "rw-wps.mapping.v1", "name": "t", "format": "grib2",
            "coordinates": {"vertical": {"kind": "pressure", "units": "Pa",
                "levels": [1000, 5000]}},
            "fields": {"cloud_water_mixing_ratio": {"selectors": [{"format": "grib2",
                 "discipline": 0, "category": 1, "parameter": 22, "level_type": 100}],
               "units": {"source": "kg kg-1", "target": "kg kg-1"},
               "source_axes": ["vertical", "y", "x"], "target_axes": ["vertical", "y", "x"],
               "location": "mass", "missing": {"kind": "reject"},
               "published_levels": {"levels": [5000], "absent": "nan"}}}}"#;
        let bytes = text.as_bytes().to_vec();
        let mapping = Mapping {
            sha256: crate::digest::bytes_sha256(&bytes),
            doc: crate::node::Node::parse(&bytes).unwrap(),
            path: "<test>".to_owned(),
        };
        let records = partly_published_records(&[], &[5000.0]);
        let refusal = assemble_grib(&mapping, &records).unwrap_err();
        assert!(refusal.message.contains("published_levels.absent"), "{}", refusal.message);
    }

    #[test]
    fn pv_half_count_is_held_to_the_declared_ladder() {
        let mapping = hybrid_mapping(None);
        let long_pv: Vec<f64> = (0..12).map(f64::from).collect();
        let refusal = resolve_hybrid_coefficients(&mapping, 3, &long_pv).unwrap_err();
        assert!(refusal.message.contains('6'), "{}", refusal.message);
        assert!(refusal.message.contains('4'), "{}", refusal.message);
    }
}
