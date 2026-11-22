//! Horizontal donor alignment on the decoded geographic lattice.
//! Continuous values use bilinear weights; masks/categories use nearest.
//! Soil and water state renormalize over finite corners of the target's
//! nearest land/sea class, preserving the other class's missing mask.
use ndarray::{ArrayD, Dimension, IxDyn};
use rayon::prelude::*;
use serde_json::{json, Value};

use crate::assemble::{DecodedCollection, DirectValue};
use crate::refusal::{frame_invalid, Result};

pub const ALIGNMENT: &str = "bilinear_mask_aware_nearest";
/// The exact contiguous subset when the donor publishes every primary
/// point; otherwise [`ALIGNMENT`] with a one-donor-step edge allowance.
/// On matching grids it is byte-identical to `exact_coordinate_subset`.
pub const EXACT_OR_REGRID: &str = "exact_coordinate_subset_or_bilinear_mask_aware_nearest";
/// How far past the donor's box a primary point may lie under
/// [`EXACT_OR_REGRID`], in donor steps.
pub const EXACT_OR_REGRID_EDGE_CELLS: f64 = 1.0;

/// Largest distance from a primary coordinate to its nearest donor
/// coordinate, degrees (longitude folded modulo 360): how far apart two
/// lattices that should share their points actually sit.
pub fn lattice_offset(source: &[f64], target: &[f64], cyclic: bool) -> f64 {
    target.iter().map(|t| source.iter().map(|s| {
        let d = (s - t).abs();
        if cyclic { let d = d.rem_euclid(360.0); d.min(360.0 - d) } else { d }
    }).fold(f64::INFINITY, f64::min)).fold(0.0, f64::max)
}

#[derive(Clone, Copy)]
struct Bracket { lo: usize, hi: usize, weight: f64 }

#[cfg(test)]
fn brackets(source: &[f64], target: &[f64], cyclic: bool) -> Result<Vec<Bracket>> {
    brackets_with_edge(source, target, cyclic, 0.0).map(|(b, _)| b)
}

/// `edge_cells` > 0 lets a target lie up to that many donor steps past the
/// donor's first or last coordinate; it takes the edge cell (weight 0).
/// Returns the brackets and how many targets were clamped that way.
fn brackets_with_edge(source: &[f64], target: &[f64], cyclic: bool, edge_cells: f64) -> Result<(Vec<Bracket>, usize)> {
    if source.is_empty() || source.iter().chain(target).any(|v| !v.is_finite()) {
        return Err(frame_invalid("horizontal regrid needs finite, nonempty coordinate axes"));
    }
    let mut ordered: Vec<(f64, usize)> = source.iter().enumerate()
        .map(|(i, v)| (if cyclic { v.rem_euclid(360.0) } else { *v }, i)).collect();
    ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
    if ordered.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(frame_invalid("horizontal regrid donor axis has duplicate coordinates"));
    }
    // Cut a regional longitude axis at its largest gap. A global axis
    // closes evenly and includes the last-to-first interpolation cell.
    let mut periodic = false;
    if cyclic && ordered.len() > 1 {
        let gaps: Vec<f64> = (0..ordered.len()).map(|i|
            (ordered[(i + 1) % ordered.len()].0 - ordered[i].0).rem_euclid(360.0)).collect();
        let cut = (0..gaps.len()).max_by(|a,b| gaps[*a].total_cmp(&gaps[*b])).unwrap();
        let min_gap = gaps.iter().copied().fold(f64::INFINITY, f64::min);
        periodic = gaps[cut] < 1.5 * min_gap;
        if !periodic {
            let count = ordered.len();
            ordered.rotate_left((cut + 1) % count);
            let start = ordered[0].0;
            for (v, _) in &mut ordered { if *v < start { *v += 360.0; } }
        } else {
            ordered.push((ordered[0].0 + 360.0, ordered[0].1));
        }
    }
    let end = ordered.len() - 1;
    let step = ordered.windows(2).map(|w| w[1].0 - w[0].0).fold(f64::INFINITY, f64::min);
    let reach = if edge_cells > 0.0 && step.is_finite() { edge_cells * step * (1.0 + 1e-9) } else { 0.0 };
    let mut clamped = 0usize;
    let out = target.iter().map(|v| {
        let mut v = if cyclic { v.rem_euclid(360.0) } else { *v };
        if cyclic && v < ordered[0].0 {
            // A regional longitude just west of the donor's first column
            // stays west of it rather than wrapping to the far side.
            if !(reach > 0.0 && ordered[0].0 - v <= reach) { v += 360.0; }
        }
        if v < ordered[0].0 && ordered[0].0 - v <= reach {
            clamped += 1;
            return Ok(Bracket { lo: ordered[0].1, hi: ordered[0].1, weight: 0.0 });
        }
        if v > ordered[end].0 && v - ordered[end].0 <= reach {
            clamped += 1;
            return Ok(Bracket { lo: ordered[end].1, hi: ordered[end].1, weight: 0.0 });
        }
        if v < ordered[0].0 || v > ordered[end].0 {
            return Err(frame_invalid(format!("horizontal regrid target coordinate {v} outside donor coverage {}..{} (periodic={periodic})", ordered[0].0, ordered[end].0)));
        }
        let hi = ordered.partition_point(|p| p.0 < v).min(end);
        let lo = if hi == 0 || ordered[hi].0 == v { hi } else { hi - 1 };
        Ok(Bracket { lo: ordered[lo].1, hi: ordered[hi].1,
            weight: if lo == hi { 0.0 } else { (v - ordered[lo].0) / (ordered[hi].0 - ordered[lo].0) } })
    }).collect::<Result<Vec<_>>>()?;
    Ok((out, clamped))
}

fn categorical(name: &str) -> bool {
    name == "land_fraction" || name.ends_with("_mask") || name.ends_with("_category")
        || matches!(name, "soil_type" | "land_use" | "land_use_category" | "vegetation_type")
}

fn interpolate(value: &DirectValue, ys: &[Bracket], xs: &[Bracket], mask: Option<&DirectValue>) -> Result<ArrayD<f64>> {
    let y = value.axes.iter().position(|a| a == "y").ok_or_else(|| frame_invalid("regrid field has no y axis"))?;
    let x = value.axes.iter().position(|a| a == "x").ok_or_else(|| frame_invalid("regrid field has no x axis"))?;
    let mut shape = value.values.shape().to_vec();
    shape[y] = ys.len(); shape[x] = xs.len();
    let dims = IxDyn(&shape);
    let strides: Vec<usize> = (0..shape.len()).map(|i| shape[i+1..].iter().product()).collect();
    let restricted = match value.name.as_str() {
        "soil_temperature" | "volumetric_soil_moisture" => Some(true),
        "sea_surface_temperature" | "sea_ice_fraction" => Some(false),
        _ => None,
    };
    let mask_sensitive = restricted.is_some() || value.name == "skin_temperature";
    if mask_sensitive && mask.is_none() {
        return Err(frame_invalid(format!("horizontal regrid {} needs donor land_fraction", value.name)));
    }
    let values: Vec<f64> = (0..dims.size()).into_par_iter().map(|flat| {
        let mut index: Vec<usize> = strides.iter().zip(&shape).map(|(s,n)| (flat / s) % n).collect();
        let by = ys[index[y]]; let bx = xs[index[x]];
        let ny = if by.weight <= 0.5 { by.lo } else { by.hi };
        let nx = if bx.weight <= 0.5 { bx.lo } else { bx.hi };
        if categorical(&value.name) {
            index[y] = ny; index[x] = nx;
            return value.values[IxDyn(&index)];
        }
        let land = mask.map(|m| m.values[IxDyn(&[ny,nx])] >= 0.5);
        if restricted.is_some() && land != restricted { return f64::NAN; }
        let mut sum = 0.0; let mut weight = 0.0;
        for (iy, wy) in [(by.lo, 1.0-by.weight), (by.hi, by.weight)] {
            for (ix, wx) in [(bx.lo, 1.0-bx.weight), (bx.hi, bx.weight)] {
                let w = wy * wx;
                if w == 0.0 { continue; }
                if mask_sensitive && mask.is_some_and(|m| (m.values[IxDyn(&[iy,ix])] >= 0.5) != land.unwrap()) { continue; }
                index[y] = iy; index[x] = ix;
                let v = value.values[IxDyn(&index)];
                if v.is_finite() { sum += w*v; weight += w; }
            }
        }
        if weight > 0.0 { sum / weight } else { f64::NAN }
    }).collect();
    ArrayD::from_shape_vec(dims, values).map_err(|e| frame_invalid(e.to_string()))
}

/// Consume and replace donor arrays once, before the temporal/vertical join.
pub fn align(donor: DecodedCollection, latitude: &[f64], longitude: &[f64]) -> Result<(DecodedCollection, Value)> {
    align_with_edge(donor, latitude, longitude, 0.0)
}

/// [`align`], letting primary points up to `edge_cells` donor steps past the
/// donor's box take the donor's edge cell.  Two ERA5 files requested over
/// the same area from two CDS services can sit on lattices shifted by part
/// of a cell (era5-137 grid mismatch, 2026-11-10), so the atmosphere's
/// outer row or column can lie just outside the surface file.
pub fn align_with_edge(mut donor: DecodedCollection, latitude: &[f64], longitude: &[f64], edge_cells: f64) -> Result<(DecodedCollection, Value)> {
    let started = std::time::Instant::now();
    let source_shape = [donor.latitude.len(), donor.longitude.len()];
    let (ys, edge_rows) = brackets_with_edge(&donor.latitude, latitude, false, edge_cells)?;
    let (xs, edge_columns) = brackets_with_edge(&donor.longitude, longitude, true, edge_cells)?;
    let results: Vec<Result<_>> = crate::threads::install(|| donor.direct.par_iter().map(|(key, value)| {
        let mask = donor.direct.get(&(key.0, key.1.clone(), "land_fraction".to_owned()));
        if let Some(m) = mask
            && (m.axes != ["y", "x"] || m.values.shape() != source_shape || m.values.iter().any(|v| !v.is_finite())) {
                return Err(frame_invalid("horizontal regrid land_fraction must be finite y,x on the donor grid"));
            }
        let values = interpolate(value, &ys, &xs, mask)?;
        let output = DirectValue { name: value.name.clone(), valid_time: value.valid_time,
            source_cycle: value.source_cycle, member: value.member.clone(), axes: value.axes.clone(),
            missing_count: values.iter().filter(|v| v.is_nan()).count(), values,
            references: value.references.clone() };
        Ok((key.clone(), output))
    }).collect());
    donor.direct = crate::threads::in_order(results)?.into_iter().collect();
    donor.latitude = latitude.to_vec(); donor.longitude = longitude.to_vec();
    let mut receipt = json!({"method": ALIGNMENT, "continuous": "bilinear_finite_weight_renormalization",
        "masks_categories": "nearest", "surface_mask": "nearest_land_fraction_threshold_0.5",
        "source_shape": source_shape, "target_shape": [latitude.len(),longitude.len()],
        "seconds": started.elapsed().as_secs_f64(), "threads": crate::threads::width(),
        "fields": donor.direct.len()});
    if edge_cells > 0.0 {
        receipt["edge_cells_allowed"] = json!(edge_cells);
        receipt["edge_clamped_rows"] = json!(edge_rows);
        receipt["edge_clamped_columns"] = json!(edge_columns);
    }
    Ok((donor, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn analytic_bilinear_between_grids() {
        let y = [3.0, 1.0, -1.0]; let x = [358.0, 0.0, 2.0];
        let ty = [2.5, 0.5, -0.5]; let tx = [359.0, 0.5, 1.5];
        let time = chrono::NaiveDate::from_ymd_opt(2024,1,1).unwrap().and_hms_opt(0,0,0).unwrap();
        let value = DirectValue { name: "air_temperature".into(), valid_time: time,
            source_cycle: time, member: None, axes: vec!["y".into(),"x".into()],
            values: ArrayD::from_shape_vec(IxDyn(&[3,3]), y.iter().flat_map(|a| x.iter().map(move |b| {
                let b = if *b > 180.0 { *b - 360.0 } else { *b };
                270.0 + 2.0*a + 3.0*b + 0.25*a*b
            })).collect()).unwrap(), missing_count: 0, references: vec![] };
        let out = crate::threads::install(|| interpolate(&value, &brackets(&y,&ty,false).unwrap(), &brackets(&x,&tx,true).unwrap(), None)).unwrap();
        for (j,a) in ty.iter().enumerate() { for (i,b) in tx.iter().enumerate() {
            let b = if *b > 180.0 { *b - 360.0 } else { *b };
            assert!((out[IxDyn(&[j,i])] - (270.0+2.0*a+3.0*b+0.25*a*b)).abs() < 1e-10);
        }}
    }

    #[test]
    fn shifted_lattice_edge_takes_edge_cell_within_one_step_only() {
        // Surface on the 0.25 lattice; atmosphere anchored at 52.1/-100.3.
        let donor_lat = [52.0, 51.75, 51.5]; let donor_lon = [259.75, 260.0, 260.25];
        let target_lat = [52.1, 51.85, 51.6]; let target_lon = [-100.3, -100.05, -99.8];
        let (ys, rows) = brackets_with_edge(&donor_lat, &target_lat, false, 1.0).unwrap();
        let (xs, cols) = brackets_with_edge(&donor_lon, &target_lon, true, 1.0).unwrap();
        assert_eq!((rows, cols), (1, 1));
        assert_eq!((ys[0].lo, ys[0].hi, ys[0].weight), (0, 0, 0.0));
        assert_eq!((xs[0].lo, xs[0].hi, xs[0].weight), (0, 0, 0.0));
        assert!((ys[1].weight - 0.4).abs() < 1e-9 && (xs[1].weight - 0.8).abs() < 1e-9);
        // Without the allowance the same edge still refuses, and anything
        // past one step refuses even with it.
        assert!(brackets(&donor_lat, &target_lat, false).is_err());
        assert!(brackets_with_edge(&donor_lat, &[52.3], false, 1.0).is_err());
        assert!(brackets_with_edge(&donor_lon, &[-100.6], true, 1.0).is_err());
    }
}
