//! GRIB records, decoded in process.
//!
//! Port of `mapped_source._grib2_inventory` / `_grib2_records` /
//! `_regular_latlon_frame` / `_selector_matches_record`, with the
//! subprocess and its TSV removed.  The numbers are unchanged: the Python
//! engine's TSV carried `grib-core`'s own struct fields through Rust's
//! shortest-round-trip `Display` and Python's exact `float()`, so reading
//! the same fields directly yields the same f64 bit patterns.  The one
//! place the TSV spelling still matters is the grid fingerprint, which
//! hashes the RENDERED strings; `grid_fingerprint` below reproduces that
//! rendering exactly, including the `0x40` scan-mode spelling.

use std::collections::BTreeMap;

use chrono::{Duration, NaiveDateTime};
use grib_core::grib2::{unpack_message, Grib2File, Grib2Message};
use ndarray::{ArrayD, IxDyn};

use crate::model::{FieldSpec, GridDeclaration, Mapping, GRIB2_GRID_RELATIVE_WIND_BIT};
use crate::refusal::{decode_failed, grid_mismatch, Result};

/// One decoded GRIB record: `mapped_source._GribRecord`.
#[derive(Debug, Clone)]
pub struct GribRecord {
    /// The path provenance is bound to: the object acquisition delivered,
    /// NOT the staged decompressed twin the parser read.
    pub source: String,
    pub index: usize,
    pub reference_time: NaiveDateTime,
    pub valid_time: NaiveDateTime,
    pub member: Option<String>,
    pub parameter: i64,
    pub level_type: i64,
    pub level_value: f64,
    pub table_version: Option<i64>,
    pub center: Option<i64>,
    pub subcenter: Option<i64>,
    pub master_table_version: Option<i64>,
    pub local_table_version: Option<i64>,
    pub discipline: Option<i64>,
    pub category: Option<i64>,
    pub second_level_type: Option<i64>,
    pub second_level_value: Option<f64>,
    pub process_identity: Option<(i64, i64)>,
    pub time_semantics: Vec<i64>,
    /// Section 4's optional coordinate list (the pv octets): the
    /// half-level A (Pa) then B hybrid coefficients on model-level
    /// records.  Empty when the message carries none.
    pub coordinate_values: Vec<f64>,
    pub values: ArrayD<f64>,
    pub latitude: Vec<f64>,
    pub longitude: Vec<f64>,
    pub grid_fingerprint: String,
    /// The record's full-plane pressure-ladder value (median / 100,
    /// `window::pressure_levels`) when the record was decoded over an
    /// atmospheric window but answers `air_pressure`, whose whole plane
    /// states the frame's per-level pressure; `None` otherwise.
    pub plane_ladder_hpa: Option<f64>,
}

/// The selector-visible identity of a record, without its bytes: the
/// lightweight record `_grib2_wanted_indices` builds from an inventory row.
#[derive(Debug, Clone)]
pub struct RecordIdentity {
    pub index: usize,
    pub member: Option<String>,
    pub parameter: i64,
    pub level_type: i64,
    pub level_value: f64,
    pub table_version: Option<i64>,
    pub center: Option<i64>,
    pub subcenter: Option<i64>,
    pub master_table_version: Option<i64>,
    pub local_table_version: Option<i64>,
    pub discipline: Option<i64>,
    pub category: Option<i64>,
    pub second_level_type: Option<i64>,
    pub second_level_value: Option<f64>,
    pub time_semantics: Vec<i64>,
}

impl GribRecord {
    pub fn identity(&self) -> RecordIdentity {
        RecordIdentity {
            index: self.index,
            member: self.member.clone(),
            parameter: self.parameter,
            level_type: self.level_type,
            level_value: self.level_value,
            table_version: self.table_version,
            center: self.center,
            subcenter: self.subcenter,
            master_table_version: self.master_table_version,
            local_table_version: self.local_table_version,
            discipline: self.discipline,
            category: self.category,
            second_level_type: self.second_level_type,
            second_level_value: self.second_level_value,
            time_semantics: self.time_semantics.clone(),
        }
    }
}

fn selector_int(selector: &crate::node::Node, key: &str) -> Option<i64> {
    selector.field(key).and_then(crate::node::Node::as_i64)
}

fn selector_float(selector: &crate::node::Node, key: &str) -> Option<f64> {
    selector.field(key).and_then(crate::node::Node::as_f64)
}

fn close(left: f64, right: f64) -> bool {
    // `math.isclose(..., abs_tol=1e-9)` with the default rel_tol of 1e-9.
    let difference = (left - right).abs();
    difference <= 1e-9 || difference <= 1e-9 * left.abs().max(right.abs())
}

/// `mapped_source._selector_matches_record`.
pub fn selector_matches(
    selector: &crate::node::Node,
    record: &RecordIdentity,
    source_format: &str,
) -> bool {
    if source_format == "grib1" {
        let Some(parameter) = selector_int(selector, "parameter") else {
            return false;
        };
        return record.parameter == parameter
            && selector_int(selector, "table_version")
                .is_none_or(|value| record.table_version == Some(value))
            && selector_int(selector, "center").is_none_or(|value| record.center == Some(value))
            && selector_int(selector, "level_type")
                .is_none_or(|value| record.level_type == value)
            && selector_float(selector, "level_value")
                .is_none_or(|value| close(record.level_value, value));
    }
    let (Some(discipline), Some(category), Some(parameter)) = (
        selector_int(selector, "discipline"),
        selector_int(selector, "category"),
        selector_int(selector, "parameter"),
    ) else {
        return false;
    };
    let second_surface = {
        let declared_type = selector_int(selector, "second_level_type");
        match declared_type {
            None => matches!(record.second_level_type, None | Some(255)),
            Some(kind) => {
                record.second_level_value.is_some()
                    && record.second_level_type == Some(kind)
                    && selector_float(selector, "second_level_value").is_some_and(|declared| {
                        close(record.second_level_value.unwrap_or(f64::NAN), declared)
                    })
            }
        }
    };
    record.discipline == Some(discipline)
        && record.category == Some(category)
        && record.parameter == parameter
        && selector_int(selector, "center").is_none_or(|value| record.center == Some(value))
        && selector_int(selector, "subcenter").is_none_or(|value| record.subcenter == Some(value))
        && selector_int(selector, "master_table_version")
            .is_none_or(|value| record.master_table_version == Some(value))
        && selector_int(selector, "local_table_version")
            .is_none_or(|value| record.local_table_version == Some(value))
        && selector_int(selector, "level_type").is_none_or(|value| record.level_type == value)
        && selector_float(selector, "level_value")
            .is_none_or(|value| close(record.level_value, value))
        && second_surface
        && selector
            .field("member")
            .and_then(crate::node::Node::as_str)
            .is_none_or(|value| record.member.as_deref() == Some(value))
        && selector_int(selector, "pdt")
            .is_none_or(|value| record.time_semantics.first() == Some(&value))
}

/// `mapped_source._declared_vertical_admits`.
pub fn declared_vertical_admits(
    declared_levels: &[f64],
    interface_levels: &[f64],
    field: &FieldSpec<'_>,
    level_value: f64,
) -> Result<bool> {
    if !field.source_axes()?.iter().any(|axis| axis == "vertical" || axis == "half_level") {
        return Ok(true);
    }
    let declared_levels = if field.source_axes()?.iter().any(|axis| axis == "half_level") {
        interface_levels
    } else {
        declared_levels
    };
    if declared_levels.is_empty() {
        return Ok(true);
    }
    Ok(declared_levels
        .iter()
        .any(|level| (level_value - level).abs() <= 1e-6))
}

/// `mapped_source._embedded_valid_time` for GRIB2 (edition 2 factors).
pub fn embedded_valid_time(
    reference: NaiveDateTime,
    unit: u8,
    amount: i64,
    edition: u8,
) -> Result<NaiveDateTime> {
    let factor = if edition == 1 {
        match unit {
            0 => Duration::minutes(1),
            1 => Duration::hours(1),
            2 => Duration::days(1),
            10 => Duration::hours(3),
            11 => Duration::hours(6),
            12 => Duration::hours(12),
            254 => Duration::seconds(1),
            _ => {
                return Err(decode_failed(format!(
                    "unsupported GRIB1 forecast time unit {unit}"
                )))
            }
        }
    } else {
        match unit {
            0 => Duration::minutes(1),
            1 => Duration::hours(1),
            2 => Duration::days(1),
            10 => Duration::hours(3),
            11 => Duration::hours(6),
            12 => Duration::hours(12),
            13 => Duration::seconds(1),
            _ => {
                return Err(decode_failed(format!(
                    "unsupported GRIB2 forecast time unit {unit}"
                )))
            }
        }
    };
    Ok(reference + factor * amount as i32)
}

/// The grid fingerprint the Python engine hashes: a compact JSON object of
/// the TSV-RENDERED grid columns, keys sorted.
pub fn grid_fingerprint(message: &Grib2Message) -> String {
    let grid = &message.grid;
    let mut payload: BTreeMap<&str, String> = BTreeMap::new();
    payload.insert("gdt", grid.template.to_string());
    payload.insert("nx", grid.nx.to_string());
    payload.insert("ny", grid.ny.to_string());
    payload.insert("lat1", grid.lat1.to_string());
    payload.insert("lon1", grid.lon1.to_string());
    payload.insert("dx", grid.dx.to_string());
    payload.insert("dy", grid.dy.to_string());
    payload.insert("latin1", grid.latin1.to_string());
    payload.insert("latin2", grid.latin2.to_string());
    payload.insert("lov", grid.lov.to_string());
    payload.insert("scan_mode", format!("0x{:02x}", grid.scan_mode));
    payload.insert("shape_of_earth", grid.shape_of_earth.to_string());
    payload.insert(
        "resolution_flags",
        format!("0x{:02x}", grid.resolution_flags),
    );
    let mut text = String::from("{");
    for (position, (key, value)) in payload.iter().enumerate() {
        if position > 0 {
            text.push(',');
        }
        text.push_str(&serde_json::Value::String((*key).to_owned()).to_string());
        text.push(':');
        text.push_str(&serde_json::Value::String(value.clone()).to_string());
    }
    text.push('}');
    crate::digest::bytes_sha256(text.as_bytes())
}

/// The canonical longitude axis of a regular GDT-0 grid and, for each
/// canonical column, the stored column it reads: the x-axis half of
/// [`regular_latlon_frame`] (`mapped_source._regular_latlon_frame`).
fn regular_longitude_axis(lon1: f64, dx: f64, nx: usize) -> (Vec<f64>, Vec<usize>) {
    let mut wrapped: Vec<f64> = (0..nx)
        .map(|column| {
            let raw_longitude = lon1 + column as f64 * dx;
            (raw_longitude + 180.0).rem_euclid(360.0) - 180.0
        })
        .collect();
    // `mapped_source._regular_latlon_frame`: a regional crop across the
    // antimeridian (the GDT-101 remap writes one whenever the domain
    // crosses it) is continuous in its own column order.  Wrapping and
    // sorting it put a jump of nearly a full turn inside the axis and every
    // such domain was refused as "not a regular axis"; the columns past the
    // seam are carried one turn on instead.  A crop clear of the seam has
    // no descending step and is untouched.  A grid within half a cell of
    // the full circle is a whole ring whose spacing was rounded to the
    // octets' micro-degrees (a 1/12-degree ring stores 4320 x 0.083333),
    // and it keeps the -180 origin it always had.
    if (nx as f64 + 0.5) * dx.abs() < 360.0 {
        let seams: Vec<usize> = (1..nx).filter(|k| wrapped[*k] < wrapped[*k - 1]).collect();
        if seams.len() == 1 {
            for value in wrapped.iter_mut().skip(seams[0]) {
                *value += 360.0;
            }
        }
    }
    // `np.argsort` is a STABLE sort for the default kind on ties; equal
    // longitudes keep their original order, which matters on a grid whose
    // first and last column alias after the wrap.
    let mut order: Vec<usize> = (0..nx).collect();
    order.sort_by(|left, right| {
        wrapped[*left]
            .total_cmp(&wrapped[*right])
            .then(left.cmp(right))
    });
    let longitude: Vec<f64> = order.iter().map(|index| wrapped[*index]).collect();
    (longitude, order)
}

/// The canonical axes of a regular GDT-0 record, read from Section 3
/// alone: ascending latitude, the canonical longitude axis, the stored
/// column each canonical column reads, and whether stored rows run
/// north to south (scan 0x00) and are reversed into canonical order.
pub struct RegularAxes {
    pub latitude: Vec<f64>,
    pub longitude: Vec<f64>,
    pub order: Vec<usize>,
    pub reversed_rows: bool,
}

pub fn regular_latlon_axes(message: &Grib2Message) -> Result<RegularAxes> {
    let grid = &message.grid;
    let scan = grid.scan_mode;
    if grid.template != 0 || !matches!(scan, 0x40 | 0x00) || grid.is_reduced {
        return Err(grid_mismatch(format!(
            "a canonical window needs a regular latitude/longitude GDT 0 grid \
             with scan mode 0x40 or 0x00; got GDT {} scan mode 0x{scan:02x}",
            grid.template
        )));
    }
    let ny = grid.ny as usize;
    let mut latitude: Vec<f64> = (0..ny)
        .map(|row| {
            if scan == 0x40 {
                grid.lat1 + row as f64 * grid.dy
            } else {
                grid.lat1 - row as f64 * grid.dy
            }
        })
        .collect();
    if scan == 0x00 {
        latitude.reverse();
    }
    let (longitude, order) = regular_longitude_axis(grid.lon1, grid.dx, grid.nx as usize);
    Ok(RegularAxes { latitude, longitude, order, reversed_rows: scan == 0x00 })
}

/// One record decoded over canonical rows `rows` and columns `columns`
/// only, row-major in canonical order: exactly the crop of
/// [`regular_latlon_frame`]'s values, without decoding the rest of a
/// JPEG2000 codestream (`grib_core::grib2::unpack_message_stored_rect`).
pub fn regular_latlon_window(
    message: &Grib2Message,
    axes: &RegularAxes,
    rows: [usize; 2],
    columns: [usize; 2],
) -> std::result::Result<Vec<f64>, grib_core::GribError> {
    let ny = axes.latitude.len();
    let nx = axes.longitude.len();
    let stored_rows = if axes.reversed_rows { (ny - rows[1], ny - rows[0]) } else { (rows[0], rows[1]) };
    let stored = &axes.order[columns[0]..columns[1]];
    let contiguous = stored.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let stored_columns = if contiguous { (stored[0], stored[stored.len() - 1] + 1) } else { (0, nx) };
    let rect = grib_core::grib2::StoredRect { rows: stored_rows, columns: stored_columns };
    let decoded = decode_rect_shared(message, rect)?;
    let width = stored_columns.1 - stored_columns.0;
    let height = stored_rows.1 - stored_rows.0;
    let mut values = Vec::with_capacity(height * stored.len());
    for canonical in 0..height {
        let row = if axes.reversed_rows { height - 1 - canonical } else { canonical };
        let line = &decoded[row * width..(row + 1) * width];
        if contiguous {
            values.extend_from_slice(line);
        } else {
            values.extend(stored.iter().map(|column| line[*column]));
        }
    }
    Ok(values)
}

/// `mapped_source._regular_latlon_frame`: one canonical ascending-latitude
/// frame from a regular GDT-0 record.
pub fn regular_latlon_frame(
    message: &Grib2Message,
    raw: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let grid = &message.grid;
    let scan = grid.scan_mode;
    if grid.template != 0 || !matches!(scan, 0x40 | 0x00) {
        return Err(grid_mismatch(format!(
            "generic GRIB2 frame export requires regular latitude/longitude \
             GDT 0 with scan mode 0x40 (rows south-to-north) or 0x00 (rows \
             north-to-south, normalized at decode), or a mapping.grid \
             declaration for a supported projected family; got GDT {} scan \
             mode 0x{scan:02x}",
            grid.template
        )));
    }
    let nx = grid.nx as usize;
    let ny = grid.ny as usize;
    let mut latitude: Vec<f64> = (0..ny)
        .map(|row| {
            if scan == 0x40 {
                grid.lat1 + row as f64 * grid.dy
            } else {
                grid.lat1 - row as f64 * grid.dy
            }
        })
        .collect();
    let (longitude, order) = regular_longitude_axis(grid.lon1, grid.dx, nx);
    let mut values = vec![0.0f64; nx * ny];
    // A regional row is normally one contiguous run; a global ring is
    // two runs at the longitude cut. Copy those slices directly, folding
    // latitude reversal into the same pass instead of allocating and
    // copying a second full field. Preserve the stable column ordering
    // for unusual grids (including duplicate seam columns).
    if nx > 0 {
        let start = order[0];
        let cyclic = order
            .iter()
            .enumerate()
            .all(|(column, stored)| *stored == (start + column) % nx);
        let copy_row = |(row, destination): (usize, &mut [f64])| {
            let stored_row = if scan == 0x00 { ny - 1 - row } else { row };
            let source = &raw[stored_row * nx..(stored_row + 1) * nx];
            if cyclic {
                let tail = nx - start;
                destination[..tail].copy_from_slice(&source[start..]);
                destination[tail..].copy_from_slice(&source[..start]);
            } else {
                for (value, column) in destination.iter_mut().zip(&order) {
                    *value = source[*column];
                }
            }
        };
        // Message decode already spans the pool. Keep its row copies on
        // that worker: nested dispatch made isolated complex-packed
        // records slower by waking idle workers between serial unpackers.
        // A caller outside the pool can instead split a large frame by row.
        if ny < 64 || rayon::current_thread_index().is_some() {
            values.chunks_mut(nx).enumerate().for_each(copy_row);
        } else {
            crate::threads::install(|| {
                use rayon::prelude::*;
                values
                    .par_chunks_mut(nx)
                    .with_min_len(32)
                    .enumerate()
                    .for_each(copy_row);
            });
        }
    }
    if scan == 0x00 {
        latitude.reverse();
    }
    Ok((latitude, longitude, values))
}

/// `grib2_inventory::validate_envelopes`: the envelope hygiene the
/// subprocess route ran before any parse.
pub fn validate_grib2_envelopes(bytes: &[u8], label: &str) -> Result<usize> {
    if bytes.is_empty() {
        return Err(decode_failed(format!("GRIB2 input {label} is empty")));
    }
    let mut offset = 0usize;
    let mut count = 0usize;
    while offset < bytes.len() {
        if bytes.len() - offset < 20 {
            return Err(decode_failed(format!(
                "truncated GRIB2 envelope {count} at byte {offset} of {label}: \
                 fewer than 20 bytes"
            )));
        }
        if &bytes[offset..offset + 4] != b"GRIB" {
            return Err(decode_failed(format!(
                "invalid GRIB2 envelope {count} at byte {offset} of {label}: \
                 missing GRIB marker"
            )));
        }
        let edition = bytes[offset + 7];
        if edition != 2 {
            return Err(decode_failed(format!(
                "envelope {count} at byte {offset} of {label} is GRIB edition \
                 {edition}, expected 2"
            )));
        }
        let declared = u64::from_be_bytes(
            bytes[offset + 8..offset + 16]
                .try_into()
                .expect("validated eight-byte slice"),
        ) as usize;
        if declared < 20 {
            return Err(decode_failed(format!(
                "invalid GRIB2 envelope {count} at byte {offset} of {label}: \
                 length {declared} is too short"
            )));
        }
        let Some(end) = offset.checked_add(declared).filter(|end| *end <= bytes.len()) else {
            return Err(decode_failed(format!(
                "truncated GRIB2 envelope {count} at byte {offset} of {label}: \
                 declared length {declared} runs past the file's {} bytes",
                bytes.len()
            )));
        };
        if &bytes[end - 4..end] != b"7777" {
            return Err(decode_failed(format!(
                "invalid GRIB2 envelope {count} at byte {offset} of {label}: \
                 missing 7777 terminator"
            )));
        }
        offset = end;
        count += 1;
    }
    Ok(count)
}

/// The identity of every message in one GRIB2 object, without unpacking.
pub fn grib2_identities(messages: &[Grib2Message]) -> Vec<RecordIdentity> {
    messages
        .iter()
        .enumerate()
        .map(|(index, message)| RecordIdentity {
            index,
            member: message
                .product
                .perturbation_number
                .map(|value| value.to_string()),
            parameter: message.product.parameter_number as i64,
            level_type: message.product.level_type as i64,
            level_value: message.product.level_value,
            table_version: None,
            center: Some(message.identification.center_id as i64),
            subcenter: Some(message.identification.subcenter_id as i64),
            master_table_version: Some(message.identification.master_table_version as i64),
            local_table_version: Some(message.identification.local_table_version as i64),
            discipline: Some(message.discipline as i64),
            category: Some(message.product.parameter_category as i64),
            second_level_type: Some(message.product.second_level_type as i64),
            second_level_value: Some(message.product.second_level_value),
            time_semantics: vec![message.product.template as i64],
        })
        .collect()
}

/// `mapped_source._grib2_wanted_indices`.
pub fn wanted_indices(mapping: &Mapping, identities: &[RecordIdentity]) -> Result<Vec<usize>> {
    let declared_levels = mapping.declared_levels()?;
    let interface_levels = mapping.interface_levels()?;
    let fields = mapping.fields()?;
    let mut wanted = Vec::new();
    for identity in identities {
        for field in &fields {
            let matched = field
                .selectors()
                .iter()
                .any(|selector| selector_matches(selector, identity, "grib2"));
            if matched && declared_vertical_admits(&declared_levels, &interface_levels, field, identity.level_value)? {
                wanted.push(identity.index);
                break;
            }
        }
    }
    Ok(wanted)
}

/// One `mapping.record_aliases` entry (`mapped_source._record_aliases`).
///
/// A publisher can change how it spells a record between releases of one
/// product while the record itself stays the same: ECMWF's open data
/// wrote its four soil layers as depth-below-land layers on WMO and
/// local parameters before it moved them to its own ordinal soil levels.
/// `record` names the earlier spelling by the keys it declares (a key it
/// leaves out is not compared) and `reads_as` the spelling the mapping's
/// selectors read, so every later check -- the selectors, the soil
/// layer contract, the duplicate refusals -- sees one spelling.
#[derive(Debug, Clone)]
pub struct RecordAlias {
    pub record: crate::node::Node,
    pub reads_as: crate::node::Node,
}

/// `mapping.record_aliases`, in declared order; empty when there are none.
pub fn record_aliases(mapping: &Mapping) -> Result<Vec<RecordAlias>> {
    let Some(raw) = mapping.doc.get("record_aliases") else {
        return Ok(Vec::new());
    };
    if !raw.is_array() {
        return Err(crate::refusal::mapping_invalid(
            "mapping.record_aliases must be a list",
        ));
    }
    raw.items()
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let part = |key: &str| {
                entry
                    .get(key)
                    .filter(|node| node.is_object())
                    .cloned()
                    .ok_or_else(|| {
                        crate::refusal::mapping_invalid(format!(
                            "mapping.record_aliases[{index}].{key} must be an object"
                        ))
                    })
            };
            Ok(RecordAlias {
                record: part("record")?,
                reads_as: part("reads_as")?,
            })
        })
        .collect()
}

/// The identity keys an alias compares and rewrites.
struct AliasKeys {
    discipline: Option<i64>,
    category: Option<i64>,
    parameter: i64,
    level_type: i64,
    level_value: f64,
    second_level_type: Option<i64>,
    second_level_value: Option<f64>,
    pdt: Option<i64>,
}

impl AliasKeys {
    fn of_identity(identity: &RecordIdentity) -> Self {
        AliasKeys {
            discipline: identity.discipline,
            category: identity.category,
            parameter: identity.parameter,
            level_type: identity.level_type,
            level_value: identity.level_value,
            second_level_type: identity.second_level_type,
            second_level_value: identity.second_level_value,
            pdt: identity.time_semantics.first().copied(),
        }
    }

    fn of_record(record: &GribRecord) -> Self {
        AliasKeys {
            discipline: record.discipline,
            category: record.category,
            parameter: record.parameter,
            level_type: record.level_type,
            level_value: record.level_value,
            second_level_type: record.second_level_type,
            second_level_value: record.second_level_value,
            pdt: record.time_semantics.first().copied(),
        }
    }

    /// Whether `pattern` (an alias's `record`) names these keys.
    fn named_by(&self, pattern: &crate::node::Node) -> bool {
        selector_int(pattern, "discipline").is_none_or(|value| self.discipline == Some(value))
            && selector_int(pattern, "category").is_none_or(|value| self.category == Some(value))
            && selector_int(pattern, "parameter").is_none_or(|value| self.parameter == value)
            && selector_int(pattern, "level_type").is_none_or(|value| self.level_type == value)
            && selector_float(pattern, "level_value")
                .is_none_or(|value| close(self.level_value, value))
            && selector_int(pattern, "second_level_type")
                .is_none_or(|value| self.second_level_type == Some(value))
            && selector_float(pattern, "second_level_value").is_none_or(|value| {
                self.second_level_value
                    .is_some_and(|observed| close(observed, value))
            })
            && selector_int(pattern, "pdt").is_none_or(|value| self.pdt == Some(value))
    }

    /// The keys `reads_as` declares, written over these.  A `reads_as`
    /// without a second surface reads as a record that has none.
    fn rewritten(mut self, reads_as: &crate::node::Node) -> Self {
        if let Some(value) = selector_int(reads_as, "discipline") {
            self.discipline = Some(value);
        }
        if let Some(value) = selector_int(reads_as, "category") {
            self.category = Some(value);
        }
        if let Some(value) = selector_int(reads_as, "parameter") {
            self.parameter = value;
        }
        if let Some(value) = selector_int(reads_as, "level_type") {
            self.level_type = value;
        }
        if let Some(value) = selector_float(reads_as, "level_value") {
            self.level_value = value;
        }
        match selector_int(reads_as, "second_level_type") {
            Some(value) => {
                self.second_level_type = Some(value);
                self.second_level_value = selector_float(reads_as, "second_level_value");
            }
            None => self.second_level_type = Some(255),
        }
        self
    }
}

/// The alias that names `keys`, if any.  Validation refuses two aliases
/// that can name one record, so the first hit is the only one.
fn alias_for<'a>(aliases: &'a [RecordAlias], keys: &AliasKeys) -> Option<&'a RecordAlias> {
    aliases.iter().find(|alias| keys.named_by(&alias.record))
}

/// Rewrite every identity an alias names; returns which were rewritten.
pub fn alias_identities(aliases: &[RecordAlias], identities: &mut [RecordIdentity]) -> Vec<bool> {
    identities
        .iter_mut()
        .map(|identity| {
            let keys = AliasKeys::of_identity(identity);
            let Some(alias) = alias_for(aliases, &keys) else {
                return false;
            };
            let keys = keys.rewritten(&alias.reads_as);
            identity.discipline = keys.discipline;
            identity.category = keys.category;
            identity.parameter = keys.parameter;
            identity.level_type = keys.level_type;
            identity.level_value = keys.level_value;
            identity.second_level_type = keys.second_level_type;
            identity.second_level_value = keys.second_level_value;
            true
        })
        .collect()
}

/// Rewrite every decoded record an alias names, exactly as
/// [`alias_identities`] rewrote its inventory identity.
pub fn alias_records(aliases: &[RecordAlias], records: &mut [GribRecord]) {
    for record in records.iter_mut() {
        let keys = AliasKeys::of_record(record);
        let Some(alias) = alias_for(aliases, &keys) else {
            continue;
        };
        let keys = keys.rewritten(&alias.reads_as);
        record.discipline = keys.discipline;
        record.category = keys.category;
        record.parameter = keys.parameter;
        record.level_type = keys.level_type;
        record.level_value = keys.level_value;
        record.second_level_type = keys.second_level_type;
        record.second_level_value = keys.second_level_value;
    }
}

/// Whether an observed first point is one of the mapping's published
/// encodings of its declared first point (`mapping.grid.published_first_points`),
/// both coordinates at the grid check's own 1e-6.
fn published_first_point(declaration: &GridDeclaration, lat1: f64, lon1: f64) -> bool {
    let close = |observed: f64, declared: f64| (observed - declared).abs() <= 1e-6;
    declaration
        .published_first_points
        .iter()
        .any(|&(lat, lon)| close(lat1, lat) && close(lon1, lon))
}

/// The canonical (projected y, x) axes of a declared Lambert grid, read
/// from Section 3 of one message before anything is unpacked; `None` when
/// the message is not the declared grid in the layout the decode accepts
/// (GDT 30, scan 0x40), which the record decode then refuses as before.
pub fn declared_lambert_axes(
    message: &Grib2Message,
    index: usize,
    declaration: &GridDeclaration,
    source: &str,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let template = declared_template(declaration)?;
    if message.grid.template != template || message.grid.scan_mode != 0x40 {
        return None;
    }
    require_declared_grid(message, index, declaration, source).ok()?;
    declaration.projected_axes()
}

/// The GRIB2 grid definition template a declared projected family is
/// published under: 3.30 for Lambert, 3.1 for a rotated lattice.
fn declared_template(declaration: &GridDeclaration) -> Option<u16> {
    if declaration.is_lambert() {
        Some(30)
    } else if declaration.is_rotated() {
        Some(1)
    } else {
        None
    }
}

/// `require_declared_grid` for a declared rotated lattice (GDT 3.1): every
/// Section 3 number the declaration states, at the Lambert check's 1e-6.
fn require_declared_rotated_grid(
    message: &Grib2Message,
    index: usize,
    declaration: &GridDeclaration,
    source: &str,
) -> Result<()> {
    let parameters = declaration
        .rotated
        .as_ref()
        .expect("rotated declarations carry parameters");
    let grid = &message.grid;
    let mut mismatched: Vec<String> = Vec::new();
    let mut compare_float = |key: &str, observed: f64, declared: f64| {
        if (observed - declared).abs() > 1e-6 {
            mismatched.push(format!("{key}: declared {declared}, observed {observed}"));
        }
    };
    compare_float("lat1", grid.lat1, parameters.lat1);
    compare_float("lon1", grid.lon1, parameters.lon1);
    compare_float("dlat_deg", grid.dy, parameters.dlat_deg);
    compare_float("dlon_deg", grid.dx, parameters.dlon_deg);
    compare_float("south_pole_lat", grid.south_pole_lat, parameters.south_pole_lat);
    compare_float("south_pole_lon", grid.south_pole_lon, parameters.south_pole_lon);
    compare_float("rotation_angle", grid.rotation_angle, parameters.rotation_angle);
    if grid.nx as i64 != parameters.nx {
        mismatched.push(format!("nx: declared {}, observed {}", parameters.nx, grid.nx));
    }
    if grid.ny as i64 != parameters.ny {
        mismatched.push(format!("ny: declared {}, observed {}", parameters.ny, grid.ny));
    }
    if grid.shape_of_earth as i64 != parameters.shape_of_earth {
        mismatched.push(format!(
            "shape_of_earth: declared {}, observed {}",
            parameters.shape_of_earth, grid.shape_of_earth
        ));
    }
    if !mismatched.is_empty() {
        mismatched.sort();
        return Err(grid_mismatch(format!(
            "GRIB2 field {index} in {source} is not on the declared \
             rotated_latitude_longitude grid; {}",
            mismatched.join("; ")
        )));
    }
    Ok(())
}

/// `mapped_source._require_declared_grib2_grid`.
fn require_declared_grid(
    message: &Grib2Message,
    index: usize,
    declaration: &GridDeclaration,
    source: &str,
) -> Result<()> {
    if declaration.is_rotated() {
        require_declared_rotated_grid(message, index, declaration, source)?;
        return require_declared_wind_basis(message, index, declaration, source);
    }
    let parameters = declaration
        .parameters
        .as_ref()
        .expect("lambert declarations carry parameters");
    let grid = &message.grid;
    let mut mismatched: Vec<String> = Vec::new();
    let mut compare_float = |key: &str, observed: f64, declared: f64| {
        if (observed - declared).abs() > 1e-6 {
            mismatched.push(format!("{key}: declared {declared}, observed {observed}"));
        }
    };
    // The first point is one coordinate pair: the declared one, or one of
    // the mapping's published encodings of it, both halves together.
    if !published_first_point(declaration, grid.lat1, grid.lon1) {
        compare_float("lat1", grid.lat1, parameters.lat1);
        compare_float("lon1", grid.lon1, parameters.lon1);
    }
    compare_float("dx_m", grid.dx, parameters.dx_m);
    compare_float("dy_m", grid.dy, parameters.dy_m);
    compare_float("latin1", grid.latin1, parameters.latin1);
    compare_float("latin2", grid.latin2, parameters.latin2);
    compare_float("lov", grid.lov, parameters.lov);
    if grid.nx as i64 != parameters.nx {
        mismatched.push(format!(
            "nx: declared {}, observed {}",
            parameters.nx, grid.nx
        ));
    }
    if grid.ny as i64 != parameters.ny {
        mismatched.push(format!(
            "ny: declared {}, observed {}",
            parameters.ny, grid.ny
        ));
    }
    if grid.shape_of_earth as i64 != parameters.shape_of_earth {
        mismatched.push(format!(
            "shape_of_earth: declared {}, observed {}",
            parameters.shape_of_earth, grid.shape_of_earth
        ));
    }
    if !mismatched.is_empty() {
        mismatched.sort();
        return Err(grid_mismatch(format!(
            "GRIB2 field {index} in {source} is not on the declared \
             lambert_conformal grid; {}",
            mismatched.join("; ")
        )));
    }
    require_declared_wind_basis(message, index, declaration, source)
}

/// The record's vector-component basis (resolution flags bit 0x08) is the
/// one the mapping declares.
fn require_declared_wind_basis(
    message: &Grib2Message,
    index: usize,
    declaration: &GridDeclaration,
    source: &str,
) -> Result<()> {
    let grid = &message.grid;
    let grid_relative = grid.resolution_flags & GRIB2_GRID_RELATIVE_WIND_BIT != 0;
    if grid_relative != declaration.rotates_winds() {
        return Err(grid_mismatch(format!(
            "GRIB2 field {index} in {source} declares {}-relative vector \
             components (resolution_flags bit 0x08) while the mapping \
             declares wind_basis='{}'",
            if grid_relative { "grid" } else { "earth" },
            declaration.wind_basis
        )));
    }
    Ok(())
}

/// GRIB2 Data Representation Template 5.40: JPEG2000 packing.
const JPEG2000_TEMPLATE: u16 = 40;

/// `unpack_message`, with the one decoder path that touches process-global
/// state serialized against itself.
///
/// `openjp2`'s HTJ2K block decoder (`ht_dec`, ITU-T T.814 code-blocks)
/// keeps a process-global `static mut` (`only_cleanup_pass_is_decoded`)
/// that two threads may write at once.  It only decides whether one
/// "malformed HT codeblock" warning is printed (the pass count it guards is
/// set the same way either way), but a racing write is still a data race,
/// so a codestream that can reach that decoder takes [`JPEG2000_HT`].
///
/// Classic JPEG2000 (ISO 15444-1), which is what GRIB2 Template 5.40 packs
/// (RAP/NAM AWIPS grids, MSC GDPS), never reaches it: `openjp2` enters the
/// HT decoder only for a code-block whose coding style carries the HT bit
/// (`J2K_CCP_CBLKSTY_HT`, 0x40, `t1.rs`), and every other table it reads
/// is immutable.  So only a codestream whose COD/COC markers declare HT
/// code-blocks, or one this scan cannot read, is serialized
/// ([`jpeg2000_may_decode_ht`]).
///
/// Named breakage this split fixes: the lock used to cover EVERY Template
/// 5.40 decode, so a RAP awp130 native compose (all 5.40) decoded one
/// message at a time across the whole process: on a 491-core box the
/// 13-time initial compose ran 131 s with about one core busy for 110 s
/// of it, and the 36-time boundary compose about 6 min, while the card
/// waited (V59, 2026-10-21).
fn unpack_shared(message: &Grib2Message) -> std::result::Result<Vec<f64>, grib_core::GribError> {
    if message.data_rep.template != JPEG2000_TEMPLATE || !jpeg2000_may_decode_ht(&message.raw_data) {
        return unpack_message(message);
    }
    // A panic inside one message's decode must not turn every later
    // HT message into a poisoned-lock refusal about threading.
    let _guard = JPEG2000_HT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    unpack_message(message)
}

/// [`unpack_shared`] for a stored rectangle: a JPEG2000 area decode runs
/// the same `openjp2` decoder, so it takes the same lock on the same terms.
fn decode_rect_shared(
    message: &Grib2Message,
    rect: grib_core::grib2::StoredRect,
) -> std::result::Result<Vec<f64>, grib_core::GribError> {
    if message.data_rep.template != JPEG2000_TEMPLATE || !jpeg2000_may_decode_ht(&message.raw_data) {
        return grib_core::grib2::unpack_message_stored_rect(message, rect);
    }
    let _guard = JPEG2000_HT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    grib_core::grib2::unpack_message_stored_rect(message, rect)
}

/// The lock every `openjp2` decode that may reach its HT block decoder takes.
static JPEG2000_HT: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Whether `openjp2` may run its HT block decoder on this codestream:
/// true when any COD or COC marker, in the main header or in any
/// tile-part header, declares HT code-blocks (style bit 0x40), and true
/// whenever the codestream cannot be walked to its end (not a raw J2K
/// codestream, a truncated or malformed marker segment) -- an unreadable
/// header is serialized rather than guessed at.
fn jpeg2000_may_decode_ht(data: &[u8]) -> bool {
    const HT: u8 = 0x40;
    let be16 = |at: usize| data.get(at..at + 2).map(|b| u16::from_be_bytes([b[0], b[1]]) as usize);
    let be32 = |at: usize| data.get(at..at + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]) as usize);
    if data.get(0..2) != Some(&[0xFF, 0x4F]) {
        return true;
    }
    // Csiz decides whether a COC names its component in one octet or two.
    let mut wide_components = None;
    let mut position = 2usize;
    // Where the tile-part being read ends, once its SOT has been read.
    let mut tile_part_end: Option<usize> = None;
    loop {
        if data.get(position) != Some(&0xFF) {
            return true;
        }
        let Some(&marker) = data.get(position + 1) else { return true };
        match marker {
            // EOC: every header has been read.
            0xD9 => return false,
            // SOD: the tile-part's data follows; resume at its end.
            0x93 => match tile_part_end {
                Some(end) if end > position => {
                    if end >= data.len() {
                        return false;
                    }
                    position = end;
                    tile_part_end = None;
                    continue;
                }
                _ => return true,
            },
            _ => {}
        }
        let Some(length) = be16(position + 2) else { return true };
        if length < 2 || position + 2 + length > data.len() {
            return true;
        }
        let segment = &data[position + 2..position + 2 + length];
        match marker {
            // SIZ: Lsiz Rsiz Xsiz Ysiz XOsiz YOsiz XTsiz YTsiz XTOsiz YTOsiz Csiz.
            0x51 => {
                let Some(count) = be16(position + 2 + 36) else { return true };
                wide_components = Some(count >= 257);
            }
            // SOT: Lsot Isot Psot TPsot TNsot; Psot 0 runs to the end.
            0x90 => {
                let Some(psot) = be32(position + 6) else { return true };
                tile_part_end = Some(if psot == 0 { data.len() } else { position + psot });
            }
            // COD: Lcod Scod SGcod(4) then levels, xcb, ycb, style.
            0x52 => match segment.get(10) {
                Some(style) if style & HT == 0 => {}
                _ => return true,
            },
            // COC: Lcoc Ccoc(1|2) Scoc then levels, xcb, ycb, style.
            0x53 => {
                let Some(wide) = wide_components else { return true };
                let at = 2 + if wide { 2 } else { 1 } + 1 + 3;
                match segment.get(at) {
                    Some(style) if style & HT == 0 => {}
                    _ => return true,
                }
            }
            _ => {}
        }
        position += 2 + length;
    }
}

/// `mapped_source._grib2_records`: the selected records of one object,
/// decoded CONCURRENTLY.
///
/// Takes the object ALREADY PARSED, because its one caller has parsed
/// it to inventory it: the byte-taking spelling this replaces parsed
/// the same half-gigabyte file a second time and held both copies.
///
/// Message decode is pure per-message work (unpack a byte range into a
/// fresh array, read the grid octets, build the frame) so the messages
/// of one object are independent. They are decoded into pre-assigned
/// slots (`crate::threads`) and drained in `wanted` order, so the
/// returned vector, and the refusal if one is raised, are exactly what
/// the serial loop produced.
pub fn grib2_records(
    file: &Grib2File,
    source_label: &str,
    wanted: &[usize],
    declaration: &GridDeclaration,
) -> Result<Vec<GribRecord>> {
    grib2_records_within(file, source_label, wanted, declaration, None)
}

/// A canonical rectangle, and the records of one object decoded over it
/// alone (by their message index in that object).
pub struct RecordWindow<'a> {
    pub rows: [usize; 2],
    pub columns: [usize; 2],
    pub records: &'a std::collections::BTreeSet<usize>,
    /// Those of `records` that answer `air_pressure`: each keeps its
    /// full-plane ladder value (`GribRecord::plane_ladder_hpa`), taken
    /// from the whole unpacked plane before it is cropped.  Only a
    /// projected record, which is unpacked whole, can state it.
    pub ladder: &'a std::collections::BTreeSet<usize>,
}

/// [`grib2_records`], with the records `window` names decoded over its
/// rectangle only.  Such a record keeps the source's full canonical
/// axes and carries the rectangle's values.
pub fn grib2_records_within(
    file: &Grib2File,
    source_label: &str,
    wanted: &[usize],
    declaration: &GridDeclaration,
    window: Option<&RecordWindow<'_>>,
) -> Result<Vec<GribRecord>> {
    if wanted.is_empty() {
        return Ok(Vec::new());
    }
    let slots: Vec<Result<GribRecord>> = crate::threads::install(|| {
        use rayon::prelude::*;
        wanted
            .par_iter()
            .map(|index| {
                let window = window.filter(|window| window.records.contains(index));
                grib2_record(file, source_label, *index, declaration, window)
            })
            .collect()
    });
    crate::threads::in_order(slots)
}

/// One selected GRIB2 message, decoded.
fn grib2_record(
    file: &Grib2File,
    source_label: &str,
    index: usize,
    declaration: &GridDeclaration,
    window: Option<&RecordWindow<'_>>,
) -> Result<GribRecord> {
    let message = file.messages.get(index).ok_or_else(|| {
        decode_failed(format!(
            "GRIB2 object {source_label} has no field {index}"
        ))
    })?;
    // A projected (Lambert) record has no row/column-addressable packing,
    // so it is unpacked whole below and cropped to the window at once: one
    // plane is transient on its worker, and only the window is retained.
    let projected_window = window.filter(|_| declaration.is_projected());
    if let Some(window) = window.filter(|_| !declaration.is_projected()) {
        if window.ladder.contains(&index) {
            return Err(decode_failed(format!(
                "GRIB2 field {index} in {source_label}: a regular-grid window decode \n                 never unpacks the whole plane its pressure ladder is taken from"
            )));
        }
        let axes = regular_latlon_axes(message)?;
        let values = regular_latlon_window(message, &axes, window.rows, window.columns)
            .map_err(|error| {
                decode_failed(format!(
                    "GRIB2 field {index} in {source_label} failed to decode rows \
                     {}..{} and columns {}..{} of its canonical grid: {error}",
                    window.rows[0], window.rows[1], window.columns[0], window.columns[1]
                ))
            })?;
        let shape = [window.rows[1] - window.rows[0], window.columns[1] - window.columns[0]];
        let array = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|error| {
            decode_failed(format!(
                "GRIB2 field {index} in {source_label} does not fill its window: {error}"
            ))
        })?;
        return record_from(message, source_label, index, array, axes.latitude, axes.longitude);
    }
    let nx = message.grid.nx as usize;
    let ny = message.grid.ny as usize;
    let raw = unpack_shared(message).map_err(|error| {
        decode_failed(format!(
            "GRIB2 field {index} in {source_label} failed to decode: {error}"
        ))
    })?;
    if raw.len() != nx * ny {
        return Err(decode_failed(format!(
            "GRIB2 field {index} decoded count {} differs from grid {nx}x{ny}",
            raw.len()
        )));
    }
    let (latitude, longitude, values) = if declaration.is_projected() {
        let template = declared_template(declaration).expect("projected family");
        if message.grid.template != template || message.grid.scan_mode != 0x40 {
            return Err(grid_mismatch(format!(
                "the mapping declares a {} source grid, \
                 which requires GRIB2 GDT {template} with scan mode 0x40; field \
                 {index} in {source_label} carries GDT {} scan 0x{:02x}",
                declaration.family, message.grid.template, message.grid.scan_mode
            )));
        }
        require_declared_grid(message, index, declaration, source_label)?;
        let (y, x) = declaration
            .projected_axes()
            .expect("projected declarations carry parameters");
        (y, x, raw)
    } else {
        regular_latlon_frame(message, &raw)?
    };
    if let Some(window) = projected_window {
        let (ny, nx) = (latitude.len(), longitude.len());
        let ([r0, r1], [c0, c1]) = (window.rows, window.columns);
        if !(r0 < r1 && r1 <= ny && c0 < c1 && c1 <= nx) || values.len() != ny * nx {
            return Err(decode_failed(format!(
                "GRIB2 field {index} in {source_label}: rows {r0}..{r1} and columns \
                 {c0}..{c1} do not lie inside its {ny}x{nx} canonical grid"
            )));
        }
        let mut cropped = Vec::with_capacity((r1 - r0) * (c1 - c0));
        for row in r0..r1 {
            cropped.extend_from_slice(&values[row * nx + c0..row * nx + c1]);
        }
        // The frame states `air_pressure`'s per-level pressure as the
        // median of its whole plane; that plane is here, so the value is
        // the one the full decode states (`frames::field_work`).
        let ladder = if window.ladder.contains(&index) {
            Some(crate::window::pressure_levels(&values, values.len())?[0])
        } else {
            None
        };
        drop(values);
        let array = ArrayD::from_shape_vec(IxDyn(&[r1 - r0, c1 - c0]), cropped).map_err(|error| {
            decode_failed(format!(
                "GRIB2 field {index} in {source_label} does not fill its window: {error}"
            ))
        })?;
        let mut record = record_from(message, source_label, index, array, latitude, longitude)?;
        record.plane_ladder_hpa = ladder;
        return Ok(record);
    }
    let array = ArrayD::from_shape_vec(IxDyn(&[latitude.len(), longitude.len()]), values)
        .map_err(|error| {
            decode_failed(format!(
                "GRIB2 field {index} in {source_label} does not fill its grid: {error}"
            ))
        })?;
    record_from(message, source_label, index, array, latitude, longitude)
}

/// The record a decoded message becomes, its values already laid out.
fn record_from(
    message: &Grib2Message,
    source_label: &str,
    index: usize,
    array: ArrayD<f64>,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
) -> Result<GribRecord> {
    let valid_time = embedded_valid_time(
        message.reference_time,
        message.product.time_range_unit,
        message.product.forecast_time as i64,
        2,
    )?;
    Ok(GribRecord {
        source: source_label.to_owned(),
        index,
        reference_time: message.reference_time,
        valid_time,
        member: message
            .product
            .perturbation_number
            .map(|value| value.to_string()),
        parameter: message.product.parameter_number as i64,
        level_type: message.product.level_type as i64,
        level_value: message.product.level_value,
        table_version: None,
        center: Some(message.identification.center_id as i64),
        subcenter: Some(message.identification.subcenter_id as i64),
        master_table_version: Some(message.identification.master_table_version as i64),
        local_table_version: Some(message.identification.local_table_version as i64),
        discipline: Some(message.discipline as i64),
        category: Some(message.product.parameter_category as i64),
        second_level_type: Some(message.product.second_level_type as i64),
        second_level_value: Some(message.product.second_level_value),
        process_identity: Some((
            message.product.generating_process as i64,
            message.product.forecast_generating_process_id as i64,
        )),
        time_semantics: vec![message.product.template as i64],
        coordinate_values: message.product.coordinate_values.clone(),
        values: array,
        latitude,
        longitude,
        grid_fingerprint: grid_fingerprint(message),
        plane_ladder_hpa: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Node;

    #[test]
    fn the_hrrr_mappings_admit_the_hrrrv1_millidegree_corner_and_nothing_else() {
        // HRRRv1 wrfprs/wrfnat (to 2016-08-23) publish 21.138, 237.28 for
        // the grid current files publish as 21.138123, 237.280472.
        for text in [
            include_str!("../../../../../woof/authorities/rw-wps-hrrr-prs-grib2.mapping.json"),
            include_str!("../../../../../woof/authorities/rw-wps-hrrr-native-grib2.mapping.json"),
        ] {
            let bytes = text.as_bytes().to_vec();
            let mapping = crate::model::Mapping {
                sha256: crate::digest::bytes_sha256(&bytes),
                doc: Node::parse(&bytes).unwrap(),
                path: "<packaged>".to_owned(),
            };
            let declaration = mapping.grid_declaration().unwrap();
            assert_eq!(declaration.published_first_points, vec![(21.138, 237.28)]);
            assert!(published_first_point(&declaration, 21.138, 237.28));
            // Half an encoding is not the encoding.
            assert!(!published_first_point(&declaration, 21.138, 237.280472));
            assert!(!published_first_point(&declaration, 21.139, 237.28));
            assert!(!published_first_point(&declaration, 21.138, 237.281));
        }
    }

    #[test]
    fn a_grib2_selector_without_a_second_surface_admits_only_the_missing_code() {
        let selector = Node::parse(br#"{"discipline": 0, "category": 0, "parameter": 0}"#).unwrap();
        let mut identity = RecordIdentity {
            index: 0,
            member: None,
            parameter: 0,
            level_type: 100,
            level_value: 85000.0,
            table_version: None,
            center: Some(7),
            subcenter: Some(0),
            master_table_version: Some(2),
            local_table_version: Some(1),
            discipline: Some(0),
            category: Some(0),
            second_level_type: Some(255),
            second_level_value: Some(0.0),
            time_semantics: vec![0],
        };
        assert!(selector_matches(&selector, &identity, "grib2"));
        identity.second_level_type = Some(106);
        assert!(!selector_matches(&selector, &identity, "grib2"));
    }

    #[test]
    fn a_selector_pinning_producer_identity_refuses_the_other_producer() {
        let selector = Node::parse(
            br#"{"discipline": 0, "category": 0, "parameter": 0, "center": 7, "subcenter": 4}"#,
        )
        .unwrap();
        let mut identity = RecordIdentity {
            index: 0,
            member: None,
            parameter: 0,
            level_type: 100,
            level_value: 85000.0,
            table_version: None,
            center: Some(7),
            subcenter: Some(4),
            master_table_version: Some(2),
            local_table_version: Some(1),
            discipline: Some(0),
            category: Some(0),
            second_level_type: Some(255),
            second_level_value: Some(0.0),
            time_semantics: vec![0],
        };
        assert!(selector_matches(&selector, &identity, "grib2"));
        identity.subcenter = Some(0);
        assert!(!selector_matches(&selector, &identity, "grib2"));
    }

    fn depth_layer(parameter: i64, discipline: i64, category: i64, top: f64, bottom: f64) -> RecordIdentity {
        RecordIdentity {
            index: 0,
            member: None,
            parameter,
            level_type: 106,
            level_value: top,
            table_version: None,
            center: Some(98),
            subcenter: Some(0),
            master_table_version: Some(27),
            local_table_version: Some(0),
            discipline: Some(discipline),
            category: Some(category),
            second_level_type: Some(106),
            second_level_value: Some(bottom),
            time_semantics: vec![0],
        }
    }

    fn soil_alias_mapping() -> Mapping {
        let text = br#"{"schema": "rw-wps.mapping.v1", "name": "t", "format": "grib2",
            "record_aliases": [
              {"record": {"discipline": 192, "category": 128, "parameter": 170,
                          "level_type": 106, "level_value": 7},
               "reads_as": {"discipline": 2, "category": 3, "parameter": 18,
                            "level_type": 151, "level_value": 1,
                            "second_level_type": 151, "second_level_value": 2}},
              {"record": {"discipline": 192, "category": 128, "parameter": 236,
                          "level_type": 106, "level_value": 100},
               "reads_as": {"discipline": 2, "category": 3, "parameter": 18,
                            "level_type": 151, "level_value": 3,
                            "second_level_type": 151, "second_level_value": 4}}
            ]}"#;
        Mapping {
            sha256: crate::digest::bytes_sha256(text),
            doc: Node::parse(text).unwrap(),
            path: "<test>".to_owned(),
        }
    }

    #[test]
    fn an_aliased_record_reads_as_the_spelling_the_selectors_name() {
        let aliases = record_aliases(&soil_alias_mapping()).unwrap();
        let mut identities = vec![
            depth_layer(170, 192, 128, 7.0, 28.0),
            // The deepest layer's bottom is written as the all-ones
            // missing value; the alias leaves it out and still names it.
            depth_layer(236, 192, 128, 100.0, 4294967295.0),
            // Not named by any alias: untouched.
            depth_layer(39, 192, 128, 0.0, 7.0),
        ];
        let aliased = alias_identities(&aliases, &mut identities);
        assert_eq!(aliased, vec![true, true, false]);
        let selector = Node::parse(
            br#"{"discipline": 2, "category": 3, "parameter": 18, "level_type": 151,
                 "level_value": 3, "second_level_type": 151, "second_level_value": 4}"#,
        )
        .unwrap();
        assert!(selector_matches(&selector, &identities[1], "grib2"));
        assert!(!selector_matches(&selector, &identities[0], "grib2"));
        assert_eq!(identities[0].level_value, 1.0);
        assert_eq!(identities[0].second_level_value, Some(2.0));
        assert_eq!(identities[2].parameter, 39);
        assert_eq!(identities[2].discipline, Some(192));
    }

    #[test]
    fn a_mapping_without_aliases_reads_every_record_as_spelled() {
        let text = br#"{"schema": "rw-wps.mapping.v1", "name": "t", "format": "grib2"}"#;
        let mapping = Mapping {
            sha256: crate::digest::bytes_sha256(text),
            doc: Node::parse(text).unwrap(),
            path: "<test>".to_owned(),
        };
        let aliases = record_aliases(&mapping).unwrap();
        let mut identities = vec![depth_layer(170, 192, 128, 7.0, 28.0)];
        assert_eq!(alias_identities(&aliases, &mut identities), vec![false]);
        assert_eq!(identities[0].parameter, 170);
    }

    #[test]
    fn a_regional_crop_across_the_antimeridian_keeps_its_column_order() {
        // The GDT-101 remap's crop for a domain on the seam: west 175,
        // 0.5 degree, 41 columns, so its stored columns run 175..195.
        let (longitude, order) = regular_longitude_axis(175.0, 0.5, 41);
        assert_eq!(order, (0..41).collect::<Vec<usize>>());
        assert_eq!(longitude.first(), Some(&175.0));
        assert_eq!(longitude.last(), Some(&195.0));
        assert!(longitude.windows(2).all(|pair| pair[1] - pair[0] == 0.5));
    }

    #[test]
    fn a_window_decode_is_the_crop_of_the_canonical_frame() {
        use grib_core::grib2::{DataRepresentation, GridDefinition};
        // A whole 8-column ring stored from 90E, so the canonical axis
        // (-180 first) reads stored columns 2..8 then 0..2: a window over
        // canonical columns 4..8 reaches across the stored cut.
        for scan in [0x40u8, 0x00] {
            let message = Grib2Message {
                discipline: 0,
                identification: Default::default(),
                reference_time: chrono::NaiveDate::from_ymd_opt(2026, 9, 29)
                    .unwrap()
                    .and_hms_opt(12, 0, 0)
                    .unwrap(),
                grid: GridDefinition {
                    template: 0,
                    nx: 8,
                    ny: 5,
                    num_data_points: 40,
                    lat1: if scan == 0x40 { -60.0 } else { 60.0 },
                    lon1: 90.0,
                    dx: 45.0,
                    dy: 30.0,
                    scan_mode: scan,
                    ..GridDefinition::default()
                },
                product: Default::default(),
                data_rep: DataRepresentation {
                    template: 0,
                    bits_per_value: 8,
                    section5_num_data_points: 40,
                    ..DataRepresentation::default()
                },
                bitmap: None,
                raw_data: (0..40).map(|value| value * 3 + 1).collect(),
            };
            let raw = unpack_message(&message).unwrap();
            let (latitude, longitude, values) = regular_latlon_frame(&message, &raw).unwrap();
            let axes = regular_latlon_axes(&message).unwrap();
            assert_eq!(axes.latitude, latitude);
            assert_eq!(axes.longitude, longitude);
            for (rows, columns) in [([1, 4], [4, 8]), ([0, 2], [1, 3]), ([2, 5], [0, 8])] {
                let window = regular_latlon_window(&message, &axes, rows, columns).unwrap();
                let expected: Vec<f64> = (rows[0]..rows[1])
                    .flat_map(|row| (columns[0]..columns[1]).map(move |column| (row, column)))
                    .map(|(row, column)| values[row * 8 + column])
                    .collect();
                assert_eq!(window, expected, "scan 0x{scan:02x} rows {rows:?} columns {columns:?}");
            }
        }
    }

    #[test]
    fn parallel_frame_export_preserves_every_bit_and_stable_column_order() {
        use grib_core::grib2::{DataRepresentation, GridDefinition};
        // Cover one/two contiguous runs, duplicate seam columns, a
        // noncyclic descending order, and enough rows to split the pool.
        for (nx, lon1, dx) in [
            (8, 90.0, 45.0),
            (9, 0.0, 45.0),
            (8, 175.0, 0.5),
            (8, 90.0, -45.0),
        ] {
            for scan in [0x40, 0x00] {
                let ny = 67;
                let message = Grib2Message {
                    discipline: 0,
                    identification: Default::default(),
                    reference_time: chrono::NaiveDate::from_ymd_opt(2026, 10, 18)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap(),
                    grid: GridDefinition {
                        template: 0,
                        nx: nx as u32,
                        ny: ny as u32,
                        lat1: if scan == 0x40 { -33.0 } else { 33.0 },
                        lon1,
                        dx,
                        dy: 1.0,
                        scan_mode: scan,
                        ..Default::default()
                    },
                    product: Default::default(),
                    data_rep: DataRepresentation::default(),
                    bitmap: None,
                    raw_data: Vec::new(),
                };
                let raw: Vec<_> = (0..nx * ny)
                    .map(|i| match i % 7 {
                        0 => f64::from_bits(0x7ff8_0000_0000_0001 + i as u64),
                        1 => -0.0,
                        _ => i as f64,
                    })
                    .collect();
                let (latitude, longitude, actual) = regular_latlon_frame(&message, &raw).unwrap();
                let (_, _, worker_actual) = crate::threads::install(|| {
                    regular_latlon_frame(&message, &raw).unwrap()
                });
                let (expected_longitude, order) = regular_longitude_axis(lon1, dx, nx);
                assert_eq!(longitude, expected_longitude);
                assert_eq!(
                    latitude,
                    (0..ny).map(|i| i as f64 - 33.0).collect::<Vec<_>>()
                );
                for row in 0..ny {
                    let stored_row = if scan == 0x00 { ny - 1 - row } else { row };
                    for (column, source_column) in order.iter().enumerate() {
                        assert_eq!(
                            actual[row * nx + column].to_bits(),
                            raw[stored_row * nx + source_column].to_bits()
                        );
                        assert_eq!(worker_actual[row * nx + column].to_bits(),
                                   actual[row * nx + column].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn a_crop_clear_of_the_seam_and_a_whole_ring_read_as_before() {
        // Clear of the seam: the wrapped values, in stored order.
        let (longitude, order) = regular_longitude_axis(336.5, 0.5, 11);
        assert_eq!(order, (0..11).collect::<Vec<usize>>());
        assert_eq!(longitude.first(), Some(&-23.5));
        assert_eq!(longitude.last(), Some(&-18.5));
        // A whole ring keeps the -180..180 cut its decode has always stored.
        let (longitude, order) = regular_longitude_axis(0.0, 1.0, 360);
        assert_eq!(longitude.first(), Some(&-180.0));
        assert_eq!(longitude.last(), Some(&179.0));
        assert_eq!(order[0], 180);
        // A 1/12-degree ring whose octets round the spacing to micro-degrees
        // spans a hair under 360 degrees; it is still a ring, not a crop.
        let (longitude, order) = regular_longitude_axis(0.0, 0.083333, 4320);
        assert_eq!(order[0], 2161);
        assert!(longitude[0] >= -180.0 && longitude[0] < -179.9);
        assert!(longitude[4319] > 179.9 && longitude[4319] < 180.0);
    }

    #[test]
    fn envelope_validation_names_a_truncated_object() {
        let refusal = validate_grib2_envelopes(b"GRIB\x00\x00\x00\x02", "sample").unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::DECODE_FAILED);
        assert!(refusal.message.contains("fewer than 20 bytes"));
    }

    #[test]
    fn envelope_validation_refuses_a_grib1_object_by_edition() {
        let mut bytes = vec![0u8; 24];
        bytes[..4].copy_from_slice(b"GRIB");
        bytes[7] = 1;
        let refusal = validate_grib2_envelopes(&bytes, "sample").unwrap_err();
        assert!(refusal.message.contains("GRIB edition 1"));
    }

    #[test]
    fn embedded_valid_time_uses_the_edition_two_second_code() {
        let reference = NaiveDateTime::parse_from_str("2026-08-17 00:00:00", "%Y-%m-%d %H:%M:%S")
            .unwrap();
        let valid = embedded_valid_time(reference, 13, 90, 2).unwrap();
        assert_eq!((valid - reference).num_seconds(), 90);
        // The same code means DAYS in no edition; 254 is edition 1's second.
        assert!(embedded_valid_time(reference, 254, 1, 2).is_err());
        assert!(embedded_valid_time(reference, 254, 1, 1).is_ok());
    }

    /// A raw J2K codestream: SOC, SIZ (one component), COD with code-block
    /// `style`, then one tile-part whose header carries `tile_header`.
    fn codestream(style: u8, tile_header: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0xFF, 0x4F];
        // SIZ: Lsiz 41, Rsiz, sizes and offsets (32 octets), Csiz 1, one component.
        bytes.extend([0xFF, 0x51, 0x00, 41, 0x00, 0x00]);
        bytes.extend([0u8; 32]);
        bytes.extend([0x00, 0x01, 0x07, 0x01, 0x01]);
        // COD: Lcod 12, Scod, SGcod(4), levels, xcb, ycb, style, transform.
        bytes.extend([0xFF, 0x52, 0x00, 12, 0x00, 0x00, 0x00, 0x01, 0x00, 0x05, 0x04, 0x04, style, 0x01]);
        let sot = bytes.len();
        let psot = (12 + tile_header.len() + 2 + 3) as u32;
        bytes.extend([0xFF, 0x90, 0x00, 10, 0x00, 0x00]);
        bytes.extend(psot.to_be_bytes());
        bytes.extend([0x00, 0x01]);
        bytes.extend(tile_header);
        bytes.extend([0xFF, 0x93, 0xAB, 0xFF, 0x52]);
        assert_eq!(bytes.len() - sot, psot as usize);
        bytes.extend([0xFF, 0xD9]);
        bytes
    }

    #[test]
    fn a_classic_jpeg2000_codestream_decodes_without_the_ht_lock() {
        // Tile data that happens to look like a COD marker is skipped, not read.
        assert!(!jpeg2000_may_decode_ht(&codestream(0x00, &[])));
        // A classic COC in a tile-part header leaves it classic.
        assert!(!jpeg2000_may_decode_ht(&codestream(0x00, &[0xFF, 0x53, 0x00, 9, 0x00, 0x00, 0x05, 0x04, 0x04, 0x00, 0x01])));
    }

    #[test]
    fn a_codestream_that_may_reach_the_ht_decoder_takes_the_lock() {
        // HT code-blocks declared in the main header COD.
        assert!(jpeg2000_may_decode_ht(&codestream(0x40, &[])));
        // ... or only in a tile-part header's COC.
        assert!(jpeg2000_may_decode_ht(&codestream(0x00, &[0xFF, 0x53, 0x00, 9, 0x00, 0x00, 0x05, 0x04, 0x04, 0x40, 0x01])));
        // A header this scan cannot read is serialized, not guessed at.
        assert!(jpeg2000_may_decode_ht(&[0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50]));
        let whole = codestream(0x00, &[]);
        assert!(jpeg2000_may_decode_ht(&whole[..40]));
    }
}
