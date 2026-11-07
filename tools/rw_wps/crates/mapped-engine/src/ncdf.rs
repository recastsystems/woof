//! NetCDF sources, decoded in process through `netcrust`.
//!
//! Port of `mapped_source._decode_netcdf` and the helpers above it
//! (`_match_nc_variables`, `_read_nc_values`, `_nc_coordinate_values`,
//! `_require_geographic_horizontal`).  The Python engine reached the same
//! bytes through the `rw_netcdf` exe; the CF decode that exe performs:
//! mask against the STORED representation first, then
//! `scale_factor`/`add_offset`: is transcribed here so the numbers are
//! identical, and the time decode is the same
//! `"<unit> since <timestamp>"` grammar with the same UTC-offset
//! handling: a well-formed offset designator folds into the epoch, a
//! malformed one is refused with a sentence naming what is wrong.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, NaiveTime, Utc};
use ndarray::{ArrayD, IxDyn};
use netcrust::{File as NcFile, Variable};
use rayon::prelude::*;

use crate::array;
use crate::assemble::{DecodedCollection, DirectKey, DirectValue, TimeKey};
use crate::model::{Mapping, GridDeclaration};
use crate::node::Node;
use crate::refusal::{decode_failed, frame_invalid, mapping_invalid, missing_input, Result};

/// How a selector was satisfied: `mapped_source.NC_EVIDENCE_*`.
pub const EVIDENCE_NAME: &str = "name";
pub const EVIDENCE_STANDARD_NAME: &str = "standard_name";

// Both unit sets are stored in Python's `sorted()` order, because the
// refusal interpolates `sorted(accepted_units)` and the two engines'
// sentences are compared byte for byte.  `_` sorts after the uppercase
// letters and before the lowercase ones, which is why `degreeN` leads
// and `degrees_north` trails.
const CF_DEGREES_NORTH: [&str; 6] = [
    "degreeN",
    "degree_N",
    "degree_north",
    "degreesN",
    "degrees_N",
    "degrees_north",
];
const CF_DEGREES_EAST: [&str; 6] = [
    "degreeE",
    "degree_E",
    "degree_east",
    "degreesE",
    "degrees_E",
    "degrees_east",
];
/// CF standard names that mark a PROJECTED axis, which this decode
/// refuses rather than reading as geographic degrees.
///
/// All six the Python engine carries.  The engine used to list four: a
/// file whose axes declare `projection_x_angular_coordinate` sailed
/// past the projection check here and was refused there, so the pair
/// disagreed on whether the source was usable at all.
const CF_PROJECTION_STANDARD_NAMES: [&str; 6] = [
    "projection_x_coordinate",
    "projection_y_coordinate",
    "projection_x_angular_coordinate",
    "projection_y_angular_coordinate",
    "grid_latitude",
    "grid_longitude",
];
const SUPPORTED_GRID_MAPPING_NAMES: [&str; 1] = ["latitude_longitude"];

/// CF string attributes probed by name on a recovered dimension scale.
///
/// `rw_netcdf`'s list, kept identical on purpose: the raw-HDF5 escape
/// hatch reads ONE named string attribute at a time (there is no
/// enumeration) so the set has to be written down, and a set that
/// differed from the shipped bridge's would resolve selectors
/// differently from the Python engine reading the same file.
const RECOVERED_ATTRIBUTES: [&str; 5] =
    ["units", "standard_name", "long_name", "calendar", "axis"];

/// The netCDF-4 marker on an HDF5 dimension scale with no coordinate
/// variable behind it.  The C library writes this exact sentence into the
/// scale's `NAME` attribute and hides such datasets from `variables`, so
/// recovering them would invent variables netCDF4-python does not report.
const PHONY_DIMENSION_PREFIX: &str =
    "This is a netCDF dimension but not a netCDF variable.";

/// Where one variable's bytes and attributes come from.
enum NcSource {
    /// A variable the NetCDF index reports, read through netcrust's
    /// normal `Variable` surface.
    Indexed(Variable),
    /// An HDF5 dimension scale the NetCDF index omits.  Values come back
    /// through `File::read_array_f64`, which already falls back to a
    /// raw-HDF5 read by name; only the probed CF STRING attributes are
    /// reachable, because the escape hatch has no attribute enumeration.
    DimensionScale {
        file: NcFile,
        attributes: BTreeMap<String, String>,
    },
}

/// One NetCDF variable a selector can resolve against.
///
/// This exists because `netcrust::File::variables()` reports the NetCDF-4
/// variable index, and in NetCDF-4 a coordinate variable is stored as an
/// HDF5 *dimension scale* which that index omits.  An ERA5 or CM1 file
/// therefore offers every field and NONE of its axes (`latitude`,
/// `longitude`, `level`, `time` are simply absent) which are exactly the
/// variables the coordinate contract resolves against.  The shipped
/// `rw_netcdf` bridge, which is how the Python engine reaches these same
/// bytes, recovers them; an engine that did not would refuse files the
/// Python engine reads, on every NetCDF-4 source gpuwm has.
struct NcVariable {
    name: String,
    dimensions: Vec<String>,
    source: NcSource,
    /// String attributes the mapping's `producer_attributes` table
    /// declares for this variable and the file does not carry (see
    /// [`apply_producer_attributes`]).  Consulted only after the file's
    /// own attributes, so a file attribute always wins and a conflicting
    /// one was already refused.
    declared: BTreeMap<String, String>,
}

impl NcVariable {
    fn name(&self) -> &str {
        &self.name
    }

    /// The variable's own dimension names, in storage order.
    fn dimensions(&self) -> &[String] {
        &self.dimensions
    }

    fn attribute_string(&self, key: &str) -> Option<String> {
        self.file_attribute_string(key)
            .or_else(|| self.declared.get(key).cloned())
    }

    fn file_attribute_string(&self, key: &str) -> Option<String> {
        match &self.source {
            NcSource::Indexed(variable) => variable
                .attribute(key)
                .and_then(|a| a.as_string().map(str::to_owned)),
            NcSource::DimensionScale { attributes, .. } => attributes.get(key).cloned(),
        }
    }

    /// A numeric attribute, or `None` on a recovered dimension scale.
    ///
    /// Not an oversight and not a zero/one assumption: the raw-HDF5
    /// by-name escape hatch reads strings only, so numeric CF attributes
    /// are genuinely unreachable there.  `rw_netcdf` has the same hole in
    /// the same place, and the Python engine consumes `rw_netcdf`, so
    /// answering `None` here is what keeps the two engines identical.
    /// Coordinate axes are not packed in any format this reads, which is
    /// why the hole has never cost a number.
    fn attribute_number(&self, key: &str) -> Option<f64> {
        match &self.source {
            NcSource::Indexed(variable) => {
                variable.attribute(key).and_then(netcrust::Attribute::as_f64)
            }
            NcSource::DimensionScale { .. } => None,
        }
    }

    fn array_f64(&self) -> netcrust::Result<netcrust::DataArray> {
        match &self.source {
            NcSource::Indexed(variable) => variable.array_f64(),
            NcSource::DimensionScale { file, .. } => file.read_array_f64(&self.name),
        }
    }

    /// The stored type, when the NetCDF index reports one.
    ///
    /// A recovered dimension scale has none: the raw-HDF5 index does not
    /// expose it, and claiming `F64` would be a claim about storage this
    /// path cannot make.
    fn dtype(&self) -> Option<&netcrust::DataType> {
        match &self.source {
            NcSource::Indexed(variable) => Some(variable.dtype()),
            NcSource::DimensionScale { .. } => None,
        }
    }
}

fn attribute_string(variable: &NcVariable, key: &str) -> Option<String> {
    variable.attribute_string(key)
}

fn attribute_number(variable: &NcVariable, key: &str) -> Option<f64> {
    variable.attribute_number(key)
}

/// `mapped_source._attributes_match`.
fn attributes_match(variable: &NcVariable, selector: &Node) -> bool {
    let Some(declared) = selector.field("attributes") else {
        return true;
    };
    for (key, expected) in declared.entries() {
        match expected {
            Node::String(text) => {
                if attribute_string(variable, key).as_deref() != Some(text.as_str()) {
                    return false;
                }
            }
            other => {
                let Some(wanted) = other.as_f64() else {
                    return false;
                };
                match attribute_number(variable, key) {
                    Some(observed) if observed == wanted => {}
                    _ => return false,
                }
            }
        }
    }
    true
}

fn selector_names(selector: &Node) -> Vec<String> {
    selector
        .field("name")
        .and_then(Node::as_string_list)
        .unwrap_or_default()
}

/// `mapped_source._match_nc_variables`: name first, CF standard name as the
/// reported rescue.  Requiring BOTH would defeat the standard name.
fn match_variables<'a>(
    variables: &'a [NcVariable],
    selector: &Node,
) -> Vec<(&'a NcVariable, &'static str)> {
    let accepted = selector_names(selector);
    let by_name: Vec<&NcVariable> = variables
        .iter()
        .filter(|variable| {
            accepted.iter().any(|name| name == variable.name()) && attributes_match(variable, selector)
        })
        .collect();
    if !by_name.is_empty() {
        return by_name
            .into_iter()
            .map(|variable| (variable, EVIDENCE_NAME))
            .collect();
    }
    let Some(expected) = selector.field("standard_name").and_then(Node::as_str) else {
        return Vec::new();
    };
    variables
        .iter()
        .filter(|variable| {
            attribute_string(variable, "standard_name").as_deref() == Some(expected)
                && attributes_match(variable, selector)
        })
        .map(|variable| (variable, EVIDENCE_STANDARD_NAME))
        .collect()
}

/// What the file actually offers, for both-vocabularies refusals.
fn vocabulary(variables: &[NcVariable]) -> String {
    let mut entries: Vec<String> = variables
        .iter()
        .map(|variable| match attribute_string(variable, "standard_name") {
            Some(standard) => format!("{} (standard_name={standard})", variable.name()),
            None => variable.name().to_owned(),
        })
        .collect();
    entries.sort();
    if entries.len() > 40 {
        let total = entries.len();
        entries.truncate(40);
        format!("{}, ... ({total} total)", entries.join(", "))
    } else {
        entries.join(", ")
    }
}

/// `repr()` of an optional string: `'K'`, or the bare word `None`.
fn optional_repr(value: Option<&str>) -> String {
    match value {
        Some(text) => crate::refusal::python_repr(text),
        None => "None".to_owned(),
    }
}

/// `repr()` of one JSON value, for a selector's declared attributes.
fn node_repr(value: &Node) -> String {
    match value {
        Node::String(text) => crate::refusal::python_repr(text),
        Node::Bool(true) => "True".to_owned(),
        Node::Bool(false) => "False".to_owned(),
        Node::Null => "None".to_owned(),
        Node::Integer(number) => number.to_string(),
        other => other
            .as_f64()
            .map(crate::refusal::python_float_repr)
            .unwrap_or_else(|| "None".to_owned()),
    }
}

/// `mapped_source._nc_selector_text`: what the mapping asked for.
///
/// The attribute and layer clauses are not decoration: they are how a
/// reader tells four soil selectors apart in a refusal that would
/// otherwise print the same variable name four times.
fn selector_text(selector: &Node) -> String {
    let names = selector_names(selector);
    let mut parts: Vec<String> = Vec::new();
    if !names.is_empty() {
        parts.push(if names.len() == 1 {
            format!("name={}", crate::refusal::python_repr(&names[0]))
        } else {
            format!("name in {}", crate::refusal::python_list_repr(&names))
        });
    }
    if let Some(standard) = selector.field("standard_name").and_then(Node::as_str) {
        parts.push(format!(
            "standard_name={}",
            crate::refusal::python_repr(standard)
        ));
    }
    let mut text = if parts.is_empty() {
        "<empty selector>".to_owned()
    } else {
        parts.join(" or ")
    };
    if let Some(attributes) = selector.field("attributes") {
        let mut entries: Vec<&(String, Node)> = attributes.entries().iter().collect();
        if !entries.is_empty() {
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let rendered: Vec<String> = entries
                .iter()
                .map(|(key, value)| format!("{key}={}", node_repr(value)))
                .collect();
            text.push_str(" with ");
            text.push_str(&rendered.join(", "));
        }
    }
    if let Some(value) = selector.field("layer_value") {
        let dimension = selector
            .field("layer_dimension")
            .and_then(Node::as_str)
            .unwrap_or_default();
        let units = selector
            .field("layer_units")
            .and_then(Node::as_str)
            .unwrap_or_default();
        text.push_str(&format!(
            " at {dimension}={} {units}",
            json_number_str(value)
        ));
    }
    text
}

/// `mapped_source._resolve_nc_variable`: exactly one variable, or a
/// refusal that prints both vocabularies.
fn resolve_variable<'a>(
    variables: &'a [NcVariable],
    selector: &Node,
    label: &str,
) -> Result<&'a NcVariable> {
    let matched = match_variables(variables, selector);
    if matched.len() == 1 {
        return Ok(matched[0].0);
    }
    // Naming only the count made a mapping/producer mismatch read as a
    // data fault.  Both vocabularies go in the message: what the mapping
    // asked for, and what this file actually contains.  Two matches never
    // collapse to a guess, both are named.
    let ambiguous = if matched.len() < 2 {
        String::new()
    } else {
        let mut names: Vec<&str> = matched.iter().map(|(variable, _)| variable.name()).collect();
        names.sort_unstable();
        format!("\n  ambiguous, all of: {}", names.join(", "))
    };
    Err(crate::refusal::selector_unmatched(format!(
        "{label} selector resolved {} NetCDF variables; expected exactly \
         one.{ambiguous}\n  mapping asked for: {}\n  file contains: {}",
        matched.len(),
        selector_text(selector),
        vocabulary(variables)
    )))
}

/// CF decoding, transcribed from `rw_netcdf`'s dump path.
struct CfPolicy {
    scale_factor: Option<f64>,
    add_offset: Option<f64>,
    fill_value: Option<f64>,
    missing_value: Option<f64>,
}

impl CfPolicy {
    fn of(variable: &NcVariable) -> Self {
        Self {
            scale_factor: attribute_number(variable, "scale_factor"),
            add_offset: attribute_number(variable, "add_offset"),
            fill_value: attribute_number(variable, "_FillValue"),
            missing_value: attribute_number(variable, "missing_value"),
        }
    }

    /// Mask against the STORED representation, THEN scale.  Applying the
    /// packing first would compare a physical quantity against a sentinel
    /// that was never in that space.
    fn apply(&self, values: &mut [f64]) -> Vec<bool> {
        let mut mask = vec![false; values.len()];
        // Coordinates are tiny; fields can contain millions of stored values.
        // Use the engine's bounded pool without changing the per-cell order
        // of masking, multiplication and addition (never a fused mul_add).
        const CHUNK: usize = 32 * 1024;
        if values.len() >= 2 * CHUNK {
            let chunk = values
                .len()
                .div_ceil(crate::threads::width().saturating_mul(4))
                .max(CHUNK);
            crate::threads::install(|| {
                values
                    .par_chunks_mut(chunk)
                    .zip(mask.par_chunks_mut(chunk))
                    .for_each(|(values, mask)| self.apply_chunk(values, mask));
            });
        } else {
            self.apply_chunk(values, &mut mask);
        }
        mask
    }

    fn apply_chunk(&self, values: &mut [f64], mask: &mut [bool]) {
        for (position, value) in values.iter_mut().enumerate() {
            let is_missing = self.fill_value.is_some_and(|fill| *value == fill)
                || self.missing_value.is_some_and(|marker| *value == marker);
            if is_missing || !value.is_finite() {
                mask[position] = is_missing;
                *value = f64::NAN;
                continue;
            }
            if let Some(scale) = self.scale_factor {
                *value *= scale;
            }
            if let Some(offset) = self.add_offset {
                *value += offset;
            }
        }
    }
}

fn read_values(variable: &NcVariable) -> Result<(Vec<usize>, Vec<f64>)> {
    let (shape, values, _mask) = read_values_masked(variable)?;
    Ok((shape, values))
}

/// The same read, keeping the CF mask the coordinate contract needs.
///
/// A coordinate whose values were MASKED and one whose values are merely
/// non-finite are different faults with different sentences, and the mask
/// is the only thing that tells them apart after the fill markers have
/// become NaN.
fn read_values_masked(variable: &NcVariable) -> Result<(Vec<usize>, Vec<f64>, Vec<bool>)> {
    // A flavor this decode does not read, named before the reader
    // produces a confusing type error from several layers down.  netcrust
    // promotes numeric variables to f64 and exposes no character or
    // string read at all, so a mapping that selects WRF's `Times` or
    // ERA5's `expver` as a field cannot be served, and the Python engine
    // reaches the same wall one layer later, as numpy's "could not
    // convert string to float", which names neither the variable nor the
    // reason.  Same refusal class, better sentence.
    if let Some(dtype @ (netcrust::DataType::Char | netcrust::DataType::String)) = variable.dtype()
    {
        return Err(decode_failed(format!(
            "NetCDF variable {} is a {dtype:?} variable; this engine decodes \
             numeric variables only, because the vendored netcrust reader \
             promotes to f64 and exposes no character read. Select a numeric \
             variable, or read this one another way.",
            crate::refusal::python_repr(variable.name())
        )));
    }
    let array = variable.array_f64().map_err(|error| {
        decode_failed(format!(
            "NetCDF variable {} did not decode: {error}",
            variable.name()
        ))
    })?;
    let shape = array.shape().to_vec();
    let mut values = array.into_values();
    let mask = CfPolicy::of(variable).apply(&mut values);
    Ok((shape, values, mask))
}

/// `mapped_source._nc_coordinate_values`.
fn coordinate_values(variable: &NcVariable, expected_units: Option<&str>, label: &str) -> Result<Vec<f64>> {
    if let Some(expected) = expected_units {
        let observed = attribute_string(variable, "units");
        if observed.as_deref() != Some(expected) {
            return Err(mapping_invalid(format!(
                "{label} units {} differ from mapping {}",
                optional_repr(observed.as_deref()),
                crate::refusal::python_repr(expected)
            )));
        }
    }
    let (shape, values, mask) = read_values_masked(variable)?;
    if mask.iter().any(|flag| *flag) {
        return Err(frame_invalid(format!(
            "{label} coordinate contains missing values"
        )));
    }
    if shape.len() != 1 || values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(frame_invalid(format!(
            "{label} coordinate must be finite non-empty 1-D"
        )));
    }
    Ok(values)
}

/// `mapped_source._require_geographic_coordinate` + `_require_geographic_range`.
fn require_geographic(variable: &NcVariable, values: &[f64], axis: &str) -> Result<()> {
    let accepted: &[&str] = if axis == "latitude" {
        &CF_DEGREES_NORTH
    } else {
        &CF_DEGREES_EAST
    };
    let canonical = if axis == "latitude" {
        "degrees_north"
    } else {
        "degrees_east"
    };
    let units = attribute_string(variable, "units");
    let standard_name = attribute_string(variable, "standard_name");
    if let Some(standard) = standard_name.as_deref()
        && CF_PROJECTION_STANDARD_NAMES.contains(&standard) {
            return Err(frame_invalid(format!(
                "{axis} selector resolved NetCDF variable {} whose CF \
                 standard_name is {}; that is a projection axis, not \
                 geographic {axis}. A projected source needs a declared \
                 Lambert grid in the mapping; otherwise supply geographic \
                 coordinate variables with units {}.",
                crate::refusal::python_repr(variable.name()),
                crate::refusal::python_repr(standard),
                crate::refusal::python_repr(canonical)
            )));
        }
    let units_ok = units
        .as_deref()
        .is_some_and(|value| accepted.contains(&value));
    let standard_ok = standard_name.as_deref() == Some(axis);
    if !units_ok && !standard_ok {
        return Err(frame_invalid(format!(
            "{axis} selector resolved NetCDF variable {} with units={} \
             and standard_name={}; RW-WPS cannot confirm it holds \
             geographic {axis} in degrees. Declare CF units {} (or one of {}) \
             or standard_name={} on that variable.",
            crate::refusal::python_repr(variable.name()),
            optional_repr(units.as_deref()),
            optional_repr(standard_name.as_deref()),
            crate::refusal::python_repr(canonical),
            crate::refusal::python_list_repr(accepted),
            crate::refusal::python_repr(axis)
        )));
    }
    let limit = if axis == "latitude" { 90.0 } else { 360.0 };
    let extreme = values.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
    if extreme > limit {
        return Err(frame_invalid(format!(
            "{axis} coordinate reaches {}, outside the valid geographic \
             {axis} range +/-{} degrees; the source grid is not a regular \
             latitude/longitude grid.",
            python_format_g(extreme),
            python_format_g(limit)
        )));
    }
    Ok(())
}

/// One f64 as Python's `f"{value:g}"` writes it.
///
/// `%g` is six significant digits, trailing zeros stripped, switching to
/// exponent form below 1e-4 or at 1e6 and above.  Rust's `{}` never
/// rounds to six and never switches; `{:e}` spells `1e6` where Python
/// spells `1e+06`.
fn python_format_g(value: f64) -> String {
    if !value.is_finite() {
        return crate::refusal::python_float_repr(value);
    }
    if value == 0.0 {
        return "0".to_owned();
    }
    let exponent = value.abs().log10().floor() as i32;
    let trim = |text: String| -> String {
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            text
        }
    };
    if !(-4..6).contains(&exponent) {
        let mantissa = trim(format!("{:.5}", value / 10f64.powi(exponent)));
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    trim(format!("{:.*}", (5 - exponent).max(0) as usize, value))
}

/// `mapped_source._cf_grid_mapping_names`, refusing anything but a plain
/// geographic grid.
/// `mapped_source._cf_grid_mapping_names`: declared projections, and the
/// variables that claim each one.
///
/// A container is discovered two ways (a data variable naming it through
/// `grid_mapping`, and a variable carrying `grid_mapping_name` directly)
/// and both are reported so a refusal can name the projection AND the
/// variables that claim it.  Naming only the projection left a reader
/// hunting through a 200-variable file for which field brought it in.
fn cf_grid_mapping_names(variables: &[NcVariable]) -> BTreeMap<String, Vec<String>> {
    let mut declared: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for variable in variables {
        if let Some(container_name) = attribute_string(variable, "grid_mapping")
            && !container_name.trim().is_empty() {
                // CF-1.7 extended syntax: "crs: x y", the container is
                // token 0, and the colon is a separator rather than part
                // of the name.
                let token = container_name
                    .split(':')
                    .next()
                    .unwrap_or_default()
                    .split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .to_owned();
                // Looked up in the SAME inventory the selectors see,
                // recovered dimension scales included: that is the set
                // `dataset.variables` holds on the Python side.
                let projection = variables
                    .iter()
                    .find(|item| item.name() == token)
                    .and_then(|container| attribute_string(container, "grid_mapping_name"));
                let key = projection.unwrap_or_else(|| {
                    format!("<unresolved {}>", crate::refusal::python_repr(&token))
                });
                declared
                    .entry(key)
                    .or_default()
                    .push(variable.name().to_owned());
            }
        if let Some(projection) = attribute_string(variable, "grid_mapping_name")
            && !projection.trim().is_empty() {
                declared.entry(projection).or_default();
            }
    }
    declared
}

fn require_supported_grid_mapping(variables: &[NcVariable], label: &str) -> Result<()> {
    let declared = cf_grid_mapping_names(variables);
    let unsupported: Vec<(&String, &Vec<String>)> = declared
        .iter()
        .filter(|(name, _)| !SUPPORTED_GRID_MAPPING_NAMES.contains(&name.as_str()))
        .collect();
    if !unsupported.is_empty() {
        let detail: Vec<String> = unsupported
            .iter()
            .map(|(projection, claimants)| {
                let mut names = (*claimants).clone();
                names.sort();
                names.truncate(6);
                if names.is_empty() {
                    crate::refusal::python_repr(projection)
                } else {
                    format!(
                        "{} (used by {})",
                        crate::refusal::python_repr(projection),
                        names.join(", ")
                    )
                }
            })
            .collect();
        return Err(frame_invalid(format!(
            "{label} declares CF grid mapping(s) {}. RW-WPS mapped \
             geographic NetCDF path requires regular latitude/longitude grids \
             (grid_mapping_name={} or no grid mapping at all). A \
             lambert_conformal_conic source needs a declared Lambert grid \
             in the mapping; other projections need regridding before this \
             geographic path can decode them.",
            detail.join("; "),
            crate::refusal::python_repr(SUPPORTED_GRID_MAPPING_NAMES[0])
        )));
    }
    Ok(())
}

/// Resolve geographic axes or a declared rectilinear Lambert grid. Auxiliary
/// lat/lon arrays are validation evidence, never treated as separable axes.
fn horizontal_coordinates(
    mapping: &Mapping, variables: &[NcVariable], lat: &NcVariable,
    lon: &NcVariable, source: &str,
) -> Result<(Vec<f64>, Vec<f64>, String, String)> {
    let declaration = mapping.grid_declaration()?;
    if !declaration.is_lambert() {
        require_supported_grid_mapping(variables, source)?;
        let latitude = coordinate_values(lat, None, "latitude")?;
        let longitude = coordinate_values(lon, None, "longitude")?;
        require_geographic(lat, &latitude, "latitude")?;
        require_geographic(lon, &longitude, "longitude")?;
        return Ok((latitude, longitude, coordinate_dimension(lat, "latitude")?, coordinate_dimension(lon, "longitude")?));
    }
    validate_cf_lambert(variables, &declaration)?;
    let parameters = declaration.parameters.as_ref().unwrap();
    if parameters.nx <= 0 || parameters.ny <= 0 {
        return Err(mapping_invalid("Lambert NetCDF dimensions must be positive"));
    }
    if lat.dimensions().len() == 1 && lon.dimensions().len() == 1 {
        let crs = variables.iter().find(|variable|
            attribute_string(variable, "grid_mapping_name").as_deref() == Some("lambert_conformal_conic"))
            .ok_or_else(|| frame_invalid("projected NetCDF axes require a CF lambert_conformal_conic grid_mapping"))?;
        let y = coordinate_values(lat, Some("m"), "projection y")?;
        let x = coordinate_values(lon, Some("m"), "projection x")?;
        if y.len() != parameters.ny as usize || x.len() != parameters.nx as usize
            || attribute_string(lat, "standard_name").as_deref() != Some("projection_y_coordinate")
            || attribute_string(lon, "standard_name").as_deref() != Some("projection_x_coordinate")
            || lat.dimensions()[0] == lon.dimensions()[0] {
            return Err(frame_invalid("CF Lambert axes must declare projection_y_coordinate/projection_x_coordinate and mapping dimensions"));
        }
        let origin = attribute_number(crs, "latitude_of_projection_origin")
            .ok_or_else(|| frame_invalid("CF Lambert grid_mapping needs latitude_of_projection_origin"))?;
        let grid = crate::lambert::LambertGrid::new(origin, crate::lambert::wrap180(parameters.lov),
            parameters.latin1, parameters.latin2, crate::lambert::wrap180(parameters.lov),
            parameters.dx_m * crate::lambert::EARTH_RADIUS_M / parameters.earth_radius_m, 1.0, 1.0);
        let (i, j) = grid.latlon_to_ij(parameters.lat1, crate::lambert::wrap180(parameters.lon1));
        let first_x = (i-1.0)*parameters.dx_m + attribute_number(crs, "false_easting").unwrap_or(0.0);
        let first_y = (j-1.0)*parameters.dx_m + attribute_number(crs, "false_northing").unwrap_or(0.0);
        if x.iter().enumerate().any(|(k,value)| (*value-first_x-k as f64*parameters.dx_m).abs() > parameters.dx_m*0.1)
            || y.iter().enumerate().any(|(k,value)| (*value-first_y-k as f64*parameters.dy_m).abs() > parameters.dy_m*0.1) {
            return Err(frame_invalid("CF Lambert projected axes disagree with the declared source window"));
        }
        let (y, x) = parameters.projected_axes();
        return Ok((y, x, lat.dimensions()[0].clone(), lon.dimensions()[0].clone()));
    }
    let (lat_shape, latitude, lat_mask) = read_values_masked(lat)?;
    let (lon_shape, longitude, lon_mask) = read_values_masked(lon)?;
    let expected = vec![parameters.ny as usize, parameters.nx as usize];
    if lat_shape != expected || lon_shape != expected || lat.dimensions() != lon.dimensions()
        || lat.dimensions().len() != 2 || lat.dimensions()[0] == lat.dimensions()[1] {
        return Err(frame_invalid("Lambert NetCDF latitude/longitude must share declared (y,x) dimensions and shape"));
    }
    if lat_mask.iter().chain(&lon_mask).any(|missing| *missing)
        || latitude.iter().chain(&longitude).any(|value| !value.is_finite()) {
        return Err(frame_invalid("Lambert NetCDF latitude/longitude contains missing or nonfinite coordinates"));
    }
    require_geographic(lat, &latitude, "latitude")?;
    require_geographic(lon, &longitude, "longitude")?;
    validate_lambert_points(parameters, &latitude, &longitude)?;
    let (y, x) = parameters.projected_axes();
    Ok((y, x, lat.dimensions()[0].clone(), lat.dimensions()[1].clone()))
}

fn validate_lambert_points(parameters: &crate::model::LambertParameters, latitude: &[f64], longitude: &[f64]) -> Result<()> {
    let grid = crate::lambert::LambertGrid::from_declaration(parameters);
    let nx = parameters.nx as usize;
    // A tenth of one cell admits float32 geographic coordinates while
    // rejecting wrong windows, swapped axes, reversed scans and curvilinear grids.
    let tolerance_deg = parameters.dx_m.min(parameters.dy_m) * 0.1 / 111_000.0;
    let bad = crate::threads::install(|| latitude.par_iter().zip(longitude).enumerate().find_first(|(i, (lat, lon))| {
        let (expected_lat, expected_lon) = grid.ij_to_latlon((i % nx + 1) as f64,
            (i / nx) as f64 * parameters.dy_m / parameters.dx_m + 1.0);
        (**lat - expected_lat).abs() > tolerance_deg
            || crate::lambert::wrap180(**lon - expected_lon).abs() > tolerance_deg
    }));
    if let Some((index, _)) = bad {
        return Err(frame_invalid(format!("NetCDF geographic coordinate at y={}, x={} differs from declared Lambert grid by more than a tenth cell", index / nx, index % nx)));
    }
    Ok(())
}

/// CF metadata, when present, must agree with the mapping's source-window
/// contract. Files with only 2-D lat/lon use that same contract directly.
fn validate_cf_lambert(variables: &[NcVariable], declaration: &GridDeclaration) -> Result<()> {
    let names = cf_grid_mapping_names(variables);
    if names.keys().any(|name| name != "lambert_conformal_conic") {
        return Err(frame_invalid("declared Lambert NetCDF grid conflicts with file grid_mapping"));
    }
    let p = declaration.parameters.as_ref().unwrap();
    for crs in variables.iter().filter(|variable|
        attribute_string(variable, "grid_mapping_name").as_deref() == Some("lambert_conformal_conic")) {
        let parallels = match &crs.source {
            NcSource::Indexed(variable) => variable.attribute("standard_parallel").and_then(|a| a.value().as_f64_vec()),
            _ => None,
        }.ok_or_else(|| frame_invalid("CF Lambert grid_mapping needs standard_parallel"))?;
        if parallels.is_empty() || parallels.len() > 2 || (parallels[0] - p.latin1).abs() > 1e-6
            || (parallels[parallels.len()-1] - p.latin2).abs() > 1e-6 {
            return Err(frame_invalid("CF Lambert standard_parallel differs from mapping grid"));
        }
        let lov = attribute_number(crs, "longitude_of_central_meridian")
            .ok_or_else(|| frame_invalid("CF Lambert grid_mapping needs longitude_of_central_meridian"))?;
        if crate::lambert::wrap180(lov - p.lov).abs() > 1e-6 {
            return Err(frame_invalid("CF Lambert central meridian differs from mapping grid"));
        }
        // The mapped projection uses a sphere. An ellipsoidal CF CRS must
        // not silently inherit its radius from a spherical mapping.
        let radius = attribute_number(crs, "earth_radius").or_else(|| {
            let major = attribute_number(crs, "semi_major_axis")?;
            let inverse = attribute_number(crs, "inverse_flattening");
            let minor = attribute_number(crs, "semi_minor_axis");
            if inverse == Some(0.0) || minor == Some(major) { Some(major) } else { None }
        }).ok_or_else(|| frame_invalid("CF Lambert grid_mapping must declare a spherical earth_radius"))?;
        if (radius - p.earth_radius_m).abs() > 1.0 {
            return Err(frame_invalid("CF Lambert earth radius differs from mapping grid"));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CfUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
}

#[derive(Debug)]
struct CfReference {
    unit: CfUnit,
    epoch: DateTime<Utc>,
}

/// `rw_netcdf::parse_cf_units`: strict about the unit, forgiving about
/// the timestamp spelling, because that is where real files vary:
/// `1970-01-01`, `1970-01-01 00:00:0.0`, `1970-01-01T00:00:00Z`, and a
/// trailing `+05:30` or ` UTC` all appear in published archives.  A
/// well-formed UTC offset is APPLIED (the epoch converts to UTC) so
/// the decoded instants land where the equivalent UTC-spelled units
/// would put them.
///
/// `Ok(None)` when the units string is not a CF reference time at all
/// (the common case: `K`, `m s-1`); `Err` when it plainly is one but
/// its UTC-offset designator is malformed, because decoding such a
/// source with the designator ignored would silently move every
/// instant read off the time axis.
fn parse_cf_units(units: &str) -> Result<Option<CfReference>> {
    let lowered = units.trim().to_ascii_lowercase();
    let Some((unit_text, rest)) = lowered.split_once(" since ") else {
        return Ok(None);
    };
    let unit = match unit_text.trim() {
        "s" | "sec" | "secs" | "second" | "seconds" => CfUnit::Seconds,
        "min" | "mins" | "minute" | "minutes" => CfUnit::Minutes,
        "h" | "hr" | "hrs" | "hour" | "hours" => CfUnit::Hours,
        "d" | "day" | "days" => CfUnit::Days,
        _ => return Ok(None),
    };
    Ok(parse_cf_epoch(rest.trim())?.map(|epoch| CfReference { unit, epoch }))
}

/// The widest UTC-offset designator accepted, in seconds: 18 hours
/// either side.  The widest civil offset on Earth is +14:00 (the Line
/// Islands) and the udunits/`java.time` grammars both cap the field at
/// +/-18:00; a larger number is a mistyped timestamp, not a timezone,
/// and applying it would move every decoded instant by most of a day.
const MAX_UTC_OFFSET_SECONDS: i64 = 18 * 3600;

/// `Ok(None)` when the text is not a timestamp this parser reads;
/// `Err` when its UTC-offset designator cannot be trusted, saying why.
fn parse_cf_epoch(text: &str) -> Result<Option<DateTime<Utc>>> {
    let mut stamp = text.trim();
    // Textual zone designators, every one of which NAMES the zero
    // offset.  udunits reads them; so do published archives.
    let mut named_utc = false;
    for suffix in [" utc", "z", " gmt"] {
        if let Some(head) = stamp.strip_suffix(suffix) {
            stamp = head.trim();
            named_utc = true;
        }
    }
    let body = stamp.replace('t', " ");
    let mut tokens = body.split_whitespace();
    let Some(date_text) = tokens.next() else {
        return Ok(None);
    };
    // A signed token is a numeric UTC offset -- a time of day is never
    // signed.  The offset may ride attached to the tail of the time
    // token (`00:00:00+05:30`) or stand alone after the date or time
    // (`1992-10-8 15:15:42.5 -6:00`, the canonical udunits spelling).
    // It is never recognised attached to a bare date, whose own `-`
    // separators make that spelling unreadable without guessing.
    let (time_text, offset_text) = match (tokens.next(), tokens.next(), tokens.next()) {
        (None, _, _) => ("", None),
        (Some(second), None, _) => match second.find(['+', '-']) {
            Some(at) => (&second[..at], Some(&second[at..])),
            None => (second, None),
        },
        (Some(second), Some(third), None) => {
            if !third.starts_with(['+', '-']) {
                return Ok(None);
            }
            if second.contains(['+', '-']) {
                return Err(mapping_invalid(format!(
                    "reference time {text:?} carries two UTC offsets; \
                     one epoch cannot be shifted twice"
                )));
            }
            (second, Some(third))
        }
        _ => return Ok(None),
    };
    let offset_seconds = match offset_text {
        Some(token) => {
            if named_utc {
                return Err(mapping_invalid(format!(
                    "reference time {text:?} names UTC and also carries \
                     the numeric offset {token:?}; an epoch with two zone \
                     designators cannot be placed on the timeline"
                )));
            }
            parse_utc_offset(token, text)?
        }
        None => 0,
    };
    let date = match NaiveDate::parse_from_str(date_text, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(date_text, "%Y-%-m-%-d"))
    {
        Ok(date) => date,
        Err(_) => return Ok(None),
    };
    let time = if time_text.is_empty() {
        NaiveTime::from_hms_opt(0, 0, 0).expect("midnight is a valid time")
    } else {
        match ["%H:%M:%S%.f", "%H:%M:%S", "%H:%M", "%H"]
            .iter()
            .find_map(|format| NaiveTime::parse_from_str(time_text, format).ok())
        {
            Some(time) => time,
            None => return Ok(None),
        }
    };
    // A timestamp at +05:00 reads five hours AHEAD of UTC, so the UTC
    // instant it names is the naive reading minus the offset.
    let epoch = NaiveDateTime::new(date, time) - Duration::seconds(offset_seconds);
    Ok(Some(epoch.and_utc()))
}

/// One numeric UTC-offset designator -- a sign, then `hh:mm`, `hhmm`,
/// or a bare hour count `h`/`hh` -- as signed seconds east of UTC.
/// These are the spellings udunits reads.  Anything else is refused
/// with the reason, because a mis-read offset does not crash: it
/// silently moves every instant decoded from the source.
fn parse_utc_offset(token: &str, timestamp: &str) -> Result<i64> {
    let malformed = |why: String| {
        mapping_invalid(format!(
            "UTC offset {token:?} in reference time {timestamp:?} is \
             malformed: {why}"
        ))
    };
    let (sign, magnitude) = if let Some(rest) = token.strip_prefix('+') {
        (1i64, rest)
    } else if let Some(rest) = token.strip_prefix('-') {
        (-1i64, rest)
    } else {
        return Err(malformed("it does not start with a sign".into()));
    };
    if magnitude.is_empty() {
        return Err(malformed("the sign has no digits behind it".into()));
    }
    let digits = |text: &str| text.bytes().all(|byte| byte.is_ascii_digit());
    let (hour_text, minute_text) = match magnitude.split_once(':') {
        Some((hours, minutes)) => {
            if minutes.len() != 2 {
                return Err(malformed(format!(
                    "minutes must be exactly two digits, got {minutes:?}; \
                     an offset carries no seconds field"
                )));
            }
            (hours, minutes)
        }
        None => match magnitude.len() {
            1 | 2 => (magnitude, "0"),
            4 => magnitude.split_at(2),
            _ => {
                return Err(malformed(format!(
                    "a colonless offset must be one or two hour digits or \
                     exactly four digits (hhmm); {magnitude:?} cannot be \
                     split into hours and minutes without guessing"
                )));
            }
        },
    };
    if hour_text.is_empty() || hour_text.len() > 2 || !digits(hour_text) || !digits(minute_text) {
        return Err(malformed(format!(
            "{hour_text:?} hours and {minute_text:?} minutes are not one- \
             or two-digit numbers"
        )));
    }
    let hours: i64 = hour_text.parse().expect("checked ascii digits");
    let minutes: i64 = minute_text.parse().expect("checked ascii digits");
    if minutes >= 60 {
        return Err(malformed(format!(
            "{minutes} is not a minute count below 60"
        )));
    }
    let magnitude_seconds = hours * 3600 + minutes * 60;
    if magnitude_seconds > MAX_UTC_OFFSET_SECONDS {
        return Err(malformed(
            "its magnitude passes 18:00, and no timezone is that far from \
             UTC (the widest civil offset on Earth is +14:00)"
            .into(),
        ));
    }
    Ok(sign * magnitude_seconds)
}

fn decode_times(reference: &CfReference, values: &[f64]) -> Result<Vec<NaiveDateTime>> {
    values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                return Err(decode_failed(format!(
                    "time coordinate is not finite: {value}"
                )));
            }
            // Whole seconds and the nanosecond remainder separately: an
            // offset from an epoch two thousand years back (Earth2Studio's
            // "hours since 0001-01-01") is about 6e19 ns, past i64, while
            // its seconds fit with room to spare.
            let seconds = match reference.unit {
                CfUnit::Seconds => *value,
                CfUnit::Minutes => value * 60.0,
                CfUnit::Hours => value * 3600.0,
                CfUnit::Days => value * 86_400.0,
            };
            if seconds.abs() >= 1e15 {
                return Err(decode_failed(format!(
                    "time coordinate out of representable range: {value}"
                )));
            }
            let whole = seconds.floor();
            let nanos = ((seconds - whole) * 1e9).round() as i64;
            reference
                .epoch
                .checked_add_signed(
                    Duration::seconds(whole as i64) + Duration::nanoseconds(nanos),
                )
                .map(|instant| instant.naive_utc())
                .ok_or_else(|| decode_failed(format!("time coordinate out of range: {value}")))
        })
        .collect()
}

/// One variable's own dimension name for a coordinate.
fn coordinate_dimension(variable: &NcVariable, label: &str) -> Result<String> {
    let dimensions = variable.dimensions();
    if dimensions.len() != 1 {
        return Err(frame_invalid(format!(
            "{label} coordinate variable '{}' must have exactly one dimension",
            variable.name()
        )));
    }
    Ok(dimensions[0].clone())
}

/// `mapped_source._declared_level_indices`.
fn declared_level_indices(declared: &[f64], offered: &[f64], label: &str) -> Result<Option<Vec<usize>>> {
    if declared == offered {
        return Ok(None);
    }
    let mut positions = Vec::with_capacity(declared.len());
    for value in declared {
        let matches: Vec<usize> = offered
            .iter()
            .enumerate()
            .filter(|(_index, level)| *level == value)
            .map(|(index, _)| index)
            .collect();
        if matches.len() != 1 {
            return Err(selector_level_refusal(label, offered, *value, matches.len(), declared));
        }
        positions.push(matches[0]);
    }
    Ok(Some(positions))
}

fn selector_level_refusal(
    label: &str,
    offered: &[f64],
    value: f64,
    count: usize,
    declared: &[f64],
) -> crate::refusal::Refusal {
    crate::refusal::selector_unmatched(format!(
        "{label} offers vertical levels {}, in which the mapping's \
         declared level {} appears {count} times; every declared level \
         must appear exactly once (declared: {})",
        crate::refusal::python_float_list_repr(offered),
        crate::refusal::python_float_repr(value),
        crate::refusal::python_float_list_repr(declared)
    ))
}

/// `mapped_source._layer_slice_position`: which slice of a producer's own
/// layer dimension this selector addresses.
///
/// `(axis, index)` in the variable's own dimension order, or `None` when
/// the selector addresses the whole variable.  The index is found by
/// matching the declared value on the layer coordinate VARIABLE (never
/// by position) so a producer that reorders its layers between cycles
/// cannot silently swap two of them under a mapping that names depths.
fn layer_slice_position(
    variables: &[NcVariable],
    variable: &NcVariable,
    selector: &Node,
    field_name: &str,
) -> Result<Option<(usize, usize)>> {
    let Some(declared) = selector.field("layer_value") else {
        return Ok(None);
    };
    let wanted = declared.as_f64().ok_or_else(|| {
        mapping_invalid(format!(
            "fields.{field_name} selector layer_value must be a number"
        ))
    })?;
    let dimension = selector
        .field("layer_dimension")
        .and_then(Node::as_str)
        .ok_or_else(|| {
            mapping_invalid(format!(
                "fields.{field_name} selector layer_value needs a layer_dimension"
            ))
        })?;
    let dimensions = variable.dimensions();
    let Some(axis) = dimensions.iter().position(|item| item == dimension) else {
        return Err(frame_invalid(format!(
            "{field_name} selector addresses layer dimension '{dimension}', \
             which variable '{}' does not use (it uses {})",
            variable.name(),
            crate::refusal::python_list_repr(dimensions)
        )));
    };
    let Some(coordinate) = variables.iter().find(|item| item.name() == dimension) else {
        return Err(frame_invalid(format!(
            "{field_name} selector addresses layer dimension '{dimension}' by \
             value, but that dimension carries no coordinate variable to read \
             the value from"
        )));
    };
    let units = selector
        .field("layer_units")
        .and_then(Node::as_str)
        .ok_or_else(|| {
            mapping_invalid(format!(
                "fields.{field_name} selector layer_value needs layer_units"
            ))
        })?;
    let values = coordinate_values(
        coordinate,
        Some(units),
        &format!("{field_name} layer coordinate"),
    )?;
    let matches: Vec<usize> = values
        .iter()
        .enumerate()
        .filter(|(_index, value)| **value == wanted)
        .map(|(index, _value)| index)
        .collect();
    if matches.len() != 1 {
        return Err(frame_invalid(format!(
            "{field_name} selector layer_value={} matches {} entries of \
             '{dimension}' ({}); expected exactly one",
            crate::refusal::python_float_repr(wanted),
            matches.len(),
            crate::refusal::python_float_list_repr(&values)
        )));
    }
    Ok(Some((axis, matches[0])))
}

/// How one resolved selector claims its variable, for the check that no
/// two direct fields quietly share one.
///
/// A layer-addressed selector claims only its slice, so four soil fields
/// may legitimately name the same `tsoil` variable at four depths; a
/// selector with no layer claims the whole variable.
fn direct_claim(variable: &str, selector: &Node) -> String {
    match (
        selector.field("layer_value"),
        selector.field("layer_dimension").and_then(Node::as_str),
    ) {
        (Some(value), Some(dimension)) => format!(
            "{variable}[{dimension}={}]",
            // Python interpolates the mapping's own JSON value with
            // `str()`, which keeps an integer an integer; the
            // `layer_value={value!r}` refusal above casts to float first
            // and so spells the same 1 as `1.0`.  The two really do
            // differ, and both are reproduced rather than unified.
            json_number_str(value)
        ),
        _ => variable.to_owned(),
    }
}

/// A JSON number as Python's `str()` writes it, integers included.
fn json_number_str(value: &Node) -> String {
    match value {
        Node::Integer(number) => number.to_string(),
        other => other
            .as_f64()
            .map(crate::refusal::python_float_repr)
            .unwrap_or_default(),
    }
}

/// Open one dataset the way the shipped `rw_netcdf` bridge opens it.
///
/// The open itself is LAZY: a NetCDF-4 file whose metadata cannot be
/// reconstructed strictly still returns a handle, and only says so when the
/// dimension or variable tables are first built.  So strictness is decided
/// by PROBING those tables, and a strict failure falls back to the
/// size-inferred (`Lossy`) metadata mode before refusing.
///
/// This ladder is not a nicety: the Python engine reaches these bytes
/// through `rw_netcdf`, which has exactly this fallback, so an engine
/// without it would refuse files the Python engine reads, a parity
/// difference that is entirely about the open path and not about the data.
fn open_dataset(source: &str) -> Result<Vec<NcVariable>> {
    let strict_error = match NcFile::open(source) {
        Ok(file) => match file.variables() {
            Ok(variables) if file.dimensions().is_ok() => {
                return Ok(inventory(&file, variables))
            }
            Ok(_) => "dimension table could not be built".to_owned(),
            Err(error) => error.to_string(),
        },
        Err(error) => error.to_string(),
    };
    let options = netcrust::NcOpenOptions {
        metadata_mode: netcrust::NcMetadataMode::Lossy,
        ..Default::default()
    };
    let file = NcFile::open_with_options(source, options).map_err(|lossy_error| {
        missing_input(format!(
            "cannot open NetCDF {source}: {strict_error} (and size-inferred \
             metadata also failed: {lossy_error})"
        ))
    })?;
    let variables = file.variables().map_err(|lossy_error| {
        decode_failed(format!(
            "cannot read NetCDF {source}: {strict_error} (and size-inferred \
             metadata also failed: {lossy_error})"
        ))
    })?;
    Ok(inventory(&file, variables))
}

/// What a whole-object decode of `files` can hold at once, in bytes.
///
/// Read from the headers only.  Every indexed variable of every object is
/// counted as if decoded, promoted to `f64` and held twice (its values and
/// one working copy while it is stacked and converted), plus each object's
/// own bytes three times (read, staged, parsed), the factor the GRIB2
/// price uses.  The selectors read a subset, so this bounds the decode
/// from above; `threads::admit_width` holds it to a declared budget.
pub fn whole_object_price(files: &[String]) -> Result<u64> {
    let mut total: u64 = 0;
    for source in files {
        let bytes = std::fs::metadata(source).map(|meta| meta.len()).unwrap_or(0);
        let mut elements: u64 = 0;
        for variable in open_dataset(source)? {
            if let NcSource::Indexed(indexed) = &variable.source {
                let count = indexed
                    .shape()
                    .iter()
                    .fold(1u64, |acc, &length| acc.saturating_mul(length as u64));
                elements = elements.saturating_add(count);
            }
        }
        total = total
            .saturating_add(elements.saturating_mul(16))
            .saturating_add(bytes.saturating_mul(3));
    }
    Ok(total)
}

/// Everything the file offers a selector: the NetCDF variable index, plus
/// the NetCDF-4 coordinate variables that index omits.
fn inventory(file: &NcFile, indexed: Vec<Variable>) -> Vec<NcVariable> {
    let mut variables: Vec<NcVariable> = indexed
        .into_iter()
        .map(|variable| NcVariable {
            name: variable.name().to_owned(),
            dimensions: variable
                .dimensions()
                .iter()
                .map(|dimension| dimension.name().to_owned())
                .collect(),
            source: NcSource::Indexed(variable),
            declared: BTreeMap::new(),
        })
        .collect();
    recover_dimension_scales(file, &mut variables);
    variables
}

/// Add back the NetCDF-4 coordinate variables the variable table omits.
///
/// Grafted from `rw_netcdf` (`tools/rustwx/crates/rw-netcdf/src/main.rs`,
/// `recover_dimension_scales`), which is the bridge the Python engine
/// reads these same bytes through, so this is the reference
/// implementation, not a reinvention.  Recovery uses only public netcrust
/// API: `hdf5_root_datasets` for the names and shapes,
/// `hdf5_dataset_attribute_string` for the CF string attributes, and
/// `read_array_f64`, which already falls back to a raw-HDF5 read by name
/// for values.  Nothing in the vendored tree is modified.
///
/// A 1-D dataset whose length matches a declared dimension is given that
/// dimension.  Ambiguity is not guessed at: a dataset whose own name is a
/// dimension takes that dimension (the CF coordinate-variable
/// convention), and anything still ambiguous gets an empty dimension list
/// rather than a plausible-looking wrong one, which the coordinate
/// contract then refuses by name instead of silently mis-naming an axis.
fn recover_dimension_scales(file: &NcFile, variables: &mut Vec<NcVariable>) {
    let (Ok(datasets), Ok(dimensions)) = (file.hdf5_root_datasets(), file.dimensions()) else {
        return;
    };
    let known: BTreeSet<String> = variables.iter().map(|v| v.name.clone()).collect();
    for dataset in datasets {
        let name = dataset.name().to_owned();
        if known.contains(&name) {
            continue;
        }
        // A dimension with no coordinate variable behind it is still
        // stored as an HDF5 dimension scale, and netCDF-4 marks it with
        // the sentence above.  netCDF4-python hides exactly those, so
        // recovering them would invent variables the C library does not
        // report, which is the convention's own answer, not a heuristic.
        if file
            .hdf5_dataset_attribute_string(&name, "NAME")
            .is_some_and(|value| value.starts_with(PHONY_DIMENSION_PREFIX))
        {
            continue;
        }
        let shape: Vec<usize> = dataset.shape().iter().map(|&n| n as usize).collect();
        let mut axes = Vec::with_capacity(shape.len());
        for &length in &shape {
            // The coordinate-variable convention first: a 1-D dataset
            // named after a dimension IS that dimension.
            let named = (shape.len() == 1)
                .then(|| {
                    dimensions
                        .iter()
                        .find(|d| d.name() == name && d.len() == length)
                })
                .flatten();
            let resolved = named.or_else(|| {
                let mut matches = dimensions.iter().filter(|d| d.len() == length);
                // Only accept a length match when it is unique; two
                // dimensions of equal length cannot be told apart here.
                match (matches.next(), matches.next()) {
                    (Some(only), None) => Some(only),
                    _ => None,
                }
            });
            match resolved {
                Some(dimension) => axes.push(dimension.name().to_owned()),
                None => {
                    axes.clear();
                    break;
                }
            }
        }
        let mut attributes = BTreeMap::new();
        for key in RECOVERED_ATTRIBUTES {
            if let Some(value) = file.hdf5_dataset_attribute_string(&name, key) {
                attributes.insert(key.to_owned(), value);
            }
        }
        variables.push(NcVariable {
            name,
            dimensions: axes,
            source: NcSource::DimensionScale {
                file: file.clone(),
                attributes,
            },
            declared: BTreeMap::new(),
        });
    }
}

/// `mapped_source._apply_producer_attributes`.
///
/// Some producers define their variables' units in a published
/// vocabulary instead of writing CF attributes into every file:
/// Earth2Studio's NetCDF writer emits `t850(time, lead_time, lat, lon)`
/// with no `units` on the field and none on `lat`/`lon`, because its
/// lexicon fixes them (`t850` is kelvin, `z500` is geopotential in
/// m2 s-2, `lat` is degrees north).  The mapping's `producer_attributes`
/// table carries that vocabulary as data, keyed by exact variable name.
/// An attribute the file lacks is supplied from it; one the file carries
/// must agree with it, because a file that says otherwise is not the
/// vocabulary the mapping was written for.
fn apply_producer_attributes(
    variables: &mut [NcVariable],
    mapping: &Mapping,
    source: &str,
) -> Result<()> {
    let Some(table) = mapping.doc.get("producer_attributes") else {
        return Ok(());
    };
    for (name, attributes) in table.entries() {
        let Some(variable) = variables.iter_mut().find(|variable| variable.name == *name) else {
            continue;
        };
        for (key, value) in attributes.entries() {
            let declared = value.as_str().ok_or_else(|| {
                mapping_invalid(format!(
                    "producer_attributes.{name}.{key} must be a string"
                ))
            })?;
            match variable.file_attribute_string(key) {
                Some(observed) if observed != declared => {
                    return Err(frame_invalid(format!(
                        "NetCDF {source} variable {} carries {key}={}, and the \
                         producer vocabulary this mapping declares says {}",
                        crate::refusal::python_repr(name),
                        crate::refusal::python_repr(&observed),
                        crate::refusal::python_repr(declared)
                    )));
                }
                Some(_) => {}
                None => {
                    variable.declared.insert(key.clone(), declared.to_owned());
                }
            }
        }
    }
    Ok(())
}

/// The seconds in one unit of a lead-time axis.
fn lead_unit_seconds(units: &str) -> Result<i64> {
    match units.trim().to_ascii_lowercase().as_str() {
        "days" | "day" | "d" => Ok(86_400),
        "hours" | "hour" | "h" => Ok(3_600),
        "minutes" | "minute" | "min" => Ok(60),
        "seconds" | "second" | "s" => Ok(1),
        other => Err(mapping_invalid(format!(
            "coordinates.time.lead.units '{other}' is not days, hours, minutes \
             or seconds"
        ))),
    }
}

/// `mapped_source._decode_netcdf`.
pub fn decode_netcdf(mapping: &Mapping, files: &[String]) -> Result<DecodedCollection> {
    let coordinates = mapping.coordinates()?;
    let horizontal = coordinates
        .get("horizontal")
        .ok_or_else(|| mapping_invalid("mapping.coordinates.horizontal is required"))?;
    let vertical_contract = mapping.vertical()?;
    let time_contract = coordinates
        .get("time")
        .ok_or_else(|| mapping_invalid("mapping.coordinates.time is required"))?;
    let member_contract = coordinates.field("member");
    let declared_levels = mapping.declared_levels()?;
    let soil_layer_count = mapping.soil_layer_count()?;

    let mut direct: BTreeMap<DirectKey, DirectValue> = BTreeMap::new();
    let mut cycles: BTreeMap<TimeKey, NaiveDateTime> = BTreeMap::new();
    let mut reference_latitude: Option<Vec<f64>> = None;
    let mut reference_longitude: Option<Vec<f64>> = None;
    let mut reference_vertical: Option<Vec<f64>> = None;
    let mut grid_fingerprint: Option<String> = None;
    let mut owed: BTreeSet<String> = BTreeSet::new();
    for field in mapping.fields()? {
        if field.derivation().is_none() && !field.optional()? {
            owed.insert(field.name.clone());
        }
    }
    let mut supplied: BTreeSet<String> = BTreeSet::new();
    // A producer that writes one variable per pressure level (`t50` ..
    // `t1000`) and no level coordinate stacks them on `vertical`; the
    // mapping's declared ladder is then the vertical coordinate.
    let mut stacks_vertical = false;
    for field in mapping.fields()? {
        if field.selector_stack_axis() == Some("vertical") {
            stacks_vertical = true;
        }
    }
    if stacks_vertical && declared_levels.is_empty() {
        return Err(mapping_invalid(
            "a field stacks per-level variables on vertical, and \
             coordinates.vertical declares no levels",
        ));
    }

    for source in files {
        let mut variables = open_dataset(source)?;
        apply_producer_attributes(&mut variables, mapping, source)?;

        let latitude_variable = resolve_variable(
            &variables,
            horizontal
                .get("latitude")
                .ok_or_else(|| mapping_invalid("coordinates.horizontal.latitude is required"))?,
            "latitude",
        )?;
        let longitude_variable = resolve_variable(
            &variables,
            horizontal
                .get("longitude")
                .ok_or_else(|| mapping_invalid("coordinates.horizontal.longitude is required"))?,
            "longitude",
        )?;
        let (latitude, longitude, latitude_dimension, longitude_dimension) =
            horizontal_coordinates(mapping, &variables, latitude_variable, longitude_variable, source)?;

        // The vertical coordinate resolves LAZILY: a file carrying only
        // surface or soil quantities has none, and refusing it there would
        // make a multi-file source undecodable for a reason that is not
        // about the data.  The failure is kept and re-raised the moment a
        // field with a vertical axis is read from this file.
        let vertical_selector = vertical_contract
            .get("selector")
            .ok_or_else(|| mapping_invalid("coordinates.vertical.selector is required"))?;
        let vertical_units = vertical_contract
            .get("units")
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid("coordinates.vertical.units is required"))?;
        let vertical_resolution = resolve_variable(&variables, vertical_selector, "vertical")
            .and_then(|variable| {
                let values = coordinate_values(variable, Some(vertical_units), "vertical")?;
                Ok((variable, values))
            });
        let mut vertical_values = vertical_resolution
            .as_ref()
            .ok()
            .map(|(_variable, values)| values.clone());
        let vertical_error = vertical_resolution.as_ref().err().cloned();
        let vertical_dimension = match &vertical_resolution {
            Ok((variable, _values)) => Some(coordinate_dimension(variable, "vertical")?),
            Err(_) => None,
        };
        let mut vertical_selection: Option<(String, Vec<usize>)> = None;
        if let (Some(dimension), Some(offered)) = (&vertical_dimension, vertical_values.clone()) {
            let distinct: BTreeSet<&String> =
                [&latitude_dimension, &longitude_dimension, dimension].into_iter().collect();
            if distinct.len() != 3 {
                return Err(frame_invalid(
                    "NetCDF latitude, longitude, and vertical coordinates must \
                     use distinct dimensions",
                ));
            }
            if !declared_levels.is_empty()
                && let Some(indices) = declared_level_indices(&declared_levels, &offered, source)? {
                    vertical_selection = Some((dimension.clone(), indices));
                    vertical_values = Some(declared_levels.clone());
                }
        }
        if stacks_vertical && vertical_values.is_none() {
            vertical_values = Some(declared_levels.clone());
        }

        let time_selector = time_contract
            .get("selector")
            .ok_or_else(|| mapping_invalid("coordinates.time.selector is required"))?;
        let time_variable = resolve_variable(&variables, time_selector, "time")?;
        let time_dimension = coordinate_dimension(time_variable, "time")?;
        let declared_time_units = time_contract
            .get("units")
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid("coordinates.time.units is required"))?;
        if attribute_string(time_variable, "units").as_deref() != Some(declared_time_units) {
            return Err(mapping_invalid("NetCDF time units differ from mapping"));
        }
        let calendar = time_contract
            .field("calendar")
            .and_then(Node::as_str)
            .map(str::to_owned)
            .or_else(|| attribute_string(time_variable, "calendar"))
            .unwrap_or_else(|| "standard".to_owned());
        if !matches!(
            calendar.as_str(),
            "standard" | "gregorian" | "proleptic_gregorian"
        ) {
            return Err(mapping_invalid(format!(
                "calendar '{calendar}' is not supported for WRF initialization"
            )));
        }
        // A malformed UTC-offset designator propagates its own refusal
        // out of the parse; `None` is the units not being a reference
        // time at all.
        let reference = parse_cf_units(declared_time_units)?.ok_or_else(|| {
            mapping_invalid(format!(
                "time units '{declared_time_units}' are not a CF reference time"
            ))
        })?;
        let (_shape, raw_times) = read_values(time_variable)?;
        let times = decode_times(&reference, &raw_times)?;
        let unique: BTreeSet<&NaiveDateTime> = times.iter().collect();
        if unique.len() != times.len() {
            return Err(frame_invalid(format!(
                "NetCDF file {source} contains duplicate valid times"
            )));
        }
        // A forecast file written as (reference time, lead time): the time
        // coordinate is the initialization and the `lead` axis the offset,
        // so each valid time is reference + lead and its source cycle is
        // the reference.  One reference per file: two initializations in
        // one file would put two forecasts' states at the same valid time.
        let mut lead_dimension: Option<String> = None;
        let valid_times: Vec<(NaiveDateTime, NaiveDateTime)> = match time_contract.get("lead") {
            None => times.iter().map(|time| (*time, *time)).collect(),
            Some(lead) => {
                let lead_selector = lead.get("selector").ok_or_else(|| {
                    mapping_invalid("coordinates.time.lead.selector is required")
                })?;
                let lead_units = lead.get("units").and_then(Node::as_str).ok_or_else(|| {
                    mapping_invalid("coordinates.time.lead.units is required")
                })?;
                let seconds = lead_unit_seconds(lead_units)?;
                let lead_variable = resolve_variable(&variables, lead_selector, "lead time")?;
                let observed = attribute_string(lead_variable, "units");
                if observed.as_deref() != Some(lead_units) {
                    return Err(mapping_invalid(format!(
                        "NetCDF lead time units {} differ from mapping {}",
                        optional_repr(observed.as_deref()),
                        crate::refusal::python_repr(lead_units)
                    )));
                }
                lead_dimension = Some(coordinate_dimension(lead_variable, "lead time")?);
                if times.len() != 1 {
                    return Err(frame_invalid(format!(
                        "NetCDF file {source} holds {} reference times beside a \
                         lead axis; pass one initialization per file so every \
                         valid time belongs to one forecast",
                        times.len()
                    )));
                }
                let reference = times[0];
                let (_shape, leads) = read_values(lead_variable)?;
                let mut valid = Vec::with_capacity(leads.len());
                for lead_value in leads {
                    let offset = lead_value * seconds as f64;
                    if !offset.is_finite() || offset.fract() != 0.0 || offset.abs() >= 1e15 {
                        return Err(frame_invalid(format!(
                            "NetCDF file {source} lead time {lead_value} {lead_units} is \
                             not a whole number of seconds"
                        )));
                    }
                    let instant = reference
                        .checked_add_signed(Duration::seconds(offset as i64))
                        .ok_or_else(|| {
                            decode_failed(format!("lead time out of range: {lead_value}"))
                        })?;
                    valid.push((instant, reference));
                }
                let distinct: BTreeSet<&NaiveDateTime> = valid.iter().map(|(time, _)| time).collect();
                if distinct.len() != valid.len() {
                    return Err(frame_invalid(format!(
                        "NetCDF file {source} contains duplicate lead times"
                    )));
                }
                valid
            }
        };

        let mut member_dimension: Option<String> = None;
        let mut member_value: Option<String> = None;
        if let Some(contract) = member_contract {
            if contract.get("kind").and_then(Node::as_str) != Some("dimension") {
                return Err(mapping_invalid(
                    "NetCDF member coordinate must be a dimension",
                ));
            }
            let selector = contract
                .get("selector")
                .ok_or_else(|| mapping_invalid("coordinates.member.selector is required"))?;
            let variable = resolve_variable(&variables, selector, "member")?;
            member_dimension = Some(coordinate_dimension(variable, "member")?);
            let (_shape, values) = read_values(variable)?;
            if values.len() != 1 {
                return Err(frame_invalid(
                    "mapped WRF initialization requires exactly one NetCDF \
                     ensemble member",
                ));
            }
            member_value = Some(format_member(values[0]));
        }

        match (&reference_latitude, &reference_longitude) {
            (None, None) => {
                reference_latitude = Some(latitude.clone());
                reference_longitude = Some(longitude.clone());
            }
            (Some(known_latitude), Some(known_longitude)) => {
                if *known_latitude != latitude || *known_longitude != longitude {
                    return Err(frame_invalid(
                        "NetCDF source horizontal coordinates change between files",
                    ));
                }
            }
            _ => unreachable!("both references are set together"),
        }
        if let Some(levels) = &vertical_values {
            match &reference_vertical {
                None => {
                    reference_vertical = Some(levels.clone());
                    grid_fingerprint = Some(axis_fingerprint(&[&latitude, &longitude, levels]));
                }
                Some(known) if known != levels => {
                    return Err(frame_invalid(
                        "NetCDF source vertical coordinate changes between files",
                    ));
                }
                _ => {}
            }
        }

        // Per FILE, spanning fields: which direct field already claimed
        // each variable slice in this file.
        let mut claimed_direct_variables: BTreeMap<String, String> = BTreeMap::new();
        for name in mapping.field_names()? {
            let field = mapping.field(&name)?;
            if field.derivation().is_some() {
                continue;
            }
            let selectors = field.selectors();
            let stack_axis = field.selector_stack_axis();
            let mut resolved: Vec<(&NcVariable, &Node)> = Vec::new();
            // A selector's claim on a variable is its SLICE, not the whole
            // variable: four soil fields legitimately name one `tsoil` at
            // four depths, so keying this on the name alone refused every
            // layer-addressed mapping as a duplicate.
            let mut claimed_slices: BTreeSet<(String, Option<String>, Option<String>)> =
                BTreeSet::new();
            let mut absent = 0usize;
            for selector in selectors {
                let matched = match_variables(&variables, selector);
                if matched.len() > 1 {
                    return Err(frame_invalid(format!(
                        "{name} selector resolves multiple NetCDF variables"
                    )));
                }
                match matched.first() {
                    None => absent += 1,
                    Some((variable, _evidence)) => {
                        let slice_key = (
                            variable.name().to_owned(),
                            selector
                                .field("layer_dimension")
                                .and_then(Node::as_str)
                                .map(str::to_owned),
                            selector.field("layer_value").map(json_number_str),
                        );
                        if !claimed_slices.insert(slice_key) {
                            if stack_axis.is_some() {
                                return Err(frame_invalid(format!(
                                    "{name} stacked selectors resolve duplicate \
                                     variable '{}'",
                                    variable.name()
                                )));
                            }
                            continue;
                        }
                        resolved.push((variable, selector));
                    }
                }
            }
            if resolved.is_empty() {
                // Not in THIS file; `owed` remembers it and the refusal at
                // the end names every file that was read.
                continue;
            }
            if stack_axis.is_some() && absent > 0 {
                return Err(frame_invalid(format!(
                    "{name} stacked selector inventory is split across files: \
                     {} of {} members resolve in {source}",
                    resolved.len(),
                    selectors.len()
                )));
            }
            if stack_axis.is_none() && resolved.len() != 1 {
                return Err(frame_invalid(format!(
                    "{name} resolves multiple NetCDF variables; rw-wps.mapping.v1 \
                     does not yet declare how alternatives become one field"
                )));
            }
            if stack_axis.is_some() && resolved.len() != selectors.len() {
                return Err(frame_invalid(format!(
                    "{name} stacked selector inventory is incomplete"
                )));
            }
            // Two DIFFERENT direct fields resolving one variable slice is a
            // mapping that says the same bytes are two quantities; the
            // remedy is an explicit derive alias, so say that rather than
            // decoding the same array twice under two names.
            for (variable, selector) in &resolved {
                let claim = direct_claim(variable.name(), selector);
                match claimed_direct_variables.get(&claim) {
                    Some(previous) if previous != &name => {
                        return Err(frame_invalid(format!(
                            "NetCDF variable {} directly provides both {} and {}; \
                             derive aliases explicitly",
                            crate::refusal::python_repr(&claim),
                            crate::refusal::python_repr(previous),
                            crate::refusal::python_repr(&name)
                        )));
                    }
                    _ => {}
                }
                claimed_direct_variables.insert(claim, name.clone());
            }
            let source_axes = field.source_axes()?;
            let mut variable_axes = source_axes.clone();
            if let Some(axis) = stack_axis {
                variable_axes.retain(|item| item != axis);
            }
            if variable_axes.iter().any(|axis| axis == "vertical") && vertical_values.is_none() {
                let detail = vertical_error
                    .as_ref()
                    .map(|error| error.message.clone())
                    .unwrap_or_else(|| "no vertical coordinate resolved".to_owned());
                return Err(frame_invalid(format!(
                    "{name} declares a vertical axis, and {source} has no \
                     readable vertical coordinate: {detail}"
                )));
            }
            let declared_source_units = field
                .raw
                .get("units")
                .and_then(|units| units.get("source"))
                .and_then(Node::as_str)
                .ok_or_else(|| {
                    mapping_invalid(format!("fields.{name}.units.source must be a string"))
                })?;

            let mut arrays: Vec<ArrayD<f64>> = Vec::with_capacity(resolved.len());
            let mut reference_dimensions: Option<Vec<String>> = None;
            // Selector outputs were already retained together in the serial
            // path. Price that retained storage plus concurrent read/mask/
            // slice scratch, then use the existing bounded worker pool.
            let sizes: Option<Vec<u64>> = resolved.iter().map(|(variable, _)| {
                match &variable.source {
                    NcSource::Indexed(value) => Some(value.shape().iter()
                        .fold(1u64, |n, size| n.saturating_mul(*size as u64))),
                    NcSource::DimensionScale { .. } => None,
                }
            }).collect();
            let selector_width = if let Some(sizes) = sizes {
                let output = sizes.iter().fold(0u64, |n, size| n.saturating_add(size.saturating_mul(8)));
                let scratch = sizes.iter().copied().max().unwrap_or(1).max(1).saturating_mul(48);
                let budget = crate::threads::usable_budget(0)?;
                budget.map(|bytes| (bytes.saturating_sub(output) / scratch).max(1) as usize)
                    .unwrap_or(crate::threads::width()).min(crate::threads::width()).max(1)
            } else { 1 };
            for batch in resolved.chunks(selector_width) {
                let loaded: Vec<Result<(Vec<String>, ArrayD<f64>)>> = crate::threads::install(|| {
                    use rayon::prelude::*;
                    batch.par_iter().map(|(variable, selector)| {
                        let observed_units = attribute_string(variable, "units");
                        if observed_units.as_deref() != Some(declared_source_units) {
                            return Err(mapping_invalid(format!(
                                "{name} source units {} differ from mapping {}",
                                optional_repr(observed_units.as_deref()),
                                crate::refusal::python_repr(declared_source_units)
                            )));
                        }
                        let layer = layer_slice_position(&variables, variable, selector, &name)?;
                        let mut dimensions: Vec<String> = variable.dimensions().to_vec();
                        let (shape, mut values) = read_values(variable)?;
                        let mut shape = shape;
                        // The layer slice goes FIRST and drops its axis, so the
                        // vertical take below indexes the dimension order that
                        // survives it: the order `_read_nc_values` documents.
                        if let Some((axis, index)) = layer {
                            let (taken_shape, taken) = take_layer(&shape, &values, axis, index);
                            shape = taken_shape;
                            values = taken;
                            dimensions.remove(axis);
                        }
                        if let Some((dimension, indices)) = &vertical_selection
                            && let Some(axis) = dimensions.iter().position(|item| item == dimension) {
                                let (taken_shape, taken) = take_axis(&shape, &values, axis, indices);
                                shape = taken_shape;
                                values = taken;
                            }
                        let missing: Vec<bool> = values.iter().map(|value| !value.is_finite()).collect();
                        let policy = field.missing_kind()?;
                        if ["reject", "composition_fill"].contains(&policy) && missing.iter().any(|flag| *flag) {
                            return Err(frame_invalid(format!(
                                "{name} contains missing/non-finite source values"
                            )));
                        }
                        if policy == "value" {
                            let replacement = field.missing_value()?;
                            for (value, flagged) in values.iter_mut().zip(missing.iter()) {
                                if *flagged {
                                    *value = replacement;
                                }
                            }
                        } else {
                            for (value, flagged) in values.iter_mut().zip(missing.iter()) {
                                if *flagged {
                                    *value = f64::NAN;
                                }
                            }
                        }
                        if variable_axes.len() != shape.len() {
                            return Err(frame_invalid(format!(
                                "{name} source_axes rank differs from NetCDF variable {}",
                                variable.name()
                            )));
                        }
                        for (role, expected) in [
                            ("vertical", vertical_dimension.clone()),
                            ("y", Some(latitude_dimension.clone())),
                            ("x", Some(longitude_dimension.clone())),
                            ("time", Some(time_dimension.clone())),
                            ("lead", lead_dimension.clone()),
                            ("member", member_dimension.clone()),
                        ] {
                            let Some(axis) = variable_axes.iter().position(|item| item == role) else {
                                continue;
                            };
                            let Some(expected) = expected else {
                                return Err(frame_invalid(format!(
                                    "{name} has a {role} axis without a {role} coordinate"
                                )));
                            };
                            if dimensions[axis] != expected {
                                return Err(frame_invalid(format!(
                                    "{name} {role} axis does not use the declared coordinate \
                                     dimension '{expected}'"
                                )));
                            }
                        }
                        let has_time = variable_axes.iter().any(|axis| axis == "time");
                        let has_lead = variable_axes.iter().any(|axis| axis == "lead");
                        if lead_dimension.is_some() && has_time != has_lead {
                            return Err(frame_invalid(format!(
                                "{name} must carry both the time and the lead axis, or \
                                 neither, in a file whose valid times are reference plus lead"
                            )));
                        }
                        if lead_dimension.is_none() && has_lead {
                            return Err(mapping_invalid(format!(
                                "{name} declares a lead axis and coordinates.time declares no lead"
                            )));
                        }
                        if !has_time && lead_dimension.is_none() && times.len() != 1 {
                            return Err(frame_invalid(format!(
                                "static field {name} is ambiguous across a multi-time \
                                 NetCDF file"
                            )));
                        }
                        let values = ArrayD::from_shape_vec(IxDyn(&shape), values)
                            .map_err(|error| decode_failed(format!("{name}: {error}")))?;
                        Ok((dimensions, values))
                    }).collect()
                });
                // Indexed collection and this drain preserve selector order,
                // including the first refusal and cross-selector shape check.
                for value in loaded {
                    let (dimensions, values) = value?;
                    match &reference_dimensions {
                        None => reference_dimensions = Some(dimensions.clone()),
                        Some(known) if *known != dimensions => {
                            return Err(frame_invalid(format!(
                                "{name} stacked NetCDF variables have different shapes \
                                 or dimensions"
                            )))
                        }
                        _ => {}
                    }
                    arrays.push(values);
                }
            }

            let data = if let Some(axis_name) = stack_axis {
                if axis_name == "vertical" {
                    // Each member variable IS one level; its selector says
                    // which, and the stack must be the declared ladder in
                    // the declared order.
                    let offered = resolved
                        .iter()
                        .map(|(_variable, selector)| {
                            selector.field("level_value").and_then(Node::as_f64)
                        })
                        .collect::<Option<Vec<f64>>>()
                        .ok_or_else(|| {
                            mapping_invalid(format!(
                                "{name} stacks on vertical, and every selector \
                                 needs a numeric level_value"
                            ))
                        })?;
                    if offered != declared_levels {
                        if field.missing_kind()? != "composition_fill"
                            || offered.iter().any(|level| !declared_levels.contains(level))
                            || offered.iter().enumerate().any(|(i, level)| offered[..i].contains(level))
                        {
                            return Err(mapping_invalid(format!(
                                "{name} sparse selector stack requires composition_fill and unique declared levels"
                            )));
                        }
                        let shape = arrays[0].raw_dim();
                        arrays = declared_levels.iter().map(|level| {
                            offered.iter().position(|offered| offered == level)
                                .map(|index| arrays[index].clone())
                                .unwrap_or_else(|| ArrayD::from_elem(shape.clone(), f64::NAN))
                        }).collect();
                    }
                } else {
                    let Some(expected) = soil_layer_count else {
                        return Err(mapping_invalid(format!(
                            "{name} stacks soil without target.soil_layer_count"
                        )));
                    };
                    if arrays.len() as i64 != expected {
                        return Err(frame_invalid(format!(
                            "{name} has {} stacked soil selectors; target declares {expected}",
                            arrays.len()
                        )));
                    }
                }
                let axis = source_axes
                    .iter()
                    .position(|item| item == axis_name)
                    .ok_or_else(|| {
                        mapping_invalid(format!(
                            "{name} selector_stack_axis '{axis_name}' is not in source_axes"
                        ))
                    })?;
                array::stack(&arrays, axis, &name)?
            } else {
                arrays.remove(0)
            };

            let references: Vec<String> = resolved
                .iter()
                .map(|(variable, _selector)| format!("{source}:{}", variable.name()))
                .collect();
            for (time_index, (valid_time, source_cycle)) in valid_times.iter().enumerate() {
                // Select borrowed planes first. Copying before dropping time,
                // lead and member axes duplicates the entire source series.
                let mut selected = data.view();
                let mut selected_axes = source_axes.clone();
                if lead_dimension.is_some() {
                    if let Some(axis) = selected_axes.iter().position(|item| item == "lead") {
                        selected = selected.index_axis_move(ndarray::Axis(axis), time_index);
                        selected_axes.remove(axis);
                    }
                    if let Some(axis) = selected_axes.iter().position(|item| item == "time") {
                        selected = selected.index_axis_move(ndarray::Axis(axis), 0);
                        selected_axes.remove(axis);
                    }
                } else if let Some(axis) = selected_axes.iter().position(|item| item == "time") {
                    selected = selected.index_axis_move(ndarray::Axis(axis), time_index);
                    selected_axes.remove(axis);
                }
                if let Some(axis) = selected_axes.iter().position(|item| item == "member") {
                    selected = selected.index_axis_move(ndarray::Axis(axis), 0);
                    selected_axes.remove(axis);
                }
                let converted = array::unit_transform(
                    selected.as_standard_layout().into_owned(),
                    field.unit_scale(),
                    field.unit_offset(),
                    &name,
                )?;
                let target_axes = field.target_axes()?;
                let converted =
                    array::transpose_to_target(converted, &selected_axes, &target_axes, &name)?;
                let missing_count = array::count_nan(&converted);
                let key = (*valid_time, member_value.clone(), name.clone());
                if direct.contains_key(&key) {
                    return Err(frame_invalid(format!(
                        "duplicate mapped field {name} at {valid_time}"
                    )));
                }
                direct.insert(
                    key,
                    DirectValue {
                        name: name.clone(),
                        valid_time: *valid_time,
                        member: member_value.clone(),
                        source_cycle: *source_cycle,
                        axes: target_axes,
                        values: converted,
                        missing_count,
                        references: references.clone(),
                    },
                );
                cycles.insert((*valid_time, member_value.clone()), *source_cycle);
            }
            supplied.insert(name.clone());
        }
    }

    let unsupplied: Vec<&String> = owed.iter().filter(|name| !supplied.contains(*name)).collect();
    if !unsupplied.is_empty() {
        return Err(crate::refusal::selector_unmatched(format!(
            "mapped field(s) {} have no matching variable in any of \
             the {} supplied NetCDF file(s): {}",
            crate::refusal::python_list_repr(&unsupplied),
            files.len(),
            files.join(", ")
        )));
    }
    let Some(latitude) = reference_latitude else {
        return Err(crate::refusal::selector_unmatched(
            "no NetCDF source data were decoded",
        ));
    };
    let longitude = reference_longitude.expect("both references are set together");
    let (vertical_values, grid_fingerprint) = match (reference_vertical, grid_fingerprint) {
        (Some(levels), Some(fingerprint)) => (levels, fingerprint),
        _ => (
            // Nothing here has a vertical axis (the composition's
            // terrain-only partition is exactly this shape) so the grid
            // identity is the horizontal one.  A field that DID declare a
            // vertical axis was already refused with the coordinate's own
            // error, so this is never a quiet substitution.
            Vec::new(),
            axis_fingerprint(&[&latitude, &longitude]),
        ),
    };
    let (hybrid_a, hybrid_b) = if mapping.vertical_kind()? == "hybrid_sigma_pressure"
        && !vertical_values.is_empty()
    {
        // NetCDF bytes carry no pv channel; a hybrid NetCDF source
        // rides entirely on the mapping's inline literals.
        crate::assemble::resolve_hybrid_literals_only(mapping, vertical_values.len())?
    } else {
        (Vec::new(), Vec::new())
    };
    let declaration = mapping.grid_declaration()?;
    let grid_fingerprint = if declaration.is_lambert() {
        crate::digest::bytes_sha256(format!("{grid_fingerprint}:{}",
            mapping.doc.get("grid").unwrap().to_value()).as_bytes())
    } else { grid_fingerprint };
    let direct = if declaration.rotates_winds() {
        crate::assemble::rotate_grid_relative_winds(direct, &declaration, None)?
    } else { direct };
    Ok(DecodedCollection {
        latitude,
        longitude,
        vertical_values,
        direct,
        source_cycles: cycles,
        grid_fingerprint,
        hybrid_a,
        hybrid_b,
        original_pressure_hpa: None,
    })
}

/// `str(np.asarray(member_values)[0])` for the integral ids real files carry.
fn format_member(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 9.0e15 {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// sha256 over the concatenated C-contiguous float64 axis bytes.
fn axis_fingerprint(axes: &[&Vec<f64>]) -> String {
    let mut bytes = Vec::new();
    for axis in axes {
        for value in axis.iter() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    crate::digest::bytes_sha256(&bytes)
}

/// `np.take(values, indices, axis)` on a flat C-order buffer.
fn take_axis(
    shape: &[usize],
    values: &[f64],
    axis: usize,
    indices: &[usize],
) -> (Vec<usize>, Vec<f64>) {
    let outer: usize = shape[..axis].iter().product();
    let extent = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut result = Vec::with_capacity(outer * indices.len() * inner);
    for block in 0..outer {
        for index in indices {
            let start = (block * extent + index) * inner;
            result.extend_from_slice(&values[start..start + inner]);
        }
    }
    let mut taken = shape.to_vec();
    taken[axis] = indices.len();
    (taken, result)
}

/// `np.take(values, index, axis)` on the flat representation: unlike
/// [`take_axis`], the addressed axis is DROPPED rather than narrowed to
/// one, which is what selecting a single soil layer out of a producer's
/// own layer dimension means.
fn take_layer(shape: &[usize], values: &[f64], axis: usize, index: usize) -> (Vec<usize>, Vec<f64>) {
    let (mut taken_shape, taken) = take_axis(shape, values, axis, &[index]);
    taken_shape.remove(axis);
    (taken_shape, taken)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projected_fixture_mapping(cf_axes: bool) -> Mapping {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/projected.mapping.json");
        let mut mapping = Mapping::load(path.to_str().unwrap()).unwrap();
        if cf_axes {
            let mut doc = mapping.doc.to_value();
            doc["coordinates"]["horizontal"]["latitude"]["name"] = serde_json::json!(["y"]);
            doc["coordinates"]["horizontal"]["longitude"]["name"] = serde_json::json!(["x"]);
            mapping.doc = Node::parse(&serde_json::to_vec(&doc).unwrap()).unwrap();
        }
        mapping
    }

    #[test]
    fn projected_cf_axes_and_geographic_auxiliaries_decode_to_the_same_window_and_winds() {
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let geographic = decode_netcdf(&projected_fixture_mapping(false),
            &[fixtures.join("projected-geographic.nc").display().to_string()]).unwrap();
        let cf = decode_netcdf(&projected_fixture_mapping(true),
            &[fixtures.join("projected-cf.nc").display().to_string()]).unwrap();
        assert_eq!(geographic.latitude, vec![0.0,0.03,0.06]);
        assert_eq!(geographic.longitude, vec![0.0,0.03,0.06,0.09]);
        assert_eq!(geographic.grid_fingerprint, cf.grid_fingerprint);
        for (key, value) in &geographic.direct {
            assert_eq!(value.values, cf.direct[key].values);
        }
        let (_, u) = geographic.direct.iter().find(|(key,_)| key.2 == "eastward_wind").unwrap();
        let (_, v) = geographic.direct.iter().find(|(key,_)| key.2 == "northward_wind").unwrap();
        // Independent geographic convergence at the HRRR SW corner.
        let alpha = (38.5_f64.to_radians()).sin() * (122.719528_f64 - 97.5).to_radians();
        assert!((u.values.as_slice().unwrap()[0] - 10.0*alpha.cos()).abs() < 1e-10);
        assert!((v.values.as_slice().unwrap()[0] - 10.0*alpha.sin()).abs() < 1e-10);
    }

    #[test]
    fn projected_coordinates_refuse_the_wrong_window_or_conflicting_cf_projection() {
        let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let mut mapping = projected_fixture_mapping(false);
        let mut doc = mapping.doc.to_value();
        doc["grid"]["parameters"]["lat1"] = serde_json::json!(22.0);
        mapping.doc = Node::parse(&serde_json::to_vec(&doc).unwrap()).unwrap();
        let error = decode_netcdf(&mapping, &[fixtures.join("projected-geographic.nc").display().to_string()]).unwrap_err();
        assert!(error.message.contains("differs from declared Lambert grid"));
        let mut mapping = projected_fixture_mapping(true);
        let mut doc = mapping.doc.to_value();
        doc["grid"]["parameters"]["lov"] = serde_json::json!(260.0);
        mapping.doc = Node::parse(&serde_json::to_vec(&doc).unwrap()).unwrap();
        let error = decode_netcdf(&mapping, &[fixtures.join("projected-cf.nc").display().to_string()]).unwrap_err();
        assert!(error.message.contains("central meridian differs"));
    }

    /// A parsed CF reference, for units that must be well-formed ones.
    fn reference(units: &str) -> CfReference {
        parse_cf_units(units)
            .unwrap_or_else(|refusal| panic!("{units}: {refusal}"))
            .unwrap_or_else(|| panic!("{units} is a CF reference time"))
    }

    #[test]
    fn cf_units_parse_the_spellings_real_archives_publish() {
        for text in [
            "hours since 1900-01-01 00:00:00",
            "seconds since 1970-01-01",
            "days since 1800-01-01T00:00:00Z",
        ] {
            reference(text);
        }
        assert!(parse_cf_units("K").expect("nothing to malform").is_none());
    }

    /// The one exact equivalence the offset handling promises: units
    /// carrying a UTC-offset designator decode to the SAME instants as
    /// the same epoch respelled in UTC by hand.  Every offset spelling
    /// udunits reads is here -- `+hh:mm`, `+hhmm`, bare `-h`/`-hh`,
    /// attached and spaced, and the zero-offset designators.
    #[test]
    fn offset_references_decode_like_their_utc_respelling() {
        for (offset_spelling, utc_respelling) in [
            // +05:00 reads five hours EAST: the epoch is EARLIER in UTC.
            (
                "hours since 2020-01-01 00:00:00 +05:00",
                "hours since 2019-12-31 19:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00+0500",
                "hours since 2019-12-31 19:00:00",
            ),
            // Negative forms read WEST: the epoch is LATER in UTC.
            (
                "hours since 2020-01-01 00:00:00 -06:00",
                "hours since 2020-01-01 06:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00-0600",
                "hours since 2020-01-01 06:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00 -6",
                "hours since 2020-01-01 06:00:00",
            ),
            // A half-hour zone exercises both halves of h*3600 + m*60.
            (
                "hours since 2020-01-01 00:00:00 +05:30",
                "hours since 2019-12-31 18:30:00",
            ),
            (
                "hours since 2020-01-01 00:00:00+0530",
                "hours since 2019-12-31 18:30:00",
            ),
            // Zero-offset designators are exactly UTC.
            (
                "hours since 2020-01-01 00:00:00Z",
                "hours since 2020-01-01 00:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00 +00:00",
                "hours since 2020-01-01 00:00:00",
            ),
            (
                "hours since 2020-01-01 00:00:00-0000",
                "hours since 2020-01-01 00:00:00",
            ),
            // A date-only stamp still takes a spaced offset (midnight).
            (
                "hours since 2020-01-01 +01:00",
                "hours since 2019-12-31 23:00:00",
            ),
        ] {
            let with_offset = reference(offset_spelling);
            let respelled = reference(utc_respelling);
            assert_eq!(
                with_offset.epoch, respelled.epoch,
                "{offset_spelling} vs {utc_respelling}"
            );
            assert_eq!(
                decode_times(&with_offset, &[0.0, 1.5]).expect("decoded"),
                decode_times(&respelled, &[0.0, 1.5]).expect("decoded"),
                "{offset_spelling}"
            );
        }
    }

    /// Ground truth for the shift direction, pinned as absolute strings
    /// so a sign error that broke both sides of the equivalence test the
    /// same way would still be caught.
    #[test]
    fn a_positive_offset_epoch_lands_earlier_on_the_utc_timeline() {
        let reference = reference("hours since 2020-01-01 00:00:00 +05:30");
        let decoded = decode_times(&reference, &[0.0, 5.5]).expect("decoded");
        assert_eq!(
            crate::frames::naive_isoformat(decoded[0]),
            "2019-12-31T18:30:00"
        );
        assert_eq!(
            crate::frames::naive_isoformat(decoded[1]),
            "2020-01-01T00:00:00"
        );
    }

    #[test]
    fn a_negative_offset_epoch_lands_later_on_the_utc_timeline() {
        let reference = reference("seconds since 2020-06-01 12:00:00 -06");
        let decoded = decode_times(&reference, &[0.0]).expect("decoded");
        assert_eq!(
            crate::frames::naive_isoformat(decoded[0]),
            "2020-06-01T18:00:00"
        );
    }

    /// The designator bounds: 59 minutes and 18:00 are the last values
    /// inside the grammar, and the first value past each is refused by
    /// name.  udunits and java.time both cap the field at +/-18:00; the
    /// widest civil offset on Earth is +14:00.
    #[test]
    fn offset_bounds_admit_18_hours_and_59_minutes_and_nothing_past_them() {
        assert_eq!(
            reference("hours since 2020-01-01 00:00:00 +18:00").epoch,
            reference("hours since 2019-12-31 06:00:00").epoch,
        );
        assert_eq!(
            reference("hours since 2020-01-01 00:59:00 +00:59").epoch,
            reference("hours since 2020-01-01 00:00:00").epoch,
        );
        for (units, names) in [
            ("hours since 2020-01-01 00:00:00 +18:01", "18:00"),
            ("hours since 2020-01-01 00:00:00 -19:00", "18:00"),
            ("hours since 2020-01-01 00:00:00 +05:60", "below 60"),
            ("hours since 2020-01-01 00:00:00 +05:61", "below 60"),
        ] {
            let refusal = match parse_cf_units(units) {
                Err(refusal) => refusal,
                Ok(parsed) => panic!("{units} was not refused: {parsed:?}"),
            };
            assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID, "{units}");
            assert!(refusal.message.contains("malformed"), "{units}: {refusal}");
            assert!(refusal.message.contains(names), "{units}: {refusal}");
        }
    }

    /// Every refusal names what is wrong; nothing is guessed at and
    /// nothing decodes with the designator ignored.  Returning `None`
    /// was the old behaviour, and at the decode call site it read as
    /// "not a CF reference time" -- the wrong sentence for units whose
    /// author plainly wrote one down.
    #[test]
    fn malformed_offsets_are_refused_with_the_reason() {
        for (units, names) in [
            // Three packed digits cannot be split into hours and
            // minutes without guessing between 5:30 and 53:0.
            ("hours since 2020-01-01 00:00:00 +530", "guessing"),
            ("hours since 2020-01-01 00:00:00 +05300", "guessing"),
            // udunits offsets carry no seconds field.
            ("hours since 2020-01-01 00:00:00 +05:30:00", "two digits"),
            ("hours since 2020-01-01 00:00:00 +05:3", "two digits"),
            // A sign with nothing behind it, a missing hours field, and
            // letters where digits go.
            ("hours since 2020-01-01 00:00:00 +", "no digits"),
            ("hours since 2020-01-01 00:00:00 +:30", "hours"),
            ("hours since 2020-01-01 00:00:00 +0a", "hours"),
            // Two zone designators cannot both place the epoch.
            ("hours since 2020-01-01 00:00:00+05:00Z", "names utc"),
            (
                "hours since 2020-01-01 00:00:00+05:00 +06:00",
                "two utc offsets",
            ),
        ] {
            let refusal = match parse_cf_units(units) {
                Err(refusal) => refusal,
                Ok(parsed) => panic!("{units} was not refused: {parsed:?}"),
            };
            assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID, "{units}");
            assert!(
                refusal.message.to_ascii_lowercase().contains(names),
                "{units}: {refusal}"
            );
        }
    }

    #[test]
    fn seconds_since_the_unix_epoch_decode_to_the_expected_instant() {
        let reference = reference("seconds since 1970-01-01");
        let decoded = decode_times(&reference, &[1_768_608_000.0]).unwrap();
        assert_eq!(
            crate::frames::naive_isoformat(decoded[0]),
            "2026-01-17T00:00:00"
        );
    }

    #[test]
    fn cf_masking_tests_the_stored_value_before_scaling() {
        let policy = CfPolicy {
            scale_factor: Some(0.5),
            add_offset: Some(10.0),
            fill_value: Some(-9999.0),
            missing_value: None,
        };
        let mut values = vec![-9999.0, 4.0];
        let mask = policy.apply(&mut values);
        assert!(values[0].is_nan());
        assert_eq!(values[1], 12.0);
        assert_eq!(mask, vec![true, false]);
    }

    #[test]
    fn cf_parallel_mask_keeps_stored_markers_nonfinite_and_unfused_words() {
        let policy = CfPolicy {
            scale_factor: Some(1.0 - 2.0_f64.powi(-27)),
            add_offset: Some(-1.0),
            fill_value: Some(-9999.0),
            missing_value: Some(-8888.0),
        };
        // The finite rounding witness becomes exactly zero with separate
        // multiply/add; an FMA would return -2^-54. Include a partial chunk.
        let stored = [
            -9999.0, -8888.0, f64::INFINITY, f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0123), 1.0 + 2.0_f64.powi(-27),
            -0.0, f64::from_bits(1),
        ];
        let expected = [
            f64::NAN, f64::NAN, f64::NAN, f64::NAN,
            f64::NAN, 0.0, -1.0, -1.0,
        ];
        let mut values: Vec<f64> = stored.into_iter().cycle().take(131_077).collect();
        let mask = policy.apply(&mut values);
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                value.to_bits(), expected[index % stored.len()].to_bits(), "{index}"
            );
            assert_eq!(mask[index], index % stored.len() < 2, "{index}");
        }
    }

    #[test]
    fn take_axis_selects_the_declared_levels_in_the_mappings_order() {
        let shape = vec![3, 2];
        let values = vec![1., 2., 3., 4., 5., 6.];
        let (taken_shape, taken) = take_axis(&shape, &values, 0, &[2, 0]);
        assert_eq!(taken_shape, vec![2, 2]);
        assert_eq!(taken, vec![5., 6., 1., 2.]);
    }

    #[test]
    fn declared_levels_that_appear_twice_refuse() {
        let error = declared_level_indices(&[500.0], &[500.0, 500.0], "file").unwrap_err();
        assert!(error.message.contains("appears 2 times"), "{}", error.message);
    }

    #[test]
    fn the_accepted_unit_sets_are_in_pythons_sorted_order() {
        // The refusal interpolates `sorted(accepted_units)` and the two
        // engines' sentences are compared byte for byte, so the ORDER is
        // part of the contract.  Measured against CPython's `sorted()`
        // on the same six spellings: `_` sorts after the uppercase
        // letters and before the lowercase ones.
        assert_eq!(
            crate::refusal::python_list_repr(&CF_DEGREES_NORTH),
            "['degreeN', 'degree_N', 'degree_north', 'degreesN', \
             'degrees_N', 'degrees_north']"
        );
        assert_eq!(
            crate::refusal::python_list_repr(&CF_DEGREES_EAST),
            "['degreeE', 'degree_E', 'degree_east', 'degreesE', \
             'degrees_E', 'degrees_east']"
        );
    }

    #[test]
    fn the_g_format_matches_pythons() {
        // Measured against CPython: `f"{v:g}"` for each value.  Rust's
        // `{}` never rounds to six significant digits and never switches
        // to exponent form, so a coordinate that reaches 1234567 would
        // have printed a different sentence on each engine.
        assert_eq!(python_format_g(400.5), "400.5");
        assert_eq!(python_format_g(90.0), "90");
        assert_eq!(python_format_g(360.0), "360");
        assert_eq!(python_format_g(1_234_567.0), "1.23457e+06");
        assert_eq!(python_format_g(1.234e-5), "1.234e-05");
        assert_eq!(python_format_g(1e6), "1e+06");
        assert_eq!(python_format_g(123.456_789), "123.457");
        assert_eq!(python_format_g(0.0), "0");
    }

    #[test]
    fn a_selector_prints_its_attributes_and_its_layer() {
        // Without these clauses a refusal about four soil selectors
        // printed the same sentence four times and named no difference.
        let selector = Node::parse(
            br#"{"name": ["tsoil"], "attributes": {"level": 2, "cell": "soil"},
                 "layer_dimension": "depth", "layer_value": 0.05,
                 "layer_units": "m"}"#,
        )
        .expect("selector parses");
        assert_eq!(
            selector_text(&selector),
            "name='tsoil' with cell='soil', level=2 at depth=0.05 m"
        );
    }

    #[test]
    fn an_optional_string_renders_as_pythons_repr() {
        assert_eq!(optional_repr(Some("K")), "'K'");
        assert_eq!(optional_repr(None), "None");
    }

    #[test]
    fn a_layer_take_drops_its_axis_rather_than_narrowing_it() {
        // The breakage: narrowing to extent 1 leaves a rank the field's
        // declared source_axes does not have, and the rank check then
        // refuses a soil mapping that is entirely correct.
        let shape = vec![2, 3, 2];
        let values: Vec<f64> = (0..12).map(f64::from).collect();
        let (taken_shape, taken) = take_layer(&shape, &values, 1, 2);
        assert_eq!(taken_shape, vec![2, 2]);
        assert_eq!(taken, vec![4., 5., 10., 11.]);
    }

    #[test]
    fn a_layer_addressed_claim_names_its_slice_not_the_whole_variable() {
        // Four soil fields legitimately name ONE `tsoil` at four depths.
        // Keying the claim on the variable name alone refused every
        // layer-addressed mapping as a duplicate.
        let selector = Node::parse(
            br#"{"layer_dimension": "depth", "layer_value": 0.05}"#,
        )
        .expect("selector parses");
        assert_eq!(direct_claim("tsoil", &selector), "tsoil[depth=0.05]");

        let whole = Node::parse(br#"{"name": ["tsoil"]}"#).expect("selector parses");
        assert_eq!(direct_claim("tsoil", &whole), "tsoil");
    }

    #[test]
    fn a_json_integer_layer_value_stays_an_integer() {
        // Python interpolates the mapping's own value with `str()`, so a
        // declared `1` claims `[level=1]`, not `[level=1.0]`.
        assert_eq!(json_number_str(&Node::Integer(1)), "1");
        assert_eq!(json_number_str(&Node::Number(0.05)), "0.05");
    }

    #[test]
    fn an_exact_level_match_needs_no_reindexing() {
        assert_eq!(
            declared_level_indices(&[1000.0, 850.0], &[1000.0, 850.0], "file").unwrap(),
            None
        );
    }
}
