//! Native GRIB2 grid-point writer following WMO FM 92, Edition 2.
//! Supports simple packing (5.0), complex spatial packing (5.3), IEEE
//! floating point (5.4, lossless), WRF projections, and statistical
//! intervals (4.8).

use super::parser::{GridDefinition, ProductDefinition};

/// Preserve geographic grid locations when encoding a projected grid on a
/// different spherical Earth. Projected coordinates scale with the radius;
/// latitude/longitude increments are angular and stay unchanged.
pub fn rescale_projected_grid_radius(
    grid: &GridDefinition,
    native_radius: f64,
    encoded_radius: f64,
) -> Result<GridDefinition, String> {
    if !native_radius.is_finite()
        || native_radius <= 0.0
        || !encoded_radius.is_finite()
        || encoded_radius <= 0.0
    {
        return Err("Grid Earth radii must be finite and positive".into());
    }
    let mut encoded = grid.clone();
    match grid.template {
        0 => {}
        10 | 20 | 30 => {
            if !grid.dx.is_finite() || grid.dx <= 0.0 || !grid.dy.is_finite() || grid.dy <= 0.0 {
                return Err("Projected grid spacing must be finite and positive".into());
            }
            let ratio = encoded_radius / native_radius;
            encoded.dx *= ratio;
            encoded.dy *= ratio;
            if !encoded.dx.is_finite() || !encoded.dy.is_finite() {
                return Err("Projected grid spacing overflows the encoded Earth radius".into());
            }
        }
        _ => {
            return Err(format!(
                "Grid template {} has no spherical radius transform", grid.template,
            ));
        }
    }
    Ok(encoded)
}
use chrono::{Datelike, NaiveDateTime, Timelike};

/// Packing method for encoding data values.
#[derive(Debug, Clone)]
pub enum PackingMethod {
    /// Template 5.0: Simple grid-point packing.
    Simple {
        /// Number of bits per packed value (e.g., 16, 24).
        /// If 0, automatically chosen based on data range.
        bits_per_value: u8,
    },
    /// Template 5.3: group packing with first or second spatial differences.
    /// `bits_per_value` controls quantization before differencing (0 = 16).
    ComplexSpatial { bits_per_value: u8, order: u8 },
    /// Template 5.4: IEEE 754 binary32 values, big-endian (precision code 1).
    /// Lossless for values that are binary32 already: each value's 32 bits
    /// are written as they are, with no scaling or rounding (a binary64
    /// value is narrowed to binary32, so pass binary32 values widened).
    Ieee32,
}

/// A continuous time interval encoded using product template 4.8.
/// The product's forecast time is the START of this interval.
#[derive(Debug, Clone)]
pub struct StatisticalInterval {
    pub end_time: NaiveDateTime,
    /// WMO Code Table 4.10: 0 = average, 1 = accumulation, 2 = maximum,
    /// 3 = minimum. Other defined table values are also accepted.
    pub statistical_process: u8,
    /// WMO Code Table 4.4 (1 = hours, 13 = seconds).
    pub time_unit: u8,
    pub length: u32,
}

impl Default for PackingMethod {
    fn default() -> Self {
        PackingMethod::Simple { bits_per_value: 16 }
    }
}

/// Builder for a single GRIB2 message (one field/variable).
#[derive(Debug, Clone)]
pub struct MessageBuilder {
    discipline: u8,
    center: u16,
    subcenter: u16,
    reference_time: NaiveDateTime,
    grid: GridDefinition,
    product: ProductDefinition,
    values: Vec<f64>,
    bitmap: Option<Vec<bool>>,
    packing: PackingMethod,
    earth_radius: f64,
    earth_shape: u8,
    resolution_flags: Option<u8>,
    generating_process_identifier: u8,
    southern_pole: Option<(f64, f64)>,
    interval: Option<StatisticalInterval>,
    second_surface: Option<(u8, f64)>,
    local_use: Option<Vec<u8>>,
    local_table_version: u8,
    master_table_version: u8,
}

impl MessageBuilder {
    /// Create a new message builder with the given discipline and data values.
    ///
    /// - `discipline`: WMO discipline (0=Meteorological, 1=Hydrological, 2=Land surface, 10=Oceanographic)
    /// - `values`: The data values for all grid points (ny * nx elements).
    pub fn new(discipline: u8, values: Vec<f64>) -> Self {
        Self {
            discipline,
            center: 0, // 0 = WMO Secretariat
            subcenter: 0,
            reference_time: chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition::default(),
            product: ProductDefinition::default(),
            values,
            bitmap: None,
            packing: PackingMethod::default(),
            earth_radius: 6_370_000.0,
            earth_shape: 1,
            resolution_flags: None,
            generating_process_identifier: 0,
            southern_pole: None,
            interval: None,
            second_surface: None,
            local_use: None,
            local_table_version: 1,
            master_table_version: 35,
        }
    }

    /// Set the originating center and subcenter.
    pub fn center(mut self, center: u16, subcenter: u16) -> Self {
        self.center = center;
        self.subcenter = subcenter;
        self
    }

    /// Set the reference time.
    pub fn reference_time(mut self, time: NaiveDateTime) -> Self {
        self.reference_time = time;
        self
    }

    /// Set the grid definition.
    pub fn grid(mut self, grid: GridDefinition) -> Self {
        self.grid = grid;
        self
    }

    /// Set the product definition.
    pub fn product(mut self, product: ProductDefinition) -> Self {
        self.product = product;
        self
    }

    /// Set the packing method.
    pub fn packing(mut self, method: PackingMethod) -> Self {
        self.packing = method;
        self
    }

    /// Set the spherical radius in metres. WRF uses 6,370,000 m.
    pub fn earth_radius(mut self, meters: f64) -> Self {
        self.earth_radius = meters;
        self.earth_shape = 1;
        self
    }

    /// Earth shape 1 specifies the supplied radius; shape 6 is the WMO
    /// standard 6,371,229 m sphere used by UPP grid definitions.
    pub fn earth_shape(mut self, shape: u8) -> Self {
        self.earth_shape = shape;
        self
    }

    /// Grid component flags: bit 5 (0x08) denotes grid-relative winds.
    pub fn resolution_flags(mut self, flags: u8) -> Self {
        self.resolution_flags = Some(flags);
        self
    }

    /// Originating-center analysis/forecast process identifier.
    pub fn generating_process_identifier(mut self, identifier: u8) -> Self {
        self.generating_process_identifier = identifier;
        self
    }

    /// Set the Lambert template's optional southern projection pole.
    pub fn southern_pole(mut self, latitude: f64, longitude: f64) -> Self {
        self.southern_pole = Some((latitude, longitude));
        self
    }

    /// Encode a statistical interval using PDT 4.8.
    pub fn statistical_interval(mut self, interval: StatisticalInterval) -> Self {
        self.interval = Some(interval);
        self
    }

    /// Set a second fixed surface, defining a vertical layer.
    pub fn second_surface(mut self, level_type: u8, level_value: f64) -> Self {
        self.second_surface = Some((level_type, level_value));
        self
    }

    /// Set the local table version used by local parameter definitions.
    pub fn local_table_version(mut self, version: u8) -> Self {
        self.local_table_version = version;
        self
    }

    /// Select the WMO master parameter table version (default 35).
    pub fn master_table_version(mut self, version: u8) -> Self {
        self.master_table_version = version;
        self
    }

    /// Include an opaque Section 2 local definition, for example UTF-8 JSON
    /// describing a local parameter's label, units, and calculation.
    pub fn local_use(mut self, definition: Vec<u8>) -> Self {
        self.local_use = Some(definition);
        self
    }

    /// Set a bitmap (true = value present, false = missing).
    /// When a bitmap is used, only values where `bitmap[i] == true` are packed.
    /// The bitmap length must match the values length.
    pub fn bitmap(mut self, bitmap: Vec<bool>) -> Self {
        self.bitmap = Some(bitmap);
        self
    }
}

/// Builder for creating complete GRIB2 files containing one or more messages.
#[derive(Debug, Clone)]
pub struct Grib2Writer {
    messages: Vec<MessageBuilder>,
}

impl Grib2Writer {
    /// Create a new empty GRIB2 writer.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Add a message/field to the GRIB2 file.
    pub fn add_message(mut self, msg: MessageBuilder) -> Self {
        self.messages.push(msg);
        self
    }

    /// Write the complete GRIB2 file to bytes.
    ///
    /// Each message is written as a separate GRIB2 message (with its own
    /// indicator and end sections), concatenated together as is standard
    /// for multi-message GRIB2 files.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for msg in &self.messages {
            let msg_bytes = encode_message(msg)?;
            out.extend_from_slice(&msg_bytes);
        }
        Ok(out)
    }

    /// Write to a file.
    pub fn write_file(&self, path: &str) -> Result<(), String> {
        let data = self.to_bytes()?;
        std::fs::write(path, &data)
            .map_err(|e| format!("Failed to write GRIB2 file '{}': {}", path, e))
    }
}

impl Default for Grib2Writer {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════
// Internal encoding functions
// ═══════════════════════════════════════════════════════════

/// Encode a single GRIB2 message to bytes.
fn encode_message(msg: &MessageBuilder) -> Result<Vec<u8>, String> {
    // We build sections 1, 3, 4, 5, 6, 7 first, then prepend section 0
    // and append section 8, computing total length.
    let total_points = msg.grid.nx as usize * msg.grid.ny as usize;
    if total_points != msg.values.len() {
        return Err(format!(
            "Grid expects {} points ({}x{}), but values has {} elements",
            total_points,
            msg.grid.nx,
            msg.grid.ny,
            msg.values.len()
        ));
    }

    let sec1 = encode_section1(msg);
    let sec2 = match &msg.local_use {
        Some(definition) => {
            let mut sec = Vec::with_capacity(5 + definition.len());
            sec.extend_from_slice(&((5 + definition.len()) as u32).to_be_bytes());
            sec.push(2);
            sec.extend_from_slice(definition);
            sec
        }
        None => Vec::new(),
    };
    let mut sec3 = encode_section3(&msg.grid)?;
    if !msg.earth_radius.is_finite() || !(1.0..=u32::MAX as f64).contains(&msg.earth_radius) {
        return Err("Earth radius must be finite and representable in metres".into());
    }
    match msg.earth_shape {
        1 => {
            sec3[14] = 1;
            sec3[15] = 0;
            sec3[16..20].copy_from_slice(&(msg.earth_radius.round() as u32).to_be_bytes());
            sec3[20..30].fill(255);
        }
        6 => { sec3[14] = 6; sec3[15..30].fill(0); }
        shape => return Err(format!("Earth shape {shape} needs geometry this sphere-only writer does not encode")),
    }
    if let Some(flags) = msg.resolution_flags {
        sec3[if msg.grid.template == 0 { 54 } else { 46 }] = flags;
    }
    if let Some((latitude, longitude)) = msg.southern_pole {
        if msg.grid.template != 30 { return Err("Southern projection pole is only defined for Lambert template 3.30".into()); }
        sec3[73..77].copy_from_slice(&encode_signed_u32(latitude, 1_000_000.0));
        sec3[77..81].copy_from_slice(&encode_longitude(longitude));
    }
    let sec4 = encode_section4(msg)?;

    // Determine which values to pack and which bitmap to emit.
    let (bitmap, pack_values) = prepare_bitmap_and_values(msg)?;

    let (sec5, sec7) = encode_data(&pack_values, &msg.packing)?;
    let sec6 = encode_section6(&bitmap, total_points);

    // Total length = sec0(16) + sec1 + sec3 + sec4 + sec5 + sec6 + sec7 + sec8(4)
    let total_length: u64 = 16
        + sec1.len() as u64
        + sec2.len() as u64
        + sec3.len() as u64
        + sec4.len() as u64
        + sec5.len() as u64
        + sec6.len() as u64
        + sec7.len() as u64
        + 4;

    let mut out = Vec::with_capacity(total_length as usize);

    // Section 0: Indicator Section (16 bytes)
    out.extend_from_slice(b"GRIB"); // octets 1-4: "GRIB"
    out.extend_from_slice(&[0, 0]); // octets 5-6: reserved
    out.push(msg.discipline); // octet 7: discipline
    out.push(2); // octet 8: GRIB edition number = 2
    out.extend_from_slice(&total_length.to_be_bytes()); // octets 9-16: total length

    // Sections 1-7
    out.extend_from_slice(&sec1);
    out.extend_from_slice(&sec2);
    out.extend_from_slice(&sec3);
    out.extend_from_slice(&sec4);
    out.extend_from_slice(&sec5);
    out.extend_from_slice(&sec6);
    out.extend_from_slice(&sec7);

    // Section 8: End Section
    out.extend_from_slice(b"7777");

    debug_assert_eq!(out.len() as u64, total_length);

    Ok(out)
}

fn prepare_bitmap_and_values(
    msg: &MessageBuilder,
) -> Result<(Option<Vec<bool>>, Vec<f64>), String> {
    match &msg.bitmap {
        Some(bitmap) => {
            if bitmap.len() != msg.values.len() {
                return Err(format!(
                    "Bitmap length {} does not match values length {}",
                    bitmap.len(),
                    msg.values.len()
                ));
            }

            let mut pack_values =
                Vec::with_capacity(bitmap.iter().filter(|&&present| present).count());
            for (idx, (&value, &present)) in msg.values.iter().zip(bitmap.iter()).enumerate() {
                if present {
                    if !value.is_finite() {
                        return Err(format!(
                            "Non-finite value at index {} is marked present in the bitmap",
                            idx
                        ));
                    }
                    pack_values.push(value);
                }
            }

            Ok((Some(bitmap.clone()), pack_values))
        }
        None => {
            if msg.values.iter().all(|v| v.is_finite()) {
                Ok((None, msg.values.clone()))
            } else {
                let bitmap: Vec<bool> = msg.values.iter().map(|v| v.is_finite()).collect();
                let pack_values = msg
                    .values
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                Ok((Some(bitmap), pack_values))
            }
        }
    }
}

/// Section 1: Identification Section.
///
/// 21 bytes total (standard for GRIB2 Section 1).
fn encode_section1(msg: &MessageBuilder) -> Vec<u8> {
    let mut sec = Vec::with_capacity(21);
    let dt = msg.reference_time;
    sec.extend_from_slice(&21u32.to_be_bytes()); // length
    sec.push(1); // section number
    sec.extend_from_slice(&msg.center.to_be_bytes());
    sec.extend_from_slice(&msg.subcenter.to_be_bytes());
    sec.push(msg.master_table_version);
    sec.push(msg.local_table_version);
    sec.push(1); // significance of reference time

    sec.extend_from_slice(&(dt.year() as u16).to_be_bytes());
    sec.push(dt.month() as u8);
    sec.push(dt.day() as u8);
    sec.push(dt.hour() as u8);
    sec.push(dt.minute() as u8);
    sec.push(dt.second() as u8);
    // Octet 20: production status (0 = operational)
    sec.push(0);
    // Octet 21: type of processed data (1 = forecast)
    sec.push(1);

    debug_assert_eq!(sec.len(), 21);
    sec
}

/// Section 3: Grid Definition Section.
fn encode_section3(grid: &GridDefinition) -> Result<Vec<u8>, String> {
    match grid.template {
        0 => encode_grid_template_0(grid),
        10 => encode_grid_template_10(grid),
        20 => encode_grid_template_20(grid),
        30 => encode_grid_template_30(grid),
        _ => Err(format!(
            "Unsupported grid template {} for writing. Supported: 0, 10, 20, 30",
            grid.template
        )),
    }
}

/// Grid Definition Template 3.0: Latitude/Longitude (Equidistant Cylindrical).
///
/// Section 3 for template 0 is 72 bytes.
fn encode_grid_template_0(grid: &GridDefinition) -> Result<Vec<u8>, String> {
    let section_len: u32 = 72;
    let mut sec = Vec::with_capacity(section_len as usize);

    // Octets 1-4: length of section
    sec.extend_from_slice(&section_len.to_be_bytes());
    // Octet 5: section number
    sec.push(3);
    // Octet 6: source of grid definition (0 = specified in Code Table 3.1)
    sec.push(0);
    // Octets 7-10: number of data points
    let npoints = grid.nx * grid.ny;
    sec.extend_from_slice(&npoints.to_be_bytes());
    // Octet 11: number of octets for optional list of numbers
    sec.push(0);
    // Octet 12: interpretation of list of numbers
    sec.push(0);
    // Octets 13-14: grid definition template number
    sec.extend_from_slice(&0u16.to_be_bytes());

    // Template 3.0 specific fields:
    // Octet 15: shape of earth (6 = spherical, radius 6371229m)
    sec.push(6);
    // Octet 16: scale factor of radius
    sec.push(0);
    // Octets 17-20: scaled value of radius
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octet 21: scale factor of major axis
    sec.push(0);
    // Octets 22-25: scaled value of major axis
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octet 26: scale factor of minor axis
    sec.push(0);
    // Octets 27-30: scaled value of minor axis
    sec.extend_from_slice(&0u32.to_be_bytes());

    // Octets 31-34: Ni (nx)
    sec.extend_from_slice(&grid.nx.to_be_bytes());
    // Octets 35-38: Nj (ny)
    sec.extend_from_slice(&grid.ny.to_be_bytes());

    // Octets 39-42: basic angle (0)
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octets 43-46: subdivisions of basic angle (0 = use 10^6)
    sec.extend_from_slice(&0u32.to_be_bytes());

    // Octets 47-50: La1 (latitude of first grid point), signed, microdegrees
    sec.extend_from_slice(&encode_signed_u32(grid.lat1, 1_000_000.0));
    // Octets 51-54: Lo1 (longitude of first grid point), signed, microdegrees
    sec.extend_from_slice(&encode_longitude(grid.lon1));
    // Octet 55: resolution and component flags
    sec.push(0x30); // bit 3+4 set: i and j direction increments given
                    // Octets 56-59: La2 (latitude of last grid point)
    sec.extend_from_slice(&encode_signed_u32(grid.lat2, 1_000_000.0));
    // Octets 60-63: Lo2 (longitude of last grid point)
    sec.extend_from_slice(&encode_longitude(grid.lon2));
    // Octets 64-67: Di (i direction increment), unsigned microdegrees
    sec.extend_from_slice(&((grid.dx * 1_000_000.0).round() as u32).to_be_bytes());
    // Octets 68-71: Dj (j direction increment), unsigned microdegrees
    sec.extend_from_slice(&((grid.dy * 1_000_000.0).round() as u32).to_be_bytes());
    // Octet 72: scanning mode
    sec.push(grid.scan_mode);

    debug_assert_eq!(sec.len(), section_len as usize);
    Ok(sec)
}

/// Grid Definition Template 3.30: Lambert Conformal.
///
/// Section 3 for template 30 is 81 bytes.
fn encode_grid_template_30(grid: &GridDefinition) -> Result<Vec<u8>, String> {
    let section_len: u32 = 81;
    let mut sec = Vec::with_capacity(section_len as usize);

    // Octets 1-4: length
    sec.extend_from_slice(&section_len.to_be_bytes());
    // Octet 5: section number
    sec.push(3);
    // Octet 6: source of grid definition
    sec.push(0);
    // Octets 7-10: number of data points
    let npoints = grid.nx * grid.ny;
    sec.extend_from_slice(&npoints.to_be_bytes());
    // Octet 11: optional list
    sec.push(0);
    // Octet 12: interpretation
    sec.push(0);
    // Octets 13-14: template number = 30
    sec.extend_from_slice(&30u16.to_be_bytes());

    // Template 3.30 specific fields:
    // Octet 15: shape of earth (6 = spherical 6371229m)
    sec.push(6);
    // Octet 16: scale factor of radius
    sec.push(0);
    // Octets 17-20: scaled value of radius
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octet 21: scale factor of major axis
    sec.push(0);
    // Octets 22-25: scaled value of major axis
    sec.extend_from_slice(&0u32.to_be_bytes());
    // Octet 26: scale factor of minor axis
    sec.push(0);
    // Octets 27-30: scaled value of minor axis
    sec.extend_from_slice(&0u32.to_be_bytes());

    // Octets 31-34: Nx
    sec.extend_from_slice(&grid.nx.to_be_bytes());
    // Octets 35-38: Ny
    sec.extend_from_slice(&grid.ny.to_be_bytes());
    // Octets 39-42: La1 (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(grid.lat1, 1_000_000.0));
    // Octets 43-46: Lo1 (microdegrees)
    sec.extend_from_slice(&encode_longitude(grid.lon1));
    // Octet 47: resolution and component flags
    sec.push(0x30);

    // Octets 48-51: LaD (latitude where Dx/Dy are specified), for Lambert this
    // is sometimes set to latin1. Use latin1 if lad is 0.
    let lad = if grid.lad != 0.0 {
        grid.lad
    } else {
        grid.latin1
    };
    sec.extend_from_slice(&encode_signed_u32(lad, 1_000_000.0));
    // Octets 52-55: LoV (microdegrees)
    sec.extend_from_slice(&encode_longitude(grid.lov));
    // Octets 56-59: Dx (millimeters)
    sec.extend_from_slice(&((grid.dx * 1000.0).round() as u32).to_be_bytes());
    // Octets 60-63: Dy (millimeters)
    sec.extend_from_slice(&((grid.dy * 1000.0).round() as u32).to_be_bytes());
    // Octet 64: projection center flag
    sec.push(grid.projection_center_flag);
    // Octet 65: scanning mode
    sec.push(grid.scan_mode);
    // Octets 66-69: Latin1 (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(grid.latin1, 1_000_000.0));
    // Octets 70-73: Latin2 (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(grid.latin2, 1_000_000.0));

    // Octets 74-77: Latitude of southern pole (microdegrees)
    sec.extend_from_slice(&encode_signed_u32(-90.0, 1_000_000.0));
    // Octets 78-81: Longitude of southern pole (microdegrees)
    sec.extend_from_slice(&0u32.to_be_bytes());

    debug_assert_eq!(sec.len(), section_len as usize);
    Ok(sec)
}

/// Template 3.10: Mercator. Grid spacing is in millimetres at LaD.
fn encode_grid_template_10(grid: &GridDefinition) -> Result<Vec<u8>, String> {
    let mut sec = grid_header(grid, 72);
    sec.extend_from_slice(&encode_signed_u32(grid.lat1, 1_000_000.0));
    sec.extend_from_slice(&encode_longitude(grid.lon1));
    sec.push(0x30); // increments given; earth-relative wind components
    sec.extend_from_slice(&encode_signed_u32(grid.lad, 1_000_000.0));
    sec.extend_from_slice(&encode_signed_u32(grid.lat2, 1_000_000.0));
    sec.extend_from_slice(&encode_longitude(grid.lon2));
    sec.push(grid.scan_mode);
    sec.extend_from_slice(&0u32.to_be_bytes()); // grid orientation
    sec.extend_from_slice(&((grid.dx * 1000.0).round() as u32).to_be_bytes());
    sec.extend_from_slice(&((grid.dy * 1000.0).round() as u32).to_be_bytes());
    debug_assert_eq!(sec.len(), 72);
    Ok(sec)
}

/// Template 3.20: polar stereographic, with the hemisphere in flag bit 1.
fn encode_grid_template_20(grid: &GridDefinition) -> Result<Vec<u8>, String> {
    let mut sec = grid_header(grid, 65);
    sec.extend_from_slice(&encode_signed_u32(grid.lat1, 1_000_000.0));
    sec.extend_from_slice(&encode_longitude(grid.lon1));
    sec.push(0); // earth-relative wind components
    sec.extend_from_slice(&encode_signed_u32(grid.lad, 1_000_000.0));
    sec.extend_from_slice(&encode_longitude(grid.lov));
    sec.extend_from_slice(&((grid.dx * 1000.0).round() as u32).to_be_bytes());
    sec.extend_from_slice(&((grid.dy * 1000.0).round() as u32).to_be_bytes());
    sec.push(grid.projection_center_flag);
    sec.push(grid.scan_mode);
    debug_assert_eq!(sec.len(), 65);
    Ok(sec)
}

fn grid_header(grid: &GridDefinition, length: u32) -> Vec<u8> {
    let mut sec = Vec::with_capacity(length as usize);
    sec.extend_from_slice(&length.to_be_bytes());
    sec.extend_from_slice(&[3, 0]);
    sec.extend_from_slice(&(grid.nx * grid.ny).to_be_bytes());
    sec.extend_from_slice(&[0, 0]);
    sec.extend_from_slice(&grid.template.to_be_bytes());
    sec.extend_from_slice(&[1, 0]);
    sec.extend_from_slice(&6_370_000u32.to_be_bytes());
    sec.extend_from_slice(&[255; 10]);
    sec.extend_from_slice(&grid.nx.to_be_bytes());
    sec.extend_from_slice(&grid.ny.to_be_bytes());
    sec
}

/// Section 4: Product Definition Section.
///
/// Template 4.0: Analysis or forecast at a horizontal level at a point in time.
/// 34 bytes total.
fn encode_section4(msg: &MessageBuilder) -> Result<Vec<u8>, String> {
    let prod = &msg.product;
    if prod.template != 0 && prod.template != 8 {
        return Err(format!(
            "Unsupported product template {} for writing",
            prod.template
        ));
    }
    if prod.template == 8 && msg.interval.is_none() {
        return Err("Product template 4.8 requires a statistical interval".into());
    }
    if prod.forecast_time > 0x7fff_ffff {
        return Err("Forecast time exceeds the positive GRIB sign-magnitude range".into());
    }
    if let Some(interval) = &msg.interval {
        fn seconds(unit: u8) -> Option<i64> {
            match unit {
                0 => Some(60),
                1 => Some(3600),
                2 => Some(86400),
                10 => Some(10800),
                11 => Some(21600),
                12 => Some(43200),
                13 => Some(1),
                _ => None,
            }
        }
        if let (Some(start_unit), Some(length_unit)) =
            (seconds(prod.time_range_unit), seconds(interval.time_unit))
        {
            let elapsed =
                prod.forecast_time as i64 * start_unit + interval.length as i64 * length_unit;
            let expected = msg
                .reference_time
                .checked_add_signed(chrono::Duration::seconds(elapsed));
            if expected != Some(interval.end_time) {
                return Err("Statistical interval end must equal reference time plus forecast start plus window length".into());
            }
        }
    }

    let section_len: u32 = if msg.interval.is_some() { 58 } else { 34 };
    let mut sec = Vec::with_capacity(section_len as usize);

    // Octets 1-4: length
    sec.extend_from_slice(&section_len.to_be_bytes());
    // Octet 5: section number
    sec.push(4);
    // Octets 6-7: number of coordinate values after template (0)
    sec.extend_from_slice(&0u16.to_be_bytes());
    // Octets 8-9: product definition template number
    sec.extend_from_slice(&(if msg.interval.is_some() { 8u16 } else { 0u16 }).to_be_bytes());

    // Template 4.0 fields:
    // Octet 10: parameter category
    sec.push(prod.parameter_category);
    // Octet 11: parameter number
    sec.push(prod.parameter_number);
    // Octet 12: type of generating process (2 = forecast)
    sec.push(prod.generating_process);
    // Octet 13: background generating process identifier
    sec.push(0);
    // Octet 14: analysis or forecast generating process identified
    sec.push(msg.generating_process_identifier);
    // Octets 15-16: hours of observational data cutoff after reference time
    sec.extend_from_slice(&0u16.to_be_bytes());
    // Octet 17: minutes of observational data cutoff after reference time
    sec.push(0);
    // Octet 18: indicator of unit of time range
    sec.push(prod.time_range_unit);
    // Octets 19-22: forecast time in units defined by octet 18
    sec.extend_from_slice(&prod.forecast_time.to_be_bytes());
    // Octet 23: type of first fixed surface (level type)
    sec.push(prod.level_type);

    // Octets 24-28: scale factor and scaled value of first fixed surface
    let (scale_factor, scaled_value) = encode_level_value(prod.level_value)?;
    sec.push(scale_factor);
    sec.extend_from_slice(&scaled_value.to_be_bytes());

    // Octet 29: type of second fixed surface (255 = missing)
    if let Some((level_type, value)) = msg.second_surface {
        sec.push(level_type);
        let (scale_factor, scaled_value) = encode_level_value(value)?;
        sec.push(scale_factor);
        sec.extend_from_slice(&scaled_value.to_be_bytes());
    } else {
        sec.extend_from_slice(&[255; 6]);
    }

    if let Some(interval) = &msg.interval {
        let dt = interval.end_time;
        sec.extend_from_slice(&(dt.year() as u16).to_be_bytes());
        sec.extend_from_slice(&[
            dt.month() as u8,
            dt.day() as u8,
            dt.hour() as u8,
            dt.minute() as u8,
            dt.second() as u8,
        ]);
        sec.push(1); // one time-range specification
        sec.extend_from_slice(&0u32.to_be_bytes()); // no missing source values
        sec.push(interval.statistical_process);
        sec.push(2); // fixed reference time, forecast time increments
        sec.push(interval.time_unit);
        sec.extend_from_slice(&interval.length.to_be_bytes());
        sec.push(255); // increment unit is not applicable to a continuous process
        sec.extend_from_slice(&0u32.to_be_bytes()); // continuous accumulation/extreme
    }

    debug_assert_eq!(sec.len(), section_len as usize);
    Ok(sec)
}

/// Section 5: Data Representation Section (Template 5.0: Simple Packing).
/// Section 7: Data Section (packed values).
///
/// Returns (section5_bytes, section7_bytes).
fn encode_data(values: &[f64], packing: &PackingMethod) -> Result<(Vec<u8>, Vec<u8>), String> {
    match packing {
        PackingMethod::Simple { bits_per_value } => encode_simple_packing(values, *bits_per_value),
        PackingMethod::ComplexSpatial {
            bits_per_value,
            order,
        } => encode_complex_spatial(values, *bits_per_value, *order),
        PackingMethod::Ieee32 => Ok(encode_ieee32(values)),
    }
}

/// Template 5.4 (IEEE floating point, precision 1 = 32-bit) and its data
/// section: every packed value's binary32 bits, big-endian, as they are.
fn encode_ieee32(values: &[f64]) -> (Vec<u8>, Vec<u8>) {
    let mut sec5 = Vec::with_capacity(12);
    sec5.extend_from_slice(&12u32.to_be_bytes()); // octets 1-4: length
    sec5.push(5); // octet 5: section number
    sec5.extend_from_slice(&(values.len() as u32).to_be_bytes()); // octets 6-9: values packed
    sec5.extend_from_slice(&4u16.to_be_bytes()); // octets 10-11: template 5.4
    sec5.push(1); // octet 12: precision, code table 5.7: 1 = IEEE 32-bit
    let mut data = Vec::with_capacity(values.len() * 4);
    for &v in values {
        data.extend_from_slice(&(v as f32).to_bits().to_be_bytes());
    }
    (sec5, encode_section7(&data))
}

/// Quantize once for either packing method: Y = R + X * 2^E.
/// R is rounded down to binary32 and aligned to the binary scale grid,
/// so zero is exactly representable and sign bounds survive packing.
fn quantize(values: &[f64], bits: u8) -> Result<(f32, i16, u8, Vec<u64>), String> {
    let bits = if bits == 0 { 16 } else { bits };
    if bits > 32 {
        return Err("Packing precision must be between 1 and 32 bits (0 = 16)".into());
    }
    if values.is_empty() {
        return Ok((0.0, 0, 0, Vec::new()));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err("Non-finite values must be represented by a bitmap".into());
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut reference = min as f32;
    if !reference.is_finite() || !(max as f32).is_finite() {
        return Err("Values exceed the GRIB binary32 reference range".into());
    }
    if min == max {
        return Ok((reference, 0, 0, vec![0; values.len()]));
    }
    if reference as f64 > min {
        reference = if reference == 0.0 {
            -f32::from_bits(1)
        } else if reference > 0.0 {
            f32::from_bits(reference.to_bits() - 1)
        } else {
            f32::from_bits(reference.to_bits() + 1)
        };
    }
    let max_int = (1u64 << bits) - 1;
    let exponent = ((max - reference as f64) / max_int as f64).log2().ceil();
    if !exponent.is_finite() || !(-32767.0..=32767.0).contains(&exponent) {
        return Err("Binary packing exponent is not representable".into());
    }
    let mut exponent = exponent as i16;
    loop {
        let step = 2.0_f64.powi(exponent as i32);
        if !step.is_finite() || step == 0.0 {
            return Err("Binary packing step is outside the floating-point range".into());
        }
        // A dyadic multiple that fits binary32 is exact on conversion. If the
        // reference has more significant bits, its binary32 spacing is already
        // coarser than the step, so it remains a dyadic multiple.
        reference = ((reference as f64 / step).floor() * step) as f32;
        if !reference.is_finite() {
            return Err("Aligned reference exceeds the GRIB binary32 range".into());
        }
        let max_code = ((max - reference as f64) / step).round();
        if max_code <= max_int as f64 {
            let ints: Vec<u64> = values
                .iter()
                .map(|v| ((v - reference as f64) / step).round() as u64)
                .collect();
            return Ok((reference, exponent, bits, ints));
        }
        // Alignment can add one code at the upper edge. Coarsen the step and
        // align again instead of clipping the maximum or exceeding the width.
        exponent = exponent
            .checked_add(1)
            .ok_or("Binary packing exponent is not representable")?;
    }
}

fn encode_simple_packing(values: &[f64], bits: u8) -> Result<(Vec<u8>, Vec<u8>), String> {
    let (reference, exponent, bits, ints) = quantize(values, bits)?;
    let mut data = vec![0u8; (ints.len() * bits as usize + 7) / 8];
    for (i, value) in ints.iter().enumerate() {
        write_bits(&mut data, i * bits as usize, bits as usize, *value);
    }
    Ok((
        encode_section5_simple(reference, exponent, 0, bits, values.len() as u32),
        encode_section7(&data),
    ))
}

fn bit_width(value: u64) -> u8 {
    (64 - value.leading_zeros()) as u8
}

/// Template 5.3 with general group splitting (5.4 code 1). Group widths adapt
/// to local differences; choose the fixed block length with the lowest actual
/// encoded bit count, including group descriptors. Values share the initial
/// quantization used by simple packing.
fn encode_complex_spatial(
    values: &[f64],
    bits: u8,
    order: u8,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    if order != 1 && order != 2 {
        return Err("Spatial difference order must be 1 or 2".into());
    }
    let (reference, exponent, _, ints) = quantize(values, bits)?;
    let order = if ints.len() < 2 { 1 } else { order };
    let mut diffs = vec![0i64; ints.len()];
    for i in order as usize..ints.len() {
        diffs[i] = if order == 1 {
            ints[i] as i64 - ints[i - 1] as i64
        } else {
            ints[i] as i64 - 2 * ints[i - 1] as i64 + ints[i - 2] as i64
        };
    }
    let minimum = diffs
        .iter()
        .skip(order as usize)
        .copied()
        .min()
        .unwrap_or(0);
    let adjusted: Vec<u64> = diffs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if i < order as usize {
                0
            } else {
                (v - minimum) as u64
            }
        })
        .collect();
    let extra_max = ints
        .iter()
        .take(order as usize)
        .copied()
        .max()
        .unwrap_or(0)
        .max(minimum.unsigned_abs());
    let extra_bytes = ((bit_width(extra_max) as usize + 1 + 7) / 8).max(1);
    let mut descriptors = vec![0; (order as usize + 1) * extra_bytes];
    for i in 0..order as usize {
        write_bits(
            &mut descriptors,
            i * extra_bytes * 8,
            extra_bytes * 8,
            ints.get(i).copied().unwrap_or(0),
        );
    }
    let sign_bit = if minimum < 0 {
        1u64 << (extra_bytes * 8 - 1)
    } else {
        0
    };
    write_bits(
        &mut descriptors,
        order as usize * extra_bytes * 8,
        extra_bytes * 8,
        sign_bit | minimum.unsigned_abs(),
    );

    let mut best = None;
    for block_length in [16usize, 32, 64, 128, 256] {
        let groups: Vec<(u64, u8, usize)> = adjusted
            .chunks(block_length)
            .map(|chunk| {
                let low = chunk.iter().copied().min().unwrap();
                let high = chunk.iter().copied().max().unwrap();
                (low, bit_width(high - low), chunk.len())
            })
            .collect();
        let ref_bits = bit_width(groups.iter().map(|g| g.0).max().unwrap_or(0));
        let width_ref = groups.iter().map(|g| g.1).min().unwrap_or(0);
        let width_bits =
            bit_width((groups.iter().map(|g| g.1).max().unwrap_or(0) - width_ref) as u64);
        let cost = (groups.len() * ref_bits as usize + 7) / 8 * 8
            + (groups.len() * width_bits as usize + 7) / 8 * 8
            + groups.iter().map(|g| g.1 as usize * g.2).sum::<usize>();
        if best
            .as_ref()
            .map_or(true, |(old_cost, _, _, _, _, _)| cost < *old_cost)
        {
            best = Some((cost, block_length, groups, ref_bits, width_ref, width_bits));
        }
    }
    let (_, block_length, groups, ref_bits, width_ref, width_bits) = best.unwrap();
    let mut data = descriptors;
    append_bit_array(&mut data, ref_bits, groups.iter().map(|g| g.0));
    append_bit_array(
        &mut data,
        width_bits,
        groups.iter().map(|g| (g.1 - width_ref) as u64),
    );
    // Group lengths have zero bit width: block_length, except explicit final length.
    let start = data.len() * 8;
    let total_bits: usize = groups.iter().map(|g| g.1 as usize * g.2).sum();
    data.resize(data.len() + (total_bits + 7) / 8, 0);
    let mut offset = start;
    let mut index = 0;
    for &(group_ref, width, length) in &groups {
        for &value in &adjusted[index..index + length] {
            write_bits(&mut data, offset, width as usize, value - group_ref);
            offset += width as usize;
        }
        index += length;
    }
    let mut sec = encode_section5_simple(reference, exponent, 0, ref_bits, values.len() as u32);
    sec[0..4].copy_from_slice(&49u32.to_be_bytes());
    sec[9..11].copy_from_slice(&3u16.to_be_bytes());
    sec.push(1); // general group splitting
    sec.push(0); // missing values are handled by Section 6
    sec.extend_from_slice(&[255; 8]); // missing value substitutes not used
    sec.extend_from_slice(&(groups.len() as u32).to_be_bytes());
    sec.push(width_ref);
    sec.push(width_bits);
    sec.extend_from_slice(&(block_length as u32).to_be_bytes());
    sec.push(1); // group length increment
    sec.extend_from_slice(&(groups.last().map_or(0, |g| g.2) as u32).to_be_bytes());
    sec.push(0); // fixed group length, no length bitstream
    sec.push(order);
    sec.push(extra_bytes as u8);
    debug_assert_eq!(sec.len(), 49);
    Ok((sec, encode_section7(&data)))
}

fn append_bit_array(out: &mut Vec<u8>, width: u8, values: impl Iterator<Item = u64>) {
    let start = out.len() * 8;
    let values: Vec<_> = values.collect();
    out.resize(out.len() + (values.len() * width as usize + 7) / 8, 0);
    for (i, value) in values.iter().enumerate() {
        write_bits(out, start + i * width as usize, width as usize, *value);
    }
}

/// Build Section 5 for simple packing (Template 5.0).
///
/// 21 bytes total.
fn encode_section5_simple(
    reference_value: f32,
    binary_scale: i16,
    decimal_scale: i16,
    bits_per_value: u8,
    num_points: u32,
) -> Vec<u8> {
    let section_len: u32 = 21;
    let mut sec = Vec::with_capacity(section_len as usize);

    // Octets 1-4: length
    sec.extend_from_slice(&section_len.to_be_bytes());
    // Octet 5: section number
    sec.push(5);
    // Octets 6-9: number of data points
    sec.extend_from_slice(&num_points.to_be_bytes());
    // Octets 10-11: data representation template number (0 = simple packing)
    sec.extend_from_slice(&0u16.to_be_bytes());

    // Template 5.0 fields:
    // Octets 12-15: reference value (IEEE 754 single precision)
    sec.extend_from_slice(&reference_value.to_be_bytes());
    // Octets 16-17: binary scale factor (sign-magnitude)
    sec.extend_from_slice(&encode_signed_u16_grib(binary_scale));
    // Octets 18-19: decimal scale factor (sign-magnitude)
    sec.extend_from_slice(&encode_signed_u16_grib(decimal_scale));
    // Octet 20: number of bits per packed value
    sec.push(bits_per_value);
    // Octet 21: type of original field values (0 = floating point)
    sec.push(0);

    debug_assert_eq!(sec.len(), section_len as usize);
    sec
}

/// Section 6: Bitmap Section.
fn encode_section6(bitmap: &Option<Vec<bool>>, total_points: usize) -> Vec<u8> {
    match bitmap {
        None => {
            // No bitmap, indicator = 255
            let section_len: u32 = 6;
            let mut sec = Vec::with_capacity(section_len as usize);
            sec.extend_from_slice(&section_len.to_be_bytes());
            sec.push(6); // section number
            sec.push(255); // bitmap indicator: not present
            sec
        }
        Some(bm) => {
            // Bitmap present, indicator = 0
            // Pack bits: 1 byte per 8 grid points, MSB first
            let bitmap_bytes = (total_points + 7) / 8;
            let section_len = 6 + bitmap_bytes as u32;
            let mut sec = Vec::with_capacity(section_len as usize);
            sec.extend_from_slice(&section_len.to_be_bytes());
            sec.push(6); // section number
            sec.push(0); // bitmap indicator: bitmap follows

            let mut bytes = vec![0u8; bitmap_bytes];
            for (i, &present) in bm.iter().enumerate().take(total_points) {
                if present {
                    let byte_idx = i / 8;
                    let bit_idx = 7 - (i % 8);
                    bytes[byte_idx] |= 1 << bit_idx;
                }
            }
            sec.extend_from_slice(&bytes);

            debug_assert_eq!(sec.len(), section_len as usize);
            sec
        }
    }
}

/// Section 7: Data Section.
fn encode_section7(packed_data: &[u8]) -> Vec<u8> {
    let section_len = 5 + packed_data.len() as u32;
    let mut sec = Vec::with_capacity(section_len as usize);
    sec.extend_from_slice(&section_len.to_be_bytes());
    sec.push(7); // section number
    sec.extend_from_slice(packed_data);
    sec
}

// ═══════════════════════════════════════════════════════════
// Helper functions
// ═══════════════════════════════════════════════════════════

/// Encode a floating point value as a signed 32-bit GRIB2 value
/// using sign-magnitude format (MSB = sign, rest = magnitude).
///
/// The value is multiplied by `scale` first (e.g., 1_000_000 for microdegrees).
fn encode_signed_u32(value: f64, scale: f64) -> [u8; 4] {
    let scaled = (value * scale).round() as i64;
    let sign: u32 = if scaled < 0 { 1 << 31 } else { 0 };
    let magnitude = scaled.unsigned_abs() as u32 & 0x7FFF_FFFF;
    (sign | magnitude).to_be_bytes()
}

/// Encode a signed 16-bit value in GRIB2 sign-magnitude format.
fn encode_signed_u16_grib(value: i16) -> [u8; 2] {
    let sign: u16 = if value < 0 { 1 << 15 } else { 0 };
    let magnitude = value.unsigned_abs() & 0x7FFF;
    (sign | magnitude).to_be_bytes()
}

/// Encode a level value into (scale_factor, scaled_value) for Section 4.
///
/// For integer levels (e.g., 2 m, 10 m, 100000 Pa), the scale factor is 0
/// and scaled value is the integer. For fractional levels, we find the
/// smallest scale factor that represents the value exactly.
fn encode_longitude(value: f64) -> [u8; 4] {
    ((value.rem_euclid(360.0) * 1_000_000.0).round() as u32 % 360_000_000).to_be_bytes()
}

fn encode_level_value(value: f64) -> Result<(u8, u32), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(
            "Fixed surface value must be finite and nonnegative (unsigned GRIB level)".into(),
        );
    }
    for sf in 0u8..7 {
        let scaled = value * 10.0_f64.powi(sf as i32);
        let rounded = scaled.round();
        if (scaled - rounded).abs() < 1e-6 && rounded < u32::MAX as f64 {
            return Ok((sf, rounded as u32));
        }
    }
    Err(format!(
        "Fixed surface value {} cannot be represented accurately",
        value
    ))
}

/// Big-endian write, one partial or full octet at a time.
fn write_bits(buf: &mut [u8], mut offset: usize, mut count: usize, value: u64) {
    while count != 0 {
        let available = 8 - offset % 8;
        let take = available.min(count);
        let mask = (1u64 << take) - 1;
        let chunk = (value >> (count - take)) & mask;
        buf[offset / 8] |= (chunk << (available - take)) as u8;
        offset += take;
        count -= take;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grib2::parser::{Grib2File, GridDefinition, ProductDefinition};
    use crate::grib2::unpack::unpack_message;

    fn square_grid(nx: u32, ny: u32) -> GridDefinition {
        GridDefinition {
            nx,
            ny,
            lat1: -30.0,
            lon1: -100.0,
            lat2: -30.0 + ny as f64 - 1.0,
            lon2: -100.0 + nx as f64 - 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0x40,
            ..Default::default()
        }
    }

    fn section(bytes: &[u8], number: u8) -> &[u8] {
        let mut offset = 16;
        while offset < bytes.len() - 4 {
            let len = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            if bytes[offset + 4] == number {
                return &bytes[offset..offset + len];
            }
            offset += len;
        }
        panic!("Section {} absent", number);
    }

    #[test]
    fn checked_in_golden_fixture_roundtrip() {
        let bytes = include_bytes!("fixtures/native-writer.grib2");
        let parsed = Grib2File::from_bytes(bytes).unwrap();
        assert_eq!(parsed.messages.len(), 17);
        for (i, message) in parsed.messages.iter().enumerate() {
            let values = unpack_message(message).unwrap();
            assert_eq!(values.len(), 35);
            if i < 12 {
                assert_eq!(message.grid.template, [0, 10, 20, 30][i / 3]);
                for (j, value) in values.iter().enumerate() {
                    if j == 6 {
                        assert!(value.is_nan());
                    } else {
                        assert_eq!(*value, 270.0 + j as f64 * 0.125);
                    }
                }
            } else if i == 13 {
                assert!(values.iter().all(|v| v.is_nan()));
            } else {
                assert!(values.iter().all(|v| *v == 1.0));
            }
        }
    }

    /// Independent validation against installed ecCodes and wgrib2 tools.
    /// Both paths are explicit so ordinary unit tests require no external tools.
    #[test]
    fn golden_fixture_external_decoders() {
        let Some(wgrib2) = std::env::var_os("GRIB2_WRITER_WGRIB2") else {
            return;
        };
        let Some(grib_get_data) = std::env::var_os("GRIB2_WRITER_ECCODES_DATA") else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("grib2-external-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("golden.grib2");
        let output = dir.join("decoded.txt");
        std::fs::write(&file, include_bytes!("fixtures/native-writer.grib2")).unwrap();
        for i in 1..=17 {
            let result = std::process::Command::new(&wgrib2)
                .arg(&file)
                .args(["-d", &i.to_string(), "-text"])
                .arg(&output)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "wgrib2 message {i}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            let text = std::fs::read_to_string(&output).unwrap();
            let values: Vec<f64> = text
                .lines()
                .skip(1)
                .map(|line| line.parse().unwrap())
                .collect();
            assert_eq!(values.len(), 35);
            for (j, actual) in values.iter().enumerate() {
                if i == 14 || (i <= 12 && j == 6) {
                    assert!(*actual > 9.0e20);
                } else {
                    let expected = if i <= 12 {
                        270.0 + j as f64 * 0.125
                    } else {
                        1.0
                    };
                    assert_eq!(*actual, expected, "wgrib2 message {i}, point {j}");
                }
            }
            let result = std::process::Command::new(&grib_get_data)
                .args(["-m", "MISSING", "-w", &format!("count={i}"), "-F", "%.17g"])
                .arg(&file)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "ecCodes message {i}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            let text = String::from_utf8(result.stdout).unwrap();
            let values: Vec<&str> = text
                .lines()
                .skip(1)
                .map(|line| line.split_whitespace().last().unwrap())
                .collect();
            assert_eq!(values.len(), 35);
            for (j, actual) in values.iter().enumerate() {
                if i == 14 || (i <= 12 && j == 6) {
                    assert_eq!(*actual, "MISSING");
                } else {
                    let expected = if i <= 12 {
                        270.0 + j as f64 * 0.125
                    } else {
                        1.0
                    };
                    assert_eq!(
                        actual.parse::<f64>().unwrap(),
                        expected,
                        "ecCodes message {i}, point {j}"
                    );
                }
            }
        }
        // Only remove exact files created by this test.
        std::fs::remove_file(output).unwrap();
        std::fs::remove_file(file).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn packing_preserves_exact_zero_and_physical_sign_bounds() {
        for min in [-110.102, -30_000.102, -0.0039, -6_001.1] {
            for positive in [false, true] {
                let values = if positive {
                    vec![0.0, -min * 0.001, -min]
                } else {
                    vec![min, min * 0.001, 0.0]
                };
                for bits in [1, 7, 16, 20, 24, 32] {
                    for packing in [
                        PackingMethod::Simple {
                            bits_per_value: bits,
                        },
                        PackingMethod::ComplexSpatial {
                            bits_per_value: bits,
                            order: 1,
                        },
                        PackingMethod::ComplexSpatial {
                            bits_per_value: bits,
                            order: 2,
                        },
                    ] {
                        let bytes = Grib2Writer::new()
                            .add_message(
                                MessageBuilder::new(0, values.clone())
                                    .grid(square_grid(3, 1))
                                    .packing(packing),
                            )
                            .to_bytes()
                            .unwrap();
                        let parsed = Grib2File::from_bytes(&bytes).unwrap();
                        let message = &parsed.messages[0];
                        let decoded = unpack_message(message).unwrap();
                        let step = 2f64.powi(message.data_rep.binary_scale as i32);
                        assert_eq!(decoded[if positive { 0 } else { 2 }], 0.0);
                        assert!(decoded.iter().all(|v| if positive {
                            *v >= 0.0
                        } else {
                            *v <= 0.0
                        }));
                        for (actual, expected) in decoded.iter().zip(&values) {
                            assert!(
                                (actual - expected).abs() <= step * 0.501,
                                "{actual} vs {expected}, step {step}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn alignment_rechecks_code_width_instead_of_clipping() {
        let values = [0.875, 127.875];
        let (reference, exponent, bits, codes) = quantize(&values, 7).unwrap();
        assert_eq!(reference, 0.0);
        assert_eq!(exponent, 1); // step 1 would need code 128 after alignment
        assert_eq!(bits, 7);
        assert!(codes.iter().all(|code| *code <= 127));
        for (original, code) in values.iter().zip(codes) {
            let decoded = reference as f64 + code as f64 * 2f64.powi(exponent as i32);
            assert!((decoded - original).abs() <= 1.0);
        }
    }

    #[test]
    fn complex_roundtrip_both_orders_and_precisions() {
        let values: Vec<f64> = (0..713)
            .map(|i| 250.0 + (i as f64 * 0.11).sin() * 18.0 + (i % 97) as f64 * 0.037)
            .collect();
        for order in [1, 2] {
            for bits in [1, 7, 12, 16, 24, 32] {
                let bytes = Grib2Writer::new()
                    .add_message(
                        MessageBuilder::new(0, values.clone())
                            .grid(square_grid(31, 23))
                            .packing(PackingMethod::ComplexSpatial {
                                bits_per_value: bits,
                                order,
                            }),
                    )
                    .to_bytes()
                    .unwrap();
                let parsed = Grib2File::from_bytes(&bytes).unwrap();
                let message = &parsed.messages[0];
                assert_eq!(message.data_rep.template, 3);
                assert_eq!(message.data_rep.spatial_diff_order, order);
                assert!(message.data_rep.num_groups > 1);
                let decoded = unpack_message(message).unwrap();
                let tolerance = 2f64.powi(message.data_rep.binary_scale as i32) * 0.501;
                assert_eq!(decoded.len(), values.len());
                for (i, (actual, expected)) in decoded.iter().zip(&values).enumerate() {
                    assert!((actual - expected).abs() <= tolerance,
                        "order {order}, bits {bits}, i {i}: {actual} vs {expected} (+/- {tolerance})");
                }
            }
        }
    }

    #[test]
    fn complex_preserves_bitmap_constant_all_missing_and_tiny_grids() {
        for values in [
            vec![273.15; 9],
            vec![f64::NAN; 9],
            vec![1.0, f64::NAN, 2.0, 4.0, 9.0, f64::INFINITY, -2.0, 7.0, 3.0],
        ] {
            for order in [1, 2] {
                let bytes = Grib2Writer::new()
                    .add_message(
                        MessageBuilder::new(0, values.clone())
                            .grid(square_grid(3, 3))
                            .packing(PackingMethod::ComplexSpatial {
                                bits_per_value: 16,
                                order,
                            }),
                    )
                    .to_bytes()
                    .unwrap();
                let parsed = Grib2File::from_bytes(&bytes).unwrap();
                let decoded = unpack_message(&parsed.messages[0]).unwrap();
                assert_eq!(decoded.len(), 9);
                for (a, b) in decoded.iter().zip(&values) {
                    if b.is_finite() {
                        assert!((a - b).abs() < 0.001);
                    } else {
                        assert!(a.is_nan());
                    }
                }
            }
        }
        for n in [1, 2] {
            let values: Vec<_> = (0..n).map(|i| 270.5 + i as f64).collect();
            let bytes = Grib2Writer::new()
                .add_message(
                    MessageBuilder::new(0, values.clone())
                        .grid(square_grid(n as u32, 1))
                        .packing(PackingMethod::ComplexSpatial {
                            bits_per_value: 16,
                            order: 2,
                        }),
                )
                .to_bytes()
                .unwrap();
            let parsed = Grib2File::from_bytes(&bytes).unwrap();
            assert_eq!(unpack_message(&parsed.messages[0]).unwrap(), values);
        }
    }

    #[test]
    fn spatial_groups_compress_a_smooth_field() {
        let values: Vec<_> = (0..16_384).map(|i| i as f64 * 0.125).collect();
        let simple = Grib2Writer::new()
            .add_message(
                MessageBuilder::new(0, values.clone())
                    .grid(square_grid(128, 128))
                    .packing(PackingMethod::Simple { bits_per_value: 16 }),
            )
            .to_bytes()
            .unwrap();
        let complex = Grib2Writer::new()
            .add_message(
                MessageBuilder::new(0, values)
                    .grid(square_grid(128, 128))
                    .packing(PackingMethod::ComplexSpatial {
                        bits_per_value: 16,
                        order: 2,
                    }),
            )
            .to_bytes()
            .unwrap();
        assert!(
            complex.len() * 10 < simple.len(),
            "{} vs {}",
            complex.len(),
            simple.len()
        );
    }

    #[test]
    fn all_four_projections_preserve_grid_and_wrf_radius() {
        for template in [0, 10, 20, 30] {
            let mut grid = square_grid(3, 3);
            grid.template = template;
            if template != 0 {
                grid.dx = 3000.0;
                grid.dy = 4500.0;
            }
            grid.lad = -30.0;
            grid.latin1 = -30.0;
            grid.latin2 = -60.0;
            grid.lov = -95.0;
            grid.projection_center_flag = 0x80;
            let bytes = Grib2Writer::new()
                .add_message(MessageBuilder::new(0, vec![1.0; 9]).grid(grid.clone()))
                .to_bytes()
                .unwrap();
            let sec = section(&bytes, 3);
            assert_eq!(sec[14], 1);
            assert_eq!(
                u32::from_be_bytes(sec[16..20].try_into().unwrap()),
                6_370_000
            );
            let parsed = Grib2File::from_bytes(&bytes).unwrap();
            let decoded = &parsed.messages[0].grid;
            assert_eq!(decoded.template, template);
            assert_eq!(decoded.nx, grid.nx);
            assert_eq!(decoded.ny, grid.ny);
            assert_eq!(decoded.lat1, grid.lat1);
            assert_eq!(decoded.lon1, 260.0);
            assert_eq!(decoded.dx, grid.dx);
            assert_eq!(decoded.dy, grid.dy);
            assert_eq!(decoded.scan_mode, grid.scan_mode);
            if template != 0 {
                assert_eq!(decoded.lad, grid.lad);
            }
            if template == 20 || template == 30 {
                assert_eq!(decoded.projection_center_flag, 0x80);
                assert_eq!(decoded.lov, 265.0);
            }
        }
    }

    #[test]
    fn accumulation_and_extreme_windows_encode_pdt_48() {
        let reference = chrono::NaiveDate::from_ymd_opt(2024, 5, 25)
            .unwrap()
            .and_hms_opt(18, 0, 0)
            .unwrap();
        for (start, length, process) in [(0u32, 6u32, 1u8), (5, 1, 1), (5, 1, 2), (5, 1, 3)] {
            let end_time = reference + chrono::Duration::hours((start + length) as i64);
            let bytes = Grib2Writer::new()
                .add_message(
                    MessageBuilder::new(0, vec![1.0; 9])
                        .grid(square_grid(3, 3))
                        .reference_time(reference)
                        .product(ProductDefinition {
                            forecast_time: start,
                            time_range_unit: 1,
                            level_type: 1,
                            ..Default::default()
                        })
                        .statistical_interval(StatisticalInterval {
                            end_time,
                            statistical_process: process,
                            time_unit: 1,
                            length,
                        }),
                )
                .to_bytes()
                .unwrap();
            let sec = section(&bytes, 4);
            assert_eq!(sec.len(), 58);
            assert_eq!(u16::from_be_bytes(sec[7..9].try_into().unwrap()), 8);
            assert_eq!(u32::from_be_bytes(sec[18..22].try_into().unwrap()), start);
            assert_eq!(u16::from_be_bytes(sec[34..36].try_into().unwrap()), 2024);
            assert_eq!(&sec[36..41], &[5, 26, 0, 0, 0]);
            assert_eq!(sec[41], 1);
            assert_eq!(sec[46], process);
            assert_eq!(sec[47], 2);
            assert_eq!(sec[48], 1);
            assert_eq!(u32::from_be_bytes(sec[49..53].try_into().unwrap()), length);
            assert_eq!(sec[53], 255); // no discrete time increment
            assert_eq!(u32::from_be_bytes(sec[54..58].try_into().unwrap()), 0);
        }
    }

    #[test]
    fn upp_metadata_for_each_native_projection() {
        for template in [0, 10, 20, 30] {
            let mut grid = square_grid(7, 5);
            grid.template = template;
            grid.lat1 = 30.0;
            grid.lon1 = -100.0;
            grid.lat2 = 34.0;
            grid.lon2 = -94.0;
            grid.lad = 30.0;
            grid.lov = -95.0;
            grid.latin1 = 30.0;
            grid.latin2 = 60.0;
            if template != 0 {
                grid.dx = 3000.0;
                grid.dy = 3000.0;
            }
            let grid = rescale_projected_grid_radius(&grid,6_370_000.0,6_371_229.0).unwrap();
            let expected_dx=grid.dx;let expected_dy=grid.dy;
            let mut message = MessageBuilder::new(0, vec![1.5; 35]).grid(grid)
                    .earth_shape(6).resolution_flags(8)
                    .generating_process_identifier(116).master_table_version(2)
                    .local_table_version(1).second_surface(255, 0.0);
            if template == 30 {message=message.southern_pole(0.0, 0.0);}
            let bytes = Grib2Writer::new().add_message(message).to_bytes().unwrap();
            let identification = section(&bytes, 1);
            assert_eq!(&identification[9..11], &[2, 1]);
            let geometry = section(&bytes, 3);
            assert_eq!(geometry[14], 6);
            assert!(geometry[15..30].iter().all(|value| *value == 0));
            assert_eq!(geometry[if template == 0 {54} else {46}], 8);
            if template == 30 {assert!(geometry[73..81].iter().all(|value| *value == 0));}
            let product = section(&bytes, 4);
            assert_eq!(product[13], 116);
            assert_eq!(&product[28..34], &[255, 0, 0, 0, 0, 0]);
            let parsed = Grib2File::from_bytes(&bytes).unwrap();
            assert_eq!(parsed.messages[0].grid.template, template);
            let unit=if template==0 {1_000_000.0} else {1000.0};
            assert_eq!(parsed.messages[0].grid.dx,(expected_dx*unit).round()/unit);
            assert_eq!(parsed.messages[0].grid.dy,(expected_dy*unit).round()/unit);
            assert_eq!(unpack_message(&parsed.messages[0]).unwrap(), vec![1.5; 35]);

            // Optional independent decoders validate new earth/wind metadata
            // on every template, rather than only our internal parser.
            let (Some(wgrib2), Some(eccodes)) = (
                std::env::var_os("GRIB2_WRITER_WGRIB2"),
                std::env::var_os("GRIB2_WRITER_ECCODES_DATA"),
            ) else {continue;};
            let dir=std::env::temp_dir().join(format!("grib2-upp-grid-{}-{template}",std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let file=dir.join("native.grib2"); let decoded=dir.join("decoded.txt");
            std::fs::write(&file,&bytes).unwrap();
            let result=std::process::Command::new(wgrib2).arg(&file).arg("-text").arg(&decoded).output().unwrap();
            assert!(result.status.success(),"wgrib2 template {template}: {}",String::from_utf8_lossy(&result.stderr));
            let text=std::fs::read_to_string(&decoded).unwrap();
            assert_eq!(text.lines().skip(1).map(|v|v.parse::<f64>().unwrap()).collect::<Vec<_>>(),vec![1.5;35]);
            let result=std::process::Command::new(eccodes).args(["-F","%.17g"]).arg(&file).output().unwrap();
            assert!(result.status.success(),"ecCodes template {template}: {}",String::from_utf8_lossy(&result.stderr));
            let text=String::from_utf8(result.stdout).unwrap();
            assert_eq!(text.lines().skip(1).map(|v|v.split_whitespace().last().unwrap().parse::<f64>().unwrap()).collect::<Vec<_>>(),vec![1.5;35]);
            std::fs::remove_file(decoded).unwrap(); std::fs::remove_file(file).unwrap();std::fs::remove_dir(dir).unwrap();
        }
    }

    #[test]
    fn radius_transform_leaves_angular_geometry_unchanged() {
        let grid=square_grid(7,5);
        let encoded=rescale_projected_grid_radius(&grid,6_370_000.0,6_371_229.0).unwrap();
        assert_eq!((encoded.dx,encoded.dy,encoded.lat1,encoded.lon1,encoded.lat2,encoded.lon2),
            (grid.dx,grid.dy,grid.lat1,grid.lon1,grid.lat2,grid.lon2));
        for radius in [0.0,-1.0,f64::NAN,f64::INFINITY] {
            assert!(rescale_projected_grid_radius(&grid,radius,6_371_229.0).is_err());
            assert!(rescale_projected_grid_radius(&grid,6_370_000.0,radius).is_err());
        }
    }

    #[test]
    fn rescaled_sphere_preserves_projected_decoder_corners() {
        let (Some(wgrib2),Some(eccodes))=(std::env::var_os("GRIB2_WRITER_WGRIB2"),
            std::env::var_os("GRIB2_WRITER_ECCODES_DATA")) else {return;};
        let nx=31u32;let ny=21u32;
        for template in [10,20,30] {
            let mut grid=square_grid(nx,ny);grid.template=template;
            grid.dx=60_000.0;grid.dy=80_000.0;grid.lov=-100.0;
            grid.lat1=if template==20 {70.0} else {30.0};grid.lon1=-110.0;
            grid.lad=if template==20 {60.0} else {30.0};grid.latin1=30.0;grid.latin2=60.0;
            if template==10 {
                let scale=6_370_000.0*grid.lad.to_radians().cos();
                let y0=scale*(std::f64::consts::FRAC_PI_4+grid.lat1.to_radians()*0.5).tan().ln();
                grid.lat2=(2.0*((y0+(ny-1) as f64*grid.dy)/scale).exp().atan()-std::f64::consts::FRAC_PI_2).to_degrees();
                grid.lon2=grid.lon1+((nx-1) as f64*grid.dx/scale).to_degrees();
            }
            let corrected=rescale_projected_grid_radius(&grid,6_370_000.0,6_371_229.0).unwrap();
            let dir=std::env::temp_dir().join(format!("grib2-radius-{}-{template}",std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let mut coordinates=Vec::new();
            for (name,geometry,shape) in [("native",grid.clone(),1u8),("corrected",corrected,6),("uncorrected",grid,6)] {
                let bytes=Grib2Writer::new().add_message(MessageBuilder::new(0,vec![1.5;(nx*ny) as usize])
                    .grid(geometry).earth_radius(6_370_000.0).earth_shape(shape)
                    .resolution_flags(if shape==6 {8} else {48})).to_bytes().unwrap();
                let file=dir.join(format!("{name}.grib2"));std::fs::write(&file,bytes).unwrap();
                let result=std::process::Command::new(&eccodes).args(["-L","%.12f %.12f ","-F","%.17g"]).arg(&file).output().unwrap();
                assert!(result.status.success(),"ecCodes GDT{template}: {}",String::from_utf8_lossy(&result.stderr));
                let text=String::from_utf8(result.stdout).unwrap();
                let points=text.lines().skip(1).map(|row|{
                    let mut columns=row.split_whitespace();
                    (columns.next().unwrap().parse::<f64>().unwrap(),columns.next().unwrap().parse::<f64>().unwrap())
                }).collect::<Vec<_>>();assert_eq!(points.len(),(nx*ny) as usize);coordinates.push(points);
                let result=std::process::Command::new(&wgrib2).arg(&file).args(["-grid","-stats"]).output().unwrap();
                assert!(result.status.success(),"wgrib2 GDT{template}: {}",String::from_utf8_lossy(&result.stderr));
                std::fs::remove_file(file).unwrap();
            }
            let distance=|(a,b):(f64,f64),(c,d):(f64,f64)|{
                let lat=(c-a).to_radians();let lon=(d-b).to_radians();
                let x=(lat*0.5).sin().powi(2)+a.to_radians().cos()*c.to_radians().cos()*(lon*0.5).sin().powi(2);
                2.0*6_370_000.0*x.sqrt().asin()
            };
            let corners=[0,(nx-1) as usize,((ny-1)*nx) as usize,(nx*ny-1) as usize,((ny/2)*nx+nx/2) as usize];
            for point in corners {
                let error=distance(coordinates[0][point],coordinates[1][point]);
                assert!(error<0.05,"GDT{template} point{point}: {error}m after sphere correction");
            }
            let original_error=distance(coordinates[0][(nx*ny-1) as usize],coordinates[2][(nx*ny-1) as usize]);
            assert!(original_error>100.0,"GDT{template}: regression did not reproduce native-location error ({original_error}m)");
            std::fs::remove_dir(dir).unwrap();
        }
    }

    #[test]
    fn local_use_section_and_vertical_layer_are_explicit() {
        let bytes = Grib2Writer::new()
            .add_message(
                MessageBuilder::new(0, vec![1.0; 9])
                    .grid(square_grid(3, 3))
                    .center(255, 0)
                    .local_table_version(1)
                    .local_use(
                        br#"{"label":"infrared brightness temperature proxy","units":"K"}"#
                            .to_vec(),
                    )
                    .product(ProductDefinition {
                        level_type: 103,
                        level_value: 0.0,
                        ..Default::default()
                    })
                    .second_surface(103, 3000.0),
            )
            .to_bytes()
            .unwrap();
        assert!(std::str::from_utf8(&section(&bytes, 2)[5..])
            .unwrap()
            .contains("proxy"));
        let sec = section(&bytes, 4);
        assert_eq!(sec[28], 103);
        assert_eq!(u32::from_be_bytes(sec[30..34].try_into().unwrap()), 3000);
    }

    #[test]
    fn rejects_invalid_packing_and_incomplete_pdt() {
        for packing in [
            PackingMethod::Simple { bits_per_value: 65 },
            PackingMethod::ComplexSpatial {
                bits_per_value: 33,
                order: 2,
            },
            PackingMethod::ComplexSpatial {
                bits_per_value: 16,
                order: 3,
            },
        ] {
            assert!(Grib2Writer::new()
                .add_message(
                    MessageBuilder::new(0, vec![1.0; 9])
                        .grid(square_grid(3, 3))
                        .packing(packing)
                )
                .to_bytes()
                .is_err());
        }
        assert!(Grib2Writer::new()
            .add_message(
                MessageBuilder::new(0, vec![1.0; 9])
                    .grid(square_grid(3, 3))
                    .product(ProductDefinition {
                        template: 8,
                        ..Default::default()
                    })
            )
            .to_bytes()
            .unwrap_err()
            .contains("requires a statistical interval"));
        assert!(encode_level_value(-1.0).is_err());
        let reference = chrono::NaiveDate::from_ymd_opt(2024, 5, 25)
            .unwrap()
            .and_hms_opt(18, 0, 0)
            .unwrap();
        assert!(Grib2Writer::new()
            .add_message(
                MessageBuilder::new(0, vec![1.0; 9])
                    .grid(square_grid(3, 3))
                    .reference_time(reference)
                    .product(ProductDefinition {
                        time_range_unit: 1,
                        ..Default::default()
                    })
                    .statistical_interval(StatisticalInterval {
                        end_time: reference,
                        statistical_process: 1,
                        time_unit: 1,
                        length: 1
                    })
            )
            .to_bytes()
            .unwrap_err()
            .contains("Statistical interval end"));
    }

    /// A deterministic small external-reader fixture. Set the environment
    /// variable to retain it for ecCodes and wgrib2 validation.
    #[test]
    fn external_reader_fixture() {
        let Some(folder) = std::env::var_os("GRIB2_WRITER_GOLDEN_DIR") else {
            return;
        };
        std::fs::create_dir_all(&folder).unwrap();
        let reference = chrono::NaiveDate::from_ymd_opt(2024, 5, 25)
            .unwrap()
            .and_hms_opt(18, 0, 0)
            .unwrap();
        let mut writer = Grib2Writer::new();
        for template in [0, 10, 20, 30] {
            for order in [0, 1, 2] {
                let mut grid = square_grid(7, 5);
                grid.template = template;
                grid.lat1 = 30.0;
                grid.lat2 = if template == 0 { 34.0 } else { 30.12 };
                grid.lon2 = if template == 0 { -94.0 } else { -99.81 };
                grid.lad = 30.0;
                grid.latin1 = 30.0;
                grid.latin2 = 60.0;
                grid.lov = -95.0;
                if template != 0 {
                    grid.dx = 3000.0;
                    grid.dy = 3000.0;
                }
                let values: Vec<_> = (0..35)
                    .map(|i| {
                        if i == 6 {
                            f64::NAN
                        } else {
                            270.0 + i as f64 * 0.125
                        }
                    })
                    .collect();
                writer = writer.add_message(
                    MessageBuilder::new(0, values)
                        .center(7, 0)
                        .grid(grid)
                        .reference_time(reference)
                        .product(ProductDefinition {
                            parameter_category: 0,
                            parameter_number: 0,
                            level_type: 103,
                            level_value: 2.0,
                            ..Default::default()
                        })
                        .packing(if order == 0 {
                            PackingMethod::Simple { bits_per_value: 16 }
                        } else {
                            PackingMethod::ComplexSpatial {
                                bits_per_value: 16,
                                order,
                            }
                        }),
                );
            }
        }
        for values in [vec![1.0; 35], vec![f64::NAN; 35]] {
            writer = writer.add_message(
                MessageBuilder::new(0, values)
                    .center(7, 0)
                    .grid(square_grid(7, 5))
                    .reference_time(reference)
                    .packing(PackingMethod::ComplexSpatial {
                        bits_per_value: 16,
                        order: 2,
                    }),
            );
        }
        for process in [1, 2, 3] {
            writer = writer.add_message(
                MessageBuilder::new(0, vec![1.0; 35])
                    .center(7, 0)
                    .grid(square_grid(7, 5))
                    .reference_time(reference)
                    .product(ProductDefinition {
                        parameter_category: 1,
                        parameter_number: 8,
                        forecast_time: 5,
                        time_range_unit: 1,
                        level_type: 1,
                        ..Default::default()
                    })
                    .statistical_interval(StatisticalInterval {
                        end_time: reference + chrono::Duration::hours(6),
                        statistical_process: process,
                        time_unit: 1,
                        length: 1,
                    })
                    .packing(PackingMethod::ComplexSpatial {
                        bits_per_value: 16,
                        order: 2,
                    }),
            );
        }
        let path = std::path::PathBuf::from(folder).join("native-writer.grib2");
        std::fs::write(path, writer.to_bytes().unwrap()).unwrap();
    }

    #[test]
    fn roundtrip_simple_constant() {
        // All values are the same
        let values = vec![273.15; 9];
        let grid = GridDefinition {
            template: 0,
            nx: 3,
            ny: 3,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 2.0,
            lon2: 2.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();

        // Parse back
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(grib.messages.len(), 1);
        let msg = &grib.messages[0];
        assert_eq!(msg.grid.nx, 3);
        assert_eq!(msg.grid.ny, 3);

        let unpacked = unpack_message(msg).unwrap();
        assert_eq!(unpacked.len(), 9);
        for (i, &v) in unpacked.iter().enumerate() {
            assert!(
                (v - 273.15).abs() < 0.01,
                "Value[{}]: expected 273.15, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn roundtrip_simple_ramp() {
        // Values from 0.0 to 8.0
        let values: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let grid = GridDefinition {
            template: 0,
            nx: 3,
            ny: 3,
            lat1: 30.0,
            lon1: -100.0,
            lat2: 32.0,
            lon2: -98.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let product = ProductDefinition {
            template: 0,
            parameter_category: 0, // Temperature
            parameter_number: 0,   // Temperature
            generating_process: 2,
            forecast_time: 0,
            time_range_unit: 1, // Hour
            level_type: 103,    // Height above ground
            level_value: 2.0,   // 2 m
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .product(product)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();

        let grib = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(grib.messages.len(), 1);
        let msg = &grib.messages[0];
        assert_eq!(msg.discipline, 0);
        assert_eq!(msg.product.parameter_category, 0);
        assert_eq!(msg.product.parameter_number, 0);
        assert_eq!(msg.product.level_type, 103);

        let unpacked = unpack_message(msg).unwrap();
        assert_eq!(unpacked.len(), 9);
        for (i, &v) in unpacked.iter().enumerate() {
            let expected = i as f64;
            assert!(
                (v - expected).abs() < 0.01,
                "Value[{}]: expected {}, got {}",
                i,
                expected,
                v
            );
        }
    }

    #[test]
    fn roundtrip_temperature_kelvin() {
        // Realistic 2m temperature range in Kelvin
        let values: Vec<f64> = (0..100).map(|i| 250.0 + i as f64 * 0.5).collect();
        let grid = GridDefinition {
            template: 0,
            nx: 10,
            ny: 10,
            lat1: 30.0,
            lon1: -100.0,
            lat2: 39.0,
            lon2: -91.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        let unpacked = unpack_message(&grib.messages[0]).unwrap();
        assert_eq!(unpacked.len(), 100);

        for (i, (&orig, &unpk)) in values.iter().zip(unpacked.iter()).enumerate() {
            assert!(
                (orig - unpk).abs() < 0.01,
                "Value[{}]: expected {}, got {} (diff={})",
                i,
                orig,
                unpk,
                (orig - unpk).abs()
            );
        }
    }

    #[test]
    fn roundtrip_with_bitmap() {
        // 3x3 grid, center value is missing
        let values = vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0, 7.0, 8.0, 9.0];
        let bitmap = vec![true, true, true, true, false, true, true, true, true];

        let grid = GridDefinition {
            template: 0,
            nx: 3,
            ny: 3,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 2.0,
            lon2: 2.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .bitmap(bitmap)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        let msg = &grib.messages[0];
        assert!(msg.bitmap.is_some());

        let unpacked = unpack_message(msg).unwrap();
        // Octet padding in a bitmap is not part of the physical grid.
        assert_eq!(unpacked.len(), 9);
        assert!((unpacked[0] - 1.0).abs() < 0.01);
        assert!((unpacked[1] - 2.0).abs() < 0.01);
        assert!((unpacked[3] - 4.0).abs() < 0.01);
        assert!(unpacked[4].is_nan(), "Index 4 should be NaN");
        assert!((unpacked[5] - 6.0).abs() < 0.01);
        assert!((unpacked[8] - 9.0).abs() < 0.01);
    }

    #[test]
    fn roundtrip_nonfinite_values_auto_bitmap() {
        let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY];
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values)
                .grid(grid)
                .packing(PackingMethod::Simple { bits_per_value: 16 }),
        );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        let msg = &grib.messages[0];
        assert!(
            msg.bitmap.is_some(),
            "non-finite values should create a bitmap"
        );

        let unpacked = unpack_message(msg).unwrap();
        assert!((unpacked[0] - 1.0).abs() < 0.01);
        assert!(unpacked[1].is_nan());
        assert!((unpacked[2] - 3.0).abs() < 0.01);
        assert!(unpacked[3].is_nan());
    }

    #[test]
    fn bitmap_length_mismatch_is_error() {
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, vec![1.0, 2.0, 3.0, 4.0])
                .grid(grid)
                .bitmap(vec![true, false, true]),
        );

        let err = writer.to_bytes().unwrap_err();
        assert!(err.contains("Bitmap length 3 does not match values length 4"));
    }

    #[test]
    fn bitmap_present_nonfinite_value_is_error() {
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, vec![1.0, f64::NAN, 3.0, 4.0])
                .grid(grid)
                .bitmap(vec![true, true, true, true]),
        );

        let err = writer.to_bytes().unwrap_err();
        assert!(err.contains("Non-finite value at index 1 is marked present"));
    }

    #[test]
    fn roundtrip_lambert_grid() {
        // Lambert Conformal (HRRR-like) grid
        let values: Vec<f64> = (0..25).map(|i| 270.0 + i as f64 * 0.1).collect();
        let grid = GridDefinition {
            template: 30,
            nx: 5,
            ny: 5,
            lat1: 21.138123,
            lon1: 237.280472,
            lat2: 0.0,
            lon2: 0.0,
            dx: 3000.0,
            dy: 3000.0,
            latin1: 38.5,
            latin2: 38.5,
            lov: 262.5,
            scan_mode: 0x40,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(
            MessageBuilder::new(0, values.clone())
                .grid(grid)
                .packing(PackingMethod::Simple { bits_per_value: 24 }),
        );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(grib.messages.len(), 1);

        let msg = &grib.messages[0];
        assert_eq!(msg.grid.template, 30);
        assert_eq!(msg.grid.nx, 5);
        assert_eq!(msg.grid.ny, 5);
        assert!((msg.grid.latin1 - 38.5).abs() < 0.001);
        assert!((msg.grid.latin2 - 38.5).abs() < 0.001);
        assert!((msg.grid.lov - 262.5).abs() < 0.001);
    }

    #[test]
    fn roundtrip_multi_message() {
        let grid = GridDefinition {
            template: 0,
            nx: 4,
            ny: 4,
            lat1: 30.0,
            lon1: -100.0,
            lat2: 33.0,
            lon2: -97.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let temp_values: Vec<f64> = (0..16).map(|i| 273.0 + i as f64).collect();
        let wind_values: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect();

        let writer = Grib2Writer::new()
            .add_message(
                MessageBuilder::new(0, temp_values.clone())
                    .grid(grid.clone())
                    .product(ProductDefinition {
                        parameter_category: 0,
                        parameter_number: 0,
                        level_type: 103,
                        level_value: 2.0,
                        ..ProductDefinition::default()
                    })
                    .packing(PackingMethod::Simple { bits_per_value: 16 }),
            )
            .add_message(
                MessageBuilder::new(0, wind_values.clone())
                    .grid(grid.clone())
                    .product(ProductDefinition {
                        parameter_category: 2,
                        parameter_number: 2,
                        level_type: 103,
                        level_value: 10.0,
                        ..ProductDefinition::default()
                    })
                    .packing(PackingMethod::Simple { bits_per_value: 16 }),
            );

        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        assert_eq!(grib.messages.len(), 2);

        // Check first message (temperature)
        let msg0 = &grib.messages[0];
        assert_eq!(msg0.product.parameter_category, 0);
        assert_eq!(msg0.product.parameter_number, 0);
        let vals0 = unpack_message(msg0).unwrap();
        assert_eq!(vals0.len(), 16);
        assert!((vals0[0] - 273.0).abs() < 0.01);

        // Check second message (wind)
        let msg1 = &grib.messages[1];
        assert_eq!(msg1.product.parameter_category, 2);
        assert_eq!(msg1.product.parameter_number, 2);
        let vals1 = unpack_message(msg1).unwrap();
        assert_eq!(vals1.len(), 16);
        assert!((vals1[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn roundtrip_write_file() {
        let values: Vec<f64> = (0..4).map(|i| i as f64 * 10.0).collect();
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer =
            Grib2Writer::new().add_message(MessageBuilder::new(0, values.clone()).grid(grid));

        let tmp = std::env::temp_dir().join("metrust_test_writer.grib2");
        let path = tmp.to_str().unwrap();
        writer.write_file(path).unwrap();

        // Read back
        let grib = Grib2File::open(path).unwrap();
        assert_eq!(grib.messages.len(), 1);
        let unpacked = unpack_message(&grib.messages[0]).unwrap();
        assert_eq!(unpacked.len(), 4);
        for (i, &v) in unpacked.iter().enumerate() {
            let expected = i as f64 * 10.0;
            assert!(
                (v - expected).abs() < 0.1,
                "Value[{}]: expected {}, got {}",
                i,
                expected,
                v
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ieee32_round_trips_every_bit() {
        let raw: Vec<f32> = vec![0.0, -1.5, 1.0e-38, f32::from_bits(1), 3.402_823_5e38, 287.153_2, 9.969_21e36, -273.15];
        let grid = GridDefinition { template: 0, ..square_grid(4, 2) };
        let msg = MessageBuilder::new(0, raw.iter().map(|&v| v as f64).collect())
            .grid(grid)
            .product(ProductDefinition { template: 0, parameter_category: 0, parameter_number: 0,
                generating_process: 2, forecast_time: 1, time_range_unit: 1, level_type: 1, level_value: 0.0 })
            .packing(PackingMethod::Ieee32);
        let bytes = Grib2Writer::new().add_message(msg).to_bytes().unwrap();
        let sec5 = section(&bytes, 5);
        assert_eq!(u16::from_be_bytes([sec5[9], sec5[10]]), 4);
        assert_eq!(sec5[11], 1);
        let file = Grib2File::from_bytes(&bytes).unwrap();
        let back = unpack_message(&file.messages[0]).unwrap();
        for (a, b) in raw.iter().zip(back.iter()) {
            assert_eq!(a.to_bits(), (*b as f32).to_bits());
        }
    }

    #[test]
    fn write_bits_basic() {
        let mut buf = vec![0u8; 3];

        // Write 0xFF (8 bits) at offset 0
        write_bits(&mut buf, 0, 8, 0xFF);
        assert_eq!(buf[0], 0xFF);

        // Write 0x5 (4 bits) at offset 8
        write_bits(&mut buf, 8, 4, 0x5);
        assert_eq!(buf[1] & 0xF0, 0x50);

        // Write 0xA (4 bits) at offset 12
        write_bits(&mut buf, 12, 4, 0xA);
        assert_eq!(buf[1], 0x5A);
    }

    #[test]
    fn encode_signed_u32_positive() {
        let bytes = encode_signed_u32(45.5, 1_000_000.0);
        let raw = u32::from_be_bytes(bytes);
        assert_eq!(raw & 0x80000000, 0); // positive
        assert_eq!(raw, 45_500_000);
    }

    #[test]
    fn encode_signed_u32_negative() {
        let bytes = encode_signed_u32(-45.5, 1_000_000.0);
        let raw = u32::from_be_bytes(bytes);
        assert_ne!(raw & 0x80000000, 0); // negative (sign bit set)
        assert_eq!(raw & 0x7FFFFFFF, 45_500_000);
    }

    #[test]
    fn grib2_magic_and_end_marker() {
        let values = vec![1.0; 4];
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new().add_message(MessageBuilder::new(0, values).grid(grid));
        let bytes = writer.to_bytes().unwrap();

        // Check magic
        assert_eq!(&bytes[0..4], b"GRIB");
        // Check edition
        assert_eq!(bytes[7], 2);
        // Check end marker
        assert_eq!(&bytes[bytes.len() - 4..], b"7777");

        // Check total length matches
        let total_len = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(total_len as usize, bytes.len());
    }

    #[test]
    fn reference_time_roundtrip() {
        let dt = chrono::NaiveDate::from_ymd_opt(2025, 6, 15)
            .unwrap()
            .and_hms_opt(12, 30, 45)
            .unwrap();

        let values = vec![1.0; 4];
        let grid = GridDefinition {
            template: 0,
            nx: 2,
            ny: 2,
            lat1: 0.0,
            lon1: 0.0,
            lat2: 1.0,
            lon2: 1.0,
            dx: 1.0,
            dy: 1.0,
            scan_mode: 0,
            ..GridDefinition::default()
        };

        let writer = Grib2Writer::new()
            .add_message(MessageBuilder::new(0, values).grid(grid).reference_time(dt));
        let bytes = writer.to_bytes().unwrap();
        let grib = Grib2File::from_bytes(&bytes).unwrap();
        let msg = &grib.messages[0];
        assert_eq!(msg.reference_time, dt);
    }
}
