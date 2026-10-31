use super::parser::{DataRepresentation, Grib2Message};

/// Bit reader for extracting packed values from GRIB2 data sections.
pub struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, bit_pos: 0 }
    }

    /// Read `n` bits as an unsigned integer (up to 64 bits).
    ///
    /// Optimized to read whole bytes at a time when possible, falling back
    /// to bit-by-bit only for partial-byte boundaries.
    pub fn read_bits(&mut self, n: usize) -> u64 {
        if n == 0 {
            return 0;
        }

        let byte_offset = self.bit_pos / 8;
        let bit_offset = self.bit_pos % 8;

        // Fast path: all bits fit within available bytes and n <= 56
        // We can grab up to 8 bytes from the data, shift and mask.
        if n <= 56 && byte_offset + (bit_offset + n + 7) / 8 <= self.data.len() {
            // Read up to 8 bytes starting at byte_offset into a u64 (big-endian)
            let bytes_needed = (bit_offset + n + 7) / 8;
            let mut raw: u64 = 0;
            for i in 0..bytes_needed {
                raw = (raw << 8) | self.data[byte_offset + i] as u64;
            }
            // The bits we want start at bit_offset from the MSB of the first byte
            // and we read `bytes_needed * 8` bits total. We need bits
            // [bit_offset .. bit_offset + n] counting from the MSB.
            let shift = bytes_needed * 8 - bit_offset - n;
            let mask = (1u64 << n) - 1;
            self.bit_pos += n;
            return (raw >> shift) & mask;
        }

        // Slow path: bit-by-bit (for edge cases or very large reads)
        let mut result: u64 = 0;
        for _ in 0..n {
            let bi = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            if bi < self.data.len() {
                result = (result << 1) | ((self.data[bi] >> bit_idx) as u64 & 1);
            } else {
                result <<= 1;
            }
            self.bit_pos += 1;
        }
        result
    }

    /// Strict bit read used by scientific decoders: never synthesize zero
    /// bits beyond the declared Section-7 payload.
    pub fn read_bits_checked(&mut self, n: usize) -> Result<u64, String> {
        if n > 64 {
            return Err(format!("cannot read {n} bits into u64"));
        }
        if self.remaining_bits() < n {
            return Err(format!(
                "truncated packed payload: requested {n} bits with {} remaining",
                self.remaining_bits()
            ));
        }
        Ok(self.read_bits(n))
    }

    /// Read `n` bits as a signed integer using sign-magnitude convention.
    /// MSB = sign (1 = negative), remaining bits = magnitude.
    pub fn read_signed_bits(&mut self, n: usize) -> i64 {
        if n == 0 {
            return 0;
        }
        if n == 1 {
            let _bit = self.read_bits(1);
            return 0;
        }
        let sign = self.read_bits(1);
        let magnitude = self.read_bits(n - 1) as i64;
        if sign == 1 {
            -magnitude
        } else {
            magnitude
        }
    }

    /// Align the bit position to the next byte boundary.
    pub fn align_to_byte(&mut self) {
        let rem = self.bit_pos % 8;
        if rem != 0 {
            self.bit_pos += 8 - rem;
        }
    }

    /// Number of bits remaining.
    pub fn remaining_bits(&self) -> usize {
        let total = self.data.len() * 8;
        if self.bit_pos >= total {
            0
        } else {
            total - self.bit_pos
        }
    }
}

// Section 5 counts coded values, so a complete all-missing bitmap declares
// zero. Complex packing has zero groups; spatial packing may retain only its
// canonical zero initial/minimum descriptors. Never turn data bytes or a
// malformed bitmap/group table into a missing field.
fn all_missing_bitmap_field(msg: &Grib2Message, num_points: usize) -> bool {
    let dr = &msg.data_rep;
    if num_points == 0 || num_points > 100_000_000 || dr.section5_num_data_points != 0 {
        return false;
    }
    let Some(bitmap) = msg.bitmap.as_ref() else { return false; };
    // Some parsers retain the final octet's unused bits. Both representations
    // must cover exactly this grid, with canonical zero padding.
    let padded_points = (num_points + 7) / 8 * 8;
    if (bitmap.len() != num_points && bitmap.len() != padded_points)
        || bitmap.iter().any(|present| *present) {
        return false;
    }
    if dr.template == 0 {
        return msg.raw_data.is_empty();
    }
    if !matches!(dr.template, 2 | 3) || dr.num_groups != 0
        || dr.bits_per_value != 0 || dr.group_width_ref != 0 || dr.group_width_bits != 0
        || dr.last_group_length != 0 || dr.group_length_bits != 0
        || dr.group_splitting_method != 1 || dr.missing_value_management != 0
        || !matches!(dr.original_field_type, 0 | 1) || !dr.reference_value.is_finite() {
        return false;
    }
    if dr.template == 2 {
        return dr.spatial_diff_order == 0 && dr.spatial_diff_bytes == 0 && msg.raw_data.is_empty();
    }
    if !matches!(dr.spatial_diff_order, 1 | 2) || dr.spatial_diff_bytes > 8 {
        return false;
    }
    let descriptor_bytes = (dr.spatial_diff_order as usize + 1) * dr.spatial_diff_bytes as usize;
    msg.raw_data.len() == descriptor_bytes && msg.raw_data.iter().all(|byte| *byte == 0)
}

/// Unpack a GRIB2 message's data section to floating-point values.
pub fn unpack_message(msg: &Grib2Message) -> crate::Result<Vec<f64>> {
    unpack_message_with_representation(msg, &msg.data_rep)
}

/// Decode integer-packed fields using WPS/g2lib's default REAL arithmetic.
///
/// g2lib simunpack/comunpack evaluate each operation in binary32:
/// `(REAL(X) * (2.0 ** REAL(E)) + R) * (10.0 ** REAL(-D))`.
/// Rounding a fully scaled binary64 value only at output is not equivalent.
/// Recover the exact packed integers through the same validated unpacker,
/// with neutral scale factors, then apply the native arithmetic explicitly.
/// Bitmap holes remain NaN for callers to apply their own missing convention.
/// The general decoder remains binary64 for existing non-WPS consumers.
pub fn unpack_message_wps32(msg: &Grib2Message) -> crate::Result<Vec<f64>> {
    let dr = &msg.data_rep;
    if !matches!(dr.template, 0 | 2 | 3 | 40 | 41 | 42) {
        return unpack_message(msg);
    }
    let mut integers = dr.clone();
    integers.reference_value = 0.0;
    integers.binary_scale = 0;
    integers.decimal_scale = 0;
    let mut values = unpack_message_with_representation(msg, &integers)?;
    let constant = match dr.template {
        2 | 3 => dr.num_groups == 0,
        _ => dr.bits_per_value == 0,
    };
    let binary = 2.0f32.powf(dr.binary_scale as f32);
    let decimal = 10.0f32.powf(-(dr.decimal_scale as f32));
    for value in &mut values {
        if value.is_nan() {
            continue;
        }
        // g2lib's constant-field branch returns R without decimal scaling.
        *value = if constant {
            dr.reference_value as f64
        } else {
            let scaled = (*value as f32) * binary;
            let shifted = scaled + dr.reference_value;
            (shifted * decimal) as f64
        };
    }
    Ok(values)
}

fn unpack_message_with_representation(
    msg: &Grib2Message,
    dr: &DataRepresentation,
) -> crate::Result<Vec<f64>> {

    // Use num_data_points when available and sane to avoid overflow on reduced
    // Gaussian grids where nx is a sentinel value (0xFFFFFFFE or 0xFFFFFFFF).
    let num_points = if msg.grid.num_data_points > 0 && msg.grid.num_data_points < 100_000_000 {
        msg.grid.num_data_points as usize
    } else if msg.grid.is_reduced {
        // For reduced Gaussian grids, the total number of points is the sum
        // of the pl array (number of points per latitude row).
        msg.grid
            .pl
            .as_ref()
            .map(|pl| pl.iter().sum::<u32>() as usize)
            .unwrap_or(0)
    } else {
        msg.grid.nx as usize * msg.grid.ny as usize
    };

    // Sanity check: refuse to allocate more than 100 million points
    if num_points > 100_000_000 {
        return Err(crate::GribError::Unpack(format!(
            "grid has {} points which exceeds the 100M sanity limit (nx={}, ny={}, is_reduced={})",
            num_points, msg.grid.nx, msg.grid.ny, msg.grid.is_reduced
        )));
    }
    if num_points == 0 {
        return Err(crate::GribError::Unpack(
            "grid has 0 data points".to_string(),
        ));
    }
    let declared_values = dr.section5_num_data_points as usize;
    if declared_values == 0 {
        if all_missing_bitmap_field(msg, num_points) {
            return Ok(vec![f64::NAN; num_points]);
        }
        return Err(crate::GribError::Unpack(
            "Section 5 declares 0 data points without an exact all-missing bitmap and canonical empty packing".to_string(),
        ));
    }
    if dr.template == 0 {
        let bpv = dr.bits_per_value as usize;
        let expected_bytes = if bpv == 0 {
            0
        } else {
            declared_values
                .checked_mul(bpv)
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8)
                .ok_or_else(|| {
                    crate::GribError::Unpack("simple-packed bit count overflows usize".to_string())
                })?
        };
        if msg.raw_data.len() != expected_bytes {
            return Err(crate::GribError::Unpack(format!(
                "simple-packed Section 7 has {} bytes; Section 5 requires {expected_bytes}",
                msg.raw_data.len()
            )));
        }
    }
    // Convergence conflict C2 (see lib.rs): IEEE floating-point packing
    // (Template 5.4) declares its point count in Section 5, and that count is
    // the authority.  Section 7 is byte-padded by some encoders, so bytes
    // BEYOND `count * width` are tolerated and dropped below; bytes SHORT of
    // it are refused, because a short IEEE payload silently shortens the field
    // and the missing tail reaches artifacts as good data.
    if dr.template == 4 {
        let width = match dr.bits_per_value {
            32 => 4usize,
            64 => 8usize,
            // Any other width is refused by `unpack_ieee`, which names it.
            _ => 0,
        };
        if width != 0 {
            let required = declared_values.checked_mul(width).ok_or_else(|| {
                crate::GribError::Unpack("IEEE byte count overflows usize".to_string())
            })?;
            if msg.raw_data.len() < required {
                return Err(crate::GribError::Unpack(format!(
                    "IEEE-packed Section 7 has {} bytes; Section 5 declares \
                     {declared_values} values needing {required}",
                    msg.raw_data.len()
                )));
            }
        }
    }
    let mut values = match dr.template {
        0 => unpack_simple(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        2 => unpack_complex(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        3 => unpack_complex_spatial(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        4 => unpack_ieee(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        40 => unpack_jpeg2000(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        41 => unpack_png(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        42 => unpack_ccsds(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        50 => unpack_spectral_simple(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        51 => unpack_spectral_complex(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        61 => unpack_simple_log(&msg.raw_data, dr).map_err(crate::GribError::Unpack)?,
        200 => unpack_rle(&msg.raw_data, dr, num_points).map_err(crate::GribError::Unpack)?,
        _ => {
            return Err(crate::GribError::UnsupportedTemplate {
                template: dr.template,
                detail: "data representation template".to_string(),
            })
        }
    };

    // A complex-packed field with zero groups is how GRIB2 spells a constant
    // field: Section 7 carries no coded values even though Section 5 still
    // declares the logical point count.  Expand it before the count check
    // below, which would otherwise read the constant as a truncated payload.
    if matches!(dr.template, 2 | 3) && dr.num_groups == 0 && values.is_empty() {
        values = vec![dr.reference_value as f64; declared_values];
    }

    if dr.template == 0 && dr.bits_per_value != 0 {
        if values.len() < declared_values {
            return Err(crate::GribError::Unpack(format!(
                "simple-packed payload decoded {} values; Section 5 declares {declared_values}",
                values.len()
            )));
        }
        values.truncate(declared_values);
    } else if dr.template == 4 {
        // Trailing IEEE padding tolerated (see the C2 note above); a short
        // payload was already refused before decode.
        values.truncate(declared_values);
    } else if matches!(dr.template, 2 | 3) && values.len() != declared_values {
        return Err(crate::GribError::Unpack(format!(
            "complex-packed payload decoded {} values; Section 5 declares {declared_values}",
            values.len()
        )));
    }

    // Expand constant fields (bpv=0) to full grid size.
    // When bits_per_value is 0, all grid points share the same reference value,
    // but unpack_simple only returns a single element.
    let values = if dr.bits_per_value == 0 && values.len() <= 1 {
        let fill = values.first().copied().unwrap_or(dr.reference_value as f64);
        vec![fill; declared_values]
    } else {
        values
    };

    // Apply bitmap if present
    let values = if let Some(ref bitmap) = msg.bitmap {
        let n = bitmap.len();
        let mut result = vec![f64::NAN; n];
        let mut val_idx = 0;
        for i in 0..n {
            if bitmap[i] {
                if val_idx >= values.len() {
                    return Err(crate::GribError::Unpack(
                        "bitmap requests more values than Section 7 decoded".to_string(),
                    ));
                }
                result[i] = values[val_idx];
                val_idx += 1;
            }
        }
        if val_idx != values.len() {
            return Err(crate::GribError::Unpack(format!(
                "bitmap consumed {val_idx} values but Section 7 decoded {}",
                values.len()
            )));
        }
        result
    } else {
        if values.len() != num_points {
            return Err(crate::GribError::Unpack(format!(
                "payload decoded {} values without bitmap; grid declares {num_points}",
                values.len()
            )));
        }
        values
    };

    // Some IEEE-packed messages carry pad bytes in section 7. Trim back to the
    // declared point count so downstream grid/value length checks stay aligned.
    let values = if values.len() > num_points {
        values.into_iter().take(num_points).collect()
    } else {
        values
    };

    // Preserve original scan order by default (matches ecCodes behavior).
    // Users who need north-to-south (top-down) row order for rendering
    // should call `flip_scan_order()` on the values.

    Ok(values)
}

/// Flip rows of a 2D field to convert between scan orders.
///
/// GRIB2 scan_mode bit 6 (0x40) indicates +j direction (south-to-north).
/// This function reverses the row order, converting south-to-north to
/// north-to-south (top-down) or vice versa.
///
/// Call this after `unpack_message` if you need north-to-south order
/// for rendering/display and the message has `scan_mode & 0x40 != 0`.
pub fn flip_rows(values: &mut [f64], nx: usize, ny: usize) {
    if nx == 0 || ny == 0 || values.len() != nx * ny {
        return;
    }
    for j in 0..ny / 2 {
        let j_rev = ny - 1 - j;
        let (top, bot) = values.split_at_mut(j_rev * nx);
        let top_row = &mut top[j * nx..j * nx + nx];
        let bot_row = &mut bot[..nx];
        top_row.swap_with_slice(bot_row);
    }
}

/// Unpack a message and normalize to north-to-south row order.
///
/// This is a convenience that calls `unpack_message` then `flip_rows`
/// if scan_mode bit 6 is set. Useful for rendering pipelines.
pub fn unpack_message_normalized(msg: &Grib2Message) -> crate::Result<Vec<f64>> {
    let mut values = unpack_message(msg)?;
    if msg.grid.scan_mode & 0x40 != 0 {
        flip_rows(&mut values, msg.grid.nx as usize, msg.grid.ny as usize);
    }
    Ok(values)
}

/// Unpack a north-to-south row window without materializing the whole grid.
///
/// The returned vector contains `(y_end - y_start) * nx` values in row-major
/// order, after the same scan-mode normalization as `unpack_message_normalized`.
/// Missing bitmap cells are emitted as `NaN`.
pub fn unpack_message_scan_normalized_row_window(
    msg: &Grib2Message,
    y_start: usize,
    y_end: usize,
) -> crate::Result<Vec<f64>> {
    let nx = msg.grid.nx as usize;
    let ny = msg.grid.ny as usize;
    if nx == 0 || ny == 0 {
        return Err(crate::GribError::Unpack(
            "grid has 0 data points".to_string(),
        ));
    }
    if msg.grid.is_reduced {
        return Err(crate::GribError::Unpack(
            "row-window unpack does not support reduced grids".to_string(),
        ));
    }
    if y_start > y_end || y_end > ny {
        return Err(crate::GribError::Unpack(format!(
            "invalid row window {y_start}..{y_end} for ny={ny}"
        )));
    }

    // The window path decodes the same Section 7 `unpack_message` does, so it
    // owes the same Section 5/7 preconditions.  Without them a short or
    // over-declared payload renders as reference-value fill instead of
    // refusing, and the renderer reports the tile as decoded.
    let dr = &msg.data_rep;
    let declared_values = dr.section5_num_data_points as usize;
    if declared_values == 0 {
        if all_missing_bitmap_field(msg, nx.checked_mul(ny).unwrap_or(usize::MAX)) {
            return Ok(vec![f64::NAN; nx * (y_end - y_start)]);
        }
        return Err(crate::GribError::Unpack(
            "Section 5 declares 0 data points without an exact all-missing bitmap and canonical empty packing".to_string(),
        ));
    }
    if dr.template == 0 {
        let bpv = dr.bits_per_value as usize;
        let expected_bytes = if bpv == 0 {
            0
        } else {
            declared_values
                .checked_mul(bpv)
                .and_then(|bits| bits.checked_add(7))
                .map(|bits| bits / 8)
                .ok_or_else(|| {
                    crate::GribError::Unpack("simple-packed bit count overflows usize".to_string())
                })?
        };
        if msg.raw_data.len() != expected_bytes {
            return Err(crate::GribError::Unpack(format!(
                "simple-packed Section 7 has {} bytes; Section 5 requires {expected_bytes}",
                msg.raw_data.len()
            )));
        }
    }

    match msg.data_rep.template {
        0 => unpack_simple_scan_normalized_row_window(msg, nx, ny, y_start, y_end)
            .map_err(crate::GribError::Unpack),
        3 => unpack_complex_spatial_scan_normalized_row_window(msg, nx, ny, y_start, y_end)
            .map_err(crate::GribError::Unpack),
        template => Err(crate::GribError::UnsupportedTemplate {
            template,
            detail: "row-window data representation".to_string(),
        }),
    }
}

/// A rectangle of a regular grid in STORED order: rows `rows.0..rows.1`
/// and columns `columns.0..columns.1` of Section 7's row-major layout,
/// first stored row first, before any scan-mode normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredRect {
    pub rows: (usize, usize),
    pub columns: (usize, usize),
}

/// Whether a raw J2K codestream's main header declares the reversible
/// 5/3 wavelet for its one component: a COD whose transform octet is 1
/// and no COC that could restate it.  Only then does an area decode
/// reconstruct integers that are the whole decode's integers exactly;
/// the irreversible 9/7 path computes in floating point over a
/// different span, so it is never trusted to agree bit for bit.
fn jpeg2000_main_header_is_reversible(data: &[u8]) -> bool {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0x4F {
        return false;
    }
    let mut position = 2usize;
    let mut reversible = false;
    while position + 4 <= data.len() {
        if data[position] != 0xFF {
            return false;
        }
        let marker = data[position + 1];
        if marker == 0x90 {
            // SOT: the main header is over.
            return reversible;
        }
        let length = u16::from_be_bytes([data[position + 2], data[position + 3]]) as usize;
        if length < 2 || position + 2 + length > data.len() {
            return false;
        }
        match marker {
            // COD: Lcod(2) Scod(1) SGcod(4) then SPcod levels, xcb, ycb,
            // style and the transform octet.
            0x52 => {
                if length < 12 {
                    return false;
                }
                reversible = data[position + 13] == 1;
            }
            // COC restates coding per component; not interpreted here.
            0x53 => return false,
            _ => {}
        }
        position += 2 + length;
    }
    false
}

/// Unpack only a stored-order rectangle of a regular grid.
///
/// The result is `(rows.1 - rows.0) * (columns.1 - columns.0)` values,
/// row-major in stored order, and every value equals the value
/// `unpack_message` returns at that cell.  A JPEG2000 message (Template
/// 5.40) with no bitmap, a non-constant payload and the reversible 5/3
/// wavelet decodes through OpenJPEG's decode area, so its cost follows
/// the rectangle instead of the grid; every other message is unpacked
/// whole and cropped.
pub fn unpack_message_stored_rect(
    msg: &Grib2Message,
    rect: StoredRect,
) -> crate::Result<Vec<f64>> {
    let nx = msg.grid.nx as usize;
    let ny = msg.grid.ny as usize;
    if msg.grid.is_reduced || nx == 0 || ny == 0 {
        return Err(crate::GribError::Unpack(
            "a stored rectangle needs a regular grid".to_string(),
        ));
    }
    if rect.rows.0 >= rect.rows.1
        || rect.rows.1 > ny
        || rect.columns.0 >= rect.columns.1
        || rect.columns.1 > nx
    {
        return Err(crate::GribError::Unpack(format!(
            "rows {}..{} and columns {}..{} are not a rectangle of the {nx}x{ny} grid",
            rect.rows.0, rect.rows.1, rect.columns.0, rect.columns.1
        )));
    }
    let dr = &msg.data_rep;
    let area = dr.template == 40
        && msg.bitmap.is_none()
        && dr.bits_per_value != 0
        && dr.section5_num_data_points as usize == nx * ny
        && jpeg2000_main_header_is_reversible(&msg.raw_data);
    if area {
        let values = unpack_jpeg2000_area(&msg.raw_data, dr, (nx, ny), rect)
            .map_err(crate::GribError::Unpack)?;
        let expected = (rect.rows.1 - rect.rows.0) * (rect.columns.1 - rect.columns.0);
        if values.len() != expected {
            return Err(crate::GribError::Unpack(format!(
                "JPEG2000 decode area returned {} values; the rectangle holds {expected}",
                values.len()
            )));
        }
        return Ok(values);
    }
    let full = unpack_message(msg)?;
    if full.len() != nx * ny {
        return Err(crate::GribError::Unpack(format!(
            "decoded {} values; the grid declares {nx}x{ny}",
            full.len()
        )));
    }
    let mut values = Vec::with_capacity((rect.rows.1 - rect.rows.0) * (rect.columns.1 - rect.columns.0));
    for row in rect.rows.0..rect.rows.1 {
        values.extend_from_slice(&full[row * nx + rect.columns.0..row * nx + rect.columns.1]);
    }
    Ok(values)
}

/// Apply the GRIB2 scaling formula: Y = (R + X * 2^E) * 10^(-D)
fn apply_scaling(raw: &[i64], dr: &DataRepresentation) -> Vec<f64> {
    let r = dr.reference_value as f64;
    let e = dr.binary_scale as f64;
    let d = dr.decimal_scale as f64;
    let two_e = 2.0_f64.powf(e);
    let ten_neg_d = 10.0_f64.powf(-d);

    raw.iter()
        .map(|&x| (r + x as f64 * two_e) * ten_neg_d)
        .collect()
}

fn scale_raw_value(raw: i64, dr: &DataRepresentation) -> f64 {
    let r = dr.reference_value as f64;
    let two_e = 2.0_f64.powi(dr.binary_scale as i32);
    let ten_neg_d = 10.0_f64.powi(-(dr.decimal_scale as i32));
    (r + raw as f64 * two_e) * ten_neg_d
}

fn unpack_simple_scan_normalized_row_window(
    msg: &Grib2Message,
    nx: usize,
    ny: usize,
    y_start: usize,
    y_end: usize,
) -> Result<Vec<f64>, String> {
    let dr = &msg.data_rep;
    if dr.bits_per_value == 0 {
        return fill_scan_normalized_row_window_from_dense_values(
            msg,
            nx,
            ny,
            y_start,
            y_end,
            || Ok(Some(dr.reference_value as f64)),
        );
    }

    let bpv = dr.bits_per_value as usize;
    if bpv > 63 {
        return Err(format!(
            "simple packing bits_per_value {bpv} exceeds the 63-bit decode width"
        ));
    }
    let mut reader = BitReader::new(&msg.raw_data);
    fill_scan_normalized_row_window_from_dense_values(msg, nx, ny, y_start, y_end, || {
        Ok(Some(scale_raw_value(
            i64::try_from(reader.read_bits_checked(bpv)?)
                .map_err(|_| "simple-packed value exceeds i64".to_string())?,
            dr,
        )))
    })
}

fn unpack_complex_spatial_scan_normalized_row_window(
    msg: &Grib2Message,
    nx: usize,
    ny: usize,
    y_start: usize,
    y_end: usize,
) -> Result<Vec<f64>, String> {
    let dr = &msg.data_rep;
    let mode = missing_value_mode(dr)?;
    let order = dr.spatial_diff_order as usize;
    let extra_bytes = dr.spatial_diff_bytes as usize;

    // Zero groups is the GRIB2 spelling of a constant field: there are no
    // coded values to consume even though Section 5 still declares the
    // logical point count.  Handled here so the window agrees with the full
    // decode's constant expansion instead of coming back all-NaN.
    if dr.num_groups == 0 {
        return fill_scan_normalized_row_window_from_dense_values(
            msg,
            nx,
            ny,
            y_start,
            y_end,
            || Ok(Some(dr.reference_value as f64)),
        );
    }

    if order == 0 || extra_bytes == 0 {
        let mut groups = ComplexGroups::new(&msg.raw_data, dr, mode)?;
        return fill_scan_normalized_row_window_from_dense_values(
            msg,
            nx,
            ny,
            y_start,
            y_end,
            || {
                Ok(groups.next_value()?.map(|value| match value {
                    GroupValue::Present(raw) => scale_raw_value(raw, dr),
                    GroupValue::Missing => f64::NAN,
                }))
            },
        );
    }

    let nbits = extra_bytes * 8;
    let mut reader = BitReader::new(&msg.raw_data);

    let mut initial_values = Vec::with_capacity(order);
    for _ in 0..order {
        initial_values.push(
            i64::try_from(reader.read_bits_checked(nbits)?)
                .map_err(|_| "spatial-differencing initial value exceeded i64".to_string())?,
        );
    }

    let sign = reader.read_bits_checked(1)?;
    let magnitude = reader.read_bits_checked(nbits - 1)? as i64;
    let minimum = if sign == 1 { -magnitude } else { magnitude };

    reader.align_to_byte();
    let consumed_bytes = (reader.bit_pos + 7) / 8;
    let remaining_data = msg
        .raw_data
        .get(consumed_bytes..)
        .ok_or_else(|| "spatial differencing header exceeded data length".to_string())?;
    let mut groups = ComplexGroups::new(remaining_data, dr, mode)?;

    // `present_idx` counts PRESENT cells, not grid cells: a cell marked
    // missing in band carries no residual and must not advance the
    // difference chain (see `unpack_complex_spatial`).
    let mut present_idx = 0usize;
    let mut previous = None::<i64>;
    let mut previous_previous = None::<i64>;
    fill_scan_normalized_row_window_from_dense_values(msg, nx, ny, y_start, y_end, || {
        let raw = match groups.next_value()? {
            None => return Ok(None),
            Some(GroupValue::Missing) => return Ok(Some(f64::NAN)),
            Some(GroupValue::Present(raw)) => raw,
        };
        // The same checked recurrence the full decode runs; see
        // `unpack_complex_spatial` for why every step is checked.
        let reconstructed = if present_idx < order {
            initial_values[present_idx]
        } else {
            let difference = raw.checked_add(minimum).ok_or_else(|| {
                "spatial-differencing difference overflow while adding minimum".to_string()
            })?;
            match order {
                1 => difference
                    .checked_add(previous.unwrap_or(0))
                    .ok_or_else(|| "first-order spatial recurrence overflow".to_string())?,
                2 => previous
                    .unwrap_or(0)
                    .checked_mul(2)
                    .and_then(|value| value.checked_sub(previous_previous.unwrap_or(0)))
                    .and_then(|value| value.checked_add(difference))
                    .ok_or_else(|| "second-order spatial recurrence overflow".to_string())?,
                _ => difference,
            }
        };

        present_idx += 1;
        previous_previous = previous;
        previous = Some(reconstructed);
        Ok(Some(scale_raw_value(reconstructed, dr)))
    })
}

struct ComplexGroups<'a> {
    reader: BitReader<'a>,
    group_refs: Vec<i64>,
    group_widths: Vec<usize>,
    group_lengths: Vec<usize>,
    group_idx: usize,
    value_idx_in_group: usize,
    mode: MissingValueMode,
    constant_markers: MissingMarkers,
}

impl<'a> ComplexGroups<'a> {
    fn new(
        data: &'a [u8],
        dr: &DataRepresentation,
        mode: MissingValueMode,
    ) -> Result<Self, String> {
        let constant_markers = MissingMarkers::for_width(dr.bits_per_value as usize, mode);
        let ng = dr.num_groups as usize;
        if ng == 0 {
            return Ok(Self {
                reader: BitReader::new(data),
                group_refs: Vec::new(),
                group_widths: Vec::new(),
                group_lengths: Vec::new(),
                group_idx: 0,
                value_idx_in_group: 0,
                mode,
                constant_markers,
            });
        }

        let mut reader = BitReader::new(data);
        let bpv = dr.bits_per_value as usize;
        let mut group_refs = Vec::with_capacity(ng);
        for _ in 0..ng {
            group_refs.push(reader.read_bits_checked(bpv)? as i64);
        }
        reader.align_to_byte();

        let gwb = dr.group_width_bits as usize;
        let mut group_widths = Vec::with_capacity(ng);
        for _ in 0..ng {
            group_widths.push(reader.read_bits_checked(gwb)? as usize + dr.group_width_ref as usize);
        }
        reader.align_to_byte();

        let glb = dr.group_length_bits as usize;
        let mut group_lengths = Vec::with_capacity(ng);
        for _ in 0..ng {
            let stored = reader.read_bits_checked(glb)? as usize;
            group_lengths
                .push(stored * dr.group_length_inc as usize + dr.group_length_ref as usize);
        }
        if ng > 0 {
            group_lengths[ng - 1] = dr.last_group_length as usize;
        }
        reader.align_to_byte();
        refuse_ambiguous_constant_group(bpv, mode, &group_widths)?;
        // The same group-table/Section-5 agreement `unpack_complex` requires.
        let total_values: usize = group_lengths.iter().sum();
        if dr.section5_num_data_points != 0 && total_values != dr.section5_num_data_points as usize {
            return Err(format!(
                "complex group lengths declare {total_values} values, Section 5 declares {}",
                dr.section5_num_data_points
            ));
        }

        Ok(Self {
            reader,
            group_refs,
            group_widths,
            group_lengths,
            group_idx: 0,
            value_idx_in_group: 0,
            mode,
            constant_markers,
        })
    }

    /// `Ok(None)` means the group table is exhausted.  A bit stream that ends
    /// before the group table does is an `Err`: the unchecked reader
    /// synthesizes zero bits past the payload, which decodes a truncated
    /// Section 7 as reference-value fill.
    fn next_value(&mut self) -> Result<Option<GroupValue>, String> {
        while self.group_idx < self.group_lengths.len()
            && self.value_idx_in_group >= self.group_lengths[self.group_idx]
        {
            self.group_idx += 1;
            self.value_idx_in_group = 0;
        }
        if self.group_idx >= self.group_lengths.len() {
            return Ok(None);
        }

        let width = self.group_widths[self.group_idx];
        let gref = self.group_refs[self.group_idx];
        self.value_idx_in_group += 1;
        if width == 0 {
            return Ok(Some(if self.constant_markers.marks(gref as u64) {
                GroupValue::Missing
            } else {
                GroupValue::Present(gref)
            }));
        }
        let stored = self.reader.read_bits_checked(width)?;
        Ok(Some(
            if MissingMarkers::for_width(width, self.mode).marks(stored) {
                GroupValue::Missing
            } else {
                GroupValue::Present(gref + stored as i64)
            },
        ))
    }
}

fn fill_scan_normalized_row_window_from_dense_values<F>(
    msg: &Grib2Message,
    nx: usize,
    ny: usize,
    y_start: usize,
    y_end: usize,
    mut next_value: F,
) -> Result<Vec<f64>, String>
where
    F: FnMut() -> Result<Option<f64>, String>,
{
    let total_points = nx
        .checked_mul(ny)
        .ok_or_else(|| "grid point count overflow".to_string())?;
    if total_points > 100_000_000 {
        return Err(format!(
            "grid has {total_points} points which exceeds the 100M sanity limit"
        ));
    }
    if let Some(bitmap) = msg.bitmap.as_ref() {
        if bitmap.len() < total_points {
            return Err(format!(
                "row-window bitmap length {} was shorter than grid point count {}",
                bitmap.len(),
                total_points
            ));
        }
    }

    let mut window = vec![f64::NAN; (y_end - y_start) * nx];
    let flip_y = msg.grid.scan_mode & 0x40 != 0;
    for idx in 0..total_points {
        let active = msg
            .bitmap
            .as_ref()
            .map(|bitmap| bitmap[idx])
            .unwrap_or(true);
        if !active {
            continue;
        }
        // Running out of coded values mid-window used to `break`, leaving the
        // untouched tail of the window at its NaN fill -- a short payload
        // reaching a tile as a partly-missing field rather than as an error.
        let value = next_value()?.ok_or_else(|| {
            "packed data ended before every grid/bitmap point was decoded".to_string()
        })?;

        let source_y = idx / nx;
        let normalized_y = if flip_y { ny - 1 - source_y } else { source_y };
        if normalized_y < y_start || normalized_y >= y_end {
            continue;
        }
        let x = idx % nx;
        let out_idx = (normalized_y - y_start) * nx + x;
        window[out_idx] = value;
    }

    Ok(window)
}

/// Template 5.0: Simple packing.
fn unpack_simple(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    let bpv = dr.bits_per_value as usize;
    if bpv == 0 {
        // All values are the reference value
        let n = if !data.is_empty() { 1 } else { 0 };
        return Ok(vec![dr.reference_value as f64; n]);
    }
    if bpv > 63 {
        return Err(format!(
            "simple packing bits_per_value {bpv} exceeds the signed 63-bit decoder limit"
        ));
    }

    let total_bits = data.len() * 8;
    let n = total_bits / bpv;
    let mut reader = BitReader::new(data);
    let mut raw = Vec::with_capacity(n);
    for _ in 0..n {
        raw.push(
            i64::try_from(reader.read_bits_checked(bpv)?)
                .map_err(|_| "simple-packed value exceeds i64".to_string())?,
        );
    }

    Ok(apply_scaling(&raw, dr))
}

/// Code Table 5.5: how Templates 5.2 and 5.3 carry missing values.
///
/// A complex-packed message may declare missing cells IN BAND instead of
/// shipping a Section-6 bitmap: octet 23 of Section 5 reserves the
/// all-ones bit pattern of each group (and, in mode 2, the pattern one
/// below it) as a marker rather than a value.  A decoder that never
/// reads octet 23 hands those markers back as physical data -- for a
/// 4-bit group that is fifteen counts above the group reference, scaled
/// like everything else, with no error and no NaN.  Mode 0 is the
/// common NCEP case and is the only one this decoder used to accept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingValueMode {
    /// Octet 23 = 0: no in-band markers; a Section-6 bitmap, if any,
    /// carries the missing cells.
    None,
    /// Octet 23 = 1: the primary substitute is reserved in band.
    Primary,
    /// Octet 23 = 2: primary and secondary substitutes are reserved.
    PrimaryAndSecondary,
}

/// Read Code Table 5.5 off the data representation, refusing what this
/// decoder cannot represent rather than guessing.
///
/// The refusal names its breakage: a reserved code means the encoder
/// reserved bit patterns this decoder cannot identify, and decoding
/// anyway is the original defect by another route.
///
/// The companion `bits_per_value == 0` refusal lives in
/// `refuse_ambiguous_constant_group`, which runs once the group widths are
/// known.  It used to sit here and refused every mode-1/2 field whose
/// `bits_per_value` was 0, including fields whose groups all carry a real
/// width from `group_width_ref` -- those decode unambiguously and the
/// blanket refusal turned them into decode failures.
pub fn missing_value_mode(dr: &DataRepresentation) -> Result<MissingValueMode, String> {
    match dr.missing_value_management {
        0 => Ok(MissingValueMode::None),
        1 => Ok(MissingValueMode::Primary),
        2 => Ok(MissingValueMode::PrimaryAndSecondary),
        other => Err(format!(
            "complex packing missing-value management {other} is reserved in Code \
             Table 5.5; refusing rather than decoding as data the bit patterns it \
             reserves as missing-value markers"
        )),
    }
}

/// Refuse a width-zero (constant) group whose reference is simultaneously
/// the only representable value and the missing-value marker.
///
/// A width-zero group takes its value from the group reference, which is
/// `bits_per_value` bits wide; the marker for that group is the all-ones
/// pattern of the same width.  When `bits_per_value` is 0 the reference can
/// only ever be 0 and the "all-ones pattern of zero bits" is not a number,
/// so every such group is either wholly missing or the marker means nothing.
/// Picking one silently is a guess about somebody's data.
///
/// Groups whose width came from `group_width_ref` are unaffected: their
/// markers are the all-ones pattern of that width, which is well defined.
fn refuse_ambiguous_constant_group(
    bits_per_value: usize,
    mode: MissingValueMode,
    group_widths: &[usize],
) -> Result<(), String> {
    if mode == MissingValueMode::None || bits_per_value != 0 {
        return Ok(());
    }
    if group_widths.iter().any(|width| *width == 0) {
        return Err(format!(
            "complex packing declares missing-value management {} with bits_per_value 0 \
             for a zero-width group, which reserves the only representable group \
             reference as the missing-value marker; refusing rather than publishing a \
             group that is either wholly missing or wholly marker",
            match mode {
                MissingValueMode::Primary => 1,
                MissingValueMode::PrimaryAndSecondary => 2,
                MissingValueMode::None => 0,
            }
        ));
    }
    Ok(())
}

/// One decoded complex-packing cell: an unscaled integer, or a cell the
/// encoder marked missing in band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupValue {
    Present(i64),
    Missing,
}

/// The all-ones pattern for `width` bits, or `None` when no pattern can
/// be reserved: a zero-width read carries no bits, and a width past 64
/// is refused by the bit reader before it gets here.
fn all_ones(width: usize) -> Option<u64> {
    match width {
        0 => None,
        64 => Some(u64::MAX),
        width if width > 64 => None,
        width => Some((1u64 << width) - 1),
    }
}

/// The bit patterns one group reserves as missing-value markers.
#[derive(Debug, Clone, Copy)]
struct MissingMarkers {
    primary: Option<u64>,
    secondary: Option<u64>,
}

impl MissingMarkers {
    fn for_width(width: usize, mode: MissingValueMode) -> Self {
        match mode {
            MissingValueMode::None => Self {
                primary: None,
                secondary: None,
            },
            MissingValueMode::Primary => Self {
                primary: all_ones(width),
                secondary: None,
            },
            MissingValueMode::PrimaryAndSecondary => {
                let primary = all_ones(width);
                Self {
                    primary,
                    // all-ones-minus-one; `all_ones` is at least 1 for
                    // every width it answers for, so this cannot wrap.
                    secondary: primary.map(|pattern| pattern - 1),
                }
            }
        }
    }

    fn marks(&self, stored: u64) -> bool {
        self.primary == Some(stored) || self.secondary == Some(stored)
    }
}

/// Apply the GRIB2 scaling formula to decoded cells, carrying in-band
/// missing cells out as NaN -- the same representation this decoder
/// already uses for cells a Section-6 bitmap excludes, so a consumer
/// that handles one handles the other.
fn apply_scaling_values(values: &[GroupValue], dr: &DataRepresentation) -> Vec<f64> {
    let r = dr.reference_value as f64;
    let e = dr.binary_scale as f64;
    let d = dr.decimal_scale as f64;
    let two_e = 2.0_f64.powf(e);
    let ten_neg_d = 10.0_f64.powf(-d);

    values
        .iter()
        .map(|value| match value {
            GroupValue::Present(raw) => (r + *raw as f64 * two_e) * ten_neg_d,
            GroupValue::Missing => f64::NAN,
        })
        .collect()
}

/// Template 5.2: Complex packing.
fn unpack_complex(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    let mode = missing_value_mode(dr)?;
    let ng = dr.num_groups as usize;
    if ng == 0 {
        return Ok(Vec::new());
    }

    let bpv = dr.bits_per_value as usize;
    let mut reader = BitReader::new(data);

    // 1. Read group reference values (each is bits_per_value bits)
    let mut group_refs = Vec::with_capacity(ng);
    for _ in 0..ng {
        group_refs.push(reader.read_bits_checked(bpv)? as i64);
    }
    reader.align_to_byte();

    // 2. Read group widths (each is group_width_bits bits)
    let gwb = dr.group_width_bits as usize;
    let mut group_widths = Vec::with_capacity(ng);
    for _ in 0..ng {
        group_widths.push(reader.read_bits_checked(gwb)? as usize + dr.group_width_ref as usize);
    }
    reader.align_to_byte();

    // 3. Read group lengths: read ALL ng values, then overwrite last with DRS value
    let glb = dr.group_length_bits as usize;
    let mut group_lengths = Vec::with_capacity(ng);
    for _ in 0..ng {
        let stored = reader.read_bits_checked(glb)? as usize;
        group_lengths.push(stored * dr.group_length_inc as usize + dr.group_length_ref as usize);
    }
    if ng > 0 {
        group_lengths[ng - 1] = dr.last_group_length as usize;
    }
    reader.align_to_byte();

    // 4. Unpack each group's values
    let total_values: usize = group_lengths.iter().sum();
    if dr.section5_num_data_points != 0 && total_values != dr.section5_num_data_points as usize {
        return Err(format!(
            "complex group lengths declare {total_values} values, Section 5 declares {}",
            dr.section5_num_data_points
        ));
    }
    let raw = read_group_values(
        &mut reader,
        &group_refs,
        &group_widths,
        &group_lengths,
        total_values,
        dr.bits_per_value as usize,
        mode,
    )?;

    Ok(apply_scaling_values(&raw, dr))
}

/// Read every group's cells, classifying in-band missing markers.
///
/// A group of non-zero width reserves the all-ones pattern OF THAT
/// WIDTH; a zero-width (constant) group carries no per-cell bits, so its
/// marker is the all-ones pattern of the field's `bits_per_value`
/// compared against the group reference itself.  Both branches match
/// NCEP g2c's `comunpack.c`.
fn read_group_values(
    reader: &mut BitReader<'_>,
    group_refs: &[i64],
    group_widths: &[usize],
    group_lengths: &[usize],
    total_values: usize,
    bits_per_value: usize,
    mode: MissingValueMode,
) -> Result<Vec<GroupValue>, String> {
    refuse_ambiguous_constant_group(bits_per_value, mode, group_widths)?;
    let constant_markers = MissingMarkers::for_width(bits_per_value, mode);
    let mut raw = Vec::with_capacity(total_values);

    for g in 0..group_lengths.len() {
        let width = group_widths[g];
        let length = group_lengths[g];
        let gref = group_refs[g];

        if width == 0 {
            let value = if constant_markers.marks(gref as u64) {
                GroupValue::Missing
            } else {
                GroupValue::Present(gref)
            };
            for _ in 0..length {
                raw.push(value);
            }
            continue;
        }

        let markers = MissingMarkers::for_width(width, mode);
        for _ in 0..length {
            let stored = reader.read_bits_checked(width)?;
            raw.push(if markers.marks(stored) {
                GroupValue::Missing
            } else {
                GroupValue::Present(gref + stored as i64)
            });
        }
    }

    Ok(raw)
}

/// Template 5.3: Complex packing with spatial differencing.
fn unpack_complex_spatial(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    let mode = missing_value_mode(dr)?;
    let order = dr.spatial_diff_order as usize;
    let extra_bytes = dr.spatial_diff_bytes as usize;

    if order == 0 || extra_bytes == 0 {
        return unpack_complex(data, dr);
    }

    let nbits = extra_bytes * 8;
    let mut reader = BitReader::new(data);

    // Read the initial values (1 for order=1, 2 for order=2)
    let mut initial_values = Vec::with_capacity(order);
    for _ in 0..order {
        let val = i64::try_from(reader.read_bits_checked(nbits)?)
            .map_err(|_| "spatial-differencing initial value exceeded i64".to_string())?;
        initial_values.push(val);
    }

    // Read the minimum value (sign-magnitude)
    let sign = reader.read_bits_checked(1)?;
    let magnitude = reader.read_bits_checked(nbits - 1)? as i64;
    let minimum = if sign == 1 { -magnitude } else { magnitude };

    reader.align_to_byte();

    // Now read the rest as complex-packed groups from current position
    let consumed_bytes = (reader.bit_pos + 7) / 8;
    let remaining_data = &data[consumed_bytes..];

    let ng = dr.num_groups as usize;
    if ng == 0 {
        return Ok(Vec::new());
    }

    let bpv = dr.bits_per_value as usize;
    let mut greader = BitReader::new(remaining_data);

    // Read group references
    let mut group_refs = Vec::with_capacity(ng);
    for _ in 0..ng {
        group_refs.push(greader.read_bits_checked(bpv)? as i64);
    }
    greader.align_to_byte();

    // Read group widths
    let gwb = dr.group_width_bits as usize;
    let mut group_widths = Vec::with_capacity(ng);
    for _ in 0..ng {
        group_widths.push(greader.read_bits_checked(gwb)? as usize + dr.group_width_ref as usize);
    }
    greader.align_to_byte();

    // Read group lengths: must read ALL ng values from the stream (not ng-1),
    // then overwrite the last one with the true last group length from the DRS.
    // This matches g2clib behavior and ensures correct bit alignment for packed data.
    let glb = dr.group_length_bits as usize;
    let mut group_lengths = Vec::with_capacity(ng);
    for _ in 0..ng {
        let stored = greader.read_bits_checked(glb)? as usize;
        group_lengths.push(stored * dr.group_length_inc as usize + dr.group_length_ref as usize);
    }
    // Overwrite last group length with the true value from DRS
    if ng > 0 {
        group_lengths[ng - 1] = dr.last_group_length as usize;
    }
    greader.align_to_byte();

    // Unpack group values
    let total_values: usize = group_lengths.iter().sum();
    if dr.section5_num_data_points != 0 && total_values != dr.section5_num_data_points as usize {
        return Err(format!(
            "spatial complex group lengths declare {total_values} values, Section 5 declares {}",
            dr.section5_num_data_points
        ));
    }
    // Spatial differencing runs over the PRESENT cells only.  A cell the
    // encoder marked missing carries no residual, so integrating across
    // it does not merely lose that cell: every value after it inherits
    // the marker.  g2c does the same by packing the present residuals
    // densely and integrating that sequence.
    //
    // The present cells are read straight into that dense sequence, with
    // the missing ones kept as a position mask, instead of as one tagged
    // cell per grid point that is then filtered, re-tagged and scaled.
    // The arithmetic and the order in which every refusal can be raised
    // are unchanged (see `read_spatial_residuals`); what goes is about two
    // thirds of the memory traffic, which is what bounded a decode running
    // on every core: each of those tagged copies is 16 bytes per cell.
    let (mut present, missing) = read_spatial_residuals(
        &mut greader,
        &group_refs,
        &group_widths,
        &group_lengths,
        total_values,
        dr.bits_per_value as usize,
        mode,
        &initial_values,
        minimum,
    )?;

    // Every step of the integration is checked.  An encoder that ships a
    // descriptor wide enough to hold near-i64 initial values can drive the
    // recurrence past i64; unchecked, that is a debug panic or a release
    // wrap-around, and a wrapped residual is published as physics.
    if order == 1 {
        for i in 1..present.len() {
            present[i] = present[i]
                .checked_add(present[i - 1])
                .ok_or_else(|| "first-order spatial recurrence overflow".to_string())?;
        }
    } else if order == 2 {
        for i in 2..present.len() {
            present[i] = present[i - 1]
                .checked_mul(2)
                .and_then(|value| value.checked_sub(present[i - 2]))
                .and_then(|value| value.checked_add(present[i]))
                .ok_or_else(|| "second-order spatial recurrence overflow".to_string())?;
        }
    }

    // `apply_scaling_values`, cell for cell: the same three constants and
    // the same expression, over the present sequence and the mask.
    let r = dr.reference_value as f64;
    let two_e = 2.0_f64.powf(dr.binary_scale as f64);
    let ten_neg_d = 10.0_f64.powf(-(dr.decimal_scale as f64));
    let scale = |raw: i64| (r + raw as f64 * two_e) * ten_neg_d;
    Ok(match missing {
        None => present.into_iter().map(scale).collect(),
        Some(missing) => {
            let mut values = present.into_iter();
            missing
                .into_iter()
                .map(|is_missing| {
                    if is_missing {
                        f64::NAN
                    } else {
                        values.next().map_or(f64::NAN, scale)
                    }
                })
                .collect()
        }
    })
}

/// The present cells of a spatially differenced field, read DENSELY: the
/// first `initial_values.len()` present residuals replaced by those
/// absolute initial values, every later one offset by `minimum`, and the
/// missing cells as a position mask (`None` when the field cannot carry
/// in-band missing markers, so there is no mask to keep).
///
/// This is `read_group_values` followed by the filter, the initial-value
/// replacement and the minimum pass that `unpack_complex_spatial` used to
/// run over it, with every refusal raised in the same order: the constant
/// group check first, then a truncated payload (the whole stream is read
/// before anything else is judged), then an overflow while adding the
/// minimum -- recorded where it happens and raised once the stream has
/// been read, exactly as the separate pass raised it.
#[allow(clippy::too_many_arguments)]
fn read_spatial_residuals(
    reader: &mut BitReader<'_>,
    group_refs: &[i64],
    group_widths: &[usize],
    group_lengths: &[usize],
    total_values: usize,
    bits_per_value: usize,
    mode: MissingValueMode,
    initial_values: &[i64],
    minimum: i64,
) -> Result<(Vec<i64>, Option<Vec<bool>>), String> {
    refuse_ambiguous_constant_group(bits_per_value, mode, group_widths)?;
    let constant_markers = MissingMarkers::for_width(bits_per_value, mode);
    let mut present: Vec<i64> = Vec::with_capacity(total_values);
    let mut missing: Option<Vec<bool>> = match mode {
        MissingValueMode::None => None,
        _ => Some(Vec::with_capacity(total_values)),
    };
    let mut minimum_overflow = false;
    let mut push = |raw: Option<i64>, present: &mut Vec<i64>| {
        match raw {
            Some(raw) => {
                let index = present.len();
                let value = if index < initial_values.len() {
                    initial_values[index]
                } else {
                    match raw.checked_add(minimum) {
                        Some(value) => value,
                        None => {
                            minimum_overflow = true;
                            raw
                        }
                    }
                };
                present.push(value);
                if let Some(missing) = missing.as_mut() {
                    missing.push(false);
                }
            }
            None => {
                if let Some(missing) = missing.as_mut() {
                    missing.push(true);
                }
            }
        }
    };

    for g in 0..group_lengths.len() {
        let width = group_widths[g];
        let length = group_lengths[g];
        let gref = group_refs[g];

        if width == 0 {
            let value = if constant_markers.marks(gref as u64) { None } else { Some(gref) };
            for _ in 0..length {
                push(value, &mut present);
            }
            continue;
        }

        let markers = MissingMarkers::for_width(width, mode);
        for _ in 0..length {
            let stored = reader.read_bits_checked(width)?;
            push(
                if markers.marks(stored) { None } else { Some(gref + stored as i64) },
                &mut present,
            );
        }
    }
    drop(push);
    if minimum_overflow {
        return Err("spatial-differencing difference overflow while adding minimum".to_string());
    }
    if let Some(mask) = missing.as_ref() {
        if !mask.iter().any(|is_missing| *is_missing) {
            // No cell is missing: the mask carries nothing.
            missing = None;
        }
    }
    Ok((present, missing))
}

/// Template 5.40: JPEG2000 packing (stub for platforms without openjp2).
#[cfg(not(feature = "jpeg2000"))]
fn unpack_jpeg2000(_data: &[u8], _dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    Err("JPEG2000 decoding not available (openjp2 feature disabled)".into())
}

#[cfg(not(feature = "jpeg2000"))]
fn unpack_jpeg2000_area(
    _data: &[u8],
    _dr: &DataRepresentation,
    _image: (usize, usize),
    _area: StoredRect,
) -> Result<Vec<f64>, String> {
    Err("JPEG2000 decoding not available (openjp2 feature disabled)".into())
}

/// Template 5.40: JPEG2000 packing.
///
/// Uses openjp2's C-style API to decode a JPEG2000 codestream embedded in GRIB2 Section 7.
/// GRIB2 JPEG2000 data is always a raw J2K codestream (starts with FF 4F).
#[cfg(feature = "jpeg2000")]
fn unpack_jpeg2000(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    decode_jpeg2000(data, dr, None)
}

/// Template 5.40, only the stored rectangle `area` of an `image` =
/// (columns, rows) codestream, through OpenJPEG's decode area: the
/// code-blocks the rectangle does not reach are never entropy-decoded
/// and the inverse wavelet runs only over the rows and columns the
/// rectangle needs.
#[cfg(feature = "jpeg2000")]
fn unpack_jpeg2000_area(
    data: &[u8],
    dr: &DataRepresentation,
    image: (usize, usize),
    area: StoredRect,
) -> Result<Vec<f64>, String> {
    decode_jpeg2000(data, dr, Some((image, area)))
}

#[cfg(feature = "jpeg2000")]
fn decode_jpeg2000(
    data: &[u8],
    dr: &DataRepresentation,
    area: Option<((usize, usize), StoredRect)>,
) -> Result<Vec<f64>, String> {
    use openjp2::openjpeg::*;
    use std::ffi::c_void;

    if data.is_empty() {
        return Ok(Vec::new());
    }

    // Detect format - GRIB2 embeds raw J2K codestreams (0xFF 0x4F)
    let format = openjp2::detect_format(data)
        .map_err(|e| format!("JPEG2000 format detection failed: {}", e))?;
    let codec_format = match format {
        openjp2::J2KFormat::J2K => OPJ_CODEC_J2K,
        openjp2::J2KFormat::JP2 => OPJ_CODEC_JP2,
        openjp2::J2KFormat::JPT => OPJ_CODEC_JPT,
    };

    // We use the C-style FFI API because the Rust Stream type has no public
    // constructor for in-memory buffers.

    // WrappedSlice for the stream callbacks
    struct WrappedSlice {
        offset: usize,
        len: usize,
        ptr: *const u8,
    }

    extern "C" fn j2k_read(p_buffer: *mut c_void, nb_bytes: usize, p_data: *mut c_void) -> usize {
        if p_buffer.is_null() || nb_bytes == 0 {
            return usize::MAX;
        }
        let slice = unsafe { &mut *(p_data as *mut WrappedSlice) };
        let remaining = slice.len - slice.offset;
        if remaining == 0 {
            return usize::MAX;
        }
        let n = remaining.min(nb_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.ptr.add(slice.offset), p_buffer as *mut u8, n);
        }
        slice.offset += n;
        n
    }

    extern "C" fn j2k_skip(nb_bytes: i64, p_data: *mut c_void) -> i64 {
        let slice = unsafe { &mut *(p_data as *mut WrappedSlice) };
        let new_off = (slice.offset as i64 + nb_bytes).max(0) as usize;
        slice.offset = new_off.min(slice.len);
        nb_bytes
    }

    extern "C" fn j2k_seek(nb_bytes: i64, p_data: *mut c_void) -> i32 {
        let slice = unsafe { &mut *(p_data as *mut WrappedSlice) };
        let off = nb_bytes as usize;
        if off <= slice.len {
            slice.offset = off;
            1
        } else {
            0
        }
    }

    extern "C" fn j2k_free(p_data: *mut c_void) {
        drop(unsafe { Box::from_raw(p_data as *mut WrappedSlice) });
    }

    let data_len = data.len();
    let wrapped = Box::new(WrappedSlice {
        offset: 0,
        len: data_len,
        ptr: data.as_ptr(),
    });
    let p_data = Box::into_raw(wrapped) as *mut c_void;

    // Create stream
    let stream = unsafe {
        let s = opj_stream_default_create(1);
        if s.is_null() {
            // Clean up wrapped data
            drop(Box::from_raw(p_data as *mut WrappedSlice));
            return Err("Failed to create JPEG2000 stream".into());
        }
        opj_stream_set_read_function(s, Some(j2k_read));
        opj_stream_set_skip_function(s, Some(j2k_skip));
        opj_stream_set_seek_function(s, Some(j2k_seek));
        opj_stream_set_user_data_length(s, data_len as u64);
        opj_stream_set_user_data(s, p_data, Some(j2k_free));
        s
    };

    // Create codec
    let codec = opj_create_decompress(codec_format);
    if codec.is_null() {
        unsafe {
            opj_stream_destroy(stream);
        }
        return Err("Failed to create JPEG2000 decoder".into());
    }

    // Setup decoder
    let mut params = opj_dparameters_t::default();
    let ret = unsafe { opj_setup_decoder(codec, &mut params) };
    if ret == 0 {
        unsafe {
            opj_destroy_codec(codec);
            opj_stream_destroy(stream);
        }
        return Err("Failed to setup JPEG2000 decoder".into());
    }

    // Read header
    let mut image: *mut opj_image_t = std::ptr::null_mut();
    let ret = unsafe { opj_read_header(stream, codec, &mut image) };
    if ret == 0 || image.is_null() {
        unsafe {
            opj_destroy_codec(codec);
            opj_stream_destroy(stream);
            if !image.is_null() {
                opj_image_destroy(image);
            }
        }
        return Err("Failed to read JPEG2000 header".into());
    }

    if let Some(((columns, rows), rect)) = area {
        // The codestream must BE the grid: an image of another size is a
        // bitmap-packed or otherwise reshaped payload, and a rectangle of
        // grid cells would then name other samples.
        let img = unsafe { &*image };
        let geometry_matches = img.x0 == 0
            && img.y0 == 0
            && img.x1 as usize == columns
            && img.y1 as usize == rows
            && img.numcomps >= 1;
        let set = geometry_matches
            && unsafe {
                opj_set_decode_area(
                    codec,
                    image,
                    rect.columns.0 as i32,
                    rect.rows.0 as i32,
                    rect.columns.1 as i32,
                    rect.rows.1 as i32,
                )
            } != 0;
        if !set {
            unsafe {
                opj_destroy_codec(codec);
                opj_stream_destroy(stream);
                opj_image_destroy(image);
            }
            return Err(format!(
                "JPEG2000 codestream does not cover the {columns}x{rows} grid, so \
                 rows {}..{} and columns {}..{} of it cannot be decoded as an area",
                rect.rows.0, rect.rows.1, rect.columns.0, rect.columns.1
            ));
        }
    }

    // Decode
    let ret = unsafe { opj_decode(codec, stream, image) };
    if ret == 0 {
        unsafe {
            opj_destroy_codec(codec);
            opj_stream_destroy(stream);
            opj_image_destroy(image);
        }
        return Err("JPEG2000 decode failed".into());
    }

    // End decompress
    unsafe {
        opj_end_decompress(codec, stream);
    }

    // Extract data from first component
    let img = unsafe { &*image };
    let numcomps = img.numcomps as usize;
    if numcomps == 0 || img.comps.is_null() {
        unsafe {
            opj_destroy_codec(codec);
            opj_stream_destroy(stream);
            opj_image_destroy(image);
        }
        return Err("JPEG2000 image has no components".into());
    }

    let comp = unsafe { &*img.comps };
    let n = (comp.w * comp.h) as usize;
    if let Some((_, rect)) = area {
        let expected = (rect.columns.1 - rect.columns.0, rect.rows.1 - rect.rows.0);
        if (comp.w as usize, comp.h as usize) != expected {
            unsafe {
                opj_destroy_codec(codec);
                opj_stream_destroy(stream);
                opj_image_destroy(image);
            }
            return Err(format!(
                "JPEG2000 decode area returned {}x{} samples; the rectangle is {}x{}",
                comp.w, comp.h, expected.0, expected.1
            ));
        }
    }
    let raw: Vec<i64> = if let Some(data_slice) = comp.data() {
        data_slice.iter().take(n).map(|&v| v as i64).collect()
    } else {
        unsafe {
            opj_destroy_codec(codec);
            opj_stream_destroy(stream);
            opj_image_destroy(image);
        }
        return Err("JPEG2000 component has no data".into());
    };

    // Cleanup
    unsafe {
        opj_destroy_codec(codec);
        opj_stream_destroy(stream);
        opj_image_destroy(image);
    }

    Ok(apply_scaling(&raw, dr))
}

/// Template 5.41: PNG packing (stub for platforms without png crate).
#[cfg(not(feature = "png_codec"))]
fn unpack_png(_data: &[u8], _dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    Err("PNG decoding not available (png_codec feature disabled)".into())
}

/// Template 5.41: PNG packing.
#[cfg(feature = "png_codec")]
fn unpack_png(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let decoder = png::Decoder::new(std::io::Cursor::new(data));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("PNG decode error: {}", e))?;

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("PNG frame error: {}", e))?;
    let bytes = &buf[..info.buffer_size()];

    let bpv = dr.bits_per_value as usize;
    let mut raw = Vec::new();

    match bpv {
        8 => {
            for &b in bytes {
                raw.push(b as i64);
            }
        }
        16 => {
            for chunk in bytes.chunks_exact(2) {
                raw.push(u16::from_be_bytes([chunk[0], chunk[1]]) as i64);
            }
        }
        24 => {
            for chunk in bytes.chunks_exact(3) {
                let v = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | chunk[2] as u32;
                raw.push(v as i64);
            }
        }
        32 => {
            for chunk in bytes.chunks_exact(4) {
                raw.push(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64);
            }
        }
        _ => {
            for &b in bytes {
                raw.push(b as i64);
            }
        }
    }

    Ok(apply_scaling(&raw, dr))
}

/// Template 5.4 / 7.4: IEEE Floating Point packing.
///
/// Values are stored as IEEE 754 floats directly: no packing or scaling.
/// Supports 32-bit (f32) and 64-bit (f64) precision based on `bits_per_value`.
fn unpack_ieee(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let bpv = dr.bits_per_value as usize;
    match bpv {
        32 => {
            let n = data.len() / 4;
            let mut values = Vec::with_capacity(n);
            for chunk in data.chunks_exact(4) {
                let v = f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                values.push(v as f64);
            }
            Ok(values)
        }
        64 => {
            let n = data.len() / 8;
            let mut values = Vec::with_capacity(n);
            for chunk in data.chunks_exact(8) {
                let v = f64::from_be_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                values.push(v);
            }
            Ok(values)
        }
        _ => Err(format!(
            "IEEE float packing: unsupported bits_per_value={} (expected 32 or 64)",
            bpv
        )),
    }
}

/// Template 5.42 / 7.42: CCSDS (AEC/SZIP) packing.
///
/// Decodes data compressed with the CCSDS Adaptive Entropy Coding (AEC) algorithm,
/// also known as SZIP. This is the primary packing used by ERA5 and many ECMWF products.
///
/// When the `ccsds` feature is enabled, uses the libaec C library via FFI for maximum
/// compatibility. Otherwise, uses a pure-Rust AEC decoder.
fn unpack_ccsds(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let bits_per_sample = dr.bits_per_value as usize;
    if bits_per_sample == 0 {
        return Ok(Vec::new());
    }

    let block_size = dr.ccsds_block_size as u32;
    let rsi = dr.ccsds_rsi as u32;
    let flags = dr.ccsds_flags as u8;

    // Compute the decoder's bytes_per_sample following eccodes/libaec convention:
    // ceil(bits/8), then upgrade 3 to 4 (since we always clear AEC_DATA_3BYTE).
    let mut decoder_bps = (bits_per_sample + 7) / 8;
    if decoder_bps == 3 {
        decoder_bps = 4;
    }

    // Use the exact number of data points from Section 5 to compute avail_out,
    // matching eccodes: avail_out = n_vals * decoder_bps.
    let n_vals = if dr.section5_num_data_points > 0 {
        dr.section5_num_data_points as usize
    } else {
        // Fallback: estimate from compressed data size
        let rsi_block_samples = rsi as usize * block_size as usize;
        let estimated = (data.len() * 8 / bits_per_sample.max(1)).max(rsi_block_samples);
        ((estimated + rsi_block_samples - 1) / rsi_block_samples) * rsi_block_samples
    };
    let avail_out = n_vals * decoder_bps;

    // Decode using FFI or pure Rust
    let decoded_values = ccsds_decode(data, block_size, rsi, flags, bits_per_sample, avail_out)?;

    Ok(apply_scaling(&decoded_values, dr))
}

/// Decode CCSDS/AEC compressed data using libaec FFI (when `ccsds` feature is enabled).
#[cfg(feature = "ccsds")]
fn ccsds_decode(
    data: &[u8],
    block_size: u32,
    rsi: u32,
    flags_byte: u8,
    bits_per_sample: usize,
    avail_out: usize,
) -> Result<Vec<i64>, String> {
    use libaec_sys::*;
    use std::mem;

    let mut strm: aec_stream = unsafe { mem::zeroed() };
    strm.bits_per_sample = bits_per_sample as u32;
    strm.block_size = block_size;
    strm.rsi = rsi;

    // The GRIB2 compression_options_mask byte uses the same bit layout as
    // libaec flags. Following eccodes: apply modify_aec_flags to clear
    // AEC_DATA_3BYTE and set endianness based on native platform.
    let mut aec_flags = flags_byte as u32;
    aec_flags &= !(AEC_DATA_3BYTE); // clear 3-byte mode
    if cfg!(target_endian = "big") {
        aec_flags |= AEC_DATA_MSB;
    } else {
        aec_flags &= !(AEC_DATA_MSB);
    }
    strm.flags = aec_flags;

    // Allocate output buffer
    let mut output = vec![0u8; avail_out];

    strm.next_in = data.as_ptr();
    strm.avail_in = data.len();
    strm.next_out = output.as_mut_ptr();
    strm.avail_out = avail_out;

    // Initialize decoder
    let ret = unsafe { aec_decode_init(&mut strm) };
    if ret != AEC_OK as i32 {
        return Err(format!("libaec aec_decode_init failed with code {}", ret));
    }

    // Decode
    let ret = unsafe { aec_decode(&mut strm, AEC_FLUSH as i32) };
    let bytes_written = avail_out - strm.avail_out;

    // Cleanup
    unsafe {
        aec_decode_end(&mut strm);
    }

    if ret != AEC_OK as i32 {
        return Err(format!("libaec aec_decode failed with code {}", ret));
    }

    // The decoder's bytes_per_sample: ceil(bps/8), with 3 upgraded to 4
    // (since AEC_DATA_3BYTE is cleared). This matches eccodes.
    let mut decoder_bps = (bits_per_sample + 7) / 8;
    if decoder_bps == 3 {
        decoder_bps = 4;
    }

    // Convert decoded bytes to integer values using native endianness.
    // The libaec output is in native byte order (we set MSB/LSB accordingly).
    let num_samples = bytes_written / decoder_bps;
    let mut values = Vec::with_capacity(num_samples);
    let out = &output[..bytes_written];

    for i in 0..num_samples {
        let start = i * decoder_bps;
        let val = match decoder_bps {
            1 => out[start] as i64,
            2 => {
                if cfg!(target_endian = "big") {
                    u16::from_be_bytes([out[start], out[start + 1]]) as i64
                } else {
                    u16::from_le_bytes([out[start], out[start + 1]]) as i64
                }
            }
            4 => {
                if cfg!(target_endian = "big") {
                    u32::from_be_bytes([out[start], out[start + 1], out[start + 2], out[start + 3]])
                        as i64
                } else {
                    u32::from_le_bytes([out[start], out[start + 1], out[start + 2], out[start + 3]])
                        as i64
                }
            }
            _ => return Err(format!("Unsupported decoder_bps: {}", decoder_bps)),
        };
        values.push(val);
    }

    Ok(values)
}

/// Pure-Rust CCSDS/AEC decoder (used when `ccsds` feature is not enabled).
///
/// This is a port of the AEC decoding algorithm based on the CCSDS 121.0-B-3
/// Lossless Data Compression standard. The algorithm uses adaptive entropy coding
/// with fundamental sequence (FS) encoding and optional preprocessing.
#[cfg(not(feature = "ccsds"))]
fn ccsds_decode(
    data: &[u8],
    block_size: u32,
    rsi: u32,
    flags_byte: u8,
    bits_per_sample: usize,
    avail_out: usize,
) -> Result<Vec<i64>, String> {
    aec_pure::decode(
        data,
        block_size,
        rsi,
        flags_byte,
        bits_per_sample,
        avail_out,
    )
}

/// Pure-Rust AEC decoder module.
///
/// Implements the CCSDS 121.0-B-3 Adaptive Entropy Coding algorithm in pure Rust.
/// This is a self-contained implementation with no external dependencies.
mod aec_pure {
    use std::collections::VecDeque;

    const ROS: u32 = 5;
    const SE_TABLE_SIZE: usize = 90;

    /// AEC compression flags (maps to libaec flag bits).
    const AEC_DATA_SIGNED: u8 = 0x01;
    const AEC_DATA_3BYTE: u8 = 0x02;
    const AEC_DATA_MSB: u8 = 0x04;
    const AEC_DATA_PREPROCESS: u8 = 0x08;
    #[allow(dead_code)]
    const AEC_RESTRICTED: u8 = 0x10;
    const AEC_PAD_RSI: u8 = 0x20;

    /// AEC decoder internal state machine modes.
    #[derive(Debug, Clone)]
    enum Mode {
        Id,
        LowEntropy,
        LowEntropyRef,
        ZeroBlock,
        ZeroOutput,
        SE,
        SEIncremental,
        Uncomp,
        UncompCopy,
        Split,
        SplitFs,
        SplitOutput,
        NextCds,
    }

    /// Result of a single decode step.
    enum StepResult {
        Continue,
        Exit,
        Error(String),
    }

    /// Internal state for the AEC decoder.
    struct State {
        bits_per_sample: usize,
        block_size: u32,
        flags: u8,
        next_in: VecDeque<u8>,
        next_out: Vec<u8>,
        avail_in: usize,
        avail_out: usize,
        flush_start: usize,
        bitp: usize,
        acc: u64,
        fs: u32,
        id: u32,
        id_len: usize,
        id_table: Vec<Mode>,
        xmax: u32,
        xmin: u32,
        bytes_per_sample: usize,
        out_blklen: usize,
        in_blklen: usize,
        encoded_block_size: u32,
        sample_counter: u32,
        mode: Mode,
        se_table: [i32; 2 * (SE_TABLE_SIZE + 1)],
        rsi_size: usize,
        reff: usize,
        last_out: i32,
        rsip: usize,
        rsi: u32,
        rsi_buffer: Vec<u32>,
        pp: bool,
    }

    impl State {
        fn new(
            bits_per_sample: usize,
            block_size: u32,
            rsi: u32,
            flags: u8,
            avail_out: usize,
            data: &[u8],
        ) -> Result<State, String> {
            if bits_per_sample == 0 || bits_per_sample > 32 {
                return Err(format!("Invalid bits_per_sample: {}", bits_per_sample));
            }

            let (bytes_per_sample, id_len) = match bits_per_sample {
                25..=32 => (4, 5),
                17..=24 => {
                    let bps = if flags & AEC_DATA_3BYTE != 0 { 3 } else { 4 };
                    (bps, 5)
                }
                9..=16 => (2, 4),
                5..=8 => (1, 3),
                4 => (1, 2),
                2..=3 => (1, 1),
                _ => return Err(format!("Invalid bits_per_sample: {}", bits_per_sample)),
            };

            let (xmin, xmax) = if flags & AEC_DATA_SIGNED != 0 {
                let xmax = ((1i64 << (bits_per_sample - 1)) - 1) as u32;
                (!xmax, xmax)
            } else {
                (0, ((1u64 << bits_per_sample) - 1) as u32)
            };

            let modi = 1usize << id_len;
            let mut id_table = vec![Mode::Split; modi];
            id_table[0] = Mode::LowEntropy;
            id_table[modi - 1] = Mode::Uncomp;

            let rsi_size = (rsi * block_size) as usize;
            let pp = flags & AEC_DATA_PREPROCESS != 0;
            let reff: usize = if pp { 1 } else { 0 };
            let encoded_block_size = block_size - reff as u32;
            let out_blklen = (block_size as usize) * bytes_per_sample;
            let in_blklen = ((block_size as usize) * bits_per_sample + id_len) / 8 + 16;

            Ok(State {
                bits_per_sample,
                block_size,
                flags,
                next_in: data.iter().copied().collect(),
                next_out: Vec::new(),
                avail_in: data.len(),
                avail_out,
                flush_start: 0,
                bitp: 0,
                acc: 0,
                fs: 0,
                id: 0,
                id_len,
                id_table,
                xmax,
                xmin,
                bytes_per_sample,
                out_blklen,
                in_blklen,
                encoded_block_size,
                sample_counter: 0,
                mode: Mode::Id,
                se_table: create_se_table(),
                rsi_size,
                reff,
                last_out: 0,
                rsip: 0,
                rsi,
                rsi_buffer: vec![0u32; rsi_size],
                pp,
            })
        }

        fn run(&mut self) -> StepResult {
            match self.mode {
                Mode::Id => self.run_id(),
                Mode::LowEntropy => self.run_low_entropy(),
                Mode::LowEntropyRef => self.run_low_entropy_ref(),
                Mode::ZeroBlock => self.run_zero_block(),
                Mode::ZeroOutput => self.run_zero_output(),
                Mode::SE => self.run_se(),
                Mode::SEIncremental => self.run_se_decode(),
                Mode::Uncomp => self.run_uncomp(),
                Mode::UncompCopy => self.run_uncomp_copy(),
                Mode::Split => self.run_split(),
                Mode::SplitFs => self.run_split_fs(),
                Mode::SplitOutput => self.run_split_output(),
                Mode::NextCds => self.run_next_cds(),
            }
        }

        // ---- Bit I/O primitives ----

        fn ask_byte(&mut self) -> bool {
            if self.avail_in == 0 {
                return false;
            }
            self.avail_in -= 1;
            let byte: u64 = self.next_in.pop_front().unwrap() as u64;
            self.acc = (self.acc << 8) | byte;
            self.bitp += 8;
            true
        }

        fn bits_ask(&mut self, n: usize) -> bool {
            while self.bitp < n {
                if !self.ask_byte() {
                    return false;
                }
            }
            true
        }

        fn bits_get(&self, n: usize) -> u32 {
            ((self.acc >> (self.bitp - n)) & (u64::MAX >> (64 - n))) as u32
        }

        fn bits_drop(&mut self, n: usize) {
            self.bitp -= n;
        }

        fn fs_ask(&mut self) -> bool {
            if !self.bits_ask(1) {
                return false;
            }
            while (self.acc & (1u64 << (self.bitp - 1))) == 0 {
                if self.bitp == 1 && !self.ask_byte() {
                    return false;
                }
                self.fs += 1;
                self.bitp -= 1;
            }
            true
        }

        fn fs_drop(&mut self) {
            self.fs = 0;
            self.bitp -= 1;
        }

        fn direct_drain(&mut self, b: usize) {
            let mut shift = b * 8;
            let mut acc = self.acc << shift;
            for byte in self.next_in.drain(..b) {
                shift -= 8;
                acc |= (byte as u64) << shift;
            }
            self.acc = acc;
            self.avail_in -= b;
        }

        fn direct_get(&mut self, n: usize) -> u32 {
            if self.bitp < n {
                let b = (63 - self.bitp) >> 3;
                self.direct_drain(b);
                self.bitp += b << 3;
            }
            self.bitp -= n;
            ((self.acc >> self.bitp) & (u64::MAX >> (64 - n as u64))) as u32
        }

        fn direct_get_fs(&mut self) -> u32 {
            let mut fs: u32 = 0;
            if self.bitp > 0 {
                self.acc &= u64::MAX >> (64 - self.bitp);
            } else {
                self.acc = 0;
            }
            while self.acc == 0 {
                if self.avail_in < 7 {
                    return 0;
                }
                self.direct_drain(7);
                fs += self.bitp as u32;
                self.bitp = 56;
            }
            let i = 63 - self.acc.leading_zeros() as usize;
            fs += (self.bitp - i - 1) as u32;
            self.bitp = i;
            fs
        }

        fn buffer_space(&self) -> bool {
            self.avail_in >= self.in_blklen && self.avail_out >= self.out_blklen
        }

        // ---- Sample output ----

        fn put_sample(&mut self, sample: u32) {
            self.rsi_buffer[self.rsip] = sample;
            self.rsip += 1;
            self.avail_out -= self.bytes_per_sample;
        }

        fn put_sample_signed(&mut self, sample: i32) {
            self.rsi_buffer[self.rsip] = sample as u32;
            self.rsip += 1;
            self.avail_out -= self.bytes_per_sample;
        }

        fn put_bytes(&mut self, data: u32) {
            for i in 0..self.bytes_per_sample {
                if self.flags & AEC_DATA_MSB != 0 {
                    self.next_out
                        .push((data >> (8 * (self.bytes_per_sample - i - 1))) as u8);
                } else {
                    self.next_out.push((data >> (8 * i)) as u8);
                }
            }
        }

        fn copysample(&mut self) -> bool {
            if !self.bits_ask(self.bits_per_sample) || self.avail_out < self.bytes_per_sample {
                return false;
            }
            let sample = self.bits_get(self.bits_per_sample);
            self.put_sample(sample);
            self.bits_drop(self.bits_per_sample);
            true
        }

        fn rsi_used_size(&self) -> usize {
            self.rsip
        }

        // ---- Flush: convert RSI buffer to output bytes with optional preprocessing ----

        fn flush_kind(&mut self) {
            let flush_end = self.rsip;

            if self.pp {
                if self.flush_start == 0 && self.rsip > 0 {
                    self.last_out = self.rsi_buffer[0] as i32;

                    if self.flags & AEC_DATA_SIGNED != 0 {
                        let m = 1u32 << (self.bits_per_sample - 1);
                        let m2 = m as i32;
                        self.last_out = (self.last_out ^ m2) - m2;
                    }

                    self.put_bytes(self.last_out as u32);
                    self.flush_start += 1;
                }

                let mut data: u32 = self.last_out as u32;
                let xmax = self.xmax;

                if self.xmin == 0 {
                    // Unsigned data path
                    let med = self.xmax / 2 + 1;

                    for i in self.flush_start..flush_end {
                        let d = self.rsi_buffer[i];
                        let half_d = (d >> 1) + (d & 1);
                        let mask = if (data & med) == 0 { 0 } else { xmax };

                        if half_d <= (mask ^ data) {
                            data = data.wrapping_add((d >> 1) ^ (!(d & 1).wrapping_sub(1)));
                        } else {
                            data = mask ^ d;
                        }

                        self.put_bytes(data);
                    }
                    self.last_out = data as i32;
                } else {
                    // Signed data path
                    for i in self.flush_start..flush_end {
                        let d = self.rsi_buffer[i];
                        let half_d = (d >> 1) + (d & 1);
                        if (data as i32) < 0 {
                            if half_d <= xmax + data + 1 {
                                data = data.wrapping_add((d >> 1) ^ (!(d & 1).wrapping_sub(1)));
                            } else {
                                data = d - xmax - 1;
                            }
                        } else if half_d <= xmax - data {
                            data = data.wrapping_add((d >> 1) ^ (!(d & 1).wrapping_sub(1)));
                        } else {
                            data = xmax - d;
                        }

                        self.put_bytes(data);
                    }
                    self.last_out = data as i32;
                }
            } else {
                // No preprocessing: straight copy
                for i in self.flush_start..flush_end {
                    self.put_bytes(self.rsi_buffer[i]);
                }
            }

            self.flush_start = self.rsip;
        }

        // ---- Mode handlers ----

        fn run_id(&mut self) -> StepResult {
            if self.avail_in >= self.in_blklen {
                self.id = self.direct_get(self.id_len);
            } else {
                if !self.bits_ask(self.id_len) {
                    self.mode = Mode::Id;
                    return StepResult::Exit;
                }
                self.id = self.bits_get(self.id_len);
                self.bits_drop(self.id_len);
            }
            self.mode = self.id_table[self.id as usize].clone();
            StepResult::Continue
        }

        fn run_low_entropy(&mut self) -> StepResult {
            if !self.bits_ask(1) {
                return StepResult::Exit;
            }
            self.id = self.bits_get(1);
            self.bits_drop(1);
            self.mode = Mode::LowEntropyRef;
            StepResult::Continue
        }

        fn run_low_entropy_ref(&mut self) -> StepResult {
            if self.reff != 0 && !self.copysample() {
                return StepResult::Exit;
            }
            if self.id == 1 {
                self.mode = Mode::SE;
            } else {
                self.mode = Mode::ZeroBlock;
            }
            StepResult::Continue
        }

        fn run_zero_block(&mut self) -> StepResult {
            if !self.fs_ask() {
                return StepResult::Exit;
            }
            let mut zero_blocks = self.fs + 1;
            self.fs_drop();

            if zero_blocks == ROS {
                let b = self.rsi_used_size() as i32 / self.block_size as i32;
                zero_blocks = std::cmp::min(self.rsi as i32 - b, 64 - (b % 64)) as u32;
            } else if zero_blocks > ROS {
                zero_blocks -= 1;
            }

            let reff = self.reff as u32;
            let zero_samples = (zero_blocks * self.block_size - reff) as usize;

            if (self.rsi_size - self.rsi_used_size()) < zero_samples {
                return StepResult::Error(format!(
                    "AEC: not enough RSI buffer space for {} zero samples \
                     (size={}, used={})",
                    zero_samples,
                    self.rsi_size,
                    self.rsi_used_size()
                ));
            }

            let zero_bytes = zero_samples * self.bytes_per_sample;
            if self.avail_out >= zero_bytes {
                for _ in 0..zero_samples {
                    self.rsi_buffer[self.rsip] = 0;
                    self.rsip += 1;
                    self.avail_out -= self.bytes_per_sample;
                }
                self.mode = Mode::NextCds;
            } else {
                self.sample_counter = zero_samples as u32;
                self.mode = Mode::ZeroOutput;
            }
            StepResult::Continue
        }

        fn run_zero_output(&mut self) -> StepResult {
            loop {
                if self.avail_out < self.bytes_per_sample {
                    return StepResult::Exit;
                }
                self.put_sample(0);
                self.sample_counter -= 1;
                if self.sample_counter == 0 {
                    break;
                }
            }
            self.mode = Mode::NextCds;
            StepResult::Continue
        }

        fn run_se(&mut self) -> StepResult {
            if self.buffer_space() {
                let mut i = self.reff as u32;
                while i < self.block_size {
                    let m = self.direct_get_fs();
                    if m > SE_TABLE_SIZE as u32 {
                        return StepResult::Error(format!(
                            "AEC: SE table index out of bounds: {}",
                            m
                        ));
                    }
                    let d1 = (m as i32) - self.se_table[(2 * m + 1) as usize];
                    if (i & 1) == 0 {
                        self.put_sample_signed(self.se_table[(2 * m) as usize] - d1);
                        i += 1;
                    }
                    self.put_sample_signed(d1);
                    i += 1;
                }
                self.mode = Mode::NextCds;
            } else {
                self.sample_counter = self.reff as u32;
                self.mode = Mode::SEIncremental;
            }
            StepResult::Continue
        }

        fn run_se_decode(&mut self) -> StepResult {
            while self.sample_counter < self.block_size {
                if !self.fs_ask() {
                    return StepResult::Exit;
                }
                let m = self.fs as i32;
                if m > SE_TABLE_SIZE as i32 {
                    return StepResult::Error(format!("AEC: SE table index out of bounds: {}", m));
                }
                let d1 = m - self.se_table[(2 * m + 1) as usize];
                if (self.sample_counter & 1) == 0 {
                    if self.avail_out < self.bytes_per_sample {
                        return StepResult::Exit;
                    }
                    self.put_sample_signed(self.se_table[(2 * m) as usize] - d1);
                    self.sample_counter += 1;
                }
                if self.avail_out < self.bytes_per_sample {
                    return StepResult::Exit;
                }
                self.put_sample_signed(d1);
                self.sample_counter += 1;
                self.fs_drop();
            }
            self.mode = Mode::NextCds;
            StepResult::Continue
        }

        fn run_uncomp(&mut self) -> StepResult {
            if self.buffer_space() {
                for _ in 0..self.block_size {
                    self.rsi_buffer[self.rsip] = self.direct_get(self.bits_per_sample);
                    self.rsip += 1;
                }
                self.avail_out -= self.out_blklen;
                self.mode = Mode::NextCds;
            } else {
                self.sample_counter = self.block_size;
                self.mode = Mode::UncompCopy;
            }
            StepResult::Continue
        }

        fn run_uncomp_copy(&mut self) -> StepResult {
            loop {
                if !self.copysample() {
                    return StepResult::Exit;
                }
                self.sample_counter -= 1;
                if self.sample_counter == 0 {
                    break;
                }
            }
            self.mode = Mode::NextCds;
            StepResult::Continue
        }

        fn run_split(&mut self) -> StepResult {
            if self.buffer_space() {
                let k = (self.id as i32) - 1;
                let _binary_part = ((k as usize) * self.encoded_block_size as usize) / 8 + 9;

                if self.reff != 0 {
                    self.rsi_buffer[self.rsip] = self.direct_get(self.bits_per_sample);
                    self.rsip += 1;
                }

                for i in 0..self.encoded_block_size {
                    self.rsi_buffer[self.rsip + i as usize] = self.direct_get_fs() << k;
                }

                if k != 0 {
                    for _ in 0..self.encoded_block_size {
                        self.rsi_buffer[self.rsip] += self.direct_get(k as usize);
                        self.rsip += 1;
                    }
                } else {
                    self.rsip += self.encoded_block_size as usize;
                }

                self.avail_out -= self.out_blklen;
                self.mode = Mode::NextCds;
            } else {
                if self.reff != 0 && !self.copysample() {
                    return StepResult::Exit;
                }
                self.sample_counter = 0;
                self.mode = Mode::SplitFs;
            }
            StepResult::Continue
        }

        fn run_split_fs(&mut self) -> StepResult {
            let k = self.id - 1;
            loop {
                if !self.fs_ask() {
                    return StepResult::Exit;
                }
                self.rsi_buffer[self.rsip + self.sample_counter as usize] = self.fs << k;
                self.fs_drop();
                self.sample_counter += 1;
                if self.sample_counter >= self.encoded_block_size {
                    break;
                }
            }
            self.sample_counter = 0;
            self.mode = Mode::SplitOutput;
            StepResult::Continue
        }

        fn run_split_output(&mut self) -> StepResult {
            let k = self.id - 1;
            loop {
                if !self.bits_ask(k as usize) || self.avail_out < self.bytes_per_sample {
                    return StepResult::Exit;
                }
                if k != 0 {
                    self.rsi_buffer[self.rsip] += self.bits_get(k as usize);
                    self.rsip += 1;
                } else {
                    self.rsip += 1;
                }
                self.avail_out -= self.bytes_per_sample;
                self.bits_drop(k as usize);
                self.sample_counter += 1;
                if self.sample_counter >= self.encoded_block_size {
                    break;
                }
            }
            self.mode = Mode::NextCds;
            StepResult::Continue
        }

        fn run_next_cds(&mut self) -> StepResult {
            if self.rsi_size == self.rsi_used_size() {
                self.flush_kind();
                self.flush_start = 0;
                self.rsip = 0;

                if self.pp {
                    self.reff = 1;
                    self.encoded_block_size = self.block_size - 1;
                }

                if self.flags & AEC_PAD_RSI != 0 {
                    self.bitp -= self.bitp % 8;
                }
            } else {
                self.reff = 0;
                self.encoded_block_size = self.block_size;
            }

            self.run_id()
        }
    }

    /// Build the second-extension lookup table.
    pub(super) fn create_se_table() -> [i32; 2 * (SE_TABLE_SIZE + 1)] {
        let mut table = [0i32; 2 * (SE_TABLE_SIZE + 1)];
        let mut k: i32 = 0;
        for i in 0..13 {
            let ms = k;
            for _ in 0..=i {
                let kk = k as usize;
                table[2 * kk] = i;
                table[2 * kk + 1] = ms;
                k += 1;
            }
        }
        table
    }

    /// Modify AEC flags for GRIB2 CCSDS usage.
    ///
    /// The GRIB2 compression options mask has different bit assignments than libaec:
    ///   GRIB2 bit 5 (value 0x20): use AEC restricted mode
    ///   GRIB2 bit 4 (value 0x10): use AEC padding
    ///   GRIB2 remainder maps to preprocessing, signed, etc.
    ///
    /// For the pure-Rust decoder we remap to the internal flag format and ensure
    /// endianness is set correctly (native LE on x86, no 3-byte mode).
    pub(super) fn modify_flags(flags_byte: u8) -> u8 {
        let mut f = flags_byte;
        // Clear 3-byte flag (not relevant for our output)
        f &= !AEC_DATA_3BYTE;
        // Set MSB/LSB based on native platform endianness
        if cfg!(target_endian = "big") {
            f |= AEC_DATA_MSB;
        } else {
            f &= !AEC_DATA_MSB;
        }
        f
    }

    /// Convert raw decoded output bytes to integer sample values.
    pub(super) fn bytes_to_values(
        bytes: &[u8],
        bytes_per_sample: usize,
        big_endian: bool,
    ) -> Vec<i64> {
        let num_samples = bytes.len() / bytes_per_sample;
        let mut values = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let s = i * bytes_per_sample;
            let val: i64 = match bytes_per_sample {
                1 => bytes[s] as i64,
                2 => {
                    if big_endian {
                        u16::from_be_bytes([bytes[s], bytes[s + 1]]) as i64
                    } else {
                        u16::from_le_bytes([bytes[s], bytes[s + 1]]) as i64
                    }
                }
                3 => {
                    if big_endian {
                        ((bytes[s] as u32) << 16 | (bytes[s + 1] as u32) << 8 | bytes[s + 2] as u32)
                            as i64
                    } else {
                        (bytes[s] as u32 | (bytes[s + 1] as u32) << 8 | (bytes[s + 2] as u32) << 16)
                            as i64
                    }
                }
                4 => {
                    if big_endian {
                        u32::from_be_bytes([bytes[s], bytes[s + 1], bytes[s + 2], bytes[s + 3]])
                            as i64
                    } else {
                        u32::from_le_bytes([bytes[s], bytes[s + 1], bytes[s + 2], bytes[s + 3]])
                            as i64
                    }
                }
                _ => 0,
            };
            values.push(val);
        }

        values
    }

    /// Main entry point: decode CCSDS/AEC compressed data to integer sample values.
    pub fn decode(
        data: &[u8],
        block_size: u32,
        rsi: u32,
        flags_byte: u8,
        bits_per_sample: usize,
        avail_out: usize,
    ) -> Result<Vec<i64>, String> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        if bits_per_sample == 0 {
            return Ok(Vec::new());
        }
        if bits_per_sample > 32 {
            return Err(format!(
                "AEC: bits_per_sample {} exceeds maximum of 32",
                bits_per_sample
            ));
        }

        let flags = modify_flags(flags_byte);
        let big_endian = flags & AEC_DATA_MSB != 0;

        let mut state = State::new(bits_per_sample, block_size, rsi, flags, avail_out, data)?;

        // The decoder's internal bytes_per_sample (used for output format).
        // This is what the decoder uses when writing samples via put_bytes.
        let decoder_bps = state.bytes_per_sample;

        // Run the state machine until completion
        loop {
            match state.run() {
                StepResult::Continue => continue,
                StepResult::Exit => break,
                StepResult::Error(msg) => {
                    return Err(format!("AEC decode error: {}", msg));
                }
            }
        }

        // Final flush for any remaining RSI data
        state.flush_kind();

        // Convert output bytes to integer values using the decoder's
        // bytes_per_sample (not ceil(bps/8)), since the decoder may use
        // 4 bytes for 24-bit data when AEC_DATA_3BYTE is cleared.
        let values = bytes_to_values(&state.next_out, decoder_bps, big_endian);
        Ok(values)
    }
}

/// Template 5.61 / 7.61: Simple packing with logarithm pre-processing.
///
/// Values were stored as log10(value + 1) before simple packing.
/// We unpack using simple packing, then reverse the transform: value = 10^x - 1.
fn unpack_simple_log(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    let packed = unpack_simple(data, dr)?;
    Ok(packed.iter().map(|&v| 10f64.powf(v) - 1.0).collect())
}

/// Template 5.200 / 7.200: Run Length Encoding (NCEP local extension).
///
/// Used for categorical/discrete data (e.g., precipitation type, soil type).
/// Data is encoded as run-length encoded values where each entry consists of
/// a level value and a run length packed into `bits_per_value` bits.
///
/// The encoding uses a variable-length scheme where the most significant bit(s)
/// determine the level value and the remaining bits encode the run count.
fn unpack_rle(data: &[u8], dr: &DataRepresentation, num_points: usize) -> Result<Vec<f64>, String> {
    if data.is_empty() {
        return Ok(vec![0.0; num_points]);
    }

    let bpv = dr.bits_per_value as usize;
    if bpv == 0 {
        return Ok(vec![dr.reference_value as f64; num_points]);
    }

    let mut reader = BitReader::new(data);
    let mut result = Vec::with_capacity(num_points);

    // NCEP RLE encoding for template 5.200:
    // Read alternating (value, count) pairs, each `bpv` bits wide.
    // The value is the category/level, count is how many consecutive
    // grid points have that value.
    while result.len() < num_points && reader.remaining_bits() >= bpv * 2 {
        let value = reader.read_bits(bpv);
        let count = reader.read_bits(bpv) as usize;
        let count = count.max(1); // ensure at least 1

        let scaled_value = dr.reference_value as f64
            + value as f64
                * 2.0_f64.powi(dr.binary_scale as i32)
                * 10.0_f64.powi(-(dr.decimal_scale as i32));

        let remaining = num_points - result.len();
        let actual_count = count.min(remaining);
        for _ in 0..actual_count {
            result.push(scaled_value);
        }
    }

    // If we haven't filled all points, pad with reference value
    while result.len() < num_points {
        result.push(dr.reference_value as f64);
    }

    Ok(result)
}

/// Template 5.50 / 7.50: Spectral Data, Simple Packing.
///
/// Used by global spectral models. The data contains spectral coefficients
/// packed using simple packing. The first value is the real part of the (0,0)
/// coefficient stored as an IEEE f32, followed by simple-packed remaining coefficients.
fn unpack_spectral_simple(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    if data.len() < 4 {
        return Ok(Vec::new());
    }

    // The first 4 bytes are the real value of the (0,0) coefficient as IEEE f32
    let coeff_00 = f32::from_be_bytes([data[0], data[1], data[2], data[3]]) as f64;

    // Remaining data is simple-packed
    let remaining = &data[4..];
    let bpv = dr.bits_per_value as usize;
    if bpv == 0 || remaining.is_empty() {
        return Ok(vec![coeff_00]);
    }

    let total_bits = remaining.len() * 8;
    let n = total_bits / bpv;
    let mut reader = BitReader::new(remaining);
    let mut raw = Vec::with_capacity(n);
    for _ in 0..n {
        raw.push(reader.read_bits(bpv) as i64);
    }

    let mut values = Vec::with_capacity(n + 1);
    values.push(coeff_00);
    values.extend(apply_scaling(&raw, dr));
    Ok(values)
}

/// Template 5.51 / 7.51: Spectral Data, Complex Packing.
///
/// Similar to spectral simple but uses complex packing for the coefficients
/// after the (0,0) term. This is rarely encountered in gridded output.
fn unpack_spectral_complex(data: &[u8], dr: &DataRepresentation) -> Result<Vec<f64>, String> {
    if data.len() < 4 {
        return Ok(Vec::new());
    }

    // The first 4 bytes are the real value of the (0,0) coefficient as IEEE f32
    let coeff_00 = f32::from_be_bytes([data[0], data[1], data[2], data[3]]) as f64;

    // Remaining data uses complex packing
    let remaining = &data[4..];
    if remaining.is_empty() {
        return Ok(vec![coeff_00]);
    }

    let complex_values = unpack_complex(remaining, dr)?;
    let mut values = Vec::with_capacity(complex_values.len() + 1);
    values.push(coeff_00);
    values.extend(complex_values);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- BitReader tests ----

    #[test]
    fn test_bitreader_read_zero_bits() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(0), 0);
        assert_eq!(reader.remaining_bits(), 8);
    }

    #[test]
    fn test_bitreader_read_single_bit() {
        let data = [0b1010_0000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(1), 1);
        assert_eq!(reader.read_bits(1), 0);
        assert_eq!(reader.read_bits(1), 1);
        assert_eq!(reader.read_bits(1), 0);
    }

    #[test]
    fn test_bitreader_read_full_byte() {
        let data = [0xA5]; // 1010_0101
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(8), 0xA5);
    }

    #[test]
    fn test_bitreader_read_across_byte_boundary() {
        let data = [0xFF, 0x00]; // 11111111 00000000
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(4), 0xF); // 1111
        assert_eq!(reader.read_bits(8), 0xF0); // 1111_0000
        assert_eq!(reader.read_bits(4), 0x00); // 0000
    }

    #[test]
    fn test_bitreader_read_16_bits() {
        let data = [0xAB, 0xCD];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(16), 0xABCD);
    }

    #[test]
    fn test_bitreader_read_beyond_data() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        // Reading past the end should pad with zeros
        let val = reader.read_bits(16);
        assert_eq!(val, 0xFF00);
    }

    #[test]
    fn test_bitreader_remaining_bits() {
        let data = [0xFF, 0x00];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.remaining_bits(), 16);
        reader.read_bits(5);
        assert_eq!(reader.remaining_bits(), 11);
        reader.read_bits(11);
        assert_eq!(reader.remaining_bits(), 0);
    }

    #[test]
    fn test_bitreader_remaining_bits_empty() {
        let data: [u8; 0] = [];
        let reader = BitReader::new(&data);
        assert_eq!(reader.remaining_bits(), 0);
    }

    #[test]
    fn test_bitreader_align_to_byte() {
        let data = [0xFF, 0xAA];
        let mut reader = BitReader::new(&data);
        reader.read_bits(3);
        assert_eq!(reader.remaining_bits(), 13);
        reader.align_to_byte();
        assert_eq!(reader.remaining_bits(), 8);
        assert_eq!(reader.read_bits(8), 0xAA);
    }

    #[test]
    fn test_bitreader_align_already_aligned() {
        let data = [0xFF, 0xAA];
        let mut reader = BitReader::new(&data);
        reader.read_bits(8);
        reader.align_to_byte(); // already aligned, should be no-op
        assert_eq!(reader.remaining_bits(), 8);
    }

    #[test]
    fn test_bitreader_read_signed_zero_bits() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_signed_bits(0), 0);
    }

    #[test]
    fn test_bitreader_read_signed_one_bit() {
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        // 1 bit: sign-magnitude with just sign bit returns 0
        assert_eq!(reader.read_signed_bits(1), 0);
    }

    #[test]
    fn test_bitreader_read_signed_positive() {
        // 0_0000101 = +5 in 8-bit sign-magnitude
        let data = [0b0_0000101];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_signed_bits(8), 5);
    }

    #[test]
    fn test_bitreader_read_signed_negative() {
        // 1_0000101 = -5 in 8-bit sign-magnitude
        let data = [0b1_0000101];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_signed_bits(8), -5);
    }

    #[test]
    fn test_bitreader_read_signed_negative_large() {
        // 1_1111111 = -127 in 8-bit sign-magnitude
        let data = [0xFF];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_signed_bits(8), -127);
    }

    #[test]
    fn test_bitreader_read_signed_positive_zero() {
        // 0_0000000 = +0
        let data = [0x00];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_signed_bits(8), 0);
    }

    #[test]
    fn test_bitreader_sequential_reads() {
        // Test reading various bit widths sequentially
        let data = [0b11010110, 0b10000000];
        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(2), 0b11); // 3
        assert_eq!(reader.read_bits(3), 0b010); // 2
        assert_eq!(reader.read_bits(3), 0b110); // 6
        assert_eq!(reader.read_bits(1), 1);
        assert_eq!(reader.remaining_bits(), 7);
    }

    // ---- apply_scaling tests ----

    #[test]
    fn test_apply_scaling_identity() {
        let dr = DataRepresentation {
            template: 0,
            reference_value: 0.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 8,
            ..make_default_dr()
        };
        let raw = vec![1, 2, 3, 4, 5];
        let result = apply_scaling(&raw, &dr);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - (i as f64 + 1.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_apply_scaling_with_reference() {
        let dr = DataRepresentation {
            template: 0,
            reference_value: 100.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 8,
            ..make_default_dr()
        };
        let raw = vec![0, 1, 2];
        let result = apply_scaling(&raw, &dr);
        assert!((result[0] - 100.0).abs() < 1e-10);
        assert!((result[1] - 101.0).abs() < 1e-10);
        assert!((result[2] - 102.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_scaling_with_binary_scale() {
        let dr = DataRepresentation {
            template: 0,
            reference_value: 0.0,
            binary_scale: 1, // multiply by 2
            decimal_scale: 0,
            bits_per_value: 8,
            ..make_default_dr()
        };
        let raw = vec![1, 2, 3];
        let result = apply_scaling(&raw, &dr);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
        assert!((result[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_scaling_with_decimal_scale() {
        let dr = DataRepresentation {
            template: 0,
            reference_value: 0.0,
            binary_scale: 0,
            decimal_scale: 1, // divide by 10
            bits_per_value: 8,
            ..make_default_dr()
        };
        let raw = vec![10, 20, 30];
        let result = apply_scaling(&raw, &dr);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }

    // ---- complex-packing missing-value management ----

    /// Section 5 for Template 5.2 over the four-point field the tests
    /// below encode: reference 0.0, no scaling, 4-bit values, two
    /// two-value groups of width 4.
    fn section5_complex(mode: u8, template: u16) -> Vec<u8> {
        let mut sec = vec![0u8; if template == 3 { 49 } else { 47 }];
        let len = sec.len() as u32;
        sec[0..4].copy_from_slice(&len.to_be_bytes());
        sec[4] = 5;
        sec[5..9].copy_from_slice(&if template == 3 { 5u32 } else { 4u32 }.to_be_bytes());
        sec[9..11].copy_from_slice(&template.to_be_bytes());
        sec[11..15].copy_from_slice(&0.0f32.to_be_bytes()); // reference value
        sec[15..17].copy_from_slice(&0i16.to_be_bytes()); // binary scale
        sec[17..19].copy_from_slice(&0i16.to_be_bytes()); // decimal scale
        sec[19] = 4; // bits per value
        sec[20] = 0; // original field type: floating point
        sec[21] = 1; // group splitting: general
        sec[22] = mode; // MISSING VALUE MANAGEMENT, Code Table 5.5
        sec[23..27].copy_from_slice(&9999.0f32.to_be_bytes()); // primary substitute
        sec[27..31].copy_from_slice(&9998.0f32.to_be_bytes()); // secondary substitute
        if template == 3 {
            sec[31..35].copy_from_slice(&1u32.to_be_bytes()); // one group
            sec[42..46].copy_from_slice(&5u32.to_be_bytes()); // last group length
            sec[47] = 1; // first-order spatial differencing
            sec[48] = 2; // two extra descriptor bytes
        } else {
            sec[31..35].copy_from_slice(&2u32.to_be_bytes()); // two groups
            sec[42..46].copy_from_slice(&2u32.to_be_bytes()); // last group length
        }
        sec[35] = 0; // group width reference
        sec[36] = 8; // group width bits
        sec[37..41].copy_from_slice(&0u32.to_be_bytes()); // group length reference
        sec[41] = 1; // group length increment
        sec[46] = 8; // group length bits
        sec
    }

    /// Wrap one Section 5 and one Section 7 payload in a complete GRIB2
    /// envelope.  The parser enforces exact envelope coverage, the
    /// 1/3/4/5/6/7 state machine and Section-3/5 cardinality, so a
    /// fixture has to be a real message rather than a fragment.
    fn synthetic_message(points: u32, section5: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut section1 = vec![0u8; 21];
        section1[0..4].copy_from_slice(&21u32.to_be_bytes());
        section1[4] = 1;
        section1[9] = 2; // master table version
        section1[12..14].copy_from_slice(&2026u16.to_be_bytes());
        section1[14] = 8; // month
        section1[15] = 16; // day

        let mut section3 = vec![0u8; 72];
        section3[0..4].copy_from_slice(&72u32.to_be_bytes());
        section3[4] = 3;
        section3[6..10].copy_from_slice(&points.to_be_bytes());
        section3[12..14].copy_from_slice(&0u16.to_be_bytes()); // template 3.0
        section3[14] = 6; // spherical earth
        section3[30..34].copy_from_slice(&points.to_be_bytes()); // Ni
        section3[34..38].copy_from_slice(&1u32.to_be_bytes()); // Nj
        section3[63..67].copy_from_slice(&1_000_000u32.to_be_bytes()); // Di
        section3[67..71].copy_from_slice(&1_000_000u32.to_be_bytes()); // Dj

        let mut section4 = vec![0u8; 34];
        section4[0..4].copy_from_slice(&34u32.to_be_bytes());
        section4[4] = 4;
        section4[7..9].copy_from_slice(&0u16.to_be_bytes()); // template 4.0
        section4[22] = 1; // ground or water surface
        section4[28] = 255; // no second fixed surface

        let mut section6 = vec![0u8; 6];
        section6[0..4].copy_from_slice(&6u32.to_be_bytes());
        section6[4] = 6;
        section6[5] = 255; // no bitmap

        let mut section7 = vec![0u8; 5];
        section7[4] = 7;
        section7.extend_from_slice(payload);
        let section7_len = section7.len() as u32;
        section7[0..4].copy_from_slice(&section7_len.to_be_bytes());

        let body_len = section1.len()
            + section3.len()
            + section4.len()
            + section5.len()
            + section6.len()
            + section7.len();
        let total = (16 + body_len + 4) as u64;

        let mut message = Vec::with_capacity(total as usize);
        message.extend_from_slice(b"GRIB");
        message.extend_from_slice(&[0, 0, 0, 2]); // reserved, discipline 0, edition 2
        message.extend_from_slice(&total.to_be_bytes());
        message.extend_from_slice(&section1);
        message.extend_from_slice(&section3);
        message.extend_from_slice(&section4);
        message.extend_from_slice(section5);
        message.extend_from_slice(&section6);
        message.extend_from_slice(&section7);
        message.extend_from_slice(b"7777");
        message
    }

    /// Section 7 for the four-point Template 5.2 field: group
    /// references 3 and 5, both groups width 4 and length 2.  The
    /// all-ones nibble `F` is the primary missing marker mode 1 and 2
    /// reserve; `E` is the secondary marker mode 2 reserves.
    fn payload_complex(values: [u8; 4]) -> Vec<u8> {
        vec![
            0x35, // group references 3 and 5, four bits each
            0x04,
            0x04, // group widths
            0x02,
            0x02, // group lengths
            (values[0] << 4) | values[1],
            (values[2] << 4) | values[3],
        ]
    }

    /// Section 7 for the five-point Template 5.3 field: one group,
    /// width 4, length 5, first-order spatial differencing with initial
    /// value 100 and overall minimum 0.
    fn payload_complex_spatial(values: [u8; 5]) -> Vec<u8> {
        vec![
            0x00,
            0x64, // initial value 100, two bytes
            0x00,
            0x00, // overall minimum 0, sign-magnitude, two bytes
            0x00, // one group reference, four bits, byte aligned
            0x04, // group width
            0x05, // group length
            (values[0] << 4) | values[1],
            (values[2] << 4) | values[3],
            values[4] << 4,
        ]
    }

    fn decode_message(bytes: &[u8]) -> Vec<f64> {
        let file = crate::grib2::parser::Grib2File::from_bytes(bytes).expect("message parses");
        assert_eq!(file.messages.len(), 1);
        unpack_message(&file.messages[0]).expect("message unpacks")
    }

    fn assert_field(got: &[f64], want: &[Option<f64>]) {
        assert_eq!(got.len(), want.len(), "point count: {got:?}");
        for (index, (value, expected)) in got.iter().zip(want.iter()).enumerate() {
            match expected {
                None => assert!(value.is_nan(), "point {index} should be missing, got {value}"),
                Some(expected) => assert!(
                    (value - expected).abs() < 1e-9,
                    "point {index}: got {value}, want {expected}"
                ),
            }
        }
    }

    #[test]
    fn test_complex_packing_decodes_primary_missing_values_as_nan() {
        // Mode 1 reserves the all-ones pattern of each group's width as
        // the missing marker.  Before octet 23 was decoded, the two
        // marked cells came out as 3+15=18 and 5+15=20 -- plausible
        // numbers that are not data.
        let bytes = synthetic_message(4, &section5_complex(1, 2), &payload_complex([1, 15, 2, 15]));
        assert_field(
            &decode_message(&bytes),
            &[Some(4.0), None, Some(7.0), None],
        );
    }

    #[test]
    fn test_complex_packing_decodes_secondary_missing_values_as_nan() {
        // Mode 2 reserves all-ones AND all-ones-minus-one.  Mode 1 must
        // NOT treat 14 as missing, which is what separates the two.
        let secondary =
            synthetic_message(4, &section5_complex(2, 2), &payload_complex([1, 15, 14, 2]));
        assert_field(
            &decode_message(&secondary),
            &[Some(4.0), None, None, Some(7.0)],
        );

        let primary_only =
            synthetic_message(4, &section5_complex(1, 2), &payload_complex([1, 15, 14, 2]));
        assert_field(
            &decode_message(&primary_only),
            &[Some(4.0), None, Some(19.0), Some(7.0)],
        );
    }

    #[test]
    fn test_complex_packing_mode_zero_decodes_every_pattern_as_data() {
        // The regression guard on the fix: with no missing-value
        // management declared, an all-ones nibble is data and stays
        // data.
        let bytes = synthetic_message(4, &section5_complex(0, 2), &payload_complex([1, 15, 2, 15]));
        assert_field(
            &decode_message(&bytes),
            &[Some(4.0), Some(18.0), Some(7.0), Some(20.0)],
        );
    }

    #[test]
    fn test_zero_width_group_reference_carries_the_missing_marker() {
        // A constant group has no per-value bits, so its marker is the
        // all-ones pattern of bits_per_value applied to the group
        // reference itself (NCEP g2c comunpack.c takes the same branch).
        let dr = DataRepresentation {
            template: 2,
            bits_per_value: 4,
            missing_value_management: 1,
            num_groups: 2,
            group_width_bits: 8,
            group_length_bits: 8,
            group_length_inc: 1,
            last_group_length: 2,
            section5_num_data_points: 4,
            ..make_default_dr()
        };
        // references 15 (the marker) and 5; both groups width 0, length 2.
        let data = [0xF5, 0x00, 0x00, 0x02, 0x02];
        let values = unpack_complex(&data, &dr).expect("constant groups decode");
        assert_field(&values, &[None, None, Some(5.0), Some(5.0)]);
    }

    #[test]
    fn test_spatial_differencing_excludes_missing_cells_from_the_chain() {
        // The defect is worse than two wrong cells: with first-order
        // differencing every value AFTER a marker inherits it.  Decoded
        // as data the field reads 100, 115, 117, 120, 135; the true
        // field is 100, missing, 102, 105, missing.
        let bytes = synthetic_message(
            5,
            &section5_complex(1, 3),
            &payload_complex_spatial([0, 15, 2, 3, 15]),
        );
        assert_field(
            &decode_message(&bytes),
            &[Some(100.0), None, Some(102.0), Some(105.0), None],
        );
    }

    #[test]
    fn test_spatial_differencing_mode_zero_is_unchanged() {
        let bytes = synthetic_message(
            5,
            &section5_complex(0, 3),
            &payload_complex_spatial([0, 15, 2, 3, 15]),
        );
        assert_field(
            &decode_message(&bytes),
            &[
                Some(100.0),
                Some(115.0),
                Some(117.0),
                Some(120.0),
                Some(135.0),
            ],
        );
    }

    #[test]
    fn test_row_window_decode_marks_missing_cells_nan() {
        // The streaming row-band path decodes the same bytes through a
        // different loop, so it needs its own proof.
        let bytes = synthetic_message(
            5,
            &section5_complex(1, 3),
            &payload_complex_spatial([0, 15, 2, 3, 15]),
        );
        let file = crate::grib2::parser::Grib2File::from_bytes(&bytes).expect("message parses");
        let window = unpack_message_scan_normalized_row_window(&file.messages[0], 0, 1)
            .expect("row window decodes");
        assert_field(
            &window,
            &[Some(100.0), None, Some(102.0), Some(105.0), None],
        );
    }

    #[test]
    fn test_reserved_missing_value_management_is_refused() {
        // Code Table 5.5 defines 0, 1 and 2.  Everything else is
        // reserved: a decoder that guessed would be back to publishing
        // whatever the encoder reserved as physical data.
        for mode in [3u8, 7, 255] {
            let dr = DataRepresentation {
                template: 2,
                bits_per_value: 8,
                missing_value_management: mode,
                num_groups: 1,
                group_width_bits: 8,
                group_length_bits: 8,
                last_group_length: 1,
                section5_num_data_points: 1,
                ..make_default_dr()
            };
            let refusal = unpack_complex(&[0u8; 16], &dr).unwrap_err();
            assert!(
                refusal.contains(&format!("missing-value management {mode}")),
                "{refusal}"
            );
            let spatial = DataRepresentation {
                template: 3,
                spatial_diff_order: 2,
                spatial_diff_bytes: 2,
                ..dr.clone()
            };
            let refusal = unpack_complex_spatial(&[0u8; 16], &spatial).unwrap_err();
            assert!(
                refusal.contains(&format!("missing-value management {mode}")),
                "{refusal}"
            );
        }
    }

    #[test]
    fn test_missing_values_with_zero_bits_per_value_are_refused() {
        // bits_per_value 0 makes the only representable group reference
        // the marker itself, so the field is either all-missing or the
        // marker is meaningless.  Refuse instead of picking one.
        let dr = DataRepresentation {
            template: 2,
            bits_per_value: 0,
            missing_value_management: 1,
            num_groups: 1,
            group_width_bits: 8,
            group_length_bits: 8,
            last_group_length: 1,
            section5_num_data_points: 1,
            ..make_default_dr()
        };
        let refusal = unpack_complex(&[0u8; 16], &dr).unwrap_err();
        assert!(refusal.contains("bits_per_value 0"), "{refusal}");
    }

    // ---- unpack_simple tests ----

    #[test]
    fn test_unpack_simple_zero_bpv() {
        let dr = DataRepresentation {
            template: 0,
            reference_value: 42.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 0,
            original_field_type: 0,
            ..make_default_dr()
        };
        let data = [0u8; 1];
        let result = unpack_simple(&data, &dr).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_unpack_simple_8bit() {
        let dr = DataRepresentation {
            template: 0,
            reference_value: 0.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 8,
            ..make_default_dr()
        };
        let data = [10, 20, 30];
        let result = unpack_simple(&data, &dr).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 20.0).abs() < 1e-10);
        assert!((result[2] - 30.0).abs() < 1e-10);
    }

    // ---- unpack_ieee tests ----

    #[test]
    fn test_unpack_ieee_f32() {
        let dr = DataRepresentation {
            template: 4,
            bits_per_value: 32,
            ..make_default_dr()
        };
        let val: f32 = 3.14;
        let bytes = val.to_be_bytes();
        let result = unpack_ieee(&bytes, &dr).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_unpack_ieee_f64() {
        let dr = DataRepresentation {
            template: 4,
            bits_per_value: 64,
            ..make_default_dr()
        };
        let val: f64 = 2.71828;
        let bytes = val.to_be_bytes();
        let result = unpack_ieee(&bytes, &dr).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.71828).abs() < 1e-10);
    }

    #[test]
    fn test_unpack_ieee_empty() {
        let dr = DataRepresentation {
            template: 4,
            bits_per_value: 32,
            ..make_default_dr()
        };
        let result = unpack_ieee(&[], &dr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_unpack_ieee_unsupported_bpv() {
        let dr = DataRepresentation {
            template: 4,
            bits_per_value: 16,
            ..make_default_dr()
        };
        let data = [0u8; 4];
        let result = unpack_ieee(&data, &dr);
        assert!(result.is_err());
    }

    #[test]
    fn test_unpack_message_ieee_matches_declared_grid_point_count() {
        use crate::grib2::parser::{GridDefinition, ProductDefinition};

        let values = [1.0f32, 2.0];
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>();
        let msg = Grib2Message {
            discipline: 0,
            identification: crate::grib2::parser::Identification::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 4, 14)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                nx: 2,
                ny: 1,
                num_data_points: 2,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 4,
                bits_per_value: 32,
                section5_num_data_points: 2,
                ..make_default_dr()
            },
            bitmap: None,
            raw_data,
        };

        let unpacked = unpack_message(&msg).unwrap();
        assert_eq!(unpacked, vec![1.0, 2.0]);
    }

    #[test]
    fn test_stored_rect_of_a_simple_packed_grid_is_the_crop_of_the_whole_unpack() {
        use crate::grib2::parser::{GridDefinition, ProductDefinition};

        let msg = Grib2Message {
            discipline: 0,
            identification: crate::grib2::parser::Identification::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 9, 29)
                .unwrap()
                .and_hms_opt(12, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                nx: 4,
                ny: 3,
                num_data_points: 12,
                scan_mode: 0x00,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 0,
                bits_per_value: 8,
                section5_num_data_points: 12,
                ..make_default_dr()
            },
            bitmap: None,
            raw_data: (1..=12).collect(),
        };
        let full = unpack_message(&msg).unwrap();
        let rect = StoredRect { rows: (1, 3), columns: (1, 3) };
        let window = unpack_message_stored_rect(&msg, rect).unwrap();
        assert_eq!(window, vec![full[5], full[6], full[9], full[10]]);
        // A rectangle that is not inside the grid is refused, not clipped.
        for bad in [
            StoredRect { rows: (0, 4), columns: (0, 1) },
            StoredRect { rows: (2, 2), columns: (0, 1) },
            StoredRect { rows: (0, 1), columns: (3, 5) },
        ] {
            assert!(unpack_message_stored_rect(&msg, bad).is_err(), "{bad:?}");
        }
    }

    /// SOC, SIZ (38 bytes of body), COD with `transform`, optional COC,
    /// then SOT: the main header shape GRIB2 JPEG2000 encoders write.
    fn j2k_main_header(transform: u8, coc: bool) -> Vec<u8> {
        let mut bytes = vec![0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x29];
        bytes.resize(bytes.len() + 0x29 - 2, 0u8);
        bytes.extend([0xFF, 0x52, 0x00, 0x0C, 0x00, 0x00, 0x00, 0x01, 0x00, 0x05, 0x04, 0x04, 0x00, transform]);
        if coc {
            bytes.extend([0xFF, 0x53, 0x00, 0x09, 0x00, 0x00, 0x05, 0x04, 0x04, 0x00, 0x00]);
        }
        bytes.extend([0xFF, 0x90, 0x00, 0x0A, 0, 0, 0, 0, 0, 0, 0, 0]);
        bytes
    }

    #[test]
    fn test_only_a_reversible_main_header_is_trusted_for_an_area_decode() {
        assert!(jpeg2000_main_header_is_reversible(&j2k_main_header(1, false)));
        // The irreversible 9/7 wavelet reconstructs in floating point over
        // another span, so an area of it is not the whole decode's crop.
        assert!(!jpeg2000_main_header_is_reversible(&j2k_main_header(0, false)));
        // A COC can restate the transform; it is not read, so not trusted.
        assert!(!jpeg2000_main_header_is_reversible(&j2k_main_header(1, true)));
        let mut truncated = j2k_main_header(1, false);
        truncated.truncate(50);
        assert!(!jpeg2000_main_header_is_reversible(&truncated));
        assert!(!jpeg2000_main_header_is_reversible(&[0x00, 0x4F, 0xFF, 0x51]));
    }

    #[test]
    fn test_row_window_simple_matches_normalized_crop_with_scan_flip() {
        use crate::grib2::parser::{GridDefinition, ProductDefinition};

        let msg = Grib2Message {
            discipline: 0,
            identification: crate::grib2::parser::Identification::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 4, 14)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                nx: 3,
                ny: 2,
                num_data_points: 6,
                scan_mode: 0x40,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 0,
                bits_per_value: 8,
                section5_num_data_points: 6,
                ..make_default_dr()
            },
            bitmap: None,
            raw_data: vec![1, 2, 3, 4, 5, 6],
        };

        let full = unpack_message_normalized(&msg).unwrap();
        let window = unpack_message_scan_normalized_row_window(&msg, 0, 1).unwrap();
        assert_eq!(full[0..3], window);
    }

    #[test]
    fn test_row_window_simple_preserves_bitmap_nan_cells() {
        use crate::grib2::parser::{GridDefinition, ProductDefinition};

        let msg = Grib2Message {
            discipline: 0,
            identification: crate::grib2::parser::Identification::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 4, 14)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                nx: 3,
                ny: 2,
                num_data_points: 6,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 0,
                bits_per_value: 8,
                section5_num_data_points: 4,
                ..make_default_dr()
            },
            bitmap: Some(vec![true, false, true, true, true, false]),
            raw_data: vec![10, 20, 30, 40],
        };

        let full = unpack_message_normalized(&msg).unwrap();
        let window = unpack_message_scan_normalized_row_window(&msg, 0, 2).unwrap();
        assert_eq!(full.len(), window.len());
        for (expected, actual) in full.iter().zip(window.iter()) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(*expected, *actual);
            }
        }
    }

    #[test]
    fn test_row_window_simple_refuses_a_short_section7() {
        use crate::grib2::parser::{GridDefinition, ProductDefinition};

        // Negative control for the window path's Section 5/7 agreement.  The
        // unchecked reader synthesizes zero bits past the payload, so before
        // this gate the missing tail decoded as the reference value and the
        // renderer reported a tile it never read.
        let msg = Grib2Message {
            discipline: 0,
            identification: crate::grib2::parser::Identification::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 4, 14)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                nx: 3,
                ny: 2,
                num_data_points: 6,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 0,
                bits_per_value: 8,
                section5_num_data_points: 6,
                ..make_default_dr()
            },
            bitmap: None,
            raw_data: vec![1, 2, 3, 4, 5],
        };

        let full = unpack_message(&msg).unwrap_err().to_string();
        assert!(full.contains("Section 5 requires 6"), "{full}");
        let window = unpack_message_scan_normalized_row_window(&msg, 0, 2)
            .unwrap_err()
            .to_string();
        assert!(window.contains("Section 5 requires 6"), "{window}");
    }

    #[test]
    fn test_row_window_simple_refuses_a_bitmap_that_outruns_section7() {
        use crate::grib2::parser::{GridDefinition, ProductDefinition};

        // The declared byte count is satisfied, so only the checked bit read
        // can catch this: the bitmap marks six present cells while Section 5
        // declares four values.
        let msg = Grib2Message {
            discipline: 0,
            identification: crate::grib2::parser::Identification::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 4, 14)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                nx: 3,
                ny: 2,
                num_data_points: 6,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 0,
                bits_per_value: 8,
                section5_num_data_points: 4,
                ..make_default_dr()
            },
            bitmap: Some(vec![true; 6]),
            raw_data: vec![10, 20, 30, 40],
        };

        assert!(unpack_message(&msg).is_err());
        let window = unpack_message_scan_normalized_row_window(&msg, 0, 2)
            .unwrap_err()
            .to_string();
        assert!(window.contains("truncated packed payload"), "{window}");
    }

    #[test]
    fn test_row_window_complex_spatial_refuses_a_truncated_group_payload() {
        use crate::grib2::parser::GridDefinition;

        // Same message as test_row_window_complex_missing_matches_full_decode
        // with the last packed byte removed: the group table still declares
        // six 3-bit values but only 16 bits of payload remain.
        let mut raw_data = vec![10, 12, 0];
        let mut packed = pack_unsigned(&[0, 0, 7, 1, 6, 2], 3);
        packed.pop();
        raw_data.extend(packed);
        let msg = message_with(
            GridDefinition {
                nx: 3,
                ny: 2,
                num_data_points: 6,
                scan_mode: 0x40,
                ..GridDefinition::default()
            },
            spatial_missing_dr(2, 6),
            raw_data,
        );

        assert!(unpack_message(&msg).is_err());
        let window = unpack_message_scan_normalized_row_window(&msg, 0, 2)
            .unwrap_err()
            .to_string();
        assert!(window.contains("truncated packed payload"), "{window}");
    }

    #[test]
    fn test_row_window_complex_spatial_matches_normalized_crop() {
        use crate::grib2::parser::{GridDefinition, ProductDefinition};

        let msg = Grib2Message {
            discipline: 0,
            identification: crate::grib2::parser::Identification::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 4, 14)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid: GridDefinition {
                nx: 3,
                ny: 2,
                num_data_points: 6,
                scan_mode: 0x40,
                ..GridDefinition::default()
            },
            product: ProductDefinition::default(),
            data_rep: DataRepresentation {
                template: 3,
                bits_per_value: 0,
                num_groups: 1,
                group_width_ref: 0,
                group_width_bits: 0,
                group_length_ref: 0,
                group_length_inc: 1,
                last_group_length: 6,
                group_length_bits: 0,
                spatial_diff_order: 2,
                spatial_diff_bytes: 1,
                section5_num_data_points: 6,
                ..make_default_dr()
            },
            bitmap: None,
            raw_data: vec![1, 3, 0],
        };

        let full = unpack_message_normalized(&msg).unwrap();
        let window = unpack_message_scan_normalized_row_window(&msg, 0, 1).unwrap();
        assert_eq!(full[0..3], window);
    }

    // ---- ported donor coverage: complex/spatial missing-value family,
    // ---- IEEE padding, zero-group constants (see lib.rs convergence notes)

    fn spatial_missing_dr(management: u8, count: u32) -> DataRepresentation {
        DataRepresentation {
            template: 3,
            bits_per_value: 0,
            missing_value_management: management,
            num_groups: 1,
            group_width_ref: 3,
            group_width_bits: 0,
            group_length_ref: 0,
            group_length_inc: 1,
            last_group_length: count,
            group_length_bits: 0,
            spatial_diff_order: 2,
            spatial_diff_bytes: 1,
            section5_num_data_points: count,
            ..make_default_dr()
        }
    }

    fn pack_unsigned(values: &[u64], width: usize) -> Vec<u8> {
        let mut packed = vec![0_u8; (values.len() * width).div_ceil(8)];
        let mut bit = 0usize;
        for value in values {
            for shift in (0..width).rev() {
                if value & (1_u64 << shift) != 0 {
                    packed[bit / 8] |= 1 << (7 - bit % 8);
                }
                bit += 1;
            }
        }
        packed
    }

    fn assert_same_values(expected: &[f64], actual: &[f64]) {
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(expected.to_bits(), actual.to_bits());
            }
        }
    }

    fn message_with(
        grid: crate::grib2::parser::GridDefinition,
        data_rep: DataRepresentation,
        raw_data: Vec<u8>,
    ) -> Grib2Message {
        Grib2Message {
            discipline: 0,
            identification: crate::grib2::parser::Identification::default(),
            reference_time: chrono::NaiveDate::from_ymd_opt(2026, 8, 17)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            grid,
            product: crate::grib2::parser::ProductDefinition::default(),
            data_rep,
            bitmap: None,
            raw_data,
        }
    }

    #[test]
    fn test_unpack_message_trims_ieee_padding_to_grid_point_count() {
        // Convergence conflict C2: the Section 5 point count is the authority
        // and trailing Section 7 padding is tolerated.  The donor asserted the
        // same trim with `section5_num_data_points` left at 0; this crate
        // refuses a zero declaration outright, so the fixture states it.
        use crate::grib2::parser::GridDefinition;

        let values = [1.0f32, 2.0, 3.0];
        let raw_data = values
            .iter()
            .flat_map(|value| value.to_be_bytes())
            .collect::<Vec<_>>();
        let msg = message_with(
            GridDefinition {
                nx: 2,
                ny: 1,
                num_data_points: 2,
                ..GridDefinition::default()
            },
            DataRepresentation {
                template: 4,
                bits_per_value: 32,
                section5_num_data_points: 2,
                ..make_default_dr()
            },
            raw_data,
        );

        let unpacked = unpack_message(&msg).unwrap();
        assert_eq!(unpacked, vec![1.0, 2.0]);
    }

    #[test]
    fn test_unpack_message_refuses_short_ieee_payload() {
        // The other half of C2: bytes short of the declared count are refused,
        // because a short IEEE payload silently shortens the field.
        use crate::grib2::parser::GridDefinition;

        let raw_data = 1.0f32.to_be_bytes().to_vec();
        let msg = message_with(
            GridDefinition {
                nx: 2,
                ny: 1,
                num_data_points: 2,
                ..GridDefinition::default()
            },
            DataRepresentation {
                template: 4,
                bits_per_value: 32,
                section5_num_data_points: 2,
                ..make_default_dr()
            },
            raw_data,
        );

        let refusal = unpack_message(&msg).unwrap_err().to_string();
        assert!(refusal.contains("IEEE-packed Section 7 has 4 bytes"), "{refusal}");
    }

    #[test]
    fn test_complex_spatial_primary_missing_is_not_a_difference() {
        let dr = spatial_missing_dr(1, 4);
        let mut data = vec![10, 12, 0];
        data.extend(pack_unsigned(&[0, 0, 7, 1], 3));

        let values = unpack_complex_spatial(&data, &dr).unwrap();
        assert_eq!(values.len(), 4);
        assert_eq!(values[0], 10.0);
        assert_eq!(values[1], 12.0);
        assert!(values[2].is_nan());
        // Recurrence continues across non-missing values only:
        // 1 + 2*12 - 10 = 15, not an all-ones-code explosion.
        assert_eq!(values[3], 15.0);
    }

    #[test]
    fn test_complex_spatial_primary_and_secondary_missing_skip_recurrence() {
        let dr = spatial_missing_dr(2, 6);
        let mut data = vec![10, 12, 0];
        data.extend(pack_unsigned(&[0, 0, 7, 1, 6, 2], 3));

        let values = unpack_complex_spatial(&data, &dr).unwrap();
        assert_eq!(values.len(), 6);
        assert_eq!(values[0], 10.0);
        assert_eq!(values[1], 12.0);
        assert!(values[2].is_nan());
        assert_eq!(values[3], 15.0);
        assert!(values[4].is_nan());
        assert_eq!(values[5], 20.0);
    }

    #[test]
    fn test_complex_width_zero_groups_classify_both_missing_codes() {
        let dr = DataRepresentation {
            template: 2,
            bits_per_value: 2,
            missing_value_management: 2,
            num_groups: 3,
            group_width_ref: 0,
            group_width_bits: 0,
            group_length_ref: 1,
            group_length_inc: 0,
            last_group_length: 1,
            group_length_bits: 0,
            section5_num_data_points: 3,
            ..make_default_dr()
        };
        // Width-zero groups use the group reference itself: 3=primary,
        // 2=secondary, 1=ordinary value.  Six reference bits plus padding.
        let values = unpack_complex(&[0b1110_0100], &dr).unwrap();
        assert!(values[0].is_nan());
        assert!(values[1].is_nan());
        assert_eq!(values[2], 1.0);
    }

    #[test]
    fn test_complex_group_count_must_match_section5_points() {
        let dr = DataRepresentation {
            template: 2,
            bits_per_value: 2,
            num_groups: 3,
            group_length_ref: 1,
            last_group_length: 1,
            section5_num_data_points: 4,
            ..make_default_dr()
        };
        // The donor phrased the same disagreement as "describe 3 values,
        // expected 4"; this crate's wording names Section 5 explicitly.
        let error = unpack_complex(&[0], &dr).unwrap_err();
        assert!(
            error.contains("declare 3 values, Section 5 declares 4"),
            "{error}"
        );
    }

    #[test]
    fn test_complex_spatial_recurrence_overflow_is_an_error() {
        let dr = DataRepresentation {
            template: 3,
            bits_per_value: 0,
            num_groups: 1,
            group_width_ref: 1,
            last_group_length: 3,
            spatial_diff_order: 2,
            spatial_diff_bytes: 8,
            section5_num_data_points: 3,
            ..make_default_dr()
        };
        let mut data = Vec::new();
        data.extend_from_slice(&i64::MAX.to_be_bytes());
        data.extend_from_slice(&i64::MAX.to_be_bytes());
        data.extend_from_slice(&0_i64.to_be_bytes());
        data.extend(pack_unsigned(&[0, 0, 1], 1));
        let error = unpack_complex_spatial(&data, &dr).unwrap_err();
        assert!(
            error.contains("second-order spatial recurrence overflow"),
            "{error}"
        );
    }

    #[test]
    fn test_row_window_complex_missing_matches_full_decode() {
        use crate::grib2::parser::GridDefinition;

        let mut raw_data = vec![10, 12, 0];
        raw_data.extend(pack_unsigned(&[0, 0, 7, 1, 6, 2], 3));
        let msg = message_with(
            GridDefinition {
                nx: 3,
                ny: 2,
                num_data_points: 6,
                scan_mode: 0x40,
                ..GridDefinition::default()
            },
            spatial_missing_dr(2, 6),
            raw_data,
        );
        let full = unpack_message_normalized(&msg).unwrap();
        let window = unpack_message_scan_normalized_row_window(&msg, 0, 2).unwrap();
        assert_same_values(&full, &window);
    }

    #[test]
    fn test_zero_group_complex_constant_matches_row_window() {
        use crate::grib2::parser::GridDefinition;

        let msg = message_with(
            GridDefinition {
                nx: 3,
                ny: 2,
                num_data_points: 6,
                ..GridDefinition::default()
            },
            DataRepresentation {
                template: 3,
                reference_value: 42.5,
                bits_per_value: 0,
                num_groups: 0,
                section5_num_data_points: 6,
                ..make_default_dr()
            },
            Vec::new(),
        );

        let full = unpack_message_normalized(&msg).unwrap();
        let window = unpack_message_scan_normalized_row_window(&msg, 0, 2).unwrap();
        assert_eq!(full, vec![42.5; 6]);
        assert_eq!(window, full);
    }

    #[test]
    fn wps32_scaling_keeps_the_general_decoder_binary64() {
        let message = message_with(
            crate::grib2::parser::GridDefinition { nx: 1, ny: 1, num_data_points: 1, ..Default::default() },
            DataRepresentation { bits_per_value: 8, decimal_scale: 1, section5_num_data_points: 1, ..make_default_dr() },
            vec![9],
        );
        let general = unpack_message(&message).unwrap()[0];
        let wps = unpack_message_wps32(&message).unwrap()[0];
        assert_eq!(general.to_bits(), (9.0f64 * 0.1).to_bits());
        assert_eq!((general as f32).to_bits(), 0x3f666666);
        assert_eq!((wps as f32).to_bits(), 0x3f666667);
    }

    #[test]
    fn wps32_keeps_native_constant_and_bitmap_semantics() {
        for template in [0, 2, 3] {
            let mut message = message_with(
                crate::grib2::parser::GridDefinition { nx: 2, ny: 1, num_data_points: 2, ..Default::default() },
                DataRepresentation { template, reference_value: 42.5, decimal_scale: 2, section5_num_data_points: 1, ..make_default_dr() },
                vec![],
            );
            message.bitmap = Some(vec![true, false]);
            let values = unpack_message_wps32(&message).unwrap();
            assert_eq!(values[0], 42.5);
            assert!(values[1].is_nan());
            // The new entry retains cardinality validation.
            message.bitmap = Some(vec![true, true]);
            assert!(unpack_message_wps32(&message).is_err());
        }
    }

    #[test]
    fn wps32_scales_reconstructed_spatial_integers_with_negative_exponents() {
        let mut dr = spatial_missing_dr(0, 5);
        dr.reference_value = 0.1;
        dr.binary_scale = -1;
        dr.decimal_scale = -2;
        let mut payload = vec![1, 3, 0];
        payload.extend(pack_unsigned(&[0, 0, 1, 1, 1], 3));
        let message = message_with(
            crate::grib2::parser::GridDefinition { nx: 5, ny: 1, num_data_points: 5, ..Default::default() },
            dr, payload,
        );
        let values = unpack_message_wps32(&message).unwrap();
        let expected: Vec<f64> = [1.0f32, 3.0, 6.0, 10.0, 15.0].iter()
            .map(|x| ((x * 0.5 + 0.1) * 100.0) as f64).collect();
        assert_same_values(&expected, &values);
    }

    // Helper to create a default DataRepresentation for testing
    fn make_default_dr() -> DataRepresentation {
        DataRepresentation {
            template: 0,
            reference_value: 0.0,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 0,
            original_field_type: 0,
            group_splitting_method: 0,
            missing_value_management: 0,
            primary_missing_value: 0,
            secondary_missing_value: 0,
            num_groups: 0,
            group_width_ref: 0,
            group_width_bits: 0,
            group_length_ref: 0,
            group_length_inc: 0,
            group_length_bits: 0,
            last_group_length: 0,
            spatial_diff_order: 0,
            spatial_diff_bytes: 0,
            ccsds_flags: 0,
            ccsds_block_size: 0,
            ccsds_rsi: 0,
            section5_num_data_points: 0,
        }
    }

    // ---- CCSDS/AEC decoder tests ----

    #[test]
    fn test_unpack_ccsds_empty_data() {
        let dr = DataRepresentation {
            template: 42,
            ccsds_flags: 0,
            ccsds_block_size: 16,
            ccsds_rsi: 128,
            bits_per_value: 16,
            ..make_default_dr()
        };
        let result = unpack_ccsds(&[], &dr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_unpack_ccsds_zero_bpv() {
        let dr = DataRepresentation {
            template: 42,
            ccsds_flags: 0,
            ccsds_block_size: 16,
            ccsds_rsi: 128,
            bits_per_value: 0,
            ..make_default_dr()
        };
        let data = [0u8; 10];
        let result = unpack_ccsds(&data, &dr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_aec_pure_se_table() {
        // The SE table should have triangular number entries
        // Row i has (i+1) entries, starting at triangular(i)
        let table = aec_pure::create_se_table();
        // Entry 0: row 0, ms=0 -> table[0]=0, table[1]=0
        assert_eq!(table[0], 0);
        assert_eq!(table[1], 0);
        // Entry 1: row 1, ms=1 -> table[2]=1, table[3]=1
        assert_eq!(table[2], 1);
        assert_eq!(table[3], 1);
        // Entry 2: row 1, ms=1 -> table[4]=1, table[5]=1
        assert_eq!(table[4], 1);
        assert_eq!(table[5], 1);
        // Entry 3: row 2, ms=3 -> table[6]=2, table[7]=3
        assert_eq!(table[6], 2);
        assert_eq!(table[7], 3);
    }

    #[test]
    fn test_aec_pure_bytes_to_values_8bit() {
        let bytes = vec![0x10, 0x20, 0x30, 0x40];
        let values = aec_pure::bytes_to_values(&bytes, 1, false);
        assert_eq!(values, vec![16, 32, 48, 64]);
    }

    #[test]
    fn test_aec_pure_bytes_to_values_16bit_le() {
        // LE: [0x34, 0x12] => 0x1234 = 4660
        let bytes = vec![0x34, 0x12, 0x78, 0x56];
        let values = aec_pure::bytes_to_values(&bytes, 2, false);
        assert_eq!(values, vec![0x1234, 0x5678]);
    }

    #[test]
    fn test_aec_pure_bytes_to_values_16bit_be() {
        let bytes = vec![0x12, 0x34, 0x56, 0x78];
        let values = aec_pure::bytes_to_values(&bytes, 2, true);
        assert_eq!(values, vec![0x1234, 0x5678]);
    }

    #[test]
    fn test_aec_pure_bytes_to_values_32bit_le() {
        let bytes = vec![0x78, 0x56, 0x34, 0x12];
        let values = aec_pure::bytes_to_values(&bytes, 4, false);
        assert_eq!(values, vec![0x12345678]);
    }

    #[test]
    fn test_aec_pure_decode_empty() {
        let result = aec_pure::decode(&[], 16, 128, 0, 16, 1024);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_aec_pure_decode_zero_bits() {
        let result = aec_pure::decode(&[1, 2, 3], 16, 128, 0, 0, 1024);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_aec_pure_decode_invalid_bits() {
        let result = aec_pure::decode(&[1, 2, 3], 16, 128, 0, 33, 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_aec_pure_modify_flags() {
        // On little-endian (x86), MSB should be cleared, 3BYTE cleared
        let flags = aec_pure::modify_flags(0xFF);
        if cfg!(target_endian = "little") {
            // MSB (0x04) should be cleared, 3BYTE (0x02) should be cleared
            assert_eq!(flags & 0x04, 0, "MSB should be cleared on LE");
            assert_eq!(flags & 0x02, 0, "3BYTE should be cleared");
        }
    }

    #[test]
    fn test_unpack_ccsds_with_scaling() {
        // Test that scaling is applied after CCSDS decoding.
        // We use empty data which produces empty result, confirming
        // the function signature and pipeline work correctly.
        let dr = DataRepresentation {
            template: 42,
            reference_value: 273.15,
            binary_scale: -10,
            decimal_scale: 0,
            bits_per_value: 16,
            ccsds_flags: 0,
            ccsds_block_size: 16,
            ccsds_rsi: 128,
            ..make_default_dr()
        };
        let result = unpack_ccsds(&[], &dr).unwrap();
        assert!(result.is_empty());
    }

    /// Construct a valid AEC bitstream in uncompressed mode and verify decoding.
    ///
    /// AEC uncompressed mode: the ID field is all 1s (maximum for id_len bits),
    /// followed by block_size samples each stored in bits_per_sample bits.
    /// For 8-bit samples with block_size=4: id_len=3, so uncompressed ID = 0b111 = 7.
    #[test]
    fn test_aec_decode_uncompressed_mode() {
        // Parameters: 8-bit samples, block_size=4, rsi=1 (one block per RSI)
        let samples: [u8; 4] = [10, 20, 30, 40];

        let bits = build_aec_bitstream(7, 3, &samples, 8);

        // Decode: no preprocessing (flags=0), block_size=4, rsi=1
        let avail_out = 4; // 4 samples * 1 byte each
        let result = aec_pure::decode(&bits, 4, 1, 0, 8, avail_out);
        assert!(result.is_ok(), "decode failed: {:?}", result.err());
        let values = result.unwrap();
        assert_eq!(values.len(), 4, "expected 4 samples, got {}", values.len());
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(
                v, samples[i] as i64,
                "sample {} mismatch: expected {}, got {}",
                i, samples[i], v
            );
        }
    }

    /// Test AEC with zero-block encoding (all zeros in a block).
    /// Low entropy ID=0, sub-ID=0, FS count encodes number of zero blocks.
    #[test]
    fn test_aec_decode_zero_block() {
        // Parameters: 8-bit samples, block_size=4, rsi=1
        // id_len for 8-bit = 3
        // Low entropy: ID=0 (3 bits), then 1-bit sub-ID=0 (zero block),
        // then FS for block count: FS=0 means 1 zero block
        // FS encoding: a '1' bit (since fs=0, no leading zeros, just the stop bit)

        let mut bit_buf: u64 = 0;
        let mut bit_count: usize = 0;

        // ID = 0 (3 bits) -> low entropy
        bit_buf = (bit_buf << 3) | 0;
        bit_count += 3;

        // Sub-ID = 0 (1 bit) -> zero block
        bit_buf = (bit_buf << 1) | 0;
        bit_count += 1;

        // FS = 0 -> 1 zero block. FS encoding: stop bit '1'
        bit_buf = (bit_buf << 1) | 1;
        bit_count += 1;

        // Pad to multiple of 8
        let pad = (8 - (bit_count % 8)) % 8;
        bit_buf <<= pad;
        bit_count += pad;

        let total_bytes = bit_count / 8;
        let mut bits = Vec::new();
        for i in 0..total_bytes {
            let shift = (total_bytes - 1 - i) * 8;
            bits.push((bit_buf >> shift) as u8);
        }

        let avail_out = 4; // 4 samples * 1 byte
        let result = aec_pure::decode(&bits, 4, 1, 0, 8, avail_out);
        assert!(
            result.is_ok(),
            "zero-block decode failed: {:?}",
            result.err()
        );
        let values = result.unwrap();
        assert_eq!(values.len(), 4, "expected 4 zero samples");
        for &v in &values {
            assert_eq!(v, 0, "expected zero value");
        }
    }

    /// Test end-to-end unpack_ccsds with a valid uncompressed AEC bitstream
    /// and GRIB2 scaling parameters.
    #[test]
    fn test_unpack_ccsds_uncompressed_with_scaling() {
        // Build an uncompressed AEC bitstream for 8-bit samples
        // block_size=4, rsi=1, 4 samples of value 100
        // For 8-bit: id_len=3, uncompressed ID = 0b111 = 7
        // Bitstream: [3-bit ID=7][4 x 8-bit values] = 35 bits
        let samples = [100u8; 4];

        let data = build_aec_bitstream(7, 3, &samples, 8);

        // GRIB2 scaling: Y = (R + X * 2^E) * 10^(-D)
        // With R=273.15, E=0, D=0: Y = 273.15 + X
        let dr = DataRepresentation {
            template: 42,
            reference_value: 273.15,
            binary_scale: 0,
            decimal_scale: 0,
            bits_per_value: 8,
            ccsds_flags: 0,
            ccsds_block_size: 4,
            ccsds_rsi: 1,
            ..make_default_dr()
        };

        let result = unpack_ccsds(&data, &dr);
        assert!(result.is_ok(), "unpack_ccsds failed: {:?}", result.err());
        let values = result.unwrap();
        assert_eq!(values.len(), 4);
        for &v in &values {
            // Y = (R + X * 2^E) * 10^(-D) = (273.15 + 100) * 1 = 373.15
            // Tolerance accounts for f32 reference_value precision loss
            assert!((v - 373.15).abs() < 1e-3, "expected ~373.15, got {}", v);
        }
    }

    /// Helper: build an AEC bitstream in uncompressed mode.
    /// Writes `id_bits`-wide ID field, then `n` samples each `sample_bits` wide.
    fn build_aec_bitstream(id: u8, id_bits: usize, samples: &[u8], sample_bits: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut buf: u32 = 0;
        let mut count: usize = 0;

        // Flush helper: push complete bytes
        let flush = |bytes: &mut Vec<u8>, buf: &mut u32, count: &mut usize| {
            while *count >= 8 {
                *count -= 8;
                bytes.push((*buf >> *count) as u8);
                *buf &= (1u32 << *count) - 1;
            }
        };

        // Write ID
        buf = (buf << id_bits) | (id as u32);
        count += id_bits;
        flush(&mut bytes, &mut buf, &mut count);

        // Write samples
        for &s in samples {
            buf = (buf << sample_bits) | (s as u32);
            count += sample_bits;
            flush(&mut bytes, &mut buf, &mut count);
        }

        // Pad remaining bits to byte boundary
        if count > 0 {
            buf <<= 8 - count;
            bytes.push(buf as u8);
        }

        bytes
    }
}
