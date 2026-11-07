//! WRF OBS_DOMAIN formatted records, decoded without Python or Fortran I/O.
//!
//! Format authority: WRF v4.7.1 share/wrf_fddaobs_in.F, formats
//! 101/102/1021/103/104/105. See tools/fddaobs_wrf471_oracle/source-pin.json
//! and WRF-NOTICE.txt. This layer deliberately preserves input QC and units.
//! Projection, QC policy, duplicates, observation windows, innovations and
//! tendencies belong to the assimilation layer, not the file decoder.

use std::{error::Error, fmt, fs, path::Path, thread};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError {
    /// One-based physical line in the input file, zero for a file I/O error.
    pub line: usize,
    pub message: String,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OBS_DOMAIN line {}: {}", self.line, self.message)
    }
}
impl Error for DecodeError {}

fn fail(line: usize, message: impl Into<String>) -> DecodeError {
    DecodeError {
        line,
        message: message.into(),
    }
}

/// Value and QC are both REAL in the native format, including QC 129/256,
/// -777777 (not checked) and -888888 (missing). Do not coerce QC to bool.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Datum {
    pub value: f32,
    pub qc: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Measurements {
    /// Per level: pressure (Pa), height (m MSL), temperature (K), earth/grid
    /// U and V (m/s, QC 129 signals earth relative), relative humidity (%).
    Sounding(Vec<[Datum; 6]>),
    /// Sea-level pressure, reference pressure, height, temperature, U, V,
    /// relative humidity, surface pressure, precipitation. Native units
    /// are retained, including Pa for pressure and K for temperature.
    Surface([Datum; 9]),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub line: usize,
    /// Original 14 byte UTC date field, retained for provenance.
    pub date: String,
    /// Gregorian UTC seconds since 1970-01-01, without a local time zone.
    pub unix_seconds: i64,
    pub latitude: f32,
    pub longitude: f32,
    pub id: String,
    pub name: String,
    pub platform: String,
    pub source: String,
    pub elevation: f32,
    pub bogus: bool,
    /// Preserve the header value even for a surface report. Native code
    /// always reads one surface row, regardless of this header value.
    pub declared_count: i32,
    pub measurements: Measurements,
}

impl Report {
    pub fn levels(&self) -> usize {
        match &self.measurements {
            Measurements::Sounding(rows) => rows.len(),
            Measurements::Surface(_) => 1,
        }
    }

    /// Native rtimob = REAL(integer seconds from the original start)/3600.
    /// The caller must retain that original start across a restart.
    pub fn hours_from(&self, original_start_unix_seconds: i64) -> f32 {
        (self.unix_seconds - original_start_unix_seconds) as f32 / 3600.0
    }
}

fn field(line: &str, begin: usize, width: usize) -> String {
    // Input is checked ASCII before slicing. Formatted sequential Fortran
    // pads short records with spaces (PAD=YES by default).
    let mut out = String::with_capacity(width);
    if begin < line.len() {
        out.push_str(&line[begin..line.len().min(begin + width)]);
    }
    out.extend(std::iter::repeat_n(' ', width - out.len()));
    out
}

fn real(
    line: &str,
    begin: usize,
    width: usize,
    decimals: i32,
    n: usize,
) -> Result<f32, DecodeError> {
    // BN is the default blank interpretation. With an Fw.d input field,
    // an absent decimal point implies d fractional digits, including when
    // an exponent is supplied. Parse the decimal once to avoid a second
    // rounding caused by parsing an integer and multiplying by 10**(-d).
    let raw: String = field(line, begin, width)
        .chars()
        .filter(|c| *c != ' ')
        .collect();
    if raw.is_empty() {
        return Ok(0.0);
    }
    let mut text = raw.replace(['d', 'D'], "e");
    if !text.contains(['e', 'E']) {
        if let Some((i, _)) = text
            .char_indices()
            .skip(1)
            .find(|(_, c)| *c == '+' || *c == '-')
        {
            text.insert(i, 'e'); // Fortran also accepts 1.0-02.
        }
    }
    if !text.contains('.') {
        let e = text.find(['e', 'E']).unwrap_or(text.len());
        let (mantissa, exponent) = text.split_at(e);
        let sign_len = usize::from(mantissa.starts_with(['+', '-']));
        let (sign, digits) = mantissa.split_at(sign_len);
        if !digits.bytes().all(|b| b.is_ascii_digit()) || digits.is_empty() {
            return Err(fail(
                n,
                format!("invalid F{width}.{decimals} field at column {}", begin + 1),
            ));
        }
        let d = decimals as usize;
        text = if digits.len() > d {
            let split = digits.len() - d;
            format!("{sign}{}.{}{exponent}", &digits[..split], &digits[split..])
        } else {
            format!("{sign}0.{}{digits}{exponent}", "0".repeat(d - digits.len()))
        };
    }
    let value = text.parse::<f32>().map_err(|_| {
        fail(
            n,
            format!("invalid F{width}.{decimals} field at column {}", begin + 1),
        )
    })?;
    if !value.is_finite() {
        return Err(fail(n, format!("nonfinite real at column {}", begin + 1)));
    }
    Ok(value)
}

fn integer(line: &str, begin: usize, width: usize, n: usize) -> Result<i32, DecodeError> {
    let raw = field(line, begin, width).replace(' ', "");
    if raw.is_empty() {
        return Ok(0);
    }
    raw.parse()
        .map_err(|_| fail(n, format!("invalid I{width} field at column {}", begin + 1)))
}

fn logical(line: &str, begin: usize, n: usize) -> Result<bool, DecodeError> {
    let text = field(line, begin, 4);
    match text.trim_start().trim_start_matches('.').as_bytes().first() {
        Some(b'T' | b't') => Ok(true),
        Some(b'F' | b'f') => Ok(false),
        _ => Err(fail(n, format!("invalid L4 field at column {}", begin + 1))),
    }
}

/// Parse the 14 digit date accepted by native format 101 and fmt_date.
pub fn utc_seconds(text: &str) -> Result<i64, DecodeError> {
    if text.len() != 14 || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(fail(0, "date must contain YYYYMMDDhhmmss"));
    }
    let num = |a, b| text[a..b].parse::<i64>().unwrap();
    let (year, month, day) = (num(0, 4), num(4, 6), num(6, 8));
    let (hour, minute, second) = (num(8, 10), num(10, 12), num(12, 14));
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if year == 0
        || !(1..=12).contains(&month)
        || day < 1
        || day > days_in_month[(month - 1) as usize]
        || hour > 23
        || minute > 59
        || second > 59
    {
        return Err(fail(0, "invalid Gregorian UTC date"));
    }
    let y = year - i64::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * m + 2) / 5 + day - 1;
    let days = era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
    Ok(days * 86400 + hour * 3600 + minute * 60 + second)
}

fn row<const N: usize>(text: &str, n: usize) -> Result<[Datum; N], DecodeError> {
    let mut values = [Datum {
        value: 0.0,
        qc: 0.0,
    }; N];
    for (i, v) in values.iter_mut().enumerate() {
        v.value = real(text, 1 + 24 * i, 11, 3, n)?;
        v.qc = real(text, 13 + 24 * i, 11, 3, n)?;
    }
    Ok(values)
}

#[derive(Clone, Copy)]
struct Span {
    start: usize,
    count: usize,
    sounding: bool,
}

fn parse(lines: &[&str], span: Span) -> Result<Report, DecodeError> {
    let s = span.start;
    let date = field(lines[s], 1, 14);
    let unix_seconds = utc_seconds(&date).map_err(|mut e| {
        e.line = s + 1;
        e
    })?;
    let hdr = lines[s + 3];
    let measurements = if span.sounding {
        Measurements::Sounding(
            (0..span.count)
                .map(|i| row::<6>(lines[s + 4 + i], s + 5 + i))
                .collect::<Result<_, _>>()?,
        )
    } else {
        Measurements::Surface(row::<9>(lines[s + 4], s + 5)?)
    };
    Ok(Report {
        line: s + 1,
        date,
        unix_seconds,
        latitude: real(lines[s + 1], 2, 9, 4, s + 2)?,
        longitude: real(lines[s + 1], 12, 9, 4, s + 2)?,
        id: field(lines[s + 2], 2, 40),
        name: field(lines[s + 2], 45, 40),
        platform: field(hdr, 2, 16),
        source: field(hdr, 20, 16),
        elevation: real(hdr, 38, 8, 0, s + 4)?,
        bogus: logical(hdr, 54, s + 4)?,
        declared_count: integer(hdr, 60, 5, s + 4)?,
        measurements,
    })
}

/// Decode one file in file order with at most `workers` concurrent workers.
/// `max_obs` bounds expanded levels, matching WRF's observation-buffer unit.
/// No assimilation filtering, sorting or deduplication happens implicitly.
pub fn decode(input: &[u8], max_obs: usize, workers: usize) -> Result<Vec<Report>, DecodeError> {
    if workers == 0 {
        return Err(fail(0, "workers must be positive"));
    }
    if !input.is_ascii() {
        return Err(fail(0, "formatted input must be ASCII"));
    }
    let text = std::str::from_utf8(input).unwrap();
    let lines: Vec<_> = text.lines().collect();
    let (mut spans, mut start, mut levels) = (Vec::new(), 0usize, 0usize);
    // One light structural pass finds report boundaries. Floating-point
    // parsing and allocating the measurements run in parallel below.
    while start < lines.len() {
        if lines.len() - start < 4 {
            return Err(fail(start + 1, "incomplete report header"));
        }
        let hdr = lines[start + 3];
        let sounding = logical(hdr, 48, start + 4)?;
        let declared = integer(hdr, 60, 5, start + 4)?;
        let count = if sounding {
            if declared <= 0 {
                return Err(fail(start + 4, "sounding must contain at least one level"));
            }
            declared as usize
        } else {
            1
        };
        levels = levels
            .checked_add(count)
            .ok_or_else(|| fail(start + 4, "observation count overflow"))?;
        if levels > max_obs {
            return Err(fail(start + 4, "expanded levels exceed max_obs"));
        }
        let end = start
            .checked_add(4 + count)
            .ok_or_else(|| fail(start + 1, "report size overflow"))?;
        if end > lines.len() {
            return Err(fail(lines.len() + 1, "incomplete measurement records"));
        }
        spans.push(Span {
            start,
            count,
            sounding,
        });
        start = end;
    }
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    let available = thread::available_parallelism().map_or(1, usize::from);
    let nworkers = workers.min(available).min(spans.len());
    let chunk = spans.len().div_ceil(nworkers);
    thread::scope(|scope| {
        let handles: Vec<_> = spans
            .chunks(chunk)
            .map(|batch| {
                let lines = &lines;
                thread::Builder::new()
                    .spawn_scoped(scope, move || {
                        batch
                            .iter()
                            .map(|s| parse(lines, *s))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .map_err(|e| fail(0, format!("cannot start decoder worker: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut result = Vec::with_capacity(spans.len());
        for handle in handles {
            result.extend(
                handle
                    .join()
                    .map_err(|_| fail(0, "decoder worker panicked"))??,
            );
        }
        Ok(result)
    })
}

pub fn read(path: &Path, max_obs: usize, workers: usize) -> Result<Vec<Report>, DecodeError> {
    let input = fs::read(path).map_err(|e| fail(0, format!("{}: {e}", path.display())))?;
    decode(&input, max_obs, workers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(sounding: bool, n: usize) -> String {
        let mut s = format!(
            " 20240229235930\n  {:9.4} {:9.4} \n  {:40}   {:40}\n  {:16}  {:16}  {:8.0}  {:>4}  {:>4}  {:5}\n",
            39.875,
            -104.675,
            "0123",
            "test",
            "FM-12 SYNOP",
            "fixture",
            1500.0,
            if sounding { "T" } else { "F" },
            "F",
            n
        );
        for _ in 0..if sounding { n } else { 1 } {
            s.push(' ');
            for i in 0..if sounding { 6 } else { 9 } {
                s.push_str(&format!("{:11.3} {:11.3} ", i as f32 - 0.125, 129.0));
            }
            s.push('\n');
        }
        s
    }

    #[test]
    fn parallel_soundings_surface_and_crlf_preserve_all_fields_and_order() {
        let input = (fixture(false, 1) + &fixture(true, 3)).replace('\n', "\r\n");
        let reports = decode(input.as_bytes(), 4, 8).unwrap();
        assert_eq!(reports, decode(input.as_bytes(), 4, 1).unwrap());
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].latitude, 39.875);
        assert_eq!(reports[0].longitude, -104.675);
        assert_eq!(reports[0].elevation, 1500.0);
        assert_eq!(reports[0].id.trim(), "0123");
        assert_eq!(reports[0].source.trim(), "fixture");
        assert_eq!(reports[0].levels(), 1);
        assert_eq!(reports[1].levels(), 3);
        assert_eq!(reports[1].line, 6);
        if let Measurements::Surface(row) = &reports[0].measurements {
            assert_eq!(
                row[8],
                Datum {
                    value: 7.875,
                    qc: 129.0
                }
            );
        } else {
            panic!("surface report lost");
        }
    }

    #[test]
    fn missing_and_unchecked_qc_are_preserved() {
        let text = fixture(true, 1).replace("     -0.125     129.000", "-888888.000 -777777.000");
        let reports = decode(text.as_bytes(), 1, 2).unwrap();
        if let Measurements::Sounding(rows) = &reports[0].measurements {
            assert_eq!(
                rows[0][0],
                Datum {
                    value: -888888.0,
                    qc: -777777.0
                }
            );
        } else {
            panic!("sounding lost");
        }
    }

    #[test]
    fn fortran_fixed_width_numeric_rules() {
        assert_eq!(real("     123456", 0, 11, 3, 1).unwrap(), 123.456);
        assert_eq!(real("   12345E-2", 0, 11, 3, 1).unwrap(), 0.12345);
        assert_eq!(real(" 1.250D+02 ", 0, 11, 3, 1).unwrap(), 125.0);
        assert_eq!(real("    1.25-02", 0, 11, 3, 1).unwrap(), 0.0125);
        assert_eq!(real(" 1 2 3 . 4 ", 0, 11, 3, 1).unwrap(), 123.4);
        assert_eq!(real("", 0, 11, 3, 1).unwrap(), 0.0);
        assert_eq!(
            real(" -000000000", 0, 11, 3, 1).unwrap().to_bits(),
            (-0.0f32).to_bits()
        );
        assert!(real("        NaN", 0, 11, 3, 1).is_err());
    }

    #[test]
    fn utc_and_original_restart_clock() {
        assert_eq!(utc_seconds("19700101000000").unwrap(), 0);
        assert_eq!(utc_seconds("19691231235959").unwrap(), -1);
        assert_eq!(
            utc_seconds("20000301000000").unwrap() - utc_seconds("20000228000000").unwrap(),
            172800
        );
        assert!(utc_seconds("19000229000000").is_err());
        assert!(utc_seconds("20241101000060").is_err());
        let reports = decode(fixture(false, 1).as_bytes(), 1, 1).unwrap();
        let start = utc_seconds("20240228235930").unwrap();
        assert_eq!(reports[0].hours_from(start), 24.0);
    }

    #[test]
    fn malformed_input_and_capacity_fail_without_partial_output() {
        let text = fixture(true, 3);
        assert!(
            decode(text.as_bytes(), 2, 2)
                .unwrap_err()
                .message
                .contains("max_obs")
        );
        let truncated = text.lines().take(5).collect::<Vec<_>>().join("\n");
        assert!(
            decode(truncated.as_bytes(), 3, 2)
                .unwrap_err()
                .message
                .contains("incomplete")
        );
        assert!(decode(fixture(true, 0).as_bytes(), 1, 1).is_err());
        assert!(decode(b"", 1, 0).is_err());
        assert!(decode(b"\xff", 1, 1).is_err());
        assert!(decode(b"", 0, 1).unwrap().is_empty());
    }
}
