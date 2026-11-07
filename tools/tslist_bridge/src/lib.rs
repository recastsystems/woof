//! WRF v4.7.1 tslist reader and text writer. No forecast arithmetic runs here.
//!
//! Source: wrf-model/WRF f52c197ed39d, share/wrf_tsin.F and
//! share/wrf_timeseries.F. The accompanying WRF notice applies to the
//! transcribed layouts. See SOURCE.md for hashes and the notice.
use std::collections::HashSet;
use std::fmt::Write as FmtWrite;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, Write};
use std::path::{Path, PathBuf};
mod ffi;

const HASH_INITIAL: u64 = 0xcbf29ce484222325;
fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        hash = (hash ^ u64::from(*b)).wrapping_mul(0x100000001b3);
    }
    hash
}

pub const ENDINGS: [&str; 8] = ["TS", "UU", "VV", "WW", "PH", "TH", "QV", "PR"];
pub const SURFACE_FIELDS: usize = 14;
pub const PROFILE_FIELDS: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coordinates {
    Geographic,
    Grid,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Station {
    pub id: usize,
    pub description: String,
    pub prefix: String,
    /// Geographic latitude or one-based grid i, rounded to WRF REAL first.
    pub first: f32,
    /// Geographic longitude or one-based grid j, rounded to WRF REAL first.
    pub second: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StationList {
    pub coordinates: Coordinates,
    pub stations: Vec<Station>,
    /// WRF stops reading at max_ts_locs, including an invalid later record.
    pub truncated: bool,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn field(line: &str, start: usize, end: usize) -> &str {
    // ASCII is validated before slicing. Short Fortran records are blank padded.
    &line[start.min(line.len())..end.min(line.len())]
}

fn fortran_real(text: &str, decimals: i32) -> io::Result<f32> {
    let raw = text.trim();
    // A blank numeric input field is zero in WRF's fixed-width READ.
    if raw.is_empty() {
        return Ok(0.0);
    }
    let compact: String = raw.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let mut compact = compact.replace(['D', 'd'], "E");
    // F input also permits an exponent sign without E, as in 12345-6.
    if !compact.contains(['E', 'e']) {
        if let Some(at) = compact
            .char_indices()
            .skip(1)
            .find_map(|(i, c)| (c == '+' || c == '-').then_some(i))
        {
            compact.insert(at, 'E');
        }
    }
    let exponent = compact.find(['E', 'e']);
    let mantissa = exponent.map_or(compact.as_str(), |i| &compact[..i]);
    // Insert the implied decimal before conversion. Dividing an already
    // rounded f32 by 1000 introduces a second rounding absent in native READ.
    if !mantissa.contains('.') {
        let at = exponent.unwrap_or(compact.len());
        let sign = usize::from(mantissa.starts_with(['+', '-']));
        let digits = at - sign;
        let places = usize::try_from(decimals).map_err(|_| invalid("negative decimal count"))?;
        if digits <= places {
            compact.insert_str(sign, &format!("0.{}", "0".repeat(places - digits)));
        } else {
            compact.insert(at - places, '.');
        }
    }
    let value = compact
        .parse::<f32>()
        .map_err(|_| invalid(format!("invalid WRF real {raw:?}")))?;
    if !value.is_finite() {
        return Err(invalid("nonfinite tslist coordinate"));
    }
    Ok(value)
}

/// WRF fixed columns, including the header's two A7 fields at columns 33/42.
/// Repeated prefixes are refused because they alias the same eight files.
pub fn parse(text: &str, max_locs: usize, ideal: bool) -> io::Result<StationList> {
    if max_locs == 0 {
        return Ok(StationList {
            coordinates: Coordinates::Grid,
            stations: vec![],
            truncated: false,
        });
    }
    if !text.is_ascii() {
        return Err(invalid("tslist must use ASCII fixed columns"));
    }
    let mut lines = text.lines();
    lines
        .next()
        .ok_or_else(|| invalid("tslist lacks first header line"))?;
    let header = lines
        .next()
        .ok_or_else(|| invalid("tslist lacks coordinate header"))?;
    lines
        .next()
        .ok_or_else(|| invalid("tslist lacks third header line"))?;
    let a = field(header, 32, 39).to_ascii_uppercase();
    let b = field(header, 41, 48).to_ascii_uppercase();
    let coordinates = if a.contains("LAT") && b.contains("LON") {
        Coordinates::Geographic
    } else if a.contains('I') && b.contains('J') {
        Coordinates::Grid
    } else {
        return Err(invalid(
            "Header line requires either LAT LON or I J (WRF wrf_tsin.F:66)",
        ));
    };
    let mut stations = Vec::new();
    let mut prefixes = HashSet::new();
    for (row, line) in lines.enumerate() {
        if stations.len() == max_locs {
            return Ok(StationList {
                coordinates,
                stations,
                truncated: true,
            });
        }
        let prefix = field(line, 26, 31).trim_end();
        if prefix.is_empty()
            || !prefix
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        {
            return Err(invalid(format!(
                "tslist line {}: prefix must be 1-5 ASCII letters, digits, '_' or '-'",
                row + 4
            )));
        }
        if !prefixes.insert(prefix.to_owned()) {
            return Err(invalid(format!(
                "tslist line {}: duplicate output prefix {prefix}",
                row + 4
            )));
        }
        let numeric = |text: &str| -> io::Result<f32> {
            if ideal || coordinates == Coordinates::Grid {
                let compact: String = text.chars().filter(|c| !c.is_ascii_whitespace()).collect();
                if compact.is_empty() {
                    return Ok(0.0);
                }
                compact
                    .parse::<i32>()
                    .map(|v| v as f32)
                    .map_err(|_| invalid("invalid integer grid index"))
            } else {
                fortran_real(text, 3)
            }
        };
        let first = numeric(field(line, 32, 39))
            .map_err(|e| invalid(format!("tslist line {}: {e}", row + 4)))?;
        let second = numeric(field(line, 40, 48))
            .map_err(|e| invalid(format!("tslist line {}: {e}", row + 4)))?;
        if !ideal
            && coordinates == Coordinates::Geographic
            && (first.abs() > 90.0 || second.abs() > 360.0)
        {
            return Err(invalid(format!(
                "tslist line {}: geographic coordinate out of range",
                row + 4
            )));
        }
        stations.push(Station {
            id: row + 1,
            description: field(line, 0, 25).trim_end().to_owned(),
            prefix: prefix.to_owned(),
            first,
            second,
        });
    }
    Ok(StationList {
        coordinates,
        stations,
        truncated: false,
    })
}

#[derive(Clone, Debug)]
pub struct PlacedStation {
    pub station: Station,
    /// One-based native WRF indices. A station belongs to every containing domain.
    pub i: i32,
    pub j: i32,
    pub grid_lat: f32,
    pub grid_lon: f32,
    pub elevation: f32,
}

fn real(value: f32, width: usize, precision: usize) -> String {
    let text = if value.is_nan() {
        "NaN".to_owned()
    } else if value.is_infinite() {
        if value.is_sign_negative() {
            "-Infinity"
        } else {
            "Infinity"
        }
        .to_owned()
    } else {
        format!("{value:.precision$}")
    };
    if text.len() > width {
        "*".repeat(width)
    } else {
        format!("{text:>width$}")
    }
}

fn integer(value: impl std::fmt::Display, width: usize) -> String {
    let text = value.to_string();
    if text.len() > width {
        "*".repeat(width)
    } else {
        format!("{text:>width$}")
    }
}

pub fn header(
    station: &PlacedStation,
    coordinates: Coordinates,
    domain: u8,
    start: &str,
) -> io::Result<String> {
    if start.len() != 19 || !start.is_ascii() || start.contains(['\n', '\r']) {
        return Err(invalid("simulation start must be YYYY-MM-DD_HH:MM:SS"));
    }
    let s = &station.station;
    if s.prefix.is_empty()
        || s.prefix.len() > 5
        || !s
            .prefix
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
    {
        return Err(invalid("station prefix must fit five safe ASCII columns"));
    }
    if s.id == 0
        || station.i < 1
        || station.j < 1
        || ![
            station.grid_lat,
            station.grid_lon,
            station.elevation,
            s.first,
            s.second,
        ]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err(invalid("invalid placed station metadata"));
    }
    let (lat, lon) = if coordinates == Coordinates::Grid {
        (station.grid_lat, station.grid_lon)
    } else {
        (s.first, s.second)
    };
    if s.description.len() > 25
        || !s.description.is_ascii()
        || s.description.contains(['\n', '\r', '\t'])
    {
        return Err(invalid("station description must fit 25 ASCII columns"));
    }
    Ok(format!(
        "{:<25} {}{} {:<5} (",
        s.description,
        integer(domain, 2),
        integer(s.id, 3),
        s.prefix
    ) + &real(lat, 7, 3)
        + ","
        + &real(lon, 8, 3)
        + ") ("
        + &integer(station.i, 4)
        + ","
        + &integer(station.j, 4)
        + ") ("
        + &real(station.grid_lat, 7, 3)
        + ","
        + &real(station.grid_lon, 8, 3)
        + ") "
        + &real(station.elevation, 6, 1)
        + " meters  "
        + start
        + "\n")
}

/// Values are [hour, T2,Q2,U10,V10,PSFC,GLW,GSW,HFX,LH,TSK,TSLB1,RAINC,RAINNC,CLW,
/// UU[levels],VV[levels],WW[levels],PH[levels],TH[levels],QV[levels],PR[levels]].
pub fn surface_row(domain: u8, station: &PlacedStation, row: &[f32]) -> io::Result<String> {
    if row.len() < 1 + SURFACE_FIELDS {
        return Err(invalid("short tslist surface record"));
    }
    let mut out = integer(domain, 2)
        + &real(row[0], 13, 6)
        + &integer(station.station.id, 5)
        + &integer(station.i, 5)
        + &integer(station.j, 5)
        + " ";
    for value in &row[1..=SURFACE_FIELDS] {
        out.push_str(&real(*value, 13, 5));
        out.push(' ');
    }
    // Fortran's terminal 1X positions the cursor but does not extend the
    // record. A literal trailing space would change native file bytes.
    out.pop();
    out.push('\n');
    Ok(out)
}

pub fn profile_row(hour: f32, profile: &[f32]) -> String {
    let mut out = real(hour, 13, 6) + " ";
    for value in profile {
        out.push_str(&real(*value, 13, 5));
    }
    // WRF writes the repeat count into substring (12:14), overwriting the
    // opening parenthesis. Native gfortran accepts 0015f13.5,1x)), so the 1X
    // occurs once after all levels rather than after each value. Reproduce
    // that observed native format, including its non-extending terminal 1X.
    out.push('\n');
    out
}

/// Compact receipt. The hash binds the exact saved prefix, including headers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileReceipt {
    pub length: u64,
    pub hash: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub files: Vec<FileReceipt>,
    pub layout_hash: u64,
    pub last_hour_bits: Option<u32>,
}

fn hash_prefix(file: &mut File, length: u64) -> io::Result<u64> {
    file.rewind()?;
    let mut hash = HASH_INITIAL;
    let mut remaining = length;
    let mut buffer = [0u8; 65536];
    while remaining > 0 {
        let count = (remaining as usize).min(buffer.len());
        file.read_exact(&mut buffer[..count])?;
        hash = hash_bytes(hash, &buffer[..count]);
        remaining -= count as u64;
    }
    Ok(hash)
}

/// One station owns eight files. Independent stations are written concurrently
/// by Writer, which caps its worker count instead of starting a thread per site.
struct RecordedFile {
    file: File,
    receipt: FileReceipt,
}
impl RecordedFile {
    fn append(&mut self, bytes: &[u8]) -> io::Result<()> {
        // A failed write makes this writer unusable; no checkpoint is valid
        // until the caller has handled that error. Never hide a short write.
        self.file.write_all(bytes)?;
        self.receipt.length += bytes.len() as u64;
        self.receipt.hash = hash_bytes(self.receipt.hash, bytes);
        Ok(())
    }
}
struct StationFiles {
    files: [RecordedFile; 8],
}

pub struct Writer {
    pub stations: Vec<PlacedStation>,
    files: Vec<StationFiles>,
    pub domain: u8,
    pub levels: usize,
    pub directory: PathBuf,
    last_hour: Option<f32>,
    layout_hash: u64,
    failed: bool,
}

fn layout_hash(
    stations: &[PlacedStation],
    coordinates: Coordinates,
    domain: u8,
    levels: usize,
    start: &str,
) -> io::Result<u64> {
    let mut hash = hash_bytes(HASH_INITIAL, &(levels as u64).to_le_bytes());
    for station in stations {
        hash = hash_bytes(
            hash,
            header(station, coordinates, domain, start)?.as_bytes(),
        );
    }
    Ok(hash)
}

impl Writer {
    pub fn create(
        directory: &Path,
        stations: Vec<PlacedStation>,
        coordinates: Coordinates,
        domain: u8,
        levels: usize,
        start: &str,
    ) -> io::Result<Self> {
        if domain == 0 || domain > 99 || levels == 0 || levels > 999 {
            return Err(invalid("WRF tslist needs domain 1-99 and levels 1-999"));
        }
        std::fs::create_dir_all(directory)?;
        let mut files = Vec::new();
        let mut created = Vec::new();
        // Preflight every header and filename before writing anything.
        let mut names = HashSet::new();
        for station in &stations {
            header(station, coordinates, domain, start)?;
            if !names.insert(station.station.prefix.clone()) {
                return Err(invalid("duplicate station prefix"));
            }
            if station.station.prefix.is_empty()
                || !station
                    .station
                    .prefix
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
            {
                return Err(invalid("unsafe station output prefix"));
            }
            for ending in ENDINGS {
                if directory
                    .join(format!("{}.d{domain:02}.{ending}", station.station.prefix))
                    .symlink_metadata()
                    .is_ok()
                {
                    return Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "tslist output exists; resume with its checkpoint receipt or use a fresh member directory",
                    ));
                }
            }
        }
        let result = (|| {
            for station in &stations {
                let h = header(station, coordinates, domain, start)?;
                let mut group = Vec::new();
                for ending in ENDINGS {
                    let path =
                        directory.join(format!("{}.d{domain:02}.{ending}", station.station.prefix));
                    let f = OpenOptions::new()
                        .create_new(true)
                        .read(true)
                        .write(true)
                        .open(&path)?;
                    created.push(path);
                    let mut f = RecordedFile {
                        file: f,
                        receipt: FileReceipt {
                            length: 0,
                            hash: HASH_INITIAL,
                        },
                    };
                    f.append(h.as_bytes())?;
                    group.push(f);
                }
                files.push(StationFiles {
                    files: group
                        .try_into()
                        .map_err(|_| invalid("station file inventory"))?,
                });
            }
            Ok(())
        })();
        if let Err(error) = result {
            // Only paths create_new succeeded on are ours to remove.
            for path in created {
                let _ = std::fs::remove_file(path);
            }
            return Err(error);
        }
        let layout_hash = layout_hash(&stations, coordinates, domain, levels, start)?;
        Ok(Self {
            stations,
            files,
            domain,
            levels,
            directory: directory.to_owned(),
            last_hour: None,
            layout_hash,
            failed: false,
        })
    }

    /// A verified ensemble roster can place a member back before its first
    /// step. Validate every owned header before discarding that attempt's tail.
    /// A genuinely pending member has no files yet and follows normal create.
    pub fn rewind_initial(
        directory: &Path,
        stations: Vec<PlacedStation>,
        coordinates: Coordinates,
        domain: u8,
        levels: usize,
        start: &str,
    ) -> io::Result<Self> {
        let any_existing = stations.iter().any(|station| {
            ENDINGS.iter().any(|ending| {
                directory
                    .join(format!("{}.d{domain:02}.{ending}", station.station.prefix))
                    .symlink_metadata()
                    .is_ok()
            })
        });
        if !any_existing {
            return Self::create(directory, stations, coordinates, domain, levels, start);
        }
        let mut files = Vec::new();
        for station in &stations {
            let h = header(station, coordinates, domain, start)?;
            files.extend(
                [FileReceipt {
                    length: h.len() as u64,
                    hash: hash_bytes(HASH_INITIAL, h.as_bytes()),
                }; 8],
            );
        }
        let snapshot = Snapshot {
            files,
            layout_hash: layout_hash(&stations, coordinates, domain, levels, start)?,
            last_hour_bits: None,
        };
        Self::resume(
            directory,
            stations,
            coordinates,
            domain,
            levels,
            start,
            &snapshot,
        )
    }

    /// Check EVERY file before truncating any tail written after a checkpoint.
    /// The saved header binds station order, coordinates, domain and start;
    /// levels are also bound. A changed prefix or short file fails closed.
    pub fn resume(
        directory: &Path,
        stations: Vec<PlacedStation>,
        coordinates: Coordinates,
        domain: u8,
        levels: usize,
        start: &str,
        snapshot: &Snapshot,
    ) -> io::Result<Self> {
        Self::resume_mode(
            directory,
            stations,
            coordinates,
            domain,
            levels,
            start,
            snapshot,
            true,
        )
    }

    pub fn validate_resume(
        directory: &Path,
        stations: Vec<PlacedStation>,
        coordinates: Coordinates,
        domain: u8,
        levels: usize,
        start: &str,
        snapshot: &Snapshot,
    ) -> io::Result<()> {
        Self::resume_mode(
            directory,
            stations,
            coordinates,
            domain,
            levels,
            start,
            snapshot,
            false,
        )
        .map(|_| ())
    }

    fn resume_mode(
        directory: &Path,
        stations: Vec<PlacedStation>,
        coordinates: Coordinates,
        domain: u8,
        levels: usize,
        start: &str,
        snapshot: &Snapshot,
        truncate: bool,
    ) -> io::Result<Self> {
        if domain == 0 || domain > 99 || levels == 0 || levels > 999 {
            return Err(invalid("invalid tslist resume layout"));
        }
        let layout_hash = layout_hash(&stations, coordinates, domain, levels, start)?;
        if layout_hash != snapshot.layout_hash || snapshot.files.len() != stations.len() * 8 {
            return Err(invalid("tslist checkpoint layout mismatch"));
        }
        let last_hour = snapshot.last_hour_bits.map(f32::from_bits);
        if last_hour.is_some_and(|hour| !hour.is_finite()) {
            return Err(invalid("invalid tslist checkpoint clock"));
        }
        let mut files = Vec::new();
        let mut prefixes = HashSet::new();
        for (point, station) in stations.iter().enumerate() {
            if !prefixes.insert(&station.station.prefix)
                || station.station.prefix.is_empty()
                || !station
                    .station
                    .prefix
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
            {
                return Err(invalid("unsafe or duplicate tslist resume prefix"));
            }
            let expected = header(station, coordinates, domain, start)?;
            let mut group = Vec::new();
            for (field, ending) in ENDINGS.iter().enumerate() {
                let path =
                    directory.join(format!("{}.d{domain:02}.{ending}", station.station.prefix));
                if !path.symlink_metadata()?.file_type().is_file() {
                    return Err(invalid("tslist resume path is not a regular file"));
                }
                let receipt = snapshot.files[point * 8 + field];
                let mut file = OpenOptions::new().read(true).write(truncate).open(path)?;
                if receipt.length < expected.len() as u64 || file.metadata()?.len() < receipt.length
                {
                    return Err(invalid(
                        "tslist checkpoint file is shorter than its receipt",
                    ));
                }
                let mut actual = vec![0u8; expected.len()];
                file.read_exact(&mut actual)?;
                if actual != expected.as_bytes()
                    || hash_prefix(&mut file, receipt.length)? != receipt.hash
                {
                    return Err(invalid(
                        "tslist checkpoint file prefix differs from its receipt",
                    ));
                }
                group.push(RecordedFile { file, receipt });
            }
            files.push(StationFiles {
                files: group
                    .try_into()
                    .map_err(|_| invalid("station file inventory"))?,
            });
        }
        // Validation above is read-only, including errors on the last file.
        for station in files.iter_mut().filter(|_| truncate) {
            for saved in &mut station.files {
                saved.file.set_len(saved.receipt.length)?;
                saved.file.seek(io::SeekFrom::Start(saved.receipt.length))?;
            }
        }
        Ok(Self {
            stations,
            files,
            domain,
            levels,
            directory: directory.to_owned(),
            last_hour,
            layout_hash,
            failed: false,
        })
    }

    /// No CUDA call occurs here. The caller must wait on its batch copy event
    /// off the forecast thread before supplying the pinned host slice.
    pub fn write_batch(&mut self, values: &[f32], samples: usize) -> io::Result<()> {
        if self.failed {
            return Err(invalid(
                "tslist writer failed; restore a durable checkpoint before reuse",
            ));
        }
        let stride = 1 + SURFACE_FIELDS + PROFILE_FIELDS * self.levels;
        let points = self.stations.len();
        let expected = samples
            .checked_mul(points)
            .and_then(|n| n.checked_mul(stride))
            .ok_or_else(|| invalid("tslist batch size overflow"))?;
        if values.len() != expected {
            return Err(invalid("tslist batch shape mismatch"));
        }
        if points == 0 || samples == 0 {
            return Ok(());
        }
        let mut prior = self.last_hour;
        for sample in 0..samples {
            let hour = values[sample * points * stride];
            if !hour.is_finite() || prior.is_some_and(|p| hour <= p) {
                return Err(invalid("nonincreasing tslist sample time"));
            }
            for p in 1..points {
                if values[(sample * points + p) * stride].to_bits() != hour.to_bits() {
                    return Err(invalid("station sample clocks disagree"));
                }
            }
            prior = Some(hour);
        }
        let domain = self.domain;
        let levels = self.levels;
        let stations = &self.stations;
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8)
            .min(points);
        let chunk = points.div_ceil(workers);
        let written = std::thread::scope(|scope| -> io::Result<()> {
            let mut handles = Vec::new();
            for (group, files) in self.files.chunks_mut(chunk).enumerate() {
                handles.push(scope.spawn(move || -> io::Result<()> {
                    let first = group * chunk;
                    for (offset, station_files) in files.iter_mut().enumerate() {
                        let point = first + offset;
                        let mut text: [String; 8] = std::array::from_fn(|_| String::new());
                        for sample in 0..samples {
                            let start = (sample * points + point) * stride;
                            let row = &values[start..start + stride];
                            text[0].push_str(&surface_row(domain, &stations[point], row)?);
                            for profile in 0..PROFILE_FIELDS {
                                let start = 1 + SURFACE_FIELDS + profile * levels;
                                text[profile + 1]
                                    .push_str(&profile_row(row[0], &row[start..start + levels]));
                            }
                        }
                        for (file, text) in station_files.files.iter_mut().zip(text) {
                            file.append(text.as_bytes())?;
                        }
                    }
                    Ok(())
                }));
            }
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| io::Error::other("tslist writer worker panicked"))??;
            }
            Ok(())
        });
        if let Err(error) = written {
            self.failed = true;
            return Err(error);
        }
        self.last_hour = prior;
        Ok(())
    }

    pub fn receipt(&mut self) -> io::Result<Snapshot> {
        if self.failed {
            return Err(invalid("cannot checkpoint a failed tslist writer"));
        }
        let mut result = Vec::new();
        for station in &mut self.files {
            for file in &mut station.files {
                file.file.sync_data()?;
                result.push(file.receipt);
            }
        }
        Ok(Snapshot {
            files: result,
            layout_hash: self.layout_hash,
            last_hour_bits: self.last_hour.map(f32::to_bits),
        })
    }
}

/// Exact bit representation for parser consumers. Text never loses an f32 bit.
pub fn describe(list: &StationList) -> String {
    let mut out = format!(
        "{}\t{}\t{}\n",
        if list.coordinates == Coordinates::Grid {
            "ij"
        } else {
            "latlon"
        },
        list.stations.len(),
        u8::from(list.truncated)
    );
    for s in &list.stations {
        writeln!(
            out,
            "{}\t{}\t{:08x}\t{:08x}\t{}",
            s.id,
            s.prefix,
            s.first.to_bits(),
            s.second.to_bits(),
            s.description
        )
        .unwrap();
    }
    out
}


/// The revision stamp is exported so it survives shared-library linking.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_tslist_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0");
    SOURCE_REV_STAMP.as_ptr().cast()
}
