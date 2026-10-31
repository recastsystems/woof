//! RRFS sibling at RRFSx/MPAS-Model 1d6e370e.
//! Native intermediate formats: PREGRID3, GRIB_PREP4 and WPS5.
//! MPAS-A v8.4.2 mpas_init_atm_read_met.F:9-431. The module defines
//! _METGRID: projected spacing is multiplied by1000, longitudes above180
//! subtract360 once and start latitude is clamped to[-90,90]. Global case7
//! ignores the CENTER start indices, but they are preserved by this decoder.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::{MpasError, MpasResult};

/// The WPS intermediate version emitted by the current writer.
pub const WPS_FORMAT_VERSION: i32 = 5;

/// Original file projection codes (native read_met remaps1/3/5 to3/1/2).
pub const IPROJ_LATLON: i32 = 0;
pub const IPROJ_GAUSS: i32 = 4;
pub const IPROJ_POLAR: i32 = 5;

#[derive(Debug,Clone)]
pub struct PolarHeader {pub dx:f32,pub dy:f32,pub stdlon:f32,pub truelat:f32}

/// One field at one level, exactly as the file carries it.
#[derive(Debug, Clone)]
pub struct MetSlab {
    pub center_lat:f32,
    pub center_lon:f32,
    pub pole_lat:f32,
    pub pole_lon:f32,
    pub format_version:i32,
    pub xfcst:f32,
    pub start_i:f32,
    pub start_j:f32,
    pub wind_grid_rel:bool,
    pub dx:f32,
    pub dy:f32,
    pub stdlon:f32,
    pub truelat1:f32,
    pub truelat2:f32,
    /// `YYYY-MM-DD_HH:MM:SS`, trimmed.
    pub hdate: String,
    /// The producing centre's own 32-character label.
    pub map_source: String,
    /// `TT`, `UU`, `SOILT001`, ... trimmed.
    pub field: String,
    pub units: String,
    pub desc: String,
    /// The level tag.  `200100.0` marks the surface level; pressure levels
    /// carry their pressure in Pa.
    pub xlvl: f32,
    pub iproj: i32,
    pub nx: usize,
    pub ny: usize,
    pub start_lat: f32,
    pub start_lon: f32,
    pub delta_lat: f32,
    pub delta_lon: f32,
    pub earth_radius_km: f32,
    pub polar: Option<PolarHeader>,
    /// `nx * ny` values, x fastest.
    pub values: Vec<f32>,
}

impl MetSlab {
    /// Value at the one-based `(i, j)` the Fortran indexes with.
    #[inline]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.values[(j - 1) * self.nx + (i - 1)]
    }
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
fn read_record<R: Read>(r: &mut R) -> MpasResult<Option<Vec<u8>>> {
    let mut head = [0u8; 4];
    match r.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(head) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload)?;
    let mut tail = [0u8; 4];
    r.read_exact(&mut tail)?;
    let trailer = u32::from_be_bytes(tail) as usize;
    if trailer != len {
        return Err(MpasError::Refusal(format!(
            "intermediate record is framed {len} bytes at its head and {trailer} at its tail; \
             the file is not a Fortran sequential-unformatted stream this reader can trust"
        )));
    }
    Ok(Some(payload))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, at: 0 }
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn take(&mut self, n: usize, what: &str) -> MpasResult<&'a [u8]> {
        if self.at + n > self.bytes.len() {
            return Err(MpasError::Refusal(format!(
                "intermediate record ran out reading {what}: wanted {n} byte(s) at offset {}, \
                 record is {} byte(s)",
                self.at,
                self.bytes.len()
            )));
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn text(&mut self, n: usize, what: &str) -> MpasResult<String> {
        let raw = self.take(n, what)?;
        Ok(String::from_utf8_lossy(raw).trim_end().to_string())
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn f32(&mut self, what: &str) -> MpasResult<f32> {
        let raw = self.take(4, what)?;
        Ok(f32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn i32(&mut self, what: &str) -> MpasResult<i32> {
        let raw = self.take(4, what)?;
        Ok(i32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }
}

/// Read every field in one intermediate file, in file order.
///
/// File order is essential downstream: the level table the case-7 code
/// builds is "first sighting wins", so re-ordering the slabs re-orders the
/// first-guess levels and moves every vertically interpolated number.
/// MPAS v8.4.2 core_init_atmosphere/mpas_init_atm_read_met.F:97-412.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
pub fn read_met_file(path: &Path) -> MpasResult<Vec<MetSlab>> {read_fields(path,None)}

/// Complete surface scan, retaining only fields consumed by the two native
/// passes. RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_surface.F:176-423
/// and mpas_init_atm_read_met.F:9-450 at 1d6e370e.
pub fn read_surface_file(path: &Path) -> MpasResult<Vec<MetSlab>> {
    let mut reader=MetReader::open(path,true)?;
    let mut result=Vec::new();
    while let Some((field,_))=reader.next_field()? {
        if matches!(field.field.as_str(),"LANDSEA"|"SKINTEMP"|"SST"|"SEAICE") {result.push(field);}
    }
    Ok(result)
}
/// MPAS v8.4.2 mpas_init_atm_read_met.F:97-412; case9 preserves original projection IDs.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
pub fn read_lbc_met_file(path: &Path) -> MpasResult<Vec<(MetSlab,i32)>> {
    Ok(read_met_file(path)?.into_iter().map(|s| {let iproj=s.iproj;(s,iproj)}).collect())
}
/// MPAS cases.F:7854-7961 stops reading once the first SOILHGT is consumed.
/// Later fields must not affect the boundary blend or its I/O acceptance.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
pub fn read_first_named_field(path:&Path,name:&str)->MpasResult<Option<MetSlab>> {
    Ok(read_fields(path,Some(name))?.into_iter().find(|field|field.field==name))
}
/// MPAS v8.4.2 mpas_init_atm_read_met.F:97-412 (all three intermediate formats).
pub struct MetReader { reader: BufReader<File>, path: std::path::PathBuf }
impl MetReader {
    /// MPAS-A v8.4.2 src/core_init_atmosphere/mpas_init_atm_read_met.F:97-412.
    /// Streams the audited decoder so monthly aerosol slabs do not accumulate.
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    pub fn open(path: &Path, _allow_gaussian: bool) -> MpasResult<Self> {
        Ok(Self { reader: BufReader::with_capacity(8 << 20, File::open(path)?), path: path.to_path_buf() })
    }
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    pub fn next_field(&mut self) -> MpasResult<Option<(MetSlab,i32)>> {
        let Some(version_record) = read_record(&mut self.reader)? else {
            return Ok(None);
        };
        let mut c = Cursor::new(&version_record);
        let version = c.i32("format version")?;
        if ![3,4,5].contains(&version) {
            return Err(MpasError::Refusal(format!("{} carries unrecognized intermediate format version {version}; native expects3,4 or5 (mpas_init_atm_read_met.F:404-405)",self.path.display())));
        }

        let header = read_record(&mut self.reader)?.ok_or_else(|| {
            MpasError::Refusal(format!(
                "{} ends after a version record, with no field header behind it",
                self.path.display()
            ))
        })?;
        let mut c = Cursor::new(&header);
        let hdate = c.text(24, "hdate")?;
        let xfcst = c.f32("xfcst")?;
        let map_source = if version==3 {String::new()} else {c.text(32, "map_source")?};
        let mut field = c.text(9, "field")?;
        let units = c.text(25, "units")?;
        let desc = c.text(46, "desc")?;
        let xlvl = c.f32("xlvl")?;
        let nx = c.i32("nx")?;
        let ny = c.i32("ny")?;
        let iproj = c.i32("iproj")?;

        // read_met's own aliasing, reproduced: some producers spell the
        // geopotential height field HGT and MPAS only knows it as GHT.
        if field == "HGT" {
            field = "GHT".to_string();
        }

        if nx <= 0 || ny <= 0 {
            return Err(MpasError::Refusal(format!(
                "field {field} declares a {nx} x {ny} slab"
            )));
        }
        let nx = nx as usize;
        let ny = ny as usize;

        if ![0,1,3,5].contains(&iproj) && !(version==5 && (iproj==4 || iproj==6)) {
            return Err(MpasError::Refusal(format!("field {field} has unrecognized projection code {iproj} for intermediate version {version}; its projection-record layout is not defined by native read_met")));
        }
        let proj=read_record(&mut self.reader)?.ok_or_else(||MpasError::Refusal(format!("field {field} has no projection record")))?;
        let mut c=Cursor::new(&proj);
        let startloc=if version==3 {"SWCORNER".into()} else {c.text(8,"startloc")?};
        let start_i=if startloc=="CENTER" {nx as f32/2.} else {1.};
        let start_j=if startloc=="CENTER" {ny as f32/2.} else {1.};
        let start_lat=c.f32("startlat")?.clamp(-90.,90.);
        let mut start_lon=c.f32("startlon")?;
        if start_lon>180. {start_lon-=360.;}
        let (mut dx,mut dy,mut stdlon,mut truelat1,mut truelat2)=(0.,0.,0.,0.,0.);
        let (mut center_lat,mut center_lon,mut pole_lat,mut pole_lon)=(0.,0.,0.,0.);
        let (delta_lat,delta_lon)=if iproj==0 || iproj==4 {
            (c.f32("deltalat")?,c.f32("deltalon")?)
        } else if iproj==6 {
            dx=c.f32("dx")?;dy=c.f32("dy")?;
            center_lat=c.f32("centerlat")?;center_lon=c.f32("centerlon")?;
            if center_lat>0. {pole_lat=90.-center_lat;pole_lon=180.;stdlon= -center_lon;}
            else {pole_lat=90.+center_lat;pole_lon=0.;stdlon=180.-center_lon;}
            // RRFS fork read_met.F:369-409 at 1d6e370e. Save the angular
            // increments before its unconditional METGRID spacing conversion.
            let increments=(dy,dx);
            dx*=1000.;dy*=1000.;
            increments
        } else {
            dx=c.f32("dx")?*1000.;dy=c.f32("dy")?*1000.;
            if iproj==1 {truelat1=c.f32("truelat1")?;}
            else {
                stdlon=c.f32("xlonc")?;truelat1=c.f32("truelat1")?;
                if iproj==3 {truelat2=c.f32("truelat2")?;}
            }
            (0.,0.)
        };
        if stdlon>180. {stdlon-=360.;}
        let earth_radius_km=if version==5 {c.f32("earth_radius")?} else {6370000f32/1000.};
        let polar=(iproj==IPROJ_POLAR).then_some(PolarHeader {dx,dy,stdlon,truelat:truelat1});
        let wind_grid_rel=if version==5 {
            let wind=read_record(&mut self.reader)?.ok_or_else(||MpasError::Refusal(format!("field {field} has no is_wind_grid_rel record")))?;
            Cursor::new(&wind).i32("is_wind_grid_rel")? != 0
        } else {true};

        let slab = read_record(&mut self.reader)?
            .ok_or_else(|| MpasError::Refusal(format!("field {field} has no data record")))?;
        let want = nx * ny * 4;
        if slab.len() != want {
            return Err(MpasError::Refusal(format!(
                "field {field} declares {nx} x {ny} but its data record is {} byte(s), not {want}",
                slab.len()
            )));
        }
        let mut values = Vec::with_capacity(nx * ny);
        for chunk in slab.chunks_exact(4) {
            values.push(f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        Ok(Some((MetSlab {
            center_lat,center_lon,pole_lat,pole_lon,
            format_version:version,xfcst,start_i,start_j,wind_grid_rel,
            dx,dy,stdlon,truelat1,truelat2,
            hdate,
            map_source,
            field,
            units,
            desc,
            xlvl,
            iproj,
            nx,
            ny,
            start_lat,
            start_lon,
            delta_lat,
            delta_lon,
            earth_radius_km,
            polar,
            values,
        }, iproj)))

    }
}
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
fn read_fields(path:&Path,stop:Option<&str>)->MpasResult<Vec<MetSlab>> {
    let mut reader=MetReader::open(path,true)?;
    let mut out=Vec::new();
    while let Some((field,_))=reader.next_field()? {
        let done=stop==Some(field.field.as_str());
        out.push(field);
        if done {break;}
    }
    if out.is_empty() {return Err(MpasError::Refusal(format!("{} holds no intermediate records",path.display())));}
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn record(payload: &[u8]) -> Vec<u8> {
        let n = payload.len() as u32;
        let mut out = n.to_be_bytes().to_vec();
        out.extend_from_slice(payload);
        out.extend_from_slice(&n.to_be_bytes());
        out
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn fixed(text: &str, width: usize) -> Vec<u8> {
        let mut out = text.as_bytes().to_vec();
        out.truncate(width);
        out.resize(width, b' ');
        out
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn one_field(name: &str, iproj: i32, nx: usize, ny: usize, values: &[f32]) -> Vec<u8> {
        let mut out = record(&WPS_FORMAT_VERSION.to_be_bytes());
        let mut header = Vec::new();
        header.extend_from_slice(&fixed("2025-03-14_12:00:00", 24));
        header.extend_from_slice(&0f32.to_be_bytes());
        header.extend_from_slice(&fixed("ECMWF", 32));
        header.extend_from_slice(&fixed(name, 9));
        header.extend_from_slice(&fixed("K", 25));
        header.extend_from_slice(&fixed("Temperature", 46));
        header.extend_from_slice(&200100f32.to_be_bytes());
        header.extend_from_slice(&(nx as i32).to_be_bytes());
        header.extend_from_slice(&(ny as i32).to_be_bytes());
        header.extend_from_slice(&iproj.to_be_bytes());
        out.extend_from_slice(&record(&header));

        let mut proj = Vec::new();
        proj.extend_from_slice(&fixed("SWCORNER", 8));
        proj.extend_from_slice(&(-90f32).to_be_bytes());
        proj.extend_from_slice(&0f32.to_be_bytes());
        proj.extend_from_slice(&0.25f32.to_be_bytes());
        proj.extend_from_slice(&0.25f32.to_be_bytes());
        proj.extend_from_slice(&6371.229f32.to_be_bytes());
        out.extend_from_slice(&record(&proj));
        out.extend_from_slice(&record(&0i32.to_be_bytes()));

        let mut data = Vec::new();
        for v in values {
            data.extend_from_slice(&v.to_be_bytes());
        }
        out.extend_from_slice(&record(&data));
        out
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    #[test]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn a_latlon_field_round_trips_out_of_the_container() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let path = write_temp(
            "rwmpas_metfile_latlon.bin",
            &one_field("TT", IPROJ_LATLON, 3, 2, &values),
        );
        let slabs = read_met_file(&path).unwrap();
        assert_eq!(slabs.len(), 1);
        let s = &slabs[0];
        assert_eq!(s.field, "TT");
        assert_eq!(s.map_source, "ECMWF");
        assert_eq!((s.nx, s.ny), (3, 2));
        // Fortran order: x fastest.  (2,1) is the second value.
        assert_eq!(s.at(2, 1), 2.0);
        assert_eq!(s.at(1, 2), 4.0);
        assert_eq!(s.xlvl, 200100.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn hgt_is_read_under_the_name_the_case_code_uses() {
        let path = write_temp(
            "rwmpas_metfile_hgt.bin",
            &one_field("HGT", IPROJ_LATLON, 2, 1, &[10.0, 20.0]),
        );
        let slabs = read_met_file(&path).unwrap();
        assert_eq!(slabs[0].field, "GHT");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn an_unknown_projection_record_layout_is_refused_by_code() {
        // Lambert conformal.  Reading its slab through the cylindrical
        // equidistant inverse would sample every cell at the wrong point and
        // say nothing about it.
        let path = write_temp(
            "rwmpas_metfile_lambert.bin",
            &one_field("TT", 99, 2, 1, &[1.0, 2.0]),
        );
        let err = read_met_file(&path).unwrap_err().to_string();
        assert!(err.contains("projection code 99"), "{err}");
        assert!(err.contains("layout is not defined"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_read_met.F:9-450 at 1d6e370e.
    fn a_mismatched_record_trailer_is_refused() {
        let mut bytes = one_field("TT", IPROJ_LATLON, 2, 1, &[1.0, 2.0]);
        let n = bytes.len();
        bytes[n - 1] ^= 0xff;
        let path = write_temp("rwmpas_metfile_torn.bin", &bytes);
        let err = read_met_file(&path).unwrap_err().to_string();
        assert!(err.contains("at its tail"), "{err}");
        let _ = std::fs::remove_file(&path);
    }
}
