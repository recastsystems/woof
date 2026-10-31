//! Projected-source fields to WPS MET, using the engine's exact donor entry.
//! Geometry is built from the engine's immutable projection metadata. No weather
//! field leaves Rust between raw-window read and the verified MET publication.

// The bridge crate's MET reader, shared so the writer here is read back by
// the same code every other MET producer is; only `inventory` is used.
#[path = "../../../../grib1_bridge/src/wps_intermediate.rs"]
#[allow(dead_code)]
mod wps_intermediate;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[repr(C)]
pub struct Geometry {
    pub source_ny: usize,
    pub source_nx: usize,
    pub ntarget: usize,
    pub donor_y: *const i32,
    pub donor_x: *const i32,
    pub fraction_y: *const f32,
    pub fraction_x: *const f32,
    pub nearest_y: *const i64,
    pub nearest_x: *const i64,
    pub sin_alpha: *const f64,
    pub cos_alpha: *const f64,
    pub interpolate: Option<Interpolate>,
}

// Bind the exact-donor entry of the selected engine library rather than
// linking a second preprocessing library with a second worker pool.
pub type Interpolate = unsafe extern "C" fn(*const f32, *const i32, *const i32,
    *const f32, *const f32, *mut f32, usize, usize, usize, usize, i32, usize) -> i32;

#[derive(Deserialize)]
struct ProjectionSeed {
    hemi: f64, cone: f64, rebydx: f64, polei: f64, polej: f64,
    truelat1: f64, stand_lon: f64, i_start: i64, j_start: i64,
    source_nx: usize, source_ny: usize, native_nx: usize, native_ny: usize,
    target: Target,
}

#[repr(C)]
pub struct GeometryOwner {
    geometry: Geometry,
    iy: Vec<i32>, ix: Vec<i32>, fy: Vec<f32>, fx: Vec<f32>,
    nearest_y: Vec<i64>, nearest_x: Vec<i64>, sin: Vec<f64>, cos: Vec<f64>,
}

fn wrap180(x:f64) -> f64 { if x>180.0 { x-360.0 } else if x< -180.0 {x+360.0} else {x} }

impl ProjectionSeed {
    fn points(&self) -> Result<Vec<(f64, f64)>, String> {
        let n = self.target.nx.checked_mul(self.target.ny).ok_or("target size overflow")?;
        if n == 0 || self.cone == 0.0 || self.rebydx <= 0.0
            || self.target.dlat <= 0.0 || self.target.dlon <= 0.0
            || [self.hemi,self.cone,self.rebydx,self.polei,self.polej,self.truelat1,self.stand_lon,
                self.target.south,self.target.west,self.target.dlat,self.target.dlon].iter().any(|x| !x.is_finite()) {
            return Err("invalid projection metadata would misplace the decoder crop".into());
        }
        let radians = std::f64::consts::PI / 180.0;
        let ctl1r = (self.truelat1 * radians).cos();
        let denominator = ((90.0 * self.hemi - self.truelat1) * radians / 2.0).tan();
        let mut points = Vec::with_capacity(n);
        for j in 0..self.target.ny { for i in 0..self.target.nx {
            let latitude = self.target.south + self.target.dlat * j as f64;
            let longitude = self.target.west + self.target.dlon * i as f64;
            let radius = self.rebydx * ctl1r / self.cone *
                (((90.0 * self.hemi - latitude) * radians / 2.0).tan() / denominator).powf(self.cone);
            let angle = self.cone * (wrap180(longitude - self.stand_lon) * radians);
            let x = self.hemi * (self.polei + self.hemi * radius * angle.sin()) - 1.0;
            let y = self.hemi * (self.polej - radius * angle.cos()) - 1.0;
            if !x.is_finite() || !y.is_finite() { return Err("target maps outside the projection domain".into()); }
            points.push((x, y));
        } }
        Ok(points)
    }

    fn build(&self, interpolate:Interpolate) -> Result<Box<GeometryOwner>,String> {
        let n=self.target.nx.checked_mul(self.target.ny).ok_or("target size overflow")?;
        let ns=self.source_nx.checked_mul(self.source_ny).ok_or("source size overflow")?;
        if self.source_nx<2 || self.source_ny<2 || n==0 || self.i_start<0 || self.j_start<0
            || self.source_nx>i32::MAX as usize || self.source_ny>i32::MAX as usize
            || self.i_start as usize+self.source_nx>self.native_nx || self.j_start as usize+self.source_ny>self.native_ny
            || [self.hemi,self.cone,self.rebydx,self.polei,self.polej,self.truelat1,self.stand_lon,
                self.target.south,self.target.west,self.target.dlat,self.target.dlon].iter().any(|x|!x.is_finite())
            || self.cone==0.0 || self.rebydx<=0.0 || self.target.dlat<=0.0 || self.target.dlon<=0.0 {
            return Err("invalid projection metadata would misplace every MET donor".into());
        }
        let mut owner=Box::new(GeometryOwner {
            geometry:Geometry {source_ny:self.source_ny,source_nx:self.source_nx,ntarget:n,
                donor_y:std::ptr::null(),donor_x:std::ptr::null(),fraction_y:std::ptr::null(),fraction_x:std::ptr::null(),
                nearest_y:std::ptr::null(),nearest_x:std::ptr::null(),sin_alpha:std::ptr::null(),cos_alpha:std::ptr::null(),interpolate:Some(interpolate)},
            iy:Vec::with_capacity(n),ix:Vec::with_capacity(n),fy:Vec::with_capacity(n),fx:Vec::with_capacity(n),
            nearest_y:Vec::with_capacity(n),nearest_x:Vec::with_capacity(n),sin:Vec::with_capacity(ns),cos:Vec::with_capacity(ns),
        });
        for (global_x, global_y) in self.points()? {
            let global_ix=global_x.floor() as i64;
            let global_iy=global_y.floor() as i64;
            let x=global_ix-self.i_start; let y=global_iy-self.j_start;
            if x<0 || y<0 || x+1>=self.source_nx as i64 || y+1>=self.source_ny as i64 {
                return Err("projected window lacks the bilinear donor cell".into());
            }
            let short=[(x<1,self.i_start==0),(x+2>=self.source_nx as i64,self.i_start as usize+self.source_nx==self.native_nx),
                (y<1,self.j_start==0),(y+2>=self.source_ny as i64,self.j_start as usize+self.source_ny==self.native_ny)];
            if short.iter().any(|(missing,native_edge)|*missing && !*native_edge) { return Err("projected window lacks the four-point interpolation halo".into()); }
            owner.ix.push(x as i32); owner.iy.push(y as i32);
            // The reference first subtracts the window origin, then restores
            // it before forming the FP32 fraction. Keep that exact order.
            owner.fx.push(((global_x-self.i_start as f64)+self.i_start as f64-global_ix as f64) as f32);
            owner.fy.push(((global_y-self.j_start as f64)+self.j_start as f64-global_iy as f64) as f32);
            owner.nearest_x.push((global_x+0.5).floor() as i64-self.i_start);
            owner.nearest_y.push((global_y+0.5).floor() as i64-self.j_start);
        }
        let degrees=180.0/std::f64::consts::PI;
        for j in 0..self.source_ny { for i in 0..self.source_nx {
            let x=(self.i_start+1+i as i64) as f64;
            let y=(self.j_start+1+j as i64) as f64;
            let xx=self.hemi*x-self.polei;
            let yy=self.polej-self.hemi*y;
            let longitude=self.stand_lon+degrees*(self.hemi*xx).atan2(yy)/self.cone;
            let mut longitude=(longitude+360.0).rem_euclid(360.0);
            if xx*xx+yy*yy==0.0 {longitude=(self.stand_lon+360.0).rem_euclid(360.0);}
            longitude=wrap180(longitude);
            let difference=wrap180(self.stand_lon-longitude);
            let alpha=self.hemi*self.cone*std::f64::consts::PI/180.0*difference;
            owner.sin.push(alpha.sin()); owner.cos.push(alpha.cos());
        } }
        owner.geometry.donor_y=owner.iy.as_ptr(); owner.geometry.donor_x=owner.ix.as_ptr();
        owner.geometry.fraction_y=owner.fy.as_ptr(); owner.geometry.fraction_x=owner.fx.as_ptr();
        owner.geometry.nearest_y=owner.nearest_y.as_ptr(); owner.geometry.nearest_x=owner.nearest_x.as_ptr();
        owner.geometry.sin_alpha=owner.sin.as_ptr(); owner.geometry.cos_alpha=owner.cos.as_ptr();
        Ok(owner)
    }
}

#[derive(Deserialize)]
struct WindowRequest {
    #[serde(flatten)] seed: ProjectionSeed,
    halo: i64,
    extra_cells: i64,
}

/// # Safety
/// JSON and error buffers are valid for their lengths; `bounds` has room for
/// four i64 values in i_start, i_end, j_start, j_end order. No field is read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_projected_window_v1(
    json: *const u8, len: usize, bounds: *mut i64, error: *mut u8, error_cap: usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| -> Result<[i64; 4], String> {
        if json.is_null() || bounds.is_null() { return Err("null projected window metadata or output".into()); }
        let request: WindowRequest = serde_json::from_slice(unsafe {std::slice::from_raw_parts(json, len)})
            .map_err(|e| e.to_string())?;
        let limit = request.seed.native_nx.max(request.seed.native_ny);
        if limit == 0 || limit > i32::MAX as usize || request.halo < 0 || request.extra_cells < 0
            || request.halo as usize > limit || request.extra_cells as usize > limit {
            return Err("invalid native dimensions or halo would omit interpolation donors".into());
        }
        let points = request.seed.points()?;
        let min_x = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = points.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = points.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        // Every target point needs its bilinear donor cell (x and x+1) on the native grid.
        if min_x < 0.0 || min_y < 0.0 || max_x >= request.seed.native_nx as f64 - 1.0
            || max_y >= request.seed.native_ny as f64 - 1.0 {
            return Err("target points lie outside native source coverage".into());
        }
        // The halo and the extra cells widen the crop up to the native grid's edge and stop there.  At the
        // edge the geometry interpolates bilinearly (ProjectionSeed::build's native_edge rule), so a short halo
        // there is the source's own boundary.  THE BREAKAGE (hex HRRR start, Vermont 2023-07-10, the NE 3 km cut
        // of conus3km, whose SE corner sits 41 km inside HRRR's east edge): the window asked for column 1800 of
        // 0..1798 and refused a target whose every point had donors, and the box narrowed to get past it left
        // 1,750 mesh cells without meteorology.
        let (nx, ny) = (request.seed.native_nx as i64, request.seed.native_ny as i64);
        let b = [(min_x.floor() as i64 - 1 - request.extra_cells).max(0),
                 (max_x.floor() as i64 + request.halo + request.extra_cells).min(nx - 1),
                 (min_y.floor() as i64 - 1 - request.extra_cells).max(0),
                 (max_y.floor() as i64 + request.halo + request.extra_cells).min(ny - 1)];
        Ok(b)
    });
    match result {
        Ok(Ok(b)) => { unsafe {std::ptr::copy_nonoverlapping(b.as_ptr(), bounds, 4)}; 0 }
        Ok(Err(e)) => { write_error(error, error_cap, &e); 1 }
        Err(_) => { write_error(error, error_cap, "projected window native panic"); 2 }
    }
}

fn write_error(error:*mut u8, error_cap:usize, message:&str) {
    if !error.is_null() && error_cap>0 {let n=message.len().min(error_cap-1); unsafe {std::ptr::copy_nonoverlapping(message.as_ptr(),error,n); *error.add(n)=0;}}
}

/// # Safety
/// `seed` has `seed_len` readable JSON bytes, `error` has `error_cap` writable
/// bytes, and `interpolate` is the selected engine's exact-donor C entry, kept
/// loaded until the returned owner is released. Null means failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_projected_geometry_v1(seed:*const u8,seed_len:usize,interpolate:Option<Interpolate>,error:*mut u8,error_cap:usize) -> *mut GeometryOwner {
    let result=std::panic::catch_unwind(|| -> Result<Box<GeometryOwner>,String> {
        if seed.is_null() || interpolate.is_none() {return Err("null projection metadata or exact-donor entry".into());}
        let seed:ProjectionSeed=serde_json::from_slice(unsafe {std::slice::from_raw_parts(seed,seed_len)}).map_err(|e|e.to_string())?;
        seed.build(interpolate.unwrap())
    });
    let message=match result {Ok(Ok(owner))=>return Box::into_raw(owner),Ok(Err(e))=>e,Err(_)=>"projected geometry native panic".into()};
    write_error(error,error_cap,&message);std::ptr::null_mut()
}

/// # Safety
/// `owner` is null or a still-live handle returned by the constructor above.
/// Release exactly once after every synchronous MET call using it returns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_projected_geometry_release_v1(owner:*mut GeometryOwner) {
    if !owner.is_null() {drop(unsafe {Box::from_raw(owner)});}
}

#[derive(Deserialize)]
struct Target { south: f64, west: f64, dlat: f64, dlon: f64, nx: usize, ny: usize }
#[derive(Deserialize)]
struct Field { source: String, field: String, units: String, description: String, method: String, transform: String }
#[derive(Deserialize)]
struct Soil { temperature: String, moisture: String, nodes: Vec<f64>, midpoints: Vec<f64>, names: Vec<[String; 2]>, #[serde(default)] preserve_nodes: bool,
    /// The (temperature, moisture) record names the preserved nodes are written under, one pair per node, when the
    /// source row names them (HRRR: the RAP/HRRR Vtable's SOILT000..SOILT300 / SOILM000..SOILM300, which the RRFS
    /// fork's case 7 reads as RUC's nine levels).  Absent: the neutral ST{cm}{cm} / SM{cm}{cm} names.
    #[serde(default)] node_names: Option<Vec<[String; 2]>> }
#[derive(Deserialize)]
struct Admission { land: String, ice: String, soil_moisture: String, soil_temperature: String, humidity: Vec<String> }
#[derive(Deserialize)]
struct Job {
    output: PathBuf, receipt: PathBuf, valid_time: String, forecast_hour: f32,
    map_source: String, target: Target, levels: usize, workers: usize,
    sources: BTreeMap<String, PathBuf>, atmosphere: Vec<Field>, surface: Vec<Field>,
    wind_3d: [String; 2], wind_10m: [String; 2], soil: Soil, admission: Admission,
}

struct Plan<'a> {
    ny: usize, nx: usize, iy: &'a [i32], ix: &'a [i32], fy: &'a [f32], fx: &'a [f32],
    nearest: Vec<usize>, sin: &'a [f64], cos: &'a [f64], workers: usize,
    interpolate: Interpolate,
}
impl Plan<'_> {
    fn apply(&self, source: &[f32], method: &str) -> Result<Vec<f32>, String> {
        let nsource = self.ny * self.nx;
        if source.is_empty() || !source.len().is_multiple_of(nsource) { return Err("source field does not match the window".into()); }
        let nlead = source.len() / nsource;
        let n = self.iy.len();
        let mut out = vec![0.0; nlead * n];
        if method == "nearest" {
            for k in 0..nlead { for i in 0..n { out[k*n+i] = source[k*nsource+self.nearest[i]]; } }
            return Ok(out);
        }
        let method_id = match method { "bilinear" => 1, "parabolic" => 2, _ => return Err(format!("unknown interpolation method {method}")) };
        let code = unsafe { (self.interpolate)(source.as_ptr(), self.iy.as_ptr(), self.ix.as_ptr(),
            self.fy.as_ptr(), self.fx.as_ptr(), out.as_mut_ptr(), nlead, self.ny, self.nx, n, method_id, self.workers) };
        if code != 0 { return Err(format!("exact-donor interpolation refused with code {code}")); }
        // The projected plan uses four-point bilinear where the native edge
        // cannot hold a sixteen-point stencil. Keep the engine's order.
        if method_id == 2 {
            for k in 0..nlead { for i in 0..n {
                let (y,x) = (self.iy[i] as usize, self.ix[i] as usize);
                if y == 0 || x == 0 || y+2 >= self.ny || x+2 >= self.nx {
                    let p = k*nsource + y*self.nx+x;
                    let (fy,fx) = (self.fy[i], self.fx[i]);
                    let lower = (1.0-fx)*source[p] + fx*source[p+1];
                    let upper = (1.0-fx)*source[p+self.nx] + fx*source[p+self.nx+1];
                    out[k*n+i] = (1.0-fy)*lower + fy*upper;
                }
            } }
        }
        Ok(out)
    }
    fn wind(&self, u: &[f32], v: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        if u.len() != v.len() { return Err("wind components have different shapes".into()); }
        let n = self.ny*self.nx;
        let mut east = Vec::with_capacity(u.len());
        let mut north = Vec::with_capacity(u.len());
        for i in 0..u.len() {
            let (sin,cos) = (self.sin[i%n] as f32, self.cos[i%n] as f32);
            // Each multiplication rounds separately, as in the engine FP32
            // array expressions. Do not replace with mul_add or fast-math.
            east.push(u[i]*cos - v[i]*sin);
            north.push(v[i]*cos + u[i]*sin);
        }
        Ok((self.apply(&east,"parabolic")?, self.apply(&north,"parabolic")?))
    }
    fn soil(&self, values: &[f32], soil: &Soil) -> Result<Vec<Vec<f32>>, String> {
        let n = self.ny*self.nx;
        if values.len() != soil.nodes.len()*n { return Err("soil node field does not match the declared node table".into()); }
        soil.midpoints.iter().map(|z| {
            let upper = soil.nodes.partition_point(|depth| depth < z);
            if upper == 0 || upper == soil.nodes.len() { return Err("soil midpoint has no source bracket".into()); }
            let lower = upper-1;
            let weight = (z-soil.nodes[lower])/(soil.nodes[upper]-soil.nodes[lower]);
            // FP64 interpolation followed by a single FP32 conversion, then
            // nearest movement. Avoid materialising whole source layers.
            Ok(self.nearest.iter().map(|&i| {
                let a = values[lower*n+i] as f64;
                (a + weight*(values[upper*n+i] as f64-a)) as f32
            }).collect())
        }).collect()
    }
}

fn read_field(path: &Path, length: usize) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}",path.display()))?;
    if bytes.len() != length.checked_mul(4).ok_or("field size overflow")? { return Err(format!("{} has {} bytes, expected {}",path.display(),bytes.len(),length*4)); }
    Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
}
fn get<'a>(fields: &'a BTreeMap<String, Vec<f32>>, name: &str) -> Result<&'a [f32], String> {
    fields.get(name).map(|v| v.as_slice()).ok_or_else(|| format!("missing source field {name}"))
}
fn admit(fields: &BTreeMap<String,Vec<f32>>, rule: &Admission, n: usize) -> Result<(),String> {
    let tolerance = 4.0*f32::EPSILON;
    for name in [&rule.land, &rule.soil_moisture] {
        if get(fields,name)?.iter().any(|x| !x.is_finite() || *x < -tolerance || *x > 1.0+tolerance) {
            return Err(format!("source {name} is non-finite or outside 0..1 beyond decode roundoff"));
        }
    }
    let temperature = get(fields,&rule.soil_temperature)?;
    if temperature.iter().any(|x| !x.is_finite()) { return Err(format!("source {} is non-finite",rule.soil_temperature)); }
    let land = get(fields,&rule.land)?;
    let ice = get(fields,&rule.ice)?;
    for i in 0..n {
        if land[i] < 0.5 && !matches!(ice[i].partial_cmp(&0.5), Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)) && temperature.iter().skip(i).step_by(n).any(|x| *x<170.0 || *x>400.0) {
            return Err("source soil temperature is outside 170..400 K on open water; this is a broken soil record, not a land or ice column".into());
        }
    }
    for name in &rule.humidity {
        if get(fields,name)?.iter().any(|x| !x.is_finite() || *x<0.0 || *x>0.1) { return Err(format!("source {name} is non-finite or outside 0..0.1")); }
    }
    Ok(())
}
fn transform(values: &mut [f32], kind: &str) -> Result<(),String> {
    for value in values {
        *value = match kind {
            "none" => *value,
            "nonnegative" => if *value <= 0.0 { 0.0 } else { *value },
            "land" => if *value >= 0.5 { 1.0 } else { 0.0 },
            "fraction" => value.clamp(0.0,1.0),
            _ => return Err(format!("unknown field transform {kind}")),
        };
    }
    Ok(())
}
fn text(out: &mut Vec<u8>, value: &str, width: usize) -> Result<(),String> {
    if !value.is_ascii() || value.len()>width { return Err(format!("{value:?} does not fit a {width}-character WPS field")); }
    out.extend_from_slice(value.as_bytes()); out.resize(out.len()+width-value.len(),b' '); Ok(())
}
fn record<W:Write>(out: &mut W, payload: &[u8]) -> Result<(),String> {
    let size = i32::try_from(payload.len()).map_err(|_| "WPS record exceeds i32 length")?.to_be_bytes();
    out.write_all(&size).and_then(|_|out.write_all(payload)).and_then(|_|out.write_all(&size)).map_err(|e|e.to_string())
}
fn write_field<W:Write>(out:&mut W, job:&Job, name:&str, units:&str, description:&str, level:f32, values:&[f32]) -> Result<(),String> {
    let target = &job.target;
    if values.len() != target.nx*target.ny { return Err(format!("{name} slab does not match the target")); }
    if values.iter().any(|x| !x.is_finite()) { return Err(format!("{name} at level {level} carries a non-finite value; refusing a broken MET field")); }
    let mut header = Vec::new();
    text(&mut header,&job.valid_time,24)?; header.extend(job.forecast_hour.to_be_bytes());
    text(&mut header,&job.map_source,32)?; text(&mut header,name,9)?;
    text(&mut header,units,25)?; text(&mut header,description,46)?;
    header.extend(level.to_be_bytes()); header.extend((target.nx as i32).to_be_bytes());
    header.extend((target.ny as i32).to_be_bytes()); header.extend(0i32.to_be_bytes());
    let mut projection = Vec::new(); text(&mut projection,"SWCORNER",8)?;
    for value in [target.south,target.west,target.dlat,target.dlon,6371.229] { projection.extend((value as f32).to_be_bytes()); }
    let slab:Vec<u8> = values.iter().flat_map(|x|x.to_be_bytes()).collect();
    record(out,&5i32.to_be_bytes())?; record(out,&header)?; record(out,&projection)?;
    record(out,&0i32.to_be_bytes())?; record(out,&slab)
}

fn run(job: &Job, plan:&Plan<'_>) -> Result<(),String> {
    let nsource = plan.ny*plan.nx;
    let n = plan.iy.len();
    if job.levels == 0 || job.soil.names.len() != job.soil.midpoints.len() || job.soil.nodes.len()<2
        || job.soil.nodes.windows(2).any(|v| !v[0].is_finite() || v[0]>=v[1])
        || !job.soil.nodes.last().unwrap().is_finite() || job.soil.midpoints.iter().any(|x|!x.is_finite()) {
        return Err("invalid level or soil table would mislabel MET layers".into());
    }
    let mut fields = BTreeMap::new();
    for (name,path) in &job.sources {
        let levels = if job.atmosphere.iter().any(|f| f.source==*name) || job.wind_3d.contains(name) { job.levels }
            else if *name==job.soil.temperature || *name==job.soil.moisture { job.soil.nodes.len() } else { 1 };
        fields.insert(name.clone(),read_field(path,levels.checked_mul(nsource).ok_or("source size overflow")?)?);
    }
    admit(&fields,&job.admission,nsource)?;
    let mut atmosphere = Vec::new();
    for f in &job.atmosphere {
        let mut values = plan.apply(get(&fields,&f.source)?,&f.method)?;
        transform(&mut values,&f.transform)?; atmosphere.push(values);
    }
    let (u,v) = plan.wind(get(&fields,&job.wind_3d[0])?,get(&fields,&job.wind_3d[1])?)?;
    let partial = job.output.with_extension("partial");
    let mut out = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(&partial).map_err(|e|format!("{}: {e}",partial.display()))?);
    let mut counts:BTreeMap<String,usize> = BTreeMap::new();
    let mut emit = |name:&str, units:&str, description:&str, level:f32, values:&[f32]| -> Result<(),String> {
        write_field(&mut out,job,name,units,description,level,values)?;
        *counts.entry(name.into()).or_default()+=1; Ok(())
    };
    for k in 0..job.levels {
        for (f,values) in job.atmosphere.iter().zip(&atmosphere) { emit(&f.field,&f.units,&f.description,(k+1) as f32,&values[k*n..(k+1)*n])?; }
        emit("UU","m s-1","U",(k+1) as f32,&u[k*n..(k+1)*n])?;
        emit("VV","m s-1","V",(k+1) as f32,&v[k*n..(k+1)*n])?;
    }
    for f in &job.surface {
        let mut values = plan.apply(get(&fields,&f.source)?,&f.method)?;
        transform(&mut values,&f.transform)?; emit(&f.field,&f.units,&f.description,200100.0,&values)?;
    }
    let (u,v) = plan.wind(get(&fields,&job.wind_10m[0])?,get(&fields,&job.wind_10m[1])?)?;
    emit("UU","m s-1","U",200100.0,&u)?; emit("VV","m s-1","V",200100.0,&v)?;
    let temperature = plan.soil(get(&fields,&job.soil.temperature)?,&job.soil)?;
    let mut moisture = plan.soil(get(&fields,&job.soil.moisture)?,&job.soil)?;
    for (i,[t,m]) in job.soil.names.iter().enumerate() {
        if t.len()!=8 || m.len()!=8 || !t.is_ascii() || !m.is_ascii() { return Err("soil field name does not encode its layer bounds".into()); }
        emit(t,"K",&format!("T {}-{} cm below ground layer (Upper)",&t[2..5],&t[5..8]),200100.0,&temperature[i])?;
        transform(&mut moisture[i],"fraction")?;
        emit(m,"m3 m-3",&format!("Soil moisture of {}-{} cm ground layer",&m[2..5],&m[5..8]),200100.0,&moisture[i])?;
    }
    if let Some(names) = &job.soil.node_names {
        if names.len() != job.soil.nodes.len() || names.iter().flatten().any(|name| name.is_empty() || name.len() > 9 || !name.is_ascii()) {
            return Err("soil node names must give one WPS (temperature, moisture) pair per source node".into());
        }
    }
    if job.soil.preserve_nodes {
        for (k, depth) in job.soil.nodes.iter().enumerate() {
            let cm = (depth * 100.0).round() as usize;
            if cm > 999 || (*depth * 100.0 - cm as f64).abs() > 1e-6 {
                return Err("source node depth cannot be encoded in WPS centimetres".into());
            }
            for (pair, (source, prefix, units)) in [(&job.soil.temperature, "ST", "K"), (&job.soil.moisture, "SM", "m3 m-3")].into_iter().enumerate() {
                let values = get(&fields, source)?;
                let mut slab = plan.apply(&values[k*plan.ny*plan.nx..(k+1)*plan.ny*plan.nx], "nearest")?;
                // Named nodes are a first guess a producer reads, so their moisture is held to
                // 0..1 like the layers; the neutral names keep their bytes.
                if pair == 1 && job.soil.node_names.is_some() { transform(&mut slab, "fraction")?; }
                let name = match &job.soil.node_names {
                    Some(names) => names[k][pair].clone(),
                    None => format!("{prefix}{cm:03}{cm:03}"),
                };
                emit(&name, units, "Source soil value at depth node", 200100.0, &slab)?;
            }
        }
    }
    out.flush().map_err(|e|e.to_string())?; drop(out);
    let (records,points) = wps_intermediate::inventory(&partial).map_err(|e|e.to_string())?;
    if records != counts.values().sum::<usize>() || points!=records*n { return Err("Rust MET writer and reader disagree".into()); }
    // Hash through a bounded buffer; no second full MET image in RAM.
    let mut file = File::open(&partial).map_err(|e|e.to_string())?;
    let mut sha = Sha256::new(); let mut buffer = vec![0u8;1<<20];
    loop { let bytes=file.read(&mut buffer).map_err(|e|e.to_string())?; if bytes==0 {break;} sha.update(&buffer[..bytes]); }
    let bytes = file.metadata().map_err(|e|e.to_string())?.len(); drop(file);
    if job.output.exists() { return Err(format!("{} exists; refusing to replace a completed MET hour",job.output.display())); }
    fs::rename(&partial,&job.output).map_err(|e|e.to_string())?;
    let receipt = serde_json::json!({"path":job.output,"bytes":bytes,"sha256":format!("{:x}",sha.finalize()),
        "valid_time":job.valid_time,"forecast_hour":job.forecast_hour,"field_records":records,"fields":counts,
        "record_endian":"big","levels":job.levels,"operator":"rust-exact-donor-projected-v1"});
    fs::write(&job.receipt,serde_json::to_vec(&receipt).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
}

/// # Safety
/// The C caller owns valid, non-overlapping contiguous geometry buffers of the
/// declared lengths until this synchronous call returns. `error` has `error_cap`
/// writable bytes; `job` has `job_len` readable UTF-8 bytes. A failure never emits
/// a ready marker. Panic is caught before it reaches the foreign caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_projected_intermediate_v1(job:*const u8, job_len:usize, geometry:*const Geometry, error:*mut u8, error_cap:usize) -> i32 {
    let result = std::panic::catch_unwind(|| -> Result<(),String> {
        if job.is_null() || geometry.is_null() { return Err("null projected intermediate job or geometry".into()); }
        let job:Job = serde_json::from_slice(unsafe {std::slice::from_raw_parts(job,job_len)}).map_err(|e|e.to_string())?;
        let g = unsafe { &*geometry };
        let source_len = g.source_nx.checked_mul(g.source_ny).ok_or("source size overflow")?;
        if g.source_nx<2 || g.source_ny<2 || g.ntarget==0 || job.workers==0 || g.interpolate.is_none()
            || job.target.nx.checked_mul(job.target.ny)!=Some(g.ntarget)
            || job.target.nx>i32::MAX as usize || job.target.ny>i32::MAX as usize
            || [g.donor_y.cast::<()>(),g.donor_x.cast(),g.fraction_y.cast(),g.fraction_x.cast(),g.nearest_y.cast(),g.nearest_x.cast(),g.sin_alpha.cast(),g.cos_alpha.cast()].iter().any(|p|p.is_null()) {
            return Err("invalid projected dimensions or null plan buffer".into());
        }
        let iy = unsafe {std::slice::from_raw_parts(g.donor_y,g.ntarget)};
        let ix = unsafe {std::slice::from_raw_parts(g.donor_x,g.ntarget)};
        let fy = unsafe {std::slice::from_raw_parts(g.fraction_y,g.ntarget)};
        let fx = unsafe {std::slice::from_raw_parts(g.fraction_x,g.ntarget)};
        let nearest_y = unsafe {std::slice::from_raw_parts(g.nearest_y,g.ntarget)};
        let nearest_x = unsafe {std::slice::from_raw_parts(g.nearest_x,g.ntarget)};
        let mut nearest=Vec::with_capacity(g.ntarget);
        for i in 0..g.ntarget {
            if iy[i]<0 || ix[i]<0 || iy[i] as usize+1>=g.source_ny || ix[i] as usize+1>=g.source_nx
                || nearest_y[i]<0 || nearest_x[i]<0 || nearest_y[i] as usize>=g.source_ny || nearest_x[i] as usize>=g.source_nx
                || !fy[i].is_finite() || !fx[i].is_finite() || !(0.0..=1.0).contains(&fy[i]) || !(0.0..=1.0).contains(&fx[i]) {
                return Err("projected plan lacks a donor cell; prevents out-of-window MET reads".into());
            }
            nearest.push(nearest_y[i] as usize*g.source_nx+nearest_x[i] as usize);
        }
        let plan=Plan {ny:g.source_ny,nx:g.source_nx,iy,ix,fy,fx,nearest,
            sin:unsafe {std::slice::from_raw_parts(g.sin_alpha,source_len)},cos:unsafe {std::slice::from_raw_parts(g.cos_alpha,source_len)},workers:job.workers,interpolate:g.interpolate.unwrap()};
        run(&job,&plan)
    });
    let message=match result {Ok(Ok(()))=>return 0, Ok(Err(e))=>e, Err(_)=>"projected intermediate native panic".into()};
    write_error(error,error_cap,&message);
    1
}

/// Release-bundle handshake for the file-backed projected MET entry.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_projected_intermediate_abi_version() -> u32 { 1 }

/// The release-source stamp, `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`
/// (`build.rs` injects it). Exported so the cdylib keeps the literal the
/// release cut reads out of the bytes.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_projected_intermediate_source_revision() -> *const u8 {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"), "\0");
    SOURCE_REV_STAMP.as_ptr()
}

#[cfg(test)]
mod tests {
    //! Contract tests of this crate. They need no engine library: the
    //! exact-donor entry is a recording stub here, so the gate compiles and
    //! runs on every runner. Byte identity of whole MET files against the
    //! engine's own operator is gpuwm-hex's test_projected_intermediate_native.
    use super::*;
    use std::sync::Mutex;

    type RecordedCall = (usize, usize, usize, usize, i32, usize);
    static CALLS: Mutex<Vec<RecordedCall>> = Mutex::new(Vec::new());

    /// Records its arguments and writes 100 plus the bilinear value, so an
    /// interior parabolic point shows the engine's answer was kept.
    unsafe extern "C" fn stub(source: *const f32, iy: *const i32, ix: *const i32,
        fy: *const f32, fx: *const f32, out: *mut f32, nlead: usize, ny: usize, nx: usize,
        n: usize, method: i32, workers: usize) -> i32 {
        CALLS.lock().unwrap().push((nlead, ny, nx, n, method, workers));
        unsafe {
            for k in 0..nlead { for i in 0..n {
                let (y, x) = (*iy.add(i) as usize, *ix.add(i) as usize);
                let (wy, wx) = (*fy.add(i), *fx.add(i));
                let p = k*ny*nx + y*nx + x;
                let lower = (1.0-wx)*(*source.add(p)) + wx*(*source.add(p+1));
                let upper = (1.0-wx)*(*source.add(p+nx)) + wx*(*source.add(p+nx+1));
                *out.add(k*n+i) = 100.0 + (1.0-wy)*lower + wy*upper;
            } }
        }
        0
    }

    fn plan() -> Plan<'static> {
        // Point 0 sits inside the 16-point stencil (engine answer kept);
        // point 1 sits on the window edge (four-point bilinear override).
        Plan {ny:4,nx:4,iy:&[1,0],ix:&[1,0],fy:&[0.5,0.5],fx:&[0.5,0.5],nearest:vec![10,5],
              sin:&[0.0;16],cos:&[1.0;16],workers:2,interpolate:stub}
    }

    fn hrrr_seed(extra: &[(&str, serde_json::Value)]) -> String {
        // The HRRR Lambert grid's constants, as gpuwm.ingest.hrrr.hrrr_source_grid states them.
        let mut value = serde_json::json!({"hemi":1.0,"cone":0.6225146366376196,
            "rebydx":2123.743,"polei":900.1733808406442,"polej":3200.012920475258,"truelat1":38.5,
            "stand_lon":-97.5,"i_start":0,"j_start":0,"source_nx":1799,"source_ny":1059,
            "native_nx":1799,"native_ny":1059,
            "target":{"south":38.0,"west":-98.0,"dlat":0.05,"dlon":0.05,"nx":4,"ny":3}});
        for (key, v) in extra { value[*key] = v.clone(); }
        value.to_string()
    }

    #[test] fn interior_points_keep_the_engine_answer_and_edges_use_bilinear() {
        let values:Vec<f32>=(0..16).map(|i|i as f32+1.0).collect();
        let out = plan().apply(&values,"parabolic").unwrap();
        assert_eq!(out, vec![108.5, 3.5]);
        assert!(CALLS.lock().unwrap().contains(&(1, 4, 4, 2, 2, 2)));
        assert_eq!(plan().apply(&values,"nearest").unwrap(), vec![11.0,6.0]);
        assert!(plan().apply(&values,"cubic").is_err());
        assert!(plan().apply(&values[..15],"parabolic").is_err());
    }

    #[test] fn soil_interpolates_in_f64_before_the_nearest_gather() {
        let soil=Soil {temperature:"T".into(),moisture:"M".into(),nodes:vec![0.0,0.10,0.30],midpoints:vec![0.05,0.25],names:vec![],preserve_nodes:false,node_names:None};
        let values:Vec<f32>=(0..48).map(|i|280.0+i as f32*0.01).collect();
        let output=plan().soil(&values,&soil).unwrap();
        for (k,w,lower) in [(0,0.5,0),(1,0.75,1)] { for (i,donor) in [10,5].iter().enumerate() {
            let a=values[lower*16+donor] as f64; let b=values[(lower+1)*16+donor] as f64;
            assert_eq!(output[k][i].to_bits(),((a+w*(b-a)) as f32).to_bits());
        } }
        let outside=Soil {midpoints:vec![0.5],..soil};
        assert!(plan().soil(&values,&outside).is_err());
    }

    #[test] fn winds_rotate_with_separately_rounded_products() {
        let mut p = plan();
        p.sin=Box::leak(Box::new([0.6f64;16])); p.cos=Box::leak(Box::new([0.8f64;16]));
        let u=vec![3.0f32;16]; let v=vec![4.0f32;16];
        let (east,north)=p.wind(&u,&v).unwrap();
        let (s, c) = (0.6f64 as f32, 0.8f64 as f32);
        let e=3.0f32*c-4.0f32*s; let n=4.0f32*c+3.0f32*s;
        // Point 1 is an edge point: bilinear of a constant field is exact.
        assert_eq!(east[1].to_bits(), e.to_bits());
        assert_eq!(north[1].to_bits(), n.to_bits());
    }

    #[test] fn the_crop_window_brackets_the_target_with_its_halo() {
        let request=hrrr_seed(&[("halo",serde_json::json!(2)),("extra_cells",serde_json::json!(2))]);
        let mut bounds=[0i64;4]; let mut error=[0u8;256];
        let code=unsafe {gpuwm_projected_window_v1(request.as_ptr(),request.len(),bounds.as_mut_ptr(),error.as_mut_ptr(),error.len())};
        assert_eq!(code,0, "{}", String::from_utf8_lossy(&error));
        let seed:ProjectionSeed=serde_json::from_str(&hrrr_seed(&[])).unwrap();
        let points=seed.points().unwrap();
        // The engine's own projection (hrrr_source_grid().latlon_to_ij,
        // one-based) puts 38.0 N 98.0 W at 885.568573269496, 511.60833226685736.
        assert!((points[0].0-884.568573269496).abs()<1e-9 && (points[0].1-510.60833226685736).abs()<1e-9,
                "{:?}", points[0]);
        let min_x=points.iter().map(|p|p.0).fold(f64::INFINITY,f64::min);
        let max_y=points.iter().map(|p|p.1).fold(f64::NEG_INFINITY,f64::max);
        assert_eq!(bounds[0], min_x.floor() as i64-3);
        assert_eq!(bounds[3], max_y.floor() as i64+4);
        assert!(bounds[0]<bounds[1] && bounds[2]<bounds[3]);
        // A target whose halo reaches past the native edge takes the crop up to the edge (bilinear there).
        let east=hrrr_seed(&[("halo",serde_json::json!(2)),("extra_cells",serde_json::json!(2)),
            ("target",serde_json::json!({"south":38.27,"west":-66.105,"dlat":0.025,"dlon":0.025,"nx":3,"ny":2}))]);
        let code=unsafe {gpuwm_projected_window_v1(east.as_ptr(),east.len(),bounds.as_mut_ptr(),error.as_mut_ptr(),error.len())};
        assert_eq!(code,0, "{}", String::from_utf8_lossy(&error));
        let seed:ProjectionSeed=serde_json::from_str(&east).unwrap();
        let east_x=seed.points().unwrap().iter().map(|p|p.0).fold(f64::NEG_INFINITY,f64::max);
        assert!(east_x+4.0>seed.native_nx as f64-1.0, "the test target must reach the edge: {east_x}");
        assert_eq!(bounds[1], seed.native_nx as i64-1);
        // A target off the grid is refused by name, never clipped.
        let outside=hrrr_seed(&[("halo",serde_json::json!(2)),("extra_cells",serde_json::json!(2)),
            ("target",serde_json::json!({"south":5.0,"west":-98.0,"dlat":0.05,"dlon":0.05,"nx":4,"ny":3}))]);
        let code=unsafe {gpuwm_projected_window_v1(outside.as_ptr(),outside.len(),bounds.as_mut_ptr(),error.as_mut_ptr(),error.len())};
        assert_eq!(code,1);
        assert!(String::from_utf8_lossy(&error).contains("outside native source coverage"));
    }

    #[test] fn geometry_refuses_a_window_without_the_halo_and_releases_cleanly() {
        let whole=hrrr_seed(&[]);
        let mut error=[0u8;256];
        let owner=unsafe {gpuwm_projected_geometry_v1(whole.as_ptr(),whole.len(),Some(stub),error.as_mut_ptr(),error.len())};
        assert!(!owner.is_null(), "{}", String::from_utf8_lossy(&error));
        let g=unsafe {&(*owner).geometry};
        assert_eq!((g.ntarget,g.source_nx,g.source_ny),(12,1799,1059));
        unsafe {gpuwm_projected_geometry_release_v1(owner)};
        let tight=hrrr_seed(&[("i_start",serde_json::json!(800)),("j_start",serde_json::json!(500)),
            ("source_nx",serde_json::json!(4)),("source_ny",serde_json::json!(4))]);
        let owner=unsafe {gpuwm_projected_geometry_v1(tight.as_ptr(),tight.len(),Some(stub),error.as_mut_ptr(),error.len())};
        assert!(owner.is_null());
        assert!(String::from_utf8_lossy(&error).contains("projected window lacks"));
        let owner=unsafe {gpuwm_projected_geometry_v1(whole.as_ptr(),whole.len(),None,error.as_mut_ptr(),error.len())};
        assert!(owner.is_null());
    }

    #[test] fn the_met_writer_reads_back_through_the_shared_reader() {
        let dir=std::env::temp_dir().join(format!("projected-met-{}",std::process::id()));
        let _=fs::remove_dir_all(&dir); fs::create_dir_all(&dir).unwrap();
        let job:Job=serde_json::from_value(serde_json::json!({
            "output":dir.join("MET"),"receipt":dir.join("MET.json"),"valid_time":"2026-10-14_12:00:00",
            "forecast_hour":0.0,"map_source":"test","levels":1,"workers":1,
            "target":{"south":38.0,"west":-98.0,"dlat":0.05,"dlon":0.05,"nx":2,"ny":1},
            "sources":{},"atmosphere":[],"surface":[],"wind_3d":["U","V"],"wind_10m":["U10","V10"],
            "soil":{"temperature":"ST","moisture":"SM","nodes":[0.0,0.1],"midpoints":[0.05],"names":[["ST000010","SM000010"]]},
            "admission":{"land":"L","ice":"I","soil_moisture":"SM","soil_temperature":"ST","humidity":[]}})).unwrap();
        assert!(job.soil.node_names.is_none());
        let named:Soil=serde_json::from_value(serde_json::json!({"temperature":"ST","moisture":"SM","nodes":[0.0,0.1],
            "midpoints":[],"names":[],"preserve_nodes":true,"node_names":[["SOILT000","SOILM000"],["SOILT010","SOILM010"]]})).unwrap();
        assert_eq!(named.node_names.as_ref().unwrap()[1][0],"SOILT010");
        let path=dir.join("one.met");
        let mut out=BufWriter::new(File::create(&path).unwrap());
        write_field(&mut out,&job,"TT","K","Temperature",1.0,&[280.0,281.0]).unwrap();
        write_field(&mut out,&job,"PSFC","Pa","Surface pressure",200100.0,&[1.0e5,1.01e5]).unwrap();
        assert!(write_field(&mut out,&job,"TT","K","Temperature",1.0,&[f32::NAN,1.0]).is_err());
        assert!(write_field(&mut out,&job,"TT","K","Temperature",1.0,&[1.0]).is_err());
        drop(out);
        assert_eq!(wps_intermediate::inventory(&path).unwrap(),(2,4));
        let _=fs::remove_dir_all(&dir);
    }

    #[test] fn malformed_job_is_returned_as_an_error_across_the_ffi() {
        let mut g=Geometry {source_ny:4,source_nx:4,ntarget:2,donor_y:[3,0].as_ptr(),donor_x:[0,0].as_ptr(),fraction_y:[0.5,0.5].as_ptr(),fraction_x:[0.5,0.5].as_ptr(),nearest_y:[0,0].as_ptr(),nearest_x:[0,0].as_ptr(),sin_alpha:[0.0;16].as_ptr(),cos_alpha:[1.0;16].as_ptr(),interpolate:None};
        g.source_nx=0; let mut error=[0u8;512];
        assert_ne!(unsafe {gpuwm_projected_intermediate_v1(b"{}".as_ptr(),2,&g,error.as_mut_ptr(),error.len())},0);
        assert_ne!(error[0],0);
    }

    #[test] fn the_abi_and_the_source_stamp_are_exported() {
        assert_eq!(gpuwm_projected_intermediate_abi_version(),1);
        let stamp=unsafe {std::ffi::CStr::from_ptr(gpuwm_projected_intermediate_source_revision().cast())};
        assert!(stamp.to_str().unwrap().starts_with("GPUWM_BRIDGE_SOURCE_REV="));
    }
}
