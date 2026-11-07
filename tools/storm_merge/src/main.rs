//! Source-independent vortex merge on canonical pressure-coordinate frames.
//!
//! The nonlinear balance equation is implemented from its published equation,
//! not copied model code: laplacian(Phi) = 2 J(u,v) + f*zeta - beta*u.
//! See NOAA HWRF scientific documentation, section 6.2, and
//! doi:10.3390/atmos11090968. The solve is confined to the blend ring, with
//! Dirichlet geopotential from the donor core and background exterior.
//! Hydrostatic temperature increments follow d(Phi)/d(log p) = -Rd*Tv.
//! This is an initialization operator, not a claim that a forecast is shock-free.
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::{self, File}, io::{Read, Seek, SeekFrom, Write}, path::{Path, PathBuf}, time::Instant};

type Result<T> = std::result::Result<T, String>;
const RD: f64 = 287.05;
const G: f64 = 9.80665;
const EARTH: f64 = 6_371_000.;
const OMEGA: f64 = 7.292115e-5;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    background: PathBuf,
    donor: PathBuf,
    output: PathBuf,
    valid_time: String,
    center_latitude_deg: f64,
    center_longitude_deg: f64,
    radius_km: f64,
    blend_width_km: f64,
    #[serde(default)]
    target_grid: Option<TargetGrid>,
    #[serde(default = "tolerance")]
    balance_relative_tolerance: f64,
    #[serde(default = "iterations")]
    balance_max_iterations: usize,
    /// TC-VI1: a deterministic donor core (member null, e.g. HAFS-A) into one ensemble member background, declared
    /// explicitly by the caller; every other member mismatch still refuses
    #[serde(default)]
    deterministic_donor_into_member: bool,
    /// TC-VP1: this member's balanced perturbation of the merged core (intensity, size, position), so a cold-start
    /// ensemble spans the observed core instead of carrying one identical donor core in every member
    #[serde(default)]
    core_perturbation: Option<CorePerturbation>,
}
/// TC-VP1 per-member targets, drawn by gpuwm/storm_merge.py from the packaged table gpuwm/data/tc/vortex_perturbation.json.
/// The merged vortex's axisymmetric tangential wind is rescaled (intensity) and radially remapped (RMW), the core is
/// displaced, and geopotential, temperature and surface pressure follow from the axisymmetric gradient-wind
/// integral of the change (hydrostatic in log pressure; barotropic below the boundary-layer top, as Kurihara et al.
/// 1993 balance the pressure from boundary-layer-top winds). Everything beyond outer_radius_m is bitwise unchanged.
#[derive(Deserialize, Clone)]
#[serde(deny_unknown_fields)]
struct CorePerturbation {
    target_central_pressure_pa: f64,
    target_rmw_m: f64,
    #[serde(default)]
    shift_east_m: f64,
    #[serde(default)]
    shift_north_m: f64,
    #[serde(default = "pert_outer")]
    outer_radius_m: f64,
    #[serde(default = "pert_bl")]
    boundary_layer_top_pa: f64,
    /// full intensity change at and below the first pressure, none at and above the second (cosine in log p)
    #[serde(default = "pert_taper")]
    intensity_taper_pa: [f64; 2],
    #[serde(default = "pert_scale_bounds")]
    intensity_scale_bounds: [f64; 2],
    #[serde(default)]
    label: Option<String>,
}
fn pert_outer() -> f64 { 300_000. }
fn pert_bl() -> f64 { 85_000. }
fn pert_taper() -> [f64; 2] { [10_000., 5_000.] }
fn pert_scale_bounds() -> [f64; 2] { [0.25, 2.5] }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetGrid { south: f64, west: f64, spacing_deg: f64, nx: usize, ny: usize }
fn tolerance() -> f64 { 1e-7 }
fn iterations() -> usize { 2000 }
fn err(e: impl std::fmt::Display) -> String { e.to_string() }
fn hash(b: &[u8]) -> String { format!("{:x}", Sha256::digest(b)) }
fn array_hash(shape: &[usize], bytes: &[u8]) -> String {
    let mut digest=Sha256::new(); digest.update(b"<f8\0");
    digest.update(format!("{:?}",shape).as_bytes()); digest.update(b"\0"); digest.update(bytes);
    format!("{:x}",digest.finalize())
}
fn read_json(p: &Path) -> Result<Value> { serde_json::from_slice(&fs::read(p).map_err(err)?).map_err(err) }
fn string<'a>(j: &'a Value, key: &str) -> Result<&'a str> { j[key].as_str().ok_or_else(|| format!("missing string {key}")) }
fn usize_field(j: &Value, key: &str) -> Result<usize> { j[key].as_u64().and_then(|v| usize::try_from(v).ok()).ok_or_else(|| format!("invalid {key}")) }
fn coordinates(j: &Value) -> Result<Vec<f64>> {
    let values: Vec<f64> = serde_json::from_value(j["values"].clone()).map_err(err)?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    if values.len() != usize_field(j,"count")? || hash(&bytes) != string(j,"sha256")? || values.iter().any(|v| !v.is_finite()) {
        return Err("coordinate digest/count/finite check failed".into());
    }
    Ok(values)
}
struct Frame {
    document: Value,
    fields: BTreeMap<String, Vec<f64>>,
    lat: Vec<f64>, lon: Vec<f64>, pressure: Vec<f64>,
}
fn load(root: &Path, time: &str) -> Result<(Value, Frame)> {
    let document = read_json(&root.join("frames.json"))?;
    if !["gpuwm-mapped-frameset-v1", "gpuwm-mapped-windowed-frameset-v1"].iter().any(|v|document["schema"]==*v) || document["stream"]["path"] != "frames.f64" || document["stream"]["dtype"] != "<f8" {
        return Err("merge requires canonical unwindowed float64 frames".into());
    }
    let matches: Vec<_> = document["frames"].as_array().ok_or("missing frames")?.iter().filter(|f| f["valid_time"] == time).collect();
    if matches.len()!=1 { return Err("merge needs exactly one frame at the requested valid time (select one member)".into()); }
    let f = matches[0];
    if f.get("atmospheric_window").is_some() || f["vertical_kind"]!="pressure" || f["vertical_units"]!="Pa"
        || f["header"]["grid"]["projection"]!="regular_latitude_longitude" || f["header"]["grid"]["wind_basis"]!="earth_relative" {
        return Err("merge needs unwindowed regular latitude-longitude, earth-relative winds and pressure in Pa".into());
    }
    let lat=coordinates(&f["latitude"])?; let lon=coordinates(&f["longitude"])?; let pressure=coordinates(&f["vertical_values"])?;
    for (name,axis) in [("latitude",&lat),("longitude",&lon)] {
        if axis.len()<3 || axis.windows(2).any(|p| (p[1]-p[0])*(axis[1]-axis[0])<=0.) {
            return Err(format!("{name} must be monotone with at least three points"));
        }
        let spacing=axis[1]-axis[0];
        if axis.windows(2).any(|p| ((p[1]-p[0])-spacing).abs()>1e-6) { return Err(format!("{name} must be regular")); }
    }
    if pressure.len()<3 || pressure[0]<=0. || pressure.windows(2).any(|p|p[1]<=p[0]) { return Err("pressure must increase".into()); }
    let path=root.join("frames.f64"); let size=fs::metadata(&path).map_err(err)?.len();
    if size!=document["stream"]["bytes"].as_u64().ok_or("invalid stream size")? {return Err("stream size differs".into());}
    let fields: Result<Vec<_>> = f["fields"].as_array().ok_or("missing fields")?.par_iter().map(|field| {
        let name=string(field,"name")?.to_owned();
        let length=usize_field(field,"length")?; let offset=usize_field(field,"offset")?;
        let shape:Vec<usize>=serde_json::from_value(field["shape"].clone()).map_err(err)?;
        if field["dtype"]!="<f8" || length%8!=0 || shape.iter().product::<usize>()!=length/8 || (offset+length) as u64>size {return Err(format!("invalid descriptor {name}"));}
        let mut bytes=vec![0;length]; let mut input=File::open(&path).map_err(err)?;
        input.seek(SeekFrom::Start(offset as u64)).map_err(err)?; input.read_exact(&mut bytes).map_err(err)?;
        if array_hash(&shape,&bytes)!=string(field,"sha256")? {return Err(format!("field digest differs: {name}"));}
        let values=bytes.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())).collect();
        Ok((name,values))
    }).collect();
    Ok((document.clone(),Frame{document:f.clone(),fields:fields?.into_iter().collect(),lat,lon,pressure}))
}
fn field<'a>(f:&'a Frame,name:&str)->Result<&'a [f64]> {f.fields.get(name).map(Vec::as_slice).ok_or_else(||format!("missing {name}"))}
fn bracket(axis:&[f64],value:f64)->Result<(usize,f64)> {
    let asc=axis[1]>axis[0];
    if (asc && (value<axis[0]-1e-8 || value>axis[axis.len()-1]+1e-8)) || (!asc && (value>axis[0]+1e-8 || value<axis[axis.len()-1]-1e-8)) {return Err(format!("donor does not cover coordinate {value}"));}
    let index=axis.partition_point(|x|if asc {*x<=value} else {*x>=value}).saturating_sub(1).min(axis.len()-2);
    Ok((index,((value-axis[index])/(axis[index+1]-axis[index])).clamp(0.,1.)))
}
#[derive(Clone,Copy)] struct Stencil { indices:[usize;4], weights:[f64;4] }
fn stencil(d:&Frame,lat:f64,lon:f64)->Result<Stencil> {
    let middle=(d.lon[0]+d.lon[d.lon.len()-1])/2.; let lon=middle+(lon-middle+180.).rem_euclid(360.)-180.;
    let (j,y)=bracket(&d.lat,lat)?; let (i,x)=bracket(&d.lon,lon)?;let n=d.lon.len();
    Ok(Stencil{indices:[j*n+i,j*n+i+1,(j+1)*n+i,(j+1)*n+i+1],weights:[(1.-x)*(1.-y),x*(1.-y),(1.-x)*y,x*y]})
}
fn interpolate(a:&[f64],s:Stencil,offset:usize)->Result<f64> {
    let mut sum=0.;
    for k in 0..4 { if s.weights[k]>1e-14 {let v=a[offset+s.indices[k]];if !v.is_finite(){return Err("missing donor value inside merge footprint".into());}sum+=v*s.weights[k];} }
    Ok(sum)
}
fn axis_document(values:&[f64])->Value {
    let bytes:Vec<_>=values.iter().flat_map(|v|v.to_le_bytes()).collect();
    json!({"values":values,"count":values.len(),"sha256":hash(&bytes)})
}
fn resample_background(mut frame:Frame, grid:&TargetGrid)->Result<Frame> {
    if !grid.south.is_finite() || !grid.west.is_finite() || !grid.spacing_deg.is_finite()
        || grid.spacing_deg<=0. || grid.nx<5 || grid.ny<5
        || grid.nx.checked_mul(grid.ny).is_none_or(|n|n>8_000_000) {
        return Err("invalid target grid or more than eight million cells".into());
    }
    let lat:Vec<_>=(0..grid.ny).map(|i|grid.south+i as f64*grid.spacing_deg).collect();
    let lon:Vec<_>=(0..grid.nx).map(|i|grid.west+i as f64*grid.spacing_deg).collect();
    let n=grid.nx*grid.ny;let old_n=frame.lat.len()*frame.lon.len();
    let stencils:Result<Vec<_>>=(0..n).into_par_iter().map(|i|stencil(&frame,lat[i/grid.nx],lon[i%grid.nx])).collect();
    let stencils=stencils?;
    let land=frame.fields.get("land_fraction");
    let fields:Result<Vec<_>>=frame.fields.par_iter().map(|(name,values)|{
        if values.len()%old_n!=0 {return Err(format!("cannot resample shape of {name}"));}
        let categorical=name=="land_fraction" || name.ends_with("_category") || name.ends_with("_mask");
        let masked=matches!(name.as_str(),"soil_temperature"|"volumetric_soil_moisture"|"sea_ice_fraction"|"sea_surface_temperature"|"skin_temperature");
        let output:Result<Vec<_>>=(0..n*(values.len()/old_n)).into_par_iter().map(|i|{
            let s=stencils[i%n];let offset=i/n*old_n;
            let nearest=(0..4).max_by(|a,b|s.weights[*a].total_cmp(&s.weights[*b])).unwrap();
            if categorical {return Ok(values[offset+s.indices[nearest]]);}
            let class=land.map(|v|v[s.indices[nearest]]>=0.5);
            let mut total=0.;let mut weight=0.;
            for k in 0..4 {
                if masked && land.is_some_and(|v|Some(v[s.indices[k]]>=0.5)!=class) {continue;}
                let v=values[offset+s.indices[k]];
                if v.is_finite() {total+=s.weights[k]*v;weight+=s.weights[k];}
            }
            if weight>1e-14 {Ok(total/weight)} else if masked {Ok(f64::NAN)} else {Err(format!("background missing {name} on target grid"))}
        }).collect();
        Ok((name.clone(),output?))
    }).collect();
    frame.fields=fields?.into_iter().collect();frame.lat=lat;frame.lon=lon;
    frame.document["latitude"]=axis_document(&frame.lat);frame.document["longitude"]=axis_document(&frame.lon);
    frame.document["grid_fingerprint"]=json!(format!("merge-target:{}",hash(serde_json::to_string(&json!([frame.document["latitude"],frame.document["longitude"]])).unwrap().as_bytes())));
    frame.document["header"]["grid"]["nx"]=json!(grid.nx);frame.document["header"]["grid"]["ny"]=json!(grid.ny);
    frame.document["header"]["grid"]["scan_order"]=json!("+x,+y");
    frame.document["header"]["grid"]["parameters"]=json!({"latitude_first":frame.lat[0],"latitude_last":frame.lat[grid.ny-1],"longitude_first":frame.lon[0],"longitude_last":frame.lon[grid.nx-1]});
    for desc in frame.document["fields"].as_array_mut().ok_or("missing fields")? {
        let shape=desc["shape"].as_array_mut().ok_or("missing shape")?;let len=shape.len();shape[len-2]=json!(grid.ny);shape[len-1]=json!(grid.nx);
        let name=string(desc,"name")?.to_owned();desc["length"]=json!(frame.fields[&name].len()*8);
        desc["missing_count"]=json!(frame.fields[&name].iter().filter(|v|v.is_nan()).count());
    }
    for desc in frame.document["header"]["fields"].as_array_mut().ok_or("missing header fields")? {
        let shape=desc["shape"].as_array_mut().ok_or("missing header shape")?;let len=shape.len();shape[len-2]=json!(grid.ny);shape[len-1]=json!(grid.nx);
    }
    Ok(frame)
}
fn smooth_weight(r:f64,inner:f64,width:f64)->f64 {
    if r<=inner {1.} else if r>=inner+width {0.} else {let x=(r-inner)/width;1.-x*x*x*(10.+x*(-15.+6.*x))}
}
fn is_tracer(name:&str)->bool {name.contains("mixing_ratio") || name.ends_with("_number") || name=="specific_humidity" || name=="specific_humidity_2m"}
fn merge_field(name:&str)->bool {
    is_tracer(name) || matches!(name,"eastward_wind"|"northward_wind"|"air_temperature"|"geopotential_height"|"surface_pressure"|"air_pressure_at_mean_sea_level"|"air_temperature_2m"|"eastward_wind_10m"|"northward_wind_10m"|"vertical_velocity")
}

// Symmetric positive definite negative Laplacian on the discrete ring.
// Dirichlet correction is exactly zero in core and exterior. The rowwise
// metric uses local zonal distances; this limited-area balance omits spherical
// connection terms and is explicitly recorded as such in the receipt.
fn apply(x:&[f64],mask:&[bool],nx:usize,dx:&[f64],dy:f64)->Vec<f64>{
    x.par_iter().enumerate().map(|(i,&v)|{
        if !mask[i]{return 0.;} let ax=1./dx[i/nx].powi(2);let ay=1./dy.powi(2);
        (2.*ax+2.*ay)*v-ax*(x[i-1]+x[i+1])-ay*(x[i-nx]+x[i+nx])
    }).collect()
}
fn dot(a:&[f64],b:&[f64])->f64 {a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn solve(rhs:&[f64],mask:&[bool],nx:usize,dx:&[f64],dy:f64,tol:f64,maxit:usize)->Result<(Vec<f64>,usize,f64)> {
    let mut x=vec![0.;rhs.len()];let mut r=rhs.to_vec();let mut p=r.clone();let mut rr=dot(&r,&r);let initial=rr;
    if initial==0.{return Ok((x,0,0.));}
    for iteration in 0..maxit {
        let ap=apply(&p,mask,nx,dx,dy);let denom=dot(&p,&ap);
        if !denom.is_finite() || denom<=0.{return Err("balance operator lost positive definiteness".into());}
        let alpha=rr/denom;
        for i in 0..x.len(){x[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
        let next=dot(&r,&r);let relative=(next/initial).sqrt();
        if relative<=tol{return Ok((x,iteration+1,relative));}
        let beta=next/rr;for i in 0..p.len(){p[i]=r[i]+beta*p[i];}rr=next;
    }
    Err(format!("balance solve did not converge within {maxit} iterations"))
}

// ---------------------------------------------------------------------------------------------- TC-VP1 perturbation
const EPS_V: f64 = 0.6078;
fn local_offset(c:(f64,f64),lat:f64,lon:f64)->(f64,f64){
    let dlon=(lon-c.1+180.).rem_euclid(360.)-180.;
    (dlon.to_radians()*EARTH*c.0.to_radians().cos(),(lat-c.0).to_radians()*EARTH)
}
fn local_point(c:(f64,f64),east:f64,north:f64)->(f64,f64){
    (c.0+(north/EARTH).to_degrees(),c.1+(east/(EARTH*c.0.to_radians().cos())).to_degrees())
}
fn bilinear(a:&[f64],s:Stencil,offset:usize)->f64{
    let mut sum=0.;
    for k in 0..4 {if s.weights[k]>1e-14 {let v=a[offset+s.indices[k]];if !v.is_finite(){return f64::NAN;}sum+=v*s.weights[k];}}
    sum
}
fn reduced_pressure(ps:f64,z:f64,t2:f64,q2:f64)->f64 {
    let tv=t2*(1.+EPS_V*q2);ps*(G*z/(RD*(tv+0.5*0.0065*z))).exp()
}
fn profile_at(profile:&[f64],dr:f64,radius:f64)->f64{
    let x=(radius/dr).max(0.);let j=(x.floor() as usize).min(profile.len()-2);let t=(x-j as f64).clamp(0.,1.);
    profile[j]*(1.-t)+profile[j+1]*t
}

// Bolton (1980), liquid-water saturation. Canonical frames carry specific
// humidity (water / total moist mass), unlike WRF's dry-air mixing ratio.
fn saturation_specific_humidity(t: f64, pressure: f64) -> f64 {
    let es = 611.2 * (17.67 * (t - 273.15) / (t - 29.65)).exp();
    0.622 * es / (pressure - 0.378 * es).max(1.0)
}

/// TC-VP2: solve temperature and vapour together. The saturation fraction
/// q/qs is retained while the prescribed virtual-temperature increment is
/// met, so hydrostatic warming cannot dry a saturated eyewall.
fn warm_at_constant_rh(t: f64, q: f64, pressure: f64, dtv: f64) -> Result<(f64, f64)> {
    if ![t,q,pressure,dtv].iter().all(|v| v.is_finite()) || q < 0.0 || q >= 1.0 {
        return Err("core perturbation needs finite temperature, pressure and specific humidity".into());
    }
    if dtv == 0.0 { return Ok((t,q)); }
    let fraction = q / saturation_specific_humidity(t, pressure);
    let target = t * (1.0 + EPS_V * q) + dtv;
    let vapour = |temperature| fraction * saturation_specific_humidity(temperature, pressure);
    let virtual_t = |temperature| temperature * (1.0 + EPS_V * vapour(temperature));
    let (mut lo, mut hi) = (150.0, 350.0);
    if target <= virtual_t(lo) || target >= virtual_t(hi) {
        return Err("core perturbation creates an invalid temperature".into());
    }
    for _ in 0..48 {
        let mid = 0.5 * (lo + hi);
        if virtual_t(mid) < target { lo = mid; } else { hi = mid; }
    }
    let temperature = 0.5 * (lo + hi);
    let humidity = vapour(temperature);
    if !humidity.is_finite() || humidity >= 1.0 {
        return Err("core perturbation creates invalid specific humidity".into());
    }
    Ok((temperature, humidity))
}

/// TC-VP1: perturb the merged vortex of frame `b` toward one member's targets. See CorePerturbation.
fn perturb_core(b:&mut Frame,p:&CorePerturbation,hint:(f64,f64))->Result<Value>{
    let started=Instant::now();
    let finite=[p.target_central_pressure_pa,p.target_rmw_m,p.shift_east_m,p.shift_north_m,p.outer_radius_m,
        p.boundary_layer_top_pa,p.intensity_taper_pa[0],p.intensity_taper_pa[1],p.intensity_scale_bounds[0],p.intensity_scale_bounds[1]];
    if finite.iter().any(|v|!v.is_finite()) || !(50_000. ..110_000.).contains(&p.target_central_pressure_pa)
        || !(3_000. ..200_000.).contains(&p.target_rmw_m) || !(50_000. ..=800_000.).contains(&p.outer_radius_m)
        || p.shift_east_m.hypot(p.shift_north_m)>0.25*p.outer_radius_m
        || !(p.intensity_taper_pa[0]>p.intensity_taper_pa[1] && p.intensity_taper_pa[1]>0.)
        || !(0.<p.intensity_scale_bounds[0] && p.intensity_scale_bounds[0]<1. && p.intensity_scale_bounds[1]>1.)
        || !(p.boundary_layer_top_pa>p.intensity_taper_pa[0]) {
        return Err("invalid core perturbation targets or controls".into());
    }
    let nx=b.lon.len();let ny=b.lat.len();let n=nx*ny;let nz=b.pressure.len();
    let ps=field(b,"surface_pressure")?.to_vec();let t2=field(b,"air_temperature_2m")?.to_vec();
    let q2=b.fields.get("specific_humidity_2m").cloned().unwrap_or_else(||vec![0.;n]);
    let zs=b.fields.get("terrain_height").cloned().unwrap_or_else(||vec![0.;n]);
    let pr:Vec<f64>=(0..n).map(|i|reduced_pressure(ps[i],zs[i],t2[i],q2[i])).collect();
    // centre: reduced-pressure minimum within 150 km of the merge centre, parabolic sub-cell
    let mut best=(f64::INFINITY,0usize);
    for j in 2..ny-2 {for i in 2..nx-2 {
        let (e,nn)=local_offset(hint,b.lat[j],b.lon[i]);
        if e.hypot(nn)<=150_000. && pr[j*nx+i]<best.0 {best=(pr[j*nx+i],j*nx+i);}
    }}
    if !best.0.is_finite(){return Err("core perturbation finds no pressure minimum near the merge centre".into());}
    let (jc,ic)=(best.1/nx,best.1%nx);
    let vertex=|a:f64,m:f64,c:f64|{let d=a-2.*m+c;if d<=0.{0.}else{(0.5*(a-c)/d).clamp(-0.5,0.5)}};
    let dj=vertex(pr[best.1-nx],pr[best.1],pr[best.1+nx]);let di=vertex(pr[best.1-1],pr[best.1],pr[best.1+1]);
    let c0=(b.lat[jc]+dj*(b.lat[1]-b.lat[0]),b.lon[ic]+di*(b.lon[1]-b.lon[0]));
    let p0=best.0;let tv0=t2[best.1]*(1.+EPS_V*q2[best.1]);
    let shift=(p.shift_east_m,p.shift_north_m);let smag=shift.0.hypot(shift.1);
    // the whole disk and its displaced source must lie inside the lattice with a two-cell margin
    let reach=p.outer_radius_m+smag;
    let (la_lo,_)=local_point(c0,0.,-reach);let (la_hi,_)=local_point(c0,0.,reach);
    let (_,lo_lo)=local_point(c0,-reach,0.);let (_,lo_hi)=local_point(c0,reach,0.);
    let dlat=b.lat[1]-b.lat[0];let dlon=b.lon[1]-b.lon[0];
    if la_lo<b.lat[0]+2.*dlat || la_hi>b.lat[ny-1]-2.*dlat || lo_lo<b.lon[0]+2.*dlon || lo_hi>b.lon[nx-1]-2.*dlon {
        return Err("core perturbation disk does not fit the merge lattice; widen --storm-merge-grid".into());
    }
    // axisymmetric (72-azimuth) counter-clockwise tangential wind of the merged vortex on 500 m rings
    let dr=500.;let nr=(p.outer_radius_m/dr).ceil() as usize+1;let az=72usize;
    let ring:Result<Vec<(Stencil,f64)>>=(0..nr*az).into_par_iter().map(|index|{
        let rho=(index/az) as f64*dr;let th=2.*std::f64::consts::PI*(index%az) as f64/az as f64;
        let (la,lo)=local_point(c0,rho*th.cos(),rho*th.sin());Ok((stencil(b,la,lo)?,th))
    }).collect();let ring=ring?;
    let tangential=|u:&[f64],v:&[f64],offset:usize|->Vec<f64>{(0..nr).map(|jr|{
        let mut s=0.;for a in 0..az {let (st,th)=ring[jr*az+a];s+=-bilinear(u,st,offset)*th.sin()+bilinear(v,st,offset)*th.cos();}
        s/az as f64}).collect()};
    let (u,v)=(field(b,"eastward_wind")?,field(b,"northward_wind")?);
    let profile:Vec<Vec<f64>>=(0..nz).into_par_iter().map(|k|tangential(u,v,k*n)).collect();
    let profile10=tangential(field(b,"eastward_wind_10m")?,field(b,"northward_wind_10m")?,0);
    if profile10.iter().any(|x|!x.is_finite()){return Err("core perturbation: missing 10 m wind inside the disk".into());}
    let jr=(2..nr).max_by(|a,c|profile10[*a].total_cmp(&profile10[*c])).unwrap();
    let rmw0=jr as f64*dr;let vmax_sym=profile10[jr];
    if vmax_sym<=10. {return Err(format!("core perturbation needs a resolved vortex (axisymmetric 10 m maximum {vmax_sym:.1} m/s)"));}
    // monotone radius map rho(r)=r exp(c tau(r)): the RMW moves to the target, identity at and beyond r1->R
    // A target RMW whose map would fold inside the outer radius is held at the largest feasible change toward it
    // (receipt: target_rmw_held_by_outer_radius), the same rule as the cycle route. Refusing it dropped a member
    // of the default table (Milton 18Z member 16: RMW 14 -> 39 km inside 300 km).
    let big=p.outer_radius_m;
    let feasible=|t:f64|{let r1=1.2*rmw0.max(t);let l=(big/r1).ln();l>0.3 && 1.-(rmw0/t).ln().abs()*std::f64::consts::FRAC_PI_2/l>=0.2};
    let mut target_rmw=p.target_rmw_m;let held=!feasible(target_rmw);
    if held {
        if !feasible(rmw0) {return Err(format!("the merged RMW {:.0} km leaves no room for a radius map inside {:.0} km",rmw0/1e3,big/1e3));}
        let (mut ok,mut bad)=(rmw0,target_rmw);
        for _ in 0..60 {let mid=0.5*(ok+bad);if feasible(mid){ok=mid}else{bad=mid}}
        target_rmw=ok;
    }
    let c=(rmw0/target_rmw).ln();let r1=1.2*rmw0.max(target_rmw);
    let l=(big/r1).ln();
    let tau=|r:f64|if r<=r1{1.}else if r>=big{0.}else{0.5*(1.+(std::f64::consts::PI*(r/r1).ln()/l).cos())};
    let dtau=|r:f64|if r<=r1||r>=big{0.}else{-0.5*(std::f64::consts::PI*(r/r1).ln()/l).sin()*std::f64::consts::PI/(l*r)};
    let rho=|r:f64|r*(c*tau(r)).exp();let drho=|r:f64|(c*tau(r)).exp()*(1.+c*r*dtau(r));
    let lp:Vec<f64>=b.pressure.iter().map(|x|x.ln()).collect();
    let (pa,pb)=(p.intensity_taper_pa[0],p.intensity_taper_pa[1]);
    let weight:Vec<f64>=b.pressure.iter().map(|&x|if x>=pa{1.}else if x<=pb{0.}else{0.5*(1.-(std::f64::consts::PI*(x/pb).ln()/(pa/pb).ln()).cos())}).collect();
    let kbl=(0..nz).filter(|&k|b.pressure[k]<=p.boundary_layer_top_pa).last().ok_or("no level above the boundary-layer top")?;
    if profile[kbl].iter().any(|x|!x.is_finite()){return Err("core perturbation: missing boundary-layer-top wind inside the disk".into());}
    let f=2.*OMEGA*c0.0.to_radians().sin();
    // the merged vortex's own axisymmetric geopotential and its radial gradient, by level
    let z0=field(b,"geopotential_height")?;
    let phi_profile:Vec<Vec<f64>>=(0..nz).into_par_iter().map(|k|(0..nr).map(|jr|
        (0..az).map(|a|G*bilinear(z0,ring[jr*az+a].0,k*n)).sum::<f64>()/az as f64).collect()).collect();
    let phi_gradient:Vec<Vec<f64>>=phi_profile.iter().map(|ph|(0..nr).map(|j|if j==0{0.}else if j==nr-1{(ph[j]-ph[j-1])/dr}
        else{(ph[j+1]-ph[j-1])/(2.*dr)}).collect()).collect();
    // geopotential change at fixed pressure: the merged vortex's own radial geopotential gradient, carried by the map,
    // times the gradient-wind ratio gamma = (a^2 V^2/r + a f V) / ((V^2/rho + f V) rho') of the new to the mapped
    // tangential wind (Kurihara et al. 1993; below the boundary-layer top gamma is the boundary-layer-top ratio, as
    // friction makes the lower winds unbalanced). The mass field keeps the vortex's own vertical structure.
    let dphi_level=|k:usize,alpha:f64|->Vec<f64>{
        let kp=if b.pressure[k]>p.boundary_layer_top_pa{kbl}else{k};let prof=&profile[kp];
        if prof.iter().any(|x|!x.is_finite()) || phi_gradient[k].iter().any(|x|!x.is_finite()){return vec![0.;nr];}
        let g:Vec<f64>=(0..nr).map(|j|{if j==0{return 0.;}let r=j as f64*dr;let ae=1.+(alpha-1.)*tau(r)*weight[kp];
            let rh=rho(r);let vv=profile_at(prof,dr,rh);let dr_=drho(r);
            let gamma=if vv.abs()<1.{ae/dr_}else{(ae*ae*vv*vv/r+ae*f*vv)/((vv*vv/rh+f*vv)*dr_)};
            (gamma-1.)*profile_at(&phi_gradient[k],dr,rh)*dr_}).collect();
        let mut out=vec![0.;nr];for j in (0..nr-1).rev(){out[j]=out[j+1]-0.5*(g[j]+g[j+1])*dr;}out};
    let target=RD*tv0*(p.target_central_pressure_pa/p0).ln();
    let (mut lo,mut hi)=(p.intensity_scale_bounds[0],p.intensity_scale_bounds[1]);
    // at a column's surface the change is extrapolated linearly in log p from its two lowest above-ground levels
    // (constant virtual-temperature change), never built from extrapolated underground values
    let xc=ps[best.1].ln();let ks=lp.partition_point(|v|*v<=xc).clamp(2,nz)-1;
    let centre_dphi=|alpha:f64|{let (d0,d1)=(dphi_level(ks-1,alpha)[0],dphi_level(ks,alpha)[0]);
        d1+(d1-d0)*(xc-lp[ks])/(lp[ks]-lp[ks-1])};
    let flo=centre_dphi(lo);let fhi=centre_dphi(hi);
    let (alpha,clipped)=if target>=flo{(lo,true)}else if target<=fhi{(hi,true)}else{
        for _ in 0..60 {let mid=0.5*(lo+hi);if centre_dphi(mid)>target{lo=mid}else{hi=mid}}(0.5*(lo+hi),false)};
    let dphi:Vec<Vec<f64>>=(0..nz).into_par_iter().map(|k|dphi_level(k,alpha)).collect();
    let dtv:Vec<Vec<f64>>=(0..nz).map(|k|{let (a,c2)=if k==0{(0,1)}else if k==nz-1{(nz-2,nz-1)}else{(k-1,k+1)};
        (0..nr).map(|j|-(dphi[c2][j]-dphi[a][j])/(RD*(lp[c2]-lp[a]))).collect()}).collect();
    // dPhi at a column's own surface pressure, linear in log p between the bracketing levels
    // a column's change below its lowest above-ground level: constant virtual-temperature change from that level
    // (one-sided), geopotential hydrostatically below it; returns (dPhi, dTv) at log pressure x
    let below=|r:f64,surface:f64,x:f64|->(f64,f64){let k=lp.partition_point(|v|*v<=surface.ln()).clamp(2,nz)-1;
        let (d0,d1)=(profile_at(&dphi[k-1],dr,r),profile_at(&dphi[k],dr,r));let slope=(d1-d0)/(lp[k]-lp[k-1]);
        (d1+slope*(x-lp[k]),-slope/RD)};
    let at_surface=|r:f64,surface:f64|->f64{below(r,surface,surface.ln()).0};
    let lowest=|surface:f64|lp.partition_point(|v|*v<=surface.ln()).clamp(2,nz)-1;
    let predicted=p0*(at_surface(0.,ps[best.1])/(RD*tv0)).exp();
    // destination cells: radius and azimuth about the displaced centre, source stencil about the merged centre
    let cells:Result<Vec<Option<(Stencil,f64,f64)>>>=(0..n).into_par_iter().map(|i|{
        let (e,nn)=local_offset(c0,b.lat[i/nx],b.lon[i%nx]);let (xe,xn)=(e-shift.0,nn-shift.1);let r=xe.hypot(xn);
        if r>=big{return Ok(None);}let th=xn.atan2(xe);let t=tau(r);let rh=rho(r);
        let (la,lo)=local_point(c0,shift.0*(1.-t)+rh*th.cos(),shift.1*(1.-t)+rh*th.sin());
        Ok(Some((stencil(b,la,lo)?,r,th)))}).collect();let cells=cells?;
    let original=b.fields.clone();
    let q=original.get("specific_humidity").ok_or("missing specific_humidity")?;
    let temperature=original.get("air_temperature").ok_or("missing air_temperature")?;
    let thermodynamics:Result<Vec<(f64,f64)>>=(0..nz*n).into_par_iter().map(|idx| {
        let k=idx/n;let i=idx%n;
        let Some((st,r,_))=cells[i] else {return Ok((temperature[idx],q[idx]));};
        let surface=bilinear(&ps,st,0);
        let change=if k>=lowest(surface){below(r,surface,lp[k]).1}else{profile_at(&dtv[k],dr,r)};
        let ts=bilinear(temperature,st,k*n);let qs=bilinear(q,st,k*n);
        // A missing upper-level value remains missing; it cannot be a source
        // of a finite increment. The ordinary field checks below still refuse
        // a missing donor where the destination was valid.
        if !ts.is_finite() || !qs.is_finite() {return Ok((ts,qs));}
        warm_at_constant_rh(ts,qs,b.pressure[k],change)
    }).collect();let thermodynamics=thermodynamics?;
    let names:Vec<String>=original.keys().filter(|name|merge_field(name)).cloned().collect();
    let updated:Result<Vec<(String,Vec<f64>)>>=names.par_iter().map(|name|{
        let src=&original[name];let planes=src.len()/n;let mut out=src.clone();
        for (i,cell) in cells.iter().enumerate(){let Some((st,r,th))=*cell else{continue;};
            let t=tau(r);let radial=|table:&[f64]|profile_at(table,dr,r);
            for k in 0..planes {let idx=k*n+i;let mut value=bilinear(src,st,k*n);
                if !value.is_finite(){if src[idx].is_finite(){return Err(format!("core perturbation sampled a missing {name}"));}continue;}
                match name.as_str(){
                    "eastward_wind"|"northward_wind"=>{let vv=profile_at(&profile[k],dr,rho(r));
                        if vv.is_finite(){let ae=1.+(alpha-1.)*t*weight[k];
                        value+=(ae-1.)*vv*if name=="eastward_wind"{-th.sin()}else{th.cos()};}}
                    "eastward_wind_10m"|"northward_wind_10m"=>{let vv=profile_at(&profile10,dr,rho(r));let ae=1.+(alpha-1.)*t;
                        value+=(ae-1.)*vv*if name=="eastward_wind_10m"{-th.sin()}else{th.cos()};}
                    // levels below the column's surface carry the surface change (no temperature change), so no
                    // increment is built from extrapolated underground values
                    "geopotential_height"=>{let surface=bilinear(&ps,st,0);
                        value+=if k>=lowest(surface){below(r,surface,lp[k]).0}else{radial(&dphi[k])}/G;}
                    "air_temperature"=>{value=thermodynamics[idx].0;}
                    "specific_humidity"=>{value=thermodynamics[idx].1;}
                    "surface_pressure"=>{let tvs=bilinear(&t2,st,0)*(1.+EPS_V*bilinear(&q2,st,0));let prs=bilinear(&pr,st,0);
                        value=prs*(at_surface(r,value)/(RD*tvs)).exp()*(-G*zs[i]/(RD*(tvs+0.5*0.0065*zs[i]))).exp();
                        if !(50_000. ..110_000.).contains(&value){return Err("core perturbation creates an invalid surface pressure".into());}}
                    "air_pressure_at_mean_sea_level"=>{let tvs=bilinear(&t2,st,0)*(1.+EPS_V*bilinear(&q2,st,0));
                        value*=(at_surface(r,bilinear(&ps,st,0))/(RD*tvs)).exp();}
                    _=>{if is_tracer(name)&&value<0.{value=0.;}}
                }
                if !value.is_finite(){return Err(format!("core perturbation creates a nonfinite {name}"));}
                out[idx]=value;}
        }
        Ok((name.clone(),out))}).collect();
    for (name,values) in updated? {b.fields.insert(name,values);}
    let final_ps=field(b,"surface_pressure")?;
    let final_t2=field(b,"air_temperature_2m")?;
    let final_q2=b.fields.get("specific_humidity_2m").map(Vec::as_slice).unwrap_or(&q2);
    let (new_lat,new_lon)=local_point(c0,shift.0,shift.1);
    let resolved=(0..n).into_par_iter().filter(|&i|{
        let (e,nn)=local_offset((new_lat,new_lon),b.lat[i/nx],b.lon[i%nx]);
        e.hypot(nn)<=150_000.
    }).map(|i|reduced_pressure(final_ps[i],zs[i],final_t2[i],final_q2[i]))
      .filter(|v|v.is_finite()).reduce(||f64::INFINITY,f64::min);
    let max_dt:Vec<f64>=(0..nz).map(|k|dtv[k].iter().fold(0.,|a:f64,x|a.max(x.abs()))).collect();
    // balance audit of the merged vortex itself: its axisymmetric geopotential deficit (outer ring minus centre)
    // against the gradient-wind integral of its own tangential wind, by level
    let audit:Vec<Value>=(0..nz).map(|k|{let centre=phi_profile[k][0];let edge=phi_profile[k][nr-1];
        let prof=&profile[k];let mut gw=0.;
        for j in 1..nr {let (r0,r1b)=((j-1) as f64*dr,j as f64*dr);let g0=if j==1{0.}else{prof[j-1]*prof[j-1]/r0+f*prof[j-1]};
            gw+=0.5*(g0+prof[j]*prof[j]/r1b+f*prof[j])*dr;}
        json!({"pressure_pa":b.pressure[k],"max_tangential_m_s":prof.iter().fold(f64::NEG_INFINITY,|a:f64,x|a.max(*x)),
               "deficit_m2_s2":edge-centre,"gradient_wind_deficit_m2_s2":gw})}).collect();
    Ok(json!({"schema":"gpuwm-storm-merge-core-perturbation-v1","label":p.label,
        "merged_centre_deg":[c0.0,c0.1],"merged_minimum_reduced_pressure_pa":p0,"merged_rmw_m":rmw0,
        "merged_axisymmetric_10m_maximum_m_s":vmax_sym,"target_central_pressure_pa":p.target_central_pressure_pa,
        "target_rmw_m":p.target_rmw_m,"applied_target_rmw_m":target_rmw,"target_rmw_held_by_outer_radius":held,"shift_m":[shift.0,shift.1],"outer_radius_m":big,"map_inner_radius_m":r1,
        "map_log_ratio":c,"intensity_scale":alpha,"intensity_scale_clipped":clipped,"intensity_scale_bounds":p.intensity_scale_bounds,
        "predicted_central_pressure_pa":predicted,"boundary_layer_top_pa":b.pressure[kbl],"intensity_taper_pa":p.intensity_taper_pa,
        "resolved_minimum_reduced_pressure_pa":resolved,
        "centre_geopotential_change_m2_s2_by_level":dphi.iter().map(|x|x[0]).collect::<Vec<_>>(),
        "max_virtual_temperature_change_k_by_level":max_dt,"pressure_levels_pa":b.pressure.clone(),
        "humidity":"TC-VP2: constant q/qs over liquid water; temperature and specific humidity solved together at fixed total pressure",
        "mass_coordinate":"canonical total pressure retained; downstream initialization derives dry mass from the updated vapour (no second subtraction from surface pressure)",
        "cells_changed":cells.iter().filter(|c|c.is_some()).count(),"merged_vortex_balance_audit":audit,
        "balance":"axisymmetric gradient wind (f-plane at the centre) of the change, integrated inward from the outer radius; hydrostatic temperature in log pressure; barotropic below the boundary-layer top; asymmetries and environment carried by the map",
        "seconds":started.elapsed().as_secs_f64()}))
}

fn run(request_path:&Path)->Result<Value>{
    let started=Instant::now();let request_bytes=fs::read(request_path).map_err(err)?;
    let req:Request=serde_json::from_slice(&request_bytes).map_err(err)?;
    if req.schema!="gpuwm-storm-merge-v1" || !req.center_latitude_deg.is_finite() || req.center_latitude_deg.abs()>65.
        || !req.center_longitude_deg.is_finite() || !req.radius_km.is_finite() || req.radius_km<=0.
        || !req.blend_width_km.is_finite() || req.blend_width_km<=0. || req.radius_km+req.blend_width_km>1000.
        || !(0.<req.balance_relative_tolerance && req.balance_relative_tolerance<1.) || req.balance_max_iterations==0 {
        return Err("invalid merge geometry or solver controls; ring must lie within 1000 km and center within 65 degrees latitude".into());
    }
    if req.output.exists(){return Err("output already exists; use a new directory".into());}
    let (mut manifest,mut b)=load(&req.background,&req.valid_time)?;let (_,d)=load(&req.donor,&req.valid_time)?;
    if b.document["member"]!=d.document["member"]
        && !(req.deterministic_donor_into_member && d.document["member"].is_null()) {
        return Err("donor/background member mismatch; select a member explicitly".into());
    }
    let read_seconds=started.elapsed().as_secs_f64();let phase=Instant::now();
    if let Some(grid)=&req.target_grid {b=resample_background(b,grid)?;}
    let target_regrid_seconds=phase.elapsed().as_secs_f64();let phase=Instant::now();
    let nx=b.lon.len();let ny=b.lat.len();let n=nx*ny;let nz=b.pressure.len();let nd=d.lon.len()*d.lat.len();
    let mut weight=vec![0.;n];let mut stencils=vec![None;n];let mut ring=vec![false;n];
    let inner=req.radius_km*1000.;let width=req.blend_width_km*1000.;
    for j in 0..ny {for i in 0..nx {
        let a=req.center_latitude_deg.to_radians();let c=b.lat[j].to_radians();let dl=(b.lon[i]-req.center_longitude_deg).to_radians();
        let h=((c-a)/2.).sin().powi(2)+a.cos()*c.cos()*(dl/2.).sin().powi(2);
        let radius=2.*EARTH*h.clamp(0.,1.).sqrt().asin();let idx=j*nx+i;
        weight[idx]=smooth_weight(radius,inner,width);
        if weight[idx]>0. {
            if i<2 || j<2 || i+2>=nx || j+2>=ny{return Err("merge footprint touches background boundary; expand the background domain".into());}
            stencils[idx]=Some(stencil(&d,b.lat[j],b.lon[i])?);
            ring[idx]=weight[idx]<1.;
        }
    }}
    if !weight.iter().any(|&w|w==1.) || ring.iter().filter(|&&r|r).count()<12 {return Err("merge core/ring is unresolved on this grid".into());}
    let logd:Vec<_>=d.pressure.iter().map(|p|p.ln()).collect();
    // Do not invent donor stratosphere above its published top. These levels
    // retain the background; the receipt records the available pressure range.
    let vertical:Vec<_>=b.pressure.iter().map(|p|if *p<d.pressure[0] {Ok(None)} else {bracket(&logd,p.ln()).map(Some)}).collect::<Result<_>>()?;
    let original_ps=field(&b,"surface_pressure")?.to_vec();
    let original_q=field(&b,"specific_humidity")?.to_vec();
    let donor_missing:Vec<_>=b.fields.keys().filter(|name|merge_field(name)&&!d.fields.contains_key(*name)).cloned().collect();
    let required=["eastward_wind","northward_wind","air_temperature","specific_humidity","geopotential_height","surface_pressure"];
    for name in required {if !b.fields.contains_key(name)||!d.fields.contains_key(name){return Err(format!("both states require {name}"));}}
    // Optional condensate and number moments absent from the background have
    // an explicit zero background. Add their declarations as well as values;
    // required thermodynamic fields must always exist in both states above.
    let mut added_tracers=Vec::new();
    for desc in d.document["fields"].as_array().ok_or("missing donor fields")? {
        let name=string(desc,"name")?;
        if !(name.contains("mixing_ratio") || name.ends_with("_number")) || b.fields.contains_key(name) {continue;}
        if desc["axes"]!=json!(["vertical","y","x"]) {return Err(format!("unsupported optional tracer axes {name}"));}
        let mut descriptor=desc.clone();descriptor["shape"]=json!([nz,ny,nx]);
        descriptor["length"]=json!(n*nz*8);descriptor["missing_count"]=json!(0);
        b.document["fields"].as_array_mut().unwrap().push(descriptor);
        let mut header=d.document["header"]["fields"].as_array().ok_or("missing donor header")?.iter()
            .find(|f|f["canonical_name"]==name).ok_or("missing donor tracer header")?.clone();
        header["shape"]=json!([nz,ny,nx]);
        b.document["header"]["fields"].as_array_mut().ok_or("missing background header")?.push(header);
        b.fields.insert(name.to_owned(),vec![0.;n*nz]);added_tracers.push(name.to_owned());
    }
    let descriptors=b.document["fields"].as_array().ok_or("missing fields")?.clone();
    let blended:Result<Vec<_>>=descriptors.par_iter().filter(|f|merge_field(f["name"].as_str().unwrap_or(""))).map(|desc|{
        let name=string(desc,"name")?;let orig=field(&b,name)?;
        let Some(donor)=d.fields.get(name) else {return Ok((name.to_owned(),orig.to_vec()));};
        let donor_desc=d.document["fields"].as_array().unwrap().iter().find(|f|f["name"]==name).ok_or("donor descriptor missing")?;
        if desc["units"]!=donor_desc["units"] || desc["axes"]!=donor_desc["axes"] {return Err(format!("units/axes differ for {name}"));}
        let is3=desc["axes"]==json!(["vertical","y","x"]);
        if !is3 && desc["axes"]!=json!(["y","x"]){return Err(format!("unsupported merge axes for {name}"));}
        if orig.len()!=n*if is3{nz}else{1} || donor.len()!=nd*if is3{d.pressure.len()}else{1}{return Err(format!("shape mismatch {name}"));}
        let values:Result<Vec<_>>=orig.iter().enumerate().map(|(index,&v)|{
            let i=index%n;let Some(stencil)=stencils[i] else{return Ok(v);};
            let dv=if is3{let Some((k,t))=vertical[index/n] else {return Ok(v);};interpolate(donor,stencil,k*nd)?*(1.-t)+interpolate(donor,stencil,(k+1)*nd)?*t}else{interpolate(donor,stencil,0)?};
            let result=if weight[i]==1.{dv}else{v+weight[i]*(dv-v)};
            if !result.is_finite() || (is_tracer(name)&&result<0.) {return Err(format!("invalid merged {name}"));}
            Ok(result)
        }).collect();Ok((name.to_owned(),values?))
    }).collect();
    for (name,values) in blended? {b.fields.insert(name,values);}
    let blend_seconds=phase.elapsed().as_secs_f64();let phase=Instant::now();
    let dx:Vec<_>=b.lat.iter().map(|lat|EARTH*lat.to_radians().cos()*(b.lon[1]-b.lon[0]).to_radians()).collect();
    let dy=EARTH*(b.lat[1]-b.lat[0]).to_radians();
    if width<4.*dx.iter().fold(dy.abs(),|a,v|a.max(v.abs())) {return Err("blend width must resolve at least four cells".into());}
    let u=field(&b,"eastward_wind")?;let v=field(&b,"northward_wind")?;let z=field(&b,"geopotential_height")?;
    let solved:Result<Vec<_>>=(0..nz).into_par_iter().map(|k|{
        let mut rhs=vec![0.;n];let u=&u[k*n..(k+1)*n];let v=&v[k*n..(k+1)*n];let z=&z[k*n..(k+1)*n];
        for i in 0..n {if ring[i] && vertical[k].is_some(){let j=i/nx;let xx=dx[j];let lat=b.lat[j].to_radians();
            let ux=(u[i+1]-u[i-1])/(2.*xx);let uy=(u[i+nx]-u[i-nx])/(2.*dy);
            let vx=(v[i+1]-v[i-1])/(2.*xx);let vy=(v[i+nx]-v[i-nx])/(2.*dy);
            let desired=2.*(ux*vy-uy*vx)+2.*OMEGA*lat.sin()*(vx-uy)-2.*OMEGA*lat.cos()/EARTH*u[i];
            let lap=G*((z[i+1]-2.*z[i]+z[i-1])/xx.powi(2)+(z[i+nx]-2.*z[i]+z[i-nx])/dy.powi(2));
            rhs[i]=lap-desired;
        }}
        solve(&rhs,&ring,nx,&dx,dy,req.balance_relative_tolerance,req.balance_max_iterations)
    }).collect();let solved=solved?;
    let mut newz=z.to_vec();let mut temp=field(&b,"air_temperature")?.to_vec();let q=field(&b,"specific_humidity")?.to_vec();
    for k in 0..nz {for i in 0..n {if ring[i]{newz[k*n+i]+=solved[k].0[i]/G;}}}
    // Exact discrete hypsometry in the ring. The n-1 trapezoidal layer
    // constraints leave one degree of freedom; choose it by least squares
    // closest to the blended virtual-temperature column. No clipping conceals
    // an incompatible donor/background pair.
    let columns:Result<Vec<_>>=(0..n).into_par_iter().filter(|&i|ring[i]).map(|i|{
        let mut offset=vec![0.;nz];let mut sign=1.;let mut numerator=0.;
        for k in 0..nz {
            if k>0 {offset[k]=-2.*G*(newz[k*n+i]-newz[(k-1)*n+i])/(RD*(b.pressure[k]/b.pressure[k-1]).ln())-offset[k-1];}
            numerator+=sign*(temp[k*n+i]*(1.+0.6078*q[k*n+i])-offset[k]);sign=-sign;
        }
        let first=numerator/nz as f64;let values:Vec<_>=(0..nz).map(|k|((if k%2==0{first}else{-first})+offset[k])/(1.+0.6078*q[k*n+i])).collect();
        if values.iter().any(|t|!t.is_finite() || !(100. ..400.).contains(t)){return Err("hydrostatic balance creates invalid temperature; revise donor/background compatibility".into());}
        Ok((i,values))
    }).collect();
    for (i,values) in columns? {for k in 0..nz {temp[k*n+i]=values[k];}}
    let mut ps=field(&b,"surface_pressure")?.to_vec();
    for i in 0..n {if ring[i]{let k=nz-1;let tv=temp[k*n+i]*(1.+0.6078*q[k*n+i]);ps[i]*=(solved[k].0[i]/(RD*tv)).exp();
        if !ps[i].is_finite() || !(10000. ..120000.).contains(&ps[i]){return Err("balance creates invalid surface pressure".into());}
    }}
    if let Some(mslp)=b.fields.get_mut("air_pressure_at_mean_sea_level") {
        for i in 0..n {if ring[i]{let k=nz-1;mslp[i]*=(solved[k].0[i]/(RD*temp[k*n+i]*(1.+0.6078*q[k*n+i]))).exp();}}
    }
    b.fields.insert("geopotential_height".into(),newz);b.fields.insert("air_temperature".into(),temp);b.fields.insert("surface_pressure".into(),ps);
    let balance_seconds=phase.elapsed().as_secs_f64();let phase=Instant::now();
    // TC-VP1: this member's balanced core perturbation, after the merge and its ring balance, before the audit
    let perturbation=match &req.core_perturbation {
        Some(p)=>Some(perturb_core(&mut b,p,(req.center_latitude_deg,req.center_longitude_deg))?),None=>None};
    let perturbation_seconds=phase.elapsed().as_secs_f64();let phase=Instant::now();
    let ps=field(&b,"surface_pressure")?;let q=field(&b,"specific_humidity")?;
    let mut mass_delta=0.;let mut water_delta=0.;let mut area=0.;
    for i in 0..n {if weight[i]>0.{let cell=(dx[i/nx]*dy).abs();area+=cell;mass_delta+=(ps[i]-original_ps[i])*cell/G;
        for k in 1..nz {let dp=(b.pressure[k].min(ps[i])-b.pressure[k-1].min(ps[i])).max(0.);
            let dp0=(b.pressure[k].min(original_ps[i])-b.pressure[k-1].min(original_ps[i])).max(0.);
            water_delta+=0.5*((q[k*n+i]+q[(k-1)*n+i])*dp-(original_q[k*n+i]+original_q[(k-1)*n+i])*dp0)*cell/G;
        }
    }}
    // Never overwrite an input. The manifest is published after the complete
    // stream; interruption leaves an uncommitted output directory, not valid data.
    fs::create_dir(&req.output).map_err(err)?;
    let stream_path=req.output.join("frames.f64");let mut output=File::create(&stream_path).map_err(err)?;let mut stream_hash=Sha256::new();let mut offset=0u64;
    let mut background_stream=File::open(req.background.join("frames.f64")).map_err(err)?;
    for frame in manifest["frames"].as_array_mut().ok_or("missing frames")? {
        let selected=frame["valid_time"]==req.valid_time;
        if selected {*frame=b.document.clone();}
        for desc in frame["fields"].as_array_mut().ok_or("missing fields")? {
            let name=string(desc,"name")?.to_owned();let mut bytes;
            if selected {bytes=b.fields[&name].iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>();}
            else {bytes=vec![0;usize_field(desc,"length")?];background_stream.seek(SeekFrom::Start(usize_field(desc,"offset")? as u64)).map_err(err)?;background_stream.read_exact(&mut bytes).map_err(err)?;}
            let shape:Vec<usize>=serde_json::from_value(desc["shape"].clone()).map_err(err)?;
            if !selected && array_hash(&shape,&bytes)!=string(desc,"sha256")? {return Err("untouched frame digest differs".into());}
            desc["offset"]=json!(offset);desc["length"]=json!(bytes.len());
            if selected {desc["missing_count"]=json!(b.fields[&name].iter().filter(|v|v.is_nan()).count());}
            desc["sha256"]=json!(array_hash(&shape,&bytes));offset+=bytes.len() as u64;stream_hash.update(&bytes);output.write_all(&bytes).map_err(err)?;
            if selected && merge_field(&name) {desc["source_references"].as_array_mut().ok_or("missing source references")?.push(json!(format!("storm-merge:{}",hash(&request_bytes))));}
        }
        if selected {
            let descriptors=frame["fields"].as_array().ok_or("missing fields")?.clone();
            for header in frame["header"]["fields"].as_array_mut().ok_or("missing header fields")? {
                let name=string(header,"canonical_name")?.to_owned();
                let desc=descriptors.iter().find(|d|d["name"]==name).ok_or("missing header field payload")?;
                header["data_reference"]=json!(format!("sha256:{}",string(desc,"sha256")?));
                if merge_field(&name) {
                    header["source_field"]=json!(format!("{};storm-merge:{}",string(header,"source_field")?,hash(&request_bytes)));
                }
            }
            for (path,digest) in d.document["input_sha256"].as_object().ok_or("missing donor input digests")? {frame["input_sha256"][path]=digest.clone();}
        }
    }
    output.flush().map_err(err)?;
    manifest["stream"]["bytes"]=json!(offset);manifest["stream"]["sha256"]=json!(format!("{:x}",stream_hash.finalize()));
    let receipt=json!({"schema":"gpuwm-storm-merge-receipt-v1","request_sha256":hash(&request_bytes),
        "background_frames_sha256":hash(&fs::read(req.background.join("frames.json")).map_err(err)?),
        "donor_frames_sha256":hash(&fs::read(req.donor.join("frames.json")).map_err(err)?),
        "center_deg":[req.center_latitude_deg,req.center_longitude_deg],"radius_km":req.radius_km,"blend_width_km":req.blend_width_km,
        "weight":"quintic C2 before balance, donor core and bitwise unchanged exterior", "balance":"pressure-coordinate nonlinear balance ring solve, rowwise local metric, no spherical connection terms; exact trapezoidal hypsometry with least-squares temperature",
        "donor_pressure_range_pa":[d.pressure[0],d.pressure[d.pressure.len()-1]],"above_donor_top":"background retained before ring hydrostatic adjustment",
        "solver_iterations_by_level":solved.iter().map(|v|v.1).collect::<Vec<_>>(),"solver_relative_residual_by_level":solved.iter().map(|v|v.2).collect::<Vec<_>>(),
        "donor_missing_preserved_background":donor_missing,"donor_tracers_added_over_zero_background":added_tracers,"area_m2":area,"column_mass_change_kg":mass_delta,"resolved_pressure_layer_vapor_change_kg":water_delta,
        "conservation_note":"Insertion changes core mass and water; changes are audited, not forced to zero. Vapor integral excludes pressure above/below supplied levels.",
        "target_grid":req.target_grid.as_ref().map(|g|json!({"south":g.south,"west":g.west,"spacing_deg":g.spacing_deg,"nx":g.nx,"ny":g.ny})),
        "exterior_reference":if req.target_grid.is_some(){"resampled background on the declared target grid"}else{"original background bytes"},
        "timings_seconds":{"read":read_seconds,"target_regrid":target_regrid_seconds,"regrid_blend":blend_seconds,"balance":balance_seconds,"core_perturbation":perturbation_seconds,"write":phase.elapsed().as_secs_f64(),"total":started.elapsed().as_secs_f64()},
        "forecast_shock_validation":"required; not established by balance residual","core_perturbation":perturbation});
    fs::write(req.output.join("storm-merge-receipt.json"),serde_json::to_vec_pretty(&receipt).map_err(err)?).map_err(err)?;
    fs::write(req.output.join("frames.json"),serde_json::to_vec_pretty(&manifest).map_err(err)?).map_err(err)?;
    Ok(receipt)
}
fn main(){
    std::hint::black_box(concat!("GPUWM_BRIDGE_SOURCE_REV=",env!("GPUWM_BRIDGE_SOURCE_REV")));
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=2 || args[0]!="--request" {eprintln!("rw_storm_merge --request request.json");std::process::exit(2);}
    match run(Path::new(&args[1])) {Ok(receipt)=>println!("{receipt}"),Err(e)=>{eprintln!("{}",json!({"schema":"gpuwm-storm-merge-refusal-v1","reason":e}));std::process::exit(2);}}
}

#[cfg(test)]
mod humidity_tests {
    use super::*;
    #[test]
    fn warming_preserves_saturation_fraction_and_requested_virtual_temperature() {
        for (t,p,rh,dtv) in [(290.,85000.,0.95,8.),(245.,40000.,1.04,6.),(220.,20000.,0.8,-2.)] {
            let q=rh*saturation_specific_humidity(t,p);
            let (after,water)=warm_at_constant_rh(t,q,p,dtv).unwrap();
            assert!((water/saturation_specific_humidity(after,p)-rh).abs()<1.0e-12);
            assert!((after*(1.+EPS_V*water)-t*(1.+EPS_V*q)-dtv).abs()<1.0e-10);
            assert_eq!(warm_at_constant_rh(t,q,p,0.).unwrap(),(t,q));
        }
    }
}
