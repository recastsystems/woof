//! Complete file-to-file vertical and reconstruction artifact preparation.
//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905,
//! src/operators/mpas_vector_operations.F and mpas_vector_reconstruction.F
//! at 1d6e370e. Shared geometry sources are identical (source-identity receipt).
use std::{collections::{BTreeMap,BTreeSet},path::PathBuf};
use serde::Deserialize;
use rayon::prelude::*;
use super::artifact::{self,Variable};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub static_path:PathBuf,
    pub output_path:PathBuf,
    /// Explicit numerical parameters. Mesh arrays are supplied by Rust.
    pub vertical:serde_json::Value,
    /// Native config_specified_zeta_levels, decoded at the Rust file boundary.
    /// RRFSx/MPAS-Model cases.F:4555-4578,10387-10478 at 1d6e370e.
    #[serde(default)]
    pub specified_levels_path:Option<PathBuf>,
    pub soil_levels:usize,
    /// Reconstruction belongs to the meteorological stage, not the vertical
    /// stage. RRFSx/MPAS-Model cases.F:4905-4909,7443-7444 at 1d6e370e.
    pub met_interp:bool,
    pub variables:Vec<Variable>,
    pub attributes:BTreeMap<String,serde_json::Value>,
    /// Fields retired from the selected Registry, including fields in an older
    /// static input. The declaration table owns this list.
    #[serde(default)]
    pub omit_fields:Vec<String>,
}

/// One RKIND value per successful formatted READ. Each READ discards the
/// remaining record; blank/comment records are skipped and an invalid number
/// terminates the readable prefix. The caller checks the interface count.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:10387-10478
/// at 1d6e370e. Reuse the shared Fortran list-directed single-real conversion.
fn specified_levels(path:&std::path::Path)->Result<Vec<f64>,String> {
    let text=std::fs::read_to_string(path).map_err(|e|format!("specified zeta levels {}: {e}",path.display()))?;
    let mut values=Vec::new();
    for record in text.lines() {
        let value=record.split('!').next().unwrap_or("").trim();
        if value.is_empty() {continue;}
        match crate::static_geog::list_directed_real(value) {
            Some(number)=>values.push(f64::from(number)),
            None=>break,
        }
    }
    if values.is_empty() {return Err("specified zeta text file contains no readable real values".into());}
    Ok(values)
}

/// Stored single-precision reads, before any promotion to protocol JSON.
/// RRFSx/MPAS-Model src/framework/mpas_io.F at 1d6e370e.
fn real(f:&netcrust::File,name:&str)->Result<Vec<f32>,String> {
    f.read_array::<f32>(name).map(|a|a.into_iter().collect()).map_err(|e|e.to_string())
}
/// MPAS connectivity is one-based, with zero for an absent neighbor.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905 at 1d6e370e.
fn index(f:&netcrust::File,name:&str)->Result<Vec<i64>,String> {
    f.read_array::<i32>(name).map(|a|a.into_iter().map(|v|i64::from(v)-1).collect()).map_err(|e|e.to_string())
}
/// Native RKIND protocol output; RRFSx/MPAS-Model cases.F:4448-4905 at 1d6e370e.
fn take(raw:&[u8],cursor:&mut usize,n:usize)->Result<Vec<f32>,String> {
    let end=cursor.checked_add(n.checked_mul(4).ok_or("array size overflow")?).ok_or("array size overflow")?;
    let bytes=raw.get(*cursor..end).ok_or("truncated numerical producer output")?;*cursor=end;
    Ok(bytes.par_chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect())
}
/// Native level/entity file ordering; RRFSx/MPAS-Model Registry.xml:1105-1149 at 1d6e370e.
fn transpose(a:&[f32],levels:usize,entities:usize)->Vec<f32> {
    // Each entity's run of levels on its own core; same words, same order.
    let mut result=vec![0f32;entities*levels];
    result.par_chunks_mut(levels.max(1)).enumerate()
        .for_each(|(c,run)| {for k in 0..levels {run[k]=a[k*entities+c];}});
    result
}
/// Native vertical field declarations; RRFSx/MPAS-Model Registry.xml:1105-1149 at 1d6e370e.
fn variable(name:&str,dims:&[&str],values:Vec<f32>)->Variable {
    Variable{name:name.into(),dimensions:dims.iter().map(|s|(*s).into()).collect(),
        attributes:BTreeMap::new(),values:Some(values),text:None,only_if_zero:false,only_if_absent:false}
}

/// One whole artifact: native-order vertical construction, edge metrics,
/// spherical vector geometry/reconstruction and Rust container emission.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4448-4905 at 1d6e370e.
pub fn run(mut r:Request)->Result<artifact::Receipt,String> {
    let mut clock=super::StageClock::new("prepare");
    let mut names=std::collections::BTreeSet::new();
    if r.variables.iter().any(|v|!names.insert(v.name.clone())) {
        return Err("duplicate caller artifact declaration".into());
    }
    let f=netcrust::File::open(&r.static_path).map_err(|e|e.to_string())?;
    let dim=|name:&str|f.dimension(name).map(|d|d.len()).ok_or_else(||format!("missing dimension {name}"));
    let (nc,ne,me)=(dim("nCells")?,dim("nEdges")?,dim("maxEdges")?);
    let cfg=r.vertical.as_object_mut().ok_or("vertical parameters must be an object")?;
    let n=cfg.get("levels").and_then(|v|v.as_u64()).ok_or("missing levels")? as usize;
    if r.soil_levels==0 {return Err("soil_levels must be supplied and positive".into());}
    let counts:Vec<usize>=f.read_array::<i32>("nEdgesOnCell").map_err(|e|e.to_string())?.into_iter().map(|v|v as usize).collect();
    let coc=index(&f,"cellsOnCell")?;let eoc=index(&f,"edgesOnCell")?;let coe=index(&f,"cellsOnEdge")?;
    // Deserialize configuration only. Weather arrays stay typed so unused
    // native NaN coefficient slots survive the file-to-file path unchanged.
    for name in ["terrain","dc","dv","area","counts","cells_on_cell",
        "edges_on_cell","cells_on_edge","deriv","zgrid"] {
        cfg.insert(name.into(),serde_json::json!([]));
    }
    cfg.insert("mode".into(),serde_json::json!("grid"));
    cfg.insert("width".into(),serde_json::json!(4));
    let mut vertical:super::vertical::Request=serde_json::from_value(r.vertical).map_err(|e|e.to_string())?;
    // File values remain typed: JSON cannot carry native IEEE NaN heights.
    // RRFSx/MPAS-Model cases.F:4555-4578,10387-10478 at 1d6e370e.
    if let Some(path)=r.specified_levels_path.as_ref() {
        vertical.scheme="specified".into();
        vertical.specified=Some(specified_levels(path)?);
    }
    // The file boundary owns these mesh reads. A resolved job supplies the
    // intermediate path; orchestration need not decode weather coordinates.
    // RRFSx/MPAS-Model cases.F:4468-4483 at 1d6e370e obtains the same arrays
    // from the input mesh before blend_bdy_terrain.
    if let Some(boundary) = vertical.boundary.as_mut() {
        if boundary.lat.is_empty() && boundary.lon.is_empty() && boundary.mask.is_empty() {
            boundary.lat = real(&f,"latCell")?;
            boundary.lon = real(&f,"lonCell")?;
            boundary.mask = f.read_array::<i32>("bdyMaskCell")
                .map_err(|e|e.to_string())?.into_iter().collect();
        }
    }
    let widen=|name:&str|real(&f,name).map(|v|v.into_iter().map(f64::from).collect::<Vec<_>>());
    vertical.terrain=widen("ter")?;vertical.dc=widen("dcEdge")?;
    vertical.dv=widen("dvEdge")?;vertical.area=widen("areaCell")?;
    vertical.counts=counts.clone();
    // The per-cell and per-edge connectivity vectors (tens of millions of
    // small blocks on conus3km) are built on every core, in the same order.
    vertical.cells_on_cell=coc.par_chunks_exact(me).map(|v|v.to_vec()).collect();
    vertical.edges_on_cell=eoc.par_chunks_exact(me).map(|v|v.to_vec()).collect();
    vertical.cells_on_edge=coe.par_chunks_exact(2).map(|v|[v[0],v[1]]).collect();
    let deriv=widen("deriv_two")?;
    vertical.deriv=deriv.par_chunks_exact(30).map(|v|v.chunks_exact(15).map(|s|s.to_vec()).collect()).collect();
    clock.mark("read static mesh");
    let mut raw=Vec::new();super::vertical::run_request(&mut vertical,&mut raw)?;
    clock.mark("vertical grid");
    let mut cursor=0;let mut vectors=Vec::new();
    for len in [n+1,n,n,n,n,n,n,n,n,n,n+1] {vectors.push(take(&raw,&mut cursor,len)?);}
    let hx=take(&raw,&mut cursor,(n+1)*nc)?;let zg=take(&raw,&mut cursor,(n+1)*nc)?;
    let zz=take(&raw,&mut cursor,n*nc)?;let zxu=take(&raw,&mut cursor,n*ne)?;let dss=take(&raw,&mut cursor,n*nc)?;
    let cf=take(&raw,&mut cursor,3)?;
    if raw.len()!=cursor+8 {return Err("vertical producer output length differs from its contract".into());}
    // Blending belongs to the initial case7 terrain/grid phase. The edge
    // phase consumes that completed grid and must not attempt another blend.
    // RRFSx/MPAS-Model cases.F:4468-4483,4871-4905 at 1d6e370e.
    vertical.boundary=None;
    vertical.mode="edge".into();
    vertical.zgrid=zg.chunks_exact(nc).map(|v|v.iter().copied().map(f64::from).collect()).collect();
    raw.clear();super::vertical::run_request(&mut vertical,&mut raw)?;
    clock.mark("edge metrics");
    cursor=0;
    for name in ["zb","zb3"] {
        let values=take(&raw,&mut cursor,(n+1)*2*ne)?;let mut entity=vec![0f32;values.len()];
        // Each edge's two sides of levels on its own core; same words, same order.
        entity.par_chunks_mut(2*(n+1)).enumerate().for_each(|(e,run)| {
            for side in 0..2 {for k in 0..=n {run[side*(n+1)+k]=values[(k*2+side)*ne+e];}}
        });
        r.variables.push(variable(name,&["nEdges","TWO","nVertLevelsP1"],entity));
    }
    for (name,values,dims) in [("ter",hx[..nc].to_vec(),vec!["nCells"]),
        ("zgrid",transpose(&zg,n+1,nc),vec!["nCells","nVertLevelsP1"]),
        ("zz",transpose(&zz,n,nc),vec!["nCells","nVertLevels"]),
        ("zxu",transpose(&zxu,n,ne),vec!["nEdges","nVertLevels"]),
        ("dss",transpose(&dss,n,nc),vec!["nCells","nVertLevels"])] {r.variables.push(variable(name,&dims,values));}
    for (i,name) in [(2,"rdzw"),(4,"dzu"),(5,"rdzu"),(8,"fzp"),(9,"fzm")] {
        r.variables.push(variable(name,&["nVertLevels"],vectors[i].clone()));
    }
    for (i,name) in ["cf1","cf2","cf3"].iter().enumerate() {r.variables.push(variable(name,&[],vec![cf[i]]));}
    // The native vertical-only stage carries geometry untouched. The shared
    // geometry driver is needed only by the subsequent meteorological stage.
    clock.mark("transposes");
    if r.met_interp {
    r.variables.extend(reconstruction_geometry(&r.static_path)?);
    }
    clock.mark("reconstruction geometry");
    let mut dimensions=BTreeMap::from([("nVertLevels".into(),Some(n)),("nVertLevelsP1".into(),Some(n+1)),("Time".into(),None)]);
    if r.met_interp {dimensions.insert("nSoilLevels".into(),Some(r.soil_levels));}
    // Numerical values replace the caller's metadata-only declarations.
    let mut merged:BTreeMap<String,Variable>=BTreeMap::new();
    for mut v in r.variables {
        if let Some(previous)=merged.remove(&v.name) {
            if v.dimensions!=previous.dimensions {return Err(format!("computed {} dimensions differ from caller declaration",v.name));}
            let mut attributes=previous.attributes;attributes.extend(v.attributes);v.attributes=attributes;
        }
        merged.insert(v.name.clone(),v);
    }
    let selected=if r.omit_fields.is_empty() {None} else {
        let source=netcrust::File::open(&r.static_path).map_err(|e|e.to_string())?;
        let mut names:BTreeSet<String>=source.variables().map_err(|e|e.to_string())?
            .iter().map(|v|v.name().to_string()).chain(merged.keys().cloned()).collect();
        for name in &r.omit_fields {names.remove(name);merged.remove(name);}
        Some(names)
    };
    let written=artifact::materialize_selected(artifact::Request{overlays:Vec::new(),template:r.static_path,output:r.output_path,dimensions,variables:merged.into_values().collect(),attributes:r.attributes},selected.as_ref()).map_err(|e|e.to_string());
    clock.mark("write artifact");
    // The vertical request's per-cell vectors are freed off the main thread.
    std::thread::spawn(move || drop(vertical));
    written
}

/// Complete fresh reconstruction geometry at the case7 driver boundary.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7442-7451,
/// src/operators/mpas_vector_operations.F and mpas_vector_reconstruction.F
/// at 1d6e370e. Shared geometry arithmetic is source-identical to v8.4.2.
pub fn reconstruction_geometry(path:&std::path::Path)->Result<Vec<Variable>,String> {
    let f=netcrust::File::open(path).map_err(|e|e.to_string())?;
    let dim=|name:&str|f.dimension(name).map(|d|d.len()).ok_or_else(||format!("missing dimension {name}"));
    let (nc,ne,me)=(dim("nCells")?,dim("nEdges")?,dim("maxEdges")?);
    let coe=index(&f,"cellsOnEdge")?;let eoc=index(&f,"edgesOnCell")?;
    let counts:Vec<usize>=f.read_array::<i32>("nEdgesOnCell").map_err(|e|e.to_string())?.into_iter().map(|v|v as usize).collect();
    let mut variables=Vec::new();
    let mut input=b"HEXGEO1\0".to_vec();for v in [nc,ne,me,nc] {input.extend_from_slice(&(v as u64).to_le_bytes());}
    let sphere=f.attribute("on_a_sphere").map(|a|a.as_string().unwrap_or("").contains("YES")).unwrap_or(true);
    input.extend_from_slice(&[4,u8::from(sphere),0,0]);
    input.extend_from_slice(&0f32.to_le_bytes());input.extend_from_slice(&0f32.to_le_bytes());
    // The protocol bytes are encoded on every core into their final places;
    // same bytes, same order as appending them one value at a time.
    for names in [["xCell","yCell","zCell"],["xEdge","yEdge","zEdge"]] {
        let values=names.map(|name|real(&f,name)).into_iter().collect::<Result<Vec<_>,_>>()?;
        let start=input.len();input.resize(start+values[0].len()*12,0);
        input[start..].par_chunks_mut(12).enumerate().for_each(|(i,out)| {
            for axis in 0..3 {out[axis*4..axis*4+4].copy_from_slice(&values[axis][i].to_le_bytes());}
        });
    }
    for values in [&coe,&eoc] {
        let start=input.len();input.resize(start+values.len()*8,0);
        input[start..].par_chunks_mut(8).zip(values.par_iter()).for_each(|(out,v)|out.copy_from_slice(&v.to_le_bytes()));
    }
    let start=input.len();input.resize(start+counts.len()*8,0);
    input[start..].par_chunks_mut(8).zip(counts.par_iter()).for_each(|(out,&v)|out.copy_from_slice(&(v as i64).to_le_bytes()));
    let mut raw=Vec::new();crate::geometry::run(std::io::Cursor::new(input),&mut raw)?;let mut cursor=0;
    for (name,len) in [("edgeNormalVectors",ne*3),("localVerticalUnitVectors",nc*3),("cellTangentPlane",nc*6),("coeffs_reconstruct",nc*me*3)] {
        let values=take(&raw,&mut cursor,len)?;
        let dims=match name {
            "edgeNormalVectors"=>vec!["nEdges","R3"],
            "localVerticalUnitVectors"=>vec!["nCells","R3"],
            "cellTangentPlane"=>vec!["nCells","TWO","R3"],
            "coeffs_reconstruct"=>vec!["nCells","maxEdges","R3"],
            _=>unreachable!("closed reconstruction field inventory"),
        }.into_iter().map(str::to_string).collect();
        variables.push(Variable{name:name.into(),dimensions:dims,attributes:BTreeMap::new(),values:Some(values),text:None,only_if_zero:true,only_if_absent:false});
    }
    Ok(variables)
}
