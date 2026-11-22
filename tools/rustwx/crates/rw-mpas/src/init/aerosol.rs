//! Native aerosol augmentation. Python resolves datasets; Rust computes
//! the WRF synthetic fallback and writes the complete model species row.
use std::{collections::BTreeMap, path::Path};
use crate::error::{MpasError, MpasResult};
use super::{emit::{Computed, write_init_declaring}, Lineage};

pub fn analyzed_pair(path: &Path) -> MpasResult<bool> {
    let file=netcrust::File::open(path)?;
    let vars=file.variables()?;
    for name in ["nwfa","nifa"] {
        if !vars.iter().any(|v|v.name()==name) { return Ok(false); }
        let values=file.read_array_f64(name)?.into_values();
        if values.iter().any(|x|!x.is_finite() || *x<0.) {
            return Err(MpasError::Refusal(format!("analysis aerosol {name} is non-finite or negative")));
        }
        if !values.iter().any(|x|*x>0.) { return Ok(false); }
    }
    Ok(true)
}

fn reconciled_categories(path: &Path) -> MpasResult<(Vec<i32>, usize)> {
    let file = netcrust::File::open(path)?;
    let xland: Vec<f32> = file.read_array_f64("xland")?.into_values().into_iter().map(|x|x as f32).collect();
    let mut soil: Vec<i32> = file.read_array_f64("isltyp")?.into_values().into_iter().map(|x|x as i32).collect();
    if xland.len() != soil.len() || xland.iter().any(|x|!x.is_finite()) {
        return Err(MpasError::Refusal("soil category reconciliation requires finite surface classification for every cell".into()));
    }
    let changed = super::soil::reconcile_soil_category(&xland, &mut soil);
    Ok((soil, changed))
}

pub fn soil_category_mismatches(path: &Path) -> MpasResult<usize> {
    Ok(reconciled_categories(path)?.1)
}

/// The aerosol climatology's target geometry, written as little-endian
/// float32 files in `dir` (cell-major, as the init stores them) so a 3 km
/// mesh's columns never pass through a text encoding. Pressure is the full
/// moist pressure the MPAS dycore diagnoses from the init state,
/// p0 * (Rd rho theta (1 + qv Rv/Rd) / p0)^(cp/cv); `phb` is WRF's base
/// geopotential at w levels, zgrid * 9.81, the form the WRF climatology
/// interpolation consumes.
pub fn target(path: &Path, dir: &Path) -> MpasResult<serde_json::Value> {
    use super::dynamics::constants::{CP, CV, P0, RGAS, RVORD};
    let file=netcrust::File::open(path)?;
    let read=|name| -> MpasResult<Vec<f32>> { Ok(file.read_array_f64(name)?.into_values().into_iter().map(|x|x as f32).collect()) };
    let rho=read("rho")?; let theta=read("theta")?; let qv=read("qv")?;
    if rho.len()!=theta.len() || rho.len()!=qv.len() {
        return Err(MpasError::Refusal("aerosol target needs rho, theta and qv on one grid".into()));
    }
    let pressure:Vec<f32>=rho.iter().zip(&theta).zip(&qv).map(|((&r,&t),&q)|
        P0 * (RGAS * r * t * (1.0 + RVORD * q) / P0).powf(CP / CV)).collect();
    let lat:Vec<f32>=read("latCell")?.iter().map(|x|x.to_degrees()).collect();
    let lon:Vec<f32>=read("lonCell")?.iter().map(|x|x.to_degrees()).collect();
    let phb:Vec<f32>=read("zgrid")?.iter().map(|x| x*9.81).collect();
    let cells=lat.len();
    if cells==0 || pressure.len()%cells!=0 || phb.len()!=cells*(pressure.len()/cells+1) {
        return Err(MpasError::Refusal("aerosol target geometry is inconsistent with nCells".into()));
    }
    std::fs::create_dir_all(dir)?;
    let mut files=serde_json::Map::new();
    for (name,values) in [("latitude",&lat),("longitude",&lon),("pressure",&pressure),("phb",&phb)] {
        let file=format!("{name}.f32le");
        write_f32le(&dir.join(&file),values)?;
        files.insert(name.into(),serde_json::Value::String(file));
    }
    Ok(serde_json::json!({"cells":cells,"levels":pressure.len()/cells,"files":files}))
}

fn write_f32le(path: &Path, values: &[f32]) -> MpasResult<()> {
    let mut bytes=Vec::with_capacity(values.len()*4);
    for v in values { bytes.extend_from_slice(&v.to_le_bytes()); }
    std::fs::write(path,bytes)?;
    Ok(())
}

fn read_f32le(path: &Path, count: usize) -> MpasResult<Vec<f32>> {
    let bytes=std::fs::read(path)?;
    if bytes.len()!=count*4 {
        return Err(MpasError::Refusal(format!("{} holds {} bytes; {} float32 values expected", path.display(), bytes.len(), count)));
    }
    Ok(bytes.chunks_exact(4).map(|b|f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect())
}

pub fn synthetic(heights: &[f32]) -> MpasResult<(Vec<f32>, Vec<f32>)> {
    if heights.len()<2 || heights.iter().any(|x| !x.is_finite()) || heights[1]<=heights[0] {
        return Err(MpasError::Refusal("aerosol profile requires increasing finite w-level heights".into()));
    }
    let h = heights[0];
    let scale = if h<=1000. {0.8} else if h>=2500. {0.01} else {0.8*(h*0.001-1.).cos()};
    let profile = |bottom:f32, top:f32| {
        let decay = -(top/bottom).ln()/scale;
        (0..heights.len()-1).map(|k| {
            let z = if k==0 {heights[1]} else {heights[k]};
            top + bottom * (-((z-h)/1000.)*decay).exp()
        }).collect::<Vec<f32>>()
    };
    Ok((profile(300e6,50e6),profile(1.5e6,0.5e6)))
}

pub fn enrich(path: &Path, scalar_names: &[String], fields: Option<&Path>, reconcile_soil: bool) -> MpasResult<serde_json::Value> {
    enrich_with_fill(path, scalar_names, &BTreeMap::new(), fields, reconcile_soil)
}

/// [`enrich`] with the door's own start values for species the file lacks.
///
/// `fill` maps a species of `scalar_names` to the uniform value it starts at when the init has no
/// variable for it (the engine's source-absent contract: NSSL's CCN number starts at WRF's
/// `ccn_conc`, every other absent moment at the allocation zero).  A species the file already
/// carries is never touched, with or without a fill.
pub fn enrich_with_fill(path: &Path, scalar_names: &[String], fill: &BTreeMap<String, f32>, fields: Option<&Path>, reconcile_soil: bool) -> MpasResult<serde_json::Value> {
    for (name, value) in fill {
        if !scalar_names.contains(name) {
            return Err(MpasError::Refusal(format!("--scalar-fill names {name}, which --scalar-names does not list")));
        }
        if !value.is_finite() || *value < 0. {
            return Err(MpasError::Refusal(format!("--scalar-fill {name} is non-finite or negative")));
        }
    }
    let file=netcrust::File::open(path)?;
    let dims=file.dimensions()?;
    let n=dims.iter().find(|d|d.name()=="nCells").unwrap().len();
    let nz=dims.iter().find(|d|d.name()=="nVertLevels").unwrap().len();
    let vars=file.variables()?;
    let mut computed=BTreeMap::new();
    let mut soil_categories_reconciled = 0;
    let mut created: Vec<String> = Vec::new();
    if reconcile_soil {
        let (categories, changed) = reconciled_categories(path)?;
        soil_categories_reconciled = changed;
        if changed > 0 { computed.insert("isltyp".into(), Computed::Ints(categories)); }
    }
    for name in ["xtime","initial_time","mminlu"] {
        if vars.iter().any(|v|v.name()==name) {
            let text=crate::lbc::compare::read_char_variable(path,name)?
                .ok_or_else(||MpasError::Refusal(format!("init label {name} is not a character field")))?;
            computed.insert(name.into(),Computed::Text(text));
        }
    }
    for name in scalar_names {
        if !vars.iter().any(|v|v.name()==name) {
            let start = fill.get(name).copied().unwrap_or(0.);
            computed.insert(name.clone(),Computed::Floats(vec![start;n*nz]));
            created.push(name.clone());
        }
    }
    let mut source="analysis";
    if scalar_names.iter().any(|s|s=="nwfa") {
        if let Some(fields)=fields {
            // `fields` is a directory holding nwfa.f32le and nifa.f32le,
            // cell-major n*nz float32 each, written by the climatology door.
            for name in ["nwfa","nifa"] {
                let values=read_f32le(&fields.join(format!("{name}.f32le")),n*nz)?;
                if values.iter().any(|x|!x.is_finite() || *x<0.) {
                    return Err(MpasError::Refusal(format!("climatology {name} is non-finite or negative")));
                }
                computed.insert(name.into(),Computed::Floats(values));
            }
            source="wif-climatology";
        } else if !analyzed_pair(path)? {
            let z=file.read_array_f64("zgrid")?.into_values();
            let mut w=Vec::with_capacity(n*nz); let mut i=Vec::with_capacity(n*nz);
            for c in 0..n {
                let heights:Vec<f32>=z[c*(nz+1)..(c+1)*(nz+1)].iter().map(|&v|v as f32).collect();
                let (cw,ci)=synthetic(&heights)?; w.extend(cw); i.extend(ci);
            }
            computed.insert("nwfa".into(),Computed::Floats(w));
            computed.insert("nifa".into(),Computed::Floats(i));
            source="synthetic";
        }
    }
    let temp=path.with_extension("aerosol.tmp.nc");
    let file_id=super::capsule_file_id(path).unwrap_or_else(||super::emit::mint_file_id(&path.display().to_string()));
    let parent_text=file.attribute("parent_id").and_then(|a|a.as_string().map(str::to_string)).unwrap_or_default();
    let parents:Vec<String>=parent_text.lines().filter(|s|!s.is_empty()).map(str::to_string).collect();
    let ledger=write_init_declaring(&temp,path,computed,scalar_names,&file_id,&parents,"aerosol initialization",&Lineage::default())?;
    drop(file);
    std::fs::rename(&temp,path)?;
    let filled: BTreeMap<&String, f32> = created.iter().filter_map(|name| fill.get(name).map(|v| (name, *v))).collect();
    Ok(serde_json::json!({"source":source,"cells":n,"levels":nz,"written":ledger.computed,"soil_categories_reconciled":soil_categories_reconciled,
        "species_created":created,"species_fill":filled}))
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn synthetic_matches_retained_wrf_init_profile() {
        let csv=include_str!("../../../../../../woof/data/thompson/oracle-aero/aero-init-profile-column.csv");
        let mut lines=csv.lines();
        let names:Vec<_>=lines.next().unwrap().split(',').collect();
        let rows:Vec<Vec<&str>>=lines.map(|line|line.split(',').collect::<Vec<_>>()).filter(|row|row[0]=="before").collect();
        let values=|name:&str| {
            let col=names.iter().position(|n|*n==name).unwrap();
            rows.iter().map(|r|r[col].trim().parse::<f32>().unwrap()).collect::<Vec<_>>()
        };
        let mut heights=values("z_m");
        heights.push(heights.last().unwrap()+values("dz_m").last().unwrap());
        let (nwfa,nifa)=synthetic(&heights).unwrap();
        for (name,got) in [("nwfa_per_kg",nwfa),("nifa_per_kg",nifa)] {
            for (k,(a,b)) in got.iter().zip(values(name)).enumerate() {
                assert_eq!(a.to_bits(),b.to_bits(),"WRF {name} level {}",k+1);
            }
        }
    }
    #[test] fn profile_has_wrf_first_level_and_terrain_thresholds() {
        let (w,i)=synthetic(&[0.,50.,1000.,3000.]).unwrap();
        assert_eq!(w.len(),3); assert_eq!(w[0],w[1]); assert_eq!(i[0],i[1]);
        assert!(w[2]<w[1] && w[2]>50e6);
        let (high,_)=synthetic(&[2500.,2550.,3500.,5000.]).unwrap();
        assert!(high[0]<w[0]);
        assert!(synthetic(&[0.,0.]).is_err());
    }
    #[test] fn a_fill_is_refused_before_the_file_is_opened() {
        // A start value for a species nobody listed, or one no species can start at, is refused by
        // name before the init is read: the path here does not exist and is never opened.
        let nowhere=Path::new("/nonexistent/init.nc");
        let names=vec!["qnn".to_string()];
        let fill=|name:&str,value:f32| BTreeMap::from([(name.to_string(),value)]);
        let unlisted=enrich_with_fill(nowhere,&names,&fill("qndrop",1.),None,false).unwrap_err().to_string();
        assert!(unlisted.contains("qndrop") && unlisted.contains("--scalar-names"),"{unlisted}");
        for bad in [f32::NAN,f32::INFINITY,-1.] {
            let refused=enrich_with_fill(nowhere,&names,&fill("qnn",bad),None,false).unwrap_err().to_string();
            assert!(refused.contains("qnn"),"{refused}");
        }
        // A good fill gets as far as the file itself.
        assert!(!enrich_with_fill(nowhere,&names,&fill("qnn",4.0e8),None,false).unwrap_err().to_string().contains("--scalar-fill"));
    }
}
