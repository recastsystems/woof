//! Native panels and polar-gate scores for one S/C/X/observed snapshot.
#[path = "../src/manifest.rs"] mod manifest;
#[path = "../src/ppi.rs"] mod ppi;
use bowecho_simradar::radar_core::{MomentType,RadarVolume};
use std::path::{Path,PathBuf};
use serde_json::json;

fn read(path:&str)->Result<RadarVolume,String> {
    nexrad_io::decode_supported_volume_bytes(&std::fs::read(path).map_err(|e|e.to_string())?)
}

fn sheet(root:&Path,name:&str,paths:&[PathBuf])->Result<manifest::Artifact,String> {
    let images:Vec<_>=paths.iter().map(|p|image::open(p).map(|v|v.to_rgba8()).map_err(|e|e.to_string()))
        .collect::<Result<_,_>>()?;
    let width=images.iter().map(|v|v.width()).sum::<u32>()+16*(images.len() as u32-1);
    let height=images.iter().map(|v|v.height()).max().ok_or("empty sheet")?;
    let mut result=image::RgbaImage::from_pixel(width,height,image::Rgba([245,247,249,255]));
    let mut x=0;
    for image in &images {
        image::imageops::overlay(&mut result,image,x,0);
        x+=i64::from(image.width()+16);
    }
    let path=root.join(name);
    result.save(&path).map_err(|e|e.to_string())?;
    manifest::artifact(root,&path,"png")
}

fn scores(model:&RadarVolume,observed:&RadarVolume)->Result<serde_json::Value,String> {
    let m=model.cuts.first().ok_or("model has no sweep")?;
    let o=observed.cuts.first().ok_or("observation has no sweep")?;
    if (m.elevation_deg-o.elevation_deg).abs()>0.01 {
        return Err("comparison sweep elevations differ; select the observed elevation in the simulation".into());
    }
    let mg=m.moments.get(&MomentType::Reflectivity).ok_or("model reflectivity absent")?;
    let og=o.moments.get(&MomentType::Reflectivity).ok_or("observed reflectivity absent")?;
    let mut all=Vec::new();let mut echo=Vec::new();
    for (mr,&ray) in mg.radial_indices.iter().enumerate() {
        let az=m.radials[ray].azimuth_deg;
        let orow=og.radial_indices.iter().enumerate().min_by(|(_,a),(_,b)| {
            let distance=|i:usize| ((o.radials[i].azimuth_deg-az+180.).rem_euclid(360.)-180.).abs();
            distance(**a).total_cmp(&distance(**b))
        }).ok_or("observation has no radial")?.0;
        for gate in 0..mg.gate_range.gate_count {
            let range=mg.gate_range.first_gate_m as f64+gate as f64*mg.gate_range.gate_spacing_m as f64;
            if range<10_000. {continue;}
            let index=((range-og.gate_range.first_gate_m as f64)/og.gate_range.gate_spacing_m as f64).round();
            if index<0. || index>=og.gate_range.gate_count as f64 {continue;}
            if let (Some(a),Some(b))=(mg.scaled_value(mr,gate),og.scaled_value(orow,index as usize)) {
                if a.is_finite() && b.is_finite() {
                    all.push(f64::from(a-b));
                    if a>=20. && b>=20. {echo.push(f64::from(a-b));}
                }
            }
        }
    }
    let summarize=|values:&[f64]| {
        if values.is_empty() {return json!({"count":0});}
        let n=values.len() as f64;
        json!({"count":values.len(),"bias_dbz":values.iter().sum::<f64>()/n,
            "mae_dbz":values.iter().map(|v|v.abs()).sum::<f64>()/n,
            "rmse_dbz":(values.iter().map(|v|v*v).sum::<f64>()/n).sqrt()})
    };
    Ok(json!({"method":"nearest observed azimuth/range gate; >=10 km; only jointly finite values",
        "all_jointly_finite":summarize(&all),"both_at_least_20_dbz":summarize(&echo),
        "scope":"conditional intensity errors, not a detection or coverage score"}))
}

fn main()->Result<(),String> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=5 {return Err("usage: band_report S_VOLUME C_VOLUME X_VOLUME OBSERVED_C_VOLUME OUTPUT_DIRECTORY".into());}
    let mut volumes=args[..4].iter().map(|p|read(p)).collect::<Result<Vec<_>,_>>()?;
    for v in &volumes[1..3] {
        if v.site.id!=volumes[0].site.id || v.volume_time!=volumes[0].volume_time {
            return Err("band panels must use the same site and model time".into());
        }
    }
    let (model,obs)=(&volumes[1],&volumes[3]);
    for (a,b) in [(model.site.latitude_deg,obs.site.latitude_deg),(model.site.longitude_deg,obs.site.longitude_deg)] {
        if !matches!((a,b),(Some(a),Some(b)) if (a-b).abs()<0.0011) {
            return Err("observation coordinates do not match the simulated radar".into());
        }
    }
    let score=scores(model,obs)?;
    let times=volumes.iter().map(|v|v.volume_time.to_rfc3339()).collect::<Vec<_>>();
    let offset=(obs.volume_time-model.volume_time).num_milliseconds() as f64*0.001;
    let site=model.site.id.clone();
    let reference=volumes[0].cuts[0].moments.get(&MomentType::Reflectivity).ok_or("S reflectivity absent")?;
    let range=reference.gate_range.first_gate_m as f64+reference.gate_range.gate_count as f64*reference.gate_range.gate_spacing_m as f64;
    for volume in &mut volumes {
        for cut in &mut volume.cuts {
            cut.moments.retain(|m,_|matches!(m,MomentType::Reflectivity|MomentType::DifferentialReflectivity|MomentType::SpecificDifferentialPhase));
        }
    }
    let root=Path::new(&args[4]);
    std::fs::create_dir_all(root).map_err(|e|e.to_string())?;
    let mut limits=Vec::new();
    for (name,moment,mut lo,mut hi,step) in [
        ("zdr",MomentType::DifferentialReflectivity,-4.0_f64,8.0_f64,2.0_f64),
        ("kdp",MomentType::SpecificDifferentialPhase,-1.0_f64,7.0_f64,5.0_f64)] {
        for volume in &volumes[..3] {
            let cut=&volume.cuts[0];
            let data=&cut.moments[&moment];
            let echo=&cut.moments[&MomentType::Reflectivity];
            for row in 0..data.radial_indices.len() {
                for gate in 0..data.gate_range.gate_count {
                    if echo.scaled_value(row,gate).is_some_and(|z|z>=10.) {
                        if let Some(v)=data.scaled_value(row,gate).filter(|v|v.is_finite()) {
                            lo=lo.min(f64::from(v));hi=hi.max(f64::from(v));
                        }
                    }
                }
            }
        }
        limits.push((name,(lo/step).floor()*step,(hi/step).ceil()*step));
    }
    let mut images=Vec::new();
    for (i,label) in ["S-band simulated","C-band simulated","X-band simulated","Observed C-band"].iter().enumerate() {
        images.push(ppi::render_labeled_with_scales(root,"d01",&format!("panel-{i}"),&volumes[i],1,900,900,label,Some(range),&limits)?);
    }
    let mut products=Vec::new();
    for field in ["reflectivity","zdr","kdp"] {
        let paths=(0..3).map(|i|images[i].iter().find(|p|p.field==field)
            .map(|p|root.join(&p.artifact.path)).ok_or_else(||format!("{field} absent in band panel {i}")))
            .collect::<Result<Vec<_>,_>>()?;
        products.push(sheet(root,&format!("bands-{field}.png"),&paths)?);
    }
    // A range limit only crops the native renderer; it does not expand a
    // shorter observed scan. Crop BOTH panels to the common last gate.
    let common_range=[1,3].iter().map(|&i| {
        let grid=&volumes[i].cuts[0].moments[&MomentType::Reflectivity].gate_range;
        grid.first_gate_m as f64+(grid.gate_count-1) as f64*grid.gate_spacing_m as f64
    }).fold(f64::INFINITY,f64::min);
    for (i,label) in [(1,"C-band simulated"),(3,"Observed C-band")] {
        let mut comparison=volumes[i].clone();
        for cut in &mut comparison.cuts {cut.moments.retain(|m,_|*m==MomentType::Reflectivity);}
        images.push(ppi::render_labeled(root,"d01",&format!("comparison-{i}"),&comparison,1,900,900,label,Some(common_range))?);
    }
    let paths=[4,5].map(|i|images[i].iter().find(|p|p.field=="reflectivity")
        .map(|p|root.join(&p.artifact.path)).ok_or("comparison reflectivity absent"))
        .into_iter().collect::<Result<Vec<_>,_>>()?;
    products.push(sheet(root,"model-observation.png",&paths)?);
    let hashes=args[..4].iter().map(|p|manifest::file_hash(Path::new(p)).map(|v|v.0)).collect::<Result<Vec<_>,_>>()?;
    let receipt=json!({"site":site,"times":times,"observation_offset_seconds":offset,
        "input_sha256":hashes,"scores":score,"products":products,"images":images,"comparison_range_m":common_range,
        "shared_panel_limits":limits,
        "scale_note":"S/C/X share limits within each frame; expanded to include every finite moment at reflectivity >=10 dBZ; read each frame legend",
        "renderer":"rw_wrfbatch::panel::render_panel; native polar sampling and image assembly",
        "band_comparison_scope":"S retains its legacy bulk operator; C/X use finite-particle tables and attenuation, so S/C differences also include the operator change"});
    manifest::atomic_bytes(&root.join("report.json"),&serde_json::to_vec_pretty(&receipt).map_err(|e|e.to_string())?)?;
    println!("{}",json!({"site":site,"times":times,"scores":score}));
    Ok(())
}
