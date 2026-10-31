//! Owned whole-driver geography fixtures; all field-byte processing is Rust.
use std::{fs,path::{Path,PathBuf}};

/// Whole case7 inputs mixing finite and exceptional terrain columns.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7130-7413
/// at 1d6e370e. This generates owned synthetic inputs, not native expectations.
#[test]
#[ignore = "requires an authenticated finite static file and an owned output directory"]
fn prepare_mixed_exceptional_terrain_statics() {
    use rw_mpas::init::artifact::{materialize,Request,Variable};
    use sha2::{Digest,Sha256};
    let source=PathBuf::from(std::env::var("RRFS_MIXED_STATIC_SOURCE").unwrap());
    let target=PathBuf::from(std::env::var("RRFS_MIXED_STATIC_TARGET").unwrap());
    let sha=|path:&Path|format!("{:x}",Sha256::digest(fs::read(path).unwrap()));
    let source_sha=sha(&source);
    assert_eq!(source_sha,std::env::var("RRFS_MIXED_STATIC_SOURCE_SHA256").unwrap());
    assert!(!target.exists());fs::create_dir_all(&target).unwrap();
    let input=netcrust::File::open(&source).unwrap();let field=input.variable("ter").unwrap();
    assert_eq!(field.dtype(),&netcrust::DataType::F32);
    let original:Vec<f32>=input.read_array::<f32>("ter").unwrap().into_iter().collect();
    assert!(original.len()>16 && original.iter().all(|value|value.is_finite()));
    let mut outputs=Vec::new();
    for pattern in ["half","interleaved"] {
        let values:Vec<f32>=original.iter().enumerate().map(|(index,&value)| {
            if (pattern=="half" && index>=original.len()/2) || (pattern=="interleaved" && index%8==0) {
                f32::from_bits(0x7fc00000)
            } else {value}
        }).collect();
        let output=target.join(format!("{pattern}.nc"));
        materialize(Request{template:source.clone(),overlays:Vec::new(),output:output.clone(),dimensions:Default::default(),
            variables:vec![Variable{name:"ter".into(),dimensions:field.dimensions().iter().map(|d|d.name().into()).collect(),
                attributes:Default::default(),values:Some(values.clone()),text:None,only_if_zero:false,only_if_absent:false}],
            attributes:Default::default()}).unwrap();
        let view=netcrust::File::open(&output).unwrap();
        let copied:Vec<f32>=view.read_array::<f32>("ter").unwrap().into_iter().collect();
        assert_eq!(values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),copied.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        let comparison=rw_mpas::staticfile::compare::compare(&source,&output).unwrap();
        assert!(comparison.dimensions_agree && comparison.only_in_reference.is_empty() && comparison.only_in_candidate.is_empty());
        assert!(comparison.type_mismatch.is_empty() && comparison.shape_mismatch.is_empty());
        assert!(comparison.fields.iter().filter(|f|f.name!="ter").all(|f|f.n==f.exact));
        outputs.push(serde_json::json!({"pattern":pattern,"path":output,"sha256":sha(&output),
            "finite_cells":values.iter().filter(|v|v.is_finite()).count(),"exceptional_cells":values.iter().filter(|v|v.is_nan()).count(),
            "all_other_stored_words_preserved":true}));
    }
    assert_eq!(sha(&source),source_sha);
    fs::write(target.join("FIXTURE.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"r10-mixed-exceptional-terrain-fixture/v1","source":source,"source_sha256":source_sha,
        "producer_sha256":sha(&std::env::current_exe().unwrap()),"synthetic_input_only":true,
        "native_expectations_supplied":false,"outputs":outputs})).unwrap()).unwrap();
}

/// Whole operator driver input for the unset-pivot witness.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_atm_advection.F:567-743 at 1d6e370e.
#[test]
#[ignore]
fn prepare_singular_operator_static() {
    use rw_mpas::init::artifact::{materialize,Request,Variable};
    let source=PathBuf::from(std::env::var("RRFS_OPERATOR_SOURCE").unwrap());
    let target=PathBuf::from(std::env::var("RRFS_OPERATOR_TARGET").unwrap());
    assert!(!target.exists(),"fixture must not overwrite an existing file");
    let file=netcrust::File::open(&source).unwrap();
    let mut variables=Vec::new();
    for name in ["yCell","yEdge","yVertex"] {
        let field=file.variable(name).unwrap();
        let count=field.dimensions().iter().map(|d|d.len()).product();
        variables.push(Variable {name:name.into(),
            dimensions:field.dimensions().iter().map(|d|d.name().into()).collect(),
            attributes:Default::default(),values:Some(vec![0.;count]),text:None,
            only_if_zero:false,only_if_absent:false});
    }
    // Carry every other mesh/static word. The collapsed great-circle geometry
    // reaches the complete polynomial-fit driver, not a substituted MIGS call.
    materialize(Request {template:source,overlays:Vec::new(),output:target,dimensions:Default::default(),
        variables,attributes:Default::default()}).unwrap();
}

fn copy_tree(source:&Path,target:&Path) {
    fs::create_dir_all(target).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry=entry.unwrap();let to=target.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {copy_tree(&entry.path(),&to)}
        else {fs::copy(entry.path(),to).unwrap();}
    }
}

#[test]
#[ignore]
fn prepare_complete_static_geography() {
    let source=PathBuf::from(std::env::var("RRFS_GEOG_SOURCE").unwrap());
    let target=PathBuf::from(std::env::var("RRFS_GEOG_TARGET").unwrap());
    assert!(!target.exists(),"fixture output must be a fresh owned directory");
    copy_tree(&source,&target);
    let green=target.join("greenfrac_fpar_modis");
    let lai=target.join("lai_modis_30s");copy_tree(&green,&lai);
    let index=fs::read_to_string(lai.join("index")).unwrap();
    fs::write(lai.join("index"),index.replace("scale_factor=0.01","scale_factor=0.1")).unwrap();
    for (family,categories) in [("bnu_soiltype_top",16u16),("soiltype_top_30s",16),("modis_landuse_20class_30s",20),("modis_landuse_20class_15s",20)] {
        let dir=target.join(family);
        let index=fs::read_to_string(dir.join("index")).unwrap();
        fs::write(dir.join("index"),index.replace("category_max=21",&format!("category_max={categories}"))).unwrap();
        for entry in fs::read_dir(&dir).unwrap() {
            let path=entry.unwrap().path();if path.file_name().unwrap()=="index" {continue;}
            let mut bytes=fs::read(&path).unwrap();assert_eq!(bytes.len()%2,0);
            for word in bytes.chunks_exact_mut(2) {
                let value=u16::from_be_bytes([word[0],word[1]]);
                if value!=255 {word.copy_from_slice(&((value.saturating_sub(1)%categories)+1).to_be_bytes());}
            }
            fs::write(path,bytes).unwrap();
        }
    }
}

/// A complete one-degree terrain tile provides samples inside every coarse
/// MPAS cell. The earlier 45-degree fixture intentionally leaves empty cell
/// averages; retain it for native IEEE propagation, not finite-column replay.
#[test]
#[ignore]
fn prepare_finite_terrain_geography() {
    let source=PathBuf::from(std::env::var("RRFS_GEOG_SOURCE").unwrap());
    let target=PathBuf::from(std::env::var("RRFS_GEOG_TARGET").unwrap());
    assert!(!target.exists());copy_tree(&source,&target);
    let terrain=target.join("topo_gmted2010_30s");
    fs::remove_dir_all(&terrain).unwrap();fs::create_dir(&terrain).unwrap();
    fs::write(terrain.join("index"),"type=continuous\nprojection=regular_ll\ndx=1\ndy=1\nknown_x=1\nknown_y=1\nknown_lat=-89.5\nknown_lon=-179.5\ntile_x=360\ntile_y=180\ntile_z=1\ntile_bdr=1\nwordsize=2\nsigned=yes\nscale_factor=1\nmissing_value=-32768\n").unwrap();
    let mut bytes=Vec::with_capacity(362*182*2);
    for j in 0..182i32 {for i in 0..362i32 {
        let x=(i-1).rem_euclid(360);let y=(j-1).clamp(0,179);
        bytes.extend_from_slice(&(200i16+(x%20) as i16+(y%20) as i16).to_be_bytes());
    }}
    fs::write(terrain.join("00001-00360.00001-00180"),bytes).unwrap();
}

/// Complete fixed-format native GWD input, sharing immutable repeating tiles
/// through hard links. MPAS reads no index metadata on this branch.
#[test]
#[ignore]
fn prepare_native_gwd_geography() {
    let target=PathBuf::from(std::env::var("RRFS_GWD_TARGET").unwrap());
    assert!(!target.exists());fs::create_dir_all(target.join("topo_gmted2010_30s")).unwrap();
    fs::create_dir_all(target.join("landuse_30s")).unwrap();
    let topo=target.join("topo-template.bin");let lu=target.join("landuse-template.bin");
    let mut height=Vec::with_capacity(1206*1206*2);
    for j in 0..1206i32 {for i in 0..1206i32 {
        height.extend_from_slice(&(((i-3).rem_euclid(1200)*3+(j-3).rem_euclid(1200)*5).rem_euclid(1600).wrapping_sub(200) as i16).to_be_bytes());
    }}
    fs::write(&topo,height).unwrap();let mut mask=Vec::with_capacity(1200*1200);
    for j in 0..1200 {for i in 0..1200 {mask.push(if (i/80+j/80)%3==0 {16u8} else {1u8});}}
    fs::write(&lu,mask).unwrap();
    for y in (1..21601).step_by(1200) {for x in (1..43201).step_by(1200) {
        let name=format!("{x:05}-{:05}.{y:05}-{:05}",x+1199,y+1199);
        fs::hard_link(&topo,target.join("topo_gmted2010_30s").join(&name)).unwrap();
        fs::hard_link(&lu,target.join("landuse_30s").join(&name)).unwrap();
    }}
}
