//! Rendering support for complete native initialization-boundary forecasts.
//! RRFSx/MPAS-Model src/core_atmosphere/Registry.xml:602-695,3047-3598
//! at 1d6e370e. No forecast field is diagnosed here: stored diagnostic words
//! are copied into a disposable history view with explicit field aliases.
use std::{collections::BTreeMap, fs, io::Read, path::Path};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use rw_mpas::init::artifact::{Request, Variable, materialize};

fn sha(path: &Path) -> String {
    let mut input=fs::File::open(path).unwrap();let mut hasher=Sha256::new();
    let mut block=[0u8;65536];
    loop {let count=input.read(&mut block).unwrap();if count==0 {break;}hasher.update(&block[..count]);}
    format!("{:x}",hasher.finalize())
}
fn word_hash(values: &[f32]) -> String {
    let mut hasher=Sha256::new();
    for value in values {hasher.update(value.to_bits().to_le_bytes());}
    format!("{:x}",hasher.finalize())
}

#[test]
#[ignore = "requires authenticated history/diagnostic pairs and owned output paths"]
fn complete_history_diagnostic_render_views() {
    let plan_path=std::env::var("RRFS_RENDER_UNION_PLAN").unwrap();
    let plan:Value=serde_json::from_slice(&fs::read(&plan_path).unwrap()).unwrap();
    let rows=plan.as_array().unwrap();assert!(!rows.is_empty());let mut outputs=Vec::new();
    for row in rows {
        let history=Path::new(row["history"].as_str().unwrap());
        let diagnostic=Path::new(row["diagnostic"].as_str().unwrap());
        let mesh=Path::new(row["mesh"].as_str().unwrap());
        let output=Path::new(row["output"].as_str().unwrap());
        assert_eq!(sha(history),row["history_sha256"].as_str().unwrap());
        assert_eq!(sha(diagnostic),row["diagnostic_sha256"].as_str().unwrap());
        assert_eq!(sha(mesh),row["mesh_sha256"].as_str().unwrap());
        assert!(!output.exists());fs::create_dir_all(output.parent().unwrap()).unwrap();
        let hf=netcrust::File::open(history).unwrap();let df=netcrust::File::open(diagnostic).unwrap();
        let mf=netcrust::File::open(mesh).unwrap();
        let expected=row["valid_time"].as_str().unwrap();
        let clocks=hf.read_strings("xtime").unwrap();
        assert_eq!(clocks.len(),1);assert_eq!(clocks[0].trim_end_matches([' ','\0']),expected);
        assert_eq!(df.read_strings("xtime").unwrap(),clocks);
        let mut dimensions=BTreeMap::new();let mut variables=Vec::new();let mut fields=Vec::new();
        let aliases=row["fields"].as_object().unwrap();assert!(!aliases.is_empty());
        let mut selected:Vec<_>=aliases.iter().map(|(name,source)|(name.clone(),source.as_str().unwrap().to_owned(),false)).collect();
        selected.extend(["latCell","lonCell"].map(|name|(name.to_owned(),name.to_owned(),true)));
        for (name, source, is_mesh) in selected {
            let input=if is_mesh {&mf}else{&df};
            let field=input.variable(&source).unwrap();assert_eq!(field.dtype(),&netcrust::DataType::F32);
            let dims:Vec<String>=field.dimensions().iter().map(|dim|dim.name().to_string()).collect();
            for dim in field.dimensions() {
                if let Some(existing)=hf.dimension(dim.name()) {
                    assert_eq!(existing.len(),dim.len());assert_eq!(existing.is_unlimited(),dim.is_unlimited());
                } else {dimensions.insert(dim.name().to_string(),if dim.is_unlimited(){None}else{Some(dim.len())});}
            }
            let values:Vec<f32>=input.read_array::<f32>(&source).unwrap().into_iter().collect();
            if let Some(existing)=hf.variable(&name) {
                assert_eq!(existing.dtype(),field.dtype());
                assert_eq!(existing.dimensions().iter().map(|dim|dim.name().to_string()).collect::<Vec<_>>(),dims);
                let original:Vec<f32>=hf.read_array::<f32>(&name).unwrap().into_iter().collect();
                assert_eq!(word_hash(&original),word_hash(&values),"existing history field {name} conflicts with diagnostics");
            }
            fields.push(json!({"view_field":name,"source_field":source,"source_path":if is_mesh {mesh}else{diagnostic},"stored_values":values.len(),
                "stored_word_sha256":word_hash(&values),"dimensions":dims}));
            variables.push(Variable{name:name.clone(),dimensions:dims,attributes:BTreeMap::new(),
                values:Some(values),text:None,only_if_zero:false,only_if_absent:true});
        }
        let receipt=materialize(Request{template:history.to_path_buf(),overlays:Vec::new(),output:output.to_path_buf(),
            dimensions,variables,attributes:BTreeMap::new()}).unwrap();
        let view=netcrust::File::open(output).unwrap();
        assert_eq!(view.read_strings("xtime").unwrap(),clocks);
        for field in &fields {
            let name=field["view_field"].as_str().unwrap();
            let copied:Vec<f32>=view.read_array::<f32>(name).unwrap().into_iter().collect();
            assert_eq!(word_hash(&copied),field["stored_word_sha256"].as_str().unwrap());
            assert_eq!(copied.len(),field["stored_values"].as_u64().unwrap() as usize);
        }
        // The ordinary complete Rust comparer authenticates all original
        // history fields, including binary64 mesh coordinates and characters.
        let comparison=rw_mpas::staticfile::compare::compare(history,output).unwrap();
        assert!(comparison.dimensions_agree && comparison.only_in_reference.is_empty());
        assert!(comparison.shape_mismatch.is_empty() && comparison.type_mismatch.is_empty());
        assert!(comparison.fields.iter().all(|field|field.n==field.exact));
        assert_eq!(sha(history),row["history_sha256"].as_str().unwrap());
        assert_eq!(sha(diagnostic),row["diagnostic_sha256"].as_str().unwrap());
        assert_eq!(sha(mesh),row["mesh_sha256"].as_str().unwrap());
        outputs.push(json!({"history":history,"history_sha256":sha(history),"diagnostic":diagnostic,
            "diagnostic_sha256":sha(diagnostic),"output":output,"output_sha256":sha(output),
            "mesh":mesh,"mesh_sha256":sha(mesh),
            "valid_time":expected,"fields":fields,"artifact_receipt":receipt,
            "original_history_fields":comparison.fields.len(),"all_original_and_added_words_preserved":true}));
    }
    fs::write(std::env::var("RRFS_RENDER_UNION_RECEIPT").unwrap(),serde_json::to_vec_pretty(&json!({
        "schema":"r10-native-init-boundary-render-view/v1","producer_sha256":sha(&std::env::current_exe().unwrap()),
        "plan_sha256":sha(Path::new(&plan_path)),"front_door_proof":false,
        "scope":"native atmosphere from two init producers; diagnostic display codec only",
        "source_fields_unchanged":true,"outputs":outputs})).unwrap()).unwrap();
}
