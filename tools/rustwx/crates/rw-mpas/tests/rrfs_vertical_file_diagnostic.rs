//! Stored IEEE-word diagnostic for complete init-stage file replays.
//! RRFSx/MPAS-Model mpas_init_atm_cases.F:4043-7742 at 1d6e370e.
#[test]
#[ignore = "requires RRFS_NATIVE_FILE and RRFS_RUST_FILE from a complete file replay"]
fn stored_vertical_words() {
    let native=netcrust::File::open(std::env::var("RRFS_NATIVE_FILE").unwrap()).unwrap();
    let rust=netcrust::File::open(std::env::var("RRFS_RUST_FILE").unwrap()).unwrap();
    let mut differences=0usize;
    let mut rows=Vec::new();
    let fields=std::env::var("RRFS_FIELDS").unwrap_or_else(|_| "ter,zgrid,zb,zb3,zz,zxu,dss".into());
    for name in fields.split(',') {
        let a=native.read_array::<f32>(name).unwrap();
        let b=rust.read_array::<f32>(name).unwrap();
        assert_eq!(a.len(),b.len());
        let mut pairs=std::collections::BTreeMap::new();
        for (i,(a,b)) in a.into_iter().zip(b.into_iter()).enumerate() {
            if a.to_bits()!=b.to_bits() {
                differences+=1;
                let entry=pairs.entry((a.to_bits(),b.to_bits())).or_insert((0usize,i));
                entry.0+=1;
            }
        }
        for (&(a,b),&(n,i)) in &pairs {println!("FIELD {name} native={a:08x} rust={b:08x} count={n} first_index={i}");}
        rows.push(serde_json::json!({"field":name,"pairs":pairs.iter().take(64).map(|(&(a,b),&(n,i))|
            serde_json::json!({"native_bits":a,"rust_bits":b,"count":n,"first_index":i})).collect::<Vec<_>>(),
            "distinct_pairs":pairs.len(),"differing_words":pairs.values().map(|&(n,_)|n).sum::<usize>()}));
    }
    if let Ok(path)=std::env::var("RRFS_WORD_DIAGNOSTIC_RECEIPT") {
        std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"schema":"r10-stored-word-pair-diagnostic/v1",
            "native":std::env::var("RRFS_NATIVE_FILE").unwrap(),"rust":std::env::var("RRFS_RUST_FILE").unwrap(),
            "differing_words":differences,"fields":rows})).unwrap()).unwrap();
    }
    assert_eq!(differences,0);
}
