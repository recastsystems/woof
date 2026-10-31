//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:9670-9763
//! and mpas_init_atm_vinterp.F:25-109 at 1d6e370e.
//! Replay authentic whole-driver machine observations, not expected values
//! derived from the Rust implementation. Whole-file replay is separate.
use std::io::{BufRead, BufReader};
use rw_mpas::rrfs_init::vinterp::{AerosolWeights, TemperatureWeights, Extrap, vertical_interp};

#[test]
#[ignore = "requires the authenticated pristine whole-initializer register trace"]
fn replay_complete_pristine_register_trace() {
    let path=std::env::var("RRFS_REGISTER_TRACE").expect("RRFS_REGISTER_TRACE");
    let input=BufReader::new(std::fs::File::open(path).unwrap());
    let mut counts=std::collections::BTreeMap::new();
    for line in input.lines() {
        let row:serde_json::Value=serde_json::from_str(&line.unwrap()).unwrap();
        let word=|key:&str|f32::from_bits(u32::from_str_radix(
            row[key].as_str().unwrap().trim_start_matches("0x"),16).unwrap());
        let n=row["nz"].as_u64().unwrap() as usize;
        let mut z=vec![0.0;n];z[n-1]=1.0;
        let mut v=vec![0.0;n];v[n-1]=word("last");
        let offset=row["offset"].as_str().unwrap();
        let got=match offset {
            "0x5e5db"=>TemperatureWeights([word("xmm4"),word("xmm5")])
                .interpolate(word("target"),&z,&v,Extrap::Linear).unwrap(),
            "0x1107d3"=>AerosolWeights([word("xmm3"),word("xmm4")])
                .interpolate(word("target"),&z,&v),
            "0x550e3"|"0x552b3"=>vertical_interp(word("target"),&z,&v,
                if offset=="0x550e3" {Extrap::Linear}else{Extrap::Constant}).unwrap(),
            _=>panic!("unknown native variant {offset}"),
        };
        assert_eq!(got.to_bits(),word("result").to_bits(),"native sample {} at {offset}",row["sample"]);
        *counts.entry(offset.to_string()).or_insert(0usize)+=1;
    }
    assert_eq!(counts.len(),4,"whole trace must cover all four variants");
    assert_eq!(counts["0x5e5db"],140910);
    assert_eq!(counts["0x1107d3"],281820);
    assert_eq!(counts["0x550e3"],281820);
    assert_eq!(counts["0x552b3"],281820);
}
