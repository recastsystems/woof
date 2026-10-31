//! Complete dormant source replay, alongside whole active GWD file replays.
//! RRFSx/MPAS-Model mpas_init_atm_gwd.F:1268-1449 at 1d6e370e.
#[test]
#[ignore = "requires MPAS_GWD_ORACLE (build pristine source with the strict GNU flags)"]
fn complete_native_module_diagnostics_match_fortran_words() {
    use rw_mpas::rrfs_static_gwd::{SG_DELTA, dormant_diagnostics, pixel_dx};
    let path = std::env::var("MPAS_GWD_ORACLE").unwrap() + ".diagnostics";
    let bytes = std::fs::read(path).unwrap();
    let mut words = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()));
    assert_eq!(words.next(), Some(64));
    assert_eq!(words.next(), Some(SG_DELTA.to_bits()));
    let mut metrics = 0;
    for k in 0..64 {
        let nx = 4 + (k * 13) % 31;
        let ny = 4 + (k * 17) % 27;
        let jc = 1 + (k * 347) % 21600;
        assert_eq!(words.next(), Some(nx as u32));
        assert_eq!(words.next(), Some(ny as u32));
        let mut topo = Vec::new();
        let mut dxm = Vec::new();
        for j in 1..=ny {
            let mut jj=j as i64-(ny/2) as i64+jc as i64;
            if jj<=0 {jj=-jj+1;} if jj>21600 {jj=21600-(jj-21600-1);}
            let dx = pixel_dx(jj);
            assert_eq!(words.next(), Some(dx.to_bits()), "metric case {k} row {j}");
            metrics += 1;
            for i in 1..=nx {
                topo.push(match k % 8 {
                    0 => 200.0,
                    1 => (i * 10) as f32,
                    2 => (j * 10) as f32,
                    3 => ((i + j) * 10) as f32,
                    4 => ((i as i32 - j as i32) * 10) as f32,
                    _ => ((i * 37 + j * 101 + k * 53) % 7000) as f32 - 1000.0,
                });
                dxm.push(dx);
            }
        }
        let d = dormant_diagnostics(&topo, &dxm, nx, ny).unwrap();
        for (field, v) in [d.elvmax, d.htheta, d.hgamma, d.hsigma].iter().enumerate() {
            assert_eq!(
                words.next(),
                Some(v.to_bits()),
                "diagnostic case {k} field {field} Rust={v}"
            );
        }
    }
    assert!(words.next().is_none());
    println!(
        "NATIVE_GWD_DIAGNOSTICS fields=256 metric_words={metrics} mismatches=0 disabled_in_native_output=true"
    );
}
