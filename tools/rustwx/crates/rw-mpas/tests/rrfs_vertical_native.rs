//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:4449-4897 at 1d6e370e.
use std::io;
#[test]
#[ignore="requires native vertical driver replay fixture"]
fn vertical_probe() {
 // libtest also writes stdout; use a dedicated file for the binary protocol.
 let bytes=std::fs::read(std::env::var("RRFS_VERTICAL_REQUEST").unwrap()).unwrap();
 let out=std::fs::File::create(std::env::var("RRFS_OUTPUT").unwrap()).unwrap();
 rw_mpas::rrfs_init::vertical::run(io::Cursor::new(bytes),out).unwrap();
}
