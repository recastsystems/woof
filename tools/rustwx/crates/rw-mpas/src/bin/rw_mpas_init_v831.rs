//! Internal initialization worker selected by the resolver's source-pin row.
use std::process::ExitCode;
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP:&str=concat!("GPUWM_BRIDGE_SOURCE_REV=",env!("GPUWM_BRIDGE_SOURCE_REV"));
fn main()->ExitCode {
    let _=std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let args:Vec<String>=std::env::args().skip(1).collect();
    if args==["--abi"] {println!("{}",rw_mpas::rrfs_init::job::ABI_MARKER);return ExitCode::SUCCESS;}
    if !args.is_empty() {eprintln!("{}",rw_mpas::rrfs_init::job::ABI_MARKER);return ExitCode::FAILURE;}
    match rw_mpas::rrfs_init::job::run(std::io::stdin().lock(),std::io::BufWriter::new(std::io::stdout().lock())) {
        Ok(())=>ExitCode::SUCCESS,
        Err(e)=>{eprintln!("initialization worker: {e}");ExitCode::FAILURE}
    }
}
