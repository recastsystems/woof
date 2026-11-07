use gpuwm_tslist::{describe, parse};
use std::{
    env, fs,
    io::{self, Read},
    path::Path,
};

fn run() -> io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") => println!("rw_tslist {}", concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"))),
        Some("--abi") => println!("rw-tslist-v1 parse-fixed-columns"),
        Some("parse") if args.len() == 4 => {
            let max = args[2]
                .parse::<usize>()
                .map_err(|_| io::Error::other("invalid max_ts_locs"))?;
            let ideal = match args[3].as_str() {
                "0" => false,
                "1" => true,
                _ => return Err(io::Error::other("ideal must be 0/1")),
            };
            let text = if args[1] == "-" {
                let mut s = String::new();
                io::stdin().read_to_string(&mut s)?;
                s
            } else {
                fs::read_to_string(Path::new(&args[1]))?
            };
            print!("{}", describe(&parse(&text, max, ideal)?));
        }
        _ => {
            return Err(io::Error::other(
                "usage: rw_tslist --abi | parse FILE MAX_TS_LOCS IDEAL_0_OR_1",
            ));
        }
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("rw_tslist: {e}");
        std::process::exit(2);
    }
}
