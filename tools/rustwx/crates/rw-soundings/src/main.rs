/// Release verification reads the source revision from the executable bytes.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

fn main() {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    if std::env::args().any(|a| a == "--abi") {
        println!("gpuwm-soundings.v1 plain-run png json html columns-v1 json-columns rw-store-v2 columns-surface-v1 series-v1");
        return;
    }
    // One line naming the device renders would use (a real device is
    // created, not just enumerated); exit 1 with the cause when none can be.
    if std::env::args().any(|a| a == "--probe-device") {
        match rw_soundings::render::probe_line() {
            Ok(line) => println!("render device: {line}"),
            Err(e) => {
                eprintln!("soundings: {e}");
                std::process::exit(1);
            }
        }
        return;
    }
    // `rw_soundings --columns REQUEST.json`: write the run's any-point column store (columns.rs) instead of pictures.
    if std::env::args().nth(1).as_deref() == Some("--columns") {
        let result = (|| {
            let path = std::env::args().nth(2).ok_or("usage: rw_soundings --columns REQUEST.json")?;
            let req = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            rw_soundings::columns::run(req).map(|_| ())
        })();
        if let Err(e) = result {
            eprintln!("soundings: {e}");
            std::process::exit(1);
        }
        return;
    }
    // `rw_soundings --series STORE_DIR [--threads N] [DOMAIN...]`: each fixed grid's point-forecast time series file.
    if std::env::args().nth(1).as_deref() == Some("--series") {
        let result = (|| {
            let args: Vec<String> = std::env::args().skip(2).collect();
            let store = args.first().ok_or("usage: rw_soundings --series STORE_DIR [--threads N] [DOMAIN...]")?;
            let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            let mut domains = vec![];
            let mut it = args[1..].iter();
            while let Some(a) = it.next() {
                if a == "--threads" {
                    threads = it.next().and_then(|v| v.parse().ok()).ok_or("--threads takes a number")?;
                } else {
                    domains.push(a.clone());
                }
            }
            rw_soundings::columns::series(std::path::Path::new(store), &domains, threads).map(|_| ())
        })();
        if let Err(e) = result {
            eprintln!("soundings: {e}");
            std::process::exit(1);
        }
        return;
    }
    let result = (|| {
        let path = std::env::args()
            .nth(1)
            .ok_or("usage: rw_soundings REQUEST.json")?;
        let req = serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        rw_soundings::batch::run(req)
    })();
    if let Err(e) = result {
        eprintln!("soundings: {e}");
        std::process::exit(1);
    }
}
