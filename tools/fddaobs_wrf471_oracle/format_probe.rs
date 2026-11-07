//! Developer format comparison; compiles the actual production decoder.
#[allow(dead_code)]
#[path = "../rustwx/crates/rw-obs/src/fddaobs.rs"]
mod fddaobs;

use fddaobs::Measurements;
use std::{
    error::Error,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: format_probe INPUT OUTPUT WORKERS".into());
    }
    let reports = fddaobs::read(Path::new(&args[1]), 150000, args[3].parse()?)?;
    let mut out = BufWriter::new(File::create(&args[2])?);
    for r in reports {
        out.write_all(r.date.as_bytes())?;
        for v in [r.latitude, r.longitude] {
            out.write_all(&v.to_le_bytes())?;
        }
        for text in [&r.id, &r.name, &r.platform, &r.source] {
            out.write_all(text.as_bytes())?;
        }
        out.write_all(&r.elevation.to_le_bytes())?;
        let sounding = matches!(&r.measurements, Measurements::Sounding(_));
        for v in [i32::from(sounding), i32::from(r.bogus), r.declared_count] {
            out.write_all(&v.to_le_bytes())?;
        }
        let rows: Vec<&[fddaobs::Datum]> = match &r.measurements {
            Measurements::Sounding(rows) => rows.iter().map(|row| row.as_slice()).collect(),
            Measurements::Surface(row) => vec![row.as_slice()],
        };
        for row in rows {
            for d in row {
                out.write_all(&d.value.to_le_bytes())?;
                out.write_all(&d.qc.to_le_bytes())?;
            }
        }
    }
    out.flush()?;
    Ok(())
}
