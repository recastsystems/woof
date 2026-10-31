//! Extract WRF's packaged P3 PSD table with native float32 parsing.
//! The resulting 14,000 words must hash to
//! 45390820659a05f80c50482dacb8c886c93c379fe7c10f468fc1d9216ababaa5.
use std::{env,fs};
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=env::args().collect();
    if args.len()!=3 {return Err("usage: export_simradar_p3 INPUT OUTPUT".into());}
    let source=fs::read_to_string(&args[1])?;
    let mut lines=source.lines();
    if lines.next().map(str::trim)!=Some("LOOKUP_TABLE_1-version:  5.4_2momI") {
        return Err("P3 table version mismatch".into());
    }
    lines.next().ok_or("missing P3 separator")?;
    let mut bytes=Vec::with_capacity(56_000);
    for _ in 0..20 {
        for _ in 0..50 {
            let words:Vec<_>=lines.next().ok_or("truncated P3 table")?.split_whitespace().collect();
            if words.len()!=17 {return Err("P3 main record width mismatch".into());}
            for word in &words[3..] {bytes.extend_from_slice(&word.parse::<f32>()?.to_le_bytes());}
        }
        for _ in 0..1500 {lines.next().ok_or("truncated P3 collision table")?;}
    }
    if lines.any(|s| !s.trim().is_empty()) {return Err("extra P3 records".into());}
    fs::write(&args[2],bytes)?;
    Ok(())
}
