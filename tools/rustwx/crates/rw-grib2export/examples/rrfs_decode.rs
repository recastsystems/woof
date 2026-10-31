//! Decode GRIB2 files with the in-tree Rust reader (vendored wx-core) and
//! print one line per message: index, discipline.category.number, product
//! template, first level type and value, grid nx x ny and template, missing
//! count, min, max and mean.  The RRFS UPP-set check runs eccodes over the
//! same files and compares the two decoders message by message.
//!
//! usage: rrfs_decode FILE...

use rayon::prelude::*;

fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read");
        let file = wx_core::grib2::Grib2File::from_bytes(&data).unwrap_or_else(|e| panic!("{path}: {e}"));
        let rows: Vec<String> = file
            .messages
            .par_iter()
            .enumerate()
            .map(|(i, m)| {
                let v = wx_core::grib2::unpack_message_normalized(m).unwrap_or_else(|e| panic!("{path} message {i}: {e}"));
                let (mut miss, mut lo, mut hi, mut sum, mut cnt) = (0usize, f64::INFINITY, f64::NEG_INFINITY, 0.0f64, 0usize);
                for x in &v {
                    if x.is_finite() {
                        lo = lo.min(*x);
                        hi = hi.max(*x);
                        sum += x;
                        cnt += 1;
                    } else {
                        miss += 1;
                    }
                }
                let p = &m.product;
                format!(
                    "{}\t{}.{}.{}\t{}\t{}\t{}\t{}x{}\t{}\t{}\t{:.6e}\t{:.6e}\t{:.6e}",
                    i + 1,
                    m.discipline,
                    p.parameter_category,
                    p.parameter_number,
                    p.template,
                    p.level_type,
                    p.level_value,
                    m.grid.nx,
                    m.grid.ny,
                    m.grid.template,
                    miss,
                    lo,
                    hi,
                    if cnt > 0 { sum / cnt as f64 } else { f64::NAN }
                )
            })
            .collect();
        println!("# {path}: {} messages", rows.len());
        for r in rows {
            println!("{r}");
        }
    }
}
