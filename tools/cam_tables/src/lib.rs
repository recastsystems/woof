//! WRF 4.6.1 CAM tables. File records and Fortran element order are preserved.
//! Aerosol initialization follows module_ra_cam_support.F:3842-3893.
use rayon::prelude::*;
use std::{
    fs,
    io::{self, Write},
    path::Path,
};

#[used]
static SOURCE_REVISION: &str = concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

#[no_mangle]
pub extern "C" fn gpuwm_cam_tables_abi_version() -> u32 { 1 }

#[no_mangle]
pub extern "C" fn gpuwm_cam_tables_init_v1() -> u32 { 1 }

fn records(bytes: &[u8], lengths: &[usize]) -> Result<Vec<Vec<f64>>, String> {
    let mut pos = 0;
    let mut result = Vec::new();
    for &count in lengths {
        let n = count.checked_mul(8).ok_or("record length overflow")?;
        if pos + n + 8 > bytes.len() {
            return Err("truncated CAM record".into());
        }
        let head = u32::from_be_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let tail = u32::from_be_bytes(bytes[pos + 4 + n..pos + 8 + n].try_into().unwrap()) as usize;
        if head != n || tail != n {
            return Err(format!(
                "CAM record at {pos}: expected {n}, markers {head}/{tail}"
            ));
        }
        let values: Vec<_> = bytes[pos + 4..pos + 4 + n]
            .par_chunks_exact(8)
            .map(|b| f64::from_be_bytes(b.try_into().unwrap()))
            .collect();
        if values.par_iter().any(|v| !v.is_finite()) {
            return Err("nonfinite CAM table".into());
        }
        result.push(values);
        pos += n + 8;
    }
    if pos != bytes.len() {
        return Err("extra CAM records or trailing bytes".into());
    }
    Ok(result)
}

fn interpolate(x: &[f64], f: &[f64], y: f64, exponential: bool) -> f64 {
    let n = x.len();
    let mut k = 0;
    if y >= x[n - 1] {
        k = n - 2;
    } else if y > x[0] {
        while y > x[k + 1] && k < n - 1 {
            k += 1;
        }
    }
    if exponential {
        let a = (f[k + 1] / f[k]).ln() / (x[k + 1] - x[k]);
        f[k] * (a * (y - x[k])).exp()
    } else {
        let a = (f[k + 1] - f[k]) / (x[k + 1] - x[k]);
        f[k] + a * (y - x[k])
    }
}

fn initialized_aerosols(raw: &[Vec<f64>]) -> Vec<(&'static str, Vec<f64>)> {
    let rh = &raw[1];
    let mut tables: Vec<_> = (2..5).map(|i| raw[i].clone()).collect();
    // This PARAMETER's RHS is default REAL division, then conversion to R8.
    let wgt = (6.0_f32 / 7.0_f32) as f64;
    let mut salt = vec![vec![0.0; 8 * 19]; 3];
    for i in 0..8 * 19 {
        salt[0][i] = (1.0 - wgt) * raw[5][i] + wgt * raw[8][i];
        salt[1][i] =
            ((1.0 - wgt) * raw[5][i] * raw[6][i] + wgt * raw[8][i] * raw[9][i]) / salt[0][i];
        salt[2][i] = ((1.0 - wgt) * raw[5][i] * raw[6][i] * raw[7][i]
            + wgt * raw[8][i] * raw[9][i] * raw[10][i])
            / (salt[0][i] * salt[1][i]);
    }
    tables.extend(salt);
    tables.extend((11..14).map(|i| raw[i].clone()));
    let names = [
        "ksul", "wsul", "gsul", "ksslt", "wsslt", "gsslt", "kcphil", "wcphil", "gcphil",
    ];
    let mut result = Vec::new();
    for (i, name) in names.into_iter().enumerate() {
        let mut dst = vec![0.0; 1000 * 19];
        dst.par_chunks_mut(1000).enumerate().for_each(|(band, output)| {
            let src = &tables[i][band * 8..(band + 1) * 8];
            let normalized: Vec<_> = src.iter().map(|v| v / src[0]).collect();
            for (krh, value) in output.iter_mut().enumerate() {
                let y = (1.0_f64 / 1000.0) * (krh as f64);
                *value = interpolate(rh, &normalized, y, i % 3 == 0) * src[0];
            }
        });
        result.push((name, dst));
    }
    for (i, name) in ["kcphob", "wcphob", "gcphob"].into_iter().enumerate() {
        result.push((name, (0..19).map(|b| raw[11 + i][8 * b]).collect()));
    }
    for (i, name) in [
        "kcb", "wcb", "gcb", "kdst", "wdst", "gdst", "kbg", "wbg", "gbg", "kvolc", "wvolc", "gvolc",
    ]
    .into_iter()
    .enumerate()
    {
        result.push((name, raw[14 + i].clone()));
    }
    result
}

fn saturation_water(t: f64) -> f64 {
    // module_ra_cam_support.F:3158-3171, itype=0.
    let ps = 1013.246_f32 as f64;
    let ts = 373.16_f32 as f64;
    let e1 = (11.344_f32 as f64) * (1.0 - t / ts);
    let e2 = (-3.49149_f32 as f64) * (ts / t - 1.0);
    let f1 = (-7.90298_f32 as f64) * (ts / t - 1.0);
    let f2 = (5.02808_f32 as f64) * (ts / t).log10();
    let f3 = (-1.3816_f32 as f64) * (10.0_f64.powf(e1) - 1.0) / 10000000.0;
    let f4 = (8.1328_f32 as f64) * (10.0_f64.powf(e2) - 1.0) / 1000.0;
    let f5 = ps.log10();
    let f = f1 + f2 + f3 + f4 + f5;
    10.0_f64.powf(f) * 100.0
}

fn verified_bytes(root: &Path, name: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use sha2::{Digest, Sha256};
    let metadata: serde_json::Value =
        serde_json::from_str(include_str!("../../../woof/data/cam3/table-assets.json"))?;
    let pin = metadata["assets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["filename"] == name)
        .ok_or("unknown CAM asset")?;
    let bytes = fs::read(root.join(name))?;
    if bytes.len() as u64 != pin["size_bytes"].as_u64().unwrap() {
        return Err(format!("{name}: size differs from pinned WRF table").into());
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != pin["sha256"].as_str().unwrap() {
        return Err(format!("{name}: SHA-256 {digest} differs from pinned WRF table").into());
    }
    Ok(bytes)
}

fn table_fields(root: &Path) -> Result<Vec<(&'static str, Vec<f64>)>, Box<dyn std::error::Error>> {
    let abs = records(
        &verified_bytes(root, "CAM_ABS_DATA")?,
        &[10 * 10 * 25 * 21 * 7; 7],
    )?;
    let mut lengths = vec![19, 8];
    lengths.extend([8 * 19; 12]);
    lengths.extend([19; 3]);
    lengths.extend([4 * 19; 3]);
    lengths.extend([19; 6]);
    let aer = records(&verified_bytes(root, "CAM_AEROPT_DATA")?, &lengths)?;
    if aer[1].windows(2).any(|x| x[0] >= x[1]) {
        return Err("CAM humidity grid is not strictly increasing".into());
    }
    let names = [
        "ah2onw", "eh2onw", "ah2ow", "cn_ah2ow", "cn_eh2ow", "ln_ah2ow", "ln_eh2ow",
    ];
    let mut saturation: Vec<_> = (160..=351).map(|t| saturation_water(t as f64)).collect();
    saturation.push(0.0);
    Ok(names
        .into_iter()
        .zip(abs)
        .chain(initialized_aerosols(&aer))
        .chain([("estblh2o", saturation)])
        .collect())
}

pub fn run(root: &Path, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let fields = table_fields(root)?;
    fs::create_dir_all(out)?;
    let mut binary = io::BufWriter::new(fs::File::create(out.join("tables.bin"))?);
    let mut index = io::BufWriter::new(fs::File::create(out.join("tables.tsv"))?);
    writeln!(index, "field,count,offset_bytes")?;
    let mut offset = 0;
    for (name, values) in fields {
        writeln!(index, "{name},{},{offset}", values.len())?;
        for value in &values {
            binary.write_all(&value.to_le_bytes())?;
        }
        offset += values.len() * 8;
    }
    binary.flush()?;
    index.flush()?;
    Ok(())
}

pub fn main_cli() {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        eprintln!("usage: rw_cam_tables TABLE_ROOT OUTPUT_DIR");
        std::process::exit(2);
    }
    if let Err(err) = run(Path::new(&args[1]), Path::new(&args[2])) {
        eprintln!("CAM tables: {err}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn record_corruption_is_refused() {
        let mut good = vec![];
        good.extend(8u32.to_be_bytes());
        good.extend(1.0f64.to_be_bytes());
        good.extend(8u32.to_be_bytes());
        assert_eq!(
            records(&good, &[1]).unwrap()[0][0].to_bits(),
            1.0f64.to_bits()
        );
        good[15] ^= 1;
        assert!(records(&good, &[1]).is_err());
        assert!(records(&good[..11], &[1]).is_err());
    }
}

mod gases;
mod runtime;
pub use runtime::*;
