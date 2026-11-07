//! WRF v4.7.1 phys/module_mp_p3.F:337-398,599-670, public domain.
//! The required WRF public-domain notice is in the repository NOTICE.
//! Source sha256: 716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6.
//! Only startup runs here. Column physics and all cross-step state stay on CUDA.

use rayon::prelude::*;
use lzma_rust2::XzReader;
use rw_libm::{expf, log10f, powf};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, io::Read, path::Path};

const FILENAME: &str = "p3_lookupTable_1.dat-v5.4_2momI";
const VERSION: &str = "5.4_2momI";
const TEXT_BYTES: usize = 1_637_040;
const DIGEST: &str = "5cb47f2ad15b726abeacac21b5877c737178526051661473a465315348867c2f";
const ITAB: usize = 14_000;
const COLL: usize = 60_000;
const RAIN: usize = 3_000;
const WORDS: usize = ITAB + COLL + 3 * RAIN;
thread_local! { static ERROR: RefCell<String> = const { RefCell::new(String::new()) }; }

fn table_text(root: &Path) -> Result<String, String> {
    let raw = root.join(FILENAME);
    let mut bytes = Vec::with_capacity(TEXT_BYTES);
    std::fs::File::open(&raw)
        .map_err(|e| {
            format!(
                "P3 lookup table missing or unreadable: {}: {e}",
                raw.display()
            )
        })?
        .take((TEXT_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let text = String::from_utf8(bytes).map_err(|e| format!("P3 table is not ASCII text: {e}"))?;
    let header = text.lines().next().unwrap_or("");
    let version = header.split_ascii_whitespace().nth(1).unwrap_or("");
    if version != VERSION {
        return Err(format!(
            "P3 lookup table names version {version:?}; requires {VERSION:?}"
        ));
    }
    if text.len() != TEXT_BYTES {
        return Err(format!(
            "P3 lookup table has size {}, expected {TEXT_BYTES}",
            text.len()
        ));
    }
    let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
    if digest != DIGEST {
        return Err(format!(
            "P3 lookup table SHA-256 {digest} does not match pinned {DIGEST}"
        ));
    }
    Ok(text)
}

fn parse_block(lines: &[&str]) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut ice = Vec::with_capacity(700);
    let mut coll = Vec::with_capacity(3000);
    for (row, line) in lines.iter().enumerate() {
        let (skip, n, out) = if row < 50 {
            (3, 14, &mut ice)
        } else {
            (3, 2, &mut coll)
        };
        let tokens: Vec<_> = line.split_ascii_whitespace().collect();
        if tokens.len() != skip + n {
            return Err(format!(
                "P3 table record {row} has {} tokens, expected {}",
                tokens.len(),
                skip + n
            ));
        }
        for value in &tokens[skip..] {
            // WRF reads collection values into REAL(8), then assigns REAL(4).
            // Both arrays' measured Fortran hashes pin this conversion.
            out.push(value.parse::<f64>().map_err(|e| e.to_string())? as f32);
        }
    }
    Ok((ice, coll))
}

fn lookup(root: &Path) -> Result<(Vec<f32>, Vec<f32>), String> {
    let text = table_text(root)?;
    let records: Vec<_> = text
        .lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .collect();
    if records.len() != 5 * 4 * 1550 {
        return Err(format!(
            "P3 table has {} records, expected 31000",
            records.len()
        ));
    }
    let blocks: Result<Vec<_>, _> = records.par_chunks_exact(1550).map(parse_block).collect();
    let mut ice = Vec::with_capacity(ITAB);
    let mut coll = Vec::with_capacity(COLL);
    for (a, b) in blocks? {
        ice.extend(a);
        coll.extend(b);
    }
    Ok((ice, coll))
}

/// p3_init's 300 independent rain-size rows. Each row retains WRF's
/// sequential 10000-bin addition order; parallelism is between rows.
fn rain_tables() -> Vec<[f32; 3]> {
    let third = 1.0_f32 / 3.0;
    let sixth = 1.0_f32 / 6.0;
    let pi_over_six = 3.14159265_f32 * sixth;
    let bins: Vec<_> = (1..=10_000)
        .into_par_iter()
        .map(|k| {
            let dia = ((k as f32 * 2.0) - 1.0) * 1.0e-6;
            let mass = ((pi_over_six * 997.0) * ((dia * dia) * dia)) * 1000.0;
            let um = dia * 1.0e6;
            let vt = if um <= 134.43 {
                4.5795e3 * powf(mass, 2.0 * third)
            } else if um < 1511.64 {
                4.962e1 * powf(mass, third)
            } else if um < 3477.84 {
                1.732e1 * powf(mass, sixth)
            } else {
                9.17
            };
            let log = log10f(dia);
            let p0 = powf(10.0, 0.0 * log + 4.0 * 0.0);
            let p3 = powf(10.0, (0.0 + 3.0) * log + 4.0 * 0.0);
            let p1 = powf(10.0, (0.0 + 1.0) * log + 3.0 * 0.0);
            (dia, vt * p0, p0, vt * p3, p3, powf(vt * dia, 0.5) * p1)
        })
        .collect();
    (1..=300)
        .into_par_iter()
        .map(|j| {
            let j = j as f32;
            let dm = if j <= 20.0 {
                (j * 10.0 - 5.0) * 1.0e-6
            } else {
                ((j - 20.0) * 30.0 + 195.0) * 1.0e-6
            };
            let lambda = 1.0 / dm;
            let mut sum = [0.0_f32; 5];
            for &(dia, a, b, c, d, e) in &bins {
                let decay = expf((-lambda) * dia);
                for (s, term) in sum.iter_mut().zip([a, b, c, d, e]) {
                    *s += ((term * decay) * 2.0) * 1.0e-6;
                }
            }
            let vn = sum[0] / sum[1].max(1.0e-30);
            let vm = sum[2] / sum[3].max(1.0e-30);
            let revap = powf(
                10.0,
                (log10f(sum[4].max(1.0e-30)) + (0.0 + 1.0) * log10f(lambda)) - 3.0 * 0.0,
            );
            [vn, vm, revap]
        })
        .collect()
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_p3_abi_version() -> u32 {
    1
}

/// Source-bound p3_init cache for the pinned GNU13/glibc2.39 REAL32 oracle.
/// The ten mu_r columns are identical (native mu_r is zero); retain the ABI layout.
/// # Safety
/// `dst` must refer to `cap` writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_p3_native_gnu239_rain(dst: *mut f32, cap: usize) -> i32 {
    if dst.is_null() || cap != 3 * RAIN {
        return 1;
    }
    let data = include_bytes!("p3_gnu239_rain.bin");
    let out = unsafe { std::slice::from_raw_parts_mut(dst, cap) };
    for table in 0..3 {
        for row in 0..300 {
            let offset = (table * 300 + row) * 4;
            let value = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            out[table * RAIN + row * 10..table * RAIN + (row + 1) * 10].fill(value);
        }
    }
    0
}

/// Copy a thread-local error without a terminating NUL. Returns its length.
/// # Safety
/// A non-null dst must refer to cap writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_p3_last_error(dst: *mut u8, cap: usize) -> usize {
    ERROR.with(|e| {
        let e = e.borrow();
        if !dst.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(e.as_ptr(), dst, cap.min(e.len()));
            }
        }
        e.len()
    })
}

/// Write C-order itab, itabcoll, vn, vm and revap to caller-owned storage.
/// # Safety
/// root refers to len UTF-8 bytes. out refers to words writable f32s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_p3_init_2mom(
    root: *const u8,
    len: usize,
    out: *mut f32,
    words: usize,
) -> i32 {
    let result = std::panic::catch_unwind(|| -> Result<(), String> {
        if root.is_null() || out.is_null() || words != WORDS {
            return Err("invalid P3 bridge buffer layout".into());
        }
        let path = std::str::from_utf8(unsafe { std::slice::from_raw_parts(root, len) })
            .map_err(|e| e.to_string())?;
        let (ice, coll) = lookup(Path::new(path))?;
        let rain = rain_tables();
        let output = unsafe { std::slice::from_raw_parts_mut(out, words) };
        output[..ITAB].copy_from_slice(&ice);
        output[ITAB..ITAB + COLL].copy_from_slice(&coll);
        for field in 0..3 {
            for (j, row) in rain.iter().enumerate() {
                let start = ITAB + COLL + field * RAIN + j * 10;
                output[start..start + 10].fill(row[field]);
            }
        }
        Ok(())
    });
    match result {
        Ok(Ok(())) => {
            ERROR.with(|e| e.borrow_mut().clear());
            0
        }
        Ok(Err(message)) => {
            ERROR.with(|e| *e.borrow_mut() = message);
            1
        }
        Err(_) => {
            ERROR.with(|e| *e.borrow_mut() = "P3 initialization panicked".into());
            2
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_p3_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    SOURCE_REV_STAMP.as_ptr().cast()
}

// Three-moment table-1 shape, with the same rain generator and ABI.
const FILENAME_3: &str = "p3_lookupTable_1.dat-v5.4_3momI";
const VERSION_3: &str = "5.4_3momI";
const TEXT_BYTES_3: usize = 17_886_038;
const DIGEST_3: &str = "9a3c57ecc09498802c8d7cb3931dbb0200dcf0f51466b3e288120c080271e6dc";
const ITAB_3: usize = 165_000;
const COLL_3: usize = 660_000;
const WORDS_3: usize = ITAB_3 + COLL_3 + 3 * RAIN;

fn table_text_3(root: &Path) -> Result<String, String> {
    let raw = root.join(FILENAME_3);
    let packed = root.join(format!("{FILENAME_3}.xz"));
    let mut bytes = Vec::with_capacity(TEXT_BYTES_3);
    if packed.is_file() {
        let file = std::fs::File::open(&packed).map_err(|e| format!("{}: {e}", packed.display()))?;
        // Bound allocation even when an operator supplies a damaged stream.
        XzReader::new(file, false).take((TEXT_BYTES_3 + 1) as u64)
            .read_to_end(&mut bytes).map_err(|e| format!("{} is not a readable xz stream: {e}", packed.display()))?;
    } else {
        std::fs::File::open(&raw).map_err(|e| format!("P3 lookup_3 table missing or unreadable: {}: {e}", raw.display()))?
            .take((TEXT_BYTES_3 + 1) as u64).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    }
    let text = String::from_utf8(bytes).map_err(|e| format!("P3 table is not ASCII text: {e}"))?;
    let header = text.lines().next().unwrap_or("");
    let version = header.split_ascii_whitespace().nth(1).unwrap_or("");
    if version != VERSION_3 {
        return Err(format!("P3 lookup_3 table names version {version:?}; requires {VERSION_3:?}"));
    }
    if text.len() != TEXT_BYTES_3 {
        return Err(format!("P3 lookup_3 table has size {}, expected {TEXT_BYTES_3}", text.len()));
    }
    let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
    if digest != DIGEST_3 {
        return Err(format!("P3 lookup_3 table SHA-256 {digest} does not match pinned {DIGEST_3}"));
    }
    Ok(text)
}

fn parse_block_3(lines: &[&str]) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut ice = Vec::with_capacity(750);
    let mut coll = Vec::with_capacity(3000);
    for (row, line) in lines.iter().enumerate() {
        let (skip, n, out) = if row < 50 { (4, 15, &mut ice) } else { (3, 2, &mut coll) };
        let tokens: Vec<_> = line.split_ascii_whitespace().collect();
        if tokens.len() != skip + n {
            return Err(format!("P3 table record {row} has {} tokens, expected {}", tokens.len(), skip + n));
        }
        for value in &tokens[skip..] {
            // WRF reads collection values into REAL(8), then assigns REAL(4).
            // Both arrays' measured Fortran hashes pin this conversion.
            out.push(value.parse::<f64>().map_err(|e| e.to_string())? as f32);
        }
    }
    Ok((ice, coll))
}

fn lookup_3(root: &Path) -> Result<(Vec<f32>, Vec<f32>), String> {
    let text = table_text_3(root)?;
    let records: Vec<_> = text.lines().skip(1).filter(|line| !line.trim().is_empty()).collect();
    if records.len() != 11 * 5 * 4 * 1550 {
        return Err(format!("P3 table has {} records, expected 341000", records.len()));
    }
    let blocks: Result<Vec<_>, _> = records.par_chunks_exact(1550).map(parse_block_3).collect();
    let mut ice = Vec::with_capacity(ITAB_3);
    let mut coll = Vec::with_capacity(COLL_3);
    for (a, b) in blocks? { ice.extend(a); coll.extend(b); }
    Ok((ice, coll))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_p3_init_3mom(root: *const u8, len: usize, out: *mut f32, words: usize) -> i32 {
    let result = std::panic::catch_unwind(|| -> Result<(), String> {
        if root.is_null() || out.is_null() || words != WORDS_3 { return Err("invalid P3 bridge buffer layout".into()); }
        let path = std::str::from_utf8(unsafe { std::slice::from_raw_parts(root, len) }).map_err(|e| e.to_string())?;
        let (ice, coll) = lookup_3(Path::new(path))?;
        let rain = rain_tables();
        let output = unsafe { std::slice::from_raw_parts_mut(out, words) };
        output[..ITAB_3].copy_from_slice(&ice);
        output[ITAB_3..ITAB_3 + COLL_3].copy_from_slice(&coll);
        for field in 0..3 {
            for (j, row) in rain.iter().enumerate() {
                let start = ITAB_3 + COLL_3 + field * RAIN + j * 10;
                output[start..start + 10].fill(row[field]);
            }
        }
        Ok(())
    });
    match result {
        Ok(Ok(())) => { ERROR.with(|e| e.borrow_mut().clear()); 0 }
        Ok(Err(message)) => { ERROR.with(|e| *e.borrow_mut() = message); 1 }
        Err(_) => { ERROR.with(|e| *e.borrow_mut() = "P3 initialization panicked".into()); 2 }
    }
}

// WRF module_mp_p3.F:474-519, native table-2 ordering. Independent output
// elements are parsed in parallel, directly into Fortran element order.
const TABLE2_BYTES: usize = 15_250_032;
const TABLE2_WORDS: usize = 250_000;
const TABLE2_DIGEST: &str = "c9103041ead719cf9e2e215cf846e0ef81b2004f29386156444a079af6b24d97";

fn collection2(root: &Path) -> Result<(Vec<f32>, Vec<f32>), String> {
    let path = root.join("p3_lookupTable_2.dat-v5.3");
    let mut bytes = Vec::with_capacity(TABLE2_BYTES);
    std::fs::File::open(&path)
        .map_err(|e| format!("P3 lookup table missing or unreadable: {}: {e}", path.display()))?
        .take((TABLE2_BYTES + 1) as u64).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let version = text.lines().next().unwrap_or("").split_ascii_whitespace().nth(1).unwrap_or("");
    if version != "5.3" { return Err(format!("P3 lookup table names version {version:?}; requires 5.3")); }
    if bytes.len() != TABLE2_BYTES { return Err(format!("P3 table 2 size {} != {TABLE2_BYTES}", bytes.len())); }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    if digest != TABLE2_DIGEST { return Err(format!("P3 table 2 SHA-256 {digest} != {TABLE2_DIGEST}")); }
    let records: Vec<_> = text.lines().skip(1).filter(|l| !l.trim().is_empty()).collect();
    if records.len() != TABLE2_WORDS { return Err(format!("P3 table 2 records {} != {TABLE2_WORDS}", records.len())); }
    let pairs: Result<Vec<_>, String> = (0..TABLE2_WORDS).into_par_iter().map(|index| {
        // File loops i,jjj,jjjj,ii,jjj2,jjjj2, but Fortran i varies fastest.
        let mut rem = index;
        let i = rem % 25; rem /= 25;
        let j = rem % 4; rem /= 4;
        let d = rem % 5; rem /= 5;
        let ii = rem % 25; rem /= 25;
        let jj = rem % 4; let dd = rem / 4;
        let row = (((((i * 4 + j) * 5 + d) * 25 + ii) * 4 + jj) * 5) + dd;
        let tokens: Vec<_> = records[row].split_ascii_whitespace().collect();
        if tokens.len() != 8 { return Err(format!("P3 table 2 row {row} has {} tokens", tokens.len())); }
        let a = tokens[6].parse::<f32>().map_err(|e| e.to_string())?;
        let b = tokens[7].parse::<f32>().map_err(|e| e.to_string())?;
        Ok((a,b))
    }).collect();
    let (a,b): (Vec<f32>, Vec<f32>) = pairs?.into_iter().unzip();
    Ok((a,b))
}

/// # Safety
/// root points to len UTF-8 bytes and out to words writable float32 values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_p3_init_table2(root: *const u8, len: usize, out: *mut f32, words: usize) -> i32 {
    let result = std::panic::catch_unwind(|| -> Result<(), String> {
        if root.is_null() || out.is_null() || words != 2 * TABLE2_WORDS { return Err("invalid P3 table-2 bridge buffer layout".into()); }
        let path = std::str::from_utf8(unsafe { std::slice::from_raw_parts(root,len) }).map_err(|e| e.to_string())?;
        let (a,b) = collection2(Path::new(path))?;
        let target = unsafe { std::slice::from_raw_parts_mut(out,words) };
        target[..TABLE2_WORDS].copy_from_slice(&a);
        target[TABLE2_WORDS..].copy_from_slice(&b);
        Ok(())
    });
    match result {
        Ok(Ok(())) => { ERROR.with(|e| e.borrow_mut().clear()); 0 }
        Ok(Err(message)) => { ERROR.with(|e| *e.borrow_mut() = message); 1 }
        Err(_) => { ERROR.with(|e| *e.borrow_mut() = "P3 table-2 initialization panicked".into()); 2 }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_preserve_fortran_column_layout_and_float_words() {
        let mut values = vec!["1.00000011920928955078125", "-0.0"];
        values.extend(std::iter::repeat_n("-1.25", 12));
        let ice = format!("1 2 3 {}", values.join(" "));
        let mut lines = vec![ice.as_str(); 50];
        lines.push("1 2 3 -2.5 0.0");
        let (ice, collection) = parse_block(&lines).unwrap();
        assert_eq!(ice.len(), 700);
        assert_eq!(collection.len(), 2);
        assert_eq!(ice[0].to_bits(), 0x3f80_0001);
        assert_eq!(ice[1].to_bits(), 0x8000_0000);
        assert_eq!(ice[14].to_bits(), ice[0].to_bits());
        assert_eq!(collection[0].to_bits(), (-2.5_f32).to_bits());

        values.push("3.5");
        let ice3 = format!("1 2 3 4 {}", values.join(" "));
        let mut lines = vec![ice3.as_str(); 50];
        lines.push("1 2 3 -2.5 0.0");
        let (ice, collection) = parse_block_3(&lines).unwrap();
        assert_eq!(ice.len(), 750);
        assert_eq!(collection.len(), 2);
        assert_eq!(ice[0].to_bits(), 0x3f80_0001);
        assert_eq!(ice[1].to_bits(), 0x8000_0000);
        assert_eq!(ice[14].to_bits(), 3.5_f32.to_bits());
        assert_eq!(ice[15].to_bits(), ice[0].to_bits());
    }

    #[test]
    fn malformed_records_are_refused_before_initialization() {
        assert!(parse_block(&["1 2 3 4"]).unwrap_err().contains("tokens"));
        assert!(parse_block_3(&["1 2 3 4"]).unwrap_err().contains("tokens"));
        let bad = format!("1 2 3 invalid {}", vec!["1.0"; 13].join(" "));
        assert!(parse_block(&[bad.as_str()]).is_err());
        let bad = format!("1 2 3 4 invalid {}", vec!["1.0"; 14].join(" "));
        assert!(parse_block_3(&[bad.as_str()]).is_err());
    }

    #[test]
    fn ffi_layout_errors_leave_the_caller_buffer_untouched() {
        let mut sentinel = 17.0_f32;
        for function in [gpuwm_p3_init_2mom, gpuwm_p3_init_3mom, gpuwm_p3_init_table2] {
            let status = unsafe { function(std::ptr::null(), 0, &mut sentinel, 1) };
            assert_eq!(status, 1);
            assert_eq!(sentinel.to_bits(), 17.0_f32.to_bits());
            let needed = unsafe { gpuwm_p3_last_error(std::ptr::null_mut(), 0) };
            assert!(needed > 0);
            let mut bytes = vec![0; needed];
            let copied = unsafe { gpuwm_p3_last_error(bytes.as_mut_ptr(), bytes.len()) };
            assert_eq!(copied, needed);
            assert!(String::from_utf8(bytes).unwrap().contains("layout"));
        }
    }
}
