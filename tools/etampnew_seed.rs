//! Seed the shared ETAMPNEW device table directly from WRF record words.
//! No floating point operation is used, so every binary32 word survives.
use std::{env, fs};
const RECORDS: [(&str, usize); 13] = [
    ("VENTR1",951),("VENTR2",951),("ACCRR",951),("MASSR",951),
    ("VRAIN",951),("RRATE",951),("VENTI1",951),("VENTI2",951),
    ("ACCRI",951),("MASSI",951),("VSNOWI",951),("VEL_RF",328),
    ("MY_GROWTH",35),
];
fn seed(raw: &[u8]) -> Result<Vec<u8>, String> {
    let head = raw.get(..4).ok_or("truncated record VENTR1")?;
    let rain_words = match u32::from_be_bytes(head.try_into().unwrap()) {
        3804 => 951, 1604 => 401,
        n => return Err(format!("record VENTR1 markers {n}, expected 3804 or 1604")),
    };
    let mut off = 0;
    let mut out = Vec::new();
    for (record, (name, words)) in RECORDS.iter().enumerate() {
        let n = if record < 6 { rain_words * 4 } else { words * 4 };
        let end = off + n + 8;
        if end > raw.len() { return Err(format!("truncated record {name}")); }
        let head = u32::from_be_bytes(raw[off..off+4].try_into().unwrap()) as usize;
        let tail = u32::from_be_bytes(raw[off+4+n..end].try_into().unwrap()) as usize;
        if head != n || tail != n { return Err(format!("record {name} markers {head}/{tail}, expected {n}")); }
        if record < 12 {
            for word in raw[off+4..off+4+n].chunks_exact(4) {
                out.extend_from_slice(&u32::from_be_bytes(word.try_into().unwrap()).to_le_bytes());
            }
        }
        off = end;
    }
    if off != raw.len() { return Err("trailing bytes after MY_GROWTH".into()); }
    Ok(out)
}
fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(args.len(), 3, "usage: etampnew_seed WRF_TABLE SEEDED_F32LE");
    let bytes = fs::read(&args[1]).expect("read WRF table");
    let out = seed(&bytes).expect("valid ETAMPNEW records");
    fs::write(&args[2], &out).expect("write seeded words");
    println!("seeded {} binary32 words",out.len()/4);
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn truncated_record_is_named() {
        assert_eq!(seed(&[]).unwrap_err(), "truncated record VENTR1");
    }
    #[test] fn preserves_word_bits_and_refuses_shifted_records() {
        let mut raw=Vec::new();
        for (_,n) in RECORDS { raw.extend_from_slice(&(n as u32*4).to_be_bytes());
            for _ in 0..n { raw.extend_from_slice(&0x3f123456u32.to_be_bytes()); }
            raw.extend_from_slice(&(n as u32*4).to_be_bytes()); }
        let got=seed(&raw).unwrap();
        assert_eq!(got.len(), (11*951+328)*4);
        assert!(got.chunks_exact(4).all(|w| w == 0x3f123456u32.to_le_bytes()));
        let mut original=Vec::new();
        for (i,(_,n)) in RECORDS.into_iter().enumerate() {
            let n=if i<6 {401} else {n};
            original.extend_from_slice(&(n as u32*4).to_be_bytes());
            for _ in 0..n { original.extend_from_slice(&0x3f123456u32.to_be_bytes()); }
            original.extend_from_slice(&(n as u32*4).to_be_bytes());
        }
        let short=seed(&original).unwrap();
        assert_eq!(short.len(), (6*401+5*951+328)*4);
        assert!(short.chunks_exact(4).all(|w| w == 0x3f123456u32.to_le_bytes()));
        raw[3] ^= 1;
        assert!(seed(&raw).unwrap_err().starts_with("record VENTR1 markers"));
    }
}
