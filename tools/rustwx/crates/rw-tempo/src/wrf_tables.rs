//! WRF public-domain binary table transport, with no physics arithmetic.
//! The caller pins the hash and layout. Source bytes are read once so hash
//! validation and decoding cannot observe different file generations.

use std::path::Path;

pub(crate) fn read_f32(
    path: &Path, sha256: &str, records: usize, values: usize, output: &mut [f32],
) -> Result<(), String> {
    let width = values.checked_mul(4).ok_or("record width overflow")?;
    let stride = width.checked_add(8).ok_or("record stride overflow")?;
    let bytes = records.checked_mul(stride).ok_or("table length overflow")?;
    if records == 0 || values == 0 || records.checked_mul(values) != Some(output.len()) {
        return Err("invalid table output shape".into());
    }
    let data = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    decode_f32(&data, sha256, records, values, bytes, output)
}

fn decode_f32(data: &[u8], sha256: &str, records: usize, values: usize,
              bytes: usize, output: &mut [f32]) -> Result<(), String> {
    if data.len() != bytes {
        return Err(format!("table has {} bytes, expected {bytes}", data.len()));
    }
    let actual = sha256_hex(data);
    if actual != sha256 {
        return Err(format!("SHA-256 {actual}, expected {sha256}"));
    }
    let width = values * 4;
    let stride = width + 8;
    for (i, (record, target)) in data.chunks_exact(stride)
        .zip(output.chunks_exact_mut(values)).enumerate() {
        let lead = u32::from_be_bytes(record[..4].try_into().unwrap()) as usize;
        let tail = u32::from_be_bytes(record[stride-4..].try_into().unwrap()) as usize;
        if lead != width || tail != width {
            return Err(format!("record {i}: markers {lead}/{tail}, expected {width}"));
        }
        for (word, dest) in record[4..stride-4].chunks_exact(4).zip(target) {
            *dest = f32::from_bits(u32::from_be_bytes(word.try_into().unwrap()));
        }
    }
    debug_assert_eq!(data.len() / stride, records);
    Ok(())
}

// SHA-256 is shared in source with rustwx-products/src/publication.rs.
// Kept dependency-free like the existing rw-tempo initialization bridge.
fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut message = bytes.to_vec();
    let bit_len = (message.len() as u64) * 8;
    message.push(0x80);
    while (message.len() % 64) != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut h0 = 0x6a09e667u32;
    let mut h1 = 0xbb67ae85u32;
    let mut h2 = 0x3c6ef372u32;
    let mut h3 = 0xa54ff53au32;
    let mut h4 = 0x510e527fu32;
    let mut h5 = 0x9b05688cu32;
    let mut h6 = 0x1f83d9abu32;
    let mut h7 = 0x5be0cd19u32;

    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;
        let mut f = h5;
        let mut g = h6;
        let mut h = h7;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
        h5 = h5.wrapping_add(f);
        h6 = h6.wrapping_add(g);
        h7 = h7.wrapping_add(h);
    }

    format!("{h0:08x}{h1:08x}{h2:08x}{h3:08x}{h4:08x}{h5:08x}{h6:08x}{h7:08x}")
}
