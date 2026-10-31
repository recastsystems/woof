//! Animated radar loops encoded from the production renderer's PNG frames.
//!
//! A live forecast calls this after every history, so a loop is extended,
//! not re-encoded: each GIF frame carries its own palette and disposal and
//! depends on no other frame, so the frames already encoded are copied from
//! the previous loop and only the new ones are encoded (the result is
//! byte-identical to encoding every frame again). Re-encoding every frame on
//! every history made loop work grow with the square of the run length. The
//! loop a new one replaces is retired and deleted once the manifest no longer
//! names it.
use crate::manifest::{self, Manifest, RadarLoop};
use image::{
    Delay, Frame,
    codecs::gif::{GifEncoder, Repeat},
};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

type Frames<'a> = [(String, &'a manifest::Image)];

fn loop_key(frames: &Frames<'_>) -> Result<String, String> {
    Ok(manifest::digest(
        &serde_json::to_vec(
            &frames
                .iter()
                .map(|(time, image)| (time, &image.artifact.sha256))
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?,
    ))
}

fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Encode `frames` as one complete GIF. Every frame is decoded and encoded
/// on its own core as a one-frame GIF and the frames are joined behind the
/// first header, which is byte-identical to one encoder writing them in
/// turn (each frame carries its own palette; see the module note). A frame
/// whose header differs from the first (another size) is encoded serially.
pub(crate) fn encode(root: &Path, frames: &Frames<'_>) -> Result<Vec<u8>, String> {
    if frames.len() < 2 {
        return encode_serial(root, frames);
    }
    let pieces = frames
        .par_iter()
        .map(|frame| encode_serial(root, std::slice::from_ref(frame)))
        .collect::<Result<Vec<_>, String>>()?;
    let layouts: Option<Vec<(usize, usize)>> = pieces.iter().map(|p| layout(p)).collect();
    if let Some(layouts) = layouts {
        let header = &pieces[0][..layouts[0].0];
        if pieces
            .iter()
            .zip(&layouts)
            .all(|(piece, (first, _))| &piece[..*first] == header)
        {
            let mut bytes = Vec::with_capacity(pieces.iter().map(Vec::len).sum());
            bytes.extend_from_slice(header);
            for (piece, (first, trailer)) in pieces.iter().zip(&layouts) {
                bytes.extend_from_slice(&piece[*first..*trailer]);
            }
            bytes.push(0x3B);
            return Ok(bytes);
        }
    }
    encode_serial(root, frames)
}

fn encode_serial(root: &Path, frames: &Frames<'_>) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = GifEncoder::new_with_speed(&mut bytes, 20);
        encoder
            .set_repeat(Repeat::Infinite)
            .map_err(|e| e.to_string())?;
        for (_, frame) in frames {
            let pixels = image::open(root.join(&frame.artifact.path))
                .map_err(|e| e.to_string())?
                .to_rgba8();
            encoder
                .encode_frame(Frame::from_parts(
                    pixels,
                    0,
                    0,
                    Delay::from_numer_denom_ms(500, 1),
                ))
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(bytes)
}

fn skip_sub_blocks(bytes: &[u8], mut pos: usize) -> Option<usize> {
    loop {
        let length = usize::from(*bytes.get(pos)?);
        pos += 1;
        if length == 0 {
            return Some(pos);
        }
        pos += length;
    }
}

/// `(first frame offset, trailer offset)` of a well-formed GIF: everything
/// before the first frame is the header (screen descriptor, global palette,
/// loop extension), and the trailer is the final byte.
pub(crate) fn layout(bytes: &[u8]) -> Option<(usize, usize)> {
    if bytes.len() < 14 || !(bytes.starts_with(b"GIF89a") || bytes.starts_with(b"GIF87a")) {
        return None;
    }
    let packed = bytes[10];
    let mut pos = 13 + if packed & 0x80 != 0 { 3usize << ((packed & 7) + 1) } else { 0 };
    let mut first = None;
    loop {
        match *bytes.get(pos)? {
            0x21 => {
                // Application extensions (the loop count) belong to the
                // header; any other extension opens a frame.
                if *bytes.get(pos + 1)? != 0xFF && first.is_none() {
                    first = Some(pos);
                }
                pos = skip_sub_blocks(bytes, pos + 2)?;
            }
            0x2C => {
                if first.is_none() {
                    first = Some(pos);
                }
                let packed = *bytes.get(pos + 9)?;
                pos += 10;
                if packed & 0x80 != 0 {
                    pos += 3usize << ((packed & 7) + 1);
                }
                pos = skip_sub_blocks(bytes, pos + 1)?;
            }
            0x3B if pos + 1 == bytes.len() => return Some((first?, pos)),
            _ => return None,
        }
    }
}

/// The previous loop extended by the frames after its own, or `None` when
/// the previous loop is not an exact prefix (a replaced frame, a corrupt or
/// missing file, a different header), in which case the caller encodes all.
fn extend(
    root: &Path,
    previous: &RadarLoop,
    frames: &Frames<'_>,
    path_for: impl Fn(&str) -> std::path::PathBuf,
) -> Result<Option<Vec<u8>>, String> {
    let count = previous.frames.len();
    if count == 0 || count >= frames.len() {
        return Ok(None);
    }
    let prefix = &frames[..count];
    let same_frames = previous
        .frames
        .iter()
        .zip(&previous.valid_times)
        .zip(prefix)
        .all(|((path, time), (want_time, image))| {
            *path == image.artifact.path && time == want_time
        });
    // The loop's own name is the digest of its frames' times and hashes, so
    // a matching name proves the old GIF holds exactly these frame bytes.
    if !same_frames || relative(root, &path_for(&loop_key(prefix)?)) != previous.artifact.path {
        return Ok(None);
    }
    let old_path = root.join(&previous.artifact.path);
    let old = match std::fs::read(&old_path) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(None),
    };
    if manifest::digest(&old) != previous.artifact.sha256 || old.len() as u64 != previous.artifact.bytes {
        return Ok(None);
    }
    let tail = encode(root, &frames[count..])?;
    let (Some((old_first, old_trailer)), Some((tail_first, _))) = (layout(&old), layout(&tail))
    else {
        return Ok(None);
    };
    if old[..old_first] != tail[..tail_first] {
        return Ok(None);
    }
    let mut bytes = Vec::with_capacity(old_trailer + tail.len() - tail_first);
    bytes.extend_from_slice(&old[..old_trailer]);
    bytes.extend_from_slice(&tail[tail_first..]);
    Ok(Some(bytes))
}

pub fn update(root: &Path, inventory: &mut Manifest) -> Result<(), String> {
    let mut groups: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for volume in &inventory.volumes {
        for image in &volume.images {
            groups
                .entry((
                    volume.domain.clone(),
                    volume.site.id.clone(),
                    image.field.clone(),
                    image.tilt_index,
                ))
                .or_default()
                .push((volume.valid_time.clone(), image));
        }
    }
    // Every loop (site, field, tilt) is built at once, kept in key order.
    let previous_loops = &inventory.loops;
    let groups: Vec<_> = groups.into_iter().collect();
    let loops = groups
        .into_par_iter()
        .map(|((domain, site, field, tilt), mut frames)| -> Result<RadarLoop, String> {
        frames.sort_by(|a, b| a.0.cmp(&b.0));
        let directory = root
            .join("radar")
            .join(&domain)
            .join(&site)
            .join("loops")
            .join(&field);
        let path_for = |key: &str| directory.join(format!("tilt-{tilt:02}-{key}.gif"));
        let path = path_for(&loop_key(&frames)?);
        let valid_existing = previous_loops
            .iter()
            .find(|old| old.artifact.path == relative(root, &path))
            .is_some_and(|old| {
                manifest::file_hash(&path).is_ok_and(|(sha, bytes)| {
                    sha == old.artifact.sha256 && bytes == old.artifact.bytes
                })
            });
        if !valid_existing {
            let previous = previous_loops.iter().find(|old| {
                old.domain == domain
                    && old.site_id == site
                    && old.field == field
                    && old.tilt_index == tilt
            });
            let extended = match previous {
                Some(previous) => extend(root, previous, &frames, &path_for)?,
                None => None,
            };
            let bytes = match extended {
                Some(bytes) => bytes,
                None => encode(root, &frames)?,
            };
            manifest::atomic_bytes(&path, &bytes)?;
        }
        Ok(RadarLoop {
            domain,
            site_id: site,
            field,
            tilt_index: tilt,
            elevation_deg: frames[0].1.elevation_deg,
            valid_times: frames.iter().map(|f| f.0.clone()).collect(),
            frames: frames.iter().map(|f| f.1.artifact.path.clone()).collect(),
            artifact: manifest::artifact(root, &path, "gif")?,
        })
        })
        .collect::<Result<Vec<_>, String>>()?;
    for old in &inventory.loops {
        if !loops.iter().any(|new| new.artifact.path == old.artifact.path) {
            inventory.retired.push(old.artifact.path.clone());
        }
    }
    inventory.loops = loops;
    inventory.save(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{Artifact, Image, Site, Volume};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "simradar-loops-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn frame_volume(root: &Path, minute: u32) -> Volume {
        let time = format!("2026-10-02T00:{minute:02}:00+00:00");
        let dir = root.join(format!("radar/d01/X001/generation-{minute}/ppi/reflectivity"));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("frame_tilt-00.png");
        let mut pixels = image::RgbaImage::new(24, 16);
        for (x, y, pixel) in pixels.enumerate_pixels_mut() {
            let shade = (x * 7 + y * 3 + minute * 41) as u8;
            *pixel = image::Rgba([shade, shade.wrapping_mul(3), 255 - shade, 255]);
        }
        pixels.save(&png).unwrap();
        Volume {
            domain: "d01".into(),
            generation: format!("generation-{minute}"),
            expected_sites: vec!["X001".into()],
            implementation: serde_json::json!({}),
            site: Site {
                id: "X001".into(),
                band: None,
                latitude_deg: 35.0,
                longitude_deg: -98.0,
                antenna_height_msl_m: 100.0,
            },
            valid_time: time.clone(),
            scan_start: time.clone(),
            scan_end: time.clone(),
            timing_requested: "history".into(),
            timing_used: "history".into(),
            source_times: vec![time],
            source_history: vec![],
            tilts_deg: vec![0.5],
            fields: vec!["reflectivity".into()],
            config_sha256: "test".into(),
            reflectivity_source: "test".into(),
            dual_pol_status: None,
            writer_reports: Default::default(),
            files: vec![Artifact {
                path: "radar/unused".into(),
                format: "level2".into(),
                sha256: String::new(),
                bytes: 0,
            }],
            images: vec![Image {
                artifact: manifest::artifact(root, &png, "png").unwrap(),
                field: "reflectivity".into(),
                tilt_index: 0,
                sweep_index: 0,
                elevation_deg: 0.5,
            }],
            elapsed_seconds: 0.0,
        }
    }

    fn frames(inventory: &Manifest) -> Vec<(String, &Image)> {
        inventory
            .volumes
            .iter()
            .map(|v| (v.valid_time.clone(), &v.images[0]))
            .collect()
    }

    #[test]
    fn a_live_loop_is_extended_byte_identically_and_the_one_it_replaces_is_deleted() {
        let fixture = Fixture::new();
        let root = &fixture.0;
        let mut inventory = Manifest::load(root, "test-extraction").unwrap();
        let mut seen = Vec::new();
        for minute in 0..4 {
            inventory.upsert(frame_volume(root, minute));
            update(root, &mut inventory).unwrap();
            assert_eq!(inventory.loops.len(), 1);
            let current = root.join(&inventory.loops[0].artifact.path);
            let written = std::fs::read(&current).unwrap();
            assert_eq!(
                written,
                encode(root, &frames(&inventory)).unwrap(),
                "an extended loop must equal encoding every frame again"
            );
            assert!(layout(&written).is_some());
            for old in &seen {
                assert!(!root.join(old).exists(), "superseded loop {old} stayed on disk");
            }
            seen.push(inventory.loops[0].artifact.path.clone());
        }
        let gifs = std::fs::read_dir(root.join("radar/d01/X001/loops/reflectivity"))
            .unwrap()
            .count();
        assert_eq!(gifs, 1, "only the loop the manifest names stays on disk");
    }

    #[test]
    fn a_replaced_frame_reencodes_instead_of_extending_a_stale_prefix() {
        let fixture = Fixture::new();
        let root = &fixture.0;
        let mut inventory = Manifest::load(root, "test-extraction").unwrap();
        for minute in 0..2 {
            inventory.upsert(frame_volume(root, minute));
        }
        update(root, &mut inventory).unwrap();
        let mut replaced = frame_volume(root, 7);
        replaced.valid_time = inventory.volumes[0].valid_time.clone();
        inventory.upsert(replaced);
        inventory.upsert(frame_volume(root, 2));
        update(root, &mut inventory).unwrap();
        let written = std::fs::read(root.join(&inventory.loops[0].artifact.path)).unwrap();
        assert_eq!(written, encode(root, &frames(&inventory)).unwrap());
    }

    #[test]
    fn layout_refuses_a_truncated_gif() {
        let fixture = Fixture::new();
        let root = &fixture.0;
        let mut inventory = Manifest::load(root, "test-extraction").unwrap();
        inventory.upsert(frame_volume(root, 0));
        let bytes = encode(root, &frames(&inventory)).unwrap();
        assert!(layout(&bytes).is_some());
        assert!(layout(&bytes[..bytes.len() - 1]).is_none());
        assert!(layout(&bytes[..bytes.len() / 2]).is_none());
    }
}
