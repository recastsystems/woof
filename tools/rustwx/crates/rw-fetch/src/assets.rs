//! Pinned static objects and native decompression. No numerical transformations.
use crate::{Failure, net::Fetcher};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path};

fn options(args: &[String]) -> Result<BTreeMap<&str, &str>, Failure> {
    let mut out = BTreeMap::new();
    if args.len() % 2 != 0 { return Err(Failure::Usage("asset options need values".into())); }
    for pair in args.chunks(2) {
        if !["--url", "--output", "--input", "--bytes", "--sha256", "--datasets", "--codec", "--gzip-decoder"].contains(&pair[0].as_str())
            || out.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err(Failure::Usage(format!("unknown or repeated asset option {}", pair[0])));
        }
    }
    Ok(out)
}

pub fn hash(path: &Path) -> Result<(u64, String), String> {
    let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut digest = Sha256::new();
    let mut bytes = 0;
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        digest.update(&buffer[..n]);
        bytes += n as u64;
    }
    Ok((bytes, format!("{:x}", digest.finalize())))
}

pub fn command(kind: &str, args: &[String]) -> Result<String, Failure> {
    let opt = options(args)?;
    let get = |key| opt.get(key).copied().ok_or_else(|| Failure::Usage(format!("{kind} needs {key}")));
    let pin = || -> Result<(u64, &str), Failure> {
        let bytes = get("--bytes")?.parse::<u64>().map_err(|_| Failure::Usage("invalid --bytes pin".into()))?;
        let sha = get("--sha256")?;
        if bytes == 0 || sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
            return Err(Failure::Usage("static pin needs positive bytes and 64 lowercase hex SHA-256 digits".into()));
        }
        Ok((bytes, sha))
    };
    let doc = match kind {
        "asset" | "encoded-asset" => {
            let path = Path::new(get("--output")?);
            if let Some(parent) = path.parent() { fs::create_dir_all(parent).map_err(|e| Failure::Refused(e.to_string()))?; }
            let url = get("--url")?;
            if let Some(local) = url.strip_prefix("file://") {
                let local = local_file_path(local)?;
                let mut input = File::open(local).map_err(|e| Failure::Refused(format!("{url}: {e}")))?;
                let length = input.metadata().map_err(|e| Failure::Refused(e.to_string()))?.len();
                let (bytes, sha) = if kind == "asset" { pin()? } else { (length, "") };
                if length != bytes { return Err(Failure::Refused(format!("{url}: pin refused: object has {length} bytes, expected {bytes}"))); }
                let receipt = crate::stream::publish_pinned_with(path, bytes, sha, |writer| {
                    let mut buffer = vec![0u8; 1024 * 1024];
                    loop {
                        let n = input.read(&mut buffer).map_err(|e| crate::net::TransferFault::Refused(e.to_string()))?;
                        if n == 0 { break; }
                        writer.push(&buffer[..n]).map_err(crate::net::TransferFault::Refused)?;
                    }
                    Ok(crate::stream::unix_now())
                })?;
                return serde_json::to_string(&receipt).map_err(|e| Failure::Refused(e.to_string()));
            }
            let fetcher = Fetcher::new(None)?;
            let reporter = crate::ProgressReporter::start(fetcher.progress(), Box::new(|line| eprint!("{line}")));
            reporter.object(0);
            let receipt = if kind == "asset" {
                let (bytes, sha) = pin()?;
                fetcher.stream_pinned_file(url, path, bytes, sha)?
            } else { fetcher.stream_encoded_file(url, path)? };
            reporter.finish();
            serde_json::to_value(receipt).map_err(|e| Failure::Refused(e.to_string()))?
        }
        "hash" => {
            let path = Path::new(get("--input")?);
            let (bytes, sha) = hash(path)?;
            if opt.contains_key("--sha256") || opt.contains_key("--bytes") {
                let (expected_bytes, expected_sha) = pin()?;
                if bytes != expected_bytes || sha != expected_sha {
                    return Err(Failure::Refused(format!("{}: pin refused: {bytes} bytes SHA-256 {sha}, expected {expected_bytes} bytes SHA-256 {expected_sha}; different static bytes change geography or lookup coefficients", path.display())));
                }
            }
            serde_json::json!({"bytes": bytes, "sha256": sha})
        }
        "unpack" | "unpack-audit" => {
            let names = get("--datasets")?.split(',').map(str::to_owned).collect::<Vec<_>>();
            let counts = if let Some(program) = opt.get("--gzip-decoder") {
                let root = Path::new(get("--output")?);
                fs::create_dir(root).map_err(|e| Failure::Refused(e.to_string()))?;
                let mut reader = ParallelGzip::open(Path::new(get("--input")?), program)?;
                untar(&mut reader, root, &names, kind == "unpack-audit")?
            } else {
                unpack(Path::new(get("--input")?), Path::new(get("--output")?), &names, kind == "unpack-audit")?
            };
            serde_json::to_value(counts).map_err(|e| Failure::Refused(e.to_string()))?
        }
        "decode" => {
            let codec = get("--codec")?;
            if !["zstd", "bz2", "gz"].contains(&codec) {
                return Err(Failure::Usage(format!("decode --codec must be zstd, bz2 or gz, not {codec:?}")));
            }
            let bytes = decode(Path::new(get("--input")?), Path::new(get("--output")?), codec)?;
            serde_json::json!({"bytes": bytes})
        }
        _ => unreachable!(),
    };
    serde_json::to_string(&doc).map_err(|e| Failure::Refused(e.to_string()))
}

fn text(bytes: &[u8]) -> Result<String, String> {
    String::from_utf8(bytes.split(|b| *b == 0).next().unwrap_or_default().to_vec())
        .map_err(|_| "tar member has a non-UTF-8 name".into())
}

fn local_file_path(value: &str) -> Result<String, Failure> {
    let value = value.strip_prefix("localhost").unwrap_or(value);
    if !value.starts_with('/') {
        return Err(Failure::Refused("file mirror is not local; an unverified network path must not supply static bytes".into()));
    }
    let mut decoded = Vec::new();
    let mut chars = value.as_bytes().iter().copied();
    while let Some(byte) = chars.next() {
        if byte == b'%' {
            let mut digit = || chars.next().and_then(|b| char::from(b).to_digit(16))
                .ok_or_else(|| Failure::Refused("invalid file mirror escape; it would name another source path".into()));
            let high = digit()?;
            let low = digit()?;
            decoded.push((high * 16 + low) as u8);
        } else { decoded.push(byte); }
    }
    let path = String::from_utf8(decoded).map_err(|_| Failure::Refused("file mirror path is not UTF-8".into()))?;
    #[cfg(windows)]
    let path = if path.as_bytes().get(2) == Some(&b':') { path[1..].to_owned() } else { path };
    Ok(path)
}

fn octal(bytes: &[u8]) -> Result<u64, String> {
    let value = text(bytes)?.trim().to_owned();
    u64::from_str_radix(if value.is_empty() { "0" } else { &value }, 8)
        .map_err(|_| "tar member has invalid octal size or checksum".into())
}

fn safe_path(name: &str) -> Result<Vec<&str>, String> {
    if name.contains('\\') || name.contains(':') || Path::new(name).components().any(|c| matches!(c, Component::ParentDir | Component::RootDir | Component::Prefix(_))) {
        return Err(format!("unsafe tar member {name:?}; traversal would write outside the unpublished staging tree"));
    }
    Ok(name.split('/').filter(|p| !p.is_empty() && *p != ".").collect())
}

fn copy_exact(reader: &mut impl Read, writer: &mut impl io::Write, bytes: u64) -> Result<(), String> {
    let copied = io::copy(&mut reader.take(bytes), writer).map_err(|e| e.to_string())?;
    if copied != bytes { return Err(format!("truncated tar body: {copied} of {bytes} bytes; incomplete tiles must not be published")); }
    Ok(())
}

fn untar(reader: &mut impl Read, root: &Path, datasets: &[String], audit: bool) -> Result<BTreeMap<String, serde_json::Value>, String> {
    let mut counts = BTreeMap::new();
    let mut written = BTreeSet::new();
    let mut receipt = if audit {
        Some(File::options().write(true).create_new(true).open(root.join("unpack-audit.jsonl"))
             .map_err(|e| e.to_string())?)
    } else { None };
    for name in datasets {
        if safe_path(name)?.len() != 1 || name == "WPS_GEOG" || name == "." {
            return Err(format!("invalid dataset directory {name:?}"));
        }
        counts.insert(name.clone(), serde_json::json!({"files": 0u64, "bytes": 0u64}));
    }
    let mut long_name: Option<String> = None;
    let mut pax_name: Option<String> = None;
    loop {
        let mut h = [0u8; 512];
        reader.read_exact(&mut h).map_err(|e| format!("truncated tar header: {e}"))?;
        if h.iter().all(|b| *b == 0) {
            break;
        }
        let checksum = h.iter().enumerate().map(|(i, b)| if (148..156).contains(&i) { 32 } else { u64::from(*b) }).sum::<u64>();
        if checksum != octal(&h[148..156])? { return Err("tar header checksum refused; damaged member metadata would select wrong tile paths".into()); }
        let size = octal(&h[124..136])?;
        let padding = (512 - size % 512) % 512;
        let kind = h[156];
        if kind == b'L' || kind == b'x' || kind == b'g' {
            if size > 1024 * 1024 { return Err("tar extended metadata exceeds 1 MiB".into()); }
            let mut metadata = vec![0; size as usize];
            reader.read_exact(&mut metadata).map_err(|e| e.to_string())?;
            if kind == b'L' { long_name = Some(text(&metadata)?); }
            else {
                let payload = String::from_utf8(metadata).map_err(|e| e.to_string())?;
                let mut offset = 0;
                while offset < payload.len() {
                    let space = payload[offset..].find(' ').ok_or("invalid PAX record")? + offset;
                    let length = payload[offset..space].parse::<usize>().map_err(|_| "invalid PAX length")?;
                    let end = offset.checked_add(length).filter(|end| *end <= payload.len())
                        .ok_or("invalid PAX extent; overflowing metadata could select a wrong tile path")?;
                    if length == 0 || space + 1 >= end || !payload.is_char_boundary(end - 1)
                        || payload.as_bytes()[end - 1] != b'\n' {
                        return Err("invalid PAX extent".into());
                    }
                    let record = &payload[space + 1..end - 1];
                    if let Some(path) = record.strip_prefix("path=") { pax_name = Some(path.to_owned()); }
                    if record.starts_with("linkpath=") || record.starts_with("GNU.sparse.") || record.starts_with("size=") {
                        return Err("tar PAX link, sparse or size override refused; tile extent or path would be ambiguous".into());
                    }
                    offset = end;
                }
            }
        } else {
            let name = pax_name.take().or_else(|| long_name.take()).unwrap_or({
                let name = text(&h[..100])?;
                let prefix = if &h[257..263] == b"ustar\0" { text(&h[345..500])? } else { String::new() };
                if prefix.is_empty() { name } else { format!("{prefix}/{name}") }
            });
            let mut parts = safe_path(&name)?;
            if kind != 0 && kind != b'0' && kind != b'5' { return Err(format!("non-regular tar member {name:?} type {kind}; links or devices would escape tile staging")); }
            if parts.first() == Some(&"WPS_GEOG") { parts.remove(0); }
            let selected = parts.first().and_then(|name| counts.get_mut(*name));
            let ignored = parts.last().is_some_and(|p| p.starts_with("._") || *p == ".DS_Store");
            if let Some(count) = selected.filter(|_| !ignored) {
                let target = root.join(parts.join("/"));
                if kind == b'5' {
                    if size != 0 { return Err(format!("directory {name:?} has a nonempty body")); }
                    fs::create_dir_all(target).map_err(|e| e.to_string())?;
                } else {
                    if !written.insert(parts.join("/")) {
                        return Err(format!("duplicate tar tile {name:?}; replacing it would change the archived field inventory"));
                    }
                    if let Some(parent) = target.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
                    let mut file = File::options().write(true).create_new(true).open(&target).map_err(|e| format!("{}: {e}", target.display()))?;
                    copy_exact(reader, &mut file, size)?;
                    drop(file);
                    if let Some(receipt) = receipt.as_mut() {
                        // Audit the actual written file through an independent native read.
                        // The durable per-file receipt precedes disposal of this audit output.
                        let (bytes, sha256) = hash(&target)?;
                        let row = serde_json::json!({"path": parts.join("/"), "bytes": bytes, "sha256": sha256});
                        writeln!(receipt, "{row}").map_err(|e| e.to_string())?;
                        receipt.sync_data().map_err(|e| e.to_string())?;
                        fs::remove_file(&target).map_err(|e| e.to_string())?;
                    }
                    count["files"] = serde_json::json!(count["files"].as_u64().unwrap() + 1);
                    count["bytes"] = serde_json::json!(count["bytes"].as_u64().unwrap() + size);
                }
            } else { copy_exact(reader, &mut io::sink(), size)?; }
        }
        copy_exact(reader, &mut io::sink(), padding)?;
    }
    // Drain the codec to observe CRC failures after the tar terminator.
    io::copy(reader, &mut io::sink()).map_err(|e| e.to_string())?;
    Ok(counts)
}

/// A read error from an in-process decoder is the codec refusing the stream
/// (a CRC, checksum or framing failure), so it is named as one.
struct Decoding<R> { inner: R, label: String }

// rapidgzip decodes ordinary single-stream gzip on independent workers. A
// streaming tar walk consumes it once; seeking a gzip tar member used to
// decompress the full archive twice. CRC and child status must both pass.
struct ParallelGzip { child: std::process::Child, output: std::process::ChildStdout, ended: bool }
impl ParallelGzip {
    fn open(input: &Path, program: &str) -> Result<Self, String> {
        let workers = std::env::var("RAYON_NUM_THREADS").ok()
            .and_then(|n| n.parse::<usize>().ok()).filter(|n| *n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
        let mut child = std::process::Command::new(program)
            .args(["--decompress", "--stdout", "--verify", "--decoder-parallelism", &workers.to_string()])
            .arg(input).stdout(std::process::Stdio::piped()).spawn()
            .map_err(|e| format!("parallel gzip decoder {program}: {e}"))?;
        let output = child.stdout.take().ok_or("parallel gzip decoder has no output")?;
        Ok(Self { child, output, ended: false })
    }
}
impl Read for ParallelGzip {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let n = self.output.read(bytes)?;
        if n == 0 && !bytes.is_empty() && !self.ended {
            self.ended = true;
            let status = self.child.wait()?;
            if !status.success() {
                return Err(io::Error::other(format!("parallel gzip integrity failed: {status}")));
            }
        }
        Ok(n)
    }
}
impl Drop for ParallelGzip {
    fn drop(&mut self) {
        if !self.ended { let _ = self.child.kill(); }
        let _ = self.child.wait();
    }
}

impl<R: Read> Read for Decoding<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}; compressed stream integrity failed", self.label)))
    }
}

/// Decode `input` with `codec` (`zstd`, `bz2`, `gz`, or `none`) in this
/// process. The codecs used to be the host's `pzstd`, `bzip2` and `gzip`
/// programs, and none of them is on a stock Windows PATH (`pzstd` has no
/// Windows build at all), so every geography archive and every compressed
/// table refused there with "program not found". Each decoder reads every
/// concatenated stream or frame, as `bzip2 -d`, `gzip -d` and `pzstd -d` do:
/// parallel compressors write one stream per block.
fn decoder(input: &Path, codec: &str) -> Result<Box<dyn Read>, String> {
    let name = input.to_string_lossy().into_owned();
    let file = io::BufReader::new(File::open(input).map_err(|e| format!("{name}: {e}"))?);
    let label = format!("{codec} refused {name}");
    Ok(match codec {
        "zstd" => Box::new(Decoding { inner: zstd::stream::read::Decoder::with_buffer(file).map_err(|e| format!("{label}: {e}"))?, label }),
        // Blocks decode on every core (bz2par); the output is byte-identical to
        // MultiBzDecoder, which took 4-5 minutes on one core for WPS greenfrac.
        "bz2" => { drop(file); Box::new(Decoding { inner: crate::bz2par::open(input).map_err(|e| format!("{label}: {e}"))?, label }) }
        "gz" => Box::new(Decoding { inner: flate2::bufread::MultiGzDecoder::new(file), label }),
        "none" => Box::new(file),
        other => return Err(format!("unknown codec {other:?}; an undecoded stream would be published as the pinned bytes")),
    })
}

fn unpack(archive: &Path, root: &Path, datasets: &[String], audit: bool) -> Result<BTreeMap<String, serde_json::Value>, String> {
    let name = archive.to_string_lossy();
    let codec = if name.ends_with(".zst") { "zstd" } else if name.ends_with(".bz2") { "bz2" }
        else if name.ends_with(".gz") { "gz" } else { "none" };
    fs::create_dir(root).map_err(|e| format!("{}: staging must be newly created: {e}", root.display()))?;
    // Independent dataset unpacks run concurrently in the orchestrator.
    untar(&mut decoder(archive, codec)?, root, datasets, audit)
}

/// Decode one compressed object into a new file; the caller pins the result.
fn decode(input: &Path, output: &Path, codec: &str) -> Result<u64, String> {
    let mut reader = decoder(input, codec)?;
    let mut file = File::options().write(true).create_new(true).open(output).map_err(|e| format!("{}: {e}", output.display()))?;
    let bytes = io::copy(&mut reader, &mut file).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| format!("{}: {e}", output.display()))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuse_escape_names() {
        for name in ["/absolute", "../tile", "data/../tile", "C:/tile", "data\\tile"] { assert!(safe_path(name).is_err(), "{name}"); }
        assert_eq!(safe_path("./data/index").unwrap(), ["data", "index"]);
    }
    #[test]
    fn hash_known_answer() {
        let path = std::env::temp_dir().join(format!("rw-assets-hash-{}", std::process::id()));
        fs::write(&path, b"abc").unwrap();
        assert_eq!(hash(&path).unwrap(), (3, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad".into()));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_extended_metadata_is_a_refusal() {
        let root = std::env::temp_dir().join(format!("rw-assets-pax-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        for payload in [b"9999999999999999999 path=bad\n".as_slice(), b"0 path=bad\n", b"12 path=badX"] {
            let mut header = [0u8; 512];
            header[..4].copy_from_slice(b"pax\0");
            let size = format!("{:011o}\0", payload.len());
            header[124..136].copy_from_slice(size.as_bytes());
            header[148..156].fill(b' ');
            header[156] = b'x';
            let checksum = header.iter().map(|b| u64::from(*b)).sum::<u64>();
            header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
            let mut stream = header.to_vec();
            stream.extend_from_slice(payload);
            stream.resize(2048, 0);
            assert!(untar(&mut stream.as_slice(), &root, &["dataset".into()], false).is_err());
        }
        fs::remove_dir(root).unwrap();
    }

    fn tar_member(name: &str, body: &[u8]) -> Vec<u8> {
        let mut header = [0u8; 512];
        header[..name.len()].copy_from_slice(name.as_bytes());
        header[124..136].copy_from_slice(format!("{:011o}\0", body.len()).as_bytes());
        header[148..156].fill(b' ');
        header[156] = b'0';
        let checksum = header.iter().map(|b| u64::from(*b)).sum::<u64>();
        header[148..156].copy_from_slice(format!("{checksum:06o}\0 ").as_bytes());
        let mut member = header.to_vec();
        member.extend_from_slice(body);
        member.resize(member.len() + (512 - body.len() % 512) % 512, 0);
        member
    }

    /// Each codec decodes in process, across every concatenated stream or
    /// frame (what parallel compressors write), with no host program.
    #[test]
    fn every_codec_decodes_concatenated_streams_in_process() {
        use std::io::Write as _;
        let tile = vec![7u8; 3000];
        let mut tar = [tar_member("WPS_GEOG/keep/index", b"type=continuous\n"),
                       tar_member("WPS_GEOG/keep/00001-00002.00001-00002", &tile),
                       tar_member("WPS_GEOG/other/index", b"type=categorical\n")].concat();
        tar.resize(tar.len() + 1024, 0);
        let (first, second) = tar.split_at(1500);
        let bz2 = |part: &[u8]| { let mut e = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default()); e.write_all(part).unwrap(); e.finish().unwrap() };
        let gz = |part: &[u8]| { let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default()); e.write_all(part).unwrap(); e.finish().unwrap() };
        let zst = |part: &[u8]| zstd::encode_all(part, 3).unwrap();
        // pzstd's layout: a skippable frame naming the next frame's size.
        let pzstd = |part: &[u8]| { let frame = zst(part); [&[0x50, 0x2A, 0x4D, 0x18, 4, 0, 0, 0][..], &(frame.len() as u32).to_le_bytes(), &frame].concat() };
        let base = std::env::temp_dir().join(format!("rw-assets-codecs-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base).unwrap();
        for (ext, stream) in [("tar.bz2", [bz2(first), bz2(second)].concat()), ("tar.gz", [gz(first), gz(second)].concat()),
                              ("tar.zst", [pzstd(first), pzstd(second)].concat()), ("tar", tar.clone())] {
            let archive = base.join(format!("geo.{ext}"));
            fs::write(&archive, &stream).unwrap();
            let root = base.join(format!("out-{ext}"));
            let counts = unpack(&archive, &root, &["keep".into()], false).unwrap();
            assert_eq!(counts["keep"]["files"], 2, "{ext}");
            assert_eq!(fs::read(root.join("keep/00001-00002.00001-00002")).unwrap(), tile, "{ext}");
            assert!(!root.join("other").exists(), "{ext}");
            // The same decoders back `decode`, which the compressed table fetch uses.
            if ext != "tar" {
                let codec = match ext { "tar.bz2" => "bz2", "tar.gz" => "gz", _ => "zstd" };
                let decoded = base.join(format!("decoded-{ext}"));
                assert_eq!(decode(&archive, &decoded, codec).unwrap(), tar.len() as u64);
                assert_eq!(fs::read(&decoded).unwrap(), tar, "{ext}");
                // A truncated stream is the codec refusing it, not a short success.
                fs::write(&archive, &stream[..stream.len() - 8]).unwrap();
                let error = decode(&archive, &base.join(format!("cut-{ext}")), codec).unwrap_err();
                assert!(error.contains("compressed stream integrity failed"), "{ext}: {error}");
            }
        }
        fs::remove_dir_all(base).unwrap();
    }
}
