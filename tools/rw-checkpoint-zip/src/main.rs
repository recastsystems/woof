//! Lossless NPZ compression before checkpoint publication. Each field is
//! compressed independently; archive assembly copies compressed bytes.
use anyhow::{ensure, Result};
use rayon::prelude::*;
use std::{collections::HashSet, fs::{self, File, OpenOptions}, io::{self, Cursor}, path::Path};
use zip::{ZipArchive, ZipWriter, CompressionMethod, write::SimpleFileOptions};

fn run(input: &Path, output: &Path, workers: usize) -> Result<()> {
    ensure!(workers > 0, "workers must be positive");
    ensure!(input != output, "input and output must differ");
    let started = std::time::Instant::now();
    let mut source = ZipArchive::new(File::open(input)?)?;
    ensure!(!source.is_empty(), "empty checkpoint archive");
    let mut names = HashSet::new();
    for index in 0..source.len() {
        let entry = source.by_index(index)?;
        ensure!(entry.name().ends_with(".npy") && !entry.is_dir(), "checkpoint entry is not NPY");
        ensure!(names.insert(entry.name().to_owned()), "duplicate checkpoint entry");
    }
    let count = source.len();
    drop(source);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers.min(count)).build()?;
    let fields: Vec<Vec<u8>> = pool.install(|| (0..count).into_par_iter().map(|index| -> Result<Vec<u8>> {
        let mut source = ZipArchive::new(File::open(input)?)?;
        let mut entry = source.by_index(index)?;
        let size = entry.size();
        let mut encoded = ZipWriter::new(Cursor::new(Vec::new()));
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated)
            .compression_level(Some(1)).large_file(size >= u32::MAX as u64);
        encoded.start_file(entry.name(), options)?;
        // Reading to EOF checks the source CRC. No array decoding, casting,
        // fill-value treatment, or floating-point arithmetic occurs here.
        ensure!(io::copy(&mut entry, &mut encoded)? == size, "truncated checkpoint entry");
        Ok(encoded.finish()?.into_inner())
    }).collect::<Result<_>>())?;
    let compression_seconds = started.elapsed().as_secs_f64();
    // The caller owns publication/fsync. Never clobber an existing output.
    let file = OpenOptions::new().write(true).create_new(true).open(output)?;
    let result = (|| -> Result<()> {
        let mut destination = ZipWriter::new(file);
        for field in fields {
            let mut archive = ZipArchive::new(Cursor::new(field))?;
            destination.raw_copy_file(archive.by_index(0)?)?;
        }
        destination.finish()?.sync_all()?;
        Ok(())
    })();
    if result.is_err() { let _ = fs::remove_file(output); }
    result?;
    println!("{}", serde_json::json!({"schema":"gpuwm.restart-compression/v1",
        "method":"parallel ZIP DEFLATE level 1", "fields":count,"workers":workers.min(count),
        "input_bytes":fs::metadata(input)?.len(),"output_bytes":fs::metadata(output)?.len(),
        "compression_seconds":compression_seconds,"wall_seconds":started.elapsed().as_secs_f64()}));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 4, "usage: rw-checkpoint-zip INPUT OUTPUT WORKERS");
    run(Path::new(&args[1]), Path::new(&args[2]), args[3].parse()?)
}
