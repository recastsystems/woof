//! Durable run inventory. Files are committed before the manifest names them.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "simulated-radar.manifest/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub path: String,
    pub format: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    #[serde(flatten)]
    pub artifact: Artifact,
    pub field: String,
    pub tilt_index: usize,
    pub sweep_index: usize,
    pub elevation_deg: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Site {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band: Option<String>,
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub antenna_height_msl_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadarLoop {
    pub domain: String,
    pub site_id: String,
    pub field: String,
    pub tilt_index: usize,
    pub elevation_deg: f32,
    pub valid_times: Vec<String>,
    pub frames: Vec<String>,
    #[serde(flatten)]
    pub artifact: Artifact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Volume {
    pub domain: String,
    pub generation: String,
    pub expected_sites: Vec<String>,
    pub implementation: serde_json::Value,
    pub site: Site,
    pub valid_time: String,
    pub scan_start: String,
    pub scan_end: String,
    pub timing_requested: String,
    pub timing_used: String,
    pub source_times: Vec<String>,
    pub source_history: Vec<String>,
    pub tilts_deg: Vec<f32>,
    pub fields: Vec<String>,
    pub config_sha256: String,
    pub reflectivity_source: String,
    pub dual_pol_status: Option<String>,
    pub writer_reports: std::collections::BTreeMap<String, serde_json::Value>,
    pub files: Vec<Artifact>,
    pub images: Vec<Image>,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub simulated: bool,
    pub model: String,
    pub velocity_convention: String,
    pub bowecho_source_commit: String,
    pub bowecho_extraction_commit: String,
    pub writer_source_commit: String,
    pub updated_at: String,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub loops: Vec<RadarLoop>,
    pub volumes: Vec<Volume>,
    /// Run-relative files a replaced volume or loop named. They are deleted
    /// after a save that no longer names them, so a superseded generation
    /// or loop does not stay on disk beside the one replacing it.
    #[serde(skip)]
    pub retired: Vec<String>,
}

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn artifact(root: &Path, path: &Path, format: &str) -> Result<Artifact, String> {
    let (sha256, bytes) = file_hash(path)?;
    Ok(Artifact {
        path: path
            .strip_prefix(root)
            .map_err(|_| "artifact escaped run directory")?
            .to_string_lossy()
            .replace('\\', "/"),
        format: format.into(),
        sha256,
        bytes,
    })
}

pub fn file_hash(path: &Path) -> Result<(String, u64), String> {
    let mut file = File::open(path).map_err(|e| format!("hash {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok((
        format!("{:x}", hash.finalize()),
        file.metadata().map_err(|e| e.to_string())?.len(),
    ))
}

pub fn atomic_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("output has no parent directory")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let tmp = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    let mut file = File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    drop(file);
    fs::rename(&tmp, path).map_err(|e| format!("commit {}: {e}", path.display()))
}

/// An advisory OS file lock survives process crashes without a stale sentinel.
pub struct RunLock {
    _file: File,
}
impl RunLock {
    pub fn acquire(root: &Path) -> Result<Self, String> {
        let dir = root.join("radar");
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.join(".manifest.lock"))
            .map_err(|e| e.to_string())?;
        file.lock()
            .map_err(|e| format!("lock radar manifest: {e}"))?;
        Ok(Self { _file: file })
    }
}

impl Manifest {
    pub fn path(root: &Path) -> PathBuf {
        root.join("radar/manifest.json")
    }

    pub fn load(root: &Path, extraction: &str) -> Result<Self, String> {
        let path = Self::path(root);
        if path.exists() {
            let value: Self = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
                .map_err(|e| format!("read existing radar manifest: {e}"))?;
            if value.schema != SCHEMA || !value.simulated {
                return Err(
                    "existing radar manifest has a different schema or is not simulated".into(),
                );
            }
            return Ok(value);
        }
        Ok(Self {
            schema: SCHEMA.into(),
            simulated: true,
            model: "WOOF".into(),
            velocity_convention: "positive away from radar; negative toward radar; m s-1".into(),
            bowecho_source_commit: "eea07fc0a032d4de439d0bfe2def9098afd8b5a8".into(),
            bowecho_extraction_commit: extraction.into(),
            writer_source_commit: "c206a2495c36341caa2a62ff7be3025320dbb028".into(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            warnings: Vec::new(),
            loops: Vec::new(),
            volumes: Vec::new(),
            retired: Vec::new(),
        })
    }

    /// Drop every volume `superseded` selects and queue its files for
    /// deletion after the next save.
    pub fn retire_where(&mut self, superseded: impl Fn(&Volume) -> bool) {
        let (gone, kept): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.volumes).into_iter().partition(|v| superseded(v));
        self.volumes = kept;
        for volume in gone {
            self.retired.extend(
                volume
                    .files
                    .iter()
                    .chain(volume.images.iter().map(|i| &i.artifact))
                    .map(|a| a.path.clone()),
            );
        }
    }

    pub fn upsert(&mut self, volume: Volume) {
        self.retire_where(|old| {
            old.domain == volume.domain
                && old.site.id == volume.site.id
                && old.valid_time == volume.valid_time
        });
        self.volumes.push(volume);
        self.volumes.sort_by(|a, b| {
            (&a.domain, &a.site.id, &a.valid_time).cmp(&(&b.domain, &b.site.id, &b.valid_time))
        });
    }

    pub fn complete_generation(
        &self,
        root: &Path,
        domain: &str,
        valid: &str,
        generation: &str,
    ) -> bool {
        let entries: Vec<_> = self
            .volumes
            .iter()
            .filter(|v| v.domain == domain && v.valid_time == valid)
            .collect();
        !entries.is_empty()
            && entries.len() == entries[0].expected_sites.len()
            && entries
                .iter()
                .all(|v| v.generation == generation && self.volume_files_valid(root, v))
            && entries[0]
                .expected_sites
                .iter()
                .all(|id| entries.iter().any(|v| v.site.id == *id))
    }

    pub fn site_complete(
        &self,
        root: &Path,
        domain: &str,
        valid: &str,
        generation: &str,
        site: &str,
    ) -> bool {
        self.volumes
            .iter()
            .find(|v| {
                v.domain == domain
                    && v.valid_time == valid
                    && v.generation == generation
                    && v.site.id == site
            })
            .is_some_and(|v| self.volume_files_valid(root, v))
    }

    fn volume_files_valid(&self, root: &Path, v: &Volume) -> bool {
        !v.files.is_empty()
            && v.files
                .iter()
                .chain(v.images.iter().map(|i| &i.artifact))
                .all(|a| {
                    if Path::new(&a.path)
                        .components()
                        .any(|p| !matches!(p, std::path::Component::Normal(_)))
                    {
                        return false;
                    }
                    let path = root.join(&a.path);
                    path.is_file()
                        && artifact(root, &path, &a.format)
                            .is_ok_and(|got| got.sha256 == a.sha256 && got.bytes == a.bytes)
                })
    }

    pub fn save(&mut self, root: &Path) -> Result<(), String> {
        self.updated_at = chrono::Utc::now().to_rfc3339();
        atomic_bytes(
            &Self::path(root),
            &serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?,
        )?;
        self.sweep_retired(root);
        Ok(())
    }

    /// Delete retired files the saved manifest no longer names. A file still
    /// named (a loop frame until its loop is rebuilt, or a path a newer
    /// volume reuses) waits for a later save. A file that cannot be removed
    /// is reported, never treated as published.
    fn sweep_retired(&mut self, root: &Path) {
        if self.retired.is_empty() {
            return;
        }
        let named: std::collections::BTreeSet<&str> = self
            .volumes
            .iter()
            .flat_map(|v| v.files.iter().chain(v.images.iter().map(|i| &i.artifact)))
            .map(|a| a.path.as_str())
            .chain(self.loops.iter().flat_map(|l| {
                std::iter::once(l.artifact.path.as_str())
                    .chain(l.frames.iter().map(String::as_str))
            }))
            .collect();
        let radar = root.join("radar");
        let mut waiting = Vec::new();
        for relative in std::mem::take(&mut self.retired) {
            if named.contains(relative.as_str()) {
                waiting.push(relative);
                continue;
            }
            let inside = Path::new(&relative)
                .components()
                .all(|p| matches!(p, std::path::Component::Normal(_)))
                && relative.starts_with("radar/");
            if !inside {
                eprintln!("warning: not deleting superseded radar path outside radar/: {relative}");
                continue;
            }
            let path = root.join(&relative);
            match fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    eprintln!(
                        "warning: superseded radar file {} was left on disk: {error}",
                        path.display()
                    );
                    continue;
                }
            }
            // Empty generation, ppi and loop directories go with their last file.
            let mut parent = path.parent();
            while let Some(directory) = parent {
                if directory == radar || !directory.starts_with(&radar) {
                    break;
                }
                if fs::remove_dir(directory).is_err() {
                    break;
                }
                parent = directory.parent();
            }
        }
        self.retired = waiting;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "simradar-manifest-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn committed_volume(root: &Path, site: &str, generation: &str) -> Volume {
        let dir = root.join("radar/d01").join(site).join(generation);
        fs::create_dir_all(&dir).unwrap();
        let data = dir.join("volume.ar2v");
        let image = dir.join("reflectivity.png");
        fs::write(&data, b"opaque volume bytes").unwrap();
        fs::write(&image, b"opaque image bytes").unwrap();
        Volume {
            domain: "d01".into(),
            generation: generation.into(),
            expected_sites: vec!["X001".into(), "X002".into()],
            implementation: serde_json::json!({"native_source_sha256":"test-source"}),
            site: Site {
                id: site.into(),
                band: None,
                latitude_deg: 35.0,
                longitude_deg: -98.0,
                antenna_height_msl_m: 100.0,
            },
            valid_time: "2026-10-02T00:00:00+00:00".into(),
            scan_start: "2026-10-02T00:00:00+00:00".into(),
            scan_end: "2026-10-02T00:00:00+00:00".into(),
            timing_requested: "history".into(),
            timing_used: "history".into(),
            source_times: vec!["2026-10-02T00:00:00+00:00".into()],
            source_history: vec!["history#0:sha256=test".into()],
            tilts_deg: vec![0.5],
            fields: vec!["reflectivity".into()],
            config_sha256: "test-config".into(),
            reflectivity_source: "test".into(),
            dual_pol_status: None,
            writer_reports: BTreeMap::new(),
            files: vec![artifact(root, &data, "level2").unwrap()],
            images: vec![Image {
                artifact: artifact(root, &image, "png").unwrap(),
                field: "reflectivity".into(),
                tilt_index: 0,
                sweep_index: 0,
                elevation_deg: 0.5,
            }],
            elapsed_seconds: 0.0,
        }
    }

    #[test]
    fn partial_generation_survives_reload_and_repairs_same_size_corruption() {
        let fixture = Fixture::new();
        let root = &fixture.0;
        let mut manifest = Manifest::load(root, "test-extraction").unwrap();
        let first = committed_volume(root, "X001", "generation-a");
        let valid = first.valid_time.clone();
        let image_path = root.join(&first.images[0].artifact.path);
        manifest.upsert(first);
        manifest.save(root).unwrap();
        let mut manifest = Manifest::load(root, "test-extraction").unwrap();
        assert!(manifest.site_complete(root, "d01", &valid, "generation-a", "X001"));
        assert!(!manifest.complete_generation(root, "d01", &valid, "generation-a"));
        manifest.upsert(committed_volume(root, "X002", "generation-a"));
        manifest.save(root).unwrap();
        assert!(manifest.complete_generation(root, "d01", &valid, "generation-a"));

        let original = fs::read(&image_path).unwrap();
        let mut corrupt = original.clone();
        corrupt[0] ^= 1;
        fs::write(&image_path, &corrupt).unwrap();
        assert!(!manifest.complete_generation(root, "d01", &valid, "generation-a"));
        assert!(!manifest.site_complete(root, "d01", &valid, "generation-a", "X001"));
        assert!(manifest.site_complete(root, "d01", &valid, "generation-a", "X002"));
        fs::write(&image_path, &original).unwrap();
        assert!(manifest.complete_generation(root, "d01", &valid, "generation-a"));
    }

    #[test]
    fn generations_cannot_complete_with_a_mix_of_old_and_new_sites() {
        let fixture = Fixture::new();
        let root = &fixture.0;
        let mut manifest = Manifest::load(root, "test-extraction").unwrap();
        let first = committed_volume(root, "X001", "generation-a");
        let valid = first.valid_time.clone();
        manifest.upsert(first);
        manifest.upsert(committed_volume(root, "X002", "generation-a"));
        manifest.upsert(committed_volume(root, "X001", "generation-b"));
        assert_eq!(manifest.volumes.len(), 2);
        assert!(!manifest.complete_generation(root, "d01", &valid, "generation-a"));
        assert!(!manifest.complete_generation(root, "d01", &valid, "generation-b"));
        manifest.upsert(committed_volume(root, "X002", "generation-b"));
        assert!(manifest.complete_generation(root, "d01", &valid, "generation-b"));
    }

    #[test]
    fn a_replaced_generation_is_deleted_once_the_saved_manifest_stops_naming_it() {
        let fixture = Fixture::new();
        let root = &fixture.0;
        let mut manifest = Manifest::load(root, "test-extraction").unwrap();
        let first = committed_volume(root, "X001", "generation-a");
        let old_paths: Vec<_> = first
            .files
            .iter()
            .chain(first.images.iter().map(|i| &i.artifact))
            .map(|a| root.join(&a.path))
            .collect();
        manifest.upsert(first);
        manifest.save(root).unwrap();
        assert!(old_paths.iter().all(|p| p.is_file()));
        manifest.upsert(committed_volume(root, "X001", "generation-b"));
        assert!(
            old_paths.iter().all(|p| p.is_file()),
            "files stay until a saved manifest stops naming them"
        );
        manifest.save(root).unwrap();
        assert!(old_paths.iter().all(|p| !p.exists()));
        assert!(!root.join("radar/d01/X001/generation-a").exists());
        assert!(root.join("radar/d01/X001/generation-b/volume.ar2v").is_file());
        assert!(manifest.retired.is_empty());

        manifest.retire_where(|v| v.domain == "d01");
        manifest.save(root).unwrap();
        assert!(manifest.volumes.is_empty());
        assert!(!root.join("radar/d01/X001/generation-b").exists());
        assert!(root.join("radar/manifest.json").is_file());
    }

    #[test]
    fn empty_artifact_inventory_is_not_a_completed_site() {
        let fixture = Fixture::new();
        let root = &fixture.0;
        let mut manifest = Manifest::load(root, "test-extraction").unwrap();
        let mut volume = committed_volume(root, "X001", "generation-a");
        let valid = volume.valid_time.clone();
        volume.files.clear();
        volume.images.clear();
        volume.expected_sites = vec!["X001".into()];
        manifest.upsert(volume);
        assert!(!manifest.site_complete(root, "d01", &valid, "generation-a", "X001"));
        assert!(!manifest.complete_generation(root, "d01", &valid, "generation-a"));
    }

    #[test]
    fn replay_does_not_accept_artifact_paths_that_leave_the_run_root() {
        let fixture = Fixture::new();
        let root = fixture.0.join("run");
        fs::create_dir(&root).unwrap();
        let outside = fixture.0.join("outside.ar2v");
        fs::write(&outside, b"outside bytes").unwrap();
        let mut volume = committed_volume(&root, "X001", "generation-a");
        let valid = volume.valid_time.clone();
        let (sha256, bytes) = file_hash(&outside).unwrap();
        volume.files = vec![Artifact {
            path: "../outside.ar2v".into(),
            format: "level2".into(),
            sha256,
            bytes,
        }];
        let mut manifest = Manifest::load(&root, "test-extraction").unwrap();
        manifest.upsert(volume);
        assert!(!manifest.site_complete(&root, "d01", &valid, "generation-a", "X001"));
    }
}
