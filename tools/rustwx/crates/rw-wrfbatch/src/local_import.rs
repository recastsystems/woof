//! Local WRF / NetCDF import, updated from BowEcho v0.30.5's hardened model
//! ingest path. This is the reusable Rusty Weather owner of local model-file
//! processing; BowEcho can consume the resulting store through `rw-ui`.
//!
//! `netcrust` provides the 2D metadata (variable list, dims, units, global
//! attrs) for every file; for raw wrfout the 2D data PLANES and the isobaric
//! sounding volumes are decoded through `wrf-core`'s single-timestep reader
//! (netcrust's `hdf5-reader` path burns ~10 s + ~8M minor page faults per
//! 800×800 plane on compressed 250 m wrfouts: allocation churn, see
//! docs/wrf-import-large-grids.md, while wrf-core reads the same slice in
//! tens of ms). Plain NetCDF and post-processed climate files stay entirely
//! on netcrust. Every source time record is mapped from WRF `Times` or a CF
//! time coordinate to an exact run timeline. Whole-hour runs with an explicit
//! model reference retain the legacy v1 forecast-hour layout byte-for-byte;
//! sub-hourly records or a missing authoritative reference switch the complete
//! run to v2 ordinal storage slots carrying exact lead/valid times.
#![allow(dead_code)]
// Compatibility note: `push_direct` threads the netcrust handle + grid + selector
// as separate args, and `try_postprocessed_wrf` returns the nested field/volume
// tuple the store writer consumes. Both are the upstream API shape.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use netcrust::{File as NcFile, NcSliceInfo, NcSliceInfoElem, Variable as NcVariable};
use rustwx_core::{
    CanonicalField, FieldSelector, GridProjection, GridShape, LatLonGrid, SelectedField2D,
};
use rw_store::grid::GridFile;
use rw_store::reader::HourReader;
use rw_store::run::{RwsRunManifest, validate_store_component};
use rw_store::{
    DerivedFieldInput, RunLock, WrittenHour, write_hour_from_fields_with_derived,
    write_hour_from_fields_with_derived_exact,
};
use sha2::{Digest, Sha256};
use wrf_core::WrfFile;

use crate::wrf_volumes::{
    ColumnHeights, IsoVolume, SurfaceFallback, build_iso_volumes, preflight_iso_volume_shape,
    try_interpolate_iso_volumes,
};
use rw_isobaric::InterfaceStencil;

const LOCAL_IMPORT_MAX_SCAN_DEPTH: usize = 8;
const LOCAL_IMPORT_MAX_DISCOVERED_FILES: usize = 10_000;
/// The store key is u16 and one run must remain practical to browse. Enforce
/// this before hostile time coordinates can drive unbounded allocations.
const MAX_RUN_TIMESTEPS: usize = (u16::MAX as usize) + 1;
const MAX_TIME_LABEL_WIDTH: usize = 256;
const MAX_TIME_LABEL_ELEMENTS: usize = 4 * 1024 * 1024;
/// `Times` has whole-second precision. Permit enough error for an `XTIME`
/// minute value stored as f32 to round back to that second, while remaining
/// comfortably below the half-second boundary where the result is ambiguous.
const WRF_XTIME_SECOND_ROUNDING_TOLERANCE: f64 = 0.25;
/// Two independently rounded `XTIME` records can carry opposite-sign errors.
/// Their floating-point origins may therefore differ by twice the per-record
/// tolerance, but their integral-second origins must still match exactly.
const WRF_XTIME_ORIGIN_AGREEMENT_TOLERANCE: f64 = 2.0 * WRF_XTIME_SECOND_ROUNDING_TOLERANCE;
const SOURCE_ID_READ_BUFFER_BYTES: usize = 1_024 * 1_024;
/// Bump whenever a scientific formula, unit normalization, grid convention,
/// or field-selection meaning changes. It is embedded in every imported run
/// identity so old and new science can never be mixed under one directory.
pub(crate) const IMPORT_SCIENCE_SCHEMA_VERSION: &str = "science_v3_vpd_2m";
const STAGING_DIR_NAME: &str = ".rw-staging";
const STAGING_WORK_DIR_NAME: &str = "work";
const STAGING_BACKUP_DIR_NAME: &str = "previous-run";
const PUBLISH_JOURNAL_SCHEMA: &str = "rw-run-publish.v1";
const MAX_PUBLISH_JOURNAL_BYTES: u64 = 16 * 1024;
const MAX_STAGING_RECOVERY_ENTRIES: usize = 256;
const PUBLISH_LOCK_TIMEOUT: Duration = Duration::from_secs(60);
static STAGING_TRANSACTION_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum PublishPhase {
    Prepared,
    BackupMoved,
    FinalInstalled,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct PublishJournal {
    schema: String,
    model: String,
    run: String,
    phase: PublishPhase,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublishRecoveryAction {
    KeepFinal,
    RestoreBackup,
    RollbackInstalled,
    RemoveAbandoned,
}

/// Same-filesystem run transaction. Writers receive [`Self::staging_store_root`]
/// and use the FINAL model/run identity beneath a unique hidden transaction,
/// so hour metadata is already truthful. Publication moves the whole complete
/// run directory into place; an existing run is first moved to a transaction
/// backup and restored if the second rename or its pre-commit durability step
/// fails. `FinalInstalled` is the commit point; later cleanup is retryable and
/// cannot turn an installed run into a reported publication failure. Immutable,
/// synced phase journals let the next publisher for the same target reconcile a
/// process death at either rename boundary while holding the same per-run lock.
pub(crate) struct RunStagingPublisher {
    store_root: PathBuf,
    staging_root: PathBuf,
    transaction_root: PathBuf,
    staging_store_root: PathBuf,
    staged_run_dir: PathBuf,
    final_run_dir: PathBuf,
    backup_run_dir: PathBuf,
    model: String,
    run: String,
    publish_lock: Option<RunLock>,
    backup_active: bool,
    published: bool,
    cleanup_complete: bool,
}

impl RunStagingPublisher {
    pub(crate) fn new(store_root: &Path, model: &str, run: &str) -> Result<Self, String> {
        validate_store_component("import model", model).map_err(|err| err.to_string())?;
        validate_store_component("import run", run).map_err(|err| err.to_string())?;
        if !has_exact_science_schema_token(run) {
            return Err(format!(
                "import run '{run}' does not include required science schema {IMPORT_SCIENCE_SCHEMA_VERSION}"
            ));
        }

        std::fs::create_dir_all(store_root)
            .map_err(|err| format!("create store root {}: {err}", store_root.display()))?;
        let store_root = std::fs::canonicalize(store_root)
            .map_err(|err| format!("resolve store root {}: {err}", store_root.display()))?;
        let staging_root = store_root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root)
            .map_err(|err| format!("create staging root {}: {err}", staging_root.display()))?;
        let staging_root = checked_real_directory(&store_root, &staging_root, "staging root")?;

        let transaction_root = create_unique_transaction_dir(&staging_root)?;
        let staging_store_root = transaction_root.join(STAGING_WORK_DIR_NAME);
        if let Err(err) = std::fs::create_dir(&staging_store_root) {
            let _ = safe_remove_tree(&staging_root, &transaction_root);
            return Err(format!(
                "create staging work directory {}: {err}",
                staging_store_root.display()
            ));
        }
        let staging_store_root = match checked_real_directory(
            &transaction_root,
            &staging_store_root,
            "staging work directory",
        ) {
            Ok(path) => path,
            Err(err) => {
                let _ = safe_remove_tree(&staging_root, &transaction_root);
                return Err(err);
            }
        };
        let staged_run_dir = staging_store_root.join(model).join(run);
        let final_run_dir = store_root.join(model).join(run);
        let backup_run_dir = transaction_root.join(STAGING_BACKUP_DIR_NAME);
        Ok(Self {
            store_root,
            staging_root,
            transaction_root,
            staging_store_root,
            staged_run_dir,
            final_run_dir,
            backup_run_dir,
            model: model.to_string(),
            run: run.to_string(),
            publish_lock: None,
            backup_active: false,
            published: false,
            cleanup_complete: false,
        })
    }

    pub(crate) fn staging_store_root(&self) -> &Path {
        &self.staging_store_root
    }

    /// Validate every persisted identity and file relationship, then publish
    /// the run as one directory transaction.
    pub(crate) fn publish(mut self) -> Result<PathBuf, String> {
        self.validate_staged_run()?;
        self.publish_prevalidated_with(|source, destination| std::fs::rename(source, destination))?;
        Ok(self.final_run_dir.clone())
    }

    fn validate_staged_run(&self) -> Result<(), String> {
        let staged_run =
            checked_real_directory(&self.transaction_root, &self.staged_run_dir, "staged run")?;
        let manifest_path = checked_existing_descendant(
            &staged_run,
            &staged_run.join("run.json"),
            "staged run manifest",
        )?;

        // The staging tree is unique, but writers deliberately use the final
        // identity so embedded hour metadata needs no binary rewrite. Rewrite
        // the manifest explicitly at the publish boundary and then reload it
        // strictly; this also future-proofs callers that construct it manually.
        let mut manifest = RwsRunManifest::load_bounded(&manifest_path)
            .map_err(|err| format!("load staged manifest: {err}"))?;
        manifest.model = self.model.clone();
        manifest.run = self.run.clone();
        manifest
            .save(&manifest_path)
            .map_err(|err| format!("finalize staged manifest identity: {err}"))?;
        let manifest = RwsRunManifest::load_for_run(&manifest_path, &self.model, &self.run)
            .map_err(|err| format!("validate staged manifest: {err}"))?;
        if manifest.hours.is_empty() {
            return Err("staged run contains no timesteps".to_string());
        }

        let grid_path =
            checked_existing_descendant(&staged_run, &staged_run.join("grid.rwg"), "staged grid")?;
        let grid = GridFile::open(&grid_path)
            .map_err(|err| format!("open staged grid {}: {err}", grid_path.display()))?;
        manifest
            .validate_grid(&grid.hash, grid.nx, grid.ny)
            .map_err(|err| format!("staged grid does not match manifest: {err}"))?;

        for (&hour, entry) in &manifest.hours {
            let hour_path = checked_existing_descendant(
                &staged_run,
                &staged_run.join(&entry.file),
                &format!("staged hour F{hour:03}"),
            )?;
            let reader = HourReader::open(&hour_path).map_err(|err| {
                format!("open staged hour F{hour:03} {}: {err}", hour_path.display())
            })?;
            manifest
                .validate_hour_meta(hour, reader.meta())
                .map_err(|err| format!("staged storage slot {hour} metadata mismatch: {err}"))?;
        }
        let staged_lock = staged_run.join(rw_store::LOCK_FILE_NAME);
        if staged_lock.exists() {
            let metadata = std::fs::symlink_metadata(&staged_lock).map_err(|err| {
                format!(
                    "inspect staged advisory lock {}: {err}",
                    staged_lock.display()
                )
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(format!(
                    "staged advisory lock {} is not a regular file",
                    staged_lock.display()
                ));
            }
            let staged_lock =
                checked_existing_descendant(&staged_run, &staged_lock, "staged advisory lock")?;
            std::fs::remove_file(&staged_lock).map_err(|err| {
                format!(
                    "remove staged advisory lock {}: {err}",
                    staged_lock.display()
                )
            })?;
        }
        Ok(())
    }

    fn write_publish_phase(&self, phase: PublishPhase) -> Result<(), String> {
        write_publish_journal(&self.transaction_root, &self.model, &self.run, phase)
    }

    fn recover_interrupted_publications(&self) -> Result<(), String> {
        recover_publish_transactions_for_run(
            &self.store_root,
            &self.staging_root,
            &self.transaction_root,
            &self.model,
            &self.run,
        )
    }

    fn publish_prevalidated_with<F>(&mut self, move_staged: F) -> Result<(), String>
    where
        F: FnOnce(&Path, &Path) -> std::io::Result<()>,
    {
        let final_model_dir = self
            .final_run_dir
            .parent()
            .ok_or_else(|| "final run has no model directory".to_string())?;
        std::fs::create_dir_all(final_model_dir).map_err(|err| {
            format!(
                "create final model directory {}: {err}",
                final_model_dir.display()
            )
        })?;
        let final_model_dir =
            checked_real_directory(&self.store_root, final_model_dir, "final model directory")?;
        checked_destination_location(
            &self.store_root,
            &self.final_run_dir,
            "final run destination",
        )?;
        checked_destination(
            &self.transaction_root,
            &self.backup_run_dir,
            "run backup destination",
        )?;

        // Serialize publishers targeting the same run. All new import paths
        // honor this hidden lock; the run itself is never locked while being
        // renamed, which avoids Windows' open-handle directory-rename trap.
        let locks_root = self.staging_root.join("publish-locks");
        std::fs::create_dir_all(&locks_root)
            .map_err(|err| format!("create publish locks root {}: {err}", locks_root.display()))?;
        let locks_root =
            checked_real_directory(&self.staging_root, &locks_root, "publish locks root")?;
        let lock_dir = publish_lock_dir(&locks_root, &self.model, &self.run);
        std::fs::create_dir(&lock_dir)
            .or_else(|err| {
                if err.kind() == std::io::ErrorKind::AlreadyExists {
                    Ok(())
                } else {
                    Err(err)
                }
            })
            .map_err(|err| {
                format!(
                    "create publish lock directory {}: {err}",
                    lock_dir.display()
                )
            })?;
        let lock_dir = checked_real_directory(&locks_root, &lock_dir, "publish lock")?;
        self.publish_lock = Some(RunLock::acquire(&lock_dir, PUBLISH_LOCK_TIMEOUT).map_err(
            |err| {
                format!(
                    "acquire publish lock for {}/{}: {err}",
                    self.model, self.run
                )
            },
        )?);

        // Recovery is serialized by the same target-specific lock as the
        // live rename sequence. Only after older journals are reconciled do
        // we durably announce this transaction's intent.
        self.recover_interrupted_publications()?;
        self.write_publish_phase(PublishPhase::Prepared)?;

        if self.final_run_dir.exists() {
            let metadata = std::fs::symlink_metadata(&self.final_run_dir).map_err(|err| {
                format!(
                    "inspect existing run {}: {err}",
                    self.final_run_dir.display()
                )
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(format!(
                    "existing final run {} is not a real directory",
                    self.final_run_dir.display()
                ));
            }
            checked_rename(
                &self.store_root,
                &self.final_run_dir,
                &self.transaction_root,
                &self.backup_run_dir,
                "backup existing run",
            )?;
            self.backup_active = true;
            sync_directory(&final_model_dir)
                .map_err(|err| format!("sync final model directory after backup: {err}"))?;
            sync_directory(&self.transaction_root)
                .map_err(|err| format!("sync transaction directory after backup: {err}"))?;
            self.write_publish_phase(PublishPhase::BackupMoved)?;
        }

        let staged_run = checked_real_directory(
            &self.transaction_root,
            &self.staged_run_dir,
            "staged run before publish",
        )?;
        let staged_model_dir = staged_run
            .parent()
            .ok_or_else(|| "staged run has no model directory".to_string())?
            .to_path_buf();
        let publish_error = match move_staged(&staged_run, &self.final_run_dir) {
            Err(err) => Some(err.to_string()),
            Ok(()) if staged_run.exists() || !self.final_run_dir.is_dir() => Some(
                "publish rename returned success without moving the staged directory".to_string(),
            ),
            Ok(()) => {
                checked_real_directory(&self.store_root, &self.final_run_dir, "published final run")
                    .err()
            }
        };
        if let Some(publish_error) = publish_error {
            let had_backup = self.backup_active;
            let rollback = self.rollback_backup();
            if rollback.is_ok() {
                self.publish_lock.take();
            }
            return match rollback {
                Ok(()) if had_backup => Err(format!(
                    "publish staged run to {} failed: {publish_error}; previous run restored",
                    self.final_run_dir.display()
                )),
                Ok(()) => Err(format!(
                    "publish staged run to {} failed: {publish_error}; final path remains absent as before",
                    self.final_run_dir.display()
                )),
                Err(rollback_error) => Err(format!(
                    "publish staged run to {} failed: {publish_error}; rollback also failed: {rollback_error}; backup preserved at {}",
                    self.final_run_dir.display(),
                    self.backup_run_dir.display()
                )),
            };
        }

        // The staged -> final rename is not the commit point. Both directories
        // directly changed by the rename must be durable before the immutable
        // FinalInstalled journal makes the new run authoritative. Any failure
        // before that marker moves the new run back to staging and restores the
        // previous backup (or leaves the final path absent when there was none).
        if let Err(commit_error) = self.persist_final_install(&final_model_dir, &staged_model_dir) {
            let had_backup = self.backup_active;
            let rollback = self.rollback_uncommitted_install();
            if rollback.is_ok() {
                self.publish_lock.take();
            }
            return match rollback {
                Ok(()) if had_backup => Err(format!(
                    "publish staged run to {} was not durably committed: {commit_error}; new run returned to staging and previous run restored",
                    self.final_run_dir.display()
                )),
                Ok(()) => Err(format!(
                    "publish staged run to {} was not durably committed: {commit_error}; new run returned to staging and final path remains absent as before",
                    self.final_run_dir.display()
                )),
                Err(rollback_error) => Err(format!(
                    "publish staged run to {} was not durably committed: {commit_error}; rollback also failed: {rollback_error}; recovery transaction preserved at {}",
                    self.final_run_dir.display(),
                    self.transaction_root.display()
                )),
            };
        }
        self.published = true;

        // FinalInstalled is durable: from here on, returning an error while the
        // new final remains installed would lie to the caller. Cleanup is safe
        // to retry from Drop or the next target-locked recovery pass, so report
        // it and leave the journal/backup in place when a best-effort step fails.
        self.finish_committed_cleanup();
        self.publish_lock.take();
        debug_assert!(self.final_run_dir.starts_with(final_model_dir));
        Ok(())
    }

    fn persist_final_install(
        &self,
        final_model_dir: &Path,
        staged_model_dir: &Path,
    ) -> Result<(), String> {
        sync_directory(final_model_dir)
            .map_err(|err| format!("sync final model directory after publish: {err}"))?;
        sync_directory(staged_model_dir)
            .map_err(|err| format!("sync staged model directory after publish: {err}"))?;
        self.write_publish_phase(PublishPhase::FinalInstalled)
    }

    fn rollback_uncommitted_install(&mut self) -> Result<(), String> {
        if !self.final_run_dir.exists() {
            return Err(format!(
                "cannot roll back uncommitted install because final path {} is absent",
                self.final_run_dir.display()
            ));
        }
        if self.staged_run_dir.exists() {
            return Err(format!(
                "cannot roll back uncommitted install because staged path {} already exists",
                self.staged_run_dir.display()
            ));
        }
        let final_model_dir = self
            .final_run_dir
            .parent()
            .ok_or_else(|| "uncommitted final run has no model directory".to_string())?
            .to_path_buf();
        let staged_model_dir = self
            .staged_run_dir
            .parent()
            .ok_or_else(|| "uncommitted staged run has no model directory".to_string())?
            .to_path_buf();
        checked_rename(
            &self.store_root,
            &self.final_run_dir,
            &self.transaction_root,
            &self.staged_run_dir,
            "return uncommitted final run to staging",
        )?;
        sync_directory(&final_model_dir)
            .map_err(|err| format!("sync final model directory after commit rollback: {err}"))?;
        sync_directory(&staged_model_dir)
            .map_err(|err| format!("sync staged model directory after commit rollback: {err}"))?;
        self.rollback_backup()
    }

    fn finish_committed_cleanup(&mut self) {
        if self.backup_active {
            match safe_remove_tree(&self.transaction_root, &self.backup_run_dir) {
                Ok(()) => {
                    self.backup_active = false;
                    if let Err(err) = sync_directory(&self.transaction_root) {
                        eprintln!(
                            "published run {}/{}; transaction sync after backup cleanup deferred: {err}",
                            self.model, self.run
                        );
                    }
                }
                Err(err) => eprintln!(
                    "published run {}/{}; backup cleanup deferred in {}: {err}",
                    self.model,
                    self.run,
                    self.transaction_root.display()
                ),
            }
        }
        if !self.backup_active {
            match safe_remove_tree(&self.staging_root, &self.transaction_root) {
                Ok(()) => {
                    self.cleanup_complete = true;
                    if let Err(err) = sync_directory(&self.staging_root) {
                        eprintln!(
                            "published run {}/{}; staging-root sync after cleanup deferred: {err}",
                            self.model, self.run
                        );
                    }
                }
                Err(err) => eprintln!(
                    "published run {}/{}; transaction cleanup deferred at {}: {err}",
                    self.model,
                    self.run,
                    self.transaction_root.display()
                ),
            }
        }
    }

    fn rollback_backup(&mut self) -> Result<(), String> {
        if !self.backup_active {
            return Ok(());
        }
        if self.final_run_dir.exists() {
            return Err(format!(
                "cannot restore backup because final path {} already exists",
                self.final_run_dir.display()
            ));
        }
        checked_rename(
            &self.transaction_root,
            &self.backup_run_dir,
            &self.store_root,
            &self.final_run_dir,
            "restore previous run",
        )?;
        self.backup_active = false;
        sync_directory(
            self.final_run_dir
                .parent()
                .ok_or_else(|| "restored final run has no parent".to_string())?,
        )
        .map_err(|err| format!("sync restored final run parent: {err}"))?;
        sync_directory(&self.transaction_root)
            .map_err(|err| format!("sync transaction after rollback: {err}"))?;
        Ok(())
    }
}

impl Drop for RunStagingPublisher {
    fn drop(&mut self) {
        if self.cleanup_complete {
            return;
        }
        if self.backup_active {
            if !self.final_run_dir.exists() {
                if let Err(err) = self.rollback_backup() {
                    eprintln!(
                        "run publisher could not restore {} on drop: {err}; preserving {}",
                        self.final_run_dir.display(),
                        self.backup_run_dir.display()
                    );
                }
            } else if self.published {
                if safe_remove_tree(&self.transaction_root, &self.backup_run_dir).is_ok() {
                    self.backup_active = false;
                }
            } else {
                // The new final was installed but FinalInstalled never became
                // durable. Keep both it and the prior backup for the next
                // target-locked recovery pass; deleting either here would turn
                // an interrupted pre-commit publish into an implicit commit.
                return;
            }
        }
        if !self.backup_active {
            if !self.published && self.final_run_dir.exists() && !self.staged_run_dir.exists() {
                // Same pre-commit window for a target that had no prior run.
                // Its Prepared journal lets recovery move the new final back
                // under the transaction and restore the original absence.
                return;
            }
            let _ = safe_remove_tree(&self.staging_root, &self.transaction_root);
            self.cleanup_complete = true;
        }
    }
}

fn has_exact_science_schema_token(run: &str) -> bool {
    run.match_indices(IMPORT_SCIENCE_SCHEMA_VERSION)
        .any(|(start, _)| {
            let end = start + IMPORT_SCIENCE_SCHEMA_VERSION.len();
            (start == 0 || run.as_bytes()[start - 1] == b'_')
                && (end == run.len() || run.as_bytes()[end] == b'_')
        })
}

fn create_unique_transaction_dir(staging_root: &Path) -> Result<PathBuf, String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    for _ in 0..128 {
        let counter = STAGING_TRANSACTION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("txn-{}-{nanos:032x}-{counter:016x}", std::process::id());
        validate_store_component("staging transaction", &name).map_err(|err| err.to_string())?;
        let path = staging_root.join(name);
        match std::fs::create_dir(&path) {
            Ok(()) => match checked_real_directory(staging_root, &path, "transaction") {
                Ok(path) => return Ok(path),
                Err(err) => {
                    let _ = safe_remove_tree(staging_root, &path);
                    return Err(err);
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => {
                return Err(format!(
                    "create staging transaction {}: {err}",
                    path.display()
                ));
            }
        }
    }
    Err("could not allocate a unique staging transaction directory".to_string())
}

fn publish_lock_dir(locks_root: &Path, model: &str, run: &str) -> PathBuf {
    locks_root.join(format!("run-{}", publish_target_key(model, run)))
}

fn publish_target_key(model: &str, run: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"rw-run-publish-lock-v1\0");
    hasher.update(model.as_bytes());
    hasher.update([0u8]);
    hasher.update(run.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    sha256_hex(&digest)[..24].to_string()
}

fn publish_phase_name(phase: PublishPhase) -> &'static str {
    match phase {
        PublishPhase::Prepared => "prepared",
        PublishPhase::BackupMoved => "backup-moved",
        PublishPhase::FinalInstalled => "final-installed",
    }
}

fn publish_journal_path(
    transaction_root: &Path,
    model: &str,
    run: &str,
    phase: PublishPhase,
) -> PathBuf {
    transaction_root.join(format!(
        "publish-{}-{}.json",
        publish_target_key(model, run),
        publish_phase_name(phase)
    ))
}

fn write_publish_journal(
    transaction_root: &Path,
    model: &str,
    run: &str,
    phase: PublishPhase,
) -> Result<(), String> {
    let journal = PublishJournal {
        schema: PUBLISH_JOURNAL_SCHEMA.to_string(),
        model: model.to_string(),
        run: run.to_string(),
        phase,
    };
    let bytes =
        serde_json::to_vec(&journal).map_err(|err| format!("serialize publish journal: {err}"))?;
    if bytes.len() as u64 > MAX_PUBLISH_JOURNAL_BYTES {
        return Err(format!(
            "publish journal is {} bytes; limit is {MAX_PUBLISH_JOURNAL_BYTES}",
            bytes.len()
        ));
    }

    let destination = publish_journal_path(transaction_root, model, run, phase);
    if destination.exists() {
        let existing = read_publish_journal(&destination, model, run, phase)?;
        if existing == journal {
            return Ok(());
        }
        return Err(format!(
            "publish journal {} already exists with different contents",
            destination.display()
        ));
    }
    checked_destination(transaction_root, &destination, "publish journal")?;
    let temporary = transaction_root.join(format!(
        ".publish-{}-{}.tmp",
        publish_target_key(model, run),
        publish_phase_name(phase),
    ));
    checked_destination(
        transaction_root,
        &temporary,
        "publish journal temporary file",
    )?;

    let write_result = (|| -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|err| format!("create publish journal {}: {err}", temporary.display()))?;
        file.write_all(&bytes)
            .map_err(|err| format!("write publish journal {}: {err}", temporary.display()))?;
        file.flush()
            .map_err(|err| format!("flush publish journal {}: {err}", temporary.display()))?;
        file.sync_all()
            .map_err(|err| format!("sync publish journal {}: {err}", temporary.display()))?;
        std::fs::rename(&temporary, &destination).map_err(|err| {
            format!(
                "install publish journal {} as {}: {err}",
                temporary.display(),
                destination.display()
            )
        })?;
        sync_directory(transaction_root).map_err(|err| {
            format!(
                "sync publish transaction directory {}: {err}",
                transaction_root.display()
            )
        })?;
        Ok(())
    })();
    if write_result.is_err() && temporary.exists() {
        let _ = std::fs::remove_file(&temporary);
    }
    write_result
}

fn read_publish_journal(
    path: &Path,
    model: &str,
    run: &str,
    phase: PublishPhase,
) -> Result<PublishJournal, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|err| format!("inspect publish journal {}: {err}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "publish journal {} is not a regular file",
            path.display()
        ));
    }
    if metadata.len() > MAX_PUBLISH_JOURNAL_BYTES {
        return Err(format!(
            "publish journal {} is {} bytes; limit is {MAX_PUBLISH_JOURNAL_BYTES}",
            path.display(),
            metadata.len()
        ));
    }
    let file = std::fs::File::open(path)
        .map_err(|err| format!("open publish journal {}: {err}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_PUBLISH_JOURNAL_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| format!("read publish journal {}: {err}", path.display()))?;
    if bytes.len() as u64 > MAX_PUBLISH_JOURNAL_BYTES {
        return Err(format!(
            "publish journal {} grew beyond {MAX_PUBLISH_JOURNAL_BYTES} bytes",
            path.display()
        ));
    }
    let journal: PublishJournal = serde_json::from_slice(&bytes)
        .map_err(|err| format!("parse publish journal {}: {err}", path.display()))?;
    if journal.schema != PUBLISH_JOURNAL_SCHEMA
        || journal.model != model
        || journal.run != run
        || journal.phase != phase
    {
        return Err(format!(
            "publish journal {} identity or phase does not match {model}/{run} ({})",
            path.display(),
            publish_phase_name(phase)
        ));
    }
    Ok(journal)
}

#[cfg(windows)]
fn sync_directory(path: &Path) -> std::io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    let result = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
        .and_then(|directory| directory.sync_all());
    match result {
        Ok(()) => Ok(()),
        // Windows filesystems do not uniformly support FlushFileBuffers on
        // directory handles. Journal FILE contents were already sync_all'd;
        // do not make publication unusable when only directory fsync is
        // unavailable on the host volume.
        Err(err)
            if matches!(
                err.kind(),
                std::io::ErrorKind::PermissionDenied
                    | std::io::ErrorKind::InvalidInput
                    | std::io::ErrorKind::Unsupported
            ) =>
        {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

#[cfg(not(windows))]
fn sync_directory(path: &Path) -> std::io::Result<()> {
    std::fs::File::open(path)?.sync_all()
}

fn publish_recovery_action(
    phase: PublishPhase,
    final_exists: bool,
    backup_exists: bool,
    staged_exists: bool,
) -> Result<PublishRecoveryAction, String> {
    match phase {
        PublishPhase::Prepared => match (final_exists, backup_exists, staged_exists) {
            // Death before the first rename: the old final remains authoritative.
            (true, false, true) => Ok(PublishRecoveryAction::KeepFinal),
            // Death after staged -> final but before the durable commit marker.
            // With no old run, move the new final back and restore absence.
            (true, false, false) => Ok(PublishRecoveryAction::RollbackInstalled),
            // A backup proves the old final crossed the first rename boundary.
            (true, true, false) => Ok(PublishRecoveryAction::RollbackInstalled),
            (false, true, _) => Ok(PublishRecoveryAction::RestoreBackup),
            // Prepared is intent, not a commit: never finish an unpublished new run.
            (false, false, _) => Ok(PublishRecoveryAction::RemoveAbandoned),
            (true, true, true) => {
                Err("prepared publish has simultaneous final, backup, and staged runs".to_string())
            }
        },
        PublishPhase::BackupMoved => match (final_exists, backup_exists, staged_exists) {
            // The new final exists, but FinalInstalled was never durable.
            (true, true, false) => Ok(PublishRecoveryAction::RollbackInstalled),
            // Rollback may already have restored the old final and moved the
            // uncommitted new run back under staging before a process death.
            (true, false, true) => Ok(PublishRecoveryAction::KeepFinal),
            (false, true, _) => Ok(PublishRecoveryAction::RestoreBackup),
            state => Err(format!(
                "backup-moved publish has unrecoverable final/backup/staged state {state:?}"
            )),
        },
        PublishPhase::FinalInstalled => match (final_exists, backup_exists, staged_exists) {
            (true, _, _) => Ok(PublishRecoveryAction::KeepFinal),
            // A failed attempt to sync/install FinalInstalled can leave its
            // filename visible before rollback. Staged data is therefore not
            // sufficient evidence of a commit when the final is absent.
            (false, true, _) => Ok(PublishRecoveryAction::RestoreBackup),
            (false, false, true) => Ok(PublishRecoveryAction::RemoveAbandoned),
            (false, false, false) => {
                Err("final-installed publish has no final, backup, or staged run".to_string())
            }
        },
    }
}

fn existing_real_directory(path: &Path, label: &str) -> Result<bool, String> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(format!("inspect {label} {}: {err}", path.display())),
    };
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "{label} {} is not a real directory",
            path.display()
        ));
    }
    Ok(true)
}

fn latest_publish_journal(
    transaction_root: &Path,
    model: &str,
    run: &str,
) -> Result<Option<PublishJournal>, String> {
    for phase in [
        PublishPhase::FinalInstalled,
        PublishPhase::BackupMoved,
        PublishPhase::Prepared,
    ] {
        let path = publish_journal_path(transaction_root, model, run, phase);
        match std::fs::symlink_metadata(&path) {
            Ok(_) => return read_publish_journal(&path, model, run, phase).map(Some),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(format!("inspect publish journal {}: {err}", path.display()));
            }
        }
    }
    Ok(None)
}

fn has_target_publish_journal_temporary(
    transaction_root: &Path,
    model: &str,
    run: &str,
) -> Result<bool, String> {
    for phase in [
        PublishPhase::Prepared,
        PublishPhase::BackupMoved,
        PublishPhase::FinalInstalled,
    ] {
        let path = transaction_root.join(format!(
            ".publish-{}-{}.tmp",
            publish_target_key(model, run),
            publish_phase_name(phase)
        ));
        match std::fs::symlink_metadata(&path) {
            Ok(_) => return Ok(true),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => {
                return Err(format!(
                    "inspect temporary publish journal {}: {err}",
                    path.display()
                ));
            }
        }
    }
    Ok(false)
}

fn recover_publish_transactions_for_run(
    store_root: &Path,
    staging_root: &Path,
    current_transaction: &Path,
    model: &str,
    run: &str,
) -> Result<(), String> {
    let final_model_dir = store_root.join(model);
    std::fs::create_dir_all(&final_model_dir).map_err(|err| {
        format!(
            "create recovery model directory {}: {err}",
            final_model_dir.display()
        )
    })?;
    let final_model_dir =
        checked_real_directory(store_root, &final_model_dir, "recovery model directory")?;
    let final_run_dir = final_model_dir.join(run);
    let entries = std::fs::read_dir(staging_root)
        .map_err(|err| format!("scan staging root {}: {err}", staging_root.display()))?;
    let mut inspected = 0usize;
    for entry in entries {
        inspected += 1;
        if inspected > MAX_STAGING_RECOVERY_ENTRIES {
            return Err(format!(
                "staging recovery found more than {MAX_STAGING_RECOVERY_ENTRIES} entries in {}; refusing an unbounded scan",
                staging_root.display()
            ));
        }
        let entry = entry
            .map_err(|err| format!("read staging entry in {}: {err}", staging_root.display()))?;
        let name = entry
            .file_name()
            .to_str()
            .ok_or_else(|| "staging entry has a non-UTF-8 name".to_string())?
            .to_string();
        if !name.starts_with("txn-") {
            continue;
        }
        let transaction_root = entry.path();
        if transaction_root == current_transaction {
            continue;
        }
        let metadata = match std::fs::symlink_metadata(&transaction_root) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(format!(
                    "inspect recovery transaction {}: {err}",
                    transaction_root.display()
                ));
            }
        };
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(format!(
                "recovery transaction {} is not a real directory",
                transaction_root.display()
            ));
        }
        let transaction_root =
            match checked_real_directory(staging_root, &transaction_root, "recovery transaction") {
                Ok(path) => path,
                Err(_) if !transaction_root.exists() => continue,
                Err(err) => return Err(err),
            };
        let Some(journal) = latest_publish_journal(&transaction_root, model, run)? else {
            // A target-keyed temporary with no installed Prepared journal
            // can only precede every destructive rename. Under the target's
            // publish lock it is safe to discard that interrupted intent.
            if has_target_publish_journal_temporary(&transaction_root, model, run)? {
                safe_remove_tree(staging_root, &transaction_root)?;
                sync_directory(staging_root)
                    .map_err(|err| format!("sync orphan journal cleanup: {err}"))?;
            }
            // Otherwise this is an import still being staged or a different
            // target's transaction and must remain untouched.
            continue;
        };
        let staged_run_dir = transaction_root
            .join(STAGING_WORK_DIR_NAME)
            .join(model)
            .join(run);
        let backup_run_dir = transaction_root.join(STAGING_BACKUP_DIR_NAME);
        let final_exists = existing_real_directory(&final_run_dir, "recovery final run")?;
        let backup_exists = existing_real_directory(&backup_run_dir, "recovery backup run")?;
        let staged_exists = existing_real_directory(&staged_run_dir, "recovery staged run")?;
        let action =
            publish_recovery_action(journal.phase, final_exists, backup_exists, staged_exists)
                .map_err(|reason| {
                    format!(
                        "cannot recover interrupted publication for {model}/{run} in {}: {reason}",
                        transaction_root.display()
                    )
                })?;

        match action {
            PublishRecoveryAction::KeepFinal => {
                validate_persisted_run(
                    store_root,
                    &final_run_dir,
                    model,
                    run,
                    "recovered final run",
                )?;
            }
            PublishRecoveryAction::RestoreBackup => {
                validate_persisted_run(
                    &transaction_root,
                    &backup_run_dir,
                    model,
                    run,
                    "recovery backup run",
                )?;
                checked_rename(
                    &transaction_root,
                    &backup_run_dir,
                    store_root,
                    &final_run_dir,
                    "restore interrupted publish backup",
                )?;
                validate_persisted_run(
                    store_root,
                    &final_run_dir,
                    model,
                    run,
                    "restored previous run",
                )?;
                sync_directory(
                    final_run_dir
                        .parent()
                        .ok_or_else(|| "restored run has no parent".to_string())?,
                )
                .map_err(|err| format!("sync restored run parent: {err}"))?;
            }
            PublishRecoveryAction::RollbackInstalled => {
                // Preserve the uncommitted new run inside the transaction until
                // the old final is safely restored. If any step fails, the
                // journal plus whichever of staged/backup remains lets the next
                // target-locked recovery attempt continue without guessing.
                if backup_exists {
                    validate_persisted_run(
                        &transaction_root,
                        &backup_run_dir,
                        model,
                        run,
                        "pre-commit rollback backup run",
                    )?;
                }
                let staged_parent = staged_run_dir
                    .parent()
                    .ok_or_else(|| "rollback staged run has no parent".to_string())?;
                checked_rename(
                    store_root,
                    &final_run_dir,
                    &transaction_root,
                    &staged_run_dir,
                    "return interrupted uncommitted final to staging",
                )?;
                sync_directory(
                    final_run_dir
                        .parent()
                        .ok_or_else(|| "rollback final run has no parent".to_string())?,
                )
                .map_err(|err| format!("sync final parent during commit rollback: {err}"))?;
                sync_directory(staged_parent)
                    .map_err(|err| format!("sync staged parent during commit rollback: {err}"))?;

                if backup_exists {
                    checked_rename(
                        &transaction_root,
                        &backup_run_dir,
                        store_root,
                        &final_run_dir,
                        "restore pre-commit publish backup",
                    )?;
                    validate_persisted_run(
                        store_root,
                        &final_run_dir,
                        model,
                        run,
                        "pre-commit restored previous run",
                    )?;
                    sync_directory(
                        final_run_dir
                            .parent()
                            .ok_or_else(|| "restored pre-commit run has no parent".to_string())?,
                    )
                    .map_err(|err| format!("sync restored pre-commit run parent: {err}"))?;
                    sync_directory(&transaction_root)
                        .map_err(|err| format!("sync consumed rollback backup: {err}"))?;
                }
            }
            PublishRecoveryAction::RemoveAbandoned => {}
        }

        safe_remove_tree(staging_root, &transaction_root)?;
        sync_directory(staging_root)
            .map_err(|err| format!("sync staging recovery cleanup: {err}"))?;
    }
    Ok(())
}

fn validate_persisted_run(
    containment_root: &Path,
    run_dir: &Path,
    model: &str,
    run: &str,
    label: &str,
) -> Result<(), String> {
    let run_dir = checked_real_directory(containment_root, run_dir, label)?;
    let manifest_path = checked_regular_file_descendant(
        &run_dir,
        &run_dir.join("run.json"),
        &format!("{label} manifest"),
    )?;
    let manifest = RwsRunManifest::load_for_run(&manifest_path, model, run)
        .map_err(|err| format!("validate {label} manifest: {err}"))?;
    if manifest.hours.is_empty() {
        return Err(format!("{label} contains no timesteps"));
    }
    let grid_path = checked_regular_file_descendant(
        &run_dir,
        &run_dir.join("grid.rwg"),
        &format!("{label} grid"),
    )?;
    let grid = GridFile::open(&grid_path)
        .map_err(|err| format!("open {label} grid {}: {err}", grid_path.display()))?;
    manifest
        .validate_grid(&grid.hash, grid.nx, grid.ny)
        .map_err(|err| format!("{label} grid does not match manifest: {err}"))?;
    for (&hour, entry) in &manifest.hours {
        let hour_path = checked_regular_file_descendant(
            &run_dir,
            &run_dir.join(&entry.file),
            &format!("{label} hour F{hour:03}"),
        )?;
        let reader = HourReader::open(&hour_path)
            .map_err(|err| format!("open {label} hour F{hour:03}: {err}"))?;
        manifest
            .validate_hour_meta(hour, reader.meta())
            .map_err(|err| format!("{label} storage slot {hour} metadata mismatch: {err}"))?;
    }
    Ok(())
}

fn checked_existing_descendant(root: &Path, path: &Path, label: &str) -> Result<PathBuf, String> {
    let root = std::fs::canonicalize(root)
        .map_err(|err| format!("resolve containment root {}: {err}", root.display()))?;
    let path = std::fs::canonicalize(path)
        .map_err(|err| format!("resolve {label} {}: {err}", path.display()))?;
    if path == root || !path.starts_with(&root) {
        return Err(format!(
            "{label} {} is not a strict descendant of {}",
            path.display(),
            root.display()
        ));
    }
    Ok(path)
}

fn checked_regular_file_descendant(
    root: &Path,
    path: &Path,
    label: &str,
) -> Result<PathBuf, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|err| format!("inspect {label} {}: {err}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    checked_existing_descendant(root, path, label)
}

fn checked_real_directory(root: &Path, path: &Path, label: &str) -> Result<PathBuf, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|err| format!("inspect {label} {}: {err}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "{label} {} is not a real directory",
            path.display()
        ));
    }
    checked_existing_descendant(root, path, label)
}

fn checked_destination(root: &Path, path: &Path, label: &str) -> Result<(), String> {
    if path.exists() {
        return Err(format!("{label} {} already exists", path.display()));
    }
    checked_destination_location(root, path, label)
}

fn checked_destination_location(root: &Path, path: &Path, label: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{label} {} has no parent", path.display()))?;
    let root = std::fs::canonicalize(root)
        .map_err(|err| format!("resolve {label} root {}: {err}", root.display()))?;
    let parent = std::fs::canonicalize(parent)
        .map_err(|err| format!("resolve {label} parent {}: {err}", parent.display()))?;
    if parent != root && !parent.starts_with(&root) {
        return Err(format!(
            "{label} parent {} is outside {}",
            parent.display(),
            root.display()
        ));
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{label} {} has no UTF-8 filename", path.display()))?;
    validate_store_component(label, name).map_err(|err| err.to_string())
}

fn checked_rename(
    source_root: &Path,
    source: &Path,
    destination_root: &Path,
    destination: &Path,
    label: &str,
) -> Result<(), String> {
    let source = checked_existing_descendant(source_root, source, &format!("{label} source"))?;
    checked_destination(
        destination_root,
        destination,
        &format!("{label} destination"),
    )?;
    std::fs::rename(&source, destination).map_err(|err| {
        format!(
            "{label}: rename {} to {} failed: {err}",
            source.display(),
            destination.display()
        )
    })
}

fn safe_remove_tree(root: &Path, target: &Path) -> Result<(), String> {
    if !target.exists() {
        return Ok(());
    }
    let metadata = std::fs::symlink_metadata(target)
        .map_err(|err| format!("inspect cleanup path {}: {err}", target.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "refusing to recursively remove symlink {}",
            target.display()
        ));
    }
    let target = checked_existing_descendant(root, target, "cleanup path")?;
    if !metadata.is_dir() {
        return Err(format!(
            "cleanup path {} is not a directory",
            target.display()
        ));
    }
    std::fs::remove_dir_all(&target)
        .map_err(|err| format!("remove cleanup tree {}: {err}", target.display()))
}

/// One source-file record with an exact UTC valid time. Every ingest path must
/// settle the complete cross-file axis through [`ForecastHourTimeline`] before
/// it writes anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceTimeRecord {
    pub(crate) time_index: usize,
    pub(crate) valid_unix: i64,
    pub(crate) label: String,
}

/// Exact time metadata discovered in one WRF/NetCDF file. `reference_unix`
/// is a real model initialization time (WRF `START_DATE`) when available; a
/// generic CF `units = "hours since ..."` epoch is intentionally not treated
/// as a forecast reference time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceTimeAxis {
    pub(crate) records: Vec<SourceTimeRecord>,
    pub(crate) reference_unix: Option<i64>,
}

/// A source record after the complete run timeline has been validated. For a
/// legacy v1 run `storage_slot` is the true forecast hour and `exact_time` is
/// `None`. For a v2 run the slot is only a stable ordinal identity; the exact
/// lead and valid time are essential metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedSourceTime {
    pub(crate) time_index: usize,
    pub(crate) storage_slot: u16,
    pub(crate) exact_time: Option<rw_store::RwsExactTime>,
    pub(crate) valid_unix: i64,
    pub(crate) label: String,
}

impl PlannedSourceTime {
    pub(crate) fn display_key(&self) -> String {
        match self.exact_time {
            Some(exact) => format!(
                "+{} · {}",
                format_lead_duration(exact.lead_seconds),
                self.label
            ),
            None => format!("f{:03}", self.storage_slot),
        }
    }
}

/// Complete cross-file timeline plan. Planning is deliberately global: an
/// exact-time run's ordinal slots are assigned only after every valid time is
/// known and sorted, so filename/selection order cannot change store identity.
#[derive(Debug, Clone)]
pub(crate) struct ForecastHourTimeline {
    origin_unix: i64,
    exact_time_axis: bool,
    records: Vec<Vec<PlannedSourceTime>>,
}

impl ForecastHourTimeline {
    /// Resolve the store run key from exact initialization time, stable source
    /// identity, and the processing profile. The profile is part of the key
    /// because rw-store replaces an existing forecast-hour file atomically;
    /// light and full imports of the same source must never erase each other.
    pub(crate) fn run_name(&self, source_identity: &str, processing_profile: &str) -> String {
        let stamp = chrono::DateTime::from_timestamp(self.origin_unix, 0)
            .map(|time| time.format("%Y%m%d%H%M%S").to_string())
            .unwrap_or_else(|| self.origin_unix.to_string());
        format!(
            "local_{stamp}_{source_identity}_{processing_profile}_{IMPORT_SCIENCE_SCHEMA_VERSION}"
        )
    }

    pub(crate) fn is_exact_time_axis(&self) -> bool {
        self.exact_time_axis
    }

    pub(crate) fn records_for_source(&self, index: usize) -> Option<&[PlannedSourceTime]> {
        self.records.get(index).map(Vec::as_slice)
    }

    pub(crate) fn plan_all(sources: &[(PathBuf, SourceTimeAxis)]) -> Result<Self, String> {
        if sources.is_empty() {
            return Err("forecast timeline has no sources".to_string());
        }

        let total_records = sources.iter().try_fold(0_usize, |total, (_, source)| {
            total
                .checked_add(source.records.len())
                .ok_or_else(|| "forecast timeline record count overflowed usize".to_string())
        })?;
        if total_records > MAX_RUN_TIMESTEPS {
            return Err(format!(
                "forecast timeline contains {total_records} records; rw-store supports at most {MAX_RUN_TIMESTEPS} timesteps per run"
            ));
        }

        let mut reference = None::<(PathBuf, i64)>;
        let mut earliest_valid = None::<i64>;
        for (path, source) in sources {
            if source.records.is_empty() {
                return Err(format!("{} has an empty time axis", path.display()));
            }
            for pair in source.records.windows(2) {
                if pair[1].valid_unix <= pair[0].valid_unix {
                    return Err(format!(
                        "{} has a non-increasing or duplicate time axis at records {} ({}) and {} ({})",
                        path.display(),
                        pair[0].time_index,
                        pair[0].label,
                        pair[1].time_index,
                        pair[1].label
                    ));
                }
            }
            if let Some(candidate) = source.reference_unix {
                if let Some((prior_path, prior)) = &reference {
                    if *prior != candidate {
                        return Err(format!(
                            "{} belongs to a different forecast run: reference time {} does not match {} from {}",
                            path.display(),
                            format_valid_unix(candidate),
                            format_valid_unix(*prior),
                            prior_path.display()
                        ));
                    }
                } else {
                    reference = Some((path.clone(), candidate));
                }
            }
            for record in &source.records {
                earliest_valid = Some(
                    earliest_valid
                        .map(|prior| prior.min(record.valid_unix))
                        .unwrap_or(record.valid_unix),
                );
            }
        }
        let has_authoritative_reference = reference.is_some();
        let origin_unix = reference
            .as_ref()
            .map(|(_, value)| *value)
            .or(earliest_valid)
            .ok_or_else(|| "forecast timeline has no valid times".to_string())?;

        #[derive(Debug)]
        struct FlatRecord<'a> {
            source_index: usize,
            record: &'a SourceTimeRecord,
            lead_seconds: u64,
        }
        let mut flat = Vec::<FlatRecord<'_>>::new();
        for (source_index, (path, source)) in sources.iter().enumerate() {
            for record in &source.records {
                let delta = record.valid_unix.checked_sub(origin_unix).ok_or_else(|| {
                    format!(
                        "{} time {} cannot be differenced from run origin {}",
                        path.display(),
                        record.label,
                        format_valid_unix(origin_unix)
                    )
                })?;
                let lead_seconds = u64::try_from(delta).map_err(|_| {
                    format!(
                        "{} time {} precedes run origin {}",
                        path.display(),
                        record.label,
                        format_valid_unix(origin_unix)
                    )
                })?;
                flat.push(FlatRecord {
                    source_index,
                    record,
                    lead_seconds,
                });
            }
        }
        flat.sort_by_key(|item| item.record.valid_unix);
        if let Some(pair) = flat
            .windows(2)
            .find(|pair| pair[0].record.valid_unix == pair[1].record.valid_unix)
        {
            let left_path = &sources[pair[0].source_index].0;
            let right_path = &sources[pair[1].source_index].0;
            return Err(format!(
                "duplicate forecast valid time {} appears in {} record {} and {} record {}; refusing to overwrite a timestep",
                pair[0].record.label,
                left_path.display(),
                pair[0].record.time_index,
                right_path.display(),
                pair[1].record.time_index
            ));
        }

        // Without a real model initialization timestamp, even an apparently
        // whole-hour cadence is not proof that ordinal labels are forecast
        // hours. Long integrations also need v2: their elapsed hours do not
        // fit the legacy u16 slot, but exact metadata can retain their true
        // origin and lead without truncating or relabelling a reanalysis.
        // An origin off the whole hour needs v2 too. WHAT BREAKAGE THIS
        // PREVENTS (gate law): the legacy axis keeps only the cycle HOUR, so
        // the +3 h frame of a forecast from a 10:15 analysis was drawn as
        // "Init 10Z | F003 | Valid 13Z" when it is valid at 13:15 (and its
        // +0 frame as "Valid 10Z"), while its +30 min frame, already on the
        // exact axis, said 10:45 (DA environment panels, 2026-10-29).
        let exact_time_axis = !has_authoritative_reference
            || origin_unix.rem_euclid(3_600) != 0
            || flat.iter().any(|item|
                item.lead_seconds % 3_600 != 0 || item.lead_seconds / 3_600 > u64::from(u16::MAX));
        let mut planned = vec![Vec::<PlannedSourceTime>::new(); sources.len()];
        for (ordinal, item) in flat.into_iter().enumerate() {
            let (storage_slot, exact_time) = if exact_time_axis {
                let slot = u16::try_from(ordinal)
                    .map_err(|_| "exact-time WRF ordinal slot exceeds u16 range".to_string())?;
                (
                    slot,
                    Some(rw_store::RwsExactTime::new(
                        item.lead_seconds,
                        item.record.valid_unix,
                    )),
                )
            } else {
                let hour = item.lead_seconds / 3_600;
                let slot = u16::try_from(hour).map_err(|_| {
                    format!(
                        "time {} is forecast hour {hour}, beyond rw-store's u16 hour range",
                        item.record.label
                    )
                })?;
                (slot, None)
            };
            planned[item.source_index].push(PlannedSourceTime {
                time_index: item.record.time_index,
                storage_slot,
                exact_time,
                valid_unix: item.record.valid_unix,
                label: item.record.label.clone(),
            });
        }
        for records in &mut planned {
            records.sort_by_key(|record| record.time_index);
        }
        Ok(Self {
            origin_unix,
            exact_time_axis,
            records: planned,
        })
    }
}

fn format_lead_duration(seconds: u64) -> String {
    let hours = seconds / 3_600;
    let minutes = (seconds % 3_600) / 60;
    let seconds = seconds % 60;
    format!("{hours:03}:{minutes:02}:{seconds:02}")
}

#[derive(Debug)]
pub struct LocalImportTask {
    pub label: String,
    pub rx: Receiver<LocalImportMessage>,
}

/// Worker → UI messages, same shape as `wrf_process::WrfProcessMessage`: the
/// dock shows the latest `Progress` line while the import runs (on a 250 m
/// grid the light path is legitimately minutes per file, an anonymous
/// spinner reads as a hang), then a single terminal `Done`.
#[derive(Debug)]
pub enum LocalImportMessage {
    Progress(String),
    Done(Result<LocalImportSummary, String>),
}

#[derive(Debug, Clone)]
pub struct LocalImportSummary {
    pub store_root: PathBuf,
    pub model: String,
    pub run: String,
    pub files_seen: usize,
    pub hours_written: usize,
    pub variables: Vec<String>,
    /// Per-file degradations that did not fail the import (e.g. isobaric
    /// sounding volumes unavailable), surfaced in the completion status line.
    pub notes: Vec<String>,
}

struct ImportedWrfFields {
    canonical: Vec<(String, SelectedField2D)>,
    raw_2d: Vec<RawField2D>,
    grid: LatLonGrid,
    projection: Option<GridProjection>,
}

/// One raw 2-D plane under the light-import `wrf_*` store naming.
/// `pub(crate)`: the wrf2d route hands these to BOTH import workers through
/// [`PostprocessedWrfHour`], and `wrf_process` maps them into its derived-
/// field refs itself.
pub(crate) struct RawField2D {
    pub(crate) name: String,
    pub(crate) units: String,
    pub(crate) values: Vec<f32>,
}

pub fn spawn_import_paths(paths: Vec<PathBuf>, store_root: PathBuf) -> LocalImportTask {
    let label = if paths.len() == 1 {
        format!("Import {}", display_name(&paths[0]))
    } else {
        format!("Import {} local files", paths.len())
    };
    let (tx, rx) = channel();
    let worker_tx = tx.clone();
    let spawn_result = std::thread::Builder::new()
        .name("rw-ui-local-import".to_string())
        .spawn(move || {
            let result = crate::wrf_process::isolate_panics("local model import worker", || {
                crate::wrf_process::lower_import_thread_priority();
                let mut progress = |message: String| {
                    let _ = worker_tx.send(LocalImportMessage::Progress(message));
                };
                import_paths(&paths, &store_root, &mut progress).map_err(|err| err.to_string())
            });
            let _ = worker_tx.send(LocalImportMessage::Done(result));
        });
    if let Err(err) = spawn_result {
        let _ = tx.send(LocalImportMessage::Done(Err(format!(
            "could not start local import worker: {err}"
        ))));
    }
    LocalImportTask { label, rx }
}

pub fn supported_files_in_folder(folder: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let mut stack = vec![(folder.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && is_supported_model_file(&path) {
                paths.push(path);
                if paths.len() >= LOCAL_IMPORT_MAX_DISCOVERED_FILES {
                    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
                    return paths;
                }
            } else if depth < LOCAL_IMPORT_MAX_SCAN_DEPTH && path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    paths
}

pub fn is_supported_model_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    name.starts_with("wrfout")
        || matches!(
            path.extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_ascii_lowercase())
                .as_deref(),
            Some("nc" | "nc4" | "cdf")
        )
        // GRIB Edition 1 (.grb/.grib: ERA-20C / GDEX reanalysis); routed to
        // `grib_import` below. GRIB2 extensions stay unsupported here.
        || crate::grib_import::is_grib1_file(path)
}

fn combine_source_fingerprints(mut fingerprints: Vec<[u8; 32]>) -> [u8; 32] {
    fingerprints.sort_unstable();
    let mut hasher = Sha256::new();
    hasher.update(b"rw-source-set-v2\0");
    hasher.update((fingerprints.len() as u64).to_le_bytes());
    for fingerprint in fingerprints {
        hasher.update(fingerprint);
    }
    hasher.finalize().into()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceFileRevision {
    canonical_path: PathBuf,
    len: u64,
    modified: SystemTime,
    created: Option<SystemTime>,
}

#[derive(Debug, Clone)]
pub(crate) struct SourceIdentitySnapshot {
    pub(crate) identity: String,
    files: Vec<(PathBuf, SourceFileRevision)>,
    /// History files the identity may lose (snowfall dependencies).
    history: Vec<PathBuf>,
    /// History files already gone when the identity was taken.
    gone: Vec<PathBuf>,
}

impl SourceIdentitySnapshot {
    /// The history files that were already gone when the identity was taken.
    pub(crate) fn gone(&self) -> &[PathBuf] {
        &self.gone
    }
}

fn inspect_source_file_revision(path: &Path) -> Result<SourceFileRevision, String> {
    let canonical_path = std::fs::canonicalize(path)
        .map_err(|err| format!("cannot resolve source file '{}': {err}", path.display()))?;
    let metadata = std::fs::metadata(&canonical_path).map_err(|err| {
        format!(
            "cannot inspect source file '{}': {err}",
            canonical_path.display()
        )
    })?;
    if !metadata.is_file() {
        return Err(format!(
            "source path '{}' is not a regular file",
            path.display()
        ));
    }
    let modified = metadata.modified().map_err(|err| {
        format!(
            "cannot read source modification time '{}': {err}",
            canonical_path.display()
        )
    })?;
    Ok(SourceFileRevision {
        canonical_path,
        len: metadata.len(),
        modified,
        created: metadata.created().ok(),
    })
}

/// The piece size of a source file's fingerprint.  Part of the identity
/// (it is hashed into it), so changing it renames every store run.
const SOURCE_ID_CHUNK_BYTES: u64 = 64 * 1_024 * 1_024;

/// Read exactly `buffer.len()` bytes at `offset`, without moving a shared
/// file cursor, so every piece of one file is read at once.
fn read_exact_at(file: &std::fs::File, buffer: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    let mut filled = 0usize;
    while filled < buffer.len() {
        #[cfg(unix)]
        let read = std::os::unix::fs::FileExt::read_at(file, &mut buffer[filled..], offset)?;
        #[cfg(windows)]
        let read = std::os::windows::fs::FileExt::seek_read(file, &mut buffer[filled..], offset)?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "file ended before its recorded length",
            ));
        }
        filled += read;
        offset += read as u64;
    }
    Ok(())
}

/// `rw-source-file-v3`: SHA-256 over the name, the length, the piece size
/// and the SHA-256 of each SOURCE_ID_CHUNK_BYTES piece in file order.  The
/// pieces are hashed in parallel; the result depends only on the bytes.
fn chunked_file_digest(file: &std::fs::File, name: &str, length: u64) -> std::io::Result<[u8; 32]> {
    use rayon::prelude::*;
    let pieces = length.div_ceil(SOURCE_ID_CHUNK_BYTES).max(1);
    let digests = (0..pieces)
        .into_par_iter()
        .map(|piece| -> std::io::Result<[u8; 32]> {
            let start = piece * SOURCE_ID_CHUNK_BYTES;
            let end = (start + SOURCE_ID_CHUNK_BYTES).min(length);
            let mut hasher = Sha256::new();
            let mut buffer = vec![0u8; SOURCE_ID_READ_BUFFER_BYTES];
            let mut offset = start;
            while offset < end {
                let take = ((end - offset) as usize).min(buffer.len());
                read_exact_at(file, &mut buffer[..take], offset)?;
                hasher.update(&buffer[..take]);
                offset += take as u64;
            }
            Ok(hasher.finalize().into())
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    let mut hasher = Sha256::new();
    hasher.update(b"rw-source-file-v3\0");
    hasher.update((name.len() as u64).to_le_bytes());
    hasher.update(name.as_bytes());
    hasher.update(length.to_le_bytes());
    hasher.update(SOURCE_ID_CHUNK_BYTES.to_le_bytes());
    for digest in digests {
        hasher.update(digest);
    }
    Ok(hasher.finalize().into())
}

fn source_file_fingerprint(path: &Path) -> Result<([u8; 32], SourceFileRevision), String> {
    let before = inspect_source_file_revision(path)?;
    let file = std::fs::File::open(&before.canonical_path).map_err(|err| {
        format!(
            "cannot open source identity file '{}': {err}",
            path.display()
        )
    })?;
    let length = before.len;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("source path '{}' has no UTF-8 filename", path.display()))?;

    // The same unchanged file hashed by an earlier render (the frame's own
    // live render, every later frame's snowfall dependencies, the windowed
    // pass) is not read again: see `crate::source_memo` for the breakage
    // and for what "unchanged" is proven by.
    let stamp = crate::source_memo::file_stamp(&before.canonical_path);
    if stamp.is_some() {
        if let Some(digest) = crate::source_memo::recall_fingerprint(&before.canonical_path, name) {
            let after = inspect_source_file_revision(path)?;
            if before == after {
                return Ok((digest, after));
            }
        }
    }

    // Every byte of the file still enters the digest, but in fixed
    // SOURCE_ID_CHUNK_BYTES pieces hashed at once across the pool and
    // then hashed in order, so a frame's identity costs its read rather
    // than one core's SHA-256.  THE BREAKAGE: one SHA-256 stream over a
    // 16 GB 1799x1059x50 history frame held each live render on ONE core
    // for ~12 s before anything else could start (an 8x5090 host, 2.9.3).
    // A bounded first/last sample would alias two same-name, same-size
    // files whose interior differs, so the pieces cover the whole file.
    let digest = chunked_file_digest(&file, name, length).map_err(|err| {
        format!("cannot hash source file '{}': {err}", path.display())
    })?;
    let after = inspect_source_file_revision(path)?;
    if before != after {
        return Err(format!(
            "source file '{}' changed while its identity was hashed; import an immutable snapshot",
            path.display()
        ));
    }
    if let Some(stamp) = &stamp {
        crate::source_memo::remember_fingerprint(&before.canonical_path, name, stamp, &digest);
    }
    Ok((digest, after))
}

fn sha256_hex(digest: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest {
        // Writing to String is infallible; deliberately ignore fmt::Result
        // rather than introduce a panic path while formatting an identifier.
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

/// Stable identity for one selected source set. Per-file fingerprints include
/// basename, length, and a SHA-256 digest of every source byte, then are sorted
/// and hashed again. Absolute cache/worktree paths and selection order do not
/// affect the result; any content change does. Light and full imports of the
/// same bytes share this base identity, while [`ForecastHourTimeline::run_name`]
/// adds a profile suffix so their store hours remain isolated.
pub(crate) fn source_set_identity(paths: &[PathBuf]) -> Result<String, String> {
    capture_source_set_identity(paths).map(|snapshot| snapshot.identity)
}

pub(crate) fn capture_source_set_identity(
    paths: &[PathBuf],
) -> Result<SourceIdentitySnapshot, String> {
    capture_source_set_identity_with_history(paths, &[])
}

/// The identity of `required` (every one must be there) and of the history
/// files in `history` that are still there.
///
/// WHAT BREAKAGE THIS PREVENTS: a frame's snowfall windows name every
/// earlier history file of its domain, and a host that removes old history
/// to save disk (an 8x5090 host, 2026-10-21: an operator script deleting each frame
/// once its own pictures were out) failed EVERY later frame whole: each
/// render exited 1 with no picture ("cannot resolve source file
/// ...wrfout_d01_2026-10-21_00_00_00").  A history file that is gone before
/// it is read leaves the identity, and one that goes during the import is
/// let go at the end; the snowfall windows that needed it say by name that
/// they are not drawn, and every other product of the frame is.  A history
/// file that is still there but CHANGED is refused as before.
pub(crate) fn capture_source_set_identity_with_history(
    required: &[PathBuf],
    history: &[PathBuf],
) -> Result<SourceIdentitySnapshot, String> {
    if required.is_empty() && history.is_empty() {
        return Err("cannot identify an empty source set".to_string());
    }
    let history: Vec<PathBuf> = history
        .iter()
        .filter(|path| !required.contains(path))
        .cloned()
        .collect();
    // Every file streams through its own SHA-256 at once; the first
    // failure in path order is the one reported.
    use rayon::prelude::*;
    let paths: Vec<&PathBuf> = required.iter().chain(history.iter()).collect();
    let hashed = paths
        .par_iter()
        .map(|path| source_file_fingerprint(path))
        .collect::<Vec<_>>();
    let mut fingerprints = Vec::with_capacity(paths.len());
    let mut files = Vec::with_capacity(paths.len());
    let mut gone = Vec::new();
    for (index, (path, hashed)) in paths.iter().zip(hashed).enumerate() {
        let optional = index >= required.len();
        match hashed {
            Ok((fingerprint, revision)) => {
                fingerprints.push(fingerprint);
                files.push(((*path).clone(), revision));
            }
            Err(_) if optional && !path.exists() => gone.push((*path).clone()),
            Err(error) => return Err(error),
        }
    }
    if fingerprints.is_empty() {
        return Err("cannot identify an empty source set".to_string());
    }
    Ok(SourceIdentitySnapshot {
        identity: sha256_hex(&combine_source_fingerprints(fingerprints)),
        files,
        history,
        gone,
    })
}

pub(crate) fn verify_source_set_unchanged(snapshot: &SourceIdentitySnapshot) -> Result<(), String> {
    for (path, expected) in &snapshot.files {
        if snapshot.history.contains(path) && !path.exists() {
            // A history file removed during the import: whatever read it
            // read it whole, and whatever needed it after says so by name.
            continue;
        }
        let current = inspect_source_file_revision(path)?;
        if &current != expected {
            return Err(format!(
                "source file '{}' changed after identity capture; refusing to publish a run under stale source provenance",
                path.display()
            ));
        }
    }
    Ok(())
}

fn is_time_dimension(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "time" | "times" | "xtime" | "valid_time" | "forecast_time" | "t"
    ) || lower.starts_with("time")
}

fn record_selection(ndim: usize, time_index: usize) -> Result<NcSliceInfo, ImportError> {
    let index = u64::try_from(time_index)
        .map_err(|_| ImportError::TimeAxis("time index exceeds u64".to_string()))?;
    let mut selections = Vec::with_capacity(ndim);
    selections.push(NcSliceInfoElem::Index(index));
    selections.extend((1..ndim).map(|_| NcSliceInfoElem::Slice {
        start: 0,
        end: u64::MAX,
        step: 1,
    }));
    Ok(NcSliceInfo { selections })
}

fn read_array_f64_record_or_all(
    nc: &NcFile,
    name: &str,
    time_index: usize,
) -> Result<netcrust::DataArray, ImportError> {
    let Some(variable) = nc.variable(name) else {
        let index = u64::try_from(time_index)
            .map_err(|_| ImportError::TimeAxis("time index exceeds u64".to_string()))?;
        return nc
            .read_array_f64_record_or_all(name, index)
            .map_err(ImportError::from);
    };
    let dimensions = variable.dimensions();
    let shape = variable.shape();
    if dimensions.len() != shape.len() {
        return Err(ImportError::BadShape(name.to_string(), shape));
    }
    if dimensions
        .first()
        .map(|dimension| is_time_dimension(dimension.name()))
        .unwrap_or(false)
    {
        let record_count = shape.first().copied().unwrap_or(0);
        if time_index >= record_count {
            return Err(ImportError::TimeAxis(format!(
                "variable {name} has {record_count} time records, cannot read record {time_index}"
            )));
        }
        return nc
            .read_array_f64_slice(name, &record_selection(shape.len(), time_index)?)
            .map_err(ImportError::from);
    }
    nc.read_array_f64(name).map_err(ImportError::from)
}

pub(crate) fn parse_utc_timestamp(value: &str) -> Option<i64> {
    let trimmed = value.trim().trim_end_matches('\0').trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(parsed) = chrono::DateTime::parse_from_rfc3339(trimmed) {
        return Some(parsed.timestamp());
    }
    for format in [
        "%Y-%m-%d %H:%M:%S%.f %:z",
        "%Y-%m-%d %H:%M:%S%.f%:z",
        "%Y-%m-%d %H:%M:%S%.f %z",
        "%Y-%m-%dT%H:%M:%S%.f %:z",
    ] {
        if let Ok(parsed) = chrono::DateTime::<chrono::FixedOffset>::parse_from_str(trimmed, format)
        {
            return Some(parsed.timestamp());
        }
    }
    let without_zone = trimmed
        .strip_suffix(" UTC")
        .or_else(|| trimmed.strip_suffix(" utc"))
        .or_else(|| trimmed.strip_suffix(" GMT"))
        .or_else(|| trimmed.strip_suffix(" gmt"))
        .or_else(|| trimmed.strip_suffix('Z'))
        .unwrap_or(trimmed)
        .trim();
    let formats = [
        "%Y-%m-%d_%H:%M:%S%.f",
        "%Y-%m-%d_%H_%M_%S%.f",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S%.f",
    ];
    for format in formats {
        if let Ok(parsed) = chrono::NaiveDateTime::parse_from_str(without_zone, format) {
            return Some(parsed.and_utc().timestamp());
        }
    }
    chrono::NaiveDate::parse_from_str(without_zone, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .map(|parsed| parsed.and_utc().timestamp())
}

fn format_valid_unix(unix: i64) -> String {
    chrono::DateTime::from_timestamp(unix, 0)
        .map(|time| time.format("%Y-%m-%d %H:%M:%SZ").to_string())
        .unwrap_or_else(|| format!("unix:{unix}"))
}

/// The run origin of a file only the netcrust reader opens.  The same rule
/// as the raw wrf-core route ([`wrf_run_origin`]): a wrfout that falls back
/// to netcrust carries the same two stamps, and a delayed nest must not draw
/// under one reader and be refused under the other.
fn explicit_netcdf_reference_time(nc: &NcFile) -> Result<Option<i64>, ImportError> {
    let read = |name: &str| -> Result<Option<i64>, ImportError> {
        let Some(attribute) = nc.attribute(name) else {
            return Ok(None);
        };
        let Some(value) = attribute.as_string() else {
            return Ok(None);
        };
        parse_utc_timestamp(value).map(Some).ok_or_else(|| {
            ImportError::TimeAxis(format!(
                "global attribute {name} is not a valid UTC timestamp"
            ))
        })
    };
    let start_date = read("START_DATE")?;
    let simulation_start = read("SIMULATION_START_DATE")?;
    wrf_run_origin(start_date, simulation_start).map_err(ImportError::TimeAxis)
}

fn source_records_from_labels(
    labels: Vec<String>,
    expected_records: usize,
    context: &str,
) -> Result<Vec<SourceTimeRecord>, ImportError> {
    if expected_records > MAX_RUN_TIMESTEPS {
        return Err(ImportError::TimeAxis(format!(
            "{context} contains {expected_records} records; the per-run limit is {MAX_RUN_TIMESTEPS}"
        )));
    }
    if labels.len() != expected_records {
        return Err(ImportError::TimeAxis(format!(
            "{context} contains {} timestamps for {expected_records} time records",
            labels.len()
        )));
    }
    labels
        .into_iter()
        .enumerate()
        .map(|(time_index, label)| {
            if label.len() > MAX_TIME_LABEL_WIDTH {
                return Err(ImportError::TimeAxis(format!(
                    "{context} record {time_index} timestamp is {} bytes; the limit is {MAX_TIME_LABEL_WIDTH}",
                    label.len()
                )));
            }
            let valid_unix = parse_utc_timestamp(&label).ok_or_else(|| {
                ImportError::TimeAxis(format!(
                    "{context} record {time_index} has invalid timestamp {label:?}"
                ))
            })?;
            Ok(SourceTimeRecord {
                time_index,
                valid_unix,
                label: format_valid_unix(valid_unix),
            })
        })
        .collect()
}

fn wrf_times_from_netcdf(
    nc: &NcFile,
    time_dimension: &str,
    record_count: usize,
) -> Result<Option<Vec<SourceTimeRecord>>, ImportError> {
    let Some(labels) = wrf_time_labels_from_netcdf(nc, time_dimension, record_count)? else {
        return Ok(None);
    };
    source_records_from_labels(labels, record_count, "Times").map(Some)
}

/// WRF `Times` as one label per record, in each form a NetCDF file carries:
///
/// * `char Times(Time, DateStrLen)` in a classic file, which `read_strings`
///   returns as one string per record;
/// * the same char matrix in a NetCDF-4 file (what `ncks -4` writes from a
///   wrfout), which the HDF5 layer stores as one-byte strings, so
///   `read_strings` returns one string per character and they are joined
///   back into records here (before, such a cut was refused with "Times
///   decoded 19 records, expected 1");
/// * `string Times(Time)` (NetCDF-4 NC_STRING), one string per record.
fn wrf_time_labels_from_netcdf(
    nc: &NcFile,
    time_dimension: &str,
    record_count: usize,
) -> Result<Option<Vec<String>>, ImportError> {
    let Some(variable) = nc.variable("Times") else {
        return Ok(None);
    };
    let shape = variable.shape();
    let dimensions = variable.dimensions();
    let leading_is_time = !dimensions.is_empty()
        && dimensions.len() == shape.len()
        && dimensions[0].name() == time_dimension
        && shape[0] == record_count;
    let width = match shape.len() {
        2 if leading_is_time && shape[1] > 0 => Some(shape[1]),
        1 if leading_is_time => None,
        _ => {
            return Err(ImportError::TimeAxis(format!(
                "Times has shape {shape:?}; expected char [{record_count}, DateStrLen] or string [{record_count}] on dimension {time_dimension}"
            )));
        }
    };
    if let Some(width) = width {
        if width > MAX_TIME_LABEL_WIDTH {
            return Err(ImportError::TimeAxis(format!(
                "Times DateStrLen is {width}; the supported maximum is {MAX_TIME_LABEL_WIDTH} bytes"
            )));
        }
        let elements = record_count.checked_mul(width).ok_or_else(|| {
            ImportError::TimeAxis("Times element count overflowed usize".to_string())
        })?;
        if elements > MAX_TIME_LABEL_ELEMENTS {
            return Err(ImportError::TimeAxis(format!(
                "Times contains {elements} character elements; the safety limit is {MAX_TIME_LABEL_ELEMENTS}"
            )));
        }
    }
    // Read the char matrix as text, one string per record.  This used to
    // go through `read_array_f64` and reassemble bytes from promoted
    // doubles, which only ever worked on HDF5 tapes (the numeric read
    // refuses Char; the raw-HDF5 fallback did the promotion).  gpuwm's
    // product tape is written by the Rust CLASSIC writer by default since
    // its 1ea745e32, and a classic container has no HDF5 fallback, so a
    // postprocessed-shaped classic tape failed here with "expected
    // numeric type, got Char".  `read_strings` answers on both
    // containers.
    let mut raw_labels = nc
        .read_strings("Times")
        .map_err(|err| ImportError::TimeAxis(format!("Times could not be read as text: {err}")))?;
    if let Some(width) = width {
        if width > 1 && raw_labels.len() == record_count * width {
            // NetCDF-4 char: one element per character.  Each must be one
            // character (or an empty NUL pad); anything longer is not a
            // char matrix and is refused rather than spliced into a label.
            if let Some(index) = raw_labels
                .iter()
                .position(|element| element.trim_end_matches('\0').len() > 1)
            {
                return Err(ImportError::TimeAxis(format!(
                    "Times decoded {} elements for a [{record_count}, {width}] char matrix, and element {index} holds {} bytes instead of one character",
                    raw_labels.len(),
                    raw_labels[index].len()
                )));
            }
            raw_labels = raw_labels
                .chunks(width)
                .map(|characters| characters.concat())
                .collect();
        }
    }
    if raw_labels.len() != record_count {
        return Err(ImportError::TimeAxis(format!(
            "Times decoded {} records, expected {record_count}",
            raw_labels.len()
        )));
    }
    Ok(Some(
        raw_labels
            .into_iter()
            .map(|label| label.trim_end_matches('\0').trim().to_string())
            .collect(),
    ))
}

fn cf_time_unit(units: &str) -> Result<(f64, i64), ImportError> {
    let lower = units.to_ascii_lowercase();
    let Some(separator) = lower.find(" since ") else {
        return Err(ImportError::TimeAxis(format!(
            "CF time units {units:?} do not contain 'since'"
        )));
    };
    let unit = lower[..separator].trim();
    let scale_seconds = match unit {
        "second" | "seconds" | "sec" | "secs" | "s" => 1.0,
        "minute" | "minutes" | "min" | "mins" => 60.0,
        "hour" | "hours" | "hr" | "hrs" | "h" => 3_600.0,
        "day" | "days" | "d" => 86_400.0,
        _ => {
            return Err(ImportError::TimeAxis(format!(
                "unsupported CF time unit {unit:?} in {units:?}"
            )));
        }
    };
    let origin_text = units[separator + " since ".len()..].trim();
    let origin = parse_utc_timestamp(origin_text).ok_or_else(|| {
        ImportError::TimeAxis(format!(
            "CF time origin {origin_text:?} is not an unambiguous UTC timestamp"
        ))
    })?;
    Ok((scale_seconds, origin))
}

fn cf_time_records(
    nc: &NcFile,
    time_dimension: &str,
    record_count: usize,
) -> Result<Option<Vec<SourceTimeRecord>>, ImportError> {
    let variables = nc.variables()?;
    let mut candidates = variables
        .into_iter()
        .filter(|variable| {
            let dimensions = variable.dimensions();
            if dimensions.len() != 1 || dimensions[0].name() != time_dimension {
                return false;
            }
            variable.name() == time_dimension
                || variable.name().eq_ignore_ascii_case("time")
                || variable.name().eq_ignore_ascii_case("xtime")
                || variable
                    .attribute("axis")
                    .and_then(|attribute| attribute.as_string())
                    .map(|axis| axis.eq_ignore_ascii_case("T"))
                    .unwrap_or(false)
                || variable
                    .attribute("standard_name")
                    .and_then(|attribute| attribute.as_string())
                    .map(|name| name.eq_ignore_ascii_case("time"))
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(None);
    }
    if candidates.len() > 1 {
        let exact = candidates
            .iter()
            .filter(|variable| variable.name() == time_dimension)
            .count();
        if exact == 1 {
            candidates.retain(|variable| variable.name() == time_dimension);
        } else {
            let names = candidates
                .iter()
                .map(|variable| variable.name().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(ImportError::TimeAxis(format!(
                "ambiguous time coordinates for dimension {time_dimension}: {names}"
            )));
        }
    }
    let variable = candidates
        .pop()
        .ok_or_else(|| ImportError::TimeAxis("time coordinate disappeared".to_string()))?;
    let coordinate_shape = variable.shape();
    if coordinate_shape.len() != 1 || coordinate_shape[0] != record_count {
        return Err(ImportError::TimeAxis(format!(
            "time coordinate {} has shape {:?}, expected [{record_count}]",
            variable.name(),
            coordinate_shape
        )));
    }
    if let Some(calendar) = variable
        .attribute("calendar")
        .and_then(|attribute| attribute.as_string())
    {
        if !matches!(
            calendar.to_ascii_lowercase().as_str(),
            "standard" | "gregorian" | "proleptic_gregorian"
        ) {
            return Err(ImportError::TimeAxis(format!(
                "time coordinate {} uses unsupported calendar {calendar:?}",
                variable.name()
            )));
        }
    }
    let units = variable
        .attribute("units")
        .and_then(|attribute| attribute.as_string())
        .ok_or_else(|| {
            ImportError::TimeAxis(format!(
                "time coordinate {} is missing CF units",
                variable.name()
            ))
        })?;
    let (scale_seconds, origin) = cf_time_unit(units)?;
    let array = nc.read_array_f64(variable.name())?;
    if array.shape().len() != 1 || array.shape()[0] != record_count {
        return Err(ImportError::TimeAxis(format!(
            "time coordinate {} decoded with shape {:?}, expected [{record_count}]",
            variable.name(),
            array.shape()
        )));
    }
    let mut records = Vec::with_capacity(record_count);
    for (time_index, value) in array.values().iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ImportError::TimeAxis(format!(
                "time coordinate {} record {time_index} is non-finite",
                variable.name()
            )));
        }
        let offset = value * scale_seconds;
        let rounded = offset.round();
        if !offset.is_finite()
            || (offset - rounded).abs() > 1.0e-6
            || rounded < i64::MIN as f64
            || rounded > i64::MAX as f64
        {
            return Err(ImportError::TimeAxis(format!(
                "time coordinate {} record {time_index} cannot be represented as exact seconds",
                variable.name()
            )));
        }
        let valid_unix = origin.checked_add(rounded as i64).ok_or_else(|| {
            ImportError::TimeAxis(format!(
                "time coordinate {} record {time_index} overflows UTC seconds",
                variable.name()
            ))
        })?;
        records.push(SourceTimeRecord {
            time_index,
            valid_unix,
            label: format_valid_unix(valid_unix),
        });
    }
    Ok(Some(records))
}

/// Discover every exact record time in a generic NetCDF/WRF file. Multi-time
/// files without an unambiguous `Times` or CF coordinate are rejected; a
/// single-record file may use an exact WRF timestamp embedded in its path.
pub(crate) fn netcdf_source_times(nc: &NcFile, path: &Path) -> Result<SourceTimeAxis, ImportError> {
    let dimensions = nc.dimensions()?;
    let time_dimensions = dimensions
        .iter()
        .filter(|dimension| is_time_dimension(dimension.name()))
        .collect::<Vec<_>>();
    if time_dimensions.len() > 1 {
        let names = time_dimensions
            .iter()
            .map(|dimension| dimension.name().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(ImportError::TimeAxis(format!(
            "{} has ambiguous time dimensions: {names}",
            path.display()
        )));
    }
    let reference_unix = explicit_netcdf_reference_time(nc)?;
    let Some(time_dimension) = time_dimensions.first() else {
        let advertised = nc
            .variables()?
            .into_iter()
            .filter(|variable| {
                variable.dimensions().len() == 1
                    && (variable.name().eq_ignore_ascii_case("time")
                        || variable.name().eq_ignore_ascii_case("xtime")
                        || variable
                            .attribute("axis")
                            .and_then(|attribute| attribute.as_string())
                            .map(|axis| axis.eq_ignore_ascii_case("T"))
                            .unwrap_or(false)
                        || variable
                            .attribute("standard_name")
                            .and_then(|attribute| attribute.as_string())
                            .map(|name| name.eq_ignore_ascii_case("time"))
                            .unwrap_or(false))
            })
            .map(|variable| format!("{}({})", variable.name(), variable.dimensions()[0].name()))
            .collect::<Vec<_>>();
        if !advertised.is_empty() {
            return Err(ImportError::TimeAxis(format!(
                "{} advertises time coordinate(s) {} on unrecognized dimension names; refusing to read the file as a single snapshot",
                path.display(),
                advertised.join(", ")
            )));
        }
        let valid_unix = valid_unix_from_file_name(path, "has no time dimension")?;
        return Ok(SourceTimeAxis {
            records: vec![SourceTimeRecord {
                time_index: 0,
                valid_unix,
                label: format_valid_unix(valid_unix),
            }],
            reference_unix,
        });
    };
    let record_count = time_dimension.len();
    if record_count == 0 {
        return Err(ImportError::TimeAxis(format!(
            "{} has an empty {} dimension",
            path.display(),
            time_dimension.name()
        )));
    }
    if record_count > MAX_RUN_TIMESTEPS {
        return Err(ImportError::TimeAxis(format!(
            "{} has {record_count} time records; the per-run limit is {MAX_RUN_TIMESTEPS}",
            path.display()
        )));
    }
    let wrf_times = wrf_times_from_netcdf(nc, time_dimension.name(), record_count);
    let cf_times = cf_time_records(nc, time_dimension.name(), record_count);
    let records = match (wrf_times, cf_times) {
        (Ok(Some(wrf_records)), Ok(Some(cf_records))) => {
            let agree = wrf_records
                .iter()
                .zip(&cf_records)
                .all(|(wrf, cf)| wrf.valid_unix == cf.valid_unix);
            if !agree || wrf_records.len() != cf_records.len() {
                return Err(ImportError::TimeAxis(format!(
                    "{} has conflicting WRF Times and CF time coordinates",
                    path.display()
                )));
            }
            wrf_records
        }
        (Ok(Some(records)), Ok(None)) | (Ok(Some(records)), Err(_)) => records,
        (Ok(None), Ok(Some(records))) | (Err(_), Ok(Some(records))) => records,
        (Err(times_err), Ok(None)) => return Err(times_err),
        (_, Err(cf_err)) => return Err(cf_err),
        (Ok(None), Ok(None)) if record_count == 1 => {
            let valid_unix = valid_unix_from_file_name(
                path,
                "has one time record but no Times or CF time coordinate",
            )?;
            vec![SourceTimeRecord {
                time_index: 0,
                valid_unix,
                label: format_valid_unix(valid_unix),
            }]
        }
        (Ok(None), Ok(None)) => {
            return Err(ImportError::TimeAxis(format!(
                "{} has {record_count} time records but no unambiguous Times or CF time coordinate",
                path.display()
            )));
        }
    };
    Ok(SourceTimeAxis {
        records,
        reference_unix,
    })
}

/// The initial-condition provenance block a gpuwm-written wrfout carries,
/// reduced to the one line a plot can show.
///
/// Contract (`docs/public/DATA.md`, "The initial-condition provenance a
/// wrfout carries"), and the writer guarantees all three:
///
/// * `GPUWM_INITIAL_FORECAST_LEAD_HOURS` present and `> 0` is the only
///   predicate.  Disclose exactly then.
/// * A genuine analysis says so explicitly (lead `0`); it is not silence.
/// * **Absence means UNKNOWN, never analysis.**  Idealized cases, stock
///   WRF archives and every file written before 1.4.1 carry none of
///   these attributes, and must render exactly as they do today.  This
///   returns `None` for all of them, and stamps nothing.
///
/// The source and the cycle are read from the file, never re-derived
/// from a run slug: the slug carries the model start, which is a
/// different time and is already the `Init` token.
pub(crate) fn initial_condition_disclosure(file: &WrfFile) -> Option<String> {
    let lead = file
        .global_attr_i32("GPUWM_INITIAL_FORECAST_LEAD_HOURS")
        .ok()?;
    let source = file
        .global_attr_str("GPUWM_INITIAL_CONDITION_SOURCE")
        .ok()?;
    let cycle = file
        .global_attr_str("GPUWM_INITIAL_CONDITION_CYCLE")
        .ok()?;
    compose_initial_condition_disclosure(i64::from(lead), &source, &cycle)
}

/// The predicate and the spelling, over values rather than over a reader.
///
/// Both readers reduce to this, so the two cannot disagree, and the rules
/// can be tested without a NetCDF file: disclose only for a whole
/// positive lead with a named source and a parseable cycle; anything
/// else is UNKNOWN and stamps nothing.
pub(crate) fn compose_initial_condition_disclosure(
    lead_hours: i64,
    source: &str,
    cycle: &str,
) -> Option<String> {
    if lead_hours <= 0 {
        return None;
    }
    let source = source.trim();
    if source.is_empty() {
        return None;
    }
    let cycle = format_cycle_label(parse_utc_timestamp(cycle)?);
    Some(format!("{source} {cycle} f{lead_hours:03}"))
}

/// The same disclosure, for a file only the netcrust reader could open.
///
/// Both readers must answer this identically.  A wrfout that falls back
/// to netcrust -- a compression or chunking wrf-core's pure-Rust reader
/// declines, which is not rare -- carries exactly the same provenance
/// attributes, and a run that discloses its lead under one reader and
/// hides it under the other would be worse than one that never
/// disclosed: the label would depend on the file's encoding rather than
/// on what the run was.
pub(crate) fn netcdf_initial_condition_disclosure(nc: &NcFile) -> Option<String> {
    let lead = nc
        .attribute("GPUWM_INITIAL_FORECAST_LEAD_HOURS")
        .and_then(|attribute| attribute.as_f64())?;
    // NC_INT in the file; read as f64 and required to be a whole positive
    // number of hours, so a malformed value discloses nothing rather than
    // rounding into a claim.
    if !(lead.is_finite() && lead > 0.0 && lead.fract() == 0.0) {
        return None;
    }
    let source_attribute = nc.attribute("GPUWM_INITIAL_CONDITION_SOURCE")?;
    let source = source_attribute.as_string()?;
    let cycle_attribute = nc.attribute("GPUWM_INITIAL_CONDITION_CYCLE")?;
    let cycle = cycle_attribute.as_string()?;
    compose_initial_condition_disclosure(lead as i64, source, cycle)
}

/// The producing model's name a file carries for the metadata row
/// (`GPUWM_MODEL_LABEL`, see `rustwx_products::shared_context`), read the
/// same way under both readers.  A file the WOOF engine wrote
/// (`GPUWM_VERSION`) and that names no model is `WOOF`.  `Ok(None)` when
/// the file names no model and the engine did not write it, which is
/// every stock WRF file; `Err` names a label the row cannot take.
pub(crate) fn model_label(file: &WrfFile) -> Result<Option<String>, String> {
    use rustwx_products::shared_context::{
        ENGINE_VERSION_ATTRIBUTE, MODEL_LABEL_ATTRIBUTE, file_model_label,
    };
    let named = file.global_attr_str(MODEL_LABEL_ATTRIBUTE).ok();
    let engine_written = file.global_attr_str(ENGINE_VERSION_ATTRIBUTE).is_ok();
    file_model_label(named.as_deref(), engine_written)
}

/// [`model_label`] for a file only the netcrust reader could open.
pub(crate) fn netcdf_model_label(nc: &NcFile) -> Result<Option<String>, String> {
    use rustwx_products::shared_context::{
        ENGINE_VERSION_ATTRIBUTE, MODEL_LABEL_ATTRIBUTE, file_model_label,
    };
    let named = nc
        .attribute(MODEL_LABEL_ATTRIBUTE)
        .and_then(|attribute| attribute.as_string().map(str::to_string));
    let engine_written = nc.attribute(ENGINE_VERSION_ATTRIBUTE).is_some();
    file_model_label(named.as_deref(), engine_written)
}

/// `2026-08-01T00:00:00Z` -> `08/01 00Z`, the spelling the `Init` and
/// `Valid` tokens already use, so one subtitle holds one date format.
fn format_cycle_label(unix_seconds: i64) -> String {
    chrono::DateTime::from_timestamp(unix_seconds, 0)
        .map(|moment| moment.format("%m/%d %HZ").to_string())
        .unwrap_or_else(|| format_valid_unix(unix_seconds))
}

/// The run origin a WRF file's two start stamps give.
///
/// WRF writes a domain's OWN start into `START_DATE` and the head grid's
/// start into `SIMULATION_START_DATE` (`share/output_wrf.F`; gpuwm's writer,
/// `gpuwm/io/wrfout.py` `wrf_global_attrs`, does the same), so the two differ
/// by design on a nest that starts after the forecast and on a restart
/// file.  The run origin every lead is measured from, and the store run is
/// keyed on, is `SIMULATION_START_DATE`: it is identical on every domain of
/// one run, so a delayed nest's frame valid at 14Z reads F+02 beside the
/// parent's F+02.  Neither stamp is a frame's valid time; that comes from
/// `Times`.  With only one stamp readable, that one is the origin, as
/// before.
///
/// Refused: a `START_DATE` before `SIMULATION_START_DATE`.  No WRF domain
/// starts before its simulation, so one of the two stamps is wrong, and
/// either choice would measure every lead, the `Init` label and the run's
/// store key from a guessed origin.
pub(crate) fn wrf_run_origin(
    start_date: Option<i64>,
    simulation_start: Option<i64>,
) -> Result<Option<i64>, String> {
    match (start_date, simulation_start) {
        (Some(start), Some(simulation)) if start < simulation => Err(format!(
            "conflicting WRF references START_DATE={} and SIMULATION_START_DATE={}: the domain starts before its simulation, which no WRF writer produces, so one stamp is wrong and every lead, the Init label and the run's store key would be measured from a guessed origin",
            format_valid_unix(start),
            format_valid_unix(simulation)
        )),
        (_, Some(simulation)) => Ok(Some(simulation)),
        (start, None) => Ok(start),
    }
}

fn parse_matching_wrf_reference_attributes(
    attributes: &[(&str, String)],
) -> Result<(Option<i64>, Vec<String>), String> {
    let mut start_date = None::<i64>;
    let mut simulation_start = None::<i64>;
    let mut malformed = Vec::new();
    for (name, value) in attributes {
        let Some(parsed) = parse_utc_timestamp(value) else {
            malformed.push(format!("{name}={value:?} is not a valid UTC timestamp"));
            continue;
        };
        match *name {
            "START_DATE" => start_date = Some(parsed),
            "SIMULATION_START_DATE" => simulation_start = Some(parsed),
            other => {
                return Err(format!(
                    "{other} is not a WRF run-origin attribute; only START_DATE and SIMULATION_START_DATE are read as one, and an unknown stamp would be dropped without a word"
                ))
            }
        }
    }
    Ok((wrf_run_origin(start_date, simulation_start)?, malformed))
}

/// Infer the WRF initialization time from authoritative absolute `Times` and
/// the standard WRF `XTIME` coordinate (minutes since initialization). This is
/// deliberately strict: every record must participate, conversion to the
/// whole-second precision of `Times` must be unambiguous, and both the raw
/// floating-point and rounded integral origins must agree across the file.
fn wrf_reference_from_xtime(
    records: &[SourceTimeRecord],
    xtime_minutes: &[f64],
) -> Result<i64, String> {
    if records.is_empty() {
        return Err("cannot derive a WRF run origin from an empty Times axis".to_string());
    }
    if xtime_minutes.len() != records.len() {
        return Err(format!(
            "XTIME contains {} values for {} WRF Times records",
            xtime_minutes.len(),
            records.len()
        ));
    }

    let mut reference_unix = None::<i64>;
    let mut floating_reference = None::<f64>;
    for (record, &minutes) in records.iter().zip(xtime_minutes) {
        if !minutes.is_finite() || minutes < 0.0 {
            return Err(format!(
                "XTIME record {} must be finite and nonnegative, got {minutes:?}",
                record.time_index
            ));
        }
        let seconds = minutes * 60.0;
        if !seconds.is_finite() {
            return Err(format!(
                "XTIME record {} overflows when converted from minutes to seconds",
                record.time_index
            ));
        }
        let rounded_seconds = seconds.round();
        let rounding_error = (seconds - rounded_seconds).abs();
        if rounding_error > WRF_XTIME_SECOND_ROUNDING_TOLERANCE {
            return Err(format!(
                "XTIME record {} ({minutes:?} minutes) is {rounding_error:.6} seconds from whole-second WRF Times precision; tolerance is {:.3} seconds",
                record.time_index, WRF_XTIME_SECOND_ROUNDING_TOLERANCE
            ));
        }
        if rounded_seconds < 0.0 || rounded_seconds >= i64::MAX as f64 {
            return Err(format!(
                "XTIME record {} cannot be represented as an integral lead time in seconds",
                record.time_index
            ));
        }
        let lead_seconds = rounded_seconds as i64;
        let candidate = record.valid_unix.checked_sub(lead_seconds).ok_or_else(|| {
            format!(
                "XTIME record {} underflows UTC while deriving the WRF run origin",
                record.time_index
            )
        })?;
        let floating_candidate = record.valid_unix as f64 - seconds;

        if let Some(prior) = floating_reference {
            let difference = (floating_candidate - prior).abs();
            if difference > WRF_XTIME_ORIGIN_AGREEMENT_TOLERANCE {
                return Err(format!(
                    "XTIME-derived origins disagree at record {} by {difference:.6} seconds; tolerance is {:.3} seconds",
                    record.time_index, WRF_XTIME_ORIGIN_AGREEMENT_TOLERANCE
                ));
            }
        } else {
            floating_reference = Some(floating_candidate);
        }
        if let Some(prior) = reference_unix {
            if prior != candidate {
                return Err(format!(
                    "XTIME-derived integral origins disagree at record {}: {} versus {}",
                    record.time_index,
                    format_valid_unix(prior),
                    format_valid_unix(candidate)
                ));
            }
        } else {
            reference_unix = Some(candidate);
        }
    }

    reference_unix.ok_or_else(|| "XTIME did not yield a WRF run origin".to_string())
}

/// Exact WRF time discovery for the raw wrf-core path. `Times` must cover
/// every record; missing or truncated labels are never replaced by ordinals.
/// Parseable WRF reference attributes are preferred. A malformed or unreadable
/// attribute can fall back only to a complete, self-consistent `XTIME` axis.
pub(crate) fn wrf_source_times(file: &WrfFile, path: &Path) -> Result<SourceTimeAxis, String> {
    if file.nt > MAX_RUN_TIMESTEPS {
        return Err(format!(
            "{} has {} WRF time records; the per-run limit is {MAX_RUN_TIMESTEPS}",
            path.display(),
            file.nt
        ));
    }
    let labels = match file.times() {
        Ok(labels) => labels,
        // wrf-core reads Times only as a char matrix; a NetCDF-4 cut whose
        // Times is NC_STRING[Time] fails there ("Unsupported datatype
        // class 9") and is read through netcrust's text decode instead.
        Err(core_err) => netcrust::open(path)
            .map_err(|err| err.to_string())
            .and_then(|nc| {
                wrf_time_labels_from_netcdf(&nc, "Time", file.nt).map_err(|err| err.to_string())
            })
            .and_then(|labels| labels.ok_or_else(|| "Times is absent".to_string()))
            .map_err(|err| {
                format!(
                    "Read WRF Times from {} failed: {core_err}; the text decode failed too: {err}",
                    path.display()
                )
            })?,
    };
    let records = source_records_from_labels(labels, file.nt, "WRF Times")
        .map_err(|err| format!("{}: {err}", path.display()))?;
    let mut attributes = Vec::<(&str, String)>::new();
    let mut attribute_diagnostics = Vec::new();
    for name in ["START_DATE", "SIMULATION_START_DATE"] {
        match file.global_attr_str(name) {
            Ok(value) => attributes.push((name, value)),
            Err(err) => attribute_diagnostics.push(format!("{name} unavailable ({err})")),
        }
    }
    let (reference_unix, malformed) = parse_matching_wrf_reference_attributes(&attributes)
        .map_err(|err| format!("{} has {err}", path.display()))?;
    if let Some(reference_unix) = reference_unix {
        return Ok(SourceTimeAxis {
            records,
            reference_unix: Some(reference_unix),
        });
    }
    attribute_diagnostics.extend(malformed);

    // WRF defines one-dimensional XTIME as minutes since initialization.
    // wrf-core returns the complete vector for a one-dimensional variable.
    let xtime_result = if file.has_var("XTIME") {
        file.read_var("XTIME", 0)
            .map_err(|err| format!("reading XTIME failed ({err})"))
    } else {
        Err("XTIME is unavailable".to_string())
    };
    let reference_unix = xtime_result
        .and_then(|xtime| wrf_reference_from_xtime(&records, &xtime))
        .map_err(|xtime_err| {
            let attributes = if attribute_diagnostics.is_empty() {
                "no parseable START_DATE or SIMULATION_START_DATE attribute".to_string()
            } else {
                attribute_diagnostics.join("; ")
            };
            format!(
                "{} has no sound WRF run origin: {attributes}; XTIME fallback failed: {xtime_err}",
                path.display()
            )
        })?;
    Ok(SourceTimeAxis {
        records,
        reference_unix: Some(reference_unix),
    })
}

#[derive(Debug, Clone)]
struct LocalSourcePlan {
    path: PathBuf,
    records: Vec<PlannedSourceTime>,
}

/// Open every selected source and settle the complete time/geometry plan
/// before the first staged hour is written. Coordinate values are still
/// checked bit-for-bit by rw-store as staged hours are appended; this catches
/// every cheap metadata-level mismatch up front without decoding every field
/// twice.
fn preflight_local_sources(
    files: &[PathBuf],
    source_identity: &str,
    processing_profile: &str,
) -> Result<(Vec<LocalSourcePlan>, String), ImportError> {
    let mut expected_shape = None::<(usize, usize)>;
    let mut sources = Vec::<(PathBuf, SourceTimeAxis)>::with_capacity(files.len());
    let mut model_labels = Vec::<Option<String>>::with_capacity(files.len());
    for path in files {
        // Probe wrf-core first so common raw wrfouts do not pay netcrust's
        // expensive eager metadata indexing twice (preflight + processing).
        let raw = crate::wrf_process::isolate_panics("preflight WRF file", || {
            WrfFile::open(path).map_err(|err| err.to_string())
        });
        let (source_times, shape) = match raw {
            Ok(file) => {
                // Every source of one invocation belongs to one run (the
                // timeline planner refuses sources whose reference times
                // disagree), so the disclosure is a property of the run and
                // is recorded once here, where the file is already open.
                // Absent attributes leave it cleared and nothing is stamped.
                rustwx_products::shared_context::set_initial_condition_disclosure(
                    initial_condition_disclosure(&file),
                );
                model_labels.push(model_label(&file).unwrap_or_else(|reason| {
                    eprintln!("IMPORT_NOTE	{reason}; the plots keep the store identity");
                    None
                }));
                (
                    wrf_source_times(&file, path).map_err(ImportError::TimeAxis)?,
                    (file.nx, file.ny),
                )
            }
            Err(_) => {
                let nc = netcrust::open(path)?;
                rustwx_products::shared_context::set_initial_condition_disclosure(
                    netcdf_initial_condition_disclosure(&nc),
                );
                model_labels.push(netcdf_model_label(&nc).unwrap_or_else(|reason| {
                    eprintln!("IMPORT_NOTE	{reason}; the plots keep the store identity");
                    None
                }));
                (
                    netcdf_source_times(&nc, path)?,
                    netcdf_grid_shape(&nc, path)?,
                )
            }
        };
        merge_preflight_grid_shape(&mut expected_shape, shape, path)
            .map_err(ImportError::TimeAxis)?;
        sources.push((path.clone(), source_times));
    }
    // One run per invocation, as the disclosure above: the model the files
    // name is the run's, when every file names the same one.
    rustwx_products::shared_context::set_model_label(
        rustwx_products::shared_context::agreed_model_label(&model_labels),
    );
    let timeline = ForecastHourTimeline::plan_all(&sources).map_err(ImportError::TimeAxis)?;
    let mut plans = Vec::with_capacity(sources.len());
    for (index, (path, _)) in sources.into_iter().enumerate() {
        let records = timeline
            .records_for_source(index)
            .ok_or_else(|| {
                ImportError::TimeAxis(format!(
                    "internal error: forecast timeline omitted source {}",
                    path.display()
                ))
            })?
            .to_vec();
        plans.push(LocalSourcePlan { path, records });
    }
    let run = timeline.run_name(source_identity, processing_profile);
    Ok((plans, run))
}

pub(crate) fn netcdf_grid_shape(nc: &NcFile, path: &Path) -> Result<(usize, usize), ImportError> {
    let find_shape = |names: &[&str]| {
        names.iter().find_map(|name| {
            let variable = nc.variable(name)?;
            let shape = variable.shape();
            (shape.len() >= 2).then(|| (shape[shape.len() - 1], shape[shape.len() - 2]))
        })
    };
    let lat = find_shape(&["XLAT", "XLAT_M", "lat", "latitude"])
        .ok_or_else(|| ImportError::MissingAny(vec!["XLAT".to_string(), "lat".to_string()]))?;
    let lon = find_shape(&["XLONG", "XLONG_M", "lon", "longitude"])
        .ok_or_else(|| ImportError::MissingAny(vec!["XLONG".to_string(), "lon".to_string()]))?;
    if lat != lon || lat.0 == 0 || lat.1 == 0 {
        return Err(ImportError::TimeAxis(format!(
            "{} has incompatible latitude/longitude grid shapes {:?} and {:?}",
            path.display(),
            lat,
            lon
        )));
    }
    Ok(lat)
}

pub(crate) fn merge_preflight_grid_shape(
    expected: &mut Option<(usize, usize)>,
    shape: (usize, usize),
    path: &Path,
) -> Result<(), String> {
    GridShape::new(shape.0, shape.1).map_err(|err| {
        format!(
            "{} declares an unsafe or unsupported grid {}x{}: {err}",
            path.display(),
            shape.0,
            shape.1
        )
    })?;
    match expected {
        Some(prior) if *prior != shape => Err(format!(
            "{} uses grid {}x{}, but the selected run was preflighted as {}x{}",
            path.display(),
            shape.0,
            shape.1,
            prior.0,
            prior.1
        )),
        Some(_) => Ok(()),
        None => {
            *expected = Some(shape);
            Ok(())
        }
    }
}

fn import_paths(
    paths: &[PathBuf],
    store_root: &Path,
    progress: &mut dyn FnMut(String),
) -> Result<LocalImportSummary, ImportError> {
    if paths.is_empty() {
        return Err(ImportError::NoFiles);
    }
    let mut files: Vec<PathBuf> = paths
        .iter()
        .filter(|path| is_supported_model_file(path))
        .cloned()
        .collect();
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    if files.is_empty() {
        return Err(ImportError::NoSupportedFiles);
    }
    if files.len() > u16::MAX as usize {
        return Err(ImportError::TooManyFiles(files.len()));
    }

    // GRIB1 files carry many timesteps each and decode through grib-core,
    // not netcrust: the whole selection routes to `grib_import`. Mixed
    // selections are refused rather than guessed at: both importers derive
    // exact valid-time hours, but use different source metadata and run
    // identity rules; interleaving them would create an ambiguous run.
    if files
        .iter()
        .any(|path| crate::grib_import::is_grib1_file(path))
    {
        if !files
            .iter()
            .all(|path| crate::grib_import::is_grib1_file(path))
        {
            return Err(ImportError::MixedGribSelection);
        }
        return crate::grib_import::import_grib1_files(&files, store_root, progress)
            .map_err(ImportError::Grib);
    }

    let source_snapshot =
        capture_source_set_identity(&files).map_err(ImportError::SourceIdentity)?;
    let source_identity = &source_snapshot.identity;
    let model = "wrf".to_string();
    let (plans, run) = preflight_local_sources(&files, source_identity, "light")?;
    let publisher =
        RunStagingPublisher::new(store_root, &model, &run).map_err(ImportError::Publish)?;
    let staging_store_root = publisher.staging_store_root().to_path_buf();
    let total = plans.len();
    let mut all_vars = Vec::new();
    let mut written = Vec::<WrittenHour>::new();
    let mut notes = Vec::<String>::new();
    for (index, plan) in plans.iter().enumerate() {
        let path = &plan.path;
        // Every stage line carries the file position, so a folder import reads
        // "file 3/10 (wrfout_…): interpolating …" rather than a bare spinner.
        let tag = format!("file {}/{total} ({})", index + 1, display_name(path));
        // One netcrust handle per file: `netcrust::open` eagerly indexes the
        // NetCDF-4 metadata twice over (NcFile + Hdf5File, ~57 s of
        // hdf5-reader churn on a 2 GB Enderlin wrfout, measured), so the
        // post-processed gate and the 2D reader below must share it. A file
        // netcrust can't open would have failed `read_wrf_2d_fields` with
        // this same error before; it just surfaces one stage earlier now.
        let nc = netcrust::open(path)?;
        let is_postprocessed = is_postprocessed_wrf(&nc);
        let wrf_file = if is_postprocessed {
            None
        } else {
            crate::wrf_process::isolate_panics("open WRF file", || {
                WrfFile::open(path).map_err(|err| err.to_string())
            })
            .ok()
        };
        progress(format!(
            "{tag}: discovered {} exact time record(s)",
            plan.records.len()
        ));
        for record in &plan.records {
            let storage_slot = record.storage_slot;
            let record_tag = format!(
                "{tag}, time {} ({}) -> {}",
                record.time_index,
                record.label,
                record.display_key()
            );
            // Post-processed climate wrfout (CONUS-I/II, GDEX: derived TK/Z/P, no
            // raw T/PB) can't go through the raw-wrfout reader, build it directly.
            // (Bound before the `if let` so the prefixing closure's borrow of
            // `progress` ends before the block uses `progress` again.)
            let postprocessed = try_postprocessed_wrf_shared(
                &nc,
                path,
                record.time_index,
                false,
                &mut |message| progress(format!("{record_tag}: {message}")),
            )?;
            if let Some((canonical, severe, volumes, raw_2d, hour_notes)) = postprocessed {
                for note in hour_notes {
                    notes.push(format!(
                        "{} time {}: {note}",
                        display_name(path),
                        record.label
                    ));
                }
                let refs = canonical
                    .iter()
                    .map(|(name, field)| (name.as_str(), field))
                    .collect::<Vec<_>>();
                // Light import requests no computed severe fields. The derived
                // slot remains here for the shared heavy path's explicit
                // `approx_*` products and for raw wrf2d fields below.
                let mut derived_refs = severe
                    .iter()
                    .map(|field| DerivedFieldInput {
                        name: field.name,
                        units: field.units,
                        values: field.values.as_slice(),
                    })
                    .collect::<Vec<_>>();
                // Raw `wrf_*` planes from the 2-D wrf2d route share that slot,
                // the same convention the raw-wrfout light import uses (empty on
                // the 3-D route, so its hours are written exactly as before).
                derived_refs.extend(raw_2d.iter().map(|field| DerivedFieldInput {
                    name: field.name.as_str(),
                    units: field.units.as_str(),
                    values: field.values.as_slice(),
                }));
                let volume_inputs = volumes.iter().map(IsoVolume::as_input).collect::<Vec<_>>();
                progress(format!("{record_tag}: writing to store"));
                let result = match record.exact_time {
                    Some(exact_time) => write_hour_from_fields_with_derived_exact(
                        &staging_store_root,
                        &model,
                        &run,
                        storage_slot,
                        exact_time,
                        &refs,
                        &derived_refs,
                        &volume_inputs,
                        writer_build(),
                        now_unix(),
                    ),
                    None => write_hour_from_fields_with_derived(
                        &staging_store_root,
                        &model,
                        &run,
                        storage_slot,
                        &refs,
                        &derived_refs,
                        &volume_inputs,
                        writer_build(),
                        now_unix(),
                    ),
                }
                .map_err(|source| ImportError::StoreWrite {
                    context: record_tag.clone(),
                    source,
                })?;
                all_vars.extend(result.vars.iter().cloned());
                written.push(result);
                continue;
            }
            if is_postprocessed {
                return Err(ImportError::TimeAxis(format!(
                    "{} changed post-processed classification while reading record {}",
                    path.display(),
                    record.time_index
                )));
            }
            progress(format!("{record_tag}: reading 2D surface fields"));
            // One wrf-core handle per raw wrfout, shared by the fast 2D plane
            // reads AND the isobaric volume build. `None` (plain NetCDF, or a
            // panic on a pathological header) keeps every 2D read on netcrust
            // and skips the volumes, exactly the pre-fast-path behavior.
            let mut fields = read_wrf_2d_fields(
                &nc,
                path,
                wrf_file.as_ref(),
                record.time_index,
                &mut |message| progress(format!("{record_tag}: {message}")),
            )?;
            if fields.canonical.is_empty() {
                return Err(ImportError::NoFields(path.clone()));
            }
            // Isobaric sounding volumes + lowest-model-level surface fallback, so an
            // imported WRF run makes soundings. Built through wrf-core; a plain
            // NetCDF wrf-core can't open yields neither. Fill any surface field the
            // 2D read missed (e.g. PSFC in a split wrf3d file) from the fallback.
            let (iso_volumes, surface_fallback, volume_note) =
                read_iso_volumes(wrf_file.as_ref(), record.time_index, &mut |message| {
                    progress(format!("{record_tag}: {message}"))
                });
            if let Some(note) = volume_note {
                progress(format!("{record_tag}: {note}"));
                notes.push(format!(
                    "{} time {}: {note}",
                    display_name(path),
                    record.label
                ));
            }
            if let Some(surface) = surface_fallback {
                fill_missing_surface(&mut fields, surface);
            }
            let refs = fields
                .canonical
                .iter()
                .map(|(name, field)| (name.as_str(), field))
                .collect::<Vec<_>>();
            let raw_refs = fields
                .raw_2d
                .iter()
                .map(|field| DerivedFieldInput {
                    name: field.name.as_str(),
                    units: field.units.as_str(),
                    values: field.values.as_slice(),
                })
                .collect::<Vec<_>>();
            // Volume planes come from wrf-core, the 2D grid from netcrust; if they
            // ever disagree on grid size, drop volumes rather than fail the hour.
            let grid_cells = fields.grid.shape.len();
            let volumes_match = iso_volumes.iter().all(|volume| {
                volume
                    .levels
                    .iter()
                    .all(|(_, plane)| plane.len() == grid_cells)
            });
            let volume_inputs = if volumes_match {
                iso_volumes
                    .iter()
                    .map(IsoVolume::as_input)
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            progress(format!("{record_tag}: writing to store"));
            let result = match record.exact_time {
                Some(exact_time) => write_hour_from_fields_with_derived_exact(
                    &staging_store_root,
                    &model,
                    &run,
                    storage_slot,
                    exact_time,
                    &refs,
                    &raw_refs,
                    &volume_inputs,
                    writer_build(),
                    now_unix(),
                ),
                None => write_hour_from_fields_with_derived(
                    &staging_store_root,
                    &model,
                    &run,
                    storage_slot,
                    &refs,
                    &raw_refs,
                    &volume_inputs,
                    writer_build(),
                    now_unix(),
                ),
            }
            .map_err(|source| ImportError::StoreWrite {
                context: record_tag.clone(),
                source,
            })?;
            all_vars.extend(result.vars.iter().cloned());
            written.push(result);
        }
    }
    all_vars.sort();
    all_vars.dedup();
    verify_source_set_unchanged(&source_snapshot).map_err(ImportError::SourceIdentity)?;
    progress(format!("Publishing complete run {model}/{run}"));
    publisher.publish().map_err(ImportError::Publish)?;
    Ok(LocalImportSummary {
        store_root: store_root.to_path_buf(),
        model,
        run,
        files_seen: files.len(),
        hours_written: written.len(),
        variables: all_vars,
        notes,
    })
}

fn read_wrf_2d_fields(
    nc: &NcFile,
    path: &Path,
    wrf: Option<&WrfFile>,
    time_index: usize,
    progress: &mut dyn FnMut(String),
) -> Result<ImportedWrfFields, ImportError> {
    let declared_shape = match wrf {
        Some(file) => (file.nx, file.ny),
        None => netcdf_grid_shape(nc, path)?,
    };
    GridShape::new(declared_shape.0, declared_shape.1)?;
    let src = PlaneSource::new(nc, wrf, time_index);
    let lat = read_first_2d_any(&src, &["XLAT", "XLAT_M", "lat", "latitude"])?;
    let lon = read_first_2d_any(&src, &["XLONG", "XLONG_M", "lon", "longitude"])?;
    if lat.nx != lon.nx || lat.ny != lon.ny || lat.values.len() != lon.values.len() {
        return Err(ImportError::GridMismatch(path.to_path_buf()));
    }
    let shape = GridShape::new(lat.nx, lat.ny)?;
    let grid = LatLonGrid::new(shape, lat.values, lon.values)?;
    let projection = wrf_projection(nc);
    let mut canonical = Vec::new();
    push_canonical_surface_fields(&mut canonical, &src, &grid, &projection)?;

    let raw_2d = read_raw_wrf_mass_grid_fields(&src, lat.nx, lat.ny, progress)?;

    // Surface how the planes were actually decoded: the stage timestamps in
    // the instrumented harness (and the dock) should show whether the fast
    // path engaged and which planes, if any, wrf-core could not read.
    if wrf.is_some() {
        let fallbacks = src.netcrust_fallbacks.borrow();
        if fallbacks.is_empty() {
            progress(format!(
                "read {} 2D planes via wrf-core reader",
                src.wrf_reads.get()
            ));
        } else {
            progress(format!(
                "read {} 2D planes via wrf-core reader; {} fell back to netcrust: {}",
                src.wrf_reads.get(),
                fallbacks.len(),
                fallbacks.join(", ")
            ));
        }
    }

    Ok(ImportedWrfFields {
        canonical,
        raw_2d,
        grid,
        projection,
    })
}

/// The canonical surface-field suite shared by the raw-wrfout 2-D read and
/// the post-processed 2-D (`wrf2d`) route: direct T2/U10/V10/PSFC/HGT/SLP/
/// REFD_MAX planes plus the derived 10 m wind speed, 2 m dewpoint /
/// relative humidity, and total-precipitation fields, each pushed only when
/// its source planes exist in the file. Body moved unchanged from
/// `read_wrf_2d_fields` (only the borrow spellings changed for the
/// by-reference parameters).
fn push_canonical_surface_fields(
    canonical: &mut Vec<(String, SelectedField2D)>,
    src: &PlaneSource,
    grid: &LatLonGrid,
    projection: &Option<GridProjection>,
) -> Result<(), ImportError> {
    push_direct(
        canonical,
        src,
        grid,
        projection.clone(),
        "T2",
        "temperature_2m",
        FieldSelector::height_agl(CanonicalField::Temperature, 2),
        Some("K"),
    )?;
    push_pressure_direct(
        canonical,
        src,
        grid,
        projection.clone(),
        "PSFC",
        "surface_pressure",
        FieldSelector::surface(CanonicalField::Pressure),
        1.0,
    )?;
    push_direct(
        canonical,
        src,
        grid,
        projection.clone(),
        "HGT",
        "orography",
        FieldSelector::surface(CanonicalField::GeopotentialHeight),
        Some("m"),
    )?;
    push_pressure_direct(
        canonical,
        src,
        grid,
        projection.clone(),
        "SLP",
        "mslp",
        FieldSelector::mean_sea_level(CanonicalField::PressureReducedToMeanSeaLevel),
        // WRF's diagnostic SLP convention is hPa when an older file omits
        // the units attribute; explicit Pa/hPa metadata always wins.
        100.0,
    )?;
    push_direct(
        canonical,
        src,
        grid,
        projection.clone(),
        "REFD_MAX",
        "composite_reflectivity",
        FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity),
        Some("dBZ"),
    )?;
    // WSPD10MAX is NOT published under a canonical selector, and in
    // particular not under `height_agl(WindGust, 10)`, which is where it
    // used to go.
    //
    // WRF 4.6.1 defines no wind gust.  `phys/module_diag_nwp.F` defines
    // WSPD10MAX, Registry title "WIND SPD MAX 10 M": the running maximum
    // of sqrt(u10^2 + v10^2) between history writes, reset after each
    // dump.  AFWA's `module_diag_afwa.F` writes the same array with a
    // precipitation blend and still names it a maximum.  Publishing it
    // as the gust plane made the catalog's "10m AGL Wind Gusts" chart
    // render a running maximum under a title naming a quantity this
    // model does not compute -- and it did so silently, because the
    // selector matched.
    //
    // Nor may it be published under `height_agl(WindSpeed, 10)`: this
    // route computes the true instantaneous speed from U10/V10 a few
    // lines below, the store's selector map is first-write-wins, and a
    // maximum pushed first would SHADOW the instantaneous field that
    // `10m_wind_speed_and_direction` renders.  A time maximum has no
    // canonical selector in this catalog; until one exists with its own
    // accurate label, the plane stays out of the canonical map rather than
    // borrowing a name that means something else.

    // WRF U10/V10 are grid-relative. Publish canonical vector components
    // only after applying the same SINALPHA/COSALPHA rotation as wrf-python's
    // uvmet10. The scalar speed remains valid without rotation.
    if let (Some(u10), Some(v10)) = (read_first_2d(src, "U10")?, read_first_2d(src, "V10")?) {
        if let Some(rotation) = read_wrf_wind_rotation(src, projection.as_ref(), grid.shape.len())?
        {
            let (u_earth, v_earth) = rotation.rotate_f32_pair(&u10, &v10)?;
            push_computed(
                canonical,
                grid,
                projection.clone(),
                "u_10m",
                FieldSelector::height_agl(CanonicalField::UWind, 10),
                "m/s",
                u_earth,
            )?;
            push_computed(
                canonical,
                grid,
                projection.clone(),
                "v_10m",
                FieldSelector::height_agl(CanonicalField::VWind, 10),
                "m/s",
                v_earth,
            )?;
        }
        let values = combine_same_grid(&u10, &v10, |u, v| (u.mul_add(u, v * v)).sqrt())?;
        push_computed(
            canonical,
            grid,
            projection.clone(),
            "wind_speed_10m",
            FieldSelector::height_agl(CanonicalField::WindSpeed, 10),
            "m/s",
            values,
        )?;
    }

    if let (Some(t2), Some(q2), Some(psfc)) = (
        read_first_2d(src, "T2")?,
        read_first_2d(src, "Q2")?,
        read_pressure_plane_pa(src, "PSFC", 1.0)?,
    ) {
        let dewpoint = derive_dewpoint_k(&t2, &q2, &psfc)?;
        push_computed(
            canonical,
            grid,
            projection.clone(),
            "dewpoint_2m",
            FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
            "K",
            dewpoint,
        )?;
        let rh = derive_relative_humidity_percent(&t2, &q2, &psfc)?;
        push_computed(
            canonical,
            grid,
            projection.clone(),
            "relative_humidity_2m",
            FieldSelector::height_agl(CanonicalField::RelativeHumidity, 2),
            "%",
            rh,
        )?;
    }

    if let (Some(rainc), Some(rainnc)) =
        (read_first_2d(src, "RAINC")?, read_first_2d(src, "RAINNC")?)
    {
        let rainsh = read_first_2d(src, "RAINSH")?;
        let values = combine_precip(&rainc, &rainnc, rainsh.as_ref())?;
        push_computed(
            canonical,
            grid,
            projection.clone(),
            "apcp",
            FieldSelector::surface(CanonicalField::TotalPrecipitation),
            "kg/m^2",
            values,
        )?;
    }

    Ok(())
}

fn push_direct(
    out: &mut Vec<(String, SelectedField2D)>,
    src: &PlaneSource,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    wrf_name: &str,
    store_name: &str,
    selector: FieldSelector,
    units_override: Option<&str>,
) -> Result<(), ImportError> {
    let Some(plane) = read_first_2d(src, wrf_name)? else {
        return Ok(());
    };
    let units = units_override
        .map(str::to_string)
        .or_else(|| variable_units(src.nc, wrf_name))
        .unwrap_or_else(|| selector.native_units().to_string());
    push_computed(
        out,
        grid,
        projection,
        store_name,
        selector,
        &units,
        plane.values,
    )
}

fn push_pressure_direct(
    out: &mut Vec<(String, SelectedField2D)>,
    src: &PlaneSource,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    wrf_name: &str,
    store_name: &str,
    selector: FieldSelector,
    missing_units_scale_to_pa: f32,
) -> Result<(), ImportError> {
    let Some(plane) = read_pressure_plane_pa(src, wrf_name, missing_units_scale_to_pa)? else {
        return Ok(());
    };
    push_computed(
        out,
        grid,
        projection,
        store_name,
        selector,
        "Pa",
        plane.values,
    )
}

fn read_pressure_plane_pa(
    src: &PlaneSource,
    name: &str,
    missing_units_scale_to_pa: f32,
) -> Result<Option<Plane2D>, ImportError> {
    let Some(mut plane) = read_first_2d(src, name)? else {
        return Ok(None);
    };
    let source_units = variable_units(src.nc, name);
    let scale = pressure_scale_to_pa(source_units.as_deref(), missing_units_scale_to_pa)
        .ok_or_else(|| ImportError::UnsupportedPressureUnits {
            variable: name.to_string(),
            units: source_units.unwrap_or_else(|| "<missing>".to_string()),
        })?;
    if scale != 1.0 {
        for value in &mut plane.values {
            if value.is_finite() {
                *value *= scale;
            }
        }
    }
    Ok(Some(plane))
}

fn pressure_scale_to_pa(units: Option<&str>, missing_scale: f32) -> Option<f32> {
    let Some(units) = units else {
        return missing_scale.is_finite().then_some(missing_scale);
    };
    let normalized = units
        .trim()
        .chars()
        .filter(|ch| !matches!(ch, ' ' | '_' | '-'))
        .flat_map(char::to_lowercase)
        .collect::<String>();
    match normalized.as_str() {
        "pa" | "pascal" | "pascals" => Some(1.0),
        "hpa" | "hectopascal" | "hectopascals" | "mb" | "mbar" | "millibar" | "millibars" => {
            Some(100.0)
        }
        "kpa" | "kilopascal" | "kilopascals" => Some(1_000.0),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum PostprocessedUnitKind {
    TemperatureKelvin,
    PressurePascal,
    HeightMeter,
    MixingRatioKgKg,
    WindMetersPerSecond,
}

impl PostprocessedUnitKind {
    fn expected(self) -> &'static str {
        match self {
            Self::TemperatureKelvin => "K or degC",
            Self::PressurePascal => "Pa, hPa/mb, or kPa",
            Self::HeightMeter => "m/gpm or km",
            Self::MixingRatioKgKg => "kg/kg or g/kg",
            Self::WindMetersPerSecond => "m/s or knots",
        }
    }

    fn plausible(self, value: f64) -> bool {
        match self {
            Self::TemperatureKelvin => (100.0..=450.0).contains(&value),
            Self::PressurePascal => (1.0..=200_000.0).contains(&value),
            Self::HeightMeter => (-2_000.0..=100_000.0).contains(&value),
            Self::MixingRatioKgKg => (-0.001..=0.2).contains(&value),
            Self::WindMetersPerSecond => value.abs() <= 500.0,
        }
    }
}

fn normalized_unit_token(units: &str) -> String {
    units
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '/')
        .flat_map(char::to_lowercase)
        .collect()
}

fn postprocessed_unit_affine(units: &str, kind: PostprocessedUnitKind) -> Option<(f64, f64)> {
    let units = normalized_unit_token(units);
    match kind {
        PostprocessedUnitKind::TemperatureKelvin => match units.as_str() {
            "k" | "kelvin" | "kelvins" => Some((1.0, 0.0)),
            "c" | "degc" | "degreec" | "degreesc" | "celsius" | "degreecelsius"
            | "degreescelsius" => Some((1.0, 273.15)),
            _ => None,
        },
        PostprocessedUnitKind::PressurePascal => match units.as_str() {
            "pa" | "pascal" | "pascals" => Some((1.0, 0.0)),
            "hpa" | "hectopascal" | "hectopascals" | "mb" | "mbar" | "millibar" | "millibars" => {
                Some((100.0, 0.0))
            }
            "kpa" | "kilopascal" | "kilopascals" => Some((1_000.0, 0.0)),
            _ => None,
        },
        PostprocessedUnitKind::HeightMeter => match units.as_str() {
            "m" | "meter" | "meters" | "metre" | "metres" | "gpm" => Some((1.0, 0.0)),
            "km" | "kilometer" | "kilometers" | "kilometre" | "kilometres" => Some((1_000.0, 0.0)),
            _ => None,
        },
        PostprocessedUnitKind::MixingRatioKgKg => match units.as_str() {
            "1" | "dimensionless" | "kg/kg" | "kgkg" | "kgkg1" => Some((1.0, 0.0)),
            "g/kg" | "gkg" | "gkg1" => Some((0.001, 0.0)),
            _ => None,
        },
        PostprocessedUnitKind::WindMetersPerSecond => match units.as_str() {
            "m/s" | "ms" | "ms1" | "meterpersecond" | "meterspersecond" | "metrepersecond"
            | "metrespersecond" => Some((1.0, 0.0)),
            "kt" | "kts" | "knot" | "knots" => Some((0.514_444, 0.0)),
            _ => None,
        },
    }
}

fn normalize_postprocessed_values(
    nc: &NcFile,
    name: &str,
    values: &mut [f64],
    kind: PostprocessedUnitKind,
) -> Result<(), ImportError> {
    let units = variable_units(nc, name).ok_or_else(|| ImportError::UnsupportedFieldUnits {
        variable: name.to_string(),
        units: "<missing>".to_string(),
        expected: kind.expected(),
    })?;
    let (scale, offset) = postprocessed_unit_affine(&units, kind).ok_or_else(|| {
        ImportError::UnsupportedFieldUnits {
            variable: name.to_string(),
            units,
            expected: kind.expected(),
        }
    })?;
    let mut plausible = 0usize;
    for value in values {
        if !value.is_finite() || value.abs() >= 1.0e30 || *value <= -9_998.0 {
            *value = f64::NAN;
            continue;
        }
        *value = (*value).mul_add(scale, offset);
        if kind.plausible(*value) {
            plausible += 1;
        } else {
            *value = f64::NAN;
        }
    }
    if plausible == 0 {
        return Err(ImportError::NoPlausibleValues(name.to_string()));
    }
    Ok(())
}

fn push_computed(
    out: &mut Vec<(String, SelectedField2D)>,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    store_name: &str,
    selector: FieldSelector,
    units: &str,
    values: Vec<f32>,
) -> Result<(), ImportError> {
    let mut field = SelectedField2D::new(selector, units, grid.clone(), values)?;
    if let Some(projection) = projection {
        field = field.with_projection(projection);
    }
    out.push((store_name.to_string(), field));
    Ok(())
}

fn read_raw_wrf_mass_grid_fields(
    src: &PlaneSource,
    nx: usize,
    ny: usize,
    progress: &mut dyn FnMut(String),
) -> Result<Vec<RawField2D>, ImportError> {
    read_raw_wrf_mass_grid_fields_where(src, nx, ny, progress, &mut |_, _| true)
}

/// The raw `(Time, south_north, west_east)` plane sweep, with a caller
/// predicate over `(wrf_name, store_name)`.
///
/// The full science import (`wrf_process`) shares this so a plane a USER
/// added to their WRF Registry reaches the store under the same
/// `wrf_<sanitized>` name the light import gives it, which is the exact
/// spelling `--products var:<name>` resolves. It needs the predicate
/// because that route has already computed a large catalog of fields under
/// their own names and must neither duplicate nor shadow them, and because
/// its `--only`/`--skip` token grammar has to reach these planes too.
pub(crate) fn read_raw_wrf_mass_grid_fields_where(
    src: &PlaneSource,
    nx: usize,
    ny: usize,
    progress: &mut dyn FnMut(String),
    accept: &mut dyn FnMut(&str, &str) -> bool,
) -> Result<Vec<RawField2D>, ImportError> {
    let mut seen = HashSet::<String>::new();
    let mut raw = Vec::new();
    for var in src.nc.variables()? {
        let wrf_name = var.name();
        if !is_raw_wrf_mass_grid_variable(&var, nx, ny) || !raw_wrf_variable_allowed(wrf_name) {
            continue;
        }
        let name = format!("wrf_{}", sanitize_store_var_name(wrf_name));
        if name == "wrf_" || !accept(wrf_name, &name) {
            continue;
        }
        // One line per raw plane: on a compressed 250 m wrfout each first-
        // record read decompresses real data, and there are dozens of them.
        progress(format!("reading raw 2D field {wrf_name}"));
        let Some(plane) = read_first_2d(src, wrf_name)? else {
            continue;
        };
        if plane.nx != nx || plane.ny != ny {
            continue;
        }
        if !seen.insert(name.clone()) {
            continue;
        }
        raw.push(RawField2D {
            name,
            units: variable_units(src.nc, wrf_name).unwrap_or_else(|| "1".to_string()),
            values: plane.values,
        });
    }
    Ok(raw)
}

fn is_raw_wrf_mass_grid_variable(var: &NcVariable, nx: usize, ny: usize) -> bool {
    let dims = var.dimensions();
    let shape = var.shape();
    dims.len() == 3
        && shape.len() == 3
        && is_time_dimension(dims[0].name())
        && dims[1].name() == "south_north"
        && dims[2].name() == "west_east"
        && shape[1] == ny
        && shape[2] == nx
}

fn raw_wrf_variable_allowed(name: &str) -> bool {
    !matches!(
        name.to_ascii_uppercase().as_str(),
        "XLAT"
            | "XLONG"
            | "XLAT_M"
            | "XLONG_M"
            | "CLAT"
            | "NEST_POS"
            | "AREA2D"
            | "DX2D"
            | "MAPFAC_M"
            | "MAPFAC_MX"
            | "MAPFAC_MY"
            | "F"
            | "E"
            | "SINALPHA"
            | "COSALPHA"
    )
}

fn sanitize_store_var_name(name: &str) -> String {
    let mut out = String::new();
    let mut last_was_underscore = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !last_was_underscore {
            out.push('_');
            last_was_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    out
}

#[derive(Debug, Clone)]
struct Plane2D {
    nx: usize,
    ny: usize,
    values: Vec<f32>,
}

/// Rotation from WRF grid-relative U/V to earth-relative east/north. WRF
/// stores SINALPHA/COSALPHA on the horizontal mass grid and reuses the same
/// pair at every vertical level. Ordinary, unrotated MAP_PROJ=6 files may
/// legitimately omit both planes; projected and rotated-pole grids may not
/// silently assume identity.
enum WrfWindRotation {
    Identity,
    Angles { sin: Vec<f32>, cos: Vec<f32> },
}

impl WrfWindRotation {
    fn rotate_f32_pair(
        &self,
        u: &Plane2D,
        v: &Plane2D,
    ) -> Result<(Vec<f32>, Vec<f32>), ImportError> {
        if u.nx != v.nx || u.ny != v.ny || u.values.len() != v.values.len() {
            return Err(ImportError::PlaneMismatch);
        }
        match self {
            Self::Identity => Ok((u.values.clone(), v.values.clone())),
            Self::Angles { sin, cos }
                if sin.len() == u.values.len() && cos.len() == u.values.len() =>
            {
                let mut u_earth = Vec::with_capacity(u.values.len());
                let mut v_earth = Vec::with_capacity(v.values.len());
                for (((&u, &v), &sin), &cos) in u.values.iter().zip(&v.values).zip(sin).zip(cos) {
                    u_earth.push(u.mul_add(cos, -v * sin));
                    v_earth.push(u.mul_add(sin, v * cos));
                }
                Ok((u_earth, v_earth))
            }
            Self::Angles { .. } => Err(ImportError::PlaneMismatch),
        }
    }

    fn rotate_f64_levels_in_place(
        &self,
        u: &mut [f64],
        v: &mut [f64],
        cells: usize,
    ) -> Result<(), ImportError> {
        if cells == 0 || u.len() != v.len() || u.len() % cells != 0 {
            return Err(ImportError::PlaneMismatch);
        }
        match self {
            Self::Identity => Ok(()),
            Self::Angles { sin, cos } if sin.len() == cells && cos.len() == cells => {
                for index in 0..u.len() {
                    let cell = index % cells;
                    let old_u = u[index];
                    let old_v = v[index];
                    let sin = f64::from(sin[cell]);
                    let cos = f64::from(cos[cell]);
                    u[index] = old_u.mul_add(cos, -old_v * sin);
                    v[index] = old_u.mul_add(sin, old_v * cos);
                }
                Ok(())
            }
            Self::Angles { .. } => Err(ImportError::PlaneMismatch),
        }
    }
}

fn read_wrf_wind_rotation(
    src: &PlaneSource,
    projection: Option<&GridProjection>,
    cells: usize,
) -> Result<Option<WrfWindRotation>, ImportError> {
    let sin = read_first_2d(src, "SINALPHA")?;
    let cos = read_first_2d(src, "COSALPHA")?;
    let (sin, cos) = match (sin, cos) {
        (None, None) if matches!(projection, Some(GridProjection::Geographic)) => {
            return Ok(Some(WrfWindRotation::Identity));
        }
        (None, None) | (Some(_), None) | (None, Some(_)) => return Ok(None),
        (Some(sin), Some(cos)) => (sin, cos),
    };
    if sin.nx != cos.nx
        || sin.ny != cos.ny
        || sin.values.len() != cells
        || cos.values.len() != cells
    {
        return Ok(None);
    }

    // Normalize small roundoff away. A malformed or missing angle at one
    // cell poisons only that column's vector products instead of rotating it
    // with arbitrary numbers or failing otherwise usable thermodynamic data.
    let mut normalized_sin = Vec::with_capacity(cells);
    let mut normalized_cos = Vec::with_capacity(cells);
    let mut valid = 0usize;
    for (&sin, &cos) in sin.values.iter().zip(&cos.values) {
        let norm = sin.hypot(cos);
        if sin.is_finite() && cos.is_finite() && norm.is_finite() && norm > 1.0e-6 {
            normalized_sin.push(sin / norm);
            normalized_cos.push(cos / norm);
            valid += 1;
        } else {
            normalized_sin.push(f32::NAN);
            normalized_cos.push(f32::NAN);
        }
    }
    if valid == 0 {
        return Ok(None);
    }
    Ok(Some(WrfWindRotation::Angles {
        sin: normalized_sin,
        cos: normalized_cos,
    }))
}

/// One file's 2-D plane source. `netcrust` always provides the metadata
/// (variable existence, dims, shapes, units); when `wrf` is set (raw wrfout)
/// the plane DATA is decoded through wrf-core's single-timestep reader
/// instead of netcrust's `hdf5-reader` path, which was measured at ~10.3 s
/// and ~8M minor page faults per 800×800 plane on compressed 250 m wrfouts
/// (allocation churn: docs/wrf-import-large-grids.md) versus tens of ms
/// for wrf-core reading the same slice.
pub(crate) struct PlaneSource<'a> {
    nc: &'a NcFile,
    wrf: Option<&'a WrfFile>,
    time_index: usize,
    /// Planes decoded via wrf-core (the fast path actually engaged).
    wrf_reads: Cell<usize>,
    /// WRF-layout planes wrf-core failed to read, served by netcrust instead.
    netcrust_fallbacks: RefCell<Vec<String>>,
}

impl<'a> PlaneSource<'a> {
    pub(crate) fn new(nc: &'a NcFile, wrf: Option<&'a WrfFile>, time_index: usize) -> Self {
        Self {
            nc,
            wrf,
            time_index,
            wrf_reads: Cell::new(0),
            netcrust_fallbacks: RefCell::new(Vec::new()),
        }
    }

    fn netcrust_only(nc: &'a NcFile, time_index: usize) -> Self {
        Self::new(nc, None, time_index)
    }
}

fn read_first_2d_any(src: &PlaneSource, names: &[&str]) -> Result<Plane2D, ImportError> {
    for name in names {
        if let Some(plane) = read_first_2d(src, name)? {
            return Ok(plane);
        }
    }
    Err(ImportError::MissingAny(
        names.iter().map(|value| value.to_string()).collect(),
    ))
}

fn read_first_2d(src: &PlaneSource, name: &str) -> Result<Option<Plane2D>, ImportError> {
    // Fast path: for the `[Time, …, ny, nx]` record layout, decode the
    // selected record through wrf-core. Identical value positions: both paths
    // yield the record's `…, ny, nx` values, and `plane_from_last_record`
    // applies the same tail-plane + f32 narrowing to either. Anything else
    // (no Time dim, rank < 3, unexpected length, wrf-core read error) keeps
    // the legacy netcrust read byte-for-byte. Names the netcrust metadata
    // listing misses (netcdf-reader index gaps: CONUS-II wrf2d) skip the
    // fast path and fall through to the netcrust read, whose raw-HDF5
    // by-name fallback can still resolve them.
    if let (Some(wrf), Some(var)) = (src.wrf, src.nc.variable(name)) {
        let dims = var.dimensions();
        let shape = var.shape();
        if dims.len() >= 3 && shape.len() == dims.len() && is_time_dimension(dims[0].name()) {
            let expected = shape[1..]
                .iter()
                .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
            let outcome = match expected {
                None => Err("dimension product overflows usize".to_string()),
                Some(expected) => match wrf.read_var(name, src.time_index) {
                    Ok(values) if values.len() == expected => Ok(values),
                    Ok(values) => Err(format!("expected {expected} values, got {}", values.len())),
                    Err(err) => Err(err.to_string()),
                },
            };
            match outcome {
                Ok(values) => {
                    src.wrf_reads.set(src.wrf_reads.get() + 1);
                    return plane_from_last_record(name, &shape[1..], &values);
                }
                // Carry the WHY: the fallback summary line is how a plane
                // that is genuinely only reachable via netcrust gets reported.
                Err(reason) => src
                    .netcrust_fallbacks
                    .borrow_mut()
                    .push(format!("{name} ({reason})")),
            }
        }
    }
    read_first_2d_netcrust(src.nc, name, src.time_index)
}

/// Legacy netcrust plane read: the pre-fast-path implementation, kept intact
/// as the fallback for non-wrfout files (and the reference side of the
/// value-identity fixture test).
///
/// Listing-missed names: netcrust's `netcdf-reader` metadata index can drop
/// datasets (measured on the real CONUS-II wrf2d file: 5 of 192,
/// U10/MUCAPE/SRH03/SWUPT/ACEDIR), so a name absent from the LISTING is
/// attempted through the selected-record reader, whose by-name HDF5 fallback
/// supports arbitrary records too. Dense-group internal-node records are now
/// included by the vendored HDF5 v2 B-tree traversal, so those exact-name
/// reads resolve as well. A name absent from both readers stays `None`;
/// listed variables propagate errors.
fn read_first_2d_netcrust(
    nc: &NcFile,
    name: &str,
    time_index: usize,
) -> Result<Option<Plane2D>, ImportError> {
    let listed = nc.variable(name).is_some() || nc.has_hdf5_dataset(name);
    let array = match read_array_f64_record_or_all(nc, name, time_index) {
        Ok(array) => array,
        Err(_) if !listed => return Ok(None),
        Err(err) => return Err(err),
    };
    plane_from_last_record(name, array.shape(), array.values())
}

/// Build a [`Plane2D`] from the LAST `ny * nx` values of a decoded record,
/// with the non-finite → NaN f32 narrowing both read paths share. `shape` is
/// the decoded record's shape (`…, ny, nx`); a leading level dimension means
/// the deepest plane wins: the tail-of-record convention the netcrust read
/// has always used.
fn plane_from_last_record(
    name: &str,
    shape: &[usize],
    values: &[f64],
) -> Result<Option<Plane2D>, ImportError> {
    if shape.len() < 2 {
        return Ok(None);
    }
    let ny = shape[shape.len() - 2];
    let nx = shape[shape.len() - 1];
    let cells = nx
        .checked_mul(ny)
        .ok_or_else(|| ImportError::BadShape(name.to_string(), shape.to_vec()))?;
    if values.len() < cells {
        return Err(ImportError::BadShape(name.to_string(), shape.to_vec()));
    }
    let offset = values.len() - cells;
    Ok(Some(Plane2D {
        nx,
        ny,
        values: values[offset..]
            .iter()
            .map(|value| {
                if value.is_finite() {
                    *value as f32
                } else {
                    f32::NAN
                }
            })
            .collect(),
    }))
}

pub(crate) fn variable_units(nc: &NcFile, name: &str) -> Option<String> {
    nc.variable(name)
        .and_then(|variable| {
            variable
                .attribute("units")
                .and_then(|attribute| attribute.as_string())
                .map(str::to_string)
        })
        .or_else(|| nc.hdf5_dataset_attribute_string(name, "units"))
}

fn combine_same_grid(
    a: &Plane2D,
    b: &Plane2D,
    f: impl Fn(f32, f32) -> f32,
) -> Result<Vec<f32>, ImportError> {
    ensure_same_grid(a, b)?;
    Ok(a.values
        .iter()
        .zip(&b.values)
        .map(|(&a, &b)| {
            if a.is_finite() && b.is_finite() {
                f(a, b)
            } else {
                f32::NAN
            }
        })
        .collect())
}

fn combine_precip(
    rainc: &Plane2D,
    rainnc: &Plane2D,
    rainsh: Option<&Plane2D>,
) -> Result<Vec<f32>, ImportError> {
    ensure_same_grid(rainc, rainnc)?;
    if let Some(rainsh) = rainsh {
        ensure_same_grid(rainc, rainsh)?;
    }
    Ok((0..rainc.values.len())
        .map(|idx| {
            let mut value = 0.0;
            let mut valid = true;
            for plane in [Some(rainc), Some(rainnc), rainsh].into_iter().flatten() {
                let v = plane.values[idx];
                if v.is_finite() {
                    value += v;
                } else {
                    valid = false;
                }
            }
            if valid { value } else { f32::NAN }
        })
        .collect())
}

fn derive_dewpoint_k(t2: &Plane2D, q2: &Plane2D, psfc: &Plane2D) -> Result<Vec<f32>, ImportError> {
    ensure_same_grid(t2, q2)?;
    ensure_same_grid(t2, psfc)?;
    Ok((0..t2.values.len())
        .map(|idx| dewpoint_from_q_psfc(q2.values[idx], psfc.values[idx]))
        .collect())
}

fn derive_relative_humidity_percent(
    t2: &Plane2D,
    q2: &Plane2D,
    psfc: &Plane2D,
) -> Result<Vec<f32>, ImportError> {
    ensure_same_grid(t2, q2)?;
    ensure_same_grid(t2, psfc)?;
    Ok((0..t2.values.len())
        .map(|idx| {
            relative_humidity_from_t_q_psfc(t2.values[idx], q2.values[idx], psfc.values[idx])
        })
        .collect())
}

/// Vapour pressure (Pa) from a water-vapour MIXING RATIO `w` (kg per kg of
/// dry air) and pressure (Pa): `e = w p / (0.622 + w)`.
///
/// WRF's `Q2` and `QVAPOR` are mixing ratios, so every wrfout humidity
/// diagnostic in this import goes through this one helper. The 2 m dewpoint
/// and RH used to apply the specific-humidity form `q p / (0.622 + 0.378 q)`
/// to `Q2`, which read Td2 about 0.1 K moist over dry land and up to 0.25 K
/// over the warm Gulf (lane/dewpoint-mixing-ratio). A specific humidity `q`
/// (GRIB SPFH) must be converted with `w = q / (1 - q)` before it gets here.
fn vapour_pressure_pa_from_mixing_ratio(w: f64, p_pa: f64) -> f64 {
    w * p_pa / (0.622 + w)
}

fn dewpoint_from_q_psfc(q: f32, p_pa: f32) -> f32 {
    if !q.is_finite() || !p_pa.is_finite() || q <= 0.0 || p_pa <= 0.0 {
        return f32::NAN;
    }
    let q = q as f64;
    let p = p_pa as f64;
    let e = vapour_pressure_pa_from_mixing_ratio(q, p).max(1.0);
    let ln = (e / 611.2).ln();
    let td_c = 243.5 * ln / (17.67 - ln);
    (td_c + 273.15) as f32
}

fn relative_humidity_from_t_q_psfc(t_k: f32, q: f32, p_pa: f32) -> f32 {
    if !t_k.is_finite() || !q.is_finite() || !p_pa.is_finite() || t_k <= 0.0 {
        return f32::NAN;
    }
    let e = vapour_pressure_pa_from_mixing_ratio(q as f64, p_pa as f64);
    let t_c = t_k as f64 - 273.15;
    let es = 611.2 * (17.67 * t_c / (t_c + 243.5)).exp();
    (100.0 * e / es).clamp(0.0, 100.0) as f32
}

#[cfg(test)]
mod mixing_ratio_humidity_tests {
    use super::*;

    fn bolton_dewpoint_k(e_pa: f64) -> f64 {
        let ln = (e_pa / 611.2).ln();
        243.5 * ln / (17.67 - ln) + 273.15
    }

    /// Specific-humidity vapour pressure, the form HRRR's own DPT is
    /// consistent with when fed its 2 m SPFH.
    fn vapour_pressure_pa_from_specific_humidity(q: f64, p_pa: f64) -> f64 {
        q * p_pa / (0.622 + 0.378 * q)
    }

    /// (specific humidity, pressure Pa, dewpoint K) worked independently
    /// (Python, Bolton over water, e from q p / (0.622 + 0.378 q)).
    const SPECIFIC_CASES: [(f64, f64, f64); 3] = [
        (0.0150, 100_000.0, 293.5113271127926),
        (0.0040, 85_000.0, 271.58785978403705),
        (0.0200, 101_500.0, 298.449408723055),
    ];

    #[test]
    fn specific_humidity_and_its_mixing_ratio_give_one_dewpoint() {
        for (q, p, td_ref) in SPECIFIC_CASES {
            let td_sh = bolton_dewpoint_k(vapour_pressure_pa_from_specific_humidity(q, p));
            assert!((td_sh - td_ref).abs() < 0.02, "SPFH path {td_sh} vs {td_ref}");
            // The same air as a mixing ratio, through the import's helper.
            let w = q / (1.0 - q);
            let td_w = bolton_dewpoint_k(vapour_pressure_pa_from_mixing_ratio(w, p));
            assert!((td_w - td_sh).abs() < 1e-9, "f64 mixing-ratio path {td_w} vs {td_sh}");
            // And through the f32 Q2/PSFC product function (f32 in and out).
            let td_product = dewpoint_from_q_psfc(w as f32, p as f32) as f64;
            assert!((td_product - td_ref).abs() < 1e-3, "Td2 product {td_product} vs {td_ref}");
        }
    }

    /// Real HRRR rows (hrrr.20260802 t18z wrfsfcf01, six random CONUS
    /// cells): 2 m SPFH (kg/kg), surface PRES (Pa), 2 m DPT (K), exactly as
    /// decoded. HRRR packs DPT in steps of 1/16 K and SPFH in steps of 1e-5,
    /// so a single cell can only agree to within one DPT packing step; over
    /// the whole grid Bolton on the specific-humidity form reproduces DPT to
    /// a mean of -0.010 K (the mixing-ratio form on SPFH: -0.20 K).
    const HRRR_ROWS: [(f64, f64, f64); 6] = [
        (0.02012, 101_770.0, 298.63739013671875),
        (0.01919, 101_930.0, 297.82489013671875),
        (0.0046, 72_630.0, 271.32489013671875),
        (0.00869, 102_830.0, 285.44989013671875),
        (0.01471, 94_890.0, 292.38739013671875),
        (0.01902, 99_860.0, 297.38739013671875),
    ];

    #[test]
    fn hrrr_spfh_reproduces_hrrr_dpt_and_q2_path_agrees() {
        let mut signed = 0.0;
        for (q, p, dpt) in HRRR_ROWS {
            let td_sh = bolton_dewpoint_k(vapour_pressure_pa_from_specific_humidity(q, p));
            assert!((td_sh - dpt).abs() < 1.0 / 16.0, "HRRR SPFH {q} -> {td_sh} vs DPT {dpt}");
            signed += td_sh - dpt;
            // The same air handed over as a WRF Q2 (mixing ratio) gives the
            // same dewpoint through the product function.
            let w = q / (1.0 - q);
            let td_q2 = dewpoint_from_q_psfc(w as f32, p as f32) as f64;
            assert!((td_q2 - td_sh).abs() < 1e-3, "Q2 path {td_q2} vs SPFH path {td_sh}");
        }
        let mean = signed / HRRR_ROWS.len() as f64;
        assert!(mean.abs() < 0.02, "mean HRRR SPFH->DPT error {mean}");
    }

    #[test]
    fn specific_humidity_form_on_q2_reads_measurably_moist() {
        // The defect this lane fixes: Q2 = 0.015 kg/kg at 1000 hPa.
        let (w, p) = (0.015_f64, 100_000.0_f64);
        let old = bolton_dewpoint_k(vapour_pressure_pa_from_specific_humidity(w, p));
        let new = dewpoint_from_q_psfc(w as f32, p as f32) as f64;
        assert!(old - new > 0.2, "old {old} new {new}");
        assert!((new - bolton_dewpoint_k(w * p / (0.622 + w))).abs() < 1e-3);
    }

    #[test]
    fn relative_humidity_uses_the_mixing_ratio_vapour_pressure() {
        let (t_k, w, p) = (303.15_f32, 0.015_f32, 100_000.0_f32);
        let e = (w as f64) * (p as f64) / (0.622 + w as f64);
        let t_c = t_k as f64 - 273.15;
        let es = 611.2 * (17.67 * t_c / (t_c + 243.5)).exp();
        let expected = (100.0 * e / es) as f32;
        assert_eq!(relative_humidity_from_t_q_psfc(t_k, w, p), expected);
        let old = (100.0 * vapour_pressure_pa_from_specific_humidity(w as f64, p as f64) / es) as f32;
        assert!(old - expected > 0.5, "old {old} new {expected}");
    }

    #[test]
    fn three_d_dewpoint_shares_the_helper_bit_for_bit() {
        for (q, p) in [(0.012_f64, 92_000.0_f64), (1.0e-5, 20_000.0)] {
            let direct = {
                let e = (q * p / (0.622 + q)).max(1.0);
                let ln = (e / 611.2).ln();
                243.5 * ln / (17.67 - ln) + 273.15
            };
            assert_eq!(dewpoint_k_from_q_p(q, p).to_bits(), direct.to_bits());
        }
    }
}

fn ensure_same_grid(a: &Plane2D, b: &Plane2D) -> Result<(), ImportError> {
    if a.nx == b.nx && a.ny == b.ny && a.values.len() == b.values.len() {
        Ok(())
    } else {
        Err(ImportError::PlaneMismatch)
    }
}

/// Match wrf-python's polar stereographic convention: the projection pole is
/// selected from `TRUELAT1`, not `CEN_LAT` (which may describe a nested domain
/// centered across the equator from its projection pole).
pub(crate) fn wrf_polar_uses_south_pole(truelat1: f64) -> bool {
    truelat1 < 0.0
}

/// wrf-python treats an absent, non-finite, or out-of-range second Lambert
/// standard parallel as a one-standard-parallel projection.
pub(crate) fn normalize_lambert_truelat2(truelat1: f64, truelat2: Option<f64>) -> f64 {
    truelat2
        .filter(|value| value.is_finite() && value.abs() <= 90.0)
        .unwrap_or(truelat1)
}

/// WRF Mercator uses STAND_LON and defaults it to zero. CEN_LON describes the
/// domain center and must not be substituted for a missing standard longitude.
pub(crate) fn wrf_mercator_central_longitude(stand_lon: Option<f64>) -> f64 {
    stand_lon.unwrap_or(0.0)
}

/// Whether MAP_PROJ=6 is ordinary (unrotated) latitude/longitude. A partially
/// specified or non-default pole is intentionally not called Geographic:
/// rw-store cannot encode a rotated pole, while the actual XLAT/XLONG arrays
/// still let the curvilinear grid render correctly when projection is `None`.
pub(crate) fn wrf_latlon_is_unrotated(pole_lat: Option<f64>, pole_lon: Option<f64>) -> bool {
    matches!((pole_lat, pole_lon), (None, None))
        || matches!((pole_lat, pole_lon), (Some(lat), Some(lon)) if lat == 90.0 && lon == 0.0)
}

fn wrf_projection(nc: &NcFile) -> Option<GridProjection> {
    let map_proj = global_attr_f64(nc, "MAP_PROJ")? as i32;
    match map_proj {
        1 => {
            let truelat1 = global_attr_f64(nc, "TRUELAT1").unwrap_or(30.0);
            Some(GridProjection::LambertConformal {
                standard_parallel_1_deg: truelat1,
                standard_parallel_2_deg: normalize_lambert_truelat2(
                    truelat1,
                    global_attr_f64(nc, "TRUELAT2"),
                ),
                central_meridian_deg: global_attr_f64(nc, "STAND_LON")
                    .or_else(|| global_attr_f64(nc, "CEN_LON"))
                    .unwrap_or(0.0),
            })
        }
        2 => {
            let truelat1 = global_attr_f64(nc, "TRUELAT1").unwrap_or(60.0);
            Some(GridProjection::PolarStereographic {
                true_latitude_deg: truelat1,
                central_meridian_deg: global_attr_f64(nc, "STAND_LON")
                    .or_else(|| global_attr_f64(nc, "CEN_LON"))
                    .unwrap_or(0.0),
                south_pole_on_projection_plane: wrf_polar_uses_south_pole(truelat1),
            })
        }
        3 => Some(GridProjection::Mercator {
            latitude_of_true_scale_deg: global_attr_f64(nc, "TRUELAT1").unwrap_or(0.0),
            central_meridian_deg: wrf_mercator_central_longitude(global_attr_f64(nc, "STAND_LON")),
        }),
        6 if wrf_latlon_is_unrotated(
            global_attr_f64(nc, "POLE_LAT"),
            global_attr_f64(nc, "POLE_LON"),
        ) =>
        {
            Some(GridProjection::Geographic)
        }
        6 => None,
        // Unknown WRF projections have no trustworthy native renderer
        // mapping. Keep the exact curvilinear XLAT/XLONG grid and omit a
        // projection claim, matching the full/raw and Formula Lab paths.
        _ => None,
    }
}

fn global_attr_f64(nc: &NcFile, name: &str) -> Option<f64> {
    nc.attribute(name).and_then(|attr| attr.as_f64())
}

/// The first WRF-shaped timestamp (`YYYY-MM-DD_HH:MM:SS` or
/// `YYYY-MM-DD_HH_MM_SS`) in a path's file name, exactly as spelled there.
fn wrf_timestamp_in_file_name(path: &Path) -> Option<&str> {
    let name = path.file_name()?.to_str()?;
    let bytes = name.as_bytes();
    for start in 0..bytes.len().saturating_sub(18) {
        let slice = name.get(start..start + 19)?;
        if is_wrf_timestamp(slice) {
            return Some(slice);
        }
    }
    None
}

/// The valid time of a single-record file with no `Times` and no CF time
/// coordinate, read from the WRF timestamp in its file name.
///
/// The file's `START_DATE` is never used as the valid time: it is the run's
/// start, so a forecast frame labelled with it is drawn as the initial time
/// (an hour-12 frame valid at 00Z read "Valid 12Z F000" with no warning).
/// A name with no timestamp, or with one that is not a real UTC time, is
/// refused by name instead.
fn valid_unix_from_file_name(path: &Path, context: &str) -> Result<i64, ImportError> {
    let Some(stamp) = wrf_timestamp_in_file_name(path) else {
        return Err(ImportError::TimeAxis(format!(
            "{} {context} and no WRF timestamp (YYYY-MM-DD_HH:MM:SS or YYYY-MM-DD_HH_MM_SS) in its file name; its START_DATE is the run's start, not this frame's valid time, so the frame is refused rather than labelled with the start time",
            path.display()
        )));
    };
    parse_utc_timestamp(stamp).ok_or_else(|| {
        ImportError::TimeAxis(format!(
            "{} {context}, and the timestamp {stamp:?} in its file name is not a valid UTC time; the frame is refused rather than labelled with another time",
            path.display()
        ))
    })
}

fn is_wrf_timestamp(value: &str) -> bool {
    let b = value.as_bytes();
    b.len() == 19
        && b[4] == b'-'
        && b[7] == b'-'
        && b[10] == b'_'
        && matches!(b[13], b':' | b'_')
        && matches!(b[16], b':' | b'_')
        && b.iter()
            .enumerate()
            .all(|(idx, byte)| matches!(idx, 4 | 7 | 10 | 13 | 16) || byte.is_ascii_digit())
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("local file")
        .to_string()
}

fn writer_build() -> &'static str {
    concat!(
        "rusty-weather-wrf-local-import-",
        env!("CARGO_PKG_VERSION"),
        "-science_v3_vpd_2m"
    )
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

/// Build isobaric sounding volumes for one selected WRF time via wrf-core.
/// `file` is the handle `import_paths` opened for the fast 2D reads; `None`
/// (plain NetCDF wrf-core can't open)
/// yields no volumes so the 2D import still succeeds. The third element
/// carries the human-readable reason when the volumes degraded on a file
/// wrf-core DID open, or when their heights took the named mass-level
/// fallback ([`crate::wrf_volumes::mass_level_heights_note`]).
fn read_iso_volumes(
    file: Option<&WrfFile>,
    time_index: usize,
    progress: &mut dyn FnMut(String),
) -> (Vec<IsoVolume>, Option<SurfaceFallback>, Option<String>) {
    let Some(file) = file else {
        // Plain NetCDF with no WRF 3D state, a 2D-only import, not a note.
        return (Vec::new(), None, None);
    };
    let cells = file.nx.saturating_mul(file.ny);
    // Same per-field panic isolation as the heavy path's `compute_var`: a
    // wrf-core panic on a pathological grid must degrade to "no soundings",
    // not unwind the rw-ui-local-import worker and lose the whole import.
    let result = crate::wrf_process::isolate_panics("isobaric volumes", || {
        build_iso_volumes(file, time_index, cells, progress)
    });
    match result {
        // The light path publishes no chart-level selector planes -- the
        // full science import (wrf_process) owns those; here only the
        // sounding volumes and the surface fallback are consumed.
        Ok((volumes, _chart_volumes, surface, height_note)) => {
            (volumes, Some(surface), height_note)
        }
        Err(err) => (
            Vec::new(),
            None,
            Some(format!("isobaric sounding volumes unavailable: {err}")),
        ),
    }
}

/// Add any skew-T surface field the netcrust 2D read did not provide, from the
/// wrf-core lowest-model-level fallback, so a split `wrf3d` file (which omits
/// `PSFC`) still sounds. Fields already present are kept; planes that don't
/// match the hour grid are skipped.
fn fill_missing_surface(fields: &mut ImportedWrfFields, surface: SurfaceFallback) {
    let cells = fields.grid.shape.len();
    let entries: [(&str, &str, FieldSelector, &str, Vec<f32>); 5] = [
        (
            "surface_pressure",
            "approx_surface_pressure",
            FieldSelector::surface(CanonicalField::Pressure),
            "Pa",
            surface.surface_pressure_pa,
        ),
        (
            "temperature_2m",
            "approx_temperature_2m",
            FieldSelector::surface(CanonicalField::Temperature),
            "K",
            surface.temperature_2m_k,
        ),
        (
            "dewpoint_2m",
            "approx_dewpoint_2m",
            FieldSelector::surface(CanonicalField::Dewpoint),
            "K",
            surface.dewpoint_2m_k,
        ),
        (
            "u_10m",
            "approx_u_10m",
            FieldSelector::surface(CanonicalField::UWind),
            "m/s",
            surface.u_10m,
        ),
        (
            "v_10m",
            "approx_v_10m",
            FieldSelector::surface(CanonicalField::VWind),
            "m/s",
            surface.v_10m,
        ),
    ];
    for (exact_name, name, selector, units, values) in entries {
        if values.len() != cells
            || fields
                .canonical
                .iter()
                .any(|(existing, _)| existing == exact_name)
        {
            continue;
        }
        if let Ok(field) = SelectedField2D::new(selector, units, fields.grid.clone(), values) {
            let field = match &fields.projection {
                Some(projection) => field.with_projection(projection.clone()),
                None => field,
            };
            fields.canonical.push((name.to_string(), field));
        }
    }
}

/// Build a soundable store hour from a POST-PROCESSED climate wrfout (NCAR
/// CONUS-I/II, GDEX): these ship derived `TK` (K), `Z` (m MSL), `P` (full
/// pressure, Pa) and staggered `U`/`V` instead of the raw `T`/`PB`/`PH`/`PHB`
/// the wrf-core reader needs, and carry no surface fields. Returns the
/// synthesized surface 2D fields + an optional `approx_*` severe/thermo suite
/// + the isobaric volumes (+ raw `wrf_*` planes for pure 2-D `wrf2d` surface
/// archive), or `None` if this isn't a post-processed WRF file (so the
/// caller falls back to the raw path). `progress` streams the stage messages
/// both import paths show in the dock.
fn is_postprocessed_wrf(nc: &NcFile) -> bool {
    nc.variable("TK").is_some()
        && nc.variable("Z").is_some()
        && nc.variable("P").is_some()
        && nc.variable("PB").is_none()
}

pub(crate) fn try_postprocessed_wrf(
    path: &Path,
    time_index: usize,
    compute_severe: bool,
    progress: &mut dyn FnMut(String),
) -> Result<Option<PostprocessedWrfHour>, ImportError> {
    // If netcrust can't open it at all, it's not our post-processed case,
    // let the caller's raw-wrfout path try instead of failing here.
    let Ok(nc) = netcrust::open(path) else {
        return Ok(None);
    };
    try_postprocessed_wrf_shared(&nc, path, time_index, compute_severe, progress)
}

/// Vertically destagger a `(nz+1) x cells` w-level field to `nz` mass levels
/// in place (mass level k = mean of staggered levels k and k+1), then
/// truncate. In-place forward iteration is safe (the write slot k is only
/// read again by iteration k itself) and avoids allocating a second
/// multi-hundred-MB buffer on CONUS-II grids.
fn destagger_z_to_mass_levels(
    values: &mut Vec<f64>,
    nz: usize,
    cells: usize,
) -> Result<(), ImportError> {
    let staggered_values = nz
        .checked_add(1)
        .and_then(|levels| levels.checked_mul(cells))
        .ok_or(ImportError::PlaneMismatch)?;
    let mass_values = nz.checked_mul(cells).ok_or(ImportError::PlaneMismatch)?;
    if values.len() != staggered_values {
        return Err(ImportError::PlaneMismatch);
    }
    for k in 0..nz {
        for i in 0..cells {
            let lo = values[k * cells + i];
            let hi = values[(k + 1) * cells + i];
            values[k * cells + i] = 0.5 * (lo + hi);
        }
    }
    values.truncate(mass_values);
    Ok(())
}

/// The eta-level stencil of a post-processed file whose Z is on the layer
/// interfaces: from ZNW, and ZNU when the file carries it (otherwise the
/// middle of each layer in ZNW, as WRF defines it).  `Err` names why the
/// file has none.
fn postprocessed_interface_stencil(
    nc: &NcFile,
    time_index: usize,
    nz: usize,
) -> Result<InterfaceStencil, String> {
    if nc.variable("ZNW").is_none() {
        return Err("the file carries Z on its layer interfaces but no eta levels (ZNW)".into());
    }
    let read = |name: &str| -> Result<Vec<f64>, String> {
        read_array_f64_record_or_all(nc, name, time_index)
            .map(|array| array.into_values())
            .map_err(|err| format!("read {name}: {err}"))
    };
    let eta_interface = read("ZNW")?;
    let stencil = if nc.variable("ZNU").is_some() {
        InterfaceStencil::new(&read("ZNU")?, &eta_interface)
    } else {
        InterfaceStencil::from_interfaces(&eta_interface)
    }
    .map_err(|err| format!("the file's eta levels: {err}"))?;
    if stencil.levels() != nz {
        return Err(format!(
            "the file's eta levels describe {} layers and its fields {nz}",
            stencil.levels()
        ));
    }
    Ok(stencil)
}

/// Everything one post-processed hour yields: the synthesized surface 2D
/// fields, the optional `approx_*` severe/thermo suite (written through the
/// derived-field slot), the isobaric sounding volumes, and (for the
/// 2-D-only `wrf2d` route) every mass-grid data plane as a raw `wrf_*`
/// field (same derived-slot convention as the raw-wrfout light import; the
/// 3-D route always returns this empty) -- and the notes the import must
/// carry, such as which heights the isobaric height volume was read from.
pub(crate) type PostprocessedWrfHour = (
    Vec<(String, SelectedField2D)>,
    Vec<crate::postproc_severe::SevereField>,
    Vec<IsoVolume>,
    Vec<RawField2D>,
    Vec<String>,
);

/// Post-processed climate-WRF routing rule: TRUE when the `TK` variable is a
/// single 2-D surface plane, i.e. after dropping one leading record
/// ("Time") dimension, exactly `[ny, nx]` remains. That is the CONUS-II
/// `wrf2d` surface-archive dialect (every data variable at the lowest model
/// level / surface, `(Time=1, ny, nx)`). `wrf3d`-style archives carry TK on
/// model levels (`[Time, nz, ny, nx]` or `[nz, ny, nx]`) and return FALSE so
/// the existing 3-D reader (including its staggered-Z destagger) is
/// untouched. The Time-squeeze mirrors what
/// the selected-record reader does on the read side.
fn postproc_tk_is_2d(dim_names: &[&str], shape: &[usize]) -> bool {
    if dim_names.len() != shape.len() {
        return false;
    }
    let squeezed_rank = if dim_names
        .first()
        .map(|name| is_time_dimension(name))
        .unwrap_or(false)
    {
        shape.len() - 1
    } else {
        shape.len()
    };
    squeezed_rank == 2
}

/// A `wrf2d`-style data variable for the post-processed 2-D route: a single
/// plane on the `ny x nx` mass grid (shaped `[Time, ny, nx]`, `[1, ny, nx]`,
/// or `[ny, nx]`) that is not a coordinate/bookkeeping variable (the raw
/// wrfout blocklist plus the lat/lon/time axis names the grid reader
/// consumes). Staggered planes and model-level stacks never match the shape
/// rule.
fn is_postproc_2d_data_plane(
    name: &str,
    dim_names: &[&str],
    shape: &[usize],
    ny: usize,
    nx: usize,
) -> bool {
    if !raw_wrf_variable_allowed(name) || is_coordinate_axis_name(name) {
        return false;
    }
    if dim_names.len() != shape.len() {
        return false;
    }
    match shape {
        [y, x] => *y == ny && *x == nx,
        [t, y, x] => (*t == 1 || is_time_dimension(dim_names[0])) && *y == ny && *x == nx,
        _ => false,
    }
}

/// Raw-HDF5 counterpart to [`is_postproc_2d_data_plane`]. Dataset names and
/// shapes survive even when the NetCDF-4 variable index omits an entry, but
/// dimension labels do not. A rank-three plane is therefore accepted only
/// for a singleton leading axis or when HDF5 explicitly marks that axis as
/// unlimited.
fn is_postproc_2d_hdf5_data_plane(
    name: &str,
    shape: &[u64],
    has_leading_record_axis: bool,
    ny: usize,
    nx: usize,
) -> bool {
    if !raw_wrf_variable_allowed(name) || is_coordinate_axis_name(name) {
        return false;
    }
    let (Ok(ny), Ok(nx)) = (u64::try_from(ny), u64::try_from(nx)) else {
        return false;
    };
    match shape {
        [y, x] => *y == ny && *x == nx,
        [t, y, x] => (*t == 1 || has_leading_record_axis) && *y == ny && *x == nx,
        _ => false,
    }
}

const CONUS_II_WRF2D_SHAPE: (usize, usize) = (1429, 1419);
const CONUS_II_WRF2D_REQUIRED_RAW_FIELDS: [&str; 5] = [
    "wrf_u10",
    "wrf_mucape",
    "wrf_srh03",
    "wrf_swupt",
    "wrf_acedir",
];
const CONUS_II_WRF2D_REQUIRED_CANONICAL_WINDS: [&str; 3] = ["u_10m", "v_10m", "wind_speed_10m"];

fn missing_conus_ii_wrf2d_fields<'a>(
    nx: usize,
    ny: usize,
    raw_names: impl IntoIterator<Item = &'a str>,
    canonical_names: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    if (nx, ny) != CONUS_II_WRF2D_SHAPE {
        return Vec::new();
    }
    let raw_names = raw_names.into_iter().collect::<HashSet<_>>();
    let canonical_names = canonical_names.into_iter().collect::<HashSet<_>>();
    CONUS_II_WRF2D_REQUIRED_RAW_FIELDS
        .into_iter()
        .filter(|name| !raw_names.contains(name))
        .chain(
            CONUS_II_WRF2D_REQUIRED_CANONICAL_WINDS
                .into_iter()
                .filter(|name| !canonical_names.contains(name)),
        )
        .map(str::to_string)
        .collect()
}

/// Coordinate-axis variable names the 2-D enumeration must skip (the grid
/// reader consumes these; they are not data planes). The uppercase WRF forms
/// (XLAT/XLONG/…) are already on `raw_wrf_variable_allowed`'s blocklist.
fn is_coordinate_axis_name(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "lat" | "lon" | "latitude" | "longitude" | "time" | "times" | "xtime"
    )
}

/// Build the store hour for a PURE 2-D post-processed surface archive
/// (CONUS-II GDEX `wrf2d`: `Time(1)` + ~190 single-plane data variables, no
/// model-level stacks, owner-reported "bad shape for variable TK" when the
/// 3-D reader claimed one). Yields the canonical surface suite (these files
/// ship T2/Q2/PSFC/U10/V10/WSPD10MAX) plus EVERY mass-grid data plane as a
/// raw `wrf_*` field through the same derived-slot store-write convention
/// the raw-wrfout light import uses, so picker labels and Solar fallback
/// styles resolve identically. No isobaric volumes and no computed severe
/// suite: nothing 3-D exists to build them from; the files carry their own
/// pre-computed severe planes (SBCAPE/MUCAPE/SRH01/…), which land as raw
/// fields. Memory stays flat: one netcrust f64 record (~16 MB on the
/// 1419 x 1429 CONUS-II grid) is narrowed to f32 and dropped per plane.
fn postprocessed_wrf2d_hour(
    nc: &NcFile,
    path: &Path,
    time_index: usize,
    progress: &mut dyn FnMut(String),
) -> Result<PostprocessedWrfHour, ImportError> {
    // Same netcrust-only source as the 3-D post-processed route: wrf-core
    // cannot open these files (no raw T/PB).
    let src = PlaneSource::netcrust_only(nc, time_index);
    let lat = read_first_2d_any(&src, &["XLAT", "XLAT_M", "lat", "latitude"])?;
    let lon = read_first_2d_any(&src, &["XLONG", "XLONG_M", "lon", "longitude"])?;
    if lat.nx != lon.nx || lat.ny != lon.ny || lat.values.len() != lon.values.len() {
        return Err(ImportError::GridMismatch(path.to_path_buf()));
    }
    let (nx, ny) = (lat.nx, lat.ny);
    let shape = GridShape::new(nx, ny)?;
    let grid = LatLonGrid::new(shape, lat.values, lon.values)?;
    let projection = wrf_projection(nc);

    let mut canonical = Vec::new();
    push_canonical_surface_fields(&mut canonical, &src, &grid, &projection)?;
    // The store writer needs at least one extracted 2-D field to carry the
    // hour grid. Real wrf2d archives ship PSFC/T2/U10/…, so this fallback is
    // for pathological surface archives only: lowest-model-level P (always
    // present, it is part of the post-processed gate) as the
    // surface-pressure proxy, the same approximation the 3-D route documents
    // for its parcel state.
    if canonical.is_empty() {
        push_direct(
            &mut canonical,
            &src,
            &grid,
            projection.clone(),
            "P",
            "approx_surface_pressure",
            FieldSelector::surface(CanonicalField::Pressure),
            Some("Pa"),
        )?;
    }
    if canonical.is_empty() {
        return Err(ImportError::NoFields(path.to_path_buf()));
    }

    // Every mass-grid data plane, read-narrow-drop, one at a time. A count
    // line every few planes keeps the dock's progress live without spamming
    // the channel (~190 variables on a real wrf2d file).
    //
    // The raw HDF5 walk is intentional. netcdf-reader's index omitted five
    // real CONUS-II datasets (U10/MUCAPE/SRH03/SWUPT/ACEDIR); hdf5-reader's
    // dense-group B-tree traversal now includes internal-node records, and
    // this second metadata source ensures those recovered datasets enter the
    // import plan instead of remaining reachable only by an exact-name read.
    let variables = nc.variables()?;
    let mut planned = variables
        .iter()
        .filter(|var| {
            let names: Vec<&str> = var.dimensions().iter().map(|dim| dim.name()).collect();
            is_postproc_2d_data_plane(var.name(), &names, &var.shape(), ny, nx)
        })
        .map(|var| var.name().to_string())
        .collect::<Vec<_>>();
    let mut planned_names = planned.iter().cloned().collect::<HashSet<_>>();
    for dataset in nc.hdf5_root_datasets()? {
        if is_postproc_2d_hdf5_data_plane(
            dataset.name(),
            dataset.shape(),
            dataset.has_leading_record_axis(),
            ny,
            nx,
        ) && planned_names.insert(dataset.name().to_string())
        {
            planned.push(dataset.name().to_string());
        }
    }
    let total = planned.len();
    progress(format!("reading {total} 2-D surface planes"));
    let mut seen = HashSet::<String>::new();
    let mut raw_2d = Vec::new();
    for (index, wrf_name) in planned.iter().enumerate() {
        if index % 10 == 0 {
            progress(format!(
                "reading 2-D surface plane {}/{total} ({wrf_name})",
                index + 1
            ));
        }
        let Some(plane) = read_first_2d(&src, wrf_name)? else {
            continue;
        };
        if plane.nx != nx || plane.ny != ny {
            continue;
        }
        let name = format!("wrf_{}", sanitize_store_var_name(wrf_name));
        if name == "wrf_" || !seen.insert(name.clone()) {
            continue;
        }
        raw_2d.push(RawField2D {
            name,
            units: variable_units(nc, wrf_name).unwrap_or_else(|| "1".to_string()),
            values: plane.values,
        });
    }
    let missing = missing_conus_ii_wrf2d_fields(
        nx,
        ny,
        raw_2d.iter().map(|field| field.name.as_str()),
        canonical.iter().map(|(name, _)| name.as_str()),
    );
    if !missing.is_empty() {
        return Err(ImportError::IncompleteWrf2d {
            path: path.to_path_buf(),
            missing,
        });
    }
    progress(format!(
        "read {} 2-D surface planes ({} canonical fields)",
        raw_2d.len(),
        canonical.len()
    ));

    Ok((canonical, Vec::new(), Vec::new(), raw_2d, Vec::new()))
}

/// [`try_postprocessed_wrf`] against an already-open netcrust handle, so the
/// light import's per-file loop pays `netcrust::open`'s eager NetCDF-4
/// metadata indexing once, not once per stage (~57 s per open on a 2 GB
/// compressed 250 m wrfout: docs/wrf-import-large-grids.md).
pub(crate) fn try_postprocessed_wrf_shared(
    nc: &NcFile,
    path: &Path,
    time_index: usize,
    compute_severe: bool,
    progress: &mut dyn FnMut(String),
) -> Result<Option<PostprocessedWrfHour>, ImportError> {
    let is_postprocessed = is_postprocessed_wrf(nc);
    if !is_postprocessed {
        return Ok(None);
    }
    let declared_shape = netcdf_grid_shape(nc, path)?;
    GridShape::new(declared_shape.0, declared_shape.1)?;
    // CONUS-II `wrf2d` surface archives pass the TK/Z/P gate too, but carry
    // every variable as a SINGLE lowest-model-level / surface plane: the 3-D
    // reader below would fail on them ("bad shape for variable TK",
    // owner-reported). Route them to the 2-D-only import instead.
    let tk_is_2d = nc
        .variable("TK")
        .map(|var| {
            let names: Vec<&str> = var.dimensions().iter().map(|dim| dim.name()).collect();
            postproc_tk_is_2d(&names, &var.shape())
        })
        .unwrap_or(false);
    if tk_is_2d {
        return postprocessed_wrf2d_hour(nc, path, time_index, progress).map(Some);
    }

    // Post-processed climate files stay entirely on netcrust: wrf-core can't
    // open them (no raw T/PB), so there is no fast plane path here.
    let src = PlaneSource::netcrust_only(nc, time_index);
    let lat = read_first_2d_any(&src, &["XLAT", "XLAT_M", "lat", "latitude"])?;
    let lon = read_first_2d_any(&src, &["XLONG", "XLONG_M", "lon", "longitude"])?;
    if lat.nx != lon.nx || lat.ny != lon.ny {
        return Err(ImportError::GridMismatch(path.to_path_buf()));
    }
    let (nx, ny) = (lat.nx, lat.ny);
    let cells = nx
        .checked_mul(ny)
        .ok_or_else(|| ImportError::BadShape("grid".to_string(), vec![ny, nx]))?;
    let shape = GridShape::new(nx, ny)?;
    let grid = LatLonGrid::new(shape, lat.values, lon.values)?;
    let projection = wrf_projection(nc);

    // 3D mass-point state. `read3d` verifies the horizontal shape and returns
    // the level count. `into_values` hands back the decoded buffer without a
    // copy, each of these is `nz * cells * 8` bytes (hundreds of MB on a
    // CONUS-II grid), so `values().to_vec()` would double the transient cost.
    let read3d = |name: &str| -> Result<(Vec<f64>, usize), ImportError> {
        let array = read_array_f64_record_or_all(nc, name, time_index)?;
        let s = array.shape().to_vec();
        if s.len() != 3 || s[1] != ny || s[2] != nx {
            return Err(ImportError::BadShape(name.to_string(), s));
        }
        let nz = s[0];
        Ok((array.into_values(), nz))
    };
    // Read only TK's small metadata first. The aggregate working-set gate
    // must know nz, but it must run before any multi-hundred-MB 3-D value
    // buffer is decoded or allocated.
    let tk_var = nc
        .variable("TK")
        .ok_or_else(|| ImportError::MissingAny(vec!["TK".to_string()]))?;
    let tk_shape = tk_var.shape();
    let tk_dims = tk_var.dimensions();
    let tk_data_shape = if tk_dims
        .first()
        .is_some_and(|dimension| is_time_dimension(dimension.name()))
    {
        tk_shape.get(1..).unwrap_or_default()
    } else {
        tk_shape.as_slice()
    };
    if tk_data_shape.len() != 3 || tk_data_shape[1] != ny || tk_data_shape[2] != nx {
        return Err(ImportError::BadShape("TK".to_string(), tk_shape.clone()));
    }
    let preflight_nz = tk_data_shape[0];
    let _working_set_bytes = preflight_iso_volume_shape(preflight_nz, cells)
        .map_err(ImportError::PostprocessedVolume)?;
    progress("reading post-processed 3D fields (TK/P/Z/QVAPOR)".to_string());
    // `tk` and `z_m` are `mut`: after the iso interpolation they are converted
    // in place (K -> C, MSL -> AGL) for the severe suite below, instead of
    // allocating two more full-3D arrays (hundreds of MB each on CONUS-II).
    let (mut tk, nz) = read3d("TK")?;
    if nz != preflight_nz {
        return Err(ImportError::BadShape("TK".to_string(), vec![nz, ny, nx]));
    }
    let (mut p_pa, _) = read3d("P")?;
    let (mut z_m, z_nz) = read3d("Z")?;
    let (mut qv, _) = read3d("QVAPOR")?;
    // CONUS-II era quirk: the CTRL/history wrf3d files carry Z on the
    // STAGGERED vertical grid (w-levels, nz+1 = bottom_top_stag, like W),
    // while the future-era files carry it destaggered on mass levels (nz).
    // Staggered Z is the layer interfaces' own height: the isobaric heights
    // are read between those interfaces when the file states its eta levels
    // (ZNW), and Z is destaggered after that for the severe suite.  Without
    // either, the height volume takes the named mass-level fallback.
    let mut notes = Vec::new();
    let z_staggered = nz.checked_add(1) == Some(z_nz);
    let interface_stencil = if z_staggered {
        match postprocessed_interface_stencil(nc, time_index, nz) {
            Ok(stencil) => Some(stencil),
            Err(reason) => {
                destagger_z_to_mass_levels(&mut z_m, nz, cells)?;
                notes.push(crate::wrf_volumes::mass_level_heights_note(&reason));
                None
            }
        }
    } else {
        notes.push(crate::wrf_volumes::mass_level_heights_note(
            "the file carries Z on its mass levels only",
        ));
        None
    };
    let expected = nz.checked_mul(cells).unwrap_or(0);
    let z_expected = if interface_stencil.is_some() {
        nz.checked_add(1)
            .and_then(|levels| levels.checked_mul(cells))
            .unwrap_or(0)
    } else {
        expected
    };
    if expected == 0
        || [tk.len(), p_pa.len(), qv.len()]
            .iter()
            .any(|len| *len != expected)
        || z_m.len() != z_expected
    {
        return Err(ImportError::PlaneMismatch);
    }
    normalize_postprocessed_values(nc, "TK", &mut tk, PostprocessedUnitKind::TemperatureKelvin)?;
    normalize_postprocessed_values(nc, "P", &mut p_pa, PostprocessedUnitKind::PressurePascal)?;
    normalize_postprocessed_values(nc, "Z", &mut z_m, PostprocessedUnitKind::HeightMeter)?;
    normalize_postprocessed_values(
        nc,
        "QVAPOR",
        &mut qv,
        PostprocessedUnitKind::MixingRatioKgKg,
    )?;

    // Destagger the C-grid winds to mass points, then rotate them to
    // earth-relative east/north before any canonical vector or sounding
    // volume is created. If a projected file omits the rotation planes, its
    // thermodynamic data remains usable but vector products are withheld.
    progress("destaggering U/V winds to mass points".to_string());
    let mut u_mass = destagger_x(nc, "U", time_index, nz, ny, nx)?;
    let mut v_mass = destagger_y(nc, "V", time_index, nz, ny, nx)?;
    normalize_postprocessed_values(
        nc,
        "U",
        &mut u_mass,
        PostprocessedUnitKind::WindMetersPerSecond,
    )?;
    normalize_postprocessed_values(
        nc,
        "V",
        &mut v_mass,
        PostprocessedUnitKind::WindMetersPerSecond,
    )?;
    let wind_rotation = read_wrf_wind_rotation(&src, projection.as_ref(), cells)?;
    let winds_are_earth_relative = wind_rotation.is_some();
    if let Some(rotation) = &wind_rotation {
        progress("rotating U/V winds to earth-relative components".to_string());
        rotation.rotate_f64_levels_in_place(&mut u_mass, &mut v_mass, cells)?;
    } else {
        progress(
            "SINALPHA/COSALPHA unavailable for projected grid; withholding canonical wind vectors"
                .to_string(),
        );
    }

    let p_hpa: Vec<f64> = p_pa.iter().map(|pa| pa / 100.0).collect();
    let dewpoint_k: Vec<f64> = qv
        .iter()
        .zip(&p_pa)
        .map(|(&q, &pa)| dewpoint_k_from_q_p(q, pa))
        .collect();

    let heights = match &interface_stencil {
        Some(stencil) => ColumnHeights::Interfaces {
            interface: &z_m,
            per_metre: 1.0,
            stencil,
        },
        None => ColumnHeights::MassLevels(&z_m),
    };
    let (mut volumes, surface) = try_interpolate_iso_volumes(
        &p_hpa,
        &tk,
        &dewpoint_k,
        heights,
        &u_mass,
        &v_mass,
        nz,
        cells,
        progress,
    )
    .map_err(ImportError::PostprocessedVolume)?;
    // Every other use of Z here is at the mass levels.
    if interface_stencil.is_some() {
        destagger_z_to_mass_levels(&mut z_m, nz, cells)?;
    }
    for note in &notes {
        progress(note.clone());
    }
    if !winds_are_earth_relative {
        volumes.retain(|volume| !matches!(volume.name.as_str(), "u_iso" | "v_iso"));
    }

    // The 3D file carries no true surface fields. Expose the lowest-model-
    // level anchors under explicit approx_* names so map products never
    // mistake a ~25-50 m model level for exact 2 m/10 m observations.
    let mut canonical = Vec::new();
    let SurfaceFallback {
        surface_pressure_pa,
        temperature_2m_k,
        dewpoint_2m_k,
        u_10m,
        v_10m,
    } = surface;
    let mut surface_entries: Vec<(&str, FieldSelector, &str, Vec<f32>)> = vec![
        (
            "approx_surface_pressure",
            FieldSelector::surface(CanonicalField::Pressure),
            "Pa",
            surface_pressure_pa,
        ),
        (
            "approx_temperature_2m",
            FieldSelector::surface(CanonicalField::Temperature),
            "K",
            temperature_2m_k,
        ),
        (
            "approx_dewpoint_2m",
            FieldSelector::surface(CanonicalField::Dewpoint),
            "K",
            dewpoint_2m_k,
        ),
    ];
    if winds_are_earth_relative {
        surface_entries.push((
            "approx_u_10m",
            FieldSelector::surface(CanonicalField::UWind),
            "m/s",
            u_10m,
        ));
        surface_entries.push((
            "approx_v_10m",
            FieldSelector::surface(CanonicalField::VWind),
            "m/s",
            v_10m,
        ));
    }
    for (name, selector, units, values) in surface_entries {
        push_computed(
            &mut canonical,
            &grid,
            projection.clone(),
            name,
            selector,
            units,
            values,
        )?;
    }

    if !compute_severe {
        progress("skipping approximate post-processed severe suite (light import)".to_string());
        return Ok(Some((canonical, Vec::new(), volumes, Vec::new(), notes)));
    }

    // Severe/thermo suite via the wrf-core met kernels (postproc_severe.rs
    // documents the approximations). Memory discipline: reuse the 3-D buffers
    // above with two in-place unit conversions instead of new allocations,
    // and give back `dewpoint_k` (only the iso interpolation needed it)
    // before the parcel lifts start.
    drop(dewpoint_k);
    // Surface parcel state from the lowest model level: the post-processed
    // files carry no PSFC/T2/Q2 (same approximation as the synthesized 2 m /
    // 10 m fields above). t2 must be captured in Kelvin BEFORE the in-place
    // Celsius conversion; psfc/q2 borrow the lowest-level planes directly.
    let t2_k: Vec<f64> = tk[..cells].to_vec();
    for value in tk.iter_mut() {
        *value -= 273.15;
    }
    // Height MSL -> AGL with the lowest model level as the terrain proxy (no
    // HGT in these files; documented approximation). Walk levels top-down so
    // the level-0 plane (the terrain itself) is consumed last and zeroes.
    for k in (0..nz).rev() {
        let base = k * cells;
        for cell in 0..cells {
            let terrain = z_m[cell];
            z_m[base + cell] -= terrain;
        }
    }
    let severe_inputs = crate::postproc_severe::SevereInputs {
        nx,
        ny,
        nz,
        pressure_pa: &p_pa,
        pressure_hpa: &p_hpa,
        temperature_c: &tk,
        qvapor: &qv,
        height_agl_m: &z_m,
        u_ms: &u_mass,
        v_ms: &v_mass,
        psfc_pa: &p_pa[..cells],
        t2_k: &t2_k,
        q2_kgkg: &qv[..cells],
    };
    // A pathological column must degrade to "no severe fields for this hour",
    // never fail the import (the heavy getvar loop's isolate_panics rule).
    let severe = match crate::wrf_process::isolate_panics("post-processed severe suite", || {
        Ok::<_, String>(crate::postproc_severe::compute(
            &severe_inputs,
            winds_are_earth_relative,
            &mut *progress,
        ))
    }) {
        Ok(fields) => fields,
        Err(err) => {
            progress(format!("severe suite skipped: {err}"));
            Vec::new()
        }
    };

    Ok(Some((canonical, severe, volumes, Vec::new(), notes)))
}

/// Destagger a `[nz, ny, nx+1]` (west_east_stag) field to `[nz, ny, nx]` mass
/// points by averaging adjacent x faces.
fn destagger_x(
    nc: &NcFile,
    name: &str,
    time_index: usize,
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<Vec<f64>, ImportError> {
    let array = read_array_f64_record_or_all(nc, name, time_index)?;
    let s = array.shape();
    let nxs = nx
        .checked_add(1)
        .ok_or_else(|| ImportError::BadShape(name.to_string(), s.to_vec()))?;
    if s.len() != 3 || s[0] != nz || s[1] != ny || s[2] != nxs {
        return Err(ImportError::BadShape(name.to_string(), s.to_vec()));
    }
    let src = array.values();
    let source_len = nz
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nxs))
        .ok_or_else(|| ImportError::BadShape(name.to_string(), s.to_vec()))?;
    if src.len() != source_len {
        return Err(ImportError::BadShape(name.to_string(), s.to_vec()));
    }
    let output_len = nz
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nx))
        .ok_or_else(|| ImportError::BadShape(name.to_string(), s.to_vec()))?;
    let mut out = vec![0f64; output_len];
    for k in 0..nz {
        for y in 0..ny {
            let base_s = (k * ny + y) * nxs;
            let base_d = (k * ny + y) * nx;
            for x in 0..nx {
                out[base_d + x] = 0.5 * (src[base_s + x] + src[base_s + x + 1]);
            }
        }
    }
    Ok(out)
}

/// Destagger a `[nz, ny+1, nx]` (south_north_stag) field to `[nz, ny, nx]` mass
/// points by averaging adjacent y faces.
fn destagger_y(
    nc: &NcFile,
    name: &str,
    time_index: usize,
    nz: usize,
    ny: usize,
    nx: usize,
) -> Result<Vec<f64>, ImportError> {
    let array = read_array_f64_record_or_all(nc, name, time_index)?;
    let s = array.shape();
    let nys = ny
        .checked_add(1)
        .ok_or_else(|| ImportError::BadShape(name.to_string(), s.to_vec()))?;
    if s.len() != 3 || s[0] != nz || s[1] != nys || s[2] != nx {
        return Err(ImportError::BadShape(name.to_string(), s.to_vec()));
    }
    let src = array.values();
    let source_len = nz
        .checked_mul(nys)
        .and_then(|value| value.checked_mul(nx))
        .ok_or_else(|| ImportError::BadShape(name.to_string(), s.to_vec()))?;
    if src.len() != source_len {
        return Err(ImportError::BadShape(name.to_string(), s.to_vec()));
    }
    let output_len = nz
        .checked_mul(ny)
        .and_then(|value| value.checked_mul(nx))
        .ok_or_else(|| ImportError::BadShape(name.to_string(), s.to_vec()))?;
    let mut out = vec![0f64; output_len];
    for k in 0..nz {
        for y in 0..ny {
            let base_lo = (k * nys + y) * nx;
            let base_hi = (k * nys + y + 1) * nx;
            let base_d = (k * ny + y) * nx;
            for x in 0..nx {
                out[base_d + x] = 0.5 * (src[base_lo + x] + src[base_hi + x]);
            }
        }
    }
    Ok(out)
}

/// Dewpoint (K) from water-vapor mixing ratio (kg/kg) and pressure (Pa), via
/// vapor pressure and the Bolton inversion: the 3D analog of the 2 m
/// `dewpoint_from_q_psfc` used above.
fn dewpoint_k_from_q_p(q: f64, p_pa: f64) -> f64 {
    if !q.is_finite() || !p_pa.is_finite() || q <= 0.0 || p_pa <= 0.0 {
        return f64::NAN;
    }
    let e = vapour_pressure_pa_from_mixing_ratio(q, p_pa).max(1.0);
    let ln = (e / 611.2).ln();
    let td_c = 243.5 * ln / (17.67 - ln);
    td_c + 273.15
}

// ---------------------------------------------------------------------------
// Shared heavy (ECAPE-class) recipe inputs for the wrfout lane.
//
// The heavy recipes live in ONE place: `rustwx_products::derived::
// compute_store_heavy_grids`, the same entry point the GRIB ingest lane
// calls. What the wrfout lane was missing is not a recipe, it is the pair
// of products-side input structs that entry point takes: the import builds
// rw-store surface planes plus the five `*_iso` volumes, and nothing turned
// those into a `SurfaceFields` / `PressureFields` pair. That assembly is
// what lives below. No ECAPE solve, no parcel code and no per-lane recipe
// is written here, so the two lanes cannot drift apart.
// ---------------------------------------------------------------------------

/// Which store planes an assembled heavy input pair actually leaned on, and
/// how many isobaric levels it carried. `substituted` names every plane that
/// came from the `approx_*` lowest-model-level stand-in a split `wrf3d` file
/// carries instead of a true 2 m / 10 m field: the heavy grids are computed
/// from it rather than refused, and the caller states that basis once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct WrfHeavyInputBasis {
    pub(crate) substituted: Vec<String>,
    pub(crate) levels: usize,
}

impl WrfHeavyInputBasis {
    /// One line naming the conservative basis the hour was priced from, or
    /// `None` when every input was the exact plane it claims to be.
    pub(crate) fn note(&self) -> Option<String> {
        if self.substituted.is_empty() {
            return None;
        }
        Some(format!(
            "shared heavy (ECAPE) recipe grids computed from lowest-model-level \
             surface stand-ins for {}: this file carries no true 2 m / 10 m plane \
             for them",
            self.substituted.join(", ")
        ))
    }
}

/// One 2-D plane of an imported hour, by its canonical store name.
fn wrf_plane_values<'a>(
    canonical: &'a [(String, SelectedField2D)],
    name: &str,
) -> Option<&'a [f32]> {
    canonical
        .iter()
        .find(|(store_name, _)| store_name == name)
        .map(|(_, field)| field.values.as_slice())
}

/// One surface input for the heavy lane as f64: the exact plane when the
/// import stored it, otherwise the named lowest-model-level stand-in (whose
/// use is recorded on `basis`). A plane that is neither is a missing input,
/// and the error names it and what produces it.
fn wrf_heavy_surface_input(
    canonical: &[(String, SelectedField2D)],
    exact: &str,
    substitute: Option<&str>,
    produced_by: &str,
    cells: usize,
    basis: &mut WrfHeavyInputBasis,
) -> Result<Vec<f64>, String> {
    let (name, values) = match wrf_plane_values(canonical, exact) {
        Some(values) => (exact, values),
        None => {
            let substitute = substitute.ok_or_else(|| {
                format!("the heavy recipe inputs need the '{exact}' plane ({produced_by})")
            })?;
            let values = wrf_plane_values(canonical, substitute).ok_or_else(|| {
                format!(
                    "the heavy recipe inputs need the '{exact}' plane ({produced_by}); \
                     this hour stored neither it nor its '{substitute}' stand-in"
                )
            })?;
            basis.substituted.push(substitute.to_string());
            (substitute, values)
        }
    };
    if values.len() != cells {
        return Err(format!(
            "the '{name}' plane has {} values for a {cells}-cell grid",
            values.len()
        ));
    }
    Ok(values.iter().map(|value| f64::from(*value)).collect())
}

/// One isobaric volume of an imported hour, by its canonical store name.
fn wrf_heavy_volume<'a>(volumes: &'a [IsoVolume], name: &str) -> Result<&'a IsoVolume, String> {
    volumes
        .iter()
        .find(|volume| volume.name == name)
        .ok_or_else(|| {
            format!(
                "the heavy recipe inputs need the '{name}' isobaric volume, which the \
                 sounding volume build produces"
            )
        })
}

/// One f64 heavy-input volume buffer, reserved fallibly.
fn try_heavy_volume_buffer(name: &str, elements: usize) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|err| {
        format!("reserve the {elements}-element '{name}' heavy recipe input volume: {err}")
    })?;
    Ok(values)
}

/// Assemble the products-side `SurfaceFields` / `PressureFields` pair the
/// SHARED heavy recipe lane takes, from the 2-D planes and `*_iso` volumes
/// one imported wrfout hour already produced.
///
/// Conventions this has to match, because the shared lane assumes them:
///
/// * volumes are indexed `k * cells + ij`, bottom level first, which for the
///   GRIB lane means DESCENDING pressure (`gridded.rs` sorts its records that
///   way) and is what `compute_height_agl_3d`'s monotonic pass reads. The
///   `*_iso` volumes are built ascending (100..=1000 hPa), so the level order
///   is reversed here.
/// * `temperature_c_3d` is Celsius; `temperature_iso` is K.
/// * `qvapor_kgkg_3d` is a mixing ratio; the volumes carry dewpoint, so each
///   level goes through the products lane's OWN `mixing_ratio_from_dewpoint_k`
///   rather than a second formula written here.
/// * levels outside a column's model range are NaN in the `*_iso` planes.
///   That is exactly what the shared column builder drops (metrust's
///   `push_ecape_level` refuses a non-finite level), so below-ground and
///   above-top levels are skipped per column instead of being invented.
///
/// The three `*_ecape_native_cape_ratio` recipes divide by the SOURCE model's
/// own decoded CAPE plane. A wrfout carries no such message, so the native
/// planes are left `None` on purpose and the shared lane records those three
/// as its documented skip rather than a failure.
pub(crate) fn wrf_heavy_compute_inputs(
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    canonical: &[(String, SelectedField2D)],
    volumes: &[IsoVolume],
) -> Result<
    (
        rustwx_products::gridded::SurfaceFields,
        rustwx_products::gridded::PressureFields,
        WrfHeavyInputBasis,
    ),
    String,
> {
    let cells = grid.shape.len();
    if cells == 0 {
        return Err("the heavy recipe inputs need a non-empty grid".to_string());
    }
    let mut basis = WrfHeavyInputBasis::default();

    let orog_m = wrf_heavy_surface_input(
        canonical,
        "orography",
        None,
        "the WRF terrain field",
        cells,
        &mut basis,
    )?;
    let psfc_pa = wrf_heavy_surface_input(
        canonical,
        "surface_pressure",
        Some("approx_surface_pressure"),
        "WRF PSFC",
        cells,
        &mut basis,
    )?;
    let t2_k = wrf_heavy_surface_input(
        canonical,
        "temperature_2m",
        Some("approx_temperature_2m"),
        "the WRF 2 m temperature diagnostic",
        cells,
        &mut basis,
    )?;
    let dewpoint_2m_k = wrf_heavy_surface_input(
        canonical,
        "dewpoint_2m",
        Some("approx_dewpoint_2m"),
        "the WRF 2 m dewpoint diagnostic",
        cells,
        &mut basis,
    )?;
    let u10_ms = wrf_heavy_surface_input(
        canonical,
        "u_10m",
        Some("approx_u_10m"),
        "the earth-relative uvmet10 diagnostic",
        cells,
        &mut basis,
    )?;
    let v10_ms = wrf_heavy_surface_input(
        canonical,
        "v_10m",
        Some("approx_v_10m"),
        "the earth-relative uvmet10 diagnostic",
        cells,
        &mut basis,
    )?;

    let temperature = wrf_heavy_volume(volumes, "temperature_iso")?;
    let dewpoint = wrf_heavy_volume(volumes, "dewpoint_iso")?;
    let u_wind = wrf_heavy_volume(volumes, "u_iso")?;
    let v_wind = wrf_heavy_volume(volumes, "v_iso")?;
    let height = wrf_heavy_volume(volumes, "height_iso")?;

    let reference: Vec<u16> = temperature.levels.iter().map(|(hpa, _)| *hpa).collect();
    if reference.len() < 2 {
        return Err(format!(
            "the heavy recipe inputs need at least two isobaric levels, this hour has {}",
            reference.len()
        ));
    }
    for volume in [dewpoint, u_wind, v_wind, height] {
        let levels: Vec<u16> = volume.levels.iter().map(|(hpa, _)| *hpa).collect();
        if levels != reference {
            return Err(format!(
                "the '{}' volume's levels differ from 'temperature_iso'; the heavy \
                 recipe inputs need one shared level set",
                volume.name
            ));
        }
    }

    // Bottom level first, i.e. descending pressure: the order the shared
    // height-AGL prep and the shared column builder both read.
    let mut order: Vec<usize> = (0..reference.len()).collect();
    order.sort_by(|a, b| reference[*b].cmp(&reference[*a]));
    let nz = order.len();
    basis.levels = nz;

    let elements = nz
        .checked_mul(cells)
        .ok_or_else(|| "the heavy recipe input volume size overflows usize".to_string())?;
    // Fallible reservation, the same shape `wrf_volumes` uses: these five
    // f64 volumes are twice the size of the f32 `*_iso` set they come from,
    // and on a large grid that is hundreds of megabytes each. A machine
    // that cannot hold them degrades to a note through the caller instead
    // of aborting the import.
    let mut temperature_c_3d = try_heavy_volume_buffer("temperature_c_3d", elements)?;
    let mut qvapor_kgkg_3d = try_heavy_volume_buffer("qvapor_kgkg_3d", elements)?;
    let mut u_ms_3d = try_heavy_volume_buffer("u_ms_3d", elements)?;
    let mut v_ms_3d = try_heavy_volume_buffer("v_ms_3d", elements)?;
    let mut gh_m_3d = try_heavy_volume_buffer("gh_m_3d", elements)?;
    for &level_index in &order {
        let level_hpa = f64::from(reference[level_index]);
        let temperature_plane = &temperature.levels[level_index].1;
        let dewpoint_plane = &dewpoint.levels[level_index].1;
        let u_plane = &u_wind.levels[level_index].1;
        let v_plane = &v_wind.levels[level_index].1;
        let height_plane = &height.levels[level_index].1;
        for (name, plane) in [
            ("temperature_iso", temperature_plane),
            ("dewpoint_iso", dewpoint_plane),
            ("u_iso", u_plane),
            ("v_iso", v_plane),
            ("height_iso", height_plane),
        ] {
            if plane.len() != cells {
                return Err(format!(
                    "the '{name}' {level_hpa} hPa plane has {} values for a {cells}-cell grid",
                    plane.len()
                ));
            }
        }
        for cell in 0..cells {
            let temperature_k = f64::from(temperature_plane[cell]);
            temperature_c_3d.push(temperature_k - 273.15);
            let dewpoint_k = f64::from(dewpoint_plane[cell]);
            qvapor_kgkg_3d.push(if dewpoint_k.is_finite() {
                rustwx_products::gridded::mixing_ratio_from_dewpoint_k(level_hpa, dewpoint_k)
            } else {
                f64::NAN
            });
            u_ms_3d.push(f64::from(u_plane[cell]));
            v_ms_3d.push(f64::from(v_plane[cell]));
            gh_m_3d.push(f64::from(height_plane[cell]));
        }
    }

    let q2_kgkg = psfc_pa
        .iter()
        .zip(dewpoint_2m_k.iter())
        .map(|(pressure_pa, dewpoint_k)| {
            if pressure_pa.is_finite() && *pressure_pa > 0.0 && dewpoint_k.is_finite() {
                rustwx_products::gridded::mixing_ratio_from_dewpoint_k(
                    pressure_pa / 100.0,
                    *dewpoint_k,
                )
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<_>>();

    let surface = rustwx_products::gridded::SurfaceFields {
        lat: grid.lat_deg.iter().map(|value| f64::from(*value)).collect(),
        lon: grid.lon_deg.iter().map(|value| f64::from(*value)).collect(),
        nx: grid.shape.nx,
        ny: grid.shape.ny,
        projection,
        psfc_pa,
        orog_m,
        // WRF terrain is the model's real orography, never a pressure-derived
        // proxy, so the shared prep must not substitute one.
        orog_is_proxy: false,
        t2_k,
        q2_kgkg,
        u10_ms,
        v10_ms,
        // A wrfout carries no decoded CAPE message of its own; see the
        // native-ratio note above.
        native_sbcape_jkg: None,
        native_mlcape_jkg: None,
        native_mucape_jkg: None,
        native_pblh_m: None,
    };
    let pressure = rustwx_products::gridded::PressureFields {
        pressure_levels_hpa: order
            .iter()
            .map(|level_index| f64::from(reference[*level_index]))
            .collect(),
        pressure_3d_pa: None,
        temperature_c_3d,
        qvapor_kgkg_3d,
        u_ms_3d,
        v_ms_3d,
        gh_m_3d,
        omega_pa_s_3d: None,
        absolute_vorticity_s_3d: None,
        cloud_liquid_kgkg_3d: None,
        cloud_ice_kgkg_3d: None,
        rain_kgkg_3d: None,
        snow_kgkg_3d: None,
        graupel_kgkg_3d: None,
    };
    Ok((surface, pressure, basis))
}

/// Run the SHARED heavy (ECAPE-class) recipe lane over one imported wrfout
/// hour: assemble its products-side inputs with [`wrf_heavy_compute_inputs`],
/// then hand them to `rustwx_products::derived::compute_store_heavy_grids`.
/// That call is the whole point: the ml/mu parcel grids and the ECAPE /
/// derived-CAPE ratio pairs resolve here through the exact recipe code the
/// GRIB heavy ingest runs, with no second ECAPE path on this lane.
pub(crate) fn compute_wrf_heavy_store_grids(
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    canonical: &[(String, SelectedField2D)],
    volumes: &[IsoVolume],
) -> Result<(rustwx_products::derived::StoreHeavyGrids, WrfHeavyInputBasis), String> {
    let (surface, pressure, basis) =
        wrf_heavy_compute_inputs(grid, projection, canonical, volumes)?;
    let grids = rustwx_products::derived::compute_store_heavy_grids(&surface, &pressure)
        .map_err(|err| format!("shared heavy recipe lane: {err}"))?;
    Ok((grids, basis))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A four-cell wrfout hour: the six surface planes the shared heavy lane
    /// reads plus the five `*_iso` volumes, built the way the import builds
    /// them (levels ASCENDING, NaN where a level is outside the column).
    /// Cell 3 has a NaN 1000 hPa row, the below-ground case.
    fn synthetic_wrf_heavy_hour() -> (
        LatLonGrid,
        Vec<(String, SelectedField2D)>,
        Vec<crate::wrf_volumes::IsoVolume>,
    ) {
        let shape = GridShape::new(2, 2).expect("2x2 grid");
        let grid = LatLonGrid::new(
            shape,
            vec![35.0, 35.0, 35.5, 35.5],
            vec![-97.5, -97.0, -97.5, -97.0],
        )
        .expect("grid");
        let plane = |name: &str, selector: FieldSelector, units: &str, values: Vec<f32>| {
            (
                name.to_string(),
                SelectedField2D::new(selector, units, grid.clone(), values).expect("plane"),
            )
        };
        let canonical = vec![
            plane(
                "orography",
                FieldSelector::surface(CanonicalField::GeopotentialHeight),
                "m",
                vec![10.0; 4],
            ),
            plane(
                "surface_pressure",
                FieldSelector::surface(CanonicalField::Pressure),
                "Pa",
                vec![100_000.0; 4],
            ),
            plane(
                "temperature_2m",
                FieldSelector::height_agl(CanonicalField::Temperature, 2),
                "K",
                vec![303.15; 4],
            ),
            plane(
                "dewpoint_2m",
                FieldSelector::height_agl(CanonicalField::Dewpoint, 2),
                "K",
                vec![297.15; 4],
            ),
            plane(
                "u_10m",
                FieldSelector::height_agl(CanonicalField::UWind, 10),
                "m/s",
                vec![2.0; 4],
            ),
            plane(
                "v_10m",
                FieldSelector::height_agl(CanonicalField::VWind, 10),
                "m/s",
                vec![1.0; 4],
            ),
        ];

        // (hPa, T K, Td K, height m, u m/s, v m/s) -- ascending pressure, the
        // order `build_iso_volumes` packs.
        let profile: [(u16, f32, f32, f32, f32, f32); 5] = [
            (300, 233.15, 213.15, 9_500.0, 32.0, 12.0),
            (500, 263.15, 253.15, 5_800.0, 22.0, 9.0),
            (700, 280.15, 274.15, 3_100.0, 14.0, 6.0),
            (850, 290.15, 286.15, 1_500.0, 8.0, 3.0),
            (1000, 300.15, 296.15, 110.0, 3.0, 1.0),
        ];
        let volume = |name: &str, units: &str, pick: fn(&(u16, f32, f32, f32, f32, f32)) -> f32| {
            crate::wrf_volumes::IsoVolume {
                name: name.to_string(),
                units: units.to_string(),
                levels: profile
                    .iter()
                    .map(|row| {
                        let value = pick(row);
                        let mut values = vec![value; 4];
                        if row.0 == 1000 {
                            // Below ground in this column: the import leaves the
                            // level NaN and the shared column builder drops it.
                            values[3] = f32::NAN;
                        }
                        (row.0, values)
                    })
                    .collect(),
            }
        };
        let volumes = vec![
            volume("temperature_iso", "K", |row| row.1),
            volume("dewpoint_iso", "K", |row| row.2),
            volume("u_iso", "m/s", |row| row.4),
            volume("v_iso", "m/s", |row| row.5),
            volume("height_iso", "gpm", |row| row.3),
        ];
        (grid, canonical, volumes)
    }

    /// WHAT BREAKAGE THIS PREVENTS (gate law): the wrfout lane built rw-store
    /// surface planes and `*_iso` volumes and nothing else, so the shared
    /// heavy recipe entry point -- which takes a products-side
    /// SurfaceFields/PressureFields pair -- could not be called at all, and
    /// the ml/mu parcel grids plus the ECAPE/derived-CAPE ratio pairs were
    /// unreachable on this lane while running fine on the other one.
    #[test]
    fn the_assembled_wrfout_inputs_feed_the_shared_heavy_recipe_lane() {
        let (grid, canonical, volumes) = synthetic_wrf_heavy_hour();
        let (surface, pressure, basis) =
            wrf_heavy_compute_inputs(&grid, None, &canonical, &volumes).expect("assembled inputs");

        // Bottom level first (descending pressure), the order the shared
        // height-AGL prep and column builder read.
        assert_eq!(
            pressure.pressure_levels_hpa,
            vec![1000.0, 850.0, 700.0, 500.0, 300.0],
            "the shared lane reads bottom level first"
        );
        assert_eq!(basis.levels, 5);
        assert!(
            basis.substituted.is_empty(),
            "every plane was exact: {:?}",
            basis.substituted
        );
        // Celsius, not K: `temperature_c_3d` is what the shared lane names.
        assert!((pressure.temperature_c_3d[0] - 27.0).abs() < 1.0e-3);
        // Moisture goes through the products lane's own converter, never a
        // second formula written on this lane.
        let expected_q =
            rustwx_products::gridded::mixing_ratio_from_dewpoint_k(1000.0, 296.15_f32 as f64);
        assert!((pressure.qvapor_kgkg_3d[0] - expected_q).abs() < 1.0e-9);
        assert!(
            pressure.qvapor_kgkg_3d[3].is_nan(),
            "a below-ground level stays NaN instead of being invented"
        );
        // A wrfout carries no decoded CAPE message of its own.
        assert!(surface.native_sbcape_jkg.is_none());
        assert!(surface.native_mlcape_jkg.is_none());
        assert!(surface.native_mucape_jkg.is_none());

        let (heavy, _) = compute_wrf_heavy_store_grids(&grid, None, &canonical, &volumes)
            .expect("shared heavy recipe lane");
        let realized: Vec<&str> = heavy.grids.iter().map(|grid| grid.slug).collect();
        for slug in ["mlecape", "muecape", "sbecin", "mlecin"] {
            assert!(
                realized.contains(&slug),
                "the shared lane must realize '{slug}' on the wrfout lane: {realized:?}"
            );
        }
        for slug in [
            "sb_ecape_derived_cape_ratio",
            "ml_ecape_derived_cape_ratio",
            "mu_ecape_derived_cape_ratio",
        ] {
            assert!(
                realized.contains(&slug),
                "the derived-CAPE ratio pairs must resolve here too: {realized:?}"
            );
        }
        let skipped: Vec<&str> = heavy.skipped.iter().map(|skip| skip.slug).collect();
        assert_eq!(
            skipped,
            vec![
                "sb_ecape_native_cape_ratio",
                "ml_ecape_native_cape_ratio",
                "mu_ecape_native_cape_ratio",
            ],
            "only the native-CAPE ratio pairs are skippable here"
        );
        for skip in &heavy.skipped {
            assert!(
                skip.reason.contains("native"),
                "the skip must name its cause: {}",
                skip.reason
            );
        }

        // The assembled pair really solves: a column with a 30 C / 24 C
        // surface under a -10 C 500 mb temperature has positive ECAPE.
        let mlecape = heavy
            .grids
            .iter()
            .find(|grid| grid.slug == "mlecape")
            .expect("mlecape grid");
        assert_eq!(mlecape.values.len(), 4);
        assert!(
            mlecape.values[0].is_finite() && mlecape.values[0] > 0.0,
            "the assembled inputs must produce a real ML ECAPE value, got {:?}",
            mlecape.values[0]
        );
    }

    /// A missing surface plane is named, with what produces it, instead of
    /// failing the hour or silently dropping the heavy family.
    #[test]
    fn a_missing_heavy_input_plane_names_itself_and_its_source() {
        let (grid, canonical, volumes) = synthetic_wrf_heavy_hour();
        let without_psfc: Vec<(String, SelectedField2D)> = canonical
            .into_iter()
            .filter(|(name, _)| name != "surface_pressure")
            .collect();
        let err = wrf_heavy_compute_inputs(&grid, None, &without_psfc, &volumes)
            .expect_err("a missing surface plane must refuse");
        assert!(err.contains("surface_pressure"), "{err}");
        assert!(err.contains("PSFC"), "{err}");
        assert!(err.contains("approx_surface_pressure"), "{err}");
    }

    fn test_time_axis(reference_unix: Option<i64>, offsets: &[i64]) -> SourceTimeAxis {
        let origin = reference_unix.unwrap_or(1_700_000_000);
        SourceTimeAxis {
            records: offsets
                .iter()
                .enumerate()
                .map(|(time_index, offset)| {
                    let valid_unix = origin + offset;
                    SourceTimeRecord {
                        time_index,
                        valid_unix,
                        label: format_valid_unix(valid_unix),
                    }
                })
                .collect(),
            reference_unix,
        }
    }

    #[test]
    fn forecast_timeline_uses_exact_elapsed_hours_not_ordinals() {
        // A whole-hour origin (2023-11-14 22:00:00Z): only those keep the
        // legacy forecast-hour slots (see the off-hour test below).
        let origin = 1_699_999_200;
        let sources = vec![(
            PathBuf::from("wrfout.nc"),
            test_time_axis(Some(origin), &[0, 3 * 3_600, 9 * 3_600]),
        )];
        let timeline = ForecastHourTimeline::plan_all(&sources).expect("integral timeline");
        let planned = timeline.records_for_source(0).unwrap();
        assert_eq!(
            planned
                .iter()
                .map(|record| record.storage_slot)
                .collect::<Vec<_>>(),
            vec![0, 3, 9]
        );
        assert!(planned.iter().all(|record| record.exact_time.is_none()));
        assert_eq!(
            timeline.run_name("0123456789abcdef", "light"),
            format!("local_20231114220000_0123456789abcdef_light_{IMPORT_SCIENCE_SCHEMA_VERSION}")
        );
        assert_ne!(
            timeline.run_name("0123456789abcdef", "light"),
            timeline.run_name("fedcba9876543210", "light")
        );
        assert_ne!(
            timeline.run_name("0123456789abcdef", "light"),
            timeline.run_name("0123456789abcdef", "full_deadbeef")
        );
    }

    #[test]
    fn forecast_timeline_retains_long_integration_origin_in_exact_slots() {
        let origin = 1_699_999_200; // whole hour, so the boundary case below stays legacy
        let last_legacy = i64::from(u16::MAX) * 3_600;
        let sources = vec![(PathBuf::from("long_integration.nc"),
            test_time_axis(Some(origin), &[last_legacy, last_legacy + 3_600]))];
        let timeline = ForecastHourTimeline::plan_all(&sources).unwrap();
        assert!(timeline.is_exact_time_axis());
        assert_eq!(timeline.origin_unix, origin);
        let records = timeline.records_for_source(0).unwrap();
        for (ordinal, row) in records.iter().enumerate() {
            let lead = last_legacy as u64 + ordinal as u64 * 3_600;
            assert_eq!(row.storage_slot, ordinal as u16);
            assert_eq!(row.valid_unix, origin + lead as i64);
            assert_eq!(row.exact_time.unwrap().lead_seconds, lead);
        }
        let boundary = vec![(PathBuf::from("legacy_boundary.nc"),
            test_time_axis(Some(origin), &[last_legacy]))];
        let legacy = ForecastHourTimeline::plan_all(&boundary).unwrap();
        assert!(!legacy.is_exact_time_axis());
        assert_eq!(legacy.records_for_source(0).unwrap()[0].storage_slot, u16::MAX);
    }

    #[test]
    fn off_hour_origin_with_whole_hour_leads_keeps_exact_times() {
        // A forecast from a 10:15Z analysis (2023-07-10), frames at +0 and +3 h:
        // the legacy axis would label the +3 h frame "Init 10Z | F003 | Valid 13Z".
        let origin = 1_688_984_100;
        let sources = vec![(
            PathBuf::from("wrfout_d01_2023-07-10_13_15_00"),
            test_time_axis(Some(origin), &[0, 3 * 3_600]),
        )];
        let timeline = ForecastHourTimeline::plan_all(&sources).unwrap();
        assert!(timeline.is_exact_time_axis());
        let records = timeline.records_for_source(0).unwrap();
        assert_eq!(records[1].exact_time.unwrap().lead_seconds, 10_800);
        assert_eq!(records[1].valid_unix, origin + 10_800);
        assert!(timeline.run_name("id", "light").starts_with("local_20230710101500_"));
    }

    #[test]
    fn source_set_identity_is_order_and_parent_path_independent() {
        let left = temp_dir("source-id-left");
        let right = temp_dir("source-id-right");
        std::fs::create_dir_all(&left).unwrap();
        std::fs::create_dir_all(&right).unwrap();
        let left_a = left.join("wrfout_d01_a.nc");
        let left_b = left.join("wrfout_d01_b.nc");
        let right_a = right.join("wrfout_d01_a.nc");
        let right_b = right.join("wrfout_d01_b.nc");
        std::fs::write(&left_a, b"same-a").unwrap();
        std::fs::write(&left_b, b"same-b").unwrap();
        std::fs::write(&right_a, b"same-a").unwrap();
        std::fs::write(&right_b, b"same-b").unwrap();

        let forward = source_set_identity(&[left_a.clone(), left_b.clone()]).unwrap();
        let reversed = source_set_identity(&[right_b.clone(), right_a.clone()]).unwrap();
        assert_eq!(forward, reversed);
        assert_eq!(forward.len(), 64, "source identity must be a full SHA-256");

        // Same filename and same byte length, different content.
        std::fs::write(&right_b, b"same-z").unwrap();
        let different = source_set_identity(&[right_a, right_b]).unwrap();
        assert_ne!(forward, different);
        let _ = std::fs::remove_dir_all(left);
        let _ = std::fs::remove_dir_all(right);
    }

    #[test]
    fn source_set_identity_detects_middle_only_changes() {
        let dir = temp_dir("source-id-middle");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("wrfout_d01.nc");
        let mut bytes = vec![0x31; 3 * 64 * 1_024];
        std::fs::write(&path, &bytes).unwrap();
        let before = source_set_identity(std::slice::from_ref(&path)).unwrap();

        bytes[96 * 1_024] = 0x32;
        std::fs::write(&path, &bytes).unwrap();
        let after = source_set_identity(std::slice::from_ref(&path)).unwrap();
        assert_ne!(before, after);
        let _ = std::fs::remove_dir_all(dir);
    }

    // The breakage: a host removing earlier history frames to save disk
    // failed every later frame whole ("cannot resolve source file"), because
    // the snowfall history was part of the source identity.
    #[test]
    fn a_removed_history_file_leaves_the_identity_instead_of_failing_it() {
        let dir = temp_dir("source-id-history");
        std::fs::create_dir_all(&dir).unwrap();
        let frame = dir.join("wrfout_d01_2026-01-01_02_00_00");
        let earlier = dir.join("wrfout_d01_2026-01-01_01_00_00");
        let removed = dir.join("wrfout_d01_2026-01-01_00_00_00");
        std::fs::write(&frame, b"frame").unwrap();
        std::fs::write(&earlier, b"earlier").unwrap();

        // Gone before the identity is taken: left out, and named.
        let history = vec![removed.clone(), earlier.clone(), frame.clone()];
        let snapshot =
            capture_source_set_identity_with_history(std::slice::from_ref(&frame), &history)
                .unwrap();
        assert_eq!(snapshot.gone(), std::slice::from_ref(&removed));
        assert_eq!(
            snapshot.identity,
            source_set_identity(&[frame.clone(), earlier.clone()]).unwrap()
        );

        // Gone during the import: let go at the end.
        std::fs::remove_file(&earlier).unwrap();
        verify_source_set_unchanged(&snapshot).unwrap();

        // A history file still there but rewritten is refused as before.
        std::fs::write(&earlier, b"earlier").unwrap();
        let snapshot =
            capture_source_set_identity_with_history(std::slice::from_ref(&frame), &history)
                .unwrap();
        std::fs::write(&earlier, b"rewritten earlier").unwrap();
        assert!(verify_source_set_unchanged(&snapshot).is_err());

        // The frame itself is never optional.
        std::fs::remove_file(&frame).unwrap();
        assert!(capture_source_set_identity_with_history(
            std::slice::from_ref(&frame),
            &history
        )
        .is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn forecast_timeline_preserves_subhourly_times_and_rejects_duplicates() {
        let origin = 1_700_000_000;
        let sources = vec![(
            PathBuf::from("wrfout_1974-04-03_17_48_00"),
            test_time_axis(Some(origin), &[31_680, 31_740]),
        )];
        let timeline = ForecastHourTimeline::plan_all(&sources).expect("one-minute timeline");
        assert!(timeline.is_exact_time_axis());
        let records = timeline.records_for_source(0).unwrap();
        assert_eq!(
            records
                .iter()
                .map(|record| record.storage_slot)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            records
                .iter()
                .map(|record| record.exact_time.unwrap().lead_seconds)
                .collect::<Vec<_>>(),
            vec![31_680, 31_740]
        );

        let duplicates = vec![
            (
                PathBuf::from("first.nc"),
                test_time_axis(Some(origin), &[0]),
            ),
            (
                PathBuf::from("duplicate.nc"),
                test_time_axis(Some(origin), &[0]),
            ),
        ];
        let err = ForecastHourTimeline::plan_all(&duplicates)
            .expect_err("duplicate valid time must not overwrite");
        assert!(err.contains("duplicate forecast valid time"));
    }

    #[test]
    fn timeline_without_authoritative_origin_always_persists_exact_times() {
        let sources = vec![(
            PathBuf::from("snapshot_1974-04-03_17_48_00.nc"),
            test_time_axis(None, &[0, 3_600]),
        )];
        let timeline = ForecastHourTimeline::plan_all(&sources).unwrap();
        assert!(timeline.is_exact_time_axis());
        let records = timeline.records_for_source(0).unwrap();
        assert_eq!(records[0].exact_time.unwrap().lead_seconds, 0);
        assert_eq!(records[1].exact_time.unwrap().lead_seconds, 3_600);
    }

    #[test]
    fn time_label_conversion_rejects_oversized_axes_before_labels_are_read() {
        let error =
            source_records_from_labels(Vec::new(), MAX_RUN_TIMESTEPS + 1, "hostile WRF Times")
                .unwrap_err()
                .to_string();
        assert!(error.contains("per-run limit"), "{error}");
    }

    #[test]
    fn malformed_wrf_reference_can_fall_back_to_consistent_xtime() {
        let origin = parse_utc_timestamp("1974-04-03_23:00:00").unwrap();
        let records = test_time_axis(Some(origin), &[2_220, 2_281]).records;
        let attributes = vec![
            ("START_DATE", "1".to_string()),
            ("SIMULATION_START_DATE", "also-invalid".to_string()),
        ];
        let (attribute_reference, malformed) =
            parse_matching_wrf_reference_attributes(&attributes).unwrap();
        assert_eq!(attribute_reference, None);
        assert_eq!(malformed.len(), 2);

        // Exercise the f32 representation a WRF XTIME variable commonly uses:
        // 2,281 seconds is a repeating fractional minute, but its conversion
        // is still unambiguously within the whole-second tolerance.
        let xtime = [37.0, f64::from(2_281.0_f32 / 60.0_f32)];
        assert_eq!(wrf_reference_from_xtime(&records, &xtime), Ok(origin));
    }

    /// C-03: a run whose initial state was itself a 174-hour forecast
    /// printed exactly what a genuine analysis run prints, in all 159 of
    /// its PNGs.  The predicate is the lead alone.
    #[test]
    fn a_positive_lead_discloses_its_source_cycle_and_lead() {
        assert_eq!(
            compose_initial_condition_disclosure(174, "GFS", "2026-08-01_00:00:00"),
            Some("GFS 08/01 00Z f174".to_string())
        );
        // Three digits, like every other lead token on the sheet.
        assert_eq!(
            compose_initial_condition_disclosure(6, "GFS", "2026-08-01_00:00:00"),
            Some("GFS 08/01 00Z f006".to_string())
        );
        // Leads past 999 h are still spelled in full rather than truncated.
        assert_eq!(
            compose_initial_condition_disclosure(1_000, "GFS", "2026-08-01_00:00:00"),
            Some("GFS 08/01 00Z f1000".to_string())
        );
    }

    /// Absence means UNKNOWN, never analysis: an idealized case, a stock
    /// WRF archive and every file written before the provenance block
    /// existed must render exactly as they did, and a genuine analysis
    /// (lead 0) adds nothing either.
    #[test]
    fn a_zero_or_missing_lead_discloses_nothing() {
        assert_eq!(
            compose_initial_condition_disclosure(0, "GFS", "2026-08-01_00:00:00"),
            None
        );
        assert_eq!(
            compose_initial_condition_disclosure(-3, "GFS", "2026-08-01_00:00:00"),
            None
        );
        // A lead with nothing to attribute it to is not a disclosure.
        assert_eq!(
            compose_initial_condition_disclosure(174, "   ", "2026-08-01_00:00:00"),
            None
        );
        // ... and neither is one whose cycle cannot be read.
        assert_eq!(
            compose_initial_condition_disclosure(174, "GFS", "not-a-timestamp"),
            None
        );
    }

    #[test]
    fn matching_wrf_reference_attributes_are_preferred_and_conflicts_fail() {
        let expected = parse_utc_timestamp("1974-04-03_23:00:00").unwrap();
        let attributes = vec![
            ("START_DATE", "1974-04-03_23:00:00".to_string()),
            ("SIMULATION_START_DATE", "1".to_string()),
        ];
        let (reference, malformed) = parse_matching_wrf_reference_attributes(&attributes).unwrap();
        assert_eq!(reference, Some(expected));
        assert_eq!(malformed.len(), 1);

        let conflicting = vec![
            ("START_DATE", "1974-04-03_23:00:00".to_string()),
            ("SIMULATION_START_DATE", "1974-04-03_23:01:00".to_string()),
        ];
        let error = parse_matching_wrf_reference_attributes(&conflicting).unwrap_err();
        assert!(error.contains("conflicting WRF references"), "{error}");
    }

    /// A137: WRF writes a nest's own start into START_DATE and the head
    /// grid's into SIMULATION_START_DATE, so a nest that starts after the
    /// forecast carries a later START_DATE by design, and every one of its
    /// frames was refused as "conflicting WRF references".
    #[test]
    fn a_delayed_nest_start_measures_leads_from_the_simulation_start() {
        let simulation = parse_utc_timestamp("2026-09-29_12:00:00").unwrap();
        let nest = parse_utc_timestamp("2026-09-29_13:00:00").unwrap();
        let attributes = |start: &str, simulation: &str| {
            vec![
                ("START_DATE", start.to_string()),
                ("SIMULATION_START_DATE", simulation.to_string()),
            ]
        };

        // The delayed nest: the run's origin, not the nest's own start.
        let (reference, malformed) = parse_matching_wrf_reference_attributes(&attributes(
            "2026-09-29_13:00:00",
            "2026-09-29_12:00:00",
        ))
        .unwrap();
        assert_eq!(reference, Some(simulation));
        assert!(malformed.is_empty(), "{malformed:?}");
        // The order the stamps are read in does not choose the answer.
        let swapped = vec![
            ("SIMULATION_START_DATE", "2026-09-29_12:00:00".to_string()),
            ("START_DATE", "2026-09-29_13:00:00".to_string()),
        ];
        assert_eq!(
            parse_matching_wrf_reference_attributes(&swapped).unwrap().0,
            Some(simulation)
        );

        // A domain that starts with the run is unchanged.
        assert_eq!(
            parse_matching_wrf_reference_attributes(&attributes(
                "2026-09-29_12:00:00",
                "2026-09-29_12:00:00"
            ))
            .unwrap()
            .0,
            Some(simulation)
        );
        // One readable stamp is the origin, as before.
        let (reference, malformed) = parse_matching_wrf_reference_attributes(&attributes(
            "2026-09-29_13:00:00",
            "not-a-time",
        ))
        .unwrap();
        assert_eq!(reference, Some(nest));
        assert_eq!(malformed.len(), 1);

        // A domain that starts before its simulation is refused, naming
        // both stamps and what a guess would break.
        let error = parse_matching_wrf_reference_attributes(&attributes(
            "2026-09-29_11:00:00",
            "2026-09-29_12:00:00",
        ))
        .unwrap_err();
        assert!(
            error.contains("START_DATE=2026-09-29 11:00:00Z")
                && error.contains("SIMULATION_START_DATE=2026-09-29 12:00:00Z")
                && error.contains("guessed origin"),
            "{error}"
        );

        // The section renderer shares the rule.
        assert_eq!(wrf_run_origin(Some(nest), Some(simulation)), Ok(Some(simulation)));
        assert_eq!(wrf_run_origin(Some(nest), None), Ok(Some(nest)));
        assert_eq!(wrf_run_origin(None, Some(simulation)), Ok(Some(simulation)));
        assert_eq!(wrf_run_origin(None, None), Ok(None));
        assert!(wrf_run_origin(Some(simulation), Some(nest)).is_err());
    }

    #[test]
    fn xtime_reference_fallback_fails_closed_on_unsound_axes() {
        let origin = parse_utc_timestamp("1974-04-03_23:00:00").unwrap();
        let records = test_time_axis(Some(origin), &[2_220, 2_280]).records;

        let error = wrf_reference_from_xtime(&records, &[37.0]).unwrap_err();
        assert!(
            error.contains("1 values for 2 WRF Times records"),
            "{error}"
        );

        let error = wrf_reference_from_xtime(&records, &[37.0, f64::NAN]).unwrap_err();
        assert!(error.contains("finite and nonnegative"), "{error}");

        let error = wrf_reference_from_xtime(&records, &[37.0, -1.0]).unwrap_err();
        assert!(error.contains("finite and nonnegative"), "{error}");

        let error = wrf_reference_from_xtime(&records, &[37.005, 38.0]).unwrap_err();
        assert!(
            error.contains("whole-second WRF Times precision"),
            "{error}"
        );

        let error = wrf_reference_from_xtime(&records, &[37.0, 39.0]).unwrap_err();
        assert!(error.contains("origins disagree"), "{error}");
    }

    #[test]
    fn exact_timeline_slots_are_independent_of_filename_order() {
        let origin = 1_700_000_000;
        let late = (
            PathBuf::from("aaa-late.nc"),
            test_time_axis(Some(origin), &[31_740]),
        );
        let early = (
            PathBuf::from("zzz-early.nc"),
            test_time_axis(Some(origin), &[31_680]),
        );
        let forward = ForecastHourTimeline::plan_all(&[late.clone(), early.clone()]).unwrap();
        let reverse = ForecastHourTimeline::plan_all(&[early, late]).unwrap();
        assert_eq!(forward.records_for_source(0).unwrap()[0].storage_slot, 1);
        assert_eq!(forward.records_for_source(1).unwrap()[0].storage_slot, 0);
        assert_eq!(reverse.records_for_source(0).unwrap()[0].storage_slot, 0);
        assert_eq!(reverse.records_for_source(1).unwrap()[0].storage_slot, 1);
    }

    #[test]
    fn preflight_grid_shape_rejects_cross_file_geometry_before_writes() {
        let mut expected = None;
        merge_preflight_grid_shape(&mut expected, (600, 500), Path::new("first.nc")).unwrap();
        merge_preflight_grid_shape(&mut expected, (600, 500), Path::new("second.nc")).unwrap();
        let err = merge_preflight_grid_shape(&mut expected, (601, 500), Path::new("different.nc"))
            .unwrap_err();
        assert!(
            err.contains("different.nc") && err.contains("601x500"),
            "{err}"
        );

        let mut hostile = None;
        let err =
            merge_preflight_grid_shape(&mut hostile, (usize::MAX, 2), Path::new("hostile.nc"))
                .unwrap_err();
        assert!(
            err.contains("hostile.nc") && err.contains("unsafe"),
            "{err}"
        );
    }

    #[test]
    fn publish_recovery_state_machine_covers_every_rename_boundary() {
        use PublishRecoveryAction::{KeepFinal, RemoveAbandoned, RestoreBackup, RollbackInstalled};

        // Journal durable, death before the first rename: retain the old run.
        assert_eq!(
            publish_recovery_action(PublishPhase::Prepared, true, false, true).unwrap(),
            KeepFinal
        );
        // Death after final -> backup, both before and after its phase marker.
        assert_eq!(
            publish_recovery_action(PublishPhase::Prepared, false, true, true).unwrap(),
            RestoreBackup
        );
        assert_eq!(
            publish_recovery_action(PublishPhase::BackupMoved, false, true, true).unwrap(),
            RestoreBackup
        );
        // Death after staged -> final remains pre-commit until FinalInstalled.
        assert_eq!(
            publish_recovery_action(PublishPhase::BackupMoved, true, true, false).unwrap(),
            RollbackInstalled
        );
        assert_eq!(
            publish_recovery_action(PublishPhase::FinalInstalled, true, true, false).unwrap(),
            KeepFinal
        );
        // Death after backup cleanup leaves a complete final and journal only.
        assert_eq!(
            publish_recovery_action(PublishPhase::FinalInstalled, true, false, false).unwrap(),
            KeepFinal
        );
        // Prepared is intent only: with no previous run, abandon the stage and
        // preserve the target's original absence.
        assert_eq!(
            publish_recovery_action(PublishPhase::Prepared, false, false, true).unwrap(),
            RemoveAbandoned
        );
        assert_eq!(
            publish_recovery_action(PublishPhase::Prepared, true, false, false).unwrap(),
            RollbackInstalled
        );
        // A visible marker without its final can be the residue of a failed
        // marker-directory sync followed by rollback; it must not republish.
        assert_eq!(
            publish_recovery_action(PublishPhase::FinalInstalled, false, false, true).unwrap(),
            RemoveAbandoned
        );
        assert!(
            publish_recovery_action(PublishPhase::BackupMoved, false, false, true).is_err(),
            "lost backup after the destructive phase must remain preserved for inspection"
        );
    }

    #[test]
    fn publish_recovery_scan_is_bounded() {
        let root = temp_dir("publisher-recovery-bound");
        let staging_root = root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root).unwrap();
        for index in 0..=MAX_STAGING_RECOVERY_ENTRIES {
            std::fs::create_dir(staging_root.join(format!("unrelated-{index:04}"))).unwrap();
        }
        let run = format!("recover_bound_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        let error = recover_publish_transactions_for_run(
            &root,
            &staging_root,
            &staging_root.join("current-not-present"),
            "wrf",
            &run,
        )
        .unwrap_err();
        assert!(error.contains("refusing an unbounded scan"), "{error}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn publish_recovery_cleans_orphaned_prepared_journal_temporary() {
        let root = temp_dir("publisher-recovery-journal-temp");
        let staging_root = root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root).unwrap();
        let transaction = create_unique_transaction_dir(&staging_root).unwrap();
        std::fs::create_dir(transaction.join(STAGING_WORK_DIR_NAME)).unwrap();
        let model = "wrf";
        let run = format!("recover_temp_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        std::fs::write(
            transaction.join(format!(
                ".publish-{}-prepared.tmp",
                publish_target_key(model, &run)
            )),
            b"partial",
        )
        .unwrap();

        recover_publish_transactions_for_run(
            &root,
            &staging_root,
            &staging_root.join("current-not-present"),
            model,
            &run,
        )
        .unwrap();
        assert!(!transaction.exists());
        assert!(!root.join(model).join(&run).exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn publish_recovery_restores_previous_run_after_first_rename() {
        let root = temp_dir("publisher-recover-backup");
        let model = "wrf";
        let run = format!("recover_backup_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        write_valid_test_run(&root, model, &run, 270.0);
        let final_run = root.join(model).join(&run);
        std::fs::write(final_run.join("old.marker"), b"old").unwrap();

        let staging_root = root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root).unwrap();
        let transaction = create_unique_transaction_dir(&staging_root).unwrap();
        let work = transaction.join(STAGING_WORK_DIR_NAME);
        std::fs::create_dir(&work).unwrap();
        write_valid_test_run(&work, model, &run, 300.0);
        std::fs::write(work.join(model).join(&run).join("new.marker"), b"new").unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::Prepared).unwrap();
        std::fs::rename(&final_run, transaction.join(STAGING_BACKUP_DIR_NAME)).unwrap();

        recover_publish_transactions_for_run(
            &root,
            &staging_root,
            &staging_root.join("current-not-this-transaction"),
            model,
            &run,
        )
        .unwrap();
        assert!(final_run.join("old.marker").is_file());
        assert!(!final_run.join("new.marker").exists());
        assert!(!transaction.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn publish_recovery_preserves_invalid_backup_instead_of_restoring_it() {
        let root = temp_dir("publisher-recover-invalid-backup");
        let model = "wrf";
        let run = format!("recover_invalid_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        write_valid_test_run(&root, model, &run, 270.0);
        let final_run = root.join(model).join(&run);
        let staging_root = root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root).unwrap();
        let transaction = create_unique_transaction_dir(&staging_root).unwrap();
        std::fs::create_dir(transaction.join(STAGING_WORK_DIR_NAME)).unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::Prepared).unwrap();
        let backup = transaction.join(STAGING_BACKUP_DIR_NAME);
        std::fs::rename(&final_run, &backup).unwrap();
        std::fs::write(backup.join("run.json"), b"not valid JSON").unwrap();

        let error = recover_publish_transactions_for_run(
            &root,
            &staging_root,
            &staging_root.join("current-not-this-transaction"),
            model,
            &run,
        )
        .unwrap_err();
        assert!(
            error.contains("validate recovery backup run manifest"),
            "{error}"
        );
        assert!(!final_run.exists());
        assert!(backup.is_dir(), "invalid backup must remain for inspection");
        assert!(transaction.is_dir());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn publish_recovery_rolls_back_second_rename_without_commit_marker() {
        let root = temp_dir("publisher-recover-final");
        let model = "wrf";
        let run = format!("recover_final_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        write_valid_test_run(&root, model, &run, 270.0);
        let final_run = root.join(model).join(&run);
        std::fs::write(final_run.join("old.marker"), b"old").unwrap();

        let staging_root = root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root).unwrap();
        let transaction = create_unique_transaction_dir(&staging_root).unwrap();
        let work = transaction.join(STAGING_WORK_DIR_NAME);
        std::fs::create_dir(&work).unwrap();
        let staged_run = work.join(model).join(&run);
        write_valid_test_run(&work, model, &run, 300.0);
        std::fs::write(staged_run.join("new.marker"), b"new").unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::Prepared).unwrap();
        std::fs::rename(&final_run, transaction.join(STAGING_BACKUP_DIR_NAME)).unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::BackupMoved).unwrap();
        std::fs::rename(&staged_run, &final_run).unwrap();

        recover_publish_transactions_for_run(
            &root,
            &staging_root,
            &staging_root.join("current-not-this-transaction"),
            model,
            &run,
        )
        .unwrap();
        assert!(final_run.join("old.marker").is_file());
        assert!(!final_run.join("new.marker").exists());
        assert!(!transaction.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn publish_recovery_keeps_second_rename_after_durable_commit_marker() {
        let root = temp_dir("publisher-recover-committed-final");
        let model = "wrf";
        let run = format!("recover_committed_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        write_valid_test_run(&root, model, &run, 270.0);
        let final_run = root.join(model).join(&run);
        std::fs::write(final_run.join("old.marker"), b"old").unwrap();

        let staging_root = root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root).unwrap();
        let transaction = create_unique_transaction_dir(&staging_root).unwrap();
        let work = transaction.join(STAGING_WORK_DIR_NAME);
        std::fs::create_dir(&work).unwrap();
        let staged_run = work.join(model).join(&run);
        write_valid_test_run(&work, model, &run, 300.0);
        std::fs::write(staged_run.join("new.marker"), b"new").unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::Prepared).unwrap();
        std::fs::rename(&final_run, transaction.join(STAGING_BACKUP_DIR_NAME)).unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::BackupMoved).unwrap();
        std::fs::rename(&staged_run, &final_run).unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::FinalInstalled).unwrap();

        recover_publish_transactions_for_run(
            &root,
            &staging_root,
            &staging_root.join("current-not-this-transaction"),
            model,
            &run,
        )
        .unwrap();
        assert!(final_run.join("new.marker").is_file());
        assert!(!final_run.join("old.marker").exists());
        assert!(!transaction.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn publish_recovery_discards_prepared_stage_when_no_previous_run_existed() {
        let root = temp_dir("publisher-recover-new");
        let model = "wrf";
        let run = format!("recover_new_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        std::fs::create_dir_all(&root).unwrap();
        let staging_root = root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root).unwrap();
        let transaction = create_unique_transaction_dir(&staging_root).unwrap();
        let work = transaction.join(STAGING_WORK_DIR_NAME);
        std::fs::create_dir(&work).unwrap();
        let staged_run = work.join(model).join(&run);
        write_valid_test_run(&work, model, &run, 300.0);
        std::fs::write(staged_run.join("new.marker"), b"new").unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::Prepared).unwrap();

        recover_publish_transactions_for_run(
            &root,
            &staging_root,
            &staging_root.join("current-not-this-transaction"),
            model,
            &run,
        )
        .unwrap();
        assert!(!root.join(model).join(&run).exists());
        assert!(!staged_run.exists());
        assert!(!transaction.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn publish_recovery_removes_uncommitted_final_when_no_previous_run_existed() {
        let root = temp_dir("publisher-recover-uncommitted-new");
        let model = "wrf";
        let run = format!("recover_uncommitted_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        std::fs::create_dir_all(&root).unwrap();
        let staging_root = root.join(STAGING_DIR_NAME);
        std::fs::create_dir_all(&staging_root).unwrap();
        let transaction = create_unique_transaction_dir(&staging_root).unwrap();
        let work = transaction.join(STAGING_WORK_DIR_NAME);
        std::fs::create_dir(&work).unwrap();
        let staged_run = work.join(model).join(&run);
        write_valid_test_run(&work, model, &run, 300.0);
        std::fs::write(staged_run.join("new.marker"), b"new").unwrap();
        write_publish_journal(&transaction, model, &run, PublishPhase::Prepared).unwrap();
        let final_run = root.join(model).join(&run);
        std::fs::create_dir_all(final_run.parent().unwrap()).unwrap();
        std::fs::rename(&staged_run, &final_run).unwrap();

        recover_publish_transactions_for_run(
            &root,
            &staging_root,
            &staging_root.join("current-not-this-transaction"),
            model,
            &run,
        )
        .unwrap();
        assert!(!final_run.exists());
        assert!(!transaction.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn staging_publisher_swaps_complete_directory_and_removes_backup() {
        let root = temp_dir("publisher-success");
        let run = format!("publisher_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        let mut publisher = RunStagingPublisher::new(&root, "wrf", &run).unwrap();
        std::fs::create_dir_all(&publisher.staged_run_dir).unwrap();
        std::fs::write(publisher.staged_run_dir.join("new.marker"), b"new").unwrap();
        std::fs::create_dir_all(&publisher.final_run_dir).unwrap();
        std::fs::write(publisher.final_run_dir.join("old.marker"), b"old").unwrap();
        let transaction = publisher.transaction_root.clone();

        publisher
            .publish_prevalidated_with(|source, destination| std::fs::rename(source, destination))
            .unwrap();
        assert!(publisher.final_run_dir.join("new.marker").is_file());
        assert!(!publisher.final_run_dir.join("old.marker").exists());
        assert!(
            !transaction.exists(),
            "successful transaction must be cleaned"
        );
        drop(publisher);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn staging_publisher_rolls_back_existing_run_when_publish_rename_fails() {
        let root = temp_dir("publisher-rollback");
        let run = format!("publisher_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        let mut publisher = RunStagingPublisher::new(&root, "wrf", &run).unwrap();
        std::fs::create_dir_all(&publisher.staged_run_dir).unwrap();
        std::fs::write(publisher.staged_run_dir.join("new.marker"), b"new").unwrap();
        std::fs::create_dir_all(&publisher.final_run_dir).unwrap();
        std::fs::write(publisher.final_run_dir.join("old.marker"), b"old").unwrap();
        let final_run = publisher.final_run_dir.clone();
        let transaction = publisher.transaction_root.clone();

        let err = publisher
            .publish_prevalidated_with(|_, _| {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "injected publish failure",
                ))
            })
            .unwrap_err();
        assert!(err.contains("previous run restored"), "{err}");
        assert!(final_run.join("old.marker").is_file());
        assert!(!final_run.join("new.marker").exists());
        assert!(publisher.staged_run_dir.join("new.marker").is_file());
        drop(publisher);
        assert!(
            !transaction.exists(),
            "failed transaction staging must be cleaned"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn staging_publisher_rolls_back_installed_run_before_commit_marker() {
        let root = temp_dir("publisher-precommit-rollback");
        let run = format!("publisher_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        let mut publisher = RunStagingPublisher::new(&root, "wrf", &run).unwrap();
        std::fs::create_dir_all(&publisher.staged_run_dir).unwrap();
        std::fs::write(publisher.staged_run_dir.join("new.marker"), b"new").unwrap();
        std::fs::create_dir_all(&publisher.final_run_dir).unwrap();
        std::fs::write(publisher.final_run_dir.join("old.marker"), b"old").unwrap();
        let transaction = publisher.transaction_root.clone();

        std::fs::rename(&publisher.final_run_dir, &publisher.backup_run_dir).unwrap();
        publisher.backup_active = true;
        std::fs::rename(&publisher.staged_run_dir, &publisher.final_run_dir).unwrap();
        publisher.rollback_uncommitted_install().unwrap();

        assert!(publisher.final_run_dir.join("old.marker").is_file());
        assert!(!publisher.final_run_dir.join("new.marker").exists());
        assert!(publisher.staged_run_dir.join("new.marker").is_file());
        assert!(!publisher.backup_run_dir.exists());
        drop(publisher);
        assert!(
            !transaction.exists(),
            "completed rollback may discard staging"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn staging_publisher_drop_preserves_uncommitted_final_and_backup() {
        let root = temp_dir("publisher-precommit-preserve");
        let run = format!("publisher_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        let mut publisher = RunStagingPublisher::new(&root, "wrf", &run).unwrap();
        std::fs::create_dir_all(&publisher.staged_run_dir).unwrap();
        std::fs::write(publisher.staged_run_dir.join("new.marker"), b"new").unwrap();
        std::fs::create_dir_all(&publisher.final_run_dir).unwrap();
        std::fs::write(publisher.final_run_dir.join("old.marker"), b"old").unwrap();
        let transaction = publisher.transaction_root.clone();
        let final_run = publisher.final_run_dir.clone();
        let backup = publisher.backup_run_dir.clone();

        std::fs::rename(&publisher.final_run_dir, &publisher.backup_run_dir).unwrap();
        publisher.backup_active = true;
        std::fs::rename(&publisher.staged_run_dir, &publisher.final_run_dir).unwrap();
        drop(publisher);

        assert!(final_run.join("new.marker").is_file());
        assert!(backup.join("old.marker").is_file());
        assert!(
            transaction.is_dir(),
            "pre-commit recovery evidence must survive Drop"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn staging_publisher_panic_restores_existing_run_before_releasing_lock() {
        let root = temp_dir("publisher-panic");
        let run = format!("publisher_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        let mut publisher = RunStagingPublisher::new(&root, "wrf", &run).unwrap();
        std::fs::create_dir_all(&publisher.staged_run_dir).unwrap();
        std::fs::write(publisher.staged_run_dir.join("new.marker"), b"new").unwrap();
        std::fs::create_dir_all(&publisher.final_run_dir).unwrap();
        std::fs::write(publisher.final_run_dir.join("old.marker"), b"old").unwrap();
        let final_run = publisher.final_run_dir.clone();
        let transaction = publisher.transaction_root.clone();

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _ = publisher.publish_prevalidated_with(|_, _| {
                panic!("injected panic while publishing staged run")
            });
        }));
        assert!(unwind.is_err(), "injected publisher panic must unwind");
        assert!(final_run.join("old.marker").is_file());
        assert!(!final_run.join("new.marker").exists());
        assert!(!transaction.exists(), "panic cleanup must remove staging");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn staging_publisher_drop_cleans_unpublished_work_and_rejects_traversal() {
        let root = temp_dir("publisher-drop");
        let run = format!("publisher_{IMPORT_SCIENCE_SCHEMA_VERSION}");
        let transaction = {
            let publisher = RunStagingPublisher::new(&root, "wrf", &run).unwrap();
            let resolved_root = std::fs::canonicalize(&root).unwrap();
            assert!(
                publisher
                    .transaction_root
                    .starts_with(resolved_root.join(STAGING_DIR_NAME)),
                "staging must stay under the hidden non-model subtree"
            );
            assert!(
                !publisher.transaction_root.join("run.json").exists(),
                "transaction root must never look like an enumerable model run"
            );
            std::fs::create_dir_all(&publisher.staged_run_dir).unwrap();
            std::fs::write(publisher.staged_run_dir.join("partial.marker"), b"partial").unwrap();
            publisher.transaction_root.clone()
        };
        assert!(!transaction.exists(), "drop must clean unpublished staging");
        assert!(RunStagingPublisher::new(&root, "../wrf", &run).is_err());
        assert!(RunStagingPublisher::new(&root, "wrf", "../science_v1").is_err());
        assert!(RunStagingPublisher::new(&root, "wrf", "local_notscience_v1ish").is_err());
        assert!(has_exact_science_schema_token(IMPORT_SCIENCE_SCHEMA_VERSION));
        assert!(has_exact_science_schema_token(&format!("local_{IMPORT_SCIENCE_SCHEMA_VERSION}")));
        assert!(has_exact_science_schema_token(&format!("archive_{IMPORT_SCIENCE_SCHEMA_VERSION}_deadbeef")));
        assert!(!has_exact_science_schema_token(&format!("local_not{IMPORT_SCIENCE_SCHEMA_VERSION}")));
        assert!(!has_exact_science_schema_token(&format!("local_{IMPORT_SCIENCE_SCHEMA_VERSION}0")));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn polar_projection_pole_follows_truelat1_not_domain_center() {
        let truelat1 = -60.0;
        let cen_lat = 10.0;
        assert!(cen_lat > 0.0, "fixture must exercise conflicting metadata");
        assert!(
            wrf_polar_uses_south_pole(truelat1),
            "TRUELAT1={truelat1} must select the south pole even when CEN_LAT={cen_lat}"
        );

        let truelat1 = 60.0;
        let cen_lat = -10.0;
        assert!(cen_lat < 0.0, "fixture must exercise conflicting metadata");
        assert!(
            !wrf_polar_uses_south_pole(truelat1),
            "TRUELAT1={truelat1} must select the north pole even when CEN_LAT={cen_lat}"
        );
    }

    #[test]
    fn projection_metadata_fallbacks_match_wrf_python() {
        assert_eq!(normalize_lambert_truelat2(30.0, None), 30.0);
        assert_eq!(normalize_lambert_truelat2(30.0, Some(95.0)), 30.0);
        assert_eq!(normalize_lambert_truelat2(30.0, Some(f64::NAN)), 30.0);
        assert_eq!(normalize_lambert_truelat2(30.0, Some(60.0)), 60.0);

        // A nested domain's CEN_LON is deliberately irrelevant here: missing
        // STAND_LON on Mercator means zero, exactly as in wrf-python.
        let cen_lon = -97.5;
        assert_ne!(cen_lon, 0.0);
        assert_eq!(wrf_mercator_central_longitude(None), 0.0);
        assert_eq!(wrf_mercator_central_longitude(Some(12.5)), 12.5);

        assert!(wrf_latlon_is_unrotated(None, None));
        assert!(wrf_latlon_is_unrotated(Some(90.0), Some(0.0)));
        assert!(!wrf_latlon_is_unrotated(Some(45.0), Some(180.0)));
        assert!(!wrf_latlon_is_unrotated(Some(90.0), None));
        assert!(!wrf_latlon_is_unrotated(None, Some(0.0)));
    }

    #[test]
    fn canonical_pressure_units_convert_to_pa_or_fail_closed() {
        for units in ["Pa", "pascal", "PASCALS"] {
            assert_eq!(pressure_scale_to_pa(Some(units), 99.0), Some(1.0));
        }
        for units in ["hPa", "mb", "mbar", "hecto_pascals"] {
            assert_eq!(pressure_scale_to_pa(Some(units), 99.0), Some(100.0));
        }
        assert_eq!(pressure_scale_to_pa(Some("kPa"), 99.0), Some(1_000.0));
        assert_eq!(pressure_scale_to_pa(None, 100.0), Some(100.0));
        assert_eq!(pressure_scale_to_pa(Some("psi"), 100.0), None);
    }

    #[test]
    fn postprocessed_3d_unit_conversions_are_explicit() {
        assert_eq!(
            postprocessed_unit_affine("degC", PostprocessedUnitKind::TemperatureKelvin),
            Some((1.0, 273.15))
        );
        assert_eq!(
            postprocessed_unit_affine("hPa", PostprocessedUnitKind::PressurePascal),
            Some((100.0, 0.0))
        );
        assert_eq!(
            postprocessed_unit_affine("g kg-1", PostprocessedUnitKind::MixingRatioKgKg),
            Some((0.001, 0.0))
        );
        assert_eq!(
            postprocessed_unit_affine("knots", PostprocessedUnitKind::WindMetersPerSecond),
            Some((0.514_444, 0.0))
        );
        assert!(postprocessed_unit_affine("psi", PostprocessedUnitKind::PressurePascal).is_none());
    }

    #[test]
    fn wrf_wind_rotation_matches_uvmet_convention_at_every_level() {
        let rotation = WrfWindRotation::Angles {
            sin: vec![1.0, 0.0],
            cos: vec![0.0, 1.0],
        };
        let u = Plane2D {
            nx: 2,
            ny: 1,
            values: vec![2.0, 4.0],
        };
        let v = Plane2D {
            nx: 2,
            ny: 1,
            values: vec![3.0, 5.0],
        };
        let (ue, ve) = rotation.rotate_f32_pair(&u, &v).unwrap();
        assert_eq!(ue, vec![-3.0, 4.0]);
        assert_eq!(ve, vec![2.0, 5.0]);

        let mut u3 = vec![2.0, 4.0, 6.0, 8.0];
        let mut v3 = vec![3.0, 5.0, 7.0, 9.0];
        rotation
            .rotate_f64_levels_in_place(&mut u3, &mut v3, 2)
            .unwrap();
        assert_eq!(u3, vec![-3.0, 4.0, -7.0, 8.0]);
        assert_eq!(v3, vec![2.0, 5.0, 6.0, 9.0]);
    }

    #[test]
    fn wrf_and_cf_timestamp_parsing_is_utc_exact() {
        assert_eq!(
            parse_utc_timestamp("2026-07-09_18:30:00"),
            parse_utc_timestamp("2026-07-09T18:30:00Z")
        );
        let (scale, origin) =
            cf_time_unit("hours since 2026-07-09 18:30:00 UTC").expect("CF hour units");
        assert_eq!(scale, 3_600.0);
        assert_eq!(origin, parse_utc_timestamp("2026-07-09T18:30:00Z").unwrap());
    }

    #[test]
    fn destagger_z_averages_adjacent_w_levels_and_truncates() {
        // 3 mass levels, 2 columns; staggered = 4 levels. Level-major layout:
        // [k0c0, k0c1, k1c0, k1c1, ...].
        let mut z = vec![
            0.0, 100.0, // stag level 0
            10.0, 110.0, // stag level 1
            30.0, 130.0, // stag level 2
            70.0, 170.0, // stag level 3
        ];
        destagger_z_to_mass_levels(&mut z, 3, 2).expect("valid staggered dimensions");
        assert_eq!(z, vec![5.0, 105.0, 20.0, 120.0, 50.0, 150.0]);

        let mut malformed = vec![0.0; 7];
        assert!(
            destagger_z_to_mass_levels(&mut malformed, 3, 2).is_err(),
            "a shape/data-length mismatch must return an import error, not index-panic"
        );
    }

    fn temp_dir(name: &str) -> PathBuf {
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        // The process id too: this module's tests compile into BOTH the lib and the rw_wrfbatch bin test binaries, which
        // the cargo gate runs at once, and the two counters repeat (2.9.4 R7b route A, Linux container, 2026-10-21: the
        // other binary's fixture wrote old.marker into this test's freshly published run directory).
        let unique = now_unix();
        std::env::temp_dir().join(format!("rw-local-import-{name}-{}-{unique}-{}", std::process::id(), NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)))
    }

    fn write_valid_test_run(store_root: &Path, model: &str, run: &str, value: f32) {
        let shape = GridShape::new(2, 2).unwrap();
        let grid = LatLonGrid::new(
            shape,
            vec![40.0, 40.0, 39.0, 39.0],
            vec![-101.0, -100.0, -101.0, -100.0],
        )
        .unwrap();
        let field = SelectedField2D::new(
            FieldSelector::height_agl(CanonicalField::Temperature, 2),
            "K",
            grid,
            vec![value; 4],
        )
        .unwrap();
        write_hour_from_fields_with_derived(
            store_root,
            model,
            run,
            0,
            &[("temperature_2m", &field)],
            &[],
            &[],
            "publish-recovery-test",
            1,
        )
        .unwrap();
    }

    /// The post-processed routing rule (owner-reported CONUS-II wrf2d
    /// misroute, "bad shape for variable TK: [1419, 1429]"): single-plane TK
    /// (surface archive) routes 2-D, model-level TK (wrf3d, either Z era)
    /// stays on the 3-D path. Dim names/shapes taken from the real GDEX
    /// files.
    #[test]
    fn postproc_routing_separates_wrf2d_from_wrf3d() {
        // Real wrf2d TK: (Time, south_north, west_east) = (1, 1419, 1429).
        assert!(postproc_tk_is_2d(
            &["Time", "south_north", "west_east"],
            &[1, 1419, 1429]
        ));
        // Real wrf3d TK: (Time, bottom_top, south_north, west_east).
        assert!(!postproc_tk_is_2d(
            &["Time", "bottom_top", "south_north", "west_east"],
            &[1, 50, 1419, 1429]
        ));
        // No record dim: bare planes route 2-D, bare stacks route 3-D.
        assert!(postproc_tk_is_2d(
            &["south_north", "west_east"],
            &[1419, 1429]
        ));
        assert!(!postproc_tk_is_2d(
            &["bottom_top", "south_north", "west_east"],
            &[50, 1419, 1429]
        ));
        // Degenerate ranks and dims/shape disagreement never claim 2-D.
        assert!(!postproc_tk_is_2d(&["Time"], &[1]));
        assert!(!postproc_tk_is_2d(
            &["Time", "south_north"],
            &[1, 1419, 1429]
        ));
    }

    /// The wrf2d plane scanner: accepts single mass-grid planes in all three
    /// stored shapes, rejects coordinates, bookkeeping axes, staggered
    /// single-level winds, and model-level stacks. Names/shapes from the
    /// real wrf2d probe (192 variables, 185 mass-grid data planes).
    #[test]
    fn wrf2d_plane_scanner_selects_mass_grid_data_vars() {
        let (ny, nx) = (1419usize, 1429usize);
        let t = &["Time", "south_north", "west_east"][..];
        // Data planes: float and int-bucket vars alike.
        assert!(is_postproc_2d_data_plane("TK", t, &[1, ny, nx], ny, nx));
        assert!(is_postproc_2d_data_plane("SBCAPE", t, &[1, ny, nx], ny, nx));
        assert!(is_postproc_2d_data_plane(
            "I_ACLWDNB",
            t,
            &[1, ny, nx],
            ny,
            nx
        ));
        // Bare (ny, nx) planes count too, as does a non-Time leading dim of
        // length 1, and a multi-record Time dim (the requested record is read).
        assert!(is_postproc_2d_data_plane(
            "TSK",
            &["south_north", "west_east"],
            &[ny, nx],
            ny,
            nx
        ));
        assert!(is_postproc_2d_data_plane(
            "TSK",
            &["level", "south_north", "west_east"],
            &[1, ny, nx],
            ny,
            nx
        ));
        assert!(is_postproc_2d_data_plane("T2", t, &[4, ny, nx], ny, nx));
        // Coordinates and bookkeeping are never data.
        assert!(!is_postproc_2d_data_plane(
            "XLAT",
            &["south_north", "west_east"],
            &[ny, nx],
            ny,
            nx
        ));
        assert!(!is_postproc_2d_data_plane(
            "lat",
            &["south_north", "west_east"],
            &[ny, nx],
            ny,
            nx
        ));
        assert!(!is_postproc_2d_data_plane("XTIME", &["Time"], &[1], ny, nx));
        assert!(!is_postproc_2d_data_plane(
            "Times",
            &["Time", "DateStrLen"],
            &[1, 19],
            ny,
            nx
        ));
        // Staggered single-level winds do not sit on the mass grid.
        assert!(!is_postproc_2d_data_plane(
            "U",
            &["Time", "south_north", "west_east_stag"],
            &[1, ny, nx + 1],
            ny,
            nx
        ));
        assert!(!is_postproc_2d_data_plane(
            "V",
            &["Time", "south_north_stag", "west_east"],
            &[1, ny + 1, nx],
            ny,
            nx
        ));
        // Model-level stacks belong to the 3-D route.
        assert!(!is_postproc_2d_data_plane(
            "TK",
            &["Time", "bottom_top", "south_north", "west_east"],
            &[1, 50, ny, nx],
            ny,
            nx
        ));

        assert!(is_postproc_2d_hdf5_data_plane(
            "U10",
            &[1, ny as u64, nx as u64],
            false,
            ny,
            nx
        ));
        assert!(is_postproc_2d_hdf5_data_plane(
            "MUCAPE",
            &[4, ny as u64, nx as u64],
            true,
            ny,
            nx
        ));
        assert!(!is_postproc_2d_hdf5_data_plane(
            "MUCAPE",
            &[4, ny as u64, nx as u64],
            false,
            ny,
            nx
        ));
        assert!(!is_postproc_2d_hdf5_data_plane(
            "U",
            &[1, ny as u64, nx as u64 + 1],
            true,
            ny,
            nx
        ));
    }

    #[test]
    fn conus_ii_wrf2d_completeness_fails_closed_on_reader_omissions() {
        let raw = ["wrf_u10", "wrf_mucape", "wrf_srh03", "wrf_swupt"];
        let canonical = ["u_10m", "v_10m"];
        assert_eq!(
            missing_conus_ii_wrf2d_fields(
                1429,
                1419,
                raw.iter().copied(),
                canonical.iter().copied()
            ),
            vec!["wrf_acedir", "wind_speed_10m"]
        );

        // The exact field contract is specific to the measured CONUS-II
        // archive geometry; another wrf2d dialect is not rejected for a
        // scientifically legitimate, smaller diagnostic suite.
        assert!(
            missing_conus_ii_wrf2d_fields(100, 100, std::iter::empty(), std::iter::empty())
                .is_empty()
        );
    }

    #[test]
    fn conus_ii_wrf2d_completeness_accepts_all_recovered_fields() {
        assert!(
            missing_conus_ii_wrf2d_fields(
                1429,
                1419,
                CONUS_II_WRF2D_REQUIRED_RAW_FIELDS.iter().copied(),
                CONUS_II_WRF2D_REQUIRED_CANONICAL_WINDS.iter().copied(),
            )
            .is_empty()
        );
    }

    /// Real-data proof for the CONUS-II `wrf2d` 2-D route (the owner's
    /// failing file): runs the full light import on the surface archive
    /// named by `RW_WRF2D_FIXTURE` and asserts the canonical suite + raw
    /// `wrf_*` planes land with physical values and NO iso volumes. Skips
    /// (passing) when the env var is unset.
    #[test]
    fn optional_wrf2d_fixture_imports_surface_planes() {
        let Ok(fixture) = std::env::var("RW_WRF2D_FIXTURE") else {
            eprintln!("skipping; set RW_WRF2D_FIXTURE to a CONUS-II wrf2d file");
            return;
        };
        let store_root = temp_dir("wrf2d");
        let start = std::time::Instant::now();
        let summary = import_paths(&[PathBuf::from(&fixture)], &store_root, &mut |message| {
            eprintln!("[{:9.2?}] {message}", start.elapsed());
        })
        .unwrap();
        eprintln!(
            "[{:9.2?}] DONE: {} hour(s), {} variables; peak RSS {}",
            start.elapsed(),
            summary.hours_written,
            summary.variables.len(),
            peak_rss_label()
        );
        assert_eq!(summary.model, "wrf");
        assert_eq!(summary.hours_written, 1);
        // Canonical suite from the file's own T2/Q2/PSFC/U10/V10 planes.
        // These wind assertions prove the formerly omitted internal B-tree
        // record for U10 is now both discoverable and readable.
        for var in [
            "temperature_2m",
            "dewpoint_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "u_10m",
            "v_10m",
            "wind_speed_10m",
        ] {
            assert!(
                summary.variables.iter().any(|name| name == var),
                "{var} missing: {:?}",
                summary.variables
            );
        }
        // WSPD10MAX is a running maximum, not a gust and not the
        // instantaneous speed; it must not appear under a canonical
        // selector at all.  Its raw plane is still readable under the
        // light-import `wrf_*` naming.
        assert!(
            !summary
                .variables
                .iter()
                .any(|name| name == "wind_speed_10m_max"),
            "WSPD10MAX must not be published as a canonical plane: {:?}",
            summary.variables
        );
        // Raw planes: the misrouting trio plus a severe plane, an
        // accumulated-flux plane, and every formerly omitted internal-node
        // record, under the light-import wrf_* naming.
        for var in [
            "wrf_tk",
            "wrf_z",
            "wrf_p",
            "wrf_sbcape",
            "wrf_aclwdnb",
            "wrf_u10",
            "wrf_v10",
            "wrf_mucape",
            "wrf_srh03",
            "wrf_swupt",
            "wrf_acedir",
        ] {
            assert!(
                summary.variables.iter().any(|name| name == var),
                "{var} missing: {:?}",
                summary.variables
            );
        }
        // The 2-D route must land the full plane set the metadata listing
        // exposes (185 raw plus the canonical suite on the real file).
        assert!(
            summary.variables.len() >= 190,
            "expected the full 2-D plane set, got {} variables: {:?}",
            summary.variables.len(),
            summary.variables
        );
        // A pure 2-D archive must not synthesize sounding volumes.
        assert!(
            !summary.variables.iter().any(|name| name.ends_with("_iso")),
            "unexpected iso volumes: {:?}",
            summary.variables
        );

        // Value roundtrip through the store: lowest-model-level TK must be
        // physical air temperature over the CONUS grid.
        let hour = store_root
            .join(&summary.model)
            .join(&summary.run)
            .join("f000.rws");
        let reader = rw_store::reader::HourReader::open(&hour).expect("open hour");
        let tk = reader.read_full_2d("wrf_tk").expect("read wrf_tk");
        let finite = tk.iter().filter(|value| value.is_finite()).count();
        assert!(
            finite > tk.len() / 2,
            "wrf_tk mostly NaN: {finite}/{}",
            tk.len()
        );
        for value in tk.iter().filter(|value| value.is_finite()) {
            assert!((180.0..=340.0).contains(value), "TK {value} K non-physical");
        }

        let _ = std::fs::remove_dir_all(store_root);
    }

    #[test]
    fn wrf_timestamp_accepts_colon_and_underscore_time() {
        // Both spellings must resolve to the same exact UTC time.  The old
        // assertion stopped at a normalised "19740403_090000" string that
        // the timestamp parser could not read, so this test passed while
        // every file-name time fell through to the run's start.
        let colon = Path::new("wrfout_d02_1974-04-03_09:00:00");
        let underscore = Path::new("wrfout_d02_1974-04-03_09_00_00");
        let expected = parse_utc_timestamp("1974-04-03T09:00:00Z").unwrap();
        assert_eq!(
            wrf_timestamp_in_file_name(colon),
            Some("1974-04-03_09:00:00")
        );
        assert_eq!(
            wrf_timestamp_in_file_name(underscore),
            Some("1974-04-03_09_00_00")
        );
        assert_eq!(valid_unix_from_file_name(colon, "test").unwrap(), expected);
        assert_eq!(
            valid_unix_from_file_name(underscore, "test").unwrap(),
            expected
        );
    }

    #[test]
    fn folder_scan_finds_extensionless_nested_wrf_files() {
        let root = temp_dir("scan");
        let nested = root.join("member").join("d02");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::File::create(nested.join("wrfout_d02_1974-04-03_09_00_00")).unwrap();
        std::fs::File::create(root.join("not_a_model.txt")).unwrap();

        let files = supported_files_in_folder(&root);
        assert_eq!(files.len(), 1);
        assert!(
            files[0]
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("wrfout")
        );

        let _ = std::fs::remove_dir_all(root);
    }

    /// Instrumented real-fixture guard for the light import (the "📄 WRF/
    /// NetCDF file…" dock path): runs `import_paths` on the wrfout named by
    /// `RW_LOCAL_IMPORT_FIXTURE`, forwarding every progress line to stderr
    /// with a timestamp and printing peak RSS (`VmHWM`) at the end, the
    /// before/after measurement harness for the large-grid memory fix
    /// (docs/wrf-import-large-grids.md). Release builds only on large grids.
    #[test]
    fn optional_wrf_fixture_imports_to_store() {
        let Ok(fixture) = std::env::var("RW_LOCAL_IMPORT_FIXTURE") else {
            eprintln!("skipping WRF import fixture; set RW_LOCAL_IMPORT_FIXTURE");
            return;
        };
        let store_root = temp_dir("store");
        let start = std::time::Instant::now();
        let mut lines = Vec::new();
        let summary = import_paths(&[PathBuf::from(&fixture)], &store_root, &mut |message| {
            eprintln!("[{:9.2?}] {message}", start.elapsed());
            lines.push(message);
        })
        .unwrap();
        eprintln!(
            "[{:9.2?}] DONE: {} hour(s), {} variables; peak RSS {}",
            start.elapsed(),
            summary.hours_written,
            summary.variables.len(),
            peak_rss_label()
        );
        assert_eq!(summary.model, "wrf");
        assert_eq!(summary.hours_written, 1);
        assert!(summary.variables.iter().any(|var| var == "temperature_2m"));
        assert!(summary.variables.iter().any(|var| var == "dewpoint_2m"));
        assert!(summary.variables.iter().any(|var| var == "wind_speed_10m"));
        // No `apcp` assert: this harness runs against ANY wrfout, and some
        // (e.g. the Enderlin 250 m d03 outputs) carry no RAINC/RAINNC, the
        // variables line above shows what the file actually yielded.
        // Progress must stream per-stage detail, not one line per file: the
        // 2D read, each wrf-core sounding field, interpolation percentages,
        // and the store write all pass through the same channel the dock
        // renders.
        assert!(
            lines.iter().any(|l| l.contains("file 1/1")),
            "stage lines must carry the file position: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains("reading 2D surface fields")),
            "missing 2D-read stage: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("sounding field 5/5")),
            "missing per-field getvar stages: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("isobaric levels")),
            "missing interpolation stages: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l.contains("writing to store")),
            "missing store-write stage: {lines:?}"
        );

        let _ = std::fs::remove_dir_all(store_root);
    }

    /// Value-identity proof for the fast 2-D read path
    /// (docs/wrf-import-large-grids.md): on the real wrfout fixture
    /// (`RW_LOCAL_IMPORT_FIXTURE`, same resolution as
    /// `optional_wrf_fixture_imports_to_store`), every `[Time, …, ny, nx]`
    /// plane must be BIT-identical between the legacy netcrust read
    /// (`read_first_2d_netcrust`) and the wrf-core fast path, the fast path
    /// must actually engage for every such plane (no silent netcrust
    /// fallback), and the end-to-end 2-D field set the import builds
    /// (canonical + raw + grid) must match name-for-name, unit-for-unit,
    /// bit-for-bit between `read_wrf_2d_fields` with and without the wrf-core
    /// handle.
    #[test]
    fn optional_wrf_fixture_fast_and_netcrust_2d_reads_match() {
        let Ok(fixture) = std::env::var("RW_LOCAL_IMPORT_FIXTURE") else {
            eprintln!("skipping WRF read-path identity; set RW_LOCAL_IMPORT_FIXTURE");
            return;
        };
        let path = PathBuf::from(&fixture);
        let nc = netcrust::open(&path).expect("netcrust opens the fixture");
        let wrf = WrfFile::open(&path).expect("wrf-core opens the wrfout fixture");

        // Per-plane sweep: every WRF-record-layout variable in the file, a
        // superset of the planes the import reads (canonical names, derived
        // inputs, and the raw mass-grid loop all go through read_first_2d).
        let slow = PlaneSource::netcrust_only(&nc, 0);
        let fast = PlaneSource::new(&nc, Some(&wrf), 0);
        let mut compared = 0usize;
        for var in nc.variables().expect("list fixture variables") {
            let name = var.name();
            let dims = var.dimensions();
            if dims.len() < 3 || dims[0].name() != "Time" {
                continue;
            }
            let legacy = read_first_2d(&slow, name)
                .unwrap_or_else(|err| panic!("{name}: netcrust read failed: {err}"))
                .unwrap_or_else(|| panic!("{name}: netcrust read yielded no plane"));
            let routed = read_first_2d(&fast, name)
                .unwrap_or_else(|err| panic!("{name}: fast-path read failed: {err}"))
                .unwrap_or_else(|| panic!("{name}: fast-path read yielded no plane"));
            assert_eq!(
                (legacy.nx, legacy.ny),
                (routed.nx, routed.ny),
                "{name}: plane shape differs between read paths"
            );
            assert_bits_eq(name, &legacy.values, &routed.values);
            compared += 1;
        }
        assert!(
            compared >= 20,
            "fixture only exposed {compared} record-layout planes, wrong fixture?"
        );
        assert_eq!(
            fast.wrf_reads.get(),
            compared,
            "every record-layout plane must take the wrf-core fast path"
        );
        assert!(
            fast.netcrust_fallbacks.borrow().is_empty(),
            "unexpected netcrust fallbacks: {:?}",
            fast.netcrust_fallbacks.borrow()
        );
        eprintln!("read-path identity: {compared} planes bit-identical");

        // End-to-end: the exact field set the import writes, both routes.
        let legacy_fields =
            read_wrf_2d_fields(&nc, &path, None, 0, &mut |_: String| {}).expect("legacy 2D read");
        let fast_fields = read_wrf_2d_fields(&nc, &path, Some(&wrf), 0, &mut |_: String| {})
            .expect("fast-path 2D read");
        assert_bits_eq(
            "grid latitudes",
            &legacy_fields.grid.lat_deg,
            &fast_fields.grid.lat_deg,
        );
        assert_bits_eq(
            "grid longitudes",
            &legacy_fields.grid.lon_deg,
            &fast_fields.grid.lon_deg,
        );
        assert_eq!(
            legacy_fields
                .canonical
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            fast_fields
                .canonical
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            "canonical field names/ordering changed"
        );
        for ((name, legacy), (_, routed)) in
            legacy_fields.canonical.iter().zip(&fast_fields.canonical)
        {
            assert_eq!(legacy.units, routed.units, "{name}: units changed");
            assert_bits_eq(name, &legacy.values, &routed.values);
        }
        assert_eq!(
            legacy_fields
                .raw_2d
                .iter()
                .map(|field| field.name.as_str())
                .collect::<Vec<_>>(),
            fast_fields
                .raw_2d
                .iter()
                .map(|field| field.name.as_str())
                .collect::<Vec<_>>(),
            "raw field names/ordering changed"
        );
        for (legacy, routed) in legacy_fields.raw_2d.iter().zip(&fast_fields.raw_2d) {
            assert_eq!(legacy.units, routed.units, "{}: units changed", legacy.name);
            assert_bits_eq(&legacy.name, &legacy.values, &routed.values);
        }
        eprintln!(
            "end-to-end identity: {} canonical + {} raw fields bit-identical",
            legacy_fields.canonical.len(),
            legacy_fields.raw_2d.len()
        );
    }

    /// Bitwise f32 equality (NaN == NaN: both read paths narrow every
    /// non-finite source value to the same `f32::NAN` constant).
    fn assert_bits_eq(name: &str, legacy: &[f32], routed: &[f32]) {
        assert_eq!(legacy.len(), routed.len(), "{name}: plane length differs");
        for (index, (a, b)) in legacy.iter().zip(routed).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "{name}[{index}]: {a} ({:#010x}) != {b} ({:#010x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    /// Peak resident set (Linux `VmHWM`), for the instrumented fixture runs on
    /// the verify node; other platforms report unavailable.
    fn peak_rss_label() -> String {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find(|line| line.starts_with("VmHWM"))
                    .map(|line| {
                        line.split_whitespace()
                            .skip(1)
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
            })
            .unwrap_or_else(|| "unavailable (no /proc)".to_string())
    }

    /// The dock consumes the worker through the message channel: a failing
    /// import must still deliver a terminal `Done(Err)` (never a hang, never
    /// a bare disconnect), the UI's completion path depends on it.
    #[test]
    fn spawn_import_delivers_done_error_for_bad_selection() {
        let task = spawn_import_paths(
            vec![PathBuf::from("definitely-missing.wrfout.nc")],
            temp_dir("spawn-err"),
        );
        loop {
            match task.rx.recv() {
                Ok(LocalImportMessage::Progress(_)) => continue,
                Ok(LocalImportMessage::Done(result)) => {
                    result.expect_err("missing file must fail the import");
                    break;
                }
                Err(err) => panic!("worker died without Done: {err}"),
            }
        }
    }

    /// End-to-end guard for the post-processed climate-wrfout path (TK/Z/P, no
    /// raw T/PB, no surface fields): the store must land the `*_iso` volumes +
    /// an explicitly approximate surface-pressure anchor, with physical temps, monotonic height,
    /// and sane winds. Gated on `RW_POSTPROCESSED_WRF_FIXTURE` (a `wrf3d`-style
    /// CONUS-I/II / GDEX file).
    #[test]
    fn optional_postprocessed_fixture_sounds() {
        let Ok(fixture) = std::env::var("RW_POSTPROCESSED_WRF_FIXTURE") else {
            eprintln!("skipping; set RW_POSTPROCESSED_WRF_FIXTURE to a TK/Z/P wrf3d file");
            return;
        };
        let store_root = temp_dir("postproc");
        let summary = import_paths(&[PathBuf::from(&fixture)], &store_root, &mut |message| {
            eprintln!("{message}");
        })
        .unwrap();
        assert_eq!(summary.model, "wrf");
        assert_eq!(summary.hours_written, 1);
        assert!(
            summary
                .variables
                .iter()
                .all(|name| !crate::postproc_severe::APPROX_SEVERE_SLUGS.contains(&name.as_str())),
            "light import must not run the approximate severe suite: {:?}",
            summary.variables
        );

        let hour = store_root
            .join(&summary.model)
            .join(&summary.run)
            .join("f000.rws");
        let reader = rw_store::reader::HourReader::open(&hour).expect("open hour");
        for name in [
            "temperature_iso",
            "dewpoint_iso",
            "u_iso",
            "v_iso",
            "height_iso",
        ] {
            let var = reader
                .variable(name)
                .unwrap_or_else(|| panic!("{name} missing"));
            assert_eq!(var.kind, "pressure3d", "{name} should be a volume");
            assert!(!var.levels_hpa.is_empty(), "{name} has no levels");
        }
        assert!(
            reader.variable("approx_surface_pressure").is_some(),
            "approx_surface_pressure must be synthesized from the lowest level"
        );

        let temps = reader.read_profile_3d("temperature_iso", 5.0, 5.0).unwrap();
        let heights = reader.read_profile_3d("height_iso", 5.0, 5.0).unwrap();
        let us = reader.read_profile_3d("u_iso", 5.0, 5.0).unwrap();
        let vs = reader.read_profile_3d("v_iso", 5.0, 5.0).unwrap();

        let finite_t = temps.iter().filter(|value| value.is_finite()).count();
        assert!(finite_t >= 5, "expected finite temps, got {finite_t}");
        for temp in &temps {
            if temp.is_finite() {
                assert!((180.0..=330.0).contains(temp), "T {temp} K non-physical");
            }
        }
        let mut last = f32::NEG_INFINITY;
        for height in &heights {
            if height.is_finite() {
                assert!(*height > last, "height {height} after {last}");
                last = *height;
            }
        }
        for (u, v) in us.iter().zip(&vs) {
            if u.is_finite() {
                assert!(u.abs() < 150.0, "u {u} m/s implausible");
            }
            if v.is_finite() {
                assert!(v.abs() < 150.0, "v {v} m/s implausible");
            }
        }

        let _ = std::fs::remove_dir_all(store_root);
    }

    /// Real-data proof + timing harness for the post-processed severe suite:
    /// runs the full diagnostics path on the GDEX wrf3d file named by
    /// `RW_POSTPROC_SEVERE_FIXTURE` and asserts every wrf-core-met severe
    /// slug lands in the store with physically sane values. Every progress
    /// line is timestamped (the `severe suite [..]: done` line carries the
    /// suite's own wall time) and peak RSS prints at the end. `#[ignore]`d:
    /// needs a multi-GB real file and minutes of parcel lifts, run once on a
    /// verify node, release build:
    /// `RW_POSTPROC_SEVERE_FIXTURE=/tmp/wrf3d_... cargo test --release
    ///  -p app_ui optional_postproc_severe -- --ignored --nocapture`
    #[test]
    #[ignore = "needs RW_POSTPROC_SEVERE_FIXTURE (real post-processed wrf3d file); run release on a node"]
    fn optional_postproc_severe_fixture_lands_sane_fields() {
        let fixture = std::env::var("RW_POSTPROC_SEVERE_FIXTURE")
            .expect("set RW_POSTPROC_SEVERE_FIXTURE to a TK/Z/P wrf3d file");
        let store_root = temp_dir("postproc-severe");
        let start = std::time::Instant::now();
        let task = crate::wrf_process::spawn_process_paths(
            vec![PathBuf::from(&fixture)],
            store_root.clone(),
            crate::wrf_process::WrfProcessOptions::default(),
        );
        let summary = loop {
            match task.rx.recv() {
                Ok(crate::wrf_process::WrfProcessMessage::Progress(message)) => {
                    eprintln!("[{:9.2?}] {message}", start.elapsed());
                }
                Ok(crate::wrf_process::WrfProcessMessage::Done(result)) => {
                    break result.expect("post-processed severe processing");
                }
                Err(err) => panic!("post-processed severe worker stopped: {err}"),
            }
        };
        eprintln!(
            "[{:9.2?}] DONE: {} hour(s), {} variables; peak RSS {}",
            start.elapsed(),
            summary.hours_written,
            summary.variables.len(),
            peak_rss_label()
        );
        assert_eq!(summary.model, "wrf");
        assert_eq!(summary.hours_written, 1);

        const SEVERE_SLUGS: [&str; 16] = [
            "approx_sbcape",
            "approx_sbcin",
            "approx_mlcape",
            "approx_mlcin",
            "approx_mucape",
            "approx_mucin",
            "approx_lcl",
            "approx_lfc",
            "approx_el",
            "approx_srh_0_1km",
            "approx_srh_0_3km",
            "approx_bulk_shear_0_1km",
            "approx_bulk_shear_0_6km",
            "approx_stp",
            "approx_scp",
            "approx_ehi_0_1km",
        ];
        for slug in SEVERE_SLUGS {
            assert!(
                summary.variables.iter().any(|name| name == slug),
                "{slug} missing from import summary: {:?}",
                summary.variables
            );
        }

        let hour = store_root
            .join(&summary.model)
            .join(&summary.run)
            .join("f000.rws");
        let reader = rw_store::reader::HourReader::open(&hour).expect("open hour");
        let plane = |slug: &str| -> Vec<f32> {
            reader
                .read_full_2d(slug)
                .unwrap_or_else(|err| panic!("{slug}: read_full_2d failed: {err}"))
        };
        let finite_stats = |slug: &str, values: &[f32]| -> (usize, f32, f32) {
            let mut count = 0usize;
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &value in values {
                if value.is_finite() {
                    count += 1;
                    min = min.min(value);
                    max = max.max(value);
                }
            }
            assert!(count > 0, "{slug}: entirely NaN");
            eprintln!("{slug}: {count} finite, min {min}, max {max}");
            (count, min, max)
        };

        // CAPE: nonnegative and physically bounded on every parcel flavor.
        for slug in ["approx_sbcape", "approx_mlcape", "approx_mucape"] {
            let values = plane(slug);
            let (_, min, max) = finite_stats(slug, &values);
            assert!(min >= 0.0, "{slug}: negative CAPE {min}");
            assert!(max <= 8000.0, "{slug}: implausible CAPE {max}");
        }
        // CIN: never positive (kernel accumulates negative buoyancy only).
        for slug in ["approx_sbcin", "approx_mlcin", "approx_mucin"] {
            let values = plane(slug);
            let (_, _, max) = finite_stats(slug, &values);
            assert!(max <= 0.0, "{slug}: positive CIN {max}");
        }
        // Parcel levels: nonnegative heights below the model top.
        for slug in ["approx_lcl", "approx_lfc", "approx_el"] {
            let values = plane(slug);
            let (_, min, max) = finite_stats(slug, &values);
            assert!(min >= 0.0, "{slug}: negative height {min} m AGL");
            assert!(max < 25_000.0, "{slug}: height {max} m above model top");
        }
        // Kinematics: bounded magnitudes.
        for slug in ["approx_srh_0_1km", "approx_srh_0_3km"] {
            let values = plane(slug);
            let (_, min, max) = finite_stats(slug, &values);
            assert!(
                min > -3000.0 && max < 3000.0,
                "{slug}: implausible SRH range {min}..{max}"
            );
        }
        for slug in ["approx_bulk_shear_0_1km", "approx_bulk_shear_0_6km"] {
            let values = plane(slug);
            let (_, min, max) = finite_stats(slug, &values);
            assert!(min >= 0.0, "{slug}: negative shear magnitude {min}");
            assert!(max < 150.0, "{slug}: shear {max} m/s implausible");
        }
        // Composites: finite (finite_stats already proves that) and STP/SCP
        // nonnegative by construction.
        for slug in ["approx_stp", "approx_scp"] {
            let values = plane(slug);
            let (_, min, _) = finite_stats(slug, &values);
            assert!(min >= 0.0, "{slug}: negative composite {min}");
        }
        finite_stats("approx_ehi_0_1km", &plane("approx_ehi_0_1km"));

        let _ = std::fs::remove_dir_all(store_root);
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ImportError {
    #[error("no files selected")]
    NoFiles,
    #[error("no supported local model files found in selection")]
    NoSupportedFiles,
    #[error("folder contains too many files to map into rw-store forecast-hour slots: {0}")]
    TooManyFiles(usize),
    #[error("missing any required grid variable: {0:?}")]
    MissingAny(Vec<String>),
    #[error("bad shape for variable {0}: {1:?}")]
    BadShape(String, Vec<usize>),
    #[error("XLAT/XLONG grid dimensions do not match in {0}")]
    GridMismatch(PathBuf),
    #[error("WRF planes do not share the same grid shape")]
    PlaneMismatch,
    #[error("no importable 2D WRF fields found in {0}")]
    NoFields(PathBuf),
    #[error("invalid or unrepresentable model time axis: {0}")]
    TimeAxis(String),
    #[error("cannot identify selected source set: {0}")]
    SourceIdentity(String),
    #[error("cannot atomically publish imported run: {0}")]
    Publish(String),
    #[error("GRIB1 import failed: {0}")]
    Grib(String),
    #[error("selection mixes GRIB1 and WRF/NetCDF files, import them separately")]
    MixedGribSelection,
    #[error("cannot map {variable} to canonical pressure: unsupported units {units:?}")]
    UnsupportedPressureUnits { variable: String, units: String },
    #[error(
        "cannot normalize post-processed {variable}: unsupported units {units:?}; expected {expected}"
    )]
    UnsupportedFieldUnits {
        variable: String,
        units: String,
        expected: &'static str,
    },
    #[error("post-processed variable {0} has no finite physically plausible values")]
    NoPlausibleValues(String),
    #[error(
        "incomplete CONUS-II wrf2d dataset in {path}: missing required fields {missing:?}; refusing a partial import"
    )]
    IncompleteWrf2d { path: PathBuf, missing: Vec<String> },
    #[error("post-processed WRF volume allocation is unsupported: {0}")]
    PostprocessedVolume(String),
    #[error("{context}: store write failed: {source}")]
    StoreWrite {
        context: String,
        #[source]
        source: rw_store::RwStoreError,
    },
    #[error(transparent)]
    Netcdf(#[from] netcrust::Error),
    #[error(transparent)]
    Core(#[from] rustwx_core::RustwxError),
    #[error(transparent)]
    Store(#[from] rw_store::RwStoreError),
}

/// Time-axis reads on tiny wrfout cuts made by NCO (regenerated by
/// `tests/time_axis_fixtures/make_fixtures.sh`).  Every fixture's run starts
/// at 2025-03-15 12:00Z and its one record is valid at 2025-03-16 00:00Z, so
/// a reader that falls back to the start time is visibly wrong.
#[cfg(test)]
mod source_time_fixture_tests {
    use super::*;

    const START: &str = "2025-03-15T12:00:00Z";
    const VALID: &str = "2025-03-16T00:00:00Z";

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("time_axis_fixtures")
            .join(name)
    }

    /// A copy of `source` under `file_name` in a fresh folder, because the
    /// file name is what these tests are about.
    fn named_copy(source: &str, tag: &str, file_name: &str) -> (PathBuf, PathBuf) {
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let folder = std::env::temp_dir().join(format!(
            "rw-time-axis-{tag}-{}-{}-{}",
            std::process::id(),
            now_unix(),
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join(file_name);
        std::fs::copy(fixture(source), &path).unwrap();
        (folder, path)
    }

    fn source_times(path: &Path) -> Result<SourceTimeAxis, ImportError> {
        let nc = netcrust::open(path).expect("netcrust opens the fixture");
        netcdf_source_times(&nc, path)
    }

    fn refusal(result: Result<SourceTimeAxis, ImportError>) -> String {
        match result {
            Ok(axis) => panic!(
                "expected a refusal, but the frame was labelled valid at {}",
                axis.records[0].label
            ),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn a_wrfout_with_no_times_reads_its_valid_time_from_an_underscore_file_name() {
        let (folder, path) = named_copy(
            "no_times_cut.nc",
            "underscore",
            "wrfout_d01_2025-03-16_00_00_00",
        );
        let result = source_times(&path);
        let _ = std::fs::remove_dir_all(&folder);
        let axis = result.expect("the file name carries the valid time");
        assert_eq!(axis.records.len(), 1);
        assert_eq!(
            axis.records[0].valid_unix,
            parse_utc_timestamp(VALID).unwrap(),
            "the frame was labelled valid at {}, not at its file-name time",
            axis.records[0].label
        );
        assert_eq!(axis.reference_unix, parse_utc_timestamp(START));
    }

    #[cfg(unix)]
    #[test]
    fn a_wrfout_with_no_times_reads_its_valid_time_from_a_colon_file_name() {
        let (folder, path) =
            named_copy("no_times_cut.nc", "colon", "wrfout_d01_2025-03-16_00:00:00");
        let result = source_times(&path);
        let _ = std::fs::remove_dir_all(&folder);
        let axis = result.expect("the file name carries the valid time");
        assert_eq!(
            axis.records[0].valid_unix,
            parse_utc_timestamp(VALID).unwrap(),
            "the frame was labelled valid at {}, not at its file-name time",
            axis.records[0].label
        );
    }

    #[test]
    fn a_wrfout_with_no_times_and_no_file_name_time_is_refused_by_name() {
        let (folder, path) = named_copy("no_times_cut.nc", "unnamed", "hour12_cut.nc");
        let result = source_times(&path);
        let _ = std::fs::remove_dir_all(&folder);
        let message = refusal(result);
        assert!(
            message.contains("hour12_cut.nc") && message.contains("START_DATE"),
            "the refusal must name the file and say why the start time is not used: {message}"
        );
    }

    #[test]
    fn a_file_name_time_that_is_not_a_real_time_is_refused_by_name() {
        let (folder, path) = named_copy(
            "no_times_cut.nc",
            "badname",
            "wrfout_d01_2025-13-16_00_00_00",
        );
        let result = source_times(&path);
        let _ = std::fs::remove_dir_all(&folder);
        let message = refusal(result);
        assert!(
            message.contains("wrfout_d01_2025-13-16_00_00_00")
                && message.contains("not a valid UTC time"),
            "the refusal must name the file and its unreadable timestamp: {message}"
        );
    }

    fn assert_hour_twelve(axis: &SourceTimeAxis, what: &str) {
        assert_eq!(axis.records.len(), 1, "{what}");
        assert_eq!(
            axis.records[0].valid_unix,
            parse_utc_timestamp(VALID).unwrap(),
            "{what}: labelled valid at {}",
            axis.records[0].label
        );
        assert_eq!(axis.reference_unix, parse_utc_timestamp(START), "{what}");
    }

    #[test]
    fn an_ncks_netcdf4_cut_with_char_times_reads_its_valid_time() {
        let path = fixture("times_char_netcdf4_cut.nc");
        let axis = source_times(&path).expect("NetCDF-4 char Times decodes");
        assert_hour_twelve(&axis, "NetCDF-4 char Times");
    }

    #[test]
    fn an_ncks_netcdf4_cut_with_string_times_reads_its_valid_time() {
        let path = fixture("times_string_netcdf4_cut.nc");
        let axis = source_times(&path).expect("NetCDF-4 string Times decodes");
        assert_hour_twelve(&axis, "NetCDF-4 string Times");
    }

    #[test]
    fn the_raw_wrfout_route_reads_both_netcdf4_times_forms() {
        for name in ["times_char_netcdf4_cut.nc", "times_string_netcdf4_cut.nc"] {
            let path = fixture(name);
            let file = WrfFile::open(&path)
                .unwrap_or_else(|err| panic!("{name}: the raw-wrfout reader opens it: {err}"));
            let axis = wrf_source_times(&file, &path).unwrap_or_else(|err| panic!("{name}: {err}"));
            assert_hour_twelve(&axis, name);
        }
    }
}

/// A137 on a written file, through both readers: a wrfout-shaped d03 that
/// starts an hour after its simulation, stamped the way WRF and gpuwm's
/// writer stamp a delayed nest (START_DATE its own start, XTIME minutes
/// since it, SIMULATION_START_DATE the run's).
#[cfg(test)]
mod delayed_nest_time_tests {
    use super::*;
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

    const SIMULATION: &str = "2026-09-29_12:00:00";
    const NEST: &str = "2026-09-29_13:00:00";
    const FRAMES: [&str; 2] = ["2026-09-29_13:00:00", "2026-09-29_14:00:00"];

    fn write_nest(start_date: &str, tag: &str) -> (PathBuf, PathBuf) {
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let folder = std::env::temp_dir().join(format!(
            "rw-delayed-nest-{tag}-{}-{}-{}",
            std::process::id(),
            now_unix(),
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join(format!("wrfout_d03_{}", FRAMES[0].replace(':', "_")));
        let mut schema = Schema::new(NcFormat::Offset64);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
        let bottom_top = schema.def_dim("bottom_top", 1, false).unwrap();
        let south_north = schema.def_dim("south_north", 2, false).unwrap();
        let west_east = schema.def_dim("west_east", 2, false).unwrap();
        for (name, value) in [
            ("TITLE", " OUTPUT FROM WRF V4.6.1 MODEL"),
            ("START_DATE", start_date),
            ("SIMULATION_START_DATE", SIMULATION),
        ] {
            schema
                .put_global_attr(name, AttrValue::Text(value.into()))
                .unwrap();
        }
        for (name, value) in [("MAP_PROJ", 1i32), ("GRID_ID", 3), ("PARENT_ID", 2)] {
            schema
                .put_global_attr(name, AttrValue::Ints(vec![value]))
                .unwrap();
        }
        for (name, value) in [
            ("DX", 1333.3334f32),
            ("DY", 1333.3334),
            ("TRUELAT1", 30.0),
            ("TRUELAT2", 60.0),
            ("STAND_LON", -97.5),
            ("CEN_LAT", 35.5),
            ("CEN_LON", -97.5),
        ] {
            schema
                .put_global_attr(name, AttrValue::Floats(vec![value]))
                .unwrap();
        }
        let times = schema
            .def_var("Times", NcType::Char, &[time, strlen])
            .unwrap();
        let xtime = schema.def_var("XTIME", NcType::Float, &[time]).unwrap();
        let origin = format!("minutes since {}", start_date.replacen('_', " ", 1));
        schema
            .put_var_attr(xtime, "units", AttrValue::Text(origin.clone()))
            .unwrap();
        schema
            .put_var_attr(xtime, "description", AttrValue::Text(origin))
            .unwrap();
        let plane = |schema: &mut Schema, name: &str| {
            schema
                .def_var(name, NcType::Float, &[time, south_north, west_east])
                .unwrap()
        };
        let xlat = plane(&mut schema, "XLAT");
        let xlong = plane(&mut schema, "XLONG");
        let t2 = plane(&mut schema, "T2");
        // The four-dimensional T the raw-wrfout reader keys on.
        let theta = schema
            .def_var("T", NcType::Float, &[time, bottom_top, south_north, west_east])
            .unwrap();
        let mut writer = NcWriter::create(&path, schema).unwrap();
        let start = parse_utc_timestamp(start_date).unwrap();
        for (record, label) in FRAMES.iter().enumerate() {
            let record_number = record as u64;
            let minutes = (parse_utc_timestamp(label).unwrap() - start) as f32 / 60.0;
            writer
                .write_record(record_number, times, VarData::Char(label.as_bytes()))
                .unwrap();
            writer
                .write_record(record_number, xtime, VarData::F32(&[minutes]))
                .unwrap();
            writer
                .write_record(record_number, xlat, VarData::F32(&[35.4, 35.4, 35.41, 35.41]))
                .unwrap();
            writer
                .write_record(record_number, xlong, VarData::F32(&[-97.5, -97.49, -97.5, -97.49]))
                .unwrap();
            writer
                .write_record(record_number, t2, VarData::F32(&[290.0, 291.0, 292.0, 293.0]))
                .unwrap();
            writer
                .write_record(record_number, theta, VarData::F32(&[1.0, 2.0, 3.0, 4.0]))
                .unwrap();
        }
        writer.finish().unwrap();
        (folder, path)
    }

    fn both_readers(path: &Path) -> [(&'static str, Result<SourceTimeAxis, String>); 2] {
        let raw = WrfFile::open(path)
            .map_err(|err| err.to_string())
            .and_then(|file| wrf_source_times(&file, path));
        let generic = netcrust::open(path)
            .map_err(|err| err.to_string())
            .and_then(|nc| netcdf_source_times(&nc, path).map_err(|err| err.to_string()));
        [("wrf-core", raw), ("netcrust", generic)]
    }

    #[test]
    fn a_delayed_nest_reads_its_frames_on_the_runs_lead_clock_through_both_readers() {
        let (folder, path) = write_nest(NEST, "later");
        let answers = both_readers(&path);
        let _ = std::fs::remove_dir_all(&folder);
        for (reader, answer) in answers {
            let axis = answer.unwrap_or_else(|err| panic!("{reader} refused the delayed nest: {err}"));
            assert_eq!(axis.reference_unix, parse_utc_timestamp(SIMULATION), "{reader}");
            let valid = axis.records.iter().map(|r| r.valid_unix).collect::<Vec<_>>();
            let expected = FRAMES
                .iter()
                .map(|label| parse_utc_timestamp(label).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(valid, expected, "{reader}: valid times come from Times");

            // The planned timeline puts 13Z at F+01 and 14Z at F+02, the
            // leads the parent's frames of the same instants carry.
            let timeline = ForecastHourTimeline::plan_all(&[(path.clone(), axis)])
                .unwrap_or_else(|err| panic!("{reader}: {err}"));
            assert!(!timeline.is_exact_time_axis(), "{reader}");
            let slots = timeline.records_for_source(0).unwrap();
            assert_eq!(
                slots.iter().map(|r| r.storage_slot).collect::<Vec<_>>(),
                vec![1, 2],
                "{reader}"
            );
            assert!(
                timeline
                    .run_name("identity", "light")
                    .starts_with("local_20260929120000_"),
                "{reader}: the store run is keyed on the run's start"
            );
        }
    }

    #[test]
    fn a_domain_that_starts_before_its_simulation_is_refused_by_both_readers() {
        let (folder, path) = write_nest("2026-09-29_11:00:00", "earlier");
        let answers = both_readers(&path);
        let _ = std::fs::remove_dir_all(&folder);
        for (reader, answer) in answers {
            let error = match answer {
                Ok(axis) => panic!("{reader} accepted it with origin {:?}", axis.reference_unix),
                Err(error) => error,
            };
            assert!(
                error.contains("START_DATE=2026-09-29 11:00:00Z")
                    && error.contains("SIMULATION_START_DATE=2026-09-29 12:00:00Z")
                    && error.contains("guessed origin"),
                "{reader}: {error}"
            );
        }
    }
}

/// The model label a wrfout-shaped file names (`GPUWM_MODEL_LABEL`, which
/// a hex frame and a global tape carry so their maps stop saying WRF), read
/// from a written file through both readers, so a reader that lost the
/// attribute would fail here rather than as a map saying WRF again.
#[cfg(test)]
mod model_label_reader_tests {
    use super::*;
    use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

    const FRAME: &str = "2026-09-29_06:00:00";

    fn write_frame(label: Option<&str>, tag: &str) -> (PathBuf, PathBuf) {
        write_frame_from(label, false, tag)
    }

    /// `engine` stamps the attribute every WOOF history file carries.
    fn write_frame_from(label: Option<&str>, engine: bool, tag: &str) -> (PathBuf, PathBuf) {
        // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
        static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let folder = std::env::temp_dir().join(format!(
            "rw-model-label-{tag}-{}-{}-{}",
            std::process::id(),
            now_unix(),
            NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join(format!("wrfout_d01_{}", FRAME.replace(':', "_")));
        let mut schema = Schema::new(NcFormat::Offset64);
        let time = schema.def_dim("Time", 0, true).unwrap();
        let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
        let bottom_top = schema.def_dim("bottom_top", 1, false).unwrap();
        let south_north = schema.def_dim("south_north", 2, false).unwrap();
        let west_east = schema.def_dim("west_east", 2, false).unwrap();
        for (name, value) in [
            ("TITLE", " OUTPUT FROM WRF V4.6.1 MODEL"),
            ("START_DATE", "2026-09-29_00:00:00"),
            ("SIMULATION_START_DATE", "2026-09-29_00:00:00"),
        ] {
            schema
                .put_global_attr(name, AttrValue::Text(value.into()))
                .unwrap();
        }
        if engine {
            schema
                .put_global_attr(
                    rustwx_products::shared_context::ENGINE_VERSION_ATTRIBUTE,
                    AttrValue::Text("2.8.6".into()),
                )
                .unwrap();
        }
        if let Some(label) = label {
            schema
                .put_global_attr(
                    rustwx_products::shared_context::MODEL_LABEL_ATTRIBUTE,
                    AttrValue::Text(label.into()),
                )
                .unwrap();
        }
        for (name, value) in [("MAP_PROJ", 1i32), ("GRID_ID", 1), ("PARENT_ID", 0)] {
            schema
                .put_global_attr(name, AttrValue::Ints(vec![value]))
                .unwrap();
        }
        for (name, value) in [
            ("DX", 750.0f32),
            ("DY", 750.0),
            ("TRUELAT1", 30.0),
            ("TRUELAT2", 60.0),
            ("STAND_LON", -122.0),
            ("CEN_LAT", 37.7),
            ("CEN_LON", -122.3),
        ] {
            schema
                .put_global_attr(name, AttrValue::Floats(vec![value]))
                .unwrap();
        }
        let times = schema
            .def_var("Times", NcType::Char, &[time, strlen])
            .unwrap();
        let plane = |schema: &mut Schema, name: &str| {
            schema
                .def_var(name, NcType::Float, &[time, south_north, west_east])
                .unwrap()
        };
        let xlat = plane(&mut schema, "XLAT");
        let xlong = plane(&mut schema, "XLONG");
        // The four-dimensional T the raw-wrfout reader keys on.
        let theta = schema
            .def_var("T", NcType::Float, &[time, bottom_top, south_north, west_east])
            .unwrap();
        let mut writer = NcWriter::create(&path, schema).unwrap();
        writer
            .write_record(0, times, VarData::Char(FRAME.as_bytes()))
            .unwrap();
        writer
            .write_record(0, xlat, VarData::F32(&[37.7, 37.7, 37.71, 37.71]))
            .unwrap();
        writer
            .write_record(0, xlong, VarData::F32(&[-122.3, -122.29, -122.3, -122.29]))
            .unwrap();
        writer
            .write_record(0, theta, VarData::F32(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        writer.finish().unwrap();
        (folder, path)
    }

    fn both_readers(path: &Path) -> [(&'static str, Result<Option<String>, String>); 2] {
        let raw = WrfFile::open(path)
            .map_err(|err| err.to_string())
            .and_then(|file| model_label(&file));
        let generic = netcrust::open(path)
            .map_err(|err| err.to_string())
            .and_then(|nc| netcdf_model_label(&nc));
        [("wrf-core", raw), ("netcrust", generic)]
    }

    #[test]
    fn both_readers_take_the_model_label_a_file_names_and_none_from_a_stock_wrfout() {
        for (label, expected, tag) in [
            (Some("WOOF Hex"), Some("WOOF Hex"), "hex"),
            (Some("  ArWen   Global "), Some("ArWen Global"), "global"),
            (None, None, "stock"),
        ] {
            let (folder, path) = write_frame(label, tag);
            let answers = both_readers(&path);
            let _ = std::fs::remove_dir_all(&folder);
            for (reader, answer) in answers {
                let read = answer.unwrap_or_else(|err| panic!("{reader} ({tag}): {err}"));
                assert_eq!(read.as_deref(), expected, "{reader} ({tag})");
            }
        }
    }

    /// A history file the WOOF engine wrote names no model of its own; both
    /// readers call it WOOF, and a label it does name still wins.
    #[test]
    fn both_readers_call_an_engine_written_file_woof() {
        for (label, expected, tag) in [
            (None, "WOOF", "engine"),
            (Some("WOOF Hex"), "WOOF Hex", "engine-named"),
            (Some("   "), "WOOF", "engine-blank"),
        ] {
            let (folder, path) = write_frame_from(label, true, tag);
            let answers = both_readers(&path);
            let _ = std::fs::remove_dir_all(&folder);
            for (reader, answer) in answers {
                let read = answer.unwrap_or_else(|err| panic!("{reader} ({tag}): {err}"));
                assert_eq!(read.as_deref(), Some(expected), "{reader} ({tag})");
            }
        }
    }

    #[test]
    fn both_readers_refuse_a_label_the_metadata_row_cannot_take() {
        let (folder, path) = write_frame(Some("Hex | F001"), "split");
        let answers = both_readers(&path);
        let _ = std::fs::remove_dir_all(&folder);
        for (reader, answer) in answers {
            let error = answer.expect_err(reader);
            assert!(error.contains("separator"), "{reader}: {error}");
        }
    }
}
