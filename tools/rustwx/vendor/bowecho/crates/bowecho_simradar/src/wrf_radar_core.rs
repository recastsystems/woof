// Shared app and library simulation source.


use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};

use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use radar_core::{
    ElevationCut, GateRange, MomentGrid, MomentStorage, MomentType, RadarSite, RadarVolume, Radial,
    RayInstrumentMetadata, ScanLegMetadata, ScanMode, VcpInfo, VolumeMetadata, beam_ground_range_m,
    beam_height_above_radar_m,
};
use rayon::prelude::*;
use wrf_core::{ComputeOpts, WrfFile, getvar};

use crate::model_layer::{
    InverseLut, neighboring_cell_starts, solve_bilinear_coords, unwrap_lon_near,
};
use crate::wrf_radar_estimator::{
    CustomSinglePrf, DopplerSpectrumConfig, GateCoverageExplanation, GateExplanation, GateIdentity,
    GatePropagationExplanation, GateTimeExplanation, GateVelocityExplanation,
    HydrometeorGateContribution, IdealMoments, MatchedFilterRangeResponse, MeasuredMoments,
    MomentEstimatorConfig, NoiseKey, PresentationConfig, PresentedMoments, PrfSpecification,
    RadarInstrument, RadarMomentValues, ResolvedEstimatorSampling, ResolvedSinglePrf,
    SpeciesDopplerMode, WhyThisGate, WhyThisGateUnavailable, estimate_measured_moments,
    present_measured_moments, resolve_estimator_sampling, resolve_prf,
    synthesize_selected_gate_spectrum,
};
use bowecho_simradar::geo::aeqd_inverse_km;

use app_ui::vcp_catalog::{
    BUILD_24_SOURCE, Build24Vcp, DopplerPrfValue, MomentCoverage, PhysicalScanRow, PulseLength,
    VcpDefinition, Waveform, build_24_definition,
};
use app_ui::wrf_radar_validation::{
    CompactPolarPrecisionAudit, ExactScanTemplate, GateQuality, GateQualityFractions,
    QualityMoment, UnavailableObservedMoment, build_difference_volume_overlap,
    compact_quality_grid, encode_quality_fraction, relative_error,
};
use app_ui::wrf_scene_adapter::inventory_selected_wrf_paths;
use app_ui::wrf_scene_inventory::{WrfScene, WrfSceneGroup, WrfSceneTime, WrfSourceIdentity};
use app_ui::wrf_temporal::{
    AtmosphereTimeMode, HoldReason, MissingNeighborPolicy, RawGateState, RawStateLinearEndpoint,
    TemporalMemoryEstimate, TemporalSamplingOutcome, TemporalScenePlan, TwoSceneCache,
    interpolate_raw_state_linear, plan_for_scene,
};

/// Operational adaptations intentionally outside the checked base-pattern
/// catalog and this synthetic renderer.
pub const BUILD_24_NO_ADAPTATIONS_CAVEAT: &str = "Base pattern only: SAILS, MRLE, AVSET, Add-MPDA, and site-specific low-tilt adaptations are absent.";

/// Stable worker result used to distinguish a user cancellation from a
/// science/backend failure all the way through the model-window job pump.
pub const SYNTHETIC_RADAR_CANCELLED: &str = "Synthetic radar cancelled";

#[must_use]
pub fn is_synthetic_radar_cancelled(error: &str) -> bool {
    error == SYNTHETIC_RADAR_CANCELLED
        || error.ends_with("synthetic-radar T-matrix evaluation cancelled")
        || error.ends_with("synthetic-radar T-matrix raw batch cancelled")
        || error.ends_with("synthetic-radar T-matrix scene build cancelled")
}

fn check_synthetic_radar_cancel(cancel: &AtomicBool) -> Result<(), String> {
    if cancel.load(Ordering::Acquire) {
        Err(SYNTHETIC_RADAR_CANCELLED.to_owned())
    } else {
        Ok(())
    }
}

fn normalize_synthetic_radar_cancel<T>(
    result: Result<T, String>,
    cancel: &AtomicBool,
) -> Result<T, String> {
    if cancel.load(Ordering::Acquire) {
        Err(SYNTHETIC_RADAR_CANCELLED.to_owned())
    } else {
        result
    }
}

/// Default WSR-88D-like elevation ladder (deg). Covers the low tilts that
/// dominate a plan-view display plus enough high tilts for cross-sections.
pub const DEFAULT_ELEVATIONS_DEG: &[f64] = &[
    0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5,
];

/// Optional sub-0.5° tilt (deg) prepended below [`DEFAULT_ELEVATIONS_DEG`] when
/// the user opts in. The community wrf-python/GR2 exports start their ladder
/// here; the 0.1° beam samples roughly half the height of our standard 0.5°
/// lowest tilt at range, so a hook echo is better defined near the ground.
pub const LOW_TILT_DEG: f64 = 0.1;

/// Build the elevation ladder for a synthetic scan. With `include_low_tilt`
/// the [`LOW_TILT_DEG`] tilt is prepended below the standard
/// [`DEFAULT_ELEVATIONS_DEG`] ladder; otherwise the standard ladder is
/// returned unchanged (bit-identical to the historical default).
pub fn elevation_ladder(include_low_tilt: bool) -> Vec<f64> {
    if include_low_tilt {
        std::iter::once(LOW_TILT_DEG)
            .chain(DEFAULT_ELEVATIONS_DEG.iter().copied())
            .collect()
    } else {
        DEFAULT_ELEVATIONS_DEG.to_vec()
    }
}

/// Reflectivity source label stamped when the model's own Thompson `REFL_10CM`
/// field is used directly.
pub const REFL_10CM_SOURCE: &str = "REFL_10CM";
/// Reflectivity source label stamped when the model-native operator falls back
/// to the computed `dbz` because the file carries no `REFL_10CM`.
pub const CALCDBZ_SOURCE: &str = "dbz/CALCDBZ";
/// Reflectivity source label stamped when the classic Stoelinga operator is
/// chosen, forcing the computed `dbz` (CALCDBZ) even when the file carries
/// `REFL_10CM`.
pub const STOELINGA_SOURCE: &str = "dbz/CALCDBZ (Stoelinga)";
/// Reflectivity source stamped when raw WRF hydrometeors feed the internally
/// consistent bulk S-band polarimetric scattering state.
pub const BULK_S_BAND_SOURCE: &str = "scheme-aware bulk S-band ZH (Rayleigh v1)";
pub const PROPERTY_TMATRIX_S_BAND_SOURCE: &str =
    "P3/ISHMAEL property-aware S-band T-matrix ZH (research v1)";
pub const PROPERTY_TMATRIX_C_BAND_SOURCE: &str =
    "P3/ISHMAEL property-aware C-band T-matrix ZH (research)";
pub const PROPERTY_TMATRIX_X_BAND_SOURCE: &str =
    "P3/ISHMAEL property-aware X-band T-matrix ZH (research)";
pub const PROPERTY_HYBRID_S_BAND_SOURCE: &str =
    "P3 Hybrid S-band ZH (property T-matrix + audited bulk Rayleigh v1)";
pub const PROPERTY_HYBRID_C_BAND_SOURCE: &str =
    "P3 Hybrid C-band ZH (property T-matrix + audited bulk Rayleigh v1)";
pub const PROPERTY_HYBRID_X_BAND_SOURCE: &str =
    "P3 Hybrid X-band ZH (property T-matrix + audited bulk Rayleigh v1)";

fn property_tmatrix_ref_source(
    kernel: PolarimetricKernel,
    frequency_mhz: u32,
) -> Result<&'static str, String> {
    match (kernel, frequency_mhz) {
        (PolarimetricKernel::PropertyTMatrixHybridV1, 2_800) => Ok(PROPERTY_HYBRID_S_BAND_SOURCE),
        (PolarimetricKernel::PropertyTMatrixHybridV1, 5_600) => Ok(PROPERTY_HYBRID_C_BAND_SOURCE),
        (PolarimetricKernel::PropertyTMatrixHybridV1, 9_400) => Ok(PROPERTY_HYBRID_X_BAND_SOURCE),
        (PolarimetricKernel::PropertyTMatrixResearchV1, 2_800) => {
            Ok(PROPERTY_TMATRIX_S_BAND_SOURCE)
        }
        (PolarimetricKernel::PropertyTMatrixResearchV1, 5_600) => {
            Ok(PROPERTY_TMATRIX_C_BAND_SOURCE)
        }
        (PolarimetricKernel::PropertyTMatrixResearchV1, 9_400) => {
            Ok(PROPERTY_TMATRIX_X_BAND_SOURCE)
        }
        (PolarimetricKernel::BulkRayleighV1, _) => {
            Err("bulk Rayleigh does not have a property T-matrix reflectivity source".to_owned())
        }
        (_, other) => Err(format!(
            "unsupported exact property T-matrix frequency {other} MHz"
        )),
    }
}

/// Which reflectivity operator a synthetic scan uses. Both diagnostics are
/// legitimate: `REFL_10CM` is the model's own Thompson-native 10-cm
/// reflectivity (hotter/fatter in graupel/big-drop/melting regions of a
/// supercell), while the classic Stoelinga `dbz` (wrf-core `CALCDBZ`,
/// fixed Marshall-Palmer intercepts) is what the community wrf-python/GR2
/// pipeline renders. The choice is a per-import operator difference of roughly
/// +10..+20 dB in those regions — not a rendering artifact.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReflectivityOperator {
    /// Prefer the model's own `REFL_10CM` when the file carries it, else fall
    /// back to the computed `dbz` (`CALCDBZ`). The historical default.
    #[default]
    ModelNative,
    /// Always compute `dbz` (wrf-core `CALCDBZ`, Stoelinga 2005 fixed
    /// intercepts) — the community wrf-python/GR2 look — even when the file
    /// carries `REFL_10CM`.
    ClassicStoelinga,
}

/// Domain used while interpolating model reflectivity onto a radar pulse.
/// `LinearZ` is the physically meaningful received-power average; `LegacyDbz`
/// remains available so old presentation renders can be reproduced exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReflectivitySampling {
    LegacyDbz,
    #[default]
    LinearZ,
}

/// Polarimetric scattering implementation selected for synthetic radar.
///
/// The T-matrix variant is intentionally explicit about its evidence status:
/// its bundled tables are reproducible research assets, not independently
/// validated operational calibration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolarimetricKernel {
    #[default]
    BulkRayleighV1,
    /// Native property/T-matrix where the pinned tables and source state
    /// support the complete cell; explicit audited bulk Rayleigh v1 for
    /// table-domain/shape omissions or the typed WRF 2 µm source-state mass gap.
    PropertyTMatrixHybridV1,
    /// Full native property/T-matrix; experimental and always fail-closed.
    PropertyTMatrixResearchV1,
}

impl PolarimetricKernel {
    pub const fn is_property_tmatrix(self) -> bool {
        matches!(
            self,
            Self::PropertyTMatrixHybridV1 | Self::PropertyTMatrixResearchV1
        )
    }

    pub const fn is_hybrid(self) -> bool {
        matches!(self, Self::PropertyTMatrixHybridV1)
    }

    pub const fn scattering_policy(self) -> app_ui::wrf_tmatrix_scene::WrfTMatrixScatteringPolicy {
        match self {
            Self::BulkRayleighV1 | Self::PropertyTMatrixResearchV1 => {
                app_ui::wrf_tmatrix_scene::WrfTMatrixScatteringPolicy::StrictFailClosed
            }
            Self::PropertyTMatrixHybridV1 => {
                app_ui::wrf_tmatrix_scene::WrfTMatrixScatteringPolicy::HybridBulkRayleighV1
            }
        }
    }
}

/// Beam-path geometry used to map slant range/elevation into model latitude,
/// longitude, and MSL height. The default preserves the historical 4/3-Earth
/// operator exactly; WRF refractivity is explicit research mode and fails
/// closed when its thermodynamic profile cannot be qualified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropagationGeometry {
    #[default]
    StandardFourThirdsEarth,
    WrfRefractivityResearch,
}

/// Deterministic antenna/pulse quadrature. The balanced rule is a symmetric
/// nine-point cubature; reference is the full 3x3x3 tensor rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeamIntegration {
    #[default]
    Center,
    Balanced,
    Reference,
}

impl BeamIntegration {
    /// Number of deterministic antenna/range quadrature points evaluated for
    /// every rendered radar gate. This is exposed for the WRF UI workload
    /// estimate; the execution path uses the same rules in
    /// [`quadrature_points`].
    pub const fn pulse_volume_sample_count(self) -> usize {
        match self {
            Self::Center => 1,
            Self::Balanced => 9,
            Self::Reference => 27,
        }
    }
}

/// User preference for the native P3/ISHMAEL T-matrix execution backend.
///
/// This is a preference rather than output provenance: `Auto` may choose a
/// qualified NVIDIA device, and every CUDA selection must retain a normal CPU
/// fallback when the runtime or a requested operation is unavailable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticRadarComputePreference {
    /// Prefer CUDA when a supported runtime/device is available, otherwise CPU.
    #[default]
    Auto,
    /// Always run the portable CPU reference implementation.
    Cpu,
    /// Prefer the optional NVIDIA CUDA path, with automatic CPU fallback.
    NvidiaCuda,
}

impl SyntheticRadarComputePreference {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::Cpu => "CPU",
            Self::NvidiaCuda => "NVIDIA CUDA",
        }
    }
}

/// User intent for a synthetic volume. Fine-grained controls remain editable,
/// but named modes make it explicit whether a volume is model truth, a virtual
/// instrument measurement, or a display-oriented presentation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationMode {
    Truth,
    Instrument,
    #[default]
    Presentation,
}

/// Whether all rays represent one model instant or carry acquisition times for
/// a rotating volume scan. Atmosphere sampling is configured independently by
/// [`SyntheticRadarConfig::atmosphere_time_mode`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanTiming {
    #[default]
    InstantaneousTruth,
    TimedVolume,
}

/// Scan strategy used by the WRF synthetic-radar forward operator.
///
/// `CustomLegacy` is deliberately the serde/default variant: settings written
/// before the Build 24 catalog existed, and programmatic callers that use
/// [`SyntheticRadarConfig::default`], retain the historical fourteen-cut
/// ladder and every custom timing/PRF knob exactly as before.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticScanStrategy {
    #[default]
    CustomLegacy,
    Build24Vcp12,
    Build24Vcp34,
    Build24Vcp35,
    Build24Vcp112,
    Build24Vcp212,
    Build24Vcp215,
}

impl SyntheticScanStrategy {
    pub const BUILD_24: [Self; 6] = [
        Self::Build24Vcp12,
        Self::Build24Vcp34,
        Self::Build24Vcp35,
        Self::Build24Vcp112,
        Self::Build24Vcp212,
        Self::Build24Vcp215,
    ];

    pub const fn vcp(self) -> Option<Build24Vcp> {
        match self {
            Self::CustomLegacy => None,
            Self::Build24Vcp12 => Some(Build24Vcp::Vcp12),
            Self::Build24Vcp34 => Some(Build24Vcp::Vcp34),
            Self::Build24Vcp35 => Some(Build24Vcp::Vcp35),
            Self::Build24Vcp112 => Some(Build24Vcp::Vcp112),
            Self::Build24Vcp212 => Some(Build24Vcp::Vcp212),
            Self::Build24Vcp215 => Some(Build24Vcp::Vcp215),
        }
    }

    pub fn definition(self) -> Option<&'static VcpDefinition> {
        build_24_definition(self.vcp()?.number())
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::CustomLegacy => "Custom legacy ladder",
            Self::Build24Vcp12 => "Build 24 VCP 12",
            Self::Build24Vcp34 => "Build 24 VCP 34",
            Self::Build24Vcp35 => "Build 24 VCP 35",
            Self::Build24Vcp112 => "Build 24 VCP 112",
            Self::Build24Vcp212 => "Build 24 VCP 212",
            Self::Build24Vcp215 => "Build 24 VCP 215",
        }
    }

    pub const fn is_named_vcp(self) -> bool {
        !matches!(self, Self::CustomLegacy)
    }
}

/// One physical antenna rotation to sample. Named VCPs produce one leg for
/// every catalog row, including equal-elevation split cuts. The legacy custom
/// mode produces one all-moment leg per configured elevation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyntheticScanLeg {
    pub elevation_deg: f64,
    pub azimuth_rate_deg_per_second: f32,
    pub source_period_seconds: f32,
    pub transition_after_seconds: f32,
    pub moments: MomentCoverage,
    pub waveform: Option<Waveform>,
    pub source_row_index: Option<usize>,
    pub source_row: Option<&'static PhysicalScanRow>,
}

impl ReflectivityOperator {
    /// Whether to prefer the model's own `REFL_10CM` when the file carries it.
    /// Only the model-native operator does; Stoelinga always recomputes.
    pub fn prefers_refl_10cm(self) -> bool {
        matches!(self, Self::ModelNative)
    }

    /// The source label stamped when this operator uses the computed `dbz`
    /// (`CALCDBZ`) path — distinguishing a deliberate Stoelinga choice from a
    /// model-native fallback.
    pub fn computed_dbz_source(self) -> &'static str {
        match self {
            Self::ModelNative => CALCDBZ_SOURCE,
            Self::ClassicStoelinga => STOELINGA_SOURCE,
        }
    }
}

/// The reflectivity source label a synthetic scan will stamp, given the
/// operator choice and whether the file carries `REFL_10CM`. Assumes the
/// `REFL_10CM` read succeeds when attempted; a failed/short read falls through
/// to the computed `dbz`, mirrored exactly by [`read_reflectivity`]. This is
/// the pure decision the reflectivity read makes, factored out so both operator
/// branches are testable without a WRF file on disk.
pub fn planned_ref_source(
    operator: ReflectivityOperator,
    file_has_refl_10cm: bool,
) -> &'static str {
    if operator.prefers_refl_10cm() && file_has_refl_10cm {
        REFL_10CM_SOURCE
    } else {
        operator.computed_dbz_source()
    }
}

/// Nyquist velocity (m/s) stamped on every radial when velocity folding is OFF
/// — deliberately large so the native, forward-modelled Vr is treated as
/// already-unfolded TRUE velocity by the dealias/readout code (it never folds).
/// This is the historical stamp the community WRF→GR2 script also implied by
/// leaving the data unfolded.
pub const UNFOLDED_NYQUIST_MPS: f32 = 320.0;

/// Default FOLDING Nyquist (m/s) when the realistic-Nyquist option is enabled —
/// a typical WSR-88D low-PRF Doppler Nyquist. Real VELocity Nyquists cluster
/// around here (roughly 8–33 m/s depending on VCP/PRF).
pub const DEFAULT_FOLD_NYQUIST_MPS: f32 = 25.0;

/// Exact transmit frequency of the coherent BowEcho S research-pack recipe.
pub const PROPERTY_TMATRIX_RESEARCH_FREQUENCY_MHZ: u32 = 2_800;
pub const PROPERTY_TMATRIX_EXACT_FREQUENCIES_MHZ: [u32; 3] = [2_800, 5_600, 9_400];

/// Explicit rain/melting sensitivity for property T-matrix evaluation.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PropertyTMatrixRainSensitivity {
    /// Evaluate standalone rain and qualified wet frozen/rain coexistence.
    #[default]
    FullProperty,
    /// Deliberately omit rain and wet coexistence; frozen categories stay dry.
    FrozenOnly,
}

impl PropertyTMatrixRainSensitivity {
    fn runtime(self) -> app_ui::wrf_tmatrix_scene::WrfTMatrixRainMode {
        match self {
            Self::FullProperty => app_ui::wrf_tmatrix_scene::WrfTMatrixRainMode::FullProperty,
            Self::FrozenOnly => app_ui::wrf_tmatrix_scene::WrfTMatrixRainMode::FrozenOnly,
        }
    }
}
/// View-angle applicability includes every Build-24/optional-cut center plus
/// the default one-degree beam's ±sigma pulse-volume quadrature offsets.
pub const PROPERTY_TMATRIX_MIN_VIEW_ELEVATION_DEG: f64 = -0.5;
pub const PROPERTY_TMATRIX_MAX_VIEW_ELEVATION_DEG: f64 = 20.0;

#[inline]
fn gaussian_beam_sigma_deg(fwhm_deg: f64) -> f64 {
    fwhm_deg / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt())
}

/// Configuration for one synthetic scan.
#[derive(Clone, Debug)]
pub struct SyntheticRadarConfig {
    /// Named operating intent. This is also stamped in export provenance.
    pub simulation_mode: SimulationMode,
    /// Preferred execution backend for the native P3/ISHMAEL property
    /// T-matrix workload. It is intentionally excluded from the scientific
    /// data fingerprint: parity-capable backends compute the same product.
    pub compute_preference: SyntheticRadarComputePreference,
    /// Job-scoped accelerator handle. This is runtime-only: it is never
    /// serialized, fingerprinted, or retained by installed-frame refresh
    /// descriptors. The outer WRF worker owns the last strong reference and
    /// therefore shuts the CUDA worker down when that job finishes.
    pub(crate) runtime_cuda_service:
        Option<Arc<app_ui::wrf_tmatrix_cuda::WrfTMatrixCudaBatchService>>,
    /// Physical scan definition. The historical custom ladder remains the
    /// default; a Build 24 selection owns its rows, rates, periods, waveform,
    /// moment coverage and PRF-code provenance.
    pub scan_strategy: SyntheticScanStrategy,
    /// Exact observed acquisition geometry for Level-II replay. This preserves
    /// source cuts/rays/gates/times/moment availability verbatim and takes
    /// precedence over `scan_strategy`; it is not a VCP reconstruction.
    pub exact_replay_template: Option<Arc<ExactScanTemplate>>,
    /// Site id stamped on the volume (drives the loop-engine cross-site guard —
    /// every hour of one run must share it).
    pub site_id: String,
    pub site_name: Option<String>,
    /// Antenna position. `None` places it at the WRF domain centre.
    pub site_lat_deg: Option<f64>,
    pub site_lon_deg: Option<f64>,
    /// Antenna altitude MSL (m). `None` uses the model terrain at the site plus
    /// [`DEFAULT_TOWER_M`].
    pub antenna_msl_m: Option<f64>,
    pub elevations_deg: Vec<f64>,
    /// Azimuth samples per sweep, clockwise from north (e.g. 360 → 1.0°, 720 →
    /// 0.5°).
    pub azimuth_count: usize,
    pub gate_spacing_m: f64,
    pub max_range_m: f64,
    /// Set the effective gate spacing equal to the WRF grid resolution (the
    /// file's `DX` global attribute, metres) instead of [`Self::gate_spacing_m`],
    /// so a coarse grid is not oversampled — 250 m gates on a 3 km grid imply
    /// ~12 redundant gates per model cell, more resolution than the model has.
    /// Resolved per file at build time by [`effective_gate_spacing`] and clamped
    /// to `[GRID_GATE_MIN_M, GRID_GATE_MAX_M]` so a garbage `DX` cannot produce an
    /// absurd gate count; a file with a missing/invalid `DX` falls back to
    /// [`Self::gate_spacing_m`]. OFF by default — an OFF build uses the configured
    /// gate spacing exactly as before (bit-identical to a build without this).
    pub match_gate_to_grid: bool,
    /// Reflectivity floor (dBZ): gates below this — and their velocity — are
    /// left NaN so clear air renders transparent, like a real scope.
    pub ref_floor_dbz: f32,
    /// The FOLDING Nyquist velocity (m/s). Used ONLY when [`Self::fold_velocity`]
    /// is set: each gate's Vr is aliased into `[-nyquist_mps, +nyquist_mps)` and
    /// this value is stamped on every radial, so the dealiaser/readout see the
    /// real velocity ambiguity. When folding is OFF the historical
    /// [`UNFOLDED_NYQUIST_MPS`] (320) is stamped instead (via
    /// [`Self::stamped_nyquist_mps`]), so the native forward-modelled Vr reads as
    /// already-unfolded TRUE velocity that never folds. Default
    /// [`DEFAULT_FOLD_NYQUIST_MPS`].
    pub nyquist_mps: f32,
    /// Fold the forward-modelled radial velocity like a real pulse-pair radar:
    /// alias every velocity gate into the [`Self::nyquist_mps`] co-interval so VEL
    /// FOLDS (and the velocity dealiaser has genuine work — a practice ground on
    /// known ground truth). OFF by default — the native Vr is the exact TRUE wind
    /// projection and [`UNFOLDED_NYQUIST_MPS`] is stamped, so an OFF build is
    /// bit-identical to a build without the feature.
    pub fold_velocity: bool,
    /// Which reflectivity operator populates the sampled `dbz`: the model's own
    /// Thompson-native `REFL_10CM` ([`ReflectivityOperator::ModelNative`]) or the
    /// classic Stoelinga `CALCDBZ` community diagnostic
    /// ([`ReflectivityOperator::ClassicStoelinga`]).
    pub reflectivity_operator: ReflectivityOperator,
    /// Interpolate/average received power in linear Z (scientific default) or
    /// reproduce the historical direct-dBZ interpolation.
    pub reflectivity_sampling: ReflectivitySampling,
    /// Antenna/pulse-volume integration tier.
    pub beam_integration: BeamIntegration,
    /// Horizontal/vertical one-way 3-dB beam width (degrees).
    pub beam_width_deg: f32,
    /// Transmitted pulse duration (microseconds).
    pub pulse_width_us: f32,
    /// Radar transmit frequency. 2.8 GHz is a representative S-band system.
    pub radar_frequency_mhz: u32,
    /// Finite-particle C/X tables. None preserves the established S-band path.
    pub scattering_band: Option<String>,
    /// Use hydrometeor/scatterer fall speed in the Doppler projection.
    pub terminal_fall_speed: bool,
    /// Apply cumulative terrain-horizon partial beam blockage.
    pub terrain_blockage: bool,
    /// Emit Doppler spectrum width from pulse-volume velocity variance, model
    /// TKE when available, terminal-speed diversity, and the instrument floor.
    pub spectrum_width: bool,
    pub spectrum_width_floor_mps: f32,
    /// Build scheme-aware S-band ZH/ZV/covariance/KDP/attenuation from raw WRF
    /// hydrometeors. Unsupported schemes fall back to scalar REF/VEL with an
    /// explicit note instead of fabricating polarimetric fields.
    pub dual_pol: bool,
    /// Scattering kernel behind dual-pol fields. Research T-matrix mode is
    /// fail-closed when the file/category/table applicability contract does
    /// not match exactly; it never silently falls back to Rayleigh.
    pub polarimetric_kernel: PolarimetricKernel,
    /// Exact property-table source. The on-demand BowEcho S research pack is
    /// the coherent default; external selection never falls back or chooses a
    /// nearest band.
    pub property_tmatrix_table_source: app_ui::wrf_tmatrix_assets::PropertyTMatrixTableSourceKind,
    /// Whether the property operator includes rain/melting sensitivity.
    pub property_tmatrix_rain_sensitivity: PropertyTMatrixRainSensitivity,
    /// Integrate KDP/Ah/Av along each radial into PhiDP and attenuation.
    pub propagation: bool,
    /// Geometric beam propagation. This is independent of polarimetric path
    /// attenuation above.
    pub propagation_geometry: PropagationGeometry,
    /// Optional synthetic-system calibration offsets.
    pub system_phidp_deg: f32,
    pub zdr_bias_db: f32,
    /// Acquisition timing and nominal instrument cadence.
    pub scan_timing: ScanTiming,
    /// Sample every ray from the anchor WRF scene or interpolate its atmosphere
    /// from the adjacent compatible model scene at the ray acquisition time.
    pub atmosphere_time_mode: AtmosphereTimeMode,
    /// Explicit behavior when a later compatible WRF scene is unavailable or
    /// cannot cover the complete scan without extrapolation.
    pub missing_neighbor_policy: MissingNeighborPolicy,
    /// Peak-memory ceiling for a rolling two-scene temporal build. The worker
    /// preflights before reading a second scene and never silently exceeds it.
    pub temporal_memory_budget_mib: usize,
    pub rotation_rate_deg_s: f32,
    pub transition_delay_s: f32,
    /// Pulse repetition frequency used for CfRadial metadata. Velocity folding
    /// remains controlled independently by `nyquist_mps`.
    pub prf_hz: f32,
    /// Opt into the physically coupled custom single-PRF instrument. When on,
    /// exact frequency + PRF jointly determine wavelength, Nyquist and
    /// unambiguous range; the moment estimator owns sensitivity, uncertainty,
    /// bias and folding. Named VCP PRF codes fail closed because they are not
    /// frequencies. Off preserves the v0.33.1 path bit-for-bit.
    pub coupled_single_prf_estimator: bool,
    /// Nominal dwell used by the coupled moment estimator.
    pub estimator_dwell_ms: f32,
    /// Optional authoritative transmitted-pulse count. `None` derives
    /// `floor(dwell * PRF)`.
    pub estimator_pulse_count: Option<u32>,
    /// Fraction of transmitted pulses treated as statistically independent.
    pub estimator_independent_sample_fraction: f32,
    /// Minimum received SNR admitted by the coupled estimator.
    pub estimator_minimum_snr_db: f32,
    /// Emit opt-in f32 Ideal (`I*`) and Measured (`M*`) diagnostic fields for
    /// REF/VEL/SW/ZDR/rhoHV/KDP. Canonical fields remain Presented.
    pub emit_stage_diagnostics: bool,
    /// Range-dependent sensitivity/noise model. The threshold follows the
    /// radar equation from this 1-km dBZ reference when enabled.
    pub instrument_noise: bool,
    pub sensitivity_dbz_at_1km: f32,
    /// "Gate texture" on REFLECTIVITY: deterministic, range-correlated speckle
    /// added to the sampled dBZ so the synthetic gates read like real Level-II
    /// texture instead of a perfectly smooth trilinear field. ON is the product
    /// default (owner: the smooth field "looks garbage without" it). When OFF,
    /// the reflectivity perturbation code never touches a value, so an OFF build
    /// is bit-identical to a build without the feature.
    pub ref_gate_texture: bool,
    /// "Gate texture" on VELOCITY: a gentle ±0.5 m/s wobble on the forward-
    /// modelled radial velocity. OFF by default and kept opt-in — the clean Vr
    /// feeds the velocity dealias / GBVTD consumers downstream, and a noisy Vr
    /// would pollute them. Separate from [`Self::ref_gate_texture`] on purpose:
    /// reflectivity wants the speckle, velocity does not.
    pub vel_gate_texture: bool,
    /// Ground-clutter amount, 0.0..=1.0. Our forward operator is pure physics
    /// and produces ZERO clutter; this dials in a fabricated near-radar
    /// ground-return look modelled on the community WRF→GR2 export script
    /// (`add_ground_clutter`). `0.0` (the default) skips the clutter path
    /// entirely, so the output is bit-identical to a build without the feature;
    /// `1.0` ≈ the community-script intensity. Intermediate values scale the
    /// clutter *probability* and *brightness* together. The pattern is
    /// deterministic per forecast frame (seeded from site id + valid time +
    /// tilt) so a loop never shimmers between rebuilds of the same hour. See the
    /// ground-clutter section (`ground_clutter_dbz`) for the model.
    pub clutter_intensity: f32,
    /// Emit compact per-gate pulse-volume support diagnostics (MCOV, TUNB,
    /// MSIG). These never alter the physical samples and use three bytes per
    /// gate in the finished volume.
    pub emit_quality_fields: bool,
    /// Mask physical moments where less than this fraction of the configured
    /// quadrature support is covered by the model domain. Quality grids remain
    /// present so a masked gate can still explain why it was rejected.
    pub minimum_model_coverage_fraction: f32,
}

/// Antenna height above model terrain when no explicit MSL altitude is given.
pub const DEFAULT_TOWER_M: f64 = 10.0;

impl Default for SyntheticRadarConfig {
    fn default() -> Self {
        Self {
            simulation_mode: SimulationMode::Presentation,
            compute_preference: SyntheticRadarComputePreference::Auto,
            runtime_cuda_service: None,
            scan_strategy: SyntheticScanStrategy::CustomLegacy,
            exact_replay_template: None,
            site_id: "WRF".to_string(),
            site_name: Some("Simulated WRF radar".to_string()),
            site_lat_deg: None,
            site_lon_deg: None,
            antenna_msl_m: None,
            elevations_deg: DEFAULT_ELEVATIONS_DEG.to_vec(),
            azimuth_count: 720,
            gate_spacing_m: 250.0,
            max_range_m: 230_000.0,
            // Off by default: the effective gate spacing is the configured
            // `gate_spacing_m`, so the default build is bit-identical to a build
            // without the grid-matching feature.
            match_gate_to_grid: false,
            ref_floor_dbz: 0.0,
            // The folding Nyquist (used only when `fold_velocity` is set). With
            // folding OFF (the default) `UNFOLDED_NYQUIST_MPS` (320) is stamped
            // and no gate folds, so the default build is bit-identical to before.
            nyquist_mps: DEFAULT_FOLD_NYQUIST_MPS,
            fold_velocity: false,
            // Default operator. Flip this one line to ClassicStoelinga to
            // re-default the community look after the owner's comparison.
            reflectivity_operator: ReflectivityOperator::ModelNative,
            reflectivity_sampling: ReflectivitySampling::LinearZ,
            beam_integration: BeamIntegration::Center,
            beam_width_deg: 0.95,
            pulse_width_us: 1.57,
            radar_frequency_mhz: 2_800,
            scattering_band: None,
            terminal_fall_speed: false,
            terrain_blockage: false,
            spectrum_width: false,
            spectrum_width_floor_mps: 0.5,
            dual_pol: false,
            polarimetric_kernel: PolarimetricKernel::BulkRayleighV1,
            property_tmatrix_table_source:
                app_ui::wrf_tmatrix_assets::PropertyTMatrixTableSourceKind::LegacyEmbeddedSResearchV1,
            property_tmatrix_rain_sensitivity: PropertyTMatrixRainSensitivity::FullProperty,
            propagation: false,
            propagation_geometry: PropagationGeometry::StandardFourThirdsEarth,
            system_phidp_deg: 0.0,
            zdr_bias_db: 0.0,
            scan_timing: ScanTiming::InstantaneousTruth,
            atmosphere_time_mode: AtmosphereTimeMode::FrozenAtVolumeStart,
            missing_neighbor_policy: MissingNeighborPolicy::HoldAnchor,
            temporal_memory_budget_mib: 65_536,
            rotation_rate_deg_s: 18.0,
            transition_delay_s: 3.5,
            prf_hz: 1_000.0,
            coupled_single_prf_estimator: false,
            estimator_dwell_ms: 50.0,
            estimator_pulse_count: None,
            estimator_independent_sample_fraction: 0.5,
            estimator_minimum_snr_db: 0.0,
            emit_stage_diagnostics: false,
            instrument_noise: false,
            sensitivity_dbz_at_1km: -40.0,
            // Reflectivity texture ON by default (owner verdict: a smooth
            // simulated field "looks garbage without" it); velocity texture
            // OFF so the clean Vr keeps feeding dealias/GBVTD.
            ref_gate_texture: true,
            vel_gate_texture: false,
            // No clutter by default: the operator stays pure physics and the
            // output is bit-identical to a build without the feature.
            clutter_intensity: 0.0,
            emit_quality_fields: true,
            minimum_model_coverage_fraction: 0.0,
        }
    }
}

impl SyntheticRadarConfig {
    /// Resolve configuration into physical antenna rotations before any model
    /// sampling begins. This is intentionally separate from the renderer loop:
    /// duplicate split cuts must remain duplicate cuts, not be collapsed into
    /// a unique elevation ladder.
    pub fn physical_scan_legs(&self) -> Vec<SyntheticScanLeg> {
        if let Some(definition) = self.scan_strategy.definition() {
            return definition
                .rows
                .iter()
                .enumerate()
                .map(|(source_row_index, row)| SyntheticScanLeg {
                    elevation_deg: f64::from(row.elevation_deg),
                    azimuth_rate_deg_per_second: row.azimuth_rate_deg_per_second,
                    source_period_seconds: row.source_period_seconds,
                    transition_after_seconds: 0.0,
                    moments: row.moments,
                    waveform: Some(row.waveform),
                    source_row_index: Some(source_row_index),
                    source_row: Some(row),
                })
                .collect();
        }

        let rate = self.rotation_rate_deg_s.max(0.1);
        self.elevations_deg
            .iter()
            .copied()
            .map(|elevation_deg| SyntheticScanLeg {
                elevation_deg,
                azimuth_rate_deg_per_second: rate,
                source_period_seconds: 360.0 / rate,
                transition_after_seconds: self.transition_delay_s.max(0.0),
                moments: MomentCoverage::ALL,
                waveform: None,
                source_row_index: None,
                source_row: None,
            })
            .collect()
    }

    /// Reject a research-table request before decoding a multi-gigabyte WRF
    /// scene. These are exact applicability checks, not automatic clamps or
    /// fallback rules.
    pub fn validate_science_contract(&self) -> Result<(), String> {
        if self.exact_replay_template.is_some() && self.coupled_single_prf_estimator {
            return Err(
                "exact observed replay cannot use custom coupled-PRF timing; the source radial Nyquist metadata owns replay ambiguity"
                    .to_string(),
            );
        }
        if self.emit_stage_diagnostics && !self.coupled_single_prf_estimator {
            return Err(
                "Ideal/Measured stage diagnostics require the coupled single-PRF estimator"
                    .to_string(),
            );
        }
        if self.coupled_single_prf_estimator {
            validate_coupled_estimator_inputs(self)?;
        }
        let property_tmatrix = self.polarimetric_kernel.is_property_tmatrix();
        if matches!(
            self.atmosphere_time_mode,
            AtmosphereTimeMode::RawStateLinear
        ) && !property_tmatrix
        {
            return Err(
                "RawStateLinear atmosphere timing is available only with the P3/ISHMAEL property T-matrix kernel"
                    .to_string(),
            );
        }
        if !property_tmatrix {
            return Ok(());
        }
        if matches!(
            self.atmosphere_time_mode,
            AtmosphereTimeMode::RawStateLinear
        ) && self.polarimetric_kernel.is_hybrid()
        {
            return Err(
                "P3 Hybrid currently requires Frozen or additive adjacent-scene timing; RawStateLinear is available only with experimental Full P3/ISHMAEL T-matrix strict mode"
                    .to_owned(),
            );
        }
        if !self.dual_pol {
            return Err("Property T-matrix research mode requires dual polarization".to_string());
        }
        if !PROPERTY_TMATRIX_EXACT_FREQUENCIES_MHZ.contains(&self.radar_frequency_mhz) {
            return Err(format!(
                "Property T-matrix frequency must be one exact supported choice (2800, 5600, or 9400 MHz), got {} MHz; no nearest-band substitution is allowed",
                self.radar_frequency_mhz,
            ));
        }
        let exact_frequency_hz = f64::from(self.radar_frequency_mhz) * 1.0e6;
        if matches!(
            self.atmosphere_time_mode,
            AtmosphereTimeMode::RawStateLinear
        ) && (!matches!(
            self.property_tmatrix_table_source,
            app_ui::wrf_tmatrix_assets::PropertyTMatrixTableSourceKind::LegacyEmbeddedSResearchV1
        ) || self.radar_frequency_mhz != PROPERTY_TMATRIX_RESEARCH_FREQUENCY_MHZ
            || !matches!(
                self.property_tmatrix_rain_sensitivity,
                PropertyTMatrixRainSensitivity::FullProperty
            ))
        {
            return Err(
                "RawStateLinear currently requires the BowEcho S research pack at exactly 2800 MHz with Full property rain/melting; external packs and Frozen-only use Frozen or additive adjacent-scene timing"
                    .to_owned(),
            );
        }
        // Keep validation pure: first-use pack acquisition belongs to the
        // worker where progress reaches the UI before the heavy WRF read.
        app_ui::wrf_tmatrix_assets::validate_property_tmatrix_table_source_frequency(
            self.property_tmatrix_table_source,
            exact_frequency_hz,
        )?;
        if !matches!(self.reflectivity_sampling, ReflectivitySampling::LinearZ) {
            return Err("Property T-matrix research mode requires linear-Z sampling".to_string());
        }
        let beam_width_deg = f64::from(self.beam_width_deg);
        if !beam_width_deg.is_finite() || beam_width_deg <= 0.0 {
            return Err(format!(
                "Property T-matrix beam width must be finite and positive, got {beam_width_deg}°"
            ));
        }
        let legs = self.physical_scan_legs();
        if legs.is_empty() {
            return Err("Property T-matrix scan has no physical cuts".to_string());
        }
        // Match the exact lower bound used by sample_gate so validation and
        // runtime can never disagree at a table-elevation boundary.
        let effective_beam_width_deg = beam_width_deg.max(0.05);
        let beam_sigma_deg = gaussian_beam_sigma_deg(effective_beam_width_deg);
        for leg in legs {
            for point in quadrature_points(self.beam_integration) {
                let view_elevation_deg = leg.elevation_deg + point.el_sigma * beam_sigma_deg;
                if !(PROPERTY_TMATRIX_MIN_VIEW_ELEVATION_DEG
                    ..=PROPERTY_TMATRIX_MAX_VIEW_ELEVATION_DEG)
                    .contains(&view_elevation_deg)
                {
                    return Err(format!(
                        "Property T-matrix view elevation {view_elevation_deg:.3}° for the {:.3}° cut is outside the exact [{:.1}°, {:.1}°] table range; reduce beam width/change cuts or use Bulk Rayleigh",
                        leg.elevation_deg,
                        PROPERTY_TMATRIX_MIN_VIEW_ELEVATION_DEG,
                        PROPERTY_TMATRIX_MAX_VIEW_ELEVATION_DEG,
                    ));
                }
            }
        }
        Ok(())
    }

    /// Apply a coherent named preset while leaving site/range/geometry choices
    /// alone. The UI calls this only when the user selects a different mode;
    /// subsequent expert edits are intentionally preserved.
    pub fn apply_mode_preset(&mut self, mode: SimulationMode) {
        self.simulation_mode = mode;
        match mode {
            SimulationMode::Truth => {
                self.reflectivity_sampling = ReflectivitySampling::LinearZ;
                self.beam_integration = BeamIntegration::Center;
                self.terminal_fall_speed = false;
                self.terrain_blockage = false;
                self.spectrum_width = false;
                self.dual_pol = false;
                self.polarimetric_kernel = PolarimetricKernel::BulkRayleighV1;
                self.propagation = false;
                self.propagation_geometry = PropagationGeometry::StandardFourThirdsEarth;
                self.scan_timing = ScanTiming::InstantaneousTruth;
                self.atmosphere_time_mode = AtmosphereTimeMode::FrozenAtVolumeStart;
                self.instrument_noise = false;
                self.ref_gate_texture = false;
                self.vel_gate_texture = false;
                self.clutter_intensity = 0.0;
                self.fold_velocity = false;
            }
            SimulationMode::Instrument => {
                self.reflectivity_sampling = ReflectivitySampling::LinearZ;
                self.beam_integration = BeamIntegration::Balanced;
                self.terminal_fall_speed = true;
                self.terrain_blockage = true;
                self.spectrum_width = true;
                self.dual_pol = true;
                self.polarimetric_kernel = PolarimetricKernel::BulkRayleighV1;
                self.propagation = true;
                self.propagation_geometry = PropagationGeometry::StandardFourThirdsEarth;
                self.scan_timing = ScanTiming::TimedVolume;
                self.atmosphere_time_mode = AtmosphereTimeMode::LinearAdjacent;
                self.instrument_noise = true;
                self.ref_gate_texture = false;
                self.vel_gate_texture = false;
                self.clutter_intensity = 0.0;
                self.fold_velocity = true;
            }
            SimulationMode::Presentation => {
                self.reflectivity_sampling = ReflectivitySampling::LinearZ;
                self.beam_integration = BeamIntegration::Center;
                self.terminal_fall_speed = false;
                self.terrain_blockage = false;
                self.spectrum_width = false;
                self.dual_pol = false;
                self.polarimetric_kernel = PolarimetricKernel::BulkRayleighV1;
                self.propagation = false;
                self.propagation_geometry = PropagationGeometry::StandardFourThirdsEarth;
                self.scan_timing = ScanTiming::InstantaneousTruth;
                self.atmosphere_time_mode = AtmosphereTimeMode::FrozenAtVolumeStart;
                self.instrument_noise = false;
                self.ref_gate_texture = true;
                self.vel_gate_texture = false;
                self.fold_velocity = false;
            }
        }
    }

    /// A 64-bit fingerprint of every field that changes the SAMPLED DATA of a
    /// synthetic scan. The loop-engine dedupe keys a re-import on scan time +
    /// site id (see [`crate`]'s install path); that alone cannot tell a
    /// genuinely re-configured build from an identical rebuild, so the install
    /// path folds THIS fingerprint into the per-frame history path key. An
    /// UNCHANGED config re-imports to the same key and the engine reuses the
    /// stored volume (upsert rule (b)); ANY change here yields a new key, so the
    /// freshly-built volume replaces the stale one (rule (c), equal status +
    /// different path) and flows to the display.
    ///
    /// Included — everything that alters a gate value or its position: the
    /// antenna placement (`site_id`, `site_lat_deg`, `site_lon_deg`,
    /// `antenna_msl_m`), the elevation ladder (which already encodes the
    /// optional 0.1° low tilt), `azimuth_count`, `gate_spacing_m`, `max_range_m`,
    /// `match_gate_to_grid` (the BOOL, not the resolved spacing: the effective
    /// spacing is derived from the file's `DX` at build time and the file is fixed
    /// for a run, so the same file + bool always resolves to the same spacing —
    /// hashing the bool is sufficient to force a rebuild when it toggles),
    /// `ref_floor_dbz`, `nyquist_mps`, `fold_velocity` (toggling folding
    /// re-aliases every velocity gate and re-stamps the Nyquist, so it must
    /// rebuild), `reflectivity_operator`, both gate-texture flags, and
    /// `clutter_intensity` (so moving the clutter slider rebuilds and replaces
    /// the stale volume on re-import). Deliberately EXCLUDED: `site_name` (a
    /// label — never changes a sample).
    pub fn data_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.site_id.hash(&mut hasher);
        if let Some(template) = self.exact_replay_template.as_deref() {
            "exact-observed-replay-v1".hash(&mut hasher);
            template.geometry_fingerprint().hash(&mut hasher);
        }
        // Do not hash the CustomLegacy discriminant: omitting that no-op value
        // preserves the pre-catalog default fingerprint exactly. A named VCP,
        // however, is version-locked to its primary source and every physical
        // row so a catalog revision cannot silently reuse cached scan data.
        if let Some(definition) = self.scan_strategy.definition() {
            "build-24-vcp".hash(&mut hasher);
            definition.vcp.number().hash(&mut hasher);
            definition.source.document_number.hash(&mut hasher);
            definition.source.revision.hash(&mut hasher);
            definition.source.rda_build.hash(&mut hasher);
            definition.rows.len().hash(&mut hasher);
            for row in definition.rows {
                hash_vcp_row(row, &mut hasher);
            }
        }
        hash_opt_f64(self.site_lat_deg, &mut hasher);
        hash_opt_f64(self.site_lon_deg, &mut hasher);
        hash_opt_f64(self.antenna_msl_m, &mut hasher);
        self.elevations_deg.len().hash(&mut hasher);
        for elevation in &self.elevations_deg {
            elevation.to_bits().hash(&mut hasher);
        }
        self.azimuth_count.hash(&mut hasher);
        self.gate_spacing_m.to_bits().hash(&mut hasher);
        self.max_range_m.to_bits().hash(&mut hasher);
        self.match_gate_to_grid.hash(&mut hasher);
        self.ref_floor_dbz.to_bits().hash(&mut hasher);
        self.nyquist_mps.to_bits().hash(&mut hasher);
        self.fold_velocity.hash(&mut hasher);
        (self.reflectivity_operator as u8).hash(&mut hasher);
        (self.simulation_mode as u8).hash(&mut hasher);
        (self.reflectivity_sampling as u8).hash(&mut hasher);
        (self.beam_integration as u8).hash(&mut hasher);
        self.beam_width_deg.to_bits().hash(&mut hasher);
        self.pulse_width_us.to_bits().hash(&mut hasher);
        self.radar_frequency_mhz.hash(&mut hasher);
        if let Some(band) = &self.scattering_band { band.hash(&mut hasher); }
        self.terminal_fall_speed.hash(&mut hasher);
        self.terrain_blockage.hash(&mut hasher);
        self.spectrum_width.hash(&mut hasher);
        self.spectrum_width_floor_mps.to_bits().hash(&mut hasher);
        self.dual_pol.hash(&mut hasher);
        (self.polarimetric_kernel as u8).hash(&mut hasher);
        self.property_tmatrix_table_source.hash(&mut hasher);
        self.property_tmatrix_rain_sensitivity.hash(&mut hasher);
        self.propagation.hash(&mut hasher);
        if matches!(
            self.propagation_geometry,
            PropagationGeometry::WrfRefractivityResearch
        ) {
            "wrf-refractivity-research-v1".hash(&mut hasher);
        }
        self.system_phidp_deg.to_bits().hash(&mut hasher);
        self.zdr_bias_db.to_bits().hash(&mut hasher);
        (self.scan_timing as u8).hash(&mut hasher);
        (self.atmosphere_time_mode as u8).hash(&mut hasher);
        (self.missing_neighbor_policy as u8).hash(&mut hasher);
        self.rotation_rate_deg_s.to_bits().hash(&mut hasher);
        self.transition_delay_s.to_bits().hash(&mut hasher);
        self.prf_hz.to_bits().hash(&mut hasher);
        self.coupled_single_prf_estimator.hash(&mut hasher);
        self.estimator_dwell_ms.to_bits().hash(&mut hasher);
        self.estimator_pulse_count.hash(&mut hasher);
        self.estimator_independent_sample_fraction
            .to_bits()
            .hash(&mut hasher);
        self.estimator_minimum_snr_db.to_bits().hash(&mut hasher);
        self.emit_stage_diagnostics.hash(&mut hasher);
        self.instrument_noise.hash(&mut hasher);
        self.sensitivity_dbz_at_1km.to_bits().hash(&mut hasher);
        self.ref_gate_texture.hash(&mut hasher);
        self.vel_gate_texture.hash(&mut hasher);
        self.clutter_intensity.to_bits().hash(&mut hasher);
        self.emit_quality_fields.hash(&mut hasher);
        self.minimum_model_coverage_fraction
            .to_bits()
            .hash(&mut hasher);
        hasher.finish()
    }

    pub fn clamped_minimum_model_coverage_fraction(&self) -> f32 {
        if self.minimum_model_coverage_fraction.is_finite() {
            self.minimum_model_coverage_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// The Nyquist velocity (m/s) stamped on every radial of the built volume.
    /// With folding ON it is the folding Nyquist [`Self::nyquist_mps`] — the
    /// truth the dealiaser/readout/fold-warnings must see. With folding OFF it is
    /// the historical [`UNFOLDED_NYQUIST_MPS`] (320), so the unfolded TRUE Vr is
    /// treated as already-dealiased (the dealiaser becomes a no-op because no
    /// gate exceeds 320).
    pub fn stamped_nyquist_mps(&self) -> f32 {
        if self.fold_velocity {
            self.nyquist_mps
        } else {
            UNFOLDED_NYQUIST_MPS
        }
    }

    /// Latest ray-acquisition offset in the configured physical scan. This is
    /// the duration supplied to temporal planning, not the end of an unused
    /// transition after the final cut.
    pub fn planned_scan_duration_ms(&self) -> i64 {
        if let Some(template) = self.exact_replay_template.as_deref() {
            return template.latest_acquisition_offset_ms().max(0);
        }
        if !matches!(self.scan_timing, ScanTiming::TimedVolume) {
            return 0;
        }
        let legs = self.physical_scan_legs();
        let mut cut_start_ms = 0i32;
        let mut latest_ray_ms = 0i32;
        for (cut_index, leg) in legs.iter().enumerate() {
            let rays = plan_synthetic_rays(
                cut_index,
                legs.len(),
                self.azimuth_count.max(1),
                self.scan_timing,
                leg.azimuth_rate_deg_per_second,
                cut_start_ms,
            );
            latest_ray_ms =
                latest_ray_ms.max(rays.last().map_or(cut_start_ms, |ray| ray.time_offset_ms));
            cut_start_ms = advance_cut_start_ms(self, leg, cut_start_ms);
        }
        i64::from(latest_ray_ms)
    }
}

const IDEAL_STAGE_DEFINITION: &str = "Ideal=pulse-volume moments after propagation, before receiver censoring, uncertainty, bias, ambiguity, texture, or clutter";
const MEASURED_STAGE_DEFINITION: &str = "Measured=Ideal plus PRF/dwell/pulse-count SNR estimator, receiver censoring, moment bias/uncertainty, and PRF-derived velocity ambiguity";
const PRESENTED_STAGE_DEFINITION: &str = "Presented=Measured plus optional deterministic display texture and stylized ground clutter; canonical output fields use this stage";

fn validate_coupled_estimator_inputs(config: &SyntheticRadarConfig) -> Result<(), String> {
    if let Some(definition) = config.scan_strategy.definition() {
        return Err(format!(
            "coupled single-PRF estimator cannot use named VCP {}: Appendix C PRF codes are identifiers, not frequencies",
            definition.vcp.number()
        ));
    }
    if config.radar_frequency_mhz == 0 {
        return Err("coupled estimator radar_frequency_mhz is invalid: 0".to_string());
    }
    for (name, value, positive) in [
        ("pulse_width_us", config.pulse_width_us, true),
        ("prf_hz", config.prf_hz, true),
        ("estimator_dwell_ms", config.estimator_dwell_ms, true),
        (
            "estimator_independent_sample_fraction",
            config.estimator_independent_sample_fraction,
            true,
        ),
        (
            "sensitivity_dbz_at_1km",
            config.sensitivity_dbz_at_1km,
            false,
        ),
        (
            "estimator_minimum_snr_db",
            config.estimator_minimum_snr_db,
            false,
        ),
        ("zdr_bias_db", config.zdr_bias_db, false),
    ] {
        if !value.is_finite() || (positive && value <= 0.0) {
            return Err(format!("coupled estimator {name} is invalid: {value}"));
        }
    }
    if config.estimator_independent_sample_fraction > 1.0 {
        return Err(format!(
            "coupled estimator independent-sample fraction exceeds 1: {}",
            config.estimator_independent_sample_fraction
        ));
    }
    if config.estimator_pulse_count == Some(0) {
        return Err("coupled estimator pulse count must be positive".to_string());
    }
    if config.estimator_pulse_count.is_none()
        && f64::from(config.estimator_dwell_ms) * 1.0e-3 * f64::from(config.prf_hz) < 1.0
    {
        return Err(
            "coupled estimator dwell and PRF resolve to fewer than one transmitted pulse"
                .to_string(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct PhysicalQuadraturePoint {
    az_sigma: f64,
    el_sigma: f64,
    range_offset_m: f64,
    weight: f64,
}

struct CoupledInstrumentContext {
    instrument: RadarInstrument,
    timing: ResolvedSinglePrf,
    estimator_config: MomentEstimatorConfig,
    sampling: ResolvedEstimatorSampling,
    range_resolution_m: f64,
    balanced_quadrature: Vec<PhysicalQuadraturePoint>,
    reference_quadrature: Vec<PhysicalQuadraturePoint>,
}

impl CoupledInstrumentContext {
    fn quadrature(&self, mode: BeamIntegration) -> Option<&[PhysicalQuadraturePoint]> {
        match mode {
            BeamIntegration::Center => None,
            BeamIntegration::Balanced => Some(&self.balanced_quadrature),
            BeamIntegration::Reference => Some(&self.reference_quadrature),
        }
    }

    fn stamped_nyquist_mps(&self) -> f32 {
        self.timing.nyquist_velocity_mps as f32
    }
}

fn resolve_coupled_instrument(
    config: &SyntheticRadarConfig,
) -> Result<Option<CoupledInstrumentContext>, String> {
    if !config.coupled_single_prf_estimator {
        return Ok(None);
    }
    if let Some(definition) = config.scan_strategy.definition() {
        let source_code = definition
            .rows
            .iter()
            .find_map(|row| {
                row.doppler_prfs
                    .iter()
                    .find(|cell| cell.is_default)
                    .map(|cell| cell.code)
                    .or_else(|| row.surveillance_prf.map(|prf| prf.code))
            })
            .unwrap_or(0);
        return match resolve_prf(
            &RadarInstrument::new(
                "BowEcho virtual radar",
                f64::from(config.radar_frequency_mhz) * 1.0e6,
                f64::from(config.pulse_width_us) * 1.0e-6,
            )
            .map_err(|error| format!("resolve coupled radar instrument: {error}"))?,
            PrfSpecification::NamedVcpCode {
                vcp: definition.vcp.number(),
                code: source_code,
            },
        ) {
            Ok(_) => {
                Err("named VCP PRF code unexpectedly resolved as a physical frequency".to_string())
            }
            Err(error) => Err(format!("resolve coupled single-PRF estimator: {error}")),
        };
    }

    let instrument = RadarInstrument::new(
        "BowEcho virtual radar",
        f64::from(config.radar_frequency_mhz) * 1.0e6,
        f64::from(config.pulse_width_us) * 1.0e-6,
    )
    .map_err(|error| format!("resolve coupled radar instrument: {error}"))?;
    let custom_prf = CustomSinglePrf::new(f64::from(config.prf_hz))
        .map_err(|error| format!("resolve coupled custom PRF: {error}"))?;
    let timing = resolve_prf(&instrument, PrfSpecification::CustomSinglePrf(custom_prf))
        .map_err(|error| format!("resolve coupled custom PRF: {error}"))?;
    let estimator_config = MomentEstimatorConfig {
        dwell_s: f64::from(config.estimator_dwell_ms) * 1.0e-3,
        pulse_count: config.estimator_pulse_count,
        independent_sample_fraction: f64::from(config.estimator_independent_sample_fraction),
        sensitivity_dbz_at_1km: f64::from(config.sensitivity_dbz_at_1km),
        minimum_snr_db: f64::from(config.estimator_minimum_snr_db),
        zdr_system_bias_db: f64::from(config.zdr_bias_db),
        kdp_baseline_km: (config.gate_spacing_m / 1_000.0).max(0.001),
    };
    let sampling = resolve_estimator_sampling(&timing, &estimator_config)
        .map_err(|error| format!("resolve coupled moment-estimator sampling: {error}"))?;
    // Retain the three central nodes of the five-node midpoint response for
    // the compact beam quadratures, then renormalize after truncating its two
    // outer-tail nodes.
    let response = MatchedFilterRangeResponse::new(instrument.pulse_width_s, 5)
        .map_err(|error| format!("resolve matched-filter range response: {error}"))?;
    let samples = &response.samples()[1..4];
    let retained_range_weight = samples.iter().map(|sample| sample.weight).sum::<f64>();
    let negative = samples[0];
    let center = samples[1];
    let positive = samples[2];
    let mut balanced_quadrature = Vec::with_capacity(9);
    balanced_quadrature.push(PhysicalQuadraturePoint {
        az_sigma: 0.0,
        el_sigma: 0.0,
        range_offset_m: center.offset_m,
        weight: center.weight / retained_range_weight,
    });
    for &(az_sigma, el_sigma) in &[(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
        for range in [negative, positive] {
            balanced_quadrature.push(PhysicalQuadraturePoint {
                az_sigma,
                el_sigma,
                range_offset_m: range.offset_m,
                weight: range.weight / retained_range_weight / 4.0,
            });
        }
    }

    let coordinates = [-1.0, 0.0, 1.0];
    let angular_weights = [0.25, 0.5, 0.25];
    let mut reference_quadrature = Vec::with_capacity(27);
    for (az_index, az_sigma) in coordinates.iter().copied().enumerate() {
        for (el_index, el_sigma) in coordinates.iter().copied().enumerate() {
            for range in samples {
                reference_quadrature.push(PhysicalQuadraturePoint {
                    az_sigma,
                    el_sigma,
                    range_offset_m: range.offset_m,
                    weight: angular_weights[az_index] * angular_weights[el_index] * range.weight
                        / retained_range_weight,
                });
            }
        }
    }
    Ok(Some(CoupledInstrumentContext {
        instrument,
        timing,
        estimator_config,
        sampling,
        range_resolution_m: response.range_resolution_m,
        balanced_quadrature,
        reference_quadrature,
    }))
}

fn advance_cut_start_ms(
    config: &SyntheticRadarConfig,
    leg: &SyntheticScanLeg,
    cut_start_ms: i32,
) -> i32 {
    let (sweep_ms, transition_ms) = if config.scan_strategy.is_named_vcp() {
        (
            1_000.0 * leg.source_period_seconds.max(0.0),
            1_000.0 * leg.transition_after_seconds.max(0.0),
        )
    } else {
        // Preserve the historical custom-mode arithmetic order: the f32
        // rounding of `360_000 / rate` is part of existing ray timestamps.
        (
            360_000.0 / config.rotation_rate_deg_s.max(0.1),
            1_000.0 * config.transition_delay_s.max(0.0),
        )
    };
    (f64::from(cut_start_ms) + f64::from(sweep_ms + transition_ms))
        .round()
        .clamp(0.0, f64::from(i32::MAX)) as i32
}

/// Lower clamp for a grid-matched gate spacing (m). A WRF `DX` finer than this
/// is treated as this value so an unusually fine nest cannot blow up the gate
/// count; also the floor for a garbage-but-positive `DX`.
pub const GRID_GATE_MIN_M: f64 = 100.0;
/// Upper clamp for a grid-matched gate spacing (m). A `DX` coarser than this is
/// capped so a bogus attribute cannot collapse the scan to a handful of gates.
pub const GRID_GATE_MAX_M: f64 = 10_000.0;

/// The WRF `DX` (metres) if it is a usable grid resolution — finite and
/// positive — else `None`. Shared by [`effective_gate_spacing`] and the import
/// note so both agree on whether the grid resolution was actually applied.
fn valid_grid_dx(dx_m: Option<f64>) -> Option<f64> {
    dx_m.filter(|dx| dx.is_finite() && *dx > 0.0)
}

fn hash_vcp_row(row: &PhysicalScanRow, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;

    row.elevation_deg.to_bits().hash(hasher);
    row.azimuth_rate_deg_per_second.to_bits().hash(hasher);
    row.source_period_seconds.to_bits().hash(hasher);
    row.waveform.abbreviation().hash(hasher);
    for present in [
        row.moments.has_reflectivity(),
        row.moments.has_velocity(),
        row.moments.has_spectrum_width(),
        row.moments.has_differential_reflectivity(),
        row.moments.has_correlation_coefficient(),
        row.moments.has_differential_phase(),
    ] {
        present.hash(hasher);
    }
    row.surveillance_prf
        .map(|prf| (prf.code, prf.pulse_count))
        .hash(hasher);
    row.doppler_prfs.len().hash(hasher);
    for cell in row.doppler_prfs {
        cell.code.hash(hasher);
        cell.is_default.hash(hasher);
        match cell.value {
            DopplerPrfValue::PulseCount(count) => {
                0u8.hash(hasher);
                count.hash(hasher);
            }
            DopplerPrfValue::AzimuthRateDegPerSecond(rate) => {
                1u8.hash(hasher);
                rate.to_bits().hash(hasher);
            }
        }
    }
}

fn scan_leg_metadata(leg: &SyntheticScanLeg) -> ScanLegMetadata {
    let Some(row) = leg.source_row else {
        return ScanLegMetadata::default();
    };
    let default_doppler = row.doppler_prfs.iter().find(|cell| cell.is_default);
    ScanLegMetadata {
        source_row_index: leg
            .source_row_index
            .and_then(|index| u16::try_from(index).ok()),
        elevation_deg: Some(row.elevation_deg),
        azimuth_rate_deg_per_second: Some(row.azimuth_rate_deg_per_second),
        source_period_seconds: Some(row.source_period_seconds),
        waveform: Some(row.waveform.abbreviation().to_owned()),
        moment_coverage: Some(
            if row.moments == MomentCoverage::SURVEILLANCE {
                "surveillance"
            } else if row.moments == MomentCoverage::DOPPLER {
                "doppler"
            } else {
                "all"
            }
            .to_owned(),
        ),
        surveillance_prf_code: row.surveillance_prf.map(|prf| prf.code),
        surveillance_pulse_count: row.surveillance_prf.map(|prf| prf.pulse_count),
        doppler_prf_code: default_doppler.map(|cell| cell.code),
        doppler_pulse_count: default_doppler.and_then(|cell| match cell.value {
            DopplerPrfValue::PulseCount(count) => Some(count),
            DopplerPrfValue::AzimuthRateDegPerSecond(_) => None,
        }),
    }
}

/// The gate spacing (m) one synthetic scan actually uses, given its config and
/// the source file's `DX` global attribute (`None` when absent/unreadable).
///
/// - [`SyntheticRadarConfig::match_gate_to_grid`] OFF → the configured
///   [`SyntheticRadarConfig::gate_spacing_m`] (today's behaviour, unchanged).
/// - ON with a finite, positive `DX` → that `DX`, clamped to
///   `[GRID_GATE_MIN_M, GRID_GATE_MAX_M]`, so the gate spacing matches the model
///   grid instead of oversampling it.
/// - ON with a missing / non-finite / non-positive `DX` → falls back to the
///   configured `gate_spacing_m`.
///
/// Pure (no file handle) so the resolution is unit-testable in isolation.
pub fn effective_gate_spacing(config: &SyntheticRadarConfig, dx_m: Option<f64>) -> f64 {
    if config.match_gate_to_grid
        && let Some(dx) = valid_grid_dx(dx_m)
    {
        dx.clamp(GRID_GATE_MIN_M, GRID_GATE_MAX_M)
    } else {
        config.gate_spacing_m
    }
}

/// Whether the effective gate spacing came from the grid `DX` (matching is on
/// AND the file supplied a usable `DX`) — drives the self-documenting import
/// note, distinguishing a genuine grid match from a matched-but-fell-back run.
fn matched_grid_dx(config: &SyntheticRadarConfig, dx_m: Option<f64>) -> bool {
    config.match_gate_to_grid && valid_grid_dx(dx_m).is_some()
}

/// Alias one true (unfolded) radial velocity `v` into the Nyquist co-interval
/// `[-nyquist, +nyquist)`, exactly as a real pulse-pair velocity estimator
/// reports it (WSR-88D Level II convention: `-nyquist` is representable,
/// `+nyquist` aliases to `-nyquist`).
///
/// The alias is `(v + nyquist).rem_euclid(2·nyquist) − nyquist`, the same wrap
/// the velocity dealiaser's own fixtures assume for real folded data (`render2d`
/// `dealias_v4`), so synthetic folds and the unfolder speak one language. This is
/// algebraically identical to the textbook `v − 2·nyquist·round(v/(2·nyquist))`
/// everywhere except the measure-zero boundary `v = −nyquist`: round-half-away
/// sends it to `+nyquist`, while the half-open interval keeps it at `−nyquist`.
///
/// A velocity ALREADY inside `[−nyquist, +nyquist)` is returned bit-for-bit
/// unchanged — a real estimator aliases only out-of-Nyquist velocities, so
/// in-Nyquist gates (clutter/near-zero returns, sub-Nyquist winds) are never even
/// perturbed by float rounding. Only genuinely out-of-range gates take the wrap.
///
/// A non-finite `v` (NaN / ±inf), or a non-finite / non-positive `nyquist`,
/// passes through UNCHANGED, so missing/blank gates and a degenerate Nyquist
/// never fabricate a value.
pub fn fold_velocity_mps(v: f32, nyquist: f32) -> f32 {
    if !v.is_finite() || !nyquist.is_finite() || nyquist <= 0.0 {
        return v;
    }
    // In-Nyquist velocities are reported exactly (no rounding); only out-of-range
    // gates are aliased into the co-interval.
    if (-nyquist..nyquist).contains(&v) {
        return v;
    }
    (v + nyquist).rem_euclid(2.0 * nyquist) - nyquist
}

/// Hash an optional `f64` by its bit pattern (floats are not `Hash`), with a
/// present/absent tag so `Some(0.0)` and `None` never collide.
fn hash_opt_f64(value: Option<f64>, hasher: &mut impl std::hash::Hasher) {
    use std::hash::Hash;
    match value {
        Some(value) => {
            1u8.hash(hasher);
            value.to_bits().hash(hasher);
        }
        None => 0u8.hash(hasher),
    }
}

/// The 3-D model fields one synthetic scan samples, read once per forecast
/// time and flattened to `f32` on the WRF unstaggered grid.
///
/// All 3-D arrays are row-major `[nz, ny, nx]` (index `k * ny*nx + j*nx + i`);
/// lat/lon are `[ny, nx]`. `dbz` is dBZ, winds are earth-relative m·s⁻¹,
/// `height_msl` is metres MSL.
pub struct WrfRadarFields {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub height_msl: Vec<f32>,
    pub dbz: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    pub terrain_m: Vec<f32>,
    /// Sparse additive scattering precomputed from native P3/ISHMAEL state for
    /// the opt-in research T-matrix kernel. Raw model-property arrays are
    /// dropped after closure and LUT evaluation; gate sampling interpolates
    /// only additive scattering quantities.
    property_scattering: Option<Arc<app_ui::wrf_tmatrix_scene::WrfTMatrixScene>>,
    /// Exact session table identity retained beside the derived scene so pack
    /// provenance survives through volume construction without table borrows.
    property_table_identity: Option<app_ui::wrf_tmatrix_band_assets::TMatrixBandPackIdentity>,
    property_table_resident_bytes: usize,
    /// Exact normalized P3/ISHMAEL source state retained only for
    /// RawStateLinear. Spatial and temporal weights are applied before the
    /// single per-quadrature closure/T-matrix evaluation.
    raw_property_scene: Option<Arc<app_ui::wrf_property_reader::WrfPropertyScene>>,
    /// Compact site column and bounded elevation/gate trace cache used only by
    /// the explicit WRF-refractivity research geometry.
    refractivity_model: Option<Arc<crate::wrf_refractivity::WrfRefractivityModel>>,
    /// Compact scheme-aware polarimetric state. Seven one-byte planes and one
    /// signed two-byte KDP plane retain
    /// the ratios/propagation/fall moments while `dbz` remains the full-precision
    /// ZH plane, avoiding several additional 193 MiB f32 volumes on the
    /// documented 800x800x79 nest.
    polarimetric: Option<CompactPolarFields>,
    /// Why requested dual-pol was unavailable (unsupported scheme or missing
    /// required raw fields). Scalar REF/VEL remain usable in this case.
    pub dual_pol_status: Option<String>,
    /// Optional one-byte turbulent kinetic energy (0.1 m2/s2 increments).
    tke_tenths_m2s2: Option<Vec<u8>>,
    /// Which reflectivity source populated `dbz` ("REFL_10CM" or "dbz/CALCDBZ").
    pub ref_source: &'static str,
    /// The source file's `DX` global attribute (grid resolution, metres) if the
    /// file carries a readable one, else `None`. Read once with the fields (they
    /// come from the same file) so [`build_synthetic_volume_reporting`] can size
    /// gates to the grid via [`effective_gate_spacing`] without re-opening it.
    pub dx_m: Option<f64>,
    /// Optional source labels supplied by a non-WRF adapter that deliberately
    /// reuses this renderer. Native wrfout readers leave these unset.
    source_model_override: Option<String>,
    source_microphysics_override: Option<String>,
    source_scattering_override: Option<String>,
    source_provenance_fragment: Option<String>,
    lut: InverseLut,
}

const COMPACT_ZDR_STEP_DB: f32 = 0.05;
const COMPACT_PHASE_STEP_DEG: f32 = 0.1;
const COMPACT_KDP_STEP_DEG_KM: f32 = 0.1;
const COMPACT_ATTEN_STEP_DB_KM: f32 = 0.001;
const COMPACT_FALL_STEP_MPS: f32 = 0.1;
const COMPACT_FALL_STD_STEP_MPS: f32 = 0.05;

struct CompactPolarFields {
    profile: crate::wrf_radar_physics::SchemeProfile,
    present_fields: Vec<String>,
    zdr: Vec<i8>,
    rho: Vec<u8>,
    covariance_phase: Vec<i8>,
    kdp: Vec<i16>,
    ah: Vec<u8>,
    adp: Vec<i8>,
    fall_speed: Vec<u8>,
    fall_speed_std: Vec<u8>,
    precision_audit: CompactPolarPrecisionAudit,
    /// C/X retains additive moments at each table elevation without the
    /// S-band compact attenuation ceiling of 0.255 dB/km.
    band_moments: Option<Vec<[crate::wrf_radar_physics::BulkContribution; 5]>>,
}

impl CompactPolarFields {
    fn new(
        len: usize,
        profile: crate::wrf_radar_physics::SchemeProfile,
        present_fields: Vec<String>,
    ) -> Self {
        Self {
            profile,
            present_fields,
            zdr: vec![0; len],
            rho: vec![u8::MAX; len],
            covariance_phase: vec![0; len],
            kdp: vec![0; len],
            ah: vec![0; len],
            adp: vec![0; len],
            fall_speed: vec![0; len],
            fall_speed_std: vec![0; len],
            precision_audit: CompactPolarPrecisionAudit::default(),
            band_moments: None,
        }
    }

    fn contribution_at(&self, index: usize, zh: f32, elevation: f32) -> crate::wrf_radar_physics::BulkContribution {
        if let Some(moments) = &self.band_moments {
            return crate::wrf_band_scattering::at_elevation(&moments[index], elevation);
        }
        if !zh.is_finite() || zh <= 0.0 || index >= self.zdr.len() {
            return crate::wrf_radar_physics::BulkContribution::default();
        }
        let zdr = self.zdr[index] as f32 * COMPACT_ZDR_STEP_DB;
        let zv = zh / 10.0f32.powf(zdr * 0.1);
        let rho = self.rho[index] as f32 / u8::MAX as f32;
        let covariance = rho * (zh * zv).max(0.0).sqrt();
        let phase = (self.covariance_phase[index] as f32 * COMPACT_PHASE_STEP_DEG).to_radians();
        let ah = self.ah[index] as f32 * COMPACT_ATTEN_STEP_DB_KM;
        let adp = self.adp[index] as f32 * COMPACT_ATTEN_STEP_DB_KM;
        let fall_std = self.fall_speed_std[index] as f32 * COMPACT_FALL_STD_STEP_MPS;
        crate::wrf_radar_physics::BulkContribution {
            zh,
            zv,
            cov_re: covariance * phase.cos(),
            cov_im: covariance * phase.sin(),
            kdp_deg_km: self.kdp[index] as f32 * COMPACT_KDP_STEP_DEG_KM,
            ah_db_km: ah,
            av_db_km: (ah - adp).max(0.0),
            fall_speed_mps: self.fall_speed[index] as f32 * COMPACT_FALL_STEP_MPS,
            fall_speed_variance_m2s2: fall_std * fall_std,
        }
    }

    fn store(&mut self, index: usize, sample: crate::wrf_radar_physics::IntrinsicPolarSample) {
        self.zdr[index] = quantize_i8(sample.zdr_db, COMPACT_ZDR_STEP_DB);
        self.rho[index] = (sample.rho_hv.clamp(0.0, 1.0) * u8::MAX as f32).round() as u8;
        let phase_deg = sample.cov_im.atan2(sample.cov_re).to_degrees();
        self.covariance_phase[index] = quantize_i8(phase_deg, COMPACT_PHASE_STEP_DEG);
        self.kdp[index] = quantize_i16(sample.kdp_deg_km, COMPACT_KDP_STEP_DEG_KM);
        self.ah[index] = quantize_u8(sample.ah_db_km, COMPACT_ATTEN_STEP_DB_KM);
        self.adp[index] = quantize_i8(sample.ah_db_km - sample.av_db_km, COMPACT_ATTEN_STEP_DB_KM);
        self.fall_speed[index] = quantize_u8(sample.fall_speed_mps, COMPACT_FALL_STEP_MPS);
        self.fall_speed_std[index] = quantize_u8(
            sample.fall_speed_variance_m2s2.max(0.0).sqrt(),
            COMPACT_FALL_STD_STEP_MPS,
        );

        // Audit the exact compact representation after encoding. The stored
        // codes and reconstruction path above are unchanged; this only records
        // loss/saturation so production runs can prove whether the memory
        // compression remained inside its declared precision envelope.
        let reconstructed_zdr = self.zdr[index] as f32 * COMPACT_ZDR_STEP_DB;
        let reconstructed_zv = sample.zh / 10.0f32.powf(reconstructed_zdr * 0.1);
        let reconstructed_rho = self.rho[index] as f32 / u8::MAX as f32;
        let reconstructed_covariance =
            reconstructed_rho * (sample.zh * reconstructed_zv).max(0.0).sqrt();
        let reconstructed_phase = self.covariance_phase[index] as f32 * COMPACT_PHASE_STEP_DEG;
        let source_phase = sample.cov_im.atan2(sample.cov_re).to_degrees();
        let source_adp = sample.ah_db_km - sample.av_db_km;
        let reconstructed_kdp = self.kdp[index] as f32 * COMPACT_KDP_STEP_DEG_KM;
        let reconstructed_ah = self.ah[index] as f32 * COMPACT_ATTEN_STEP_DB_KM;
        let encoded_adp = self.adp[index] as f32 * COMPACT_ATTEN_STEP_DB_KM;
        let reconstructed_av = (reconstructed_ah - encoded_adp).max(0.0);
        let reconstructed_adp = reconstructed_ah - reconstructed_av;
        let reconstructed_fall = self.fall_speed[index] as f32 * COMPACT_FALL_STEP_MPS;
        let source_fall_std = sample.fall_speed_variance_m2s2.max(0.0).sqrt();
        let reconstructed_fall_std = self.fall_speed_std[index] as f32 * COMPACT_FALL_STD_STEP_MPS;
        let reconstructed_fall_variance = reconstructed_fall_std * reconstructed_fall_std;

        self.precision_audit.zdr_db.observe(
            sample.zdr_db,
            reconstructed_zdr,
            i8::MIN as f32 * COMPACT_ZDR_STEP_DB,
            i8::MAX as f32 * COMPACT_ZDR_STEP_DB,
        );
        self.precision_audit
            .rho_hv
            .observe(sample.rho_hv, reconstructed_rho, 0.0, 1.0);
        self.precision_audit.covariance_phase_deg.observe(
            source_phase,
            reconstructed_phase,
            i8::MIN as f32 * COMPACT_PHASE_STEP_DEG,
            i8::MAX as f32 * COMPACT_PHASE_STEP_DEG,
        );
        self.precision_audit.kdp_deg_km.observe(
            sample.kdp_deg_km,
            reconstructed_kdp,
            i16::MIN as f32 * COMPACT_KDP_STEP_DEG_KM,
            i16::MAX as f32 * COMPACT_KDP_STEP_DEG_KM,
        );
        self.precision_audit.ah_db_km.observe(
            sample.ah_db_km,
            reconstructed_ah,
            0.0,
            u8::MAX as f32 * COMPACT_ATTEN_STEP_DB_KM,
        );
        self.precision_audit.adp_db_km.observe(
            source_adp,
            reconstructed_adp,
            i8::MIN as f32 * COMPACT_ATTEN_STEP_DB_KM,
            i8::MAX as f32 * COMPACT_ATTEN_STEP_DB_KM,
        );
        self.precision_audit.fall_speed_mps.observe(
            sample.fall_speed_mps,
            reconstructed_fall,
            0.0,
            u8::MAX as f32 * COMPACT_FALL_STEP_MPS,
        );
        self.precision_audit.fall_speed_std_mps.observe(
            source_fall_std,
            reconstructed_fall_std,
            0.0,
            u8::MAX as f32 * COMPACT_FALL_STD_STEP_MPS,
        );
        self.precision_audit.max_zv_relative_error = self
            .precision_audit
            .max_zv_relative_error
            .max(relative_error(sample.zv, reconstructed_zv));
        self.precision_audit.max_covariance_magnitude_relative_error = self
            .precision_audit
            .max_covariance_magnitude_relative_error
            .max(relative_error(
                sample.covariance_magnitude,
                reconstructed_covariance,
            ));
        self.precision_audit.max_av_abs_error_db_km = self
            .precision_audit
            .max_av_abs_error_db_km
            .max((reconstructed_av - sample.av_db_km).abs());
        self.precision_audit.max_fall_variance_abs_error_m2s2 = self
            .precision_audit
            .max_fall_variance_abs_error_m2s2
            .max((reconstructed_fall_variance - sample.fall_speed_variance_m2s2).abs());
    }
}

fn quantize_u8(value: f32, step: f32) -> u8 {
    if value.is_finite() && step > 0.0 {
        (value.max(0.0) / step).round().clamp(0.0, u8::MAX as f32) as u8
    } else {
        0
    }
}

fn quantize_i8(value: f32, step: f32) -> i8 {
    if value.is_finite() && step > 0.0 {
        (value / step).round().clamp(i8::MIN as f32, i8::MAX as f32) as i8
    } else {
        0
    }
}

fn quantize_i16(value: f32, step: f32) -> i16 {
    if value.is_finite() && step > 0.0 {
        (value / step)
            .round()
            .clamp(i16::MIN as f32, i16::MAX as f32) as i16
    } else {
        0
    }
}

impl WrfRadarFields {
    fn cells(&self) -> usize {
        self.nx * self.ny
    }

    fn has_polarimetric_input(&self) -> bool {
        self.polarimetric.is_some()
            || self.property_scattering.is_some()
            || self.raw_property_scene.is_some()
    }

    fn base_retained_bytes_without_property_scattering(&self) -> usize {
        let f32_values = self
            .lat
            .len()
            .saturating_add(self.lon.len())
            .saturating_add(self.height_msl.len())
            .saturating_add(self.dbz.len())
            .saturating_add(self.u.len())
            .saturating_add(self.v.len())
            .saturating_add(self.w.len())
            .saturating_add(self.terrain_m.len());
        let compact_polar_bytes = self.polarimetric.as_ref().map_or(0, |polar| {
            polar.zdr.len()
                + polar.rho.len()
                + polar.covariance_phase.len()
                + polar.kdp.len() * std::mem::size_of::<i16>()
                + polar.ah.len()
                + polar.adp.len()
                + polar.fall_speed.len()
                + polar.fall_speed_std.len()
                + polar.band_moments.as_ref().map_or(0, |v| v.len()*std::mem::size_of::<[crate::wrf_radar_physics::BulkContribution;5]>())
        });
        std::mem::size_of::<Self>()
            .saturating_add(f32_values.saturating_mul(std::mem::size_of::<f32>()))
            .saturating_add(compact_polar_bytes)
            .saturating_add(
                self.tke_tenths_m2s2
                    .as_ref()
                    .map_or(0, |values| values.len()),
            )
            .saturating_add(self.lut.retained_bytes())
            .saturating_add(
                self.dual_pol_status
                    .as_ref()
                    .map_or(0, std::string::String::len),
            )
            .saturating_add(
                self.source_model_override
                    .as_ref()
                    .map_or(0, std::string::String::len),
            )
            .saturating_add(
                self.source_microphysics_override
                    .as_ref()
                    .map_or(0, std::string::String::len),
            )
            .saturating_add(
                self.source_scattering_override
                    .as_ref()
                    .map_or(0, std::string::String::len),
            )
            .saturating_add(
                self.source_provenance_fragment
                    .as_ref()
                    .map_or(0, std::string::String::len),
            )
            .saturating_add(self.refractivity_model.as_deref().map_or(0, |model| {
                std::mem::size_of::<crate::wrf_refractivity::WrfRefractivityModel>()
                    + std::mem::size_of_val(model.profile().levels())
            }))
    }

    fn retained_bytes_estimate(&self) -> usize {
        self.base_retained_bytes_without_property_scattering()
            .saturating_add(
                self.property_scattering
                    .as_deref()
                    .map_or(0, |scene| scene.memory_estimate().retained_bytes()),
            )
            .saturating_add(
                self.raw_property_scene
                    .as_deref()
                    .map_or(0, |scene| scene.memory_estimate().retained_bytes()),
            )
    }

    /// Domain-centre grid cell (used for the default antenna position).
    fn center_cell(&self) -> usize {
        (self.ny / 2) * self.nx + (self.nx / 2)
    }
}

fn resolve_radar_site(
    fields: &WrfRadarFields,
    config: &SyntheticRadarConfig,
) -> Result<(f64, f64, f64), String> {
    let center = fields.center_cell();
    let site_lat = config
        .site_lat_deg
        .unwrap_or_else(|| f64::from(fields.lat[center]));
    let site_lon = config
        .site_lon_deg
        .unwrap_or_else(|| f64::from(fields.lon[center]));
    if !site_lat.is_finite()
        || !site_lon.is_finite()
        || !(-90.0..=90.0).contains(&site_lat)
        || !(-180.0..=180.0).contains(&site_lon)
    {
        return Err(format!(
            "synthetic radar site is invalid: lat={site_lat}, lon={site_lon}"
        ));
    }
    let antenna_msl = config.antenna_msl_m.unwrap_or_else(|| {
        let site_cell = fields
            .lut
            .lookup(site_lat as f32, site_lon as f32)
            .unwrap_or(center);
        f64::from(fields.terrain_m[site_cell]) + DEFAULT_TOWER_M
    });
    if !antenna_msl.is_finite() {
        return Err(format!(
            "synthetic radar antenna altitude is invalid: {antenna_msl} m MSL"
        ));
    }
    Ok((site_lat, site_lon, antenna_msl))
}

fn remaining_property_tmatrix_build_budget_bytes(
    config: &SyntheticRadarConfig,
    fields: &WrfRadarFields,
    expected_cells: usize,
    reserved_memory_bytes: usize,
    selected_table_bytes: usize,
) -> Result<usize, String> {
    let budget_bytes = config
        .temporal_memory_budget_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| "Simulated-radar memory budget overflows address space".to_string())?;
    checked_property_tmatrix_build_remainder(
        budget_bytes,
        fields.base_retained_bytes_without_property_scattering(),
        if config.spectrum_width {
            expected_cells
        } else {
            0
        },
        selected_table_bytes,
        reserved_memory_bytes,
        minimum_property_tmatrix_owned_peak_bytes(expected_cells)?,
    )
}

fn ensure_raw_property_retention_budget(
    config: &SyntheticRadarConfig,
    fields: &WrfRadarFields,
    property_scene: &app_ui::wrf_property_reader::WrfPropertyScene,
    expected_cells: usize,
    reserved_memory_bytes: usize,
) -> Result<(), String> {
    let budget_bytes = config
        .temporal_memory_budget_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| "Raw-state temporal memory budget overflows address space".to_string())?;
    let tke_bytes = if config.spectrum_width {
        expected_cells
    } else {
        0
    };
    let required = fields
        .base_retained_bytes_without_property_scattering()
        .checked_add(property_scene.memory_estimate().retained_bytes())
        .and_then(|value| {
            value.checked_add(app_ui::wrf_tmatrix_assets::embedded_lut_memory_bytes())
        })
        .and_then(|value| value.checked_add(tke_bytes))
        .and_then(|value| value.checked_add(reserved_memory_bytes))
        .ok_or_else(|| "Raw-state temporal retained-memory estimate overflowed".to_string())?;
    if required > budget_bytes {
        return Err(format!(
            "RawStateLinear needs {:.2} GiB for retained atmosphere, exact property state, embedded tables, TKE, outputs, and cache, above the configured {:.2} GiB budget",
            required as f64 / 1024.0_f64.powi(3),
            budget_bytes as f64 / 1024.0_f64.powi(3),
        ));
    }
    Ok(())
}

fn minimum_property_tmatrix_owned_peak_bytes(expected_cells: usize) -> Result<usize, String> {
    // The raw source owns dense temperature, pressure, moist-density, and
    // dry-density f32 planes.
    // While that source is retained, the scattering scene must also allocate
    // one dense u32 full-cell -> sparse-row map. Sparse categories, additive
    // output rows, provenance, and worker scratch only increase this bound.
    let unavoidable_dense_bytes = expected_cells
        .checked_mul(4 * std::mem::size_of::<f32>() + std::mem::size_of::<u32>())
        .ok_or_else(|| "Property T-matrix minimum source-memory estimate overflowed".to_string())?;
    unavoidable_dense_bytes
        .checked_add(std::mem::size_of::<
            app_ui::wrf_property_reader::WrfPropertyScene,
        >())
        .and_then(|value| {
            value.checked_add(std::mem::size_of::<
                app_ui::wrf_tmatrix_scene::WrfTMatrixScene,
            >())
        })
        .ok_or_else(|| "Property T-matrix minimum source-memory estimate overflowed".to_string())
}

fn checked_property_tmatrix_build_remainder(
    budget_bytes: usize,
    retained_field_bytes: usize,
    spectrum_width_bytes: usize,
    embedded_lut_bytes: usize,
    reserved_memory_bytes: usize,
    minimum_owned_peak_bytes: usize,
) -> Result<usize, String> {
    let current_base_bytes = retained_field_bytes
        .checked_add(spectrum_width_bytes)
        .and_then(|value| value.checked_add(embedded_lut_bytes))
        .and_then(|value| value.checked_add(reserved_memory_bytes))
        .ok_or_else(|| "Property T-matrix build-memory reservation overflowed".to_string())?;
    let remaining = budget_bytes.saturating_sub(current_base_bytes);
    if remaining >= minimum_owned_peak_bytes {
        Ok(remaining)
    } else {
        Err(format!(
            "Property T-matrix raw read and dense lookup need at least {:.2} GiB after retained fields, tables, outputs, and cache, but only {:.2} GiB remains inside the configured {:.2} GiB budget",
            minimum_owned_peak_bytes as f64 / 1024.0_f64.powi(3),
            remaining as f64 / 1024.0_f64.powi(3),
            budget_bytes as f64 / 1024.0_f64.powi(3),
        ))
    }
}

/// Read the WRF 3-D reflectivity + earth-relative winds + height for one time.
///
/// Reflectivity: `REFL_10CM` (the model's own Thompson 10-cm reflectivity) when
/// present and the operator is [`ReflectivityOperator::ModelNative`], else
/// `wrf-core`'s `dbz` (`CALCDBZ`, Stoelinga 2005) — the same diagnostic
/// BowEcho's composite reflectivity uses, so a synthetic scan co-locates with
/// the model's own composite. [`ReflectivityOperator::ClassicStoelinga`] forces
/// the `CALCDBZ` path even when the file carries `REFL_10CM`. Raw wrfout carries
/// the hydrometeor mixing ratios `dbz` needs; a post-processed / climate wrfout
/// may not — this returns an `Err` (empty/absent reflectivity) so the caller can
/// warn rather than emit an all-NaN scan.
pub fn read_wrf_radar_fields(
    file: &WrfFile,
    timeidx: usize,
    operator: ReflectivityOperator,
) -> Result<WrfRadarFields, String> {
    read_wrf_radar_fields_reporting(file, timeidx, operator, &|_| {})
}

/// Read one time's fields, streaming stage labels through `progress` so the UI
/// can show "Reading …" instead of freezing.
///
/// PERF: the four heavy 3-D fields (height, reflectivity, earth-relative winds,
/// vertical velocity) are the whole cost of a synthetic scan — on a 250 m
/// 800×800×79 wrfout the NetCDF decompress dominates (~7 s serial), while the
/// polar sampling that follows is <0.1 s. Each is an independent variable, so
/// they are read/decompressed on separate threads with `std::thread::scope`.
/// The pure-Rust HDF5 reader guards only the file handle with a mutex and
/// decompresses (the expensive part) without it, so the inflates overlap:
/// wall time drops to the single longest field (~2.5–3 s here, ~2.5× faster).
/// Each thread calls the exact same `getvar`/`read_var` entry points as before,
/// so the sampled output is byte-for-byte unchanged — this is a speed change,
/// not an accuracy change.
pub fn read_wrf_radar_fields_reporting(
    file: &WrfFile,
    timeidx: usize,
    operator: ReflectivityOperator,
    progress: &dyn Fn(&str),
) -> Result<WrfRadarFields, String> {
    read_wrf_radar_fields_reporting_inner(file, timeidx, operator, progress, true)
}

fn read_wrf_radar_fields_reporting_inner(
    file: &WrfFile,
    timeidx: usize,
    operator: ReflectivityOperator,
    progress: &dyn Fn(&str),
    require_native_reflectivity: bool,
) -> Result<WrfRadarFields, String> {
    let nx = file.nx;
    let ny = file.ny;
    let nz = file.nz;
    let cells = nx * ny;
    if cells == 0 || nz == 0 {
        return Err("WRF grid has zero cells".to_string());
    }

    // Grid resolution (metres) for the optional "match gate size to grid" mode.
    // Read here where the file handle lives and carried on the fields; a missing
    // or unreadable attribute is `None` and the builder falls back to the
    // configured gate spacing. WRF writes `DX` in metres.
    let dx_m = file.global_attr_f64("DX").ok();

    let lat = file
        .xlat(timeidx)
        .map_err(|err| format!("read XLAT: {err}"))?;
    let lon = file
        .xlong(timeidx)
        .map_err(|err| format!("read XLONG: {err}"))?;
    if lat.len() != cells || lon.len() != cells {
        return Err(format!(
            "WRF lat/lon size mismatch: expected {cells}, got lat {} lon {}",
            lat.len(),
            lon.len()
        ));
    }

    progress(if require_native_reflectivity {
        "reading model fields (reflectivity, winds, height)…"
    } else {
        "reading model fields (winds and height; reflectivity comes from property T-matrix)…"
    });

    // Read the four heavy 3-D fields concurrently. Placeholders are overwritten
    // inside the scope; the scope join guarantees they are all set on exit.
    let mut height_res: Result<Vec<f32>, String> = Err("height not read".to_string());
    let mut refl_res: Result<(Vec<f32>, &'static str), String> = Err("refl not read".to_string());
    let mut winds_res: Result<(Vec<f32>, Vec<f32>), String> = Err("winds not read".to_string());
    let mut w_res: Result<Vec<f32>, String> = Err("wa not read".to_string());
    let mut terrain_m: Vec<f32> = Vec::new();
    std::thread::scope(|scope| {
        let th_height = scope.spawn(|| read_3d(file, "height", timeidx, nz * cells));
        let th_refl = scope.spawn(|| {
            if require_native_reflectivity {
                read_reflectivity(file, timeidx, nz * cells, operator)
            } else {
                Ok((Vec::new(), PROPERTY_TMATRIX_S_BAND_SOURCE))
            }
        });
        let th_winds = scope.spawn(|| read_earth_relative_winds(file, timeidx, nz * cells));
        let th_w = scope.spawn(|| read_3d(file, "wa", timeidx, nz * cells));
        let th_terrain = scope.spawn(|| read_terrain_m(file, timeidx, cells));

        height_res = join_read(th_height, "height");
        refl_res = join_read(th_refl, "reflectivity");
        winds_res = join_read(th_winds, "winds");
        w_res = join_read(th_w, "wa");
        terrain_m = th_terrain.join().unwrap_or_else(|_| vec![0.0; cells]);
    });

    let height = height_res?;
    let (dbz, ref_source) = refl_res?;
    if require_native_reflectivity && dbz.iter().all(|value| !value.is_finite()) {
        return Err(format!(
            "WRF reflectivity ({ref_source}) is entirely missing — is this a \
             post-processed/climate wrfout without hydrometeor mixing ratios?"
        ));
    }
    let (u, v) = winds_res?;
    let w = w_res?;

    progress("building geolocation index…");
    let lat_f32 = to_f32(&lat);
    let lon_f32 = to_f32(&lon);
    // Domain-bounded: a WRF grid's row/col perimeter IS its true data edge,
    // so gates past that edge (but still inside the rectangular lat/lon
    // bbox) must sample nothing — without the bound, the LUT's hole-fill
    // dilation leaked nearest-edge values into a ~3-bin smeared ring around
    // the domain boundary.
    let lut = InverseLut::build_with_shape_domain_bounded(&lat_f32, &lon_f32, nx, ny)
        .ok_or_else(|| "failed to build WRF inverse geolocation LUT".to_string())?;

    Ok(WrfRadarFields {
        nx,
        ny,
        nz,
        lat: lat_f32,
        lon: lon_f32,
        height_msl: height,
        dbz,
        u,
        v,
        w,
        terrain_m,
        property_scattering: None,
        property_table_identity: None,
        property_table_resident_bytes: 0,
        raw_property_scene: None,
        refractivity_model: None,
        polarimetric: None,
        dual_pol_status: None,
        tke_tenths_m2s2: None,
        ref_source,
        dx_m,
        source_model_override: None,
        source_microphysics_override: None,
        source_scattering_override: None,
        source_provenance_fragment: None,
        lut,
    })
}

/// Config-aware field reader used by the production worker. The public
/// operator-only reader above remains the compatibility/fixture seam; advanced
/// modes add compact microphysical scattering and optional TKE here.
pub fn read_wrf_radar_fields_for_config_reporting(
    file: &WrfFile,
    source_identity: &WrfSourceIdentity,
    timeidx: usize,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    reserved_memory_bytes: usize,
    cancel: Option<&AtomicBool>,
) -> Result<WrfRadarFields, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let property_tmatrix = config.polarimetric_kernel.is_property_tmatrix();
    let mut fields = read_wrf_radar_fields_reporting_inner(
        file,
        timeidx,
        config.reflectivity_operator,
        progress,
        !property_tmatrix,
    )?;
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let expected = fields.nx * fields.ny * fields.nz;

    if matches!(
        config.propagation_geometry,
        PropagationGeometry::WrfRefractivityResearch
    ) {
        progress("reading WRF refractivity profile at the radar site…");
        let (site_lat, site_lon, antenna_msl) = resolve_radar_site(&fields, config)?;
        let model = crate::wrf_refractivity::read_wrf_refractivity_model(
            file,
            timeidx,
            crate::wrf_refractivity::WrfRefractivityGrid {
                nx: fields.nx,
                ny: fields.ny,
                nz: fields.nz,
                latitude_deg: &fields.lat,
                longitude_deg: &fields.lon,
                height_msl: &fields.height_msl,
                site_latitude_deg: site_lat,
                site_longitude_deg: site_lon,
                antenna_msl_m: antenna_msl,
            },
        )
        .map_err(|error| format!("build WRF refractivity geometry: {error}"))?;
        fields.refractivity_model = Some(Arc::new(model));
    }

    if property_tmatrix {
        if !config.dual_pol {
            return Err("Property T-matrix research mode requires dual polarization".to_string());
        }
        let raw_state_linear = matches!(
            config.atmosphere_time_mode,
            AtmosphereTimeMode::RawStateLinear
        );
        if raw_state_linear && config.polarimetric_kernel.is_hybrid() {
            return Err(
                "P3 Hybrid does not support RawStateLinear; choose Frozen/additive adjacent-scene timing or experimental Full P3/ISHMAEL T-matrix strict mode"
                    .to_owned(),
            );
        }
        let table_owner =
            app_ui::wrf_tmatrix_assets::load_property_tmatrix_tables_exact_with_progress(
                config.property_tmatrix_table_source,
                f64::from(config.radar_frequency_mhz) * 1.0e6,
                progress,
            )?;
        let table_identity = table_owner.identity();
        let table_resident_bytes = table_owner.retained_source_bytes();
        let maximum_owned_peak_bytes = if raw_state_linear {
            None
        } else {
            Some(remaining_property_tmatrix_build_budget_bytes(
                config,
                &fields,
                expected,
                reserved_memory_bytes,
                table_resident_bytes,
            )?)
        };
        progress("reading native P3/ISHMAEL property state…");
        let property_scene = app_ui::wrf_property_reader::read_wrf_property_scene(
            file,
            source_identity.clone(),
            timeidx,
        )
        .map_err(|error| format!("read native P3/ISHMAEL property state: {error}"))?;
        if let Some(cancel) = cancel {
            check_synthetic_radar_cancel(cancel)?;
        }
        if property_scene.cell_count() != expected {
            return Err(format!(
                "property scene has {} cells, expected {expected}",
                property_scene.cell_count()
            ));
        }
        if raw_state_linear {
            app_ui::wrf_tmatrix_assets::preload_embedded_property_tmatrix_for_scheme(
                property_scene.microphysics_scheme_id(),
                progress,
            )
            .map_err(|error| format!("preload property T-matrix scheme assets: {error}"))?;
        }
        let fields_read = property_scene
            .source_fields()
            .iter()
            .map(app_ui::wrf_property_reader::SourceFieldProvenance::source_name)
            .collect::<Vec<_>>()
            .join(",");
        if raw_state_linear {
            progress("retaining raw property state for pre-closure interpolation…");
        } else {
            progress("evaluating property T-matrix scene…");
        }
        fields.ref_source =
            property_tmatrix_ref_source(config.polarimetric_kernel, config.radar_frequency_mhz)?;
        fields.property_table_identity = Some(table_identity.clone());
        fields.property_table_resident_bytes = table_resident_bytes;
        if raw_state_linear {
            ensure_raw_property_retention_budget(
                config,
                &fields,
                &property_scene,
                expected,
                reserved_memory_bytes,
            )?;
            fields.dual_pol_status = Some(format!(
                "property-aware T-matrix raw_state_linear input (mp_physics={}; table={}; frequency={:.0} Hz; rain={:?}; raw fields={fields_read}; retained raw state {:.2} GiB)",
                property_scene.microphysics_scheme_id(),
                table_identity.pack_id(),
                table_identity.exact_frequency_hz(),
                config.property_tmatrix_rain_sensitivity,
                property_scene.memory_estimate().retained_bytes() as f64 / 1024.0_f64.powi(3),
            ));
            fields.raw_property_scene = Some(Arc::new(property_scene));
        } else {
            let maximum_owned_peak_bytes =
                maximum_owned_peak_bytes.expect("non-raw property build has a budget");
            let scattering_policy = config.polarimetric_kernel.scattering_policy();
            let built = if let Some(cancel) = cancel {
                app_ui::wrf_tmatrix_assets::build_property_tmatrix_scene_with_policy_and_cancel(
                    &property_scene,
                    maximum_owned_peak_bytes,
                    table_owner,
                    config.property_tmatrix_rain_sensitivity.runtime(),
                    scattering_policy,
                    progress,
                    config.runtime_cuda_service.as_deref(),
                    cancel,
                )
            } else {
                match config.runtime_cuda_service.as_deref() {
                    Some(cuda) => {
                        app_ui::wrf_tmatrix_assets::build_property_tmatrix_scene_with_policy_and_cuda(
                            &property_scene,
                            maximum_owned_peak_bytes,
                            table_owner,
                            config.property_tmatrix_rain_sensitivity.runtime(),
                            scattering_policy,
                            progress,
                            cuda,
                        )
                    }
                    None => app_ui::wrf_tmatrix_assets::build_property_tmatrix_scene_with_policy(
                        &property_scene,
                        maximum_owned_peak_bytes,
                        table_owner,
                        config.property_tmatrix_rain_sensitivity.runtime(),
                        scattering_policy,
                        progress,
                    ),
                }
            }
            .map_err(|error| format!("build property T-matrix scene: {error}"))?;
            if let Some(cancel) = cancel {
                check_synthetic_radar_cancel(cancel)?;
            }
            let property_scattering = built.scene;
            let counts = property_scattering.provenance().counts;
            fields.dual_pol_status = Some(format!(
                "property-aware scattering input (mp_physics={}; policy={}; table={}; source={:?}; frequency={:.0} Hz; rain={:?}; native PSD populations={}; hybrid bulk cells={}; hybrid bulk populations={}; raw fields={fields_read}; build peak {:.2} GiB)",
                property_scattering.microphysics_scheme_id(),
                built.scattering_policy.identifier(),
                built.table_identity.pack_id(),
                built.table_source,
                built.exact_frequency_hz,
                built.rain_mode,
                counts.scheme_native_psd_populations,
                counts.hybrid_bulk_rayleigh_cells,
                counts.hybrid_bulk_rayleigh_populations,
                built.peak.estimated_peak_bytes as f64 / 1024.0_f64.powi(3),
            ));
            fields.property_scattering = Some(Arc::new(property_scattering));
        }
    } else if let Some(band) = &config.scattering_band {
        let (polar, dbz, audit) = build_band_polar_fields(file, timeidx, expected, band)?;
        fields.dual_pol_status = Some(audit);
        fields.dbz = dbz;
        fields.ref_source = crate::wrf_band_scattering::MODEL_ID;
        fields.source_scattering_override = Some(crate::wrf_band_scattering::MODEL_ID.into());
        fields.polarimetric = Some(polar);
    } else if config.dual_pol || config.terminal_fall_speed {
        progress("deriving scheme-aware hydrometeor scattering…");
        match build_compact_polar_fields(file, timeidx, expected, progress) {
            Ok((polar, bulk_zh_dbz)) => {
                let audit = &polar.precision_audit;
                fields.dual_pol_status = Some(format!(
                    "{} ({:?}{}; compact audit: clamps={}, quantized-to-zero={}, max error ZDR={:.4} dB rho={:.5} KDP={:.4} deg/km AH={:.4} dB/km)",
                    polar.profile.name,
                    polar.profile.capability,
                    if polar.profile.assumption_heavy {
                        "; assumed PSD parameters"
                    } else {
                        ""
                    },
                    audit.total_clamps(),
                    audit.total_quantized_to_zero(),
                    audit.zdr_db.max_abs_reconstruction_error,
                    audit.rho_hv.max_abs_reconstruction_error,
                    audit.kdp_deg_km.max_abs_reconstruction_error,
                    audit.ah_db_km.max_abs_reconstruction_error,
                ));
                if config.dual_pol {
                    fields.dbz = bulk_zh_dbz;
                    fields.ref_source = BULK_S_BAND_SOURCE;
                }
                fields.polarimetric = Some(polar);
            }
            Err(reason) => {
                fields.dual_pol_status = Some(reason);
            }
        }
    }

    if config.spectrum_width {
        fields.tke_tenths_m2s2 = read_compact_tke(file, timeidx, expected);
    }
    Ok(fields)
}

const MICROPHYSICS_FIELD_NAMES: &[&str] = &[
    "QCLOUD",
    "QRAIN",
    "QICE",
    "QICE2",
    "QICE3",
    "QSNOW",
    "QGRAUP",
    "QGRAUPEL",
    "QHAIL",
    "QNDROP",
    "QNCLOUD",
    "QNRAIN",
    "QNICE",
    "QNICE2",
    "QNICE3",
    "QNSNOW",
    "QNGRAUP",
    "QNGRAUPEL",
    "QNHAIL",
    "QVGRAUPEL",
    "QVHAIL",
];

struct SpeciesReadSpec {
    kind: crate::wrf_radar_physics::HydrometeorKind,
    mass: &'static [&'static str],
    number: &'static [&'static str],
    volume: &'static [&'static str],
}

const SPECIES_READ_SPECS: &[SpeciesReadSpec] = &[
    SpeciesReadSpec {
        kind: crate::wrf_radar_physics::HydrometeorKind::CloudWater,
        mass: &["QCLOUD"],
        number: &["QNCLOUD", "QNDROP"],
        volume: &[],
    },
    SpeciesReadSpec {
        kind: crate::wrf_radar_physics::HydrometeorKind::Rain,
        mass: &["QRAIN"],
        number: &["QNRAIN"],
        volume: &[],
    },
    SpeciesReadSpec {
        kind: crate::wrf_radar_physics::HydrometeorKind::CloudIce,
        mass: &["QICE", "QICE2", "QICE3"],
        number: &["QNICE", "QNICE2", "QNICE3"],
        volume: &[],
    },
    SpeciesReadSpec {
        kind: crate::wrf_radar_physics::HydrometeorKind::Snow,
        mass: &["QSNOW"],
        number: &["QNSNOW"],
        volume: &[],
    },
    SpeciesReadSpec {
        kind: crate::wrf_radar_physics::HydrometeorKind::Graupel,
        mass: &["QGRAUP", "QGRAUPEL"],
        number: &["QNGRAUPEL", "QNGRAUP"],
        volume: &["QVGRAUPEL"],
    },
    SpeciesReadSpec {
        kind: crate::wrf_radar_physics::HydrometeorKind::Hail,
        mass: &["QHAIL"],
        number: &["QNHAIL"],
        volume: &["QVHAIL"],
    },
];

fn build_compact_polar_fields(
    file: &WrfFile,
    timeidx: usize,
    expected: usize,
    progress: &dyn Fn(&str),
) -> Result<(CompactPolarFields, Vec<f32>), String> {
    let present_fields: Vec<String> = MICROPHYSICS_FIELD_NAMES
        .iter()
        .filter(|name| file.has_var(name))
        .map(|name| (*name).to_string())
        .collect();
    let scheme_id = file.global_attr_i32("MP_PHYSICS").ok();
    let profile = crate::wrf_radar_physics::detect_scheme(scheme_id, &present_fields);
    if profile.capability == crate::wrf_radar_physics::MicrophysicsCapability::Unsupported {
        return Err(format!(
            "dual-pol unavailable: {} cannot be represented by the bulk-category operator",
            profile.name
        ));
    }

    let temperature = file
        .temperature(timeidx)
        .map_err(|error| format!("dual-pol temperature: {error}"))?
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    file.clear_cache();
    if temperature.len() != expected {
        return Err(format!(
            "dual-pol temperature has {} values, expected {expected}",
            temperature.len()
        ));
    }

    let qv = if file.has_var("QVAPOR") {
        file.read_var("QVAPOR", timeidx)
            .map_err(|error| format!("dual-pol QVAPOR: {error}"))?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>()
    } else {
        vec![0.0; expected]
    };
    let pressure = file
        .full_pressure(timeidx)
        .map_err(|error| format!("dual-pol pressure: {error}"))?;
    if pressure.len() != expected || qv.len() != expected {
        file.clear_cache();
        return Err("dual-pol pressure/QVAPOR grid shape mismatch".to_string());
    }
    let air_density = pressure
        .iter()
        .zip(&temperature)
        .zip(&qv)
        .map(|((&pressure, &temperature), &qv)| {
            crate::wrf_radar_physics::air_density(pressure as f32, temperature, qv)
        })
        .collect::<Vec<_>>();
    drop(pressure);
    file.clear_cache();

    let mut compact = CompactPolarFields::new(expected, profile.clone(), present_fields);
    let mut zh_linear = vec![0.0f32; expected];
    let mut species_used = 0usize;
    for spec in SPECIES_READ_SPECS {
        let Some(mass_name) = first_existing_var(file, spec.mass) else {
            continue;
        };
        progress(&format!("scattering {mass_name}…"));
        let number = read_optional_first_var_f32(file, timeidx, spec.number, expected)?;
        let volume = read_optional_first_var_f32(file, timeidx, spec.volume, expected)?;
        let mass = file
            .read_var(mass_name, timeidx)
            .map_err(|error| format!("dual-pol {mass_name}: {error}"))?;
        if mass.len() != expected {
            return Err(format!(
                "dual-pol {mass_name} has {} values, expected {expected}",
                mass.len()
            ));
        }
        species_used += 1;
        for index in 0..expected {
            let q = mass[index] as f32;
            if !q.is_finite() || q <= 0.0 {
                continue;
            }
            let contribution = crate::wrf_radar_physics::bulk_sband_contribution(
                crate::wrf_radar_physics::BulkSpeciesInput {
                    kind: spec.kind,
                    q_kgkg: q,
                    number_per_kg: number.as_ref().map(|values| values[index]),
                    volume_m3_per_kg: volume.as_ref().map(|values| values[index]),
                    temperature_k: temperature[index],
                    air_density_kgm3: air_density[index],
                },
                &profile,
            );
            if contribution.zh <= 0.0 {
                continue;
            }
            let mut accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
            if zh_linear[index] > 0.0 {
                accumulator.add(1.0, compact.contribution_at(index, zh_linear[index], 0.));
            }
            accumulator.add(1.0, contribution);
            let combined = accumulator.finalize();
            zh_linear[index] = combined.zh;
            compact.store(index, combined);
        }
    }
    if species_used == 0 || zh_linear.iter().all(|value| *value <= 0.0) {
        return Err(format!(
            "dual-pol unavailable: {} contains no usable bulk hydrometeor mass fields",
            profile.name
        ));
    }
    let zh_dbz = zh_linear
        .into_iter()
        .map(|value| {
            if value.is_finite() && value > 0.0 {
                crate::wrf_radar_physics::z_to_dbz(value)
            } else {
                f32::NAN
            }
        })
        .collect();
    Ok((compact, zh_dbz))
}

fn first_existing_var<'a>(file: &WrfFile, names: &'a [&str]) -> Option<&'a str> {
    names.iter().copied().find(|name| file.has_var(name))
}

/// Read the same scheme carriers as the polarimetric operator, then run the
/// independent cell quadratures on Rayon. No forecast arrays are modified.
fn build_band_polar_fields(
    file: &WrfFile, timeidx: usize, expected: usize, band: &str,
) -> Result<(CompactPolarFields,Vec<f32>,String),String> {
    use crate::wrf_band_scattering as scattering;
    let scheme=file.global_attr_i32("MP_PHYSICS").map_err(|_| "radar band scattering requires MP_PHYSICS to select its PSD")?;
    scattering::validate_scheme(scheme)?;
    let table=scattering::table(band)?;
    let read=|names:&[&str]| -> Result<Vec<f32>,String> {
        read_optional_first_var_f32(file,timeidx,names,expected)
            .map(|v|v.unwrap_or_else(||vec![0.;expected]))
    };
    let mass=[read(&["QRAIN"] )?,read(&["QSNOW"] )?,read(&["QICE"] )?,
        read(&["QGRAUP","QGRAUPEL"] )?,read(&["QHAIL"] )?];
    let number=[read(&["QNRAIN"] )?,read(&["QNSNOW"] )?,read(&["QNICE"] )?,
        read(&["QNGRAUPEL","QNGRAUP"] )?,read(&["QNHAIL"] )?];
    for (s,names) in [vec!["QNRAIN"],vec!["QNSNOW"],vec!["QNICE"],vec!["QNGRAUPEL","QNGRAUP"],vec!["QNHAIL"]].iter().enumerate() {
        if !(scheme==8 && (s==1 || s==3)) && !names.iter().any(|n|file.has_var(n)) && mass[s].iter().any(|q|*q>1e-12) {
            return Err(format!("radar PSD number field missing for species {s}; retain the scheme number moments in history"));
        }
    }
    let volume=[read(&["QVGRAUPEL"] )?,read(&["QVHAIL"] )?];
    let rime=read(&["QIR"] )?;let rime_volume=read(&["QIB"] )?;
    let vapor=read(&["QVAPOR"] )?;
    let pressure=file.full_pressure(timeidx).map_err(|e|format!("radar pressure: {e}"))?;
    let mut temperature:Vec<f32>=file.temperature(timeidx).map_err(|e|format!("radar temperature: {e}"))?
        .iter().map(|v|*v as f32).collect();
    if pressure.len()!=expected || temperature.len()!=expected {return Err("radar thermodynamic grid shape mismatch".into());}
    if scheme>=50 {
        let theta=read(&["T"] )?;
        for i in 0..expected {temperature[i]=(theta[i]+300.)*(pressure[i] as f32*1e-5).powf(287.15/1005.);}
        if !file.has_var("QIR") || !file.has_var("QIB") {return Err("radar P3 requires QIR and QIB; omitting rime changes ice density and scattering".into());}
    }
    let altitude=read_optional_first_var_f32(file,timeidx,&["ALT"],expected)?;
    let p3=if scheme>=50 {scattering::p3_table()} else {&[]};
    let evaluated:Vec<scattering::Scattering>=(0..expected).into_par_iter().map(|i| {
        let density=if scheme>=50 {pressure[i] as f32/(287.15*temperature[i])}
            else if scheme==18 && altitude.is_some() {1./altitude.as_ref().unwrap()[i]}
            else {pressure[i] as f32/(287.04*temperature[i]*(1.+vapor[i]/0.622))};
        if !density.is_finite() || density<=0. || !temperature[i].is_finite() {
            return Err(format!("radar thermodynamics invalid at model cell {i}"));
        }
        let mut cell=scattering::Cell {mass:std::array::from_fn(|s|mass[s][i]),
            number:std::array::from_fn(|s|number[s][i]),volume:[volume[0][i],volume[1][i]],
            rime:rime[i],rime_volume:rime_volume[i],density,temperature:temperature[i]};
        if scheme==8 {cell.number[3]=scattering::thompson_graupel_number(cell.mass[3],density);}
        scattering::integrate(table,cell,scheme,p3)
    }).collect::<Result<_,String>>()?;
    let mass_error=evaluated.iter().map(|s|s.mass_error).fold(0f32,f32::max);
    let clipped=evaluated.iter().map(|s|s.clipped_mass_fraction).fold(0f32,f32::max);
    let dbz=evaluated.iter().map(|s|{
        let z=s.moments[0].zh;
        if z>0. {crate::wrf_radar_physics::z_to_dbz(z)} else {-40.}
    }).collect();
    let present:Vec<String>=MICROPHYSICS_FIELD_NAMES.iter().filter(|v|file.has_var(v)).map(|v|v.to_string()).collect();
    let mut profile=crate::wrf_radar_physics::detect_scheme(Some(scheme),&present);
    profile.capability=crate::wrf_radar_physics::MicrophysicsCapability::FullTwoMoment;
    profile.assumption_heavy=false;
    let mut compact=CompactPolarFields::new(expected,profile,present);
    compact.band_moments=Some(evaluated.into_iter().map(|s|s.moments).collect());
    file.clear_cache();
    Ok((compact,dbz,format!("{}; band={band}; wavelength_mm={}; full-precision additive moments; weighted melting; max PSD mass error={mass_error}; max clamped mass fraction={clipped}",scattering::MODEL_ID,table.wavelength_mm)))
}

fn read_optional_first_var_f32(
    file: &WrfFile,
    timeidx: usize,
    names: &[&str],
    expected: usize,
) -> Result<Option<Vec<f32>>, String> {
    let Some(name) = first_existing_var(file, names) else {
        return Ok(None);
    };
    let values = file
        .read_var(name, timeidx)
        .map_err(|error| format!("dual-pol {name}: {error}"))?;
    if values.len() != expected {
        return Err(format!(
            "dual-pol {name} has {} values, expected {expected}",
            values.len()
        ));
    }
    Ok(Some(values.into_iter().map(|value| value as f32).collect()))
}

fn read_compact_tke(file: &WrfFile, timeidx: usize, expected: usize) -> Option<Vec<u8>> {
    let name = first_existing_var(file, &["TKE_PBL", "TKE", "QKE"])?;
    let values = file.read_var(name, timeidx).ok()?;
    if values.len() != expected {
        return None;
    }
    let qke_halving = if name == "QKE" { 0.5 } else { 1.0 };
    Some(
        values
            .into_iter()
            .map(|value| quantize_u8((value as f32 * qke_halving).max(0.0), 0.1))
            .collect(),
    )
}

/// Earth-relative winds. `uvmet` returns `[u_earth.., v_earth..]`
/// (2 * nz * cells); fall back to grid-relative `ua`/`va` if unavailable.
/// Extracted verbatim from the original inline logic so the values match.
fn read_earth_relative_winds(
    file: &WrfFile,
    timeidx: usize,
    expected: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    match getvar(file, "uvmet", Some(timeidx), &ComputeOpts::default()) {
        Ok(uvmet) if uvmet.data.len() == 2 * expected => {
            let (ue, ve) = uvmet.data.split_at(expected);
            Ok((to_f32(ue), to_f32(ve)))
        }
        _ => {
            let ua = read_3d(file, "ua", timeidx, expected)?;
            let va = read_3d(file, "va", timeidx, expected)?;
            Ok((ua, va))
        }
    }
}

fn read_terrain_m(file: &WrfFile, timeidx: usize, cells: usize) -> Vec<f32> {
    file.terrain(timeidx)
        .map(|ter| ter.iter().map(|value| *value as f32).collect::<Vec<_>>())
        .unwrap_or_else(|_| vec![0.0; cells])
}

/// Join a scoped read thread, turning a thread panic into a readable error.
fn join_read<T>(
    handle: std::thread::ScopedJoinHandle<'_, Result<T, String>>,
    what: &str,
) -> Result<T, String> {
    match handle.join() {
        Ok(inner) => inner,
        Err(_) => Err(format!("WRF {what} read thread panicked")),
    }
}

fn read_reflectivity(
    file: &WrfFile,
    timeidx: usize,
    expected: usize,
    operator: ReflectivityOperator,
) -> Result<(Vec<f32>, &'static str), String> {
    // The same decision the tests assert through `planned_ref_source`: only the
    // model-native operator reads REFL_10CM, and only when the file carries it.
    if planned_ref_source(operator, file.has_var("REFL_10CM")) == REFL_10CM_SOURCE
        && let Ok(raw) = file.read_var("REFL_10CM", timeidx)
        && raw.len() == expected
    {
        return Ok((to_f32(&raw), REFL_10CM_SOURCE));
    }
    // wrf-core `dbz` = CALCDBZ (Stoelinga 2005), the same source BowEcho's
    // composite reflectivity uses. Constant intercepts / no bright-band
    // correction (ComputeOpts default) to match that composite exactly.
    // The classic-Stoelinga operator forces this path even when the file
    // carries REFL_10CM; the source label records which case applied.
    let dbz = read_3d(file, "dbz", timeidx, expected)
        .map_err(|err| format!("no REFL_10CM and computed dbz failed: {err}"))?;
    Ok((dbz, operator.computed_dbz_source()))
}

fn read_3d(
    file: &WrfFile,
    name: &str,
    timeidx: usize,
    expected: usize,
) -> Result<Vec<f32>, String> {
    let out = getvar(file, name, Some(timeidx), &ComputeOpts::default())
        .map_err(|err| format!("read WRF {name}: {err}"))?;
    if out.data.len() != expected {
        return Err(format!(
            "WRF {name} has {} values, expected {expected}",
            out.data.len()
        ));
    }
    Ok(to_f32(&out.data))
}

fn to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

/// Build one synthetic [`RadarVolume`] from pre-read [`WrfRadarFields`].
///
/// `valid_time` is the volume's scan time (the WRF forecast valid time), which
/// keys the frame in a loop.
pub fn build_synthetic_volume(
    fields: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
) -> RadarVolume {
    build_synthetic_volume_reporting(fields, valid_time, config, &|_| {})
}

/// As [`build_synthetic_volume`], but streams a per-tilt progress label so the
/// UI shows "building tilt k/n…" instead of freezing while the polar volume is
/// traced. The per-tilt work itself is unchanged.
pub fn build_synthetic_volume_reporting(
    fields: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
) -> RadarVolume {
    try_build_synthetic_volume_reporting(fields, valid_time, config, progress)
        .expect("validated single-scene synthetic-radar render")
}

/// Exact-geometry replay products returned together so the caller cannot
/// accidentally compare against a separately regridded synthetic volume.
pub struct ExactReplayProducts {
    pub observed: Arc<RadarVolume>,
    pub simulated: RadarVolume,
    pub difference: RadarVolume,
    pub unavailable_observed_moments: Vec<UnavailableObservedMoment>,
}

fn is_dual_pol_replay_moment(moment: &MomentType) -> bool {
    matches!(
        moment,
        MomentType::DifferentialReflectivity
            | MomentType::CorrelationCoefficient
            | MomentType::DifferentialPhase
            | MomentType::SpecificDifferentialPhase
    )
}

fn normalize_replay_config_for_observed(config: &mut SyntheticRadarConfig, observed: &RadarVolume) {
    // The explicit Replay action intentionally replaces custom instrument
    // ambiguity with source radial Nyquist. Direct hand-built contradictory
    // configs still fail closed in `validate_science_contract`.
    config.coupled_single_prf_estimator = false;
    config.emit_stage_diagnostics = false;
    config.fold_velocity = false;
    if observed
        .cuts
        .iter()
        .any(|cut| cut.moments.contains_key(&MomentType::SpectrumWidth))
    {
        config.spectrum_width = true;
    }
    if observed
        .cuts
        .iter()
        .flat_map(|cut| cut.moments.keys())
        .any(is_dual_pol_replay_moment)
    {
        config.dual_pol = true;
    }
}

/// Render the WRF atmosphere on an observed scan's exact acquisition geometry
/// and return both the simulated and `simulated - observed` volumes. This is a
/// scan replay, not a reconstruction of the numbered VCP carried by the file.
pub fn build_exact_replay_products(
    fields: &WrfRadarFields,
    observed: Arc<RadarVolume>,
    config: &SyntheticRadarConfig,
) -> Result<ExactReplayProducts, String> {
    let template = ExactScanTemplate::from_volume(&observed)
        .map_err(|error| format!("extract exact observed scan template: {error}"))?;
    let mut replay_config = config.clone();
    normalize_replay_config_for_observed(&mut replay_config, &observed);
    replay_config.exact_replay_template = Some(Arc::new(template));
    let mut simulated = try_build_synthetic_volume_reporting(
        fields,
        observed.volume_time,
        &replay_config,
        &|_| {},
    )?;
    copy_observed_ray_instrument_metadata(&observed, &mut simulated)?;
    let overlap = build_difference_volume_overlap(&observed, &simulated)
        .map_err(|error| format!("build exact-geometry replay difference: {error}"))?;
    Ok(ExactReplayProducts {
        observed,
        simulated,
        difference: overlap.volume,
        unavailable_observed_moments: overlap.unavailable_observed_moments,
    })
}

fn copy_observed_ray_instrument_metadata(
    observed: &RadarVolume,
    simulated: &mut RadarVolume,
) -> Result<(), String> {
    if observed.cuts.len() != simulated.cuts.len() {
        return Err(format!(
            "exact replay has {} observed cuts but {} simulated cuts while copying per-ray instrument metadata",
            observed.cuts.len(),
            simulated.cuts.len(),
        ));
    }
    for (cut_index, (simulated_cut, observed_cut)) in
        simulated.cuts.iter_mut().zip(&observed.cuts).enumerate()
    {
        let metadata = observed_cut
            .aligned_ray_instrument_metadata()
            .map_err(|error| {
                format!(
                    "exact replay observed cut {} has invalid per-ray instrument metadata: {error}",
                    cut_index + 1
                )
            })?;
        if let Some(metadata) = metadata {
            if metadata.len() != simulated_cut.radials.len() {
                return Err(format!(
                    "exact replay cut {} per-ray instrument metadata has {} entries for {} simulated radials",
                    cut_index + 1,
                    metadata.len(),
                    simulated_cut.radials.len(),
                ));
            }
            simulated_cut.ray_instrument_metadata = metadata.to_vec();
        }
    }
    Ok(())
}

/// Background form of [`build_exact_replay_products`]. The observed volume is
/// retained by `Arc` through completion and returned in the product bundle.
pub fn spawn_exact_replay_worker(
    fields: Arc<WrfRadarFields>,
    observed: Arc<RadarVolume>,
    config: SyntheticRadarConfig,
) -> Receiver<Result<ExactReplayProducts, String>> {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("wrf-exact-radar-replay".to_string())
        .spawn(move || {
            let result = build_exact_replay_products(&fields, observed, &config);
            let _ = tx.send(result);
        })
        .expect("spawn exact WRF radar replay worker");
    rx
}

fn try_build_synthetic_volume_reporting(
    fields: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
) -> Result<RadarVolume, String> {
    build_synthetic_volume_reporting_inner(fields, valid_time, config, progress, None, None, None)
}

fn try_build_synthetic_volume_reporting_cancellable(
    fields: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    cancel: &AtomicBool,
) -> Result<RadarVolume, String> {
    build_synthetic_volume_reporting_inner(
        fields,
        valid_time,
        config,
        progress,
        None,
        None,
        Some(cancel),
    )
}

fn try_build_synthetic_volume_reporting_previewing(
    fields: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    cut_completed: &dyn Fn(&RadarVolume, usize, usize),
) -> Result<RadarVolume, String> {
    build_synthetic_volume_reporting_inner(
        fields,
        valid_time,
        config,
        progress,
        None,
        Some(cut_completed),
        None,
    )
}

fn try_build_synthetic_volume_reporting_previewing_cancellable(
    fields: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    cut_completed: &dyn Fn(&RadarVolume, usize, usize),
    cancel: &AtomicBool,
) -> Result<RadarVolume, String> {
    build_synthetic_volume_reporting_inner(
        fields,
        valid_time,
        config,
        progress,
        None,
        Some(cut_completed),
        Some(cancel),
    )
}

struct TemporalRenderContext<'a> {
    neighbor: &'a WrfRadarFields,
    plan: &'a TemporalScenePlan,
}

pub fn build_synthetic_volume_reporting_temporal(
    fields: &WrfRadarFields,
    neighbor: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    plan: &TemporalScenePlan,
) -> Result<RadarVolume, String> {
    build_synthetic_volume_reporting_inner(
        fields,
        valid_time,
        config,
        progress,
        Some(TemporalRenderContext { neighbor, plan }),
        None,
        None,
    )
}

fn build_synthetic_volume_reporting_temporal_previewing(
    fields: &WrfRadarFields,
    neighbor: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    plan: &TemporalScenePlan,
    cut_completed: &dyn Fn(&RadarVolume, usize, usize),
) -> Result<RadarVolume, String> {
    build_synthetic_volume_reporting_inner(
        fields,
        valid_time,
        config,
        progress,
        Some(TemporalRenderContext { neighbor, plan }),
        Some(cut_completed),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_synthetic_volume_reporting_temporal_previewing_cancellable(
    fields: &WrfRadarFields,
    neighbor: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    plan: &TemporalScenePlan,
    cut_completed: &dyn Fn(&RadarVolume, usize, usize),
    cancel: &AtomicBool,
) -> Result<RadarVolume, String> {
    build_synthetic_volume_reporting_inner(
        fields,
        valid_time,
        config,
        progress,
        Some(TemporalRenderContext { neighbor, plan }),
        Some(cut_completed),
        Some(cancel),
    )
}

type CutCompletedCallback<'a> = dyn Fn(&RadarVolume, usize, usize) + 'a;

fn build_synthetic_volume_reporting_inner(
    fields: &WrfRadarFields,
    valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    temporal: Option<TemporalRenderContext<'_>>,
    cut_completed: Option<&CutCompletedCallback<'_>>,
    cancel: Option<&AtomicBool>,
) -> Result<RadarVolume, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    config.validate_science_contract()?;
    if let Some(template) = config.exact_replay_template.as_deref() {
        return build_exact_replay_volume(
            fields,
            valid_time,
            config,
            template,
            progress,
            temporal.as_ref(),
            cancel,
        );
    }
    let coupled_instrument = resolve_coupled_instrument(config)?;
    let cells = fields.cells();
    let (site_lat, site_lon, antenna_msl) = resolve_radar_site(fields, config)?;

    let naz = config.azimuth_count.max(1);
    // Match-gate-to-grid resolves against the file's DX (carried on `fields`);
    // off, this is the configured `gate_spacing_m`, so the build is unchanged.
    let spacing = effective_gate_spacing(config, fields.dx_m).max(1.0);
    let gate_count = ((config.max_range_m / spacing).floor() as usize).max(1);
    let gate_range = GateRange {
        first_gate_m: 0,
        gate_spacing_m: spacing.round() as i32,
        gate_count,
    };
    let beam_geometry =
        BeamPropagationGeometry::resolve(fields, config, site_lat, site_lon, antenna_msl)?;

    let mut site = RadarSite::new(config.site_id.clone());
    site.name = config.site_name.clone();
    site.latitude_deg = Some(site_lat as f32);
    site.longitude_deg = Some(site_lon as f32);
    site.elevation_m = Some(antenna_msl as f32);

    let mut volume = RadarVolume::new(site, valid_time);
    let scan_legs = config.physical_scan_legs();
    let microphysics_inventory = fields
        .property_scattering
        .as_ref()
        .map(|scene| {
            scene
                .source_fields()
                .iter()
                .map(app_ui::wrf_property_reader::SourceFieldProvenance::source_name)
                .collect::<Vec<_>>()
                .join(",")
        })
        .or_else(|| {
            fields.raw_property_scene.as_ref().map(|scene| {
                scene
                    .source_fields()
                    .iter()
                    .map(app_ui::wrf_property_reader::SourceFieldProvenance::source_name)
                    .collect::<Vec<_>>()
                    .join(",")
            })
        })
        .or_else(|| {
            fields
                .polarimetric
                .as_ref()
                .map(|polar| polar.present_fields.join(","))
        })
        .unwrap_or_default();
    let named_vcp = config.scan_strategy.definition();
    if let Some(definition) = named_vcp {
        volume.vcp = Some(VcpInfo {
            pattern: definition.vcp.number(),
        });
    }
    let mut forward_operator_config = format!(
        "mode={:?}; reflectivity_sampling={:?}; beam_integration={:?}; \
         fall_speed={}; terrain_blockage={}; spectrum_width={}; dual_pol={}; polarimetric_kernel={:?}; \
         propagation={}; quality_fields={}; minimum_model_coverage_fraction={:.4}; microphysics_fields={microphysics_inventory}",
        config.simulation_mode,
        config.reflectivity_sampling,
        config.beam_integration,
        config.terminal_fall_speed,
        config.terrain_blockage,
        config.spectrum_width,
        config.dual_pol,
        config.polarimetric_kernel,
        config.propagation,
        config.emit_quality_fields,
        config.clamped_minimum_model_coverage_fraction(),
    );
    if config.polarimetric_kernel.is_property_tmatrix() {
        use std::fmt::Write as _;
        let _ = write!(
            forward_operator_config,
            "; tmatrix_scattering_policy={}",
            config.polarimetric_kernel.scattering_policy().identifier(),
        );
        if let Some(scene) = fields.property_scattering.as_ref() {
            let counts = scene.provenance().counts;
            let _ = write!(
                forward_operator_config,
                "; tmatrix_native_psd_populations={}; tmatrix_hybrid_bulk_cells={}; tmatrix_hybrid_bulk_populations={}",
                counts.scheme_native_psd_populations,
                counts.hybrid_bulk_rayleigh_cells,
                counts.hybrid_bulk_rayleigh_populations,
            );
        }
        if let Some(service) = config.runtime_cuda_service.as_deref() {
            let report = service.report();
            if let Some(reason) = report.fallback_reason.as_ref() {
                let _ = write!(
                    forward_operator_config,
                    "; tmatrix_compute_preference={}; tmatrix_execution_initial_backend=cuda_then_cpu_replay; cuda_device={}; cuda_compute={}.{}; cuda_kernel={}; cuda_nodes_completed={}; cuda_fallback={reason}",
                    config.compute_preference.label(),
                    report.device.name,
                    report.device.compute_capability_major,
                    report.device.compute_capability_minor,
                    report.device.kernel_artifact,
                    report.nodes_completed,
                );
            } else {
                let _ = write!(
                    forward_operator_config,
                    "; tmatrix_compute_preference={}; tmatrix_execution_initial_backend=nvidia_cuda; cuda_device={}; cuda_compute={}.{}; cuda_kernel={}",
                    config.compute_preference.label(),
                    report.device.name,
                    report.device.compute_capability_major,
                    report.device.compute_capability_minor,
                    report.device.kernel_artifact,
                );
            }
        } else {
            let _ = write!(
                forward_operator_config,
                "; tmatrix_compute_preference={}; tmatrix_execution_initial_backend=cpu_reference",
                config.compute_preference.label(),
            );
        }
    }
    if let Some(polar) = fields.polarimetric.as_ref() {
        use std::fmt::Write as _;
        let audit = &polar.precision_audit;
        let _ = write!(
            forward_operator_config,
            "; compact_dual_pol_audit={}; compact_dual_pol_clamps={}; compact_dual_pol_quantized_to_zero={}; compact_dual_pol_max_zv_relative_error={:.6}; compact_dual_pol_max_covariance_relative_error={:.6}; compact_dual_pol_max_av_error_db_km={:.6}; compact_dual_pol_max_fall_variance_error_m2s2={:.6}",
            audit.provenance_fragment(),
            audit.total_clamps(),
            audit.total_quantized_to_zero(),
            audit.max_zv_relative_error,
            audit.max_covariance_magnitude_relative_error,
            audit.max_av_abs_error_db_km,
            audit.max_fall_variance_abs_error_m2s2,
        );
    }
    if let Some(coupled) = coupled_instrument.as_ref() {
        use std::fmt::Write as _;
        let _ = write!(
            forward_operator_config,
            "; estimator=CustomSinglePrfV1; frequency_hz={:.3}; wavelength_m={:.12}; \
             prf_hz={:.6}; prt_s={:.9}; nyquist_mps={:.6}; unambiguous_range_km={:.6}; \
             pulse_range_response=matched_filter_triangular; pulse_range_resolution_m={:.6}; \
             dwell_ms={:.3}; transmitted_pulses={}; independent_samples={:.3}; \
             minimum_snr_db={:.3}; velocity_folding=always_prf_derived; \
             stage_diagnostics={}; stage_definitions={} | {} | {}",
            coupled.timing.frequency_hz,
            coupled.timing.wavelength_m,
            coupled.timing.prf_hz,
            coupled.timing.prt_s,
            coupled.timing.nyquist_velocity_mps,
            coupled.timing.unambiguous_range_m / 1_000.0,
            coupled.range_resolution_m,
            coupled.sampling.dwell_s * 1_000.0,
            coupled.sampling.transmitted_pulses,
            coupled.sampling.independent_samples,
            coupled.estimator_config.minimum_snr_db,
            config.emit_stage_diagnostics,
            IDEAL_STAGE_DEFINITION,
            MEASURED_STAGE_DEFINITION,
            PRESENTED_STAGE_DEFINITION,
        );
    }
    if let Some(definition) = named_vcp {
        use std::fmt::Write as _;
        let _ = write!(
            forward_operator_config,
            "; scan_strategy=Build24Vcp{}; vcp_source={} rev {}; physical_rows={}; config_fingerprint={:016x}",
            definition.vcp.number(),
            definition.source.document_number,
            definition.source.revision,
            definition.rows.len(),
            config.data_fingerprint(),
        );
    }
    if let Some(scene) = fields.property_scattering.as_deref() {
        use std::fmt::Write as _;
        let provenance = scene.provenance();
        let tables = provenance
            .tables
            .iter()
            .map(|table| format!("{}={}@{}", table.role, table.table_id, table.file_sha256))
            .collect::<Vec<_>>()
            .join(",");
        let counts = provenance.counts;
        let _ = write!(
            forward_operator_config,
            "; tmatrix_status={}; tmatrix_frequency_hz={:.0}; tmatrix_orientation={:?}; \
             tmatrix_rain_mode={:?}; tmatrix_fall_policy_closed={:?}; \
             tmatrix_fall_policy_wet={:?}; tmatrix_tables={tables}; \
             tmatrix_populations=source_cells:{},characteristic_frozen:{},scheme_native_psd:{},dry_frozen:{},wet_frozen:{},residual_rain:{},standalone_rain:{}; \
             tmatrix_interpolation=source_cell_closure_then_additive_space_beam_and_time",
            provenance.status,
            provenance.frequency_hz,
            provenance.orientation,
            provenance.rain_mode,
            provenance.fall_moment_policy.closed_category,
            provenance.fall_moment_policy.diagnostic_wet_category,
            counts.source_cells,
            counts.characteristic_frozen_populations,
            counts.scheme_native_psd_populations,
            counts.dry_frozen_populations,
            counts.wet_frozen_populations,
            counts.residual_rain_populations,
            counts.standalone_rain_populations,
        );
        if let Some(identity) = fields.property_table_identity.as_ref() {
            let _ = write!(
                forward_operator_config,
                "; tmatrix_table_source={:?}; tmatrix_pack_id={}; tmatrix_pack_band={:?}; tmatrix_pack_exact_frequency_hz={:.0}; tmatrix_rain_sensitivity={:?}",
                config.property_tmatrix_table_source,
                identity.pack_id(),
                identity.band(),
                identity.exact_frequency_hz(),
                config.property_tmatrix_rain_sensitivity,
            );
        }
    }
    if let Some(scene) = fields.raw_property_scene.as_deref() {
        use std::fmt::Write as _;
        let _ = write!(
            forward_operator_config,
            "; tmatrix_status=research_only_unvalidated; tmatrix_frequency_hz={:.0}; \
             tmatrix_table_source={:?}; tmatrix_pack_id={}; tmatrix_rain_sensitivity={:?}; \
             tmatrix_raw_mp_physics={}; tmatrix_raw_source={}:time{}; \
             tmatrix_interpolation=raw_state_linear_spatial_and_temporal_preclosure_then_single_scattering_evaluation",
            f64::from(config.radar_frequency_mhz) * 1.0e6,
            config.property_tmatrix_table_source,
            fields
                .property_table_identity
                .as_ref()
                .map_or("unresolved", |identity| identity.pack_id()),
            config.property_tmatrix_rain_sensitivity,
            scene.microphysics_scheme_id(),
            scene.identity().source_identity.0.as_str(),
            scene.identity().time_index,
        );
    }
    if let Some(source_provenance) = fields.source_provenance_fragment.as_deref() {
        forward_operator_config.push_str("; ");
        forward_operator_config.push_str(source_provenance);
    }
    append_beam_geometry_provenance(
        &mut forward_operator_config,
        fields,
        config.propagation_geometry,
    );
    let source_model_label = fields.source_model_override.as_deref().unwrap_or("WRF");
    let archive_version = fields.source_model_override.as_ref().map_or_else(
        || "simulated-wrf".to_string(),
        |source| format!("simulated-{}", source.to_ascii_lowercase()),
    );
    volume.metadata = VolumeMetadata {
        source_path: None,
        archive_version: Some(archive_version),
        compression: None,
        message_count: 0,
        decoded_radial_count: 0,
        skipped_message_count: 0,
        scan_mode: Some(ScanMode::Ppi),
        radar_frequency_mhz: Some(config.radar_frequency_mhz),
        beam_width_h_deg: Some(config.beam_width_deg),
        beam_width_v_deg: Some(config.beam_width_deg),
        pulse_width_us: Some(config.pulse_width_us),
        // Appendix C supplies numbered PRF codes, not Hz. Never turn those
        // codes into a fictitious standard PRT/unambiguous range.
        prt_s: coupled_instrument.as_ref().map_or_else(
            || {
                named_vcp
                    .is_none()
                    .then(|| {
                        (config.prf_hz.is_finite() && config.prf_hz > 0.0)
                            .then_some(1.0 / config.prf_hz)
                    })
                    .flatten()
            },
            |coupled| Some(coupled.timing.prt_s as f32),
        ),
        unambiguous_range_km: coupled_instrument.as_ref().map_or_else(
            || {
                named_vcp
                    .is_none()
                    .then(|| {
                        (config.prf_hz.is_finite() && config.prf_hz > 0.0)
                            .then_some(299_792.47 / (2.0 * config.prf_hz))
                    })
                    .flatten()
            },
            |coupled| Some((coupled.timing.unambiguous_range_m / 1_000.0) as f32),
        ),
        scan_name: Some(if let Some(definition) = named_vcp {
            format!("Build 24 VCP {} baseline", definition.vcp.number())
        } else {
            match config.scan_timing {
                ScanTiming::InstantaneousTruth => "BowEcho instantaneous model volume".to_string(),
                ScanTiming::TimedVolume => "BowEcho timed synthetic volume".to_string(),
            }
        }),
        scan_id: Some(if let Some(definition) = named_vcp {
            format!(
                "bowecho-build24-vcp{}-{}-rev{}-{}rows",
                definition.vcp.number(),
                BUILD_24_SOURCE.document_number,
                BUILD_24_SOURCE.revision,
                definition.rows.len(),
            )
        } else {
            "bowecho-synthetic-14-cut".to_string()
        }),
        vcp_source_document: named_vcp
            .map(|definition| definition.source.document_number.to_owned()),
        vcp_source_revision: named_vcp.map(|definition| definition.source.revision.to_owned()),
        vcp_source_rda_build: named_vcp.map(|definition| definition.source.rda_build.to_owned()),
        vcp_source_figure: named_vcp.map(|definition| definition.source_figure.to_owned()),
        vcp_pulse_length: named_vcp.map(|definition| match definition.pulse_length {
            PulseLength::Short => "short".to_owned(),
            PulseLength::Long => "long".to_owned(),
        }),
        vcp_adaptations: named_vcp.map(|_| BUILD_24_NO_ADAPTATIONS_CAVEAT.to_owned()),
        scan_legs: named_vcp
            .map(|_| scan_legs.iter().map(scan_leg_metadata).collect())
            .unwrap_or_default(),
        polarization: Some(if config.dual_pol && fields.has_polarimetric_input() {
            "simultaneous horizontal/vertical".to_string()
        } else {
            "horizontal".to_string()
        }),
        calibration: Some(format!(
            "ZDR bias={:.2} dB; system PhiDP={:.2} deg",
            config.zdr_bias_db, config.system_phidp_deg
        )),
        forward_operator: Some(if coupled_instrument.is_some() {
            format!(
                "BowEcho {source_model_label} polar-volume forward operator v4 (coupled single-PRF estimator)"
            )
        } else {
            format!("BowEcho {source_model_label} polar-volume forward operator v3")
        }),
        forward_operator_config: Some(forward_operator_config),
        source_model: fields
            .source_model_override
            .clone()
            .or_else(|| Some("WRF".to_string())),
        microphysics_scheme: fields.source_microphysics_override.clone().or_else(|| {
            fields
                .property_scattering
                .as_ref()
                .map(|scene| format!("WRF mp_physics={}", scene.microphysics_scheme_id()))
                .or_else(|| {
                    fields
                        .raw_property_scene
                        .as_ref()
                        .map(|scene| format!("WRF mp_physics={}", scene.microphysics_scheme_id()))
                })
                .or_else(|| {
                    fields.polarimetric.as_ref().map(|polar| {
                        format!("{} ({:?})", polar.profile.name, polar.profile.capability)
                    })
                })
        }),
        scattering_model: if config.dual_pol && fields.has_polarimetric_input() {
            fields.source_scattering_override.clone().or_else(|| Some(match config.polarimetric_kernel {
                PolarimetricKernel::BulkRayleighV1 => format!(
                    "{} (scheme-aware bulk S-band Rayleigh; not T-matrix)",
                    crate::wrf_radar_physics::bulk_sband_model_id()
                ),
                PolarimetricKernel::PropertyTMatrixHybridV1 => {
                    let identity = fields
                        .property_table_identity
                        .as_ref()
                        .map(|identity| identity.pack_id())
                        .unwrap_or("unresolved");
                    let counts = fields
                        .property_scattering
                        .as_ref()
                        .map(|scene| scene.provenance().counts)
                        .unwrap_or_default();
                    format!(
                        "P3 Hybrid ({identity}; {} MHz; {:?}; policy={}; native_psd_populations={}; hybrid_bulk_cells={}; hybrid_bulk_populations={}; research_only_unvalidated)",
                        config.radar_frequency_mhz,
                        config.property_tmatrix_rain_sensitivity,
                        app_ui::wrf_tmatrix_scene::WrfTMatrixScatteringPolicy::HybridBulkRayleighV1.identifier(),
                        counts.scheme_native_psd_populations,
                        counts.hybrid_bulk_rayleigh_cells,
                        counts.hybrid_bulk_rayleigh_populations,
                    )
                }
                PolarimetricKernel::PropertyTMatrixResearchV1 => {
                    let identity = fields
                        .property_table_identity
                        .as_ref()
                        .map(|identity| identity.pack_id())
                        .unwrap_or("unresolved");
                    format!(
                        "Full P3/ISHMAEL property-aware PSD T-matrix ({identity}; {} MHz; {:?}; experimental; strict_fail_closed; research_only_unvalidated)",
                        config.radar_frequency_mhz,
                        config.property_tmatrix_rain_sensitivity,
                    )
                }
            }))
        } else {
            None
        },
    };

    // One deterministic seed per forecast frame, folded into every clutter
    // draw so the same hour rebuilds bit-identically (no loop shimmer) while
    // distinct frames get a distinct clutter pattern (the community operator's
    // per-time-step variation). Site id + valid time; the tilt index is mixed
    // in per cut. Cheap to compute even when clutter is off (unused then).
    let frame_seed = clutter_frame_seed(&config.site_id, valid_time);

    let terrain_horizon =
        (config.terrain_blockage && !beam_geometry.is_wrf_refractivity()).then(|| {
            progress("tracing terrain horizon…");
            TerrainHorizon::build(
                fields,
                site_lat,
                site_lon,
                antenna_msl,
                naz,
                gate_count,
                0,
                spacing,
            )
        });

    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }

    let mut decoded_radials = 0usize;
    let mut cut_start_ms = 0i32;
    let tilt_total = scan_legs.len();
    let mut cuda_fallback_reported = false;
    for (cut_index, leg) in scan_legs.iter().enumerate() {
        if let Some(cancel) = cancel {
            check_synthetic_radar_cancel(cancel)?;
        }
        if !cuda_fallback_reported
            && let Some(reason) = config
                .runtime_cuda_service
                .as_deref()
                .and_then(|service| service.report().fallback_reason)
        {
            progress(&format!(
                "CUDA disabled ({reason}); replaying affected and remaining T-matrix work on CPU"
            ));
            cuda_fallback_reported = true;
        }
        let elevation_deg = leg.elevation_deg;
        progress(&format!(
            "building tilt {}/{tilt_total} ({elevation_deg:.1}°)…",
            cut_index + 1
        ));
        let cut = build_cut(
            fields,
            temporal.as_ref(),
            cells,
            site_lat,
            site_lon,
            antenna_msl,
            elevation_deg,
            cut_index,
            naz,
            spacing,
            gate_range.clone(),
            beam_geometry,
            config,
            leg,
            tilt_total,
            frame_seed,
            terrain_horizon.as_ref(),
            cut_start_ms,
            coupled_instrument.as_ref(),
            None,
            cancel,
        )?;
        decoded_radials += cut.radials.len();
        volume.cuts.push(cut);
        volume.metadata.decoded_radial_count = decoded_radials;
        if !cuda_fallback_reported
            && let Some(reason) = config
                .runtime_cuda_service
                .as_deref()
                .and_then(|service| service.report().fallback_reason)
        {
            progress(&format!(
                "CUDA disabled ({reason}); replaying affected and remaining T-matrix work on CPU"
            ));
            cuda_fallback_reported = true;
        }
        refresh_in_progress_tmatrix_execution_provenance(&mut volume, config);
        if let Some(cut_completed) = cut_completed {
            cut_completed(&volume, cut_index + 1, tilt_total);
        }
        if let Some(cancel) = cancel {
            check_synthetic_radar_cancel(cancel)?;
        }
        if matches!(config.scan_timing, ScanTiming::TimedVolume) {
            cut_start_ms = advance_cut_start_ms(config, leg, cut_start_ms);
        }
    }
    volume.metadata.decoded_radial_count = decoded_radials;
    Ok(volume)
}

fn build_exact_replay_volume(
    fields: &WrfRadarFields,
    model_valid_time: DateTime<Utc>,
    config: &SyntheticRadarConfig,
    template: &ExactScanTemplate,
    progress: &dyn Fn(&str),
    temporal: Option<&TemporalRenderContext<'_>>,
    cancel: Option<&AtomicBool>,
) -> Result<RadarVolume, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let cells = fields.cells();
    let site_lat = f64::from(
        template
            .site
            .latitude_deg
            .ok_or_else(|| format!("exact replay site {} has no latitude", template.site.id))?,
    );
    let site_lon = f64::from(
        template
            .site
            .longitude_deg
            .ok_or_else(|| format!("exact replay site {} has no longitude", template.site.id))?,
    );
    let center = fields.center_cell();
    let antenna_msl = template.site.elevation_m.map(f64::from).unwrap_or_else(|| {
        let site_cell = fields
            .lut
            .lookup(site_lat as f32, site_lon as f32)
            .unwrap_or(center);
        f64::from(fields.terrain_m[site_cell]) + DEFAULT_TOWER_M
    });
    let geometry_fingerprint = template.geometry_fingerprint();
    let mut volume = RadarVolume::new(template.site.clone(), template.volume_time);
    volume.vcp = template.vcp.clone();
    volume.metadata = VolumeMetadata {
        archive_version: Some("simulated-wrf-exact-observed-replay-v1".to_string()),
        decoded_radial_count: template.cuts.iter().map(|cut| cut.rays.len()).sum(),
        scan_mode: Some(ScanMode::Ppi),
        radar_frequency_mhz: Some(config.radar_frequency_mhz),
        beam_width_h_deg: Some(config.beam_width_deg),
        beam_width_v_deg: Some(config.beam_width_deg),
        pulse_width_us: Some(config.pulse_width_us),
        prt_s: template.source_prt_s,
        unambiguous_range_km: template.source_unambiguous_range_km,
        scan_name: Some("Exact observed scan-geometry replay (not a VCP reconstruction)".into()),
        scan_id: Some(format!("bowecho-exact-replay-{geometry_fingerprint:016x}")),
        polarization: Some(if config.dual_pol && fields.has_polarimetric_input() {
            "simultaneous horizontal/vertical".to_string()
        } else {
            "horizontal".to_string()
        }),
        calibration: Some(format!(
            "ZDR bias={:.2} dB; system PhiDP={:.2} deg",
            config.zdr_bias_db, config.system_phidp_deg
        )),
        forward_operator: Some("BowEcho WRF exact observed-geometry replay v1".to_string()),
        forward_operator_config: Some(format!(
            "scan_source=exact_observed_geometry; geometry_fingerprint={geometry_fingerprint:016x}; \
             vcp_reconstruction=false; cut_order=source; split_cuts=preserved; missing_sectors=preserved; \
             radial_geometry=source; moment_availability=source; acquisition_offsets=source_normalized_utc; \
             radial_nyquist=source; volume_prt=source; source_time_encoding={:?}; \
             model_valid_time={}; config_fingerprint={:016x}",
            template.time_encoding,
            model_valid_time.to_rfc3339(),
            config.data_fingerprint(),
        )),
        source_model: Some("WRF".to_string()),
        microphysics_scheme: fields
            .property_scattering
            .as_ref()
            .map(|scene| format!("WRF mp_physics={}", scene.microphysics_scheme_id()))
            .or_else(|| {
                fields
                    .raw_property_scene
                    .as_ref()
                    .map(|scene| format!("WRF mp_physics={}", scene.microphysics_scheme_id()))
            })
            .or_else(|| {
                fields
                    .polarimetric
                    .as_ref()
                    .map(|polar| format!("{} ({:?})", polar.profile.name, polar.profile.capability))
            }),
        ..VolumeMetadata::default()
    };
    if let Some(provenance) = volume.metadata.forward_operator_config.as_mut() {
        if let Some(identity) = fields.property_table_identity.as_ref() {
            use std::fmt::Write as _;
            let _ = write!(
                provenance,
                "; tmatrix_table_source={:?}; tmatrix_pack_id={}; tmatrix_pack_band={:?}; tmatrix_pack_exact_frequency_hz={:.0}; tmatrix_rain_sensitivity={:?}",
                config.property_tmatrix_table_source,
                identity.pack_id(),
                identity.band(),
                identity.exact_frequency_hz(),
                config.property_tmatrix_rain_sensitivity,
            );
            volume.metadata.scattering_model = Some(format!(
                "Property-aware PSD T-matrix ({}; {} MHz; {:?}; research_only_unvalidated)",
                identity.pack_id(),
                config.radar_frequency_mhz,
                config.property_tmatrix_rain_sensitivity,
            ));
        }
        append_beam_geometry_provenance(provenance, fields, config.propagation_geometry);
    }

    let frame_seed = clutter_frame_seed(&template.site.id, model_valid_time);
    let mut render_config = config.clone();
    render_config.exact_replay_template = None;
    render_config.scan_strategy = SyntheticScanStrategy::CustomLegacy;
    render_config.site_id = template.site.id.clone();
    render_config.site_name = template.site.name.clone();
    render_config.site_lat_deg = Some(site_lat);
    render_config.site_lon_deg = Some(site_lon);
    render_config.antenna_msl_m = Some(antenna_msl);
    // Replay ambiguity is applied from each observed radial's own Nyquist.
    render_config.fold_velocity = false;
    render_config.coupled_single_prf_estimator = false;
    render_config.emit_stage_diagnostics = false;
    if template
        .cuts
        .iter()
        .flat_map(|cut| &cut.moments)
        .any(|plan| plan.moment == MomentType::SpectrumWidth)
    {
        render_config.spectrum_width = true;
    }
    if template
        .cuts
        .iter()
        .flat_map(|cut| &cut.moments)
        .any(|plan| is_dual_pol_replay_moment(&plan.moment))
    {
        render_config.dual_pol = true;
    }
    // Replay always retains pulse-volume support fields. They are synthetic-
    // only diagnostics and the exact-geometry difference builder ignores them.
    render_config.emit_quality_fields = true;
    let beam_geometry =
        BeamPropagationGeometry::resolve(fields, &render_config, site_lat, site_lon, antenna_msl)?;
    let replay_leg = SyntheticScanLeg {
        elevation_deg: 0.0,
        azimuth_rate_deg_per_second: render_config.rotation_rate_deg_s,
        source_period_seconds: 0.0,
        transition_after_seconds: 0.0,
        moments: MomentCoverage::ALL,
        waveform: None,
        source_row_index: None,
        source_row: None,
    };
    let mut unavailable_observed_moments = Vec::new();

    for (cut_index, replay_cut) in template.cuts.iter().enumerate() {
        progress(&format!(
            "replaying observed cut {}/{} ({:.2} deg)...",
            cut_index + 1,
            template.cuts.len(),
            replay_cut.elevation_deg
        ));
        let mut output_cut =
            ElevationCut::new(replay_cut.elevation_deg, replay_cut.elevation_number);
        for ray in &replay_cut.rays {
            let time_offset_ms = i32::try_from(ray.acquisition_offset_ms).map_err(|_| {
                format!(
                    "exact replay cut {} radial {} acquisition offset {} ms exceeds i32",
                    cut_index + 1,
                    ray.source_radial_index + 1,
                    ray.acquisition_offset_ms
                )
            })?;
            output_cut.radials.push(Radial {
                azimuth_deg: ray.azimuth_deg,
                elevation_deg: ray.elevation_deg,
                time_offset_ms,
                gate_range: ray.gate_range.clone(),
                nyquist_velocity_mps: ray.nyquist_velocity_mps,
                radial_status: ray.radial_status,
            });
        }

        let quality_source_index = replay_cut
            .moments
            .iter()
            .position(|plan| plan.moment == MomentType::Reflectivity)
            .or_else(|| (!replay_cut.moments.is_empty()).then_some(0));
        for (moment_index, moment_plan) in replay_cut.moments.iter().enumerate() {
            let supported = match &moment_plan.moment {
                MomentType::Reflectivity | MomentType::Velocity => true,
                MomentType::SpectrumWidth => render_config.spectrum_width,
                moment if is_dual_pol_replay_moment(moment) => {
                    render_config.dual_pol && fields.has_polarimetric_input()
                }
                MomentType::Unknown(_) => true,
                _ => false,
            };
            if !supported && !matches!(moment_plan.moment, MomentType::Unknown(_)) {
                unavailable_observed_moments.push(format!(
                    "cut{}:{}",
                    cut_index + 1,
                    moment_plan.moment.short_name()
                ));
            }
            if !supported && quality_source_index != Some(moment_index) {
                continue;
            }
            let gate_count = moment_plan.gate_range.gate_count;
            let mut values = vec![f32::NAN; moment_plan.radial_indices.len() * gate_count];
            if !moment_plan.radial_indices.is_empty() {
                let mut ray_plan = Vec::with_capacity(moment_plan.radial_indices.len());
                for &source_index in &moment_plan.radial_indices {
                    let source = replay_cut.rays.get(source_index).ok_or_else(|| {
                        format!(
                            "exact replay cut {} moment {} refers to missing radial {}",
                            cut_index + 1,
                            moment_plan.moment.short_name(),
                            source_index
                        )
                    })?;
                    let sampling_offset_ms =
                        temporal.map_or(source.acquisition_offset_ms, |context| {
                            (source.acquisition_time_utc.to_owned()
                                - context.plan.anchor_time.to_owned())
                            .num_milliseconds()
                        });
                    ray_plan.push(SyntheticRayPlan {
                        source_index,
                        azimuth_deg: source.azimuth_deg,
                        elevation_deg: source.elevation_deg,
                        time_offset_ms: i32::try_from(sampling_offset_ms).map_err(|_| {
                            format!(
                                "exact replay cut {} radial {} temporal acquisition offset exceeds i32",
                                cut_index + 1,
                                source_index + 1
                            )
                        })?,
                        atmosphere_alpha: 0.0,
                        radial_status: source
                            .radial_status
                            .unwrap_or(radar_core::RadialStatus::Intermediate),
                    });
                }
                let spacing = f64::from(moment_plan.gate_range.gate_spacing_m);
                let terrain_horizon = (render_config.terrain_blockage
                    && !beam_geometry.is_wrf_refractivity())
                .then(|| {
                    TerrainHorizon::build(
                        fields,
                        site_lat,
                        site_lon,
                        antenna_msl,
                        // Fixed azimuth lookup sampled by each ray's actual
                        // angle; never treat irregular source indices as bins.
                        720,
                        gate_count,
                        moment_plan.gate_range.first_gate_m,
                        spacing,
                    )
                });
                let mut rendered = build_cut(
                    fields,
                    temporal,
                    cells,
                    site_lat,
                    site_lon,
                    antenna_msl,
                    f64::from(replay_cut.elevation_deg),
                    cut_index,
                    replay_cut.rays.len(),
                    spacing,
                    moment_plan.gate_range.clone(),
                    beam_geometry,
                    &render_config,
                    &replay_leg,
                    template.cuts.len(),
                    frame_seed,
                    terrain_horizon.as_ref(),
                    0,
                    None,
                    Some(&ray_plan),
                    cancel,
                )?;
                if quality_source_index == Some(moment_index) {
                    for quality in QualityMoment::ALL {
                        let moment = quality.moment_type();
                        if let Some(mut grid) = rendered.moments.remove(&moment) {
                            grid.radial_indices = moment_plan.radial_indices.clone();
                            output_cut.moments.insert(moment, grid);
                        }
                    }
                }
                if let Some(grid) = rendered.moments.remove(&moment_plan.moment)
                    && let MomentStorage::F32(rendered_values) = grid.storage
                    && rendered_values.len() == values.len()
                {
                    values = rendered_values;
                }
            }
            if moment_plan.moment == MomentType::Velocity {
                for (row, &source_index) in moment_plan.radial_indices.iter().enumerate() {
                    let Some(nyquist) = replay_cut.rays[source_index]
                        .nyquist_velocity_mps
                        .filter(|value| value.is_finite() && *value > 0.0)
                    else {
                        continue;
                    };
                    for value in &mut values[row * gate_count..(row + 1) * gate_count] {
                        if value.is_finite() {
                            *value = fold_velocity_mps(*value, nyquist);
                        }
                    }
                }
            }
            if supported {
                output_cut.moments.insert(
                    moment_plan.moment.clone(),
                    f32_grid(
                        moment_plan.moment.clone(),
                        moment_plan.gate_range.clone(),
                        moment_plan.radial_indices.clone(),
                        values,
                    ),
                );
            }
        }
        volume.cuts.push(output_cut);
    }
    if !unavailable_observed_moments.is_empty()
        && let Some(provenance) = volume.metadata.forward_operator_config.as_mut()
    {
        provenance.push_str("; unavailable_observed_moments=");
        provenance.push_str(&unavailable_observed_moments.join(","));
    }
    Ok(volume)
}

/// Cumulative apparent terrain elevation for every azimuth/range cell. The
/// running maximum is the terrain horizon seen by the virtual antenna; storing
/// it once makes blockage independent of elevation-cut count.
struct TerrainHorizon {
    azimuth_count: usize,
    gate_count: usize,
    elevation_deg: Vec<f32>,
}

impl TerrainHorizon {
    #[allow(clippy::too_many_arguments)]
    fn build_row(
        fields: &WrfRadarFields,
        site_lat: f64,
        site_lon: f64,
        antenna_msl: f64,
        azimuth_deg: f64,
        gate_count: usize,
        first_gate_m: i32,
        spacing_m: f64,
    ) -> Vec<f32> {
        let azimuth = azimuth_deg.to_radians();
        let mut horizon = f32::NEG_INFINITY;
        let mut row = Vec::with_capacity(gate_count);
        for gate in 0..gate_count {
            let slant_m = f64::from(first_gate_m) + gate as f64 * spacing_m;
            let ground_m = beam_ground_range_m(slant_m, 0.0);
            if ground_m < spacing_m.max(1.0) {
                row.push(horizon);
                continue;
            }
            let east_km = ground_m * azimuth.sin() / 1_000.0;
            let north_km = ground_m * azimuth.cos() / 1_000.0;
            let (lat, lon) = aeqd_inverse_km(site_lat, site_lon, east_km, north_km);
            if let Some(terrain_m) = sample_terrain(fields, lat as f32, lon as f32) {
                let ae = radar_core::EFFECTIVE_EARTH_RADIUS_M;
                let phi = ground_m / ae;
                let terrain_radius = ae + f64::from(terrain_m);
                let horizontal = terrain_radius * phi.sin();
                let vertical = terrain_radius * phi.cos() - (ae + antenna_msl);
                let apparent = vertical.atan2(horizontal).to_degrees() as f32;
                if apparent.is_finite() {
                    horizon = horizon.max(apparent);
                }
            }
            row.push(horizon);
        }
        row
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        fields: &WrfRadarFields,
        site_lat: f64,
        site_lon: f64,
        antenna_msl: f64,
        azimuth_count: usize,
        gate_count: usize,
        first_gate_m: i32,
        spacing_m: f64,
    ) -> Self {
        let rows: Vec<Vec<f32>> = (0..azimuth_count)
            .into_par_iter()
            .map(|iaz| {
                let azimuth_deg = iaz as f64 * 360.0 / azimuth_count as f64;
                Self::build_row(
                    fields,
                    site_lat,
                    site_lon,
                    antenna_msl,
                    azimuth_deg,
                    gate_count,
                    first_gate_m,
                    spacing_m,
                )
            })
            .collect();
        Self {
            azimuth_count,
            gate_count,
            elevation_deg: rows.into_iter().flatten().collect(),
        }
    }

    /// Exact lookup support for one requested radial. Only the two azimuth
    /// rows needed by interpolation are traced; all other rows remain empty.
    #[allow(clippy::too_many_arguments)]
    fn build_selected_azimuth(
        fields: &WrfRadarFields,
        site_lat: f64,
        site_lon: f64,
        antenna_msl: f64,
        azimuth_count: usize,
        selected_azimuth_deg: f64,
        gate_count: usize,
        first_gate_m: i32,
        spacing_m: f64,
    ) -> Self {
        let azimuth_count = azimuth_count.max(1);
        let az = selected_azimuth_deg.rem_euclid(360.0) * azimuth_count as f64 / 360.0;
        let lo = az.floor() as usize % azimuth_count;
        let hi = (lo + 1) % azimuth_count;
        let mut elevation_deg = vec![f32::NEG_INFINITY; azimuth_count * gate_count];
        for index in [lo, hi] {
            let azimuth_deg = index as f64 * 360.0 / azimuth_count as f64;
            let row = Self::build_row(
                fields,
                site_lat,
                site_lon,
                antenna_msl,
                azimuth_deg,
                gate_count,
                first_gate_m,
                spacing_m,
            );
            elevation_deg[index * gate_count..(index + 1) * gate_count].copy_from_slice(&row);
        }
        Self {
            azimuth_count,
            gate_count,
            elevation_deg,
        }
    }

    fn at(&self, azimuth_deg: f64, gate: usize) -> f32 {
        if self.azimuth_count == 0 || self.gate_count == 0 {
            return f32::NEG_INFINITY;
        }
        let az = azimuth_deg.rem_euclid(360.0) * self.azimuth_count as f64 / 360.0;
        let lo = az.floor() as usize % self.azimuth_count;
        let hi = (lo + 1) % self.azimuth_count;
        let t = (az - az.floor()) as f32;
        let gate = gate.min(self.gate_count - 1);
        let a = self.elevation_deg[lo * self.gate_count + gate];
        let b = self.elevation_deg[hi * self.gate_count + gate];
        if a.is_finite() && b.is_finite() {
            a + t * (b - a)
        } else if a.is_finite() {
            a
        } else {
            b
        }
    }
}

fn sample_terrain(fields: &WrfRadarFields, lat: f32, lon: f32) -> Option<f32> {
    let mut weighted = 0.0f32;
    let mut weight_sum = 0.0f32;
    for (column, weight) in horizontal_stencil(fields, lat, lon)? {
        let value = fields.terrain_m.get(column).copied()?;
        if weight > 0.0 && value.is_finite() {
            weighted += weight * value;
            weight_sum += weight;
        }
    }
    (weight_sum > 0.0).then_some(weighted / weight_sum)
}

#[derive(Clone, Copy)]
struct QuadraturePoint {
    az_sigma: f64,
    el_sigma: f64,
    range_gate: f64,
    weight: f64,
}

const CENTER_QUADRATURE: [QuadraturePoint; 1] = [QuadraturePoint {
    az_sigma: 0.0,
    el_sigma: 0.0,
    range_gate: 0.0,
    weight: 1.0,
}];

// Center plus all eight corners of a symmetric 3-D pulse cubature. The center
// receives four times one corner's weight; normalization happens per gate.
const BALANCED_QUADRATURE: [QuadraturePoint; 9] = [
    QuadraturePoint {
        az_sigma: 0.0,
        el_sigma: 0.0,
        range_gate: 0.0,
        weight: 4.0,
    },
    QuadraturePoint {
        az_sigma: -1.0,
        el_sigma: -1.0,
        range_gate: -0.35,
        weight: 1.0,
    },
    QuadraturePoint {
        az_sigma: -1.0,
        el_sigma: -1.0,
        range_gate: 0.35,
        weight: 1.0,
    },
    QuadraturePoint {
        az_sigma: -1.0,
        el_sigma: 1.0,
        range_gate: -0.35,
        weight: 1.0,
    },
    QuadraturePoint {
        az_sigma: -1.0,
        el_sigma: 1.0,
        range_gate: 0.35,
        weight: 1.0,
    },
    QuadraturePoint {
        az_sigma: 1.0,
        el_sigma: -1.0,
        range_gate: -0.35,
        weight: 1.0,
    },
    QuadraturePoint {
        az_sigma: 1.0,
        el_sigma: -1.0,
        range_gate: 0.35,
        weight: 1.0,
    },
    QuadraturePoint {
        az_sigma: 1.0,
        el_sigma: 1.0,
        range_gate: -0.35,
        weight: 1.0,
    },
    QuadraturePoint {
        az_sigma: 1.0,
        el_sigma: 1.0,
        range_gate: 0.35,
        weight: 1.0,
    },
];

const fn make_reference_quadrature() -> [QuadraturePoint; 27] {
    let coordinates = [-1.0, 0.0, 1.0];
    let weights = [1.0, 2.0, 1.0];
    let mut points = [CENTER_QUADRATURE[0]; 27];
    let mut index = 0usize;
    let mut iaz = 0usize;
    while iaz < 3 {
        let mut iel = 0usize;
        while iel < 3 {
            let mut irange = 0usize;
            while irange < 3 {
                points[index] = QuadraturePoint {
                    az_sigma: coordinates[iaz],
                    el_sigma: coordinates[iel],
                    range_gate: 0.35 * coordinates[irange],
                    weight: weights[iaz] * weights[iel] * weights[irange],
                };
                index += 1;
                irange += 1;
            }
            iel += 1;
        }
        iaz += 1;
    }
    points
}

const REFERENCE_QUADRATURE: [QuadraturePoint; 27] = make_reference_quadrature();

fn quadrature_points(mode: BeamIntegration) -> &'static [QuadraturePoint] {
    let points: &'static [QuadraturePoint] = match mode {
        BeamIntegration::Center => &CENTER_QUADRATURE,
        BeamIntegration::Balanced => &BALANCED_QUADRATURE,
        BeamIntegration::Reference => &REFERENCE_QUADRATURE,
    };
    debug_assert_eq!(points.len(), mode.pulse_volume_sample_count());
    points
}

#[derive(Clone, Copy)]
struct ResolvedBeamSamplePoint {
    az_sigma: f64,
    el_sigma: f64,
    range_offset_m: f64,
    weight: f64,
}

fn resolved_beam_sample_points(
    config: &SyntheticRadarConfig,
    coupled_instrument: Option<&CoupledInstrumentContext>,
    spacing_m: f64,
) -> Vec<ResolvedBeamSamplePoint> {
    if let Some(points) =
        coupled_instrument.and_then(|instrument| instrument.quadrature(config.beam_integration))
    {
        points
            .iter()
            .map(|point| ResolvedBeamSamplePoint {
                az_sigma: point.az_sigma,
                el_sigma: point.el_sigma,
                range_offset_m: point.range_offset_m,
                weight: point.weight,
            })
            .collect()
    } else {
        quadrature_points(config.beam_integration)
            .iter()
            .map(|point| ResolvedBeamSamplePoint {
                az_sigma: point.az_sigma,
                el_sigma: point.el_sigma,
                range_offset_m: point.range_gate * spacing_m,
                weight: point.weight,
            })
            .collect()
    }
}

#[derive(Clone, Copy)]
enum BeamPropagationGeometry<'a> {
    StandardFourThirdsEarth,
    WrfRefractivity(&'a crate::wrf_refractivity::WrfRefractivityModel),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct BeamGeometryPoint {
    ground_range_m: f64,
    height_above_radar_m: f64,
}

#[derive(Clone, Copy)]
struct BeamColumnPoint {
    azimuth_deg: f64,
    elevation_deg: f64,
    latitude: f32,
    longitude: f32,
    z_msl: f64,
    geometry: BeamGeometryPoint,
}

#[allow(clippy::too_many_arguments)]
fn beam_column_point(
    site_lat: f64,
    site_lon: f64,
    antenna_msl: f64,
    center_azimuth_deg: f64,
    center_elevation_deg: f64,
    center_slant_m: f64,
    gate_index: usize,
    gate_range: &GateRange,
    beam_geometry: BeamPropagationGeometry<'_>,
    beam_sigma_deg: f64,
    point: ResolvedBeamSamplePoint,
) -> Result<BeamColumnPoint, String> {
    let azimuth_deg = center_azimuth_deg + point.az_sigma * beam_sigma_deg;
    let elevation_deg = center_elevation_deg + point.el_sigma * beam_sigma_deg;
    let geometry = beam_geometry.point(
        elevation_deg,
        gate_range,
        gate_index,
        point.range_offset_m,
        center_slant_m,
    )?;
    let z_msl = antenna_msl + geometry.height_above_radar_m;
    let azimuth = azimuth_deg.to_radians();
    let east_km = geometry.ground_range_m * azimuth.sin() / 1_000.0;
    let north_km = geometry.ground_range_m * azimuth.cos() / 1_000.0;
    let (latitude, longitude) = aeqd_inverse_km(site_lat, site_lon, east_km, north_km);
    Ok(BeamColumnPoint {
        azimuth_deg,
        elevation_deg,
        latitude: latitude as f32,
        longitude: longitude as f32,
        z_msl,
        geometry,
    })
}

impl<'a> BeamPropagationGeometry<'a> {
    fn resolve(
        fields: &'a WrfRadarFields,
        config: &SyntheticRadarConfig,
        site_lat: f64,
        site_lon: f64,
        antenna_msl: f64,
    ) -> Result<Self, String> {
        match config.propagation_geometry {
            PropagationGeometry::StandardFourThirdsEarth => Ok(Self::StandardFourThirdsEarth),
            PropagationGeometry::WrfRefractivityResearch => {
                let model = fields.refractivity_model.as_deref().ok_or_else(|| {
                    if fields.source_model_override.is_some() {
                        "WRF refractivity geometry is unavailable for operational HRRR/RRFS GRIB; use Standard 4/3 Earth"
                            .to_owned()
                    } else {
                        "WRF refractivity geometry was selected but no qualified P/PB/T/QVAPOR profile was retained"
                            .to_owned()
                    }
                })?;
                let provenance = model.provenance();
                let horizontal = &provenance.horizontal;
                if (horizontal.site_latitude_deg - site_lat).abs() > 1.0e-6
                    || (horizontal.site_longitude_deg - site_lon).abs() > 1.0e-6
                    || (provenance.antenna_msl_m - antenna_msl).abs() > 1.0e-3
                {
                    return Err(
                        "retained WRF refractivity profile does not match the active radar site/antenna"
                            .to_owned(),
                    );
                }
                Ok(Self::WrfRefractivity(model))
            }
        }
    }

    const fn is_wrf_refractivity(self) -> bool {
        matches!(self, Self::WrfRefractivity(_))
    }

    fn point(
        self,
        elevation_deg: f64,
        gate_range: &GateRange,
        gate_index: usize,
        range_offset_m: f64,
        standard_center_slant_m: f64,
    ) -> Result<BeamGeometryPoint, String> {
        match self {
            Self::StandardFourThirdsEarth => {
                let slant_m = (standard_center_slant_m + range_offset_m).max(0.0);
                Ok(BeamGeometryPoint {
                    ground_range_m: beam_ground_range_m(slant_m, elevation_deg),
                    height_above_radar_m: beam_height_above_radar_m(slant_m, elevation_deg),
                })
            }
            Self::WrfRefractivity(model) => {
                let offset_m = range_offset_m
                    .round()
                    .clamp(i32::MIN as f64, i32::MAX as f64) as i32;
                let adjusted = GateRange {
                    first_gate_m: gate_range.first_gate_m.saturating_add(offset_m).max(0),
                    gate_spacing_m: gate_range.gate_spacing_m,
                    gate_count: gate_range.gate_count,
                };
                let trace = model
                    .trace_for_gate_range(elevation_deg, &adjusted)
                    .map_err(|error| format!("trace WRF refracted beam: {error}"))?;
                let point = trace.point(gate_index).ok_or_else(|| {
                    format!(
                        "WRF refracted trace has no gate {gate_index} of {}",
                        adjusted.gate_count
                    )
                })?;
                Ok(BeamGeometryPoint {
                    ground_range_m: point.ground_range_m,
                    height_above_radar_m: point.height_above_radar_m,
                })
            }
        }
    }
}

fn append_beam_geometry_provenance(
    output: &mut String,
    fields: &WrfRadarFields,
    geometry: PropagationGeometry,
) {
    use std::fmt::Write as _;
    match geometry {
        PropagationGeometry::StandardFourThirdsEarth => {
            output.push_str("; propagation_geometry=standard_four_thirds_earth");
        }
        PropagationGeometry::WrfRefractivityResearch => {
            if let Some(model) = fields.refractivity_model.as_deref() {
                let provenance = model.provenance();
                let gradients = &provenance.gradients;
                let _ = write!(
                    output,
                    "; propagation_geometry=wrf_refractivity_research; refractivity_fields={}/{}/{}; refractivity_humidity=QVAPOR_dry_mixing_ratio_to_moist_specific_humidity; refractivity_formula={}; refractivity_time_index={}; refractivity_profile_levels={}; refractivity_antenna_gradient_n_per_km={:.6}; refractivity_antenna_regime={:?}; refractivity_min_gradient_n_per_km={:.6}; refractivity_max_gradient_n_per_km={:.6}; refractivity_contains_ducting_layer={}; refractivity_antenna_level_inserted={}; refractivity_antenna_extrapolation_m={:.3}",
                    provenance.pressure_source,
                    provenance.temperature_source,
                    provenance.humidity_source,
                    provenance.formula,
                    provenance.time_index,
                    provenance.retained_level_count,
                    gradients.antenna_gradient_n_per_km,
                    gradients.antenna_regime,
                    gradients.minimum_gradient_n_per_km,
                    gradients.maximum_gradient_n_per_km,
                    gradients.contains_ducting_layer,
                    provenance.antenna_level_inserted,
                    provenance.antenna_extrapolation_m,
                );
            }
        }
    }
}

fn refractivity_run_note(fields: &WrfRadarFields) -> Option<String> {
    let model = fields.refractivity_model.as_deref()?;
    let gradients = &model.provenance().gradients;
    Some(format!(
        "WRF refractivity: antenna gradient {:.2} N/km ({:?}), profile {:.2}..{:.2} N/km{}",
        gradients.antenna_gradient_n_per_km,
        gradients.antenna_regime,
        gradients.minimum_gradient_n_per_km,
        gradients.maximum_gradient_n_per_km,
        if gradients.contains_ducting_layer {
            "; ⚠ DUCTING LAYER PRESENT"
        } else {
            ""
        }
    ))
}

#[derive(Clone, Copy)]
struct GatePhysicalSample {
    z_linear: f32,
    velocity_mps: f32,
    spectrum_width_mps: f32,
    polar: Option<crate::wrf_radar_physics::IntrinsicPolarSample>,
    air_velocity_mps: f32,
    terminal_fall_correction_mps: f32,
    pulse_volume_variance_m2s2: f32,
    terminal_variance_m2s2: f32,
    turbulence_variance_m2s2: f32,
}

struct GateSampleResult {
    physical: Option<GatePhysicalSample>,
    quality: GateQualityFractions,
}

/// Keep the host-side prepare/evaluate/finish footprint near the original
/// eight-gate Reference bound while widening sparse Center/Balanced work.
/// The 64-gate cap also bounds cancellation latency and retained preparations.
pub(crate) const RAW_TMATRIX_PIPELINE_TARGET_COLUMNS: usize = 8 * 27;
pub(crate) const RAW_TMATRIX_PIPELINE_MAX_GATE_CHUNK: usize = 64;

pub(crate) const fn raw_tmatrix_pipeline_gate_chunk(point_count: usize) -> usize {
    if point_count == 0 {
        return 1;
    }
    let gates = RAW_TMATRIX_PIPELINE_TARGET_COLUMNS.saturating_add(point_count.saturating_sub(1))
        / point_count;
    if gates == 0 {
        1
    } else if gates > RAW_TMATRIX_PIPELINE_MAX_GATE_CHUNK {
        RAW_TMATRIX_PIPELINE_MAX_GATE_CHUNK
    } else {
        gates
    }
}

struct RawGateColumnBatch {
    first_gate: usize,
    gate_count: usize,
    points: Vec<ResolvedBeamSamplePoint>,
    center_point_index: Option<usize>,
    locations: Vec<BeamColumnPoint>,
    samples: Vec<Option<ColumnSample>>,
}

impl RawGateColumnBatch {
    fn position(&self, gate_index: usize, point_index: usize) -> Result<usize, String> {
        let local_gate = gate_index.checked_sub(self.first_gate).ok_or_else(|| {
            format!(
                "raw T-matrix gate batch starts at {}, queried gate {gate_index}",
                self.first_gate
            )
        })?;
        if local_gate >= self.gate_count || point_index >= self.points.len() {
            return Err(format!(
                "raw T-matrix gate batch does not contain gate {gate_index}, point {point_index}"
            ));
        }
        let position = local_gate
            .checked_mul(self.points.len())
            .and_then(|offset| offset.checked_add(point_index))
            .ok_or_else(|| "raw T-matrix gate-batch lookup overflowed".to_owned())?;
        Ok(position)
    }

    fn location(&self, gate_index: usize, point_index: usize) -> Result<BeamColumnPoint, String> {
        let position = self.position(gate_index, point_index)?;
        self.locations
            .get(position)
            .copied()
            .ok_or_else(|| "raw T-matrix gate-batch geometry is incomplete".to_owned())
    }

    fn sample(
        &self,
        gate_index: usize,
        point_index: usize,
    ) -> Result<Option<ColumnSample>, String> {
        let position = self.position(gate_index, point_index)?;
        self.samples
            .get(position)
            .copied()
            .ok_or_else(|| "raw T-matrix gate-batch storage is incomplete".to_owned())
    }

    fn center_geometry(&self, gate_index: usize) -> Result<Option<BeamGeometryPoint>, String> {
        self.center_point_index
            .map(|point_index| {
                self.location(gate_index, point_index)
                    .map(|location| location.geometry)
            })
            .transpose()
    }
}

fn exact_center_point_index(points: &[ResolvedBeamSamplePoint]) -> Option<usize> {
    points.iter().position(|point| {
        point.az_sigma == 0.0 && point.el_sigma == 0.0 && point.range_offset_m == 0.0
    })
}

#[allow(clippy::too_many_arguments)]
fn build_raw_gate_column_batch(
    fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    atmosphere_alpha: f64,
    cells: usize,
    site_lat: f64,
    site_lon: f64,
    antenna_msl: f64,
    center_azimuth_deg: f64,
    center_elevation_deg: f64,
    spacing_m: f64,
    gate_range: &GateRange,
    beam_geometry: BeamPropagationGeometry<'_>,
    config: &SyntheticRadarConfig,
    coupled_instrument: Option<&CoupledInstrumentContext>,
    first_gate: usize,
    end_gate: usize,
    cuda: &app_ui::wrf_tmatrix_cuda::WrfTMatrixCudaBatchService,
    cancel: Option<&AtomicBool>,
) -> Result<RawGateColumnBatch, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let gate_count = end_gate.saturating_sub(first_gate);
    let points = resolved_beam_sample_points(config, coupled_instrument, spacing_m);
    if points.is_empty() {
        return Err("raw T-matrix gate batch requires at least one beam sample point".to_owned());
    }
    let center_point_index = exact_center_point_index(&points);
    let work_count = gate_count
        .checked_mul(points.len())
        .ok_or_else(|| "raw T-matrix gate-batch work size overflowed".to_owned())?;
    let beam_sigma_deg = gaussian_beam_sigma_deg(f64::from(config.beam_width_deg.max(0.05)));
    let prepared = (0..work_count)
        .into_par_iter()
        .map(
            |position| -> Result<(BeamColumnPoint, Option<PreparedRawColumnSample>), String> {
                let local_gate = position / points.len();
                let point_index = position % points.len();
                let gate_index = first_gate + local_gate;
                let center_slant_m =
                    f64::from(gate_range.first_gate_m) + gate_index as f64 * spacing_m;
                let location = beam_column_point(
                    site_lat,
                    site_lon,
                    antenna_msl,
                    center_azimuth_deg,
                    center_elevation_deg,
                    center_slant_m,
                    gate_index,
                    gate_range,
                    beam_geometry,
                    beam_sigma_deg,
                    points[point_index],
                )?;
                let prepared = prepare_column_raw_state_linear(
                    fields,
                    neighbor_fields,
                    atmosphere_alpha,
                    cells,
                    location.latitude,
                    location.longitude,
                    location.z_msl as f32,
                    cancel,
                )?;
                Ok((location, prepared))
            },
        )
        .collect::<Vec<_>>();
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    // Indexed Rayon collection preserves work position. Selecting errors in
    // this serial pass makes (gate, beam point) ordering deterministic.
    let mut locations = Vec::with_capacity(prepared.len());
    let mut ordered = Vec::with_capacity(prepared.len());
    for result in prepared {
        let (location, prepared) = result?;
        locations.push(location);
        ordered.push(prepared);
    }
    let evaluated = {
        let requests = ordered
            .iter()
            .enumerate()
            .filter_map(|(position, prepared)| {
                prepared.as_ref().map(|prepared| {
                    app_ui::wrf_tmatrix_scene::WrfTMatrixRawBatchRequest::new(
                        &prepared.property_cell,
                        locations[position].elevation_deg,
                    )
                })
            })
            .collect::<Vec<_>>();
        app_ui::wrf_tmatrix_assets::evaluate_embedded_raw_property_batch_with_cuda_and_cancel(
            &requests,
            Some(cuda),
            cancel,
        )
        .map_err(|error| {
            format!("ordered raw T-matrix gate batch {first_gate}..{end_gate} failed: {error}")
        })?
    };
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let expected_evaluated = ordered.iter().filter(|prepared| prepared.is_some()).count();
    if evaluated.len() != expected_evaluated {
        return Err(format!(
            "raw T-matrix batch returned {} covered samples, expected {expected_evaluated}",
            evaluated.len()
        ));
    }
    let mut evaluated = evaluated.into_iter();
    let mut samples = Vec::with_capacity(ordered.len());
    for prepared in ordered {
        samples.push(match prepared {
            Some(prepared) => Some(finish_prepared_raw_column(
                prepared,
                evaluated
                    .next()
                    .expect("covered raw columns and batch answers have equal length"),
                cancel,
            )?),
            None => None,
        });
    }
    debug_assert!(evaluated.next().is_none());
    Ok(RawGateColumnBatch {
        first_gate,
        gate_count,
        points,
        center_point_index,
        locations,
        samples,
    })
}

#[allow(clippy::too_many_arguments)]
fn sample_gate_with_quality_instrument(
    fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    atmosphere_alpha: f64,
    cells: usize,
    site_lat: f64,
    site_lon: f64,
    antenna_msl: f64,
    center_azimuth_deg: f64,
    center_elevation_deg: f64,
    center_slant_m: f64,
    gate_index: usize,
    spacing_m: f64,
    gate_range: &GateRange,
    beam_geometry: BeamPropagationGeometry<'_>,
    config: &SyntheticRadarConfig,
    terrain_horizon: Option<&TerrainHorizon>,
    mut refracted_terrain_blockage: Option<&mut [bool]>,
    coupled_instrument: Option<&CoupledInstrumentContext>,
    raw_batch: Option<&RawGateColumnBatch>,
    cancel: Option<&AtomicBool>,
) -> Result<GateSampleResult, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let beam_sigma_deg = gaussian_beam_sigma_deg(f64::from(config.beam_width_deg.max(0.05)));
    let owned_points;
    let points = if let Some(raw_batch) = raw_batch {
        raw_batch.points.as_slice()
    } else {
        owned_points = resolved_beam_sample_points(config, coupled_instrument, spacing_m);
        owned_points.as_slice()
    };
    let total_weight = points.iter().map(|point| point.weight).sum::<f64>();
    let mut valid_weight = 0.0f64;
    let mut terrain_unblocked_weight = 0.0f64;
    let mut signal_weight = 0.0f64;
    let mut sum_z = 0.0f64;
    let mut sum_z_vr = 0.0f64;
    let mut sum_z_vr2 = 0.0f64;
    let mut sum_z_air_velocity = 0.0f64;
    let mut sum_z_terminal_variance = 0.0f64;
    let mut sum_z_turbulence_variance = 0.0f64;
    let mut polar_accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
    let mut any_polar = false;

    let mut accumulate_point =
        |point_index: usize, point: ResolvedBeamSamplePoint| -> Result<(), String> {
            let location = if let Some(raw_batch) = raw_batch {
                raw_batch.location(gate_index, point_index)?
            } else {
                beam_column_point(
                    site_lat,
                    site_lon,
                    antenna_msl,
                    center_azimuth_deg,
                    center_elevation_deg,
                    center_slant_m,
                    gate_index,
                    gate_range,
                    beam_geometry,
                    beam_sigma_deg,
                    point,
                )?
            };
            let azimuth_deg = location.azimuth_deg;
            let elevation_deg = location.elevation_deg;
            let z_msl = location.z_msl;
            let azimuth = azimuth_deg.to_radians();
            let refracted_blocked =
                if let Some(blocked_by_point) = refracted_terrain_blockage.as_deref_mut() {
                    let blocked = blocked_by_point.get_mut(point_index).ok_or_else(|| {
                        "WRF refractivity terrain state does not match beam quadrature".to_owned()
                    })?;
                    if sample_terrain(fields, location.latitude, location.longitude)
                        .is_some_and(|terrain_m| f64::from(terrain_m) >= z_msl)
                    {
                        *blocked = true;
                    }
                    Some(*blocked)
                } else {
                    None
                };
            let sampled_column = if let Some(raw_batch) = raw_batch {
                raw_batch.sample(gate_index, point_index)?
            } else {
                sample_column_temporal(
                    fields,
                    neighbor_fields,
                    atmosphere_alpha,
                    cells,
                    location.latitude,
                    location.longitude,
                    location.z_msl as f32,
                    elevation_deg,
                    config.atmosphere_time_mode,
                    config.reflectivity_sampling,
                    config.runtime_cuda_service.as_deref(),
                    cancel,
                )?
            };
            let Some(sample) = sampled_column else {
                return Ok(());
            };
            valid_weight += point.weight;

            let blocked = refracted_blocked.unwrap_or_else(|| {
                terrain_horizon.is_some_and(|horizon| {
                    let horizon_deg = horizon.at(azimuth_deg, gate_index);
                    horizon_deg.is_finite() && elevation_deg as f32 <= horizon_deg
                })
            });
            if blocked {
                return Ok(());
            }
            terrain_unblocked_weight += point.weight;

            let el = elevation_deg.to_radians();
            let fall_speed = if config.terminal_fall_speed {
                sample.polar.map_or(0.0, |polar| polar.fall_speed_mps)
            } else {
                0.0
            };
            let air_velocity = f64::from(sample.u) * azimuth.sin() * el.cos()
                + f64::from(sample.v) * azimuth.cos() * el.cos()
                + f64::from(sample.w) * el.sin();
            let vr = air_velocity - f64::from(fall_speed) * el.sin();
            let z = f64::from(sample.z_linear);
            if !z.is_finite() || z <= 0.0 || !vr.is_finite() {
                return Ok(());
            }
            signal_weight += point.weight;
            sum_z += point.weight * z;
            sum_z_vr += point.weight * z * vr;
            sum_z_vr2 += point.weight * z * vr * vr;
            sum_z_air_velocity += point.weight * z * air_velocity;
            let terminal_variance = if config.terminal_fall_speed {
                sample
                    .polar
                    .map_or(0.0, |polar| f64::from(polar.fall_speed_variance_m2s2))
                    * el.sin().powi(2)
            } else {
                0.0
            };
            let turbulent_variance = if config.spectrum_width {
                // Isotropic one-dimensional variance from TKE = 1/2(u'^2+v'^2+w'^2).
                f64::from((2.0 / 3.0) * sample.tke_m2s2.max(0.0))
            } else {
                0.0
            };
            sum_z_terminal_variance += point.weight * z * terminal_variance;
            sum_z_turbulence_variance += point.weight * z * turbulent_variance;
            if let Some(polar) = sample.polar {
                any_polar = true;
                polar_accumulator.add(point.weight as f32, intrinsic_as_contribution(polar));
            }
            Ok(())
        };
    for (point_index, &point) in points.iter().enumerate() {
        accumulate_point(point_index, point)?;
    }

    let quality = GateQuality {
        total_weight,
        model_covered_weight: valid_weight,
        terrain_unblocked_weight,
        meteorological_signal_weight: signal_weight,
    }
    .fractions();
    if valid_weight <= 0.0 || signal_weight <= 0.0 || sum_z <= 0.0 {
        return Ok(GateSampleResult {
            physical: None,
            quality,
        });
    }
    // Missing model-domain quadrature points are renormalized away. Terrain-
    // blocked points remain in `valid_weight` but contribute no signal, so
    // partial blockage correctly reduces received power.
    let mut polar =
        any_polar.then(|| normalize_intrinsic(polar_accumulator.finalize(), valid_weight as f32));
    let z_linear = polar
        .filter(|sample| sample.zh > 0.0)
        .map_or((sum_z / valid_weight) as f32, |sample| sample.zh);
    let velocity = sum_z_vr / sum_z;
    let pulse_volume_variance = (sum_z_vr2 / sum_z - velocity * velocity).max(0.0);
    let terminal_variance = sum_z_terminal_variance / sum_z;
    let turbulence_variance = sum_z_turbulence_variance / sum_z;
    let variance = pulse_volume_variance + terminal_variance + turbulence_variance;
    let floor = if config.spectrum_width {
        f64::from(config.spectrum_width_floor_mps.max(0.0))
    } else {
        0.0
    };
    Ok(GateSampleResult {
        physical: Some(GatePhysicalSample {
            z_linear,
            velocity_mps: velocity as f32,
            spectrum_width_mps: (variance + floor * floor).sqrt() as f32,
            polar: polar.take(),
            air_velocity_mps: (sum_z_air_velocity / sum_z) as f32,
            terminal_fall_correction_mps: (velocity - sum_z_air_velocity / sum_z) as f32,
            pulse_volume_variance_m2s2: pulse_volume_variance as f32,
            terminal_variance_m2s2: terminal_variance as f32,
            turbulence_variance_m2s2: turbulence_variance as f32,
        }),
        quality,
    })
}

#[allow(clippy::too_many_arguments)]
fn sample_gate_with_quality(
    fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    atmosphere_alpha: f64,
    cells: usize,
    site_lat: f64,
    site_lon: f64,
    antenna_msl: f64,
    center_azimuth_deg: f64,
    center_elevation_deg: f64,
    center_slant_m: f64,
    gate_index: usize,
    spacing_m: f64,
    config: &SyntheticRadarConfig,
    terrain_horizon: Option<&TerrainHorizon>,
) -> Result<GateSampleResult, String> {
    let gate_range = GateRange {
        first_gate_m: (center_slant_m - gate_index as f64 * spacing_m)
            .round()
            .max(0.0) as i32,
        gate_spacing_m: spacing_m.round().max(1.0) as i32,
        gate_count: gate_index.saturating_add(1),
    };
    sample_gate_with_quality_instrument(
        fields,
        neighbor_fields,
        atmosphere_alpha,
        cells,
        site_lat,
        site_lon,
        antenna_msl,
        center_azimuth_deg,
        center_elevation_deg,
        center_slant_m,
        gate_index,
        spacing_m,
        &gate_range,
        BeamPropagationGeometry::StandardFourThirdsEarth,
        config,
        terrain_horizon,
        None,
        None,
        None,
        None,
    )
}

/// Compatibility seam for focused physics tests and callers that need only
/// the sampled moments. Production cut building uses the quality-bearing form.
#[allow(clippy::too_many_arguments)]
fn sample_gate(
    fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    atmosphere_alpha: f64,
    cells: usize,
    site_lat: f64,
    site_lon: f64,
    antenna_msl: f64,
    center_azimuth_deg: f64,
    center_elevation_deg: f64,
    center_slant_m: f64,
    gate_index: usize,
    spacing_m: f64,
    config: &SyntheticRadarConfig,
    terrain_horizon: Option<&TerrainHorizon>,
) -> Result<Option<GatePhysicalSample>, String> {
    Ok(sample_gate_with_quality(
        fields,
        neighbor_fields,
        atmosphere_alpha,
        cells,
        site_lat,
        site_lon,
        antenna_msl,
        center_azimuth_deg,
        center_elevation_deg,
        center_slant_m,
        gate_index,
        spacing_m,
        config,
        terrain_horizon,
    )?
    .physical)
}

struct CutMomentRow {
    reflectivity: Vec<f32>,
    velocity: Vec<f32>,
    spectrum_width: Vec<f32>,
    zdr: Vec<f32>,
    rho_hv: Vec<f32>,
    phi_dp: Vec<f32>,
    kdp: Vec<f32>,
    ah: Vec<f32>,
    pia: Vec<f32>,
    ref_corrected: Vec<f32>,
    adp: Vec<f32>,
    pida: Vec<f32>,
    zdr_corrected: Vec<f32>,
    model_coverage: Vec<u8>,
    terrain_unblocked: Vec<u8>,
    meteorological_signal: Vec<u8>,
    stage_diagnostics: Option<StageDiagnosticRow>,
}

struct StageDiagnosticRow {
    ideal_reflectivity: Vec<f32>,
    ideal_velocity: Vec<f32>,
    ideal_spectrum_width: Vec<f32>,
    ideal_zdr: Vec<f32>,
    ideal_rho_hv: Vec<f32>,
    ideal_kdp: Vec<f32>,
    measured_reflectivity: Vec<f32>,
    measured_velocity: Vec<f32>,
    measured_spectrum_width: Vec<f32>,
    measured_zdr: Vec<f32>,
    measured_rho_hv: Vec<f32>,
    measured_kdp: Vec<f32>,
}

fn diagnostic_value(value: Option<f64>) -> f32 {
    value
        .filter(|value| value.is_finite())
        .map_or(f32::NAN, |value| value as f32)
}

impl StageDiagnosticRow {
    fn with_len(len: usize) -> Self {
        let blank = || vec![f32::NAN; len];
        Self {
            ideal_reflectivity: blank(),
            ideal_velocity: blank(),
            ideal_spectrum_width: blank(),
            ideal_zdr: blank(),
            ideal_rho_hv: blank(),
            ideal_kdp: blank(),
            measured_reflectivity: blank(),
            measured_velocity: blank(),
            measured_spectrum_width: blank(),
            measured_zdr: blank(),
            measured_rho_hv: blank(),
            measured_kdp: blank(),
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        let values = || Vec::with_capacity(capacity);
        Self {
            ideal_reflectivity: values(),
            ideal_velocity: values(),
            ideal_spectrum_width: values(),
            ideal_zdr: values(),
            ideal_rho_hv: values(),
            ideal_kdp: values(),
            measured_reflectivity: values(),
            measured_velocity: values(),
            measured_spectrum_width: values(),
            measured_zdr: values(),
            measured_rho_hv: values(),
            measured_kdp: values(),
        }
    }

    fn record(&mut self, gate: usize, ideal: IdealMoments, measured: MeasuredMoments) {
        self.ideal_reflectivity[gate] = diagnostic_value(ideal.values.reflectivity_dbz);
        self.ideal_velocity[gate] = diagnostic_value(ideal.values.velocity_mps);
        self.ideal_spectrum_width[gate] = diagnostic_value(ideal.values.spectrum_width_mps);
        self.ideal_zdr[gate] = diagnostic_value(ideal.values.zdr_db);
        self.ideal_rho_hv[gate] = diagnostic_value(ideal.values.rho_hv);
        self.ideal_kdp[gate] = diagnostic_value(ideal.values.kdp_deg_km);
        self.measured_reflectivity[gate] = diagnostic_value(measured.values.reflectivity_dbz);
        self.measured_velocity[gate] = diagnostic_value(measured.values.velocity_mps);
        self.measured_spectrum_width[gate] = diagnostic_value(measured.values.spectrum_width_mps);
        self.measured_zdr[gate] = diagnostic_value(measured.values.zdr_db);
        self.measured_rho_hv[gate] = diagnostic_value(measured.values.rho_hv);
        self.measured_kdp[gate] = diagnostic_value(measured.values.kdp_deg_km);
    }

    fn append(&mut self, row: Self) {
        self.ideal_reflectivity.extend(row.ideal_reflectivity);
        self.ideal_velocity.extend(row.ideal_velocity);
        self.ideal_spectrum_width.extend(row.ideal_spectrum_width);
        self.ideal_zdr.extend(row.ideal_zdr);
        self.ideal_rho_hv.extend(row.ideal_rho_hv);
        self.ideal_kdp.extend(row.ideal_kdp);
        self.measured_reflectivity.extend(row.measured_reflectivity);
        self.measured_velocity.extend(row.measured_velocity);
        self.measured_spectrum_width
            .extend(row.measured_spectrum_width);
        self.measured_zdr.extend(row.measured_zdr);
        self.measured_rho_hv.extend(row.measured_rho_hv);
        self.measured_kdp.extend(row.measured_kdp);
    }
}

/// Acquisition geometry and timing resolved before gate sampling begins.
///
/// Temporal WRF sampling needs one interpolation weight for the whole ray, so
/// acquisition time cannot be invented after the parallel sampler has already
/// read the atmosphere. Keeping this plan separate also makes the timestamp
/// written to [`Radial`] exactly the timestamp used by the forward operator.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SyntheticRayPlan {
    source_index: usize,
    azimuth_deg: f32,
    elevation_deg: f32,
    time_offset_ms: i32,
    atmosphere_alpha: f64,
    radial_status: radar_core::RadialStatus,
}

fn plan_synthetic_rays(
    cut_index: usize,
    cut_count: usize,
    naz: usize,
    scan_timing: ScanTiming,
    azimuth_rate_deg_per_second: f32,
    cut_start_ms: i32,
) -> Vec<SyntheticRayPlan> {
    (0..naz)
        .map(|source_index| {
            let azimuth_deg = source_index as f32 * 360.0 / naz as f32;
            let time_offset_ms = match scan_timing {
                ScanTiming::InstantaneousTruth => 0,
                ScanTiming::TimedVolume => {
                    let radial_ms = 1_000.0 * azimuth_deg / azimuth_rate_deg_per_second.max(0.1);
                    cut_start_ms.saturating_add(radial_ms.round() as i32)
                }
            };
            let radial_status = if source_index == 0 && cut_index == 0 {
                radar_core::RadialStatus::StartVolume
            } else if source_index == 0 {
                radar_core::RadialStatus::StartElevation
            } else if source_index + 1 == naz && cut_index + 1 == cut_count {
                radar_core::RadialStatus::EndVolume
            } else if source_index + 1 == naz {
                radar_core::RadialStatus::EndElevation
            } else {
                radar_core::RadialStatus::Intermediate
            };
            SyntheticRayPlan {
                source_index,
                azimuth_deg,
                elevation_deg: 0.0,
                time_offset_ms,
                atmosphere_alpha: 0.0,
                radial_status,
            }
        })
        .collect()
}

impl CutMomentRow {
    fn blank(gates: usize, emit_stage_diagnostics: bool) -> Self {
        let blank = || vec![f32::NAN; gates];
        Self {
            reflectivity: blank(),
            velocity: blank(),
            spectrum_width: blank(),
            zdr: blank(),
            rho_hv: blank(),
            phi_dp: blank(),
            kdp: blank(),
            ah: blank(),
            pia: blank(),
            ref_corrected: blank(),
            adp: blank(),
            pida: blank(),
            zdr_corrected: blank(),
            model_coverage: vec![0; gates],
            terrain_unblocked: vec![0; gates],
            meteorological_signal: vec![0; gates],
            stage_diagnostics: emit_stage_diagnostics.then(|| StageDiagnosticRow::with_len(gates)),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_cut(
    fields: &WrfRadarFields,
    temporal: Option<&TemporalRenderContext<'_>>,
    cells: usize,
    site_lat: f64,
    site_lon: f64,
    antenna_msl: f64,
    elevation_deg: f64,
    cut_index: usize,
    naz: usize,
    spacing: f64,
    gate_range: GateRange,
    beam_geometry: BeamPropagationGeometry<'_>,
    config: &SyntheticRadarConfig,
    scan_leg: &SyntheticScanLeg,
    scan_leg_count: usize,
    frame_seed: u32,
    terrain_horizon: Option<&TerrainHorizon>,
    cut_start_ms: i32,
    coupled_instrument: Option<&CoupledInstrumentContext>,
    ray_plan_override: Option<&[SyntheticRayPlan]>,
    cancel: Option<&AtomicBool>,
) -> Result<ElevationCut, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let gate_count = gate_range.gate_count;
    // Instrument mode already applies its range-varying radar-equation
    // sensitivity gate below; a fixed 0 dBZ presentation floor would erase
    // valid weak nearby returns a real instrument can see.
    let floor = if config.instrument_noise {
        f32::NEG_INFINITY
    } else {
        config.ref_floor_dbz
    };

    // Ground clutter (opt-in). When the amount is 0 the whole clutter path is
    // skipped and the output is bit-identical to a build without the feature.
    // The near-radar hotspots are precomputed once per tilt (serial, ≤8 rects)
    // so the parallel per-gate work only does an O(hotspots) membership test.
    let clutter_amount = config.clutter_intensity.clamp(0.0, 1.0);
    let clutter_on = clutter_amount > 0.0 && cut_index < CLUTTER_TILT_LIMIT;
    let hotspots = if clutter_on {
        clutter_hotspots(frame_seed, cut_index, naz, gate_count)
    } else {
        Vec::new()
    };

    let mut ray_plan = if let Some(override_plan) = ray_plan_override {
        override_plan.to_vec()
    } else {
        let mut planned = plan_synthetic_rays(
            cut_index,
            scan_leg_count,
            naz,
            config.scan_timing,
            scan_leg.azimuth_rate_deg_per_second,
            cut_start_ms,
        );
        for ray in &mut planned {
            ray.elevation_deg = elevation_deg as f32;
        }
        planned
    };
    if let Some(temporal) = temporal {
        for ray in &mut ray_plan {
            ray.atmosphere_alpha = temporal
                .plan
                .ray_alpha(i64::from(ray.time_offset_ms))
                .map_err(|error| {
                    format!(
                        "temporal weight for cut {} ray {} at {} ms: {error}",
                        cut_index + 1,
                        ray.source_index + 1,
                        ray.time_offset_ms
                    )
                })?;
        }
    }
    let rendered_ray_count = ray_plan.len();

    // One row per radial, sampled in parallel. Each row is `gate_count` REF and
    // `gate_count` VEL f32 values (NaN = no data / below floor / off-domain).
    let row_results: Vec<Result<CutMomentRow, String>> = ray_plan
        .par_iter()
        .map(|ray| -> Result<CutMomentRow, String> {
            if let Some(cancel) = cancel {
                check_synthetic_radar_cancel(cancel)?;
            }
            let iaz = ray.source_index;
            let az_deg = f64::from(ray.azimuth_deg);
            let ray_elevation_deg = f64::from(ray.elevation_deg);
            let mut row = CutMomentRow::blank(gate_count, config.emit_stage_diagnostics);
            let dr_km = (spacing / 1_000.0).max(0.0) as f32;
            let mut previous_kdp = 0.0f32;
            let mut previous_ah = 0.0f32;
            let mut previous_adp = 0.0f32;
            let mut phi_path = 0.0f32;
            let mut tau_h = 0.0f32;
            let mut tau_dp = 0.0f32;
            let quadrature_count = coupled_instrument
                .and_then(|instrument| instrument.quadrature(config.beam_integration))
                .map_or_else(
                    || quadrature_points(config.beam_integration).len(),
                    |points| points.len(),
                );
            let mut refracted_terrain_blockage = (config.terrain_blockage
                && beam_geometry.is_wrf_refractivity())
            .then(|| vec![false; quadrature_count]);
            let raw_cuda = matches!(
                config.atmosphere_time_mode,
                AtmosphereTimeMode::RawStateLinear
            )
            .then(|| config.runtime_cuda_service.as_deref())
            .flatten();
            let raw_gate_chunk = raw_tmatrix_pipeline_gate_chunk(quadrature_count);
            let mut raw_batch = None;
            for gate in 0..gate_count {
                if let Some(cuda) = raw_cuda
                    && gate % raw_gate_chunk == 0
                {
                    let end_gate = gate.saturating_add(raw_gate_chunk).min(gate_count);
                    raw_batch = Some(
                        build_raw_gate_column_batch(
                            fields,
                            temporal.map(|context| context.neighbor),
                            ray.atmosphere_alpha,
                            cells,
                            site_lat,
                            site_lon,
                            antenna_msl,
                            az_deg,
                            ray_elevation_deg,
                            spacing,
                            &gate_range,
                            beam_geometry,
                            config,
                            coupled_instrument,
                            gate,
                            end_gate,
                            cuda,
                            cancel,
                        )
                        .map_err(|error| {
                            format!(
                                "prepare raw T-matrix cut {} ray {} gates {}..{}: {error}",
                                cut_index + 1,
                                iaz + 1,
                                gate + 1,
                                end_gate
                            )
                        })?,
                    );
                }
                if gate % 8 == 0
                    && let Some(cancel) = cancel
                {
                    check_synthetic_radar_cancel(cancel)?;
                }
                let slant_m = f64::from(gate_range.first_gate_m) + gate as f64 * spacing;
                let cached_center_geometry = match raw_batch.as_ref() {
                    Some(batch) => batch.center_geometry(gate)?,
                    None => None,
                };
                let center_geometry = match cached_center_geometry {
                    Some(geometry) => geometry,
                    None => {
                        beam_geometry.point(ray_elevation_deg, &gate_range, gate, 0.0, slant_m)?
                    }
                };
                let beam_height_m = center_geometry.height_above_radar_m;
                let ground_m = center_geometry.ground_range_m;
                let sampled = sample_gate_with_quality_instrument(
                    fields,
                    temporal.map(|context| context.neighbor),
                    ray.atmosphere_alpha,
                    cells,
                    site_lat,
                    site_lon,
                    antenna_msl,
                    az_deg,
                    ray_elevation_deg,
                    slant_m,
                    gate,
                    spacing,
                    &gate_range,
                    beam_geometry,
                    config,
                    terrain_horizon,
                    refracted_terrain_blockage.as_deref_mut(),
                    coupled_instrument,
                    raw_batch.as_ref(),
                    cancel,
                )?;
                if config.emit_quality_fields {
                    row.model_coverage[gate] =
                        encode_quality_fraction(sampled.quality.model_coverage_fraction);
                    row.terrain_unblocked[gate] =
                        encode_quality_fraction(sampled.quality.terrain_unblocked_fraction);
                    row.meteorological_signal[gate] =
                        encode_quality_fraction(sampled.quality.meteorological_signal_fraction);
                }
                if sampled.quality.model_coverage_fraction
                    < config.clamped_minimum_model_coverage_fraction()
                {
                    continue;
                }
                let Some(sample) = sampled.physical else {
                    continue;
                };
                // Reflectivity gate texture (default ON): perturb BEFORE the
                // floor test so echo edges go ragged the way marginal-SNR gates
                // do on a real scope. When off, `dbz` is `sample.dbz` untouched
                // — the output stays bit-identical to a textureless build.
                let intrinsic_dbz = z_to_dbz(sample.z_linear);
                let mut intrinsic_zdr = f32::NAN;
                let mut rho_hv = f32::NAN;
                let mut kdp = f32::NAN;
                let mut ah = f32::NAN;
                let mut adp = f32::NAN;
                let mut pia = f32::NAN;
                let mut pida = f32::NAN;
                let mut phi_dp = f32::NAN;
                if let Some(polar) = sample.polar {
                    intrinsic_zdr = polar.zdr_db;
                    rho_hv = polar.rho_hv;
                    kdp = polar.kdp_deg_km;
                    ah = polar.ah_db_km;
                    adp = polar.ah_db_km - polar.av_db_km;
                    if config.propagation && gate > 0 {
                        phi_path += (previous_kdp + kdp) * dr_km;
                        tau_h += 0.5 * (previous_ah + ah) * dr_km;
                        tau_dp += 0.5 * (previous_adp + adp) * dr_km;
                    }
                    previous_kdp = kdp;
                    previous_ah = ah;
                    previous_adp = adp;
                    pia = if config.propagation { 2.0 * tau_h } else { 0.0 };
                    pida = if config.propagation {
                        2.0 * tau_dp
                    } else {
                        0.0
                    };
                    phi_dp =
                        config.system_phidp_deg + if config.propagation { phi_path } else { 0.0 };
                }
                if let Some(coupled) = coupled_instrument {
                    let propagated_reflectivity =
                        intrinsic_dbz - if pia.is_finite() { pia } else { 0.0 };
                    let ideal = IdealMoments {
                        values: RadarMomentValues {
                            reflectivity_dbz: propagated_reflectivity
                                .is_finite()
                                .then_some(f64::from(propagated_reflectivity)),
                            velocity_mps: sample
                                .velocity_mps
                                .is_finite()
                                .then_some(f64::from(sample.velocity_mps)),
                            spectrum_width_mps: (config.spectrum_width
                                && sample.spectrum_width_mps.is_finite())
                            .then_some(f64::from(sample.spectrum_width_mps.max(0.0))),
                            zdr_db: (config.dual_pol && intrinsic_zdr.is_finite()).then_some(
                                f64::from(
                                    intrinsic_zdr - if pida.is_finite() { pida } else { 0.0 },
                                ),
                            ),
                            rho_hv: (config.dual_pol && rho_hv.is_finite())
                                .then_some(f64::from(rho_hv.clamp(0.0, 1.0))),
                            kdp_deg_km: (config.dual_pol && kdp.is_finite())
                                .then_some(f64::from(kdp)),
                        },
                    };
                    let noise_key = NoiseKey {
                        seed: u64::from(frame_seed),
                        frame: 0,
                        cut: u16::try_from(cut_index).unwrap_or(u16::MAX),
                        ray: u32::try_from(iaz).unwrap_or(u32::MAX),
                        gate: u32::try_from(gate).unwrap_or(u32::MAX),
                    };
                    let measured = estimate_measured_moments(
                        ideal,
                        &coupled.instrument,
                        &coupled.timing,
                        &coupled.estimator_config,
                        slant_m,
                        noise_key,
                    )
                    .map_err(|error| {
                        format!(
                            "estimate coupled moments for cut {} ray {} gate {}: {error}",
                            cut_index + 1,
                            iaz + 1,
                            gate + 1
                        )
                    })?;
                    if let Some(diagnostics) = row.stage_diagnostics.as_mut() {
                        diagnostics.record(gate, ideal, measured);
                    }
                    let clutter_reflectivity_dbz = clutter_on
                        .then(|| {
                            ground_clutter_dbz(
                                frame_seed,
                                cut_index,
                                iaz,
                                gate,
                                az_deg as f32,
                                (ground_m / 1_000.0) as f32,
                                beam_height_m as f32,
                                in_clutter_hotspot(&hotspots, iaz, gate),
                                clutter_amount,
                            )
                        })
                        .flatten();
                    let presentation = PresentationConfig {
                        reflectivity_texture_sigma_db: if config.ref_gate_texture {
                            f64::from(REF_TEXTURE_CORRELATED_DB.hypot(REF_TEXTURE_JITTER_DB))
                        } else {
                            0.0
                        },
                        velocity_texture_sigma_mps: if config.vel_gate_texture {
                            f64::from(VEL_TEXTURE_MPS)
                        } else {
                            0.0
                        },
                        zdr_display_bias_db: 0.0,
                        reflectivity_display_floor_dbz: None,
                        clutter_reflectivity_dbz: clutter_reflectivity_dbz.map(f64::from),
                        clutter_velocity_mps: if clutter_reflectivity_dbz.is_some() {
                            f64::from(clutter_velocity_mps(frame_seed, cut_index, iaz, gate))
                        } else {
                            0.0
                        },
                    };
                    let presented: PresentedMoments =
                        present_measured_moments(measured, &presentation, noise_key);
                    let mut presented_values = presented.values;
                    if let Some(velocity) = presented_values.velocity_mps.as_mut() {
                        *velocity = f64::from(fold_velocity_mps(
                            *velocity as f32,
                            coupled.stamped_nyquist_mps(),
                        ));
                    }
                    if presented.adjustment.clutter_replaced
                        && let Some(width) = presented_values.spectrum_width_mps.as_mut()
                    {
                        *width = (*width).max(f64::from(config.spectrum_width_floor_mps.max(0.0)));
                    }
                    if let Some(value) = presented_values.reflectivity_dbz {
                        row.reflectivity[gate] = value as f32;
                    }
                    if let Some(value) = presented_values.velocity_mps {
                        row.velocity[gate] = value as f32;
                    }
                    if let Some(value) = presented_values.spectrum_width_mps {
                        row.spectrum_width[gate] = value as f32;
                    }
                    if config.dual_pol
                        && !presented.adjustment.clutter_replaced
                        && presented_values.reflectivity_dbz.is_some()
                    {
                        if let Some(value) = presented_values.zdr_db {
                            row.zdr[gate] = value as f32;
                        }
                        if let Some(value) = presented_values.rho_hv {
                            row.rho_hv[gate] = value as f32;
                        }
                        if let Some(value) = presented_values.kdp_deg_km {
                            row.kdp[gate] = value as f32;
                        }
                        row.phi_dp[gate] = phi_dp;
                        row.ah[gate] = ah;
                        row.pia[gate] = pia;
                        row.ref_corrected[gate] = intrinsic_dbz;
                        row.adp[gate] = adp;
                        row.pida[gate] = pida;
                        row.zdr_corrected[gate] = intrinsic_zdr;
                    }
                    continue;
                }
                let mut dbz = intrinsic_dbz - if pia.is_finite() { pia } else { 0.0 };
                if config.ref_gate_texture {
                    dbz += ref_gate_texture_db(cut_index, iaz, gate);
                }
                let range_km = (slant_m / 1_000.0).max(1.0) as f32;
                let sensitivity = config.sensitivity_dbz_at_1km + 20.0 * range_km.log10();
                if config.instrument_noise {
                    let snr_db = dbz - sensitivity;
                    if !snr_db.is_finite() || snr_db < 0.0 {
                        continue;
                    }
                    let sigma_db = (1.5 / (1.0 + snr_db / 6.0)).clamp(0.12, 1.5);
                    dbz += sigma_db * clutter_signed(frame_seed, cut_index, iaz, gate, 0x534e_525a);
                }
                // Physics reflectivity + velocity for this gate. Both stay NaN
                // below the floor (clear air) so ground clutter can fill them,
                // exactly as the community operator fills the clear-air gates
                // near the radar. With clutter off, `ref_val`/`vel_val` are
                // committed unchanged, so the output is bit-identical.
                let mut ref_val = f32::NAN;
                let mut vel_val = f32::NAN;
                let mut sw_val = f32::NAN;
                let mut clutter_replaced = false;
                if dbz.is_finite() && dbz >= floor {
                    ref_val = dbz;
                    // Radial velocity: wind projected onto the beam unit vector
                    // (east, north, up) = (sinAz·cosEl, cosAz·cosEl, sinEl), with
                    // azimuth clockwise from north. Positive = away from the radar
                    // (NEXRAD convention). Sun & Crook (1997).
                    let mut vr = sample.velocity_mps;
                    let mut velocity_noise_variance = 0.0f32;
                    if config.instrument_noise {
                        let snr_db = (dbz - sensitivity).max(0.0);
                        let sigma_velocity = (1.2 / (1.0 + snr_db / 8.0)).clamp(0.08, 1.2);
                        velocity_noise_variance = sigma_velocity * sigma_velocity;
                        vr += sigma_velocity
                            * clutter_signed(frame_seed, cut_index, iaz, gate, 0x534e_5256);
                    }
                    if vr.is_finite() {
                        // Velocity gate texture (default OFF): opt-in wobble; the
                        // clean Vr is what dealias/GBVTD consume.
                        vel_val = if config.vel_gate_texture {
                            vr + vel_gate_texture_mps(cut_index, iaz, gate)
                        } else {
                            vr
                        };
                    }
                    if config.spectrum_width && sample.spectrum_width_mps.is_finite() {
                        sw_val = (sample.spectrum_width_mps.max(0.0).powi(2)
                            + velocity_noise_variance)
                            .sqrt();
                    }
                }

                // Ground clutter (opt-in), applied AFTER the reflectivity floor
                // and only into gates weaker than the clutter value, so storms
                // are never overwritten (the community rule).
                if clutter_on
                    && let Some(cv) = ground_clutter_dbz(
                        frame_seed,
                        cut_index,
                        iaz,
                        gate,
                        az_deg as f32,
                        (ground_m / 1000.0) as f32,
                        beam_height_m as f32,
                        in_clutter_hotspot(&hotspots, iaz, gate),
                        clutter_amount,
                    )
                    && (!ref_val.is_finite() || ref_val < cv)
                {
                    clutter_replaced = true;
                    ref_val = cv;
                    // The ground is stationary: where clutter dominates a gate
                    // that already carried a velocity, replace the wind
                    // projection with a near-zero return. A clear-air clutter
                    // gate had no Vr to speak of, so it is left blank (velocity
                    // is gated on data existing).
                    if vel_val.is_finite() {
                        vel_val = clutter_velocity_mps(frame_seed, cut_index, iaz, gate);
                        if config.spectrum_width {
                            sw_val = sw_val.max(config.spectrum_width_floor_mps.max(0.0));
                        }
                    }
                }

                // Realistic Nyquist (opt-in): alias the true Vr into the folding
                // co-interval, AFTER the clutter VEL replacement. Applied last so
                // it folds exactly what the display will store. Clutter gates are
                // ~0 ± 0.5 m/s — well inside any sane Nyquist — so folding leaves
                // them untouched. With folding OFF this branch never runs, so the
                // output is bit-identical to a build without the feature.
                if config.fold_velocity && vel_val.is_finite() {
                    vel_val = fold_velocity_mps(vel_val, config.nyquist_mps);
                }

                if ref_val.is_finite() {
                    row.reflectivity[gate] = ref_val;
                }
                if vel_val.is_finite() {
                    row.velocity[gate] = vel_val;
                }
                if sw_val.is_finite() {
                    row.spectrum_width[gate] = sw_val;
                }
                if config.dual_pol
                    && !clutter_replaced
                    && ref_val.is_finite()
                    && intrinsic_zdr.is_finite()
                {
                    row.zdr[gate] = intrinsic_zdr + config.zdr_bias_db - pida;
                    row.rho_hv[gate] = rho_hv.clamp(0.0, 1.0);
                    row.phi_dp[gate] = phi_dp;
                    row.kdp[gate] = kdp;
                    row.ah[gate] = ah;
                    row.pia[gate] = pia;
                    row.ref_corrected[gate] = intrinsic_dbz;
                    row.adp[gate] = adp;
                    row.pida[gate] = pida;
                    row.zdr_corrected[gate] = intrinsic_zdr;
                }
            }
            Ok(row)
        })
        .collect();
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    // Rayon completion order must not select the externally visible failure.
    // Resolve indexed radial slots serially so the first ray in cut order wins.
    let rows = row_results.into_iter().collect::<Result<Vec<_>, _>>()?;

    let mut cut = ElevationCut::new(elevation_deg as f32, u8::try_from(cut_index + 1).ok());
    let mut ref_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut vel_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut sw_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut zdr_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut rho_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut phi_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut kdp_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut ah_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut pia_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut refc_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut adp_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut pida_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut zdrc_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut model_coverage_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut terrain_unblocked_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut meteorological_signal_values = Vec::with_capacity(rendered_ray_count * gate_count);
    let mut stage_diagnostics = config
        .emit_stage_diagnostics
        .then(|| StageDiagnosticRow::with_capacity(rendered_ray_count * gate_count));
    for (ray, row) in ray_plan.into_iter().zip(rows) {
        cut.radials.push(Radial {
            azimuth_deg: ray.azimuth_deg,
            elevation_deg: ray.elevation_deg,
            time_offset_ms: ray.time_offset_ms,
            gate_range: gate_range.clone(),
            nyquist_velocity_mps: Some(coupled_instrument.map_or_else(
                || config.stamped_nyquist_mps(),
                CoupledInstrumentContext::stamped_nyquist_mps,
            )),
            radial_status: Some(ray.radial_status),
        });
        if let Some(coupled) = coupled_instrument {
            cut.ray_instrument_metadata.push(RayInstrumentMetadata {
                prt_s: Some(coupled.timing.prt_s as f32),
                unambiguous_range_km: Some((coupled.timing.unambiguous_range_m / 1_000.0) as f32),
                pulse_count: Some(coupled.sampling.transmitted_pulses),
                independent_samples: Some(coupled.sampling.independent_samples as f32),
            });
        }
        ref_values.extend(row.reflectivity);
        vel_values.extend(row.velocity);
        sw_values.extend(row.spectrum_width);
        zdr_values.extend(row.zdr);
        rho_values.extend(row.rho_hv);
        phi_values.extend(row.phi_dp);
        kdp_values.extend(row.kdp);
        ah_values.extend(row.ah);
        pia_values.extend(row.pia);
        refc_values.extend(row.ref_corrected);
        adp_values.extend(row.adp);
        pida_values.extend(row.pida);
        zdrc_values.extend(row.zdr_corrected);
        model_coverage_values.extend(row.model_coverage);
        terrain_unblocked_values.extend(row.terrain_unblocked);
        meteorological_signal_values.extend(row.meteorological_signal);
        if let (Some(all_diagnostics), Some(row_diagnostics)) =
            (stage_diagnostics.as_mut(), row.stage_diagnostics)
        {
            all_diagnostics.append(row_diagnostics);
        }
    }

    let radial_indices: Vec<usize> = (0..rendered_ray_count).collect();
    if scan_leg.moments.has_reflectivity() {
        cut.moments.insert(
            MomentType::Reflectivity,
            f32_grid(
                MomentType::Reflectivity,
                gate_range.clone(),
                radial_indices.clone(),
                ref_values,
            ),
        );
    }
    if scan_leg.moments.has_velocity() {
        cut.moments.insert(
            MomentType::Velocity,
            f32_grid(
                MomentType::Velocity,
                gate_range.clone(),
                radial_indices.clone(),
                vel_values,
            ),
        );
    }
    if config.spectrum_width && scan_leg.moments.has_spectrum_width() {
        cut.moments.insert(
            MomentType::SpectrumWidth,
            f32_grid(
                MomentType::SpectrumWidth,
                gate_range.clone(),
                radial_indices.clone(),
                sw_values,
            ),
        );
    }
    if config.dual_pol && fields.has_polarimetric_input() && scan_leg.moments.has_reflectivity() {
        let polar_moments = [
            (MomentType::DifferentialReflectivity, zdr_values),
            (MomentType::CorrelationCoefficient, rho_values),
            (MomentType::DifferentialPhase, phi_values),
            (MomentType::SpecificDifferentialPhase, kdp_values),
            (MomentType::Unknown("AH".to_string()), ah_values),
            (MomentType::Unknown("PIA".to_string()), pia_values),
            (MomentType::Unknown("REFC".to_string()), refc_values),
            (MomentType::Unknown("ADP".to_string()), adp_values),
            (MomentType::Unknown("PIDA".to_string()), pida_values),
            (MomentType::Unknown("ZDRC".to_string()), zdrc_values),
        ];
        for (moment, values) in polar_moments {
            cut.moments.insert(
                moment.clone(),
                f32_grid(moment, gate_range.clone(), radial_indices.clone(), values),
            );
        }
    }
    if config.emit_quality_fields {
        for (quality, values) in [
            (QualityMoment::ModelCoverage, model_coverage_values),
            (QualityMoment::TerrainUnblocked, terrain_unblocked_values),
            (
                QualityMoment::MeteorologicalSignal,
                meteorological_signal_values,
            ),
        ] {
            let grid =
                compact_quality_grid(quality, gate_range.clone(), radial_indices.clone(), values)?;
            cut.moments.insert(grid.moment.clone(), grid);
        }
    }
    if let Some(diagnostics) = stage_diagnostics {
        for (name, values) in [
            ("IREF", diagnostics.ideal_reflectivity),
            ("IVEL", diagnostics.ideal_velocity),
            ("ISW", diagnostics.ideal_spectrum_width),
            ("IZDR", diagnostics.ideal_zdr),
            ("IRHO", diagnostics.ideal_rho_hv),
            ("IKDP", diagnostics.ideal_kdp),
            ("MREF", diagnostics.measured_reflectivity),
            ("MVEL", diagnostics.measured_velocity),
            ("MSW", diagnostics.measured_spectrum_width),
            ("MZDR", diagnostics.measured_zdr),
            ("MRHO", diagnostics.measured_rho_hv),
            ("MKDP", diagnostics.measured_kdp),
        ] {
            let moment = MomentType::Unknown(name.to_string());
            cut.moments.insert(
                moment.clone(),
                f32_grid(moment, gate_range.clone(), radial_indices.clone(), values),
            );
        }
    }
    Ok(cut)
}

fn f32_grid(
    moment: MomentType,
    gate_range: GateRange,
    radial_indices: Vec<usize>,
    values: Vec<f32>,
) -> MomentGrid {
    // True physical units: dBZ / m·s⁻¹ stored directly (scale 1, offset 0), so
    // the render/dealias/GBVTD F32 paths and the standard colour tables read
    // them without a raw→scaled conversion.
    MomentGrid {
        moment,
        gate_range,
        scale: 1.0,
        offset: 0.0,
        nodata: None,
        range_folded: None,
        radial_indices,
        storage: MomentStorage::F32(values),
    }
}

/// One trilinearly-sampled model column value at a gate.
#[derive(Clone, Copy)]
struct ColumnSample {
    /// Equivalent reflectivity factor in linear mm^6 m^-3. Keeping received
    /// power linear through every interpolation and pulse-volume average is the
    /// core scientific correction; convert to dBZ only at the finished gate.
    z_linear: f32,
    u: f32,
    v: f32,
    w: f32,
    polar: Option<crate::wrf_radar_physics::IntrinsicPolarSample>,
    tke_m2s2: f32,
}

// These inputs are kept explicit because they cross the spatial, beam and
// atmosphere-time sampling boundary; bundling them would hide unit contracts.
#[allow(clippy::too_many_arguments)]
fn sample_column_temporal(
    fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    alpha: f64,
    cells: usize,
    lat: f32,
    lon: f32,
    z_msl: f32,
    beam_elevation_deg: f64,
    atmosphere_time_mode: AtmosphereTimeMode,
    reflectivity_sampling: ReflectivitySampling,
    cuda: Option<&app_ui::wrf_tmatrix_cuda::WrfTMatrixCudaBatchService>,
    cancel: Option<&AtomicBool>,
) -> Result<Option<ColumnSample>, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(format!(
            "temporal atmosphere weight {alpha} is outside the closed [0,1] interval"
        ));
    }
    if matches!(atmosphere_time_mode, AtmosphereTimeMode::RawStateLinear) {
        return sample_column_raw_state_linear(
            fields,
            neighbor_fields,
            alpha,
            cells,
            lat,
            lon,
            z_msl,
            beam_elevation_deg,
            cuda,
            cancel,
        );
    }
    if alpha <= 0.0 {
        return sample_column(
            fields,
            cells,
            lat,
            lon,
            z_msl,
            beam_elevation_deg,
            reflectivity_sampling,
        );
    }
    let neighbor = neighbor_fields.ok_or_else(|| {
        "temporal sampling has a positive atmosphere weight but no adjacent WRF scene".to_string()
    })?;
    if alpha >= 1.0 {
        return sample_column(
            neighbor,
            cells,
            lat,
            lon,
            z_msl,
            beam_elevation_deg,
            reflectivity_sampling,
        );
    }
    let Some(left) = sample_column(
        fields,
        cells,
        lat,
        lon,
        z_msl,
        beam_elevation_deg,
        reflectivity_sampling,
    )?
    else {
        return Ok(None);
    };
    let Some(right) = sample_column(
        neighbor,
        cells,
        lat,
        lon,
        z_msl,
        beam_elevation_deg,
        reflectivity_sampling,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(blend_column_samples(left, right, alpha as f32)))
}

/// Blend quantities that remain linear/additive through the radar operator.
/// ZH/ZV/covariance/KDP/attenuation are combined as additive scattering
/// quantities; ratios such as ZDR and rhoHV are derived only afterwards.
fn blend_column_samples(left: ColumnSample, right: ColumnSample, alpha: f32) -> ColumnSample {
    let alpha = alpha.clamp(0.0, 1.0);
    let beta = 1.0 - alpha;
    let polar = match (left.polar, right.polar) {
        (Some(left), Some(right)) => {
            let mut accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
            accumulator.add(beta, intrinsic_as_contribution(left));
            accumulator.add(alpha, intrinsic_as_contribution(right));
            Some(accumulator.finalize())
        }
        (Some(left), None) => {
            let mut accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
            accumulator.add(beta, intrinsic_as_contribution(left));
            Some(accumulator.finalize())
        }
        (None, Some(right)) => {
            let mut accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
            accumulator.add(alpha, intrinsic_as_contribution(right));
            Some(accumulator.finalize())
        }
        (None, None) => None,
    };
    ColumnSample {
        z_linear: beta * left.z_linear + alpha * right.z_linear,
        u: beta * left.u + alpha * right.u,
        v: beta * left.v + alpha * right.v,
        w: beta * left.w + alpha * right.w,
        polar,
        tke_m2s2: beta * left.tke_m2s2 + alpha * right.tke_m2s2,
    }
}

struct RawSpatialEndpoint {
    property_weights: [(usize, f64); 8],
    property_weight_count: usize,
    gate_state: RawGateState,
    tke_m2s2: f32,
}

struct PreparedRawColumnSample {
    property_cell: app_ui::wrf_property_reader::RawPropertyCell,
    gate_state: RawGateState,
    tke_m2s2: f32,
}

#[allow(clippy::too_many_arguments)]
fn sample_column_raw_state_linear(
    fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    alpha: f64,
    cells: usize,
    lat: f32,
    lon: f32,
    z_msl: f32,
    beam_elevation_deg: f64,
    cuda: Option<&app_ui::wrf_tmatrix_cuda::WrfTMatrixCudaBatchService>,
    cancel: Option<&AtomicBool>,
) -> Result<Option<ColumnSample>, String> {
    let Some(prepared) = prepare_column_raw_state_linear(
        fields,
        neighbor_fields,
        alpha,
        cells,
        lat,
        lon,
        z_msl,
        cancel,
    )?
    else {
        return Ok(None);
    };
    let polar =
        app_ui::wrf_tmatrix_assets::evaluate_embedded_raw_property_cell_with_cuda_and_cancel(
            &prepared.property_cell,
            beam_elevation_deg,
            cuda,
            cancel,
        )?;
    finish_prepared_raw_column(prepared, polar, cancel).map(Some)
}

#[allow(clippy::too_many_arguments)]
fn prepare_column_raw_state_linear(
    fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    alpha: f64,
    cells: usize,
    lat: f32,
    lon: f32,
    z_msl: f32,
    cancel: Option<&AtomicBool>,
) -> Result<Option<PreparedRawColumnSample>, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let left = if alpha < 1.0 {
        raw_spatial_endpoint(fields, cells, lat, lon, z_msl)?
    } else {
        None
    };
    let right = if alpha > 0.0 {
        let neighbor = neighbor_fields.ok_or_else(|| {
            "RawStateLinear has a positive temporal weight but no adjacent WRF scene".to_string()
        })?;
        raw_spatial_endpoint(neighbor, cells, lat, lon, z_msl)?
    } else {
        None
    };
    if (alpha < 1.0 && left.is_none()) || (alpha > 0.0 && right.is_none()) {
        return Ok(None);
    }
    let left_endpoint = left.as_ref().map(|endpoint| {
        RawStateLinearEndpoint::with_spatial_weights(
            fields
                .raw_property_scene
                .as_deref()
                .expect("raw spatial endpoint requires a raw property scene"),
            &endpoint.property_weights[..endpoint.property_weight_count],
            &endpoint.gate_state,
        )
    });
    let right_endpoint = right.as_ref().map(|endpoint| {
        RawStateLinearEndpoint::with_spatial_weights(
            neighbor_fields
                .and_then(|neighbor| neighbor.raw_property_scene.as_deref())
                .expect("raw adjacent endpoint requires a raw property scene"),
            &endpoint.property_weights[..endpoint.property_weight_count],
            &endpoint.gate_state,
        )
    });
    let blended = interpolate_raw_state_linear(left_endpoint, right_endpoint, alpha)
        .map_err(|error| format!("RawStateLinear pre-closure blend: {error}"))?;
    let tke_m2s2 = match (left.as_ref(), right.as_ref()) {
        (Some(left), Some(right)) => {
            (f64::from(left.tke_m2s2)
                + (f64::from(right.tke_m2s2) - f64::from(left.tke_m2s2)) * alpha) as f32
        }
        (Some(left), None) => left.tke_m2s2,
        (None, Some(right)) => right.tke_m2s2,
        (None, None) => return Ok(None),
    };
    Ok(Some(PreparedRawColumnSample {
        property_cell: blended.property_cell,
        gate_state: blended.gate_state,
        tke_m2s2,
    }))
}

fn finish_prepared_raw_column(
    prepared: PreparedRawColumnSample,
    polar: Option<radar_scattering::PolarAccumulatorQuantities>,
    cancel: Option<&AtomicBool>,
) -> Result<ColumnSample, String> {
    if let Some(cancel) = cancel {
        check_synthetic_radar_cancel(cancel)?;
    }
    let polar = polar.map(|quantities| {
        let mut accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
        accumulator.add(1.0, tmatrix_as_contribution(quantities));
        accumulator.finalize()
    });
    Ok(ColumnSample {
        z_linear: polar.map_or(0.0, |sample| sample.zh),
        u: prepared.gate_state.wind_u_mps,
        v: prepared.gate_state.wind_v_mps,
        w: prepared.gate_state.wind_w_mps,
        polar,
        tke_m2s2: prepared.tke_m2s2,
    })
}

fn raw_spatial_endpoint(
    fields: &WrfRadarFields,
    cells: usize,
    lat: f32,
    lon: f32,
    z_msl: f32,
) -> Result<Option<RawSpatialEndpoint>, String> {
    let property_scene = fields.raw_property_scene.as_deref().ok_or_else(|| {
        "RawStateLinear requires retained native P3/ISHMAEL property state".to_string()
    })?;
    if property_scene.cell_count() != fields.nx * fields.ny * fields.nz {
        return Err(format!(
            "RawStateLinear property coverage has {} cells, expected {}",
            property_scene.cell_count(),
            fields.nx * fields.ny * fields.nz
        ));
    }
    let Some(stencil) = horizontal_stencil(fields, lat, lon) else {
        return Ok(None);
    };
    let mut property_weights = [(0usize, 0.0f64); 8];
    let mut property_weight_count = 0usize;
    let mut wind_u = 0.0f64;
    let mut wind_v = 0.0f64;
    let mut wind_w = 0.0f64;
    let mut tke = 0.0f64;
    let mut weight_sum = 0.0f64;
    for (column, horizontal_weight) in stencil {
        if horizontal_weight <= 0.0 {
            continue;
        }
        let Some((k, vertical_fraction)) = bracket_height(fields, cells, column, z_msl) else {
            continue;
        };
        let lower = k * cells + column;
        let upper = (k + 1) * cells + column;
        let (Some(&u0), Some(&u1), Some(&v0), Some(&v1), Some(&w0), Some(&w1)) = (
            fields.u.get(lower),
            fields.u.get(upper),
            fields.v.get(lower),
            fields.v.get(upper),
            fields.w.get(lower),
            fields.w.get(upper),
        ) else {
            return Err("RawStateLinear wind coverage is internally inconsistent".to_string());
        };
        if [u0, u1, v0, v1, w0, w1]
            .into_iter()
            .any(|value| !value.is_finite())
        {
            continue;
        }
        let lower_weight = f64::from(horizontal_weight) * f64::from(1.0 - vertical_fraction);
        let upper_weight = f64::from(horizontal_weight) * f64::from(vertical_fraction);
        for (index, weight, u, v, w) in [
            (lower, lower_weight, u0, v0, w0),
            (upper, upper_weight, u1, v1, w1),
        ] {
            if weight <= 0.0 {
                continue;
            }
            if property_weight_count >= property_weights.len() {
                return Err("RawStateLinear spatial stencil exceeded eight cells".to_string());
            }
            property_weights[property_weight_count] = (index, weight);
            property_weight_count += 1;
            weight_sum += weight;
            wind_u += weight * f64::from(u);
            wind_v += weight * f64::from(v);
            wind_w += weight * f64::from(w);
            if let Some(tke_grid) = &fields.tke_tenths_m2s2 {
                let value = tke_grid.get(index).copied().ok_or_else(|| {
                    "RawStateLinear TKE coverage is internally inconsistent".to_string()
                })?;
                tke += weight * f64::from(value) * 0.1;
            }
        }
    }
    if property_weight_count == 0 || !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Ok(None);
    }
    for (_, weight) in &mut property_weights[..property_weight_count] {
        *weight /= weight_sum;
    }
    let normalized_sum = property_weights[..property_weight_count]
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<f64>();
    property_weights[property_weight_count - 1].1 += 1.0 - normalized_sum;

    let (first_cell_index, first_weight) = property_weights[0];
    let mut endpoint_samples = [app_ui::wrf_property_reader::WeightedRawPropertyCell::new(
        property_scene,
        first_cell_index,
        first_weight,
    ); 8];
    for (sample, &(cell_index, weight)) in endpoint_samples
        .iter_mut()
        .zip(&property_weights[..property_weight_count])
    {
        *sample = app_ui::wrf_property_reader::WeightedRawPropertyCell::new(
            property_scene,
            cell_index,
            weight,
        );
    }
    let endpoint_environment = app_ui::wrf_property_reader::blend_raw_property_environment(
        &endpoint_samples[..property_weight_count],
    )
    .map_err(|error| format!("RawStateLinear spatial environment blend: {error}"))?;
    Ok(Some(RawSpatialEndpoint {
        property_weights,
        property_weight_count,
        gate_state: RawGateState {
            wind_u_mps: (wind_u / weight_sum) as f32,
            wind_v_mps: (wind_v / weight_sum) as f32,
            wind_w_mps: (wind_w / weight_sum) as f32,
            temperature_k: endpoint_environment.environment().temperature_k() as f32,
            pressure_pa: endpoint_environment.pressure_pa() as f32,
            air_density_kgm3: endpoint_environment.environment().air_density_kg_m3() as f32,
            fields: Default::default(),
        },
        tke_m2s2: (tke / weight_sum) as f32,
    }))
}

fn intrinsic_as_contribution(
    sample: crate::wrf_radar_physics::IntrinsicPolarSample,
) -> crate::wrf_radar_physics::BulkContribution {
    crate::wrf_radar_physics::BulkContribution {
        zh: sample.zh,
        zv: sample.zv,
        cov_re: sample.cov_re,
        cov_im: sample.cov_im,
        kdp_deg_km: sample.kdp_deg_km,
        ah_db_km: sample.ah_db_km,
        av_db_km: sample.av_db_km,
        fall_speed_mps: sample.fall_speed_mps,
        fall_speed_variance_m2s2: sample.fall_speed_variance_m2s2,
    }
}

fn tmatrix_as_contribution(
    sample: radar_scattering::PolarAccumulatorQuantities,
) -> crate::wrf_radar_physics::BulkContribution {
    crate::wrf_radar_physics::BulkContribution {
        zh: sample.zh,
        zv: sample.zv,
        cov_re: sample.cov_re,
        cov_im: sample.cov_im,
        kdp_deg_km: sample.kdp_deg_km,
        ah_db_km: sample.ah_db_km,
        av_db_km: sample.av_db_km,
        fall_speed_mps: sample.fall_speed_mps,
        fall_speed_variance_m2s2: sample.fall_speed_variance_m2s2,
    }
}

fn normalize_intrinsic(
    mut sample: crate::wrf_radar_physics::IntrinsicPolarSample,
    divisor: f32,
) -> crate::wrf_radar_physics::IntrinsicPolarSample {
    if divisor.is_finite() && divisor > 0.0 {
        sample.zh /= divisor;
        sample.zv /= divisor;
        sample.cov_re /= divisor;
        sample.cov_im /= divisor;
        sample.covariance_magnitude /= divisor;
        sample.kdp_deg_km /= divisor;
        sample.ah_db_km /= divisor;
        sample.av_db_km /= divisor;
    }
    sample
}

/// Sample the 3-D model fields at (lat, lon, MSL height) by horizontal 2×2
/// bilinear weights (over the curvilinear WRF grid) combined with a
/// per-corner vertical bracket in MSL height — i.e. trilinear. Returns `None`
/// off the domain or when the height sits below terrain / above the model top
/// at every contributing corner.
fn sample_column(
    fields: &WrfRadarFields,
    cells: usize,
    lat: f32,
    lon: f32,
    z_msl: f32,
    beam_elevation_deg: f64,
    reflectivity_sampling: ReflectivitySampling,
) -> Result<Option<ColumnSample>, String> {
    let Some(stencil) = horizontal_stencil(fields, lat, lon) else {
        return Ok(None);
    };

    let mut wsum = 0.0f32;
    let mut reflectivity = 0.0f32;
    let mut u = 0.0f32;
    let mut v = 0.0f32;
    let mut w = 0.0f32;
    let mut tke = 0.0f32;
    let mut polar_accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
    let property_scattering = fields.property_scattering.as_deref();
    let mut property_weights = [(0usize, 0.0f64); 8];
    let mut property_weight_count = 0usize;
    for (col, weight) in stencil {
        if weight <= 0.0 {
            continue;
        }
        let Some((k, t)) = bracket_height(fields, cells, col, z_msl) else {
            continue;
        };
        let i0 = k * cells + col;
        let i1 = (k + 1) * cells + col;
        let (Some(su), Some(sv), Some(sw)) = (
            lerp(fields.u[i0], fields.u[i1], t),
            lerp(fields.v[i0], fields.v[i1], t),
            lerp(fields.w[i0], fields.w[i1], t),
        ) else {
            continue;
        };

        if property_scattering.is_some() {
            let lower_weight = f64::from(weight) * f64::from(1.0 - t);
            if lower_weight > 0.0 {
                property_weights[property_weight_count] = (i0, lower_weight);
                property_weight_count += 1;
            }
            let upper_weight = f64::from(weight) * f64::from(t);
            if upper_weight > 0.0 {
                property_weights[property_weight_count] = (i1, upper_weight);
                property_weight_count += 1;
            }
        } else {
            let Some(d) = (match reflectivity_sampling {
                ReflectivitySampling::LegacyDbz => lerp(fields.dbz[i0], fields.dbz[i1], t),
                ReflectivitySampling::LinearZ => {
                    lerp(dbz_to_z(fields.dbz[i0]), dbz_to_z(fields.dbz[i1]), t)
                }
            }) else {
                continue;
            };
            reflectivity += weight * d;
        }

        if property_scattering.is_none()
            && let Some(polar) = &fields.polarimetric
        {
            let mut vertical = crate::wrf_radar_physics::PolarAccumulator::default();
            vertical.add(1.0 - t, polar.contribution_at(i0, dbz_to_z(fields.dbz[i0]), beam_elevation_deg as f32));
            vertical.add(t, polar.contribution_at(i1, dbz_to_z(fields.dbz[i1]), beam_elevation_deg as f32));
            polar_accumulator.add(weight, intrinsic_as_contribution(vertical.finalize()));
        }
        if let Some(tke_grid) = &fields.tke_tenths_m2s2 {
            let t0 = tke_grid[i0] as f32 * 0.1;
            let t1 = tke_grid[i1] as f32 * 0.1;
            tke += weight * (t0 + t * (t1 - t0));
        }
        wsum += weight;
        u += weight * su;
        v += weight * sv;
        w += weight * sw;
    }
    if wsum <= 0.0 {
        return Ok(None);
    }
    let (z_linear, polar) = if let Some(property) = property_scattering {
        if property_weight_count == 0 {
            return Ok(None);
        }
        let weights = &mut property_weights[..property_weight_count];
        let weight_sum = weights.iter().map(|(_, weight)| *weight).sum::<f64>();
        if !weight_sum.is_finite() || weight_sum <= 0.0 {
            return Err(format!(
                "property T-matrix trilinear weights have invalid sum {weight_sum}"
            ));
        }
        for (_, weight) in weights.iter_mut() {
            *weight /= weight_sum;
        }
        let normalized_sum = weights.iter().map(|(_, weight)| *weight).sum::<f64>();
        if let Some((_, last_weight)) = weights.last_mut() {
            *last_weight += 1.0 - normalized_sum;
        }
        let quantities = property
            .weighted_polar_at(weights, beam_elevation_deg)
            .map_err(|error| format!("query weighted property T-matrix column: {error}"))?;
        let mut accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
        accumulator.add(1.0, tmatrix_as_contribution(quantities));
        let sample = accumulator.finalize();
        (sample.zh, (sample.zh > 0.0).then_some(sample))
    } else {
        let reflectivity = reflectivity / wsum;
        let z_linear = match reflectivity_sampling {
            ReflectivitySampling::LegacyDbz => dbz_to_z(reflectivity),
            ReflectivitySampling::LinearZ => reflectivity,
        };
        let polar = fields.polarimetric.as_ref().and_then(|_| {
            let sample = normalize_intrinsic(polar_accumulator.finalize(), wsum);
            (sample.zh > 0.0).then_some(sample)
        });
        (z_linear, polar)
    };
    Ok(Some(ColumnSample {
        z_linear: polar.map_or(z_linear, |sample| sample.zh),
        u: u / wsum,
        v: v / wsum,
        w: w / wsum,
        polar,
        tke_m2s2: tke / wsum,
    }))
}

#[inline]
fn dbz_to_z(dbz: f32) -> f32 {
    if dbz.is_finite() {
        crate::wrf_radar_physics::dbz_to_z(dbz)
    } else {
        f32::NAN
    }
}

#[inline]
fn z_to_dbz(z: f32) -> f32 {
    if z.is_finite() && z > 0.0 {
        10.0 * z.log10()
    } else {
        f32::NAN
    }
}

/// Up to four `(column index, horizontal weight)` pairs for the WRF cell
/// containing (lat, lon), via the inverse LUT + a 2×2 bilinear solve. Falls
/// back to nearest-neighbour when the point is not cleanly inside a cell.
fn horizontal_stencil(fields: &WrfRadarFields, lat: f32, lon: f32) -> Option<[(usize, f32); 4]> {
    let nx = fields.nx;
    let ny = fields.ny;
    let nearest = fields.lut.lookup(lat, lon)?;
    if nx < 2 || ny < 2 {
        return Some([
            (nearest, 1.0),
            (nearest, 0.0),
            (nearest, 0.0),
            (nearest, 0.0),
        ]);
    }
    if let Some(stencil) = horizontal_stencil_near(fields, lat, lon, nearest) {
        return Some(stencil);
    }
    // A LUT bin spans more than one model cell and retains one seed. Its
    // seed can be farther than the four surrounding cells, especially after
    // hole filling. Descend through geographic neighbors until a containing
    // cell is found. Distance decreases strictly, so this cannot cycle.
    let distance = |index: usize| {
        let dy = f64::from(fields.lat[index]) - f64::from(lat);
        let dx = unwrap_lon_near(f64::from(fields.lon[index]), f64::from(lon))
            - f64::from(lon);
        dx * dx + dy * dy
    };
    let mut current = nearest;
    let mut best_distance = distance(current);
    loop {
        let row = current / nx;
        let col = current % nx;
        let mut next = current;
        for y in row.saturating_sub(1)..=(row + 1).min(ny - 1) {
            for x in col.saturating_sub(1)..=(col + 1).min(nx - 1) {
                let candidate = y * nx + x;
                let candidate_distance = distance(candidate);
                if candidate_distance < best_distance {
                    next = candidate;
                    best_distance = candidate_distance;
                }
            }
        }
        if next == current {
            // No cell contains this coordinate. A distant point value would
            // create constant wedges across otherwise smooth model fields.
            return None;
        }
        if let Some(stencil) = horizontal_stencil_near(fields, lat, lon, next) {
            return Some(stencil);
        }
        current = next;
    }
}

fn horizontal_stencil_near(
    fields: &WrfRadarFields,
    lat: f32,
    lon: f32,
    nearest: usize,
) -> Option<[(usize, f32); 4]> {
    let nx = fields.nx;
    let ny = fields.ny;
    let row = nearest / nx;
    let col = nearest % nx;
    let target_lon = f64::from(lon);
    let target_lat = f64::from(lat);
    for y0 in neighboring_cell_starts(row, ny).into_iter().flatten() {
        for x0 in neighboring_cell_starts(col, nx).into_iter().flatten() {
            let i00 = y0 * nx + x0;
            let i10 = i00 + 1;
            let i01 = i00 + nx;
            let i11 = i01 + 1;
            let corners = [
                (
                    unwrap_lon_near(f64::from(fields.lon[i00]), target_lon),
                    f64::from(fields.lat[i00]),
                ),
                (
                    unwrap_lon_near(f64::from(fields.lon[i10]), target_lon),
                    f64::from(fields.lat[i10]),
                ),
                (
                    unwrap_lon_near(f64::from(fields.lon[i01]), target_lon),
                    f64::from(fields.lat[i01]),
                ),
                (
                    unwrap_lon_near(f64::from(fields.lon[i11]), target_lon),
                    f64::from(fields.lat[i11]),
                ),
            ];
            let Some((uu, vv)) = solve_bilinear_coords(corners, target_lon, target_lat) else {
                continue;
            };
            if !((-0.02..=1.02).contains(&uu) && (-0.02..=1.02).contains(&vv)) {
                continue;
            }
            let uu = uu.clamp(0.0, 1.0) as f32;
            let vv = vv.clamp(0.0, 1.0) as f32;
            return Some([
                (i00, (1.0 - uu) * (1.0 - vv)),
                (i10, uu * (1.0 - vv)),
                (i01, (1.0 - uu) * vv),
                (i11, uu * vv),
            ]);
        }
    }
    None
}

/// Bracket a target MSL height in a WRF column (height increases with model
/// level index k). Returns the lower level and the linear weight, or `None`
/// when the target is below the lowest level (below terrain) or above the top.
fn bracket_height(
    fields: &WrfRadarFields,
    cells: usize,
    col: usize,
    z: f32,
) -> Option<(usize, f32)> {
    let nz = fields.nz;
    let h0 = fields.height_msl[col];
    let htop = fields.height_msl[(nz - 1) * cells + col];
    if !h0.is_finite() || !htop.is_finite() || z < h0 || z > htop {
        return None;
    }
    // WRF mass-level heights are monotonic in ordinary columns. Binary search
    // matters once a pulse gate carries 9 or 27 quadrature points: the former
    // O(nz) scan otherwise dominates the forward operator on deep nests.
    let mut lo = 0usize;
    let mut hi = nz - 1;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let hm = fields.height_msl[mid * cells + col];
        if !hm.is_finite() {
            break;
        }
        if hm <= z {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let hlo = fields.height_msl[lo * cells + col];
    let hhi = fields.height_msl[(lo + 1) * cells + col];
    if hlo.is_finite() && hhi.is_finite() && hhi > hlo && z >= hlo && z <= hhi {
        return Some((lo, (z - hlo) / (hhi - hlo)));
    }

    // A malformed/non-monotonic column is uncommon but should not turn into a
    // fabricated gate. Fall back to the former defensive linear search.
    (0..nz - 1).find_map(|k| {
        let hk = fields.height_msl[k * cells + col];
        let hk1 = fields.height_msl[(k + 1) * cells + col];
        (hk.is_finite() && hk1.is_finite() && hk1 > hk && z >= hk && z <= hk1)
            .then_some((k, (z - hk) / (hk1 - hk)))
    })
}

fn lerp(a: f32, b: f32, t: f32) -> Option<f32> {
    (a.is_finite() && b.is_finite()).then_some(a + t * (b - a))
}

// ── Gate texture (opt-in speckle) ─────────────────────────────────────────────
//
// Deliberately hash-based, NOT an RNG: every perturbation is a pure function
// of (tilt, azimuth, gate), so rebuilding a frame is bit-identical and a loop
// never shimmers between rebuilds of the same hour. Magnitudes are tuned
// against real Level-II: reflectivity texture on the order of ±2 dB,
// correlated over a few gates in range (the pulse volume smears neighbours)
// plus a smaller fully independent per-gate jitter; velocity gets only a
// gentle ±0.5 m/s wobble — pulse-pair Vr estimates are far smoother than
// reflectivity at decent SNR, and a noisy Vr would pollute the dealias/GBVTD
// consumers downstream.

/// Peak amplitude (dB) of the range-correlated reflectivity texture.
const REF_TEXTURE_CORRELATED_DB: f32 = 1.8;
/// Peak amplitude (dB) of the independent per-gate reflectivity jitter.
const REF_TEXTURE_JITTER_DB: f32 = 0.7;
/// Peak amplitude (m/s) of the (correlated) velocity wobble.
const VEL_TEXTURE_MPS: f32 = 0.5;
/// Range correlation length of the correlated components, in gates.
const TEXTURE_CORR_GATES: usize = 3;

const TEXTURE_SALT_REF: u32 = 0x5265_6631; // "Ref1"
const TEXTURE_SALT_REF_JITTER: u32 = 0x5265_664A; // "RefJ"
const TEXTURE_SALT_VEL: u32 = 0x5665_6C31; // "Vel1"

/// 32-bit avalanche mix (splitmix32 finalizer) — platform-independent.
fn texture_mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

/// Uniform noise in [-1, 1) from (tilt, azimuth, knot) + a salt.
fn texture_noise(tilt: usize, az: usize, knot: usize, salt: u32) -> f32 {
    let mixed = texture_mix(
        (tilt as u32).wrapping_mul(0x9e37_79b9)
            ^ (az as u32).wrapping_mul(0x85eb_ca6b)
            ^ (knot as u32).wrapping_mul(0xc2b2_ae35)
            ^ salt,
    );
    ((mixed >> 8) as f32) * (2.0 / 16_777_216.0) - 1.0
}

/// Range-correlated value noise in [-1, 1): hashed knots every
/// [`TEXTURE_CORR_GATES`] gates, linearly interpolated between them.
fn texture_correlated_noise(tilt: usize, az: usize, gate: usize, salt: u32) -> f32 {
    let knot = gate / TEXTURE_CORR_GATES;
    let t = (gate % TEXTURE_CORR_GATES) as f32 / TEXTURE_CORR_GATES as f32;
    let n0 = texture_noise(tilt, az, knot, salt);
    let n1 = texture_noise(tilt, az, knot + 1, salt);
    n0 + t * (n1 - n0)
}

/// Reflectivity gate-texture perturbation (dB), peak ±2.5 dB.
fn ref_gate_texture_db(tilt: usize, az: usize, gate: usize) -> f32 {
    REF_TEXTURE_CORRELATED_DB * texture_correlated_noise(tilt, az, gate, TEXTURE_SALT_REF)
        + REF_TEXTURE_JITTER_DB * texture_noise(tilt, az, gate, TEXTURE_SALT_REF_JITTER)
}

/// Velocity gate-texture perturbation (m/s), peak ±0.5 m/s.
fn vel_gate_texture_mps(tilt: usize, az: usize, gate: usize) -> f32 {
    VEL_TEXTURE_MPS * texture_correlated_noise(tilt, az, gate, TEXTURE_SALT_VEL)
}

// ── Ground clutter (opt-in, fabricated) ───────────────────────────────────────
//
// Our forward operator is pure physics and produces ZERO clutter. The community
// WRF→GR2 export script fabricates a near-radar ground-return look
// (`add_ground_clutter`): an exponential range falloff (hard-capped at 40 km),
// a height-AGL falloff, an elevation falloff on the lowest ~7 tilts, a
// sinusoidal azimuthal ripple, and a handful of random near-in hotspots, applied
// only where the existing echo is weaker. This is a native port of that look,
// dialled by `SyntheticRadarConfig::clutter_intensity` (0 = off, 1 ≈ the script).
//
// Like the gate texture above, every draw is a pure hash of
// (frame seed, tilt, azimuth, gate) — deterministic, so a rebuilt frame is
// bit-identical and a loop never shimmers. UNLIKE the texture, the FRAME SEED
// (site id + valid time) is folded in, so distinct forecast hours get distinct
// clutter, reproducing the script's per-time-step variation without a live RNG.
//
// Antenna height: the script keys clutter on beam height AGL. We do not have a
// per-gate terrain lookup inside the sampling loop, so we use the beam height
// ABOVE THE ANTENNA as the AGL proxy — exact at the radar and a good
// approximation within the 40 km clutter cap, where terrain relief is a
// second-order effect on near-in ground return.

/// Clutter is confined to the lowest tilts. The community script applies it to
/// the first 7 elevation cuts (`n < 7`); we mirror that by cut index, so it
/// tracks the near-ground tilts whether or not the optional 0.1° tilt is added.
const CLUTTER_TILT_LIMIT: usize = 7;
/// Maximum ground range for any clutter (km). Beyond this the script produces
/// none; near-radar concentration is the whole point.
const CLUTTER_MAX_RANGE_KM: f32 = 40.0;
/// Base clutter reflectivity (dBZ) before the combined-factor scaling, matching
/// the script's `clutter_base = 22 * combined_factor`.
const CLUTTER_BASE_DBZ: f32 = 22.0;
/// Multiplier applied inside a hotspot rectangle (the script's `*= 1.8`).
const CLUTTER_HOTSPOT_BOOST: f32 = 1.8;

const CLUTTER_SALT_ROLL: u32 = 0x436c_5250; // "ClRP" — apply/skip roll
const CLUTTER_SALT_NOISE: u32 = 0x436c_4e5a; // "ClNZ" — dBZ noise
const CLUTTER_SALT_TEXTURE: u32 = 0x436c_5458; // "ClTX" — dBZ texture
const CLUTTER_SALT_VEL: u32 = 0x436c_564c; // "ClVL" — near-zero velocity jitter
const CLUTTER_SALT_HOTSPOT: u32 = 0x436c_4853; // "ClHS" — hotspot stream seed

/// One deterministic seed per forecast frame: site id + valid-time seconds,
/// avalanche-mixed. Folded into every clutter draw so the SAME hour rebuilds
/// identically while DISTINCT hours vary (the script's `time_step` role).
fn clutter_frame_seed(site_id: &str, valid_time: DateTime<Utc>) -> u32 {
    // FNV-1a over the site id bytes, then fold in the timestamp seconds.
    let mut acc = 0x811c_9dc5u32;
    for byte in site_id.as_bytes() {
        acc ^= u32::from(*byte);
        acc = acc.wrapping_mul(0x0100_0193);
    }
    let secs = valid_time.timestamp();
    acc ^= (secs as u64 as u32).wrapping_mul(0x9e37_79b9);
    acc ^= ((secs as u64 >> 32) as u32).wrapping_mul(0x85eb_ca6b);
    texture_mix(acc)
}

/// A tiny counter-based splitmix32 stream (state += golden ratio, then the
/// [`texture_mix`] avalanche) for the SERIAL per-tilt hotspot layout. Reusing
/// the gate-texture finalizer keeps every random draw on one platform-
/// independent mixer.
struct SplitMix32 {
    state: u32,
}

impl SplitMix32 {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x9e37_79b9);
        texture_mix(self.state)
    }

    /// Uniform in [0, 1).
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Uniform integer in [0, n); 0 when `n == 0`.
    fn next_below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            ((self.next_unit() * n as f32) as usize).min(n - 1)
        }
    }
}

/// A near-radar hotspot rectangle in (azimuth index, gate index) space — the
/// script's `hotspot_mask[ray_min:ray_max, gate_min:gate_max]`.
struct ClutterRect {
    ray_min: usize,
    ray_max: usize,
    gate_min: usize,
    gate_max: usize,
}

impl ClutterRect {
    /// Whether gate (azimuth index, range index) falls inside this rectangle.
    fn contains(&self, az: usize, gate: usize) -> bool {
        let in_az = (self.ray_min..self.ray_max).contains(&az);
        let in_gate = (self.gate_min..self.gate_max).contains(&gate);
        in_az && in_gate
    }
}

/// The 3–8 near-radar hotspots for one tilt, laid out deterministically from
/// the frame seed + tilt index. Positions/sizes shrink with elevation, mirroring
/// the script (`max_gate = max(10, 100*(1 - n/8))`, ray/gate sizes `* (1 - n/8)`).
fn clutter_hotspots(
    frame_seed: u32,
    tilt: usize,
    naz: usize,
    gate_count: usize,
) -> Vec<ClutterRect> {
    let tilt_seed =
        texture_mix(frame_seed ^ (tilt as u32).wrapping_mul(0x9e37_79b9) ^ CLUTTER_SALT_HOTSPOT);
    let mut rng = SplitMix32::new(tilt_seed);
    let count = 3 + rng.next_below(6); // 3..=8, like the script
    let shrink = (1.0 - tilt as f32 / 8.0).max(0.0);
    let max_gate = ((100.0 * shrink) as usize).max(10).min(gate_count.max(1));
    let ray_size = ((20.0 * shrink) as usize).max(4);
    let gate_size = ((15.0 * shrink) as usize).max(4);
    let gate_lo = 5.min(max_gate.saturating_sub(1));
    let mut rects = Vec::with_capacity(count);
    for _ in 0..count {
        let ray = rng.next_below(naz.max(1));
        let center_gate = gate_lo + rng.next_below((max_gate - gate_lo).max(1));
        rects.push(ClutterRect {
            ray_min: ray.saturating_sub(ray_size / 2),
            ray_max: (ray + ray_size / 2 + 1).min(naz),
            gate_min: center_gate.saturating_sub(gate_size / 2),
            gate_max: (center_gate + gate_size / 2 + 1).min(gate_count),
        });
    }
    rects
}

/// Whether gate (az index, range index) falls inside any hotspot rectangle.
fn in_clutter_hotspot(hotspots: &[ClutterRect], az: usize, gate: usize) -> bool {
    hotspots.iter().any(|h| h.contains(az, gate))
}

/// A hashed uniform in [0, 1) from (frame seed, tilt, az, gate) + a salt — the
/// parallel-safe per-gate analogue of [`texture_noise`], order-independent so it
/// is identical whatever order rayon visits the radials in.
fn clutter_unit(frame_seed: u32, tilt: usize, az: usize, gate: usize, salt: u32) -> f32 {
    let mixed = texture_mix(
        frame_seed
            ^ (tilt as u32).wrapping_mul(0x9e37_79b9)
            ^ (az as u32).wrapping_mul(0x85eb_ca6b)
            ^ (gate as u32).wrapping_mul(0xc2b2_ae35)
            ^ salt,
    );
    (mixed >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Hashed signed noise in [-1, 1).
fn clutter_signed(frame_seed: u32, tilt: usize, az: usize, gate: usize, salt: u32) -> f32 {
    clutter_unit(frame_seed, tilt, az, gate, salt) * 2.0 - 1.0
}

/// The fabricated ground-clutter reflectivity (dBZ) for one gate, or `None` when
/// no clutter falls here (out of range, past the tilt limit, or the probability
/// roll misses). `intensity` (0..=1) scales BOTH the probability and the
/// brightness, so intermediate values are rarer AND dimmer; at 1.0 the model
/// matches the community script (bar its live-RNG boil, which the frame seed
/// reproduces deterministically). Native port of `add_ground_clutter`.
#[allow(clippy::too_many_arguments)]
fn ground_clutter_dbz(
    frame_seed: u32,
    tilt: usize,
    az: usize,
    gate: usize,
    az_deg: f32,
    ground_range_km: f32,
    beam_height_agl_m: f32,
    in_hotspot: bool,
    intensity: f32,
) -> Option<f32> {
    if tilt >= CLUTTER_TILT_LIMIT || ground_range_km > CLUTTER_MAX_RANGE_KM {
        return None;
    }
    // Combined intensity factor (the script's product of falloffs).
    let tilt_factor = (-(tilt as f32) * 0.8).exp();
    let distance_factor = (-ground_range_km / 15.0).exp();
    let height_factor = (-beam_height_agl_m.max(0.0) / 100.0).exp();
    let azimuthal = (az_deg * 3.0).to_radians().sin() * 0.3 + 0.7;
    let mut combined = distance_factor * height_factor * tilt_factor * azimuthal;
    if in_hotspot {
        combined *= CLUTTER_HOTSPOT_BOOST;
    }
    if combined <= 0.0 {
        return None;
    }

    // Probability of a clutter gate (the script's `combined * (0.9 - n*0.08)`),
    // scaled by the user amount. A per-gate roll decides apply vs skip.
    let base_prob = (combined * (0.9 - tilt as f32 * 0.08)).max(0.0);
    let prob = base_prob * intensity;
    if clutter_unit(frame_seed, tilt, az, gate, CLUTTER_SALT_ROLL) >= prob {
        return None;
    }

    // Base ≈ 22·combined dBZ + range-dependent noise, clipped 5..35, plus a
    // little texture — then scaled by the user amount so lower settings are
    // dimmer as well as rarer.
    let noise_factor = 4.0 - 3.0 * tilt as f32 / 7.0;
    let noise = clutter_signed(frame_seed, tilt, az, gate, CLUTTER_SALT_NOISE) * noise_factor;
    let mut value = (CLUTTER_BASE_DBZ * combined + noise).clamp(5.0, 35.0);
    value += clutter_signed(frame_seed, tilt, az, gate, CLUTTER_SALT_TEXTURE) * combined;
    Some(value * intensity)
}

/// The near-zero radial velocity (m/s) stamped on a clutter-dominated gate:
/// the ground is stationary, so ~0 with a small deterministic ±0.5 m/s jitter.
fn clutter_velocity_mps(frame_seed: u32, tilt: usize, az: usize, gate: usize) -> f32 {
    0.5 * clutter_signed(frame_seed, tilt, az, gate, CLUTTER_SALT_VEL)
}

/// Parse a WRF `Times` string ("YYYY-MM-DD_HH:MM:SS") to a UTC scan time.
fn parse_wrf_time(raw: &str) -> Option<DateTime<Utc>> {
    let cleaned = raw.trim().replace('_', " ");
    NaiveDateTime::parse_from_str(&cleaned, "%Y-%m-%d %H:%M:%S")
        .ok()
        .map(|naive| DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
}

/// The loop-engine history path key for one synthetic forecast frame. The key
/// is the dedupe/refresh discriminator the upsert rules key on
/// ([`ui_core::loop_engine::LoopEngine::install_batch`]): a re-import whose
/// [`SyntheticRadarConfig::data_fingerprint`] matches produces the SAME key (the
/// engine reuses the stored volume, rule (b)), while ANY config change produces
/// a DIFFERENT key so the freshly-built volume replaces the stale one (rule (c),
/// equal status + different path). `index` + `stamp` keep it unique per frame
/// within one build; `site_id` keeps it human-readable in status/telemetry.
pub fn synthetic_frame_path(
    site_id: &str,
    config_fingerprint: u64,
    index: usize,
    stamp: &str,
) -> PathBuf {
    PathBuf::from(format!(
        "wrf-synth://{site_id}/{config_fingerprint:016x}/{index:04}_{stamp}"
    ))
}

// ── Background job: WRF file(s) → Vec<Arc<RadarVolume>> ───────────────────────

#[derive(Debug)]
pub enum SyntheticRadarMessage {
    Progress(String),
    /// One-cut preview of the first forecast frame. This is intentionally sent
    /// only once: cloning every growing volume would make an already expensive
    /// raw P3 render progressively slower and consume quadratic memory traffic.
    Preview(SyntheticRadarPreview),
    Done(Result<SyntheticRadarOutput, String>),
}

#[derive(Debug)]
pub struct SyntheticRadarPreview {
    pub label: String,
    pub volume: Arc<RadarVolume>,
    pub config_fingerprint: u64,
    pub completed_cuts: usize,
    pub total_cuts: usize,
}

/// Private, session-only source contract for reopening one installed WRF
/// synthetic frame. Paths intentionally live only in this sidecar and are
/// never copied into RadarVolume metadata or exported provenance.
#[derive(Clone)]
pub(crate) struct SyntheticFrameSourceDescriptor {
    source: SyntheticFrameSource,
    config: SyntheticRadarConfig,
    config_fingerprint: u64,
    witness: SyntheticFrameWitness,
}

#[derive(Clone)]
enum SyntheticFrameSource {
    Wrf(Box<SyntheticWrfFrameSource>),
    OperationalForecast,
}

#[derive(Clone)]
struct SyntheticWrfFrameSource {
    anchor: WrfScene,
    neighbor: Option<WrfScene>,
    temporal_plan: Option<TemporalScenePlan>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SyntheticFrameWitness {
    frame_ordinal: usize,
    site_id: String,
    volume_time_unix_ms: i64,
    cut_count: usize,
    decoded_radial_count: usize,
    geometry_digest: u64,
}

impl SyntheticFrameWitness {
    fn from_volume(frame_ordinal: usize, config_fingerprint: u64, volume: &RadarVolume) -> Self {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        volume.site.id.hash(&mut hasher);
        volume.volume_time.timestamp_millis().hash(&mut hasher);
        config_fingerprint.hash(&mut hasher);
        volume.metadata.archive_version.hash(&mut hasher);
        volume.metadata.forward_operator.hash(&mut hasher);
        volume.metadata.forward_operator_config.hash(&mut hasher);
        volume.cuts.len().hash(&mut hasher);
        for cut in &volume.cuts {
            cut.elevation_deg.to_bits().hash(&mut hasher);
            cut.radials.len().hash(&mut hasher);
            for radial in &cut.radials {
                radial.azimuth_deg.to_bits().hash(&mut hasher);
                radial.elevation_deg.to_bits().hash(&mut hasher);
                radial.time_offset_ms.hash(&mut hasher);
                radial.gate_range.first_gate_m.hash(&mut hasher);
                radial.gate_range.gate_spacing_m.hash(&mut hasher);
                radial.gate_range.gate_count.hash(&mut hasher);
                radial
                    .nyquist_velocity_mps
                    .map(f32::to_bits)
                    .hash(&mut hasher);
            }
        }
        Self {
            frame_ordinal,
            site_id: volume.site.id.clone(),
            volume_time_unix_ms: volume.volume_time.timestamp_millis(),
            cut_count: volume.cuts.len(),
            decoded_radial_count: volume.metadata.decoded_radial_count,
            geometry_digest: hasher.finish(),
        }
    }

    fn matches(&self, config_fingerprint: u64, volume: &RadarVolume) -> bool {
        Self::from_volume(self.frame_ordinal, config_fingerprint, volume) == *self
    }
}

impl SyntheticFrameSourceDescriptor {
    fn retained_config(config: &SyntheticRadarConfig) -> SyntheticRadarConfig {
        let mut retained = config.clone();
        retained.runtime_cuda_service = None;
        retained
    }

    fn wrf(
        frame_ordinal: usize,
        volume: &RadarVolume,
        anchor: WrfScene,
        neighbor: Option<WrfScene>,
        temporal_plan: Option<TemporalScenePlan>,
        config: &SyntheticRadarConfig,
        config_fingerprint: u64,
    ) -> Self {
        Self {
            source: SyntheticFrameSource::Wrf(Box::new(SyntheticWrfFrameSource {
                anchor,
                neighbor,
                temporal_plan,
            })),
            config: Self::retained_config(config),
            config_fingerprint,
            witness: SyntheticFrameWitness::from_volume(frame_ordinal, config_fingerprint, volume),
        }
    }

    fn operational(
        frame_ordinal: usize,
        volume: &RadarVolume,
        config: &SyntheticRadarConfig,
        config_fingerprint: u64,
    ) -> Self {
        Self {
            source: SyntheticFrameSource::OperationalForecast,
            config: Self::retained_config(config),
            config_fingerprint,
            witness: SyntheticFrameWitness::from_volume(frame_ordinal, config_fingerprint, volume),
        }
    }

    pub(crate) fn matches_volume(&self, volume: &RadarVolume) -> bool {
        self.witness.matches(self.config_fingerprint, volume)
    }
}

pub(crate) struct GateExplanationTask {
    pub(crate) rx: Receiver<WhyThisGate>,
}

pub(crate) fn spawn_gate_explanation(
    descriptor: SyntheticFrameSourceDescriptor,
    volume: Arc<RadarVolume>,
    identity: GateIdentity,
) -> GateExplanationTask {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("wrf-selected-gate-explanation".to_owned())
        .spawn(move || {
            let result = explain_selected_gate(&descriptor, &volume, identity)
                .map(|explanation| WhyThisGate::Available(Box::new(explanation)))
                .unwrap_or_else(WhyThisGate::Unavailable);
            let _ = tx.send(result);
        })
        .expect("spawn selected-gate explanation worker");
    GateExplanationTask { rx }
}

fn explain_selected_gate(
    descriptor: &SyntheticFrameSourceDescriptor,
    volume: &RadarVolume,
    identity: GateIdentity,
) -> Result<GateExplanation, WhyThisGateUnavailable> {
    if !descriptor.matches_volume(volume) {
        return Err(WhyThisGateUnavailable::StaleFrameWitness);
    }
    if identity.frame_index != descriptor.witness.frame_ordinal {
        return Err(WhyThisGateUnavailable::StaleFrameWitness);
    }
    let SyntheticFrameSource::Wrf(source) = &descriptor.source else {
        return Err(WhyThisGateUnavailable::UnsupportedSourceContract);
    };
    let SyntheticWrfFrameSource {
        anchor,
        neighbor,
        temporal_plan,
    } = source.as_ref();
    let cut = volume
        .cuts
        .get(identity.cut_index)
        .ok_or(WhyThisGateUnavailable::StaleFrameWitness)?;
    let radial = cut
        .radials
        .get(identity.radial_index)
        .ok_or(WhyThisGateUnavailable::StaleFrameWitness)?;
    if identity.gate_index >= radial.gate_range.gate_count
        || (f64::from(radial.azimuth_deg) - identity.azimuth_deg).abs() > 1.0e-5
        || (f64::from(radial.elevation_deg) - identity.elevation_deg).abs() > 1.0e-5
    {
        return Err(WhyThisGateUnavailable::StaleFrameWitness);
    }
    let expected_slant_m = f64::from(radial.gate_range.first_gate_m)
        + identity.gate_index as f64 * f64::from(radial.gate_range.gate_spacing_m);
    if (expected_slant_m - identity.slant_range_m).abs() > 0.51 {
        return Err(WhyThisGateUnavailable::StaleFrameWitness);
    }

    let config = inspector_render_config(&descriptor.config, volume);
    let anchor_file = reopen_wrf_scene(anchor)?;
    let progress = |_: &str| {};
    let anchor_fields = read_wrf_radar_fields_for_config_reporting(
        &anchor_file,
        &anchor.source_identity,
        anchor.time_index,
        &config,
        &progress,
        0,
        None,
    )
    .map_err(WhyThisGateUnavailable::WorkerFailed)?;
    let neighbor_pair = if let Some(neighbor) = neighbor {
        let file = reopen_wrf_scene(neighbor)?;
        let fields = read_wrf_radar_fields_for_config_reporting(
            &file,
            &neighbor.source_identity,
            neighbor.time_index,
            &config,
            &progress,
            0,
            None,
        )
        .map_err(WhyThisGateUnavailable::WorkerFailed)?;
        validate_temporal_field_pair(&anchor_fields, &fields)
            .map_err(WhyThisGateUnavailable::WorkerFailed)?;
        Some((file, fields))
    } else {
        None
    };
    let neighbor_fields = neighbor_pair.as_ref().map(|(_, fields)| fields);
    let ray_offset_ms = i64::from(radial.time_offset_ms);
    let temporal_alpha = if neighbor_fields.is_some() {
        temporal_plan
            .as_ref()
            .map(|plan| plan.ray_alpha(ray_offset_ms))
            .transpose()
            .map_err(|error| WhyThisGateUnavailable::WorkerFailed(error.to_string()))?
            .unwrap_or(0.0)
    } else {
        0.0
    };

    let (site_lat, site_lon, antenna_msl) = resolve_radar_site(&anchor_fields, &config)
        .map_err(WhyThisGateUnavailable::WorkerFailed)?;
    let beam_geometry =
        BeamPropagationGeometry::resolve(&anchor_fields, &config, site_lat, site_lon, antenna_msl)
            .map_err(WhyThisGateUnavailable::WorkerFailed)?;
    let coupled =
        resolve_coupled_instrument(&config).map_err(WhyThisGateUnavailable::WorkerFailed)?;
    let spacing_m = f64::from(radial.gate_range.gate_spacing_m);
    let prefix_gate_count = identity.gate_index + 1;
    let horizon_azimuth_count = if descriptor.config.exact_replay_template.is_some() {
        720
    } else {
        config.azimuth_count.max(1)
    };
    let terrain_horizon =
        (config.terrain_blockage && !beam_geometry.is_wrf_refractivity()).then(|| {
            TerrainHorizon::build_selected_azimuth(
                &anchor_fields,
                site_lat,
                site_lon,
                antenna_msl,
                horizon_azimuth_count,
                identity.azimuth_deg,
                prefix_gate_count,
                radial.gate_range.first_gate_m,
                spacing_m,
            )
        });
    let quadrature_count = coupled
        .as_ref()
        .and_then(|instrument| instrument.quadrature(config.beam_integration))
        .map_or_else(
            || quadrature_points(config.beam_integration).len(),
            |points| points.len(),
        );
    let mut refracted_terrain_blockage = (config.terrain_blockage
        && beam_geometry.is_wrf_refractivity())
    .then(|| vec![false; quadrature_count]);

    let mut previous_kdp = 0.0f32;
    let mut previous_ah = 0.0f32;
    let mut previous_adp = 0.0f32;
    let mut phi_path = 0.0f32;
    let mut tau_h = 0.0f32;
    let mut tau_dp = 0.0f32;
    let dr_km = (spacing_m / 1_000.0).max(0.0) as f32;
    let mut selected_quality = GateQualityFractions::default();
    let mut selected_physical = None;
    let mut selected_polar = None;
    let mut selected_pia = f32::NAN;
    let mut selected_pida = f32::NAN;
    let mut selected_phi_dp = f32::NAN;
    let mut selected_intrinsic_dbz = f32::NAN;
    let mut selected_geometry = None;
    for gate in 0..prefix_gate_count {
        let slant_m = f64::from(radial.gate_range.first_gate_m) + gate as f64 * spacing_m;
        let center_geometry = beam_geometry
            .point(
                identity.elevation_deg,
                &radial.gate_range,
                gate,
                0.0,
                slant_m,
            )
            .map_err(WhyThisGateUnavailable::WorkerFailed)?;
        let sampled = sample_gate_with_quality_instrument(
            &anchor_fields,
            neighbor_fields,
            temporal_alpha,
            anchor_fields.cells(),
            site_lat,
            site_lon,
            antenna_msl,
            identity.azimuth_deg,
            identity.elevation_deg,
            slant_m,
            gate,
            spacing_m,
            &radial.gate_range,
            beam_geometry,
            &config,
            terrain_horizon.as_ref(),
            refracted_terrain_blockage.as_deref_mut(),
            coupled.as_ref(),
            None,
            None,
        )
        .map_err(WhyThisGateUnavailable::WorkerFailed)?;
        if gate == identity.gate_index {
            selected_geometry = Some(center_geometry);
            selected_quality = sampled.quality;
        }
        if sampled.quality.model_coverage_fraction
            < config.clamped_minimum_model_coverage_fraction()
        {
            continue;
        }
        let Some(physical) = sampled.physical else {
            continue;
        };
        let intrinsic_dbz = z_to_dbz(physical.z_linear);
        let mut pia = f32::NAN;
        let mut pida = f32::NAN;
        let mut phi_dp = f32::NAN;
        if let Some(polar) = physical.polar {
            let kdp = polar.kdp_deg_km;
            let ah = polar.ah_db_km;
            let adp = polar.ah_db_km - polar.av_db_km;
            if config.propagation && gate > 0 {
                phi_path += (previous_kdp + kdp) * dr_km;
                tau_h += 0.5 * (previous_ah + ah) * dr_km;
                tau_dp += 0.5 * (previous_adp + adp) * dr_km;
            }
            previous_kdp = kdp;
            previous_ah = ah;
            previous_adp = adp;
            pia = if config.propagation { 2.0 * tau_h } else { 0.0 };
            pida = if config.propagation {
                2.0 * tau_dp
            } else {
                0.0
            };
            phi_dp = config.system_phidp_deg + if config.propagation { phi_path } else { 0.0 };
        }
        if gate == identity.gate_index {
            selected_intrinsic_dbz = intrinsic_dbz;
            selected_polar = physical.polar;
            selected_pia = pia;
            selected_pida = pida;
            selected_phi_dp = phi_dp;
            selected_physical = Some(physical);
        }
    }

    let coverage = selected_quality;
    let physical = selected_physical;
    let observed_dbz = selected_intrinsic_dbz
        - if selected_pia.is_finite() {
            selected_pia
        } else {
            0.0
        };
    let ideal = IdealMoments {
        values: RadarMomentValues {
            reflectivity_dbz: observed_dbz.is_finite().then_some(f64::from(observed_dbz)),
            velocity_mps: physical
                .filter(|sample| sample.velocity_mps.is_finite())
                .map(|sample| f64::from(sample.velocity_mps)),
            spectrum_width_mps: physical
                .filter(|sample| config.spectrum_width && sample.spectrum_width_mps.is_finite())
                .map(|sample| f64::from(sample.spectrum_width_mps.max(0.0))),
            zdr_db: selected_polar
                .filter(|_| config.dual_pol)
                .map(|polar| f64::from(polar.zdr_db - selected_pida.max(0.0))),
            rho_hv: selected_polar
                .filter(|_| config.dual_pol)
                .map(|polar| f64::from(polar.rho_hv.clamp(0.0, 1.0))),
            kdp_deg_km: selected_polar
                .filter(|_| config.dual_pol)
                .map(|polar| f64::from(polar.kdp_deg_km)),
        },
    };
    let frame_seed = clutter_frame_seed(&volume.site.id, volume.volume_time);
    let noise_key = NoiseKey {
        seed: u64::from(frame_seed),
        frame: 0,
        cut: u16::try_from(identity.cut_index).unwrap_or(u16::MAX),
        ray: u32::try_from(identity.radial_index).unwrap_or(u32::MAX),
        gate: u32::try_from(identity.gate_index).unwrap_or(u32::MAX),
    };
    let (instrument, timing, measured, presented) = explain_instrument_stages(
        &config,
        coupled.as_ref(),
        ideal,
        identity,
        selected_geometry,
        frame_seed,
        cut.radials.len(),
        radial.gate_range.gate_count,
        noise_key,
    )?;

    let hydrometeor_details = if physical.is_some()
        && matches!(
            config.polarimetric_kernel,
            PolarimetricKernel::BulkRayleighV1
        ) {
        bulk_species_gate_contributions(
            &anchor_file,
            anchor.time_index,
            neighbor_pair.as_ref().map(|(file, _)| file),
            neighbor.as_ref().map(|scene| scene.time_index),
            &anchor_fields,
            neighbor_fields,
            temporal_alpha,
            site_lat,
            site_lon,
            antenna_msl,
            identity,
            &radial.gate_range,
            beam_geometry,
            &config,
            terrain_horizon.as_ref(),
            coupled.as_ref(),
            coverage.model_coverage_fraction as f64,
        )?
    } else {
        Vec::new()
    };
    let hydrometeors = hydrometeor_details
        .iter()
        .map(|detail| HydrometeorGateContribution {
            name: detail.name.clone(),
            zh_linear: f64::from(detail.contribution.zh),
            zv_linear: f64::from(detail.contribution.zv),
            kdp_deg_km: f64::from(detail.contribution.kdp_deg_km),
            ah_db_km: f64::from(detail.contribution.ah_db_km),
            fall_speed_mps: f64::from(detail.contribution.fall_speed_mps),
        })
        .collect::<Vec<_>>();
    let spectrum = physical.and_then(|physical| {
        let el = identity.elevation_deg.to_radians();
        let modes = hydrometeor_details
            .iter()
            .filter(|detail| detail.contribution.zh.is_finite() && detail.contribution.zh > 0.0)
            .map(|detail| SpeciesDopplerMode {
                name: detail.name.clone(),
                power_linear: f64::from(detail.contribution.zh),
                air_velocity_mps: f64::from(physical.air_velocity_mps),
                fall_velocity_projection_mps: -f64::from(detail.contribution.fall_speed_mps)
                    * el.sin(),
                intrinsic_width_mps: f64::from(
                    detail.contribution.fall_speed_variance_m2s2.max(0.0).sqrt(),
                ) * el.sin().abs(),
                beam_shear_width_mps: f64::from(
                    physical.pulse_volume_variance_m2s2.max(0.0).sqrt(),
                ),
                turbulence_width_mps: f64::from(physical.turbulence_variance_m2s2.max(0.0).sqrt()),
            })
            .collect::<Vec<_>>();
        if modes.is_empty() {
            return None;
        }
        let nyquist = timing
            .map(|value| value.nyquist_velocity_mps)
            .or_else(|| radial.nyquist_velocity_mps.map(f64::from));
        let extent = nyquist.unwrap_or(75.0).max(20.0);
        synthesize_selected_gate_spectrum(
            &modes,
            DopplerSpectrumConfig {
                true_min_velocity_mps: -extent * 2.0,
                true_max_velocity_mps: extent * 2.0,
                true_bin_count: 512,
                output_bin_count: 256,
                nyquist_velocity_mps: nyquist,
                white_noise_power_per_bin: measured
                    .snr_db
                    .map(|snr| f64::from(physical.z_linear) / 10.0f64.powf(snr / 10.0) / 256.0)
                    .unwrap_or(0.0),
                noise_key,
            },
        )
        .ok()
    });

    let anchor_time = anchor
        .time
        .valid_time()
        .ok_or(WhyThisGateUnavailable::StaleFrameWitness)?;
    let neighbor_time = neighbor
        .as_ref()
        .and_then(|scene| scene.time.valid_time())
        .map(|time| time.timestamp_millis());
    let mut provenance = vec![
        format!("config_fingerprint={:016x}", descriptor.config_fingerprint),
        format!(
            "frame_geometry_witness={:016x}",
            descriptor.witness.geometry_digest
        ),
        format!("anchor_source_identity={}", anchor.source_identity.0),
        format!("anchor_time_index={}", anchor.time_index),
        format!(
            "atmosphere_mode={}",
            config.atmosphere_time_mode.provenance_name()
        ),
        "source_paths=session_private_not_exported".to_owned(),
    ];
    if let Some(neighbor) = neighbor {
        provenance.push(format!(
            "neighbor_source_identity={}",
            neighbor.source_identity.0
        ));
        provenance.push(format!("neighbor_time_index={}", neighbor.time_index));
    }
    if config.polarimetric_kernel.is_property_tmatrix() {
        provenance.push(
            "hydrometeor_decomposition=unavailable_without_qualified_property_category_seam"
                .to_owned(),
        );
        provenance.push("doppler_spectrum=unavailable_without_species_modes".to_owned());
    }
    Ok(GateExplanation {
        identity,
        time: GateTimeExplanation {
            ray_offset_ms,
            anchor_unix_ms: anchor_time.timestamp_millis(),
            neighbor_unix_ms: neighbor_time,
            temporal_alpha,
            held_anchor: neighbor.is_none() && temporal_plan.is_some(),
        },
        coverage: GateCoverageExplanation {
            model_coverage_fraction: f64::from(coverage.model_coverage_fraction),
            terrain_unblocked_fraction: f64::from(coverage.terrain_unblocked_fraction),
            meteorological_signal_fraction: f64::from(coverage.meteorological_signal_fraction),
            unblocked_power_fraction: f64::from(coverage.terrain_unblocked_fraction),
        },
        hydrometeors,
        velocity: GateVelocityExplanation {
            air_velocity_mps: physical.map_or(0.0, |value| f64::from(value.air_velocity_mps)),
            terminal_fall_correction_mps: physical
                .map_or(0.0, |value| f64::from(value.terminal_fall_correction_mps)),
            scatterer_velocity_mps: physical.map_or(0.0, |value| f64::from(value.velocity_mps)),
            pulse_volume_variance_m2s2: physical
                .map_or(0.0, |value| f64::from(value.pulse_volume_variance_m2s2)),
            terminal_variance_m2s2: physical
                .map_or(0.0, |value| f64::from(value.terminal_variance_m2s2)),
            turbulence_variance_m2s2: physical
                .map_or(0.0, |value| f64::from(value.turbulence_variance_m2s2)),
            instrument_variance_m2s2: measured.uncertainty.velocity_sigma_mps.powi(2),
        },
        propagation: GatePropagationExplanation {
            intrinsic_reflectivity_dbz: selected_intrinsic_dbz
                .is_finite()
                .then_some(f64::from(selected_intrinsic_dbz)),
            observed_reflectivity_dbz: observed_dbz.is_finite().then_some(f64::from(observed_dbz)),
            intrinsic_zdr_db: selected_polar.map(|value| f64::from(value.zdr_db)),
            observed_zdr_db: selected_polar.map(|value| {
                f64::from(
                    value.zdr_db
                        - if selected_pida.is_finite() {
                            selected_pida
                        } else {
                            0.0
                        },
                )
            }),
            phi_dp_deg: selected_phi_dp
                .is_finite()
                .then_some(f64::from(selected_phi_dp)),
            pia_db: selected_pia.is_finite().then_some(f64::from(selected_pia)),
            pida_db: selected_pida
                .is_finite()
                .then_some(f64::from(selected_pida)),
        },
        instrument,
        timing,
        ideal,
        measured,
        presented,
        spectrum,
        provenance,
    })
}

fn reopen_wrf_scene(scene: &WrfScene) -> Result<WrfFile, WhyThisGateUnavailable> {
    if !scene.path.is_file() {
        return Err(WhyThisGateUnavailable::SourceFileUnavailable);
    }
    let selected =
        inventory_selected_wrf_paths(std::slice::from_ref(&scene.path)).map_err(|error| {
            if scene.path.is_file() {
                WhyThisGateUnavailable::WorkerFailed(error)
            } else {
                WhyThisGateUnavailable::SourceFileUnavailable
            }
        })?;
    let current = selected
        .group
        .scenes
        .iter()
        .find(|candidate| candidate.time_index == scene.time_index)
        .ok_or(WhyThisGateUnavailable::StaleFrameWitness)?;
    if current.source_identity != scene.source_identity
        || current.run_domain != scene.run_domain
        || current.grid_signature != scene.grid_signature
        || current.time.valid_time() != scene.time.valid_time()
    {
        return Err(WhyThisGateUnavailable::StaleFrameWitness);
    }
    WrfFile::open(&scene.path).map_err(|error| {
        if scene.path.is_file() {
            WhyThisGateUnavailable::WorkerFailed(format!("reopen retained WRF source: {error}"))
        } else {
            WhyThisGateUnavailable::SourceFileUnavailable
        }
    })
}

fn inspector_render_config(
    source: &SyntheticRadarConfig,
    volume: &RadarVolume,
) -> SyntheticRadarConfig {
    let mut config = source.clone();
    if source.exact_replay_template.is_some() {
        config.exact_replay_template = None;
        config.scan_strategy = SyntheticScanStrategy::CustomLegacy;
        config.site_id = volume.site.id.clone();
        config.site_name = volume.site.name.clone();
        config.site_lat_deg = volume.site.latitude_deg.map(f64::from);
        config.site_lon_deg = volume.site.longitude_deg.map(f64::from);
        config.antenna_msl_m = volume.site.elevation_m.map(f64::from);
        config.fold_velocity = false;
        config.coupled_single_prf_estimator = false;
        config.emit_stage_diagnostics = false;
        config.emit_quality_fields = true;
        config.spectrum_width = volume
            .cuts
            .iter()
            .any(|cut| cut.moments.contains_key(&MomentType::SpectrumWidth));
        config.dual_pol = volume
            .cuts
            .iter()
            .flat_map(|cut| cut.moments.keys())
            .any(is_dual_pol_replay_moment);
    }
    config
}

#[allow(clippy::too_many_arguments)]
fn explain_instrument_stages(
    config: &SyntheticRadarConfig,
    coupled: Option<&CoupledInstrumentContext>,
    ideal: IdealMoments,
    identity: GateIdentity,
    geometry: Option<BeamGeometryPoint>,
    frame_seed: u32,
    radial_count: usize,
    gate_count: usize,
    noise_key: NoiseKey,
) -> Result<
    (
        RadarInstrument,
        Option<ResolvedSinglePrf>,
        MeasuredMoments,
        PresentedMoments,
    ),
    WhyThisGateUnavailable,
> {
    let instrument = coupled.map_or_else(
        || {
            RadarInstrument::new(
                "BowEcho virtual radar",
                f64::from(config.radar_frequency_mhz) * 1.0e6,
                f64::from(config.pulse_width_us) * 1.0e-6,
            )
            .map_err(|error| WhyThisGateUnavailable::WorkerFailed(error.to_string()))
        },
        |context| Ok(context.instrument.clone()),
    )?;
    let timing = if let Some(coupled) = coupled {
        Some(coupled.timing)
    } else {
        let prf = CustomSinglePrf::new(f64::from(config.prf_hz))
            .map_err(|error| WhyThisGateUnavailable::WorkerFailed(error.to_string()))?;
        resolve_prf(&instrument, PrfSpecification::CustomSinglePrf(prf)).ok()
    };
    let clutter_amount = config.clutter_intensity.clamp(0.0, 1.0);
    let clutter_on = clutter_amount > 0.0 && identity.cut_index < CLUTTER_TILT_LIMIT;
    let clutter_reflectivity = if clutter_on {
        let geometry = geometry.unwrap_or(BeamGeometryPoint {
            ground_range_m: identity.slant_range_m,
            height_above_radar_m: 0.0,
        });
        let hotspots = clutter_hotspots(frame_seed, identity.cut_index, radial_count, gate_count);
        ground_clutter_dbz(
            frame_seed,
            identity.cut_index,
            identity.radial_index,
            identity.gate_index,
            identity.azimuth_deg as f32,
            (geometry.ground_range_m / 1_000.0) as f32,
            geometry.height_above_radar_m as f32,
            in_clutter_hotspot(&hotspots, identity.radial_index, identity.gate_index),
            clutter_amount,
        )
    } else {
        None
    };

    if let Some(coupled) = coupled {
        let measured = estimate_measured_moments(
            ideal,
            &instrument,
            &coupled.timing,
            &coupled.estimator_config,
            identity.slant_range_m,
            noise_key,
        )
        .map_err(|error| WhyThisGateUnavailable::WorkerFailed(error.to_string()))?;
        let presentation = PresentationConfig {
            reflectivity_texture_sigma_db: if config.ref_gate_texture {
                f64::from(REF_TEXTURE_CORRELATED_DB.hypot(REF_TEXTURE_JITTER_DB))
            } else {
                0.0
            },
            velocity_texture_sigma_mps: if config.vel_gate_texture {
                f64::from(VEL_TEXTURE_MPS)
            } else {
                0.0
            },
            zdr_display_bias_db: 0.0,
            reflectivity_display_floor_dbz: None,
            clutter_reflectivity_dbz: clutter_reflectivity.map(f64::from),
            clutter_velocity_mps: if clutter_reflectivity.is_some() {
                f64::from(clutter_velocity_mps(
                    frame_seed,
                    identity.cut_index,
                    identity.radial_index,
                    identity.gate_index,
                ))
            } else {
                0.0
            },
        };
        let mut presented = present_measured_moments(measured, &presentation, noise_key);
        if let Some(value) = presented.values.velocity_mps.as_mut() {
            *value = f64::from(fold_velocity_mps(
                *value as f32,
                coupled.stamped_nyquist_mps(),
            ));
        }
        if presented.adjustment.clutter_replaced
            && let Some(width) = presented.values.spectrum_width_mps.as_mut()
        {
            *width = (*width).max(f64::from(config.spectrum_width_floor_mps.max(0.0)));
        }
        return Ok((instrument, timing, measured, presented));
    }

    let mut measured = MeasuredMoments {
        values: ideal.values,
        ..MeasuredMoments::default()
    };
    if let Some(dbz) = measured.values.reflectivity_dbz.as_mut() {
        if config.ref_gate_texture {
            *dbz += f64::from(ref_gate_texture_db(
                identity.cut_index,
                identity.radial_index,
                identity.gate_index,
            ));
        }
        let range_km = (identity.slant_range_m / 1_000.0).max(1.0);
        let sensitivity = f64::from(config.sensitivity_dbz_at_1km) + 20.0 * range_km.log10();
        measured.sensitivity_dbz = Some(sensitivity);
        measured.snr_db = Some(*dbz - sensitivity);
        if config.instrument_noise {
            let snr = *dbz - sensitivity;
            if !snr.is_finite() || snr < 0.0 {
                measured.censored = true;
                measured.values = RadarMomentValues::default();
            } else {
                let sigma_db = (1.5 / (1.0 + snr / 6.0)).clamp(0.12, 1.5);
                measured.uncertainty.reflectivity_sigma_db = sigma_db;
                *dbz += sigma_db
                    * f64::from(clutter_signed(
                        frame_seed,
                        identity.cut_index,
                        identity.radial_index,
                        identity.gate_index,
                        0x534e_525a,
                    ));
            }
        } else if *dbz < f64::from(config.ref_floor_dbz) {
            measured.censored = true;
            measured.values = RadarMomentValues::default();
        }
    }
    if !measured.censored
        && let Some(velocity) = measured.values.velocity_mps.as_mut()
    {
        if config.instrument_noise {
            let snr = measured.snr_db.unwrap_or(0.0).max(0.0);
            let sigma = (1.2 / (1.0 + snr / 8.0)).clamp(0.08, 1.2);
            measured.uncertainty.velocity_sigma_mps = sigma;
            *velocity += sigma
                * f64::from(clutter_signed(
                    frame_seed,
                    identity.cut_index,
                    identity.radial_index,
                    identity.gate_index,
                    0x534e_5256,
                ));
        }
        if config.vel_gate_texture {
            *velocity += f64::from(vel_gate_texture_mps(
                identity.cut_index,
                identity.radial_index,
                identity.gate_index,
            ));
        }
    }
    let mut presented = PresentedMoments {
        values: measured.values,
        ..PresentedMoments::default()
    };
    if let (Some(clutter), Some(reflectivity)) = (
        clutter_reflectivity.map(f64::from),
        presented.values.reflectivity_dbz.as_mut(),
    ) && *reflectivity < clutter
    {
        presented.adjustment.clutter_replaced = true;
        presented.adjustment.reflectivity_db = clutter - *reflectivity;
        *reflectivity = clutter;
        if let Some(velocity) = presented.values.velocity_mps.as_mut() {
            let clutter_velocity = f64::from(clutter_velocity_mps(
                frame_seed,
                identity.cut_index,
                identity.radial_index,
                identity.gate_index,
            ));
            presented.adjustment.velocity_mps = clutter_velocity - *velocity;
            *velocity = clutter_velocity;
        }
        presented.values.zdr_db = None;
        presented.values.rho_hv = None;
        presented.values.kdp_deg_km = None;
    }
    if !presented.adjustment.clutter_replaced
        && let Some(zdr) = presented.values.zdr_db.as_mut()
    {
        *zdr += f64::from(config.zdr_bias_db);
        presented.adjustment.zdr_db = f64::from(config.zdr_bias_db);
    }
    if config.fold_velocity
        && let Some(velocity) = presented.values.velocity_mps.as_mut()
    {
        *velocity = f64::from(fold_velocity_mps(*velocity as f32, config.nyquist_mps));
    }
    Ok((instrument, timing, measured, presented))
}

#[derive(Clone)]
struct BulkHydrometeorDetail {
    name: String,
    contribution: crate::wrf_radar_physics::BulkContribution,
}

#[derive(Clone)]
struct InspectorSpatialTarget {
    latitude_deg: f32,
    longitude_deg: f32,
    height_msl_m: f32,
    elevation_deg: f64,
    point_weight: f64,
    blocked: bool,
}

#[allow(clippy::too_many_arguments)]
fn bulk_species_gate_contributions(
    anchor_file: &WrfFile,
    anchor_time_index: usize,
    neighbor_file: Option<&WrfFile>,
    neighbor_time_index: Option<usize>,
    anchor_fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    temporal_alpha: f64,
    site_lat: f64,
    site_lon: f64,
    antenna_msl: f64,
    identity: GateIdentity,
    gate_range: &GateRange,
    beam_geometry: BeamPropagationGeometry<'_>,
    config: &SyntheticRadarConfig,
    terrain_horizon: Option<&TerrainHorizon>,
    coupled: Option<&CoupledInstrumentContext>,
    _model_coverage_fraction: f64,
) -> Result<Vec<BulkHydrometeorDetail>, WhyThisGateUnavailable> {
    let targets = inspector_spatial_targets(
        anchor_fields,
        neighbor_fields,
        temporal_alpha,
        site_lat,
        site_lon,
        antenna_msl,
        identity,
        gate_range,
        beam_geometry,
        config,
        terrain_horizon,
        coupled,
    )?;
    let valid_weight = targets
        .iter()
        .map(|target| target.point_weight)
        .sum::<f64>();
    if valid_weight <= 0.0 {
        return Ok(Vec::new());
    }
    let anchor =
        read_bulk_species_targets(anchor_file, anchor_time_index, anchor_fields, &targets)?;
    let neighbor = match (neighbor_file, neighbor_fields, neighbor_time_index) {
        (Some(file), Some(fields), Some(time_index)) => Some(read_bulk_species_targets(
            file, time_index, fields, &targets,
        )?),
        _ => None,
    };
    let mut output = Vec::new();
    for (species_index, spec) in SPECIES_READ_SPECS.iter().enumerate() {
        let mut gate = crate::wrf_radar_physics::PolarAccumulator::default();
        for (target_index, target) in targets.iter().enumerate() {
            if target.blocked {
                continue;
            }
            let mut temporal = crate::wrf_radar_physics::PolarAccumulator::default();
            temporal.add(
                (1.0 - temporal_alpha) as f32,
                anchor[species_index][target_index],
            );
            if let Some(neighbor) = &neighbor {
                temporal.add(temporal_alpha as f32, neighbor[species_index][target_index]);
            }
            gate.add(
                target.point_weight as f32,
                intrinsic_as_contribution(temporal.finalize()),
            );
        }
        let contribution =
            intrinsic_as_contribution(normalize_intrinsic(gate.finalize(), valid_weight as f32));
        if contribution.zh > 0.0 {
            output.push(BulkHydrometeorDetail {
                name: hydrometeor_name(spec.kind).to_owned(),
                contribution,
            });
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn inspector_spatial_targets(
    fields: &WrfRadarFields,
    neighbor_fields: Option<&WrfRadarFields>,
    temporal_alpha: f64,
    site_lat: f64,
    site_lon: f64,
    antenna_msl: f64,
    identity: GateIdentity,
    gate_range: &GateRange,
    beam_geometry: BeamPropagationGeometry<'_>,
    config: &SyntheticRadarConfig,
    terrain_horizon: Option<&TerrainHorizon>,
    coupled: Option<&CoupledInstrumentContext>,
) -> Result<Vec<InspectorSpatialTarget>, WhyThisGateUnavailable> {
    let beam_sigma_deg = gaussian_beam_sigma_deg(f64::from(config.beam_width_deg.max(0.05)));
    let physical_points =
        coupled.and_then(|instrument| instrument.quadrature(config.beam_integration));
    let points = physical_points
        .map(|points| {
            points
                .iter()
                .map(|point| {
                    (
                        point.az_sigma,
                        point.el_sigma,
                        point.range_offset_m,
                        point.weight,
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            quadrature_points(config.beam_integration)
                .iter()
                .map(|point| {
                    (
                        point.az_sigma,
                        point.el_sigma,
                        point.range_gate * f64::from(gate_range.gate_spacing_m),
                        point.weight,
                    )
                })
                .collect()
        });
    let mut targets = Vec::new();
    for (az_sigma, el_sigma, range_offset_m, point_weight) in points {
        let azimuth_deg = identity.azimuth_deg + az_sigma * beam_sigma_deg;
        let elevation_deg = identity.elevation_deg + el_sigma * beam_sigma_deg;
        let center_slant_m = f64::from(gate_range.first_gate_m)
            + identity.gate_index as f64 * f64::from(gate_range.gate_spacing_m);
        let geometry = beam_geometry
            .point(
                elevation_deg,
                gate_range,
                identity.gate_index,
                range_offset_m,
                center_slant_m,
            )
            .map_err(WhyThisGateUnavailable::WorkerFailed)?;
        let z_msl = antenna_msl + geometry.height_above_radar_m;
        let azimuth = azimuth_deg.to_radians();
        let east_km = geometry.ground_range_m * azimuth.sin() / 1_000.0;
        let north_km = geometry.ground_range_m * azimuth.cos() / 1_000.0;
        let (lat, lon) = aeqd_inverse_km(site_lat, site_lon, east_km, north_km);
        if sample_column_temporal(
            fields,
            neighbor_fields,
            temporal_alpha,
            fields.cells(),
            lat as f32,
            lon as f32,
            z_msl as f32,
            elevation_deg,
            config.atmosphere_time_mode,
            config.reflectivity_sampling,
            None,
            None,
        )
        .map_err(WhyThisGateUnavailable::WorkerFailed)?
        .is_none()
        {
            continue;
        }
        let blocked = if beam_geometry.is_wrf_refractivity() && config.terrain_blockage {
            let mut blocked = false;
            for gate in 0..=identity.gate_index {
                let slant_m = f64::from(gate_range.first_gate_m)
                    + gate as f64 * f64::from(gate_range.gate_spacing_m);
                let point = beam_geometry
                    .point(elevation_deg, gate_range, gate, range_offset_m, slant_m)
                    .map_err(WhyThisGateUnavailable::WorkerFailed)?;
                let ground_m = point.ground_range_m;
                let east_km = ground_m * azimuth.sin() / 1_000.0;
                let north_km = ground_m * azimuth.cos() / 1_000.0;
                let (gate_lat, gate_lon) = aeqd_inverse_km(site_lat, site_lon, east_km, north_km);
                if sample_terrain(fields, gate_lat as f32, gate_lon as f32).is_some_and(|terrain| {
                    f64::from(terrain) >= antenna_msl + point.height_above_radar_m
                }) {
                    blocked = true;
                    break;
                }
            }
            blocked
        } else {
            terrain_horizon.is_some_and(|horizon| {
                let horizon_deg = horizon.at(azimuth_deg, identity.gate_index);
                horizon_deg.is_finite() && elevation_deg as f32 <= horizon_deg
            })
        };
        targets.push(InspectorSpatialTarget {
            latitude_deg: lat as f32,
            longitude_deg: lon as f32,
            height_msl_m: z_msl as f32,
            elevation_deg,
            point_weight,
            blocked,
        });
    }
    Ok(targets)
}

fn read_bulk_species_targets(
    file: &WrfFile,
    time_index: usize,
    fields: &WrfRadarFields,
    targets: &[InspectorSpatialTarget],
) -> Result<Vec<Vec<crate::wrf_radar_physics::BulkContribution>>, WhyThisGateUnavailable> {
    let target_weights = targets
        .iter()
        .map(|target| inspector_cell_weights(fields, target))
        .collect::<Vec<_>>();
    let needed_indices = target_weights
        .iter()
        .flatten()
        .map(|(index, _)| *index)
        .collect::<std::collections::BTreeSet<_>>();
    let expected = fields.cells() * fields.nz;
    let temperature = file.temperature(time_index).map_err(|error| {
        WhyThisGateUnavailable::WorkerFailed(format!("read species temperature: {error}"))
    })?;
    if temperature.len() != expected {
        return Err(WhyThisGateUnavailable::StaleFrameWitness);
    }
    let temperature = needed_indices
        .iter()
        .map(|&index| (index, temperature[index] as f32))
        .collect::<HashMap<_, _>>();
    file.clear_cache();
    let pressure = file.full_pressure(time_index).map_err(|error| {
        WhyThisGateUnavailable::WorkerFailed(format!("read species pressure: {error}"))
    })?;
    if pressure.len() != expected {
        return Err(WhyThisGateUnavailable::StaleFrameWitness);
    }
    let qv = if file.has_var("QVAPOR") {
        file.read_var("QVAPOR", time_index).map_err(|error| {
            WhyThisGateUnavailable::WorkerFailed(format!("read species QVAPOR: {error}"))
        })?
    } else {
        vec![0.0; expected]
    };
    if qv.len() != expected {
        return Err(WhyThisGateUnavailable::StaleFrameWitness);
    }
    let air_density = needed_indices
        .iter()
        .map(|&index| {
            (
                index,
                crate::wrf_radar_physics::air_density(
                    pressure[index] as f32,
                    temperature[&index],
                    qv[index] as f32,
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    drop(pressure);
    drop(qv);
    file.clear_cache();

    let present_fields = MICROPHYSICS_FIELD_NAMES
        .iter()
        .filter(|name| file.has_var(name))
        .map(|name| (*name).to_owned())
        .collect::<Vec<_>>();
    let profile = crate::wrf_radar_physics::detect_scheme(
        file.global_attr_i32("MP_PHYSICS").ok(),
        &present_fields,
    );
    if profile.capability == crate::wrf_radar_physics::MicrophysicsCapability::Unsupported {
        return Err(WhyThisGateUnavailable::UnsupportedSourceContract);
    }
    let mut output = Vec::with_capacity(SPECIES_READ_SPECS.len());
    for spec in SPECIES_READ_SPECS {
        let Some(mass_name) = first_existing_var(file, spec.mass) else {
            output.push(vec![
                crate::wrf_radar_physics::BulkContribution::default();
                targets.len()
            ]);
            continue;
        };
        let number = read_optional_first_var_f32(file, time_index, spec.number, expected)
            .map_err(WhyThisGateUnavailable::WorkerFailed)?;
        let volume = read_optional_first_var_f32(file, time_index, spec.volume, expected)
            .map_err(WhyThisGateUnavailable::WorkerFailed)?;
        let mass = file.read_var(mass_name, time_index).map_err(|error| {
            WhyThisGateUnavailable::WorkerFailed(format!("read species {mass_name}: {error}"))
        })?;
        if mass.len() != expected {
            return Err(WhyThisGateUnavailable::StaleFrameWitness);
        }
        let contributions = needed_indices
            .iter()
            .map(|&index| {
                let q = mass[index] as f32;
                let contribution = if q.is_finite() && q > 0.0 {
                    crate::wrf_radar_physics::bulk_sband_contribution(
                        crate::wrf_radar_physics::BulkSpeciesInput {
                            kind: spec.kind,
                            q_kgkg: q,
                            number_per_kg: number.as_ref().map(|values| values[index]),
                            volume_m3_per_kg: volume.as_ref().map(|values| values[index]),
                            temperature_k: temperature[&index],
                            air_density_kgm3: air_density[&index],
                        },
                        &profile,
                    )
                } else {
                    crate::wrf_radar_physics::BulkContribution::default()
                };
                (index, contribution)
            })
            .collect::<HashMap<_, _>>();
        let mut sampled = Vec::with_capacity(targets.len());
        for weights in &target_weights {
            let mut accumulator = crate::wrf_radar_physics::PolarAccumulator::default();
            for &(index, weight) in weights {
                accumulator.add(weight, contributions[&index]);
            }
            sampled.push(intrinsic_as_contribution(accumulator.finalize()));
        }
        output.push(sampled);
        drop(mass);
        drop(number);
        drop(volume);
        file.clear_cache();
    }
    Ok(output)
}

fn inspector_cell_weights(
    fields: &WrfRadarFields,
    target: &InspectorSpatialTarget,
) -> Vec<(usize, f32)> {
    let Some(stencil) = horizontal_stencil(fields, target.latitude_deg, target.longitude_deg)
    else {
        return Vec::new();
    };
    let cells = fields.cells();
    let mut weights = Vec::with_capacity(8);
    for (column, horizontal_weight) in stencil {
        if horizontal_weight <= 0.0 {
            continue;
        }
        let Some((k, vertical_fraction)) =
            bracket_height(fields, cells, column, target.height_msl_m)
        else {
            continue;
        };
        let lower = k * cells + column;
        let upper = (k + 1) * cells + column;
        if lerp(fields.u[lower], fields.u[upper], vertical_fraction).is_none()
            || lerp(fields.v[lower], fields.v[upper], vertical_fraction).is_none()
            || lerp(fields.w[lower], fields.w[upper], vertical_fraction).is_none()
        {
            continue;
        }
        weights.push((lower, horizontal_weight * (1.0 - vertical_fraction)));
        weights.push((upper, horizontal_weight * vertical_fraction));
    }
    let sum = weights.iter().map(|(_, weight)| *weight).sum::<f32>();
    if sum > 0.0 {
        for (_, weight) in &mut weights {
            *weight /= sum;
        }
    }
    weights
}

fn hydrometeor_name(kind: crate::wrf_radar_physics::HydrometeorKind) -> &'static str {
    use crate::wrf_radar_physics::HydrometeorKind;
    match kind {
        HydrometeorKind::CloudWater => "Cloud water",
        HydrometeorKind::Rain => "Rain",
        HydrometeorKind::CloudIce => "Cloud ice",
        HydrometeorKind::Snow => "Snow",
        HydrometeorKind::Graupel => "Graupel",
        HydrometeorKind::Hail => "Hail",
    }
}

/// Result of a finished synthetic-radar job: one volume per WRF forecast time,
/// ready to feed the loop engine as a looping sequence.
pub struct SyntheticRadarOutput {
    pub label: String,
    pub volumes: Vec<Arc<RadarVolume>>,
    pub notes: Vec<String>,
    /// Effective build fingerprint: [`SyntheticRadarConfig::data_fingerprint`]
    /// plus the run/domain/grid and basename-free scene identities/time indices
    /// that built these volumes. The install path folds it into each frame's
    /// history key so either changed settings OR changed WRF input replaces a
    /// stale result without exposing private absolute paths.
    pub config_fingerprint: u64,
    /// Aligned one-for-one with `volumes`; contains private paths and therefore
    /// must never be serialized or copied into volume provenance.
    pub(crate) frame_sources: Vec<SyntheticFrameSourceDescriptor>,
}

impl std::fmt::Debug for SyntheticRadarOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyntheticRadarOutput")
            .field("label", &self.label)
            .field("volumes", &self.volumes.len())
            .field("notes", &self.notes)
            .field("config_fingerprint", &self.config_fingerprint)
            .field("frame_sources", &self.frame_sources.len())
            .finish()
    }
}

pub struct SyntheticRadarTask {
    pub label: String,
    pub rx: Receiver<SyntheticRadarMessage>,
    pub(crate) cancel: Arc<AtomicBool>,
}

impl SyntheticRadarTask {
    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn cancellation_requested(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }
}

pub enum SyntheticRadarReplayMessage {
    Progress(String),
    Done(Result<SyntheticRadarReplayOutput, String>),
}

pub struct SyntheticRadarReplayFrame {
    pub observed: Arc<RadarVolume>,
    pub simulated: Arc<RadarVolume>,
    pub difference: Arc<RadarVolume>,
    pub unavailable_observed_moments: Vec<UnavailableObservedMoment>,
    pub(crate) source_descriptor: SyntheticFrameSourceDescriptor,
}

pub struct SyntheticRadarReplayOutput {
    pub label: String,
    pub frames: Vec<SyntheticRadarReplayFrame>,
    pub notes: Vec<String>,
    pub config_fingerprint: u64,
}

pub struct SyntheticRadarReplayTask {
    pub label: String,
    pub rx: Receiver<SyntheticRadarReplayMessage>,
    pub(crate) cancel: Arc<AtomicBool>,
}

impl SyntheticRadarReplayTask {
    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn cancellation_requested(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }
}

/// Existing-file worker seam for the ThreeStacked validation workspace. It
/// reuses the normal inventory, WRF field-read, temporal-plan, and progress
/// worker, then attaches the retained observed source and exact differences
/// without rereading either WRF or observed gate values.
pub fn spawn_synthetic_radar_replay(
    paths: Vec<PathBuf>,
    mut config: SyntheticRadarConfig,
    observed: Arc<RadarVolume>,
) -> SyntheticRadarReplayTask {
    let label = format!("Exact radar replay from {} WRF source(s)", paths.len());
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let template = match ExactScanTemplate::from_volume(&observed) {
        Ok(template) => template,
        Err(error) => {
            let _ = tx.send(SyntheticRadarReplayMessage::Done(Err(format!(
                "extract exact observed scan template: {error}"
            ))));
            return SyntheticRadarReplayTask { label, rx, cancel };
        }
    };
    normalize_replay_config_for_observed(&mut config, &observed);
    if matches!(
        config.propagation_geometry,
        PropagationGeometry::WrfRefractivityResearch
    ) {
        config.site_lat_deg = observed.site.latitude_deg.map(f64::from);
        config.site_lon_deg = observed.site.longitude_deg.map(f64::from);
        config.antenna_msl_m = observed.site.elevation_m.map(f64::from);
    }
    config.exact_replay_template = Some(Arc::new(template));
    let base = spawn_synthetic_radar(paths, config);
    let cancel = Arc::clone(&base.cancel);
    let bridge_cancel = Arc::clone(&cancel);
    std::thread::Builder::new()
        .name("wrf-exact-replay-products".to_string())
        .spawn(move || {
            while let Ok(message) = base.rx.recv() {
                match message {
                    SyntheticRadarMessage::Progress(progress) => {
                        let _ = tx.send(SyntheticRadarReplayMessage::Progress(progress));
                    }
                    // Exact replay remains all-or-nothing because its observed,
                    // simulated, and difference panes must stay aligned.
                    SyntheticRadarMessage::Preview(_) => {}
                    SyntheticRadarMessage::Done(result) => {
                        let replay_result = result.and_then(|output| {
                            check_synthetic_radar_cancel(&bridge_cancel)?;
                            let mut frames = Vec::with_capacity(output.volumes.len());
                            let mut notes = output.notes;
                            notes.push(
                                "Exact replay uses observed radial Nyquist/PRT; custom coupled PRF, stage diagnostics, and manual folding were disabled"
                                    .to_string(),
                            );
                            for (mut simulated, source_descriptor) in
                                output.volumes.into_iter().zip(output.frame_sources)
                            {
                                check_synthetic_radar_cancel(&bridge_cancel)?;
                                copy_observed_ray_instrument_metadata(
                                    &observed,
                                    Arc::make_mut(&mut simulated),
                                )?;
                                let overlap = build_difference_volume_overlap(&observed, &simulated)
                                    .map_err(|error| {
                                        format!(
                                            "build exact-geometry replay difference: {error}"
                                        )
                                    })?;
                                if !overlap.unavailable_observed_moments.is_empty() {
                                    notes.push(format!(
                                        "Replay unavailable observed moments: {}",
                                        overlap
                                            .unavailable_observed_moments
                                            .iter()
                                            .map(|entry| format!(
                                                "cut {} {} ({})",
                                                entry.cut, entry.moment, entry.reason
                                            ))
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    ));
                                }
                                frames.push(SyntheticRadarReplayFrame {
                                    observed: Arc::clone(&observed),
                                    simulated,
                                    difference: Arc::new(overlap.volume),
                                    unavailable_observed_moments: overlap
                                        .unavailable_observed_moments,
                                    source_descriptor,
                                });
                            }
                            Ok(SyntheticRadarReplayOutput {
                                label: output.label,
                                frames,
                                notes,
                                config_fingerprint: output.config_fingerprint,
                            })
                        });
                        let replay_result =
                            normalize_synthetic_radar_cancel(replay_result, &bridge_cancel);
                        let _ = tx.send(SyntheticRadarReplayMessage::Done(replay_result));
                        break;
                    }
                }
            }
        })
        .expect("spawn exact replay product bridge");
    SyntheticRadarReplayTask { label, rx, cancel }
}

/// Spawn a worker that turns each forecast time of the given wrfout file(s)
/// into a simulated [`RadarVolume`]. Streams progress, then a `Done`.
pub fn spawn_synthetic_radar(
    paths: Vec<PathBuf>,
    config: SyntheticRadarConfig,
) -> SyntheticRadarTask {
    let mut label = if paths.len() == 1 {
        format!("Simulated radar from {}", display_name(&paths[0]))
    } else {
        format!("Simulated radar from {} WRF files", paths.len())
    };
    if config.polarimetric_kernel.is_property_tmatrix() {
        label.push_str(if config.polarimetric_kernel.is_hybrid() {
            " [P3 Hybrid]"
        } else {
            " [Full P3 T-matrix experimental]"
        });
    }
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&cancel);
    let label_for_thread = label.clone();
    std::thread::Builder::new()
        .name("rw-ui-wrf-synth-radar".to_string())
        .spawn(move || {
            let result =
                build_synthetic_from_paths(&paths, &config, &label_for_thread, &tx, &worker_cancel);
            let result = normalize_synthetic_radar_cancel(result, &worker_cancel);
            let _ = tx.send(SyntheticRadarMessage::Done(result));
        })
        .expect("spawn WRF synthetic-radar worker");
    SyntheticRadarTask { label, rx, cancel }
}

/// Spawn native HRRR/RRFS GRIB -> ordinary synthetic `RadarVolume` frames.
/// The worker reuses the same message/result contract as wrfout simulation so
/// installation, looping, Refresh, and CfRadial export need no parallel path.
enum TMatrixExecutionRuntime {
    NotApplicable,
    CpuReference {
        reason: String,
    },
    NvidiaCuda {
        service: Arc<app_ui::wrf_tmatrix_cuda::WrfTMatrixCudaBatchService>,
    },
}

fn resolve_tmatrix_execution_runtime(
    config: &SyntheticRadarConfig,
    tx: &Sender<SyntheticRadarMessage>,
    cancel: &AtomicBool,
) -> Result<(SyntheticRadarConfig, TMatrixExecutionRuntime), String> {
    check_synthetic_radar_cancel(cancel)?;
    let mut runtime_config = config.clone();
    runtime_config.runtime_cuda_service = None;
    if !config.polarimetric_kernel.is_property_tmatrix() {
        return Ok((runtime_config, TMatrixExecutionRuntime::NotApplicable));
    }
    if matches!(
        config.compute_preference,
        SyntheticRadarComputePreference::Cpu
    ) {
        return Ok((
            runtime_config,
            TMatrixExecutionRuntime::CpuReference {
                reason: "CPU selected by user".to_owned(),
            },
        ));
    }
    if !matches!(
        config.property_tmatrix_table_source,
        app_ui::wrf_tmatrix_assets::PropertyTMatrixTableSourceKind::LegacyEmbeddedSResearchV1
    ) || config.radar_frequency_mhz != PROPERTY_TMATRIX_RESEARCH_FREQUENCY_MHZ
    {
        return Ok((
            runtime_config,
            TMatrixExecutionRuntime::CpuReference {
                reason: "CUDA v1 requires the authenticated BowEcho 2.8 GHz S research pack"
                    .to_owned(),
            },
        ));
    }

    let _ = tx.send(SyntheticRadarMessage::Progress(
        "Initializing NVIDIA CUDA T-matrix worker and uploading dry LUTs…".to_owned(),
    ));
    match app_ui::wrf_tmatrix_cuda::WrfTMatrixCudaBatchService::open_preferred() {
        Ok(service) => {
            let service = Arc::new(service);
            let report = service.report();
            let _ = tx.send(SyntheticRadarMessage::Progress(format!(
                "CUDA active: {} · compute {}.{} · {}; CPU retains PSD weights and ordered reductions",
                report.device.name,
                report.device.compute_capability_major,
                report.device.compute_capability_minor,
                report.device.kernel_artifact,
            )));
            runtime_config.runtime_cuda_service = Some(Arc::clone(&service));
            check_synthetic_radar_cancel(cancel)?;
            Ok((
                runtime_config,
                TMatrixExecutionRuntime::NvidiaCuda { service },
            ))
        }
        Err(error) => {
            let reason = error.to_string();
            let _ = tx.send(SyntheticRadarMessage::Progress(format!(
                "CUDA unavailable ({reason}); continuing on the exact CPU reference path"
            )));
            check_synthetic_radar_cancel(cancel)?;
            Ok((
                runtime_config,
                TMatrixExecutionRuntime::CpuReference { reason },
            ))
        }
    }
}

fn append_execution_fragment(volume: &mut RadarVolume, fragment: &str) {
    match volume.metadata.forward_operator_config.as_mut() {
        Some(config) if !config.is_empty() => {
            config.push_str("; ");
            config.push_str(fragment);
        }
        Some(config) => config.push_str(fragment),
        None => volume.metadata.forward_operator_config = Some(fragment.to_owned()),
    }
}

/// Keep a partial first-tilt preview honest when CUDA disables after the
/// volume metadata was initialized but before that cut is published. Final
/// volumes receive the complete counters from `finish_tmatrix_execution_report`;
/// this marker exists only to prevent an in-progress preview from claiming an
/// accelerator that has already fallen back.
fn refresh_in_progress_tmatrix_execution_provenance(
    volume: &mut RadarVolume,
    config: &SyntheticRadarConfig,
) {
    let Some(service) = config.runtime_cuda_service.as_deref() else {
        return;
    };
    let report = service.report();
    let Some(reason) = report.fallback_reason.as_ref() else {
        return;
    };
    let existing = volume
        .metadata
        .forward_operator_config
        .as_deref()
        .unwrap_or_default();
    if existing.contains("tmatrix_execution_initial_backend=cuda_then_cpu_replay")
        || existing.contains("tmatrix_execution_progress_backend=cuda_then_cpu_replay")
    {
        return;
    }
    append_execution_fragment(
        volume,
        &format!(
            "tmatrix_execution_progress_backend=cuda_then_cpu_replay; cuda_nodes_completed={}; cuda_batches_completed={}; cuda_fallback={reason}",
            report.nodes_completed, report.batches_completed,
        ),
    );
}

fn finish_tmatrix_execution_report(
    output: &mut SyntheticRadarOutput,
    preference: SyntheticRadarComputePreference,
    runtime: &TMatrixExecutionRuntime,
) {
    let preference = preference.label();
    let (note, fragment) = match runtime {
        TMatrixExecutionRuntime::NotApplicable => return,
        TMatrixExecutionRuntime::CpuReference { reason } => (
            format!("T-matrix execution: CPU reference ({reason})"),
            format!(
                "tmatrix_compute_preference={preference}; tmatrix_execution_backend=cpu_reference; cuda_fallback={reason}"
            ),
        ),
        TMatrixExecutionRuntime::NvidiaCuda { service } => {
            let report = service.report();
            let identity = format!(
                "{} compute {}.{} via {}",
                report.device.name,
                report.device.compute_capability_major,
                report.device.compute_capability_minor,
                report.device.kernel_artifact,
            );
            if let Some(reason) = report.fallback_reason {
                (
                    format!(
                        "T-matrix CUDA used {identity}; completed {} nodes in {}/{} batches before fallback ({reason}); the discarded/remaining work replayed on CPU",
                        report.nodes_completed, report.batches_completed, report.batches_submitted,
                    ),
                    format!(
                        "tmatrix_compute_preference={preference}; tmatrix_execution_backend=cuda_then_cpu_replay; cuda_device={}; cuda_compute={}.{}; cuda_kernel={}; cuda_requests={}; cuda_batches_completed={}; cuda_batches_submitted={}; cuda_nodes_completed={}; cuda_nodes_submitted={}; cuda_fallback={reason}",
                        report.device.name,
                        report.device.compute_capability_major,
                        report.device.compute_capability_minor,
                        report.device.kernel_artifact,
                        report.requests_submitted,
                        report.batches_completed,
                        report.batches_submitted,
                        report.nodes_completed,
                        report.nodes_submitted,
                    ),
                )
            } else {
                (
                    format!(
                        "T-matrix CUDA used {identity}; completed {} particle nodes in {} batches ({} requests); CPU retained exact PSD reduction",
                        report.nodes_completed, report.batches_completed, report.requests_submitted,
                    ),
                    format!(
                        "tmatrix_compute_preference={preference}; tmatrix_execution_backend=nvidia_cuda; cuda_device={}; cuda_compute={}.{}; cuda_kernel={}; cuda_requests={}; cuda_batches={}; cuda_nodes={}; cuda_cpu_reduction=true",
                        report.device.name,
                        report.device.compute_capability_major,
                        report.device.compute_capability_minor,
                        report.device.kernel_artifact,
                        report.requests_submitted,
                        report.batches_completed,
                        report.nodes_completed,
                    ),
                )
            }
        }
    };
    output.notes.push(note);
    for volume in &mut output.volumes {
        append_execution_fragment(Arc::make_mut(volume), &fragment);
    }
}

fn build_synthetic_from_paths(
    paths: &[PathBuf],
    config: &SyntheticRadarConfig,
    label: &str,
    tx: &Sender<SyntheticRadarMessage>,
    cancel: &AtomicBool,
) -> Result<SyntheticRadarOutput, String> {
    check_synthetic_radar_cancel(cancel)?;
    // Reject malformed jobs before retaining a CUDA primary context or
    // uploading immutable LUTs. The inner worker repeats the cheap checks at
    // its own boundary, but accelerator startup must never precede them.
    config.validate_science_contract()?;
    if paths.is_empty() {
        return Err("No WRF files selected".to_owned());
    }
    if !paths
        .iter()
        .any(|path| is_supported_wrf_file(path))
    {
        return Err("No supported WRF files selected".to_owned());
    }
    if config.polarimetric_kernel.is_property_tmatrix() {
        let pack_progress = |stage: &str| {
            let _ = tx.send(SyntheticRadarMessage::Progress(stage.to_owned()));
        };
        app_ui::wrf_tmatrix_assets::load_property_tmatrix_tables_exact_with_progress(
            config.property_tmatrix_table_source,
            f64::from(config.radar_frequency_mhz) * 1.0e6,
            &pack_progress,
        )
        .map(drop)
        .map_err(|error| format!("prepare property T-matrix research data pack: {error}"))?;
        check_synthetic_radar_cancel(cancel)?;
    }
    let (runtime_config, runtime) = resolve_tmatrix_execution_runtime(config, tx, cancel)?;
    let mut output = build_synthetic_from_paths_inner(paths, &runtime_config, label, tx, cancel)?;
    check_synthetic_radar_cancel(cancel)?;
    finish_tmatrix_execution_report(&mut output, config.compute_preference, &runtime);
    Ok(output)
}

fn build_synthetic_from_paths_inner(
    paths: &[PathBuf],
    config: &SyntheticRadarConfig,
    label: &str,
    tx: &Sender<SyntheticRadarMessage>,
    cancel: &AtomicBool,
) -> Result<SyntheticRadarOutput, String> {
    check_synthetic_radar_cancel(cancel)?;
    config.validate_science_contract()?;
    if paths.is_empty() {
        return Err("No WRF files selected".to_string());
    }
    let files: Vec<PathBuf> = paths
        .iter()
        .filter(|path| is_supported_wrf_file(path))
        .cloned()
        .collect();
    if files.is_empty() {
        return Err("No supported WRF files selected".to_string());
    }

    let _ = tx.send(SyntheticRadarMessage::Progress(
        "Inventorying WRF run, domain, grid, and internal times…".to_owned(),
    ));
    let selected = inventory_selected_wrf_paths(&files)?;
    check_synthetic_radar_cancel(cancel)?;
    let scenes = selected.group.scenes.clone();
    let config_fingerprint = scene_build_fingerprint(config, &selected.group);
    let property_build_reservation_bytes = if config.polarimetric_kernel.is_property_tmatrix() {
        let estimate = temporal_memory_estimate(&selected.group, config, scenes.len())?;
        estimate
            .shared_static_bytes
            .checked_add(estimate.output_bytes)
            .ok_or_else(|| "Property T-matrix output-memory reservation overflowed".to_string())?
    } else {
        0
    };

    let mut notes = selected
        .notes
        .into_iter()
        .map(|note| format!("{}: {}", note.source_name, note.message))
        .collect::<Vec<_>>();
    if config.polarimetric_kernel.is_property_tmatrix() {
        notes.push(if config.polarimetric_kernel.is_hybrid() {
            format!(
                "P3 Hybrid output is research_only_unvalidated; native P3/ISHMAEL PSD T-matrix is used for supported cells and only table-domain/shape omissions or the typed WRF 2 µm source-state mass gap use audited {}; policy and fallback counts are retained in scene/volume metadata",
                app_ui::wrf_tmatrix_scene::WrfTMatrixScatteringPolicy::HybridBulkRayleighV1.identifier(),
            )
        } else if matches!(
            config.atmosphere_time_mode,
            AtmosphereTimeMode::RawStateLinear
        ) {
            "Experimental Full P3/ISHMAEL T-matrix output is research_only_unvalidated and strict fail-closed; raw_state_linear blends full spatial/temporal state before one closure/scattering evaluation"
                .to_string()
        } else {
            "Experimental Full P3/ISHMAEL T-matrix output is research_only_unvalidated and strict fail-closed; scheme-native PSD integration uses the exact selected table pack; not operational calibration"
                .to_string()
        });
    }
    if config.atmosphere_time_mode.uses_adjacent_scene() {
        return build_temporal_synthetic_from_scenes(
            &selected.group,
            config,
            label,
            tx,
            notes,
            config_fingerprint,
            cancel,
        );
    }

    let mut volumes = Vec::new();
    let mut frame_sources = Vec::new();
    let one_scene_per_file = {
        let mut identities = std::collections::BTreeSet::new();
        scenes
            .iter()
            .all(|scene| identities.insert(scene.source_identity.clone()))
    };
    let frame_total = scenes.len();
    let mut opened_path: Option<PathBuf> = None;
    let mut opened_file: Option<WrfFile> = None;
    for (frame_index, scene) in scenes.iter().enumerate() {
        check_synthetic_radar_cancel(cancel)?;
        if opened_path.as_ref() != Some(&scene.path) {
            opened_file = Some(WrfFile::open(&scene.path).map_err(|error| {
                format!(
                    "Open inventoried WRF {}: {error}",
                    display_name(&scene.path)
                )
            })?);
            opened_path = Some(scene.path.clone());
        }
        let file = opened_file
            .as_ref()
            .expect("inventoried WRF file opened above");
        let name = display_name(&scene.path);
        let timeidx = scene.time_index;
        let frame_prefix = if one_scene_per_file && frame_total > 1 {
            // Preserve the established multi-file progress vocabulary.
            format!("file {}/{frame_total} ({name}): ", frame_index + 1)
        } else if frame_total > 1 {
            format!(
                "frame {}/{frame_total} ({name}, time {}/{}): ",
                frame_index + 1,
                timeidx + 1,
                file.nt
            )
        } else if file.nt > 1 {
            format!("Simulating {name} (time {}/{}): ", timeidx + 1, file.nt)
        } else {
            format!("Simulating {name}: ")
        };
        let progress = |stage: &str| {
            let _ = tx.send(SyntheticRadarMessage::Progress(format!(
                "{frame_prefix}{stage}"
            )));
        };
        progress("reading…");
        let fields = match read_wrf_radar_fields_for_config_reporting(
            file,
            &scene.source_identity,
            timeidx,
            config,
            &progress,
            property_build_reservation_bytes,
            Some(cancel),
        ) {
            Ok(fields) => fields,
            Err(error) => {
                record_or_propagate_scene_failure(
                    config,
                    &mut notes,
                    format!("{name} time {timeidx}: {error}"),
                )?;
                continue;
            }
        };
        check_synthetic_radar_cancel(cancel)?;
        let valid_time = scene
            .time
            .valid_time()
            .cloned()
            .expect("scene adapter rejects untimed scenes");
        let first_cut_preview =
            |partial: &RadarVolume, completed_cuts: usize, total_cuts: usize| {
                if frame_index == 0 && completed_cuts == 1 {
                    send_first_cut_preview(
                        tx,
                        label,
                        config_fingerprint,
                        scene,
                        partial,
                        completed_cuts,
                        total_cuts,
                    );
                }
            };
        let mut volume = match try_build_synthetic_volume_reporting_previewing_cancellable(
            &fields,
            valid_time,
            config,
            &progress,
            &first_cut_preview,
            cancel,
        ) {
            Ok(volume) => volume,
            Err(error) => {
                record_or_propagate_scene_failure(
                    config,
                    &mut notes,
                    format!("{name} time {timeidx}: {error}"),
                )?;
                continue;
            }
        };
        append_scene_provenance(&mut volume, scene);
        // Self-document the gate spacing when grid-matching is on: record the
        // effective size and whether it came from the grid DX or fell back.
        let gate_note = if config.match_gate_to_grid {
            let eff = effective_gate_spacing(config, fields.dx_m);
            if matched_grid_dx(config, fields.dx_m) {
                format!(", gate {eff:.0} m (grid DX)")
            } else {
                format!(", gate {eff:.0} m (grid DX unavailable)")
            }
        } else {
            String::new()
        };
        let polar_note = fields
            .dual_pol_status
            .as_deref()
            .map(|status| format!(", {status}"))
            .unwrap_or_default();
        if let Some(note) = refractivity_run_note(&fields) {
            notes.push(format!("{name} time {timeidx}: {note}"));
        }
        notes.push(format!(
            "{name} time {timeidx} ({}): {} radials from {}{gate_note}{polar_note}",
            scene_time_authority(&scene.time),
            volume.metadata.decoded_radial_count,
            fields.ref_source,
        ));
        let volume = Arc::new(volume);
        frame_sources.push(SyntheticFrameSourceDescriptor::wrf(
            volumes.len(),
            &volume,
            scene.clone(),
            None,
            None,
            config,
            config_fingerprint,
        ));
        volumes.push(volume);
    }

    if volumes.is_empty() {
        return Err(if notes.is_empty() {
            "WRF produced no simulated radar volumes".to_string()
        } else {
            format!(
                "WRF produced no simulated radar volumes: {}",
                notes.join("; ")
            )
        });
    }
    // `WrfSceneInventory` already supplies strict chronological order from
    // internal Times; retain it exactly so timeidx/provenance stay aligned.
    Ok(SyntheticRadarOutput {
        label: label.to_string(),
        volumes,
        notes,
        config_fingerprint,
        frame_sources,
    })
}

fn send_first_cut_preview(
    tx: &Sender<SyntheticRadarMessage>,
    label: &str,
    config_fingerprint: u64,
    scene: &WrfScene,
    partial: &RadarVolume,
    completed_cuts: usize,
    total_cuts: usize,
) {
    debug_assert_eq!(completed_cuts, 1);
    let mut volume = partial.clone();
    append_scene_provenance(&mut volume, scene);
    volume.metadata.archive_version = Some("simulated-wrf-partial-preview".to_owned());
    let preview_fragment = format!(
        "build_complete=false; completed_tilts={completed_cuts}; planned_tilts={total_cuts}"
    );
    match volume.metadata.forward_operator_config.as_mut() {
        Some(config) => {
            config.push_str("; ");
            config.push_str(&preview_fragment);
        }
        None => volume.metadata.forward_operator_config = Some(preview_fragment),
    }
    volume.metadata.scan_name = Some(format!(
        "{} (processing: tilt {completed_cuts}/{total_cuts})",
        volume
            .metadata
            .scan_name
            .as_deref()
            .unwrap_or("BowEcho synthetic radar")
    ));
    let _ = tx.send(SyntheticRadarMessage::Preview(SyntheticRadarPreview {
        label: format!("{label} · tilt {completed_cuts}/{total_cuts} preview"),
        volume: Arc::new(volume),
        config_fingerprint,
        completed_cuts,
        total_cuts,
    }));
}

fn record_or_propagate_scene_failure(
    config: &SyntheticRadarConfig,
    notes: &mut Vec<String>,
    message: String,
) -> Result<(), String> {
    if config.polarimetric_kernel.is_property_tmatrix() {
        Err(message)
    } else {
        notes.push(message);
        Ok(())
    }
}

fn build_temporal_synthetic_from_scenes(
    group: &WrfSceneGroup,
    config: &SyntheticRadarConfig,
    label: &str,
    tx: &Sender<SyntheticRadarMessage>,
    mut notes: Vec<String>,
    config_fingerprint: u64,
    cancel: &AtomicBool,
) -> Result<SyntheticRadarOutput, String> {
    check_synthetic_radar_cancel(cancel)?;
    if !matches!(config.scan_timing, ScanTiming::TimedVolume) {
        return Err(
            "Adjacent-scene atmosphere interpolation requires Timed volume scan timing".to_string(),
        );
    }
    let scan_duration_ms = config.planned_scan_duration_ms();
    let scan_duration = Duration::milliseconds(scan_duration_ms);
    let mut plans = Vec::with_capacity(group.scenes.len());
    for (frame_index, scene) in group.scenes.iter().enumerate() {
        check_synthetic_radar_cancel(cancel)?;
        plans.push(
            plan_for_scene(
                group,
                frame_index,
                scan_duration,
                config.atmosphere_time_mode,
                config.missing_neighbor_policy,
            )
            .map_err(|error| {
                format!(
                    "Plan temporal sampling for {}: {error}",
                    display_name(&scene.path)
                )
            })?,
        );
    }
    let emitted_frame_count = plans.iter().filter(|plan| plan.is_some()).count();
    let requires_two_scenes = plans
        .iter()
        .flatten()
        .any(|plan| matches!(plan.outcome, TemporalSamplingOutcome::LinearAdjacent));
    let estimate = temporal_memory_estimate(group, config, emitted_frame_count)?;
    let budget_bytes = config
        .temporal_memory_budget_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| "Temporal memory budget overflows address space".to_string())?;
    let required_bytes = temporal_required_peak_bytes(estimate, requires_two_scenes)
        .ok_or_else(|| "Temporal memory estimate overflowed".to_string())?;
    if required_bytes <= budget_bytes {
        let scene_label = if requires_two_scenes {
            "two-scene"
        } else {
            "single-scene held"
        };
        let property_suffix = if config.polarimetric_kernel.is_property_tmatrix() {
            " minimum; exact sparse T-matrix cache is checked after each scene read"
        } else {
            ""
        };
        notes.push(format!(
            "Temporal {scene_label} preflight: {:.2} GiB estimated peak within {:.2} GiB budget{property_suffix}",
            required_bytes as f64 / 1024.0_f64.powi(3),
            budget_bytes as f64 / 1024.0_f64.powi(3),
        ));
    } else {
        return Err(temporal_build_budget_error(required_bytes, budget_bytes));
    }

    let mut cache: TwoSceneCache<(String, usize), Arc<WrfRadarFields>> = TwoSceneCache::default();
    let mut volumes = Vec::new();
    let mut frame_sources = Vec::new();
    for (frame_index, (scene, plan)) in group.scenes.iter().zip(plans).enumerate() {
        check_synthetic_radar_cancel(cancel)?;
        let Some(plan) = plan else {
            notes.push(format!(
                "{} time {} dropped: no adjacent WRF scene covers the timed scan",
                display_name(&scene.path),
                scene.time_index
            ));
            continue;
        };

        let frame_prefix = format!(
            "frame {}/{} ({}, time {}): ",
            frame_index + 1,
            group.scenes.len(),
            display_name(&scene.path),
            scene.time_index + 1,
        );
        let progress = |stage: &str| {
            let _ = tx.send(SyntheticRadarMessage::Progress(format!(
                "{frame_prefix}{stage}"
            )));
        };
        let anchor = match cached_temporal_fields(
            &mut cache,
            scene,
            config,
            &progress,
            estimate,
            budget_bytes,
            cancel,
        ) {
            Ok(fields) => fields,
            Err(error) => {
                record_or_propagate_scene_failure(
                    config,
                    &mut notes,
                    format!(
                        "{} time {}: {error}",
                        display_name(&scene.path),
                        scene.time_index
                    ),
                )?;
                continue;
            }
        };
        if let Some(status) = anchor.dual_pol_status.as_deref() {
            notes.push(format!(
                "{} time {} microphysics: {status}",
                display_name(&scene.path),
                scene.time_index
            ));
        }
        if let Some(note) = refractivity_run_note(&anchor) {
            notes.push(format!(
                "{} time {}: {note}",
                display_name(&scene.path),
                scene.time_index
            ));
        }
        if (config.dual_pol || config.terminal_fall_speed) && !anchor.has_polarimetric_input() {
            notes.push(format!(
                "{} time {}: requested polarimetric/fall-speed physics unavailable; emitting explicitly labeled scalar fallback",
                display_name(&scene.path),
                scene.time_index
            ));
        }
        let valid_time = scene
            .time
            .valid_time()
            .cloned()
            .expect("scene adapter rejects untimed scenes");
        let first_cut_preview =
            |partial: &RadarVolume, completed_cuts: usize, total_cuts: usize| {
                if frame_index == 0 && completed_cuts == 1 {
                    send_first_cut_preview(
                        tx,
                        label,
                        config_fingerprint,
                        scene,
                        partial,
                        completed_cuts,
                        total_cuts,
                    );
                }
            };

        let (mut volume, runtime_outcome) = match plan.outcome {
            TemporalSamplingOutcome::LinearAdjacent => {
                let neighbor_locator = plan
                    .neighbor
                    .as_ref()
                    .expect("linear temporal plan carries a neighbor");
                let neighbor_scene = group
                    .scenes
                    .iter()
                    .find(|candidate| {
                        candidate.source_identity == neighbor_locator.source_identity
                            && candidate.time_index == neighbor_locator.time_index
                    })
                    .expect("temporal planner selected a scene from this group");
                let neighbor = match cached_temporal_fields(
                    &mut cache,
                    neighbor_scene,
                    config,
                    &progress,
                    estimate,
                    budget_bytes,
                    cancel,
                ) {
                    Ok(fields) => TemporalNeighborResolution::Fields(fields),
                    Err(error) => match config.missing_neighbor_policy {
                        MissingNeighborPolicy::HoldAnchor => {
                            notes.push(format!(
                                "{} time {} held at anchor: adjacent scene read failed ({error})",
                                display_name(&scene.path),
                                scene.time_index
                            ));
                            let volume =
                                try_build_synthetic_volume_reporting_previewing_cancellable(
                                    &anchor,
                                    valid_time,
                                    config,
                                    &progress,
                                    &first_cut_preview,
                                    cancel,
                                )?;
                            TemporalNeighborResolution::Held(
                                volume,
                                "held_anchor_neighbor_read_error".to_string(),
                            )
                        }
                        MissingNeighborPolicy::DropFrame => {
                            notes.push(format!(
                                "{} time {} dropped: adjacent scene read failed ({error})",
                                display_name(&scene.path),
                                scene.time_index
                            ));
                            continue;
                        }
                        MissingNeighborPolicy::Error => {
                            return Err(format!(
                                "Read adjacent WRF scene for {}: {error}",
                                display_name(&scene.path)
                            ));
                        }
                    },
                };
                match neighbor {
                    TemporalNeighborResolution::Held(volume, outcome) => (volume, outcome),
                    TemporalNeighborResolution::Fields(neighbor) => {
                        if let Some(status) = neighbor.dual_pol_status.as_deref() {
                            notes.push(format!(
                                "{} time {} adjacent microphysics: {status}",
                                display_name(&neighbor_scene.path),
                                neighbor_scene.time_index
                            ));
                        }
                        if let Err(error) = validate_temporal_field_pair(&anchor, &neighbor) {
                            match config.missing_neighbor_policy {
                                MissingNeighborPolicy::HoldAnchor => {
                                    notes.push(format!(
                                        "{} time {} held at anchor: {error}",
                                        display_name(&scene.path),
                                        scene.time_index
                                    ));
                                    (
                                        try_build_synthetic_volume_reporting_previewing_cancellable(
                                            &anchor,
                                            valid_time,
                                            config,
                                            &progress,
                                            &first_cut_preview,
                                            cancel,
                                        )?,
                                        "held_anchor_property_mismatch".to_string(),
                                    )
                                }
                                MissingNeighborPolicy::DropFrame => {
                                    notes.push(format!(
                                        "{} time {} dropped: {error}",
                                        display_name(&scene.path),
                                        scene.time_index
                                    ));
                                    continue;
                                }
                                MissingNeighborPolicy::Error => return Err(error),
                            }
                        } else {
                            (
                                build_synthetic_volume_reporting_temporal_previewing_cancellable(
                                    &anchor,
                                    &neighbor,
                                    valid_time,
                                    config,
                                    &progress,
                                    &plan,
                                    &first_cut_preview,
                                    cancel,
                                )?,
                                config.atmosphere_time_mode.provenance_name().to_string(),
                            )
                        }
                    }
                }
            }
            TemporalSamplingOutcome::Frozen => (
                try_build_synthetic_volume_reporting_previewing_cancellable(
                    &anchor,
                    valid_time,
                    config,
                    &progress,
                    &first_cut_preview,
                    cancel,
                )?,
                "frozen".to_string(),
            ),
            TemporalSamplingOutcome::HeldAnchor(reason) => (
                try_build_synthetic_volume_reporting_previewing_cancellable(
                    &anchor,
                    valid_time,
                    config,
                    &progress,
                    &first_cut_preview,
                    cancel,
                )?,
                match reason {
                    HoldReason::NoLaterScene => "held_anchor_no_later_scene".to_string(),
                    HoldReason::ScanCrossesNeighbor => {
                        "held_anchor_scan_crosses_neighbor".to_string()
                    }
                },
            ),
        };
        append_scene_provenance(&mut volume, scene);
        append_temporal_provenance(&mut volume, &plan, &runtime_outcome, config, &anchor);
        notes.push(format!(
            "{} time {}: {} radials, atmosphere {runtime_outcome}",
            display_name(&scene.path),
            scene.time_index,
            volume.metadata.decoded_radial_count,
        ));
        let neighbor_scene = if runtime_outcome.starts_with("held_anchor") {
            None
        } else {
            plan.neighbor
                .as_ref()
                .and_then(|locator| {
                    group.scenes.iter().find(|candidate| {
                        candidate.source_identity == locator.source_identity
                            && candidate.time_index == locator.time_index
                    })
                })
                .cloned()
        };
        let volume = Arc::new(volume);
        frame_sources.push(SyntheticFrameSourceDescriptor::wrf(
            volumes.len(),
            &volume,
            scene.clone(),
            neighbor_scene,
            Some(plan.clone()),
            config,
            config_fingerprint,
        ));
        volumes.push(volume);
    }

    if volumes.is_empty() {
        return Err(if notes.is_empty() {
            "WRF temporal build produced no simulated radar volumes".to_string()
        } else {
            format!(
                "WRF temporal build produced no simulated radar volumes: {}",
                notes.join("; ")
            )
        });
    }
    Ok(SyntheticRadarOutput {
        label: label.to_string(),
        volumes,
        notes,
        config_fingerprint,
        frame_sources,
    })
}

fn temporal_build_budget_error(required_bytes: usize, budget_bytes: usize) -> String {
    format!(
        "Temporal build needs an estimated {:.2} GiB, above the configured {:.2} GiB budget. Change 'Temporal build RAM cap' under WRF Synthetic radar > Instrument & propagation, select fewer WRF frames, or turn off Ideal/Measured stage diagnostics; raise the cap only when that RAM is actually available",
        required_bytes as f64 / 1024.0_f64.powi(3),
        budget_bytes as f64 / 1024.0_f64.powi(3),
    )
}

// This short-lived control value favors clear ownership at the scan boundary;
// it is never stored in a collection or retained per ray/gate.
#[allow(clippy::large_enum_variant)]
enum TemporalNeighborResolution {
    Fields(Arc<WrfRadarFields>),
    Held(RadarVolume, String),
}

fn temporal_memory_estimate(
    group: &WrfSceneGroup,
    config: &SyntheticRadarConfig,
    emitted_frame_count: usize,
) -> Result<TemporalMemoryEstimate, String> {
    let grid = &group.key.grid_signature;
    let nz = grid.nz.ok_or_else(|| {
        "WRF grid has no vertical-size metadata for temporal preflight".to_string()
    })?;
    let horizontal_cells = grid
        .nx
        .checked_mul(grid.ny)
        .ok_or_else(|| "WRF horizontal grid size overflowed".to_string())?;
    let cells_per_scene = horizontal_cells
        .checked_mul(nz)
        .ok_or_else(|| "WRF 3-D grid size overflowed".to_string())?;
    let mut compact_bytes_per_scene = horizontal_cells
        .checked_mul(28)
        .ok_or_else(|| "WRF geolocation memory estimate overflowed".to_string())?;
    if config.polarimetric_kernel.is_property_tmatrix() {
        // RawStateLinear retains dense temperature, pressure, moist density,
        // and dry density. LinearAdjacent retains the compact scene's dense
        // full-cell -> sparse-row lookup. Sparse category/output bytes are
        // added from exact runtime memory estimates after each scene read.
        let property_dense_bytes_per_cell = if matches!(
            config.atmosphere_time_mode,
            AtmosphereTimeMode::RawStateLinear
        ) {
            4 * std::mem::size_of::<f32>()
        } else {
            std::mem::size_of::<u32>()
        };
        compact_bytes_per_scene = compact_bytes_per_scene
            .checked_add(
                cells_per_scene
                    .checked_mul(property_dense_bytes_per_cell)
                    .ok_or_else(|| "WRF property memory estimate overflowed".to_string())?,
            )
            .ok_or_else(|| "WRF property memory estimate overflowed".to_string())?;
    } else if config.dual_pol || config.terminal_fall_speed {
        compact_bytes_per_scene = compact_bytes_per_scene
            .checked_add(
                cells_per_scene
                    .checked_mul(9)
                    .ok_or_else(|| "WRF polar memory estimate overflowed".to_string())?,
            )
            .ok_or_else(|| "WRF polar memory estimate overflowed".to_string())?;
    }
    if config.spectrum_width {
        compact_bytes_per_scene = compact_bytes_per_scene
            .checked_add(cells_per_scene)
            .ok_or_else(|| "WRF TKE memory estimate overflowed".to_string())?;
    }

    let dx_m = grid.dx_millimeters.map(|value| value as f64 / 1_000.0);
    let spacing_m = effective_gate_spacing(config, dx_m).max(1.0);
    let gate_count = ((config.max_range_m / spacing_m).floor() as usize).max(1);
    let radial_count = config
        .physical_scan_legs()
        .len()
        .checked_mul(config.azimuth_count.max(1))
        .ok_or_else(|| "Synthetic radial count overflowed".to_string())?;
    let f32_moment_count = 2usize
        + usize::from(config.spectrum_width)
        + if config.dual_pol { 10 } else { 0 }
        + if config.emit_stage_diagnostics { 12 } else { 0 };
    let gate_samples = radial_count
        .checked_mul(gate_count)
        .ok_or_else(|| "Synthetic output sample count overflowed".to_string())?;
    let one_output_bytes = gate_samples
        .checked_mul(f32_moment_count)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|value| {
            value.checked_add(if config.emit_quality_fields {
                gate_samples.saturating_mul(QualityMoment::ALL.len())
            } else {
                0
            })
        })
        .and_then(|value| value.checked_add(radial_count.saturating_mul(96)))
        .ok_or_else(|| "Synthetic output memory estimate overflowed".to_string())?;
    let output_bytes = one_output_bytes
        .checked_mul(emitted_frame_count)
        .ok_or_else(|| "Retained loop-output memory estimate overflowed".to_string())?;
    let terrain_horizon_bytes = config
        .azimuth_count
        .max(1)
        .checked_mul(gate_count)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "Terrain-horizon memory estimate overflowed".to_string())?;
    // `CutMomentRow` owns thirteen canonical f32 moment arrays plus three
    // compact quality arrays until a cut is assembled, regardless of final VCP
    // availability. Opt-in diagnostics add twelve more f32 stage arrays.
    let cut_row_bytes_per_gate = (13 + if config.emit_stage_diagnostics { 12 } else { 0 })
        * std::mem::size_of::<f32>()
        + QualityMoment::ALL.len();
    let cut_row_scratch_bytes = config
        .azimuth_count
        .max(1)
        .checked_mul(gate_count)
        .and_then(|value| value.checked_mul(cut_row_bytes_per_gate))
        .ok_or_else(|| "Cut-row scratch memory estimate overflowed".to_string())?;
    // Concurrent WRF reads temporarily retain f64 decoder buffers while the
    // five f32 model planes are narrowed. Six planes is a conservative bound
    // for winds that arrive as two components plus height/REF/W.
    let read_conversion_scratch_bytes = cells_per_scene
        .checked_mul(6)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| "WRF read-scratch memory estimate overflowed".to_string())?;
    let shared_static_bytes = terrain_horizon_bytes
        .checked_add(cut_row_scratch_bytes)
        .and_then(|value| value.checked_add(read_conversion_scratch_bytes))
        .ok_or_else(|| "Temporal shared-memory estimate overflowed".to_string())?;

    Ok(TemporalMemoryEstimate {
        cells_per_scene,
        dense_fields_per_scene: 5,
        bytes_per_dense_value: std::mem::size_of::<f32>(),
        compact_bytes_per_scene,
        shared_static_bytes,
        output_bytes,
    })
}

fn temporal_required_peak_bytes(
    estimate: TemporalMemoryEstimate,
    requires_two_scenes: bool,
) -> Option<usize> {
    let scene_count = if requires_two_scenes { 2 } else { 1 };
    estimate
        .scene_bytes()?
        .checked_mul(scene_count)?
        .checked_add(estimate.shared_static_bytes)?
        .checked_add(estimate.output_bytes)
}

fn ensure_temporal_runtime_cache_budget(
    cache: &TwoSceneCache<(String, usize), Arc<WrfRadarFields>>,
    estimate: TemporalMemoryEstimate,
    budget_bytes: usize,
) -> Result<(), String> {
    let cached_scene_count = cache.len();
    let property_table_resident_bytes = cache
        .values()
        .map(|fields| fields.property_table_resident_bytes)
        .max()
        .unwrap_or(0);
    let required_bytes = temporal_runtime_reserved_bytes(cache, estimate)?
        .checked_add(property_table_resident_bytes)
        .ok_or_else(|| "Temporal runtime memory estimate overflowed".to_string())?;
    if required_bytes > budget_bytes {
        return Err(format!(
            "Temporal sparse T-matrix cache needs {:.2} GiB with {cached_scene_count} retained scene(s), above the configured {:.2} GiB budget",
            required_bytes as f64 / 1024.0_f64.powi(3),
            budget_bytes as f64 / 1024.0_f64.powi(3),
        ));
    }
    Ok(())
}

fn temporal_runtime_reserved_bytes(
    cache: &TwoSceneCache<(String, usize), Arc<WrfRadarFields>>,
    estimate: TemporalMemoryEstimate,
) -> Result<usize, String> {
    let shared_and_output = estimate
        .shared_static_bytes
        .checked_add(estimate.output_bytes)
        .ok_or_else(|| "Temporal runtime retained-memory estimate overflowed".to_string())?;
    cache
        .values()
        .try_fold(shared_and_output, |total, fields| {
            total.checked_add(fields.retained_bytes_estimate())
        })
        .ok_or_else(|| "Temporal runtime retained-memory estimate overflowed".to_string())
}

fn cached_temporal_fields(
    cache: &mut TwoSceneCache<(String, usize), Arc<WrfRadarFields>>,
    scene: &WrfScene,
    config: &SyntheticRadarConfig,
    progress: &dyn Fn(&str),
    estimate: TemporalMemoryEstimate,
    budget_bytes: usize,
    cancel: &AtomicBool,
) -> Result<Arc<WrfRadarFields>, String> {
    check_synthetic_radar_cancel(cancel)?;
    let key = (scene.source_identity.0.clone(), scene.time_index);
    if let Some(fields) = cache.get(&key) {
        return Ok(Arc::clone(fields));
    }
    progress(&format!(
        "reading WRF atmosphere at {}…",
        scene
            .time
            .valid_time()
            .map(DateTime::<Utc>::to_rfc3339)
            .unwrap_or_else(|| "unknown time".to_string())
    ));
    let file = WrfFile::open(&scene.path).map_err(|error| {
        format!(
            "Open inventoried WRF {}: {error}",
            display_name(&scene.path)
        )
    })?;
    let reserved_memory_bytes = temporal_runtime_reserved_bytes(cache, estimate)?;
    let fields = Arc::new(read_wrf_radar_fields_for_config_reporting(
        &file,
        &scene.source_identity,
        scene.time_index,
        config,
        progress,
        reserved_memory_bytes,
        Some(cancel),
    )?);
    check_synthetic_radar_cancel(cancel)?;
    let evicted = cache.insert(key.clone(), Arc::clone(&fields));
    if let Err(error) = ensure_temporal_runtime_cache_budget(cache, estimate, budget_bytes) {
        let _ = cache.remove(&key);
        if let Some((evicted_key, evicted_fields)) = evicted {
            cache.insert(evicted_key, evicted_fields);
        }
        return Err(error);
    }
    Ok(fields)
}

fn validate_temporal_field_pair(
    anchor: &WrfRadarFields,
    neighbor: &WrfRadarFields,
) -> Result<(), String> {
    if (anchor.nx, anchor.ny, anchor.nz) != (neighbor.nx, neighbor.ny, neighbor.nz) {
        return Err(format!(
            "Adjacent WRF field shape changed from {}x{}x{} to {}x{}x{}",
            anchor.nx, anchor.ny, anchor.nz, neighbor.nx, neighbor.ny, neighbor.nz
        ));
    }
    if anchor.ref_source != neighbor.ref_source {
        return Err(format!(
            "Adjacent WRF reflectivity source changed from {} to {}",
            anchor.ref_source, neighbor.ref_source
        ));
    }
    match (&anchor.polarimetric, &neighbor.polarimetric) {
        (None, None) => {}
        (Some(left), Some(right))
            if left.profile == right.profile && left.present_fields == right.present_fields => {}
        (Some(left), Some(right)) => {
            return Err(format!(
                "Adjacent WRF microphysics inventory changed from {} [{}] to {} [{}]",
                left.profile.name,
                left.present_fields.join(","),
                right.profile.name,
                right.present_fields.join(",")
            ));
        }
        _ => {
            return Err(
                "Adjacent WRF scene has incompatible polarimetric field availability".to_string(),
            );
        }
    }
    match (&anchor.property_scattering, &neighbor.property_scattering) {
        (None, None) => {}
        (Some(left), Some(right))
            if left.required_field_signature() == right.required_field_signature()
                && left.microphysics_scheme_id() == right.microphysics_scheme_id()
                && property_scattering_contract_matches(left, right) => {}
        (Some(left), Some(right)) => {
            return Err(format!(
                "Adjacent WRF property-scattering contracts differ (mp_physics {} vs {})",
                left.microphysics_scheme_id(),
                right.microphysics_scheme_id(),
            ));
        }
        _ => {
            return Err(
                "Adjacent WRF scene has incompatible P3/ISHMAEL property-scattering availability"
                    .to_string(),
            );
        }
    }
    if anchor.property_table_identity != neighbor.property_table_identity {
        return Err(
            "Adjacent WRF scenes resolved different property T-matrix table-pack identities"
                .to_owned(),
        );
    }
    match (&anchor.raw_property_scene, &neighbor.raw_property_scene) {
        (None, None) => {}
        (Some(left), Some(right))
            if left.microphysics_scheme_id() == right.microphysics_scheme_id()
                && left.required_field_signature() == right.required_field_signature() => {}
        (Some(left), Some(right)) => {
            return Err(format!(
                "Adjacent RawStateLinear property inventories differ (mp_physics {} vs {})",
                left.microphysics_scheme_id(),
                right.microphysics_scheme_id(),
            ));
        }
        _ => {
            return Err(
                "Adjacent WRF scene has incompatible RawStateLinear property coverage".to_string(),
            );
        }
    }
    if anchor.refractivity_model.is_some() != neighbor.refractivity_model.is_some() {
        return Err(
            "Adjacent WRF scene has incompatible refractivity-profile availability".to_owned(),
        );
    }
    if anchor.tke_tenths_m2s2.is_some() != neighbor.tke_tenths_m2s2.is_some() {
        return Err("Adjacent WRF scene has incompatible TKE availability".to_string());
    }
    Ok(())
}

fn property_scattering_contract_matches(
    left: &app_ui::wrf_tmatrix_scene::WrfTMatrixScene,
    right: &app_ui::wrf_tmatrix_scene::WrfTMatrixScene,
) -> bool {
    let left_provenance = left.provenance();
    let right_provenance = right.provenance();
    left_provenance.status == right_provenance.status
        && left_provenance.frequency_hz == right_provenance.frequency_hz
        && left_provenance.orientation == right_provenance.orientation
        && left_provenance.frozen_scattering == right_provenance.frozen_scattering
        && left_provenance.fall_moment_policy == right_provenance.fall_moment_policy
        && left_provenance.rain_mode == right_provenance.rain_mode
        && left_provenance.tables == right_provenance.tables
        && left.radar_elevations_deg() == right.radar_elevations_deg()
}

fn append_temporal_provenance(
    volume: &mut RadarVolume,
    plan: &TemporalScenePlan,
    runtime_outcome: &str,
    config: &SyntheticRadarConfig,
    anchor: &WrfRadarFields,
) {
    let neighbor = plan
        .neighbor
        .as_ref()
        .map(|scene| format!("{}:time{}", scene.source_identity.0, scene.time_index))
        .unwrap_or_else(|| "none".to_string());
    let neighbor_time = plan
        .neighbor_time
        .as_ref()
        .map(DateTime::<Utc>::to_rfc3339)
        .unwrap_or_else(|| "none".to_string());
    let policy = match config.missing_neighbor_policy {
        MissingNeighborPolicy::HoldAnchor => "hold_anchor",
        MissingNeighborPolicy::DropFrame => "drop_frame",
        MissingNeighborPolicy::Error => "error",
    };
    let physics_status = anchor
        .dual_pol_status
        .as_deref()
        .unwrap_or("not_requested")
        .replace(';', ",");
    let interpolation_space = match config.atmosphere_time_mode {
        AtmosphereTimeMode::RawStateLinear => {
            "raw_thermodynamics_winds_and_scheme_microphysics_spatial_plus_temporal_preclosure"
        }
        AtmosphereTimeMode::LinearAdjacent if anchor.property_scattering.is_some() => {
            "source_cell_closure_then_additive_scattering_and_wind"
        }
        _ => "derived_linear_z_wind_and_additive_polar",
    };
    let provenance = format!(
        "atmosphere_time_mode={}; temporal_interpolation_space={interpolation_space}; temporal_policy={policy}; temporal_outcome={runtime_outcome}; temporal_neighbor={neighbor}; temporal_neighbor_time={neighbor_time}; temporal_scan_duration_ms={}; temporal_reflectivity_source={}; temporal_polar_available={}; temporal_microphysics_status={physics_status}",
        config.atmosphere_time_mode.provenance_name(),
        plan.scan_duration_ms,
        anchor.ref_source,
        anchor.has_polarimetric_input(),
    );
    match volume.metadata.forward_operator_config.as_mut() {
        Some(config) => {
            config.push_str("; ");
            config.push_str(&provenance);
        }
        None => volume.metadata.forward_operator_config = Some(provenance),
    }
}

fn scene_time_authority(time: &WrfSceneTime) -> &'static str {
    match time {
        WrfSceneTime::InternalTimes { .. } => "internal Times",
        WrfSceneTime::FilenameFallback { .. } => "filename fallback",
        WrfSceneTime::Unavailable { .. } => "unavailable",
    }
}

fn append_scene_provenance(volume: &mut RadarVolume, scene: &WrfScene) {
    let valid_time = scene
        .time
        .valid_time()
        .expect("scene adapter rejects untimed scenes");
    let provenance = format!(
        "scene_source={}; scene_time_index={}; scene_time_authority={}; scene_valid_time={}",
        scene.source_identity.0,
        scene.time_index,
        scene_time_authority(&scene.time),
        valid_time.to_rfc3339(),
    );
    match volume.metadata.forward_operator_config.as_mut() {
        Some(config) => {
            config.push_str("; ");
            config.push_str(&provenance);
        }
        None => volume.metadata.forward_operator_config = Some(provenance),
    }
}

fn scene_build_fingerprint(config: &SyntheticRadarConfig, group: &WrfSceneGroup) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.data_fingerprint().hash(&mut hasher);
    group.key.hash(&mut hasher);
    for scene in &group.scenes {
        scene.source_identity.hash(&mut hasher);
        scene.time_index.hash(&mut hasher);
        scene
            .time
            .valid_time()
            .map(DateTime::<Utc>::timestamp_millis)
            .hash(&mut hasher);
        scene_time_authority(&scene.time).hash(&mut hasher);
    }
    hasher.finish()
}

fn display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| path.display().to_string())
}


pub fn is_supported_wrf_file(path: &Path) -> bool {
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
}
