//! Dependency-free S-band bulk polarimetric radar primitives.
//!
//! This module is deliberately an honest *bulk/Rayleigh approximation*, not a
//! T-matrix solver.  Its structure follows the forward-operator lineage of
//! Jung et al. (2008, 2010): close a particle-size distribution (PSD), add the
//! species in linear scattering space, and derive polarimetric observables
//! only after accumulation.  Rain oblateness uses the Brandes et al. drop
//! axis-ratio polynomial.  The approximation does not resolve particle
//! orientation distributions, detailed melting-layer physics, resonance, or
//! scheme-native particle-property categories. P3 and ISHMAEL therefore stay
//! unsupported by this *bulk* kernel; BowEcho's separate opt-in property-aware
//! T-matrix path owns those inputs and fails closed instead of entering here.

use std::collections::HashSet;
use std::f64::consts::PI;
use std::sync::OnceLock;

const WATER_DENSITY_KG_M3: f64 = 1_000.0;
const ICE_DENSITY_KG_M3: f64 = 917.0;
const MAX_LINEAR_Z: f64 = 1.0e15;

/// Bulk hydrometeor categories supported by the approximation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HydrometeorKind {
    CloudWater,
    Rain,
    CloudIce,
    Snow,
    Graupel,
    Hail,
}

/// How completely an output file constrains the PSDs needed by this operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MicrophysicsCapability {
    /// Mass and number are available for all represented precipitating species.
    FullTwoMoment,
    /// At least one, but not every, represented species carries number.
    PartialTwoMoment,
    /// Only mass is usable, so fixed-intercept PSD assumptions are required.
    MassOnly,
    /// Categories cannot be represented honestly by this bulk-species model.
    Unsupported,
}

/// Detected WRF microphysics and the capability of fields actually present.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchemeProfile {
    pub id: Option<i32>,
    pub name: String,
    pub capability: MicrophysicsCapability,
    /// True when one or more species needs an assumed intercept/shape closure.
    pub assumption_heavy: bool,
}

/// Detect a WRF microphysics scheme and reconcile it with available fields.
///
/// Field names are case-insensitive and may carry a `wrf_` prefix.  An empty
/// field list reports the scheme's nominal capability.  A non-empty list can
/// only downgrade that capability when number-concentration output is absent.
pub fn detect_scheme<I, S>(id: Option<i32>, present_fields: I) -> SchemeProfile
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let fields: HashSet<String> = present_fields
        .into_iter()
        .map(|field| normalize_field_name(field.as_ref()))
        .collect();

    let (name, nominal) = match id {
        Some(2) => ("Purdue Lin", MicrophysicsCapability::MassOnly),
        Some(6) => ("WSM6", MicrophysicsCapability::MassOnly),
        Some(8) => ("Thompson", MicrophysicsCapability::PartialTwoMoment),
        Some(9) => ("Milbrandt-Yau", MicrophysicsCapability::FullTwoMoment),
        Some(10) => ("Morrison", MicrophysicsCapability::FullTwoMoment),
        Some(14) => ("WDM5", MicrophysicsCapability::PartialTwoMoment),
        Some(16) => ("WDM6", MicrophysicsCapability::PartialTwoMoment),
        Some(18) => ("NSSL two-moment", MicrophysicsCapability::FullTwoMoment),
        Some(24) => ("WSM7", MicrophysicsCapability::MassOnly),
        Some(26) => ("WDM7", MicrophysicsCapability::PartialTwoMoment),
        Some(28) => (
            "Thompson aerosol-aware",
            MicrophysicsCapability::PartialTwoMoment,
        ),
        Some(29) => (
            "RCON Thompson aerosol-aware",
            MicrophysicsCapability::PartialTwoMoment,
        ),
        Some(38) => (
            "Thompson hail/graupel/aerosol",
            MicrophysicsCapability::PartialTwoMoment,
        ),
        Some(40) => (
            "Morrison with CESM aerosol",
            MicrophysicsCapability::FullTwoMoment,
        ),
        Some(50..=53) => ("P3", MicrophysicsCapability::Unsupported),
        Some(55) => ("Jensen ISHMAEL", MicrophysicsCapability::Unsupported),
        Some(other) => {
            let capability = infer_capability_from_fields(&fields);
            return SchemeProfile {
                id,
                name: format!("Unknown WRF microphysics ({other})"),
                capability,
                assumption_heavy: capability != MicrophysicsCapability::FullTwoMoment,
            };
        }
        None => {
            let capability = infer_capability_from_fields(&fields);
            return SchemeProfile {
                id,
                name: "Detected from WRF output fields".to_owned(),
                capability,
                assumption_heavy: capability != MicrophysicsCapability::FullTwoMoment,
            };
        }
    };

    let capability = reconcile_nominal_capability(nominal, &fields);
    SchemeProfile {
        id,
        name: name.to_owned(),
        capability,
        assumption_heavy: capability != MicrophysicsCapability::FullTwoMoment,
    }
}

fn normalize_field_name(field: &str) -> String {
    let upper = field.trim().to_ascii_uppercase();
    upper.strip_prefix("WRF_").unwrap_or(&upper).to_owned()
}

fn has_any(fields: &HashSet<String>, aliases: &[&str]) -> bool {
    aliases.iter().any(|alias| fields.contains(*alias))
}

fn mass_number_evidence(fields: &HashSet<String>) -> (usize, usize) {
    let pairs: [(&[&str], &[&str]); 6] = [
        (&["QCLOUD", "QC"], &["QNCLOUD", "QNC", "QNDROP", "QNCDROP"]),
        (&["QRAIN", "QR"], &["QNRAIN", "QNR"]),
        (&["QICE", "QI"], &["QNICE", "QNI"]),
        (&["QSNOW", "QS"], &["QNSNOW", "QNS"]),
        (
            &["QGRAUP", "QGRAUPEL", "QG"],
            &["QNGRAUP", "QNGRAUPEL", "QNG"],
        ),
        (&["QHAIL", "QH"], &["QNHAIL", "QNH"]),
    ];

    pairs.iter().fold((0, 0), |(mass, numbered), pair| {
        if has_any(fields, pair.0) {
            (mass + 1, numbered + usize::from(has_any(fields, pair.1)))
        } else {
            (mass, numbered)
        }
    })
}

fn infer_capability_from_fields(fields: &HashSet<String>) -> MicrophysicsCapability {
    let (mass, numbered) = mass_number_evidence(fields);
    if mass == 0 {
        MicrophysicsCapability::Unsupported
    } else if numbered == mass && mass >= 2 {
        MicrophysicsCapability::FullTwoMoment
    } else if numbered > 0 {
        MicrophysicsCapability::PartialTwoMoment
    } else {
        MicrophysicsCapability::MassOnly
    }
}

fn reconcile_nominal_capability(
    nominal: MicrophysicsCapability,
    fields: &HashSet<String>,
) -> MicrophysicsCapability {
    if fields.is_empty() || nominal == MicrophysicsCapability::Unsupported {
        return nominal;
    }

    let observed = infer_capability_from_fields(fields);
    match nominal {
        MicrophysicsCapability::FullTwoMoment => observed,
        MicrophysicsCapability::PartialTwoMoment => match observed {
            MicrophysicsCapability::FullTwoMoment | MicrophysicsCapability::PartialTwoMoment => {
                MicrophysicsCapability::PartialTwoMoment
            }
            other => other,
        },
        MicrophysicsCapability::MassOnly => MicrophysicsCapability::MassOnly,
        MicrophysicsCapability::Unsupported => MicrophysicsCapability::Unsupported,
    }
}

/// One bulk-species state at a model sample point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BulkSpeciesInput {
    pub kind: HydrometeorKind,
    /// Hydrometeor mixing ratio per kg of dry air.
    pub q_kgkg: f32,
    /// Optional total number concentration per kg of dry air.
    pub number_per_kg: Option<f32>,
    /// Optional bulk particle volume per kg of dry air (variable-density ice).
    pub volume_m3_per_kg: Option<f32>,
    pub temperature_k: f32,
    pub air_density_kgm3: f32,
}

/// A species contribution in additive, linear scattering space.
///
/// `fall_speed_mps` and `fall_speed_variance_m2s2` are the species' intrinsic
/// reflectivity-weighted moments.  [`PolarAccumulator`] converts them to
/// additive first/second moments using this contribution's `zh`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BulkContribution {
    /// Horizontal equivalent reflectivity factor, mm^6 m^-3.
    pub zh: f32,
    /// Vertical equivalent reflectivity factor, mm^6 m^-3.
    pub zv: f32,
    /// Real cross-covariance in linear reflectivity units.
    pub cov_re: f32,
    /// Imaginary cross-covariance in linear reflectivity units.
    pub cov_im: f32,
    pub kdp_deg_km: f32,
    pub ah_db_km: f32,
    pub av_db_km: f32,
    /// Positive-downward reflectivity-weighted terminal speed.
    pub fall_speed_mps: f32,
    pub fall_speed_variance_m2s2: f32,
}

#[derive(Clone, Copy, Debug)]
struct SpeciesParameters {
    density_kg_m3: f64,
    mu: f64,
    fixed_n0: f64,
    default_number_m3: Option<f64>,
    fall_a: f64,
    fall_b: f64,
    fall_max_mps: f64,
}

fn species_parameters(kind: HydrometeorKind) -> SpeciesParameters {
    match kind {
        HydrometeorKind::CloudWater => SpeciesParameters {
            density_kg_m3: WATER_DENSITY_KG_M3,
            mu: 2.0,
            fixed_n0: 1.0e16,
            default_number_m3: Some(1.0e8),
            fall_a: 0.01,
            fall_b: 2.0,
            fall_max_mps: 0.1,
        },
        HydrometeorKind::Rain => SpeciesParameters {
            density_kg_m3: WATER_DENSITY_KG_M3,
            mu: 0.0,
            // Marshall-Palmer 8,000 m^-3 mm^-1 in SI diameter units.
            fixed_n0: 8.0e6,
            default_number_m3: None,
            fall_a: 3.78,
            fall_b: 0.67,
            fall_max_mps: 10.5,
        },
        HydrometeorKind::CloudIce => SpeciesParameters {
            density_kg_m3: ICE_DENSITY_KG_M3,
            mu: 1.0,
            fixed_n0: 1.0e13,
            default_number_m3: None,
            fall_a: 0.6,
            fall_b: 0.25,
            fall_max_mps: 2.5,
        },
        HydrometeorKind::Snow => SpeciesParameters {
            density_kg_m3: 100.0,
            mu: 0.0,
            fixed_n0: 2.0e7,
            default_number_m3: None,
            fall_a: 0.8,
            fall_b: 0.2,
            fall_max_mps: 3.0,
        },
        HydrometeorKind::Graupel => SpeciesParameters {
            density_kg_m3: 400.0,
            mu: 0.0,
            fixed_n0: 4.0e6,
            default_number_m3: None,
            fall_a: 1.4,
            fall_b: 0.45,
            fall_max_mps: 9.0,
        },
        HydrometeorKind::Hail => SpeciesParameters {
            density_kg_m3: 900.0,
            mu: 0.0,
            fixed_n0: 4.0e4,
            default_number_m3: None,
            fall_a: 3.0,
            fall_b: 0.5,
            fall_max_mps: 25.0,
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct PsdClosure {
    mass_kg_m3: f64,
    air_density_kg_m3: f64,
    density_kg_m3: f64,
    mu: f64,
    lambda_m_inv: f64,
    n0: f64,
}

impl PsdClosure {
    /// Integral of D^order N(D) dD for D in metres.
    fn moment(self, order: f64) -> f64 {
        self.n0 * gamma(self.mu + order + 1.0) / self.lambda_m_inv.powf(self.mu + order + 1.0)
    }

    fn characteristic_diameter_mm(self) -> f64 {
        (1_000.0 * (self.mu + 4.0) / self.lambda_m_inv).clamp(0.001, 8.0)
    }
}

fn close_psd(input: BulkSpeciesInput, parameters: SpeciesParameters) -> Option<PsdClosure> {
    let q = f64::from(input.q_kgkg);
    let air_density = f64::from(input.air_density_kgm3);
    if !q.is_finite() || !air_density.is_finite() || q <= 0.0 || air_density <= 0.0 {
        return None;
    }

    let mass = q * air_density;
    if !mass.is_finite() || mass <= 0.0 {
        return None;
    }

    let density = input
        .volume_m3_per_kg
        .map(f64::from)
        .filter(|volume| volume.is_finite() && *volume > 0.0)
        .map_or(parameters.density_kg_m3, |volume| q / volume)
        .clamp(50.0, WATER_DENSITY_KG_M3);
    let mass_coefficient = PI * density / 6.0;

    let supplied_number = input
        .number_per_kg
        .map(f64::from)
        .filter(|number| number.is_finite() && *number > 0.0)
        .map(|number| number * air_density);
    let target_number = supplied_number.or(parameters.default_number_m3);

    let gamma_number = gamma(parameters.mu + 1.0);
    let gamma_mass = gamma(parameters.mu + 4.0);
    let (lambda, n0) = if let Some(number) = target_number {
        let lambda =
            (mass_coefficient * number * gamma_mass / (mass * gamma_number)).powf(1.0 / 3.0);
        let n0 = number * lambda.powf(parameters.mu + 1.0) / gamma_number;
        (lambda, n0)
    } else {
        let n0 = parameters.fixed_n0;
        let lambda = (mass_coefficient * n0 * gamma_mass / mass).powf(1.0 / (parameters.mu + 4.0));
        (lambda, n0)
    };

    if !lambda.is_finite() || !n0.is_finite() || lambda <= 0.0 || n0 <= 0.0 {
        return None;
    }

    Some(PsdClosure {
        mass_kg_m3: mass,
        air_density_kg_m3: air_density,
        density_kg_m3: density,
        mu: parameters.mu,
        lambda_m_inv: lambda,
        n0,
    })
}

/// Stable seam for replacing the analytic bulk kernel with a versioned offline
/// T-matrix LUT. Implementations must return additive linear moments; callers
/// never interpolate ZDR or rhoHV directly.
pub trait PolarimetricScatteringKernel {
    fn identifier(&self) -> &'static str;
    fn contribution(&self, input: BulkSpeciesInput, scheme: &SchemeProfile) -> BulkContribution;
}

/// Current dependency-free S-band Rayleigh/bulk kernel.
pub struct BulkRayleighSbandV1;

impl PolarimetricScatteringKernel for BulkRayleighSbandV1 {
    fn identifier(&self) -> &'static str {
        "bowecho-bulk-rayleigh-sband-v1"
    }

    fn contribution(&self, input: BulkSpeciesInput, scheme: &SchemeProfile) -> BulkContribution {
        bulk_sband_contribution_impl(input, scheme.capability, scheme.assumption_heavy)
    }
}

pub fn bulk_sband_model_id() -> &'static str {
    BulkRayleighSbandV1.identifier()
}

/// Versioned policy identifier for the explicit P3/ISHMAEL hybrid seam.
///
/// The default bulk dispatch continues to reject property schemes. Callers may
/// use this policy only after a native property/T-matrix cell has failed for a
/// narrowly classified table-domain/shape omission or typed native ISHMAEL
/// source-state mass-closure gap, and must retain audit counts.
pub const PROPERTY_HYBRID_BULK_RAYLEIGH_REVISION: &str =
    "bowecho-p3-ishmael-hybrid-bulk-rayleigh-v1";

/// Compute one species' intrinsic S-band bulk scattering contribution through
/// the default kernel. Frozen-particle dielectric and wetness factors are
/// smooth approximations; property-based unsupported schemes return zero
/// rather than being relabelled into false bulk categories.
pub fn bulk_sband_contribution(
    input: BulkSpeciesInput,
    scheme: &SchemeProfile,
) -> BulkContribution {
    BulkRayleighSbandV1.contribution(input, scheme)
}

/// Evaluate one explicitly audited P3/ISHMAEL frozen population through the
/// existing bulk Rayleigh v1 operator.
///
/// This does not alter the default unsupported-scheme behavior above. The
/// caller supplies native mass, number, and (when it is a true total particle
/// volume) volume moments and records the hybrid policy in scene provenance.
pub fn property_hybrid_bulk_rayleigh_contribution(input: BulkSpeciesInput) -> BulkContribution {
    bulk_sband_contribution_impl(input, MicrophysicsCapability::FullTwoMoment, true)
}

fn bulk_sband_contribution_impl(
    input: BulkSpeciesInput,
    capability: MicrophysicsCapability,
    assumption_heavy: bool,
) -> BulkContribution {
    if capability == MicrophysicsCapability::Unsupported {
        return BulkContribution::default();
    }

    let parameters = species_parameters(input.kind);
    let Some(psd) = close_psd(input, parameters) else {
        return BulkContribution::default();
    };

    let diameter_mm = psd.characteristic_diameter_mm();
    let wet_fraction = particle_wet_fraction(input.kind, f64::from(input.temperature_k));
    let dielectric = dielectric_factor(input.kind, psd.density_kg_m3, wet_fraction);
    let spherical_z = 1.0e18 * psd.moment(6.0);
    let zh = (spherical_z * dielectric).clamp(0.0, MAX_LINEAR_Z);
    let zdr_db = intrinsic_zdr_db(input.kind, diameter_mm, wet_fraction);
    let zv = (zh / 10.0_f64.powf(zdr_db / 10.0)).clamp(0.0, MAX_LINEAR_Z);

    let rho_penalty = if assumption_heavy { 0.01 } else { 0.0 };
    let rho = (intrinsic_rho_hv(input.kind, wet_fraction) - rho_penalty).clamp(0.0, 1.0);
    let covariance = rho * (zh * zv).sqrt();
    let phase = covariance_phase_rad(input.kind, wet_fraction);

    let water_content_g_m3 = psd.mass_kg_m3 * 1_000.0;
    let (kdp, ah, av) = propagation_terms(
        input.kind,
        water_content_g_m3,
        diameter_mm,
        wet_fraction,
        zdr_db,
    );
    let (fall_speed, fall_variance) = fall_speed_moments(psd, parameters, assumption_heavy);

    BulkContribution {
        zh: finite_nonnegative(zh, MAX_LINEAR_Z),
        zv: finite_nonnegative(zv, MAX_LINEAR_Z),
        cov_re: finite_signed(covariance * phase.cos(), MAX_LINEAR_Z),
        cov_im: finite_signed(covariance * phase.sin(), MAX_LINEAR_Z),
        kdp_deg_km: finite_nonnegative(kdp, 1_000.0),
        ah_db_km: finite_nonnegative(ah, 100.0),
        av_db_km: finite_nonnegative(av, 100.0),
        fall_speed_mps: finite_nonnegative(fall_speed, 50.0),
        fall_speed_variance_m2s2: finite_nonnegative(fall_variance, 2_500.0),
    }
}

fn particle_wet_fraction(kind: HydrometeorKind, temperature_k: f64) -> f64 {
    match kind {
        HydrometeorKind::CloudWater | HydrometeorKind::Rain => 1.0,
        HydrometeorKind::CloudIce => 0.0,
        HydrometeorKind::Snow | HydrometeorKind::Graupel | HydrometeorKind::Hail => {
            smoothstep(268.15, 274.15, temperature_k)
        }
    }
}

fn smoothstep(edge0: f64, edge1: f64, value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    let x = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn dielectric_factor(kind: HydrometeorKind, density: f64, wet_fraction: f64) -> f64 {
    match kind {
        HydrometeorKind::CloudWater | HydrometeorKind::Rain => 1.0,
        HydrometeorKind::CloudIce
        | HydrometeorKind::Snow
        | HydrometeorKind::Graupel
        | HydrometeorKind::Hail => {
            // Rayleigh |K_ice|^2 / |K_water|^2 at S band, with an effective-
            // medium density correction.  Mixing amplitudes (not powers)
            // avoids an artificial discontinuity at the melting level.
            let dry = 0.189 * (density / ICE_DENSITY_KG_M3).powi(2);
            ((1.0 - wet_fraction) * dry.sqrt() + wet_fraction).powi(2)
        }
    }
}

/// Brandes et al. equilibrium rain-drop minor/major axis-ratio polynomial.
fn rain_axis_ratio(diameter_mm: f64) -> f64 {
    if diameter_mm < 0.5 {
        return 1.0;
    }
    let d = diameter_mm.clamp(0.5, 8.0);
    (0.9951 + 0.025_10 * d - 0.036_44 * d.powi(2) + 0.005_303 * d.powi(3) - 0.000_249_2 * d.powi(4))
        .clamp(0.56, 1.0)
}

fn intrinsic_zdr_db(kind: HydrometeorKind, diameter_mm: f64, wet_fraction: f64) -> f64 {
    match kind {
        HydrometeorKind::CloudWater => 0.0,
        HydrometeorKind::Rain => (15.0 * (1.0 - rain_axis_ratio(diameter_mm))).clamp(0.0, 5.0),
        HydrometeorKind::CloudIce => 0.6,
        HydrometeorKind::Snow => 0.35 * (1.0 - wet_fraction) + 0.1 * wet_fraction,
        HydrometeorKind::Graupel => 0.15 * (1.0 - wet_fraction) + 0.05 * wet_fraction,
        HydrometeorKind::Hail => 0.05 * (1.0 - wet_fraction),
    }
}

fn intrinsic_rho_hv(kind: HydrometeorKind, wet_fraction: f64) -> f64 {
    let mixed_phase = 4.0 * wet_fraction * (1.0 - wet_fraction);
    match kind {
        HydrometeorKind::CloudWater => 0.9995,
        HydrometeorKind::Rain => 0.998,
        HydrometeorKind::CloudIce => 0.985,
        HydrometeorKind::Snow => 0.98 - 0.11 * mixed_phase,
        HydrometeorKind::Graupel => 0.985 - 0.09 * mixed_phase,
        HydrometeorKind::Hail => 0.99 - 0.08 * mixed_phase,
    }
}

fn covariance_phase_rad(kind: HydrometeorKind, wet_fraction: f64) -> f64 {
    let degrees: f64 = match kind {
        HydrometeorKind::CloudWater | HydrometeorKind::Rain => 0.0,
        HydrometeorKind::CloudIce => -0.5,
        HydrometeorKind::Snow => 2.0 * wet_fraction,
        HydrometeorKind::Graupel => 4.0 * wet_fraction,
        HydrometeorKind::Hail => 6.0 * wet_fraction,
    };
    degrees.to_radians()
}

fn propagation_terms(
    kind: HydrometeorKind,
    water_content_g_m3: f64,
    diameter_mm: f64,
    wet_fraction: f64,
    zdr_db: f64,
) -> (f64, f64, f64) {
    let rain_anisotropy = 1.0 - rain_axis_ratio(diameter_mm);
    let (kdp, ah) = match kind {
        HydrometeorKind::CloudWater => (0.0, 0.003 * water_content_g_m3),
        HydrometeorKind::Rain => (
            8.0 * water_content_g_m3 * rain_anisotropy,
            0.015 * water_content_g_m3 * (1.0 + 2.0 * rain_anisotropy),
        ),
        HydrometeorKind::CloudIce => (0.0, 0.0),
        HydrometeorKind::Snow => (
            0.25 * water_content_g_m3 * wet_fraction,
            0.008 * water_content_g_m3 * wet_fraction,
        ),
        HydrometeorKind::Graupel => (
            0.2 * water_content_g_m3 * wet_fraction,
            0.01 * water_content_g_m3 * wet_fraction,
        ),
        HydrometeorKind::Hail => (
            0.1 * water_content_g_m3 * wet_fraction,
            0.012 * water_content_g_m3 * wet_fraction,
        ),
    };
    let av = ah * (1.0 - 0.08 * zdr_db).clamp(0.5, 1.0);
    (kdp, ah, av)
}

fn fall_speed_moments(
    psd: PsdClosure,
    parameters: SpeciesParameters,
    assumption_heavy: bool,
) -> (f64, f64) {
    let density_correction = (1.225 / psd.air_density_kg_m3).powf(0.4).clamp(0.7, 2.0);
    let scale = parameters.fall_a * 1_000.0_f64.powf(parameters.fall_b) * density_correction;
    let (first_ratio, second_ratio) = fall_gamma_ratios(parameters.mu, parameters.fall_b);
    let raw_mean = scale * first_ratio / psd.lambda_m_inv.powf(parameters.fall_b);
    let raw_second = scale.powi(2) * second_ratio / psd.lambda_m_inv.powf(2.0 * parameters.fall_b);
    let raw_variance = (raw_second - raw_mean.powi(2)).max(0.0);

    let cap_scale = if raw_mean > parameters.fall_max_mps {
        parameters.fall_max_mps / raw_mean
    } else {
        1.0
    };
    let mean = raw_mean * cap_scale;
    let mut variance = raw_variance * cap_scale.powi(2);
    if assumption_heavy {
        variance += (0.15 * mean).powi(2);
    }
    (mean, variance)
}

/// Gamma-moment ratios are constant for each species profile. Cache the six
/// pairs once instead of evaluating Lanczos gamma twice for every wet cell in
/// a large nest.
fn fall_gamma_ratios(mu: f64, exponent: f64) -> (f64, f64) {
    type RatioEntry = ((u64, u64), (f64, f64));
    static RATIOS: OnceLock<Vec<RatioEntry>> = OnceLock::new();
    let key = (mu.to_bits(), exponent.to_bits());
    RATIOS
        .get_or_init(|| {
            [
                HydrometeorKind::CloudWater,
                HydrometeorKind::Rain,
                HydrometeorKind::CloudIce,
                HydrometeorKind::Snow,
                HydrometeorKind::Graupel,
                HydrometeorKind::Hail,
            ]
            .into_iter()
            .map(|kind| {
                let parameters = species_parameters(kind);
                let base = gamma(parameters.mu + 7.0);
                (
                    (parameters.mu.to_bits(), parameters.fall_b.to_bits()),
                    (
                        gamma(parameters.mu + 7.0 + parameters.fall_b) / base,
                        gamma(parameters.mu + 7.0 + 2.0 * parameters.fall_b) / base,
                    ),
                )
            })
            .collect()
        })
        .iter()
        .find_map(|(candidate, ratios)| (*candidate == key).then_some(*ratios))
        .unwrap_or_else(|| {
            let base = gamma(mu + 7.0);
            (
                gamma(mu + 7.0 + exponent) / base,
                gamma(mu + 7.0 + 2.0 * exponent) / base,
            )
        })
}

/// Accumulates species/beam samples before logarithmic radar products exist.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PolarAccumulator {
    zh: f64,
    zv: f64,
    cov_re: f64,
    cov_im: f64,
    kdp: f64,
    ah: f64,
    av: f64,
    fall_weight: f64,
    fall_first: f64,
    fall_second: f64,
}

impl PolarAccumulator {
    /// Add a species or quadrature sample with a nonnegative linear weight.
    pub fn add(&mut self, weight: f32, contribution: BulkContribution) {
        let weight = f64::from(weight);
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }

        self.zh += weight * f64::from(contribution.zh.max(0.0));
        self.zv += weight * f64::from(contribution.zv.max(0.0));
        self.cov_re += weight * finite_or_zero(contribution.cov_re);
        self.cov_im += weight * finite_or_zero(contribution.cov_im);
        self.kdp += weight * finite_or_zero(contribution.kdp_deg_km);
        self.ah += weight * f64::from(contribution.ah_db_km.max(0.0));
        self.av += weight * f64::from(contribution.av_db_km.max(0.0));

        let fall_weight = weight * f64::from(contribution.zh.max(0.0));
        let fall_speed = f64::from(contribution.fall_speed_mps.max(0.0));
        let fall_variance = f64::from(contribution.fall_speed_variance_m2s2.max(0.0));
        if fall_weight.is_finite() && fall_speed.is_finite() && fall_variance.is_finite() {
            self.fall_weight += fall_weight;
            self.fall_first += fall_weight * fall_speed;
            self.fall_second += fall_weight * (fall_variance + fall_speed.powi(2));
        }
    }

    /// Finalize intrinsic linear and derived polarimetric moments.
    pub fn finalize(self) -> IntrinsicPolarSample {
        let zh = self.zh.max(0.0);
        let zv = self.zv.max(0.0);
        let covariance = self.cov_re.hypot(self.cov_im);
        let zdr_db = if zh > 0.0 && zv > 0.0 {
            10.0 * (zh / zv).log10()
        } else {
            0.0
        };
        let rho_hv = if zh > 0.0 && zv > 0.0 {
            (covariance / (zh * zv).sqrt()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (fall_speed, fall_variance) = if self.fall_weight > 0.0 {
            let mean = self.fall_first / self.fall_weight;
            let variance = (self.fall_second / self.fall_weight - mean.powi(2)).max(0.0);
            (mean, variance)
        } else {
            (0.0, 0.0)
        };

        IntrinsicPolarSample {
            zh: finite_nonnegative(zh, MAX_LINEAR_Z),
            zv: finite_nonnegative(zv, MAX_LINEAR_Z),
            cov_re: finite_signed(self.cov_re, MAX_LINEAR_Z),
            cov_im: finite_signed(self.cov_im, MAX_LINEAR_Z),
            covariance_magnitude: finite_nonnegative(covariance, MAX_LINEAR_Z),
            kdp_deg_km: finite_signed(self.kdp, 1_000.0),
            ah_db_km: finite_nonnegative(self.ah, 100.0),
            av_db_km: finite_nonnegative(self.av, 100.0),
            fall_speed_mps: finite_nonnegative(fall_speed, 50.0),
            fall_speed_variance_m2s2: finite_nonnegative(fall_variance, 2_500.0),
            zdr_db: finite_signed(zdr_db, 20.0),
            rho_hv: finite_nonnegative(rho_hv, 1.0),
        }
    }
}

/// Intrinsic polarimetric sample after species/beam aggregation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct IntrinsicPolarSample {
    pub zh: f32,
    pub zv: f32,
    pub cov_re: f32,
    pub cov_im: f32,
    pub covariance_magnitude: f32,
    pub kdp_deg_km: f32,
    pub ah_db_km: f32,
    pub av_db_km: f32,
    pub fall_speed_mps: f32,
    pub fall_speed_variance_m2s2: f32,
    pub zdr_db: f32,
    pub rho_hv: f32,
}

/// Moist-air density from pressure, temperature, and water-vapor mixing ratio.
pub fn air_density(pressure_pa: f32, temperature_k: f32, qv_kgkg: f32) -> f32 {
    let pressure = f64::from(pressure_pa);
    let temperature = f64::from(temperature_k);
    let qv = f64::from(qv_kgkg);
    if !pressure.is_finite()
        || !temperature.is_finite()
        || !qv.is_finite()
        || pressure <= 0.0
        || temperature <= 0.0
    {
        return 0.0;
    }
    let virtual_temperature = temperature * (1.0 + 0.61 * qv.max(0.0));
    finite_nonnegative(pressure / (287.05 * virtual_temperature), 10.0)
}

/// Convert logarithmic dBZ to linear Z in mm^6 m^-3.
pub fn dbz_to_z(dbz: f32) -> f32 {
    if dbz.is_nan() {
        return f32::NAN;
    }
    10.0_f32.powf(dbz / 10.0)
}

/// Convert linear Z in mm^6 m^-3 to logarithmic dBZ.
pub fn z_to_dbz(z: f32) -> f32 {
    if z.is_nan() {
        return f32::NAN;
    }
    if z <= 0.0 {
        f32::NEG_INFINITY
    } else {
        10.0 * z.log10()
    }
}

fn finite_nonnegative(value: f64, maximum: f64) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, maximum) as f32
    } else {
        0.0
    }
}

fn finite_signed(value: f64, maximum_magnitude: f64) -> f32 {
    if value.is_finite() {
        value.clamp(-maximum_magnitude, maximum_magnitude) as f32
    } else {
        0.0
    }
}

fn finite_or_zero(value: f32) -> f64 {
    if value.is_finite() {
        f64::from(value)
    } else {
        0.0
    }
}

/// Lanczos gamma approximation; all PSD calls use positive arguments.
pub(crate) fn gamma(value: f64) -> f64 {
    let integer = value.round();
    if (value - integer).abs() < 1.0e-12 && (1.0..=32.0).contains(&integer) {
        // Gamma(n) = (n-1)! for positive integers.
        let mut factorial = 1.0;
        let mut factor = 2u32;
        while factor < integer as u32 {
            factorial *= f64::from(factor);
            factor += 1;
        }
        return factorial;
    }
    const COEFFICIENTS: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    debug_assert!(value >= 0.5);
    let z = value - 1.0;
    let mut series = 0.999_999_999_999_809_9;
    for (index, coefficient) in COEFFICIENTS.iter().enumerate() {
        series += coefficient / (z + index as f64 + 1.0);
    }
    let t = z + 7.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * series
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_scheme() -> SchemeProfile {
        SchemeProfile {
            id: Some(10),
            name: "Morrison".to_owned(),
            capability: MicrophysicsCapability::FullTwoMoment,
            assumption_heavy: false,
        }
    }

    fn rain(number_per_kg: f32, q_kgkg: f32) -> BulkSpeciesInput {
        BulkSpeciesInput {
            kind: HydrometeorKind::Rain,
            q_kgkg,
            number_per_kg: Some(number_per_kg),
            volume_m3_per_kg: None,
            temperature_k: 290.0,
            air_density_kgm3: 1.0,
        }
    }

    #[test]
    fn property_hybrid_seam_is_explicit_and_default_dispatch_stays_unsupported() {
        let p3 = detect_scheme(Some(50), ["QICE", "QNICE"]);
        assert_eq!(p3.capability, MicrophysicsCapability::Unsupported);
        let input = BulkSpeciesInput {
            kind: HydrometeorKind::CloudIce,
            q_kgkg: 1.0e-4,
            number_per_kg: Some(2.0e5),
            volume_m3_per_kg: None,
            temperature_k: 260.0,
            air_density_kgm3: 1.0,
        };

        assert_eq!(
            bulk_sband_contribution(input, &p3),
            BulkContribution::default()
        );
        let hybrid = property_hybrid_bulk_rayleigh_contribution(input);
        assert!(hybrid.zh > 0.0);
        assert!(hybrid.zv > 0.0);
        assert_eq!(
            PROPERTY_HYBRID_BULK_RAYLEIGH_REVISION,
            "bowecho-p3-ishmael-hybrid-bulk-rayleigh-v1"
        );
    }

    #[test]
    fn gamma_psd_closes_supplied_mass_and_number() {
        let input = rain(1.0e6, 1.0e-3);
        let psd = close_psd(input, species_parameters(input.kind)).expect("valid PSD");
        let reconstructed_number = psd.moment(0.0);
        let reconstructed_mass = PI * psd.density_kg_m3 / 6.0 * psd.moment(3.0);
        assert!((reconstructed_number / 1.0e6 - 1.0).abs() < 1.0e-10);
        assert!((reconstructed_mass / psd.mass_kg_m3 - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn fixed_intercept_psd_closes_mass() {
        let mut input = rain(1.0, 7.5e-4);
        input.number_per_kg = None;
        let psd = close_psd(input, species_parameters(input.kind)).expect("valid PSD");
        let reconstructed_mass = PI * psd.density_kg_m3 / 6.0 * psd.moment(3.0);
        assert!((reconstructed_mass / psd.mass_kg_m3 - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn small_rain_is_nearly_spherical_and_highly_correlated() {
        let contribution = bulk_sband_contribution(rain(1.0e10, 1.0e-4), &full_scheme());
        let mut accumulator = PolarAccumulator::default();
        accumulator.add(1.0, contribution);
        let sample = accumulator.finalize();
        assert!(sample.zdr_db.abs() < 0.1, "ZDR={}", sample.zdr_db);
        assert!(sample.rho_hv > 0.99, "rhoHV={}", sample.rho_hv);
    }

    #[test]
    fn polar_accumulator_preserves_signed_kdp() {
        let contribution = BulkContribution {
            zh: 10.0,
            zv: 10.0,
            kdp_deg_km: -2.5,
            ..BulkContribution::default()
        };
        let mut accumulator = PolarAccumulator::default();
        accumulator.add(2.0, contribution);
        assert_eq!(accumulator.finalize().kdp_deg_km, -5.0);

        let mut legacy_positive = PolarAccumulator::default();
        legacy_positive.add(
            2.0,
            BulkContribution {
                kdp_deg_km: 2.5,
                ..contribution
            },
        );
        assert_eq!(legacy_positive.finalize().kdp_deg_km, 5.0);
    }

    #[test]
    fn larger_rain_drops_increase_zdr() {
        let small = bulk_sband_contribution(rain(1.0e9, 5.0e-4), &full_scheme());
        let large = bulk_sband_contribution(rain(1.0e2, 5.0e-4), &full_scheme());
        let mut small_acc = PolarAccumulator::default();
        small_acc.add(1.0, small);
        let mut large_acc = PolarAccumulator::default();
        large_acc.add(1.0, large);
        let small_zdr = small_acc.finalize().zdr_db;
        let large_zdr = large_acc.finalize().zdr_db;
        assert!(
            large_zdr > small_zdr + 0.3,
            "small={small_zdr}, large={large_zdr}"
        );
    }

    #[test]
    fn mixed_species_reduce_rho_hv() {
        let rain_contribution = bulk_sband_contribution(rain(2.0e5, 8.0e-4), &full_scheme());
        let hail = BulkSpeciesInput {
            kind: HydrometeorKind::Hail,
            q_kgkg: 2.0e-3,
            number_per_kg: Some(2.0e3),
            volume_m3_per_kg: None,
            temperature_k: 271.15,
            air_density_kgm3: 1.0,
        };
        let hail_contribution = bulk_sband_contribution(hail, &full_scheme());
        let hail_weight = rain_contribution.zh / hail_contribution.zh.max(f32::MIN_POSITIVE);

        let mut rain_only = PolarAccumulator::default();
        rain_only.add(1.0, rain_contribution);
        let mut mixture = PolarAccumulator::default();
        mixture.add(1.0, rain_contribution);
        mixture.add(hail_weight, hail_contribution);
        assert!(mixture.finalize().rho_hv < rain_only.finalize().rho_hv - 0.01);
    }

    #[test]
    fn outputs_are_finite_and_nonnegative_for_all_species() {
        for kind in [
            HydrometeorKind::CloudWater,
            HydrometeorKind::Rain,
            HydrometeorKind::CloudIce,
            HydrometeorKind::Snow,
            HydrometeorKind::Graupel,
            HydrometeorKind::Hail,
        ] {
            let contribution = bulk_sband_contribution(
                BulkSpeciesInput {
                    kind,
                    q_kgkg: 1.0e-3,
                    number_per_kg: None,
                    volume_m3_per_kg: None,
                    temperature_k: 270.0,
                    air_density_kgm3: 1.0,
                },
                &full_scheme(),
            );
            for value in [
                contribution.zh,
                contribution.zv,
                contribution.kdp_deg_km,
                contribution.ah_db_km,
                contribution.av_db_km,
                contribution.fall_speed_mps,
                contribution.fall_speed_variance_m2s2,
            ] {
                assert!(value.is_finite() && value >= 0.0, "{kind:?}: {value}");
            }
            assert!(contribution.cov_re.is_finite());
            assert!(contribution.cov_im.is_finite());
        }
    }

    #[test]
    fn fall_speeds_remain_in_species_ranges() {
        let rain_speed =
            bulk_sband_contribution(rain(1.0e5, 1.0e-3), &full_scheme()).fall_speed_mps;
        let snow_speed = bulk_sband_contribution(
            BulkSpeciesInput {
                kind: HydrometeorKind::Snow,
                q_kgkg: 1.0e-3,
                number_per_kg: None,
                volume_m3_per_kg: None,
                temperature_k: 263.0,
                air_density_kgm3: 0.9,
            },
            &full_scheme(),
        )
        .fall_speed_mps;
        let hail_speed = bulk_sband_contribution(
            BulkSpeciesInput {
                kind: HydrometeorKind::Hail,
                q_kgkg: 5.0e-3,
                number_per_kg: Some(1.0e3),
                volume_m3_per_kg: None,
                temperature_k: 268.0,
                air_density_kgm3: 1.0,
            },
            &full_scheme(),
        )
        .fall_speed_mps;
        assert!((0.0..=10.5).contains(&rain_speed));
        assert!((0.0..=3.0).contains(&snow_speed));
        assert!((0.0..=25.0).contains(&hail_speed));
    }

    #[test]
    fn detects_common_schemes_and_field_downgrades() {
        for (id, expected) in [
            (2, MicrophysicsCapability::MassOnly),
            (6, MicrophysicsCapability::MassOnly),
            (8, MicrophysicsCapability::PartialTwoMoment),
            (9, MicrophysicsCapability::FullTwoMoment),
            (10, MicrophysicsCapability::FullTwoMoment),
            (14, MicrophysicsCapability::PartialTwoMoment),
            (16, MicrophysicsCapability::PartialTwoMoment),
            (18, MicrophysicsCapability::FullTwoMoment),
            (24, MicrophysicsCapability::MassOnly),
            (26, MicrophysicsCapability::PartialTwoMoment),
            (28, MicrophysicsCapability::PartialTwoMoment),
            (29, MicrophysicsCapability::PartialTwoMoment),
            (38, MicrophysicsCapability::PartialTwoMoment),
            (40, MicrophysicsCapability::FullTwoMoment),
        ] {
            assert_eq!(
                detect_scheme(Some(id), std::iter::empty::<&str>()).capability,
                expected,
                "mp_physics={id}"
            );
        }

        let morrison = detect_scheme(
            Some(10),
            ["QRAIN", "QNRAIN", "QSNOW", "QNSNOW", "QGRAUP", "QNGRAUP"],
        );
        assert_eq!(morrison.capability, MicrophysicsCapability::FullTwoMoment);
        assert!(!morrison.assumption_heavy);

        let missing_numbers = detect_scheme(Some(10), ["QRAIN", "QSNOW", "QGRAUP"]);
        assert_eq!(missing_numbers.capability, MicrophysicsCapability::MassOnly);
        assert!(missing_numbers.assumption_heavy);

        for unsupported in [50, 51, 52, 53, 55] {
            assert_eq!(
                detect_scheme(Some(unsupported), ["QRAIN", "QNRAIN"]).capability,
                MicrophysicsCapability::Unsupported
            );
        }
    }

    #[test]
    fn density_and_log_conversions_round_trip() {
        let density = air_density(100_000.0, 300.0, 0.01);
        assert!((1.0..1.3).contains(&density));
        for dbz in [-20.0, 0.0, 35.0, 70.0] {
            assert!((z_to_dbz(dbz_to_z(dbz)) - dbz).abs() < 1.0e-4);
        }
    }
}
