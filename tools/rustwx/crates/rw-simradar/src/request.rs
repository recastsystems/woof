use bowecho_simradar::wrf_temporal::{AtmosphereTimeMode, MissingNeighborPolicy};
use bowecho_simradar::{
    BeamIntegration, ScanTiming, SimulationMode, SyntheticRadarComputePreference,
    SyntheticRadarConfig, SyntheticScanStrategy, WrfRadarFields,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// The request contract. `volume_paths` and `scene_shapes` are optional
/// request keys this build accepts; a binary predating them refuses the key
/// by name (`deny_unknown_fields`), so the marker names them and a forecast
/// door refuses that binary before the forecast starts, not at its first
/// scan-timing history.
/// `bands=v1` prevents an older catalog/operator being admitted for a
/// C-band site and failing only after the forecast's first history.
pub const ABI: &str = "rw_simradar --request REQUEST.json schema=simulated-radar.request/v1 manifest=simulated-radar.manifest/v1 volume_paths=v1 scene_shapes=v1 bands=v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema: String,
    pub history_paths: Vec<PathBuf>,
    pub outdir: PathBuf,
    pub config: Config,
    /// The histories that publish volumes; absent means every history. The
    /// rest are read only as scan-timing neighbours. A live scan-timing run
    /// names the earlier of two adjacent histories here, so each volume is
    /// written once, with its successor, instead of first as a held anchor
    /// and again as a linear-adjacent scan under a second generation.
    #[serde(default)]
    pub volume_paths: Option<Vec<PathBuf>>,
    /// `[nx, ny, nz]` mass-grid shapes a forecast will write, priced by
    /// `--estimate` before any history exists. A volume request reads its
    /// real history headers instead and refuses this key.
    #[serde(default)]
    pub scene_shapes: Vec<[usize; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Sites {
    Auto(String),
    List(Vec<SiteChoice>),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SiteChoice {
    Id(String),
    Custom(CustomSite),
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomSite {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub height_m: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beamwidth_deg: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wavelength_mm: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Fields {
    Auto(String),
    List(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub enabled: bool,
    pub sites: Sites,
    pub scan_strategy: String,
    pub elevations_deg: Vec<f64>,
    pub formats: Vec<String>,
    pub timing: String,
    pub fields: Fields,
    pub range_km: f64,
    pub gate_spacing_m: f64,
    pub azimuth_step_deg: f64,
    pub volume_duration_s: f64,
    /// The colour tables the reflectivity and velocity PPIs draw with:
    /// `standard` (the radar tables) or `classic` (the reflectivity ladder
    /// and blue-red velocity scale they replaced). Left out of the
    /// serialized config when `standard`, so a request that does not name
    /// it keeps the `config_sha256` it had before the key existed. The
    /// radar files do not depend on it.
    #[serde(skip_serializing_if = "is_standard_color_tables")]
    pub color_tables: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub band: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wavelength_mm: Option<f64>,
}

fn is_standard_color_tables(name: &String) -> bool {
    name == "standard"
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            sites: Sites::Auto("auto".into()),
            scan_strategy: "vcp212".into(),
            elevations_deg: vec![
                0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5,
            ],
            formats: vec!["level2".into(), "cfradial1".into()],
            timing: "history".into(),
            fields: Fields::Auto("auto".into()),
            range_km: 230.0,
            gate_spacing_m: 250.0,
            azimuth_step_deg: 1.0,
            volume_duration_s: 300.0,
            color_tables: "standard".into(),
            band: None,
            wavelength_mm: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedSite {
    pub id: String,
    pub lat: f64,
    pub lon: f64,
    pub height_m: Option<f64>,
    pub beamwidth_deg: Option<f64>,
    pub band: String,
}

#[derive(Deserialize)]
struct CatalogSite {
    id: String,
    lat: f64,
    lon: f64,
    antenna_height_msl_m: f64,
    #[serde(default = "s_band")]
    band: String,
    #[serde(default = "automatic_site")]
    automatic: bool,
    #[serde(default)]
    beamwidth_deg: Option<f64>,
}

fn s_band() -> String { "S".into() }
fn automatic_site() -> bool { true }

pub fn resolve_band(band: Option<&str>, wavelength: Option<f64>) -> Result<Option<String>,String> {
    let mut band=band.map(str::to_ascii_uppercase);
    if band.as_deref().is_some_and(|v| !matches!(v,"S"|"C"|"X")) {
        return Err("radar_band_unsupported: band must be S, C or X".into());
    }
    if let Some(wavelength)=wavelength {
        let selected=match wavelength {107.068735=>"S",53.5=>"C",33.3=>"X",
            _=>return Err("radar_wavelength_unsupported: legacy S is 107.068735 mm (2.8 GHz); C/X tables are 53.5 and 33.3 mm; another wavelength would change the requested scattering and attenuation".into())};
        if band.as_deref().is_some_and(|b|b!=selected) {
            return Err("radar_band_wavelength_mismatch: band and wavelength select different tables".into());
        }
        band=Some(selected.into());
    }
    Ok(band)
}

fn valid_id(id: &str) -> bool {
    id.len() == 4 && id.bytes().all(|b| b.is_ascii_alphanumeric())
}

/// Upper limits on the scan geometry, each with the breakage it prevents.
/// `gpuwm.simulated_radar_config.GEOMETRY_LIMITS` carries the same rows, so
/// the Python door refuses with this sentence before the native runs.
/// `gate_spacing_m` has no universal ceiling: one gate is always budgeted,
/// and only Level II stores the spacing in a bounded field (below).
pub const GEOMETRY_LIMITS: [(&str, f64, &str); 3] = [
    (
        "range_km",
        460.0,
        "460 km is the WSR-88D's longest unambiguous range (its long-PRT surveillance cut), so a simulated S-band volume reaching farther would hold echo where that radar's own data range-folds",
    ),
    (
        "azimuth_step_deg",
        720.0,
        "the ray count is 360 / azimuth_step_deg rounded, which is zero above 720 degrees; the simulator would scan one ray while the memory and work estimates priced none",
    ),
    (
        "volume_duration_s",
        3600.0,
        "a custom ladder turns at 360 x elevations / volume_duration_s deg/s and the simulator holds that rate at no less than 0.1 deg/s, so a longer one-elevation scan would run 3600 s per sweep instead of the duration asked for",
    ),
];

/// Message 31 carries the first-gate range and gate spacing in fields that
/// hold 1 to 32767 m (the Level II writer's `MAX_RANGE_FIELD_M`).
pub const LEVEL2_MAX_GATE_SPACING_M: f64 = 32_767.0;

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        resolve_band(self.band.as_deref(),self.wavelength_mm)?;
        let values = [
            ("range_km", self.range_km),
            ("gate_spacing_m", self.gate_spacing_m),
            ("azimuth_step_deg", self.azimuth_step_deg),
            ("volume_duration_s", self.volume_duration_s),
        ];
        for (name, value) in values {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "{name} must be finite and positive: a zero, negative or nonfinite value leaves no gate, ray or scan time to sample"
                ));
            }
        }
        for (name, limit, reason) in GEOMETRY_LIMITS {
            let value = values.iter().find(|(key, _)| *key == name).map_or(0.0, |v| v.1);
            if value > limit {
                return Err(format!("{name} must be at most {limit}: {reason}"));
            }
        }
        if self.formats.iter().any(|f| f == "level2") && self.gate_spacing_m > LEVEL2_MAX_GATE_SPACING_M {
            return Err(format!(
                "gate_spacing_m must be at most {LEVEL2_MAX_GATE_SPACING_M} with level2: Message 31 stores gate spacing in a field holding 1 to 32767 m; omit level2 for coarser gates"
            ));
        }
        if self.gate_spacing_m.fract() != 0.0 {
            return Err("gate_spacing_m must be a whole metre because radar gate geometry is stored as integer metres".into());
        }
        if (360.0 / self.azimuth_step_deg).ceil() > u16::MAX as f64
            || (self.range_km * 1000.0 / self.gate_spacing_m).ceil() > u16::MAX as f64
        {
            return Err(
                "radial or gate count exceeds the Level II unsigned 16-bit geometry fields".into(),
            );
        }
        if !matches!(self.timing.as_str(), "history" | "scan") {
            return Err("timing must be history or scan".into());
        }
        if !matches!(
            self.scan_strategy.as_str(),
            "vcp212" | "low_tilts" | "custom"
        ) {
            return Err("scan_strategy must be vcp212, low_tilts, or custom".into());
        }
        if self.elevations_deg.is_empty()
            || self.elevations_deg.len() > 255
            || self
                .elevations_deg
                .iter()
                .any(|e| !e.is_finite() || !(-1.0..=90.0).contains(e))
            || self.elevations_deg.windows(2).any(|e| e[0] >= e[1])
        {
            return Err("custom elevations must be finite, increasing, between -1 and 90 degrees, with at most 255 sweeps".into());
        }
        if (self.scan_strategy == "vcp212" && self.elevations_deg != Self::default().elevations_deg)
            || (self.scan_strategy == "low_tilts" && self.elevations_deg != vec![0.5, 0.9, 1.3])
        {
            return Err("an explicit elevation ladder must match its named strategy; use custom for a different ladder".into());
        }
        if self.formats.is_empty()
            || self
                .formats
                .iter()
                .any(|f| !["level2", "cfradial1", "cfradial2", "odim"].contains(&f.as_str()))
            || self.formats.iter().collect::<BTreeSet<_>>().len() != self.formats.len()
        {
            return Err(
                "formats must contain unique level2, cfradial1, cfradial2, or odim values".into(),
            );
        }
        if (self.range_km * 1000.0 / self.gate_spacing_m).ceil() > 16384.0 {
            return Err("radar writers support at most 16384 gates per radial; increase gate_spacing_m or reduce range_km".into());
        }
        if self.formats.iter().any(|f| f == "level2") {
            if self.native().physical_scan_legs().len() > 32 {
                return Err("Level II writer supports at most 32 physical cuts; shorten the ladder or omit level2".into());
            }
            if let Sites::List(sites) = &self.sites {
                for site in sites {
                    if let SiteChoice::Custom(site) = site {
                        if site.id.to_ascii_uppercase().starts_with('T')
                            && !site.id.eq_ignore_ascii_case("TJUA")
                        {
                            return Err("custom site IDs beginning T trigger a fixed TDWR station lookup in Py-ART; use another four-character ID or omit level2".into());
                        }
                    }
                }
            }
        }
        match &self.fields {
            Fields::Auto(v) if v == "auto" => (),
            Fields::List(v)
                if !v.is_empty()
                    && v.iter().all(|f| {
                        ["reflectivity", "velocity", "zdr", "rhohv", "phidp", "kdp", "spectrum_width"]
                            .contains(&f.as_str())
                    }) =>
            {
                ()
            }
            _ => return Err("fields must be auto or a list of supported radar moments".into()),
        }
        self.color_set()?;
        Ok(())
    }

    /// The radar colour set `color_tables` names, or a refusal listing them.
    pub fn color_set(&self) -> Result<rustwx_render::RadarColorSet, String> {
        rustwx_render::RadarColorSet::parse(&self.color_tables)
            .map_err(|error| format!("color_tables: {error}"))
    }

    /// Memory needed for polar working rows, retained moments, writer copies,
    /// and the PPI canvas. Checked arithmetic rejects impossible allocations.
    pub fn scan_memory_bytes(&self) -> Result<u64, String> {
        let rays = (360.0 / self.azimuth_step_deg).round() as u64;
        let gates = (self.range_km * 1000.0 / self.gate_spacing_m).ceil() as u64;
        let cuts = self.native().physical_scan_legs().len() as u64;
        // The reference operator also holds six propagation diagnostics until
        // the public six-moment selection is applied after beam integration.
        let fields = if self.wants_dual_pol() { 12u64 } else { 2u64 };
        let checked = || {
            let bins = rays.checked_mul(gates)?;
            let retained = bins
                .checked_mul(cuts)?
                .checked_mul(fields)?
                .checked_mul(4)?
                .checked_mul(4)?;
            let rows = bins.checked_mul(55)?;
            retained.checked_add(rows)?.checked_add(128 * 1024 * 1024)
        };
        checked().ok_or_else(|| "radar sampling dimensions overflow the memory estimate".into())
    }

    pub fn check_memory(&self, extra_bytes: u64) -> Result<(), String> {
        self.check_memory_with_available(extra_bytes, rw_host_memory::available_bytes())
    }

    pub fn check_memory_with_available(&self, extra_bytes: u64, available: Option<u64>) -> Result<(), String> {
        let required = self
            .scan_memory_bytes()?
            .checked_add(extra_bytes)
            .ok_or("radar memory estimate overflow")?;
        if let Some(available) = available {
            if required > available {
                return Err(format!(
                    "radar scan and atmosphere require an estimated {required} bytes, but only {available} host bytes are available within the memory limit; use wider gates, fewer rays or fewer elevations"
                ));
            }
        } else {
            return Err("radar cannot determine available host memory; admitting an unbounded polar allocation could terminate the forecast process".into());
        }
        Ok(())
    }

    pub fn wants_dual_pol(&self) -> bool {
        if self.atmosphere_factor() > 1 { return true; }
        match &self.fields {
            Fields::Auto(_) => true,
            Fields::List(v) => v
                .iter()
                .any(|f| !["reflectivity", "velocity", "spectrum_width"].contains(&f.as_str())),
        }
    }

    pub fn native(&self) -> SyntheticRadarConfig {
        let mut c = SyntheticRadarConfig::default();
        c.simulation_mode = SimulationMode::Truth;
        c.compute_preference = SyntheticRadarComputePreference::Cpu;
        c.scan_strategy = if self.scan_strategy == "vcp212" {
            SyntheticScanStrategy::Build24Vcp212
        } else {
            SyntheticScanStrategy::CustomLegacy
        };
        c.elevations_deg = self.elevations_deg.clone();
        c.azimuth_count = (360.0 / self.azimuth_step_deg).round() as usize;
        c.gate_spacing_m = self.gate_spacing_m;
        c.max_range_m = self.range_km * 1000.0;
        c.beam_integration = BeamIntegration::Balanced;
        c.ref_gate_texture = false;
        c.vel_gate_texture = false;
        c.ref_floor_dbz = -30.0;
        c.dual_pol = self.wants_dual_pol();
        // Opt-in: automatic/default requests keep their existing moments.
        c.spectrum_width = matches!(&self.fields, Fields::List(v) if v.iter().any(|f| f == "spectrum_width"));
        c.emit_quality_fields = false;
        c.scan_timing = if self.timing == "scan" {
            ScanTiming::TimedVolume
        } else {
            ScanTiming::InstantaneousTruth
        };
        c.atmosphere_time_mode = if self.timing == "scan" {
            AtmosphereTimeMode::LinearAdjacent
        } else {
            AtmosphereTimeMode::FrozenAtVolumeStart
        };
        c.missing_neighbor_policy = MissingNeighborPolicy::HoldAnchor;
        // Named VCPs carry measured antenna rates. This duration applies to custom ladders.
        c.transition_delay_s = 0.0;
        c.rotation_rate_deg_s =
            (360.0 * self.elevations_deg.len() as f64 / self.volume_duration_s) as f32;
        c.site_name = Some("WOOF simulated radar".into());
        c
    }

    pub fn resolve_sites(&self, fields: &WrfRadarFields) -> Result<Vec<ResolvedSite>, String> {
        let rows: Vec<CatalogSite> = serde_json::from_str(include_str!("../data/sites.json"))
            .map_err(|e| format!("NEXRAD catalog: {e}"))?;
        let automatic_ids: BTreeSet<_> = rows.iter().filter(|s|s.automatic).map(|s|s.id.clone()).collect();
        let override_band=resolve_band(self.band.as_deref(),self.wavelength_mm)?;
        let catalog: Vec<_> = rows
            .into_iter()
            .map(|s| ResolvedSite {
                id: s.id,
                lat: s.lat,
                lon: s.lon,
                height_m: Some(s.antenna_height_msl_m),
                beamwidth_deg: s.beamwidth_deg,
                band: override_band.clone().unwrap_or(s.band),
            })
            .collect();
        let mut sites = match &self.sites {
            Sites::Auto(v) if v == "auto" => catalog
                .into_iter()
                .filter(|s|automatic_ids.contains(&s.id))
                .filter(|s| coverage_overlaps(s, fields, self.range_km * 1000.0))
                .collect(),
            Sites::Auto(_) => return Err("sites string must be auto".into()),
            Sites::List(choices) => {
                let mut selected = Vec::new();
                for choice in choices {
                    selected.push(match choice {
                    SiteChoice::Id(id)=>catalog.iter().find(|s| s.id.eq_ignore_ascii_case(id)).cloned()
                        .ok_or_else(|| format!("unknown NEXRAD site {id}; use a custom latitude, longitude, and antenna height"))?,
                    SiteChoice::Custom(s)=> {
                        if !valid_id(&s.id) || !s.lat.is_finite() || !s.lon.is_finite() || !s.height_m.is_finite()
                            || !(-90.0..=90.0).contains(&s.lat) || !(-180.0..=180.0).contains(&s.lon)
                            || !(-500.0..=10000.0).contains(&s.height_m) { return Err("custom site has an invalid ID, coordinate, or antenna height".into()); }
                        if s.beamwidth_deg.is_some_and(|w| !w.is_finite() || w < 0.05 || w > f32::MAX as f64) {
                            return Err("custom site beamwidth_deg must be at least 0.05 degrees and fit a finite f32; narrower beams are clamped by the sampler and overflow would make the sampled width nonfinite".into());
                        }
                        let band=override_band.clone().or(resolve_band(s.band.as_deref(),s.wavelength_mm)?).unwrap_or_else(s_band);
                        ResolvedSite{id:s.id.to_ascii_uppercase(),lat:s.lat,lon:s.lon,height_m:Some(s.height_m),beamwidth_deg:s.beamwidth_deg,band}
                    }
                });
                }
                selected
            }
        };
        sites.sort_by(|a, b| a.id.cmp(&b.id));
        if sites.windows(2).any(|s| s[0].id == s[1].id) {
            return Err("duplicate radar site IDs would overwrite volume files".into());
        }
        if sites.is_empty() && matches!(self.sites, Sites::List(_)) {
            return Err("sites list must contain at least one radar".into());
        }
        Ok(sites)
    }

    /// Additional live arrays for exact elevation-dependent scattering and
    /// the PSD inputs. Default S requests keep their previous admission.
    pub fn atmosphere_factor(&self) -> u64 {
        if self.band.as_deref().is_some_and(|b|b.eq_ignore_ascii_case("S")) || self.wavelength_mm==Some(107.068735) {return 1;}
        if self.band.is_some() || self.wavelength_mm.is_some() {return 6;}
        match &self.sites {
            Sites::List(sites) if sites.iter().any(|s|match s {
                SiteChoice::Custom(s)=>s.band.as_deref().is_some_and(|b|!b.eq_ignore_ascii_case("S")) || s.wavelength_mm.is_some_and(|w|w!=107.068735),
                SiteChoice::Id(id)=>serde_json::from_str::<Vec<CatalogSite>>(include_str!("../data/sites.json"))
                    .map(|rows|rows.iter().any(|row|row.id.eq_ignore_ascii_case(id) && row.band!="S"))
                    .unwrap_or(true),
            })=>6,
            _=>1,
        }
    }
}

fn distance_m(lat: f64, lon: f64, lat2: f64, lon2: f64) -> f64 {
    let a = ((lat2 - lat).to_radians() * 0.5).sin().powi(2)
        + lat.to_radians().cos()
            * lat2.to_radians().cos()
            * ((lon2 - lon).to_radians() * 0.5).sin().powi(2);
    2.0 * 6_371_000.0 * a.clamp(0.0, 1.0).sqrt().asin()
}

/// Intersect the coverage disk with the actual grid perimeter. Longitude is
/// unwrapped about the site so dateline-crossing domains retain their footprint.
pub fn coverage_overlaps(site: &ResolvedSite, fields: &WrfRadarFields, range: f64) -> bool {
    let mut boundary = Vec::with_capacity(2 * (fields.nx + fields.ny));
    boundary.extend(0..fields.nx);
    boundary.extend((1..fields.ny).map(|j| j * fields.nx + fields.nx - 1));
    boundary.extend(
        (0..fields.nx - 1)
            .rev()
            .map(|i| (fields.ny - 1) * fields.nx + i),
    );
    boundary.extend((1..fields.ny - 1).rev().map(|j| j * fields.nx));
    let origin = f64::from(fields.lon[boundary[0]]);
    let mut previous_lon = origin;
    let mut points: Vec<_> = boundary
        .iter()
        .map(|&p| {
            let raw = f64::from(fields.lon[p]);
            let unwrapped = previous_lon + (raw - previous_lon + 180.0).rem_euclid(360.0) - 180.0;
            previous_lon = unwrapped;
            (unwrapped, f64::from(fields.lat[p]) - site.lat)
        })
        .collect();
    let middle = (points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min)
        + points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max))
        * 0.5;
    let site_lon = site.lon + 360.0 * ((middle - site.lon) / 360.0).round();
    for point in &mut points {
        point.0 -= site_lon;
    }
    let mut inside = false;
    let mut previous = *points.last().unwrap();
    for &current in &points {
        if (current.1 > 0.0) != (previous.1 > 0.0)
            && 0.0 < (previous.0 - current.0) * (-current.1) / (previous.1 - current.1) + current.0
        {
            inside = !inside;
        }
        previous = current;
    }
    if inside {
        return true;
    }
    // A half-cell diagonal includes coverage that clips the outer grid cells.
    let padding = fields.dx_m.unwrap_or(0.0) * std::f64::consts::FRAC_1_SQRT_2;
    boundary.into_iter().any(|p| {
        distance_m(
            site.lat,
            site.lon,
            f64::from(fields.lat[p]),
            f64::from(fields.lon[p]),
        ) <= range + padding
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_defaults_keep_physical_volume_and_supported_formats() {
        let c = Config::default();
        c.validate().unwrap();
        assert_eq!(
            c.native().compute_preference,
            SyntheticRadarComputePreference::Cpu
        );
        assert_eq!(c.native().beam_integration, BeamIntegration::Balanced);
        assert!(!c.native().ref_gate_texture);
    }
    #[test]
    fn gate_limit_applies_to_every_writer() {
        let mut c = Config::default();
        c.gate_spacing_m = 1.0;
        c.range_km = 20.0;
        assert!(c.validate().unwrap_err().contains("16384"));
        c.formats = vec!["cfradial1".into()];
        assert!(c.validate().unwrap_err().contains("16384"));
        c.formats = vec!["cfradial2".into()];
        assert!(c.validate().unwrap_err().contains("16384"));
        c.formats = vec!["odim".into()];
        assert!(c.validate().unwrap_err().contains("16384"));
    }
    #[test]
    fn operational_site_table_has_real_antenna_heights() {
        let sites: Vec<CatalogSite> =
            serde_json::from_str(include_str!("../data/sites.json")).unwrap();
        assert_eq!(sites.len(), 204);
        assert_eq!(
            sites.iter().map(|s| &s.id).collect::<BTreeSet<_>>().len(),
            204
        );
        assert!(
            sites
                .iter()
                .all(|s| valid_id(&s.id) && s.antenna_height_msl_m.is_finite())
        );
        let site = sites.iter().find(|s| s.id == "KTLX").unwrap();
        assert!((site.antenna_height_msl_m - 389.5344).abs() < 1e-6);
        assert_eq!(sites.iter().filter(|s|s.band=="S" && s.automatic).count(),158);
        assert_eq!(sites.iter().filter(|s|s.band=="C" && !s.automatic).count(),46);
        let tdwr=sites.iter().find(|s|s.id=="TOKC").unwrap();
        assert!((tdwr.antenna_height_msl_m-398.9832).abs()<1e-6);
        assert_eq!(tdwr.beamwidth_deg,Some(0.55));
    }

    #[test]
    fn site_band_and_explicit_wavelength_select_the_native_operator() {
        let fields=box_fields(-98.,-97.,35.,36.);
        let mut config=Config {sites:Sites::List(vec![SiteChoice::Id("TOKC".into())]),..Default::default()};
        assert_eq!(config.resolve_sites(&fields).unwrap()[0].band,"C");
        assert!(config.atmosphere_factor()>1);
        config.band=Some("S".into());
        assert_eq!(config.resolve_sites(&fields).unwrap()[0].band,"S");
        assert_eq!(config.atmosphere_factor(),1);
        config.band=Some("s".into());
        assert_eq!(config.atmosphere_factor(),1);
        config.band=None;config.wavelength_mm=Some(107.068735);
        assert_eq!(config.resolve_sites(&fields).unwrap()[0].band,"S");
        assert_eq!(config.atmosphere_factor(),1);
        config.band=None;config.wavelength_mm=Some(33.3);
        assert_eq!(config.resolve_sites(&fields).unwrap()[0].band,"X");
        config.wavelength_mm=Some(54.);
        assert!(config.validate().unwrap_err().contains("radar_wavelength_unsupported"));
    }

    fn box_fields(west: f32, east: f32, south: f32, north: f32) -> WrfRadarFields {
        use bowecho_simradar::ModelRadarFields;
        WrfRadarFields::from_model_fields(ModelRadarFields {
            nx: 2,
            ny: 2,
            nz: 2,
            latitude_deg: vec![south, south, north, north],
            longitude_deg: vec![west, east, west, east],
            height_msl_m: [vec![0.0; 4], vec![10000.0; 4]].concat(),
            reflectivity_dbz: vec![30.0; 8],
            eastward_wind_mps: vec![5.0; 8],
            northward_wind_mps: vec![0.0; 8],
            upward_wind_mps: vec![0.0; 8],
            terrain_msl_m: vec![0.0; 4],
            grid_spacing_m: Some(3000.0),
            model_label: "coverage test".into(),
        })
        .unwrap()
    }

    fn test_site(lat: f64, lon: f64) -> ResolvedSite {
        ResolvedSite {
            id: "X001".into(),
            band: "S".into(),
            lat,
            lon,
            height_m: Some(100.0),
            beamwidth_deg: None,
        }
    }

    #[test]
    fn coverage_keeps_both_dateline_sides_and_excludes_the_opposite_world() {
        let fields = box_fields(179.0, -179.0, 10.0, 11.0);
        assert!(coverage_overlaps(&test_site(10.5, 179.5), &fields, 1000.0));
        assert!(coverage_overlaps(&test_site(10.5, -179.5), &fields, 1000.0));
        assert!(
            !coverage_overlaps(&test_site(10.5, 0.0), &fields, 230000.0),
            "a small dateline domain cannot contain a site on the opposite side of Earth"
        );
    }

    #[test]
    fn auto_site_selection_uses_coverage_not_only_antenna_inside_domain() {
        let fields = box_fields(-97.4, -97.1, 35.2, 35.5);
        let mut config = Config::default();
        config.range_km = 230.0;
        let sites = config.resolve_sites(&fields).unwrap();
        assert!(sites.iter().any(|site| site.id == "KTLX"));
        assert!(
            sites.iter().any(|site| site.id == "KVNX"),
            "an outside antenna whose coverage reaches the domain must be selected"
        );
        assert!(!sites.iter().any(|site| site.id == "KAMX"));
    }

    #[test]
    fn named_low_ladder_builds_three_tilts_and_rejects_conflicting_values() {
        let mut config: Config =
            serde_json::from_str(r#"{"scan_strategy":"low_tilts","elevations_deg":[0.5,0.9,1.3]}"#)
                .unwrap();
        config.validate().unwrap();
        let elevations: Vec<_> = config
            .native()
            .physical_scan_legs()
            .iter()
            .map(|leg| leg.elevation_deg)
            .collect();
        assert_eq!(elevations, vec![0.5, 0.9, 1.3]);
        config.elevations_deg.push(3.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn spectrum_width_is_explicit_and_does_not_enable_dual_pol() {
        let default = Config::default();
        assert!(!default.native().spectrum_width);
        let config: Config = serde_json::from_str(
            r#"{"fields":["reflectivity","velocity","spectrum_width"]}"#,
        ).unwrap();
        config.validate().unwrap();
        assert!(config.native().spectrum_width);
        assert!(!config.wants_dual_pol());
    }

    #[test]
    fn custom_beamwidth_is_optional_and_validated_without_clamping() {
        let fields = box_fields(-98.0, -97.0, 35.0, 36.0);
        let default: Config = serde_json::from_str(
            r#"{"sites":[{"id":"SIM1","lat":35.5,"lon":-97.5,"height_m":300.0}]}"#,
        ).unwrap();
        assert!(!serde_json::to_string(&default).unwrap().contains("beamwidth_deg"));
        let mut value = serde_json::to_value(&default).unwrap();
        value["sites"][0]["beamwidth_deg"] = serde_json::json!(1.5);
        let config: Config = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(config.resolve_sites(&fields).unwrap()[0].beamwidth_deg, Some(1.5));
        value["sites"][0]["beamwidth_deg"] = serde_json::json!(0.01);
        let invalid: Config = serde_json::from_value(value).unwrap();
        assert!(invalid.resolve_sites(&fields).unwrap_err().contains("beamwidth_deg"));
    }

    #[test]
    fn numerically_valid_cfradial_scan_cannot_bypass_the_host_memory_budget() {
        let mut config = Config::default();
        config.formats = vec!["cfradial1".into()];
        config.fields = Fields::List(vec!["reflectivity".into(), "velocity".into()]);
        config.scan_strategy = "custom".into();
        config.elevations_deg = (0..255).map(|index| index as f64 * 0.25).collect();
        config.azimuth_step_deg = 0.006;
        config.gate_spacing_m = 32.0;
        config.range_km = 460.0;
        config.validate().unwrap();
        let required = config.scan_memory_bytes().unwrap();
        assert!(
            required > 1_000_000_000_000,
            "format-valid ray and gate counts can still require terabytes"
        );
        if rw_host_memory::available_bytes().is_some_and(|available| required > available) {
            let error = config.check_memory(0).unwrap_err();
            assert!(error.contains("host bytes") && error.contains("fewer rays"));
        }
        assert!(
            Config::default()
                .check_memory(u64::MAX)
                .unwrap_err()
                .contains("overflow")
        );
    }
}
