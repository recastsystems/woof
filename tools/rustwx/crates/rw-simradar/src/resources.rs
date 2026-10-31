//! Allocation and sampling estimates for the same native scan that executes.
use crate::request::{Config, Fields, Request, Sites};
use bowecho_simradar::WrfFile;
use serde_json::{Value, json};

pub const SCHEMA: &str = "simulated-radar.resources/v1";

pub fn atmosphere_bytes(nx: usize, ny: usize, nz: usize) -> Result<u64, String> {
    (nx as u64).checked_mul(ny as u64)
        .and_then(|n| n.checked_mul(nz as u64))
        .and_then(|n| n.checked_mul(128))
        .ok_or_else(|| "atmosphere shape overflows memory estimate".into())
}

pub fn scan(config: &Config) -> Result<Value, String> {
    config.validate()?;
    let native = config.native();
    let rays = native.azimuth_count as u64;
    // The reference samples complete gates, with one gate for sub-gate ranges.
    let gates = (native.max_range_m / native.gate_spacing_m).floor().max(1.0) as u64;
    let cuts = native.physical_scan_legs().len() as u64;
    let samples = native.beam_integration.pulse_volume_sample_count() as u64;
    let polar = rays.checked_mul(gates).and_then(|n| n.checked_mul(cuts))
        .ok_or("radar sampling dimensions overflow the work estimate")?;
    let evaluations = polar.checked_mul(samples)
        .ok_or("radar pulse samples overflow the work estimate")?;
    let catalog: Vec<Value> = serde_json::from_str(include_str!("../data/sites.json"))
        .map_err(|error| error.to_string())?;
    Ok(json!({
        "rays_per_cut": rays, "gates_per_ray": gates, "physical_cuts": cuts,
        "polar_bins_per_site_volume": polar,
        "quadrature_points_per_bin": samples,
        "pulse_samples_per_site_volume_upper_bound": evaluations,
        "scan_working_bytes": config.scan_memory_bytes()?,
        "internal_moments_for_memory_bound": if config.wants_dual_pol() {12} else {2},
        "formats": config.formats, "fields": config.fields,
        "timing": config.timing, "sites_processed_concurrently": null,
        "site_concurrency_policy": "memory-admitted batches within the Rayon thread count; shared atmospheres and temporal read-ahead reserved",
        "site_count": match &config.sites { Sites::List(v) => Some(v.len()), _ => None },
        "auto_site_count_requires_domain_coverage": matches!(config.sites, Sites::Auto(_)),
        "catalog_site_count_upper_bound":match &config.sites {
            Sites::Auto(_)=>catalog.iter().filter(|s|s.get("automatic").and_then(Value::as_bool).unwrap_or(true)).count(),
            _=>catalog.len(),
        },
    }))
}

/// Bytes one format's file may take per polar bin and moment: Level II
/// stores 8- or 16-bit words, the other writers at most a 32-bit float.
fn word_bytes(format: &str) -> u64 {
    if format == "level2" { 2 } else { 4 }
}

/// Per-ray metadata bound (radial headers, ray variables), per file.
const RAY_METADATA_BYTES: u64 = 1024;
/// Per-file bound for headers, attributes, Level II metadata records and
/// HDF5 structure.
const FILE_OVERHEAD_BYTES: u64 = 4 * 1024 * 1024;

/// An upper bound on the bytes one site-volume leaves on disk: every
/// requested format at its widest gate word with 1% for compression framing
/// that cannot shrink the data, the PPI PNGs at their uncompressed RGBA
/// size, and each PPI's frame in its GIF loop at the 12-bit LZW worst case,
/// twice, because a loop is rewritten whole beside the one it replaces.
pub fn output_bound(config: &Config) -> Result<Value, String> {
    config.validate()?;
    let native = config.native();
    let rays = native.azimuth_count.max(1) as u64;
    let gates = (native.max_range_m / native.gate_spacing_m).floor().max(1.0) as u64;
    let cuts = native.physical_scan_legs().len() as u64;
    let moments = match &config.fields {
        Fields::List(v) => v.len() as u64,
        Fields::Auto(_) => 6,
    };
    let overflow = || "radar output dimensions overflow the disk estimate".to_string();
    let rays_written = rays.checked_mul(cuts).ok_or_else(overflow)?;
    let values = rays_written
        .checked_mul(gates)
        .and_then(|n| n.checked_mul(moments))
        .ok_or_else(overflow)?;
    let mut formats = serde_json::Map::new();
    let mut total: u64 = 0;
    for format in &config.formats {
        let bytes = values
            .checked_mul(word_bytes(format))
            .and_then(|n| n.checked_mul(101))
            .map(|n| n / 100)
            .and_then(|n| n.checked_add(rays_written.checked_mul(RAY_METADATA_BYTES)?))
            .and_then(|n| n.checked_add(FILE_OVERHEAD_BYTES))
            .ok_or_else(overflow)?;
        total = total.checked_add(bytes).ok_or_else(overflow)?;
        formats.insert(format.clone(), json!(bytes));
    }
    let mut tilts: Vec<f64> = config.elevations_deg.clone();
    tilts.dedup_by(|a, b| (*a - *b).abs() < 0.01);
    // Every PPI field a volume may carry: reflectivity, velocity and the
    // dual-pol moments drawn when the history's microphysics supplies them.
    let drawn = match &config.fields {
        Fields::List(v) => v.iter().filter(|f| crate::ppi::PPI_FIELDS.contains(&f.as_str())).count(),
        Fields::Auto(_) => crate::ppi::PPI_FIELDS.len(),
    };
    let images = (drawn * tilts.len().min(crate::PPI_TILTS)) as u64;
    let pixels = u64::from(crate::PPI_WIDTH) * u64::from(crate::PPI_HEIGHT);
    let png = (pixels * 4 + u64::from(crate::PPI_HEIGHT)) * 101 / 100 + 64 * 1024;
    let gif_frame = pixels * 3 / 2 * 256 / 255 + 2048;
    total = images
        .checked_mul(png + 2 * gif_frame)
        .and_then(|n| n.checked_add(total))
        .ok_or_else(overflow)?;
    Ok(json!({
        "output_bytes_per_site_volume_upper_bound": total,
        "format_bytes_upper_bound": formats,
        "ppi_images_per_site_volume": images,
        "png_bytes_upper_bound": png,
        "gif_frame_bytes_upper_bound": gif_frame,
        "basis": "widest gate word per format plus 1% framing, 1 KiB per ray and 4 MiB per file; uncompressed RGBA PNGs; 12-bit LZW GIF frames counted twice for loop rewrites",
    }))
}

/// Read only shapes; no atmosphere arrays, simulation, output locks or files.
pub fn estimate(request: &Request) -> Result<Value, String> {
    if request.schema != "simulated-radar.request/v1" {
        return Err("unsupported simulated radar request schema".into());
    }
    let geometry = scan(&request.config)?;
    let mut shapes = Vec::new();
    let mut largest = 0;
    for path in &request.history_paths {
        let header = WrfFile::open(path).map_err(|e| e.to_string())?;
        let bytes = atmosphere_bytes(header.nx, header.ny, header.nz)?.checked_mul(request.config.atmosphere_factor()).ok_or("radar band memory estimate overflow")?;
        largest = largest.max(bytes);
        shapes.push(json!({"path":path,"nx":header.nx,"ny":header.ny,"nz":header.nz,
                           "atmosphere_working_bytes":bytes}));
    }
    // A forecast door prices the grids it will write before any history
    // exists, with the same per-scene arithmetic the scan admits against.
    for [nx, ny, nz] in &request.scene_shapes {
        if *nx == 0 || *ny == 0 || *nz == 0 {
            return Err("scene_shapes entries need positive nx, ny and nz".into());
        }
        let bytes = atmosphere_bytes(*nx, *ny, *nz)?.checked_mul(request.config.atmosphere_factor()).ok_or("radar band memory estimate overflow")?;
        largest = largest.max(bytes);
        shapes.push(json!({"path":null,"nx":nx,"ny":ny,"nz":nz,
                           "atmosphere_working_bytes":bytes}));
    }
    let atmosphere_count = if request.config.timing == "scan" {2} else {1};
    let extra = largest.checked_mul(atmosphere_count).ok_or("radar memory estimate overflow")?;
    let required = request.config.scan_memory_bytes()?.checked_add(extra)
        .ok_or("radar memory estimate overflow")?;
    let available = rw_host_memory::available_bytes();
    let admission = request.config.check_memory_with_available(extra, available);
    let output = output_bound(&request.config)?;
    Ok(json!({"schema":SCHEMA, "geometry":geometry, "histories":shapes, "output":output,
        "atmospheres_retained_upper_bound":atmosphere_count,
        "estimated_peak_working_bytes":required, "available_host_bytes":available,
        "memory_admitted_now":admission.is_ok(), "memory_refusal":admission.err(),
        "input_fields_validated":false,
        "geometry_only":request.history_paths.is_empty() && request.scene_shapes.is_empty(),
        "work_multiplier":"sum of selected sites times committed volume times across domains",
        "work_scope":"pulse samples bound beam work, not dual-pol closure, history I/O or cumulative GIF encoding",
        "price_or_duration_guarantee":false,
        "measurement_reference":{
            "native_binary_sha256":"821ba1e703c63a640c07e26f0936f08cf1aca60e9f87461a26a9e40bdc1475f7",
            "grid":[1799,1059,50], "grid_spacing_m":3000, "sites":3, "volumes_per_site":1,
            "scan_strategy":"vcp212", "range_km":230, "gate_spacing_m":250,
            "azimuth_step_deg":1, "fields":["reflectivity","velocity"],
            "formats":["level2","cfradial1"], "cpu_threads":4,
            "elapsed_seconds":20.436, "peak_rss_kib":10196888,
            "includes":["input hashing","history read","simulation","radar writers","PPI","GIF"],
            "excludes":["forecast","upload","dual-pol timing qualification"],
            "use":"measured reference only; no automatic linear price extrapolation"
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quote_counts_match_reference_sampling_for_nonintegral_geometry() {
        let mut c = Config::default();
        c.scan_strategy = "custom".into(); c.elevations_deg = vec![0.5, 1.5];
        c.range_km = 0.62; c.gate_spacing_m = 250.0; c.azimuth_step_deg = 7.0;
        let estimate = scan(&c).unwrap();
        assert_eq!(estimate["rays_per_cut"], 51);
        assert_eq!(estimate["gates_per_ray"], 2);
        assert_eq!(estimate["physical_cuts"], 2);
        assert_eq!(estimate["pulse_samples_per_site_volume_upper_bound"], 1836);
    }
    #[test]
    fn subgate_range_still_budgets_one_gate() {
        let mut c = Config::default(); c.range_km = 0.001;
        assert_eq!(scan(&c).unwrap()["gates_per_ray"], 1);
        assert!(c.scan_memory_bytes().unwrap() > 128 * 1024 * 1024);
    }
    #[test]
    fn tiny_positive_geometry_is_refused_before_integer_cast_or_allocation() {
        let mut c = Config::default(); c.azimuth_step_deg = f64::MIN_POSITIVE;
        assert!(scan(&c).unwrap_err().contains("geometry"));
        c.azimuth_step_deg = 1.0; c.gate_spacing_m = f64::MIN_POSITIVE;
        assert!(scan(&c).unwrap_err().contains("whole metre"));
    }
    #[test]
    fn allocation_gate_is_deterministic_and_names_the_actual_budget() {
        let c = Config::default();
        let required = c.scan_memory_bytes().unwrap();
        assert!(c.check_memory_with_available(0, Some(required)).is_ok());
        assert!(c.check_memory_with_available(0, Some(required - 1)).unwrap_err().contains("host bytes"));
        assert!(c.check_memory_with_available(0, None).unwrap_err().contains("cannot determine"));
        assert!(c.check_memory_with_available(u64::MAX, Some(u64::MAX)).unwrap_err().contains("overflow"));
    }
    #[test]
    fn output_bound_counts_every_format_image_and_loop_frame() {
        let mut c = Config::default();
        c.scan_strategy = "custom".into(); c.elevations_deg = vec![0.5, 1.5];
        c.range_km = 10.0; c.gate_spacing_m = 1000.0; c.azimuth_step_deg = 10.0;
        c.fields = Fields::List(vec!["reflectivity".into()]);
        c.formats = vec!["level2".into(), "cfradial1".into()];
        let bound = output_bound(&c).unwrap();
        // 36 rays x 2 cuts x 10 gates x 1 moment.
        let values = 36 * 2 * 10;
        let level2 = values * 2 * 101 / 100 + 72 * RAY_METADATA_BYTES + FILE_OVERHEAD_BYTES;
        let cfradial = values * 4 * 101 / 100 + 72 * RAY_METADATA_BYTES + FILE_OVERHEAD_BYTES;
        assert_eq!(bound["format_bytes_upper_bound"]["level2"], level2);
        assert_eq!(bound["format_bytes_upper_bound"]["cfradial1"], cfradial);
        assert_eq!(bound["ppi_images_per_site_volume"], 2, "one field at the two lowest tilts");
        let images = 2 * (bound["png_bytes_upper_bound"].as_u64().unwrap()
            + 2 * bound["gif_frame_bytes_upper_bound"].as_u64().unwrap());
        assert_eq!(bound["output_bytes_per_site_volume_upper_bound"], level2 + cfradial + images);
        // Auto fields price every PPI field at both drawn tilts: reflectivity,
        // velocity, ZDR, CC and KDP.
        c.fields = Fields::Auto("auto".into());
        assert_eq!(output_bound(&c).unwrap()["ppi_images_per_site_volume"], 10);
        c.fields = Fields::List(vec!["zdr".into(), "phidp".into()]);
        assert_eq!(output_bound(&c).unwrap()["ppi_images_per_site_volume"], 2, "PHIDP has no PPI");
    }
    #[test]
    fn atmosphere_overflow_cannot_wrap_to_a_small_quote() {
        assert!(atmosphere_bytes(usize::MAX, usize::MAX, 50).is_err());
    }
}
