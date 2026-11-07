//! Rust stream-4 producer shared with the native NetCDF bridge.
pub fn run(
    inputs: &[std::path::PathBuf],
    output: &std::path::Path,
    surface_physics: i32,
) -> Result<serde_json::Value, crate::RwWpsError> {
    wrf_surface::run_with_surface_physics(inputs, output, surface_physics)
        .map_err(crate::RwWpsError::Output)
}
