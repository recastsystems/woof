//! Native-column visibility shared by deterministic and ensemble import.
use crate::wrf_process::{
    WrfHourFields, WrfProcessOptions, WrfProductGroup, push_canonical_values,
};
use rustwx_calc::visibility::{VisibilityInputs, compute_visibility};
use rustwx_core::{CanonicalField, FieldSelector, GridProjection, LatLonGrid};
use wrf_core::WrfFile;
pub const PRODUCTS: &[(&str, CanonicalField, &str)] = &[
    ("visibility_surface", CanonicalField::Visibility, "m"),
    ("fog_flag_surface", CanonicalField::FogFlag, "1"),
];
pub fn selected(o: &WrfProcessOptions, name: &str) -> bool {
    o.should_process(name, Some(name), WrfProductGroup::Diagnostic)
}
pub(crate) fn push(
    fields: &mut WrfHourFields,
    file: &WrfFile,
    ti: usize,
    grid: &LatLonGrid,
    projection: Option<GridProjection>,
    o: &WrfProcessOptions,
    progress: &mut impl FnMut(String),
) {
    if !PRODUCTS.iter().any(|(name, _, _)| selected(o, name)) {
        return;
    }
    let result = (|| -> Result<_, String> {
        let read = |name: &str| {
            file.read_var(name, ti)
                .map(|v| v.to_vec())
                .map_err(|e| format!("{name}: {e}"))
        };
        let optional = |name: &str| {
            if file.has_var(name) {
                read(name).map(Some)
            } else {
                Ok(None)
            }
        };
        let p = file.full_pressure(ti).map_err(|e| e.to_string())?;
        let t = file.temperature(ti).map_err(|e| e.to_string())?;
        let q = read("QVAPOR")?;
        let c = read("QCLOUD")?;
        // Rain and snow are required, not optional: without them the extinction reads clear air through
        // precipitation. THE BREAKAGE THIS PREVENTS: a hex surface export carries QCLOUD and QICE (cloud cover)
        // but no rain or snow, and would draw a visibility map with every shower left out.
        let r = Some(read("QRAIN")?);
        let ice = optional("QICE")?;
        let s = Some(read("QSNOW")?);
        let g = optional("QGRAUP")?;
        compute_visibility(VisibilityInputs {
            grid: grid.shape,
            nz: file.nz,
            pressure_pa: &p,
            temperature_k: &t,
            vapor_kgkg: &q,
            cloud_kgkg: &c,
            rain_kgkg: r.as_deref(),
            ice_kgkg: ice.as_deref(),
            snow_kgkg: s.as_deref(),
            graupel_kgkg: g.as_deref(),
        })
        .map_err(|e| e.to_string())
    })();
    match result {
        Ok(out) => {
            for ((name, field, units), values) in
                PRODUCTS.iter().zip([out.visibility_m, out.fog_flag])
            {
                if selected(o, name) {
                    push_canonical_values(
                        fields,
                        grid,
                        projection.clone(),
                        name,
                        FieldSelector::surface(*field),
                        units,
                        values.into_iter().map(|v| v as f32).collect(),
                    );
                }
            }
            let missing = ["QRAIN", "QICE", "QSNOW", "QGRAUP"]
                .into_iter()
                .filter(|n| !file.has_var(n))
                .collect::<Vec<_>>();
            progress(format!(
                "visibility: Stoelinga-Warner + NOAA GSD/RUC RH; cap 90000 m; fog_flag is vis < 1000 m; absent species {missing:?}; aerosol/smoke/blowing snow not included"
            ));
        }
        Err(e) => progress(format!("visibility and fog_flag unavailable: {e}")),
    }
}
