//! Product-dependent, lossless 2-D processing for the live viewer.
//!
//! The existing recipe selector catalog owns dependencies and the existing
//! store catalog owns availability. This module only connects those contracts
//! to the WRF processor's filter names; it implements no meteorology.
use std::{
    collections::{BTreeSet, HashSet},
    path::Path,
};

use rustwx_core::{CanonicalField as F, FieldSelector, VerticalSelector as V};
use rustwx_models::plot_recipe_store_requirements;
use rustwx_products::{derived::store_derived_recipe_slugs, windowed::HrrrWindowedProduct};
use rusty_weather::{batch_render::inspect_renderable_products, render_all::StoreFieldSource};
use serde::{Deserialize, Serialize};

use crate::wrf_process::WrfProcessOptions;

pub const PROFILE: &str = "viewer-2d-v1";
pub const DEFAULT_PRODUCTS: &[&str] = &[
    "composite_reflectivity",
    "1km_reflectivity",
    "2m_temperature",
    "2m_dewpoint",
    "2m_relative_humidity",
    "10m_wind_speed_and_direction",
    "mslp_10m_winds",
    "total_qpf",
    "precipitable_water",
    "850mb_temperature_height_winds",
    "850mb_height_winds",
    "700mb_rh_height_winds",
    "500mb_height_winds",
    "300mb_height_winds",
    "sbcape",
    "mlcape",
    "sbcin",
    "bulk_shear_0_6km",
    "srh_0_1km",
    "uh_2to5km",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProductAvailability {
    pub slug: String,
    pub available: bool,
    pub source_fields: Vec<String>,
    pub missing_reasons: Vec<String>,
}

#[derive(Debug, Clone)]
enum Requirement {
    Selector(FieldSelector),
    Derived(String),
    /// A stored 2-D field named by `var:<name>`: the renderer's generic
    /// lane draws any stored 2-D field under a request-safe name.
    Stored(String),
    Unsupported(String),
}

#[derive(Debug, Clone)]
pub struct ViewerProfile {
    pub products: Vec<String>,
    pub options: WrfProcessOptions,
    requirements: Vec<Vec<Requirement>>,
    windowed: HashSet<String>,
}

impl ViewerProfile {
    pub fn new(requested: &[String]) -> Result<Self, String> {
        if requested.len() > 128 {
            return Err("A viewer profile supports at most 128 named products".into());
        }
        let mut products: Vec<String> = if requested.is_empty() {
            DEFAULT_PRODUCTS
                .iter()
                .map(|slug| (*slug).to_owned())
                .collect()
        } else {
            // A stored field's name is spelled exactly as the store spells
            // it; only a catalog slug is case-folded.
            requested
                .iter()
                .map(|slug| {
                    let slug = slug.trim();
                    match slug.strip_prefix("var:") {
                        Some(name) => format!("var:{name}"),
                        None => slug.to_ascii_lowercase(),
                    }
                })
                .collect()
        };
        products.sort();
        products.dedup();
        let derived = store_derived_recipe_slugs();
        let mut requirements = Vec::new();
        let mut only = BTreeSet::new();
        let mut chart_selectors = Vec::new();
        let mut windowed = HashSet::new();
        let mut stored_fields = false;
        for slug in &products {
            if slug.is_empty() || slug.len() > 128 {
                return Err("Viewer product slugs must contain 1 to 128 characters".into());
            }
            let required = if let Some(name) = slug.strip_prefix("var:") {
                // The renderer's own parser decides what a `var:` request
                // may name, so the viewer takes exactly the stored-field
                // requests the renderer draws and refuses the spellings it
                // would refuse.
                let parsed = rusty_weather::render_all::partition_products(slug)
                    .map_err(|error| format!("Viewer product {slug:?}: {error}"))?;
                if parsed.generic.len() != 1 || parsed.generic[0] != name {
                    return Err(format!(
                        "Viewer product {slug:?} must name exactly one stored 2-D field"
                    ));
                }
                stored_fields = true;
                vec![Requirement::Stored(name.to_owned())]
            } else if let Ok(fields) = plot_recipe_store_requirements(slug) {
                fields
                    .into_iter()
                    .map(|field| match field.selector {
                        Some(selector) => Requirement::Selector(selector),
                        None => Requirement::Unsupported(format!(
                            "{} has no canonical store selector",
                            field.field_key
                        )),
                    })
                    .collect()
            } else if derived.iter().any(|known| *known == slug) {
                vec![Requirement::Derived(slug.clone())]
            } else if let Some(product) = HrrrWindowedProduct::from_slug(slug) {
                windowed.insert(slug.clone());
                product
                    .input_selectors()
                    .into_iter()
                    .map(Requirement::Selector)
                    .collect()
            } else {
                return Err(format!("Unknown or non-2-D viewer product {slug:?}"));
            };
            for requirement in &required {
                match requirement {
                    Requirement::Selector(selector) => {
                        only.insert(processing_name(*selector));
                        if matches!(selector.vertical, V::IsobaricHpa(_))
                            && !chart_selectors.contains(selector)
                        {
                            chart_selectors.push(*selector);
                        }
                    }
                    Requirement::Derived(name) => {
                        only.insert(name.clone());
                    }
                    Requirement::Stored(name) => {
                        only.insert(name.clone());
                        // A chart-level plane is built only for the
                        // selectors the viewer hands the processor.
                        if let Some(selector) =
                            crate::wrf_process::isobaric_recipe_selector_for_key(name)
                        {
                            if !chart_selectors.contains(&selector) {
                                chart_selectors.push(selector);
                            }
                        }
                    }
                    Requirement::Unsupported(_) => {}
                }
            }
            requirements.push(required);
        }
        // Keep the source terrain plane as a small independent map field, so
        // a valid frame can still publish its unavailable-product catalog.
        // An empty filter would mean "all" to the full-science processor.
        only.insert("orography".into());
        // A `var:` request may name a raw WRF field or one a user added to
        // their own Registry, which only the raw-extras and stored-plane
        // passes store. The `only` filter above keeps both passes to the
        // named fields, and a selection without `var:` pays for neither.
        let options = WrfProcessOptions {
            core_fields: true,
            diagnostics: true,
            heavy_ecape: false,
            raw_extras: stored_fields,
            stored_planes: stored_fields,
            only: only.into_iter().collect(),
            skip: Vec::new(),
            viewer_2d: true,
            chart_selectors,
            named_products_only: false,
            snow_since: None,
            required_fields: None,
        }
        .normalized();
        Ok(Self {
            products,
            options,
            requirements,
            windowed,
        })
    }

    pub fn availability(
        &self,
        root: &Path,
        model: &str,
        run: &str,
        slot: u16,
    ) -> Result<Vec<ProductAvailability>, String> {
        let catalog = inspect_renderable_products(root, model, run, slot)?;
        let store =
            StoreFieldSource::open(root, model, run, slot).map_err(|error| error.to_string())?;
        self.products
            .iter()
            .zip(&self.requirements)
            .map(|(slug, required)| {
                let mut source_fields = Vec::new();
                let mut missing_reasons = Vec::new();
                for requirement in required {
                    let resolved = match requirement {
                        Requirement::Selector(selector) => match store.resolve(selector) {
                            Some(name) => Some(name.to_owned()),
                            None => {
                                missing_reasons
                                    .push(format!("Missing stored selector {}", selector.key()));
                                None
                            }
                        },
                        Requirement::Derived(name) => {
                            if store.surface_variable(name).is_some() {
                                Some(name.clone())
                            } else {
                                missing_reasons.push(format!("Missing derived 2-D field {name}"));
                                None
                            }
                        }
                        Requirement::Stored(name) => {
                            if store.surface_variable(name).is_some() {
                                Some(name.clone())
                            } else {
                                missing_reasons.push(format!(
                                    "This frame stores no 2-D field named {name}"
                                ));
                                None
                            }
                        }
                        Requirement::Unsupported(reason) => {
                            missing_reasons.push(reason.clone());
                            None
                        }
                    };
                    if let Some(name) = resolved {
                        if !source_fields.contains(&name) {
                            source_fields.push(name);
                        }
                    }
                }
                if self.windowed.contains(slug) {
                    missing_reasons.push(
                        "Requires a completed, grid-compatible window of committed UTC timestamps"
                            .into(),
                    );
                }
                // The catalog leaves out `var:` spellings of a plane a named
                // product already draws, so an "all" request does not draw
                // one grid twice. An explicit `var:` request is drawn by
                // the generic lane whenever the field is stored.
                let listed = matches!(required.as_slice(), [Requirement::Stored(_)])
                    || catalog.products.iter().any(|product| product.slug == *slug);
                if !listed && missing_reasons.is_empty() {
                    missing_reasons.push(
                        "The production renderer has no available recipe for these stored fields"
                            .into(),
                    );
                }
                Ok(ProductAvailability {
                    slug: slug.clone(),
                    available: listed && missing_reasons.is_empty(),
                    source_fields,
                    missing_reasons,
                })
            })
            .collect()
    }
}

/// Existing WRF processing names for canonical surface selector addresses.
/// Isobaric planes already use selector.key() in the shared processor.
fn processing_name(selector: FieldSelector) -> String {
    if !selector.product.is_default() {
        return selector.key();
    }
    if let Some(row) = crate::wrf_process::CHEM_CORE_FIELD_CATALOG.iter().find(|row| row.selector == selector) {
        return row.store_name.to_string();
    }
    match (selector.field, selector.vertical) {
        (F::GeopotentialHeight, V::Surface) => "orography".into(),
        (F::Temperature, V::HeightAboveGroundMeters(2)) => "temperature_2m".into(),
        (F::Dewpoint, V::HeightAboveGroundMeters(2)) => "dewpoint_2m".into(),
        (F::RelativeHumidity, V::HeightAboveGroundMeters(2)) => "relative_humidity_2m".into(),
        (F::UWind, V::HeightAboveGroundMeters(10)) => "u_10m".into(),
        (F::VWind, V::HeightAboveGroundMeters(10)) => "v_10m".into(),
        (F::WindSpeed, V::HeightAboveGroundMeters(10)) => "wind_speed_10m".into(),
        (F::PressureReducedToMeanSeaLevel, V::MeanSeaLevel) => "mslp".into(),
        (F::PrecipitableWater, V::EntireAtmosphere) => "pwat".into(),
        (F::CompositeReflectivity, V::EntireAtmosphere) => "composite_reflectivity".into(),
        (F::RadarReflectivity, V::HeightAboveGroundMeters(1000)) => "reflectivity_1km".into(),
        (
            F::UpdraftHelicity,
            V::HeightAboveGroundLayerMeters {
                bottom_m: 2000,
                top_m: 5000,
            },
        ) => "updraft_helicity_2to5km".into(),
        (F::UpdraftHelicity, V::HeightAboveGroundLayerMeters { bottom_m: 0, top_m: 1000 }) => {
            "updraft_helicity_0to1km".into()
        }
        (F::UpdraftHelicity, V::HeightAboveGroundLayerMeters { bottom_m: 0, top_m: 2000 }) => {
            "updraft_helicity_0to2km".into()
        }
        (F::UpdraftHelicity, V::HeightAboveGroundLayerMeters { bottom_m: 0, top_m: 3000 }) => {
            "updraft_helicity_0to3km".into()
        }
        (F::TotalPrecipitation, V::Surface) => "apcp".into(),
        _ => selector.key(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_viewer_profile_is_twenty_products_without_volumes_or_raw_extras() {
        let profile = ViewerProfile::new(&[]).unwrap();
        assert_eq!(profile.products.len(), 20);
        assert!(profile.options.viewer_2d);
        assert!(
            !profile.options.raw_extras
                && !profile.options.stored_planes
                && !profile.options.heavy_ecape
        );
        assert_eq!(profile.options.chart_selectors.len(), 14);
        assert!(profile.options.only.contains(&"sbcape".into()));
        assert!(
            !profile
                .options
                .only
                .iter()
                .any(|name| name.ends_with("_iso"))
        );
    }
    #[test]
    fn wind_recipe_keeps_components_and_fill_while_scalar_selection_stays_scalar() {
        let wind = ViewerProfile::new(&["10m_wind_speed_and_direction".into()]).unwrap();
        assert_eq!(
            wind.options.only,
            ["orography", "u_10m", "v_10m", "wind_speed_10m"]
        );
        assert!(wind.options.chart_selectors.is_empty());
        let temp = ViewerProfile::new(&["2m_temperature".into()]).unwrap();
        assert_eq!(temp.options.only, ["orography", "temperature_2m"]);
    }
    #[test]
    fn windows_keep_source_planes_without_inventing_hour_slots() {
        let profile = ViewerProfile::new(&["qpf_1h".into()]).unwrap();
        assert_eq!(profile.options.only, ["apcp", "orography"]);
        assert!(profile.windowed.contains("qpf_1h"));
        assert!(ViewerProfile::new(&["not_a_product".into()]).is_err());
    }
    #[test]
    fn a_stored_field_selection_is_accepted_and_processes_only_that_field() {
        let profile = ViewerProfile::new(&["var:temperature_2m".into()]).unwrap();
        assert_eq!(profile.products, ["var:temperature_2m"]);
        assert_eq!(profile.options.only, ["orography", "temperature_2m"]);
        assert!(profile.options.viewer_2d);
        assert!(profile.options.raw_extras && profile.options.stored_planes);
        assert!(profile.options.chart_selectors.is_empty());
        let chart = ViewerProfile::new(&["var:geopotential_height_850hpa".into()]).unwrap();
        assert_eq!(
            chart.options.chart_selectors,
            [FieldSelector::isobaric(F::GeopotentialHeight, 850)]
        );
        assert_eq!(
            chart.options.only,
            ["geopotential_height_850hpa", "orography"]
        );
    }
    #[test]
    fn a_stored_field_selection_keeps_the_renderers_spelling_rules() {
        for slug in ["var:", "var:temperature_2m,dewpoint_2m", "var: temperature_2m"] {
            assert!(ViewerProfile::new(&[slug.into()]).is_err(), "{slug}");
        }
    }
}
