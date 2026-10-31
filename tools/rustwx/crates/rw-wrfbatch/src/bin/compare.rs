//! `rw_compare` -- a run's products beside a reference model's own fields,
//! one sheet per product and valid time.
//!
//! `rw_wrfbatch` draws one run.  A reader who wants to know how that run
//! sits against the operational model it was configured after had two
//! pictures from two tools to line up by eye, in two colour tables.  This
//! binary is the shape that question needs: the run's history frame is
//! imported through the same hardened wrfout lane, the reference model's
//! OWN message for the same cycle and lead is decoded from its GRIB2 file
//! (fetched from the public bucket when it is not already on disk), and
//! both planes go through the production render path on ONE grid, in ONE
//! projection and extent, on ONE colour scale -- the run's own operational
//! ladder for that product.  A third panel, run minus reference, is drawn
//! for the continuous fields on a diverging ladder centred on zero.
//!
//! Product against product: the run's plane is whatever its store carries
//! under the product's selector (the same plane `rw_wrfbatch` draws), and
//! the reference's plane is the message the reference centre published
//! under that name -- its composite reflectivity message, its 2 m dewpoint
//! message -- never a quantity recomputed here from other reference
//! fields.  The one derived reference plane is 10 m wind speed, which the
//! reference publishes as two components; speed is the same number in
//! grid-relative and earth-relative components, so nothing is rotated.
//!
//! Isobaric height is product against product too: the store reads it
//! between the model's layer interfaces (`rw_isobaric`), as a reference's
//! post-processor does, so the difference panel shows the forecast and not
//! a pairing of layer-mean heights with mass-level pressures.
//!
//! The grids are compared, not assumed: `compare::match_grids` reads off
//! whether the run is a window of the reference grid (point against the
//! same point) or has to be sampled at the nearest point, and the sheet's
//! header says which.
//!
//! ```text
//! rw_compare --store-root DIR --out-dir DIR --reference NAME[,NAME...]
//!            [--products LIST|all] [--difference auto|on|off]
//!            [--reference-file FILE.grib2 | --reference-dir DIR]
//!            [--reference-cache DIR] [--offline] [--cycle YYYYMMDDHH]
//!            [--layout nested|flat] [--flat-dir DIR]
//!            [--width N] [--height N] [--source-label TEXT] [--run-label TEXT]
//!            [--stations FILE.json] [--station-mode observed|error]
//!            [--observations FILE.json]
//!            [--context WRFOUT ...] WRFOUT [WRFOUT ...]
//! rw_compare --fetch-reference --store-root DIR --reference NAME
//!            --cycle YYYYMMDDHH --forecast-hour N
//! rw_compare --store-root DIR --out-dir DIR [--reference NAME[,NAME...]]
//!            --panel "LABEL=WRFOUT|mean:LIST|pmm:LIST|max:LIST" ...
//!            [--panel "LABEL=frames:ROOT@VALID|ttenref:WINDOW" ...]
//!            [--panel-grid WRFOUT] [--row --panel ...]... [--order TOKEN,...]
//!            [--sheet-name STEM]
//! rw_compare --mode increment|spread|stamp|omb ...   (see da_sheet.rs)
//! rw_compare --list-products | --help | --abi
//! ```
//!
//! The `--panel` form is the N-panel sheet (`compare_panels.rs`): any number
//! of labelled model panels beside the references, in the `--order` the
//! caller gives.  The `--mode` forms are the data-assimilation sheets
//! (`rw_wrfbatch::da_sheet`): analysis increments, ensemble spread, postage
//! stamps with paintball, and observation-minus-background dots.
//!
//! A station file is the native decoded surface-observation record. An
//! observation manifest is an array of rows `{quantity,label,path,grid_path}`.
//! Pack paths resolve relative to the manifest. Quantity and valid-time metadata
//! select the observed panel, which uses the forecasts' own colour ladder.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{DateTime, Datelike, Timelike, Utc};
use grib_core::grib2::Grib2File;
use rustwx_core::{
    CanonicalField, CycleSpec, FieldSelector, GridProjection, ModelId, ModelRunRequest, SourceId,
};
use rustwx_io::FetchRequest;
use rustwx_products::viewer::StoreVariableStyle;
use rustwx_render::{LegendControls, PngCompressionMode, PngWriteOptions};
use rusty_weather::render_all::StoreFieldSource;
use rw_wrfbatch::compare::{
    GridMatch, MatchRule, SheetHeader, compose_sheet, difference, difference_above_floor,
    difference_scale, difference_stats, difference_step, ladder_scale, match_grids, sample,
    wind_speed,
};
use rw_wrfbatch::panel::{PanelRequest, layout_path, render_panel_with_unobserved, safe_component};
use rw_wrfbatch::annotate::MapOverlays;
use rw_wrfbatch::coverage::{self, CoverageDraw, NO_COVERAGE_LEGEND};
use rw_wrfbatch::station_overlay::{StationDot, scalar_dots, append_error_key};
use rw_wrfbatch::verification_io::{GridData, RadarSpec, load_radar, load_station_observations};
use rw_wrfbatch::wrf_process::{WrfProcessMessage, WrfProcessOptions, spawn_process_paths};
use sha2::{Digest, Sha256};

/// Embed the build revision for release artifact verification.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

#[path = "../comparison_observations.rs"]
mod comparison_observations;

#[path = "../compare_panels.rs"]
mod compare_panels;

/// The `--abi` contract line: the vocabulary the PYTHON half parses.
use crate::cli_contracts::COMPARE as ABI_MARKER;

/// One reference model: where its files are and what they are called.
///
/// A row, not a code path.  Adding a reference is adding a row here (its
/// model identity, the file product that carries the surface fields, the
/// public source, how its files are named); nothing below this table names
/// a model.
#[derive(Clone)]
struct ReferenceSpec {
    name: &'static str,
    label: &'static str,
    model: ModelId,
    /// The model's file product that carries every row of [`product_specs`].
    file_product: &'static str,
    /// The public sources, tried in order: the first that serves the file
    /// wins.  A source with short retention (NOMADS keeps about two days)
    /// sits behind the unbounded archive, so a past cycle still resolves.
    sources: &'static [SourceId],
    /// What the panel's provenance stamp says.
    source_label: &'static str,
    /// `(cycle hour, forecast hour) -> file names`, as the centre publishes
    /// them: every file product that carries the whole product table, the
    /// one that is fetched first.  A directory of files already on disk
    /// (a campaign's own downloads) is searched for any of them.
    file_names: fn(u8, u16) -> Vec<String>,
    /// `YYYYMMDD -> directory` under a mirror of the public bucket.
    bucket_directory: fn(&str) -> String,
    /// The last forecast hour the cycle starting at this hour publishes.
    horizon: fn(u8) -> u16,
    /// An isobaric product can live in another published file family.
    pressure_file_product: &'static str,
    pressure_file_names: fn(u8, u16) -> Vec<String>,
}

fn reference_specs() -> Vec<ReferenceSpec> {
    vec![
        ReferenceSpec {
            name: "hrrr",
            label: "HRRR",
            model: ModelId::Hrrr,
            file_product: "sfc",
            sources: &[SourceId::Aws],
            source_label: "NOAA HRRR, noaa-hrrr-bdp-pds",
            file_names: |cycle_hour, lead| {
                ["wrfsfc", "wrfprs"]
                    .iter()
                    .map(|product| format!("hrrr.t{cycle_hour:02}z.{product}f{lead:02}.grib2"))
                    .collect()
            },
            bucket_directory: |date| format!("hrrr.{date}/conus"),
            // Forty-eight hours from the six-hourly cycles, eighteen from the
            // hours between them.
            horizon: |cycle_hour| if cycle_hour % 6 == 0 { 48 } else { 18 },
            pressure_file_product: "prs",
            pressure_file_names: |hour, lead| {
                vec![format!("hrrr.t{hour:02}z.wrfprsf{lead:02}.grib2")]
            },
        },
        ReferenceSpec {
            name: "rrfs",
            label: "RRFS",
            model: ModelId::Rrfs,
            file_product: "2dfld-conus",
            // The operational AWS bucket (the one the init fetch reads) is
            // the unbounded archive; NOMADS publishes first but keeps about
            // two days, so it is the second rung.
            sources: &[SourceId::Aws, SourceId::Nomads],
            source_label: "NOAA RRFS v1.0",
            file_names: |hour, lead| {
                vec![format!("rrfs.t{hour:02}z.2dfld.3km.f{lead:03}.conus.grib2")]
            },
            bucket_directory: |date| format!("rrfs.{date}"),
            horizon: |hour| if hour % 6 == 0 { 84 } else { 18 },
            pressure_file_product: "prs-conus",
            pressure_file_names: |hour, lead| {
                vec![format!(
                    "rrfs.t{hour:02}z.prslev.3km.f{lead:03}.conus.grib2"
                )]
            },
        },
    ]
}

fn reference_names(text: &str) -> Result<Vec<String>, String> {
    let observations = comparison_observations::specifications()?;
    let forecasts = reference_specs();
    let known: Vec<&str> = forecasts
        .iter()
        .map(|spec| spec.name)
        .chain(observations.iter().map(|spec| spec.name.as_str()))
        .collect();
    let mut selected = Vec::new();
    for name in text.split(',').map(str::trim) {
        if !known.contains(&name) {
            return Err(format!(
                "unknown --reference {name:?}; this build knows: {}",
                known.join(", ")
            ));
        }
        if !selected.iter().any(|have| have == name) {
            selected.push(name.to_string());
        }
    }
    if selected.is_empty() {
        return Err("--reference named nothing".into());
    }
    Ok(selected)
}

/// How the run's plane is read from its store.
#[derive(Clone, Copy)]
enum RunPlane {
    /// The stored plane under this selector, as `rw_wrfbatch` draws it.
    Selector(FieldSelector),
    /// A stored raw diagnostic with its file-declared units.
    Named { raw: &'static str, variable: &'static str },
    /// The run total under this selector, differenced over the hour that
    /// ends at the frame.
    HourlyAccumulation(FieldSelector),
    /// Speed from two stored component planes.
    SelectorSpeed(FieldSelector, FieldSelector),
}

/// Which of the reference's own messages make its plane.
#[derive(Clone, Copy)]
enum ReferencePlane {
    Selector(FieldSelector),
    /// Speed from the two published components.
    Speed(FieldSelector, FieldSelector),
    /// The reference's own one-hour accumulation message ending at the lead.
    HourlyAccumulation(FieldSelector),
    /// One raw message chosen by its GRIB2 keys.
    Raw(RawKey),
    /// Speed from two raw messages.
    RawSpeed(RawKey, RawKey),
}

/// A raw GRIB2 message key with the conversion into the run's units.
#[derive(Clone, Copy, Debug)]
struct RawKey {
    discipline: u8,
    category: u8,
    number: u8,
    level_type: u8,
    level_value: f64,
    second_type: u8,
    second_value: f64,
    /// None: instantaneous; Some(1): accumulation from the start; Some(2): max over the last hour.
    stat: Option<u8>,
    units: &'static str,
    factor: f32,
    offset: f32,
}

fn raw_message<'a>(grib: &'a Grib2File, key: &RawKey, lead: u16) -> Option<&'a grib_core::grib2::Grib2Message> {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-3 * a.abs().max(1.0);
    grib.messages.iter().find(|m| {
        let p = &m.product;
        if m.discipline != key.discipline || p.parameter_category != key.category || p.parameter_number != key.number {
            return false;
        }
        if p.level_type != key.level_type || !close(p.level_value, key.level_value) {
            return false;
        }
        if key.second_type != 255 && (p.second_level_type != key.second_type || !close(p.second_level_value, key.second_value)) {
            return false;
        }
        match key.stat {
            None => p.statistical_process_type.is_none(),
            Some(1) => p.statistical_process_type == Some(1)
                && message_hours(p.time_range_unit, p.forecast_time) == Some(0)
                && p.end_of_interval == Some(m.reference_time + chrono::Duration::hours(i64::from(lead))),
            Some(stat) => p.statistical_process_type == Some(stat)
                && p.end_of_interval == Some(m.reference_time + chrono::Duration::hours(i64::from(lead)))
                && (lead == 0 || message_hours(p.time_range_unit, p.forecast_time) == Some(u32::from(lead) - 1)),
        }
    })
}

fn raw_field(grib: &Grib2File, key: &RawKey, lead: u16) -> Result<Option<ReferenceField>, String> {
    let Some(message) = raw_message(grib, key, lead) else { return Ok(None) };
    let field = rustwx_io::extract_message_field(message, key.units)
        .map_err(|error| format!("decode raw {}.{}: {error}", key.category, key.number))?;
    Ok(Some(ReferenceField {
        units: key.units.to_string(),
        ny: field.grid.shape.ny,
        nx: field.grid.shape.nx,
        lat: field.grid.lat_deg,
        lon: field.grid.lon_deg,
        values: field.values.iter().map(|v| if v.is_finite() { v * key.factor + key.offset } else { *v }).collect(),
    }))
}

struct ProductSpec {
    name: &'static str,
    slug: &'static str,
    title: &'static str,
    run: RunPlane,
    reference: ReferencePlane,
    /// What the reference panel's subtitle calls the message(s) it drew.
    /// Kept short: it shares one subtitle slot with the panel's start,
    /// lead and file name, and the renderer cuts what does not fit.
    reference_message: &'static str,
    /// NCEP inventory patterns (`VAR:level`), for an indexed subset fetch.
    inventory_patterns: &'static [&'static str],
    /// A continuous field: the difference panel is drawn under `auto`.
    continuous: bool,
    /// The store variable whose production style the panels wear, when it
    /// is not the run plane's own variable.
    style_variable: Option<&'static str>,
    /// The filled ladder both panels wear when the production catalogue has
    /// no fill for this field at all (it only ever contours it).  Never
    /// consulted when a production style resolves.
    ladder: Option<Ladder>,
    /// A fixed physical range on the existing neutral generic ramp, for
    /// a field with no production palette. Both field panels share it.
    neutral_range: Option<(f32, f32)>,
    /// The sheet's own colour levels, replacing the production style's
    /// levels (its units, conversion and palette stay) where the production
    /// scale spans a range one valid time never reaches, so the values a
    /// comparison is about would sit in one sliver of the colour bar.
    sheet_scale: Option<fn() -> rustwx_render::DiscreteColorScale>,
    station_quantity: Option<&'static str>,
    observation_quantity: Option<&'static str>,
}

/// A fixed filled ladder: `bins` steps of `step` from `first`, in the units
/// the field is drawn in.
#[derive(Clone, Copy)]
struct Ladder {
    first: f64,
    step: f64,
    bins: usize,
    units: &'static str,
    /// Stored units to drawn units.
    stored_to_drawn: f32,
    palette: rustwx_render::WeatherPalette,
    /// Labelled contours every this many drawn units, on both field
    /// panels.  A ladder wide enough for a continent is one colour across
    /// a regional window; the contours are what a small domain is read by.
    contour_interval: f64,
}

/// 1 h precipitation on a comparison sheet, in inches.
///
/// The shared production QPF scale runs 0.01 to 15 in, so on its linear
/// colour bar (whole-inch ticks) every hourly amount from 0.1 to 1 in sat
/// in the bottom ~6 % and the panels could not be read against each other.
/// One hour's precipitation tops out near 3 in; above that the bar extends.
/// The production scale (rustwx-products qpf.rs) is left as it is.
fn comparison_qpf_1h_scale() -> rustwx_render::DiscreteColorScale {
    rustwx_render::weather::palette_scale(
        rustwx_render::WeatherPalette::Precip,
        vec![
            0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50,
            1.75, 2.00, 2.50, 3.00,
        ],
        rustwx_render::ExtendMode::Max,
        Some(0.01),
    )
}

fn product_specs() -> Vec<ProductSpec> {
    let refc = FieldSelector::entire_atmosphere(CanonicalField::CompositeReflectivity);
    let t2m = FieldSelector::height_agl(CanonicalField::Temperature, 2);
    let td2m = FieldSelector::height_agl(CanonicalField::Dewpoint, 2);
    let apcp = FieldSelector::surface(CanonicalField::TotalPrecipitation);
    let hgt500 = FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 500);
    let swdown = FieldSelector::surface(CanonicalField::DownwardShortwaveRadiationFlux);
    let smoke_sfc = FieldSelector::height_agl(CanonicalField::SmokeMassDensity, 8);
    let smoke_column = FieldSelector::entire_atmosphere(CanonicalField::ColumnIntegratedSmoke);
    vec![
        ProductSpec {
            name: "refc",
            station_quantity: None,
            observation_quantity: Some("composite_reflectivity"),
            slug: "composite_reflectivity",
            title: "Composite reflectivity",
            run: RunPlane::Selector(refc),
            reference: ReferencePlane::Selector(refc),
            reference_message: "REFC, entire atmosphere",
            inventory_patterns: &["REFC:entire atmosphere"],
            continuous: false,
            style_variable: None,
            ladder: None,
            neutral_range: None,
            sheet_scale: None,
        },
        ProductSpec {
            name: "t2m",
            station_quantity: Some("temperature_2m"),
            observation_quantity: None,
            slug: "2m_temperature",
            title: "2 m temperature",
            run: RunPlane::Selector(t2m),
            reference: ReferencePlane::Selector(t2m),
            reference_message: "TMP, 2 m above ground",
            inventory_patterns: &["TMP:2 m above ground"],
            continuous: true,
            style_variable: None,
            ladder: None,
            neutral_range: None,
            sheet_scale: None,
        },
        ProductSpec {
            name: "td2m",
            station_quantity: Some("dewpoint_2m"),
            observation_quantity: None,
            slug: "2m_dewpoint",
            title: "2 m dewpoint",
            run: RunPlane::Selector(td2m),
            reference: ReferencePlane::Selector(td2m),
            reference_message: "DPT, 2 m above ground",
            inventory_patterns: &["DPT:2 m above ground"],
            continuous: true,
            style_variable: None,
            ladder: None,
            neutral_range: None,
            sheet_scale: None,
        },
        ProductSpec {
            name: "wspd10",
            station_quantity: Some("wind_speed_10m"),
            observation_quantity: None,
            slug: "10m_wind_speed",
            title: "10 m wind speed",
            run: RunPlane::Selector(FieldSelector::height_agl(CanonicalField::WindSpeed, 10)),
            reference: ReferencePlane::Speed(
                FieldSelector::height_agl(CanonicalField::UWind, 10),
                FieldSelector::height_agl(CanonicalField::VWind, 10),
            ),
            reference_message: "UGRD, VGRD 10 m, speed",
            inventory_patterns: &["UGRD:10 m above ground", "VGRD:10 m above ground"],
            continuous: true,
            style_variable: None,
            ladder: None,
            neutral_range: None,
            sheet_scale: None,
        },
        ProductSpec { name: "snow10", slug: "snow_10to1_window", title: "10:1 snowfall, saved intervals",
            station_quantity: None, observation_quantity: Some("snowfall_accumulation"),
            run: RunPlane::Selector(FieldSelector::surface(CanonicalField::Snowfall10to1)),
            reference: ReferencePlane::Selector(FieldSelector::surface(CanonicalField::Snowfall10to1)),
            reference_message: "ASNOW, surface", inventory_patterns: &["ASNOW:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "snowkuchera", slug: "snow_kuchera_window", title: "Kuchera snowfall, saved intervals",
            station_quantity: None, observation_quantity: Some("snowfall_accumulation"),
            run: RunPlane::Selector(FieldSelector::surface(CanonicalField::SnowfallKuchera)),
            reference: ReferencePlane::Selector(FieldSelector::surface(CanonicalField::SnowfallKuchera)),
            reference_message: "ASNOW, surface", inventory_patterns: &["ASNOW:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "snow_depth", slug: "snow_depth_total", title: "Standing snow depth",
            station_quantity: None, observation_quantity: Some("snow_depth"),
            run: RunPlane::Selector(FieldSelector::surface(CanonicalField::SnowDepth)),
            reference: ReferencePlane::Selector(FieldSelector::surface(CanonicalField::SnowDepth)),
            reference_message: "SNOD, surface", inventory_patterns: &["SNOD:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec {
            name: "qpf1h",
            station_quantity: None,
            observation_quantity: Some("precipitation_accumulation"),
            slug: "1h_precipitation",
            title: "1 h precipitation",
            run: RunPlane::HourlyAccumulation(apcp),
            reference: ReferencePlane::HourlyAccumulation(apcp),
            reference_message: "APCP, 1 h accumulation",
            inventory_patterns: &["APCP:surface"],
            continuous: false,
            style_variable: Some("apcp_1h"),
            ladder: None,
            neutral_range: None,
            sheet_scale: Some(comparison_qpf_1h_scale),
        },
        ProductSpec {
            name: "hgt500",
            station_quantity: None,
            observation_quantity: None,
            slug: "500mb_height",
            title: "500 hPa height",
            run: RunPlane::Selector(hgt500),
            reference: ReferencePlane::Selector(hgt500),
            reference_message: "HGT, 500 mb",
            inventory_patterns: &["HGT:500 mb"],
            continuous: true,
            style_variable: None,
            // 480 to 606 dam on the 6 dam interval the height is contoured
            // at operationally, so a colour step is a contour.
            ladder: Some(Ladder {
                first: 480.0,
                step: 6.0,
                bins: 21,
                units: "dam",
                stored_to_drawn: 0.1,
                palette: rustwx_render::WeatherPalette::IsothermHeight,
                contour_interval: 3.0,
            }),
            neutral_range: None,
            sheet_scale: None,
        },
        ProductSpec {
            name: "swdown",
            station_quantity: None,
            observation_quantity: None,
            slug: "surface_downward_shortwave",
            title: "Surface downward shortwave",
            run: RunPlane::Named { raw: "SWDOWN", variable: "wrf_swdown" },
            reference: ReferencePlane::Selector(swdown),
            reference_message: "DSWRF, surface, instant",
            inventory_patterns: &["DSWRF:surface"],
            continuous: true,
            style_variable: None,
            ladder: None,
            neutral_range: Some((0.0, 1200.0)),
            sheet_scale: None,
        },
        // Smoke, product against product: the run's SMOKE_SFC / SMOKE_COLUMN
        // planes are stored under the same selectors HRRR-Smoke publishes
        // (wrf_process::CHEM_CORE_FIELD_CATALOG writes them in kg m-3 and
        // kg m-2), so both panels wear the store's own smoke styles and a
        // reference without the message (RRFS 2dfld) is SKIPPED, not
        // substituted.
        ProductSpec {
            name: "smoke_sfc",
            station_quantity: None,
            observation_quantity: None,
            slug: "smoke_near_surface",
            title: "Near-surface smoke",
            run: RunPlane::Selector(smoke_sfc),
            reference: ReferencePlane::Selector(smoke_sfc),
            reference_message: "MASSDEN, 8 m",
            inventory_patterns: &["MASSDEN:8 m above ground"],
            continuous: true,
            style_variable: Some("smoke_near_surface"),
            ladder: None,
            neutral_range: None,
            sheet_scale: None,
        },
        ProductSpec {
            name: "smoke_column",
            station_quantity: None,
            observation_quantity: None,
            slug: "smoke_column",
            title: "Column smoke",
            run: RunPlane::Selector(smoke_column),
            reference: ReferencePlane::Selector(smoke_column),
            reference_message: "COLMD, entire atmosphere",
            inventory_patterns: &["COLMD:entire atmosphere"],
            continuous: true,
            style_variable: Some("smoke_column"),
            ladder: None,
            neutral_range: None,
            sheet_scale: None,
        },
        ProductSpec { name: "200mb_height", slug: "200mb_height", title: "200 hPa height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 200)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 200)),
            reference_message: "HGT, 200 mb", inventory_patterns: &["HGT:200 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 1080.0, step: 6.0, bins: 30, units: "dam", stored_to_drawn: 0.1, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 6.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "250mb_height", slug: "250mb_height", title: "250 hPa height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 250)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 250)),
            reference_message: "HGT, 250 mb", inventory_patterns: &["HGT:250 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 960.0, step: 6.0, bins: 26, units: "dam", stored_to_drawn: 0.1, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 6.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "300mb_height", slug: "300mb_height", title: "300 hPa height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 300)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 300)),
            reference_message: "HGT, 300 mb", inventory_patterns: &["HGT:300 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 840.0, step: 6.0, bins: 28, units: "dam", stored_to_drawn: 0.1, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 6.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "700mb_height", slug: "700mb_height", title: "700 hPa height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 700)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 700)),
            reference_message: "HGT, 700 mb", inventory_patterns: &["HGT:700 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 270.0, step: 3.0, bins: 22, units: "dam", stored_to_drawn: 0.1, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 3.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "850mb_height", slug: "850mb_height", title: "850 hPa height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 850)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 850)),
            reference_message: "HGT, 850 mb", inventory_patterns: &["HGT:850 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 120.0, step: 3.0, bins: 17, units: "dam", stored_to_drawn: 0.1, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 3.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "925mb_height", slug: "925mb_height", title: "925 hPa height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 925)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::GeopotentialHeight, 925)),
            reference_message: "HGT, 925 mb", inventory_patterns: &["HGT:925 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 45.0, step: 3.0, bins: 22, units: "dam", stored_to_drawn: 0.1, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 3.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "200mb_temperature", slug: "200mb_temperature", title: "200 hPa temperature",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 200)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 200)),
            reference_message: "TMP, 200 mb", inventory_patterns: &["TMP:200 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 200.0, step: 1.0, bins: 40, units: "K", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Temperature, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "250mb_temperature", slug: "250mb_temperature", title: "250 hPa temperature",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 250)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 250)),
            reference_message: "TMP, 250 mb", inventory_patterns: &["TMP:250 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 205.0, step: 1.0, bins: 40, units: "K", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Temperature, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "300mb_temperature", slug: "300mb_temperature", title: "300 hPa temperature",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 300)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 300)),
            reference_message: "TMP, 300 mb", inventory_patterns: &["TMP:300 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 213.0, step: 1.0, bins: 40, units: "K", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Temperature, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "500mb_temperature", slug: "500mb_temperature", title: "500 hPa temperature",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 500)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 500)),
            reference_message: "TMP, 500 mb", inventory_patterns: &["TMP:500 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 233.0, step: 1.0, bins: 40, units: "K", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Temperature, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "700mb_temperature", slug: "700mb_temperature", title: "700 hPa temperature",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 700)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 700)),
            reference_message: "TMP, 700 mb", inventory_patterns: &["TMP:700 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 250.0, step: 1.0, bins: 50, units: "K", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Temperature, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "850mb_temperature", slug: "850mb_temperature", title: "850 hPa temperature",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 850)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Temperature, 850)),
            reference_message: "TMP, 850 mb", inventory_patterns: &["TMP:850 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 255.0, step: 1.0, bins: 50, units: "K", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Temperature, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "200mb_rh", slug: "200mb_rh", title: "200 hPa relative humidity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 200)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 200)),
            reference_message: "RH, 200 mb", inventory_patterns: &["RH:200 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 20, units: "%", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Rh, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "200mb_absolute_vorticity", slug: "200mb_absolute_vorticity", title: "200 hPa absolute vorticity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, 200)), reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 2, number: 10, level_type: 100, level_value: 20000.0, second_type: 255, second_value: 0.0, stat: None, units: "s-1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "ABSV, 200 mb", inventory_patterns: &["ABSV:200 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: -10.0, step: 2.0, bins: 30, units: "1e-5 s-1", stored_to_drawn: 100000.0, palette: rustwx_render::WeatherPalette::RelVort, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "300mb_rh", slug: "300mb_rh", title: "300 hPa relative humidity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 300)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 300)),
            reference_message: "RH, 300 mb", inventory_patterns: &["RH:300 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 20, units: "%", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Rh, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "300mb_absolute_vorticity", slug: "300mb_absolute_vorticity", title: "300 hPa absolute vorticity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, 300)), reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 2, number: 10, level_type: 100, level_value: 30000.0, second_type: 255, second_value: 0.0, stat: None, units: "s-1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "ABSV, 300 mb", inventory_patterns: &["ABSV:300 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: -10.0, step: 2.0, bins: 30, units: "1e-5 s-1", stored_to_drawn: 100000.0, palette: rustwx_render::WeatherPalette::RelVort, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "500mb_rh", slug: "500mb_rh", title: "500 hPa relative humidity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 500)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 500)),
            reference_message: "RH, 500 mb", inventory_patterns: &["RH:500 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 20, units: "%", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Rh, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "500mb_absolute_vorticity", slug: "500mb_absolute_vorticity", title: "500 hPa absolute vorticity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, 500)), reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 2, number: 10, level_type: 100, level_value: 50000.0, second_type: 255, second_value: 0.0, stat: None, units: "s-1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "ABSV, 500 mb", inventory_patterns: &["ABSV:500 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: -10.0, step: 2.0, bins: 30, units: "1e-5 s-1", stored_to_drawn: 100000.0, palette: rustwx_render::WeatherPalette::RelVort, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "700mb_rh", slug: "700mb_rh", title: "700 hPa relative humidity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 700)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 700)),
            reference_message: "RH, 700 mb", inventory_patterns: &["RH:700 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 20, units: "%", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Rh, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "700mb_absolute_vorticity", slug: "700mb_absolute_vorticity", title: "700 hPa absolute vorticity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, 700)), reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 2, number: 10, level_type: 100, level_value: 70000.0, second_type: 255, second_value: 0.0, stat: None, units: "s-1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "ABSV, 700 mb", inventory_patterns: &["ABSV:700 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: -10.0, step: 2.0, bins: 30, units: "1e-5 s-1", stored_to_drawn: 100000.0, palette: rustwx_render::WeatherPalette::RelVort, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "850mb_rh", slug: "850mb_rh", title: "850 hPa relative humidity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 850)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::RelativeHumidity, 850)),
            reference_message: "RH, 850 mb", inventory_patterns: &["RH:850 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 20, units: "%", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Rh, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "850mb_absolute_vorticity", slug: "850mb_absolute_vorticity", title: "850 hPa absolute vorticity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::AbsoluteVorticity, 850)), reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 2, number: 10, level_type: 100, level_value: 85000.0, second_type: 255, second_value: 0.0, stat: None, units: "s-1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "ABSV, 850 mb", inventory_patterns: &["ABSV:850 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: -10.0, step: 2.0, bins: 30, units: "1e-5 s-1", stored_to_drawn: 100000.0, palette: rustwx_render::WeatherPalette::RelVort, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "700mb_dewpoint", slug: "700mb_dewpoint", title: "700 hPa dewpoint",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Dewpoint, 700)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Dewpoint, 700)),
            reference_message: "DPT, 700 mb", inventory_patterns: &["DPT:700 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 230.0, step: 2.0, bins: 32, units: "K", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Dewpoint, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "850mb_dewpoint", slug: "850mb_dewpoint", title: "850 hPa dewpoint",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Dewpoint, 850)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Dewpoint, 850)),
            reference_message: "DPT, 850 mb", inventory_patterns: &["DPT:850 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 240.0, step: 2.0, bins: 32, units: "K", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Dewpoint, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "250mb_wind_speed", slug: "250mb_wind_speed", title: "250 hPa wind speed",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::SelectorSpeed(FieldSelector::isobaric(CanonicalField::UWind, 250), FieldSelector::isobaric(CanonicalField::VWind, 250)), reference: ReferencePlane::Speed(FieldSelector::isobaric(CanonicalField::UWind, 250), FieldSelector::isobaric(CanonicalField::VWind, 250)),
            reference_message: "UGRD, VGRD 250 mb", inventory_patterns: &["UGRD:250 mb", "VGRD:250 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 32, units: "kt", stored_to_drawn: 1.943844, palette: rustwx_render::WeatherPalette::Winds, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "300mb_wind_speed", slug: "300mb_wind_speed", title: "300 hPa wind speed",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::SelectorSpeed(FieldSelector::isobaric(CanonicalField::UWind, 300), FieldSelector::isobaric(CanonicalField::VWind, 300)), reference: ReferencePlane::Speed(FieldSelector::isobaric(CanonicalField::UWind, 300), FieldSelector::isobaric(CanonicalField::VWind, 300)),
            reference_message: "UGRD, VGRD 300 mb", inventory_patterns: &["UGRD:300 mb", "VGRD:300 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 32, units: "kt", stored_to_drawn: 1.943844, palette: rustwx_render::WeatherPalette::Winds, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "500mb_wind_speed", slug: "500mb_wind_speed", title: "500 hPa wind speed",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::SelectorSpeed(FieldSelector::isobaric(CanonicalField::UWind, 500), FieldSelector::isobaric(CanonicalField::VWind, 500)), reference: ReferencePlane::Speed(FieldSelector::isobaric(CanonicalField::UWind, 500), FieldSelector::isobaric(CanonicalField::VWind, 500)),
            reference_message: "UGRD, VGRD 500 mb", inventory_patterns: &["UGRD:500 mb", "VGRD:500 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 22, units: "kt", stored_to_drawn: 1.943844, palette: rustwx_render::WeatherPalette::Winds, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "700mb_wind_speed", slug: "700mb_wind_speed", title: "700 hPa wind speed",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::SelectorSpeed(FieldSelector::isobaric(CanonicalField::UWind, 700), FieldSelector::isobaric(CanonicalField::VWind, 700)), reference: ReferencePlane::Speed(FieldSelector::isobaric(CanonicalField::UWind, 700), FieldSelector::isobaric(CanonicalField::VWind, 700)),
            reference_message: "UGRD, VGRD 700 mb", inventory_patterns: &["UGRD:700 mb", "VGRD:700 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 22, units: "kt", stored_to_drawn: 1.943844, palette: rustwx_render::WeatherPalette::Winds, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "850mb_wind_speed", slug: "850mb_wind_speed", title: "850 hPa wind speed",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::SelectorSpeed(FieldSelector::isobaric(CanonicalField::UWind, 850), FieldSelector::isobaric(CanonicalField::VWind, 850)), reference: ReferencePlane::Speed(FieldSelector::isobaric(CanonicalField::UWind, 850), FieldSelector::isobaric(CanonicalField::VWind, 850)),
            reference_message: "UGRD, VGRD 850 mb", inventory_patterns: &["UGRD:850 mb", "VGRD:850 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 5.0, bins: 22, units: "kt", stored_to_drawn: 1.943844, palette: rustwx_render::WeatherPalette::Winds, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "500mb_omega", slug: "500mb_omega", title: "500 hPa omega",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::Omega, 500)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::Omega, 500)),
            reference_message: "VVEL, 500 mb", inventory_patterns: &["VVEL:500 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: -3.0, step: 0.25, bins: 24, units: "Pa/s", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::Advection, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "850mb_specific_humidity", slug: "850mb_specific_humidity", title: "850 hPa specific humidity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Selector(FieldSelector::isobaric(CanonicalField::SpecificHumidity, 850)), reference: ReferencePlane::Selector(FieldSelector::isobaric(CanonicalField::SpecificHumidity, 850)),
            reference_message: "SPFH, 850 mb", inventory_patterns: &["SPFH:850 mb"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 1.0, bins: 20, units: "g/kg", stored_to_drawn: 1000.0, palette: rustwx_render::WeatherPalette::Dewpoint, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "mslp", slug: "mslp", title: "Mean sea level pressure",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "mslp", variable: "mslp" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 3, number: 198, level_type: 101, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "Pa", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "MSLMA, mean sea level", inventory_patterns: &["MSLMA:mean sea level"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 960.0, step: 2.0, bins: 46, units: "hPa", stored_to_drawn: 0.01, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 4.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "2m_relative_humidity", slug: "2m_relative_humidity", title: "2 m relative humidity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "relative_humidity_2m", variable: "relative_humidity_2m" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 1, level_type: 103, level_value: 2.0, second_type: 255, second_value: 0.0, stat: None, units: "%", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "RH, 2 m", inventory_patterns: &["RH:2 m above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "10m_wind_gusts", slug: "10m_wind_gusts", title: "10 m wind gust",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "wind_gust_10m_agl", variable: "wind_gust_10m_agl" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 2, number: 22, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "m/s", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "GUST, surface", inventory_patterns: &["GUST:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "1km_reflectivity", slug: "1km_reflectivity", title: "1 km AGL reflectivity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "reflectivity_1km", variable: "reflectivity_1km" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 16, number: 195, level_type: 103, level_value: 1000.0, second_type: 255, second_value: 0.0, stat: None, units: "dBZ", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "REFD, 1000 m", inventory_patterns: &["REFD:1000 m above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "precipitable_water", slug: "precipitable_water", title: "Precipitable water",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "pwat", variable: "pwat" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 3, level_type: 200, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "mm", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "PWAT, entire atmosphere", inventory_patterns: &["PWAT:entire atmosphere (considered as a single layer)"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "visibility", slug: "visibility", title: "Surface visibility",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "visibility_surface", variable: "visibility_surface" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 19, number: 0, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "m", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "VIS, surface", inventory_patterns: &["VIS:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "isotherm_height_0c", slug: "isotherm_height_0c", title: "0 C isotherm height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "isotherm_height_0c", variable: "isotherm_height_0c" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 3, number: 5, level_type: 4, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "m", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "HGT, 0C isotherm", inventory_patterns: &["HGT:0C isotherm"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 250.0, bins: 22, units: "m", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "isotherm_height_minus10c", slug: "isotherm_height_minus10c", title: "-10 C isotherm height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "isotherm_height_minus10c", variable: "isotherm_height_minus10c" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 3, number: 5, level_type: 20, level_value: 263.0, second_type: 255, second_value: 0.0, stat: None, units: "m", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "HGT, 263 K level", inventory_patterns: &["HGT:263 K level"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 350.0, bins: 22, units: "m", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "isotherm_height_minus20c", slug: "isotherm_height_minus20c", title: "-20 C isotherm height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "isotherm_height_minus20c", variable: "isotherm_height_minus20c" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 3, number: 5, level_type: 20, level_value: 253.0, second_type: 255, second_value: 0.0, stat: None, units: "m", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "HGT, 253 K level", inventory_patterns: &["HGT:253 K level"],
            continuous: true, style_variable: None, ladder: Some(Ladder { first: 0.0, step: 450.0, bins: 22, units: "m", stored_to_drawn: 1.0, palette: rustwx_render::WeatherPalette::IsothermHeight, contour_interval: 0.0 }), neutral_range: None, sheet_scale: None },
        ProductSpec { name: "low_cloud_cover", slug: "low_cloud_cover", title: "Low cloud cover",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "low_cloud_cover_entire_atmosphere", variable: "low_cloud_cover_entire_atmosphere" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 6, number: 3, level_type: 214, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "%", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "LCDC, low cloud layer", inventory_patterns: &["LCDC:low cloud layer"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "middle_cloud_cover", slug: "middle_cloud_cover", title: "Middle cloud cover",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "middle_cloud_cover_entire_atmosphere", variable: "middle_cloud_cover_entire_atmosphere" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 6, number: 4, level_type: 224, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "%", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "MCDC, middle cloud layer", inventory_patterns: &["MCDC:middle cloud layer"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "high_cloud_cover", slug: "high_cloud_cover", title: "High cloud cover",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "high_cloud_cover_entire_atmosphere", variable: "high_cloud_cover_entire_atmosphere" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 6, number: 5, level_type: 234, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "%", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "HCDC, high cloud layer", inventory_patterns: &["HCDC:high cloud layer"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "cloud_fraction_column_max", slug: "cloud_fraction_column_max", title: "Total cloud cover",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "cloud_fraction_column_maximum", variable: "cloud_fraction_column_maximum" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 6, number: 1, level_type: 10, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "%", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "TCDC, entire atmosphere", inventory_patterns: &["TCDC:entire atmosphere"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "simulated_ir_satellite", slug: "simulated_ir_satellite", title: "Simulated IR brightness temperature",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "simulated_ir_brightness_temperature", variable: "simulated_ir_brightness_temperature" }, reference: ReferencePlane::Raw(RawKey { discipline: 3, category: 192, number: 2, level_type: 8, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "K", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "SBT124 10.7 um, TOA", inventory_patterns: &["SBT124:top of atmosphere"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "categorical_rain", slug: "categorical_rain", title: "Categorical rain",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "categorical_rain_surface", variable: "categorical_rain_surface" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 33, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "0/1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CRAIN, surface", inventory_patterns: &["CRAIN:surface"],
            continuous: false, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "categorical_snow", slug: "categorical_snow", title: "Categorical snow",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "categorical_snow_surface", variable: "categorical_snow_surface" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 36, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "0/1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CSNOW, surface", inventory_patterns: &["CSNOW:surface"],
            continuous: false, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "categorical_freezing_rain", slug: "categorical_freezing_rain", title: "Categorical freezing rain",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "categorical_freezing_rain_surface", variable: "categorical_freezing_rain_surface" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 34, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "0/1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CFRZR, surface", inventory_patterns: &["CFRZR:surface"],
            continuous: false, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "categorical_ice_pellets", slug: "categorical_ice_pellets", title: "Categorical ice pellets",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "categorical_ice_pellets_surface", variable: "categorical_ice_pellets_surface" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 35, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "0/1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CICEP, surface", inventory_patterns: &["CICEP:surface"],
            continuous: false, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "total_qpf", slug: "total_qpf", title: "Total precipitation since start",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "apcp", variable: "apcp" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 8, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: Some(1), units: "kg/m^2", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "APCP, run total", inventory_patterns: &["APCP:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        // HRRR's wrfsfc carries three HAIL (0.1.31) one-hour maxima, and NCEP UPP decides which WRF field
        // each one is (sorc/ncep_post.fd/MDL2AGL.f, parm/post_avblflds_raphrrr.xml 794/795/728):
        // entire atmosphere = HAIL_MAX2D, the 0.1 sigma level = HAIL_MAXK1, surface = HAILCAST_DIAM_MAX / 1000.
        // Breakage it prevents: hail_max_surface (HAIL_MAXK1) was drawn beside HRRR's surface message, which is
        // HAILCAST, so the sheet set the PSD diameter at the lowest level against a different diagnostic.
        ProductSpec { name: "hail_max_surface", slug: "hail_max_surface", title: "Max hail size at the lowest model level, last hour",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "hail_max_surface", variable: "hail_max_surface" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 31, level_type: 104, level_value: 0.1, second_type: 255, second_value: 0.0, stat: Some(2), units: "mm", factor: 1000.0_f32, offset: 0.0_f32 }),
            reference_message: "HAIL, 0.1 sigma, 1 h max", inventory_patterns: &["HAIL:0.1 sigma level"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "hail_max_column", slug: "hail_max_column", title: "Max hail size in the column, last hour",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "hail_max_column", variable: "hail_max_column" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 31, level_type: 10, level_value: 0.0, second_type: 255, second_value: 0.0, stat: Some(2), units: "mm", factor: 1000.0_f32, offset: 0.0_f32 }),
            reference_message: "HAIL, column, 1 h max", inventory_patterns: &["HAIL:entire atmosphere"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "hail_max_hailcast", slug: "hail_max_hailcast", title: "Max HAILCAST hail size at the surface, last hour",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "hail_max_hailcast", variable: "hail_max_hailcast" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 31, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: Some(2), units: "mm", factor: 1000.0_f32, offset: 0.0_f32 }),
            reference_message: "HAIL, surface, 1 h max", inventory_patterns: &["HAIL:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "snow_depth_sd", slug: "snow_depth_sd", title: "Snow depth",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "snow_depth_total", variable: "snow_depth_total" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 1, number: 11, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "mm", factor: 1000.0_f32, offset: 0.0_f32 }),
            reference_message: "SNOD, surface", inventory_patterns: &["SNOD:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "sbcape", slug: "sbcape", title: "Surface-based CAPE",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "sbcape", variable: "sbcape" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 7, number: 6, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "J/kg", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CAPE, surface", inventory_patterns: &["CAPE:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "sbcin", slug: "sbcin", title: "Surface-based CIN",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "sbcin", variable: "sbcin" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 7, number: 7, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "J/kg", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CIN, surface", inventory_patterns: &["CIN:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "mlcape", slug: "mlcape", title: "Mixed-layer CAPE",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "mlcape", variable: "mlcape" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 7, number: 6, level_type: 108, level_value: 9000.0, second_type: 108, second_value: 0.0, stat: None, units: "J/kg", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CAPE, 90-0 mb AGL", inventory_patterns: &["CAPE:90-0 mb above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "mlcin", slug: "mlcin", title: "Mixed-layer CIN",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "mlcin", variable: "mlcin" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 7, number: 7, level_type: 108, level_value: 9000.0, second_type: 108, second_value: 0.0, stat: None, units: "J/kg", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CIN, 90-0 mb AGL", inventory_patterns: &["CIN:90-0 mb above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "mucape", slug: "mucape", title: "Most-unstable CAPE",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "mucape", variable: "mucape" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 7, number: 6, level_type: 108, level_value: 25500.0, second_type: 108, second_value: 0.0, stat: None, units: "J/kg", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CAPE, 255-0 mb AGL", inventory_patterns: &["CAPE:255-0 mb above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "mucin", slug: "mucin", title: "Most-unstable CIN",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "mucin", variable: "mucin" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 7, number: 7, level_type: 108, level_value: 25500.0, second_type: 108, second_value: 0.0, stat: None, units: "J/kg", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "CIN, 255-0 mb AGL", inventory_patterns: &["CIN:255-0 mb above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "srh_0_1km", slug: "srh_0_1km", title: "0-1 km storm-relative helicity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "srh_0_1km", variable: "srh_0_1km" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 7, number: 8, level_type: 103, level_value: 1000.0, second_type: 103, second_value: 0.0, stat: None, units: "m2/s2", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "HLCY, 1000-0 m", inventory_patterns: &["HLCY:1000-0 m above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "srh_0_3km", slug: "srh_0_3km", title: "0-3 km storm-relative helicity",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "srh_0_3km", variable: "srh_0_3km" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 7, number: 8, level_type: 103, level_value: 3000.0, second_type: 103, second_value: 0.0, stat: None, units: "m2/s2", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "HLCY, 3000-0 m", inventory_patterns: &["HLCY:3000-0 m above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "terrain_height", slug: "terrain_height", title: "Terrain height",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "orography", variable: "orography" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 3, number: 5, level_type: 1, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "m", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "HGT, surface", inventory_patterns: &["HGT:surface"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "lightning_threat_3", slug: "lightning_threat_3", title: "Lightning threat 3",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "lightning_threat_3", variable: "lightning_threat_3" }, reference: ReferencePlane::Raw(RawKey { discipline: 0, category: 17, number: 192, level_type: 10, level_value: 0.0, second_type: 255, second_value: 0.0, stat: None, units: "flashes km^-2 (5 min)^-1", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "LTNG, entire atmosphere", inventory_patterns: &["LTNG:entire atmosphere"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "bulk_shear_0_1km", slug: "bulk_shear_0_1km", title: "0-1 km bulk shear",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "bulk_shear_0_1km", variable: "bulk_shear_0_1km" }, reference: ReferencePlane::RawSpeed(RawKey { discipline: 0, category: 2, number: 15, level_type: 103, level_value: 0.0, second_type: 103, second_value: 1000.0, stat: None, units: "m/s", factor: 1.0_f32, offset: 0.0_f32 }, RawKey { discipline: 0, category: 2, number: 16, level_type: 103, level_value: 0.0, second_type: 103, second_value: 1000.0, stat: None, units: "m/s", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "VUCSH, VVCSH 0-1000 m", inventory_patterns: &["VUCSH:0-1000 m above ground", "VVCSH:0-1000 m above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
        ProductSpec { name: "bulk_shear_0_6km", slug: "bulk_shear_0_6km", title: "0-6 km bulk shear",
            station_quantity: None, observation_quantity: None,
            run: RunPlane::Named { raw: "bulk_shear_0_6km", variable: "bulk_shear_0_6km" }, reference: ReferencePlane::RawSpeed(RawKey { discipline: 0, category: 2, number: 15, level_type: 103, level_value: 0.0, second_type: 103, second_value: 6000.0, stat: None, units: "m/s", factor: 1.0_f32, offset: 0.0_f32 }, RawKey { discipline: 0, category: 2, number: 16, level_type: 103, level_value: 0.0, second_type: 103, second_value: 6000.0, stat: None, units: "m/s", factor: 1.0_f32, offset: 0.0_f32 }),
            reference_message: "VUCSH, VVCSH 0-6000 m", inventory_patterns: &["VUCSH:0-6000 m above ground", "VVCSH:0-6000 m above ground"],
            continuous: true, style_variable: None, ladder: None, neutral_range: None, sheet_scale: None },
    ]
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DifferenceMode {
    /// The third panel for continuous fields only.
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    Nested,
    Flat,
}

struct Args {
    store_root: PathBuf,
    out_dir: PathBuf,
    reference: String,
    products: Vec<String>,
    difference: DifferenceMode,
    reference_file: Option<PathBuf>,
    reference_dir: Option<PathBuf>,
    reference_cache: Option<PathBuf>,
    offline: bool,
    cycle: Option<(String, u8)>,
    layout: Layout,
    flat_dir: Option<PathBuf>,
    width: u32,
    height: u32,
    source_label: String,
    run_label: String,
    run_source_subtitle: String,
    theme: Option<String>,
    context: Vec<PathBuf>,
    inputs: Vec<PathBuf>,
    fetch_reference: bool,
    forecast_hour: Option<u16>,
    observations: Option<PathBuf>,
    stations: Option<PathBuf>,
    station_mode: String,
    /// `--panel` rows: empty unless the sheet is an N-panel sheet.
    panels: Vec<Vec<compare_panels::PanelArg>>,
    /// `--order`: the row's columns, panel numbers and reference names.
    order: Option<Vec<String>>,
    /// `--sheet-name`: the N-panel sheet's file stem.
    sheet_name: Option<String>,
    /// `--panel-grid`: the history file whose grid a row of `frames:` or
    /// `ttenref:` panels alone is drawn on (its values are never drawn).
    panel_grid: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: rw_compare --store-root DIR --out-dir DIR --reference NAME[,NAME...] \
[--products LIST|all] [--difference auto|on|off] \
[--reference-file FILE.grib2 | --reference-dir DIR] [--reference-cache DIR] [--offline] \
[--cycle YYYYMMDDHH] [--layout nested|flat] [--flat-dir DIR] [--width N] [--height N] \
[--source-label TEXT] [--run-label TEXT] [--theme NAME|FILE] [--context WRFOUT ...] WRFOUT [WRFOUT ...]\n       \
rw_compare --fetch-reference --store-root DIR --reference NAME --cycle YYYYMMDDHH --forecast-hour N\n       \
[--observations FILE.json] [--stations FILE.json] [--station-mode observed|error]\n       \
rw_compare --store-root DIR --out-dir DIR [--reference NAME[,NAME...]] \
--panel \"LABEL=WRFOUT|mean:LIST|pmm:LIST|max:LIST|frames:ROOT@VALID|ttenref:WINDOW\" ... \
[--panel-grid WRFOUT] [--row --panel ...]... [--order TOKEN,...] [--sheet-name STEM]\n       \
rw_compare --mode latlon-pair ... (see rw_compare --mode latlon-pair --help)\n       \
rw_compare --mode increment|spread|stamp|omb ... (see rw_compare --mode stamp --help)\n       \
rw_compare --list-products | --help | --abi"
}

enum Invocation {
    Run(Box<Args>),
    ListProducts,
    Abi,
    Help,
}

fn parse_cycle(text: &str) -> Result<(String, u8), String> {
    let digits: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() != 10 {
        return Err(format!(
            "--cycle {text:?} is not YYYYMMDDHH (ten digits; separators are ignored)"
        ));
    }
    let hour: u8 = digits[8..10]
        .parse()
        .map_err(|_| format!("--cycle {text:?} has no hour"))?;
    CycleSpec::new(digits[..8].to_string(), hour)
        .map_err(|error| format!("--cycle {text:?}: {error}"))?;
    Ok((digits[..8].to_string(), hour))
}

fn parse_args() -> Result<Invocation, String> {
    let mut store_root = None;
    let mut out_dir = None;
    // No default: which model a run is compared with is the caller's
    // statement, and a reference named by a default is one a reader never
    // chose.  `--list-products` prints the names this build knows.
    let mut reference: Option<String> = None;
    let mut products = "all".to_string();
    let mut difference = DifferenceMode::Auto;
    let mut reference_file = None;
    let mut reference_dir = None;
    let mut reference_cache = None;
    let mut offline = false;
    let mut cycle = None;
    let mut layout = Layout::Nested;
    let mut flat_dir = None;
    let mut width = 1_200u32;
    let mut height = 900u32;
    let mut source_label = "WOOF".to_string();
    let mut run_label = "WOOF".to_string();
    let mut theme = None;
    let mut context = Vec::new();
    let mut inputs = Vec::new();
    let mut fetch_reference = false;
    let mut forecast_hour = None;
    let mut observations = None;
    let mut stations = None;
    let mut station_mode = "observed".to_string();
    let mut panels: Vec<Vec<compare_panels::PanelArg>> = vec![Vec::new()];
    let mut order: Option<Vec<String>> = None;
    let mut sheet_name = None;
    let mut panel_grid = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--help" | "-h" => return Ok(Invocation::Help),
            "--abi" => return Ok(Invocation::Abi),
            "--list-products" => return Ok(Invocation::ListProducts),
            "--store-root" => store_root = Some(PathBuf::from(value("--store-root")?)),
            "--out-dir" => out_dir = Some(PathBuf::from(value("--out-dir")?)),
            "--reference" => reference = Some(value("--reference")?),
            "--products" => products = value("--products")?,
            "--difference" => {
                difference = match value("--difference")?.as_str() {
                    "auto" => DifferenceMode::Auto,
                    "on" => DifferenceMode::On,
                    "off" => DifferenceMode::Off,
                    other => {
                        return Err(format!("--difference {other:?} is not auto, on or off"));
                    }
                }
            }
            "--reference-file" => reference_file = Some(PathBuf::from(value("--reference-file")?)),
            "--reference-dir" => reference_dir = Some(PathBuf::from(value("--reference-dir")?)),
            "--reference-cache" => {
                reference_cache = Some(PathBuf::from(value("--reference-cache")?))
            }
            "--offline" => offline = true,
            "--cycle" => cycle = Some(parse_cycle(&value("--cycle")?)?),
            "--layout" => {
                layout = match value("--layout")?.as_str() {
                    "nested" => Layout::Nested,
                    "flat" => Layout::Flat,
                    other => return Err(format!("--layout {other:?} is not nested or flat")),
                }
            }
            "--flat-dir" => flat_dir = Some(PathBuf::from(value("--flat-dir")?)),
            "--width" => {
                width = value("--width")?
                    .parse()
                    .map_err(|_| "--width needs a whole number".to_string())?
            }
            "--height" => {
                height = value("--height")?
                    .parse()
                    .map_err(|_| "--height needs a whole number".to_string())?
            }
            "--source-label" => source_label = value("--source-label")?,
            "--run-label" => run_label = value("--run-label")?,
            "--theme" => theme = Some(value("--theme")?),
            "--context" => context.push(PathBuf::from(value("--context")?)),
            "--fetch-reference" => fetch_reference = true,
            "--forecast-hour" => forecast_hour = Some(value("--forecast-hour")?.parse::<u16>().map_err(|_| "--forecast-hour needs a whole number")?),
            "--observations" => observations = Some(PathBuf::from(value("--observations")?)),
            "--stations" => stations = Some(PathBuf::from(value("--stations")?)),
            "--station-mode" => {
                station_mode = value("--station-mode")?;
                if !matches!(station_mode.as_str(), "observed" | "error") {
                    return Err("--station-mode is observed or error".to_string());
                }
            }
            "--panel" => panels
                .last_mut()
                .expect("one row always open")
                .push(compare_panels::parse_panel(&value("--panel")?)?),
            "--row" => {
                if panels.last().is_some_and(|row| !row.is_empty()) {
                    panels.push(Vec::new());
                }
            }
            "--order" => {
                order = Some(
                    value("--order")?
                        .split(',')
                        .map(str::trim)
                        .filter(|token| !token.is_empty())
                        .map(str::to_string)
                        .collect(),
                )
            }
            "--sheet-name" => sheet_name = Some(value("--sheet-name")?),
            "--panel-grid" => panel_grid = Some(PathBuf::from(value("--panel-grid")?)),
            // The DA sheet modes are dispatched before this parser runs.
            "--mode" => {
                let mode = value("--mode")?;
                if mode != "compare" {
                    return Err(format!("--mode {mode:?} is handled by the DA sheet entry"));
                }
            }
            other if other.starts_with("--") => return Err(format!("unknown option {other}")),
            other => inputs.push(PathBuf::from(other)),
        }
    }
    panels.retain(|row| !row.is_empty());
    if let Some(first) = panels.first() {
        if panels.iter().any(|row| row.len() != first.len()) {
            return Err("every --row holds the same number of --panel entries".into());
        }
        if !inputs.is_empty() {
            return Err(
                "--panel names every model panel with its label; a positional WRFOUT beside                  --panel would be a panel with no label"
                    .into(),
            );
        }
    } else if order.is_some() || sheet_name.is_some() || panel_grid.is_some() {
        return Err(
            "--order, --sheet-name and --panel-grid arrange an N-panel sheet; give --panel".into(),
        );
    }
    let panel_mode = !panels.is_empty();

    let store_root = store_root.ok_or("--store-root is required")?;
    let out_dir = out_dir.or_else(|| fetch_reference.then(|| store_root.clone())).ok_or("--out-dir is required")?;
    if inputs.is_empty() && !fetch_reference && !panel_mode {
        return Err("at least one WRFOUT frame is required".into());
    }
    if !(256..=4096).contains(&width) || !(256..=4096).contains(&height) {
        return Err(format!(
            "panel size {width}x{height} is outside 256..4096 on a side"
        ));
    }
    if reference_file.is_some() && reference_dir.is_some() {
        return Err("--reference-file and --reference-dir do not combine".into());
    }
    if reference_file.is_some() && inputs.len() != 1 && !fetch_reference && !panel_mode {
        return Err(
            "--reference-file names one cycle and lead, so it takes exactly one WRFOUT frame; \
             use --reference-dir for several"
                .into(),
        );
    }
    let known_references = || {
        reference_specs()
            .iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>()
            .join(", ")
    };
    // An N-panel sheet may compare model panels with each other only.
    let reference = match reference {
        Some(reference) => reference,
        None if panel_mode => String::new(),
        None => {
            return Err(format!(
                "--reference is required: the model the run is drawn beside. This build knows: {}",
                known_references()
            ));
        }
    };
    let selected = if reference.is_empty() { Vec::new() } else { reference_names(&reference)? };
    if reference_file.is_some() && (selected.len() != 1 || panels.len() > 1) {
        return Err(
            "--reference-file requires one reference; use --reference-dir for a list".into(),
        );
    }
    let reference = selected.join(",");
    if fetch_reference && (selected.len() != 1 || !reference_specs().iter().any(|row| row.name == selected[0])) {
        return Err("--fetch-reference requires exactly one forecast reference row".into());
    }
    let known = product_specs();
    let products: Vec<String> = if products.trim() == "all" {
        known.iter().map(|spec| spec.name.to_string()).collect()
    } else {
        let mut chosen = Vec::new();
        for token in products.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let spec = known
                .iter()
                .find(|spec| spec.name == token || spec.slug == token)
                .ok_or_else(|| {
                    format!(
                        "unknown product {token:?}; this build compares: {}",
                        known
                            .iter()
                            .map(|spec| spec.name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            if !chosen.iter().any(|name: &String| name == spec.name) {
                chosen.push(spec.name.to_string());
            }
        }
        if chosen.is_empty() {
            return Err("--products named nothing".into());
        }
        chosen
    };

    let run_source_subtitle = format!("source: {source_label}");
    Ok(Invocation::Run(Box::new(Args {
        store_root,
        out_dir,
        reference,
        products,
        difference,
        reference_file,
        reference_dir,
        reference_cache,
        offline,
        cycle,
        layout,
        flat_dir,
        width,
        height,
        source_label,
        run_label,
        run_source_subtitle,
        theme,
        context,
        inputs,
        fetch_reference,
        forecast_hour,
        observations,
        stations,
        station_mode,
        panels,
        order,
        sheet_name,
        panel_grid,
    })))
}

pub(crate) fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    // The DA sheet modes (increment, spread, stamp, omb) read their fields
    // from the files directly and share no option with the comparison.
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if let Some(mode) = raw
        .iter()
        .position(|arg| arg == "--mode")
        .and_then(|index| raw.get(index + 1))
        .filter(|mode| mode.as_str() != "compare")
    {
        if mode == "latlon-pair" {
            return match rw_wrfbatch::latlon_pair::cli(&raw) {
                Ok(()) => ExitCode::SUCCESS,
                Err(message) => {
                    eprintln!("{message}\n{}", rw_wrfbatch::latlon_pair::USAGE);
                    ExitCode::FAILURE
                }
            };
        }
        return match rw_wrfbatch::da_sheet::cli(&raw) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("{message}");
                eprintln!("{}", rw_wrfbatch::da_sheet::USAGE);
                ExitCode::FAILURE
            }
        };
    }
    match parse_args() {
        Ok(Invocation::Abi) => {
            println!("{ABI_MARKER}");
            ExitCode::SUCCESS
        }
        Ok(Invocation::Help) => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Ok(Invocation::ListProducts) => {
            for spec in reference_specs() {
                println!("REFERENCE\t{}\t{}", spec.name, spec.label);
                for product in product_specs() {
                    println!("CAPABILITY\t{}\t{}\tforecast", spec.name, product.name);
                }
            }
            for spec in
                comparison_observations::specifications().expect("validated observation table")
            {
                println!("REFERENCE\t{}\t{}", spec.name, spec.label);
                for product in spec.products {
                    println!(
                        "CAPABILITY\t{}\t{}\tobservation",
                        spec.name, product.product
                    );
                }
            }
            for spec in product_specs() {
                println!(
                    "PRODUCT\t{}\t{}\t{}\tdifference={}",
                    spec.name,
                    spec.slug,
                    spec.title,
                    if spec.continuous {
                        "auto"
                    } else {
                        "on-request"
                    }
                );
            }
            ExitCode::SUCCESS
        }
        Ok(Invocation::Run(args)) => match run(*args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(message) => {
                eprintln!("{message}");
                ExitCode::FAILURE
            }
        },
        Err(message) => {
            eprintln!("{message}");
            eprintln!("{}", usage());
            ExitCode::from(2)
        }
    }
}

/// One imported frame of the run.
struct Frame {
    slot: u16,
    file: PathBuf,
    lead_seconds: u64,
    valid_unix: i64,
    compared: bool,
}

fn import(
    paths: Vec<PathBuf>,
    store_root: &Path,
    options: WrfProcessOptions,
) -> Result<rw_wrfbatch::wrf_process::WrfProcessSummary, String> {
    std::fs::create_dir_all(store_root)
        .map_err(|error| format!("create {}: {error}", store_root.display()))?;
    let task = spawn_process_paths(paths, store_root.to_path_buf(), options);
    loop {
        match task
            .rx
            .recv()
            .map_err(|error| format!("the wrfout importer exited: {error}"))?
        {
            WrfProcessMessage::Progress(_) => {}
            WrfProcessMessage::Done(result) => return result,
        }
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

fn copy_flat_sheet(source: &Path, destination: &Path) -> Result<(), String> {
    // A flat gallery can be the output directory itself. Resolve aliases
    // before copying so the completed sheet is never truncated in place.
    if same_file(source, destination) {
        return Ok(());
    }
    std::fs::copy(source, destination)
        .map(|_| ())
        .map_err(|error| format!("copy to {}: {error}", destination.display()))
}

fn utc(unix: i64) -> Result<DateTime<Utc>, String> {
    DateTime::<Utc>::from_timestamp(unix, 0)
        .ok_or_else(|| format!("{unix} is not a representable UTC time"))
}

/// The names a history frame one hour before `name` may carry, in both
/// separator spellings (`06:00:00` and `06_00_00`), or nothing when `name`
/// is not a `wrfout_dNN_YYYY-MM-DD_HH:MM:SS` frame.
fn names_an_hour_earlier(name: &str) -> Vec<String> {
    let Some(rest) = name.strip_prefix("wrfout_d") else {
        return Vec::new();
    };
    let (Some(domain), Some(stamp)) = (rest.get(..2), rest.get(3..22)) else {
        return Vec::new();
    };
    if !domain.bytes().all(|byte| byte.is_ascii_digit()) || rest.as_bytes()[2] != b'_' {
        return Vec::new();
    }
    let normalized: String = stamp
        .chars()
        .enumerate()
        .map(|(index, character)| {
            if matches!(index, 13 | 16) && matches!(character, ':' | '_') {
                ':'
            } else {
                character
            }
        })
        .collect();
    let Ok(valid) = chrono::NaiveDateTime::parse_from_str(&normalized, "%Y-%m-%d_%H:%M:%S") else {
        return Vec::new();
    };
    let earlier = valid - chrono::Duration::hours(1);
    let suffix = &rest[22..];
    [":", "_"]
        .iter()
        .map(|separator| {
            format!(
                "wrfout_d{domain}_{}{separator}{}{separator}{}{suffix}",
                earlier.format("%Y-%m-%d_%H"),
                earlier.format("%M"),
                earlier.format("%S"),
            )
        })
        .collect()
}

/// The frame one hour before `path`, when it sits beside it.
///
/// An hourly accumulation is the difference of two run totals, so the
/// earlier frame is needed whenever one is drawn; looking beside the frame
/// that was named is what lets a caller name only the frames it wants
/// drawn.
fn sibling_an_hour_earlier(path: &Path) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    names_an_hour_earlier(name)
        .into_iter()
        .map(|earlier| path.with_file_name(earlier))
        .find(|candidate| candidate.is_file())
}

/// The run origin a local import's run key opens with:
/// `local_<YYYYMMDDHHMMSS>_...` (`ForecastHourTimeline::run_name`).
fn run_origin_unix(run_slug: &str) -> Option<i64> {
    let stamp = run_slug.strip_prefix("local_")?.get(..14)?;
    if !stamp.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    chrono::NaiveDateTime::parse_from_str(stamp, "%Y%m%d%H%M%S")
        .ok()
        .map(|time| time.and_utc().timestamp())
}

/// `(lead seconds, valid unix)` of one stored frame.
///
/// A store written with exact times says so itself.  One written on whole
/// forecast hours keys its slots by the hour and carries the origin in the
/// run key, so the pair is the origin plus that many hours.
fn frame_time(
    exact: Option<rw_store::RwsExactTime>,
    run_slug: &str,
    slot: u16,
) -> Option<(u64, i64)> {
    if let Some(exact) = exact {
        return Some((exact.lead_seconds, exact.valid_unix));
    }
    let origin = run_origin_unix(run_slug)?;
    let lead = u64::from(slot) * 3_600;
    Some((lead, origin + lead as i64))
}

/// `d01-3km`, from the file's own `GRID_ID` and `DX`.
fn domain_tokens(path: &Path) -> (String, Option<String>) {
    let file = wrf_core::WrfFile::open(path).ok();
    let domain = file
        .as_ref()
        .and_then(|file| file.global_attr_i32("GRID_ID").ok())
        .filter(|id| (1..=99).contains(id))
        .map(|id| format!("d{id:02}"))
        .or_else(|| {
            let name = path.file_name()?.to_str()?;
            let digits: String = name.strip_prefix("wrfout_d")?.chars().take(2).collect();
            (digits.len() == 2 && digits.chars().all(|c| c.is_ascii_digit()))
                .then(|| format!("d{digits}"))
        });
    let spacing = file
        .as_ref()
        .and_then(|file| file.global_attr_f64("DX").ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|metres| {
            let trimmed = |value: f64| {
                let text = format!("{value:.3}");
                text.trim_end_matches('0').trim_end_matches('.').to_string()
            };
            if metres.round() >= 1_000.0 {
                (
                    format!("{}km", trimmed(metres / 1_000.0)),
                    format!("{} km", trimmed(metres / 1_000.0)),
                )
            } else {
                (
                    format!("{:.0}m", metres.round()),
                    format!("{:.0} m", metres.round()),
                )
            }
        });
    match (domain, spacing) {
        (Some(domain), Some((token, label))) => (
            format!("{domain}-{token}"),
            Some(format!("{domain}, \u{0394}x {label}")),
        ),
        (Some(domain), None) => (domain.clone(), Some(domain)),
        (None, Some((token, label))) => (
            format!("native_grid-{token}"),
            Some(format!("\u{0394}x {label}")),
        ),
        (None, None) => ("native_grid".to_string(), None),
    }
}

/// The reference file for one cycle and lead, and where it came from.
struct ReferenceFile {
    grib: Grib2File,
    file_name: String,
    origin: String,
    path: PathBuf,
}

fn inventory_patterns() -> Vec<String> {
    let mut patterns: Vec<String> = Vec::new();
    for spec in product_specs() {
        for pattern in spec.inventory_patterns {
            if !patterns.iter().any(|have| have == pattern) {
                patterns.push((*pattern).to_string());
            }
        }
    }
    patterns
}

/// A short tag for the subset a cached file holds.  The cache keeps the
/// subset of EVERY product in the table, whatever this invocation asked
/// for, so one cached file serves any later request; the tag changes when
/// the table's patterns do, and an older subset is then simply not found.
fn subset_tag(patterns: &[String]) -> String {
    let mut digest = Sha256::new();
    for pattern in patterns {
        digest.update(pattern.as_bytes());
        digest.update([0u8]);
    }
    format!("{:x}", digest.finalize())[..8].to_string()
}

fn load_reference(
    args: &Args,
    spec: &ReferenceSpec,
    date: &str,
    cycle_hour: u8,
    lead: u16,
) -> Result<ReferenceFile, String> {
    let file_names = (spec.file_names)(cycle_hour, lead);
    let file_name = file_names
        .first()
        .cloned()
        .ok_or_else(|| format!("the {} reference names no file", spec.name))?;
    let parse = |bytes: &[u8], origin: String, name: String, path: PathBuf| -> Result<ReferenceFile, String> {
        let grib = Grib2File::from_bytes(bytes)
            .map_err(|error| format!("{origin}: not a readable GRIB2 file: {error}"))?;
        Ok(ReferenceFile {
            grib,
            file_name: name,
            origin,
            path,
        })
    };
    let read = |path: &Path, name: String| -> Result<ReferenceFile, String> {
        let bytes =
            std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        parse(&bytes, path.display().to_string(), name, path.to_path_buf())
    };

    if let Some(path) = &args.reference_file {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| file_name.clone());
        return read(path, name);
    }
    if let Some(dir) = &args.reference_dir {
        let bucket = (spec.bucket_directory)(date);
        for name in &file_names {
            let candidates = [
                dir.join(name),
                dir.join(spec.name).join(name),
                dir.join(&bucket).join(name),
                dir.join(&bucket)
                    .join(format!("{cycle_hour:02}"))
                    .join(name),
                dir.join(bucket.split('/').next().unwrap_or_default())
                    .join(name),
                dir.join(date).join(name),
                dir.join(date).join(format!("{cycle_hour:02}")).join(name),
            ];
            if let Some(found) = candidates.iter().find(|path| path.is_file()) {
                return read(found, name.clone());
            }
        }
    }
    let patterns = inventory_patterns();
    let cache_dir = args
        .reference_cache
        .clone()
        .unwrap_or_else(|| args.store_root.join("reference-cache"));
    let cached = cache_dir.join((spec.bucket_directory)(date)).join(format!(
        "{}.subset-{}.grib2",
        file_name.trim_end_matches(".grib2"),
        subset_tag(&patterns)
    ));
    if cached.is_file() {
        return read(&cached, file_name.clone());
    }
    if args.offline {
        return Err(format!(
            "--offline: {file_name} for cycle {date} {cycle_hour:02}Z is not under --reference-dir \
             and not in the cache ({}); nothing was fetched",
            cached.display()
        ));
    }
    let cycle = CycleSpec::new(date.to_string(), cycle_hour).map_err(|error| error.to_string())?;
    let request = ModelRunRequest::new(spec.model, cycle, lead, spec.file_product)
        .map_err(|error| error.to_string())?;
    let mut failures = Vec::new();
    let mut fetched = None;
    for source in spec.sources {
        match rustwx_io::fetch_bytes(&FetchRequest {
            request: request.clone(),
            source_override: Some(*source),
            variable_patterns: patterns.clone(),
        }) {
            Ok(result) => {
                fetched = Some(result);
                break;
            }
            Err(error) => failures.push(error.to_string()),
        }
    }
    let fetched = fetched.ok_or_else(|| {
        format!(
            "fetch {file_name} for cycle {date} {cycle_hour:02}Z: {}. If this machine has no \
             route to the public bucket, download the file and pass --reference-dir",
            failures.join(" | ")
        )
    })?;
    if let Some(parent) = cached.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    // Written beside its final name and renamed, so a killed run never
    // leaves a truncated subset a later run would read as the file.
    let partial = cached.with_extension("grib2.partial");
    std::fs::write(&partial, &fetched.bytes)
        .map_err(|error| format!("write {}: {error}", partial.display()))?;
    std::fs::rename(&partial, &cached)
        .map_err(|error| format!("rename {}: {error}", partial.display()))?;
    parse(&fetched.bytes, fetched.url, file_name.clone(), cached)
}

fn load_product_reference(
    args: &Args,
    reference: &ReferenceSpec,
    product: &ProductSpec,
    date: &str,
    hour: u8,
    lead: u16,
) -> Result<ReferenceFile, String> {
    let isobaric = matches!(
        product.run,
        RunPlane::Selector(FieldSelector { vertical: rustwx_core::VerticalSelector::IsobaricHpa(_), .. }) | RunPlane::SelectorSpeed(FieldSelector { vertical: rustwx_core::VerticalSelector::IsobaricHpa(_), .. }, _)
    );
    let reference = if isobaric {
        ReferenceSpec {
            file_product: reference.pressure_file_product,
            file_names: reference.pressure_file_names,
            ..reference.clone()
        }
    } else {
        reference.clone()
    };
    load_reference(args, &reference, date, hour, lead)
        .and_then(|file| verify_reference(&file, date, hour, lead).map(|()| file))
}

/// The first reference cycle at or after the run's start that publishes
/// the frame's valid time, as `(YYYYMMDDHH, forecast hour)`.
///
/// What a refusal offers when the run's own cycle stops short of the
/// frame: the nearest later cycle is the one whose forecast began closest
/// to the run's, which is the comparison a reader means by "the same
/// valid time".
fn cycle_reaching(
    spec: &ReferenceSpec,
    run_init: DateTime<Utc>,
    valid: DateTime<Utc>,
) -> Option<(String, u16)> {
    let first = run_init.timestamp().div_euclid(3_600) * 3_600;
    (0..48i64)
        .map(|step| first + step * 3_600)
        .filter(|cycle| *cycle >= run_init.timestamp() && *cycle <= valid.timestamp())
        .find_map(|cycle| {
            let start = utc(cycle).ok()?;
            let lead = u16::try_from((valid.timestamp() - cycle) / 3_600).ok()?;
            ((valid.timestamp() - cycle) % 3_600 == 0 && lead <= (spec.horizon)(start.hour() as u8))
                .then(|| {
                    (
                        format!(
                            "{:04}{:02}{:02}{:02}",
                            start.year(),
                            start.month(),
                            start.day(),
                            start.hour()
                        ),
                        lead,
                    )
                })
        })
}

fn message_hours(unit: u8, value: u32) -> Option<u32> {
    match unit {
        0 => (value % 60 == 0).then_some(value / 60),
        1 => Some(value),
        _ => None,
    }
}

/// Refuse a reference file that is not the cycle and lead the frame needs.
///
/// A file handed over by path or found in a directory is whatever it is;
/// pairing a frame with another cycle's field draws two different
/// forecasts side by side under one valid time, which is the one thing a
/// comparison sheet must never do.
fn verify_reference(
    file: &ReferenceFile,
    date: &str,
    cycle_hour: u8,
    lead: u16,
) -> Result<(), String> {
    let first = file
        .grib
        .messages
        .first()
        .ok_or_else(|| format!("{}: no GRIB2 messages", file.origin))?;
    let reference = first.reference_time;
    let found_date = format!(
        "{:04}{:02}{:02}",
        reference.year(),
        reference.month(),
        reference.day()
    );
    if found_date != date || reference.hour() != u32::from(cycle_hour) {
        return Err(format!(
            "{} is cycle {found_date} {:02}Z; the frame needs cycle {date} {cycle_hour:02}Z. \
             Two cycles' fields under one valid time are not a comparison",
            file.origin,
            reference.hour()
        ));
    }
    let at_lead = file.grib.messages.iter().any(|message| {
        message.product.statistical_process_type.is_none()
            && message_hours(
                message.product.time_range_unit,
                message.product.forecast_time,
            ) == Some(u32::from(lead))
    });
    if !at_lead {
        return Err(format!(
            "{} carries no instantaneous message at forecast hour {lead}; it is not the file for \
             this lead",
            file.origin
        ));
    }
    Ok(())
}

/// One reference plane on ITS OWN grid, with the coordinates it came on.
struct ReferenceField {
    values: Vec<f32>,
    units: String,
    lat: Vec<f32>,
    lon: Vec<f32>,
    ny: usize,
    nx: usize,
}

fn reference_selected(
    grib: &Grib2File,
    selector: FieldSelector,
    forecast_hour: Option<u16>,
) -> Result<Option<rustwx_core::SelectedField2D>, String> {
    let extraction = match forecast_hour {
        Some(hour) => {
            rustwx_io::extract_fields_from_grib2_partial_at_forecast_hour(grib, &[selector], hour)
        }
        None => rustwx_io::extract_fields_from_grib2_partial(grib, &[selector]),
    }
    .map_err(|error| format!("decode {}: {error}", selector.key()))?;
    Ok(extraction.extracted.into_iter().next())
}

fn reference_field(
    grib: &Grib2File,
    plane: ReferencePlane,
    lead: u16,
) -> Result<Result<ReferenceField, String>, String> {
    let wrap = |field: rustwx_core::SelectedField2D| ReferenceField {
        units: field.units.clone(),
        ny: field.grid.shape.ny,
        nx: field.grid.shape.nx,
        lat: field.grid.lat_deg,
        lon: field.grid.lon_deg,
        values: field.values,
    };
    match plane {
        ReferencePlane::Raw(key) => {
            return Ok(match raw_field(grib, &key, lead)? {
                Some(field) => Ok(field),
                None => Err(format!("the reference file has no raw message {key:?} at f{lead:03}")),
            });
        }
        ReferencePlane::RawSpeed(u, v) => {
            let (Some(uf), Some(vf)) = (raw_field(grib, &u, lead)?, raw_field(grib, &v, lead)?) else {
                return Ok(Err(format!("the reference file lacks raw {u:?} or {v:?}")));
            };
            let speed = wind_speed(&uf.values, &vf.values);
            return Ok(Ok(ReferenceField { values: speed, ..uf }));
        }
        ReferencePlane::Selector(selector) => {
            Ok(match reference_selected(grib, selector, Some(lead))? {
                Some(field) => Ok(wrap(field)),
                None => Err(format!(
                    "the reference file has no {} message",
                    selector.key()
                )),
            })
        }
        ReferencePlane::Speed(u, v) => {
            let (Some(u_field), Some(v_field)) = (
                reference_selected(grib, u, Some(lead))?,
                reference_selected(grib, v, Some(lead))?,
            ) else {
                return Ok(Err(format!(
                    "the reference file lacks {} or {}",
                    u.key(),
                    v.key()
                )));
            };
            if u_field.values.len() != v_field.values.len() {
                return Ok(Err("the reference wind components are on two grids".into()));
            }
            let speed = wind_speed(&u_field.values, &v_field.values);
            let mut field = wrap(u_field);
            field.values = speed;
            Ok(Ok(field))
        }
        ReferencePlane::HourlyAccumulation(selector) => {
            if lead == 0 {
                return Ok(Err(
                    "no hour has been accumulated at the analysis time".into()
                ));
            }
            // The window that STARTS an hour before the lead and is one
            // hour long.  Checked on the messages themselves before the
            // extraction is trusted: a run total that happens to be the
            // only accumulation in a file must not be drawn as one hour.
            let start = u32::from(lead) - 1;
            let cycle = grib.messages.first().map(|message| message.reference_time);
            let windowed = Grib2File {
                messages: grib.messages.iter().filter(|message| {
                    Some(message.reference_time) == cycle
                        && message.product.statistical_process_type == Some(1)
                        && message.product.statistical_time_range_hours() == Some(1)
                        && message_hours(message.product.time_range_unit, message.product.forecast_time)
                            == Some(start)
                        && message.product.end_of_interval == Some(
                            message.reference_time + chrono::Duration::hours(i64::from(lead))
                        )
                }).cloned().collect(),
            };
            if windowed.messages.is_empty() {
                return Ok(Err(format!(
                    "the reference file has no one-hour accumulation from hour {start} to {lead}"
                )));
            }
            Ok(match reference_selected(&windowed, selector, Some(lead - 1))? {
                Some(field) => Ok(wrap(field)),
                None => Err(format!(
                    "the reference file has no {} accumulation starting at hour {start}",
                    selector.key()
                )),
            })
        }
    }
}

/// Stored and published units that are the same unit under two spellings.
fn same_units(a: &str, b: &str) -> bool {
    let family = |text: &str| -> String {
        let key: String = text
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        match key.as_str() {
            "k" | "kelvin" => "k".into(),
            "m/s" | "ms-1" | "ms^-1" | "ms**-1" => "m/s".into(),
            "kg/m^2" | "kgm-2" | "kgm^-2" | "mm" | "kg/m2" => "kg/m^2".into(),
            "dbz" | "db" => "dbz".into(),
            "gpm" | "m" => "m".into(),
            "w/m^2" | "w/m2" | "wm-2" | "wm^-2" | "wm**-2" => "W/m^2".into(),
            other => other.to_string(),
        }
    };
    family(a) == family(b)
}

/// How one sheet's values become colour: the same for every panel that
/// shows the field itself.
struct SheetStyle {
    display_units: String,
    scale: rustwx_render::ColorScale,
    cbar_tick_step: Option<f64>,
    legend: LegendControls,
    density: rustwx_render::RenderDensity,
    convert: SheetConvert,
    /// Contour levels drawn over both field panels, in drawn units.
    contour_levels: Vec<f64>,
}

#[derive(Clone, Copy)]
enum SheetConvert {
    Production(rustwx_products::viewer::UnitConvert),
    Factor(f32),
}

impl SheetConvert {
    fn apply(self, value: f32) -> f32 {
        match self {
            Self::Production(convert) => convert.apply(value),
            Self::Factor(factor) => value * factor,
        }
    }
}

impl SheetStyle {
    fn production(style: StoreVariableStyle) -> Self {
        Self {
            display_units: style.display_units,
            scale: style.scale,
            cbar_tick_step: style.cbar_tick_step,
            legend: style.colormap_options.legend,
            density: style.colormap_options.render_density,
            convert: SheetConvert::Production(style.convert),
            contour_levels: Vec::new(),
        }
    }

    /// The product's own sheet levels, when it has them, over the
    /// production style; the colour bar then picks its ticks from them.
    fn for_sheet(mut self, spec: &ProductSpec) -> Self {
        if let Some(scale) = spec.sheet_scale {
            self.scale = rustwx_render::ColorScale::Discrete(scale());
            self.cbar_tick_step = None;
        }
        self
    }

    fn ladder(ladder: Ladder) -> Self {
        Self {
            display_units: ladder.units.to_string(),
            scale: ladder_scale(
                ladder.first,
                ladder.step,
                ladder.bins,
                &rustwx_render::weather::weather_palette(ladder.palette),
            ),
            cbar_tick_step: Some(2.0 * ladder.step),
            legend: LegendControls {
                density: rustwx_render::LevelDensity::default(),
                mode: rustwx_render::LegendMode::Stepped,
            },
            density: rustwx_render::RenderDensity::default(),
            convert: SheetConvert::Factor(ladder.stored_to_drawn),
            contour_levels: if ladder.contour_interval > 0.0 {
                let top = ladder.first + ladder.step * ladder.bins as f64;
                let count = ((top - ladder.first) / ladder.contour_interval).round() as usize;
                (0..=count)
                    .map(|index| ladder.first + index as f64 * ladder.contour_interval)
                    .collect()
            } else {
                Vec::new()
            },
        }
    }
}

struct RunField {
    values: Vec<f32>,
    units: String,
    variable: String,
    selector_json: serde_json::Value,
}

fn run_plane(source: &StoreFieldSource, selector: &FieldSelector) -> Result<RunField, String> {
    let variable = source
        .resolve(selector)
        .map(str::to_string)
        .ok_or_else(|| format!("the frame's store carries no {}", selector.key()))?;
    let field = source
        .fetch(selector)
        .map_err(|error| format!("read {variable}: {error}"))?;
    let meta = source
        .surface_variable(&variable)
        .ok_or_else(|| format!("{variable} vanished from the store"))?;
    Ok(RunField {
        values: field.values,
        units: meta.units.clone(),
        variable,
        selector_json: meta.selector.clone(),
    })
}

fn named_run_plane(source: &StoreFieldSource, variable: &str) -> Result<RunField, String> {
    let meta = source.surface_variable(variable)
        .ok_or_else(|| format!("the frame's store carries no {variable}"))?;
    let field = source.generic_grid(variable)
        .map_err(|error| format!("read {variable}: {error}"))?;
    Ok(RunField {
        values: field.values,
        units: meta.units.clone(),
        variable: variable.to_string(),
        selector_json: meta.selector.clone(),
    })
}

/// When each side of a sheet started and how far into its forecast it is.
///
/// The two sides share a VALID time and nothing else is promised: a run
/// compared with another cycle of the reference (its own cycle stops
/// short, or the question is about a later analysis) puts two different
/// starts and two different leads under one valid time.  Every label that
/// names a start or a lead is therefore made here, from both, so the sheet
/// cannot print one side's cycle over the other side's panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SheetTimes {
    run_init: DateTime<Utc>,
    run_lead_seconds: u64,
    reference_cycle: DateTime<Utc>,
    reference_lead: u16,
    valid: DateTime<Utc>,
}

/// `2026-10-03 00Z`, or `2026-10-03 00:30Z` off the hour.
fn hour_label(time: DateTime<Utc>) -> String {
    let day = format!("{:04}-{:02}-{:02}", time.year(), time.month(), time.day());
    if time.minute() == 0 {
        format!("{day} {:02}Z", time.hour())
    } else {
        format!("{day} {:02}:{:02}Z", time.hour(), time.minute())
    }
}

impl SheetTimes {
    fn same_cycle(&self) -> bool {
        self.run_init == self.reference_cycle
    }

    /// `F024`, or `F001h30m` for a run frame off its own whole hours.
    fn run_lead_label(&self) -> String {
        let hours = self.run_lead_seconds / 3_600;
        let minutes = self.run_lead_seconds % 3_600 / 60;
        if minutes == 0 {
            format!("F{hours:03}")
        } else {
            format!("F{hours:03}h{minutes:02}m")
        }
    }

    fn reference_lead_label(&self) -> String {
        format!("F{:03}", self.reference_lead)
    }

    /// What the run's panel says about itself.
    fn run_subtitle(&self) -> String {
        format!(
            "Init {} {}",
            hour_label(self.run_init),
            self.run_lead_label()
        )
    }

    /// The run panel's left subtitle when no station dots ride on it: its
    /// start and lead, then the valid time.  The run's history file name
    /// (`wrfout_d01_...`) is never shown; the file keeps its WRF-compatible
    /// name on disk, but a viewer reads it as "this picture is WRF".
    fn run_panel_subtitle(&self) -> String {
        format!("{} | Valid {}", self.run_subtitle(), hour_label(self.valid))
    }

    /// What the reference's panel says about itself.
    fn reference_subtitle(&self) -> String {
        format!(
            "Init {} {}",
            hour_label(self.reference_cycle),
            self.reference_lead_label()
        )
    }

    /// The header's time facts.  One start and one lead when the two sides
    /// share them; otherwise the valid time first and each side's own
    /// start and lead under its own name.
    fn header_facts(&self, run_label: &str, reference_label: &str) -> Vec<String> {
        let valid = format!("Valid {}", hour_label(self.valid));
        if self.same_cycle() {
            vec![
                format!("Init {}", hour_label(self.run_init)),
                self.run_lead_label(),
                valid,
            ]
        } else {
            vec![
                valid,
                format!(
                    "{run_label} init {} {}",
                    hour_label(self.run_init),
                    self.run_lead_label()
                ),
                format!(
                    "{reference_label} init {} {}",
                    hour_label(self.reference_cycle),
                    self.reference_lead_label()
                ),
            ]
        }
    }

    /// The file-name token: the run's own start and lead, and the
    /// reference's after it when they are not the same.
    fn stem_token(&self) -> String {
        let run = format!(
            "{:04}{:02}{:02}_{:02}z_{}",
            self.run_init.year(),
            self.run_init.month(),
            self.run_init.day(),
            self.run_init.hour(),
            self.run_lead_label().to_ascii_lowercase()
        );
        if self.same_cycle() {
            run
        } else {
            format!(
                "{run}_ref{:04}{:02}{:02}_{:02}z_{}",
                self.reference_cycle.year(),
                self.reference_cycle.month(),
                self.reference_cycle.day(),
                self.reference_cycle.hour(),
                self.reference_lead_label().to_ascii_lowercase()
            )
        }
    }
}

/// A panel's title and its two subtitle slots.
struct PanelText {
    title: String,
    left: String,
    right: String,
}

#[derive(serde::Deserialize)]
struct ObservationRow {
    #[serde(flatten)]
    spec: RadarSpec,
    label: String,
}

/// Request token, canonical quantity, coincidence window and canonical units.
const OBSERVATION_QUANTITY_ROWS: &[(&str, &str, i64, &str)] = &[
    ("composite_reflectivity", "composite_reflectivity", 240, "dBZ"),
    ("precipitation_accumulation", "precipitation_accumulation", 0, "mm"),
    ("precipitation_1h", "precipitation_accumulation", 0, "mm"),
];

fn observation_quantity_row(quantity: &str) -> Option<&'static (&'static str, &'static str, i64, &'static str)> {
    OBSERVATION_QUANTITY_ROWS.iter().find(|(token, _, _, _)| *token == quantity)
}

/// Observation inputs are rows keyed by quantity and the pack's valid time.
fn observed_panel(
    manifest: &Path,
    quantity: &str,
    valid: DateTime<Utc>,
    geometry: (&[f32], &[f32], Option<&GridProjection>, usize, usize),
    style: &SheetStyle,
) -> Result<Option<(PanelText, Vec<f32>, serde_json::Value)>, String> {
    let (_, canonical_quantity, window, input_units) = observation_quantity_row(quantity)
        .ok_or_else(|| format!("no observation quantity metadata for {quantity:?}"))?;
    let rows: Vec<ObservationRow> = serde_json::from_slice(
        &std::fs::read(manifest).map_err(|error| format!("read {}: {error}", manifest.display()))?
    ).map_err(|error| format!("parse {}: {error}", manifest.display()))?;
    for mut row in rows.into_iter().filter(|row| {
        observation_quantity_row(&row.spec.quantity).is_some_and(|(_, canonical, _, _)| canonical == canonical_quantity)
    }) {
        let root = manifest.parent().unwrap_or(Path::new("."));
        if row.spec.path.is_relative() { row.spec.path = root.join(&row.spec.path); }
        if row.spec.grid_path.is_relative() { row.spec.grid_path = root.join(&row.spec.grid_path); }
        let observed = load_radar(&row.spec)?;
        let stamp = observed.valid_time.trim_end_matches('Z');
        let observed_time = DateTime::parse_from_rfc3339(&format!("{stamp}Z"))
            .map_err(|error| format!("observation valid time {:?}: {error}", observed.valid_time))?
            .with_timezone(&Utc);
        let offset = (observed_time - valid).num_seconds();
        // Instantaneous products use the archive cadence. An accumulated
        // product must end at this exact hour, never the previous hour.
        if offset.abs() > *window { continue; }
        if !same_units(&observed.units, input_units) {
            return Err("observation pack units disagree with the product quantity".into());
        }
        let (lat, lon, projection, ny, nx) = geometry;
        let grid = GridData {lat: lat.to_vec(), lon: lon.to_vec(), projection: projection.cloned(), ny, nx};
        let native_values: Vec<f64> = observed.values.iter().zip(&observed.valid)
            .map(|(value, valid)| if *valid { *value } else { f64::NAN }).collect();
        let values = rw_wrfbatch::verification::mapped_fields(&grid, &observed.grid, &native_values)?;
        let missing = values.iter().filter(|value| !value.is_finite()).count();
        let values = values.into_iter().map(|value| style.convert.apply(value as f32)).collect();
        let metadata = serde_json::json!({
            "label": row.label, "quantity": observed.quantity, "valid": observed.valid_time,
            "canonical_quantity": canonical_quantity,
            "offset_seconds": offset, "path": row.spec.path, "geometry": row.spec.grid_path,
            "provenance": observed.provenance, "points_without_observation": missing,
            "sampling": "nearest grid point within the native observation lattice",
        });
        return Ok(Some((PanelText {
            title: row.label,
            left: format!("Valid {}", hour_label(observed_time)),
            right: format!("observed | offset {offset:+} s"),
        }, values, metadata)));
    }
    Ok(None)
}

fn station_marks(
    path: &Path,
    quantity: &str,
    valid: DateTime<Utc>,
    geometry: (&[f32], &[f32], Option<&GridProjection>, usize, usize),
    values: &[f32],
    style: &SheetStyle,
    error: bool,
) -> Result<MapOverlays, String> {
    let (lat, lon, projection, ny, nx) = geometry;
    let grid = GridData { lat: lat.to_vec(), lon: lon.to_vec(), projection: projection.cloned(), ny, nx };
    let values: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
    let valid_token = valid.format("%Y-%m-%dT%H:%M:%S").to_string();
    let dots = load_station_observations(path, &valid_token)?.into_iter().filter_map(|station| {
        let observed = station.values.get(quantity).copied()?;
        let position = grid.position(station.latitude, station.longitude)?;
        let forecast = grid.sample(&values, position);
        Some(StationDot {
            latitude: station.latitude, longitude: station.longitude,
            observed: f64::from(style.convert.apply(observed as f32)), forecast,
        })
    }).collect::<Vec<_>>();
    scalar_dots(&dots, &style.display_units, error)
}

/// Everything the panels of one sheet are drawn from.
struct PanelSet<'a> {
    /// The run grid: latitude, longitude, projection, rows, columns.  BOTH
    /// field panels and the difference are drawn on it, which is what
    /// makes their projection and extent one.
    geometry: (
        &'a [f32],
        &'a [f32],
        Option<&'a GridProjection>,
        usize,
        usize,
    ),
    width: u32,
    height: u32,
    slug: &'a str,
    /// The ONE colour treatment both field panels wear.
    style: &'a SheetStyle,
    scratch: &'a Path,
    run: PanelText,
    run_values: Vec<f32>,
    reference: PanelText,
    reference_values: Vec<f32>,
    run_overlays: Option<&'a MapOverlays>,
    reference_overlays: Option<&'a MapOverlays>,
    observation: Option<(PanelText, Vec<f32>)>,
    /// The third panel: its text, its plane and its ladder step.
    difference: Option<(PanelText, Vec<f32>, f64)>,
    /// The sheet's radar no-coverage footprint on the run grid: outlined on
    /// the run, forecast-reference and difference panels, filled on the
    /// observation panel.  `None` when the sheet observes nothing.
    coverage: Option<&'a [bool]>,
    /// The reference panel's own unobserved cells when the reference IS an
    /// observation: filled there instead of outlined.
    reference_coverage: Option<&'a [bool]>,
}

// Snow depth differences need an inch-scale ladder, rather than the tenth-inch
// ladder used for liquid precipitation. The quantity is declared in the table.
fn product_difference_step(spec: &ProductSpec, units: &str) -> Option<f64> {
    match (spec.observation_quantity, units) {
        (Some("snowfall_accumulation" | "snow_depth"), "in" | "inch" | "inches") => Some(2.0),
        _ => difference_step(units).or_else(|| extra_difference_step(spec.name, units)),
    }
}

/// The step of an 11-bin difference bar whose half range is about 2.5 times
/// the panel's RMS difference, rounded up to a 1, 2, 2.5 or 5 times a power
/// of ten.  None when nothing differs.
fn rms_difference_step(values: &[f32]) -> Option<f64> {
    let (mut n, mut sum) = (0f64, 0f64);
    for v in values {
        if v.is_finite() {
            n += 1.0;
            sum += f64::from(*v) * f64::from(*v);
        }
    }
    if n == 0.0 || sum <= 0.0 {
        return None;
    }
    let raw = 2.5 * (sum / n).sqrt() / 5.5;
    let decade = 10f64.powf(raw.log10().floor());
    [1.0, 2.0, 2.5, 5.0, 10.0].iter().map(|m| m * decade).find(|step| *step >= raw)
}

/// Difference steps for the extended product rows whose units the shared
/// table does not carry (11 bins, so the bar spans +-5.5 steps).
fn extra_difference_step(name: &str, units: &str) -> Option<f64> {
    let u = units.trim();
    Some(match u {
        "J/kg" | "J kg-1" => if name.contains("cin") { 25.0 } else { 200.0 },
        "m2/s2" | "m^2/s^2" | "m2 s-2" => 25.0,
        "m/s" | "m s-1" => 2.0,
        "kt" | "kts" => 4.0,
        "%" => 10.0,
        "mm" => if name.contains("snow") { 20.0 } else if name.contains("hail") { 5.0 } else { 2.0 },
        "kg/m^2" => 2.0,
        "in" | "inch" | "inches" => 0.25,
        "m" => if name.contains("visib") { 2000.0 } else if name.contains("terrain") { 20.0 } else { 100.0 },
        "mi" => 1.0,
        "km" => 2.0,
        "0/1" | "1" => 0.2,
        "1e-5 s-1" | "10^-5 s^-1" => 2.0,
        "hPa" | "mb" => 1.0,
        "Pa/s" => 0.2,
        "g/kg" => 0.5,
        "dam" => 1.0,
        "K" | "degC" | "C" => 1.0,
        "degF" | "F" => 2.0,
        "dBZ" => 5.0,
        _ if u.starts_with("flashes") => 0.5,
        _ => return None,
    })
}

fn comparison_difference_scale(theme: &rustwx_render::RenderTheme, step: f64) -> rustwx_render::ColorScale {
    let mut scale = difference_scale(step);
    if let rustwx_render::ColorScale::Discrete(discrete) = &mut scale {
        if let Some(colors) = theme.diverging_colors(discrete.colors.len()) {
            discrete.colors = colors;
        }
    }
    scale
}

fn comparison_product_key(theme: &rustwx_render::RenderTheme, slug: &str, panel: &str) -> String {
    if panel != "difference" && !theme.is_default() {
        slug.to_string()
    } else {
        format!("{slug}_{panel}")
    }
}

/// Draw a sheet's panels through the production panel path.
///
/// The run panel and the reference panel go through the SAME call with the
/// same grid, the same size and the same [`SheetStyle`]; nothing about
/// either plane reaches the other's scale, and nothing about the pair
/// reaches either (no range is fitted to the data).  That is the whole of
/// "same colour table and value range", and it is why a panel's pixels
/// depend only on its own plane.
/// Where and how big one sheet's panels are drawn: the run grid, the panel
/// size, the product slug the theme keys on, the sheet's one style and the
/// scratch directory the panel PNGs pass through.
#[derive(Clone, Copy)]
struct PanelFrame<'a> {
    geometry: (
        &'a [f32],
        &'a [f32],
        Option<&'a GridProjection>,
        usize,
        usize,
    ),
    width: u32,
    height: u32,
    slug: &'a str,
    style: &'a SheetStyle,
    scratch: &'a Path,
}

/// One panel through the production panel path.
///
/// The single call every comparison panel goes through, whether it sits in
/// a run-and-reference sheet or in an N-panel sheet: a panel's pixels depend
/// on its frame, its name (the theme's product key), its text and its plane,
/// and on nothing about the sheet it is composed into.
#[allow(clippy::too_many_arguments)]
fn draw_one_panel(
    frame: PanelFrame<'_>,
    theme: &rustwx_render::RenderTheme,
    name: &str,
    text: PanelText,
    values: Vec<f32>,
    scale: rustwx_render::ColorScale,
    tick: Option<f64>,
    legend: LegendControls,
    contour_levels: &[f64],
    overlays: Option<&MapOverlays>,
    coverage: CoverageDraw<'_>,
) -> Result<rustwx_render::RgbaImage, String> {
    let (lat, lon, projection, ny, nx) = frame.geometry;
    let style = frame.style;
    let mut contours = if contour_levels.is_empty() {
        Vec::new()
    } else {
        vec![rustwx_render::ContourLayer {
            data: values.clone(),
            levels: contour_levels.to_vec(),
            color: rustwx_render::Color::rgba(20, 20, 20, 255),
            width: 1,
            labels: true,
            show_extrema: false,
            pattern: Default::default(),
            major_every: None,
            major_width: None,
        }]
    };
    let fill = match coverage {
        CoverageDraw::None => None,
        CoverageDraw::Fill(mask) => Some(mask),
        CoverageDraw::Outline(mask) => {
            contours.extend(coverage::outline_layer(mask));
            None
        }
    };
    let themed_overlays = overlays.map(|overlays| {
        let mut overlays = overlays.clone();
        let key = rustwx_render::ProductKey::named(comparison_product_key(theme, frame.slug, name));
        let palette = theme.product_scale_override(&key, &scale).unwrap_or_else(|| scale.clone());
        for layer in &mut overlays.value_layers {
            if layer.scale.is_none() { layer.scale = Some(palette.clone()); }
        }
        overlays
    });
    let path = render_panel_with_unobserved(PanelRequest {
        lat_deg: lat,
        lon_deg: lon,
        projection,
        ny,
        nx,
        values,
        // Every themed field panel resolves the same product override.
        product_slug: comparison_product_key(theme, frame.slug, name),
        title: text.title,
        display_units: style.display_units.clone(),
        scale,
        cbar_tick_step: tick,
        legend,
        render_density: style.density,
        subtitle_left: text.left,
        subtitle_center: None,
        subtitle_right: text.right,
        width: frame.width,
        height: frame.height,
        contours,
        colorbar: true,
        overlays: themed_overlays.as_ref(),
        annotations: None,
        out_path: frame.scratch.join(format!("{name}.png")),
    }, fill)?;
    Ok(image::open(&path)
        .map_err(|error| format!("read back {}: {error}", path.display()))?
        .to_rgba8())
}

fn draw_panels(set: PanelSet<'_>) -> Result<Vec<rustwx_render::RgbaImage>, String> {
    std::fs::create_dir_all(set.scratch)
        .map_err(|error| format!("create {}: {error}", set.scratch.display()))?;
    let style = set.style;
    let theme = rustwx_render::active_theme();
    let frame = PanelFrame {
        geometry: set.geometry,
        width: set.width,
        height: set.height,
        slug: set.slug,
        style,
        scratch: set.scratch,
    };
    let panel = |name: &str,
                 text: PanelText,
                 values: Vec<f32>,
                 scale: rustwx_render::ColorScale,
                 tick: Option<f64>,
                 legend: LegendControls,
                 contour_levels: &[f64],
                 overlays: Option<&MapOverlays>,
                 coverage: CoverageDraw<'_>|
     -> Result<rustwx_render::RgbaImage, String> {
        draw_one_panel(
            frame, &theme, name, text, values, scale, tick, legend, contour_levels, overlays,
            coverage,
        )
    };
    let outline = set.coverage.map_or(CoverageDraw::None, CoverageDraw::Outline);

    let mut panels = Vec::with_capacity(3);
    let reference_coverage = set.reference_coverage.map_or(outline, CoverageDraw::Fill);
    for (name, text, values, overlays, coverage) in [
        ("run", set.run, set.run_values, set.run_overlays, outline),
        ("reference", set.reference, set.reference_values, set.reference_overlays, reference_coverage),
    ] {
        panels.push(panel(
            name,
            text,
            values,
            style.scale.clone(),
            style.cbar_tick_step,
            style.legend,
            &style.contour_levels,
            overlays,
            coverage,
        )?);
    }
    if let Some((text, values)) = set.observation {
        panels.push(panel(
            "observation", text, values, style.scale.clone(), style.cbar_tick_step,
            style.legend, &style.contour_levels, None,
            set.coverage.map_or(CoverageDraw::None, CoverageDraw::Fill),
        )?);
    }
    if let Some((text, values, step)) = set.difference {
        panels.push(panel(
            "difference",
            text,
            values,
            comparison_difference_scale(&theme, step),
            Some(step),
            LegendControls {
                density: style.legend.density,
                mode: rustwx_render::LegendMode::Stepped,
            },
            &[],
            None,
            outline,
        )?);
    }
    Ok(panels)
}

#[allow(clippy::too_many_arguments)]
fn write_sheet(
    args: &Args,
    reference: &ReferenceSpec,
    spec: &ProductSpec,
    frame: &Frame,
    times: SheetTimes,
    domain_token: &str,
    domain_label: Option<&str>,
    geometry: (&[f32], &[f32], Option<&GridProjection>, usize, usize),
    run_values: Vec<f32>,
    reference_values: Vec<f32>,
    style: &SheetStyle,
    grid_match: &GridMatch,
    reference_file: &ReferenceFile,
) -> Result<(PathBuf, Option<rw_wrfbatch::compare::DifferenceStats>), String> {
    let (_, _, _, ny, nx) = geometry;
    let units = style.display_units.clone();
    let mut step = product_difference_step(spec, &units);
    let draw_difference = match args.difference {
        DifferenceMode::Off => false,
        DifferenceMode::On => true,
        DifferenceMode::Auto => spec.continuous,
    } && step.is_some();
    // Below the field scale's own floor neither panel draws anything, so
    // two models' different spellings of "nothing" are not a difference.
    let floor = style.scale.resolved_discrete().mask_below;
    let diff_values = difference_above_floor(&run_values, &reference_values, floor);
    // RW_COMPARE_STEP=rms: the difference bar is fitted to this panel's own
    // RMS difference (half range about 2.5 RMS on a 1-2-2.5-5 ladder), so a
    // sub-unit difference is not hidden in the centre bin of a fixed bar.
    if std::env::var("RW_COMPARE_STEP").map(|v| v == "rms").unwrap_or(false) {
        if let Some(fitted) = rms_difference_step(&diff_values) {
            step = Some(fitted);
        }
    }
    // Reported only for the panel it describes.
    let stats = if draw_difference {
        difference_stats(&diff_values)
    } else {
        None
    };

    let valid = times.valid;
    let stem = format!(
        "{}_vs_{}_{}_{}_{}",
        safe_component(&args.run_label, "run"),
        reference.name,
        spec.slug,
        times.stem_token(),
        safe_component(domain_token, "native_grid"),
    );
    let scratch = args.store_root.join("panels").join(&stem);

    let station_error = args.station_mode == "error";
    let marks = |values: &[f32]| -> Result<Option<MapOverlays>, String> {
        match (args.stations.as_deref(), spec.station_quantity) {
            (Some(path), Some(quantity)) => station_marks(path, quantity, valid, geometry, values, style, station_error).map(Some),
            _ => Ok(None),
        }
    };
    let run_overlays = marks(&run_values)?;
    let reference_overlays = marks(&reference_values)?;
    let station_label = |overlays: Option<&MapOverlays>| overlays.map(|marks| {
        let count = marks.value_layers.first().map_or(0, |layer| layer.points.len());
        format!(" | station {} dots n={count}", if station_error { "obs minus forecast" } else { "observed" })
    }).unwrap_or_default();
    let observed = match (args.observations.as_deref(), spec.observation_quantity) {
        (Some(path), Some(quantity)) => observed_panel(path, quantity, valid, geometry, style)?,
        _ => None,
    };
    let observation_metadata = observed.as_ref().map(|(_, _, metadata)| metadata.clone());
    let observed_coverage = observed.as_ref().map(|(_, values, _)| coverage::mask_from_nan(values));
    let panels = draw_panels(PanelSet {
        geometry,
        width: args.width,
        height: args.height,
        slug: spec.slug,
        style,
        scratch: &scratch,
        run: PanelText {
            title: args.run_label.clone(),
            left: if run_overlays.is_some() {
                format!("{}{}", times.run_subtitle(), station_label(run_overlays.as_ref()))
            } else {times.run_panel_subtitle()},
            right: args.run_source_subtitle.clone(),
        },
        run_values,
        reference: PanelText {
            title: reference.label.to_string(),
            left: format!(
                "{} | {}{}",
                times.reference_subtitle(),
                spec.reference_message,
                station_label(reference_overlays.as_ref()),
            ),
            right: format!("source: {}", reference.source_label),
        },
        reference_values,
        run_overlays: run_overlays.as_ref(),
        reference_overlays: reference_overlays.as_ref(),
        observation: observed.map(|(text, values, _)| (text, values)),
        difference: draw_difference.then(|| {
            (
                PanelText {
                    title: format!("{} minus {}", args.run_label, reference.label),
                    left: grid_match.rule.describe(reference.label),
                    right: String::new(),
                },
                diff_values,
                step.expect("checked with draw_difference"),
            )
        }),
        coverage: observed_coverage.as_deref(),
        reference_coverage: None,
    })?;

    let mut facts = times.header_facts(&args.run_label, reference.label);
    if let Some(label) = domain_label {
        facts.push(label.to_string());
    }
    facts.push(grid_match.rule.describe(reference.label));
    if observed_coverage.is_some() {
        facts.push(NO_COVERAGE_LEGEND.to_string());
    }
    let sheet = compose_sheet(
        &panels,
        &SheetHeader {
            title: format!("{} ({units})", spec.title),
            subtitle: facts.join(" | "),
        },
    )?;
    let sheet = if station_error && run_overlays.is_some() {
        append_error_key(&sheet, &units, rustwx_render::ColormapBuildOptions {
            render_density: style.density, legend: style.legend,
        })?
    } else { sheet };

    let valid_day = format!(
        "{:04}-{:02}-{:02}",
        valid.year(),
        valid.month(),
        valid.day()
    );
    let out_path = match args.layout {
        Layout::Nested => layout_path(
            &args.out_dir,
            domain_token,
            &format!("compare_{}_{}", reference.name, spec.slug),
            &valid_day,
            &stem,
        ),
        Layout::Flat => args.out_dir.join(format!("{stem}.png")),
    };
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    rustwx_render::save_rgba_png_profile_with_options(
        &sheet,
        &out_path,
        &PngWriteOptions {
            compression: PngCompressionMode::default(),
        },
    )
    .map_err(|error| format!("write {}: {error}", out_path.display()))?;
    if let Some(flat) = &args.flat_dir {
        std::fs::create_dir_all(flat)
            .map_err(|error| format!("create {}: {error}", flat.display()))?;
        let copy = flat.join(format!("{stem}.png"));
        copy_flat_sheet(&out_path, &copy)?;
    }
    let _ = std::fs::remove_dir_all(&scratch);

    let mut panel_kinds = vec!["run", "reference"];
    if observation_metadata.is_some() { panel_kinds.push("observation"); }
    if draw_difference { panel_kinds.push("difference"); }
    let mut sidecar = serde_json::json!({
        "schema": "gpuwm.compare-sheet.v1",
        "product": spec.name,
        "slug": spec.slug,
        "title": spec.title,
        "display_units": units,
        "run_label": args.run_label,
        "run_source_label": args.source_label,
        "run_frame": frame.file,
        "reference": reference.name,
        "reference_file": reference_file.file_name,
        "reference_origin": reference_file.origin,
        "reference_message": spec.reference_message,
        "run_plane": match spec.run {
            RunPlane::Selector(_) | RunPlane::Named { .. } => "the run's stored product plane",
            RunPlane::SelectorSpeed(..) => "speed from the run's two stored component planes",
            RunPlane::HourlyAccumulation(_) =>
                "the run total differenced over the hour ending at the frame",
        },
        "valid": valid.to_rfc3339(),
        "run_init": times.run_init.to_rfc3339(),
        "run_lead_seconds": times.run_lead_seconds,
        "reference_cycle": times.reference_cycle.to_rfc3339(),
        "reference_forecast_hour": times.reference_lead,
        "same_cycle": times.same_cycle(),
        "domain": domain_token,
        "grid": {
            "ny": ny,
            "nx": nx,
            "rule": match grid_match.rule {
                MatchRule::Window { .. } => "window",
                MatchRule::Nearest => "nearest",
            },
            "window_origin": match grid_match.rule {
                MatchRule::Window {
                    i0,
                    j0,
                    rows_reversed,
                    columns_reversed,
                } => serde_json::json!({
                    "i0": i0,
                    "j0": j0,
                    "rows_reversed": rows_reversed,
                    "columns_reversed": columns_reversed,
                }),
                MatchRule::Nearest => serde_json::Value::Null,
            },
            "points_without_reference": grid_match.missing,
            "max_distance_km": grid_match.max_distance_km,
            "lattice": grid_match.lattice.map(|fit| serde_json::json!({
                "same_lattice": fit.same_lattice,
                "max_offset_cells": fit.max_offset_cells,
                "max_residual_cells": fit.max_residual_cells,
                "scale": [fit.scale.0, fit.scale.1],
                "anchor": [
                    fit.anchor.0.is_finite().then_some(fit.anchor.0),
                    fit.anchor.1.is_finite().then_some(fit.anchor.1),
                ],
            })),
        },
        "panels": panel_kinds,
        "observations": observation_metadata,
        "station_overlay": args.stations.as_ref().map(|path| serde_json::json!({
            "path": path, "mode": args.station_mode,
            "quantity": spec.station_quantity,
            "run_count": run_overlays.as_ref().and_then(|marks| marks.value_layers.first()).map(|layer| layer.points.len()),
            "reference_count": reference_overlays.as_ref().and_then(|marks| marks.value_layers.first()).map(|layer| layer.points.len()),
            "error_scale": run_overlays.as_ref().and_then(|marks| marks.value_layers.first()).and_then(|layer| layer.scale.as_ref()),
        })),
        "difference": stats.map(|stats| serde_json::json!({
            "points": stats.points,
            "mean": stats.mean,
            "rms": stats.rms,
            "max_abs": stats.max_abs,
        })),
        "sheet": out_path,
    });
    if let Some(theme) = &args.theme {
        sidecar["presentation"] = serde_json::json!({ "theme": theme,
            "run_title": args.run_label, "run_source_subtitle": args.run_source_subtitle });
    }
    let sidecar_path = out_path.with_extension("json");
    std::fs::write(
        &sidecar_path,
        serde_json::to_vec_pretty(&sidecar).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("write {}: {error}", sidecar_path.display()))?;
    Ok((out_path, stats))
}

struct ReferencePanel {
    name: String,
    label: String,
    source_label: String,
    subtitle: String,
    values: Vec<f32>,
    matched: GridMatch,
    receipt: serde_json::Value,
    /// An observed reference's unobserved cells on the run grid
    /// (`rw_wrfbatch::coverage`); `None` for a forecast reference.
    no_coverage: Option<Vec<bool>>,
}

fn observation_source_matches(metadata: &serde_json::Value, name: &str, label: &str) -> bool {
    metadata["provenance"]["source"].as_str().is_some_and(|source| source.eq_ignore_ascii_case(name))
        || metadata["label"].as_str().is_some_and(|source| source.eq_ignore_ascii_case(label))
}

fn packaged_reference(
    name: String,
    source_label: String,
    observed: (PanelText, Vec<f32>, serde_json::Value),
    geometry: (&[f32], &[f32], Option<&GridProjection>, usize, usize),
) -> Result<ReferencePanel, String> {
    let (text, values, mut receipt) = observed;
    let (lat, lon, _, ny, nx) = geometry;
    let matched = match_grids(lat, lon, ny, nx, lat, lon, ny, nx)?;
    receipt["name"] = serde_json::json!(name);
    receipt["kind"] = serde_json::json!("observation");
    // A pack's no-echo floor is a finite valid value, so its NaN cells are
    // exactly the cells nothing observed (or outside its lattice).
    let no_coverage = coverage::mask_from_nan(&values);
    receipt["no_coverage_cells"] = serde_json::json!(no_coverage.iter().filter(|cell| **cell).count());
    Ok(ReferencePanel {
        name, label: text.title, source_label,
        subtitle: format!("{} | {}", text.left, text.right), values, matched, receipt,
        no_coverage: Some(no_coverage),
    })
}

#[allow(clippy::too_many_arguments)]
fn render_reference_list(
    args: &Args,
    chosen: &[String],
    forecasts: &[ReferenceSpec],
    spec: &ProductSpec,
    frame: &Frame,
    times: SheetTimes,
    domain_token: &str,
    domain_label: Option<&str>,
    geometry: (&[f32], &[f32], Option<&GridProjection>, usize, usize),
    model_id: ModelId,
    run_field: &RunField,
) -> Result<Result<PathBuf, String>, String> {
    let (lat, lon, projection, ny, nx) = geometry;
    let ResolvedReferences { style, references } = match resolve_references(
        args, chosen, forecasts, spec, times, geometry, model_id, run_field,
    )? {
        Ok(resolved) => resolved,
        Err(reason) => return Ok(Err(reason)),
    };
    let convert = |values: &[f32]| {
        values
            .iter()
            .map(|value| style.convert.apply(*value))
            .collect::<Vec<_>>()
    };
    let run_values = convert(&run_field.values);
    render_reference_sheet(
        args, chosen, spec, frame, times, domain_token, domain_label,
        (lat, lon, projection, ny, nx), style, run_values, references,
    )
}

/// The sheet's one style and every reference's plane on the run grid.
struct ResolvedReferences {
    style: SheetStyle,
    references: Vec<ReferencePanel>,
}

/// Resolve the sheet style from the run's field and load, match and sample
/// every chosen reference onto the run grid.
///
/// Shared by the run-and-references sheet and the N-panel sheet, so a
/// reference panel is the same plane, on the same ladder, with the same
/// subtitle whichever sheet it is drawn into.
#[allow(clippy::too_many_arguments)]
fn resolve_references(
    args: &Args,
    chosen: &[String],
    forecasts: &[ReferenceSpec],
    spec: &ProductSpec,
    times: SheetTimes,
    geometry: (&[f32], &[f32], Option<&GridProjection>, usize, usize),
    model_id: ModelId,
    run_field: &RunField,
) -> Result<Result<ResolvedReferences, String>, String> {
    let observations = comparison_observations::specifications()?;
    // A sheet covers one product. Refuse a missing reference product explicitly
    // rather than substituting another quantity or silently omitting its panel.
    for name in chosen {
        if let Some(observation) = observations.iter().find(|row| &row.name == name) {
            if !observation
                .products
                .iter()
                .any(|row| row.product == spec.name)
            {
                return Ok(Err(format!(
                    "{} has no {} product in the reference table",
                    observation.label, spec.name
                )));
            }
        }
    }
    let (lat, lon, projection, ny, nx) = geometry;
    let style_variable = spec.style_variable.unwrap_or(&run_field.variable);
    let style = rustwx_products::viewer::operational_style_for_store_variable(
        style_variable,
        &run_field.selector_json,
        &run_field.units,
        model_id,
    )
    .or_else(|| {
        rustwx_products::viewer::curated_style_for_store_variable(
            style_variable,
            &run_field.selector_json,
            &run_field.units,
            model_id,
        )
    })
    .or_else(|| {
        spec.neutral_range.map(|range| {
            rustwx_products::viewer::generic_style_for_prescaled_store_variable(
                style_variable, &run_field.units, Some(range),
            )
        })
    });
    let style = match (style, spec.ladder) {
        (Some(style), _) => SheetStyle::production(style).for_sheet(spec),
        (None, Some(ladder)) => SheetStyle::ladder(ladder),
        _ => {
            return Ok(Err(format!(
                "no production colour scale resolves for {style_variable}"
            )));
        }
    };
    let convert = |values: &[f32]| {
        values
            .iter()
            .map(|value| style.convert.apply(*value))
            .collect::<Vec<_>>()
    };
    let mut packaged_observation = match (args.observations.as_deref(), spec.observation_quantity) {
        (Some(path), Some(quantity)) => observed_panel(path, quantity, times.valid, geometry, &style)?,
        _ => None,
    };
    let date = times.reference_cycle.format("%Y%m%d").to_string();
    let hour = times.reference_cycle.hour() as u8;
    let lead = times.reference_lead;
    let cache = args
        .reference_cache
        .clone()
        .unwrap_or_else(|| args.store_root.join("reference-cache"));
    let mut references = Vec::new();
    for name in chosen {
        let mut observed_coverage: Option<Vec<bool>> = None;
        if let Some(reference) = observations.iter().find(|row| &row.name == name) {
            let product = reference.products.iter().find(|row| row.product == spec.name).expect("supported above");
            if packaged_observation.as_ref().is_some_and(|(_, _, receipt)| {
                let offset = receipt["offset_seconds"].as_i64().unwrap_or(i64::MAX);
                observation_source_matches(receipt, &reference.name, &reference.label)
                    && match &product.time {
                        comparison_observations::TimePolicy::Nearest {tolerance_seconds} => offset.unsigned_abs() <= *tolerance_seconds as u64,
                        comparison_observations::TimePolicy::ExactHourEnd => offset == 0 && times.valid.minute() == 0 && times.valid.second() == 0,
                    }
            }) {
                references.push(packaged_reference(
                    name.clone(), reference.source_label.clone(), packaged_observation.take().expect("matched above"), geometry,
                )?);
                continue;
            }
        }
        let (label, source_label, subtitle, field, mut receipt) = if let Some(reference) =
            forecasts.iter().find(|row| row.name == name)
        {
            if lead > (reference.horizon)(hour) {
                return Ok(Err(format!(
                    "{} {hour:02}Z publishes to f{:03}, not f{lead:03}",
                    reference.label,
                    (reference.horizon)(hour)
                )));
            }
            let file = load_product_reference(args, reference, spec, &date, hour, lead)?;
            let field = match reference_field(&file.grib, spec.reference, lead)? {
                Ok(field) => field,
                Err(reason) => return Ok(Err(format!("{}: {reason}", reference.label))),
            };
            println!(
                "SOURCE\t{}\t{date}\t{hour:02}\tf{lead:03}\t{}",
                reference.name, file.origin
            );
            let subtitle = format!(
                "{} | {} | {}",
                times.reference_subtitle(),
                file.file_name,
                spec.reference_message
            );
            let receipt = serde_json::json!({
                "name": name, "label": reference.label, "kind": "forecast",
                "cycle": times.reference_cycle.to_rfc3339(), "forecast_hour": lead,
                "valid": times.valid.to_rfc3339(), "file": file.file_name, "origin": file.origin,
                "message": spec.reference_message, "units": field.units,
            });
            (
                reference.label.to_string(),
                reference.source_label.to_string(),
                subtitle,
                field,
                receipt,
            )
        } else {
            let reference = observations
                .iter()
                .find(|row| &row.name == name)
                .expect("validated reference");
            let product = reference
                .products
                .iter()
                .find(|row| row.product == spec.name)
                .expect("supported above");
            if matches!(spec.name, "snow10" | "snowkuchera") {
                let start = match std::env::var("GPUWM_SNOW_SINCE").ok().filter(|s| !s.trim().is_empty()) {
                    Some(s) => chrono::NaiveDateTime::parse_from_str(&s.replace(':', "_"), "%Y-%m-%d_%H_%M_%S")
                        .map_err(|e| format!("invalid snowfall baseline: {e}"))?.and_utc(),
                    None => times.run_init,
                };
                if Some((times.valid - start).num_seconds()) != product.accumulation_seconds.map(i64::from) {
                    return Err(format!("snowfall comparison window {start}..{} differs from analysis period {:?} s; set GPUWM_SNOW_SINCE to the exact matching saved baseline", times.valid, product.accumulation_seconds));
                }
            }
            let field = comparison_observations::load(
                reference,
                spec.name,
                times.valid,
                args.reference_dir.as_deref(),
                &cache,
                args.reference_file.as_deref(),
                args.offline,
            )?;
            let actual = field.valid.format("%Y-%m-%d %H:%M:%SZ").to_string();
            let period = product
                .accumulation_seconds
                .map(|seconds| format!(" | {} h ending", seconds / 3600))
                .unwrap_or_default();
            // The source file name stays in the receipt ("file", "origin");
            // in the subtitle an MRMS GRIB name overflowed a 1200 px panel.
            let reference_kind = if product.raster.is_some() { "Analysis" } else { "Observed" };
            let subtitle = format!("{reference_kind} {actual}{period} | {}", product.message);
            println!(
                "SOURCE\t{}\tobserved\t{}\t{}\t{}",
                reference.name,
                field.valid.to_rfc3339(),
                spec.name,
                field.origin
            );
            let receipt = serde_json::json!({
                "name": name, "label": reference.label, "kind": "observation",
                "valid": field.valid.to_rfc3339(), "target_valid": times.valid.to_rfc3339(),
                "offset_seconds": field.valid.timestamp() - times.valid.timestamp(),
                "accumulation_seconds": product.accumulation_seconds,
                "interval_start": product.accumulation_seconds.map(|seconds| (field.valid - chrono::Duration::seconds(i64::from(seconds))).to_rfc3339()),
                "file": field.file_name, "origin": field.origin, "candidate": field.candidate,
                "source_sha256": field.sha256, "message": product.message, "units": product.units,
                "parameter_table": reference.parameter_table, "missing_source_cells": field.missing_cells,
                "no_coverage_source_cells": field.unobserved.iter().filter(|cell| **cell).count(),
                "no_coverage_values": product.unobserved_values,
            });
            observed_coverage = Some(field.unobserved);
            (
                reference.label.clone(),
                reference.source_label.clone(),
                subtitle,
                ReferenceField {
                    values: field.values,
                    units: product.units.clone(),
                    lat: field.lat,
                    lon: field.lon,
                    ny: field.ny,
                    nx: field.nx,
                },
                receipt,
            )
        };
        if !same_units(&run_field.units, &field.units) {
            return Err(format!(
                "{}: run units {:?} differ from reference units {:?}",
                label, run_field.units, field.units
            ));
        }
        // Each source owns its match. Equal dimensions alone cannot establish
        // that two reference grids have the same coordinates or spacing.
        let matched = match_grids(lat, lon, ny, nx, &field.lat, &field.lon, field.ny, field.nx)?;
        println!(
            "MATCH\t{}\treference={name}\trun={ny}x{nx}\tsource={}x{}\tmissing={}\tmax_distance_km={:.3}\treference_spacing_km={:.3}",
            matched.rule.describe(&label),
            field.ny,
            field.nx,
            matched.missing,
            matched.max_distance_km,
            matched.source_spacing_km
        );
        if matched.missing == ny * nx {
            return Err(format!("no run grid point lies inside the {label} grid"));
        }
        receipt["grid"] = serde_json::json!({ "ny": field.ny, "nx": field.nx,
            "rule": matched.rule.describe(&label), "missing": matched.missing,
            "max_distance_km": matched.max_distance_km, "spacing_km": matched.source_spacing_km,
            "lattice": matched.lattice.map(|fit| serde_json::json!({"same_lattice": fit.same_lattice,
                "max_offset_cells": fit.max_offset_cells, "max_residual_cells": fit.max_residual_cells})) });
        let mut no_coverage = None;
        let sampled = if receipt["kind"] == "observation" {
            let target = GridData {lat:lat.to_vec(), lon:lon.to_vec(), ny, nx, projection:projection.cloned()};
            let source = GridData {lat:field.lat.clone(), lon:field.lon.clone(), ny:field.ny, nx:field.nx, projection:None};
            // One mapping carries the values and the coverage together: an
            // unobserved source cell, and a run point outside the source
            // lattice, both come back unobserved; an observed no-echo cell
            // comes back drawn as nothing but observed.
            let unobserved = observed_coverage.take().unwrap_or_else(|| coverage::mask_from_nan(&field.values));
            let mapped = rw_wrfbatch::verification::mapped_fields(&target, &source, &coverage::encode_for_mapping(&field.values, &unobserved))?;
            let (values, mask) = coverage::decode_mapped(&mapped);
            receipt["no_coverage_cells"] = serde_json::json!(mask.iter().filter(|cell| **cell).count());
            no_coverage = Some(mask);
            values
        } else { sample(&field.values, &matched) };
        receipt["defined_sampled_cells"] =
            serde_json::json!(sampled.iter().filter(|value| value.is_finite()).count());
        references.push(ReferencePanel {
            name: name.clone(),
            label,
            source_label,
            subtitle,
            values: convert(&sampled),
            matched,
            receipt,
            no_coverage,
        });
    }
    if let Some(observed) = packaged_observation {
        let source = observed.2["provenance"]["source"].as_str()
            .map(str::to_string).unwrap_or_else(|| safe_component(&observed.0.title, "observation"));
        if !references.iter().any(|reference| reference.receipt["kind"] == "observation"
            && observation_source_matches(&observed.2, &reference.name, &reference.label)) {
            let label = source.to_ascii_uppercase();
            references.push(packaged_reference(source, label, observed, geometry)?);
        }
    }
    Ok(Ok(ResolvedReferences { style, references }))
}

/// The run panel beside every resolved reference (and a difference panel
/// per reference when one is drawn), composed under one header.
#[allow(clippy::too_many_arguments)]
fn render_reference_sheet(
    args: &Args,
    chosen: &[String],
    spec: &ProductSpec,
    frame: &Frame,
    times: SheetTimes,
    domain_token: &str,
    domain_label: Option<&str>,
    geometry: (&[f32], &[f32], Option<&GridProjection>, usize, usize),
    style: SheetStyle,
    run_values: Vec<f32>,
    references: Vec<ReferencePanel>,
) -> Result<Result<PathBuf, String>, String> {
    let (lat, lon, projection, ny, nx) = geometry;
    let stem = format!(
        "{}_vs_{}_{}_{}_{}",
        safe_component(&args.run_label, "run"),
        chosen.join("_"),
        spec.slug,
        times.stem_token(),
        safe_component(domain_token, "native_grid")
    );
    let scratch = args.store_root.join("panels").join(&stem);
    let mut panels = Vec::new();
    let mut panel_names = vec!["run".to_string()];
    let mut receipts = Vec::new();
    let step = product_difference_step(spec, &style.display_units);
    let station_error = args.station_mode == "error";
    let marks = |values: &[f32]| -> Result<Option<MapOverlays>, String> {
        match (args.stations.as_deref(), spec.station_quantity) {
            (Some(path), Some(quantity)) => station_marks(path, quantity, times.valid, geometry, values, &style, station_error).map(Some),
            _ => Ok(None),
        }
    };
    let run_overlays = marks(&run_values)?;
    let station_label = |overlays: Option<&MapOverlays>| overlays.map(|marks| {
        let count = marks.value_layers.first().map_or(0, |layer| layer.points.len());
        format!(" | station {} dots n={count}", if station_error {"obs minus forecast"} else {"observed"})
    }).unwrap_or_default();
    let draw_difference = step.is_some()
        && match args.difference {
            DifferenceMode::On => true,
            DifferenceMode::Off => false,
            DifferenceMode::Auto => chosen.len() == 1 && spec.continuous,
        };
    // The first observed reference's footprint is the sheet's: outlined on
    // the run panel whichever reference it is drawn beside.
    let sheet_coverage: Option<Vec<bool>> =
        references.iter().find_map(|reference| reference.no_coverage.clone());
    for reference in references {
        let reference_overlays = marks(&reference.values)?;
        let diff_values = difference_above_floor(
            &run_values,
            &reference.values,
            style.scale.resolved_discrete().mask_below,
        );
        let stats = if draw_difference {
            difference_stats(&diff_values)
        } else {
            None
        };
        let mut images = draw_panels(PanelSet {
            geometry: (lat, lon, projection, ny, nx),
            width: args.width,
            height: args.height,
            slug: spec.slug,
            style: &style,
            scratch: &scratch.join(&reference.name),
            run: PanelText {
                title: args.run_label.clone(),
                left: if run_overlays.is_some() {
                    format!("{}{}", times.run_subtitle(), station_label(run_overlays.as_ref()))
                } else {times.run_panel_subtitle()},
                right: args.run_source_subtitle.clone(),
            },
            run_values: run_values.clone(),
            reference: PanelText {
                title: reference.label.clone(),
                left: format!("{}{}", reference.subtitle, station_label(reference_overlays.as_ref())),
                right: format!("source: {}", reference.source_label),
            },
            reference_values: reference.values,
            run_overlays: run_overlays.as_ref(),
            reference_overlays: reference_overlays.as_ref(),
            observation: None,
            difference: draw_difference.then(|| {
                (
                    PanelText {
                        title: format!("{} minus {}", args.run_label, reference.label),
                        left: reference.matched.rule.describe(&reference.label),
                        right: String::new(),
                    },
                    diff_values,
                    step.expect("checked"),
                )
            }),
            coverage: sheet_coverage.as_deref(),
            reference_coverage: reference.no_coverage.as_deref(),
        })?;
        if panels.is_empty() {
            panels.push(images.remove(0));
        } else {
            images.remove(0);
        }
        panels.push(images.remove(0));
        panel_names.push(reference.name.clone());
        if draw_difference {
            panels.push(images.remove(0));
            panel_names.push(format!("run_minus_{}", reference.name));
        }
        let mut receipt = reference.receipt;
        receipt["station_count"] = serde_json::json!(reference_overlays.as_ref().and_then(|marks|marks.value_layers.first()).map(|layer|layer.points.len()));
        receipt["difference"] = stats
            .map(|stats| {
                serde_json::json!({"points": stats.points, "mean": stats.mean,
            "rms": stats.rms, "max_abs": stats.max_abs})
            })
            .unwrap_or(serde_json::Value::Null);
        if let Some(stats) = stats {
            println!(
                "STATS\t{}\t{}\treference={}\tunits={}\tpoints={}\tmean={:.4}\trms={:.4}\tmax_abs={:.4}",
                spec.name,
                times.run_lead_label().to_ascii_lowercase(),
                reference.name,
                style.display_units,
                stats.points,
                stats.mean,
                stats.rms,
                stats.max_abs
            );
        }
        receipts.push(receipt);
    }
    let mut facts = vec![
        format!("Valid {}", hour_label(times.valid)),
        times.run_subtitle(),
    ];
    if let Some(label) = domain_label {
        facts.push(label.to_string());
    }
    facts.push(format!("References: {}", chosen.join(", ")));
    if sheet_coverage.is_some() {
        facts.push(NO_COVERAGE_LEGEND.to_string());
    }
    let sheet = compose_sheet(
        &panels,
        &SheetHeader {
            title: format!("{} ({})", spec.title, style.display_units),
            subtitle: facts.join(" | "),
        },
    )?;
    let sheet = if station_error && run_overlays.is_some() {
        append_error_key(&sheet, &style.display_units, rustwx_render::ColormapBuildOptions {
            render_density:style.density, legend:style.legend,
        })?
    } else {sheet};
    let day = times.valid.format("%Y-%m-%d").to_string();
    let path = match args.layout {
        Layout::Nested => layout_path(
            &args.out_dir,
            domain_token,
            &format!("compare_{}_{}", chosen.join("_"), spec.slug),
            &day,
            &stem,
        ),
        Layout::Flat => args.out_dir.join(format!("{stem}.png")),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    rustwx_render::save_rgba_png_profile_with_options(
        &sheet,
        &path,
        &PngWriteOptions {
            compression: PngCompressionMode::default(),
        },
    )
    .map_err(|error| format!("write {}: {error}", path.display()))?;
    if let Some(flat) = &args.flat_dir {
        std::fs::create_dir_all(flat).map_err(|error| error.to_string())?;
        copy_flat_sheet(&path, &flat.join(format!("{stem}.png")))?;
    }
    let mut receipt = serde_json::json!({ "schema": "gpuwm.compare-sheet.v2", "product": spec.name,
        "slug": spec.slug, "title": spec.title, "display_units": style.display_units,
        "run_label": args.run_label, "run_source_label": args.source_label, "run_frame": frame.file,
        "run_init": times.run_init.to_rfc3339(), "run_lead_seconds": frame.lead_seconds,
        "valid": times.valid.to_rfc3339(), "domain": domain_token, "run_grid": {"ny": ny, "nx": nx},
        "references": receipts, "panels": panel_names, "sheet": path,
        "station_overlay": args.stations.as_ref().map(|path| serde_json::json!({
            "path":path, "mode":args.station_mode, "quantity":spec.station_quantity,
            "run_count":run_overlays.as_ref().and_then(|marks|marks.value_layers.first()).map(|layer|layer.points.len()),
            "error_scale":run_overlays.as_ref().and_then(|marks|marks.value_layers.first()).and_then(|layer|layer.scale.as_ref()),
        })) });
    if let Some(theme) = &args.theme {
        receipt["presentation"] = serde_json::json!({ "theme": theme,
            "run_title": args.run_label, "run_source_subtitle": args.run_source_subtitle });
    }
    std::fs::write(
        path.with_extension("json"),
        serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(Ok(path))
}

/// Labels belong to each source, while presentation belongs to the whole sheet.
/// Resolve the run's theme labels once, then leave reference labels intact.
fn comparison_presentation(mut theme: rustwx_render::RenderTheme, run_label: &str,
                           source_subtitle: String) -> (rustwx_render::RenderTheme, String, String) {
    let run_label = theme.model_name(run_label);
    let source_subtitle = theme.source_subtitle(Some(source_subtitle)).unwrap_or_default();
    theme.model_label = None;
    theme.source_label = None;
    (theme, run_label, source_subtitle)
}

fn run(mut args: Args) -> Result<(), String> {
    let references = reference_specs();
    let chosen = if args.reference.is_empty() {
        Vec::new()
    } else {
        reference_names(&args.reference)?
    };
    let first_forecast = references
        .iter()
        .find(|spec| chosen.iter().any(|name| name == spec.name));
    let has_forecast = first_forecast.is_some();
    let single_forecast = chosen.len() == 1 && has_forecast;
    let reference = references
        .iter()
        .find(|spec| chosen.iter().any(|name| name == spec.name))
        .unwrap_or(&references[0]);
    if args.fetch_reference {
        let (date, cycle_hour) = args.cycle.as_ref().ok_or("--fetch-reference needs --cycle")?;
        let lead = args.forecast_hour.ok_or("--fetch-reference needs --forecast-hour")?;
        if lead > (reference.horizon)(*cycle_hour) {
            return Err("requested reference lead is outside the cycle's published horizon".into());
        }
        let file = load_reference(&args, reference, date, *cycle_hour, lead)?;
        verify_reference(&file, date, *cycle_hour, lead)?;
        println!("{}", serde_json::json!({
            "schema": "gpuwm.reference-input.v1",
            "path": file.path,
            "label": reference.label,
            "model": reference.name,
            "cycle": format!("{date}{cycle_hour:02}"),
            "hour": lead,
            "origin": file.origin,
        }));
        return Ok(());
    }
    let specs: Vec<ProductSpec> = product_specs()
        .into_iter()
        .filter(|spec| args.products.iter().any(|name| name == spec.name))
        .collect();

    rustwx_render::theme::set_template_version(args.source_label.split_whitespace().last()
        .filter(|token| token.starts_with(|character: char| character.is_ascii_digit())).map(str::to_string));
    let theme = args.theme.as_deref().map(rustwx_render::RenderTheme::resolve).transpose()?
        .unwrap_or_else(rustwx_render::RenderTheme::default_theme);
    let (theme, run_label, run_source_subtitle) = comparison_presentation(theme,
        &args.run_label, args.run_source_subtitle.clone());
    args.run_label = run_label;
    args.run_source_subtitle = run_source_subtitle;
    rustwx_render::install_theme(theme)?;

    if !args.panels.is_empty() {
        return compare_panels::run(&args, &specs, &references);
    }

    // --- import: the compared frames and their context, ONE store -------
    //
    // The context frames are the same run's earlier frames, wanted only so
    // an hourly accumulation can be differenced; they enter the same
    // import because rw-store merges frames written into one run and that
    // is exactly what a run's own frames are.
    let options = import_options(&specs);
    let mut all_inputs = args.inputs.clone();
    let wants_accumulation = specs
        .iter()
        .any(|spec| matches!(spec.run, RunPlane::HourlyAccumulation(_)));
    let siblings: Vec<PathBuf> = if wants_accumulation {
        args.inputs
            .iter()
            .filter_map(|path| sibling_an_hour_earlier(path))
            .collect()
    } else {
        Vec::new()
    };
    for path in args.context.iter().chain(siblings.iter()) {
        if !all_inputs.iter().any(|have| same_file(have, path)) {
            all_inputs.push(path.clone());
        }
    }
    let store_root = args.store_root.join("run");
    let summary = import(all_inputs, &store_root, options)?;
    for note in &summary.notes {
        eprintln!("IMPORT_NOTE\t{note}");
    }

    let mut frames: Vec<Frame> = Vec::new();
    for (slot, file) in &summary.frame_sources {
        let source = StoreFieldSource::open(&store_root, &summary.model, &summary.run, *slot)
            .map_err(|error| format!("open the frame of {}: {error}", file.display()))?;
        let (lead_seconds, valid_unix) = frame_time(source.exact_time(), &summary.run, *slot)
            .ok_or_else(|| {
                format!(
                    "{} was stored with no valid time this binary can read (run {:?}, slot \
                     {slot}); a frame with no valid time cannot be paired with a reference \
                     forecast",
                    file.display(),
                    summary.run
                )
            })?;
        frames.push(Frame {
            slot: *slot,
            file: file.clone(),
            lead_seconds,
            valid_unix,
            compared: args.inputs.iter().any(|input| same_file(input, file)),
        });
    }
    frames.sort_by_key(|frame| frame.valid_unix);
    let by_valid: BTreeMap<i64, u16> = frames
        .iter()
        .map(|frame| (frame.valid_unix, frame.slot))
        .collect();
    let model_id: ModelId = summary
        .model
        .parse()
        .map_err(|error| format!("store model slug {:?}: {error}", summary.model))?;

    let mut rendered = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut grid_match: Option<(usize, usize, GridMatch)> = None;

    for frame in frames.iter().filter(|frame| frame.compared) {
        let frame_label = frame.file.display().to_string();
        let valid = utc(frame.valid_unix)?;
        // The cycle is the run's own start unless the caller names another
        // (a run started from one model's analysis and compared with a
        // different cycle of the reference).
        let cycle = match args.cycle.as_ref().filter(|_| has_forecast) {
            Some((date, hour)) => {
                let text = format!(
                    "{}-{}-{}T{hour:02}:00:00Z",
                    &date[..4],
                    &date[4..6],
                    &date[6..8]
                );
                DateTime::parse_from_rfc3339(&text)
                    .map_err(|error| format!("--cycle: {error}"))?
                    .with_timezone(&Utc)
            }
            None => utc(frame.valid_unix - frame.lead_seconds as i64)?,
        };
        let lead_seconds = frame.valid_unix - cycle.timestamp();
        if has_forecast
            && (lead_seconds < 0
                || lead_seconds % 3_600 != 0
                || cycle.minute() != 0
                || cycle.second() != 0)
        {
            skipped += specs.len();
            println!(
                "SKIPPED\t*\t{frame_label}\tvalid {} is not a whole number of hours after cycle {}; \
                 the reference publishes hourly forecasts",
                valid.to_rfc3339(),
                cycle.to_rfc3339()
            );
            continue;
        }
        let lead = u16::try_from(if has_forecast {
            lead_seconds / 3_600
        } else {
            0
        })
        .map_err(|_| format!("{frame_label}: lead {lead_seconds} s is out of range"))?;
        let date = format!("{:04}{:02}{:02}", cycle.year(), cycle.month(), cycle.day());
        let cycle_hour = cycle.hour() as u8;

        let times = SheetTimes {
            run_init: utc(frame.valid_unix - frame.lead_seconds as i64)?,
            run_lead_seconds: frame.lead_seconds,
            reference_cycle: cycle,
            reference_lead: lead,
            valid,
        };
        let horizon = (reference.horizon)(cycle_hour);
        if single_forecast && lead > horizon {
            // Asked of the reference's own schedule before anything is
            // fetched: a lead the cycle never publishes is not a download
            // that failed.
            let offer = match cycle_reaching(reference, times.run_init, valid) {
                Some((later, later_lead)) => format!(
                    "The {later} cycle reaches this valid time at f{later_lead:02}: compare \
                     against it with --cycle {later} (gpuwm render: --compare-cycle {later})"
                ),
                None => format!(
                    "No {} cycle that starts at or after the run does reach this valid time",
                    reference.label
                ),
            };
            skipped += specs.len();
            println!(
                "SKIPPED\t*\t{frame_label}\t{}'s {cycle_hour:02}Z cycle publishes to f{horizon:02} \
                 and this frame is its f{lead:02}. {offer}",
                reference.label
            );
            continue;
        }
        let reference_file = if single_forecast {
            Some(
                match load_reference(&args, reference, &date, cycle_hour, lead).and_then(|file| {
                    verify_reference(&file, &date, cycle_hour, lead).map(|()| file)
                }) {
                    Ok(file) => file,
                    Err(message) => {
                        failed += specs.len();
                        eprintln!("FAILED\t*\t{frame_label}\t{message}");
                        continue;
                    }
                },
            )
        } else {
            None
        };
        if let Some(reference_file) = &reference_file {
            println!(
                "SOURCE\t{}\t{date}\t{cycle_hour:02}\tf{lead:03}\t{}",
                reference.name, reference_file.origin
            );
        }

        let source = StoreFieldSource::open(&store_root, &summary.model, &summary.run, frame.slot)
            .map_err(|error| format!("{frame_label}: open store: {error}"))?;
        let (lat, lon) = {
            let (lat, lon) = source.grid_coordinates();
            (lat.to_vec(), lon.to_vec())
        };
        let grid = source.full_grid();
        let (ny, nx) = (grid.shape.ny, grid.shape.nx);
        drop(grid);
        let projection = source.projection().cloned();
        let (domain_token, domain_label) = domain_tokens(&frame.file);
        if std::env::var_os("RW_COMPARE_DUMP_STORE").is_some() {
            for var in source.surface_variables() {
                println!("STOREVAR\t{}\t{}\t{}", var.name, var.units, var.selector);
            }
            for slug in source.derived_slugs() {
                println!("STORESLUG\t{slug}");
            }
        }
        if std::env::var_os("RW_COMPARE_INVENTORY").is_some() {
            if let Some(file) = &reference_file {
                for (i, m) in file.grib.messages.iter().enumerate() {
                    let p = &m.product;
                    println!("INV\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{}\t{}", i + 1, m.discipline, p.parameter_category, p.parameter_number, p.level_type, p.level_value, p.second_level_type, p.second_level_value, p.statistical_process_type, p.time_range_unit, p.forecast_time);
                }
            }
        }

        for spec in &specs {
            let outcome = (|| -> Result<Result<PathBuf, String>, String> {
                // --- the run's plane ------------------------------------
                let run_field = match read_run_field(
                    &source, spec, frame.lead_seconds, frame.valid_unix, &by_valid,
                    &store_root, &summary,
                )? {
                    Ok(field) => field,
                    Err(reason) => return Ok(Err(reason)),
                };

                if !single_forecast {
                    return render_reference_list(
                        &args,
                        &chosen,
                        &references,
                        spec,
                        frame,
                        times,
                        &domain_token,
                        domain_label.as_deref(),
                        (&lat, &lon, projection.as_ref(), ny, nx),
                        model_id,
                        &run_field,
                    );
                }

                // --- the reference's plane, on its own grid -------------
                let initial_file = reference_file.as_ref().expect("single forecast file");
                let fallback_file;
                let mut product_file = initial_file;
                let mut extracted = reference_field(&product_file.grib, spec.reference, lead)?;
                if extracted.is_err()
                    && matches!(
                        spec.run,
                        RunPlane::Selector(FieldSelector { vertical: rustwx_core::VerticalSelector::IsobaricHpa(_), .. }) | RunPlane::SelectorSpeed(FieldSelector { vertical: rustwx_core::VerticalSelector::IsobaricHpa(_), .. }, _)
                    )
                    && args.reference_file.is_none()
                {
                    fallback_file =
                        load_product_reference(&args, reference, spec, &date, cycle_hour, lead)?;
                    product_file = &fallback_file;
                    extracted = reference_field(&product_file.grib, spec.reference, lead)?;
                    println!(
                        "SOURCE\t{}\t{date}\t{cycle_hour:02}\tf{lead:03}\t{}",
                        reference.name, product_file.origin
                    );
                }
                let reference_plane = match extracted {
                    Ok(field) => field,
                    Err(reason) => return Ok(Err(reason)),
                };
                if !same_units(&run_field.units, &reference_plane.units) {
                    return Err(format!(
                        "the run stores {} in {:?} and the reference publishes {:?}; drawing both \
                         on one ladder would put two units on one colour bar",
                        spec.title, run_field.units, reference_plane.units
                    ));
                }

                // --- one grid ------------------------------------------
                let reusable = matches!(
                    &grid_match,
                    Some((have_ny, have_nx, _))
                        if *have_ny == reference_plane.ny && *have_nx == reference_plane.nx
                );
                if !reusable {
                    let matched = match_grids(
                        &lat,
                        &lon,
                        ny,
                        nx,
                        &reference_plane.lat,
                        &reference_plane.lon,
                        reference_plane.ny,
                        reference_plane.nx,
                    )?;
                    let rule = match matched.rule {
                        MatchRule::Window {
                            i0,
                            j0,
                            rows_reversed,
                            columns_reversed,
                        } => format!(
                            "window\ti0={i0}\tj0={j0}\trows_reversed={rows_reversed}\tcolumns_reversed={columns_reversed}"
                        ),
                        MatchRule::Nearest => "nearest".to_string(),
                    };
                    let lattice = match matched.lattice {
                        Some(fit) => format!(
                            "\tmax_offset_cells={:.3}\tscale={:.2e},{:.2e}\tanchor={:.0},{:.0}\tresidual_cells={:.3}",
                            fit.max_offset_cells,
                            fit.scale.0,
                            fit.scale.1,
                            fit.anchor.0,
                            fit.anchor.1,
                            fit.max_residual_cells
                        ),
                        None => String::new(),
                    };
                    println!(
                        "MATCH\t{rule}\trun={ny}x{nx}\treference={}x{}\tmissing={}\tmax_distance_km={:.3}\treference_spacing_km={:.3}{lattice}",
                        reference_plane.ny,
                        reference_plane.nx,
                        matched.missing,
                        matched.max_distance_km,
                        matched.source_spacing_km
                    );
                    grid_match = Some((reference_plane.ny, reference_plane.nx, matched));
                }
                let matched = &grid_match.as_ref().expect("set above").2;
                if matched.missing == ny * nx {
                    return Err(format!(
                        "no point of the run grid lies inside the {} grid",
                        reference.label
                    ));
                }
                let reference_on_run = sample(&reference_plane.values, matched);

                // --- one ladder ----------------------------------------
                let style_variable = spec.style_variable.unwrap_or(&run_field.variable);
                let style = rustwx_products::viewer::operational_style_for_store_variable(
                    style_variable,
                    &run_field.selector_json,
                    &run_field.units,
                    model_id,
                )
                .or_else(|| {
                    rustwx_products::viewer::curated_style_for_store_variable(
                        style_variable,
                        &run_field.selector_json,
                        &run_field.units,
                        model_id,
                    )
                })
                .or_else(|| {
                    spec.neutral_range.map(|range| {
                        rustwx_products::viewer::generic_style_for_prescaled_store_variable(
                            style_variable, &run_field.units, Some(range),
                        )
                    })
                });
                let style = match (style, spec.ladder) {
                    (Some(style), _) => SheetStyle::production(style).for_sheet(spec),
                    (None, Some(ladder)) => SheetStyle::ladder(ladder),
                    (None, None) => {
                        return Ok(Err(format!(
                            "no production colour scale resolves for {style_variable} and the \
                             product table gives it no ladder; a scale invented here would not be \
                             one any chart of this field is read on"
                        )));
                    }
                };
                let convert = |values: &[f32]| -> Vec<f32> {
                    values
                        .iter()
                        .map(|value| style.convert.apply(*value))
                        .collect()
                };
                {
                    let a = convert(&run_field.values);
                    let b = convert(&reference_on_run);
                    let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab, mut sd, mut sdd) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
                    let floor = style.scale.resolved_discrete().mask_below;
                    for (x, y) in a.iter().zip(b.iter()) {
                        if !x.is_finite() || !y.is_finite() { continue; }
                        let (mut x, mut y) = (*x as f64, *y as f64);
                        if let Some(f) = floor {
                            if x < f && y < f { continue; }
                            x = x.max(f); y = y.max(f);
                        }
                        n += 1.0; sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
                        sd += x - y; sdd += (x - y) * (x - y);
                    }
                    if n > 0.0 {
                        let cov = sab / n - (sa / n) * (sb / n);
                        let va = saa / n - (sa / n).powi(2);
                        let vb = sbb / n - (sb / n).powi(2);
                        let corr = if va > 0.0 && vb > 0.0 { cov / (va * vb).sqrt() } else { f64::NAN };
                        println!("FULLSTATS\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}", spec.name, times.run_lead_label().to_ascii_lowercase(), style.display_units, n as u64, sd / n, (sdd / n).sqrt(), corr, sa / n, sb / n);
                    }
                }
                let (path, stats) = write_sheet(
                    &args,
                    reference,
                    spec,
                    frame,
                    times,
                    &domain_token,
                    domain_label.as_deref(),
                    (&lat, &lon, projection.as_ref(), ny, nx),
                    convert(&run_field.values),
                    convert(&reference_on_run),
                    &style,
                    matched,
                    product_file,
                )?;
                if let Some(stats) = stats {
                    println!(
                        "STATS\t{}\t{}\tunits={}\tpoints={}\tmean={:.4}\trms={:.4}\tmax_abs={:.4}",
                        spec.name,
                        times.run_lead_label().to_ascii_lowercase(),
                        style.display_units,
                        stats.points,
                        stats.mean,
                        stats.rms,
                        stats.max_abs
                    );
                }
                Ok(Ok(path))
            })();
            match outcome {
                Ok(Ok(path)) => {
                    rendered += 1;
                    println!(
                        "RENDERED\t{}\t{}\t{}",
                        spec.name,
                        times.run_lead_label().to_ascii_lowercase(),
                        path.display()
                    );
                }
                Ok(Err(reason)) => {
                    skipped += 1;
                    println!("SKIPPED\t{}\t{frame_label}\t{reason}", spec.name);
                }
                Err(message) => {
                    failed += 1;
                    eprintln!("FAILED\t{}\t{frame_label}\t{message}", spec.name);
                }
            }
        }
    }

    println!("FINISHED rendered={rendered} skipped={skipped} failed={failed}");
    if rendered == 0 || failed > 0 {
        return Err(format!(
            "comparison incomplete: rendered={rendered} skipped={skipped} failed={failed}"
        ));
    }
    Ok(())
}

/// The import options for a set of products: the core fields, the chart
/// selectors the isobaric products need, and the raw diagnostics the named
/// products read.  One rule for every store a comparison imports into.
fn import_options(specs: &[ProductSpec]) -> WrfProcessOptions {
    let chart_selectors: Vec<FieldSelector> = specs
        .iter()
        .flat_map(|spec| match spec.run {
            RunPlane::Selector(selector)
                if matches!(
                    selector.vertical,
                    rustwx_core::VerticalSelector::IsobaricHpa(_)
                ) =>
            {
                vec![selector]
            }
            RunPlane::SelectorSpeed(u, v) => vec![u, v],
            _ => Vec::new(),
        })
        .collect();
    let raw_names: Vec<&str> = specs.iter().filter_map(|spec| match spec.run {
        RunPlane::Named { raw, .. } => Some(raw),
        _ => None,
    }).collect();
    let raw_skip = if raw_names.is_empty() {
        Vec::new()
    } else {
        rw_wrfbatch::wrf_process::RAW_EXTRA_CATALOG.iter()
            .filter(|raw| !raw_names.contains(raw))
            .map(|raw| raw.to_string()).collect()
    };
    let mut options = WrfProcessOptions {
        core_fields: true,
        diagnostics: false,
        heavy_ecape: false,
        raw_extras: !raw_names.is_empty(),
        stored_planes: false,
        only: Vec::new(),
        skip: raw_skip,
        viewer_2d: true,
        chart_selectors,
        named_products_only: false,
        snow_since: std::env::var("GPUWM_SNOW_SINCE").ok().filter(|s| !s.trim().is_empty()),
        required_fields: None,
    };
    if std::env::var_os("RW_COMPARE_FULL").is_some() {
        options.diagnostics = true;
        options.raw_extras = true;
        options.skip = Vec::new();
    }
    // Snow planes are computed diagnostics, rather than raw WRF variables.
    // Admit only the requested snow planes alongside the existing import plan;
    // enabling every diagnostic would make a comparison compute unrelated fields.
    let snow_names: Vec<String> = specs.iter().filter_map(|spec| {
        rw_wrfbatch::snowfall::PRODUCTS.iter()
            .find(|(slug, _)| *slug == spec.slug)
            .map(|(slug, _)| (*slug).to_string())
    }).collect();
    if !snow_names.is_empty() {
        let mut required = options.planned_store_fields();
        required.extend(snow_names);
        options.diagnostics = true;
        options.required_fields = Some(required);
    }
    if std::env::var_os("RW_COMPARE_FULL").is_some() {
        options.diagnostics = true;
        options.required_fields = None;
    }
    options.normalized()
}

/// The run's plane for one product at one stored frame: the stored plane,
/// a named raw diagnostic, or the run total differenced over the hour that
/// ends at the frame (read from the frame an hour earlier in the same store).
#[allow(clippy::too_many_arguments)]
fn read_run_field(
    source: &StoreFieldSource,
    spec: &ProductSpec,
    lead_seconds: u64,
    valid_unix: i64,
    by_valid: &BTreeMap<i64, u16>,
    store_root: &Path,
    summary: &rw_wrfbatch::wrf_process::WrfProcessSummary,
) -> Result<Result<RunField, String>, String> {
    Ok(Ok(match spec.run {
        RunPlane::Selector(selector) => match run_plane(source, &selector) {
            Ok(field) => field,
            Err(reason) => return Ok(Err(reason)),
        },
        RunPlane::Named { variable, .. } => match named_run_plane(source, variable) {
            Ok(field) => field,
            Err(reason) => return Ok(Err(reason)),
        },
        RunPlane::SelectorSpeed(u, v) => {
            let (uf, vf) = match (run_plane(source, &u), run_plane(source, &v)) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(r), _) | (_, Err(r)) => return Ok(Err(r)),
            };
            RunField { values: wind_speed(&uf.values, &vf.values), variable: format!("{}_speed", uf.variable), ..uf }
        }
        RunPlane::HourlyAccumulation(selector) => {
            let now = match run_plane(source, &selector) {
                Ok(field) => field,
                Err(reason) => return Ok(Err(reason)),
            };
            if lead_seconds == 0 {
                return Ok(Err(
                    "no hour has been accumulated at the run's start".into()
                ));
            }
            let earlier = valid_unix - 3_600;
            if let Some(slot) = by_valid.get(&earlier) {
                let before_source = StoreFieldSource::open(
                    store_root,
                    &summary.model,
                    &summary.run,
                    *slot,
                )
                .map_err(|error| format!("open the frame an hour earlier: {error}"))?;
                let before = match run_plane(&before_source, &selector) {
                    Ok(field) => field,
                    Err(reason) => {
                        return Ok(Err(format!("the frame an hour earlier: {reason}")));
                    }
                };
                RunField {
                    values: difference(&now.values, &before.values),
                    ..now
                }
            } else if lead_seconds == 3_600 {
                // One hour after the run's own start the run
                // total IS the hour's accumulation.
                now
            } else {
                return Ok(Err(
                    "the run's frame an hour earlier is not beside this one and was \
                     not given, so its hourly accumulation cannot be formed (keep \
                     that frame, or pass it with --context)"
                        .into(),
                ));
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rrfs_reference_reads_the_unbounded_bucket_before_nomads() {
        // NOMADS keeps about two days: a reference pinned to it 404'd on every
        // older RRFS cycle while the init fetch read noaa-rrfs-ops-pds.
        let rrfs = reference_specs().into_iter().find(|row| row.name == "rrfs").unwrap();
        assert_eq!(rrfs.sources, &[SourceId::Aws, SourceId::Nomads]);
        let cycle = CycleSpec::new("20261015".to_string(), 12).unwrap();
        let request = ModelRunRequest::new(rrfs.model, cycle, 12, rrfs.file_product).unwrap();
        let first = rustwx_models::resolve_urls(&request)
            .unwrap()
            .into_iter()
            .find(|url| url.source == rrfs.sources[0])
            .unwrap();
        assert!(first.grib_url.starts_with("https://noaa-rrfs-ops-pds.s3.amazonaws.com/"),
            "{}", first.grib_url);
    }

    #[test]
    fn snow_comparison_difference_uses_depth_units() {
        for spec in product_specs().iter().filter(|s| ["snow10", "snowkuchera", "snow_depth"].contains(&s.name)) {
            assert!(spec.continuous);
            assert_eq!(product_difference_step(spec, "in"), Some(2.0));
        }
        let liquid = product_specs().into_iter().find(|s| s.name == "qpf1h").unwrap();
        assert_eq!(product_difference_step(&liquid, "in"), Some(0.1));
    }

    #[test]
    fn vorticity_comparisons_keep_the_difference_in_production_display_units() {
        for spec in product_specs().into_iter().filter(|s| s.name.ends_with("absolute_vorticity")) {
            let RunPlane::Selector(selector) = spec.run else { panic!("vorticity selector") };
            let style = rustwx_products::viewer::operational_style_for_store_variable(
                spec.slug, &serde_json::to_value(selector).unwrap(), "s-1", ModelId::WrfGdex,
            ).or_else(|| rustwx_products::viewer::curated_style_for_store_variable(
                spec.slug, &serde_json::to_value(selector).unwrap(), "s-1", ModelId::WrfGdex,
            )).expect("production vorticity style");
            assert_eq!(style.display_units, "10^-5 s^-1");
            assert_eq!(product_difference_step(&spec, &style.display_units), Some(2.0), "{}", spec.name);
        }
    }

    #[test]
    fn snow_comparison_imports_requested_computed_planes() {
        let specs: Vec<_> = product_specs().into_iter()
            .filter(|spec| ["snow10", "snowkuchera", "snow_depth"].contains(&spec.name))
            .collect();
        assert_eq!(specs.len(), 3);
        let planned = import_options(&specs).planned_store_fields();
        for spec in specs {
            assert!(planned.iter().any(|name| name == spec.slug), "{}", spec.slug);
        }
        assert!(!planned.iter().any(|name| name == "snow_depth_change"));
    }

    #[test]
    fn observation_identity_deduplicates_ordered_and_packaged_references() {
        let packet = serde_json::json!({
            "label":"Observed radar", "provenance":{"source":"radar_reference"},
        });
        assert!(observation_source_matches(&packet, "radar_reference", "Observed radar"));
        assert!(observation_source_matches(&packet, "RADAR_REFERENCE", "Observed radar"));
        assert!(observation_source_matches(&packet, "table_alias", "Observed radar"));
        assert!(!observation_source_matches(&packet, "other_reference", "Other observation"));
    }

    #[test]
    fn an_unrelated_hourly_field_cannot_validate_a_wrong_precipitation_statistic() {
        use grib_core::grib2::{DataRepresentation, Grib2Message, GridDefinition, ProductDefinition};
        let cycle = at("2026-10-03T21:00:00Z").naive_utc();
        let message = |parameter, statistic, hours, value: f32| Grib2Message {
            discipline: 0,
            identification: Default::default(),
            reference_time: cycle,
            grid: GridDefinition {
                template: 0, nx: 2, ny: 1, lat1: 35.0, lat2: 35.0,
                lon1: -100.0, lon2: -99.0, dx: 1.0, dy: 0.0,
                num_data_points: 2, ..Default::default()
            },
            product: ProductDefinition {
                template: 8, parameter_category: 1, parameter_number: parameter,
                level_type: 1, level_value: 0.0, forecast_time: 2, time_range_unit: 1,
                statistical_process_type: Some(statistic), statistical_time_range_unit: Some(1),
                time_range_length: Some(hours),
                end_of_interval: Some(cycle + chrono::Duration::hours(2 + i64::from(hours))),
                ..Default::default()
            },
            data_rep: DataRepresentation {
                template: 4, bits_per_value: 32, section5_num_data_points: 2,
                ..Default::default()
            },
            bitmap: None,
            raw_data: [value, value].into_iter().flat_map(f32::to_be_bytes).collect(),
        };
        let selector = FieldSelector::surface(CanonicalField::TotalPrecipitation);
        let plane = ReferencePlane::HourlyAccumulation(selector);
        // Parameter 7 is an unrelated moisture field. Its valid interval
        // cannot lend an accumulation identity to maximum APCP (process 2).
        let mut grib = Grib2File { messages: vec![message(8, 2, 1, 99.0), message(7, 1, 1, 7.0)] };
        assert!(reference_field(&grib, plane, 3).unwrap().is_err());
        // A six-hour APCP accumulation starts at the same hour too. Only
        // the two-to-three-hour sum belongs to this precipitation panel.
        grib.messages.push(message(8, 1, 6, 66.0));
        grib.messages.push(message(8, 1, 1, 2.5));
        let field = reference_field(&grib, plane, 3).unwrap().unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(field.values, vec![2.5, 2.5]);
        grib.messages.last_mut().unwrap().product.end_of_interval = Some(cycle + chrono::Duration::hours(4));
        assert!(reference_field(&grib, plane, 3).unwrap().is_err());
    }

    /// The hail sheets read the HRRR message that NCEP UPP writes from the same WRF field as the run's plane:
    /// entire atmosphere = HAIL_MAX2D, the 0.1 sigma level = HAIL_MAXK1, surface = HAILCAST_DIAM_MAX
    /// (UPP MDL2AGL.f; post_avblflds_raphrrr.xml 794/795/728; HRRR wrfsfc .idx: HAIL at entire atmosphere,
    /// 0.1 sigma level and surface). Before the fix HAIL_MAXK1 was read against the surface message (HAILCAST).
    #[test]
    fn hail_sheets_pair_each_wrf_field_with_the_hrrr_message_upp_writes_from_it() {
        use grib_core::grib2::{DataRepresentation, Grib2Message, GridDefinition, ProductDefinition};
        let cycle = at("2026-10-03T21:00:00Z").naive_utc();
        let hail = |level_type: u8, level_value: f64, metres: f32| Grib2Message {
            discipline: 0,
            identification: Default::default(),
            reference_time: cycle,
            grid: GridDefinition {
                template: 0, nx: 2, ny: 1, lat1: 35.0, lat2: 35.0,
                lon1: -100.0, lon2: -99.0, dx: 1.0, dy: 0.0,
                num_data_points: 2, ..Default::default()
            },
            product: ProductDefinition {
                template: 8, parameter_category: 1, parameter_number: 31,
                level_type, level_value, forecast_time: 2, time_range_unit: 1,
                statistical_process_type: Some(2), statistical_time_range_unit: Some(1),
                time_range_length: Some(1),
                end_of_interval: Some(cycle + chrono::Duration::hours(3)),
                ..Default::default()
            },
            data_rep: DataRepresentation {
                template: 4, bits_per_value: 32, section5_num_data_points: 2,
                ..Default::default()
            },
            bitmap: None,
            raw_data: [metres, metres].into_iter().flat_map(f32::to_be_bytes).collect(),
        };
        // The order and levels of HRRR's own file (hrrr.t12z.wrfsfcf01, messages 53, 54 and 55).
        let grib = Grib2File { messages: vec![hail(10, 0.0, 0.050), hail(104, 0.1, 0.030), hail(1, 0.0, 0.020)] };
        for (name, wrf_source, pattern, millimetres) in [
            ("hail_max_column", "HAIL_MAX2D", "HAIL:entire atmosphere", 50.0_f32),
            ("hail_max_surface", "HAIL_MAXK1", "HAIL:0.1 sigma level", 30.0),
            ("hail_max_hailcast", "HAILCAST_DIAM_MAX", "HAIL:surface", 20.0),
        ] {
            let spec = product_specs().into_iter().find(|spec| spec.name == name)
                .unwrap_or_else(|| panic!("no {name} sheet"));
            let RunPlane::Named { variable, .. } = spec.run else { panic!("{name}: a named run plane") };
            let row = rw_wrfbatch::wrf_process::HAIL_CORE_FIELD_CATALOG.iter()
                .find(|row| row.store_name == variable)
                .unwrap_or_else(|| panic!("{name}: no store row {variable}"));
            assert_eq!(row.source, wrf_source, "{name}");
            assert_eq!(spec.inventory_patterns, [pattern].as_slice(), "{name}");
            let field = reference_field(&grib, spec.reference, 3).unwrap()
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(field.values.iter().all(|value| (value - millimetres).abs() < 1e-3),
                "{name}: {:?}", field.values);
        }
    }

    /// A scratch directory removed when the test ends.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            // Parallel tests read one clock tick on the 2.8.6 windows-2025 runner and collided on this name; the counter keeps each call distinct.
            static NEXT_SCRATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "rw-compare-test-{tag}-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|elapsed| elapsed.as_nanos())
                    .unwrap_or(0),
                NEXT_SCRATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ));
            std::fs::create_dir_all(&path).expect("scratch");
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_flat_gallery_alias_of_the_output_keeps_the_completed_sheet() {
        let scratch = Scratch::new("self-copy");
        let alias_directory = scratch.0.join("alias");
        std::fs::create_dir(&alias_directory).expect("alias directory");
        let sheet = scratch.0.join("sheet.png");
        let original = b"a complete renderer output";
        std::fs::write(&sheet, original).expect("sheet");

        copy_flat_sheet(&sheet, &sheet).expect("identical output and gallery");
        assert_eq!(std::fs::read(&sheet).expect("same sheet"), original);

        let alias = alias_directory.join("..").join("sheet.png");
        assert_ne!(sheet, alias, "the spellings must exercise canonical paths");
        copy_flat_sheet(&sheet, &alias).expect("canonical output and gallery alias");
        assert_eq!(std::fs::read(&sheet).expect("aliased sheet"), original);

        let separate = scratch.0.join("separate.png");
        copy_flat_sheet(&sheet, &separate).expect("a distinct gallery copy");
        assert_eq!(std::fs::read(&separate).expect("copied sheet"), original);
        assert_eq!(std::fs::read(&sheet).expect("original sheet"), original);
    }

    fn temperature_style() -> SheetStyle {
        let selector = FieldSelector::height_agl(CanonicalField::Temperature, 2);
        SheetStyle::production(
            rustwx_products::viewer::operational_style_for_store_variable(
                "temperature_2m",
                &serde_json::to_value(selector).expect("selector json"),
                "K",
                ModelId::WrfGdex,
            )
            .expect("the 2 m temperature ladder"),
        )
    }

    #[test]
    fn generic_comparison_presentation_keeps_derived_labels_exactly() {
        let (presentation, title, source) = comparison_presentation(
            rustwx_render::RenderTheme::default_theme(), "Candidate run", "source: engine 2.8.5".into());
        assert_eq!(title, "Candidate run");
        assert_eq!(source, "source: engine 2.8.5");
        assert!(presentation.is_default());
        assert_eq!(presentation.model_name("HRRR"), "HRRR");
        assert_eq!(presentation.source_subtitle(Some("source: NOAA HRRR".into())).as_deref(), Some("source: NOAA HRRR"));
    }

    #[test]
    fn comparison_difference_themes_change_colors_and_keep_scientific_ladder() {
        let default = rustwx_render::RenderTheme::default_theme();
        assert_eq!(comparison_difference_scale(&default, 2.0), difference_scale(2.0));
        let file = rustwx_render::theme::RenderThemeFile::from_json(
            r##"{"colormaps":{"diverging":["#ff0000","#0000ff"]}}"##)
            .expect("theme JSON");
        let themed = rustwx_render::RenderTheme::from_file_spec(&file, None).expect("theme");
        let original = difference_scale(2.0).resolved_discrete();
        let changed = comparison_difference_scale(&themed, 2.0).resolved_discrete();
        assert_eq!(changed.levels, original.levels);
        assert_eq!(changed.extend, original.extend);
        assert_eq!(changed.mask_below, original.mask_below);
        assert_ne!(changed.colors, original.colors);
    }

    #[test]
    fn every_themed_field_panel_resolves_one_canonical_color_table() {
        let default = rustwx_render::RenderTheme::default_theme();
        assert_eq!(comparison_product_key(&default, "test", "run"), "test_run");
        assert_eq!(comparison_product_key(&default, "test", "reference"), "test_reference");
        let file = rustwx_render::theme::RenderThemeFile::from_json(
            r##"{"colormaps":{"products":{"test":["#ff0000","#0000ff"]}}}"##)
            .expect("theme JSON");
        let theme = rustwx_render::RenderTheme::from_file_spec(&file, None).expect("theme");
        let scale = difference_scale(2.0);
        let run_key = rustwx_render::ProductKey::named(comparison_product_key(&theme, "test", "run"));
        let reference_key = rustwx_render::ProductKey::named(comparison_product_key(&theme, "test", "reference"));
        assert_eq!(run_key, reference_key);
        assert_eq!(theme.product_scale_override(&run_key, &scale), theme.product_scale_override(&reference_key, &scale));
        assert!(theme.product_scale_override(&run_key, &scale).is_some());
        let diff_key = rustwx_render::ProductKey::named(comparison_product_key(&theme, "test", "difference"));
        assert!(theme.product_scale_override(&diff_key, &scale).is_none());
    }

    #[test]
    fn branded_comparison_themes_apply_run_labels_and_preserve_reference_provenance() {
        for theme in ["woof-light", "woof-dark"] {
            let (presentation, title, source) = comparison_presentation(
                rustwx_render::RenderTheme::resolve(theme).expect("theme"),
                "Candidate run", "source: engine 2.8.5".into());
            assert_eq!(title, "WOOF");
            assert_eq!(source, "Recast WOOF");
            for reference in reference_specs() {
                assert_eq!(presentation.model_name(reference.label), reference.label);
                let derived = format!("source: {}", reference.source_label);
                assert_eq!(presentation.source_subtitle(Some(derived.clone())), Some(derived));
            }
            for reference in comparison_observations::specifications().expect("table") {
                assert_eq!(presentation.model_name(&reference.label), reference.label);
                let derived = format!("source: {}", reference.source_label);
                assert_eq!(presentation.source_subtitle(Some(derived.clone())), Some(derived));
            }
        }
    }

    /// The panels of a sheet whose two sides carry the same text, so two
    /// panels can differ only by what their planes were drawn as.
    fn panels_of(
        scratch: &Path,
        style: &SheetStyle,
        run_kelvin: &[f32],
        reference_kelvin: &[f32],
        with_difference: bool,
    ) -> Vec<rustwx_render::RgbaImage> {
        let (ny, nx) = (6usize, 8usize);
        let lat: Vec<f32> = (0..ny * nx).map(|cell| 34.0 + (cell / nx) as f32).collect();
        let lon: Vec<f32> = (0..ny * nx)
            .map(|cell| -104.0 + (cell % nx) as f32)
            .collect();
        let convert = |values: &[f32]| -> Vec<f32> {
            values
                .iter()
                .map(|value| style.convert.apply(*value))
                .collect()
        };
        let text = || PanelText {
            title: "panel".to_string(),
            left: "left".to_string(),
            right: "right".to_string(),
        };
        let (run_values, reference_values) = (convert(run_kelvin), convert(reference_kelvin));
        let difference = with_difference.then(|| {
            (
                text(),
                rw_wrfbatch::compare::difference(&run_values, &reference_values),
                difference_step(&style.display_units).expect("a degF step"),
            )
        });
        draw_panels(PanelSet {
            geometry: (&lat, &lon, None, ny, nx),
            width: 480,
            height: 360,
            slug: "test",
            style,
            scratch,
            run: text(),
            run_values,
            reference: text(),
            reference_values,
            run_overlays: None,
            reference_overlays: None,
            observation: None,
            difference,
            coverage: None,
            reference_coverage: None,
        })
        .expect("panels")
    }

    #[test]
    fn both_field_panels_wear_one_colour_scale_whatever_the_other_side_holds() {
        let scratch = Scratch::new("shared-scale");
        let style = temperature_style();
        let cold: Vec<f32> = (0..48).map(|cell| 268.0 + 0.5 * cell as f32).collect();
        let warm: Vec<f32> = cold.iter().map(|value| value + 9.0).collect();

        // The same plane on both sides is the same picture on both sides:
        // one size, one extent, one table, one range.
        let same = panels_of(&scratch.0.join("same"), &style, &cold, &cold, false);
        assert_eq!(same.len(), 2);
        assert_eq!(same[0].dimensions(), (480, 360));
        assert!(
            same[0] == same[1],
            "identical planes drew two different panels"
        );

        // Swapping the sides swaps the pictures and changes neither: a
        // panel depends on its own plane and on nothing about its partner.
        let forward = panels_of(&scratch.0.join("forward"), &style, &cold, &warm, false);
        let swapped = panels_of(&scratch.0.join("swapped"), &style, &warm, &cold, false);
        assert!(forward[0] != forward[1], "9 K apart must not draw alike");
        assert!(forward[0] == swapped[1] && forward[1] == swapped[0]);
        assert!(
            forward[0] == same[0],
            "the run panel moved with its partner"
        );

        // An outlier on one side does not re-range the other.
        let mut wild = warm.clone();
        wild[7] = 400.0;
        let outlier = panels_of(&scratch.0.join("outlier"), &style, &cold, &wild, false);
        assert!(
            outlier[0] == same[0],
            "the reference's extreme re-ranged the run"
        );
    }

    #[test]
    fn the_difference_panel_is_drawn_on_its_own_ladder_and_follows_the_difference() {
        let scratch = Scratch::new("difference");
        let style = temperature_style();
        let base: Vec<f32> = (0..48).map(|cell| 280.0 + 0.25 * cell as f32).collect();
        let warmer: Vec<f32> = base.iter().map(|value| value + 3.0).collect();
        let none = panels_of(&scratch.0.join("none"), &style, &base, &base, true);
        let some = panels_of(&scratch.0.join("some"), &style, &warmer, &base, true);
        let other = panels_of(&scratch.0.join("other"), &style, &base, &warmer, true);
        assert_eq!(none.len(), 3);
        assert_eq!(none[2].dimensions(), none[0].dimensions());
        assert!(
            none[2] != none[0],
            "the difference is not drawn on the field's ladder"
        );
        assert!(
            some[2] != none[2],
            "a 5.4 degF difference drew as no difference"
        );
        assert!(some[2] != other[2], "the sign of the difference is drawn");
    }

    #[test]
    fn the_observed_panel_uses_the_forecast_ladder_and_keeps_the_first_two_panels() {
        let scratch = Scratch::new("observed-scale");
        let style = temperature_style();
        let (ny, nx) = (6_usize, 8_usize);
        let lat: Vec<f32> = (0..ny * nx).map(|cell| 34.0 + (cell / nx) as f32).collect();
        let lon: Vec<f32> = (0..ny * nx).map(|cell| -104.0 + (cell % nx) as f32).collect();
        let native: Vec<f32> = (0..48).map(|cell| 280.0 + 0.25 * cell as f32).collect();
        let convert = || native.iter().map(|value| style.convert.apply(*value)).collect();
        let text = || PanelText {title: "panel".into(), left: "left".into(), right: "right".into()};
        let pair = panels_of(&scratch.0.join("pair"), &style, &native, &native, false);
        let observed = draw_panels(PanelSet {
            geometry: (&lat, &lon, None, ny, nx), width: 480, height: 360,
            slug: "test", style: &style, scratch: &scratch.0.join("triple"),
            run: text(), run_values: convert(), reference: text(), reference_values: convert(),
            run_overlays: None, reference_overlays: None,
            observation: Some((text(), convert())), difference: None,
            coverage: None, reference_coverage: None,
        }).unwrap();
        assert_eq!(observed.len(), 3);
        assert!(observed[0] == pair[0] && observed[1] == pair[1]);
        assert!(observed[2] == pair[0], "equal observed and forecast planes draw equal pixels");
    }

    /// Pixels of exactly `color` in a panel.
    fn count_color(image: &rustwx_render::RgbaImage, color: rustwx_render::Color) -> usize {
        image
            .pixels()
            .filter(|pixel| pixel.0 == [color.r, color.g, color.b, color.a])
            .count()
    }

    #[test]
    fn no_coverage_is_grey_on_the_observed_panel_and_outlined_on_every_other() {
        let scratch = Scratch::new("no-coverage");
        // An observed product's own ladder (hourly precipitation, which, like
        // composite reflectivity, draws nothing below its floor), on a
        // latitude-longitude grid: the inverse-raster path.
        let selector = FieldSelector::surface(CanonicalField::TotalPrecipitation);
        let style = SheetStyle::production(rustwx_products::viewer::operational_style_for_store_variable(
            "apcp_1h", &serde_json::to_value(selector).unwrap(), "mm", ModelId::WrfGdex,
        ).unwrap());
        let (ny, nx) = (12_usize, 16_usize);
        let lat: Vec<f32> = (0..ny * nx).map(|cell| 34.0 + 0.5 * (cell / nx) as f32).collect();
        let lon: Vec<f32> = (0..ny * nx).map(|cell| -104.0 + 0.5 * (cell % nx) as f32).collect();
        // The east third saw nothing (a radar outage); the rest observed.
        let mask: Vec<bool> = (0..ny * nx).map(|cell| cell % nx >= 11).collect();
        let native: Vec<f32> = (0..ny * nx).map(|cell| 0.5 + 0.05 * cell as f32).collect();
        let convert = || -> Vec<f32> { native.iter().map(|value| style.convert.apply(*value)).collect() };
        let observed: Vec<f32> = convert()
            .into_iter()
            .zip(&mask)
            .map(|(value, unobserved)| if *unobserved { f32::NAN } else { value })
            .collect();
        let text = || PanelText {title: "panel".into(), left: "left".into(), right: "right".into()};
        let draw = |name: &str, coverage: Option<&[bool]>| draw_panels(PanelSet {
            geometry: (&lat, &lon, None, ny, nx), width: 480, height: 360,
            slug: "test", style: &style, scratch: &scratch.0.join(name),
            run: text(), run_values: convert(), reference: text(), reference_values: convert(),
            run_overlays: None, reference_overlays: None,
            observation: Some((text(), observed.clone())),
            difference: Some((text(), vec![0.5; ny * nx], 1.0)),
            coverage, reference_coverage: None,
        }).unwrap();
        let plain = draw("plain", None);
        let marked = draw("marked", Some(&mask));
        assert_eq!(plain.len(), 4);
        // Observed panel: the unobserved third is flat grey, and without the
        // mask it is not (blank, like an observed nothing).
        let grey = count_color(&marked[2], coverage::NO_COVERAGE_FILL);
        assert_eq!(count_color(&plain[2], coverage::NO_COVERAGE_FILL), 0);
        let map_pixels = marked[2].width() as usize * marked[2].height() as usize;
        assert!(grey > map_pixels / 40, "the outage is visibly grey: {grey} of {map_pixels} pixels");
        // Run, forecast reference and difference panels: the same pictures
        // plus the grey outline, nothing else.
        for index in [0, 1, 3] {
            assert_ne!(plain[index], marked[index], "panel {index} carries the outline");
            let changed = plain[index]
                .pixels()
                .zip(marked[index].pixels())
                .filter(|(a, b)| a != b)
                .count();
            assert!(changed < map_pixels / 10, "panel {index}: an outline, not a fill ({changed} pixels)");
            assert_eq!(count_color(&marked[index], coverage::NO_COVERAGE_FILL), 0);
        }
        // A fully covered sheet draws exactly what it drew before.
        let covered = vec![false; ny * nx];
        let unchanged = draw("covered", Some(&covered));
        for index in [0, 1, 3] {
            assert!(unchanged[index] == plain[index], "panel {index}: no edge, no outline");
        }
    }

    #[test]
    fn hourly_precipitation_manifest_adds_a_bounded_observation_panel() {
        use rw_obs::pack::{GEO_SCHEMA, GRID_SCHEMA, PayloadBuilder, payload_digest, write_pack};
        let scratch = Scratch::new("hourly-observation");
        let metadata = serde_json::json!({
            "kind":"regular_latlon", "nx":2, "ny":2, "source_nx":2, "source_ny":2,
            "i_start":0, "j_start":0,
        });
        let mut data = PayloadBuilder::new();
        data.push_f64("values", &[25.4; 4], vec![2,2]);
        data.push_mask("valid", &[true; 4], vec![2,2]);
        let (payload, arrays) = data.finish();
        write_pack(&scratch.0.join("hour.obspack"), &serde_json::json!({
            "schema":GRID_SCHEMA, "quantity":"precipitation_accumulation", "units":"mm",
            "valid_time":"2030-01-01T01:00:00", "accumulation_seconds":3600,
            "provenance":{"product":"hourly_accumulation", "is_stub":false},
            "grid":metadata, "arrays":arrays, "content_sha256":payload_digest(&payload),
        }), &payload).unwrap();
        let mut coordinates = PayloadBuilder::new();
        coordinates.push_f64("latitude", &[30.0,30.0,31.0,31.0], vec![2,2]);
        coordinates.push_f64("longitude", &[-100.0,-99.0,-100.0,-99.0], vec![2,2]);
        let (payload, arrays) = coordinates.finish();
        write_pack(&scratch.0.join("grid.geopack"), &serde_json::json!({
            "schema":GEO_SCHEMA, "source_product":"hourly_accumulation", "grid":metadata,
            "arrays":arrays, "content_sha256":payload_digest(&payload),
        }), &payload).unwrap();
        let manifest = scratch.0.join("observations.json");
        std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!([{
            "quantity":"precipitation_1h", "label":"OBS", "path":"hour.obspack",
            "grid_path":"grid.geopack",
        }])).unwrap()).unwrap();
        let selector = FieldSelector::surface(CanonicalField::TotalPrecipitation);
        let style = SheetStyle::production(rustwx_products::viewer::operational_style_for_store_variable(
            "apcp_1h", &serde_json::to_value(selector).unwrap(), "mm", ModelId::WrfGdex,
        ).unwrap());
        let lat = [30.25,30.25,30.75,30.75];
        let lon = [-100.25,-99.75,-100.25,-99.75];
        let geometry = (&lat[..], &lon[..], None, 2, 2);
        let (text, values, receipt) = observed_panel(
            &manifest, "precipitation_accumulation", at("2030-01-01T01:00:00Z"), geometry, &style,
        ).unwrap().expect("the request alias selects its canonical observed quantity");
        assert!(values[0].is_nan() && values[2].is_nan(), "outside cells cannot acquire edge observations");
        assert_eq!(values[1], 1.0);
        assert_eq!(values[3], 1.0);
        assert_eq!(receipt["points_without_observation"], 2);
        assert_eq!(receipt["canonical_quantity"], "precipitation_accumulation");
        let title = || PanelText {title:"forecast".into(), left:"time".into(), right:String::new()};
        let panels = draw_panels(PanelSet {
            geometry, width:480, height:360, slug:"hourly_accumulation", style:&style,
            scratch:&scratch.0.join("panels"), run:title(), run_values:vec![1.0;4],
            reference:title(), reference_values:vec![1.0;4], run_overlays:None,
            reference_overlays:None, observation:Some((text, values)), difference:None,
            coverage:None, reference_coverage:None,
        }).unwrap();
        assert_eq!(panels.len(), 3, "the manifest produces the observed third panel");
    }

    #[test]
    fn every_product_names_its_reference_messages_and_patterns() {
        let specs = product_specs();
        let mut names = std::collections::BTreeSet::new();
        let mut slugs = std::collections::BTreeSet::new();
        for spec in &specs {
            assert!(names.insert(spec.name), "{} twice", spec.name);
            assert!(slugs.insert(spec.slug), "{} twice", spec.slug);
            assert!(!spec.inventory_patterns.is_empty(), "{}", spec.name);
            assert!(!spec.reference_message.is_empty(), "{}", spec.name);
            assert!(
                spec.reference_message.len() <= 24,
                "{}: {:?} will not fit beside the start, lead and file name",
                spec.name,
                spec.reference_message
            );
            let wanted = match spec.reference {
                ReferencePlane::Speed(..) | ReferencePlane::RawSpeed(..) => 2,
                _ => 1,
            };
            assert_eq!(spec.inventory_patterns.len(), wanted, "{}", spec.name);
        }
        for name in ["refc", "t2m", "td2m", "wspd10", "qpf1h", "hgt500", "swdown"] {
            assert!(names.contains(name), "{name} is in the brief");
        }
    }

    #[test]
    fn smoke_rows_compare_the_planes_the_wrfout_catalogue_stores() {
        // The run plane must be the selector SMOKE_SFC / SMOKE_COLUMN are
        // written under, or the sheet would find no run plane; the reference
        // message is HRRR-Smoke's own MASSDEN at 8 m and COLMD.
        let catalogue = rw_wrfbatch::wrf_process::CHEM_CORE_FIELD_CATALOG;
        for (product, source, message) in [
            ("smoke_sfc", "SMOKE_SFC", "MASSDEN:8 m above ground"),
            ("smoke_column", "SMOKE_COLUMN", "COLMD:entire atmosphere"),
        ] {
            let spec = product_specs().into_iter().find(|s| s.name == product)
                .unwrap_or_else(|| panic!("{product} row"));
            let row = catalogue.iter().find(|r| r.source == source).expect("catalogue row");
            match (spec.run, spec.reference) {
                (RunPlane::Selector(run), ReferencePlane::Selector(reference)) => {
                    assert_eq!(run, row.selector, "{product} run plane");
                    assert_eq!(reference, row.selector, "{product} reference plane");
                }
                _ => panic!("{product} compares one selector on both sides"),
            }
            assert_eq!(spec.inventory_patterns, &[message]);
            assert_eq!(spec.style_variable, Some(row.store_name));
            assert!(spec.continuous, "{product} draws its difference panel");
        }
    }

    #[test]
    fn the_reference_table_spells_the_published_file_names() {
        let hrrr = reference_specs()
            .into_iter()
            .find(|spec| spec.name == "hrrr")
            .expect("hrrr");
        assert_eq!(
            (hrrr.file_names)(0, 6),
            vec!["hrrr.t00z.wrfsfcf06.grib2", "hrrr.t00z.wrfprsf06.grib2"]
        );
        assert_eq!((hrrr.file_names)(21, 18)[0], "hrrr.t21z.wrfsfcf18.grib2");
        assert_eq!((hrrr.bucket_directory)("20261003"), "hrrr.20261003/conus");
        assert_eq!((hrrr.horizon)(0), 48);
        assert_eq!((hrrr.horizon)(18), 48);
        assert_eq!((hrrr.horizon)(21), 18);
    }

    #[test]
    fn ordered_reference_lists_use_the_native_table_and_reject_empty_members() {
        assert_eq!(reference_names(" hrrr, mrms,rrfs,hrrr ").expect("list"),
            vec!["hrrr", "mrms", "rrfs"]);
        for invalid in ["", "hrrr,", "hrrr,unknown"] {
            assert!(reference_names(invalid).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn forecast_rows_choose_surface_and_pressure_files_without_source_branches() {
        let reference = reference_specs().into_iter().find(|row| row.name == "rrfs").expect("RRFS row");
        assert_eq!((reference.file_names)(12, 1),
            vec!["rrfs.t12z.2dfld.3km.f001.conus.grib2"]);
        assert_eq!((reference.pressure_file_names)(12, 1),
            vec!["rrfs.t12z.prslev.3km.f001.conus.grib2"]);
        assert_eq!(reference.file_product, "2dfld-conus");
        assert_eq!(reference.pressure_file_product, "prs-conus");
        for hour in 0..24 {
            assert_eq!((reference.horizon)(hour),
                *rustwx_models::supported_forecast_hours(reference.model, hour).last().expect("hourly cycle"));
        }
    }

    #[test]
    fn a_cycle_that_stops_short_is_answered_with_the_one_that_reaches() {
        let hrrr = reference_specs()
            .into_iter()
            .find(|spec| spec.name == "hrrr")
            .expect("hrrr");
        // A 21Z run at hour 24: its own cycle stops at f18; the next
        // six-hourly cycle reaches the same valid time at f21.
        assert_eq!(
            cycle_reaching(
                &hrrr,
                at("2026-10-02T21:00:00Z"),
                at("2026-10-03T21:00:00Z")
            ),
            Some(("2026100300".to_string(), 21))
        );
        // Hour 48 of the same run is the 00Z cycle's f45.
        assert_eq!(
            cycle_reaching(
                &hrrr,
                at("2026-10-02T21:00:00Z"),
                at("2026-10-04T21:00:00Z")
            ),
            Some(("2026100300".to_string(), 45))
        );
        // A lead the run's own cycle publishes is that cycle.
        assert_eq!(
            cycle_reaching(
                &hrrr,
                at("2026-10-02T21:00:00Z"),
                at("2026-10-03T09:00:00Z")
            ),
            Some(("2026100221".to_string(), 12))
        );
        // Nothing that starts within two days of the run reaches a valid time
        // a hundred hours out.
        assert_eq!(
            cycle_reaching(
                &hrrr,
                at("2026-10-02T21:00:00Z"),
                at("2026-10-07T01:00:00Z")
            ),
            None
        );
    }

    #[test]
    fn the_subset_tag_follows_the_patterns() {
        let patterns = inventory_patterns();
        assert_eq!(subset_tag(&patterns), subset_tag(&patterns));
        let mut more = patterns.clone();
        more.push("PRES:surface".into());
        assert_ne!(subset_tag(&patterns), subset_tag(&more));
        // "ab","c" and "a","bc" are two different tables.
        assert_ne!(
            subset_tag(&["ab".into(), "c".into()]),
            subset_tag(&["a".into(), "bc".into()])
        );
    }

    #[test]
    fn units_are_compared_by_family_not_by_spelling() {
        assert!(same_units("K", "K"));
        assert!(same_units("kg/m^2", "mm"));
        assert!(same_units("m s-1", "m/s"));
        assert!(same_units("gpm", "m"));
        assert!(same_units("dBZ", "dB"));
        assert!(same_units("W m-2", "W/m^2"));
        assert!(same_units("W m**-2", "W/m2"));
        assert!(!same_units("K", "degC"));
        assert!(!same_units("Pa", "hPa"));
    }

    #[test]
    fn a_cycle_is_ten_digits_and_a_real_date() {
        assert_eq!(
            parse_cycle("2026100300").unwrap(),
            ("20261003".to_string(), 0)
        );
        assert_eq!(
            parse_cycle("2026-10-03T21").unwrap(),
            ("20261003".to_string(), 21)
        );
        assert!(parse_cycle("20261003").is_err());
        assert!(parse_cycle("2026100325").is_err());
        assert!(parse_cycle("2026133100").is_err());
    }

    #[test]
    fn a_frame_time_comes_from_the_store_or_from_the_run_key() {
        let run = "local_20261002210000_cf801f82_viewer2d_wrf_science_v5_science_v1";
        let origin = run_origin_unix(run).expect("origin");
        assert_eq!(
            utc(origin).unwrap().to_rfc3339(),
            "2026-10-02T21:00:00+00:00"
        );
        assert_eq!(frame_time(None, run, 2), Some((7_200, origin + 7_200)));
        // A store that carries exact times is believed over its run key.
        let exact = rw_store::RwsExactTime::new(300, origin + 300);
        assert_eq!(frame_time(Some(exact), run, 1), Some((300, origin + 300)));
        assert_eq!(frame_time(None, "era20c_fsr_2004010100", 0), None);
        assert_eq!(run_origin_unix("local_2026100221_short"), None);
    }

    fn at(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text)
            .expect("test time")
            .with_timezone(&Utc)
    }

    #[test]
    fn one_cycle_is_labelled_once() {
        let times = SheetTimes {
            run_init: at("2026-10-03T00:00:00Z"),
            run_lead_seconds: 6 * 3_600,
            reference_cycle: at("2026-10-03T00:00:00Z"),
            reference_lead: 6,
            valid: at("2026-10-03T06:00:00Z"),
        };
        assert!(times.same_cycle());
        assert_eq!(
            times.header_facts("WOOF", "HRRR"),
            vec!["Init 2026-10-03 00Z", "F006", "Valid 2026-10-03 06Z"]
        );
        assert_eq!(times.run_subtitle(), "Init 2026-10-03 00Z F006");
        assert_eq!(
            times.run_panel_subtitle(),
            "Init 2026-10-03 00Z F006 | Valid 2026-10-03 06Z"
        );
        assert_eq!(times.reference_subtitle(), "Init 2026-10-03 00Z F006");
        assert_eq!(times.stem_token(), "20261003_00z_f006");
    }

    #[test]
    fn two_cycles_under_one_valid_time_are_each_named_on_their_own_side() {
        // A 21Z run at hour 24 against the reference's 00Z cycle at hour
        // 21: the same valid time, two starts, two leads.
        let times = SheetTimes {
            run_init: at("2026-10-02T21:00:00Z"),
            run_lead_seconds: 24 * 3_600,
            reference_cycle: at("2026-10-03T00:00:00Z"),
            reference_lead: 21,
            valid: at("2026-10-03T21:00:00Z"),
        };
        assert!(!times.same_cycle());
        assert_eq!(
            times.header_facts("WOOF", "HRRR"),
            vec![
                "Valid 2026-10-03 21Z",
                "WOOF init 2026-10-02 21Z F024",
                "HRRR init 2026-10-03 00Z F021",
            ]
        );
        // Neither panel carries the other's start or lead.
        assert_eq!(times.run_subtitle(), "Init 2026-10-02 21Z F024");
        assert_eq!(
            times.run_panel_subtitle(),
            "Init 2026-10-02 21Z F024 | Valid 2026-10-03 21Z"
        );
        assert_eq!(times.reference_subtitle(), "Init 2026-10-03 00Z F021");
        // The file name is the run's, with the reference's after it.
        assert_eq!(times.stem_token(), "20261002_21z_f024_ref20261003_00z_f021");
    }

    #[test]
    fn a_run_frame_off_its_own_whole_hours_keeps_its_minutes() {
        let times = SheetTimes {
            run_init: at("2026-10-02T21:30:00Z"),
            run_lead_seconds: 5_400,
            reference_cycle: at("2026-10-02T21:00:00Z"),
            reference_lead: 2,
            valid: at("2026-10-02T23:00:00Z"),
        };
        assert_eq!(times.run_lead_label(), "F001h30m");
        assert_eq!(times.run_subtitle(), "Init 2026-10-02 21:30Z F001h30m");
        assert_eq!(
            times.stem_token(),
            "20261002_21z_f001h30m_ref20261002_21z_f002"
        );
    }

    #[test]
    fn the_frame_an_hour_earlier_is_named_in_both_spellings() {
        assert_eq!(
            names_an_hour_earlier("wrfout_d01_2026-10-03_00:00:00"),
            vec![
                "wrfout_d01_2026-10-02_23:00:00".to_string(),
                "wrfout_d01_2026-10-02_23_00_00".to_string(),
            ]
        );
        assert_eq!(
            names_an_hour_earlier("wrfout_d02_2026-10-03_06_30_00.nc")[1],
            "wrfout_d02_2026-10-03_05_30_00.nc"
        );
        assert!(names_an_hour_earlier("model_output.nc").is_empty());
        assert!(names_an_hour_earlier("wrfout_d01_2026-10-03").is_empty());
        assert!(names_an_hour_earlier("wrfout_dxx_2026-10-03_00:00:00").is_empty());
    }

    #[test]
    fn only_whole_hour_messages_are_hours() {
        assert_eq!(message_hours(1, 6), Some(6));
        assert_eq!(message_hours(0, 120), Some(2));
        assert_eq!(message_hours(0, 45), None);
        assert_eq!(message_hours(2, 1), None);
    }
}
