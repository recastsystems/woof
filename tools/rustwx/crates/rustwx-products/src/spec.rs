use rustwx_core::{
    CanonicalProductIdentity, ProductId, ProductKeyMetadata, ProductKind, ProductLineage,
    ProductProvenance, ProductWindowSpec, StatisticalProcess,
};
use rustwx_models::{PlotRecipe, RenderStyle, built_in_plot_recipes};
use rustwx_render::{ProductMaturity, ProductSemanticFlag};

use crate::derived::{
    BlockedDerivedRecipeInventoryEntry, DerivedRecipeInventoryEntry,
    blocked_derived_recipe_inventory, supported_derived_recipe_inventory,
};
use crate::hrrr::HrrrBatchProduct;
use crate::windowed::HrrrWindowedProduct;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProductAliasSpec {
    pub id: ProductId,
    pub slug: String,
    pub title: String,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct ProductSpec {
    pub id: ProductId,
    pub slug: String,
    pub title: String,
    pub kind: ProductKind,
    pub product_metadata: Option<ProductKeyMetadata>,
    pub maturity: ProductMaturity,
    pub flags: Vec<ProductSemanticFlag>,
    pub render_style: Option<String>,
    pub aliases: Vec<ProductAliasSpec>,
    pub notes: Vec<String>,
    pub blocked_reasons: Vec<String>,
}

impl ProductSpec {
    pub fn experimental(&self) -> bool {
        self.maturity.is_non_operational()
    }
}

#[derive(Debug, Clone, Copy)]
struct LegacyProductAliasRoute {
    alias_slug: &'static str,
    alias_title: &'static str,
    canonical_slug: &'static str,
    canonical_kind: ProductKind,
    note: &'static str,
}

const LEGACY_NON_ECAPE_ALIAS_ROUTES: &[LegacyProductAliasRoute] = &[
    LegacyProductAliasRoute {
        alias_slug: "2m_theta_e_10m_winds",
        alias_title: "2m AGL Theta-e / 10m Winds",
        canonical_slug: "theta_e_2m_10m_winds",
        canonical_kind: ProductKind::Derived,
        note: "Legacy plot-recipe slug from the big list. HRRR support lives in the derived lane, not as a native/direct GRIB recipe.",
    },
    LegacyProductAliasRoute {
        alias_slug: "2m_heat_index",
        alias_title: "2m AGL Heat Index",
        canonical_slug: "heat_index_2m",
        canonical_kind: ProductKind::Derived,
        note: "Legacy plot-recipe slug from the big list. HRRR support lives in the derived lane, not as a native/direct GRIB recipe.",
    },
    LegacyProductAliasRoute {
        alias_slug: "2m_wind_chill",
        alias_title: "2m AGL Wind Chill",
        canonical_slug: "wind_chill_2m",
        canonical_kind: ProductKind::Derived,
        note: "Legacy plot-recipe slug from the big list. HRRR support lives in the derived lane, not as a native/direct GRIB recipe.",
    },
    LegacyProductAliasRoute {
        alias_slug: "1h_qpf",
        alias_title: "1h QPF",
        canonical_slug: "qpf_1h",
        canonical_kind: ProductKind::Windowed,
        note: "Legacy plot-recipe slug from the big list. The accurate HRRR implementation is the 1-hour windowed APCP product; alias wiring belongs in the windowed lane rather than a fake native/direct recipe.",
    },
];

pub fn direct_product_specs() -> Vec<ProductSpec> {
    built_in_plot_recipes()
        .iter()
        .filter(|recipe| legacy_alias_route_for_direct_slug(recipe.slug).is_none())
        .map(direct_product_spec)
        .collect()
}

pub fn supported_derived_product_specs() -> Vec<ProductSpec> {
    supported_derived_recipe_inventory()
        .iter()
        .map(supported_derived_product_spec)
        .collect()
}

pub fn blocked_derived_product_specs() -> Vec<ProductSpec> {
    blocked_derived_recipe_inventory()
        .iter()
        .map(blocked_derived_product_spec)
        .collect()
}

pub fn heavy_product_specs() -> Vec<ProductSpec> {
    [HrrrBatchProduct::SevereProofPanel]
        .into_iter()
        .map(heavy_product_spec)
        .collect()
}

pub fn windowed_product_specs() -> Vec<ProductSpec> {
    [
        (
            HrrrWindowedProduct::Qpf1h,
            "1-h APCP accumulation ending at the requested forecast hour",
            "weather_qpf",
        ),
        (
            HrrrWindowedProduct::Qpf6h,
            "Uses direct 6-hour APCP from the ending hour when present, else sums hourly APCP increments",
            "weather_qpf",
        ),
        (
            HrrrWindowedProduct::Qpf12h,
            "Uses direct 12-hour APCP from the ending hour when present, else sums hourly APCP increments",
            "weather_qpf",
        ),
        (
            HrrrWindowedProduct::Qpf24h,
            "Uses direct 24-hour APCP from the ending hour when present, else sums hourly APCP increments",
            "weather_qpf",
        ),
        (
            HrrrWindowedProduct::QpfTotal,
            "Uses direct APCP from the ending hour when available, else sums all hourly APCP increments from F001..Fend",
            "weather_qpf",
        ),
        (
            HrrrWindowedProduct::Uh25km1h,
            "Native 2-5 km UH 1-hour max from HRRR wrfnat",
            "weather_uh",
        ),
        (
            HrrrWindowedProduct::Uh25km3h,
            "Max of trailing native hourly 2-5 km UH maxima",
            "weather_uh",
        ),
        (
            HrrrWindowedProduct::Uh25kmRunMax,
            "Run max of native hourly 2-5 km UH maxima from F001..Fend",
            "weather_uh",
        ),
        (
            HrrrWindowedProduct::Wind10m1hMax,
            "Native 10 m wind-speed 1-hour maximum ending at the requested forecast hour",
            "weather_winds",
        ),
        (
            HrrrWindowedProduct::Wind10mRunMax,
            "Run max of native hourly 10 m wind-speed maxima from F001..Fend",
            "weather_winds",
        ),
        (
            HrrrWindowedProduct::Wind10m0to24hMax,
            "Extended-cycle fixed 24 h max of native hourly 10 m wind-speed maxima from F001..F024",
            "weather_winds",
        ),
        (
            HrrrWindowedProduct::Wind10m24to48hMax,
            "Extended-cycle fixed 24 h max of native hourly 10 m wind-speed maxima from F025..F048",
            "weather_winds",
        ),
        (
            HrrrWindowedProduct::Wind10m0to48hMax,
            "Extended-cycle fixed 48 h max of native hourly 10 m wind-speed maxima from F001..F048",
            "weather_winds",
        ),
        (
            HrrrWindowedProduct::Temp2m0to24hMax,
            "Extended-cycle fixed 24 h max of hourly 2 m temperature snapshots from F001..F024",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Temp2m24to48hMax,
            "Extended-cycle fixed 24 h max of hourly 2 m temperature snapshots from F025..F048",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Temp2m0to48hMax,
            "Extended-cycle fixed 48 h max of hourly 2 m temperature snapshots from F001..F048",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Temp2m0to24hMin,
            "Extended-cycle fixed 24 h min of hourly 2 m temperature snapshots from F001..F024",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Temp2m24to48hMin,
            "Extended-cycle fixed 24 h min of hourly 2 m temperature snapshots from F025..F048",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Temp2m0to48hMin,
            "Extended-cycle fixed 48 h min of hourly 2 m temperature snapshots from F001..F048",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Temp2m0to24hRange,
            "Extended-cycle fixed 24 h max-minus-min range of hourly 2 m temperature snapshots from F001..F024",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Temp2m24to48hRange,
            "Extended-cycle fixed 24 h max-minus-min range of hourly 2 m temperature snapshots from F025..F048",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Temp2m0to48hRange,
            "Extended-cycle fixed 48 h max-minus-min range of hourly 2 m temperature snapshots from F001..F048",
            "weather_temperature",
        ),
        (
            HrrrWindowedProduct::Rh2m0to24hMax,
            "Extended-cycle fixed 24 h max of hourly 2 m relative humidity snapshots from F001..F024",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Rh2m24to48hMax,
            "Extended-cycle fixed 24 h max of hourly 2 m relative humidity snapshots from F025..F048",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Rh2m0to48hMax,
            "Extended-cycle fixed 48 h max of hourly 2 m relative humidity snapshots from F001..F048",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Rh2m0to24hMin,
            "Extended-cycle fixed 24 h min of hourly 2 m relative humidity snapshots from F001..F024",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Rh2m24to48hMin,
            "Extended-cycle fixed 24 h min of hourly 2 m relative humidity snapshots from F025..F048",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Rh2m0to48hMin,
            "Extended-cycle fixed 48 h min of hourly 2 m relative humidity snapshots from F001..F048",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Rh2m0to24hRange,
            "Extended-cycle fixed 24 h max-minus-min range of hourly 2 m relative humidity snapshots from F001..F024",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Rh2m24to48hRange,
            "Extended-cycle fixed 24 h max-minus-min range of hourly 2 m relative humidity snapshots from F025..F048",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Rh2m0to48hRange,
            "Extended-cycle fixed 48 h max-minus-min range of hourly 2 m relative humidity snapshots from F001..F048",
            "weather_rh",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m0to24hMax,
            "Extended-cycle fixed 24 h max of hourly 2 m dewpoint snapshots from F001..F024",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m24to48hMax,
            "Extended-cycle fixed 24 h max of hourly 2 m dewpoint snapshots from F025..F048",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m0to48hMax,
            "Extended-cycle fixed 48 h max of hourly 2 m dewpoint snapshots from F001..F048",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m0to24hMin,
            "Extended-cycle fixed 24 h min of hourly 2 m dewpoint snapshots from F001..F024",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m24to48hMin,
            "Extended-cycle fixed 24 h min of hourly 2 m dewpoint snapshots from F025..F048",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m0to48hMin,
            "Extended-cycle fixed 48 h min of hourly 2 m dewpoint snapshots from F001..F048",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m0to24hRange,
            "Extended-cycle fixed 24 h max-minus-min range of hourly 2 m dewpoint snapshots from F001..F024",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m24to48hRange,
            "Extended-cycle fixed 24 h max-minus-min range of hourly 2 m dewpoint snapshots from F025..F048",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Dewpoint2m0to48hRange,
            "Extended-cycle fixed 48 h max-minus-min range of hourly 2 m dewpoint snapshots from F001..F048",
            "weather_dewpoint",
        ),
        (
            HrrrWindowedProduct::Vpd2m0to24hMax,
            "Extended-cycle fixed 24 h max of hourly 2 m vapor pressure deficit snapshots from F001..F024",
            "weather_vpd",
        ),
        (
            HrrrWindowedProduct::Vpd2m24to48hMax,
            "Extended-cycle fixed 24 h max of hourly 2 m vapor pressure deficit snapshots from F025..F048",
            "weather_vpd",
        ),
        (
            HrrrWindowedProduct::Vpd2m0to48hMax,
            "Extended-cycle fixed 48 h max of hourly 2 m vapor pressure deficit snapshots from F001..F048",
            "weather_vpd",
        ),
        (
            HrrrWindowedProduct::Vpd2m0to24hMin,
            "Extended-cycle fixed 24 h min of hourly 2 m vapor pressure deficit snapshots from F001..F024",
            "weather_vpd",
        ),
        (
            HrrrWindowedProduct::Vpd2m24to48hMin,
            "Extended-cycle fixed 24 h min of hourly 2 m vapor pressure deficit snapshots from F025..F048",
            "weather_vpd",
        ),
        (
            HrrrWindowedProduct::Vpd2m0to48hMin,
            "Extended-cycle fixed 48 h min of hourly 2 m vapor pressure deficit snapshots from F001..F048",
            "weather_vpd",
        ),
        (
            HrrrWindowedProduct::Vpd2m0to24hRange,
            "Extended-cycle fixed 24 h max-minus-min range of hourly 2 m vapor pressure deficit snapshots from F001..F024",
            "weather_vpd",
        ),
        (
            HrrrWindowedProduct::Vpd2m24to48hRange,
            "Extended-cycle fixed 24 h max-minus-min range of hourly 2 m vapor pressure deficit snapshots from F025..F048",
            "weather_vpd",
        ),
        (
            HrrrWindowedProduct::Vpd2m0to48hRange,
            "Extended-cycle fixed 48 h max-minus-min range of hourly 2 m vapor pressure deficit snapshots from F001..F048",
            "weather_vpd",
        ),
    ]
    .into_iter()
    .map(|(product, note, render_style)| windowed_product_spec(product, note, render_style))
    .collect()
}

pub fn direct_product_spec(recipe: &PlotRecipe) -> ProductSpec {
    let id = ProductId::new(ProductKind::Direct, recipe.slug);
    let aliases = legacy_aliases(ProductKind::Direct, recipe.slug);
    ProductSpec {
        id: id.clone(),
        slug: recipe.slug.to_string(),
        title: recipe.title.to_string(),
        kind: ProductKind::Direct,
        product_metadata: Some(with_canonical_identity(
            recipe.product_metadata(),
            id,
            &aliases,
        )),
        maturity: if matches!(recipe.slug, "lightning_threat_1" | "lightning_threat_2" | "lightning_threat_3" | "pr92_flash_rate") {
            ProductMaturity::Experimental
        } else { ProductMaturity::Operational },
        flags: Vec::new(),
        render_style: Some(direct_render_style(recipe).to_string()),
        aliases,
        notes: direct_entry_notes(recipe.slug),
        blocked_reasons: Vec::new(),
    }
}

fn supported_derived_product_spec(recipe: &DerivedRecipeInventoryEntry) -> ProductSpec {
    let maturity = if recipe.experimental {
        ProductMaturity::Experimental
    } else {
        ProductMaturity::Operational
    };
    let flags = derived_entry_flags(recipe.slug);
    let id = ProductId::new(ProductKind::Derived, recipe.slug);
    let aliases = legacy_aliases(ProductKind::Derived, recipe.slug);
    ProductSpec {
        id: id.clone(),
        slug: recipe.slug.to_string(),
        title: recipe.title.to_string(),
        kind: ProductKind::Derived,
        product_metadata: Some(derived_product_metadata(
            recipe.title,
            maturity,
            &flags,
            id,
            &aliases,
        )),
        maturity,
        flags: sorted_flags(&flags),
        render_style: None,
        aliases,
        notes: derived_entry_notes(recipe.slug, recipe.experimental),
        blocked_reasons: Vec::new(),
    }
}

fn blocked_derived_product_spec(recipe: &BlockedDerivedRecipeInventoryEntry) -> ProductSpec {
    let id = ProductId::new(ProductKind::Derived, recipe.slug);
    ProductSpec {
        id: id.clone(),
        slug: recipe.slug.to_string(),
        title: recipe.title.to_string(),
        kind: ProductKind::Derived,
        product_metadata: Some(derived_product_metadata(
            recipe.title,
            ProductMaturity::Operational,
            &[],
            id,
            &[],
        )),
        maturity: ProductMaturity::Operational,
        flags: Vec::new(),
        render_style: None,
        aliases: Vec::new(),
        notes: Vec::new(),
        blocked_reasons: vec![recipe.reason.to_string()],
    }
}

fn heavy_product_spec(product: HrrrBatchProduct) -> ProductSpec {
    let (title, maturity, flags, notes) = match product {
        HrrrBatchProduct::SevereProofPanel => (
            "Severe Map Set",
            ProductMaturity::Proof,
            vec![
                ProductSemanticFlag::ProofOriented,
                ProductSemanticFlag::Proxy,
            ],
            vec![
                "Bundled severe map family".to_string(),
                "Generic gridded severe map path for supported built-in models"
                    .to_string(),
                "Keeps fixed-depth SCP proxy diagnostics until effective-layer SRH and EBWD are wired"
                    .to_string(),
            ],
        ),
    };
    let id = ProductId::new(ProductKind::Bundled, product.slug());

    ProductSpec {
        id: id.clone(),
        slug: product.slug().to_string(),
        title: title.to_string(),
        kind: ProductKind::Bundled,
        product_metadata: Some(heavy_product_metadata(title, maturity, &flags, id)),
        maturity,
        flags: sorted_flags(&flags),
        render_style: Some("weather_map_family".to_string()),
        aliases: Vec::new(),
        notes,
        blocked_reasons: Vec::new(),
    }
}

fn windowed_product_spec(
    product: HrrrWindowedProduct,
    note: &'static str,
    render_style: &'static str,
) -> ProductSpec {
    let id = ProductId::new(ProductKind::Windowed, product.slug());
    let aliases = legacy_aliases(ProductKind::Windowed, product.slug());
    ProductSpec {
        id: id.clone(),
        slug: product.slug().to_string(),
        title: product.title().to_string(),
        kind: ProductKind::Windowed,
        product_metadata: Some(windowed_product_metadata(
            product,
            product.title(),
            match product {
                HrrrWindowedProduct::Qpf1h
                | HrrrWindowedProduct::Qpf6h
                | HrrrWindowedProduct::Qpf12h
                | HrrrWindowedProduct::Qpf24h
                | HrrrWindowedProduct::QpfTotal => Some("mm"),
                HrrrWindowedProduct::Uh25km1h
                | HrrrWindowedProduct::Uh25km3h
                | HrrrWindowedProduct::Uh25kmRunMax => Some("m^2/s^2"),
                HrrrWindowedProduct::Wind10m1hMax
                | HrrrWindowedProduct::Wind10mRunMax
                | HrrrWindowedProduct::Wind10m0to24hMax
                | HrrrWindowedProduct::Wind10m24to48hMax
                | HrrrWindowedProduct::Wind10m0to48hMax => Some("m/s"),
                HrrrWindowedProduct::Temp2m0to24hMax
                | HrrrWindowedProduct::Temp2m24to48hMax
                | HrrrWindowedProduct::Temp2m0to48hMax
                | HrrrWindowedProduct::Temp2m0to24hMin
                | HrrrWindowedProduct::Temp2m24to48hMin
                | HrrrWindowedProduct::Temp2m0to48hMin
                | HrrrWindowedProduct::Temp2m0to24hRange
                | HrrrWindowedProduct::Temp2m24to48hRange
                | HrrrWindowedProduct::Temp2m0to48hRange => Some("K"),
                HrrrWindowedProduct::Rh2m0to24hMax
                | HrrrWindowedProduct::Rh2m24to48hMax
                | HrrrWindowedProduct::Rh2m0to48hMax
                | HrrrWindowedProduct::Rh2m0to24hMin
                | HrrrWindowedProduct::Rh2m24to48hMin
                | HrrrWindowedProduct::Rh2m0to48hMin
                | HrrrWindowedProduct::Rh2m0to24hRange
                | HrrrWindowedProduct::Rh2m24to48hRange
                | HrrrWindowedProduct::Rh2m0to48hRange => Some("%"),
                HrrrWindowedProduct::Dewpoint2m0to24hMax
                | HrrrWindowedProduct::Dewpoint2m24to48hMax
                | HrrrWindowedProduct::Dewpoint2m0to48hMax
                | HrrrWindowedProduct::Dewpoint2m0to24hMin
                | HrrrWindowedProduct::Dewpoint2m24to48hMin
                | HrrrWindowedProduct::Dewpoint2m0to48hMin
                | HrrrWindowedProduct::Dewpoint2m0to24hRange
                | HrrrWindowedProduct::Dewpoint2m24to48hRange
                | HrrrWindowedProduct::Dewpoint2m0to48hRange => Some("K"),
                HrrrWindowedProduct::Vpd2m0to24hMax
                | HrrrWindowedProduct::Vpd2m24to48hMax
                | HrrrWindowedProduct::Vpd2m0to48hMax
                | HrrrWindowedProduct::Vpd2m0to24hMin
                | HrrrWindowedProduct::Vpd2m24to48hMin
                | HrrrWindowedProduct::Vpd2m0to48hMin
                | HrrrWindowedProduct::Vpd2m0to24hRange
                | HrrrWindowedProduct::Vpd2m24to48hRange
                | HrrrWindowedProduct::Vpd2m0to48hRange => Some("hPa"),
            },
            id,
            &aliases,
        )),
        maturity: ProductMaturity::Operational,
        flags: Vec::new(),
        render_style: Some(render_style.to_string()),
        aliases,
        notes: {
            let mut notes = vec![
                note.to_string(),
                windowed_product_source_note(product).to_string(),
            ];
            notes.extend(legacy_alias_notes(ProductKind::Windowed, product.slug()));
            notes
        },
        blocked_reasons: Vec::new(),
    }
}

fn windowed_product_source_note(product: HrrrWindowedProduct) -> &'static str {
    if product.is_surface_snapshot() {
        "Computed from hourly HRRR 2 m surface snapshots pulled from wrfsfc idx subsets; fixed windows are pointwise max/min/range reductions"
    } else {
        "Backed by HRRR statistical time-window metadata surfaced through grib-core"
    }
}

fn sorted_flags(flags: &[ProductSemanticFlag]) -> Vec<ProductSemanticFlag> {
    let mut values = flags.to_vec();
    values.sort_by_key(|flag| match flag {
        ProductSemanticFlag::Proxy => 0,
        ProductSemanticFlag::Composite => 1,
        ProductSemanticFlag::Alias => 2,
        ProductSemanticFlag::ProofOriented => 3,
    });
    values.dedup();
    values
}

fn typed_metadata(
    display_name: &str,
    category: &str,
    native_units: Option<&str>,
    lineage: ProductLineage,
    maturity: ProductMaturity,
    flags: &[ProductSemanticFlag],
    window: Option<ProductWindowSpec>,
    id: ProductId,
    aliases: &[ProductAliasSpec],
) -> ProductKeyMetadata {
    let mut provenance = ProductProvenance::new(lineage, maturity.into());
    for flag in flags {
        provenance = provenance.with_flag((*flag).into());
    }
    if let Some(window) = window {
        provenance = provenance.with_window(window);
    }
    let mut metadata = ProductKeyMetadata::new(display_name)
        .with_category(category)
        .with_identity(product_identity(id, aliases))
        .with_provenance(provenance);
    if let Some(native_units) = native_units {
        metadata = metadata.with_native_units(native_units);
    }
    metadata
}

fn derived_product_metadata(
    title: &str,
    maturity: ProductMaturity,
    flags: &[ProductSemanticFlag],
    id: ProductId,
    aliases: &[ProductAliasSpec],
) -> ProductKeyMetadata {
    typed_metadata(
        title,
        "derived",
        None,
        ProductLineage::Derived,
        maturity,
        flags,
        None,
        id,
        aliases,
    )
}

fn heavy_product_metadata(
    title: &str,
    maturity: ProductMaturity,
    flags: &[ProductSemanticFlag],
    id: ProductId,
) -> ProductKeyMetadata {
    typed_metadata(
        title,
        "bundled",
        None,
        ProductLineage::Bundled,
        maturity,
        flags,
        None,
        id,
        &[],
    )
}

fn windowed_product_window(product: HrrrWindowedProduct) -> ProductWindowSpec {
    if let Some(window) = surface_snapshot_product_window(product) {
        return window;
    }
    match product {
        HrrrWindowedProduct::Qpf1h => ProductWindowSpec {
            process: StatisticalProcess::Accumulation,
            duration_hours: Some(1),
        },
        HrrrWindowedProduct::Qpf6h => ProductWindowSpec {
            process: StatisticalProcess::Accumulation,
            duration_hours: Some(6),
        },
        HrrrWindowedProduct::Qpf12h => ProductWindowSpec {
            process: StatisticalProcess::Accumulation,
            duration_hours: Some(12),
        },
        HrrrWindowedProduct::Qpf24h => ProductWindowSpec {
            process: StatisticalProcess::Accumulation,
            duration_hours: Some(24),
        },
        HrrrWindowedProduct::QpfTotal => ProductWindowSpec {
            process: StatisticalProcess::Accumulation,
            duration_hours: None,
        },
        HrrrWindowedProduct::Uh25km1h => ProductWindowSpec {
            process: StatisticalProcess::Maximum,
            duration_hours: Some(1),
        },
        HrrrWindowedProduct::Uh25km3h => ProductWindowSpec {
            process: StatisticalProcess::Maximum,
            duration_hours: Some(3),
        },
        HrrrWindowedProduct::Uh25kmRunMax => ProductWindowSpec {
            process: StatisticalProcess::Maximum,
            duration_hours: None,
        },
        HrrrWindowedProduct::Wind10m1hMax => ProductWindowSpec {
            process: StatisticalProcess::Maximum,
            duration_hours: Some(1),
        },
        HrrrWindowedProduct::Wind10mRunMax => ProductWindowSpec {
            process: StatisticalProcess::Maximum,
            duration_hours: None,
        },
        HrrrWindowedProduct::Wind10m0to24hMax | HrrrWindowedProduct::Wind10m24to48hMax => {
            ProductWindowSpec {
                process: StatisticalProcess::Maximum,
                duration_hours: Some(24),
            }
        }
        HrrrWindowedProduct::Wind10m0to48hMax => ProductWindowSpec {
            process: StatisticalProcess::Maximum,
            duration_hours: Some(48),
        },
        _ => unreachable!("surface snapshot window products are handled before match"),
    }
}

fn surface_snapshot_product_window(product: HrrrWindowedProduct) -> Option<ProductWindowSpec> {
    if !product.is_surface_snapshot() {
        return None;
    }
    let slug = product.slug();
    let process = if slug.ends_with("_max") {
        StatisticalProcess::Maximum
    } else if slug.ends_with("_min") {
        StatisticalProcess::Minimum
    } else {
        StatisticalProcess::Range
    };
    let duration_hours = if slug.contains("_0_48h_") {
        Some(48)
    } else {
        Some(24)
    };
    Some(ProductWindowSpec {
        process,
        duration_hours,
    })
}

fn windowed_product_metadata(
    product: HrrrWindowedProduct,
    title: &str,
    native_units: Option<&str>,
    id: ProductId,
    aliases: &[ProductAliasSpec],
) -> ProductKeyMetadata {
    typed_metadata(
        title,
        "windowed",
        native_units,
        ProductLineage::Windowed,
        ProductMaturity::Operational,
        &[],
        Some(windowed_product_window(product)),
        id,
        aliases,
    )
}

fn legacy_alias_route_for_direct_slug(slug: &str) -> Option<&'static LegacyProductAliasRoute> {
    LEGACY_NON_ECAPE_ALIAS_ROUTES
        .iter()
        .find(|route| route.alias_slug == slug)
}

fn legacy_aliases(kind: ProductKind, canonical_slug: &str) -> Vec<ProductAliasSpec> {
    LEGACY_NON_ECAPE_ALIAS_ROUTES
        .iter()
        .filter(|route| route.canonical_kind == kind && route.canonical_slug == canonical_slug)
        .map(|route| ProductAliasSpec {
            id: ProductId::new(kind, route.alias_slug),
            slug: route.alias_slug.to_string(),
            title: route.alias_title.to_string(),
            note: route.note.to_string(),
        })
        .collect()
}

fn legacy_alias_notes(kind: ProductKind, canonical_slug: &str) -> Vec<String> {
    LEGACY_NON_ECAPE_ALIAS_ROUTES
        .iter()
        .filter(|route| route.canonical_kind == kind && route.canonical_slug == canonical_slug)
        .map(|route| route.note.to_string())
        .collect()
}

fn product_identity(id: ProductId, aliases: &[ProductAliasSpec]) -> CanonicalProductIdentity {
    aliases
        .iter()
        .fold(CanonicalProductIdentity::new(id), |identity, alias| {
            identity.with_alias_slug(alias.id.as_slug())
        })
}

fn with_canonical_identity(
    metadata: ProductKeyMetadata,
    id: ProductId,
    aliases: &[ProductAliasSpec],
) -> ProductKeyMetadata {
    metadata.with_identity(product_identity(id, aliases))
}

fn direct_render_style(recipe: &PlotRecipe) -> &'static str {
    match recipe.slug {
        "cloud_cover_levels" | "precipitation_type" => "weather_panel_grid",
        _ => render_style_name(recipe.style),
    }
}

fn direct_entry_notes(slug: &str) -> Vec<String> {
    match slug {
        "cloud_cover_levels" => vec![
            "Rendered as an accurate HRRR direct composite panel over low, middle, and high cloud-cover component fields".to_string(),
        ],
        "precipitation_type" => vec![
            "Rendered as an accurate HRRR direct composite panel over categorical rain, freezing-rain, ice-pellet, and snow phase flags".to_string(),
        ],
        "snowfall_window" => vec!["Explicit SNOWFALLAC depth in mm of snow, from model initialization or an exact selected baseline".into()],
        "snow_10to1_window" => vec!["Interval changes in SNOWNC plus optional GRAUPELNC times 10; mm of snow; history cadence is retained".into()],
        "snow_kuchera_window" => vec!["Interval changes in frozen liquid equivalent weighted by interval-end native column Tmax through 500 hPa; empirical snow-to-liquid ratio; mm of snow".into()],
        _ => Vec::new(),
    }
}

fn derived_entry_flags(slug: &str) -> Vec<ProductSemanticFlag> {
    match slug {
        "scp_mu_0_3km_0_6km_proxy" => vec![ProductSemanticFlag::Proxy],
        _ => Vec::new(),
    }
}

fn derived_entry_notes(slug: &str, experimental: bool) -> Vec<String> {
    let mut notes = Vec::new();
    match slug {
        "ehi_0_1km" => notes.push(
            "Depth-specific EHI using sbCAPE with 0-1 km SRH; not an effective-layer diagnostic"
                .to_string(),
        ),
        "ehi_0_3km" => notes.push(
            "Depth-specific EHI using sbCAPE with 0-3 km SRH; not an effective-layer diagnostic"
                .to_string(),
        ),
        "scp_mu_0_3km_0_6km_proxy" => notes.push(
            "Uses muCAPE with 0-3 km SRH and 0-6 km bulk shear; kept explicit because effective-layer SCP is still blocked"
                .to_string(),
        ),
        _ => {}
    }
    notes.extend(legacy_alias_notes(ProductKind::Derived, slug));
    if experimental {
        notes.push(
            "Current proof/product runner labels this as a proxy or experimental diagnostic"
                .to_string(),
        );
    }
    notes
}

fn render_style_name(style: RenderStyle) -> &'static str {
    match style {
        RenderStyle::AirQualityPm25 => "air_quality_pm25",
        RenderStyle::AirQualityOzone => "air_quality_ozone",
        RenderStyle::AerosolOpticalDepth => "aerosol_optical_depth",
        RenderStyle::DustConcentration => "dust_concentration",
        RenderStyle::WeatherWbgt => "weatherwbgt",
        RenderStyle::WeatherHdw => "weatherhdw",
        RenderStyle::WeatherHaines => "weatherhaines",
        RenderStyle::WeatherLightningThreat => "weatherlightningthreat",
        RenderStyle::WeatherFlashRate => "weatherflashrate",
        RenderStyle::WeatherCape => "weather_cape",
        RenderStyle::WeatherCin => "weather_cin",
        RenderStyle::WeatherReflectivity => "weather_reflectivity",
        RenderStyle::WeatherUh => "weather_uh",
        RenderStyle::WeatherVerticalVelocity => "weather_vertical_velocity",
        RenderStyle::WeatherTemperature => "weather_temperature",
        RenderStyle::WeatherDewpoint => "weather_dewpoint",
        RenderStyle::WeatherSpecificHumidity => "weather_specific_humidity",
        RenderStyle::WeatherOmega => "weather_omega",
        RenderStyle::WeatherRh => "weather_rh",
        RenderStyle::WeatherProbability => "weather_probability",
        RenderStyle::WeatherWinds => "weather_winds",
        RenderStyle::WeatherHeight => "weather_height",
        RenderStyle::WeatherPressure => "weather_pressure",
        RenderStyle::WeatherWindGust => "weather_wind_gust",
        RenderStyle::WeatherCloudCover => "weather_cloud_cover",
        RenderStyle::WeatherPrecipitableWater => "weather_precipitable_water",
        RenderStyle::WeatherIvt => "weather_ivt",
        RenderStyle::WeatherQpf => "weather_qpf",
        RenderStyle::WeatherSnowfall => "weather_snowfall",
        RenderStyle::WeatherSnowDepthChange => "weather_snow_depth_change",
        RenderStyle::WeatherHail => "weather_hail",
        RenderStyle::WeatherCategorical => "weather_categorical",
        RenderStyle::WeatherVisibility => "weather_visibility",
        RenderStyle::WeatherRadarReflectivity => "weather_radar_reflectivity",
        RenderStyle::WeatherSatellite => "weather_satellite",
        RenderStyle::WeatherLightning => "weather_lightning",
        RenderStyle::WeatherVorticity => "weather_vorticity",
        RenderStyle::WeatherStp => "weather_stp",
        RenderStyle::WeatherScp => "weather_scp",
        RenderStyle::WeatherEhi => "weather_ehi",
        RenderStyle::WeatherTerrain => "weather_terrain",
        RenderStyle::WeatherWindComponent => "weather_wind_component",
        RenderStyle::WeatherIsothermHeight => "weather_isotherm_height",
        RenderStyle::WeatherSupercooledWaterPath => "weather_supercooled_water_path",
        RenderStyle::WeatherHydrometeorMixingRatio => "weather_hydrometeor_mixing_ratio",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_specs_skip_legacy_aliases() {
        let specs = direct_product_specs();
        assert!(specs.iter().all(|spec| spec.slug != "1h_qpf"));
        assert!(specs.iter().all(|spec| spec.kind == ProductKind::Direct));
    }

    #[test]
    fn derived_specs_capture_identity_aliases_and_provenance() {
        let theta_e = supported_derived_product_specs()
            .into_iter()
            .find(|spec| spec.slug == "theta_e_2m_10m_winds")
            .expect("theta-e spec should exist");
        assert_eq!(
            theta_e.id,
            ProductId::new(ProductKind::Derived, "theta_e_2m_10m_winds")
        );
        assert!(
            theta_e
                .aliases
                .iter()
                .any(|alias| alias.slug == "2m_theta_e_10m_winds")
        );
        assert!(
            theta_e
                .aliases
                .iter()
                .any(|alias| alias.id
                    == ProductId::new(ProductKind::Derived, "2m_theta_e_10m_winds"))
        );
        let identity = theta_e
            .product_metadata
            .as_ref()
            .and_then(|metadata| metadata.identity.as_ref())
            .expect("derived spec should expose canonical identity");
        assert_eq!(identity.canonical, theta_e.id);
        assert!(
            identity
                .alias_slugs
                .contains(&"2m_theta_e_10m_winds".to_string())
        );
        assert_eq!(
            theta_e
                .product_metadata
                .as_ref()
                .and_then(|metadata| metadata.provenance.as_ref())
                .expect("derived spec should expose provenance")
                .lineage,
            ProductLineage::Derived
        );
    }

    #[test]
    fn windowed_specs_capture_window_metadata() {
        let qpf_1h = windowed_product_specs()
            .into_iter()
            .find(|spec| spec.slug == "qpf_1h")
            .expect("qpf_1h spec should exist");
        assert_eq!(qpf_1h.id, ProductId::new(ProductKind::Windowed, "qpf_1h"));
        assert_eq!(
            qpf_1h
                .product_metadata
                .as_ref()
                .and_then(|metadata| metadata.provenance.as_ref())
                .and_then(|provenance| provenance.window.clone()),
            Some(ProductWindowSpec {
                process: StatisticalProcess::Accumulation,
                duration_hours: Some(1),
            })
        );
        assert_eq!(
            qpf_1h
                .product_metadata
                .as_ref()
                .and_then(|metadata| metadata.identity.as_ref())
                .expect("windowed spec should expose canonical identity")
                .canonical,
            qpf_1h.id
        );
    }
}
