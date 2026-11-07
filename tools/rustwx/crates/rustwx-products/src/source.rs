use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductSourceMode {
    #[default]
    Canonical,
    Fastest,
}

impl ProductSourceMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Canonical => "canonical",
            Self::Fastest => "fastest",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductSourceRoute {
    CanonicalDerived,
    DirectNativeExact,
    DirectNativeCompositeExact,
    NativeExact,
    NativeProxy,
    CheapDerived,
    BlockedNoFastRoute,
}

impl ProductSourceRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalDerived => "canonical_derived",
            Self::DirectNativeExact => "direct_native_exact",
            Self::DirectNativeCompositeExact => "direct_native_composite_exact",
            Self::NativeExact => "native_exact",
            Self::NativeProxy => "native_proxy",
            Self::CheapDerived => "cheap_derived",
            Self::BlockedNoFastRoute => "blocked_no_fast_route",
        }
    }
}

pub fn direct_route_for_recipe_slug(slug: &str) -> ProductSourceRoute {
    match slug {
        "cloud_cover_levels" | "precipitation_type" | "composite_reflectivity_uh" => {
            ProductSourceRoute::DirectNativeCompositeExact
        }
        _ if rustwx_models::is_uh_layer_overlay_recipe(slug) => {
            ProductSourceRoute::DirectNativeCompositeExact
        }
        _ => ProductSourceRoute::DirectNativeExact,
    }
}
