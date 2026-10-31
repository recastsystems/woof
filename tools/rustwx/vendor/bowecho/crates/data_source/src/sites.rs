//! One radar-site model for the whole world (v0.29 spec §1,
//! `docs/v029-engine-spec.md`).
//!
//! Today the app holds two site worlds: the US Level-II catalog
//! (`Vec<RadarSite>`, addressed by index) and the international registry
//! ([`crate::international::intl_static_sites`]), stitched together at
//! every feature by ad-hoc predicates. This module is the one catalog API
//! that Phase 3 migrates consumers onto: string-keyed [`SiteRef`]s that
//! survive catalog reorder and serialize into settings, an exhaustive
//! [`SiteKind`] so a US-only feature cannot be written without a visible
//! match on the site kind, and resolved [`SiteRecord`] rows for pickers,
//! beam rankings, markers, and engines.
//!
//! Phase 1 (spec §11 Milestone C) adds the module WITHOUT migrating any
//! consumer: everything here is compiled-in data — the embedded US table
//! (`embedded_sites.rs`), the community research-feed table
//! (`community_feeds.rs`), and the providers' embedded international
//! tables — so every function is network-free and safe on the UI thread.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::{community_feeds, embedded_sites, international};

/// The US-operated Level-II network's country label. The embedded catalog
/// is the NEXRAD/TDWR program plus community research feeds — including
/// overseas military sites (RODN Kadena, RKJK Kunsan, PGUA Andersen) —
/// so this names the operating network, not the soil under the antenna.
const US_NETWORK_COUNTRY: &str = "United States";

/// Provenance suffix for community research feeds, matching the grammar
/// `BeamCandidate.origin` (main.rs) already uses for non-CONUS rows.
const RESEARCH_FEED_ORIGIN: &str = "research feed";

/// TDWR ids in the embedded US Level-II catalog, as CATALOG DATA — the
/// replacement for `site_is_tdwr`'s `'T'`-prefix heuristic (main.rs),
/// whose one documented exception (TJUA, San Juan's S-band WSR-88D) and
/// undocumented leak class (JMA sites TAKA/TANE/TOJI share the prefix)
/// both disappear once classification is a table lookup.
///
/// Seeded by porting the current heuristic over `EMBEDDED_SITES` — every
/// `T`-prefixed id except TJUA, which keeps TLKA2 (Talkeetna) riding
/// along exactly as v0.28.2 classifies it. The heuristic itself survives
/// ONLY in this module's seed-pinning test (spec §11 C.1). Sorted, for
/// `binary_search`.
const TDWR_SITES: &[&str] = &[
    "TADW", "TATL", "TBNA", "TBOS", "TBWI", "TCLT", "TCMH", "TCVG", "TDAL", "TDAY", "TDCA", "TDEN",
    "TDFW", "TDTW", "TEWR", "TFLL", "THOU", "TIAD", "TIAH", "TICH", "TIDS", "TJFK", "TLAS",
    "TLKA2", "TLVE", "TMCI", "TMCO", "TMDW", "TMEM", "TMIA", "TMKE", "TMSP", "TMSY", "TOKC",
    "TORD", "TPBI", "TPHL", "TPHX", "TPIT", "TRDU", "TSDF", "TSJU", "TSLC", "TSTL", "TTPA", "TTUL",
];

/// One radar site anywhere on earth. Strings, never indices: survives
/// catalog reorder, serializes into settings, Eq/Hash for dedupe keys.
#[derive(Clone, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum SiteRef {
    /// Embedded US Level-II catalog id (WSR-88D, TDWR, research feeds),
    /// canonically uppercase: `"KTLX"`, `"TOKC"`, `"KCRI"`.
    Us { level2_id: String },
    /// [`crate::international`] registry site, case-significant:
    /// `{"ord","deess"}`, `{"smhi","angelholm"}`, `{"jma","TAKA"}`.
    Intl {
        provider_id: String,
        site_id: String,
    },
}

/// The exhaustive-match lever. Deliberately NOT `#[non_exhaustive]`:
/// adding a variant is a compile error at every consumer — that is the
/// type-system parity outcome.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SiteKind {
    /// WSR-88D, incl. TJUA — the exception becomes catalog data.
    Wsr88d,
    /// Terminal Doppler, from [`TDWR_SITES`] — NOT a `'T'`-prefix
    /// heuristic.
    Tdwr,
    /// Community research feed (`community_feeds.rs` — what
    /// `community_feed_for_site()` recognizes today).
    Research,
    /// International registry site.
    Intl { provider_id: String },
}

/// Resolved site row — what pickers, beam rankings, markers, and engines
/// consume.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteRecord {
    /// Known weather-radar band. Unknown/mixed-band provider rows remain
    /// None rather than being silently assigned S-band scattering.
    pub band: Option<&'static str>,
    pub site: SiteRef,
    pub kind: SiteKind,
    /// `"KTLX Norman"` | `"Ängelholm"` — the US grammar is `id name`, the
    /// international grammar is the provider's own picker label.
    pub label: String,
    /// Weak provenance suffix for rows outside the home catalog
    /// (`"SMHI Sweden"`, `"research feed"`); `None` for WSR-88D/TDWR —
    /// `BeamCandidate.origin` (main.rs) anticipated exactly this.
    pub origin: Option<String>,
    pub lat_lon: Option<(f32, f32)>,
    /// The operating network's country ([`IntlSite::country`] for
    /// international rows; [`US_NETWORK_COUNTRY`] for the US catalog).
    ///
    /// [`IntlSite::country`]: crate::international::IntlSite
    pub country: String,
}

impl SiteRef {
    /// Settings/favorites encoding: `"KTLX"` | `"intl:smhi:angelholm"`.
    /// Backward compatible — US ids never contain `':'`, so every
    /// existing settings value parses as [`SiteRef::Us`].
    pub fn settings_key(&self) -> String {
        match self {
            Self::Us { level2_id } => level2_id.clone(),
            Self::Intl {
                provider_id,
                site_id,
            } => format!("intl:{provider_id}:{site_id}"),
        }
    }

    /// Decode a settings/favorites key. `':'`-free strings are bare US
    /// ids and normalize to uppercase exactly like [`crate::RadarSite::new`]
    /// (and like the favorites store, which uppercases bare keys on
    /// write). `intl:{provider}:{site}` keys preserve case — ORD/SMHI
    /// site ids are case-significant lowercase, JMA's uppercase. Total:
    /// malformed `':'` keys fall back to a (unresolvable) US ref rather
    /// than dropping the user's data.
    pub fn parse_settings_key(key: &str) -> SiteRef {
        if let Some(rest) = key.strip_prefix("intl:")
            && let Some((provider_id, site_id)) = rest.split_once(':')
            && !provider_id.is_empty()
            && !site_id.is_empty()
        {
            return Self::Intl {
                provider_id: provider_id.to_owned(),
                site_id: site_id.to_owned(),
            };
        }
        Self::Us {
            level2_id: key.trim().to_ascii_uppercase(),
        }
    }
}

/// THE catalog lookup: resolve a ref against the compiled-in tables. No
/// network — US rows come from the embedded catalog + community feeds,
/// international rows from the providers' embedded tables — so this is
/// safe on the UI thread. `None` = not a compiled-in site (e.g. an S3
/// listing surfaced a brand-new US site, or a saved key went stale). US
/// lookup is case-insensitive (that world is case-insensitive today);
/// international lookup is exact.
pub fn resolve(site: &SiteRef) -> Option<SiteRecord> {
    match site {
        SiteRef::Us { level2_id } => catalog()
            .iter()
            .find(|record| {
                matches!(&record.site, SiteRef::Us { level2_id: known }
                    if known.eq_ignore_ascii_case(level2_id))
            })
            .cloned(),
        SiteRef::Intl { .. } => catalog()
            .iter()
            .find(|record| &record.site == site)
            .cloned(),
    }
}

/// Every compiled-in site, both worlds: the embedded US catalog (table
/// order), then community research feeds (table order), then every
/// international provider's embedded table in registry order.
pub fn all_sites() -> impl Iterator<Item = SiteRecord> {
    catalog().iter().cloned()
}

/// Sites within `radius_km` of a point, nearest first, with their
/// great-circle distances. Sites without coordinates never rank.
pub fn sites_near(lat: f32, lon: f32, radius_km: f32) -> Vec<(SiteRecord, f32)> {
    let mut hits: Vec<(SiteRecord, f32)> = catalog()
        .iter()
        .filter_map(|record| {
            let (site_lat, site_lon) = record.lat_lon?;
            let distance_km = haversine_km(lat, lon, site_lat, site_lon);
            (distance_km <= radius_km).then(|| (record.clone(), distance_km))
        })
        .collect();
    hits.sort_by(|left, right| left.1.total_cmp(&right.1));
    hits
}

/// The memoized union catalog (pure compiled-in data; built once).
fn catalog() -> &'static [SiteRecord] {
    static CATALOG: OnceLock<Vec<SiteRecord>> = OnceLock::new();
    CATALOG.get_or_init(build_catalog)
}

fn build_catalog() -> Vec<SiteRecord> {
    let mut records = Vec::new();
    let mut seen_us_ids = BTreeSet::new();
    for &(id, name, latitude_deg, longitude_deg) in embedded_sites::EMBEDDED_SITES {
        seen_us_ids.insert(id.to_ascii_uppercase());
        records.push(SiteRecord {
            band: if id.len()!=4 {None} else if matches!(us_catalog_site_kind(id),SiteKind::Tdwr) {Some("C")} else {Some("S")},
            site: SiteRef::Us {
                level2_id: id.to_owned(),
            },
            kind: us_catalog_site_kind(id),
            label: format!("{id} {name}"),
            origin: None,
            lat_lon: Some((latitude_deg, longitude_deg)),
            country: US_NETWORK_COUNTRY.to_owned(),
        });
    }
    for feed in community_feeds::community_feeds() {
        // Ids are disjoint from the embedded catalog today; the guard
        // keeps the embedded row authoritative if that ever changes
        // (same precedence as main.rs's site_catalog_with_community_feeds).
        if !seen_us_ids.insert(feed.id.to_ascii_uppercase()) {
            continue;
        }
        records.push(SiteRecord {
            band: None,
            site: SiteRef::Us {
                level2_id: feed.id.to_owned(),
            },
            kind: SiteKind::Research,
            label: format!("{} {}", feed.id, feed.label),
            origin: Some(RESEARCH_FEED_ORIGIN.to_owned()),
            lat_lon: Some((feed.latitude_deg, feed.longitude_deg)),
            country: US_NETWORK_COUNTRY.to_owned(),
        });
    }
    for provider in international::intl_providers() {
        for site in provider.static_sites() {
            records.push(SiteRecord {
                band: PROVIDER_BANDS.iter().find(|(id,_)|*id==site.provider_id).map(|(_,band)|*band),
                site: SiteRef::Intl {
                    provider_id: site.provider_id.to_owned(),
                    site_id: site.site_id.clone(),
                },
                kind: SiteKind::Intl {
                    provider_id: site.provider_id.to_owned(),
                },
                label: site.label.clone(),
                origin: Some(provider.label().to_owned()),
                lat_lon: match (site.latitude_deg, site.longitude_deg) {
                    (Some(latitude_deg), Some(longitude_deg)) => {
                        Some((latitude_deg, longitude_deg))
                    }
                    _ => None,
                },
                country: site.country.to_owned(),
            });
        }
    }
    records
}

// Agency metadata, not a geographic heuristic. Mixed-band networks need
// per-site rows and are deliberately not assigned a network-wide band.
// Sources: FMI radar network, DMI radar specifications, SMHI frequency-plan
// response (2025-01-10), DWD network description (doi:10.5194/amt-18-5177-2025),
// and SHMU radar-network technical poster (2025). See the engine radar docs.
const PROVIDER_BANDS: &[(&str,&str)] = &[
    ("dwd","C"),("fmi","C"),("dmi","C"),("smhi","C"),("shmu","C"),
];

#[cfg(test)]
mod band_tests {
    use super::*;
    #[test]
    fn weather_bands_do_not_follow_an_identifier_prefix() {
        let band=|id:&str|resolve(&SiteRef::Us {level2_id:id.into()}).unwrap().band;
        assert_eq!(band("TOKC"),Some("C"));
        assert_eq!(band("TJUA"),Some("S"));
        assert_eq!(band("TLKA2"),None);
    }
    #[test]
    fn known_agency_rows_carry_c_band_and_mixed_providers_stay_unknown() {
        let sites:Vec<_>=all_sites().collect();
        for (provider,band) in PROVIDER_BANDS {
            let rows:Vec<_>=sites.iter().filter(|s|matches!(&s.site,SiteRef::Intl {provider_id,..} if provider_id.as_str()==*provider)).collect();
            assert!(!rows.is_empty(),"missing provider {provider}");
            assert!(rows.iter().all(|s|s.band==Some(*band)));
        }
        assert!(sites.iter().filter(|s|matches!(&s.site,SiteRef::Intl {provider_id,..} if provider_id=="australia_nci"))
            .all(|s|s.band.is_none()));
    }
}

/// Kind of an embedded-US-catalog id: TDWR table membership, else
/// WSR-88D (community feeds classify at their own insertion site).
fn us_catalog_site_kind(level2_id: &str) -> SiteKind {
    if TDWR_SITES.binary_search(&level2_id).is_ok() {
        SiteKind::Tdwr
    } else {
        SiteKind::Wsr88d
    }
}

/// Great-circle distance on the R = 6371 km sphere (haversine; Sinnott
/// 1984, "Virtues of the Haversine", Sky & Telescope 68(2):159), in the
/// same atan2 formulation as `haversine_km` in main.rs so rankings agree
/// when consumers migrate in Phase 3.
fn haversine_km(lat_a: f32, lon_a: f32, lat_b: f32, lon_b: f32) -> f32 {
    let earth_radius_km = 6371.0_f32;
    let d_lat = (lat_b - lat_a).to_radians();
    let d_lon = (lon_b - lon_a).to_radians();
    let lat_a = lat_a.to_radians();
    let lat_b = lat_b.to_radians();
    let a = (d_lat / 2.0).sin().powi(2) + lat_a.cos() * lat_b.cos() * (d_lon / 2.0).sin().powi(2);
    2.0 * earth_radius_km * a.sqrt().atan2((1.0 - a).max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn us(level2_id: &str) -> SiteRef {
        SiteRef::Us {
            level2_id: level2_id.to_owned(),
        }
    }

    fn intl(provider_id: &str, site_id: &str) -> SiteRef {
        SiteRef::Intl {
            provider_id: provider_id.to_owned(),
            site_id: site_id.to_owned(),
        }
    }

    #[test]
    fn settings_key_round_trips_every_variant() {
        for site in [
            us("KTLX"),
            us("TOKC"),
            us("KCRI"),
            us("TLKA2"),
            intl("ord", "deess"),
            intl("smhi", "angelholm"),
            intl("jma", "TAKA"),
        ] {
            let key = site.settings_key();
            assert_eq!(
                SiteRef::parse_settings_key(&key),
                site,
                "round trip through '{key}'"
            );
        }
        assert_eq!(us("KTLX").settings_key(), "KTLX");
        assert_eq!(
            intl("smhi", "angelholm").settings_key(),
            "intl:smhi:angelholm"
        );
    }

    /// Backward compatibility: every existing settings value is a bare US
    /// id and must keep parsing as one, normalized uppercase exactly like
    /// `RadarSite::new` and the favorites store.
    #[test]
    fn bare_keys_parse_as_uppercased_us_refs() {
        assert_eq!(SiteRef::parse_settings_key("KTLX"), us("KTLX"));
        assert_eq!(SiteRef::parse_settings_key("ktlx"), us("KTLX"));
        assert_eq!(SiteRef::parse_settings_key(" ktlx "), us("KTLX"));
        // A bare intl-LOOKING id is still a bare key — only the
        // `intl:{provider}:{site}` grammar reaches the Intl variant.
        assert_eq!(SiteRef::parse_settings_key("deess"), us("DEESS"));
    }

    /// ORD/SMHI ids are case-significant lowercase and JMA's uppercase:
    /// intl keys must never be case-folded in either direction.
    #[test]
    fn intl_keys_preserve_case_exactly() {
        assert_eq!(
            SiteRef::parse_settings_key("intl:ord:deess"),
            intl("ord", "deess")
        );
        assert_eq!(
            SiteRef::parse_settings_key("intl:jma:TAKA"),
            intl("jma", "TAKA")
        );
        assert_eq!(intl("jma", "TAKA").settings_key(), "intl:jma:TAKA");
        // An uppercased corruption is NOT silently accepted as intl — it
        // falls back to a bare (unresolvable) US ref, stably.
        let corrupted = SiteRef::parse_settings_key("INTL:ORD:DEESS");
        assert_eq!(corrupted, us("INTL:ORD:DEESS"));
        assert_eq!(
            SiteRef::parse_settings_key(&corrupted.settings_key()),
            corrupted,
            "malformed keys still round-trip stably"
        );
    }

    /// Malformed `':'` keys degrade to unresolvable US refs instead of
    /// panicking or dropping data.
    #[test]
    fn malformed_namespaced_keys_fall_back_to_us() {
        for key in ["intl:", "intl:ord", "intl::deess", "intl:ord:", ":"] {
            let parsed = SiteRef::parse_settings_key(key);
            assert!(
                matches!(parsed, SiteRef::Us { .. }),
                "'{key}' must fall back to Us, got {parsed:?}"
            );
        }
    }

    #[test]
    fn tjua_is_a_wsr88d_not_a_tdwr() {
        // TJUA is San Juan's S-band WSR-88D (full 460 km range) and only
        // happens to share the TDWR `T` prefix — the exception is now
        // catalog data, not a special case at every gate.
        let record = resolve(&us("TJUA")).expect("TJUA resolves");
        assert_eq!(record.kind, SiteKind::Wsr88d);
        assert!(!TDWR_SITES.contains(&"TJUA"));
    }

    #[test]
    fn terminal_radars_classify_from_catalog_data() {
        for id in ["TOKC", "TDAL", "TJFK", "TLKA2"] {
            let record = resolve(&us(id)).expect("TDWR id resolves");
            assert_eq!(record.kind, SiteKind::Tdwr, "{id}");
            assert_eq!(record.origin, None, "{id}: home catalog carries no origin");
        }
        let ktlx = resolve(&us("KTLX")).expect("KTLX resolves");
        assert_eq!(ktlx.kind, SiteKind::Wsr88d);
        assert_eq!(ktlx.label, "KTLX Norman");
        assert_eq!(ktlx.country, US_NETWORK_COUNTRY);
    }

    /// The `'T'`-prefix heuristic is permitted HERE ONLY (spec §11 C.1),
    /// pinning the ported table against the embedded catalog: the two can
    /// never drift while both exist, and the shipped classification stays
    /// a data lookup.
    #[test]
    fn tdwr_table_matches_the_ported_t_prefix_seed() {
        assert!(TDWR_SITES.is_sorted(), "binary_search needs sorted data");
        let seeded: Vec<&str> = embedded_sites::EMBEDDED_SITES
            .iter()
            .map(|&(id, ..)| id)
            .filter(|id| id.starts_with('T') && !id.eq_ignore_ascii_case("TJUA"))
            .collect();
        let mut expected = seeded;
        expected.sort_unstable();
        assert_eq!(TDWR_SITES, expected.as_slice());
    }

    /// The JMA TAKA/TANE/TOJI leak (uppercase `T`-prefixed intl ids read
    /// as TDWRs by the old heuristic) is unrepresentable here: intl refs
    /// classify by provider, never by id shape.
    #[test]
    fn jma_t_prefixed_sites_classify_intl_not_tdwr() {
        for site_id in ["TAKA", "TANE", "TOJI"] {
            let record =
                resolve(&intl("jma", site_id)).unwrap_or_else(|| panic!("jma/{site_id} resolves"));
            assert_eq!(
                record.kind,
                SiteKind::Intl {
                    provider_id: "jma".to_owned()
                },
                "{site_id}"
            );
            assert_eq!(record.origin.as_deref(), Some("JMA Japan"), "{site_id}");
        }
    }

    #[test]
    fn community_feeds_classify_research_with_the_provenance_suffix() {
        for id in ["KCRI", "FWLX", "LARE", "K08D"] {
            let record = resolve(&us(id)).unwrap_or_else(|| panic!("{id} resolves"));
            assert_eq!(record.kind, SiteKind::Research, "{id}");
            assert_eq!(record.origin.as_deref(), Some(RESEARCH_FEED_ORIGIN), "{id}");
            assert_eq!(record.country, US_NETWORK_COUNTRY, "{id}");
        }
    }

    #[test]
    fn us_resolution_is_case_insensitive_with_canonical_records() {
        let record = resolve(&us("ktlx")).expect("lowercase US lookup resolves");
        assert_eq!(record.site, us("KTLX"));
        // Intl resolution is exact: the case-folded id is a different key.
        assert!(resolve(&intl("smhi", "angelholm")).is_some());
        assert!(resolve(&intl("smhi", "ANGELHOLM")).is_none());
        assert!(resolve(&us("ZZZZ")).is_none());
        assert!(resolve(&intl("nope", "nowhere")).is_none());
    }

    /// The union catalog covers both worlds completely and uniquely, and
    /// every row resolves back to itself through its own ref.
    #[test]
    fn catalog_unions_both_worlds_uniquely_and_resolves_every_row() {
        let records: Vec<SiteRecord> = all_sites().collect();
        assert_eq!(
            records.len(),
            embedded_sites::EMBEDDED_SITES.len()
                + community_feeds::community_feeds().len()
                + international::intl_static_sites().len(),
            "embedded + community + intl, no overlaps today"
        );
        let mut keys: Vec<String> = records
            .iter()
            .map(|record| record.site.settings_key())
            .collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), before, "settings keys must be unique");
        for record in &records {
            assert!(!record.label.is_empty(), "{:?}", record.site);
            assert_eq!(
                resolve(&record.site).as_ref(),
                Some(record),
                "{:?} must resolve to its own row",
                record.site
            );
        }
        // Memoized: repeated iteration is over the same backing slice.
        assert!(std::ptr::eq(catalog(), catalog()));
    }

    /// Spec §11 C.1: every international static-catalog entry resolves,
    /// with the provider-derived kind and provenance.
    #[test]
    fn every_intl_static_site_resolves_with_provider_kind() {
        for site in international::intl_static_sites() {
            let record = resolve(&intl(site.provider_id, &site.site_id))
                .unwrap_or_else(|| panic!("{}/{} resolves", site.provider_id, site.site_id));
            assert_eq!(
                record.kind,
                SiteKind::Intl {
                    provider_id: site.provider_id.to_owned()
                }
            );
            assert_eq!(record.label, site.label);
            assert_eq!(record.country, site.country);
            assert!(
                record.origin.as_deref().is_some_and(|o| !o.is_empty()),
                "{}/{}: intl rows carry provider provenance",
                site.provider_id,
                site.site_id
            );
        }
    }

    /// US and international sites rank in ONE distance ordering — the
    /// Okinawa interleave: JMA Itokazu (ITOK) sits nearer the probe point
    /// than Kadena AB's US-catalog RODN, with more JMA island-arc radars
    /// behind both.
    #[test]
    fn sites_near_interleaves_both_worlds_by_distance() {
        let hits = sites_near(26.2, 127.8, 450.0);
        assert!(hits.len() >= 4, "expected ITOK/RODN/FUNC/ISHI at least");
        for pair in hits.windows(2) {
            assert!(pair[0].1 <= pair[1].1, "nearest first");
        }
        assert!(hits.iter().all(|(_, distance_km)| *distance_km <= 450.0));
        assert_eq!(hits[0].0.site, intl("jma", "ITOK"));
        assert_eq!(hits[1].0.site, us("RODN"));
        assert!(
            hits.iter()
                .any(|(record, _)| record.site == intl("jma", "FUNC"))
        );
        assert!(
            hits.iter()
                .any(|(record, _)| record.site == intl("jma", "ISHI"))
        );

        // A tight radius keeps only the nearest site.
        let tight = sites_near(26.2, 127.8, 10.0);
        assert_eq!(
            tight
                .iter()
                .map(|(record, _)| record.site.clone())
                .collect::<Vec<_>>(),
            vec![intl("jma", "ITOK")]
        );
        assert!(sites_near(0.0, -160.0, 100.0).is_empty(), "open ocean");
    }
}
