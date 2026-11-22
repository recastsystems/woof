//! Dated public object layouts. New archive doors are metadata rows.
use crate::{ModelError, normalize_token};
use rustwx_core::{ModelRunRequest, SourceId};
use serde::Deserialize;
use std::sync::LazyLock;

#[derive(Deserialize)]
struct Segment { through: u16, step: u16 }
#[derive(Deserialize)]
struct Layout {
    source_id: String,
    rust_source: String,
    product: String,
    #[serde(default)]
    product_aliases: Vec<String>,
    #[serde(default)]
    exclusive_source: bool,
    base: String,
    record_start: String,
    cycle_until: Option<String>,
    path_template: String,
    #[serde(default)]
    lead_segments: Vec<Segment>,
}
#[derive(Deserialize)]
struct Document { layouts: Vec<Layout> }
static TABLE: LazyLock<Document> = LazyLock::new(|| serde_json::from_str(include_str!(
    "../../../../../woof/authorities/rw-wps-archive-layouts.v1.json"
)).expect("packaged archive layout table"));

pub(super) fn object_url(source: SourceId, request: &ModelRunRequest)
    -> Option<Result<String, ModelError>>
{
    let rows: Vec<_> = TABLE.layouts.iter().filter(|row|
        row.source_id == request.model.as_str() && row.rust_source == source.as_str()
    ).collect();
    if rows.is_empty() { return None; }
    let date = &request.cycle.date_yyyymmdd;
    let hour = format!("{:02}", request.cycle.hour_utc);
    let stamp = format!("{}-{}-{}T{}", &date[..4], &date[4..6], &date[6..8], hour);
    let product = normalize_token(&request.product);
    let matches_product = |row: &&Layout| normalize_token(&row.product) == product
        || row.product_aliases.iter().any(|alias| normalize_token(alias) == product);
    // Preserve existing products on nonexclusive hosts; their established
    // builder owns products this table does not override.
    if !rows.iter().any(matches_product) && !rows.iter().any(|row| row.exclusive_source) {
        return None;
    }
    for row in rows {
        if !matches_product(&row)
            || stamp < row.record_start
            || row.cycle_until.as_ref().is_some_and(|end| stamp >= *end) { continue; }
        let lead = request.forecast_hour;
        if !row.lead_segments.is_empty() && !row.lead_segments.iter().enumerate().any(|(i, seg)|
            lead <= seg.through && lead % seg.step == 0
                && (i == 0 || lead > row.lead_segments[i-1].through)) { continue; }
        let mut key = row.path_template.clone();
        for (token, value) in [
            ("YYYY", date[..4].to_string()), ("MM", date[4..6].to_string()),
            ("DD", date[6..8].to_string()), ("HH", hour.clone()),
            ("FFF", format!("{lead:03}")),
        ] {
            key = key.replace(&format!("{{{token}}}"), &value);
        }
        assert!(!key.contains('{'), "unknown archive path token");
        return Some(Ok(format!("{}/{}", row.base, key)));
    }
    // A registered source cannot fall through to a different resolution or era.
    Some(Err(ModelError::UnsupportedProduct {
        model: request.model,
        product: format!("{} at {stamp} f{:03}: no declared archive layout",
                         request.product, request.forecast_hour),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_core::{CycleSpec, ModelId};
    fn request(date: &str, hour: u8, lead: u16, product: &str) -> ModelRunRequest {
        ModelRunRequest::new(ModelId::Gfs, CycleSpec::new(date, hour).unwrap(),
                             lead, product).unwrap()
    }
    #[test]
    fn archive_objects_have_explicit_resolution_and_era() {
        let old = request("20170918", 12, 6, "pgrb2.0p25");
        assert_eq!(object_url(SourceId::Gdex, &old).unwrap().unwrap(),
            "https://data.gdex.ucar.edu/d084001/2017/20170918/gfs.0p25.2017091812.f006.grib2");
        assert!(object_url(SourceId::Aws, &old).unwrap().is_err());
        assert!(object_url(SourceId::Aws,
            &request("20261001", 0, 0, "pgrb2b.0p25")).is_none());
        assert_eq!(object_url(SourceId::Gdex,
            &request("20170918", 12, 6, "0p25")).unwrap().unwrap(),
            object_url(SourceId::Gdex, &old).unwrap().unwrap());
        assert!(object_url(SourceId::Ncei, &old).unwrap().is_err());
        let coarse = request("20170918", 12, 6, "pgrb2.0p50");
        assert!(object_url(SourceId::Ncei, &coarse).unwrap().unwrap()
                .ends_with("/gfs_4_20170918_1200_006.grb2"));
        let early = request("20210101", 0, 0, "pgrb2.0p25");
        assert!(object_url(SourceId::Aws, &early).unwrap().unwrap().contains("/00/gfs.t00z"));
        let late = request("20210322", 12, 0, "pgrb2.0p25");
        assert!(object_url(SourceId::Aws, &late).unwrap().unwrap().contains("/12/atmos/"));
    }
    #[test]
    fn gdas_archive_starts_at_the_atmos_layout() {
        // Door item 11: a GDAS cycle before 2021-03-22 12Z is refused by
        // the table, not sent to an atmos/ key the archive does not hold.
        let gdas = |date: &str, hour: u8, lead: u16, product: &str| ModelRunRequest::new(
            ModelId::Gdas, CycleSpec::new(date, hour).unwrap(), lead, product).unwrap();
        assert_eq!(object_url(SourceId::Aws, &gdas("20210322", 12, 0, "pgrb2.0p25"))
                       .unwrap().unwrap(),
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.20210322/12/atmos/gdas.t12z.pgrb2.0p25.f000");
        assert_eq!(object_url(SourceId::Aws, &gdas("20260502", 0, 3, "pgrb2.0p25"))
                       .unwrap().unwrap(),
            "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.20260502/00/atmos/gdas.t00z.pgrb2.0p25.f003");
        for (date, hour) in [("20210322", 6), ("20210115", 0), ("20190101", 0)] {
            assert!(object_url(SourceId::Aws, &gdas(date, hour, 0, "pgrb2.0p25"))
                        .unwrap().is_err(), "{date} {hour}");
        }
        // NOMADS keeps no dated rows, and the 1.0 degree product keeps its builder.
        assert!(object_url(SourceId::Nomads, &gdas("20210115", 0, 0, "pgrb2.0p25")).is_none());
        assert!(object_url(SourceId::Aws, &gdas("20261001", 0, 0, "pgrb2.1p00")).is_none());
    }
    #[test]
    fn forecast_holes_are_refused() {
        for lead in [1, 241, 243] {
            assert!(object_url(SourceId::Gdex,
                &request("20170918", 12, lead, "pgrb2.0p25")).unwrap().is_err());
        }
        assert!(object_url(SourceId::Gdex,
            &request("20170918", 12, 252, "pgrb2.0p25")).unwrap().is_ok());
    }
}
