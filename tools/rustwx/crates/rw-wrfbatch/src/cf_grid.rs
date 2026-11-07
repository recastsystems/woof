//! CF diagnostic planes. No model identity or WRF state is synthesized.
use chrono::{DateTime, Duration, NaiveDateTime, Utc};
use netcrust::{File, NcSliceInfo, NcSliceInfoElem, Variable};
use rayon::prelude::*;
use rustwx_render::{Color, ColorScale, DiscreteColorScale, ExtendMode};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct ProductRow {
    #[serde(default)]
    pub levels: Vec<f64>,
    pub names: Vec<String>,
    pub standard_names: Vec<String>,
    pub product: String,
    pub title: String,
    pub recipe: Option<String>,
    pub default_units: String,
    pub display_units: String,
    pub unit_factors: BTreeMap<String, f64>,
    pub contour_level: Option<f64>,
    pub observation_product: Option<String>,
    pub scale: Option<CfScale>,
}
/// Optional physical-unit ladder supplied by a CF product row.
#[derive(Clone, Debug, Deserialize)]
pub struct CfScale {
    pub levels: Vec<f64>,
    pub colors: Vec<[u8; 3]>,
    /// Values under this (display units) are left blank instead of clamped into the first colour. Without it a
    /// dry precipitation window (a field of zeros) painted the whole domain in the first band, which reads as
    /// rain (WOOF-NAM qpf_1h dry hours on gpuwm 2.9.11/2.9.12, 2026-11-05: the key was silently dropped).
    /// Absent = no mask, as before.
    #[serde(default)]
    pub mask_below: Option<f64>,
}
impl CfScale {
    fn color_scale(&self) -> Result<ColorScale, String> {
        if self.levels.len() < 2 || self.colors.len() + 1 != self.levels.len()
            || self.levels.iter().any(|v| !v.is_finite())
            || self.levels.windows(2).any(|v| v[0] >= v[1]) {
            return Err("CF scale needs increasing finite levels and one RGB color per interval".into());
        }
        if self.mask_below.is_some_and(|v| !v.is_finite()) {
            return Err("CF scale mask_below must be a finite value in display units".into());
        }
        Ok(ColorScale::Discrete(DiscreteColorScale {
            levels: self.levels.clone(),
            colors: self.colors.iter().map(|[r,g,b]| Color::rgba(*r,*g,*b,255)).collect(),
            extend: ExtendMode::Both,
            mask_below: self.mask_below,
        }))
    }
}

impl ProductRow {
    pub fn factor(&self, units: &str) -> Result<f64, String> {
        self.unit_factors.get(units).copied().ok_or_else(|| {
            format!(
                "{}: units {units:?} have no conversion to {} in the CF product table",
                self.product, self.display_units
            )
        })
    }
    pub fn scale(&self) -> Result<ColorScale, String> {
        if let Some(scale) = &self.scale { return scale.color_scale(); }
        if !self.levels.is_empty() {
            if self.levels.len() != 10 || self.levels.iter().any(|v| !v.is_finite())
                || self.levels.windows(2).any(|p| p[0] >= p[1]) {
                return Err("CF custom levels need ten increasing finite edges".into());
            }
            let colors = [[49,54,149],[69,117,180],[116,173,209],[171,217,233],
                [224,243,248],[254,224,144],[253,174,97],[244,109,67],[165,0,38]];
            return Ok(ColorScale::Discrete(rustwx_render::DiscreteColorScale {
                levels: self.levels.clone(),
                colors: colors.iter().map(|c| rustwx_render::Color::rgba(c[0],c[1],c[2],255)).collect(),
                extend: rustwx_render::ExtendMode::Both, mask_below: None,
            }));
        }
        match self.recipe.as_deref() {
            Some(slug) => {
                let recipe = rustwx_models::plot_recipe(slug)
                    .ok_or_else(|| format!("unknown production recipe {slug}"))?;
                let selector = recipe
                    .filled
                    .selector
                    .ok_or("recipe has no fill selector")?;
                Ok(
                    rustwx_products::plot_design::operational_fill_scale_for_recipe(
                        recipe, selector,
                    ),
                )
            }
            None => Ok(crate::scales::probability_scale()),
        }
    }
}
pub fn product_rows(path: Option<&Path>) -> Result<Vec<ProductRow>, String> {
    let text = match path {
        Some(p) => std::fs::read_to_string(p).map_err(|e| e.to_string())?,
        None => include_str!("../data/cf-products.json").to_string(),
    };
    serde_json::from_str(&text).map_err(|e| format!("CF product table: {e}"))
}
pub fn resolve<'a>(
    rows: &'a [ProductRow],
    name: &str,
    standard: &str,
) -> Result<&'a ProductRow, String> {
    let named: Vec<_> = rows
        .iter()
        .filter(|r| r.names.iter().any(|n| n == name))
        .collect();
    let selected = if named.is_empty() {
        rows.iter()
            .filter(|r| r.standard_names.iter().any(|n| n == standard))
            .collect()
    } else {
        named
    };
    match selected.as_slice() {
        [row] => Ok(row),
        [] => Err(format!(
            "no CF product row for {name:?} / {standard:?}; add a table row"
        )),
        _ => Err(format!(
            "ambiguous CF product rows for {name:?} / {standard:?}"
        )),
    }
}
fn text(var: &Variable, key: &str) -> String {
    var.attribute(key)
        .and_then(|a| a.as_string())
        .unwrap_or("")
        .to_string()
}
pub fn moment(value: &str) -> Result<DateTime<Utc>, String> {
    if let Ok(t) = DateTime::parse_from_rfc3339(value) {
        return Ok(t.with_timezone(&Utc));
    }
    let value = value
        .trim()
        .trim_end_matches(" UTC")
        .trim_end_matches('Z')
        .replace(' ', "T");
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M"] {
        if let Ok(t) = NaiveDateTime::parse_from_str(&value, fmt) {
            return Ok(t.and_utc());
        }
    }
    if let Ok(day) = chrono::NaiveDate::parse_from_str(&value, "%Y-%m-%d") {
        return Ok(day.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }
    Err(format!("invalid UTC time {value:?}"))
}
fn cf_times(file: &File, var: &Variable) -> Result<Vec<DateTime<Utc>>, String> {
    let calendar = text(var, "calendar");
    if !["", "standard", "gregorian", "proleptic_gregorian"].contains(&calendar.as_str()) {
        return Err(format!(
            "calendar {calendar:?} cannot be matched to UTC observations"
        ));
    }
    let units = text(var, "units");
    let (unit, origin) = units
        .split_once(" since ")
        .ok_or("CF time requires '<unit> since <origin>'")?;
    let seconds = match unit {
        "seconds" | "second" | "s" => 1.0,
        "minutes" | "minute" => 60.0,
        "hours" | "hour" | "h" => 3600.0,
        "days" | "day" | "d" => 86400.0,
        "milliseconds" => 0.001,
        "microseconds" => 0.000001,
        _ => return Err(format!("unsupported CF time unit {unit}")),
    };
    let origin = moment(origin)?;
    file.read_f64(var.name())
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|v| {
            // A valid reference may be thousands of years before the data.
            // Keep whole seconds separate so an i64 nanosecond span does not
            // impose an artificial 292-year limit on CF reference epochs.
            let offset = v * seconds;
            if !offset.is_finite() || offset.abs() >= i64::MAX as f64 {
                return Err("CF time offset exceeds UTC range".into());
            }
            let whole = Duration::try_seconds(offset.trunc() as i64)
                .ok_or("CF time offset exceeds UTC range")?;
            let fraction = Duration::nanoseconds((offset.fract() * 1e9).round() as i64);
            let delta = whole.checked_add(&fraction).ok_or("CF time overflow")?;
            origin.checked_add_signed(delta).ok_or("CF time overflow".into())
        })
        .collect()
}
fn coordinate(vars: &[Variable], standard: &str, aliases: &[&str]) -> Result<Variable, String> {
    vars.iter()
        .find(|v| text(v, "standard_name") == standard)
        .or_else(|| vars.iter().find(|v| aliases.contains(&v.name())))
        .cloned()
        .ok_or_else(|| format!("CF file has no {standard} coordinate"))
}

pub struct CfGrid {
    file: File,
    pub nx: usize,
    pub ny: usize,
    pub lat: Vec<f32>,
    pub lon: Vec<f32>,
    pub valid: DateTime<Utc>,
    pub source: String,
    time_dim: String,
    time_index: usize,
    spatial_dims: [String; 2],
    /// Source index after row seam rotation. Applied to every plane.
    order: Vec<usize>,
}
impl CfGrid {
    pub fn open(path: &Path, valid: Option<DateTime<Utc>>) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let vars = file.variables().map_err(|e| e.to_string())?;
        let latvar = coordinate(&vars, "latitude", &["lat", "latitude", "XLAT"])?;
        let lonvar = coordinate(&vars, "longitude", &["lon", "longitude", "XLONG"])?;
        let timevar = coordinate(&vars, "time", &["time", "valid_time"])?;
        if timevar.ndim() != 1 {
            return Err("CF time must have one dimension".into());
        }
        let times = cf_times(&file, &timevar)?;
        let ti = match valid {
            Some(t) => times
                .iter()
                .position(|v| *v == t)
                .ok_or_else(|| format!("{} has no valid time {t}", path.display()))?,
            None => 0,
        };
        let stamp = *times.get(ti).ok_or("empty CF time axis")?;
        let a = file
            .read_array_f64(latvar.name())
            .map_err(|e| e.to_string())?;
        let b = file
            .read_array_f64(lonvar.name())
            .map_err(|e| e.to_string())?;
        let (ny, nx, spatial_dims) = match (a.shape(), b.shape()) {
            ([ny], [nx]) => (
                *ny as usize,
                *nx as usize,
                [
                    latvar.dimensions()[0].name().into(),
                    lonvar.dimensions()[0].name().into(),
                ],
            ),
            ([ny, nx], [by, bx])
                if ny == by
                    && nx == bx
                    && latvar
                        .dimensions()
                        .iter()
                        .map(|d| d.name())
                        .eq(lonvar.dimensions().iter().map(|d| d.name())) =>
            {
                (
                    *ny as usize,
                    *nx as usize,
                    [
                        latvar.dimensions()[0].name().into(),
                        latvar.dimensions()[1].name().into(),
                    ],
                )
            }
            _ => {
                return Err("CF latitude/longitude must be 1-D axes or matching 2-D arrays".into());
            }
        };
        if ny < 2 || nx < 2 {
            return Err("a map needs at least two latitude rows and longitude columns".into());
        }
        let axes = a.ndim() == 1;
        let mut lat: Vec<f32> = (0..ny * nx)
            .into_par_iter()
            .map(|i| a.values()[if axes { i / nx } else { i }] as f32)
            .collect();
        let mut lon: Vec<f32> = (0..ny * nx)
            .into_par_iter()
            .map(|i| {
                ((b.values()[if axes { i % nx } else { i }] + 180.).rem_euclid(360.) - 180.) as f32
            })
            .collect();
        if lat.par_iter().any(|x| !x.is_finite() || x.abs() > 90.)
            || lon.par_iter().any(|x| !x.is_finite())
        {
            return Err("nonfinite or invalid CF grid coordinates".into());
        }
        let mut order: Vec<usize> = (0..ny * nx).into_par_iter().collect();
        lat.par_chunks_mut(nx)
            .zip(lon.par_chunks_mut(nx))
            .zip(order.par_chunks_mut(nx))
            .for_each(|((a, b), idx)| {
                // A westward axis must be reversed together with its values
                // before locating the cyclic seam.
                if b.windows(2)
                    .find(|p| (p[1] - p[0]).abs() < 180. && p[1] != p[0])
                    .is_some_and(|p| p[1] < p[0])
                {
                    a.reverse();
                    b.reverse();
                    idx.reverse();
                }
                if let Some(k) = b.windows(2).position(|p| p[1] - p[0] < -180.) {
                    a.rotate_left(k + 1);
                    b.rotate_left(k + 1);
                    idx.rotate_left(k + 1);
                }
            });
        let source = file
            .attribute("source")
            .and_then(|a| a.as_string().map(str::to_string))
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| {
                format!(
                    "CF diagnostic {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                )
            });
        Ok(Self {
            file,
            nx,
            ny,
            lat,
            lon,
            valid: stamp,
            source,
            time_dim: timevar.dimensions()[0].name().into(),
            time_index: ti,
            spatial_dims,
            order,
        })
    }
    /// Keep coordinates and field indices together across a requested map seam.
    pub fn recenter(&mut self, center: f64) {
        self.lat.par_chunks_mut(self.nx)
            .zip(self.lon.par_chunks_mut(self.nx))
            .zip(self.order.par_chunks_mut(self.nx))
            .for_each(|((lat, lon), order)| {
                for x in lon.iter_mut() {
                    *x = ((*x as f64 - center + 180.).rem_euclid(360.) - 180. + center) as f32;
                }
                if let Some(k) = lon.windows(2).position(|p| p[1] - p[0] < -180.) {
                    lat.rotate_left(k + 1);
                    lon.rotate_left(k + 1);
                    order.rotate_left(k + 1);
                }
            });
    }
    pub fn field(&self, name: &str, rows: &[ProductRow]) -> Result<(Vec<f32>, ProductRow), String> {
        let var = self
            .file
            .variable(name)
            .ok_or_else(|| format!("no CF variable {name}"))?;
        let row = resolve(rows, name, &text(&var, "standard_name"))?.clone();
        let units = text(&var, "units");
        let factor = row.factor(if units.is_empty() {
            &row.default_units
        } else {
            &units
        })?;
        let dims: Vec<_> = var.dimensions().iter().map(|d| d.name()).collect();
        let spatial: Vec<_> = self.spatial_dims.iter().map(String::as_str).collect();
        let spatial_start = dims.len().checked_sub(spatial.len())
            .ok_or_else(|| format!("{name}: dimensions {dims:?} lack the CF spatial grid"))?;
        if dims[spatial_start..] != spatial {
            return Err(format!(
                "{name}: dimensions {dims:?} do not end in the CF spatial grid {spatial:?}"
            ));
        }
        let mut selections = Vec::with_capacity(dims.len());
        for (i, dimension) in var.dimensions().iter().enumerate() {
            let selection = if i >= spatial_start {
                NcSliceInfoElem::Slice { start: 0, end: u64::MAX, step: 1 }
            } else if dimension.name() == self.time_dim {
                NcSliceInfoElem::Index(self.time_index as u64)
            } else if dimension.len() == 1 {
                // Extra producer axes are unambiguous only when singleton.
                // Do not silently choose a member, level or forecast lead.
                NcSliceInfoElem::Index(0)
            } else {
                return Err(format!(
                    "{name}: nonspatial dimension {:?} has length {}; select one plane explicitly",
                    dimension.name(), dimension.len()
                ));
            };
            selections.push(selection);
        }
        let selection = NcSliceInfo { selections };
        let data = self
            .file
            .read_array_f64_slice(name, &selection)
            .map_err(|e| e.to_string())?;
        if data.len() != self.nx * self.ny {
            return Err(format!("{name}: coordinate and value shapes differ"));
        }
        // Promoted reads preserve stored words. Mask before CF unpacking.
        let scale = var
            .attribute("scale_factor")
            .and_then(|a| a.as_f64())
            .unwrap_or(1.);
        let offset = var
            .attribute("add_offset")
            .and_then(|a| a.as_f64())
            .unwrap_or(0.);
        let missing: Vec<_> = ["_FillValue", "missing_value"]
            .iter()
            .filter_map(|k| var.attribute(k))
            .flat_map(|a| a.value().as_f64_vec().unwrap_or_default())
            .collect();
        let values = self
            .order
            .par_iter()
            .map(|&i| {
                let v = data.values()[i];
                if !v.is_finite() || missing.contains(&v) {
                    f32::NAN
                } else {
                    ((v * scale + offset) * factor) as f32
                }
            })
            .collect();
        Ok((values, row))
    }
}

/// A rectangle of source indices. The panel frame itself uses the exact requested bounds.
pub fn window(
    lat: &[f32],
    lon: &[f32],
    nx: usize,
    bounds: Option<(f64, f64, f64, f64)>,
) -> Result<Vec<usize>, String> {
    let Some((w, e, s, n)) = bounds else {
        return Ok((0..lat.len()).into_par_iter().collect());
    };
    let (mut j0, mut j1, mut i0, mut i1) = (0..lat.len())
        .into_par_iter()
        .filter(|&i| {
            let x = lon[i] as f64;
            x >= w && x <= e && lat[i] as f64 >= s && lat[i] as f64 <= n
        })
        .map(|k| (k / nx, k / nx, k % nx, k % nx))
        .reduce(
            || (usize::MAX, 0, usize::MAX, 0),
            |(a, b, c, d), (e, f, g, h)| (a.min(e), b.max(f), c.min(g), d.max(h)),
        );
    if j0 == usize::MAX {
        return Err("bounds do not intersect the diagnostic grid".into());
    }
    j0 = j0.saturating_sub(1);
    j1 = (j1 + 1).min(lat.len() / nx - 1);
    i0 = i0.saturating_sub(1);
    i1 = (i1 + 1).min(nx - 1);
    let width = i1 - i0 + 1;
    Ok((0..(j1 - j0 + 1) * width)
        .into_par_iter()
        .map(|k| (j0 + k / width) * nx + i0 + k % width)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn custom_fire_levels_are_valid_and_cover_physical_values() {
        let rows = product_rows(None).unwrap();
        for name in ["fwi_ffmc", "fwi_dmc", "fwi_dc", "fwi_isi", "fwi_bui", "fwi_fwi", "kbdi", "nfdrs_mc1", "nfdrs_mc10", "nfdrs_mc100", "nfdrs_mc1000", "nfdrs_live_herb", "nfdrs_live_wood", "nfdrs_gsi", "nfdrs_erc", "nfdrs_bi", "nfdrs_sc"] {
            let mut row = resolve(&rows, name, "").unwrap().clone();
            assert_eq!(row.factor(&row.default_units).unwrap(), 1.);
            assert!(row.scale().is_ok());
            assert!(row.levels.last().unwrap() > &0.);
            row.levels[1] = row.levels[0];
            assert!(row.scale().is_err());
            row.levels.clear();
            row.levels.push(f64::NAN);
            assert!(row.scale().is_err());
        }
    }
    #[test]
    fn cf_reader_rotates_north_to_south_classic_grid() {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        let path = std::env::temp_dir().join(format!("cf-reader-{}.nc", std::process::id()));
        let mut s = Schema::new(NcFormat::Offset64);
        let t = s.def_dim("time", 1, false).unwrap();
        let y = s.def_dim("lat", 2, false).unwrap();
        let x = s.def_dim("lon", 4, false).unwrap();
        let tv = s.def_var("time", NcType::Double, &[t]).unwrap();
        let la = s.def_var("lat", NcType::Float, &[y]).unwrap();
        let lo = s.def_var("lon", NcType::Float, &[x]).unwrap();
        let v = s.def_var("fg10m", NcType::Float, &[t, y, x]).unwrap();
        s.put_var_attr(
            tv,
            "units",
            AttrValue::Text("hours since 0001-01-01 00:00:00.0".into()),
        )
        .unwrap();
        s.put_var_attr(
            tv,
            "calendar",
            AttrValue::Text("proleptic_gregorian".into()),
        ).unwrap();
        s.put_global_attr("source", AttrValue::Text("diagnostic producer".into()))
            .unwrap();
        let mut wr = NcWriter::create(&path, s).unwrap();
        wr.write_var(tv, VarData::F64(&[17_739_672.])).unwrap();
        wr.write_var(la, VarData::F32(&[90., -90.])).unwrap();
        wr.write_var(lo, VarData::F32(&[0., 90., 180., 270.]))
            .unwrap();
        wr.write_var(v, VarData::F32(&[0., 1., 2., 3., 4., 5., 6., 7.]))
            .unwrap();
        wr.finish().unwrap();
        let mut grid = CfGrid::open(&path, Some(moment("2024-09-25T00:00:00Z").unwrap())).unwrap();
        assert_eq!(&grid.lon[..4], &[-180., -90., 0., 90.]);
        assert_eq!(&grid.lat[..4], &[90.; 4]);
        assert_eq!(grid.source, "diagnostic producer");
        let (values, _) = grid.field("fg10m", &product_rows(None).unwrap()).unwrap();
        assert!((values[0] - 2. * 1.9438445).abs() < 1e-5);
        assert!((values[4] - 6. * 1.9438445).abs() < 1e-5);
        grid.recenter(180.);
        assert_eq!(&grid.lon[..4], &[0., 90., 180., 270.]);
        let (values, _) = grid.field("fg10m", &product_rows(None).unwrap()).unwrap();
        for (i, value) in values.iter().enumerate() {
            assert!((*value - i as f32 * 1.9438445).abs() < 1e-5);
        }
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn cf_reader_selects_time_and_only_collapses_singleton_producer_axes() {
        use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};
        for extra_len in [1, 2] {
            let path = std::env::temp_dir().join(format!(
                "cf-singleton-{}-{extra_len}.nc", std::process::id()
            ));
            let mut schema = Schema::new(NcFormat::Offset64);
            let t = schema.def_dim("time", 2, false).unwrap();
            let extra = schema.def_dim("arbitrary_producer_axis", extra_len, false).unwrap();
            let y = schema.def_dim("lat", 2, false).unwrap();
            let x = schema.def_dim("lon", 4, false).unwrap();
            let tv = schema.def_var("time", NcType::Double, &[t]).unwrap();
            let la = schema.def_var("lat", NcType::Float, &[y]).unwrap();
            let lo = schema.def_var("lon", NcType::Float, &[x]).unwrap();
            let v = schema.def_var("ws10m", NcType::Float, &[t, extra, y, x]).unwrap();
            schema.put_var_attr(tv, "units", AttrValue::Text(
                "hours since 0001-01-01 00:00:00.0".into())).unwrap();
            schema.put_var_attr(tv, "calendar", AttrValue::Text(
                "proleptic_gregorian".into())).unwrap();
            schema.put_var_attr(v, "units", AttrValue::Text("m s-1".into())).unwrap();
            let mut writer = NcWriter::create(&path, schema).unwrap();
            writer.write_var(tv, VarData::F64(&[17_739_660., 17_739_672.])).unwrap();
            writer.write_var(la, VarData::F32(&[90., -90.])).unwrap();
            writer.write_var(lo, VarData::F32(&[0., 90., 180., 270.])).unwrap();
            let values: Vec<f32> = (0..2 * extra_len * 8).map(|i| i as f32).collect();
            writer.write_var(v, VarData::F32(&values)).unwrap();
            writer.finish().unwrap();
            let grid = CfGrid::open(&path, Some(moment("2024-09-25T00:00:00Z").unwrap())).unwrap();
            let result = grid.field("ws10m", &product_rows(None).unwrap());
            if extra_len == 1 {
                let (values, _) = result.unwrap();
                assert!((values[0] - 10. * 1.9438445).abs() < 1e-5);
                assert!((values[4] - 14. * 1.9438445).abs() < 1e-5);
            } else {
                assert!(result.unwrap_err().contains("nonspatial dimension"));
            }
            std::fs::remove_file(path).unwrap();
        }
    }
    #[test]
    fn cf_table_resolves_names_and_arbitrary_standard_name() {
        let rows = product_rows(None).unwrap();
        let gust = resolve(&rows, "fg10m", "").unwrap();
        assert_eq!(gust.display_units, "kt");
        assert!(gust.scale().is_ok());
        let p = resolve(&rows, "another_producer_rain", "precipitation_amount").unwrap();
        assert_eq!(p.product, "qpf_6h");
        assert!((p.factor("m").unwrap() - 39.3700787).abs() < 1e-7);
        assert!(p.scale().is_ok());
        assert_eq!(
            resolve(&rows, "anything", "atmosphere_mass_content_of_water_vapor")
                .unwrap()
                .product,
            "precipitable_water"
        );
        assert_eq!(
            resolve(&rows, "climnet_ar", "").unwrap().contour_level,
            Some(0.5)
        );
    }
}

#[cfg(test)]
mod cf_physical_scale_tests {
    use super::*;
    #[test]
    fn signed_physical_ladder_validates_before_render() {
        let scale = CfScale { levels: vec![-1., 0., 1.], colors: vec![[0,0,255],[255,0,0]], mask_below: None };
        assert!(scale.color_scale().is_ok());
        for levels in [vec![0., 0., 1.], vec![0., f64::NAN, 1.], vec![0., 1.]] {
            assert!(CfScale { levels, colors: scale.colors.clone(), mask_below: None }.color_scale().is_err());
        }
        let bad = CfScale { mask_below: Some(f64::NAN), ..scale.clone() };
        assert!(bad.color_scale().is_err());
    }
    #[test]
    fn a_row_scale_mask_below_reaches_the_colour_scale() {
        // 2.9.11/2.9.12 dropped the key, so a field of zeros painted the whole domain in the first colour.
        let row: CfScale = serde_json::from_str(
            r#"{"levels": [0.01, 0.1, 1.0], "colors": [[150,230,150],[30,150,30]], "mask_below": 0.01}"#).unwrap();
        match row.color_scale().unwrap() {
            ColorScale::Discrete(scale) => assert_eq!(scale.mask_below, Some(0.01)),
            _ => panic!("a CF row scale is discrete"),
        }
        let plain: CfScale = serde_json::from_str(r#"{"levels": [0.0, 1.0], "colors": [[0,0,0]]}"#).unwrap();
        match plain.color_scale().unwrap() {
            ColorScale::Discrete(scale) => assert_eq!(scale.mask_below, None),
            _ => panic!("a CF row scale is discrete"),
        }
    }
}
