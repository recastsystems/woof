//! WRF native-field binding for named derived heat, fire and lightning products.
use super::{
    OwnedDerivedField, WrfHourFields, WrfProcessOptions, WrfProductGroup, compute_var,
    push_canonical_values,
};
use rayon::prelude::*;
use rustwx_calc::derived_heat::{self, ThermoColumns};
use rustwx_calc::lightning_threat;
use rustwx_calc::{SurfaceInputs, compute_surface_thermo};
use rustwx_core::{CanonicalField, FieldSelector, GridProjection, LatLonGrid};
use wrf_core::WrfFile;

pub(super) const PRODUCTS: &[(&str, CanonicalField)] = &[
    ("wbgt_2m", CanonicalField::Wbgt),
    ("hot_dry_windy", CanonicalField::HotDryWindy),
    ("haines_low", CanonicalField::HainesLow),
    ("haines_mid", CanonicalField::HainesMid),
    ("haines_high", CanonicalField::HainesHigh),
    ("lightning_threat_1", CanonicalField::LightningThreat1),
    ("lightning_threat_2", CanonicalField::LightningThreat2),
    ("lightning_threat_3", CanonicalField::LightningThreat3),
    ("pr92_flash_rate", CanonicalField::Pr92FlashRate),
];
pub(super) const SURFACE: &[&str] = &[
    "heat_index_2m",
    "apparent_temperature_2m",
    "wetbulb_2m",
    "fire_weather_composite",
    "vpd_2m",
];
fn selected(o: &WrfProcessOptions, name: &str) -> bool {
    let key = PRODUCTS
        .iter()
        .find(|r| r.0 == name)
        .map(|r| FieldSelector::surface(r.1).key())
        .unwrap_or_else(|| name.to_string());
    // The viewer's filter tokens are exact store names. These planes'
    // names contain other planes' names (apparent_temperature_2m contains
    // temperature_2m), so the CLI's substring token grammar would compute
    // and store them for a viewer that asked only for the plane inside.
    if o.viewer_2d && !o.only.is_empty() {
        let exact = [super::slug(&key), super::slug(name)];
        if !o.only.iter().any(|token| exact.contains(token)) {
            return false;
        }
    }
    o.should_process(&key, Some(name), WrfProductGroup::Diagnostic)
}
pub(super) fn planned_fields(o: &WrfProcessOptions) -> Vec<String> {
    PRODUCTS
        .iter()
        .map(|r| r.0)
        .chain(SURFACE.iter().copied())
        .filter(|s| selected(o, s))
        .map(str::to_owned)
        .collect()
}
pub(super) fn planned_selectors(o: &WrfProcessOptions) -> Vec<FieldSelector> {
    PRODUCTS
        .iter()
        .filter(|r| selected(o, r.0))
        .map(|r| FieldSelector::surface(r.1))
        .collect()
}
fn read(
    f: &WrfFile,
    ti: usize,
    name: &str,
    shape: &[usize],
    units: Option<&str>,
) -> Result<Vec<f64>, String> {
    let v = compute_var(f, name, ti, units)?;
    if v.shape != shape || v.data.len() != shape.iter().product::<usize>() {
        return Err(format!("{name}: expected {shape:?}, got {:?}", v.shape));
    }
    // WRF fill values can be finite. Never turn a fill value into an extreme
    // heat or flash prediction.
    Ok(v.data
        .into_par_iter()
        .map(|x| {
            if x.is_finite() && x.abs() < 1e30 {
                x
            } else {
                f64::NAN
            }
        })
        .collect())
}
fn push(
    out: &mut WrfHourFields,
    g: &LatLonGrid,
    proj: Option<GridProjection>,
    o: &WrfProcessOptions,
    name: &str,
    units: &str,
    v: Vec<f64>,
) {
    if !selected(o, name) {
        return;
    }
    let v = v.into_par_iter().map(|x| x as f32).collect();
    if let Some((_, field)) = PRODUCTS.iter().find(|r| r.0 == name) {
        push_canonical_values(out, g, proj, name, FieldSelector::surface(*field), units, v);
    } else {
        out.derived.push(OwnedDerivedField {
            name: name.into(),
            units: units.into(),
            values: v,
        });
    }
}
pub(super) fn compute(
    out: &mut WrfHourFields,
    f: &WrfFile,
    ti: usize,
    g: &LatLonGrid,
    proj: Option<GridProjection>,
    o: &WrfProcessOptions,
) {
    if planned_fields(o).is_empty() {
        return;
    }
    // The groups below read the file and nothing another group wrote, so they
    // run at once, each into its own part, and the parts join in the order the
    // groups ran one after another: the hour holds the same fields in the same
    // order. They run before the isobaric volumes release wrf-core's cache, so
    // they read the 3-D fields the import's groups left memoized; run after the
    // release, each rebuilt its fields (7.5 s of a 1400x981x59 hex window
    // frame's import, 2026-11-15; one after another it had been 1.4 s of a
    // 2.25 km / 750 m HRRR nest frame's 5.3 s render at one core, 2026-10-24).
    let part = || WrfHourFields {
        grid: out.grid.clone(),
        projection: out.projection.clone(),
        canonical: Vec::new(),
        derived: Vec::new(),
        volumes: Vec::new(),
        notes: Vec::new(),
    };
    let mut parts: [WrfHourFields; 4] = std::array::from_fn(|_| part());
    let columns = PRODUCTS[1..].iter().any(|r| selected(o, r.0));
    {
        let [surface, wbgt, column, pr92] = &mut parts;
        let proj = &proj;
        rayon::scope(|scope| {
            scope.spawn(move |_| surface_group(surface, f, ti, g, proj.clone(), o));
            scope.spawn(move |_| wbgt_group(wbgt, f, ti, g, proj.clone(), o));
            if columns {
                scope.spawn(move |_| column_group(column, f, ti, g, proj.clone(), o));
                scope.spawn(move |_| pr92_group(pr92, f, ti, g, proj.clone(), o));
            }
        });
    }
    for part in parts {
        out.canonical.extend(part.canonical);
        out.derived.extend(part.derived);
        out.notes.extend(part.notes);
    }
}

/// Heat index, apparent temperature, wet bulb, fire-weather composite and VPD.
fn surface_group(
    out: &mut WrfHourFields,
    f: &WrfFile,
    ti: usize,
    g: &LatLonGrid,
    proj: Option<GridProjection>,
    o: &WrfProcessOptions,
) {
    let plane = [f.ny, f.nx];
    // This separate surface group does not require microphysics or radiation.
    if SURFACE.iter().any(|name| *name != "vpd_2m" && selected(o, name)) {
        let surface = (|| -> Result<Vec<(&str, &str, Vec<f64>)>, String> {
            let ps = read(f, ti, "PSFC", &plane, None)?;
            let t = read(f, ti, "T2", &plane, None)?;
            let q = read(f, ti, "Q2", &plane, None)?;
            let u = read(f, ti, "U10", &plane, None)?;
            let v = read(f, ti, "V10", &plane, None)?;
            let s = compute_surface_thermo(
                g.shape,
                SurfaceInputs {
                    psfc_pa: &ps,
                    t2_k: &t,
                    q2_kgkg: &q,
                    u10_ms: &u,
                    v10_ms: &v,
                },
            )
            .map_err(|e| e.to_string())?;
            Ok(vec![
                ("heat_index_2m", "degC", s.heat_index_2m_c),
                (
                    "apparent_temperature_2m",
                    "degC",
                    s.apparent_temperature_2m_c,
                ),
                ("wetbulb_2m", "degC", s.wetbulb_2m_c),
                ("fire_weather_composite", "1", s.fire_weather_composite),
            ])
        })();
        if let Err(e) = &surface {
            out.notes
                .push(format!("derived diagnostics surface unavailable: {e}"));
        }
        if let Ok(rows) = surface {
            for (name, units, v) in rows {
                push(out, g, proj.clone(), o, name, units, v);
            }
        }
    }
    // Day-window VPD reads this exact store name in hPa. Both the full
    // default import and the scoped end-of-run window import must publish
    // it. Only T2/Q2/PSFC are needed; absent winds cannot block VPD.
    if selected(o, "vpd_2m") {
        let result = (|| -> Result<Vec<f64>, String> {
            let ps = read(f, ti, "PSFC", &plane, None)?;
            let t = read(f, ti, "T2", &plane, None)?;
            let q = read(f, ti, "Q2", &plane, None)?;
            let pressure_hpa: Vec<_> = ps.into_iter().map(|p| p / 100.0).collect();
            let temperature_c: Vec<_> = t.into_iter().map(|t| t - 273.15).collect();
            rustwx_calc::compute_vpd_from_pressure_temperature_and_mixing_ratio(
                &pressure_hpa, &temperature_c, &q,
            ).map_err(|e| e.to_string())
        })();
        match result {
            Ok(values) => push(out, g, proj.clone(), o, "vpd_2m", "hPa", values),
            Err(e) => out.notes.push(format!("vpd_2m unavailable: {e}")),
        }
    }
}

/// Wet-bulb globe temperature.
fn wbgt_group(
    out: &mut WrfHourFields,
    f: &WrfFile,
    ti: usize,
    g: &LatLonGrid,
    proj: Option<GridProjection>,
    o: &WrfProcessOptions,
) {
    let cells = f.nx * f.ny;
    let plane = [f.ny, f.nx];
    let volume = [f.nz, f.ny, f.nx];
    if selected(o, "wbgt_2m") {
        let result = (|| -> Result<Vec<f64>, String> {
            let ps = read(f, ti, "PSFC", &plane, None)?;
            let t = read(f, ti, "T2", &plane, None)?;
            let q = read(f, ti, "Q2", &plane, None)?;
            let u = read(f, ti, "U10", &plane, None)?;
            let v = read(f, ti, "V10", &plane, None)?;
            let sw = read(f, ti, "SWDOWN", &plane, None)?;
            let times = f.times().map_err(|e| e.to_string())?;
            let stamp = times.get(ti).ok_or("missing WRF valid time")?;
            let time = chrono::NaiveDateTime::parse_from_str(stamp, "%Y-%m-%d_%H:%M:%S")
                .map_err(|e| format!("WRF valid time {stamp}: {e}"))?
                .and_utc()
                .timestamp();
            // SRDT stability uses the native 2-10 m temperature difference.
            // Linear interpolation in AGL height, with T2 as the 2 m endpoint.
            let (temp, height) = rayon::join(
                || read(f, ti, "temp", &volume, Some("K")),
                || read(f, ti, "height_agl", &volume, Some("m")),
            );
            let (temp, height) = (temp?, height?);
            Ok((0..cells)
                .into_par_iter()
                .map(|i| {
                    let mut previous = (2.0, t[i]);
                    let mut t10 = f64::NAN;
                    for k in 0..f.nz {
                        let j = k * cells + i;
                        let z = height[j];
                        if !z.is_finite() || !temp[j].is_finite() || z < previous.0 {
                            break;
                        }
                        if z >= 10.0 {
                            t10 = previous.1
                                + (temp[j] - previous.1) * (10.0 - previous.0) / (z - previous.0);
                            break;
                        }
                        previous = (z, temp[j]);
                    }
                    if !t10.is_finite() {
                        return f64::NAN;
                    }
                    let (cza, dist) = derived_heat::solar_geometry(
                        time as f64,
                        g.lat_deg[i] as f64,
                        g.lon_deg[i] as f64,
                    );
                    let wind =
                        derived_heat::sensor_wind_2m(u[i].hypot(v[i]), sw[i], cza, t10 - t[i]);
                    let rh =
                        derived_heat::vapor_hpa(ps[i], q[i]) / derived_heat::saturation_hpa(t[i]);
                    if !rh.is_finite() || rh < 0.0 {
                        return f64::NAN;
                    }
                    derived_heat::wbgt_liljegren(t[i], rh.min(1.0), ps[i], wind, sw[i], cza, dist)
                })
                .collect())
        })();
        match result {
            Ok(v) => push(out, g, proj.clone(), o, "wbgt_2m", "degC", v),
            Err(e) => out.notes.push(format!("wbgt_2m unavailable: {e}")),
        }
    }
}

/// Haines, hot-dry-windy and the McCaul lightning threats, from one set of columns.
fn column_group(
    out: &mut WrfHourFields,
    f: &WrfFile,
    ti: usize,
    g: &LatLonGrid,
    proj: Option<GridProjection>,
    o: &WrfProcessOptions,
) {
    let cells = f.nx * f.ny;
    let plane = [f.ny, f.nx];
    let volume = [f.nz, f.ny, f.nx];
    let result = (|| -> Result<(), String> {
        let staggered = [f.nz + 1, f.ny, f.nx];
        let (((t, p), (q, h)), ((ph, phb), terrain)) = rayon::join(
            || {
                rayon::join(
                    || rayon::join(|| read(f, ti, "temp", &volume, Some("K")), || read(f, ti, "pres", &volume, Some("Pa"))),
                    || rayon::join(|| read(f, ti, "QVAPOR", &volume, None), || read(f, ti, "height_agl", &volume, Some("m"))),
                )
            },
            || {
                rayon::join(
                    || rayon::join(|| read(f, ti, "PH", &staggered, None), || read(f, ti, "PHB", &staggered, None)),
                    || read(f, ti, "HGT", &plane, None),
                )
            },
        );
        let (t, p, q, h, ph, phb, terrain) = (t?, p?, q?, h?, ph?, phb?, terrain?);
        let interfaces: Vec<f64> = ph
            .par_iter()
            .zip(&phb)
            .enumerate()
            .map(|(j, (a, b))| (a + b) / 9.80665 - terrain[j % cells])
            .collect();
        let c = ThermoColumns {
            cells,
            nz: f.nz,
            pressure_pa: &p,
            temperature_k: &t,
            vapor_kgkg: &q,
            height_agl_m: &h,
            interface_agl_m: &interfaces,
        };
        c.validate()?;
        for (variant, name) in ["haines_low", "haines_mid", "haines_high"]
            .iter()
            .enumerate()
        {
            if selected(o, name) {
                push(
                    out,
                    g,
                    proj.clone(),
                    o,
                    name,
                    "1",
                    derived_heat::haines(c, variant)?,
                );
            }
        }
        if selected(o, "hot_dry_windy") {
            let hdw = (|| {
                let ps = read(f, ti, "PSFC", &plane, None)?;
                let ts = read(f, ti, "T2", &plane, None)?;
                let qs = read(f, ti, "Q2", &plane, None)?;
                let us = read(f, ti, "U10", &plane, None)?;
                let vs = read(f, ti, "V10", &plane, None)?;
                let (u, v) = rayon::join(
                    || read(f, ti, "ua", &volume, Some("m s-1")),
                    || read(f, ti, "va", &volume, Some("m s-1")),
                );
                let (u, v) = (u?, v?);
                derived_heat::hot_dry_windy(c, &ps, &ts, &qs, &us, &vs, &u, &v)
            })();
            match hdw {
                Ok(v) => push(out, g, proj.clone(), o, "hot_dry_windy", "hPa m s^-1", v),
                Err(e) => out.notes.push(format!("hot_dry_windy unavailable: {e}")),
            }
        }
        if [
            "lightning_threat_1",
            "lightning_threat_2",
            "lightning_threat_3",
        ]
        .iter()
        .any(|n| selected(o, n))
        {
            let ltg = (|| {
                let ((w, qg), (qs, qi)) = rayon::join(
                    || rayon::join(|| read(f, ti, "wa", &volume, Some("m s-1")), || read(f, ti, "QGRAUP", &volume, None)),
                    || rayon::join(|| read(f, ti, "QSNOW", &volume, None), || read(f, ti, "QICE", &volume, None)),
                );
                let (w, qg, qs, qi) = (w?, qg?, qs?, qi?);
                lightning_threat::mc_caul(c, &w, &qg, &qs, &qi)
            })();
            match ltg {
                Ok(v) => {
                    for (name, data) in [
                        ("lightning_threat_1", v.ltg1),
                        ("lightning_threat_2", v.ltg2),
                        ("lightning_threat_3", v.ltg3),
                    ] {
                        push(
                            out,
                            g,
                            proj.clone(),
                            o,
                            name,
                            "flashes km^-2 (5 min)^-1",
                            data,
                        );
                    }
                }
                Err(e) => out.notes.push(format!("LTG1/2/3 unavailable: {e}")),
            }
        }
        Ok(())
    })();
    if let Err(e) = result {
        out.notes.push(format!(
            "native derived heat/fire/lightning unavailable: {e}"
        ));
    }
}

/// The Price and Rind (1992) flash rate.
fn pr92_group(
    out: &mut WrfHourFields,
    f: &WrfFile,
    ti: usize,
    g: &LatLonGrid,
    proj: Option<GridProjection>,
    o: &WrfProcessOptions,
) {
    let cells = f.nx * f.ny;
    let volume = [f.nz, f.ny, f.nx];
    if selected(o, "pr92_flash_rate") {
        let rate = (|| {
            let (w, r) = rayon::join(
                || read(f, ti, "wa", &volume, Some("m s-1")),
                || read(f, ti, "dbz", &volume, Some("dBZ")),
            );
            let (w, r) = (w?, r?);
            lightning_threat::pr92(&w, &r, f.nz, cells, 20.0)
        })();
        match rate {
            Ok(v) => push(
                out,
                g,
                proj,
                o,
                "pr92_flash_rate",
                "flashes s^-1 cell^-1",
                v,
            ),
            Err(e) => out.notes.push(format!("pr92_flash_rate unavailable: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn named_import_uses_the_same_store_names_as_each_writer() {
        for (name, _) in PRODUCTS {
            let options = WrfProcessOptions {
                required_fields: super::super::required_fields_for_products(name),
                ..WrfProcessOptions::default()
            };
            assert_eq!(planned_fields(&options), vec![name.to_string()], "{name}");
        }
    }
    #[test]
    fn default_import_plans_all_named_diagnostics() {
        let options = WrfProcessOptions::default();
        let names = planned_fields(&options);
        for (name, _) in PRODUCTS {
            assert!(names.iter().any(|n| n == name), "{name}");
        }
        for name in SURFACE {
            assert!(names.iter().any(|n| n == name), "{name}");
        }
        assert_eq!(planned_selectors(&options).len(), PRODUCTS.len());
        // Surface indices retain their recipe-slot names, not primitive aliases.
        assert!(names.iter().any(|n| n == "heat_index_2m"));
        assert!(names.iter().any(|n| n == "apparent_temperature_2m"));
    }
}
