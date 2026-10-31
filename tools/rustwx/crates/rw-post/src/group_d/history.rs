//! Read the group D carriers of a WRF-format WOOF history (spec 3.1), first
//! time record: `REFL_10CM`, `CLDFRA`, the 550 nm aerosol extinction of the
//! lowest level, `HGT` and `OLR`.  Absent carriers stay `None`; they are never
//! replaced by zeros.

use std::path::Path;

use super::Carriers;

/// Owned group D carriers of one frame.
#[derive(Clone, Debug, Default)]
pub struct OwnedCarriers {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub refl: Option<Vec<f32>>,
    pub cldfra: Option<Vec<f32>>,
    pub aerosol_km: Option<Vec<f32>>,
    /// Which carriers the aerosol extinction came from, or "absent".
    pub aerosol_source: String,
    pub hgt: Option<Vec<f32>>,
    /// `OLR`, W m-2.
    pub olr: Option<Vec<f32>>,
}

/// km-1 per unit of the carrier, or None for units this reader cannot
/// trust.  A wrong unit would scale smoke extinction, and so visibility in
/// smoke, by a factor of 1000, so unknown units are refused.
fn per_km(units: &str) -> Option<f64> {
    match units.replace(' ', "").to_ascii_lowercase().as_str() {
        "km-1" | "km^-1" | "1/km" | "km**-1" => Some(1.0),
        "m-1" | "m^-1" | "1/m" | "m**-1" => Some(1000.0),
        _ => None,
    }
}

fn units_of(f: &netcrust::File, name: &str) -> String {
    f.variable(name)
        .and_then(|v| v.attribute("units").and_then(|a| a.as_string().map(str::to_owned)))
        .unwrap_or_default()
}

impl OwnedCarriers {
    /// Read a history file.
    pub fn read(path: &Path) -> Result<Self, String> {
        let f = netcrust::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let dim = |name: &str| f.dimension(name).map(|d| d.len()).ok_or_else(|| format!("dimension {name} absent"));
        let nx = dim("west_east")?;
        let ny = dim("south_north")?;
        let nz = dim("bottom_top")?;
        let n = nx * ny;
        let read = |name: &str| -> Result<Option<Vec<f64>>, String> {
            if f.variable(name).is_none() {
                return Ok(None);
            }
            let fill = f.variable(name).and_then(|v| v.attribute("_FillValue").and_then(|a| a.as_f64()));
            f.read_f64_first_record_or_all(name)
                .map(|v| Some(match fill { Some(fv) => v.into_iter().map(|x| if x == fv { f64::NAN } else { x }).collect(), None => v }))
                .map_err(|e| format!("read {name}: {e}"))
        };
        let as_f32 = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
        let volume = |name: &str, v: Option<Vec<f64>>| -> Result<Option<Vec<f64>>, String> {
            match v {
                Some(v) if v.len() != nz * n => Err(format!("{name}: {} words, expected a mass volume of {}", v.len(), nz * n)),
                other => Ok(other),
            }
        };

        let refl = volume("REFL_10CM", read("REFL_10CM")?)?.map(as_f32);
        // CLDFRA is a fraction in WRF histories; a percent carrier is scaled.
        let cldfra = volume("CLDFRA", read("CLDFRA")?)?.map(|v| {
            let u = units_of(&f, "CLDFRA").trim().to_ascii_lowercase();
            let scale = if u == "%" || u == "percent" { 0.01 } else { 1.0 };
            v.into_iter().map(|x| (x * scale) as f32).collect::<Vec<f32>>()
        });
        let hgt = read("HGT")?.map(as_f32);
        let olr = match read("OLR")? {
            Some(v) if v.len() != n => return Err(format!("OLR: {} words, expected a plane of {n}", v.len())),
            other => other.map(as_f32),
        };

        let mut total: Option<Vec<f64>> = None;
        let mut used = Vec::new();
        for name in ["AEXTC55", "EXTCOF55"] {
            if let Some(v) = read(name)? {
                let u = units_of(&f, name);
                let scale = per_km(&u).ok_or_else(|| format!("{name}: unreadable units {u:?}"))?;
                let low: Vec<f64> = v[..n].iter().map(|x| x * scale).collect();
                total = Some(match total {
                    None => low,
                    Some(t) => t.iter().zip(&low).map(|(a, b)| a + b).collect(),
                });
                used.push(name.to_owned());
            }
        }
        if !used.iter().any(|u| u == "AEXTC55") {
            if let Some(tau) = read("AOD3D_SMOKE")? {
                let ph = read("PH")?.ok_or("PH absent")?;
                let phb = read("PHB")?.ok_or("PHB absent")?;
                let low: Vec<f64> = (0..n)
                    .map(|i| {
                        let dz_km = ((ph[n + i] + phb[n + i]) - (ph[i] + phb[i])) / crate::consts::G as f64 / 1000.0;
                        tau[i] / dz_km
                    })
                    .collect();
                total = Some(match total {
                    None => low,
                    Some(t) => t.iter().zip(&low).map(|(a, b)| a + b).collect(),
                });
                used.push("AOD3D_SMOKE/dz".to_owned());
            }
        }
        let aerosol_source = if used.is_empty() { "absent".to_owned() } else { used.join("+") };
        Ok(Self { nx, ny, nz, refl, cldfra, aerosol_km: total.map(as_f32), aerosol_source, hgt, olr })
    }

    /// Borrowed carriers.
    #[must_use]
    pub fn carriers(&self) -> Carriers<'_> {
        Carriers {
            refl: self.refl.as_deref(),
            cldfra: self.cldfra.as_deref(),
            aerosol_km: self.aerosol_km.as_deref(),
            hgt: self.hgt.as_deref(),
            olr: self.olr.as_deref(),
        }
    }
}
