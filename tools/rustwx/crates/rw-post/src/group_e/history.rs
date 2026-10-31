//
// WOOF post-processor, group E: read the 2D carriers (and TKE) group E needs
// from a
// WRF-format WOOF history (variable names per the WRF ARW user's guide and
// spec 3.1), first time record.

use super::Carriers;

/// Owned carriers of one frame.
#[derive(Clone, Debug, Default)]
pub struct OwnedCarriers {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub vars: std::collections::BTreeMap<&'static str, Vec<f32>>,
    /// Where TKE came from (`TKE_PBL`, `QKE / 2`), when present.
    pub tke_source: Option<&'static str>,
    pub valid_time: String,
}

const REQUIRED: [&str; 5] = ["HGT", "PSFC", "U10", "V10", "XLAT"];
const OPTIONAL: [&str; 10] = ["UST", "T2", "TH2", "Q2", "HFX", "QFX", "LH", "SINALPHA", "COSALPHA", "PBLH"];

impl OwnedCarriers {
    /// Reads a history file.
    pub fn read(path: &std::path::Path) -> Result<Self, String> {
        let f = netcrust::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let dim = |name: &str| f.dimension(name).map(|d| d.len()).ok_or_else(|| format!("dimension {name} absent"));
        let nx = dim("west_east")?;
        let ny = dim("south_north")?;
        let nz = dim("bottom_top")?;
        let read = |name: &str| -> Result<Vec<f32>, String> {
            f.read_f64_first_record_or_all(name)
                .map(|v| v.into_iter().map(|x| x as f32).collect())
                .map_err(|e| format!("read {name}: {e}"))
        };
        // Every carrier on its own core with its own handle (no single-CPU
        // path): the serial reads held one core while the card waited.
        use rayon::prelude::*;
        let present: Vec<&'static str> = REQUIRED.iter().copied().chain(OPTIONAL.iter().copied().filter(|n| f.variable(n).is_some())).collect();
        let read_own = |name: &str| -> Result<Vec<f32>, String> {
            let g = netcrust::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
            let fill = g.variable(name).and_then(|v| v.attribute("_FillValue").and_then(|a| a.as_f64()));
            g.read_f64_first_record_or_all(name)
                .map(|v| v.into_iter().map(|x| if Some(x) == fill { f32::NAN } else { x as f32 }).collect())
                .map_err(|e| format!("read {name}: {e}"))
        };
        let mut vars = std::collections::BTreeMap::new();
        for (name, v) in present.iter().zip(present.par_iter().map(|n| read_own(n)).collect::<Vec<_>>()) {
            vars.insert(*name, v?);
        }
        let mut tke_source = None;
        if f.variable("TKE_PBL").is_some() {
            let mut tke = read("TKE_PBL")?;
            let ncell = nx * ny;
            if tke.len() == ncell * (nz + 1) {
                // WRF writes TKE_PBL on the full (w) levels, bottom_top_stag,
                // for MYNN (bl_pbl_physics 5 and 6); group E prices the mass
                // levels, so each layer takes the mean of its two bounding
                // interfaces.  Breakage (2.9.2 box door test): `gpuwm
                // export-grib2` of a 3 km HRRR-recipe run refused "TKE has
                // 119637840 values, the grid needs 117292000" (1988 x 1180
                // x 51 against x 50) and wrote nothing.
                let layers: Vec<f32> = (0..nz)
                    .flat_map(|k| {
                        let lower = &tke[k * ncell..(k + 1) * ncell];
                        let upper = &tke[(k + 1) * ncell..(k + 2) * ncell];
                        lower.iter().zip(upper).map(|(a, b)| 0.5 * (a + b)).collect::<Vec<f32>>()
                    })
                    .collect();
                tke = layers;
                tke_source = Some("TKE_PBL (layer mean of the full levels)");
            } else {
                tke_source = Some("TKE_PBL");
            }
            vars.insert("TKE", tke);
        } else if f.variable("QKE").is_some() {
            let q = read("QKE")?;
            vars.insert("TKE", q.into_iter().map(|x| x * 0.5).collect());
            tke_source = Some("QKE / 2");
        }
        let valid_time = crate::history::read_times(&f)
            .ok()
            .and_then(|v| v.into_iter().next())
            .unwrap_or_default();
        Ok(Self { nx, ny, nz, vars, tke_source, valid_time })
    }

    fn get(&self, name: &str) -> Option<&[f32]> {
        self.vars.get(name).map(Vec::as_slice)
    }

    /// Borrowed carriers over Group A's shared state of the same frame.
    #[must_use]
    pub fn carriers<'a>(&'a self, state: &'a crate::state::ColumnState) -> Carriers<'a> {
        let r = |n: &str| self.get(n).expect("required carrier was read");
        Carriers {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            state,
            tke: self.get("TKE"),
            hgt: r("HGT"),
            psfc: r("PSFC"),
            u10: r("U10"),
            v10: r("V10"),
            ust: self.get("UST"),
            t2: self.get("T2"),
            th2: self.get("TH2"),
            q2: self.get("Q2"),
            hfx: self.get("HFX"),
            qfx: self.get("QFX"),
            lh: self.get("LH"),
            sinalpha: self.get("SINALPHA"),
            cosalpha: self.get("COSALPHA"),
        }
    }
}
