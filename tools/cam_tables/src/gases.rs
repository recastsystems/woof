use std::{fs, path::Path};
#[derive(Clone)]
pub struct GasTable {
    pub years: Vec<i32>,
    pub days: Vec<f32>,
    pub values: Vec<[f64; 5]>,
}
fn leap(y: i32) -> bool {
    y % 4 == 0 && y % 100 != 0 || y % 400 == 0
}
impl GasTable {
    pub fn read(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = fs::read_to_string(path)?;
        let mut years = Vec::new();
        let mut days = Vec::new();
        let mut values = Vec::new();
        for (line_no, line) in text.lines().enumerate().skip(2) {
            if line.trim().is_empty() {
                continue;
            }
            if line.len() < 57 || !line.is_ascii() {
                return Err(format!(
                    "CAM gas record {} is shorter than the WRF fixed format",
                    line_no + 1
                )
                .into());
            }
            let year: i32 = line[0..4].trim().parse()?;
            let mut row: [f64; 5] = [0.; 5];
            for (i, (lo, hi)) in [(5, 13), (14, 24), (25, 35), (36, 46), (47, 57)]
                .into_iter()
                .enumerate()
            {
                row[i] = line[lo..hi].trim().parse()?;
            }
            if row.iter().any(|x| !x.is_finite()) {
                return Err("nonfinite CAM gas concentration".into());
            }
            if years.last().is_some_and(|last| *last >= year) {
                return Err("CAM gas years are not strictly increasing".into());
            }
            // read_CAMgases assigns June to every annual record and uses this FP32 date.
            years.push(year);
            days.push(if leap(year) { 166.5 } else { 165.5 });
            values.push(row);
        }
        if years.len() < 2 || years.len() > 1800 {
            return Err("CAM gas table requires 2..1800 annual records".into());
        }
        Ok(Self {
            years,
            days,
            values,
        })
    }
    pub fn interpolate(&self, year: i32, julian: f32) -> [f64; 5] {
        let n = self.years.len();
        let mut upper = None;
        let mut checked = 0;
        for i in 0..n {
            checked = i + 2;
            if self.years[i] > year || (self.years[i] == year && self.days[i] > julian) {
                upper = Some(i);
                break;
            }
        }
        if checked >= n {
            upper = Some(n - 2);
        }
        let fallback = [
            3.55e-4_f32 as f64,
            0.311e-6_f32 as f64,
            1.714e-6_f32 as f64,
            0.280e-9_f32 as f64,
            0.503e-9_f32 as f64,
        ];
        let Some(mut hi) = upper else {
            return fallback;
        };
        if hi == 0 {
            hi = 1;
        }
        // WRF chooses its interpolation window from valid CO2 rows for all five gases.
        let valid: Vec<_> = self
            .values
            .iter()
            .enumerate()
            .filter(|(_, x)| x[0] > 0.)
            .map(|(i, _)| i)
            .collect();
        if valid.len() < 2 {
            return fallback;
        }
        let lo;
        if !valid.iter().any(|i| *i < hi) {
            lo = valid[0];
            hi = valid[1];
        } else {
            if self.values[hi][0] < 0. {
                hi = valid
                    .iter()
                    .copied()
                    .find(|i| *i > hi)
                    .unwrap_or_else(|| *valid.iter().filter(|i| **i < hi).last().unwrap());
            }
            let mut lower = (upper.unwrap().saturating_sub(1)).min(hi - 1);
            if self.values[lower][0] < 0. {
                lower = *valid.iter().filter(|i| **i < lower).last().unwrap();
            }
            lo = lower;
        }
        let yi = self.years[lo];
        let yf = self.years[hi];
        let mut f2 = self.days[lo] as i32 as f32;
        let mut f1 = self.days[hi] as i32 as f32;
        let mut x = julian;
        for y in year.min(yi)..year.max(yf) {
            let nd = if leap(y) { 366.0_f32 } else { 365.0_f32 };
            if y < yi {
                f2 += nd;
            }
            if y < year {
                x += nd;
            }
            if y < yf {
                f1 += nd;
            }
        }
        let dt = f1 - f2;
        let a = (f1 - x) / dt;
        let b = (x - f2) / dt;
        let mut result = [0.; 5];
        for i in 0..5 {
            result[i] = self.values[lo][i] * (a as f64) + self.values[hi][i] * (b as f64);
        }
        if result.iter().any(|x| *x < 0.) {
            return fallback;
        }
        result[0] = result[0].max(270.);
        result[1] = result[1].max(270.);
        result[2] = result[2].max(700.);
        for (i, scale) in [1.0e-6_f32, 1.0e-9_f32, 1.0e-9_f32, 1.0e-12_f32, 1.0e-12_f32]
            .into_iter()
            .enumerate()
        {
            result[i] *= scale as f64;
        }
        result
    }
}
