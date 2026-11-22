//! Carry selected WPS soil fractions through a terrain/land-cover overlay.

use crate::error::{Result, StaticError};
use crate::types::{Field, FieldSet, Grid2, Stack3};

pub fn from_baseline(baseline: &FieldSet, landmask: &Grid2)
    -> Result<(FieldSet, serde_json::Value)>
{
    let (ny, nx) = (landmask.ny, landmask.nx);
    let n = ny * nx;
    let mut fields = FieldSet::default();
    let mut audit = serde_json::Map::new();
    for (fraction_name, dominant_name, layer) in [
        ("SOILCTOP", "SCT_DOM", "top_0_30cm"),
        ("SOILCBOT", "SCB_DOM", "bottom_30_100cm"),
    ] {
        let Some(Field::Stack(source)) = baseline.fields.get(fraction_name) else {
            return Err(StaticError::Missing(format!(
                "selected WPS soil requires {fraction_name} category fractions")));
        };
        if (source.planes, source.ny, source.nx) != (16, ny, nx) {
            return Err(StaticError::Invalid(format!(
                "{fraction_name} selected WPS soil shape differs from (16, {ny}, {nx})")));
        }
        if source.data.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err(StaticError::Invalid(format!(
                "{fraction_name} selected WPS soil contains invalid fractions")));
        }
        let dominant = crate::fields::dominant_category(source)?;
        let donors: Vec<bool> = (0..n).map(|cell| {
            landmask.data[cell] > 0.5 && dominant.data[cell] != 14.0
                && (0..16).map(|p| source.data[p*n+cell]).sum::<f64>() > 0.0
        }).collect();
        let repair: Vec<bool> = (0..n).map(|cell| {
            landmask.data[cell] > 0.5 && !donors[cell]
        }).collect();
        let repair_count = repair.iter().filter(|flag| **flag).count();
        let mut fractions: Stack3 = source.clone();
        // Soil the selected geography calls water on every cell is a grid
        // over open sea: its new land (an islet the land cover finds) has
        // no soil to take, and the merge keeps it water, because the
        // baseline has no land to donate its climatology either.  Left
        // here as the geography has it.  Water soil beside land soil the
        // mask calls water is a geography that disagrees with itself, and
        // that is still refused.
        let all_water_soil = dominant.data.iter().all(|category| *category == 14.0);
        let mut without_donor = 0;
        if repair_count > 0 && !donors.iter().any(|flag| *flag) {
            if !all_water_soil {
                return Err(StaticError::Invalid(format!(
                    "{fraction_name} selected WPS soil has {repair_count} land cells with water or missing soil and no land-soil donor")));
            }
            without_donor = repair_count;
        } else if repair_count > 0 {
            let (donor_y, donor_x) = super::nearest_donors(&donors, ny, nx)?;
            for cell in 0..n {
                if repair[cell] {
                    let donor = donor_y[cell] as usize * nx + donor_x[cell] as usize;
                    for p in 0..16 {
                        fractions.data[p*n+cell] = source.data[p*n+donor];
                    }
                }
            }
        }
        let mut water_count = 0;
        for cell in 0..n {
            if landmask.data[cell] <= 0.5 {
                water_count += 1;
                for p in 0..16 {
                    fractions.data[p*n+cell] = if p == 13 { 1.0 } else { 0.0 };
                }
            }
        }
        let dominant = crate::fields::dominant_category(&fractions)?;
        fields.fields.insert(fraction_name.into(), Field::Stack(fractions));
        fields.fields.insert(dominant_name.into(), Field::Plane(dominant));
        audit.insert(layer.into(), serde_json::json!({
            "source": "selected WPS geography",
            "retained_land_cells": donors.iter().filter(|flag| **flag).count(),
            "water_soil_land_cells_from_nearest_land": repair_count - without_donor,
            "water_soil_land_cells_without_donor": without_donor,
            "water_cells": water_count,
        }));
    }
    Ok((fields, serde_json::Value::Object(audit)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_land_fraction_bits_and_repairs_water_on_new_land() {
        let mut baseline = FieldSet::default();
        let mut source = Stack3 { planes: 16, ny: 1, nx: 3, data: vec![0.; 48] };
        source.data[3] = 0.20000000298023224;
        source.data[6] = 0.7999999970197678;
        source.data[13*3+1] = 1.;
        source.data[13*3+2] = 1.;
        for name in ["SOILCTOP", "SOILCBOT"] {
            baseline.fields.insert(name.into(), Field::Stack(source.clone()));
        }
        let mask = Grid2 { ny: 1, nx: 3, data: vec![1., 1., 0.] };
        let (fields, audit) = from_baseline(&baseline, &mask).unwrap();
        let Field::Stack(result) = &fields.fields["SOILCTOP"] else { panic!() };
        for p in 0..16 {
            assert_eq!(result.data[p*3].to_bits(), source.data[p*3].to_bits());
            assert_eq!(result.data[p*3+1].to_bits(), source.data[p*3].to_bits());
            assert_eq!(result.data[p*3+2], if p == 13 {1.} else {0.});
        }
        assert_eq!(audit["top_0_30cm"]["water_soil_land_cells_from_nearest_land"], 1);
    }

    #[test]
    fn refuses_water_soil_on_land_without_donors() {
        // Land soil exists, but only where the mask says water: the
        // geography disagrees with itself.
        let mut baseline = FieldSet::default();
        let mut source = Stack3 { planes: 16, ny: 1, nx: 2, data: vec![0.; 32] };
        source.data[13 * 2] = 1.;
        source.data[2 * 2 + 1] = 1.;
        baseline.fields.insert("SOILCTOP".into(), Field::Stack(source));
        let mask = Grid2 { ny: 1, nx: 2, data: vec![1., 0.] };
        assert!(from_baseline(&baseline, &mask).unwrap_err().to_string().contains("no land-soil donor"));
    }

    #[test]
    fn open_sea_soil_leaves_new_land_without_a_donor_to_the_merge() {
        // A grid over open sea: water soil on every cell, and the land
        // cover finds one islet.  Nothing to donate; no donor search.
        let mut baseline = FieldSet::default();
        let mut source = Stack3 { planes: 16, ny: 2, nx: 2, data: vec![0.; 64] };
        for cell in 0..4 {
            source.data[13 * 4 + cell] = 1.;
        }
        for name in ["SOILCTOP", "SOILCBOT"] {
            baseline.fields.insert(name.into(), Field::Stack(source.clone()));
        }
        let mask = Grid2 { ny: 2, nx: 2, data: vec![0., 1., 0., 0.] };
        let (fields, audit) = from_baseline(&baseline, &mask).unwrap();
        let Field::Plane(dominant) = &fields.fields["SCT_DOM"] else { panic!() };
        assert!(dominant.data.iter().all(|category| *category == 14.));
        assert_eq!(audit["top_0_30cm"]["water_soil_land_cells_without_donor"], 1);
        assert_eq!(audit["top_0_30cm"]["water_soil_land_cells_from_nearest_land"], 0);
        // And a grid with no land at all needs no donor either.
        let sea = Grid2 { ny: 2, nx: 2, data: vec![0.; 4] };
        let (_, audit) = from_baseline(&baseline, &sea).unwrap();
        assert_eq!(audit["top_0_30cm"]["water_soil_land_cells_without_donor"], 0);
    }
}
