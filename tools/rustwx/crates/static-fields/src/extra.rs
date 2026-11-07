//! Row-driven continuous fields. No field or dataset names are special.
use std::path::PathBuf;
use crate::error::{Result, StaticError};
use crate::fields::{crop_grid, landmask_from_landusef};
use crate::geog::{GeogDataset, SourceType};
use crate::interp::InterpOp;
use crate::projection::ProjectedGrid;
use crate::sampler::DomainSampler;
use crate::types::{Field, FieldSet, Stack3};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct WaterMask {
    /// Categorical land-use source used by the native static build.
    pub dataset_path: PathBuf,
    /// GEOGRID.TBL mask fill, normally the row's fill_missing.
    pub fill_missing: f64,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtraFieldSpec {
    pub output_name: String,
    pub dataset_path: PathBuf,
    pub interp_options: Vec<String>,
    /// Optional WPS average_gcell ratio; absent rows keep their existing sampler.
    #[serde(default)]
    pub average_gcell: Option<f32>,
    pub fill_missing: f64,
    pub planes: usize,
    #[serde(default)]
    pub z_dim_name: Option<String>,
    #[serde(default)]
    pub water_mask: Option<WaterMask>,
}

fn interpolation(names: &[String]) -> Result<Vec<InterpOp>> {
    if names.is_empty() {
        return Err(StaticError::Invalid("interp_options must not be empty".into()));
    }
    names.iter().map(|name| Ok(match name.as_str() {
        "nearest_neighbor" => InterpOp::NearestNeighbor,
        "four_pt" => InterpOp::FourPt,
        "average_4pt" => InterpOp::Average4Pt,
        "average_16pt" => InterpOp::Average16Pt,
        "sixteen_pt" => InterpOp::SixteenPt,
        "search" => InterpOp::Search,
        _ => return Err(StaticError::Invalid(format!("unsupported GEOGRID interpolation option {name:?}"))),
    })).collect()
}

/// Build a JSON list of continuous rows through the native sampler.
/// average_gcell is declared separately from the interpolation fallback list.
/// Unknown options are refused rather than approximated.
pub fn build_extra_fields(grid: &ProjectedGrid, specs: &[ExtraFieldSpec], halo: usize) -> Result<FieldSet> {
    let dom = DomainSampler::new(grid, halo)?;
    let mut set = FieldSet::default();
    for spec in specs {
        if spec.output_name.is_empty() || spec.output_name.len() > 128
            || !spec.output_name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            || set.fields.contains_key(&spec.output_name) {
            return Err(StaticError::Invalid(format!("invalid or duplicate output_name {:?}", spec.output_name)));
        }
        if spec.planes == 0 || !spec.fill_missing.is_finite() {
            return Err(StaticError::Invalid("planes must be positive and fill_missing finite".into()));
        }
        if spec.average_gcell.is_some_and(|r| !r.is_finite() || r <= 0.0) {
            return Err(StaticError::Invalid("average_gcell must be positive and finite".into()));
        }
        let seq = interpolation(&spec.interp_options)?;
        let ds = GeogDataset::open(&spec.dataset_path, None)?;
        if ds.index.kind != SourceType::Continuous || ds.index.nz() as usize != spec.planes {
            return Err(StaticError::Invalid(format!("{} requires a continuous dataset with {} planes; index declares {:?}, {} planes",
                spec.output_name, spec.planes, ds.index.kind, ds.index.nz())));
        }
        let active = if let Some(rule) = &spec.water_mask {
            if !rule.fill_missing.is_finite() {
                return Err(StaticError::Invalid("water_mask fill_missing must be finite".into()));
            }
            let lu = GeogDataset::open(&rule.dataset_path, None)?;
            let win = dom.window(&lu, 3)?;
            let receipt = dom.require_source_coverage(&lu, &win, &format!("{}_water_mask", spec.output_name))?;
            set.coverage_reports.insert(format!("{}_water_mask", spec.output_name), receipt);
            let frac = dom.categorical(&lu, &win, true)?;
            let iswater = lu.index.iswater.filter(|&w| w != 0).unwrap_or(17);
            let mask = landmask_from_landusef(&frac, iswater, lu.index.islake)?;
            Some(mask.data.iter().map(|&v| v != 0.0).collect::<Vec<bool>>())
        } else { None };
        let mut stack = Stack3::filled(spec.planes, dom.ny, dom.nx, spec.fill_missing);
        for z in 0..spec.planes {
            let win = dom.window_plane(&ds, 3, z)?;
            if z == 0 {
                let receipt = dom.require_source_coverage(&ds, &win, &spec.output_name)?;
                set.coverage_reports.insert(spec.output_name.clone(), receipt);
            }
            let mut plane = if let Some(ratio) = spec.average_gcell {
                dom.continuous_wps(&ds, &win, z, &seq, spec.fill_missing, true, active.as_deref(), ratio)?
            } else if seq.contains(&InterpOp::NearestNeighbor) {
                dom.continuous_wps(&ds, &win, z, &seq, spec.fill_missing, false, active.as_deref(), 4.0)?
            } else {
                dom.continuous(&ds, &win, z, &seq, spec.fill_missing, false, active.as_deref())?
            };
            if let (Some(mask), Some(rule)) = (&active, &spec.water_mask) {
                for (v, &keep) in plane.data.iter_mut().zip(mask) {
                    if !keep { *v = rule.fill_missing; }
                }
            }
            let plane = crop_grid(&dom, &plane);
            let n = dom.ny * dom.nx;
            stack.data[z*n..(z+1)*n].copy_from_slice(&plane.data);
        }
        let field = if spec.planes == 1 {
            Field::Plane(crate::types::Grid2 { ny: dom.ny, nx: dom.nx, data: stack.data })
        } else { Field::Stack(stack) };
        set.fields.insert(spec.output_name.clone(), field);
    }
    Ok(set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GridSpec, ProjectionKind};

    fn grid() -> ProjectedGrid {
        ProjectedGrid::new(GridSpec {
            kind: ProjectionKind::Mercator, ref_lat: 0.0, ref_lon: 0.0,
            truelat1: 0.0, truelat2: 0.0, stand_lon: 0.0,
            dx: 12000.0, dy: 12000.0, e_we: 5, e_sn: 4,
            known_x: 2.5, known_y: 2.0, moad_cen_lat: 0.0, moad_cen_lon: 0.0,
            lat_deg: vec![], lon0_deg: 0.0, dlon_deg: 0.0,
        }).unwrap()
    }

    #[test]
    fn arbitrary_rows_planes_fill_mask_and_refusals() {
        let root = std::env::temp_dir().join(format!("extra-fields-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let dsdir = root.join("unrelated_source");
        let ludir = root.join("water");
        for (path, kind, planes, extra) in [
            (&dsdir, "continuous", 3, "scale_factor=0.001\nmissing_value=65535\n"),
            (&ludir, "categorical", 1, "category_min=1\ncategory_max=2\niswater=2\n"),
        ] {
            std::fs::create_dir_all(path).unwrap();
            std::fs::write(path.join("index"), format!(
                "type={kind}\nprojection=regular_ll\ndx=1\ndy=1\nknown_x=1\nknown_y=1\nknown_lat=-10\nknown_lon=-10\nwordsize=2\ntile_x=21\ntile_y=21\ntile_z={planes}\nsigned=no\n{extra}"
            )).unwrap();
            let mut bytes = vec![];
            for z in 0..planes {
                for j in 0..21u16 {
                    for i in 0..21u16 {
                        let value = if kind == "categorical" { 2u16 }
                            else if z == 2 { 65535 } else { z * 100 + i*i + j*j };
                        bytes.extend(value.to_be_bytes());
                    }
                }
            }
            std::fs::write(path.join("00001-00021.00001-00021"), bytes).unwrap();
        }
        let mut row = ExtraFieldSpec {
            output_name: "UNRELATED".into(), dataset_path: dsdir,
            average_gcell: None, interp_options: vec!["four_pt".into()], fill_missing: 0.75,
            planes: 3, z_dim_name: Some("arbitrary_axis".into()), water_mask: None,
        };
        let grid = grid();
        let set = build_extra_fields(&grid, &[row.clone()], 1).unwrap();
        let f = set.get("UNRELATED").unwrap();
        assert_eq!(f.dims(), (3, 3, 4));
        let data = f.data();
        for k in 0..12 {
            assert!((data[12+k] - data[k] - 0.1).abs() < 1e-14);
            assert_eq!(data[24+k], 0.75);
        }
        // Compare the row assembly to the exact existing continuous path.
        let dom = DomainSampler::new(&grid, 1).unwrap();
        let ds = GeogDataset::open(&row.dataset_path, None).unwrap();
        let win = dom.window_plane(&ds, 3, 0).unwrap();
        let expected = crop_grid(&dom, &dom.continuous(&ds, &win, 0,
            &[InterpOp::FourPt], 0.75, false, None).unwrap());
        assert_eq!(&data[..12], expected.data.as_slice());
        row.interp_options = vec!["average_4pt".into()];
        let average = build_extra_fields(&grid, &[row.clone()], 1).unwrap();
        assert_ne!(average.get("UNRELATED").unwrap().data()[0], data[0]);
        row.water_mask = Some(WaterMask { dataset_path: ludir, fill_missing: 0.0 });
        let water = build_extra_fields(&grid, &[row.clone()], 1).unwrap();
        assert!(water.get("UNRELATED").unwrap().data().iter().all(|&v| v == 0.0));
        row.water_mask = None;
        assert!(build_extra_fields(&grid, &[row.clone(), row.clone()], 1).is_err());
        row.interp_options = vec!["wt_average_4pt".into()];
        assert!(build_extra_fields(&grid, &[row.clone()], 1).unwrap_err().to_string().contains("unsupported GEOGRID"));
        row.interp_options = vec!["four_pt".into()];
        row.planes = 2;
        assert!(build_extra_fields(&grid, &[row], 1).is_err());
        assert!(build_extra_fields(&grid, &[], 1).unwrap().fields.is_empty());
        for directory in std::fs::read_dir(&root).unwrap() {
            for entry in std::fs::read_dir(directory.unwrap().path()).unwrap() {
                let entry = entry.unwrap();
                println!("deleted fixture {} {} bytes", entry.path().display(), entry.metadata().unwrap().len());
            }
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}
