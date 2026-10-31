//! Regular latitude-longitude tapes the way a global model exports them for
//! this renderer (WOOF Global's `woof global render`/`export`): MAP_PROJ 6,
//! longitudes -180 .. 180-dlon with no duplicated wrap column for the whole
//! globe, or a lat/lon window cut from that ring for a regional crop.
//!
//! The 6 h precipitation panel is a windowed product (F006 minus F000),
//! drawn by the windowed lane; the pressure panel beside it is a direct
//! product.  Until the windowed lane carried the inverse-raster
//! projection the direct lane already used, it drew a regular lat/lon
//! grid as a forward-projected mesh: a whole-globe map smeared the cells
//! beside the date line across the panel, and a regional crop framed
//! itself to the rectangle inscribed in the grid's curved footprint, a
//! thin strip.  These tests hold both pictures to the field itself.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use netcdf_writer::{AttrValue, NcFormat, NcType, NcWriter, Schema, VarData};

const NZ: usize = 4;

mod support;
use support::Scratch;

/// A regular lat/lon axis pair: cell centres, south to north and west to east.
struct Axes {
    lat: Vec<f32>,
    lon: Vec<f32>,
}

impl Axes {
    /// The whole globe at `step` degrees, longitudes -180 .. 180-step.
    fn globe(step: f32) -> Self {
        let nlat = (180.0 / step).round() as usize;
        let nlon = (360.0 / step).round() as usize;
        Self {
            lat: (0..nlat).map(|j| -90.0 + step * (j as f32 + 0.5)).collect(),
            lon: (0..nlon).map(|i| -180.0 + step * i as f32).collect(),
        }
    }

    /// The window of [`Axes::globe`] inside the given bounds, as a bbox
    /// export cuts it.
    fn window(step: f32, lat_min: f32, lat_max: f32, lon_min: f32, lon_max: f32) -> Self {
        let globe = Self::globe(step);
        Self {
            lat: globe
                .lat
                .into_iter()
                .filter(|lat| *lat >= lat_min && *lat <= lat_max)
                .collect(),
            lon: globe
                .lon
                .into_iter()
                .filter(|lon| *lon >= lon_min && *lon <= lon_max)
                .collect(),
        }
    }

    fn nx(&self) -> usize {
        self.lon.len()
    }

    fn ny(&self) -> usize {
        self.lat.len()
    }

    fn is_whole_ring(&self) -> bool {
        let step = (self.lon[1] - self.lon[0]).abs();
        ((self.nx() as f32 * step) - 360.0).abs() < 0.5 * step
    }
}

/// One frame: the accumulated precipitation `rain(lat, lon)` (mm) and a
/// uniform 10 m wind.  Everything else is a plain resting column.
fn write_frame(
    dir: &Path,
    axes: &Axes,
    lead_hours: i64,
    rain: &dyn Fn(f32, f32) -> f32,
    wind_m_s: f32,
) -> PathBuf {
    let init = chrono::NaiveDate::from_ymd_opt(2026, 9, 20)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    let valid = (init + chrono::Duration::hours(lead_hours))
        .format("%Y-%m-%d_%H:%M:%S")
        .to_string();
    let path = dir.join(format!("wrfout_d01_{}", valid.replace(':', "_")));
    let (nx, ny) = (axes.nx(), axes.ny());
    let cells = nx * ny;
    let volume = cells * NZ;
    let step = (axes.lon[1] - axes.lon[0]).abs();
    let dx_m = step * std::f32::consts::PI / 180.0 * 6_370_000.0;

    let mut schema = Schema::new(NcFormat::Offset64);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let bottom_top = schema.def_dim("bottom_top", NZ, false).unwrap();
    let bottom_top_stag = schema.def_dim("bottom_top_stag", NZ + 1, false).unwrap();
    let south_north = schema.def_dim("south_north", ny, false).unwrap();
    let south_north_stag = schema.def_dim("south_north_stag", ny + 1, false).unwrap();
    let west_east = schema.def_dim("west_east", nx, false).unwrap();
    let west_east_stag = schema.def_dim("west_east_stag", nx + 1, false).unwrap();
    for (name, value) in [
        ("TITLE", "global lat-lon render tape"),
        ("START_DATE", "2026-09-20_00:00:00"),
        ("SIMULATION_START_DATE", "2026-09-20_00:00:00"),
        ("MAP_PROJ_CHAR", "Cylindrical Equidistant"),
    ] {
        schema
            .put_global_attr(name, AttrValue::Text(value.into()))
            .unwrap();
    }
    for (name, value) in [("MAP_PROJ", 6i32), ("GRID_ID", 1), ("PARENT_ID", 0)] {
        schema
            .put_global_attr(name, AttrValue::Ints(vec![value]))
            .unwrap();
    }
    let cen_lon = axes.lon.iter().sum::<f32>() / nx as f32;
    let cen_lat = if axes.is_whole_ring() {
        0.0
    } else {
        axes.lat.iter().sum::<f32>() / ny as f32
    };
    for (name, value) in [
        ("DX", dx_m),
        ("DY", dx_m),
        ("TRUELAT1", 0.0),
        ("TRUELAT2", 0.0),
        ("STAND_LON", 0.0),
        ("CEN_LAT", cen_lat),
        ("CEN_LON", cen_lon),
        ("POLE_LAT", 90.0),
        ("POLE_LON", 0.0),
        ("DT", 300.0),
    ] {
        schema
            .put_global_attr(name, AttrValue::Floats(vec![value]))
            .unwrap();
    }
    let times = schema
        .def_var("Times", NcType::Char, &[time, strlen])
        .unwrap();
    let plane = |schema: &mut Schema, name: &str, units: &str| {
        let id = schema
            .def_var(name, NcType::Float, &[time, south_north, west_east])
            .unwrap();
        schema
            .put_var_attr(id, "units", AttrValue::Text(units.into()))
            .unwrap();
        id
    };
    let volume_var = |schema: &mut Schema, name: &str, dims: &[usize], units: &str| {
        let id = schema.def_var(name, NcType::Float, dims).unwrap();
        schema
            .put_var_attr(id, "units", AttrValue::Text(units.into()))
            .unwrap();
        id
    };
    let mass = [time, bottom_top, south_north, west_east];
    let planes = [
        ("XLAT", "degree_north"),
        ("XLONG", "degree_east"),
        ("T2", "K"),
        ("Q2", "kg kg-1"),
        ("PSFC", "Pa"),
        ("HGT", "m"),
        ("U10", "m s-1"),
        ("V10", "m s-1"),
        ("SINALPHA", "1"),
        ("COSALPHA", "1"),
        ("TSK", "K"),
        ("LANDMASK", "1"),
        ("RAINC", "mm"),
        ("RAINNC", "mm"),
    ]
    .map(|(name, units)| (name, plane(&mut schema, name, units)));
    let t = volume_var(&mut schema, "T", &mass, "K");
    let p = volume_var(&mut schema, "P", &mass, "Pa");
    let pb = volume_var(&mut schema, "PB", &mass, "Pa");
    let qv = volume_var(&mut schema, "QVAPOR", &mass, "kg kg-1");
    let qc = volume_var(&mut schema, "QCLOUD", &mass, "kg kg-1");
    let stag_z = [time, bottom_top_stag, south_north, west_east];
    let ph = volume_var(&mut schema, "PH", &stag_z, "m2 s-2");
    let phb = volume_var(&mut schema, "PHB", &stag_z, "m2 s-2");
    let u = volume_var(
        &mut schema,
        "U",
        &[time, bottom_top, south_north, west_east_stag],
        "m s-1",
    );
    let v = volume_var(
        &mut schema,
        "V",
        &[time, bottom_top, south_north_stag, west_east],
        "m s-1",
    );

    let mut lat = Vec::with_capacity(cells);
    let mut lon = Vec::with_capacity(cells);
    let mut rain_nc = Vec::with_capacity(cells);
    let mut psfc = Vec::with_capacity(cells);
    for &cell_lat in &axes.lat {
        for &cell_lon in &axes.lon {
            lat.push(cell_lat);
            lon.push(cell_lon);
            rain_nc.push(rain(cell_lat, cell_lon));
            // A smooth periodic pressure pattern, so isobars exist.
            let rl = cell_lon.to_radians();
            let rp = cell_lat.to_radians();
            psfc.push(101_325.0 + 1_200.0 * (3.0 * rl).cos() * rp.cos().powi(2));
        }
    }
    let constant = |value: f32| vec![value; cells];
    let values: Vec<(&str, Vec<f32>)> = vec![
        ("XLAT", lat),
        ("XLONG", lon),
        ("T2", constant(288.0)),
        ("Q2", constant(0.008)),
        ("PSFC", psfc.clone()),
        ("HGT", constant(0.0)),
        ("U10", constant(wind_m_s)),
        ("V10", constant(0.0)),
        ("SINALPHA", constant(0.0)),
        ("COSALPHA", constant(1.0)),
        ("TSK", constant(288.0)),
        ("LANDMASK", constant(0.0)),
        ("RAINC", constant(0.0)),
        ("RAINNC", rain_nc),
    ];

    let sigma = [0.97f32, 0.85, 0.6, 0.3];
    let mut base_pressure = Vec::with_capacity(volume);
    for s in sigma {
        base_pressure.extend(psfc.iter().map(|p| p * s));
    }
    let mut geopotential = Vec::with_capacity(cells * (NZ + 1));
    geopotential.extend(std::iter::repeat_n(0.0f32, cells));
    for s in sigma {
        geopotential.extend(std::iter::repeat_n(287.0 * 280.0 * -s.ln(), cells));
    }

    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer
        .write_record(0, times, VarData::Char(valid.as_bytes()))
        .unwrap();
    for (name, id) in planes {
        let data = &values.iter().find(|(n, _)| *n == name).unwrap().1;
        writer
            .write_record(0, id, VarData::F32(data.as_slice()))
            .unwrap();
    }
    for (id, data) in [
        (t, vec![-12.0f32; volume]),
        (p, vec![0.0; volume]),
        (pb, base_pressure),
        (qv, vec![0.008; volume]),
        (qc, vec![0.0; volume]),
        (ph, vec![0.0; cells * (NZ + 1)]),
        (phb, geopotential),
        (u, vec![wind_m_s; (nx + 1) * ny * NZ]),
        (v, vec![0.0; nx * (ny + 1) * NZ]),
    ] {
        writer
            .write_record(0, id, VarData::F32(data.as_slice()))
            .unwrap();
    }
    writer.finish().unwrap();
    path
}

fn render(root: &Path, products: &str, inputs: &[PathBuf]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rw_wrfbatch"))
        .args(["--products", products, "--width", "1000", "--height", "620"])
        .arg("--store-root")
        .arg(root.join("store"))
        .arg("--out-dir")
        .arg(root.join("out"))
        .args(inputs)
        .env("GPUWM_NO_LOCAL_GPU", "1")
        .env("CUDA_VISIBLE_DEVICES", "-1")
        .env("RUSTWX_BATCH_RENDER_THREADS", "1")
        .output()
        .unwrap()
}

fn rendered(stdout: &str, slug: &str) -> PathBuf {
    let prefix = format!("RENDERED {slug} ");
    let paths: Vec<PathBuf> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix(prefix.as_str()).map(PathBuf::from))
        .filter(|path| path.to_string_lossy().contains("_f006"))
        .collect();
    assert_eq!(paths.len(), 1, "one F006 {slug} panel expected:\n{stdout}");
    paths.into_iter().next().unwrap()
}

/// A pixel of a filled colour scale rather than of the page, the basemap
/// or the linework: every one of those is a near-grey.
fn is_fill(pixel: &image::Rgba<u8>) -> bool {
    let [r, g, b, _] = pixel.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    max.saturating_sub(min) >= 60
}

/// Fill pixels inside the map area: everything left of the colour bar.
/// `RW_GLOBAL_PANELS_KEEP=<dir>` keeps a copy of every panel looked at.
fn fill_pixels(png: &Path, tag: &str) -> (image::RgbaImage, Vec<(u32, u32)>) {
    if let Some(keep) = std::env::var_os("RW_GLOBAL_PANELS_KEEP") {
        let keep = PathBuf::from(keep);
        std::fs::create_dir_all(&keep).unwrap();
        let name = format!("{tag}-{}", png.file_name().unwrap().to_string_lossy());
        std::fs::copy(png, keep.join(name)).unwrap();
    }
    let image = image::open(png).unwrap().to_rgba8();
    let map_right = image.width() * 88 / 100;
    let fills = image
        .enumerate_pixels()
        .filter(|(x, _, pixel)| *x < map_right && is_fill(pixel))
        .map(|(x, y, _)| (x, y))
        .collect();
    (image, fills)
}

#[test]
fn a_longitude_periodic_precipitation_field_draws_no_date_line_seam() {
    let scratch = Scratch::new("rw-global-latlon", "seam");
    let axes = Axes::globe(1.0);
    // A U of rain: a band astride the date line from 60 S to 60 N, joined
    // across the whole globe by a zonal band near 52 N.  In the grid's
    // index space that is ONE region that leaves the west edge and comes
    // back in at the east edge.  The southern-hemisphere ocean between the
    // arms of the U, far from the date line, is dry.
    let u_of_rain = |lat: f32, lon: f32| {
        let beside_date_line = ((-lon.to_radians().cos() - 0.85) / 0.15).clamp(0.0, 1.0);
        let below_60 = ((60.0 - lat.abs()) / 5.0).clamp(0.0, 1.0);
        let northern_band = (-((lat - 52.0) / 5.0).powi(2)).exp();
        40.0 * (beside_date_line * below_60).max(northern_band)
    };
    let inputs = [
        write_frame(&scratch.0, &axes, 0, &|_, _| 0.0, 5.0),
        write_frame(&scratch.0, &axes, 6, &u_of_rain, 5.0),
    ];
    let output = render(&scratch.0, "qpf_6h", &inputs);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let (image, fills) = fill_pixels(&rendered(&stdout, "qpf_6h"), "seam");
    let (width, height) = (image.width(), image.height());

    // The middle of the map's southern half holds longitudes far from the
    // date line and latitudes far from the northern band: the field is
    // zero there, so any fill is the U closed across the map from one edge
    // of the grid to the other.
    let middle = (width * 30 / 100)..(width * 58 / 100);
    let south = (height * 58 / 100)..(height * 90 / 100);
    let smeared = fills
        .iter()
        .filter(|(x, y)| middle.contains(x) && south.contains(y))
        .count();
    assert_eq!(
        smeared, 0,
        "{smeared} filled pixels in the dry southern ocean between the arms of a U of rain \
         that crosses the date line"
    );

    // The northern band is drawn across the middle, and the arms at BOTH
    // edges of the globe, in comparable amounts.
    let north_middle = fills
        .iter()
        .filter(|(x, y)| middle.contains(x) && *y < height * 45 / 100)
        .count();
    assert!(
        north_middle > 1_000,
        "the zonal band is missing ({north_middle} pixels)"
    );
    let left = fills.iter().filter(|(x, _)| *x < width * 30 / 100).count();
    let right = fills.iter().filter(|(x, _)| *x >= width * 58 / 100).count();
    assert!(
        left > 2_000 && right > 2_000,
        "rain beside the date line must draw at both edges (left {left}, right {right})"
    );
    let ratio = left as f64 / right as f64;
    assert!(
        (0.6..=1.6).contains(&ratio),
        "the two sides of the date line draw unequal amounts of one symmetric field: \
         left {left}, right {right}"
    );
}

#[test]
fn a_regional_crop_fills_both_halves_of_the_pressure_and_precipitation_pair() {
    let scratch = Scratch::new("rw-global-latlon", "crop");
    // A North America window cut from a 2 degree global ring, as a bbox
    // export writes it.
    let axes = Axes::window(2.0, 10.0, 75.0, -170.0, -50.0);
    assert!(!axes.is_whole_ring());
    let inputs = [
        write_frame(&scratch.0, &axes, 0, &|_, _| 0.0, 15.0),
        write_frame(&scratch.0, &axes, 6, &|_, _| 50.0, 15.0),
    ];
    let output = render(&scratch.0, "mslp_10m_winds,qpf_6h", &inputs);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Both fields are nonzero on every cell, so each panel's fill is the
    // grid's own footprint: the bounding box of the fill IS the drawn map.
    let mut boxes = Vec::new();
    let mut counts = Vec::new();
    for slug in ["mslp_10m_winds", "qpf_6h"] {
        let (image, fills) = fill_pixels(&rendered(&stdout, slug), "crop");
        let (w, h) = (image.width(), image.height());
        let min_x = fills.iter().map(|(x, _)| *x).min().unwrap_or(0);
        let max_x = fills.iter().map(|(x, _)| *x).max().unwrap_or(0);
        let min_y = fills.iter().map(|(_, y)| *y).min().unwrap_or(0);
        let max_y = fills.iter().map(|(_, y)| *y).max().unwrap_or(0);
        let box_w = max_x.saturating_sub(min_x) + 1;
        let box_h = max_y.saturating_sub(min_y) + 1;
        assert!(
            box_w >= w * 70 / 100 && box_h >= h * 70 / 100,
            "{slug}: the regional crop draws a {box_w} x {box_h} map on a {w} x {h} picture"
        );
        boxes.push((min_x, max_x, min_y, max_y));
        counts.push(fills.len());
    }
    // One grid, one frame: the two halves cover the same area.  (The
    // pressure half carries isobars and barbs over its fill, so it reads a
    // little lower.)
    let ratio = counts[1] as f64 / counts[0] as f64;
    assert!(
        (0.9..=1.35).contains(&ratio),
        "the precipitation half fills {} pixels where the pressure half of the same grid fills {}",
        counts[1],
        counts[0]
    );
    // The two halves of the pair frame the same region the same way.
    let (a, b) = (boxes[0], boxes[1]);
    let tolerance = 25;
    for (name, va, vb) in [
        ("left", a.0, b.0),
        ("right", a.1, b.1),
        ("top", a.2, b.2),
        ("bottom", a.3, b.3),
    ] {
        assert!(
            va.abs_diff(vb) <= tolerance,
            "the pressure and precipitation panels disagree on their {name} edge: {va} vs {vb}"
        );
    }
}
