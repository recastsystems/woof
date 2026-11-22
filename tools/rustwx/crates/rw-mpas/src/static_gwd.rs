//! Sub-grid orography statistics -- `var2d`, `con`, `oa1..4`, `ol1..4` -- on
//! the bounded, admission-gated geography reader.
//!
//! A transcription of `mpas_init_atm_gwd.F` (`compute_gwd_fields` and the
//! `get_*` statistic functions), not a reinterpretation of it. Every formula
//! below is the one MPAS evaluates, including the integer-division box halving
//! that decides which pixels count as "upstream", because the gravity-wave
//! drag scheme downstream was tuned against these exact definitions.
//!
//! These statistics are NOT cell averages. MPAS cuts a rectangular lat/lon box
//! of the 30-arc-second topography centred on each cell, sized from the cell's
//! mean `dcEdge`, and reduces that box. A Voronoi-cell average would be a
//! different quantity wearing the same name, so the box is reproduced exactly.
//!
//! # Three places this is deliberately NOT the Fortran, each named
//!
//! **The water category is read, not hardcoded.** MPAS hardcodes `WATER = 16`,
//! the USGS land-use code, when splitting land from water inside `get_con`.
//! That constant is wrong for any MODIS dataset, where 16 is barren land and 17
//! is water -- pointed at MODIS tiles unchanged it calls every desert an ocean.
//! This reads the water code from the dataset's own index. Where the dataset IS
//! USGS the two agree exactly; where it is not, this is right and MPAS is
//! wrong. A declared lake class (`islake`, 21 in the with-lakes MODIS set) is
//! water in the same statistics, as the static's own land use folds it.
//! Recorded in the receipt as `water_category_source`.
//!
//! **The source frame is read, not assumed.** MPAS holds one global array and
//! knows, by construction, that its column 1 is 180 W. A WPS_GEOG archive
//! declares its own origin, and the archives do not agree with each other:
//! measured on the staged tree, `topo_gmted2010_30s` declares
//! `known_lon=0.004166667` while `modis_landuse_20class_30s_with_lakes`
//! declares `known_lon=-179.99583`. Taking either as "column 1 is 180 W" puts
//! one of the two exactly half a globe from every cell -- terrain sampled from
//! the antipode, silently, with every value finite and plausible. Every index
//! here is derived from the dataset's own `index`, and the offset between the
//! two datasets is computed rather than assumed. See [`FrameAlignment`].
//!
//! **A cell's box never falls off the band.** The Fortran holds the whole
//! array, so a box that reflects across a pole reads real rows. A band reader
//! that widened only to the unreflected span would hand those pixels zero, and
//! WHICH pixels depends on how cells happened to group into bands -- which
//! would make the answer a function of the memory budget. Bands here are
//! widened to the reflected span, so every cell reads the same numbers at
//! every budget and every worker count. That is what [`compute`] means when it
//! promises determinism.

use rayon::prelude::*;

use crate::error::{MpasError, MpasResult};
use crate::static_geog::GeogDataset;
use crate::static_memory::checked_vec;

/// Earth radius, the value `mpas_init_atm_gwd.F` carries.
pub const RE: f64 = 6_371_229.0;

/// The ten fields this module produces, one value per cell.
#[derive(Debug, Clone, Default)]
pub struct GwdFields {
    pub var2d: Vec<f64>,
    pub con: Vec<f64>,
    pub oa: [Vec<f64>; 4],
    pub ol: [Vec<f64>; 4],
    /// Maximum sub-grid elevation per cell (m MSL), MPAS-Model PR #1408 `elvmax` (the KIM drag's
    /// `omax`). Empty unless the kim2026 recipe ran; the v8.4.2 recipe never writes it.
    pub elvmax: Vec<f64>,
    /// Dominant land mask per cell over the box, as MPAS's `hlanduse`.
    pub hlanduse: Vec<u8>,
    /// Cells whose variance was replaced by the mean of their neighbours.
    pub smoothed_cells: usize,
    /// Regional meshes only: outermost-ring cells left out of the isolated-
    /// point smooth because part of their neighbour ring was culled. 0 on a
    /// global mesh.
    pub smooth_skipped_boundary_cells: usize,
    /// Which land-use code counted as water, and where that number came from.
    pub water_category: i64,
    pub water_category_source: String,
    /// Rows of source held at once, and how many bands that took.
    pub band_rows: i64,
    pub bands: usize,
    /// Peak bytes the two source bands occupied together.
    pub band_bytes: u64,
}

/// How the land-use frame sits against the topography frame.
///
/// The two 30-arc-second products are the same grid at the same spacing and
/// place their column 1 at different longitudes, so an index in one is an
/// index in the other plus a constant. Computing that constant is the whole
/// job; asserting it is zero is the defect this type exists to stop.
#[derive(Debug, Clone, Copy)]
pub struct FrameAlignment {
    pub dx_offset: i64,
    pub dy_offset: i64,
}

impl FrameAlignment {
    /// Derive the shift that carries a topography index into a land-use index.
    ///
    /// Refuses when the two are not the same grid, or when the origins differ
    /// by a fraction of a cell rather than a whole number of them: either way
    /// the pixels cannot be matched by an index shift, and matching them
    /// approximately would put land use and terrain in different places.
    pub fn derive(topo: &GeogDataset, landuse: &GeogDataset) -> MpasResult<Self> {
        let (t, l) = (&topo.index, &landuse.index);
        if t.dx != l.dx || t.dy != l.dy {
            return Err(MpasError::Refusal(format!(
                "the sub-grid orography pass needs terrain and land use on the \
                 same grid: {} is {}x{} degrees and {} is {}x{} degrees. \
                 Resampling one to the other is not something this pass may \
                 invent -- the statistics are counts of source pixels, so a \
                 resample changes every one of them. Remedy: stage matched \
                 30-arc-second products.",
                topo.path.display(),
                t.dx,
                t.dy,
                landuse.path.display(),
                l.dx,
                l.dy
            )));
        }
        if topo.nx_global != landuse.nx_global || topo.ny_global != landuse.ny_global {
            return Err(MpasError::Refusal(format!(
                "terrain is {}x{} source pixels and land use is {}x{}; the \
                 orography box is cut in index space and cannot span two \
                 different grids",
                topo.nx_global, topo.ny_global, landuse.nx_global, landuse.ny_global
            )));
        }
        // The STEP is taken from the tile inventory, not from the header.
        //
        // A WPS_GEOG index spells its spacing to a fixed number of digits, and
        // the 30-arc-second products spell 1/120 as `0.00833333`. That is 4e-7
        // relative, harmless per pixel and 0.009 of a pixel once it is
        // multiplied by the 21,600-column offset between two archives. Reading
        // the step off the global width instead makes the arithmetic exact for
        // any product that spans the globe, which every one of these does.
        let step_x = if topo.wraps_x {
            360.0 / topo.nx_global as f64
        } else {
            t.dx.abs()
        };
        let step_y = 180.0 / topo.ny_global as f64;
        let shift = |a_known_coord: f64,
                     a_known_idx: f64,
                     b_known_coord: f64,
                     b_known_idx: f64,
                     step: f64,
                     period: Option<f64>,
                     axis: &str|
         -> MpasResult<i64> {
            let mut delta = a_known_coord - b_known_coord;
            if let Some(p) = period {
                delta = delta.rem_euclid(p);
            }
            // `a` is the terrain frame, `b` the land use: the shift carries a
            // terrain index into a land-use one, so the reference indices
            // enter as (b - a) and the coordinates as (a - b).
            let cells = delta / step + (b_known_idx - a_known_idx);
            let rounded = cells.round();
            // A twentieth of a pixel: 400 m at 30 arc seconds. Tight enough
            // that a half-pixel or whole-pixel misregistration is refused,
            // loose enough that a truncated header digit is not.
            if (cells - rounded).abs() > 0.05 {
                return Err(MpasError::Refusal(format!(
                    "the terrain and land-use origins are {cells} source cells \
                     apart along {axis}, which is not a whole number of cells. \
                     Their pixels do not line up, so every orography statistic \
                     would mix a terrain sample with the land use of a \
                     different place."
                )));
            }
            Ok(rounded as i64)
        };
        let dx_offset = shift(
            t.known_lon,
            t.known_x,
            l.known_lon,
            l.known_x,
            step_x,
            Some(360.0),
            "longitude",
        )?;
        let dy_offset = shift(
            t.known_lat,
            t.known_y,
            l.known_lat,
            l.known_y,
            step_y,
            None,
            "latitude",
        )?;
        Ok(Self { dx_offset, dy_offset })
    }
}

/// Everything about the source grid the box arithmetic needs.
#[derive(Debug, Clone, Copy)]
struct SourceFrame {
    nx: i64,
    ny: i64,
    /// Source points per degree of longitude and of latitude.
    pts_per_degree_x: f64,
    pts_per_degree_y: f64,
}

impl SourceFrame {
    fn of(ds: &GeogDataset) -> MpasResult<Self> {
        let (dx, dy) = (ds.index.dx.abs(), ds.index.dy.abs());
        if !(dx > 0.0 && dy > 0.0) {
            return Err(MpasError::Refusal(format!(
                "{} declares a zero grid spacing; the orography box has no size",
                ds.path.display()
            )));
        }
        Ok(Self {
            nx: ds.nx_global,
            ny: ds.ny_global,
            pts_per_degree_x: 1.0 / dx,
            pts_per_degree_y: 1.0 / dy,
        })
    }

    /// The box MPAS extracts for a cell at `lat` with mean edge length `dx_m`.
    ///
    /// Integer `ceiling`, exactly as the Fortran writes it: the box is the
    /// smallest one that covers the cell, and rounding it differently changes
    /// every statistic by changing the sample.
    fn box_size(&self, lat: f64, dx_m: f64) -> (i64, i64) {
        let deg = (180.0 * dx_m) / (std::f64::consts::PI * RE);
        let nx = if lat.cos() > (2.0 * deg * self.pts_per_degree_x) / self.nx as f64 {
            (deg * self.pts_per_degree_x / lat.cos()).ceil() as i64
        } else {
            self.nx / 2
        };
        let ny = (deg * self.pts_per_degree_y).ceil() as i64;
        (nx.max(1), ny.max(1))
    }

    /// MPAS's index wrap: reflect across a pole, then wrap in longitude.
    ///
    /// The half-globe shift on reflection is `nx/2`, which is what the
    /// Fortran's `TOPO_Y` happens to equal on a 43200 x 21600 grid. Spelled as
    /// the thing it means so a differently shaped archive is not silently
    /// shifted by the wrong amount.
    #[inline]
    fn wrap(&self, mut ii: i64, mut jj: i64) -> (i64, i64) {
        if jj <= 0 {
            jj = -jj + 1;
            ii += self.nx / 2;
        }
        if jj > self.ny {
            jj = self.ny - (jj - self.ny - 1);
            ii += self.nx / 2;
        }
        ii = (ii - 1).rem_euclid(self.nx) + 1;
        (ii, jj)
    }

    /// The reflected row span a box at `jc` with height `ny_box` reads.
    ///
    /// The `+ 1` is the Fortran's, not a correction of it: the gather runs
    /// `j = 1..=ny` and reads row `j - ny/2 + jc`, so the first row is
    /// `jc - ny/2 + 1` and the last is `jc - ny/2 + ny`. Spelling the span
    /// one row lower -- which the retired port did -- makes the band miss the
    /// box's top row, and a cell served by a band that stops one row short
    /// reads zero terrain there. Its own determinism gate caught it: the
    /// answer moved with the band grant.
    fn reflected_rows(&self, jc: i64, ny_box: i64) -> (i64, i64) {
        let j0 = jc - ny_box / 2 + 1;
        let j1 = j0 + ny_box - 1;
        let (mut lo, mut hi) = (i64::MAX, i64::MIN);
        let mut note = |a: i64, b: i64| {
            if b >= a {
                lo = lo.min(a);
                hi = hi.max(b);
            }
        };
        note(j0.max(1), j1.min(self.ny));
        if j0 <= 0 {
            note(1, 1 - j0);
        }
        if j1 > self.ny {
            note(2 * self.ny + 1 - j1, self.ny);
        }
        if lo > hi {
            // A box entirely off the grid cannot happen for a cell on the
            // sphere, but a clamped answer beats an inverted range.
            return (1, 1);
        }
        (lo, hi)
    }
}

/// A full-width horizontal slab of both sources, in the TOPOGRAPHY frame.
struct Band {
    y_lo: i64,
    y_hi: i64,
    nx: i64,
    topo: Vec<f32>,
    landuse: Vec<i32>,
}

impl Band {
    fn rows(&self) -> i64 {
        self.y_hi - self.y_lo + 1
    }

    /// Value at wrapped global 1-based `(ii, jj)` of the topography frame.
    #[inline]
    fn at(&self, ii: i64, jj: i64) -> Option<(f32, i32)> {
        if jj < self.y_lo || jj > self.y_hi {
            return None;
        }
        let k = ((jj - self.y_lo) * self.nx + (ii - 1)) as usize;
        Some((self.topo[k], self.landuse[k]))
    }
}

/// One cell's box, already gathered.
struct Boxed {
    nx: i64,
    ny: i64,
    topo: Vec<f64>,
    landuse: Vec<i32>,
    mean: f64,
}

impl Boxed {
    #[inline]
    fn h(&self, i: i64, j: i64) -> f64 {
        self.topo[((j - 1) * self.nx + (i - 1)) as usize]
    }
}

fn get_var(b: &Boxed) -> f64 {
    let mut s2 = 0.0;
    for v in &b.topo {
        s2 += (v - b.mean).powi(2);
    }
    (s2 / (b.nx * b.ny) as f64).sqrt()
}

fn get_con(b: &Boxed, water: i32) -> f64 {
    let n = (b.nx * b.ny) as f64;
    let (mut xland, mut mean_land, mut mean_water) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..b.topo.len() {
        if b.landuse[k] != water {
            xland += 1.0;
            mean_land += b.topo[k];
        } else {
            mean_water += b.topo[k];
        }
    }
    if xland > 0.0 {
        mean_land /= xland;
    }
    if xland < n {
        mean_water /= n - xland;
    }
    let xland = xland / n;
    let oro = if xland >= 0.5 { mean_land } else { mean_water };
    let (mut s2, mut s4) = (0.0f64, 0.0f64);
    for &v in &b.topo {
        s2 += (v - b.mean).powi(2);
        s4 += (v - oro).powi(4);
    }
    let var = s2 / n;
    if var.sqrt() < 1.0 || xland < 0.5 {
        0.0
    } else {
        s4 / (var * var * n)
    }
}

/// `(nu - nd) / (nu + nd)` over a split MPAS defines by index arithmetic.
fn asymmetry(b: &Boxed, split: impl Fn(i64, i64) -> bool) -> f64 {
    let (mut nu, mut nd) = (0i64, 0i64);
    for j in 1..=b.ny {
        for i in 1..=b.nx {
            if b.h(i, j) > b.mean {
                if split(i, j) {
                    nu += 1;
                } else {
                    nd += 1;
                }
            }
        }
    }
    if nu + nd > 0 {
        (nu - nd) as f64 / (nu + nd) as f64
    } else {
        0.0
    }
}

/// Fraction of a sub-box above the critical height.
fn fraction_above(b: &Boxed, hc: f64, include: impl Fn(i64, i64) -> bool) -> f64 {
    let (mut nw, mut nt) = (0i64, 0i64);
    for j in 1..=b.ny {
        for i in 1..=b.nx {
            if include(i, j) {
                if b.h(i, j) > hc {
                    nw += 1;
                }
                nt += 1;
            }
        }
    }
    if nt > 0 {
        nw as f64 / nt as f64
    } else {
        0.0
    }
}

/// Dominant land mask over the box: 1 where most of it is not water.
fn dominant_landmask(b: &Boxed, water: i32) -> u8 {
    let land = b.landuse.iter().filter(|&&c| c != water).count();
    u8::from(land * 2 >= b.landuse.len())
}

/// What one cell's box costs, and what a band of `rows` costs.
///
/// Both are what the admission gate is shown, so they are computed from the
/// same numbers the pass will actually allocate rather than estimated.
pub fn band_bytes(topo: &GeogDataset, rows: i64) -> u64 {
    let per_row = topo.band_row_bytes(std::mem::size_of::<f32>())
        + topo.band_row_bytes(std::mem::size_of::<i32>());
    per_row.saturating_mul(rows.max(0) as u64)
}

/// The rows a band must hold to serve every cell, before any budget applies.
///
/// A cell whose box is taller than the band cannot be served at all, so this
/// is the floor the admission gate has to clear, not a preference.
pub fn minimum_band_rows(
    topo: &GeogDataset,
    cell_lat: &[f64],
    dc_m: &[f64],
) -> MpasResult<i64> {
    let frame = SourceFrame::of(topo)?;
    let mut most = 1i64;
    for (c, &lat) in cell_lat.iter().enumerate() {
        let (_, ny) = frame.box_size(lat, dc_m[c]);
        let jc = jcentre(&frame, topo, lat);
        let (lo, hi) = frame.reflected_rows(jc, ny);
        most = most.max(hi - lo + 1);
    }
    Ok(most)
}

/// The source row a cell centre falls on, in the topography frame.
fn jcentre(frame: &SourceFrame, topo: &GeogDataset, lat: f64) -> i64 {
    let (_, y) = topo.latlon_rad_to_source_xy(lat, 0.0);
    let _ = frame;
    y.round() as i64
}

/// Compute every GWD field for a mesh.
///
/// `cell_lat`/`cell_lon` are radians; `dc_m` is each cell's mean `dcEdge` in
/// METRES on the earth radius. `band_rows` is how many source rows may be
/// resident at once -- the admission gate's grant, never a guess -- and the
/// ANSWER DOES NOT DEPEND ON IT: every cell's box is fully covered by the band
/// it is served from, at every setting, which is what makes the build
/// byte-identical under a different memory limit.
#[allow(clippy::too_many_arguments)]
pub fn compute(
    topo: &GeogDataset,
    landuse: &GeogDataset,
    cell_lat: &[f64],
    cell_lon: &[f64],
    dc_m: &[f64],
    n_edges_on_cell: &[usize],
    cells_on_cell: &[usize],
    max_edges: usize,
    band_rows: i64,
    progress: &dyn Fn(&str),
) -> MpasResult<GwdFields> {
    let n = cell_lat.len();
    let frame = SourceFrame::of(topo)?;
    let align = FrameAlignment::derive(topo, landuse)?;
    let water = landuse.index.iswater.unwrap_or(16) as i32;
    // The DATASET name, never its path on this box. This string reaches the
    // stamped bytes of a file a registry pins by sha256, and the same archive
    // named by one absolute path in its two Windows spellings (backslash and
    // forward slash) produced two different statics -- measured, 2,681,384 B
    // against 2,681,380 B, from byte-identical inputs.
    let mut water_source = match landuse.index.iswater {
        Some(_) => format!(
            "{} index iswater",
            landuse
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "landuse".to_string())
        ),
        None => "MPAS default WATER=16 (dataset index declares none)".to_string(),
    };
    let lake = landuse.index.islake.filter(|&c| c > 0).map(|c| c as i32);
    if let Some(lake) = lake {
        water_source.push_str(&format!("; index islake={lake} treated as water"));
    }

    // Box geometry per cell, in the TOPOGRAPHY dataset's own index frame.
    let mut geometry: Vec<(i64, i64, i64, i64)> = Vec::with_capacity(n); // ic, jc, nx, ny
    for c in 0..n {
        let (nx, ny) = frame.box_size(cell_lat[c], dc_m[c]);
        let (x, y) = topo.latlon_rad_to_source_xy(cell_lat[c], cell_lon[c]);
        let ic = (x.round() as i64 - 1).rem_euclid(frame.nx) + 1;
        let jc = y.round() as i64;
        geometry.push((ic, jc, nx, ny));
    }

    let needed = geometry
        .iter()
        .map(|&(_, jc, _, ny)| {
            let (lo, hi) = frame.reflected_rows(jc, ny);
            hi - lo + 1
        })
        .max()
        .unwrap_or(1);
    if band_rows < needed {
        return Err(MpasError::Refusal(format!(
            "the sub-grid orography pass was granted {band_rows} source rows but \
             the widest cell's box reads {needed} of them. Serving it from a \
             shorter band would hand that cell zero terrain over the rows the \
             band does not reach, and WHICH rows would depend on the memory \
             limit -- so the same mesh would build two different drag fields on \
             two boxes. Remedy: raise the host-memory limit, or build a coarser \
             mesh whose cells cut a smaller box."
        )));
    }

    // Process in latitude bands so the source is streamed, not resident.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&c| {
        let (_, jc, _, ny) = geometry[c];
        (frame.reflected_rows(jc, ny).0, c)
    });

    let mut var2d = vec![0.0f64; n];
    let mut con = vec![0.0f64; n];
    let mut oa: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0f64; n]);
    let mut ol: [Vec<f64>; 4] = std::array::from_fn(|_| vec![0.0f64; n]);
    let mut hlanduse = vec![0u8; n];

    let mut pos = 0usize;
    let mut bands = 0usize;
    while pos < n {
        // Take as many cells as one band of rows can serve, always covering
        // each cell's WHOLE reflected span.
        let first = order[pos];
        let (_, jc0, _, ny0) = geometry[first];
        let (mut y_lo, mut y_hi) = frame.reflected_rows(jc0, ny0);
        let mut end = pos + 1;
        while end < n {
            let c = order[end];
            let (_, jc, _, ny) = geometry[c];
            let (clo, chi) = frame.reflected_rows(jc, ny);
            let lo = clo.min(y_lo);
            let hi = chi.max(y_hi);
            if hi - lo + 1 > band_rows {
                break;
            }
            y_lo = lo;
            y_hi = hi;
            end += 1;
        }

        let missing = topo.index.missing_value;
        let scale = topo.index.scale_factor;
        let topo_band = topo.read_band_with(y_lo, y_hi, 0, 0.0f32, |raw| {
            if missing == Some(raw) {
                0.0
            } else {
                (raw as f64 * scale) as f32
            }
        })?;
        // The land-use band is read in the LAND-USE frame's own rows, then
        // indexed through the alignment; a row offset of zero is the common
        // case and a nonzero one is exactly what `FrameAlignment` exists for.
        let lu_band = landuse.read_band_with(
            y_lo + align.dy_offset,
            y_hi + align.dy_offset,
            0,
            -1i32,
            |raw| if lake == Some(raw as i32) { water } else { raw as i32 },
        )?;
        // Roll the land-use band into the topography frame's column order so
        // the box gather indexes one array with one index.
        let nx = frame.nx;
        let mut landuse_band = checked_vec::<i32>(
            topo_band.len(),
            "gwd-band",
            "land-use band in the terrain frame",
        )?;
        // Row by row on the pool: each row is its own slice (same bytes).
        landuse_band
            .par_chunks_mut(nx as usize)
            .enumerate()
            .for_each(|(r, dst)| {
                let src = r * nx as usize;
                for i in 0..nx {
                    let li = (i + align.dx_offset).rem_euclid(nx) as usize;
                    dst[i as usize] = lu_band[src + li];
                }
            });
        drop(lu_band);
        let band = Band {
            y_lo,
            y_hi,
            nx,
            topo: topo_band,
            landuse: landuse_band,
        };
        bands += 1;
        progress(&format!(
            "GWDBAND\t{}\t{}\t{}\t{}",
            band.y_lo,
            band.y_hi,
            band.rows(),
            end - pos
        ));

        let slice: Vec<usize> = order[pos..end].to_vec();
        // Every cell writes only its own slot and reads only the band, so the
        // parallel map is order-free by construction: the collect preserves
        // `slice` order and each result lands at the index it names.
        let results: Vec<(usize, f64, f64, [f64; 4], [f64; 4], u8)> = slice
            .par_iter()
            .map(|&c| {
                let (ic, jc, nx_box, ny_box) = geometry[c];
                let mut t = vec![0.0f64; (nx_box * ny_box) as usize];
                let mut l = vec![water; (nx_box * ny_box) as usize];
                let mut mean = 0.0f64;
                for j in 1..=ny_box {
                    for i in 1..=nx_box {
                        let (ii, jj) = frame.wrap(i - nx_box / 2 + ic, j - ny_box / 2 + jc);
                        let k = ((j - 1) * nx_box + (i - 1)) as usize;
                        if let Some((h, lu)) = band.at(ii, jj) {
                            t[k] = h as f64;
                            l[k] = lu;
                        }
                        mean += t[k];
                    }
                }
                mean /= (nx_box * ny_box) as f64;
                let b = Boxed { nx: nx_box, ny: ny_box, topo: t, landuse: l, mean };

                let v = get_var(&b);
                let cn = get_con(&b, water);
                let a = [
                    asymmetry(&b, |i, _| i <= b.nx / 2),
                    asymmetry(&b, |_, j| j <= b.ny / 2),
                    asymmetry(&b, |i, j| {
                        let ratio = b.ny as f64 / b.nx as f64;
                        ((i as f64 * ratio).round() as i64) < (b.ny - j)
                    }),
                    asymmetry(&b, |i, j| {
                        let ratio = b.ny as f64 / b.nx as f64;
                        ((i as f64 * ratio).round() as i64) < j
                    }),
                ];
                // Kim (1996) critical height, from this cell's own variance.
                let hc = 1116.2 - 0.878 * v;
                let o = [
                    fraction_above(&b, hc, |_, j| j >= b.ny / 4 && j <= 3 * b.ny / 4),
                    fraction_above(&b, hc, |i, _| i >= b.nx / 4 && i <= 3 * b.nx / 4),
                    fraction_above(&b, hc, |i, j| {
                        (j <= b.ny / 2 && i <= b.nx / 2) || (j > b.ny / 2 && i > b.nx / 2)
                    }),
                    fraction_above(&b, hc, |i, j| {
                        (j > b.ny / 2 && i <= b.nx / 2) || (j <= b.ny / 2 && i > b.nx / 2)
                    }),
                ];
                (c, v, cn, a, o, dominant_landmask(&b, water))
            })
            .collect();

        for (c, v, cn, a, o, hl) in results {
            var2d[c] = v;
            con[c] = cn;
            for k in 0..4 {
                oa[k][c] = a[k];
                ol[k][c] = o[k];
            }
            hlanduse[c] = hl;
        }
        pos = end;
    }

    // MPAS smooths the variance at a cell whose land/water class disagrees with
    // every one of its neighbours: an isolated island or lake would otherwise
    // carry a variance its surroundings contradict straight into the drag.
    let before = var2d.clone();
    let mut smoothed = 0usize;
    let mut smooth_skipped_boundary_cells = 0usize;
    'cells: for c in 0..n {
        let k = n_edges_on_cell[c];
        if k == 0 {
            continue;
        }
        let mut sum_landuse = 0u32;
        let mut sum_var = 0.0f64;
        for e in 0..k {
            let nb = cells_on_cell[c * max_edges + e];
            if nb >= n {
                // The regional branch: an out-of-range slot is the culler's
                // absent-neighbour sentinel on an outermost-ring cell. Native
                // v8.4.1 reads its garbage halo slot here, which is not a
                // rule; the DEFINED behaviour is that a cell missing part of
                // its ring cannot be judged "isolated", so the smooth leaves
                // it alone. Divergence from a whole-sphere answer is confined
                // to mask-7 cells and is recorded in the receipt.
                smooth_skipped_boundary_cells += 1;
                continue 'cells;
            }
            sum_landuse += hlanduse[nb] as u32;
            sum_var += before[nb];
        }
        let all_land = sum_landuse == k as u32 && hlanduse[c] == 0;
        let all_water = sum_landuse == 0 && hlanduse[c] == 1;
        if all_land || all_water {
            var2d[c] = sum_var / k as f64;
            smoothed += 1;
        }
    }

    Ok(GwdFields {
        var2d,
        con,
        oa,
        ol,
        elvmax: Vec::new(),
        hlanduse,
        smoothed_cells: smoothed,
        smooth_skipped_boundary_cells,
        water_category: water as i64,
        water_category_source: water_source,
        band_rows,
        bands,
        band_bytes: band_bytes(topo, band_rows),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // -- a synthetic pair, built from bytes this module owns -----------------

    const NX_TILES: i64 = 6;
    const NY_TILES: i64 = 3;
    const TILE: i64 = 30;
    /// 180 x 90 source cells at two degrees, covering the globe.
    const NX: i64 = NX_TILES * TILE;

    /// A field that is a function of PLACE, not of index.
    ///
    /// That is the whole point: two archives that place their column 1 at
    /// different longitudes must produce the same statistics from it, and they
    /// can only do that if the reader shifts one onto the other.
    fn place(lon_deg: f64, lat_deg: f64) -> (i64, i64) {
        (
            (lon_deg.rem_euclid(360.0) / 2.0).round() as i64,
            ((lat_deg + 89.0) / 2.0).round() as i64,
        )
    }

    fn height_at(lon_deg: f64, lat_deg: f64) -> i64 {
        let (cx, cy) = place(lon_deg, lat_deg);
        (cx * 37 + cy * 101).rem_euclid(2000) + 1
    }

    /// Steps by THREE per column, which is coprime with the eight categories.
    ///
    /// An earlier spelling stepped by six, so the fixture only ever produced
    /// the four even categories and never the water code -- and `get_con`,
    /// which splits land from water, became a function of terrain alone. The
    /// test that was meant to prove land use reaches the answer proved
    /// nothing, and said so by passing when it should not have.
    fn category_at(lon_deg: f64, lat_deg: f64) -> i64 {
        let (cx, cy) = place(lon_deg, lat_deg);
        (cx * 3 + cy * 5).rem_euclid(8) + 1
    }

    /// A global two-degree WPS_GEOG tree whose column 1 sits at `known_lon`.
    fn fixture(label: &str, kind: &str, known_lon: f64, wordsize: usize) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rw-mpas-gwd-{}-{}",
            std::process::id(),
            label
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("fixture directory");
        let mut index = format!(
            "type={kind}\nprojection=regular_ll\ndx=2.0\ndy=2.0\n\
             known_x=1.0\nknown_y=1.0\nknown_lat=-89.0\nknown_lon={known_lon}\n\
             wordsize={wordsize}\ntile_x={TILE}\ntile_y={TILE}\ntile_z=1\n\
             tile_bdr=0\nendian=big\nsigned=no\nscale_factor=1.0\n"
        );
        if kind == "categorical" {
            index.push_str("category_min=1\ncategory_max=8\niswater=3\nmminlu=FIXTURE\n");
        }
        std::fs::write(dir.join("index"), &index).expect("fixture index");
        for ty in 0..NY_TILES {
            for tx in 0..NX_TILES {
                let xs = tx * TILE + 1;
                let ys = ty * TILE + 1;
                let mut bytes = Vec::new();
                for j in 0..TILE {
                    for i in 0..TILE {
                        let lon = known_lon + (xs + i - 1) as f64 * 2.0;
                        let lat = -89.0 + (ys + j - 1) as f64 * 2.0;
                        let v = if kind == "categorical" {
                            category_at(lon, lat)
                        } else {
                            height_at(lon, lat)
                        };
                        match wordsize {
                            1 => bytes.push(v as u8),
                            _ => bytes.extend_from_slice(&(v as u16).to_be_bytes()),
                        }
                    }
                }
                std::fs::write(
                    dir.join(format!(
                        "{:05}-{:05}.{:05}-{:05}",
                        xs,
                        xs + TILE - 1,
                        ys,
                        ys + TILE - 1
                    )),
                    &bytes,
                )
                .expect("fixture tile");
            }
        }
        dir
    }

    /// A handful of cells, including one close enough to a pole that its box
    /// reflects across it.
    fn cells() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>, usize) {
        let lats_deg = [0.0f64, 33.0, -47.5, 71.0, 88.0, -88.5];
        let lons_deg = [0.0f64, -101.0, 140.0, 12.0, -170.0, 60.0];
        let lat: Vec<f64> = lats_deg.iter().map(|d| d.to_radians()).collect();
        let lon: Vec<f64> = lons_deg.iter().map(|d| d.to_radians()).collect();
        let n = lat.len();
        let dc = vec![2_000_000.0f64; n];
        let max_edges = 3usize;
        let n_edges_on_cell = vec![max_edges; n];
        let cells_on_cell: Vec<usize> = (0..n * max_edges).map(|k| (k + 1) % n).collect();
        (lat, lon, dc, n_edges_on_cell, cells_on_cell, max_edges)
    }

    fn run(topo: &GeogDataset, lu: &GeogDataset, band_rows: i64) -> GwdFields {
        let (lat, lon, dc, neoc, coc, me) = cells();
        compute(topo, lu, &lat, &lon, &dc, &neoc, &coc, me, band_rows, &|_| {})
            .expect("the drag band computes")
    }

    /// The with-lakes land use names its lakes `islake`; they are water here
    /// exactly as the water class is. Relabelling every water pixel to a
    /// declared lake class must leave all four statistics unchanged.
    #[test]
    fn declared_lake_pixels_count_as_water() {
        let topo = GeogDataset::open(&fixture("lake-topo", "continuous", -179.0, 2)).unwrap();
        let lu_path = fixture("lake-lu", "categorical", -179.0, 1);
        let water = run(&topo, &GeogDataset::open(&lu_path).unwrap(), 90);
        for entry in std::fs::read_dir(&lu_path).unwrap() {
            let path = entry.unwrap().path();
            if path.file_name().unwrap() == "index" { continue; }
            let mut bytes = std::fs::read(&path).unwrap();
            for value in &mut bytes { if *value == 3 { *value = 9; } }
            std::fs::write(path, bytes).unwrap();
        }
        let index = std::fs::read_to_string(lu_path.join("index")).unwrap()
            .replace("category_max=8", "category_max=9");
        std::fs::write(lu_path.join("index"), &index).unwrap();
        let undeclared = run(&topo, &GeogDataset::open(&lu_path).unwrap(), 90);
        assert_ne!(water.con, undeclared.con, "the fixture must exercise the land/water split");
        std::fs::write(lu_path.join("index"), format!("{index}islake=9\n")).unwrap();
        let lake = run(&topo, &GeogDataset::open(&lu_path).unwrap(), 90);
        assert_eq!(water.con, lake.con);
        assert_eq!(water.var2d, lake.var2d);
        assert_eq!(water.oa, lake.oa);
        assert_eq!(water.ol, lake.ol);
        assert_eq!(water.hlanduse, lake.hlanduse);
        assert!(lake.water_category_source.ends_with("; index islake=9 treated as water"));
        assert!(!water.water_category_source.contains("islake"));
    }

    /// THE DETERMINISM GATE FOR THIS PASS. The band is a memory budget, and a
    /// memory budget may not be visible in the answer.
    ///
    /// Cells are grouped into bands by how many source rows the group needs,
    /// so a smaller grant makes more, narrower bands. If a cell's box were
    /// ever served from a band that did not cover all of it, the pixels off
    /// the end would read zero -- and WHICH pixels depends on the grouping,
    /// which depends on the grant. The same mesh would then build two
    /// different drag fields on two boxes with different RAM.
    #[test]
    fn the_band_grant_changes_the_cost_and_not_the_answer() {
        let topo = GeogDataset::open(&fixture("inv-topo", "continuous", -179.0, 2))
            .expect("topo opens");
        let lu = GeogDataset::open(&fixture("inv-lu", "categorical", -179.0, 1))
            .expect("landuse opens");
        let (lat, _, dc, _, _, _) = cells();
        let floor = minimum_band_rows(&topo, &lat, &dc).expect("floor");
        let narrow = run(&topo, &lu, floor);
        let wide = run(&topo, &lu, 90);
        assert!(
            narrow.bands > wide.bands,
            "the two grants produced the same banding ({} vs {}), so this \
             compares nothing",
            narrow.bands,
            wide.bands
        );
        assert_eq!(narrow.var2d, wide.var2d, "var2d moved with the band grant");
        assert_eq!(narrow.con, wide.con, "con moved with the band grant");
        for k in 0..4 {
            assert_eq!(narrow.oa[k], wide.oa[k], "oa{} moved", k + 1);
            assert_eq!(narrow.ol[k], wide.ol[k], "ol{} moved", k + 1);
        }
        assert!(
            narrow.var2d.iter().any(|&v| v > 0.0),
            "the fixture produced a flat field, so the comparison is vacuous"
        );
    }

    /// A band too short for the widest box is REFUSED, not silently truncated.
    #[test]
    fn a_band_that_cannot_hold_a_box_is_refused_with_both_numbers() {
        let topo = GeogDataset::open(&fixture("short-topo", "continuous", -179.0, 2))
            .expect("topo opens");
        let lu = GeogDataset::open(&fixture("short-lu", "categorical", -179.0, 1))
            .expect("landuse opens");
        let (lat, lon, dc, neoc, coc, me) = cells();
        let err = compute(&topo, &lu, &lat, &lon, &dc, &neoc, &coc, me, 1, &|_| {})
            .expect_err("a one-row band cannot serve a nine-row box");
        let text = err.to_string();
        assert!(text.contains("granted 1 source rows"), "{text}");
        assert!(text.contains("depend on the memory limit"), "{text}");
    }

    /// THE FRAME GATE. Two archives of the same field that disagree about
    /// where their column 1 sits must produce the same statistics.
    ///
    /// This is the defect the port was carrying, measured rather than
    /// described: the staged `topo_gmted2010_30s` declares
    /// `known_lon=0.004166667` and the staged land use declares
    /// `-179.99583`, exactly half a globe apart. Reading both as if column 1
    /// were 180 W sampled terrain from the antipode of every cell, with every
    /// value finite and every field in range.
    #[test]
    fn a_land_use_archive_with_a_shifted_origin_gives_the_same_answer() {
        let topo = GeogDataset::open(&fixture("frame-topo", "continuous", -179.0, 2))
            .expect("topo opens");
        let aligned = GeogDataset::open(&fixture("frame-lu-aligned", "categorical", -179.0, 1))
            .expect("aligned landuse opens");
        // The same field, described by an archive whose column 1 is half a
        // globe away: 90 of its 180 columns.
        let shifted = GeogDataset::open(&fixture("frame-lu-shifted", "categorical", 1.0, 1))
            .expect("shifted landuse opens");

        let align = FrameAlignment::derive(&topo, &shifted).expect("alignment derives");
        assert_eq!(align.dx_offset, NX / 2, "the shift is not half the globe");
        assert_eq!(align.dy_offset, 0);
        assert_eq!(
            FrameAlignment::derive(&topo, &aligned)
                .expect("aligned")
                .dx_offset,
            0
        );

        let a = run(&topo, &aligned, 90);
        let b = run(&topo, &shifted, 90);
        assert_eq!(a.con, b.con, "convexity followed the archive's index, not the place");
        assert_eq!(a.var2d, b.var2d, "variance followed the archive's index");
        assert_eq!(a.hlanduse, b.hlanduse, "the land mask followed the index");
        assert!(
            a.con.iter().any(|&v| v != 0.0),
            "the fixture produced no convexity at all, so this is vacuous"
        );
    }

    /// And the same comparison catches a frame that really IS misregistered:
    /// a land-use archive shifted by a whole column of pixels is a different
    /// answer, which is what makes the test above evidence.
    #[test]
    fn a_genuinely_misregistered_archive_does_change_the_answer() {
        let topo = GeogDataset::open(&fixture("mis-topo", "continuous", -179.0, 2))
            .expect("topo opens");
        let accurate = GeogDataset::open(&fixture("mis-lu", "categorical", -179.0, 1))
            .expect("landuse opens");
        // The same bytes, relabelled as if they started 90 columns later.
        // Nothing shifts them back, so every pixel is read from the wrong
        // place -- the state the port was in.
        let lying = {
            let dir = fixture("mis-lu-lying", "categorical", -179.0, 1);
            let index = std::fs::read_to_string(dir.join("index")).expect("index");
            std::fs::write(
                dir.join("index"),
                index.replace("known_lon=-179", "known_lon=1"),
            )
            .expect("relabel");
            GeogDataset::open(&dir).expect("relabelled landuse opens")
        };
        let a = run(&topo, &accurate, 90);
        let b = run(&topo, &lying, 90);
        assert_ne!(
            a.con, b.con,
            "a half-globe misregistration left convexity unchanged, so the \
             frame gate above proves nothing"
        );
    }

    /// The 30-arc-second frame the staged archive carries, so the box
    /// arithmetic is exercised against the numbers it will actually see.
    fn thirty_second_frame() -> SourceFrame {
        SourceFrame {
            nx: 43_200,
            ny: 21_600,
            pts_per_degree_x: 120.0,
            pts_per_degree_y: 120.0,
        }
    }

    /// The box must grow towards the pole, where a degree of longitude is
    /// shorter, or high-latitude cells sample a box narrower than themselves.
    #[test]
    fn the_box_widens_towards_the_pole() {
        let f = thirty_second_frame();
        let dx = 25_000.0;
        let (nx_eq, ny_eq) = f.box_size(0.0, dx);
        let (nx_60, ny_60) = f.box_size(60f64.to_radians(), dx);
        assert!(nx_60 > nx_eq, "{nx_60} is not wider than {nx_eq}");
        assert_eq!(ny_eq, ny_60, "meridional size does not depend on latitude");
        // 25 km at 120 points per degree is about 27 points.
        assert_eq!(ny_eq, 27);
    }

    /// At high enough latitude the formula saturates at half the globe rather
    /// than running away.
    #[test]
    fn the_box_saturates_at_half_the_globe() {
        let f = thirty_second_frame();
        let (nx, _) = f.box_size(89.999f64.to_radians(), 25_000.0);
        assert_eq!(nx, f.nx / 2);
    }

    #[test]
    fn a_pole_crossing_index_reflects_and_shifts_half_a_globe() {
        let f = thirty_second_frame();
        assert_eq!(f.wrap(100, 0), (100 + f.nx / 2, 1));
        assert_eq!(f.wrap(100, f.ny + 1), (100 + f.nx / 2, f.ny));
        assert_eq!(f.wrap(f.nx + 5, 10), (5, 10));
        assert_eq!(f.wrap(-3, 10), (f.nx - 3, 10));
    }

    /// The band a polar box needs covers the rows the REFLECTION reads, not
    /// merely the rows the unreflected span names.
    ///
    /// Without this the reflected pixels read whatever the band happened to
    /// hold -- zero, in practice -- and which pixels those were depended on how
    /// cells grouped into bands, which depends on the memory budget. That is
    /// the difference between a build that is reproducible and one that is
    /// reproducible on one machine.
    #[test]
    fn a_polar_box_widens_the_band_to_the_rows_it_reflects_onto() {
        let f = thirty_second_frame();
        // A cell on row 11 with a 100-row box reads rows -38..61: the 39 rows
        // at or below the pole reflect onto 1..39, and the rest stay on
        // 1..61. A band that stopped at 61 without the reflection would still
        // have to reach 1, so the span is [1, 61] either way -- what the
        // reflection decides is that nothing in it is read as zero.
        assert_eq!(f.reflected_rows(11, 100), (1, 61));
        // At the north pole the reflection is what pulls the low end down:
        // rows 21541..21640 reflect their top 40 onto 21561..21600, which the
        // unreflected span already covers, so the span is [21541, 21600].
        let (lo, hi) = f.reflected_rows(f.ny - 10, 100);
        assert_eq!((lo, hi), (f.ny - 59, f.ny));
        // And a box wholly inside the grid is exactly its own height.
        let (lo, hi) = f.reflected_rows(10_000, 100);
        assert_eq!((lo, hi), (9_951, 10_050));
        assert_eq!(hi - lo + 1, 100);
    }

    /// Flat terrain has no variance and no asymmetry, whatever the box.
    #[test]
    fn flat_terrain_produces_zero_variance_and_zero_asymmetry() {
        let b = Boxed {
            nx: 8,
            ny: 6,
            topo: vec![300.0; 48],
            landuse: vec![1; 48],
            mean: 300.0,
        };
        assert_eq!(get_var(&b), 0.0);
        assert_eq!(asymmetry(&b, |i, _| i <= 4), 0.0);
        // Variance below 1 m forces convexity to zero by MPAS's own guard.
        assert_eq!(get_con(&b, 16), 0.0);
    }

    /// A ridge in the western half must read as positive asymmetry, which is
    /// the sign convention the drag scheme's upstream/downstream split needs.
    #[test]
    fn a_western_ridge_reads_as_positive_asymmetry() {
        let (nx, ny) = (8i64, 4i64);
        let mut topo = vec![0.0f64; (nx * ny) as usize];
        for j in 0..ny {
            for i in 0..nx / 2 {
                topo[(j * nx + i) as usize] = 1000.0;
            }
        }
        let mean = topo.iter().sum::<f64>() / topo.len() as f64;
        let b = Boxed { nx, ny, topo, landuse: vec![1; (nx * ny) as usize], mean };
        assert_eq!(asymmetry(&b, |i, _| i <= b.nx / 2), 1.0);
    }
}

