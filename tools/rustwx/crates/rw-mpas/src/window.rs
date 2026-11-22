//! Render windows: the structured target grids an unstructured frame is
//! gathered onto, and the WRF projection attributes that describe them.
//!
//! Two kinds, because the renderer treats them differently. A Lambert window
//! emulates a WRF domain and gets the renderer's native regional map
//! furniture; a regular latitude/longitude window is the accurate shape for a
//! global overview. The Lambert setup and the two transforms below are WRF
//! `module_llxy.F` (`set_lc`, `ijll_lc` and `llij_lc`) transcribed, in f64,
//! so the emitted `XLAT`/`XLONG` agree with the projection attributes a
//! reader recomputes from.
//!
//! Two windows are FIXED boxes -- `focus` and `global` -- and their geometry
//! is frozen: their `weights_sha256` values are recorded in evidence already
//! written, and moving a corner by a metre changes every one of those digests
//! without changing a filename. The third, [`Window::mesh_focus`], is DERIVED
//! from the mesh being converted, which is what lets a mesh refined anywhere
//! render its own fine region without a fourth box being added here.

use std::f64::consts::PI;

// Every transcendental on the window path goes through the vendored `libm`
// (pure IEEE arithmetic, `force-soft-floats`), never `f64::cos`/`powf`/`atan`.
// Those call the platform C library, and the Windows UCRT returns different
// last bits for the same argument depending on its build and on the CPU it
// dispatches for: the same rw-mpas test binary gave a different focus-window
// digest on an AMD Ryzen 9 9950X3D (UCRT 10.0.26100) and on an Intel Xeon
// 8488C (UCRT 10.0.20348), first in `pow`, then `atan2` and `atan`. With
// `libm` the coordinates are the same bits on every x86-64 CPU and OS.
use libm::{acos, asin, atan, atan2, cos, log10, pow, sin, tan};

use crate::error::{MpasError, MpasResult};

/// How far past a pole a cell latitude may sit and still be read as the pole.
///
/// MPAS meshes store `latCell` as float32, and the float32 nearest pi/2 is
/// 1.5707963705062866 rad, 4.37e-8 rad (2.5e-6 degree) PAST the pole: every
/// global mesh with a cell on a pole (x1.40962 cells 2 and 8) carries one.
/// That is rounding, not a position off the sphere, so a latitude within this
/// tolerance of +-90 is clamped to +-90. 1e-4 degree is about 11 m on the
/// ground, forty times the float32 overshoot and far under any mesh spacing;
/// a latitude further out than that is a broken file and is still refused.
pub const POLE_ROUNDING_TOLERANCE_DEGREES: f64 = 1.0e-4;

/// The cell latitudes with any pole-rounding overshoot clamped to +-90, or a
/// refusal naming the first cell that sits clearly outside [-90, 90]. Borrows
/// the input untouched when no cell needs clamping. Non-finite values pass
/// through for the caller's own by-name refusal.
fn pole_clamped_latitudes<'a>(
    latitude_degrees: &'a [f64],
    role: &str,
) -> MpasResult<std::borrow::Cow<'a, [f64]>> {
    let mut clamped: Option<Vec<f64>> = None;
    for (i, &lat) in latitude_degrees.iter().enumerate() {
        // NaN and infinity are left for the caller's "not finite" refusal; a
        // NaN would otherwise pass both comparisons and be clamped to a pole.
        if !lat.is_finite() || lat.abs() <= 90.0 {
            continue;
        }
        if lat.abs() > 90.0 + POLE_ROUNDING_TOLERANCE_DEGREES {
            return Err(MpasError::Refusal(format!(
                "{role} {i} sits at latitude {lat}, outside [-90, 90] by more than the \
                 {POLE_ROUNDING_TOLERANCE_DEGREES} degree pole-rounding tolerance"
            )));
        }
        clamped.get_or_insert_with(|| latitude_degrees.to_vec())[i] = 90.0_f64.copysign(lat);
    }
    Ok(match clamped {
        Some(v) => std::borrow::Cow::Owned(v),
        None => std::borrow::Cow::Borrowed(latitude_degrees),
    })
}

#[cfg(test)]
mod pole_rounding_tests {
    use super::{pole_clamped_latitudes, Window, POLE_ROUNDING_TOLERANCE_DEGREES};

    /// latCell of x1.40962 cells 2 and 8 as the file stores them (float32 +-pi/2).
    fn float32_pole_degrees() -> f64 {
        (std::f32::consts::FRAC_PI_2 as f64).to_degrees()
    }

    #[test]
    fn float32_pole_latitude_is_clamped_to_the_pole() {
        let north = float32_pole_degrees();
        assert!(north > 90.0 && north < 90.0 + 1.0e-5, "{north}");
        let lat = [north, -north, 45.0, -89.9];
        let out = pole_clamped_latitudes(&lat, "mesh cell").unwrap();
        assert_eq!(&*out, &[90.0, -90.0, 45.0, -89.9]);
        let odd = pole_clamped_latitudes(&[f64::NAN, f64::INFINITY], "mesh cell").unwrap();
        assert!(odd[0].is_nan() && odd[1].is_infinite(), "non-finite values must reach the caller");
        let inside = [90.0, -90.0, 0.0];
        assert!(matches!(
            pole_clamped_latitudes(&inside, "mesh cell").unwrap(),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn latitude_clearly_past_the_pole_is_still_refused_by_name() {
        let past = 90.0 + 2.0 * POLE_ROUNDING_TOLERANCE_DEGREES;
        let err = pole_clamped_latitudes(&[0.0, -past], "mesh cell").unwrap_err().to_string();
        assert!(err.contains("mesh cell 1") && err.contains("outside [-90, 90]"), "{err}");
    }

    #[test]
    fn mesh_with_a_float32_pole_cell_gets_a_window() {
        // A mesh refined near the north pole whose refined cells include the
        // float32 pole cell, the rest coarse. Before the clamp this refused
        // with "mesh cell 0 sits at latitude 90.00000250447816, outside [-90, 90]".
        let pole = float32_pole_degrees();
        let mut lat = vec![pole];
        let mut lon = vec![0.0];
        let mut spacing = vec![3_000.0];
        for k in 0..10 {
            lat.push(88.0);
            lon.push(10.0 * k as f64);
            spacing.push(3_000.0);
        }
        for k in 0..36 {
            lat.push(-30.0);
            lon.push(-180.0 + 10.0 * k as f64);
            spacing.push(60_000.0);
        }
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let (grid_lat, grid_lon) = window.coordinates().unwrap();
        assert!(!grid_lat.is_empty());
        assert!(grid_lat.iter().all(|v| v.is_finite() && v.abs() <= 90.0));
        assert!(grid_lon.iter().all(|v| v.is_finite()));
        let cap = 11;
        let regional = Window::regional_focus(&lat[..cap], &lon[..cap], &spacing[..cap]).unwrap();
        assert!(regional.coordinates().unwrap().0.iter().all(|v| v.abs() <= 90.0));
    }
}

#[cfg(test)]
mod regional_focus_tests {
    use super::Window;

    #[test]
    fn uniform_regional_footprint_derives_a_finite_window() {
        let lat = [35.0, 35.0, 40.0, 40.0];
        let lon = [-100.0, -92.0, -100.0, -92.0];
        let spacing = [12000.0; 4];
        assert!(Window::mesh_focus(&lat, &lon, &spacing).is_err());
        let window = Window::regional_focus(&lat, &lon, &spacing).unwrap();
        let (grid_lat, grid_lon) = window.coordinates().unwrap();
        assert!(!grid_lat.is_empty());
        assert!(grid_lat.iter().chain(&grid_lon).all(|v| v.is_finite()));
        assert!(window.description().contains("limited-area"));
    }

    #[test]
    fn unbounded_or_invalid_regional_footprint_is_refused() {
        assert!(Window::regional_focus(&[f64::NAN], &[0.0], &[12000.0]).is_err());
        assert!(Window::regional_focus(&[0.0; 4], &[0.0, 90.0, 180.0, -90.0], &[12000.0; 4]).is_err());
    }
}

/// WRF's spherical earth. These windows emulate a WRF domain, so they use
/// WRF's radius and not a WGS84 mean radius, or the emitted coordinates would
/// disagree with the projection attributes the renderer reads.
pub const WRF_EARTH_RADIUS_M: f64 = 6_370_000.0;
/// The radius the MPAS meshes carry, used for physical distances only.
pub const MPAS_EARTH_RADIUS_M: f64 = 6_371_229.0;

const RAD: f64 = PI / 180.0;
const DEG: f64 = 180.0 / PI;

/// A cell belongs to the refined region when its spacing is within this
/// factor of the mesh's minimum spacing.
///
/// The rule is stated in the derived window's description string so it
/// travels with the window instead of living only here. 2.0 because a graded
/// mesh's density function is smooth: the finest spacing is attained over a
/// plateau of thousands of cells rather than at one freak cell, and doubling
/// it takes the core plus the innermost ring of the transition -- the part of
/// the mesh that carries structure the background cannot hold. A tighter
/// factor puts the window edge inside the gradient; a looser one walks out
/// into the background, and the derived dx stops being the resolution the
/// mesh actually carries where the window is pointed.
/// The prefix of an explicit Lambert window spec (see [`Window::lambert_from_spec`]).
pub const LAMBERT_SPEC_PREFIX: &str = "lambert:";

pub const MESH_FOCUS_REFINEMENT_FACTOR: f64 = 2.0;

/// The factor a given mesh's refined region is cut at: the standing 2.0, or
/// the geometric mean of the mesh's own finest and coarsest spacing when that
/// is smaller.
///
/// THE BREAKAGE THIS PREVENTS (gpuwm-hex zoom globes, worker zoomna and
/// zoom36, 2026-11-14): a zoom globe refined by 2x (120 to 60 km, 192 to
/// 96 km, 72 to 36 km) has a background that is not uniform -- the
/// generator's background runs about 0.8 to 1.05 of its nominal spacing, finer
/// around the icosahedron's twelve corners -- so at 2.0x the minimum the
/// corner patches (100 to 110 km on the 120 to 60 km globe, against a 103 km
/// cut) were taken as refined, the selection spanned the planet, and
/// `--window mesh` refused with "the refined region reaches 162.36 degrees of
/// arc". The products door then drew no map of the fine region on any 2x zoom
/// globe. The geometric mean sits halfway (in ratio) between the core and the
/// background, so it takes the core and the inner transition and never the
/// background. For a mesh refined 4x or more the mean is past 2.0 and the cut
/// is the standing 2.0x, unchanged.
pub fn mesh_refinement_factor(min_spacing_m: f64, max_spacing_m: f64) -> f64 {
    MESH_FOCUS_REFINEMENT_FACTOR.min((max_spacing_m / min_spacing_m).sqrt())
}

/// The derived whole-sphere window: a lat-lon grid at the mesh's OWN
/// spacing (see [`Window::global_at_mesh_spacing`]).
pub const GLOBAL_MESH_WINDOW: &str = "global-mesh";

/// The prefix of an extent-only regional window whose spacing the mesh
/// decides (see [`Window::region_at_mesh_spacing`]).
pub const REGION_SPEC_PREFIX: &str = "region:";

/// The prefix of the same extent-only regional window held on a regular
/// latitude/longitude grid at any latitude
/// (`region-latlon:CLAT,CLON,WIDTH_KM,HEIGHT_KM`; see
/// [`Window::region_at_mesh_spacing`]).
///
/// THE BREAKAGE THIS PREVENTS: `region:` turns Lambert 10 degrees off the
/// equator, and a Lambert frame carries only 2-D `XLAT`/`XLONG`.  The
/// engine's mapped NetCDF route reads 1-D latitude and longitude, so a hex
/// regional window over the US plains could not be an ArWen parent at all.
pub const REGION_LATLON_SPEC_PREFIX: &str = "region-latlon:";

/// The largest point count a derived global window may carry.
///
/// The breakage this prevents, named: a 3 km uniform global mesh asks for
/// 13,344 x 6,673 points, and the converter writes every 3-D field at
/// `(levels, ny, nx)` float32 -- 55 levels over 89 million points is 19.6 GB
/// for ONE field, past the 4 GiB a classic 64-bit-offset variable may hold
/// and past what the host can allocate several times over. Nine million
/// points keeps one 55-level field at 2.0 GB, inside that format limit, and
/// still covers every uniform mesh down to 10 km at its own spacing (15 km
/// is 3.6 million points). When it binds, the spacing coarsens and the
/// description records what it started from.
pub const MESH_GLOBAL_MAX_POINTS: usize = 9_000_000;

/// Margin added to each half-extent of the refined region, as a fraction.
///
/// The refined core of a graded mesh is where the storm is; the transition
/// ring around it still resolves the inflow feeding it. Ten percent is one
/// panel border, not a second domain.
pub const MESH_FOCUS_MARGIN_FRACTION: f64 = 0.10;

/// The largest derived target grid, per side.
///
/// The breakage this prevents, named: a mesh refined over a whole hemisphere
/// at 3 km asks for 13,000 points a side, and the converter writes every 3-D
/// field at `(levels, ny, nx)` float32 -- 55 levels over 13,000^2 is 37 GB
/// for ONE field, and a frame carries several, so the run would allocate
/// until the host died and emit nothing. At 2,000 a side that same field is
/// 880 MB, which is the practical ceiling for a wrfout-shaped frame, and the
/// grid still carries 111x the points of the fixed CONUS `focus` box.
///
/// When the clamp binds, `dx_metres` is coarsened so the refined region stays
/// FULLY covered -- cropping would silently drop the part of the mesh the
/// caller asked to see -- and the description records what dx it started from.
pub const MESH_FOCUS_MAX_POINTS_PER_SIDE: usize = 2_000;

/// Below this |centre latitude| the derived window is emitted as lat-lon
/// rather than Lambert.
///
/// A Lambert cone's constant is `sin(|truelat|)`, so it collapses toward the
/// equator, and the projection origin `polej` in `set_lc` carries a `1/cone`
/// and recedes as it does. On the equator itself the cone is a literal 0/0 --
/// the two derived truelats sit symmetric about it, so
/// `log10(cos t1) - log10(cos t2)` is zero and so is its denominator -- and
/// EVERY emitted coordinate is NaN. That is measured, not feared:
/// `the_cone_is_why_a_near_equatorial_centre_falls_back` builds the window at
/// a centre of exactly 0 and finds `cone = 0`, `polej = inf` and no finite
/// coordinate in the grid.
///
/// Where the line sits is a judgement, and it is not a float-precision line:
/// measured, the f64 transcription still round-trips its own centre to
/// 6e-8 m at a centre of 0.5 degrees, where the cone is 0.0087 and the origin
/// sits 161,204 grid units away. What forces a line at all is that this
/// centre is DERIVED -- it is wherever the mesh's refined cells happen to
/// average out -- so a mesh refined symmetrically about the equator lands
/// arbitrarily close to that NaN, and no small number is a safe distance from
/// a singularity.
///
/// 10 degrees is where a Lambert cone is still recognisably a cone: measured,
/// its constant is 0.174 and its origin is 8,008 grid units off the grid,
/// which at 4.5 km is 5.7 earth radii. Below it the map's shape is degrading
/// toward a cylinder anyway, so emitting the cylinder is both better
/// conditioned and the accurate answer for a near-equatorial region.
pub const MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG: f64 = 10.0;

/// Half the span between the two derived true latitudes.
///
/// Bracketing the centre is the standard WRF practice the fixed `focus`
/// window follows (37 N between 30 and 60). The half-span is additionally
/// held to `|centre_lat| / 2` so both truelats stay strictly in the centre's
/// own hemisphere: `set_lc` takes `hemi` from the SIGN of `truelat1`, so a
/// truelat that crosses the equator flips the cone to the wrong pole and the
/// window lands in the opposite hemisphere.
const MESH_FOCUS_TRUELAT_HALF_SPAN_DEG: f64 = 5.0;

/// The northernmost true latitude a derived window will use. `set_lc`
/// divides by `tan((90 - truelat) * pi / 360)`, which is zero at the pole.
const MESH_FOCUS_MAX_ABS_TRUELAT_DEG: f64 = 89.0;

/// How short the mean of the selected cells' unit vectors may be before the
/// centre it names has no direction. A mean this short means the selected
/// cells are spread over the sphere with no middle -- the exact case a
/// uniform global mesh produces, where the mean is the origin.
const MESH_FOCUS_MIN_CENTROID_NORM: f64 = 1.0e-6;

/// How far a selected cell may sit from the derived centre. Past a quarter
/// turn the window has passed the projection's horizon: a Lambert cone sends
/// the antipode to infinity, so the far cells project to a radius with no
/// bound and nx and ny would be sized from it, while a cylindrical window
/// that wide has wrapped the planet and covers the same ground twice.
const MESH_FOCUS_MAX_ANGULAR_RADIUS_RAD: f64 = PI / 2.0;

/// A regular latitude/longitude window; renders under WRF `MAP_PROJ=6`.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonWindow {
    pub south: f64,
    pub north: f64,
    pub west: f64,
    pub east: f64,
    pub spacing_degrees: f64,
    pub description: String,
}

/// A Lambert conformal window emulating a WRF domain (`MAP_PROJ=1`).
#[derive(Debug, Clone, PartialEq)]
pub struct LambertWindow {
    pub centre_lat: f64,
    pub centre_lon: f64,
    pub dx_metres: f64,
    pub nx: usize,
    pub ny: usize,
    pub truelat1: f64,
    pub truelat2: f64,
    pub stand_lon: f64,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Window {
    LatLon(LatLonWindow),
    Lambert(LambertWindow),
}

/// A projection attribute value as it lands in the emitted file.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjAttr {
    Int(i32),
    Float(f64),
    Text(String),
}

impl Window {
    /// The catalogue of FIXED named windows: a coarse global overview plus
    /// one frozen CONUS box.
    ///
    /// Neither geometry may move. Both are inputs to `weights_sha256`, and
    /// those digests are recorded in evidence already written; a corner
    /// nudged by a metre invalidates every one of them silently. A mesh
    /// refined somewhere these boxes do not reach gets
    /// [`Window::mesh_focus`], not a third entry here.
    pub fn named(name: &str) -> MpasResult<Window> {
        match name {
            "global" => Ok(Window::LatLon(LatLonWindow {
                south: -90.0,
                north: 90.0,
                west: -180.0,
                east: 179.75,
                spacing_degrees: 0.25,
                description: "global 0.25 degree overview window".to_string(),
            })),
            "focus" => Ok(Window::Lambert(LambertWindow {
                centre_lat: 37.0,
                centre_lon: -96.0,
                dx_metres: 22_000.0,
                nx: 240,
                ny: 150,
                truelat1: 30.0,
                truelat2: 60.0,
                stand_lon: -96.0,
                description:
                    "CONUS focus window on a 22 km Lambert grid, matched to the x4.163842 refined-region spacing"
                        .to_string(),
            })),
            "mesh" => Err(MpasError::Refusal(
                "render window 'mesh' is derived from the mesh being converted, not served \
                 from this catalogue: build it with Window::mesh_focus once that mesh's cell \
                 coordinates and per-cell spacing have been read. Reaching here means a \
                 caller asked for the derived window without the mesh in hand, and answering \
                 with a fixed box would render a region the mesh may not refine at all"
                    .to_string(),
            )),
            spec if spec.starts_with(LAMBERT_SPEC_PREFIX) => {
                Window::lambert_from_spec(&spec[LAMBERT_SPEC_PREFIX.len()..])
            }
            spec if spec.starts_with("latlon:") => {
                let numbers: Vec<f64> = spec[7..].split(',').map(str::parse)
                    .collect::<Result<_,_>>().map_err(|_| MpasError::Refusal(
                        "latlon takes SOUTH,NORTH,WEST,EAST,SPACING_DEG".into()))?;
                if numbers.len() != 5 || numbers.iter().any(|v| !v.is_finite())
                    || numbers[0] < -90.0 || numbers[1] > 90.0
                    || numbers[1] <= numbers[0] || numbers[3] <= numbers[2]
                    || numbers[3] - numbers[2] >= 360.0 || numbers[4] <= 0.0 {
                    return Err(MpasError::Refusal("invalid explicit latlon bounds/spacing".into()));
                }
                // Check the count BEFORE allocating axes. A fine grid over a
                // hemisphere otherwise exhausts memory in shape() itself.
                for span in [numbers[1]-numbers[0], numbers[3]-numbers[2]] {
                    if (span/numbers[4]).round()+1.0 > MESH_FOCUS_MAX_POINTS_PER_SIDE as f64 {
                        return Err(MpasError::Refusal(format!(
                            "explicit latlon exceeds {MESH_FOCUS_MAX_POINTS_PER_SIDE} points per side; choose a smaller extent or coarser spacing")));
                    }
                }
                let window = Window::LatLon(LatLonWindow {
                    south:numbers[0], north:numbers[1], west:numbers[2], east:numbers[3],
                    spacing_degrees:numbers[4], description:format!("explicit fixed {spec}"),
                });
                window.shape()?; // spacing must partition the bounds exactly
                Ok(window)
            }
            GLOBAL_MESH_WINDOW => Err(MpasError::Refusal(
                "render window 'global-mesh' is a whole-sphere grid at the spacing of the mesh \
                 being converted, so it cannot be served before that mesh's per-cell spacing \
                 has been read: build it with Window::global_at_mesh_spacing. Answering with \
                 the fixed 0.25 degree box instead would draw a 15 km mesh on a 28 km grid"
                    .to_string(),
            )),
            spec if spec.starts_with(REGION_SPEC_PREFIX)
                || spec.starts_with(REGION_LATLON_SPEC_PREFIX) => Err(MpasError::Refusal(format!(
                "render window '{spec}' takes its spacing from the mesh cells inside it, so it \
                 cannot be served before that mesh has been read: build it with \
                 Window::region_at_mesh_spacing. A fixed spacing here is how a 15 km mesh came \
                 to be drawn on a 20 km grid"
            ))),
            other => Err(MpasError::Refusal(format!(
                "unknown render window '{other}' (known: focus, global, mesh, global-mesh, region:CLAT,CLON,WIDTH_KM,HEIGHT_KM[,TRUELAT1,TRUELAT2], latlon:SOUTH,NORTH,WEST,EAST,SPACING_DEG, lambert:CLAT,CLON,DX_M,NX,NY,TRUELAT1,TRUELAT2,STAND_LON)"
            ))),
        }
    }

    /// The render window a mesh implies: its own refined region, at its own
    /// spacing.
    ///
    /// The defect this exists for, measured: a global variable-resolution
    /// mesh whose refined core sits at 15-20 N, 50-57 W attains 4.53 km
    /// inside a 75 km background, and the only windows on offer were a fixed
    /// CONUS box 5,000 km away -- which contains none of that core and
    /// samples 75-80 km background cells -- and a 0.25 degree global grid,
    /// which is 28 km and cannot resolve a 4.5 km core at all. There was no
    /// way to render that mesh's fine region at its own resolution, and every
    /// future mesh placed somewhere new would have needed another hardcoded
    /// box. This derives the box instead, so a mesh refined anywhere is data
    /// rather than a code path.
    ///
    /// `cell_spacing_m` is the per-cell horizontal spacing in metres, one
    /// value per cell, on the same ordering as the coordinates --
    /// [`crate::weights::read_mesh_cell_spacing`] reads it off a mesh file.
    ///
    /// What it decides, in order:
    ///
    /// - The refined region is every cell within
    ///   [`MESH_FOCUS_REFINEMENT_FACTOR`] of the mesh's minimum spacing.
    /// - The centre is the mean of those cells' unit vectors, re-normalised.
    ///   Not a mean of longitudes: a region straddling the antimeridian
    ///   averages +179 and -179 to 0 and lands the window on the wrong side
    ///   of the planet.
    /// - `dx_metres` is the MEDIAN spacing of the selected cells, so the
    ///   window resolves what the mesh carries there rather than what its
    ///   single finest cell carries.
    /// - `nx`/`ny` cover the selected cells' extent plus
    ///   [`MESH_FOCUS_MARGIN_FRACTION`], clamped by
    ///   [`MESH_FOCUS_MAX_POINTS_PER_SIDE`]; a clamp coarsens `dx_metres`
    ///   instead of cropping.
    /// - Lambert when the centre is far enough from the equator for the cone
    ///   to be conditioned ([`MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG`]),
    ///   lat-lon when it is not.
    ///
    /// A UNIFORM mesh is refused by name rather than answered. It has no
    /// refined region to find, and the whole-mesh answer is not merely
    /// unhelpful: the unit-vector mean of a quasi-uniform global mesh IS the
    /// origin, so the centre would be float noise pointing in an arbitrary
    /// direction, and the extent would be the whole sphere, which no Lambert
    /// cone represents. `--window global` is the window a uniform mesh has.
    /// An explicit Lambert target grid:
    /// `lambert:CLAT,CLON,DX_M,NX,NY,TRUELAT1,TRUELAT2,STAND_LON`.
    ///
    /// The grid is the one WPS lays down for a domain of that projection,
    /// centred on the point `((nx+1)/2, (ny+1)/2)` -- the same convention the
    /// fixed `focus` window uses -- so the spec of another model's own mass
    /// grid (its `DX`, `TRUELAT1/2`, `STAND_LON`, dimensions and the lat/lon
    /// of its geometric centre) reproduces that grid point for point.
    ///
    /// THE BREAKAGE THIS PREVENTS: every window this converter knew was
    /// either fixed geometry or derived from the mesh itself, so a mesh
    /// forecast could never land on the grid of the WRF-type forecast it is
    /// being compared with.  A side-by-side then showed two different
    /// footprints and two different pixel grids, and any field difference
    /// needed a second resample outside the Rust data path.
    pub fn lambert_from_spec(text: &str) -> MpasResult<Window> {
        let parts: Vec<&str> = text.split(',').map(str::trim).collect();
        if parts.len() != 8 {
            return Err(MpasError::Refusal(format!(
                "a lambert window takes eight comma-separated values CLAT,CLON,DX_M,NX,NY,TRUELAT1,TRUELAT2,STAND_LON; '{text}' has {}",
                parts.len()
            )));
        }
        let number = |index: usize, role: &str| -> MpasResult<f64> {
            let value: f64 = parts[index].parse().map_err(|_| {
                MpasError::Refusal(format!(
                    "lambert window {role} '{}' is not a number",
                    parts[index]
                ))
            })?;
            if !value.is_finite() {
                return Err(MpasError::Refusal(format!(
                    "lambert window {role} must be finite"
                )));
            }
            Ok(value)
        };
        let count = |index: usize, role: &str| -> MpasResult<usize> {
            parts[index].parse::<usize>().map_err(|_| {
                MpasError::Refusal(format!(
                    "lambert window {role} '{}' is not a whole number",
                    parts[index]
                ))
            })
        };
        let centre_lat = number(0, "centre latitude")?;
        let centre_lon = number(1, "centre longitude")?;
        let dx_metres = number(2, "dx")?;
        let nx = count(3, "nx")?;
        let ny = count(4, "ny")?;
        let truelat1 = number(5, "truelat1")?;
        let truelat2 = number(6, "truelat2")?;
        let stand_lon = number(7, "stand_lon")?;
        if centre_lat.abs() > 89.0 || truelat1.abs() > 89.0 || truelat2.abs() > 89.0 {
            return Err(MpasError::Refusal(
                "lambert window latitudes must lie within 89 degrees of the equator".to_string(),
            ));
        }
        if !(dx_metres > 0.0) {
            return Err(MpasError::Refusal("lambert window dx must be positive".to_string()));
        }
        if nx < 2 || ny < 2 || nx > 10_000 || ny > 10_000 {
            return Err(MpasError::Refusal(format!(
                "lambert window {nx} x {ny} is outside 2..=10000 points per side"
            )));
        }
        Ok(Window::Lambert(LambertWindow {
            centre_lat,
            centre_lon,
            dx_metres,
            nx,
            ny,
            truelat1,
            truelat2,
            stand_lon,
            description: format!(
                "explicit Lambert window {nx} x {ny} at {} m centred {}",
                json_f64(dx_metres),
                format_centre(centre_lat, centre_lon)
            ),
        }))
    }

    /// The whole sphere on a lat-lon grid at the mesh's own spacing.
    ///
    /// THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-10): every global map
    /// of the 15 km hex mesh (x1.2621442) was drawn from the fixed `global`
    /// window, a 0.25 degree grid that is 27.8 km at the equator -- nearly
    /// twice the model's spacing -- and stamped `d01-27.794km`. The 60 km
    /// mesh came out on the same 27.8 km grid, finer than the model holds.
    /// A whole-sphere picture of a mesh forecast has to be drawn at the
    /// spacing that mesh carries, and the mesh is the only source that
    /// states that number.
    ///
    /// The spacing is the median per-cell spacing, rounded DOWN to three
    /// significant figures (never coarser than the mesh), then shortened to
    /// the nearest whole division of 180 degrees so both axes partition
    /// their bounds exactly. Longitudes run -180 to 180 minus one step.
    pub fn global_at_mesh_spacing(cell_spacing_m: &[f64]) -> MpasResult<Window> {
        let median = Self::checked_median_spacing(cell_spacing_m, "a global window")?;
        let target_m = finer_round_spacing(median);
        let half_turn_m = PI * WRF_EARTH_RADIUS_M;
        let mut intervals = (half_turn_m / target_m).ceil().max(2.0) as usize;
        let points = |n: usize| 2 * n * (n + 1);
        let mut coarsened_from: Option<f64> = None;
        if points(intervals) > MESH_GLOBAL_MAX_POINTS {
            coarsened_from = Some(180.0 / intervals as f64);
            // Largest n with 2 n (n + 1) <= the ceiling.
            intervals =
                ((-1.0 + (1.0 + 2.0 * MESH_GLOBAL_MAX_POINTS as f64).sqrt()) / 2.0).floor() as usize;
        }
        let spacing_degrees = 180.0 / intervals as f64;
        let spacing_m = spacing_degrees * RAD * WRF_EARTH_RADIUS_M;
        Ok(Window::LatLon(LatLonWindow {
            south: -90.0,
            north: 90.0,
            west: -180.0,
            east: 180.0 - spacing_degrees,
            spacing_degrees,
            description: format!(
                "mesh-derived global window (lat-lon): {}x{} points at {spacing_degrees:.6} \
                 degrees ({:.3} km), from the mesh's median cell spacing {:.3} km rounded down \
                 to {:.3} km{}",
                2 * intervals,
                intervals + 1,
                spacing_m / 1000.0,
                median / 1000.0,
                target_m / 1000.0,
                match coarsened_from {
                    Some(was) => format!(
                        "; coarsened from {was:.6} degrees because the whole sphere at that \
                         spacing needed more than {MESH_GLOBAL_MAX_POINTS} points"
                    ),
                    None => String::new(),
                }
            ),
        }))
    }

    /// A regional window of a stated ground extent whose spacing is the
    /// mesh's own spacing INSIDE it:
    /// `region:CLAT,CLON,WIDTH_KM,HEIGHT_KM[,TRUELAT1,TRUELAT2]`.
    ///
    /// THE BREAKAGE THIS PREVENTS, MEASURED (2026-10-10): the live hex
    /// gallery framed its North America, Asia and polar zooms as explicit
    /// `lambert:` specs with a hand-written 20,000 m dx, so a 15 km mesh was
    /// drawn on a 20 km grid and every title said `d01 20 km`; the same
    /// spec drew the 60 km mesh at 20 km, finer than it holds. An explicit
    /// spec has to carry a dx, and a dx typed into a caller is a constant
    /// that is wrong for every mesh but one. This form carries only the
    /// ground the caller wants to see.
    ///
    /// What it decides, in order:
    ///
    /// - Lambert, with the given true latitudes or ones bracketing the
    ///   centre the way [`Window::mesh_focus`] derives them; lat-lon when no
    ///   truelats are given and the centre is within
    ///   [`MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG`] of the equator.
    /// - The cells whose centres fall inside the box (measured on that
    ///   projection) are the region; with none inside, the cell nearest the
    ///   centre stands in.
    /// - `dx` is their median spacing rounded DOWN to three significant
    ///   figures, so the grid is never coarser than the mesh there.
    /// - `nx`/`ny` cover the extent at that dx, clamped by
    ///   [`MESH_FOCUS_MAX_POINTS_PER_SIDE`]; a clamp coarsens dx instead of
    ///   cropping, and the description says so.
    pub fn region_at_mesh_spacing(
        spec: &str,
        latitude_degrees: &[f64],
        longitude_degrees: &[f64],
        cell_spacing_m: &[f64],
    ) -> MpasResult<Window> {
        let (body, force_latlon) = match spec.strip_prefix(REGION_LATLON_SPEC_PREFIX) {
            Some(body) => (body, true),
            None => (spec.strip_prefix(REGION_SPEC_PREFIX).unwrap_or(spec), false),
        };
        let parts: Vec<&str> = body.split(',').map(str::trim).collect();
        if force_latlon && parts.len() != 4 {
            return Err(MpasError::Refusal(format!(
                "a region-latlon window takes CLAT,CLON,WIDTH_KM,HEIGHT_KM (true latitudes                  belong to a Lambert window); '{body}' has {} value(s)",
                parts.len()
            )));
        }
        if parts.len() != 4 && parts.len() != 6 {
            return Err(MpasError::Refusal(format!(
                "a region window takes CLAT,CLON,WIDTH_KM,HEIGHT_KM and optionally \
                 TRUELAT1,TRUELAT2; '{body}' has {} value(s)",
                parts.len()
            )));
        }
        let mut values = Vec::with_capacity(parts.len());
        for (index, text) in parts.iter().enumerate() {
            let value: f64 = text.parse().map_err(|_| {
                MpasError::Refusal(format!("region window value {} '{text}' is not a number", index + 1))
            })?;
            if !value.is_finite() {
                return Err(MpasError::Refusal(format!(
                    "region window value {} must be finite",
                    index + 1
                )));
            }
            values.push(value);
        }
        let (centre_lat, centre_lon) = (values[0], values[1]);
        let (width_m, height_m) = (values[2] * 1000.0, values[3] * 1000.0);
        if centre_lat.abs() > 89.0 {
            return Err(MpasError::Refusal(
                "a region window centre must lie within 89 degrees of the equator".to_string(),
            ));
        }
        // Half a meridian: past it the box has wrapped past the horizon a
        // projected window has, and would cover the same ground twice.
        let max_extent_m = PI * WRF_EARTH_RADIUS_M;
        if !(width_m > 0.0 && height_m > 0.0) || width_m > max_extent_m || height_m > max_extent_m {
            return Err(MpasError::Refusal(format!(
                "a region window's width and height must be positive and at most {:.0} km",
                max_extent_m / 1000.0
            )));
        }
        let n = latitude_degrees.len();
        if longitude_degrees.len() != n || cell_spacing_m.len() != n {
            return Err(MpasError::Refusal(format!(
                "a region window needs one latitude, longitude and spacing per cell; got {n}, \
                 {} and {}",
                longitude_degrees.len(),
                cell_spacing_m.len()
            )));
        }
        Self::checked_median_spacing(cell_spacing_m, "a region window")?;
        let explicit_truelats = (parts.len() == 6).then(|| (values[4], values[5]));
        if let Some((t1, t2)) = explicit_truelats {
            // `set_lc` takes the hemisphere from the sign of truelat1, so a
            // pair that straddles (or touches) the equator flips the cone.
            if t1.abs() > 89.0 || t2.abs() > 89.0 || t1 * t2 <= 0.0 {
                return Err(MpasError::Refusal(
                    "region window true latitudes must lie within 89 degrees of the equator, \
                     both on one side of it"
                        .to_string(),
                ));
            }
        }
        let centre = unit_vector(centre_lat, centre_lon);
        let lambert = !force_latlon
            && (explicit_truelats.is_some()
                || centre_lat.abs() >= MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG);
        let (truelat1, truelat2) = explicit_truelats.unwrap_or_else(|| {
            let hemi = if centre_lat >= 0.0 { 1.0 } else { -1.0 };
            let abs_lat = centre_lat.abs();
            let half_span = MESH_FOCUS_TRUELAT_HALF_SPAN_DEG.min(0.5 * abs_lat);
            (
                hemi * (abs_lat - half_span),
                hemi * (abs_lat + half_span).min(MESH_FOCUS_MAX_ABS_TRUELAT_DEG),
            )
        });
        // Cell membership, measured on the projection the window will use.
        // A cell more than a quarter turn from the centre is past the
        // horizon and never inside, whatever the cone maps it to.
        let half_lat_deg = 0.5 * height_m / (RAD * WRF_EARTH_RADIUS_M);
        let half_lon_deg =
            0.5 * width_m / (RAD * WRF_EARTH_RADIUS_M * cos(centre_lat * RAD).max(1.0e-6));
        let probe = LambertProjection::set_lc(
            truelat1, truelat2, centre_lon, 1.0, centre_lat, centre_lon, 0.0, 0.0,
        );
        let mut selected: Vec<usize> = Vec::new();
        let mut nearest: Option<(usize, f64)> = None;
        for i in 0..n {
            let (lat, lon) = (latitude_degrees[i], longitude_degrees[i]);
            let v = unit_vector(lat, lon);
            let dot = v[0] * centre[0] + v[1] * centre[1] + v[2] * centre[2];
            if nearest.is_none_or(|(_, best)| dot > best) {
                nearest = Some((i, dot));
            }
            if dot <= 0.0 {
                continue;
            }
            let inside = if lambert {
                let (x, y) = probe.llij(lat, lon);
                x.is_finite() && y.is_finite() && x.abs() <= 0.5 * width_m && y.abs() <= 0.5 * height_m
            } else {
                let dlon = (lon - centre_lon + 180.0).rem_euclid(360.0) - 180.0;
                (lat - centre_lat).abs() <= half_lat_deg && dlon.abs() <= half_lon_deg
            };
            if inside {
                selected.push(i);
            }
        }
        let stood_in = selected.is_empty();
        if stood_in {
            selected.push(nearest.map(|(i, _)| i).unwrap_or(0));
        }
        let median = Self::median_spacing_of(cell_spacing_m, &selected);
        let nominal_m = finer_round_spacing(median);
        let span = (MESH_FOCUS_MAX_POINTS_PER_SIDE - 1) as f64;
        let floor_m = width_m.max(height_m) / span;
        let mut dx_metres = nominal_m;
        let mut coarsened_from: Option<f64> = None;
        if dx_metres < floor_m {
            coarsened_from = Some(dx_metres);
            dx_metres = floor_m * (1.0 + 1.0e-12);
        }
        let points = |extent: f64| -> usize {
            ((extent / dx_metres).ceil() as usize + 1).clamp(2, MESH_FOCUS_MAX_POINTS_PER_SIDE)
        };
        let nx = points(width_m);
        let mut ny = points(height_m);
        let selection = format!(
            "{} of {n} cells fall inside the {:.0} by {:.0} km box centred {}{}; their median \
             spacing is {:.3} km, rounded down to {:.3} km{}",
            if stood_in { 0 } else { selected.len() },
            width_m / 1000.0,
            height_m / 1000.0,
            format_centre(centre_lat, centre_lon),
            if stood_in { " (the cell nearest the centre stands in)" } else { "" },
            median / 1000.0,
            nominal_m / 1000.0,
            match coarsened_from {
                Some(was) => format!(
                    ", coarsened from {:.3} km because that extent needed more than \
                     {MESH_FOCUS_MAX_POINTS_PER_SIDE} points a side",
                    was / 1000.0
                ),
                None => String::new(),
            }
        );
        if lambert {
            return Ok(Window::Lambert(LambertWindow {
                centre_lat,
                centre_lon,
                dx_metres,
                nx,
                ny,
                truelat1,
                truelat2,
                stand_lon: centre_lon,
                description: format!(
                    "mesh-spacing region window (Lambert): {selection}; {nx}x{ny} at dx \
                     {:.3} km; truelats {truelat1:.3} and {truelat2:.3}, stand_lon the centre \
                     longitude",
                    dx_metres / 1000.0
                ),
            }));
        }
        let spacing_degrees = dx_metres / (RAD * WRF_EARTH_RADIUS_M);
        let mut nx_ll =
            ((2.0 * half_lon_deg / spacing_degrees).ceil() as usize + 1).clamp(2, MESH_FOCUS_MAX_POINTS_PER_SIDE);
        if (nx_ll - 1) as f64 * spacing_degrees >= 360.0 {
            nx_ll = (360.0 / spacing_degrees).floor() as usize;
        }
        let room = 2.0 * (90.0 - centre_lat.abs()) / spacing_degrees;
        if (ny - 1) as f64 > room {
            ny = (room.floor() as usize + 1).max(2);
        }
        let south = centre_lat - 0.5 * (ny - 1) as f64 * spacing_degrees;
        let west = centre_lon - 0.5 * (nx_ll - 1) as f64 * spacing_degrees;
        Ok(Window::LatLon(LatLonWindow {
            south,
            north: south + (ny - 1) as f64 * spacing_degrees,
            west,
            east: west + (nx_ll - 1) as f64 * spacing_degrees,
            spacing_degrees,
            description: if force_latlon {
                format!(
                    "mesh-spacing region window (lat-lon, asked for by region-latlon:): \
                     {selection}; {nx_ll}x{ny} at {spacing_degrees:.6} degrees"
                )
            } else {
                format!(
                    "mesh-spacing region window (lat-lon): {selection}; {nx_ll}x{ny} at \
                     {spacing_degrees:.6} degrees; the centre is within \
                     {MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG:.0} degrees of the equator, where a \
                     Lambert cone degenerates"
                )
            },
        }))
    }

    /// The median of a whole mesh's per-cell spacing, refusing an empty or
    /// non-positive set by name: a window sized from a zero or a NaN is the
    /// whole planet or a single point.
    fn checked_median_spacing(cell_spacing_m: &[f64], role: &str) -> MpasResult<f64> {
        if cell_spacing_m.is_empty() {
            return Err(MpasError::Refusal(format!(
                "{role} at the mesh's own spacing needs that spacing, and the mesh handed over \
                 none"
            )));
        }
        if let Some((i, bad)) = cell_spacing_m
            .iter()
            .enumerate()
            .find(|(_, s)| !s.is_finite() || **s <= 0.0)
        {
            return Err(MpasError::Refusal(format!(
                "mesh cell {i} carries spacing {bad} m; {role} cannot be sized from a spacing \
                 that is not positive and finite"
            )));
        }
        let all: Vec<usize> = (0..cell_spacing_m.len()).collect();
        Ok(Self::median_spacing_of(cell_spacing_m, &all))
    }

    pub fn mesh_focus(
        latitude_degrees: &[f64],
        longitude_degrees: &[f64],
        cell_spacing_m: &[f64],
    ) -> MpasResult<Window> {
        let n = latitude_degrees.len();
        if n == 0 {
            return Err(MpasError::Refusal(
                "a mesh-derived window needs at least one cell; the mesh handed over none, \
                 so there is no refined region to centre on and no spacing to render at"
                    .to_string(),
            ));
        }
        if longitude_degrees.len() != n || cell_spacing_m.len() != n {
            return Err(MpasError::Refusal(format!(
                "a mesh-derived window needs one latitude, longitude and spacing per cell; \
                 got {n} latitude(s), {} longitude(s) and {} spacing(s). Zipping arrays of \
                 different lengths pairs one cell's position with another cell's resolution",
                longitude_degrees.len(),
                cell_spacing_m.len()
            )));
        }
        let latitude_degrees = &*pole_clamped_latitudes(latitude_degrees, "mesh cell")?;
        for i in 0..n {
            if !latitude_degrees[i].is_finite() || !longitude_degrees[i].is_finite() {
                return Err(MpasError::Refusal(format!(
                    "mesh cell {i} sits at latitude {} longitude {}, which is not finite; a \
                     non-finite coordinate poisons the unit-vector mean and every derived \
                     corner with it",
                    latitude_degrees[i], longitude_degrees[i]
                )));
            }
            if !cell_spacing_m[i].is_finite() || cell_spacing_m[i] <= 0.0 {
                return Err(MpasError::Refusal(format!(
                    "mesh cell {i} carries spacing {} m; a spacing has to be positive and \
                     finite or the refinement ratio it is compared against is meaningless",
                    cell_spacing_m[i]
                )));
            }
        }

        // --- the refined region -------------------------------------------
        let min_spacing = cell_spacing_m.iter().copied().fold(f64::INFINITY, f64::min);
        let max_spacing = cell_spacing_m.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let selected = Self::refined_region(cell_spacing_m)?;
        let median_spacing_m = Self::median_spacing_of(cell_spacing_m, &selected);
        let selection = format!(
            "{} of {n} cells carry spacing at or under {:.2}x the mesh minimum {:.3} km and              are taken as the refined region",
            selected.len(),
            mesh_refinement_factor(min_spacing, max_spacing),
            min_spacing / 1000.0,
        );

        Self::window_over_selected_cells(
            latitude_degrees,
            longitude_degrees,
            &selected,
            median_spacing_m,
            MESH_FOCUS_MARGIN_FRACTION,
            &selection,
            "mesh-derived focus window",
        )
    }

    /// A limited-area window covers all supplied cells at their median spacing.
    /// The caller verifies a nonempty boundary mask before choosing this path.
    pub fn regional_focus(
        latitude_degrees: &[f64],
        longitude_degrees: &[f64],
        cell_spacing_m: &[f64],
    ) -> MpasResult<Window> {
        let n = latitude_degrees.len();
        if n == 0 || longitude_degrees.len() != n || cell_spacing_m.len() != n {
            return Err(MpasError::Refusal(
                "a regional render window needs one position and spacing per cell".to_string(),
            ));
        }
        let latitude_degrees = &*pole_clamped_latitudes(latitude_degrees, "regional cell")?;
        for i in 0..n {
            if !latitude_degrees[i].is_finite()
                || !longitude_degrees[i].is_finite()
                || !cell_spacing_m[i].is_finite()
                || cell_spacing_m[i] <= 0.0
            {
                return Err(MpasError::Refusal(format!(
                    "regional cell {i} has an invalid position or spacing; the window cannot be projected"
                )));
            }
        }
        let selected: Vec<usize> = (0..n).collect();
        let spacing = Self::median_spacing_of(cell_spacing_m, &selected);
        Self::window_over_selected_cells(
            latitude_degrees,
            longitude_degrees,
            &selected,
            spacing,
            MESH_FOCUS_MARGIN_FRACTION,
            "all cells of the verified limited-area mesh",
            "mesh-derived limited-area window",
        )
    }

    /// A window covering every fine region handed over, with room around
    /// them for the coarse field they will be composited into.
    ///
    /// gpuwm addition (VENDOR.md).  `mesh_focus` frames ONE mesh's refined
    /// region tightly, which is right for rendering that core on its own
    /// and wrong for a composite: a tight frame contains no coarse field,
    /// so there is nothing to composite against and the picture is the fine
    /// render again. This takes the refined cells of every source at once
    /// -- so N placed grids get one frame, not N -- renders at the finest
    /// spacing among them, and pulls the frame out to
    /// `context_factor` times the fine footprint.
    ///
    /// The same clamp applies: past 2000 points a side the spacing
    /// coarsens rather than the frame cropping. When N grids are scattered
    /// far apart, that clamp is what a caller will hit, and it is the
    /// accurate outcome -- one frame wide enough to hold two grids on
    /// opposite sides of an ocean cannot also resolve 5 km. The
    /// description says so, in the numbers.
    pub fn composite_focus(
        latitude_degrees: &[f64],
        longitude_degrees: &[f64],
        spacing_metres: &[f64],
        context_factor: f64,
    ) -> MpasResult<Window> {
        let n = latitude_degrees.len();
        if n == 0 {
            return Err(MpasError::Refusal(
                "a composite window is sized to the fine regions being composited in; none                  was handed over, so the window would have nothing to centre on"
                    .to_string(),
            ));
        }
        if longitude_degrees.len() != n || spacing_metres.len() != n {
            return Err(MpasError::Refusal(format!(
                "a composite window needs one latitude, longitude and spacing per fine cell;                  got {n}, {} and {}",
                longitude_degrees.len(),
                spacing_metres.len()
            )));
        }
        if !(context_factor.is_finite() && context_factor >= 1.0) {
            return Err(MpasError::Refusal(format!(
                "a context factor of {context_factor} would draw a frame smaller than the                  fine regions it is meant to surround"
            )));
        }
        let selected: Vec<usize> = (0..n).collect();
        let finest = spacing_metres
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if !finest.is_finite() || finest <= 0.0 {
            return Err(MpasError::Refusal(format!(
                "the finest spacing among the composited regions is {finest} m; a window                  cannot be rendered at it"
            )));
        }
        let selection = format!(
            "{n} fine cell(s) from the composited sources, finest spacing {:.3} km, drawn              into a frame {context_factor:.1}x their own extent so the coarse field they sit              in is visible",
            finest / 1000.0
        );
        Self::window_over_selected_cells(
            latitude_degrees,
            longitude_degrees,
            &selected,
            finest,
            context_factor - 1.0,
            &selection,
            "composite window",
        )
    }

    /// The median spacing of `selected`, which is the resolution a window
    /// over them renders at.
    fn median_spacing_of(cell_spacing_m: &[f64], selected: &[usize]) -> f64 {
        let mut spacings: Vec<f64> = selected.iter().map(|&i| cell_spacing_m[i]).collect();
        let mid = spacings.len() / 2;
        spacings.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        spacings[mid]
    }

    /// The cells of one mesh that make up its refined region.
    ///
    /// The rule is the mesh's own: spacing at or under
    /// `MESH_FOCUS_REFINEMENT_FACTOR` times that mesh's minimum. It needs
    /// no cross-mesh comparison and no tunable, which is why the composite
    /// path reuses it rather than inventing a second definition of
    /// "refined" that could disagree with the window a fine core renders
    /// itself on.
    pub fn refined_region(cell_spacing_m: &[f64]) -> MpasResult<Vec<usize>> {
        let n = cell_spacing_m.len();
        if n == 0 {
            return Err(MpasError::Refusal(
                "a refined region needs at least one cell spacing; none was handed over"
                    .to_string(),
            ));
        }
        let min_spacing = cell_spacing_m.iter().copied().fold(f64::INFINITY, f64::min);
        let max_spacing = cell_spacing_m
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !min_spacing.is_finite() || min_spacing <= 0.0 {
            return Err(MpasError::Refusal(format!(
                "the mesh minimum spacing is {min_spacing}; a refinement ratio measured                  against it is meaningless"
            )));
        }
        if max_spacing <= MESH_FOCUS_REFINEMENT_FACTOR * min_spacing {
            return Err(MpasError::Refusal(format!(
                "this mesh is uniform to within the refinement factor -- spacing runs                  {:.3} km to {:.3} km, a ratio of {:.3}x against the {MESH_FOCUS_REFINEMENT_FACTOR:.2}x                  that marks a refined region -- so there is no refined region to derive a                  window from. Deriving one anyway would centre the window on the unit-vector                  mean of every cell, which for a quasi-uniform global mesh is the origin and                  therefore points nowhere, and would size it to the whole sphere, which no                  Lambert cone represents. Render a uniform mesh with --window global",
                min_spacing / 1000.0,
                max_spacing / 1000.0,
                max_spacing / min_spacing
            )));
        }
        let threshold = mesh_refinement_factor(min_spacing, max_spacing) * min_spacing;
        let selected: Vec<usize> = (0..n).filter(|&i| cell_spacing_m[i] <= threshold).collect();
        // Unreachable while the minimum is a member of the mesh, but a
        // silently empty selection would divide by zero downstream.
        if selected.is_empty() {
            return Err(MpasError::Refusal(format!(
                "no cell has spacing at or under {:.3} km, though the mesh minimum is {:.3} km",
                threshold / 1000.0,
                min_spacing / 1000.0
            )));
        }
        Ok(selected)
    }

    /// A window centred on, and sized to cover, an explicit set of cells.
    ///
    /// gpuwm addition (VENDOR.md): extracted verbatim from `mesh_focus`
    /// rather than copied, because the composite window has to be built by
    /// the same geometry as a single fine core's own window or the two
    /// frames disagree and the composite shows the fine data in the wrong
    /// place. `margin_fraction` is the one thing that differs: a fine core
    /// wants a tight frame, a composite wants surrounding context.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn window_over_selected_cells(
        latitude_degrees: &[f64],
        longitude_degrees: &[f64],
        selected: &[usize],
        median_spacing_m: f64,
        margin_fraction: f64,
        selection: &str,
        label: &str,
    ) -> MpasResult<Window> {
        // --- the centre, by unit-vector mean ------------------------------
        let mut sum = [0.0f64; 3];
        for &i in selected {
            let v = unit_vector(latitude_degrees[i], longitude_degrees[i]);
            for k in 0..3 {
                sum[k] += v[k];
            }
        }
        let count = selected.len() as f64;
        let mean = [sum[0] / count, sum[1] / count, sum[2] / count];
        let norm = (mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2]).sqrt();
        if !norm.is_finite() || norm < MESH_FOCUS_MIN_CENTROID_NORM {
            return Err(MpasError::Refusal(format!(
                "the {} selected cells have a mean direction of magnitude {norm:e}, \
                 under the {MESH_FOCUS_MIN_CENTROID_NORM:e} a centre needs to have a \
                 direction at all: they are spread over the sphere with no middle, so any \
                 point this returned would be float noise",
                selected.len()
            )));
        }
        let up = [mean[0] / norm, mean[1] / norm, mean[2] / norm];
        let centre_lat = asin(up[2].clamp(-1.0, 1.0)) * DEG;
        let centre_lon = atan2(up[1], up[0]) * DEG;

        // --- the horizon guard --------------------------------------------
        let mut max_angle_rad = 0.0f64;
        for &i in selected {
            let v = unit_vector(latitude_degrees[i], longitude_degrees[i]);
            let dot = (v[0] * up[0] + v[1] * up[1] + v[2] * up[2]).clamp(-1.0, 1.0);
            max_angle_rad = max_angle_rad.max(acos(dot));
        }
        if max_angle_rad > MESH_FOCUS_MAX_ANGULAR_RADIUS_RAD {
            return Err(MpasError::Refusal(format!(
                "the refined region reaches {:.2} degrees of arc from its own centre, past \
                 the {:.2} degree horizon a projected window has. A Lambert cone sends the \
                 antipode to infinity, so the far cells would project to a radius with no \
                 bound and nx and ny would be sized from it; a cylindrical window that wide \
                 has wrapped the planet and is covering the same ground twice. A mesh \
                 refined that widely wants --window global",
                max_angle_rad * DEG,
                MESH_FOCUS_MAX_ANGULAR_RADIUS_RAD * DEG
            )));
        }

        let selection = format!(
            "{selection}; centre {} from their unit-vector mean",
            format_centre(centre_lat, centre_lon)
        );

        // --- lat-lon, when the cone will not stand ------------------------
        //
        // Sized in DEGREES, which is exact for a cylindrical grid: the axes
        // ARE the coordinates, so no proxy geometry sits between the region
        // and the box that has to hold it.
        if centre_lat.abs() < MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG {
            let mut half_lat_deg = 0.0f64;
            let mut half_lon_deg = 0.0f64;
            for &i in selected {
                half_lat_deg = half_lat_deg.max((latitude_degrees[i] - centre_lat).abs());
                let delta =
                    (longitude_degrees[i] - centre_lon + 180.0).rem_euclid(360.0) - 180.0;
                half_lon_deg = half_lon_deg.max(delta.abs());
            }
            let nominal_degrees = median_spacing_m / (RAD * WRF_EARTH_RADIUS_M);
            half_lat_deg = (half_lat_deg * (1.0 + margin_fraction)).max(nominal_degrees);
            half_lon_deg = (half_lon_deg * (1.0 + margin_fraction)).max(nominal_degrees);

            let span = (MESH_FOCUS_MAX_POINTS_PER_SIDE - 1) as f64;
            let floor_degrees = (2.0 * half_lat_deg / span).max(2.0 * half_lon_deg / span);
            let mut spacing_degrees = nominal_degrees;
            let mut coarsened_from: Option<f64> = None;
            if spacing_degrees < floor_degrees {
                coarsened_from = Some(spacing_degrees);
                // The hair of headroom keeps the ceiling below from rounding
                // one point past the limit the clamp exists to hold.
                spacing_degrees = floor_degrees * (1.0 + 1.0e-12);
            }
            let points = |half: f64| -> usize {
                ((2.0 * half / spacing_degrees).ceil() as usize + 1)
                    .clamp(2, MESH_FOCUS_MAX_POINTS_PER_SIDE)
            };
            let nx = points(half_lon_deg);
            // The latitude axis has to stay inside [-90, 90]; `coordinates`
            // refuses a window that leaves it.
            let mut ny = points(half_lat_deg);
            let room = 2.0 * (90.0 - centre_lat.abs()) / spacing_degrees;
            if (ny - 1) as f64 > room {
                ny = (room.floor() as usize + 1).max(2);
            }
            let south = centre_lat - 0.5 * (ny - 1) as f64 * spacing_degrees;
            let north = south + (ny - 1) as f64 * spacing_degrees;
            let west = centre_lon - 0.5 * (nx - 1) as f64 * spacing_degrees;
            let east_bound = west + (nx - 1) as f64 * spacing_degrees;
            return Ok(Window::LatLon(LatLonWindow {
                south,
                north,
                west,
                east: east_bound,
                spacing_degrees,
                description: format!(
                    "{label} (lat-lon): {selection}; spacing \
                     {spacing_degrees:.6} degrees is their median {:.3} km{}; {nx}x{ny} covers \
                     their {:.3} by {:.3} degree extent including a {:.0} percent margin. The \
                     centre is {:.2} degrees from the equator, under the \
                     {MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG:.0} a Lambert cone needs: its cone \
                     constant is sin(|latitude|), which is zero on the equator, and the \
                     projection origin carries 1/cone. The longitude axis is allowed past \
                     +/-180 rather than wrapped, because an axis has to increase",
                    median_spacing_m / 1000.0,
                    match coarsened_from {
                        Some(was) => format!(
                            ", coarsened from {was:.6} degrees because that extent needed more \
                             than {MESH_FOCUS_MAX_POINTS_PER_SIDE} points a side"
                        ),
                        None => String::new(),
                    },
                    2.0 * half_lon_deg,
                    2.0 * half_lat_deg,
                    100.0 * margin_fraction,
                    centre_lat.abs()
                ),
            }));
        }

        // --- Lambert ------------------------------------------------------
        let hemi = if centre_lat >= 0.0 { 1.0 } else { -1.0 };
        let abs_lat = centre_lat.abs();
        let half_span = MESH_FOCUS_TRUELAT_HALF_SPAN_DEG.min(0.5 * abs_lat);
        let truelat1 = hemi * (abs_lat - half_span);
        let truelat2 = hemi * (abs_lat + half_span).min(MESH_FOCUS_MAX_ABS_TRUELAT_DEG);

        // The extent is measured ON the projection the window will use, not
        // on a proxy. A proxy is where a sizing goes wrong quietly: a
        // great-circle or azimuthal offset is SHORTER than the Lambert
        // easting across a wide region, so nx would come out too small and
        // the corners the window exists to show would be cropped off it.
        //
        // `dx` cancels out of the measurement -- `set_lc` divides by it once
        // through `rebydx`, so the offsets scale as 1/dx -- and the probe
        // therefore runs at the median spacing and reports metres, which is
        // what the clamp below works in.
        let probe = LambertProjection::set_lc(
            truelat1,
            truelat2,
            centre_lon,
            median_spacing_m,
            centre_lat,
            centre_lon,
            0.0,
            0.0,
        );
        let mut half_width_m = 0.0f64;
        let mut half_height_m = 0.0f64;
        for &i in selected {
            let (gi, gj) = probe.llij(latitude_degrees[i], longitude_degrees[i]);
            if !gi.is_finite() || !gj.is_finite() {
                return Err(MpasError::Refusal(format!(
                    "refined cell {i} at {} projects to ({gi}, {gj}) on the derived Lambert \
                     cone (truelats {truelat1:.3}/{truelat2:.3}); a window cannot be sized \
                     from a coordinate that is not finite",
                    format_centre(latitude_degrees[i], longitude_degrees[i])
                )));
            }
            half_width_m = half_width_m.max(gi.abs() * median_spacing_m);
            half_height_m = half_height_m.max(gj.abs() * median_spacing_m);
        }
        // A one-cell refined region has no extent; floor each half-extent at
        // one dx so the window is a 3x3 rather than a degenerate 2x2.
        half_width_m = (half_width_m * (1.0 + margin_fraction)).max(median_spacing_m);
        half_height_m = (half_height_m * (1.0 + margin_fraction)).max(median_spacing_m);

        // --- the clamp: coarsen, never crop -------------------------------
        let span = (MESH_FOCUS_MAX_POINTS_PER_SIDE - 1) as f64;
        let dx_floor = (2.0 * half_width_m / span).max(2.0 * half_height_m / span);
        let mut dx_metres = median_spacing_m;
        let mut coarsened_from: Option<f64> = None;
        if dx_metres < dx_floor {
            coarsened_from = Some(dx_metres);
            // The hair of headroom keeps the ceiling below from rounding one
            // point past the limit the clamp exists to hold.
            dx_metres = dx_floor * (1.0 + 1.0e-12);
        }
        let points = |half: f64| -> usize {
            ((2.0 * half / dx_metres).ceil() as usize + 1)
                .clamp(2, MESH_FOCUS_MAX_POINTS_PER_SIDE)
        };
        let nx = points(half_width_m);
        let ny = points(half_height_m);

        Ok(Window::Lambert(LambertWindow {
            centre_lat,
            centre_lon,
            dx_metres,
            nx,
            ny,
            truelat1,
            truelat2,
            stand_lon: centre_lon,
            description: format!(
                "{label} (Lambert): {selection}; dx {:.3} km is their \
                 median spacing{}; {nx}x{ny} covers their {:.0} by {:.0} km extent on this \
                 cone including a {:.0} percent margin; truelats {truelat1:.3} and \
                 {truelat2:.3} bracket the centre and stand_lon is the centre longitude",
                dx_metres / 1000.0,
                match coarsened_from {
                    Some(was) => format!(
                        ", coarsened from {:.3} km because that extent needed more than \
                         {MESH_FOCUS_MAX_POINTS_PER_SIDE} points a side",
                        was / 1000.0
                    ),
                    None => String::new(),
                },
                2.0 * half_width_m / 1000.0,
                2.0 * half_height_m / 1000.0,
                100.0 * margin_fraction,
            ),
        }))
    }

    pub fn description(&self) -> &str {
        match self {
            Window::LatLon(w) => &w.description,
            Window::Lambert(w) => &w.description,
        }
    }

    /// The canonical specification, as the weights digest and the emitted
    /// `MPAS_RENDER_WINDOW_SPEC` attribute serialize it: JSON object keys in
    /// sorted order, which is what makes the digest reproducible.
    pub fn spec_json(&self) -> String {
        match self {
            Window::LatLon(w) => format!(
                "{{\"east\": {}, \"kind\": \"latlon\", \"north\": {}, \"south\": {}, \"spacing_degrees\": {}, \"west\": {}}}",
                json_f64(w.east),
                json_f64(w.north),
                json_f64(w.south),
                json_f64(w.spacing_degrees),
                json_f64(w.west)
            ),
            Window::Lambert(w) => format!(
                "{{\"centre_lat\": {}, \"centre_lon\": {}, \"dx_metres\": {}, \"earth_radius_m\": {}, \"kind\": \"lambert\", \"nx\": {}, \"ny\": {}, \"stand_lon\": {}, \"truelat1\": {}, \"truelat2\": {}}}",
                json_f64(w.centre_lat),
                json_f64(w.centre_lon),
                json_f64(w.dx_metres),
                json_f64(WRF_EARTH_RADIUS_M),
                w.nx,
                w.ny,
                json_f64(w.stand_lon),
                json_f64(w.truelat1),
                json_f64(w.truelat2)
            ),
        }
    }

    /// The same specification with no spaces, which is the form the weights
    /// digest hashes (`json.dumps(separators=(",", ":"))`).
    pub fn spec_json_compact(&self) -> String {
        self.spec_json().replace(", ", ",").replace("\": ", "\":")
    }

    /// Target grid shape `(ny, nx)`.
    pub fn shape(&self) -> MpasResult<(usize, usize)> {
        match self {
            Window::Lambert(w) => Ok((w.ny, w.nx)),
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                Ok((lat.len(), lon.len()))
            }
        }
    }

    /// Target latitudes and longitudes, row-major `(ny, nx)`, in degrees.
    pub fn coordinates(&self) -> MpasResult<(Vec<f64>, Vec<f64>)> {
        match self {
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                if lat.iter().any(|v| v.abs() > 90.0) {
                    return Err(MpasError::Refusal(
                        "latitude window leaves [-90, 90]".to_string(),
                    ));
                }
                let mut lat_grid = Vec::with_capacity(lat.len() * lon.len());
                let mut lon_grid = Vec::with_capacity(lat.len() * lon.len());
                for &la in &lat {
                    for &lo in &lon {
                        lat_grid.push(la);
                        lon_grid.push(lo);
                    }
                }
                Ok((lat_grid, lon_grid))
            }
            Window::Lambert(w) => {
                if w.nx < 2 || w.ny < 2 {
                    return Err(MpasError::Refusal(
                        "a Lambert window needs at least 2x2 points".to_string(),
                    ));
                }
                if !w.dx_metres.is_finite() || w.dx_metres <= 0.0 {
                    return Err(MpasError::Refusal(
                        "Lambert dx must be positive and finite".to_string(),
                    ));
                }
                let projection = LambertProjection::set_lc(
                    w.truelat1,
                    w.truelat2,
                    w.stand_lon,
                    w.dx_metres,
                    w.centre_lat,
                    w.centre_lon,
                    0.5 * (w.nx as f64 + 1.0),
                    0.5 * (w.ny as f64 + 1.0),
                );
                let mut lat_grid = Vec::with_capacity(w.nx * w.ny);
                let mut lon_grid = Vec::with_capacity(w.nx * w.ny);
                for j in 1..=w.ny {
                    for i in 1..=w.nx {
                        let (la, lo) = projection.ijll(i as f64, j as f64);
                        lat_grid.push(la);
                        lon_grid.push(lo);
                    }
                }
                Ok((lat_grid, lon_grid))
            }
        }
    }

    /// The WRF global attributes describing this projection, in the order the
    /// emitted file carries them.
    pub fn projection_attributes(&self) -> MpasResult<Vec<(String, ProjAttr)>> {
        match self {
            Window::LatLon(w) => {
                let lat = axis(w.south, w.north, w.spacing_degrees, "latitude")?;
                let lon = axis(w.west, w.east, w.spacing_degrees, "longitude")?;
                let centre_lat = 0.5 * (lat[0] + lat[lat.len() - 1]);
                let centre_lon = 0.5 * (lon[0] + lon[lon.len() - 1]);
                // WRF writes DX/DY in metres even for a lat-lon grid; the
                // nominal spacing at the window centre is the accurate number.
                let metres = w.spacing_degrees * RAD * WRF_EARTH_RADIUS_M;
                Ok(vec![
                    ("MAP_PROJ".into(), ProjAttr::Int(6)),
                    (
                        "MAP_PROJ_CHAR".into(),
                        ProjAttr::Text("Cylindrical Equidistant".into()),
                    ),
                    ("DX".into(), ProjAttr::Float(metres)),
                    ("DY".into(), ProjAttr::Float(metres)),
                    ("TRUELAT1".into(), ProjAttr::Float(0.0)),
                    ("TRUELAT2".into(), ProjAttr::Float(0.0)),
                    ("STAND_LON".into(), ProjAttr::Float(centre_lon)),
                    ("CEN_LAT".into(), ProjAttr::Float(centre_lat)),
                    ("CEN_LON".into(), ProjAttr::Float(centre_lon)),
                    ("MOAD_CEN_LAT".into(), ProjAttr::Float(centre_lat)),
                    ("POLE_LAT".into(), ProjAttr::Float(90.0)),
                    ("POLE_LON".into(), ProjAttr::Float(0.0)),
                ])
            }
            Window::Lambert(w) => {
                let (lat, lon) = self.coordinates()?;
                let centre_j = w.ny / 2;
                let centre_i = w.nx / 2;
                let at = centre_j * w.nx + centre_i;
                Ok(vec![
                    ("MAP_PROJ".into(), ProjAttr::Int(1)),
                    (
                        "MAP_PROJ_CHAR".into(),
                        ProjAttr::Text("Lambert Conformal".into()),
                    ),
                    ("DX".into(), ProjAttr::Float(w.dx_metres)),
                    ("DY".into(), ProjAttr::Float(w.dx_metres)),
                    ("TRUELAT1".into(), ProjAttr::Float(w.truelat1)),
                    ("TRUELAT2".into(), ProjAttr::Float(w.truelat2)),
                    ("STAND_LON".into(), ProjAttr::Float(w.stand_lon)),
                    ("CEN_LAT".into(), ProjAttr::Float(lat[at])),
                    ("CEN_LON".into(), ProjAttr::Float(lon[at])),
                    ("MOAD_CEN_LAT".into(), ProjAttr::Float(lat[at])),
                    ("POLE_LAT".into(), ProjAttr::Float(90.0)),
                    ("POLE_LON".into(), ProjAttr::Float(0.0)),
                ])
            }
        }
    }
}

/// A spacing rounded DOWN to three significant figures: 14,987 m -> 14,900 m,
/// 3,012 m -> 3,010 m, 15,000 m -> 15,000 m. Down, so a derived grid is
/// never coarser than the mesh it samples; three figures, so the label a
/// reader sees (`Δx 14.9 km`) is not a nine-digit float.
pub fn finer_round_spacing(spacing_m: f64) -> f64 {
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return spacing_m;
    }
    let step = pow(10.0, log10(spacing_m).floor() - 2.0);
    // The hair keeps an exact 15,000 from flooring to 14,900 through a
    // division that lands one ulp under 150.
    let rounded = ((spacing_m / step) * (1.0 + 1.0e-12)).floor() * step;
    rounded.min(spacing_m)
}

/// The point on the unit sphere a latitude/longitude pair names.
fn unit_vector(lat_degrees: f64, lon_degrees: f64) -> [f64; 3] {
    let lat = lat_degrees * RAD;
    let lon = lon_degrees * RAD;
    let cos_lat = cos(lat);
    [cos_lat * cos(lon), cos_lat * sin(lon), sin(lat)]
}

/// A centre as a reader says it out loud: `17.312N 53.897W`.
fn format_centre(lat: f64, lon: f64) -> String {
    format!(
        "{:.3}{} {:.3}{}",
        lat.abs(),
        if lat >= 0.0 { "N" } else { "S" },
        lon.abs(),
        if lon >= 0.0 { "E" } else { "W" }
    )
}

/// Render a float the way `json.dumps` does, so the digest input matches.
fn json_f64(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

fn axis(first: f64, last: f64, step: f64, role: &str) -> MpasResult<Vec<f64>> {
    if !(first.is_finite() && last.is_finite() && step.is_finite()) {
        return Err(MpasError::Refusal(format!(
            "{role} window bounds must be finite"
        )));
    }
    if last <= first || step <= 0.0 {
        return Err(MpasError::Refusal(format!("{role} window must increase")));
    }
    let count = ((last - first) / step).round() as usize + 1;
    let mut values: Vec<f64> = (0..count).map(|k| first + k as f64 * step).collect();
    let end = values[count - 1];
    if (end - last).abs() > 1.0e-9 {
        return Err(MpasError::Refusal(format!(
            "{role} spacing does not exactly partition its bounds"
        )));
    }
    values[count - 1] = last;
    Ok(values)
}

/// WRF `module_llxy.F` Lambert state.
#[derive(Debug, Clone, Copy)]
pub struct LambertProjection {
    cone: f64,
    hemi: f64,
    rebydx: f64,
    polei: f64,
    polej: f64,
    truelat1: f64,
    truelat2: f64,
    stand_lon: f64,
}

fn lambert_cone(truelat1: f64, truelat2: f64) -> f64 {
    if (truelat1 - truelat2).abs() > 0.1 {
        let numerator = log10(cos(truelat1 * RAD)) - log10(cos(truelat2 * RAD));
        let denominator = log10(tan((45.0 - truelat1.abs() / 2.0) * RAD))
            - log10(tan((45.0 - truelat2.abs() / 2.0) * RAD));
        numerator / denominator
    } else {
        sin(truelat1.abs() * RAD)
    }
}

impl LambertProjection {
    /// WRF `set_lc`, transcribed.
    #[allow(clippy::too_many_arguments)]
    pub fn set_lc(
        truelat1: f64,
        truelat2: f64,
        stand_lon: f64,
        dx_metres: f64,
        known_lat: f64,
        known_lon: f64,
        known_i: f64,
        known_j: f64,
    ) -> Self {
        let cone = lambert_cone(truelat1, truelat2);
        let hemi = if truelat1 >= 0.0 { 1.0 } else { -1.0 };
        let rebydx = WRF_EARTH_RADIUS_M / dx_metres;
        let mut delta_lon = known_lon - stand_lon;
        if delta_lon > 180.0 {
            delta_lon -= 360.0;
        }
        if delta_lon < -180.0 {
            delta_lon += 360.0;
        }
        let ctl1r = cos(truelat1 * RAD);
        let rsw = rebydx * ctl1r / cone
            * pow(
                tan((90.0 * hemi - known_lat) * RAD / 2.0)
                    / tan((90.0 * hemi - truelat1) * RAD / 2.0),
                cone,
            );
        let arg = cone * (delta_lon * RAD);
        let polei = hemi * known_i - hemi * rsw * sin(arg);
        let polej = hemi * known_j + rsw * cos(arg);
        LambertProjection {
            cone,
            hemi,
            rebydx,
            polei,
            polej,
            truelat1,
            truelat2,
            stand_lon,
        }
    }

    /// WRF `llij_lc`, transcribed. Returns `(i, j)`, the grid coordinate a
    /// latitude/longitude pair lands on, in the same 1-based convention
    /// [`LambertProjection::ijll`] reads back.
    ///
    /// The inverse alone was enough while every window was a fixed box. A
    /// DERIVED window has to ask the opposite question -- how many points does
    /// it take to cover these cells -- and answering it with a great-circle or
    /// azimuthal proxy undersizes a wide region, because those offsets are
    /// shorter than the Lambert easting they stand in for. Measuring on the
    /// cone the window will actually use makes the coverage exact instead.
    pub fn llij(&self, latitude: f64, longitude: f64) -> (f64, f64) {
        let hemi = self.hemi;
        let mut delta_lon = longitude - self.stand_lon;
        if delta_lon > 180.0 {
            delta_lon -= 360.0;
        }
        if delta_lon < -180.0 {
            delta_lon += 360.0;
        }
        let ctl1r = cos(self.truelat1 * RAD);
        let rm = self.rebydx * ctl1r / self.cone
            * pow(
                tan((90.0 * hemi - latitude) * RAD / 2.0)
                    / tan((90.0 * hemi - self.truelat1) * RAD / 2.0),
                self.cone,
            );
        let arg = self.cone * (delta_lon * RAD);
        let i = self.polei + hemi * rm * sin(arg);
        let j = self.polej - rm * cos(arg);
        (hemi * i, hemi * j)
    }

    /// WRF `ijll_lc`, transcribed. Returns `(latitude, longitude)` in degrees.
    pub fn ijll(&self, i: f64, j: f64) -> (f64, f64) {
        let hemi = self.hemi;
        let chi1 = (90.0 - hemi * self.truelat1) * RAD;
        let chi2 = (90.0 - hemi * self.truelat2) * RAD;
        let inew = hemi * i;
        let jnew = hemi * j;
        let xx = inew - self.polei;
        let yy = self.polej - jnew;
        let radius_squared = xx * xx + yy * yy;
        let radius = radius_squared.sqrt() / self.rebydx;
        let mut longitude = self.stand_lon + DEG * atan2(hemi * xx, yy) / self.cone;
        longitude = (longitude + 360.0).rem_euclid(360.0);
        let chi = if (chi1 - chi2).abs() < 1.0e-12 {
            2.0 * atan(pow(radius / tan(chi1), 1.0 / self.cone) * tan(chi1 * 0.5))
        } else {
            2.0 * atan(pow(radius * self.cone / sin(chi1), 1.0 / self.cone) * tan(chi1 * 0.5))
        };
        let mut latitude = (90.0 - chi * DEG) * hemi;
        if radius_squared == 0.0 {
            latitude = hemi * 90.0;
            longitude = self.stand_lon;
        }
        if longitude > 180.0 {
            longitude -= 360.0;
        } else if longitude < -180.0 {
            longitude += 360.0;
        }
        (latitude, longitude)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest as _;

    /// A mesh as this derivation sees one: `(latitude, longitude, spacing)`
    /// per cell, a refined patch inside a coarse background. Nothing else is
    /// read, so a synthetic mesh built this way exercises exactly the code a
    /// real one does.
    ///
    /// `west > east` means the patch crosses the antimeridian and is walked
    /// through it.
    fn patch_in_background(
        south: f64,
        north: f64,
        west: f64,
        east: f64,
        core_spacing_m: f64,
        background_spacing_m: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        let mut spacing = Vec::new();
        let mut la = -85.0;
        while la <= 85.0 {
            let mut lo = -180.0;
            while lo < 180.0 {
                lat.push(la);
                lon.push(lo);
                spacing.push(background_spacing_m);
                lo += 5.0;
            }
            la += 5.0;
        }
        let down = ((north - south) / 0.2).round() as usize;
        let width = if east >= west { east - west } else { east + 360.0 - west };
        let across = (width / 0.2).round() as usize;
        for j in 0..=down {
            for i in 0..=across {
                lat.push(south + j as f64 * 0.2);
                lon.push(((west + i as f64 * 0.2) + 180.0).rem_euclid(360.0) - 180.0);
                spacing.push(core_spacing_m);
            }
        }
        (lat, lon, spacing)
    }

    /// Great-circle distance from a point to the nearest point of a window's
    /// grid, in km. This is what "covers" has to mean: a bounding box in
    /// latitude and longitude says nothing about a Lambert grid, whose edges
    /// are curved and whose corners are the part that gets cropped.
    fn nearest_grid_point_km(grid: &(Vec<f64>, Vec<f64>), lat: f64, lon: f64) -> f64 {
        let t = unit_vector(lat, lon);
        let mut best = f64::INFINITY;
        for k in 0..grid.0.len() {
            let v = unit_vector(grid.0[k], grid.1[k]);
            let chord = ((v[0] - t[0]).powi(2) + (v[1] - t[1]).powi(2) + (v[2] - t[2]).powi(2))
                .sqrt();
            let km = 2.0 * (chord / 2.0).clamp(0.0, 1.0).asin() * MPAS_EARTH_RADIUS_M / 1000.0;
            if km < best {
                best = km;
            }
        }
        best
    }

    fn coordinate_digest(window: &Window) -> String {
        let (lat, lon) = window.coordinates().unwrap();
        let mut hash = sha2::Sha256::new();
        for v in lat.iter().chain(lon.iter()) {
            hash.update(v.to_le_bytes());
        }
        crate::weights::hex(&hash.finalize())
    }

    /// The `focus` window's coordinate digest: ONE value, on every platform.
    ///
    /// History (#373, then 2026-10-21). Until 2026-10-21 the Lambert path
    /// called `f64::atan`, `f64::powf` and friends, which go to the platform C
    /// library, so this pin was a per-platform table: a `linux-gnu` row
    /// (glibc 2.43, `551a1ff1...`) and a `windows-msvc` row (`34e3521f...`,
    /// measured on an AMD desktop). The `windows-msvc` row was not even a
    /// platform constant: the SAME test binary gave a different digest on an
    /// Intel Xeon 8488C under Windows Server 2022 (UCRT 10.0.20348) than on
    /// the AMD Ryzen 9 9950X3D desktop (UCRT 10.0.26100), with `pow` the
    /// first call to disagree and `atan2`/`atan` after it. A Windows user's
    /// window coordinates depended on their CPU and Windows build.
    ///
    /// The path now calls the vendored `libm` (pure IEEE arithmetic,
    /// `force-soft-floats`), so the bytes belong to this source and nothing
    /// else, and the old rows were retired rather than kept: no build can
    /// produce them any more. This value was derived with the portable path
    /// and measured identical on Windows/AMD, Windows/Intel and Linux. The
    /// recorded `weights_sha256` values are taken over the window spec and
    /// the nearest-cell INDEX map, not over these bytes, and the coordinates
    /// moved by at most a few ULP (nanometres), so they stand.
    const FROZEN_FOCUS_COORDINATE_DIGEST: &str =
        "193deaf48c4b858ccbd78c2742936cab49988ac3f914d4475ee28441de493638";

    /// The `focus` grid's corner and centre coordinates, in degrees, to a
    /// tolerance that no libm can reach and no real spec change can hide in.
    ///
    /// THE BREAKAGE THIS PREVENTS is the one the digest was always for: a
    /// corner nudged by a metre invalidates every recorded `weights_sha256`
    /// while every filename stays the same. This is the half of that gate
    /// which holds on EVERY platform, including one nobody has measured yet.
    ///
    /// The tolerance is chosen from both sides, measured: the largest
    /// cross-libm disagreement in these coordinates is 1.42e-14 degrees, and
    /// one metre of ground is 9.0e-6 degrees. `1e-9` degrees is 0.11 mm: five
    /// orders of magnitude above the noise this cannot be allowed to trip on,
    /// and four orders below the smallest movement it must catch.
    const FOCUS_GEOMETRY_TOLERANCE_DEG: f64 = 1.0e-9;
    const FROZEN_FOCUS_GEOMETRY: &[(&str, usize, f64, f64)] = &[
        ("first point (j=1, i=1)", 0, 18.955_624_884_995_8, -119.806_558_555_523_42),
        ("last point (j=ny, i=nx)", 35_999, 46.496_632_054_296_91, -59.143_091_166_431_93),
    ];

    /// The two fixed windows may not move, ever. Their `weights_sha256`
    /// values are recorded in evidence already written, and the digest is
    /// taken over the window spec and the nearest-cell index map, so a corner
    /// nudged by a metre invalidates every recorded digest while every
    /// filename stays the same. This pins the spec string the digest hashes
    /// AND the coordinates the index map is built from.
    ///
    /// The geometry gate catches a moved corner; the exact digest catches a
    /// moved BIT. Both are one value on every platform, because the Lambert
    /// path calls the vendored `libm` and no platform C library (see
    /// `FROZEN_FOCUS_COORDINATE_DIGEST`).
    #[test]
    fn the_two_fixed_windows_are_frozen() {
        let focus = Window::named("focus").unwrap();
        assert_eq!(
            focus.spec_json(),
            "{\"centre_lat\": 37.0, \"centre_lon\": -96.0, \"dx_metres\": 22000.0, \
             \"earth_radius_m\": 6370000.0, \"kind\": \"lambert\", \"nx\": 240, \"ny\": 150, \
             \"stand_lon\": -96.0, \"truelat1\": 30.0, \"truelat2\": 60.0}"
        );
        assert_eq!(
            focus.spec_json_compact(),
            "{\"centre_lat\":37.0,\"centre_lon\":-96.0,\"dx_metres\":22000.0,\
             \"earth_radius_m\":6370000.0,\"kind\":\"lambert\",\"nx\":240,\"ny\":150,\
             \"stand_lon\":-96.0,\"truelat1\":30.0,\"truelat2\":60.0}"
        );
        assert_eq!(focus.shape().unwrap(), (150, 240));

        // The portable half, asserted on every platform including an
        // unmeasured one: the grid is still where it was.
        let (lat, lon) = focus.coordinates().unwrap();
        assert_eq!(lat.len(), 36_000, "the focus grid changed size");
        for (label, index, want_lat, want_lon) in FROZEN_FOCUS_GEOMETRY {
            let dlat = (lat[*index] - want_lat).abs();
            let dlon = (lon[*index] - want_lon).abs();
            assert!(
                dlat < FOCUS_GEOMETRY_TOLERANCE_DEG && dlon < FOCUS_GEOMETRY_TOLERANCE_DEG,
                "the focus window's {label} MOVED: got ({}, {}), recorded ({want_lat}, \
                 {want_lon}), off by ({dlat:.3e}, {dlon:.3e}) degrees against a tolerance of \
                 {FOCUS_GEOMETRY_TOLERANCE_DEG:.0e}. That tolerance is 0.11 mm on the ground \
                 and the largest cross-libm disagreement ever measured here is 1.42e-14 \
                 degrees, so this is not floating-point noise -- the projection or the spec \
                 changed. Every `weights_sha256` recorded in evidence was taken over this \
                 grid and none of them reproduces now, while every filename stays the same.",
                lat[*index],
                lon[*index],
            );
        }

        // The exact half: the same bytes on every CPU and OS.
        assert_eq!(
            coordinate_digest(&focus),
            FROZEN_FOCUS_COORDINATE_DIGEST,
            "the focus window's coordinate digest MOVED. The Lambert path calls only \
             IEEE arithmetic and the vendored `libm`, so this is the same on every \
             platform by construction: a move is a change to the Lambert transcription, \
             to the window spec, or to the `libm` version, not a property of this box. \
             If the geometry gate above passed, the grid is where it was to within \
             0.11 mm and only BITS changed; find the change before touching the pin."
        );

        let global = Window::named("global").unwrap();
        assert_eq!(
            global.spec_json(),
            "{\"east\": 179.75, \"kind\": \"latlon\", \"north\": 90.0, \"south\": -90.0, \
             \"spacing_degrees\": 0.25, \"west\": -180.0}"
        );
        assert_eq!(
            global.spec_json_compact(),
            "{\"east\":179.75,\"kind\":\"latlon\",\"north\":90.0,\"south\":-90.0,\
             \"spacing_degrees\":0.25,\"west\":-180.0}"
        );
        assert_eq!(global.shape().unwrap(), (721, 1440));
        // ONE value, on every platform, and that is a measured fact rather
        // than an assumption: `global` is a lat-lon window, its coordinates
        // come from `axis()` -- `first + k as f64 * step` -- and no
        // transcendental is called anywhere in that path. Measured 2026-08-27
        // on Windows/MSVC and on glibc 2.43: identical. This is the control
        // that proves the `focus` split above is the libm and not the source.
        assert_eq!(
            coordinate_digest(&global),
            "ce6e916e0c01874f7a21296232706eb9855d9422be3e255fc9201352efb5a4d9",
            "the global window's coordinate digest moved. Unlike `focus` this one \
             is platform-independent by construction -- pure `first + k * step`, no \
             libm call -- so a move here is a real change to the window spec or to \
             `axis()`, and every recorded `weights_sha256` over the global grid \
             stops reproducing while every filename stays the same."
        );
    }

    #[test]
    fn explicit_latlon_does_not_depend_on_the_segment_mesh() {
        let a = Window::named("latlon:38,42,-86,-82,0.01").unwrap();
        assert_eq!(a.shape().unwrap(),(401,401));
        let (lat,lon) = a.coordinates().unwrap();
        assert_eq!((lat[0],lon[0]),(38.0,-86.0));
        assert_eq!((*lat.last().unwrap(),*lon.last().unwrap()),(42.0,-82.0));
        for spec in ["latlon:38,42,-86,-82,0.03", "latlon:38,42,-86,-82,0.0001",
            "latlon:38,42,-86,-82,NaN", "latlon:91,92,-86,-82,0.01"] {
            assert!(Window::named(spec).is_err(),"{spec}");
        }
    }

    /// The geometry gate, validated in the direction that matters: it must
    /// SEE a corner nudged by a metre.
    ///
    /// A tolerance chosen to sit above floating-point noise is worthless
    /// unless someone proves it still sits below the breakage. This nudges
    /// `focus`'s centre by one metre of latitude (9.0e-6 degrees) (the
    /// movement the frozen test's own docstring names as fatal to every
    /// recorded `weights_sha256`) and asserts the gate's arithmetic rejects
    /// it by a wide margin. Without this test, `FOCUS_GEOMETRY_TOLERANCE_DEG`
    /// could be loosened to 1.0 and nothing would notice.
    #[test]
    fn the_geometry_gate_sees_a_corner_nudged_by_one_metre() {
        const ONE_METRE_OF_LATITUDE_DEG: f64 = 9.0e-6;
        let Window::Lambert(mut spec) = Window::named("focus").unwrap() else {
            unreachable!("focus is a Lambert window");
        };
        spec.centre_lat += ONE_METRE_OF_LATITUDE_DEG;
        let (lat, lon) = Window::Lambert(spec).coordinates().unwrap();

        let mut caught = 0usize;
        for (label, index, want_lat, want_lon) in FROZEN_FOCUS_GEOMETRY {
            let dlat = (lat[*index] - want_lat).abs();
            let dlon = (lon[*index] - want_lon).abs();
            if dlat >= FOCUS_GEOMETRY_TOLERANCE_DEG || dlon >= FOCUS_GEOMETRY_TOLERANCE_DEG {
                caught += 1;
            }
            assert!(
                dlat < 1.0e-3,
                "{label} moved {dlat:.3e} degrees for a one-metre nudge, which is far more \
                 than a metre -- the nudge is not doing what this control claims"
            );
        }
        assert_eq!(
            caught,
            FROZEN_FOCUS_GEOMETRY.len(),
            "the geometry gate did NOT see a one-metre nudge at every pinned point. \
             Its tolerance {FOCUS_GEOMETRY_TOLERANCE_DEG:.0e} degrees is too loose to \
             catch the breakage the frozen window exists to prevent, and the exact \
             digest is then the only thing holding that grid in place."
        );
    }

    /// The control #373 used to show the old per-platform `focus` rows were
    /// the C library and not the compiler: the `global` window's coordinates
    /// call no transcendental at all, so they must digest identically on
    /// every platform. Kept now that `focus` is portable too, because it pins
    /// WHY `global` is: exact `first + k * step` arithmetic.
    #[test]
    fn the_transcendental_free_window_needs_no_platform_row() {
        let global = Window::named("global").unwrap();
        let (lat, lon) = global.coordinates().unwrap();
        assert_eq!(lat.len(), 721 * 1440);
        // Every coordinate is exactly representable as `first + k * 0.25`,
        // which is what makes this digest portable. Assert the property, not
        // just the digest, so the reason survives a refactor of `axis()`.
        for (k, v) in lat.iter().take(1440 * 8).enumerate() {
            let row = (k / 1440) as f64;
            assert_eq!(
                *v,
                -90.0 + row * 0.25,
                "global latitude {k} is not exactly -90 + row*0.25; the moment this \
                 path acquires arithmetic that is not a multiply and an add, its \
                 digest becomes platform-specific and needs a table like `focus`'s"
            );
        }
        assert!(lon.iter().all(|v| v.is_finite()));
    }

    /// The measured defect, as a test: neither fixed window can render a
    /// refined core in the tropical Atlantic. `focus` puts no grid point
    /// anywhere near it, and `global` is 6x too coarse to hold it. This is
    /// the breakage the derived window prevents, so it is asserted rather
    /// than asserted about.
    #[test]
    fn neither_fixed_window_can_render_a_tropical_atlantic_core() {
        let focus = Window::named("focus").unwrap().coordinates().unwrap();
        let inside = (0..focus.0.len())
            .filter(|&k| {
                (14.6..=20.0).contains(&focus.0[k]) && (-57.3..=-50.5).contains(&focus.1[k])
            })
            .count();
        assert_eq!(
            inside, 0,
            "the CONUS focus box must contain no part of a 15-20 N, 50-57 W core"
        );

        let global = match Window::named("global").unwrap() {
            Window::LatLon(w) => w.spacing_degrees * RAD * WRF_EARTH_RADIUS_M,
            Window::Lambert(_) => unreachable!("global is a lat-lon window"),
        };
        assert!(
            global / 4_530.0 > 6.0,
            "the global overview is {global:.0} m, only {:.1}x a 4.53 km core",
            global / 4_530.0
        );
    }

    /// The case that forced this: a refined core at roughly 14.6-20.0 N,
    /// 50.5-57.3 W attaining 4.53 km inside a 75 km background.
    #[test]
    fn a_tropical_atlantic_core_derives_a_window_over_itself() {
        let (lat, lon, spacing) =
            patch_in_background(14.6, 20.0, -57.3, -50.5, 4_530.0, 75_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("17 N is well inside the Lambert branch"),
        };
        assert!(
            (w.centre_lat - 17.310).abs() < 0.01,
            "centre latitude {} is not the core's",
            w.centre_lat
        );
        assert!(
            (w.centre_lon + 53.900).abs() < 0.01,
            "centre longitude {} is not the core's",
            w.centre_lon
        );
        // The median of a uniform core is the core's own spacing, exactly.
        assert_eq!(w.dx_metres, 4_530.0);
        assert_eq!((w.nx, w.ny), (179, 148));
        // Truelats bracket the centre, both in the northern hemisphere.
        assert!((w.truelat1 - (w.centre_lat - 5.0)).abs() < 1.0e-9);
        assert!((w.truelat2 - (w.centre_lat + 5.0)).abs() < 1.0e-9);
        assert!(w.truelat1 > 0.0 && w.stand_lon == w.centre_lon);

        // Coverage, corner by corner: every corner of the core has a grid
        // point within half a cell of it. A window that merely brackets the
        // core in latitude and longitude would pass a weaker test and still
        // crop the corners.
        let grid = window.coordinates().unwrap();
        for (clat, clon) in [(14.6, -57.3), (14.6, -50.5), (20.0, -57.3), (20.0, -50.5)] {
            let km = nearest_grid_point_km(&grid, clat, clon);
            assert!(
                km <= 4.53,
                "core corner {clat} {clon} is {km:.3} km from the nearest grid point, \
                 further than one cell"
            );
        }
        assert!(window.description().contains("2.00x the mesh minimum 4.530 km"));
    }

    /// A naive mean of longitudes puts a core straddling the antimeridian at
    /// roughly the prime meridian -- half a planet from where it is. The
    /// unit-vector mean puts it on the antimeridian, which is where it is.
    #[test]
    fn a_core_across_the_antimeridian_stays_on_the_antimeridian() {
        let (lat, lon, spacing) = patch_in_background(30.0, 40.0, 175.0, -175.0, 3_000.0, 60_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("35 N is well inside the Lambert branch"),
        };
        assert!(
            w.centre_lon.abs() > 179.9,
            "centre longitude {} is not on the antimeridian; an arithmetic mean of these \
             longitudes lands near 0 and that is the bug this guards",
            w.centre_lon
        );
        assert!((w.centre_lat - 35.0).abs() < 0.1, "{}", w.centre_lat);

        // And the grid genuinely straddles the line rather than sitting on
        // one side of it.
        let grid = window.coordinates().unwrap();
        assert!(grid.1.iter().any(|&v| v > 176.0), "no grid point east of 176 E");
        assert!(grid.1.iter().any(|&v| v < -176.0), "no grid point west of 176 W");
        for (clat, clon) in [(30.0, 175.0), (30.0, -175.0), (40.0, 175.0), (40.0, -175.0)] {
            assert!(nearest_grid_point_km(&grid, clat, clon) <= 3.0, "{clat} {clon}");
        }
    }

    /// A core on the equator has no Lambert cone to sit on, so it gets the
    /// cylindrical grid instead -- at its own spacing, not the 0.25 degree
    /// overview's.
    #[test]
    fn a_near_equatorial_core_falls_back_to_lat_lon() {
        let (lat, lon, spacing) = patch_in_background(-2.0, 2.0, -20.0, -14.0, 3_000.0, 60_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::LatLon(w) => w.clone(),
            Window::Lambert(_) => panic!("a centre on the equator has no conditioned cone"),
        };
        // The derived spacing is the core's own 3 km, in degrees.
        let metres = w.spacing_degrees * RAD * WRF_EARTH_RADIUS_M;
        assert!((metres - 3_000.0).abs() < 1.0, "{metres} m is not the core's 3 km");
        assert!(w.south < -2.0 && w.north > 2.0, "{} to {}", w.south, w.north);
        assert!(w.west < -20.0 && w.east > -14.0, "{} to {}", w.west, w.east);
        assert!(window.description().contains("under the 10 a Lambert cone needs"));
        // The axes must still partition exactly, or `coordinates` refuses.
        let grid = window.coordinates().unwrap();
        for (clat, clon) in [(-2.0, -20.0), (-2.0, -14.0), (2.0, -20.0), (2.0, -14.0)] {
            assert!(nearest_grid_point_km(&grid, clat, clon) <= 3.0, "{clat} {clon}");
        }
    }

    /// Just far enough from the equator to keep the cone, and the fallback
    /// must not fire: the branch is a threshold, not a wide band.
    #[test]
    fn the_lambert_branch_holds_at_its_own_threshold() {
        let below = patch_in_background(8.0, 10.0, -20.0, -18.0, 3_000.0, 60_000.0);
        assert!(
            matches!(
                Window::mesh_focus(&below.0, &below.1, &below.2).unwrap(),
                Window::LatLon(_)
            ),
            "a centre at 9 N is under the threshold and must be cylindrical"
        );
        let above = patch_in_background(11.0, 13.0, -20.0, -18.0, 3_000.0, 60_000.0);
        let window = Window::mesh_focus(&above.0, &above.1, &above.2).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("a centre at 12 N is over the threshold"),
        };
        assert!(w.centre_lat >= MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG);
        let grid = window.coordinates().unwrap();
        assert!(grid.0.iter().chain(grid.1.iter()).all(|v| v.is_finite()));
    }

    /// The measurement behind [`MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG`]: on the
    /// equator the cone is zero, the projection origin is infinite and every
    /// coordinate is NaN, while at the threshold the same transcription is
    /// exact. Both halves are asserted, because a threshold that has only
    /// been checked on the good side has not been checked.
    #[test]
    fn the_cone_is_why_a_near_equatorial_centre_falls_back() {
        // On the equator: 0/0.
        let cone = lambert_cone(0.0, 0.0);
        assert_eq!(cone, 0.0, "a cone of zero is what a 1/cone divides by");
        let dead = LambertProjection::set_lc(0.0, 0.0, 0.0, 4_530.0, 0.0, 0.0, 90.0, 74.0);
        assert!(!dead.polej.is_finite(), "polej is {}", dead.polej);
        let (bad_lat, bad_lon) = dead.ijll(90.0, 74.0);
        assert!(
            bad_lat.is_nan() && bad_lon.is_nan(),
            "an equatorial cone must be visibly dead, not quietly wrong: got {bad_lat} {bad_lon}"
        );

        // At the threshold: cone 0.174, origin 8,008 grid units off, and the
        // centre round-trips to well under a millimetre.
        let centre = MESH_FOCUS_MIN_LAMBERT_LATITUDE_DEG;
        let (t1, t2) = (centre - 5.0, centre + 5.0);
        let cone = lambert_cone(t1, t2);
        assert!((cone - 0.1739).abs() < 1.0e-3, "cone is {cone}");
        let live = LambertProjection::set_lc(t1, t2, -53.9, 4_530.0, centre, -53.9, 90.0, 74.0);
        assert!((live.polej - 8_008.4).abs() < 1.0, "polej is {}", live.polej);
        let (lat, lon) = live.ijll(90.0, 74.0);
        let error_m = (lat - centre).abs() * RAD * WRF_EARTH_RADIUS_M;
        assert!(error_m < 1.0e-3, "centre round-trip is off by {error_m:e} m");
        assert!((lon + 53.9).abs() < 1.0e-9, "{lon}");
    }

    /// The forward transform the sizing depends on has to be the inverse of
    /// the one the coordinates come from, or the window is sized on one map
    /// and rendered on another.
    #[test]
    fn the_lambert_transforms_invert_each_other() {
        let projection =
            LambertProjection::set_lc(30.0, 60.0, -96.0, 22_000.0, 37.0, -96.0, 120.5, 75.5);
        for (i, j) in [(1.0, 1.0), (120.5, 75.5), (240.0, 150.0), (17.0, 133.0)] {
            let (lat, lon) = projection.ijll(i, j);
            let (bi, bj) = projection.llij(lat, lon);
            assert!(
                (bi - i).abs() < 1.0e-9 && (bj - j).abs() < 1.0e-9,
                "({i}, {j}) went to {lat} {lon} and came back ({bi}, {bj})"
            );
        }
    }

    /// A pathological mesh must not be able to ask for a target grid that
    /// cannot be written. The clamp coarsens dx; it does not crop, and the
    /// region it was derived from is still fully inside the result.
    #[test]
    fn the_clamp_coarsens_dx_and_still_covers_the_region() {
        let (lat, lon, spacing) =
            patch_in_background(20.0, 45.0, -110.0, -70.0, 1_000.0, 60_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("33 N is well inside the Lambert branch"),
        };
        assert!(w.nx <= MESH_FOCUS_MAX_POINTS_PER_SIDE, "nx is {}", w.nx);
        assert!(w.ny <= MESH_FOCUS_MAX_POINTS_PER_SIDE, "ny is {}", w.ny);
        assert_eq!(w.nx, MESH_FOCUS_MAX_POINTS_PER_SIDE, "the clamp must be the binding side");
        assert!(
            w.dx_metres > 1_000.0,
            "dx {} was not coarsened, so the clamp cropped instead",
            w.dx_metres
        );
        assert!(window.description().contains("coarsened from 1.000 km"));

        // Coverage survives the clamp: every corner of the region still has a
        // grid point within a cell of it.
        let grid = window.coordinates().unwrap();
        for (clat, clon) in [(20.0, -110.0), (20.0, -70.0), (45.0, -110.0), (45.0, -70.0)] {
            let km = nearest_grid_point_km(&grid, clat, clon);
            assert!(
                km <= w.dx_metres / 1000.0,
                "corner {clat} {clon} is {km:.3} km out after the clamp"
            );
        }
    }

    /// A uniform mesh has no refined region, and the answer is a refusal that
    /// says so and says what to run instead.
    #[test]
    fn a_uniform_mesh_is_refused_by_name() {
        let lat = vec![10.0, 20.0, 30.0];
        let lon = vec![0.0, 10.0, 20.0];
        let error = Window::mesh_focus(&lat, &lon, &[60_000.0; 3])
            .unwrap_err()
            .to_string();
        assert!(error.contains("uniform"), "{error}");
        assert!(error.contains("no refined region"), "{error}");
        assert!(error.contains("--window global"), "{error}");
        assert!(error.contains("60.000 km"), "{error}");
    }

    /// The refinement factor is a closed boundary: a mesh exactly at the
    /// factor is uniform, a hair past it is not. A boundary nobody pinned is
    /// a boundary that moves.
    #[test]
    fn the_refinement_factor_boundary_is_closed() {
        let lat = vec![10.0, 20.0, 30.0];
        let lon = vec![0.0, 10.0, 20.0];
        assert!(
            Window::mesh_focus(&lat, &lon, &[30_000.0, 60_000.0, 45_000.0]).is_err(),
            "a 2.000x ratio is uniform to within the factor"
        );
        // A hair past the factor the mesh is refined, and its cut is the geometric mean of
        // its own extremes (1.41x here), so the 45 km cell is not the core.
        let window =
            Window::mesh_focus(&lat, &lon, &[30_000.0, 60_001.0, 45_000.0]).unwrap();
        assert!(window.description().contains("1 of 3 cells"), "{}", window.description());
        assert!(window.description().contains("1.41x"), "{}", window.description());
    }

    /// A 2x zoom globe whose background is finer near the icosahedron's
    /// corners frames its core, not the planet. The case that forced this:
    /// the 120 to 60 km zoom globe over North America (zoomna, 2026-11-14),
    /// spacing 51.7 to 126 km, with 100 to 110 km corner patches that the
    /// 2.0x cut (103 km) took as refined; the door refused with "the refined
    /// region reaches 162.36 degrees of arc".
    #[test]
    fn a_2x_zoom_globe_with_finer_corner_patches_frames_its_core() {
        let mut lat = Vec::new();
        let mut lon = Vec::new();
        let mut spacing = Vec::new();
        // the core over North America, 52 to 62 km
        for i in 0..10 {
            for j in 0..10 {
                lat.push(30.0 + 2.0 * i as f64);
                lon.push(-115.0 + 3.0 * j as f64);
                spacing.push(52_000.0 + 1_000.0 * ((i + j) % 10) as f64);
            }
        }
        // the background, 120 to 126 km, and two corner patches at 100 to 104 km
        // on the far side of the planet
        for k in 0..36 {
            lat.push(-60.0 + 3.0 * (k % 12) as f64);
            lon.push(60.0 + 10.0 * (k / 12) as f64);
            spacing.push(120_000.0 + 6_000.0 * (k % 2) as f64);
        }
        for (la, lo) in [(-30.0, 120.0), (10.0, 80.0)] {
            lat.push(la);
            lon.push(lo);
            spacing.push(100_000.0);
            lat.push(la + 1.0);
            lon.push(lo);
            spacing.push(104_000.0);
        }
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        assert!(window.description().contains("100 of 140 cells"), "{}", window.description());
    }

    /// A mesh refined 4x or more keeps the standing 2.0x cut.
    #[test]
    fn a_4x_zoom_globe_keeps_the_standing_cut() {
        assert_eq!(mesh_refinement_factor(52_700.0, 253_400.0), MESH_FOCUS_REFINEMENT_FACTOR);
        assert!((mesh_refinement_factor(51_700.0, 126_000.0) - 1.561).abs() < 1.0e-3);
    }

    /// Every degenerate input names what it broke.
    #[test]
    fn degenerate_inputs_are_refused_by_name() {
        let empty = Window::mesh_focus(&[], &[], &[]).unwrap_err().to_string();
        assert!(empty.contains("at least one cell"), "{empty}");

        let ragged = Window::mesh_focus(&[1.0, 2.0], &[1.0], &[1.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(ragged.contains("one latitude, longitude and spacing per cell"), "{ragged}");

        let nan = Window::mesh_focus(&[1.0, f64::NAN], &[1.0, 2.0], &[1.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(nan.contains("not finite"), "{nan}");

        let off_globe = Window::mesh_focus(&[1.0, 91.0], &[1.0, 2.0], &[1.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(off_globe.contains("outside [-90, 90]"), "{off_globe}");

        let zero = Window::mesh_focus(&[1.0, 2.0], &[1.0, 2.0], &[0.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(zero.contains("positive and finite"), "{zero}");

        // Two refined cells on opposite sides of the planet average to the
        // origin, which points nowhere.
        let antipodal = Window::mesh_focus(
            &[45.0, -45.0, 0.0],
            &[0.0, 180.0, 90.0],
            &[2_000.0, 2_000.0, 60_000.0],
        )
        .unwrap_err()
        .to_string();
        assert!(antipodal.contains("mean direction of magnitude"), "{antipodal}");
        assert!(antipodal.contains("no middle"), "{antipodal}");
    }

    /// A refined region wider than a quarter turn has passed the horizon of
    /// any projection a window can be cut on.
    #[test]
    fn a_refined_region_past_the_horizon_is_refused() {
        let error = Window::mesh_focus(
            &[0.0, 0.0, 0.0, 40.0],
            &[0.0, 95.0, -95.0, 10.0],
            &[2_000.0, 2_000.0, 2_000.0, 60_000.0],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("degrees of arc from its own centre"), "{error}");
        assert!(error.contains("--window global"), "{error}");
    }

    /// A one-cell refined region has no extent of its own. It gets the
    /// smallest window `coordinates` will build rather than a 2x2 with no
    /// centre point.
    #[test]
    fn a_single_refined_cell_still_makes_a_window() {
        let window = Window::mesh_focus(
            &[45.0, 10.0, -10.0],
            &[8.0, 100.0, -100.0],
            &[2_000.0, 60_000.0, 60_000.0],
        )
        .unwrap();
        assert_eq!(window.shape().unwrap(), (3, 3));
        let grid = window.coordinates().unwrap();
        assert!(nearest_grid_point_km(&grid, 45.0, 8.0) < 0.001);
    }

    /// A southern core keeps both truelats in the south, because `set_lc`
    /// takes its hemisphere from the sign of `truelat1` and a truelat over
    /// the equator would put the window in the wrong one.
    #[test]
    fn a_southern_core_keeps_its_truelats_in_the_south() {
        let (lat, lon, spacing) = patch_in_background(-40.0, -30.0, 140.0, 152.0, 4_000.0, 60_000.0);
        let window = Window::mesh_focus(&lat, &lon, &spacing).unwrap();
        let w = match &window {
            Window::Lambert(w) => w.clone(),
            Window::LatLon(_) => panic!("35 S is well inside the Lambert branch"),
        };
        assert!(w.centre_lat < 0.0, "{}", w.centre_lat);
        assert!(w.truelat1 < 0.0 && w.truelat2 < 0.0, "{} {}", w.truelat1, w.truelat2);
        let grid = window.coordinates().unwrap();
        assert!(grid.0.iter().all(|&v| v < 0.0), "a southern window drifted north");
        for (clat, clon) in [(-40.0, 140.0), (-40.0, 152.0), (-30.0, 140.0), (-30.0, 152.0)] {
            assert!(nearest_grid_point_km(&grid, clat, clon) <= 4.0, "{clat} {clon}");
        }
    }

    /// `mesh` is not a catalogue entry, and asking the catalogue for it says
    /// why rather than handing back a box.
    #[test]
    fn the_catalogue_refuses_the_derived_name_and_says_why() {
        let error = Window::named("mesh").unwrap_err().to_string();
        assert!(error.contains("derived from the mesh being converted"), "{error}");
        assert!(error.contains("mesh_focus"), "{error}");
        let unknown = Window::named("atlantic").unwrap_err().to_string();
        assert!(unknown.contains("focus, global, mesh"), "{unknown}");
    }
}

#[cfg(test)]
mod lambert_spec_tests {
    use super::*;

    #[test]
    fn explicit_lambert_matches_the_fixed_focus_window() {
        let named = Window::named("focus").unwrap();
        let spec = Window::named("lambert:37.0,-96.0,22000,240,150,30,60,-96").unwrap();
        assert_eq!(named.coordinates().unwrap(), spec.coordinates().unwrap());
        assert_eq!(named.spec_json(), spec.spec_json());
    }

    #[test]
    fn explicit_lambert_refuses_malformed_specs() {
        assert!(Window::named("lambert:37,-96,22000,240,150,30,60").is_err());
        assert!(Window::named("lambert:37,-96,-1,240,150,30,60,-96").is_err());
        assert!(Window::named("lambert:37,-96,22000,1,150,30,60,-96").is_err());
        assert!(Window::named("lambert:37,-96,22000,240.5,150,30,60,-96").is_err());
    }
}

#[cfg(test)]
mod mesh_spacing_window_tests {
    use super::*;

    /// A quasi-uniform lattice of cells `spacing_m` apart, covering the
    /// latitude band given, as a stand-in for a uniform mesh.
    fn lattice(spacing_m: f64, south: f64, north: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let step = spacing_m / (RAD * MPAS_EARTH_RADIUS_M);
        let (mut lat, mut lon, mut sp) = (Vec::new(), Vec::new(), Vec::new());
        let mut la = south;
        while la <= north {
            let lon_step = step / (la * RAD).cos().max(0.05);
            let mut lo = -180.0;
            while lo < 180.0 {
                lat.push(la);
                lon.push(lo);
                sp.push(spacing_m);
                lo += lon_step;
            }
            la += step;
        }
        (lat, lon, sp)
    }

    #[test]
    fn spacing_rounds_down_to_three_figures_and_never_up() {
        assert_eq!(finer_round_spacing(15_000.0), 15_000.0);
        assert_eq!(finer_round_spacing(14_987.3), 14_900.0);
        assert_eq!(finer_round_spacing(29_870.0), 29_800.0);
        assert!((finer_round_spacing(3_012.0) - 3_010.0).abs() < 1.0e-9);
        for value in [999.9, 1_234.5, 59_812.0, 120_000.0] {
            assert!(finer_round_spacing(value) <= value);
        }
    }

    #[test]
    fn a_15_km_mesh_draws_globally_at_15_km_or_finer_not_28() {
        let spacing = vec![14_990.0; 1001];
        let window = Window::global_at_mesh_spacing(&spacing).unwrap();
        let Window::LatLon(grid) = &window else { panic!("global window is lat-lon") };
        let metres = grid.spacing_degrees * RAD * WRF_EARTH_RADIUS_M;
        assert!(metres <= 14_990.0, "{metres}");
        assert!(metres > 14_000.0, "{metres}");
        let (ny, nx) = window.shape().unwrap();
        assert_eq!(nx, 2 * (ny - 1));
        assert_eq!(grid.south, -90.0);
        assert_eq!(grid.north, 90.0);
        let (lat, lon) = window.coordinates().unwrap();
        assert!(lat.iter().all(|v| v.abs() <= 90.0));
        assert!(lon.iter().all(|v| (-180.0..180.0).contains(v)));
        match window.projection_attributes().unwrap().iter().find(|(k, _)| k == "DX") {
            Some((_, ProjAttr::Float(dx))) => assert!(*dx <= 14_990.0),
            other => panic!("DX attribute {other:?}"),
        }
    }

    #[test]
    fn coarse_meshes_are_not_drawn_finer_than_they_hold() {
        for (spacing, low, high) in [(59_800.0, 58_000.0, 59_800.0), (29_900.0, 29_000.0, 29_900.0)] {
            let window = Window::global_at_mesh_spacing(&[spacing; 7]).unwrap();
            let Window::LatLon(grid) = window else { panic!() };
            let metres = grid.spacing_degrees * RAD * WRF_EARTH_RADIUS_M;
            assert!(metres <= high && metres >= low, "{spacing} -> {metres}");
        }
    }

    #[test]
    fn a_global_window_too_large_to_write_coarsens_and_says_so() {
        let window = Window::global_at_mesh_spacing(&[3_000.0; 3]).unwrap();
        let (ny, nx) = window.shape().unwrap();
        assert!(nx * ny <= MESH_GLOBAL_MAX_POINTS, "{nx}x{ny}");
        assert!(window.description().contains("coarsened"), "{}", window.description());
    }

    #[test]
    fn region_takes_its_dx_from_the_mesh_not_from_the_caller() {
        let (lat, lon, sp) = lattice(15_000.0, 20.0, 70.0);
        let window = Window::region_at_mesh_spacing(
            "region:40,-105,9000,7000,30,60", &lat, &lon, &sp,
        )
        .unwrap();
        let Window::Lambert(grid) = &window else { panic!("mid-latitude region is Lambert") };
        assert_eq!(grid.dx_metres, 15_000.0);
        assert!((grid.nx - 1) as f64 * grid.dx_metres >= 9_000_000.0);
        assert!((grid.ny - 1) as f64 * grid.dx_metres >= 7_000_000.0);
        assert_eq!((grid.truelat1, grid.truelat2, grid.stand_lon), (30.0, 60.0, -105.0));
        assert!(window.coordinates().unwrap().0.iter().all(|v| v.is_finite()));

        let (lat60, lon60, sp60) = lattice(60_000.0, 20.0, 70.0);
        let coarse = Window::region_at_mesh_spacing(
            "region:40,-105,9000,7000,30,60", &lat60, &lon60, &sp60,
        )
        .unwrap();
        let Window::Lambert(grid60) = coarse else { panic!() };
        assert_eq!(grid60.dx_metres, 60_000.0);
    }

    #[test]
    fn region_truelats_default_to_bracketing_the_centre_and_tropics_fall_back() {
        let (lat, lon, sp) = lattice(30_000.0, -30.0, 89.0);
        let polar = Window::region_at_mesh_spacing("region:85,0,8400,8400", &lat, &lon, &sp).unwrap();
        let Window::Lambert(grid) = polar else { panic!() };
        assert_eq!((grid.truelat1, grid.truelat2), (80.0, 89.0));
        let tropics = Window::region_at_mesh_spacing("region:0,-60,8000,3000", &lat, &lon, &sp).unwrap();
        let Window::LatLon(band) = &tropics else { panic!("equatorial region is lat-lon") };
        assert!(band.spacing_degrees * RAD * WRF_EARTH_RADIUS_M <= 30_000.0);
        assert!(tropics.coordinates().is_ok());
    }

    #[test]
    fn region_latlon_holds_a_mid_latitude_box_on_a_regular_grid() {
        let (lat, lon, sp) = lattice(60_000.0, 20.0, 60.0);
        let window = Window::region_at_mesh_spacing(
            "region-latlon:37,-98,2400,1800", &lat, &lon, &sp,
        )
        .unwrap();
        let Window::LatLon(grid) = &window else { panic!("region-latlon is lat-lon at 37N") };
        assert!(grid.spacing_degrees * RAD * WRF_EARTH_RADIUS_M <= 60_000.0);
        assert!(grid.south <= 37.0 - 8.0 && grid.north >= 37.0 + 8.0, "{grid:?}");
        assert!(window.coordinates().is_ok());
        assert!(Window::region_at_mesh_spacing(
            "region-latlon:37,-98,2400,1800,30,45", &lat, &lon, &sp,
        )
        .is_err());
        let named = Window::named("region-latlon:37,-98,2400,1800").unwrap_err().to_string();
        assert!(named.contains("region_at_mesh_spacing"), "{named}");
    }

    #[test]
    fn region_and_global_mesh_refuse_without_the_mesh_and_on_bad_specs() {
        let need = Window::named(GLOBAL_MESH_WINDOW).unwrap_err().to_string();
        assert!(need.contains("global_at_mesh_spacing"), "{need}");
        let region = Window::named("region:40,-105,9000,7000").unwrap_err().to_string();
        assert!(region.contains("region_at_mesh_spacing"), "{region}");
        let (lat, lon, sp) = lattice(60_000.0, 0.0, 60.0);
        for bad in [
            "region:40,-105,9000",
            "region:40,-105,-1,7000",
            "region:95,-105,9000,7000",
            "region:40,-105,9000,7000,-30,60",
            "region:40,-105,90000,7000",
        ] {
            assert!(Window::region_at_mesh_spacing(bad, &lat, &lon, &sp).is_err(), "{bad}");
        }
    }
}
