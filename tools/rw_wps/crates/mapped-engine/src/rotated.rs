//! Rotated latitude/longitude lattice (GRIB2 GDT 3.1) geometry, for the
//! declared-source wind rotation.
//!
//! The twin of `gpuwm.static.rotated_latlon.RotatedLatLonGrid`.  The
//! rotated -> geographic transform is the one grib-core places GDT 3.1
//! points with (`grib2::grid::rotated_to_geographic`): the rotated frame's
//! north pole sits at geographic latitude `-south_pole_lat` on meridian
//! `south_pole_lon + 180`, the rotated origin at
//! `(90 + south_pole_lat, south_pole_lon)`, and the angle of rotation is
//! subtracted from the rotated longitude before the transform.
//!
//! ECCC's HRDPS and RDPS declare grid-relative vector components
//! (resolution flags bit 0x08): `UGRD`/`VGRD` lie along the rotated
//! lattice's own east and north.  `(SINALPHA, COSALPHA)` here is the angle
//! from geographic east to rotated-lattice east at each cell, in the same
//! convention as the Lambert rotation, so the one rotation step
//! (`assemble::rotate_grid_relative_winds`) serves both families:
//! `u_earth = u cos a - v sin a`, `v_earth = v cos a + u sin a`.

use crate::model::RotatedParameters;

/// The rotation from the rotated frame to the geographic frame, with the
/// trigonometric constants each cell reuses.
#[derive(Debug, Clone, Copy)]
pub struct RotatedFrame {
    sin_alpha: f64,
    cos_alpha: f64,
    south_pole_lon: f64,
    rotation_angle: f64,
}

impl RotatedFrame {
    pub fn from_declaration(parameters: &RotatedParameters) -> Self {
        // Latitude of the rotated north pole.
        let alpha = (-parameters.south_pole_lat).to_radians();
        Self {
            sin_alpha: alpha.sin(),
            cos_alpha: alpha.cos(),
            south_pole_lon: parameters.south_pole_lon.to_radians(),
            rotation_angle: parameters.rotation_angle.to_radians(),
        }
    }

    /// The rotated point's unit vector in the frame whose x axis lies on
    /// meridian `south_pole_lon`, and the rotated-lattice east tangent
    /// there (both before the final rotation about the polar axis, which
    /// leaves every dot product below unchanged).
    fn vectors(&self, rlat_deg: f64, rlon_deg: f64) -> ([f64; 3], [f64; 3]) {
        let rlat = rlat_deg.to_radians();
        let rlon = rlon_deg.to_radians() - self.rotation_angle;
        let (sin_lat, cos_lat) = rlat.sin_cos();
        let (sin_lon, cos_lon) = rlon.sin_cos();
        let r = [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat];
        let t = [-sin_lon, cos_lon, 0.0];
        let apply = |v: [f64; 3]| {
            [
                self.sin_alpha * v[0] - self.cos_alpha * v[2],
                v[1],
                self.cos_alpha * v[0] + self.sin_alpha * v[2],
            ]
        };
        (apply(r), apply(t))
    }

    /// Geographic `(latitude, longitude)` in degrees of a rotated point,
    /// longitude in (-180, 180].
    pub fn to_geographic(&self, rlat_deg: f64, rlon_deg: f64) -> (f64, f64) {
        let (q, _) = self.vectors(rlat_deg, rlon_deg);
        let latitude = q[2].clamp(-1.0, 1.0).asin().to_degrees();
        let mut longitude = (q[1].atan2(q[0]) + self.south_pole_lon).to_degrees();
        while longitude > 180.0 {
            longitude -= 360.0;
        }
        while longitude <= -180.0 {
            longitude += 360.0;
        }
        (latitude, longitude)
    }

    /// Rotated `(rlat, rlon)` in degrees of a geographic point: the
    /// transpose of [`Self::to_geographic`]'s rotation
    /// (`RotatedLatLonGrid.geographic_to_rotated`).  The longitude is not
    /// wrapped; a caller places it against its lattice's first meridian.
    pub fn from_geographic(&self, latitude_deg: f64, longitude_deg: f64) -> (f64, f64) {
        let lat = latitude_deg.to_radians();
        let lon = longitude_deg.to_radians() - self.south_pole_lon;
        let (sin_lat, cos_lat) = lat.sin_cos();
        let (sin_lon, cos_lon) = lon.sin_cos();
        let q = [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat];
        let r = [
            self.sin_alpha * q[0] + self.cos_alpha * q[2],
            q[1],
            -self.cos_alpha * q[0] + self.sin_alpha * q[2],
        ];
        let rlat = r[2].clamp(-1.0, 1.0).asin().to_degrees();
        let rlon = r[1].atan2(r[0]).to_degrees() + self.rotation_angle.to_degrees();
        (rlat, rlon)
    }

    /// `(sin a, cos a)`: the angle from geographic east to rotated-lattice
    /// east at a rotated point.
    pub fn wind_rotation(&self, rlat_deg: f64, rlon_deg: f64) -> (f64, f64) {
        let (q, t) = self.vectors(rlat_deg, rlon_deg);
        let horizontal = (q[0] * q[0] + q[1] * q[1]).sqrt();
        let (sin_lon, cos_lon) = if horizontal > 0.0 {
            (q[1] / horizontal, q[0] / horizontal)
        } else {
            (0.0, 1.0)
        };
        let (sin_lat, cos_lat) = (q[2], horizontal);
        let east = [-sin_lon, cos_lon, 0.0];
        let north = [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat];
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        (dot(t, north), dot(t, east))
    }
}

/// Source-grid `(SINALPHA, COSALPHA)` over rows `rows` and columns
/// `columns` of a declared rotated lattice, row major in (y, x): each value
/// is the whole grid's value at that cell, so a record decoded over an
/// atmospheric window is rotated with these.
pub fn declared_grid_rotation_within(
    parameters: &RotatedParameters,
    rows: [usize; 2],
    columns: [usize; 2],
) -> (Vec<f64>, Vec<f64>) {
    let frame = RotatedFrame::from_declaration(parameters);
    // Rows run on every core and are joined in row order: the same
    // arithmetic per cell, the same values.
    let out: Vec<(Vec<f64>, Vec<f64>)> = crate::threads::install(|| {
        use rayon::prelude::*;
        (rows[0]..rows[1])
            .into_par_iter()
            .map(|row| {
                let rlat = parameters.lat1 + row as f64 * parameters.dlat_deg;
                let width = columns[1].saturating_sub(columns[0]);
                let mut sina = Vec::with_capacity(width);
                let mut cosa = Vec::with_capacity(width);
                for column in columns[0]..columns[1] {
                    let rlon = parameters.lon1 + column as f64 * parameters.dlon_deg;
                    let (s, c) = frame.wind_rotation(rlat, rlon);
                    sina.push(s);
                    cosa.push(c);
                }
                (sina, cosa)
            })
            .collect()
    });
    let cells = rows[1].saturating_sub(rows[0]) * columns[1].saturating_sub(columns[0]);
    let mut sina = Vec::with_capacity(cells);
    let mut cosa = Vec::with_capacity(cells);
    for (row_sina, row_cosa) in out {
        sina.extend_from_slice(&row_sina);
        cosa.extend_from_slice(&row_cosa);
    }
    (sina, cosa)
}

/// [`declared_grid_rotation_within`] over the whole declared lattice.
pub fn declared_grid_rotation(parameters: &RotatedParameters) -> (Vec<f64>, Vec<f64>) {
    declared_grid_rotation_within(
        parameters,
        [0, parameters.ny as usize],
        [0, parameters.nx as usize],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// HRDPS continental 2.5 km, as MSC Datamart published it on
    /// 2026-10-28 (GDT 3.1, shape of earth 6).
    fn hrdps() -> RotatedParameters {
        RotatedParameters {
            lat1: -12.302501,
            lon1: 345.17878,
            dlat_deg: 0.0225,
            dlon_deg: 0.0225,
            nx: 2540,
            ny: 1290,
            south_pole_lat: -36.08852,
            south_pole_lon: 245.305142,
            rotation_angle: 0.0,
            earth_radius_m: 6_371_229.0,
            shape_of_earth: 6,
        }
    }

    #[test]
    fn hrdps_first_point_is_where_eccc_documents_it() {
        // MSC documents the HRDPS continental first point as about
        // 39 N, 134 W (39.626 N, 133.630 W placed exactly).
        let frame = RotatedFrame::from_declaration(&hrdps());
        let (lat, lon) = frame.to_geographic(-12.302501, 345.17878);
        assert!((lat - 39.626).abs() < 1e-3, "lat {lat}");
        assert!((lon + 133.630).abs() < 1e-3, "lon {lon}");
    }

    #[test]
    fn from_geographic_inverts_to_geographic() {
        let frame = RotatedFrame::from_declaration(&hrdps());
        for (rlat, rlon) in [(-12.302501, 345.17878), (2.0, 10.0), (16.7, 42.306283), (0.0, 0.0)] {
            let (lat, lon) = frame.to_geographic(rlat, rlon);
            let (back_lat, back_lon) = frame.from_geographic(lat, lon);
            let dlon = (back_lon - rlon + 540.0).rem_euclid(360.0) - 180.0;
            assert!((back_lat - rlat).abs() < 1e-9 && dlon.abs() < 1e-9, "{rlat} {rlon} -> {back_lat} {back_lon}");
        }
    }

    #[test]
    fn an_unrotated_frame_needs_no_wind_rotation() {
        let mut parameters = hrdps();
        parameters.south_pole_lat = -90.0;
        parameters.south_pole_lon = 0.0;
        let frame = RotatedFrame::from_declaration(&parameters);
        let (s, c) = frame.wind_rotation(40.0, 20.0);
        assert!(s.abs() < 1e-12 && (c - 1.0).abs() < 1e-12, "{s} {c}");
    }

    #[test]
    fn rotation_is_a_unit_angle_and_matches_a_finite_difference() {
        let frame = RotatedFrame::from_declaration(&hrdps());
        let (rlat, rlon) = (2.0, 10.0);
        let (s, c) = frame.wind_rotation(rlat, rlon);
        assert!((s * s + c * c - 1.0).abs() < 1e-12);
        // Step a little along the rotated parallel and read the bearing.
        let h = 1e-4;
        let (lat0, lon0) = frame.to_geographic(rlat, rlon - h);
        let (lat1, lon1) = frame.to_geographic(rlat, rlon + h);
        let dx = (lon1 - lon0) * lat0.to_radians().cos();
        let dy = lat1 - lat0;
        let norm = (dx * dx + dy * dy).sqrt();
        assert!((dx / norm - c).abs() < 1e-6 && (dy / norm - s).abs() < 1e-6,
                "{s} {c} vs {} {}", dy / norm, dx / norm);
    }
}
