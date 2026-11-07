use rayon::prelude::*;

/// Destagger a 3D field along the X (west_east) axis.
///
/// Input shape: `[nz, ny, nx_stag]` where `nx_stag = nx + 1`.
/// Output shape: `[nz, ny, nx]`.
/// Averages adjacent points: `out[k,j,i] = (in[k,j,i] + in[k,j,i+1]) / 2`.
pub fn destagger_x(data: &[f64], nz: usize, ny: usize, nx_stag: usize) -> Vec<f64> {
    let nx = nx_stag - 1;
    let mut out = vec![0.0; nz * ny * nx];
    out.par_chunks_mut(ny * nx).enumerate().for_each(|(k, plane)| {
        for j in 0..ny {
            for i in 0..nx {
                let idx_l = k * ny * nx_stag + j * nx_stag + i;
                let idx_r = idx_l + 1;
                plane[j * nx + i] = 0.5 * (data[idx_l] + data[idx_r]);
            }
        }
    });
    out
}

/// Destagger a 3D field along the Y (south_north) axis.
///
/// Input shape: `[nz, ny_stag, nx]` where `ny_stag = ny + 1`.
/// Output shape: `[nz, ny, nx]`.
pub fn destagger_y(data: &[f64], nz: usize, ny_stag: usize, nx: usize) -> Vec<f64> {
    let ny = ny_stag - 1;
    let mut out = vec![0.0; nz * ny * nx];
    out.par_chunks_mut(ny * nx).enumerate().for_each(|(k, plane)| {
        for j in 0..ny {
            for i in 0..nx {
                let idx_b = k * ny_stag * nx + j * nx + i;
                let idx_t = idx_b + nx;
                plane[j * nx + i] = 0.5 * (data[idx_b] + data[idx_t]);
            }
        }
    });
    out
}

/// Destagger a 3D field along the Z (bottom_top) axis.
///
/// Input shape: `[nz_stag, ny, nx]` where `nz_stag = nz + 1`.
/// Output shape: `[nz, ny, nx]`.
pub fn destagger_z(data: &[f64], nz_stag: usize, ny: usize, nx: usize) -> Vec<f64> {
    let nz = nz_stag - 1;
    let plane_size = ny * nx;
    let mut out = vec![0.0; nz * plane_size];
    out.par_chunks_mut(plane_size)
        .enumerate()
        .for_each(|(k, plane)| {
            let off_b = k * plane_size;
            let off_t = (k + 1) * plane_size;
            for idx in 0..plane_size {
                plane[idx] = 0.5 * (data[off_b + idx] + data[off_t + idx]);
            }
        });
    out
}

/// Destagger a 2D field along the X axis.
///
/// Input shape: `[ny, nx_stag]`. Output shape: `[ny, nx]`.
pub fn destagger_x_2d(data: &[f64], ny: usize, nx_stag: usize) -> Vec<f64> {
    destagger_x(data, 1, ny, nx_stag)
}

/// Destagger a 2D field along the Y axis.
///
/// Input shape: `[ny_stag, nx]`. Output shape: `[ny, nx]`.
pub fn destagger_y_2d(data: &[f64], ny_stag: usize, nx: usize) -> Vec<f64> {
    destagger_y(data, 1, ny_stag, nx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_destagger_x_simple() {
        // 1 level, 2 rows, 3 staggered cols -> 2 unstaggered cols
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = destagger_x(&data, 1, 2, 3);
        assert_eq!(out, vec![1.5, 2.5, 4.5, 5.5]);
    }

    #[test]
    fn test_destagger_z_simple() {
        // 3 stag levels, 1 row, 1 col
        let data = vec![0.0, 10.0, 20.0];
        let out = destagger_z(&data, 3, 1, 1);
        assert_eq!(out, vec![5.0, 15.0]);
    }
}
