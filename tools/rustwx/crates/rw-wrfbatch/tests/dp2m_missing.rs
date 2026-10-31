//! wrf-core's 2 m dewpoint keeps a missing input missing.  The MPAS
//! converter writes NaN at every window point off a limited-area hex mesh;
//! the vendored body floored Q2 with `q.max(1e-10)`, which turns NaN into
//! 1e-10, so the frame around a hex polygon was painted with the -40 F end
//! of the dewpoint scale (2026-10-29, Kansas polygon f000/f001).

use wrf_core::diag::moisture::dewpoint_2m_c;

#[test]
fn missing_q2_or_psfc_gives_missing_dewpoint() {
    assert!(dewpoint_2m_c(f64::NAN, 97_000.0).is_nan());
    assert!(dewpoint_2m_c(0.010, f64::NAN).is_nan());
    assert!(dewpoint_2m_c(f64::NAN, f64::NAN).is_nan());
    // f32 NaN widened by the reader is still NaN.
    assert!(dewpoint_2m_c(f32::NAN as f64, 97_000.0).is_nan());
}

#[test]
fn finite_inputs_keep_the_old_formula_bit_for_bit() {
    let old = |q: f64, p_pa: f64| {
        let q = q.max(1e-10);
        let p_hpa = p_pa / 100.0;
        let e = q * p_hpa / (0.622 + q);
        let ln_e = (e / 6.112).max(1e-10).ln();
        (243.5 * ln_e) / (17.67 - ln_e)
    };
    for &(q, p) in &[
        (0.010, 97_000.0),
        (0.0001, 85_000.0),
        (0.0, 101_325.0),
        (-1.0e-6, 99_000.0),
        (0.022, 100_500.0),
    ] {
        assert_eq!(dewpoint_2m_c(q, p).to_bits(), old(q, p).to_bits(), "q {q} p {p}");
    }
    // A moist plains surface: q 10 g/kg at 970 hPa is a Td near 13.5 C.
    let td = dewpoint_2m_c(0.010, 97_000.0);
    assert!((td - 13.5).abs() < 0.5, "{td}");
}
