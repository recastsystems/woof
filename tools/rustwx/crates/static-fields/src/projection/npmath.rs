//! LANE 1.  numpy-bit-parity float32 transcendentals and float helpers.
//!
//! The port's byte-parity contract is against numpy on the deployment
//! platform.  Measured on the reference box (numpy 2.2.6, Windows
//! x86-64, UCRT; `tools/static_rust_port/probe_libm.py`):
//!
//! * every float64 transcendental numpy evaluates (sin, cos, tan, asin,
//!   acos, atan, atan2, log, log10, exp, sqrt, pow, mod) is bit-equal
//!   to the UCRT libm that Rust `std` links -- `f64::sin` & co. are the
//!   correct spellings and nothing here re-implements them;
//! * float32 tan/atan/atan2/asin/acos/log10/pow use the vendored pure-Rust
//!   libm implementation. Platform libm agreement with one NumPy build
//!   is insufficient: a WPS sampling position must have the same bits
//!   on the machine that prepares a nest and the machine that moves it;
//! * sqrt remains the correctly rounded hardware operation;
//! * float32 **sin, cos, exp and log are NOT**: numpy routes them
//!   through its own SIMD kernels for both arrays and scalars, so this
//!   module ports those four kernels exactly (same constants, same
//!   fused-multiply-add sequence -- `f32::mul_add` is correctly rounded
//!   like the vector FMA the kernels use);
//! * `x ** 2.0` under numpy's power ufunc is `x * x` (measured; UCRT
//!   `pow(x, 2.0)` differs on some inputs), hence [`np_pow`]/[`np_powf`];
//! * `np.mod` is fmod adjusted to the divisor's sign, with an exact-zero
//!   result taking the divisor's sign.
//!
//! Kernel provenance (BSD-3-Clause, Copyright (c) 2005-2025 NumPy
//! Developers; vendored reference copies with origin URLs live in
//! `tools/static_rust_port/numpy_src/`):
//!
//! * sin/cos: `numpy/_core/src/umath/loops_trigonometric.dispatch.cpp`
//!   @ v2.2.6 (Cody-Waite reduction + Myklebust polynomials, libm
//!   fallback outside the reduction range);
//! * exp/log: `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src`
//!   @ v2.2.6, the FMA (AVX2) variant -- the one MSVC-built wheels
//!   compile (`SIMD_AVX512F` is explicitly disabled under `_MSC_VER`);
//!   coefficients from `npy_simd_data.h`.
//!
//! Unit-tested bit-for-bit against committed numpy outputs in
//! `tests/goldens/lane1` (`npmath` case).

// ---------------------------------------------------------------------------
// float32 sin/cos (loops_trigonometric.dispatch.cpp)
// ---------------------------------------------------------------------------

const TWO_OVER_PI: f32 = f32::from_bits(0x3F22F983);
const CODYW_PIO2_HIGH: f32 = f32::from_bits(0xBFC90FD8);
const CODYW_PIO2_MED: f32 = f32::from_bits(0xB4A8885A);
const CODYW_PIO2_LOW: f32 = f32::from_bits(0xA7C234C4);
const RINT_CVT_MAGIC: f32 = f32::from_bits(0x4B400000); // 0x1.8p23
const MAX_CODY_SIN: f32 = f32::from_bits(0x47E55DFF); // 117435.992f
const MAX_CODY_COS: f32 = f32::from_bits(0x478B9A08); // 71476.0625f

const COS_INVF8: f32 = f32::from_bits(0x37CC730B);
const COS_INVF6: f32 = f32::from_bits(0xBAB6036E);
const COS_INVF4: f32 = f32::from_bits(0x3D2AAA9E);
const COS_INVF2: f32 = -0.5;
const COS_INVF0: f32 = 1.0;

const SIN_INVF9: f32 = f32::from_bits(0x363E9DDE);
const SIN_INVF7: f32 = f32::from_bits(0xB95035DD);
const SIN_INVF5: f32 = f32::from_bits(0x3C0888CD);
const SIN_INVF3: f32 = f32::from_bits(0xBE2AAAAB);

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrigOp {
    Sin,
    Cos,
}

#[inline]
fn cosine_poly(x2: f32) -> f32 {
    let mut r = COS_INVF8.mul_add(x2, COS_INVF6);
    r = r.mul_add(x2, COS_INVF4);
    r = r.mul_add(x2, COS_INVF2);
    r = r.mul_add(x2, COS_INVF0);
    r
}

#[inline]
fn sine_poly(x: f32, x2: f32) -> f32 {
    let mut r = SIN_INVF9.mul_add(x2, SIN_INVF7);
    r = r.mul_add(x2, SIN_INVF5);
    r = r.mul_add(x2, SIN_INVF3);
    r = r.mul_add(x2, 0.0);
    r = r.mul_add(x, x);
    r
}

fn sincos(x_in: f32, op: TrigOp) -> f32 {
    if x_in.is_nan() {
        return f32::from_bits(0x7FC0_0000); // NPY_NANF
    }
    let max_cody = match op {
        TrigOp::Sin => MAX_CODY_SIN,
        TrigOp::Cos => MAX_CODY_COS,
    };
    if x_in.abs() > max_cody {
        // libm fallback exactly as the kernel's scalar loop does.
        return match op {
            TrigOp::Sin => x_in.sin(),
            TrigOp::Cos => x_in.cos(),
        };
    }
    let x = x_in;
    // Round to nearest via the magic-constant trick.  The source spells
    // this Mul-then-Add, but the MSVC-built wheel CONTRACTS it into one
    // fused multiply-add, which flips the chosen quadrant when the
    // product sits on a half-integer knife edge.  Measured: the fused
    // spelling matches numpy on a 400k-value sweep with zero mismatches
    // while the unfused one diverges on 42 large-|x| values
    // (tools/static_rust_port/hunt_variant3.py).
    let mut quadrant = x.mul_add(TWO_OVER_PI, RINT_CVT_MAGIC);
    quadrant -= RINT_CVT_MAGIC;

    // Cody-Waite range reduction (three fused steps)
    let mut reduced = quadrant.mul_add(CODYW_PIO2_HIGH, x);
    reduced = quadrant.mul_add(CODYW_PIO2_MED, reduced);
    reduced = quadrant.mul_add(CODYW_PIO2_LOW, reduced);
    let reduced2 = reduced * reduced;

    let cos = cosine_poly(reduced2);
    let sin = sine_poly(reduced, reduced2);

    let mut iquadrant = quadrant as i32; // integral by construction
    if op == TrigOp::Cos {
        iquadrant = iquadrant.wrapping_add(1);
    }
    let mut out = if iquadrant & 1 == 0 { sin } else { cos };
    if iquadrant & 2 == 2 {
        out = 0.0 - out;
    }
    out
}

/// numpy's float32 `sin` ufunc, scalar-for-scalar.
pub fn np_sinf(x: f32) -> f32 {
    sincos(x, TrigOp::Sin)
}

/// numpy's float32 `cos` ufunc, scalar-for-scalar.
pub fn np_cosf(x: f32) -> f32 {
    sincos(x, TrigOp::Cos)
}

// ---------------------------------------------------------------------------
// float32 exp/log (loops_exponent_log.dispatch.c.src, FMA variant)
// ---------------------------------------------------------------------------

const EXP_CODYW_C1: f32 = f32::from_bits(0xBF317200);
const EXP_CODYW_C2: f32 = f32::from_bits(0xB5BFBE8E);
const EXP_P0: f32 = f32::from_bits(0x3F800000);
const EXP_P1: f32 = f32::from_bits(0x3F39CBD5);
const EXP_P2: f32 = f32::from_bits(0x3E7D4C58);
const EXP_P3: f32 = f32::from_bits(0x3D517D8C);
const EXP_P4: f32 = f32::from_bits(0x3BDD7159);
const EXP_P5: f32 = f32::from_bits(0x3A053DD8);
const EXP_Q0: f32 = f32::from_bits(0x3F800000);
const EXP_Q1: f32 = f32::from_bits(0xBE8C6857);
const EXP_Q2: f32 = f32::from_bits(0x3CB0E832);
const LOG2E: f32 = f32::from_bits(0x3FB8AA3B);
const EXP_XMAX: f32 = f32::from_bits(0x42B17218); // 88.72283935546875
const EXP_XMIN: f32 = f32::from_bits(0xC2CFF1B5); // -103.97208404541015625

/// `poly * 2^quadrant` by exponent-bit arithmetic (fma_scalef_ps),
/// including the quadrant <= -125 denormal split.
#[inline]
fn scalef(poly: f32, quadrant: f32) -> f32 {
    if quadrant <= -125.0 {
        let quad_diff = -(quadrant - (-125.0));
        let two_power_diff = (1i32 << (quad_diff as i32)) as f32;
        let clamped = quadrant.max(-125.0);
        let exponent = (clamped as i32) << 23;
        let scaled =
            f32::from_bits(poly.to_bits().wrapping_add(exponent as u32));
        scaled / two_power_diff
    } else {
        let exponent = (quadrant as i32) << 23;
        f32::from_bits(poly.to_bits().wrapping_add(exponent as u32))
    }
}

/// numpy's float32 `exp` ufunc, scalar-for-scalar.
pub fn np_expf(x_in: f32) -> f32 {
    if x_in.is_nan() {
        return f32::from_bits(0x7FC0_0000);
    }
    if x_in >= EXP_XMAX {
        return f32::INFINITY;
    }
    if x_in <= EXP_XMIN {
        return 0.0;
    }
    let mut x = x_in;
    // NumPy's Windows wheel contracts this multiply/add; its Linux
    // wheel preserves the two operations in loops_exponent_log.dispatch.c.
    #[cfg(target_env = "msvc")]
    let mut quadrant = x.mul_add(LOG2E, RINT_CVT_MAGIC);
    #[cfg(not(target_env = "msvc"))]
    let mut quadrant = x * LOG2E + RINT_CVT_MAGIC;
    quadrant -= RINT_CVT_MAGIC;

    x = quadrant.mul_add(EXP_CODYW_C1, x);
    x = quadrant.mul_add(EXP_CODYW_C2, x);
    // third Cody-Waite step is fma(q, 0, x) -- a no-op, kept out.

    let mut num = EXP_P5.mul_add(x, EXP_P4);
    num = num.mul_add(x, EXP_P3);
    num = num.mul_add(x, EXP_P2);
    num = num.mul_add(x, EXP_P1);
    num = num.mul_add(x, EXP_P0);
    let mut den = EXP_Q2.mul_add(x, EXP_Q1);
    den = den.mul_add(x, EXP_Q0);
    scalef(num / den, quadrant)
}

const LOG_P1: f32 = f32::from_bits(0x3F800000);
const LOG_P2: f32 = f32::from_bits(0x4007361C);
const LOG_P3: f32 = f32::from_bits(0x3FBD70A9);
const LOG_P4: f32 = f32::from_bits(0x3EC30333);
const LOG_P5: f32 = f32::from_bits(0x3CD42BCD);
const LOG_Q0: f32 = f32::from_bits(0x3F800000);
const LOG_Q1: f32 = f32::from_bits(0x4027361C);
const LOG_Q2: f32 = f32::from_bits(0x401CFE0D);
const LOG_Q3: f32 = f32::from_bits(0x3F7C8AE4);
const LOG_Q4: f32 = f32::from_bits(0x3E1E5BF3);
const LOG_Q5: f32 = f32::from_bits(0x3BC083DF);
const LOGE2: f32 = f32::from_bits(0x3F317218);
const SQRT1_2: f32 = f32::from_bits(0x3F3504F3);
const FLT_MIN: f32 = f32::from_bits(0x0080_0000);
const TWO_POW_100: f32 = f32::from_bits(0x7180_0000);

/// fma_get_exponent: unbiased exponent + 1, with the 2^100 denormal
/// pre-scale.  Callers have already zeroed negative lanes.
#[inline]
fn get_exponent(x: f32) -> f32 {
    let (x, denormal) = if x < FLT_MIN {
        (x * TWO_POW_100, true)
    } else {
        (x, false)
    };
    let exp = ((x.to_bits() >> 23) as i32 - 0x7E) as f32;
    if denormal { exp - 100.0 } else { exp }
}

/// fma_get_mantissa: mantissa normalized into [0.5, 1).
#[inline]
fn get_mantissa(x: f32) -> f32 {
    let x = if x < FLT_MIN { x * TWO_POW_100 } else { x };
    f32::from_bits((x.to_bits() & 0x007F_FFFF) | (126u32 << 23))
}

/// numpy's float32 `log` ufunc, scalar-for-scalar.
pub fn np_logf(x_in: f32) -> f32 {
    if x_in.is_nan() {
        return f32::from_bits(0x7FC0_0000);
    }
    if x_in == f32::INFINITY {
        return f32::INFINITY;
    }
    if x_in == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x_in < 0.0 {
        return f32::from_bits(0xFFC0_0000); // -NPY_NANF
    }
    let mut exponent = get_exponent(x_in);
    let mut x = get_mantissa(x_in);
    if x <= SQRT1_2 {
        exponent -= 1.0;
        x += x;
    }
    x -= 1.0;

    let mut num = LOG_P5.mul_add(x, LOG_P4);
    num = num.mul_add(x, LOG_P3);
    num = num.mul_add(x, LOG_P2);
    num = num.mul_add(x, LOG_P1);
    num = num.mul_add(x, 0.0); // LOG_P0 = 0
    let mut den = LOG_Q5.mul_add(x, LOG_Q4);
    den = den.mul_add(x, LOG_Q3);
    den = den.mul_add(x, LOG_Q2);
    den = den.mul_add(x, LOG_Q1);
    den = den.mul_add(x, LOG_Q0);
    exponent.mul_add(LOGE2, num / den)
}

// ---------------------------------------------------------------------------
// power / remainder / spacing / nextafter helpers
// ---------------------------------------------------------------------------

/// numpy's float64 power ufunc: `x ** 2.0` is exactly `x * x`
/// (measured; libm `pow` disagrees on some inputs), general exponents
/// go to libm `pow` which numpy and Rust `std` share bit-for-bit.
pub fn np_pow(a: f64, b: f64) -> f64 {
    if b == 2.0 { a * a } else { a.powf(b) }
}

/// numpy's float32 power ufunc (same `** 2.0` special case, measured).
pub fn np_powf(a: f32, b: f32) -> f32 {
    if b == 2.0 { a * a } else { libm::powf(a, b) }
}

/// Portable float32 operations for the WPS sampling projection (cross-machine moving nests).
pub fn np_tanf(x: f32) -> f32 { libm::tanf(x) }
pub fn np_atanf(x: f32) -> f32 { libm::atanf(x) }
pub fn np_atan2f(y: f32, x: f32) -> f32 { libm::atan2f(y, x) }
pub fn np_asinf(x: f32) -> f32 { libm::asinf(x) }
pub fn np_acosf(x: f32) -> f32 { libm::acosf(x) }
pub fn np_log10f(x: f32) -> f32 { libm::log10f(x) }

/// numpy's float64 `mod`: fmod adjusted into the divisor's sign, exact
/// zero taking the divisor's sign (npy_remainder).
pub fn np_mod(a: f64, b: f64) -> f64 {
    let r = a % b;
    if r == 0.0 {
        0.0f64.copysign(b)
    } else if (r < 0.0) != (b < 0.0) {
        r + b
    } else {
        r
    }
}

/// numpy's float32 `mod`.
pub fn np_modf(a: f32, b: f32) -> f32 {
    let r = a % b;
    if r == 0.0 {
        0.0f32.copysign(b)
    } else if (r < 0.0) != (b < 0.0) {
        r + b
    } else {
        r
    }
}

/// C `nextafterf(x, +inf)` (what `np.nextafter(x, np.float32(inf))`
/// evaluates through).
pub fn nextafter_up(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1); // smallest positive subnormal
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// C `nextafterf(x, -inf)`.
pub fn nextafter_down(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(0x8000_0001); // smallest negative subnormal
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// `|np.spacing(x)|` for float32: the distance to the next
/// representable value away from zero (numpy's spacing is signed away
/// from zero; every consumer here takes the absolute value).
pub fn spacing_abs(x: f32) -> f32 {
    let away = if x >= 0.0 || x.is_nan() {
        nextafter_up(x)
    } else {
        nextafter_down(x)
    };
    (away - x).abs()
}
