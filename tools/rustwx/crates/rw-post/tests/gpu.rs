//! GPU path: the same bits as the CPU path (decision D30), and the kernel
//! images match their manifest.  The device tests need an NVIDIA GPU and
//! run with `cargo test -p rw-post --test gpu -- --ignored`.

#![cfg(any(windows, target_os = "linux"))]

mod common;

use rw_post::gpu::{GpuPost, MANIFEST};
use rw_post::solar::{SolarFrame, UtcTime};
use rw_post::state::{MassState, build_cpu};
use rw_post::surface::{SurfaceInput, SurfaceOptions, surface_cpu};
use rw_post::thermo;
use sha2::{Digest, Sha256};

fn differing(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

#[test]
fn kernel_images_match_manifest() {
    let m: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    for a in m["artifacts"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(a["file"].as_str().unwrap())).unwrap();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), a["sha256"].as_str().unwrap(), "{}", a["file"]);
    }
    for s in m["sources"].as_array().unwrap() {
        let bytes = std::fs::read(root.join(s["file"].as_str().unwrap())).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            s["sha256"].as_str().unwrap(),
            "{} changed since the kernels were built: run the rebuild_kernels example",
            s["file"]
        );
    }
    let opts: Vec<&str> = m["options"].as_array().unwrap().iter().map(|o| o.as_str().unwrap()).collect();
    assert!(opts.contains(&"--fmad=false") && opts.contains(&"--prec-div=true") && opts.contains(&"--ftz=false"));
}

fn gpu() -> GpuPost {
    GpuPost::open_with(0, rw_post::gpu::test_image()).expect("an NVIDIA GPU")
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn libm_port_is_bit_identical() {
    let g = gpu();
    // Every 4099th binary32 word (all signs, exponents, NaNs, infinities and
    // subnormals), plus the arguments the post code feeds the functions.
    let mut x: Vec<f32> = (0..=u32::MAX / 4099).map(|i| f32::from_bits(i * 4099)).collect();
    let mut rng = common::Lcg(3);
    for _ in 0..500_000 {
        x.push((200.0 * rng.next() - 100.0) as f32);
        x.push((1e5 * rng.next()) as f32);
    }
    let out = g.libm_probe(&x).unwrap();
    let mut bad = 0usize;
    for (i, &v) in x.iter().enumerate() {
        let want = [thermo::expf(v), thermo::logf(v), thermo::sinf(v), thermo::cosf(v)];
        for k in 0..4 {
            let got = out[4 * i + k];
            if got.to_bits() != want[k].to_bits() && !(got.is_nan() && want[k].is_nan()) {
                if bad < 10 {
                    eprintln!("fn {k} x {v:e} ({:#x}): gpu {got:e} cpu {:e}", v.to_bits(), want[k]);
                }
                bad += 1;
            }
        }
    }
    eprintln!("{} arguments x 4 functions, {bad} differ", x.len());
    assert_eq!(bad, 0);
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn thermo_helpers_are_bit_identical() {
    let g = gpu();
    let mut rng = common::Lcg(5);
    let n = 1_000_000;
    let t: Vec<f32> = (0..n).map(|_| (170.0 + 170.0 * rng.next()) as f32).collect();
    let p: Vec<f32> = (0..n).map(|_| (100.0 + 107000.0 * rng.next()) as f32).collect();
    let r: Vec<f32> = (0..n).map(|i| if i % 97 == 0 { -1e-5 } else { (0.04 * rng.next() * rng.next()) as f32 }).collect();
    let out = g.thermo_probe(&t, &p, &r).unwrap();
    let mut cpu = Vec::with_capacity(7 * n);
    for i in 0..n {
        let q = r[i] / (1.0 + r[i]);
        let e = thermo::vapour_pressure(q, p[i]);
        cpu.extend_from_slice(&[
            thermo::es_water(t[i]),
            thermo::es_ice(t[i]),
            e,
            thermo::dewpoint(e),
            thermo::rh(e, t[i]),
            thermo::virtual_temperature(t[i], r[i]),
            thermo::theta_e(t[i], p[i], r[i]),
        ]);
    }
    assert_eq!(differing(&out, &cpu), 0);
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn state_and_surface_are_bit_identical() {
    let g = gpu();
    for (nx, ny, nz, seed) in [(37, 29, 50, 1u64), (257, 131, 20, 2), (5, 3, 1, 3)] {
        let inp = common::synthetic_state(nx, ny, nz, seed);
        let cpu = build_cpu(&inp).unwrap();
        let dst = g.build_state(&inp).unwrap();
        let gst = g.download_state(&dst).unwrap();
        assert_eq!(differing(&cpu.mass, &gst.mass), 0, "state {nx}x{ny}x{nz}");
        assert_eq!(differing(&cpu.interface, &gst.interface), 0, "interfaces {nx}x{ny}x{nz}");

        let n = inp.shape.ncell();
        let mut s = common::synthetic_surface(&inp, seed + 10);
        s.set(SurfaceInput::TvLowest, &cpu.mass_slot(MassState::Tv)[..n]);
        let solar = SolarFrame::at(&UtcTime::parse_wrf("2026-10-03_22:00:00").unwrap());
        for (qfx, derived, th2_only) in [(0u32, 0u32, false), (1, 1, true)] {
            let mut opt = SurfaceOptions::new(&solar);
            opt.latent_from_qfx = qfx;
            opt.snow_cover_derived = derived;
            let mut s2 = s.clone();
            if derived == 1 {
                let snup = rw_post::snow::snup_plane("MODIFIED_IGBP_MODIS_NOAH", s2.plane(SurfaceInput::IvgTyp)).unwrap();
                s2.set(SurfaceInput::Snup, &snup);
            }
            if th2_only {
                s2.present &= !(1 << SurfaceInput::T2 as u32);
            }
            let c = surface_cpu(&s2, &opt);
            // TvLowest from the device-resident state, as the pipeline runs.
            let mut s3 = s2.clone();
            s3.present &= !(1 << SurfaceInput::TvLowest as u32);
            let gpu_out = g.surface(&s3, &opt, Some(&dst)).unwrap();
            assert_eq!(differing(&c, &gpu_out), 0, "surface {nx}x{ny} qfx={qfx} derived={derived}");
        }
    }
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn rotation_is_bit_identical() {
    let g = gpu();
    let mut rng = common::Lcg(8);
    let n = 10_007;
    let a: Vec<f32> = (0..n).map(|_| (6.3 * rng.next()) as f32).collect();
    let sa: Vec<f32> = a.iter().map(|&x| thermo::sinf(x)).collect();
    let ca: Vec<f32> = a.iter().map(|&x| thermo::cosf(x)).collect();
    let u: Vec<f32> = (0..3 * n).map(|_| (60.0 * rng.next() - 30.0) as f32).collect();
    let v: Vec<f32> = (0..3 * n).map(|_| (60.0 * rng.next() - 30.0) as f32).collect();
    let (gu, gv) = g.rotate(&u, &v, &sa, &ca).unwrap();
    let (cu, cv) = rw_post::wind::rotate_cpu(&u, &v, &sa, &ca);
    assert_eq!(differing(&gu, &cu), 0);
    assert_eq!(differing(&gv, &cv), 0);
}

/// The embedded PTX loads on the documented driver floor.  Breakage this
/// prevents: 2.8.7 to 2.9.0 embedded NVRTC 13.4's PTX (`.version 9.4`); a
/// CUDA 13.2 driver (595.91) refused it with
/// `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, so `--post-device auto` computed
/// the GRIB2 products of every card without a CUBIN (A100, H100, L40S, RTX
/// 30/40, T4, ...) on the CPU, and it said so nowhere but manifest.json.
/// The same PTX with an 8.0 header loaded.  ISA 8.0 is CUDA 12.0's, which
/// every driver from R525 on JIT-compiles.
#[test]
fn embedded_ptx_isa_is_within_the_driver_floor() {
    let ptx = std::str::from_utf8(rw_post::gpu::checked_in_artifact("woof_post.ptx").unwrap()).unwrap();
    let header = |key: &str| {
        ptx.lines()
            .map(str::trim)
            .find_map(|l| l.strip_prefix(key))
            .unwrap_or_else(|| panic!("the PTX has no {key} line"))
            .trim()
            .to_owned()
    };
    let version = header(".version ");
    let (major, minor) = version.split_once('.').expect("a major.minor PTX ISA version");
    let isa: (u32, u32) = (major.parse().unwrap(), minor.parse().unwrap());
    assert!(
        isa <= rw_post::gpu::PTX_ISA_CEILING,
        "the embedded PTX declares ISA {version}, newer than {:?}, which drivers older than its toolkit refuse \
         (a CUDA 13.2 driver refused 9.4); rebuild it with `rebuild_kernels ptx` under NVRTC 12.0",
        rw_post::gpu::PTX_ISA_CEILING
    );
    let target = rw_post::gpu::PTX_VIRTUAL_ARCHITECTURE.replace("compute_", "sm_");
    assert_eq!(header(".target "), target, "the PTX targets another virtual architecture");
    let m: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
    let entry = m["artifacts"].as_array().unwrap().iter().find(|a| a["file"] == "woof_post.ptx").unwrap();
    assert_eq!(entry["generator"], "NVIDIA NVRTC 12.0", "the PTX's generator is not the NVRTC whose ISA the floor names");
    assert_eq!(m["ptx_virtual_architecture"], rw_post::gpu::PTX_VIRTUAL_ARCHITECTURE);
}

/// Each embedded CUBIN serves its whole architecture family (CUDA runs an
/// sm_XY CUBIN on sm_XZ, Z >= Y); every other card takes the PTX.
#[test]
fn cubins_serve_their_architecture_family() {
    use rw_post::gpu::cubin_file_for;
    for (major, minor) in [(10, 0), (10, 3)] {
        assert_eq!(cubin_file_for(major, minor), Some("woof_post_sm100.cubin"), "sm_{major}{minor}");
    }
    for (major, minor) in [(12, 0), (12, 1)] {
        assert_eq!(cubin_file_for(major, minor), Some("woof_post_sm120.cubin"), "sm_{major}{minor}");
    }
    for (major, minor) in [(6, 0), (7, 0), (7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (11, 0)] {
        assert_eq!(cubin_file_for(major, minor), None, "sm_{major}{minor}");
    }
}

/// The forced-PTX switch really loads the PTX, so the device tests run
/// with `RW_POST_TEST_KERNEL_IMAGE=ptx` prove the PTX path.
#[test]
#[ignore = "needs an NVIDIA GPU"]
fn forced_ptx_opens_the_ptx() {
    let g = GpuPost::open_with(0, rw_post::gpu::KernelImage::Ptx).expect("an NVIDIA GPU");
    if cudarc::gpuwm_hip::requested() {
        assert!(g.device.kernel_artifact.starts_with("HIP "));
        assert!(g.device.kernel_artifact.ends_with("source-bound image"));
    } else {
        assert_eq!(g.device.kernel_artifact, format!("{} PTX", rw_post::gpu::PTX_VIRTUAL_ARCHITECTURE));
    }
    eprintln!("{}", g.device.describe());
}

#[test]
fn one_kernel_set_lists_every_kernel_and_artifact() {
    let m: serde_json::Value = serde_json::from_str(MANIFEST).unwrap();
    assert_eq!(m["kernel_set"], "woof_post");
    let listed: Vec<&str> = m["kernels"].as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
    assert_eq!(listed, rw_post::gpu::KERNELS.to_vec());
    let mut archs = Vec::new();
    for a in m["artifacts"].as_array().unwrap() {
        let file = a["file"].as_str().unwrap();
        let bytes = rw_post::gpu::checked_in_artifact(file).unwrap_or_else(|| panic!("{file} is not compiled in"));
        assert_eq!(format!("{:x}", Sha256::digest(bytes)), a["sha256"].as_str().unwrap(), "{file}");
        archs.push(a["architecture"].as_str().unwrap().to_owned());
    }
    assert!(archs.iter().any(|a| a == "sm_100") && archs.iter().any(|a| a == "sm_120"));
    let files: Vec<&str> = m["artifacts"].as_array().unwrap().iter().map(|a| a["file"].as_str().unwrap()).collect();
    assert_eq!(files, rw_post::gpu::EMBEDDED_ARTIFACTS.to_vec());
    // No source outside the crate's kernels/ directory feeds the set.
    for s in m["sources"].as_array().unwrap() {
        let f = s["file"].as_str().unwrap();
        assert!(!f.contains('/') && !f.contains(char::from(92u8)), "{f}");
    }
}

/// The embedded kernel set is exactly the two certified Blackwell CUBINs and
/// the PTX, and it stays small.  Breakage this prevents: every image is
/// compiled into each binary that links rw-post (rw_grib2export ships in the
/// wheel), and the twelve-architecture set (19.2 MB of CUBIN, 20.4 MB with the
/// PTX) put the 2.8.7 manylinux wheel at about 105 MB, over PyPI's
/// 100,000,000-byte per-file limit, so the release could not be uploaded.  The
/// three images measured 5,123,899 bytes at NVRTC 13.4; the ceiling leaves room
/// for kernel growth and none for a third CUBIN (each is about 1.9 MB).
#[test]
fn only_the_certified_cubins_and_the_ptx_are_embedded() {
    let kernels = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
    let mut images: Vec<String> = std::fs::read_dir(&kernels)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".cubin") || name.ends_with(".ptx"))
        .collect();
    images.sort();
    let mut embedded: Vec<String> = rw_post::gpu::EMBEDDED_ARTIFACTS.iter().map(|f| (*f).to_owned()).collect();
    embedded.sort();
    assert_eq!(images, embedded, "kernels/ holds an image that is not embedded, or the reverse");
    let mut total = 0usize;
    for file in rw_post::gpu::EMBEDDED_ARTIFACTS {
        total += rw_post::gpu::checked_in_artifact(file).unwrap_or_else(|| panic!("{file} is not compiled in")).len();
    }
    for absent in ["woof_post_sm75.cubin", "woof_post_sm90.cubin", "woof_post_sm103.cubin", "woof_post_sm121.cubin"] {
        assert!(rw_post::gpu::checked_in_artifact(absent).is_none(), "{absent} is embedded");
    }
    assert!(total <= 6_000_000, "the embedded kernel images are {total} bytes");
}

#[test]
#[ignore = "needs an NVIDIA GPU"]
fn binary64_maths_library_is_bit_identical() {
    let g = gpu();
    let mut rng = common::Lcg(11);
    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    // Every 2^40-th binary64 word (all signs, exponents, NaNs, infinities,
    // subnormals), then the ranges the post code feeds the functions.
    let mut w: u64 = 0;
    while let Some(next) = w.checked_add(1 << 40) {
        x.push(f64::from_bits(w));
        y.push(0.2857142857142857);
        w = next;
    }
    for _ in 0..400_000 {
        x.push(1400.0 * rng.next() - 700.0);
        y.push(12.0 * rng.next() - 6.0);
        x.push(2.0e5 * rng.next());
        y.push(rng.next());
        x.push(14.0 * rng.next() - 7.0);
        y.push(5.26);
    }
    let out = g.math64_probe(&x, &y).unwrap();
    let mut bad = 0usize;
    for i in 0..x.len() {
        let v = x[i];
        let want = [
            rw_post::math::ln(v),
            rw_post::math::exp(v),
            rw_post::math::pow(v.abs(), y[i]),
            rw_post::math::cbrt(v),
            rw_post::math::sin(v),
            rw_post::math::cos(v),
        ];
        for k in 0..6 {
            let got = out[6 * i + k];
            if got.to_bits() != want[k].to_bits() && !(got.is_nan() && want[k].is_nan()) {
                if bad < 10 {
                    eprintln!("fn {k} x {v:e} y {}: gpu {got:e} cpu {:e}", y[i], want[k]);
                }
                bad += 1;
            }
        }
    }
    eprintln!("{} arguments x 6 functions, {bad} differ", x.len());
    assert_eq!(bad, 0);
}
