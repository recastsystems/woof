//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:9670-9763
//! and mpas_init_atm_vinterp.F:25-109 at 1d6e370e.
//! Strict GNU caller state for the native unset-weight fall-through. The
//! ordinary initializer keeps its existing interpolation implementation.

pub use crate::init::vinterp::{sorted_column, AboveTopLapseRate, Extrap,
    SURFACE_LEVEL_TAG, SURFACE_SENTINEL};

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:9670-9763 at 1d6e370e.
/// The constant-propagated strict kernel retains target in xmm0 at an unset
/// bracket. Its other product reads the measured cold preceding buffer word.
pub fn vertical_interp(target: f32, coords: &[f32], values: &[f32], extrap: Extrap)
    -> Result<f32, AboveTopLapseRate>
{
    if target.is_nan() {
        return Ok(target * values[values.len()-1] + 0.0);
    }
    crate::init::vinterp::vertical_interp(target, coords, values, extrap)
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7180-7204,9670-9763 at 1d6e370e.
/// xmm4/xmm5 in the non-specialized temperature kernel. Preserve products,
/// rather than logical interpolation weights: native overwrites both registers.
#[derive(Clone, Copy)]
pub struct TemperatureWeights(pub [f32; 2]);

impl TemperatureWeights {
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7180-7199 at 1d6e370e.
    /// The logger leaves its final real argument in xmm5. The finite xmm4
    /// formatting scratch is immaterial when multiplied by the cold zero word.
    pub fn from_log(last_real: f32) -> Self { Self([1.0, last_real]) }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:9670-9763 at 1d6e370e.
    pub fn interpolate(&mut self, target:f32, z:&[f32], v:&[f32], extrap:Extrap)
        -> Result<f32, AboveTopLapseRate>
    {
        let n=z.len();
        if target < z[0] {
            if extrap==Extrap::Linear {self.0[0]=v[0];}
            return crate::init::vinterp::vertical_interp(target,z,v,extrap);
        }
        if target >= z[n-1] {
            if extrap==Extrap::Linear {
                self.0=[v[n-1],z[n-1]-z[n-2]];
            }
            return crate::init::vinterp::vertical_interp(target,z,v,extrap);
        }
        for k in 0..n-1 {
            if target>=z[k] && target<z[k+1] {
                let width=z[k+1]-z[k];
                self.0=[((z[k+1]-target)/width)*v[k],((target-z[k])/width)*v[k+1]];
                return Ok(self.0[0]+self.0[1]);
            }
        }
        self.0[0] *= 0.0;
        self.0[1] *= v[n-1];
        Ok(self.0[0]+self.0[1])
    }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7400-7413 at 1d6e370e.
    /// Call the shared host libm, carrying the strict native live registers.
    /// This is an ABI adapter, not an independent logarithm implementation.
    pub fn log(&mut self,value:f32)->f32 { unsafe { rrfs_log_live(value,self.0.as_mut_ptr()) } }

    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7408-7413 at 1d6e370e.
    pub fn exp(&mut self,value:f32)->f32 { unsafe { rrfs_exp_live(value,self.0.as_mut_ptr()) } }
}

// SysV scalar argument xmm0, register-state pointer rdi, scalar return xmm0.
// Save rbx and align the stack before the standard libm call. The two scratch
// registers are explicit state only here; no Rust caller relies on ABI garbage.
// ELF and the SysV ABI only (x86-64 Linux, where the native reference runs):
// the directives below do not assemble for COFF, so a Windows build of the
// crate stopped here (2.9.10 pass 1, R8 Windows cargo gates).
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(r#"
.text
.globl rrfs_log_live
.hidden rrfs_log_live
.type rrfs_log_live,@function
rrfs_log_live:
    push rbx
    mov rbx,rdi
    movss xmm4,[rbx]
    movss xmm5,[rbx+4]
    call logf@PLT
    movss [rbx],xmm4
    movss [rbx+4],xmm5
    pop rbx
    ret
.size rrfs_log_live,.-rrfs_log_live
.globl rrfs_exp_live
.hidden rrfs_exp_live
.type rrfs_exp_live,@function
rrfs_exp_live:
    push rbx
    mov rbx,rdi
    movss xmm4,[rbx]
    movss xmm5,[rbx+4]
    call expf@PLT
    movss [rbx],xmm4
    movss [rbx+4],xmm5
    pop rbx
    ret
.size rrfs_exp_live,.-rrfs_exp_live
"#);
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
unsafe extern "C" {
    fn rrfs_log_live(value:f32,state:*mut f32)->f32;
    fn rrfs_exp_live(value:f32,state:*mut f32)->f32;
}

// Elsewhere the platform libm answers and the carried registers keep their
// values: the native's live registers are an x86-64 Linux glibc artifact, so a
// build for another platform is not word-for-word the native fork's init.
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
unsafe fn rrfs_log_live(value:f32,_state:*mut f32)->f32 { value.ln() }
#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
unsafe fn rrfs_exp_live(value:f32,_state:*mut f32)->f32 { value.exp() }

/// Two live registers a native serial loop carries from one column to the next.
pub trait CarriedRegisters: Copy + Send + Sync {
    fn registers(&self) -> [f32; 2];
    fn from_registers(registers: [f32; 2]) -> Self;
}
impl CarriedRegisters for TemperatureWeights {
    fn registers(&self) -> [f32; 2] { self.0 }
    fn from_registers(registers: [f32; 2]) -> Self { Self(registers) }
}
impl CarriedRegisters for AerosolWeights {
    fn registers(&self) -> [f32; 2] { self.0 }
    fn from_registers(registers: [f32; 2]) -> Self { Self(registers) }
}

/// A native serial column loop on every core, with the serial loop's carried registers.
///
/// The native driver runs the columns in cell order and each column can read
/// the two registers the column before it left (TemperatureWeights in case 7,
/// AerosolWeights in the GOCART vertical pass). Run in order on one core, the
/// case-7 loop cost minutes on RRFS v2 conus3km (2,341,818 cells) while the
/// other cores waited (box V5090, 2026-10-28). The cells are cut into pieces;
/// the column just before each piece is run from two unrelated states (finite,
/// and NaN with distinct payloads). Every way a column can let its entry state
/// through (keep a register, copy it, multiply it, the 0*x fall-through)
/// carries a NaN entry through as NaN, so when both runs leave the same
/// register bits, what the serial loop carries into the piece cannot depend on
/// its own start, and the piece starts from those bits. When they differ, the
/// piece runs on from the piece before it, exactly as the serial loop does.
/// Every column therefore sees the registers the serial loop gives it, and the
/// results come back in cell order (the first error in cell order is still the
/// one returned).
pub fn carried_columns<S: CarriedRegisters, R: Send>(n: usize, first: S,
    column: &(dyn Fn(usize, &mut S) -> R + Sync)) -> Vec<R>
{
    use rayon::prelude::*;
    if n == 0 { return Vec::new(); }
    let size = n.div_ceil((rayon::current_num_threads() * 8).clamp(1, n));
    let starts: Vec<usize> = (0..n).step_by(size).collect();
    let seeds: Vec<Option<S>> = starts.par_iter().map(|&start| {
        if start == 0 { return Some(first); }
        let mut finite = S::from_registers([1.0, 2.0]);
        let mut nan = S::from_registers([f32::from_bits(0x7fc1_2345), f32::from_bits(0xffd5_4321)]);
        let _ = column(start - 1, &mut finite);
        let _ = column(start - 1, &mut nan);
        let (a, b) = (finite.registers(), nan.registers());
        let same = a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits();
        same.then_some(finite)
    }).collect();
    let mut runs: Vec<(usize, usize, S)> = Vec::with_capacity(starts.len());
    for (&start, seed) in starts.iter().zip(seeds) {
        let end = (start + size).min(n);
        match seed {
            Some(state) => runs.push((start, end, state)),
            None => runs.last_mut().expect("the first piece starts from the driver state").1 = end,
        }
    }
    let pieces: Vec<Vec<R>> = runs.into_par_iter()
        .map(|(start, end, mut state)| (start..end).map(|c| column(c, &mut state)).collect())
        .collect();
    pieces.into_iter().flatten().collect()
}

#[cfg(test)]
mod carried_tests {
    use super::*;

    /// A column that, depending on the cell, overwrites the registers, lets
    /// them through, or scales them (the 0*x fall-through), and whose result
    /// reads them: the serial loop and the splitter must agree bit for bit.
    fn column(c: usize, s: &mut AerosolWeights) -> u64 {
        match c % 7 {
            0 | 3 | 5 => s.0 = [c as f32 * 0.5, 1.0 / (c as f32 + 1.0)],
            1 => {}
            2 => { s.0[0] *= 0.0; s.0[1] *= 1.5; }
            _ => s.0[1] = c as f32,
        }
        ((s.0[0].to_bits() as u64) << 32) | s.0[1].to_bits() as u64
    }

    #[test]
    fn carried_columns_equal_the_serial_loop_bit_for_bit() {
        for n in [0usize, 1, 2, 7, 100, 10_007] {
            let first = AerosolWeights([f32::from_bits(0x7fc0_0001), 3.0]);
            let mut state = first;
            let serial: Vec<u64> = (0..n).map(|c| column(c, &mut state)).collect();
            assert_eq!(carried_columns(n, first, &column), serial, "n = {n}");
        }
        // Pass-through everywhere: no piece can start on its own, all runs serially.
        let first = AerosolWeights([1.25, -2.5]);
        let through = |c: usize, s: &mut AerosolWeights| -> f32 { s.0[0] + c as f32 };
        let mut state = first;
        let serial: Vec<f32> = (0..5000).map(|c| through(c, &mut state)).collect();
        assert_eq!(carried_columns(5000, first, &through), serial);
    }
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_vinterp.F:25-109 at 1d6e370e.
/// The separately compiled aerosol kernel stores its products in xmm3/xmm4.
#[derive(Clone, Copy)]
pub struct AerosolWeights(pub [f32;2]);
impl AerosolWeights {
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_vinterp.F:25-109 at 1d6e370e.
    pub fn interpolate(&mut self,target:f32,z:&[f32],v:&[f32])->f32 {
        let n=z.len();
        if target<z[0] {return v[0];}
        if target>=z[n-1] {return v[n-1];}
        for k in 0..n-1 {
            if target>=z[k] && target<z[k+1] {
                let width=z[k+1]-z[k];
                self.0=[((z[k+1]-target)/width)*v[k],((target-z[k])/width)*v[k+1]];
                return self.0[0]+self.0[1];
            }
        }
        self.0[0]*=0.0;
        self.0[1]*=v[n-1];
        self.0[0]+self.0[1]
    }
}
