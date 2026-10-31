//! The shared column state (specification section 3.4), built once per frame.
//!
//! | quantity | grid | definition |
//! | --- | --- | --- |
//! | theta | mass | T + 300 K [R1] |
//! | r | mass | max(QVAPOR, 0) |
//! | q | mass | r / (1 + r) |
//! | theta_v | mass | theta (1 + r/eps) / (1 + r) [R2] |
//! | p | mass | hydrostatic pressure: P_HYD, else the moist integration below (D1) |
//! | p_full | mass | P + PB |
//! | tk | mass | theta (p_full / p0)^(Rd/cp) (D1: T from full pressure) |
//! | tv | mass | tk (1 + r/eps) / (1 + r) [R2] |
//! | z_mass | mass | mean of the two interface heights (D1 note) |
//! | u, v | mass | mean of the two faces of each cell [R1] |
//! | p_int | interface | hydrostatic interface pressure, ground (k = 0) to top |
//! | z_int | interface | (PH + PHB) / g, metres MSL |
//!
//! Hydrostatic pressure without P_HYD ([R1] sections 2.1 to 2.3): the dry
//! interface pressure is p_d = C3F mu_d + C4F + P_TOP ([`crate::hybrid`]),
//! and the moist hydrostatic pressure is integrated down from P_TOP, each
//! layer weighing its dry thickness times (1 + total water mixing ratio)
//! (QVAPOR + QCLOUD + QRAIN + QICE + QSNOW + QGRAUP + QHAIL, negative
//! species counted as zero, summed in that order).  The mass level sits at
//! half its layer's weight below the interface above it.
//!
//! Layout: level-major `[k][cell]`, `cell = j * nx + i`, matching the GPU.
//! [`build_column`] is the CPU twin of `woof_post_state_v1`.

use rayon::prelude::*;

use crate::consts::*;
use crate::thermo::{powpos, virtual_temperature};

/// Slots of the packed mass-level input (`IN3_*` in the kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MassInput {
    T = 0,
    P = 1,
    PB = 2,
    QVapor = 3,
    QCloud = 4,
    QRain = 5,
    QIce = 6,
    QSnow = 7,
    QGraupel = 8,
    QHail = 9,
    PHyd = 10,
}

impl MassInput {
    pub const COUNT: usize = 11;
    pub const ALL: [MassInput; 11] = [
        Self::T,
        Self::P,
        Self::PB,
        Self::QVapor,
        Self::QCloud,
        Self::QRain,
        Self::QIce,
        Self::QSnow,
        Self::QGraupel,
        Self::QHail,
        Self::PHyd,
    ];
    /// The history variable that fills the slot.
    pub fn carrier(&self) -> &'static str {
        match self {
            Self::T => "T",
            Self::P => "P",
            Self::PB => "PB",
            Self::QVapor => "QVAPOR",
            Self::QCloud => "QCLOUD",
            Self::QRain => "QRAIN",
            Self::QIce => "QICE",
            Self::QSnow => "QSNOW",
            Self::QGraupel => "QGRAUP",
            Self::QHail => "QHAIL",
            Self::PHyd => "P_HYD",
        }
    }
    /// Required for the state; the others are optional carriers.
    pub fn required(&self) -> bool {
        matches!(self, Self::T | Self::P | Self::PB | Self::QVapor)
    }
}

/// Slots of the packed mass-level state (`ST_*` in the kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum MassState {
    Theta = 0,
    Q = 1,
    P = 2,
    PFull = 3,
    Tk = 4,
    Tv = 5,
    ZMass = 6,
    U = 7,
    V = 8,
    /// Water-vapour mixing ratio max(QVAPOR, 0).
    R = 9,
    /// Virtual potential temperature theta (1 + r/eps) / (1 + r) [R2].
    ThetaV = 10,
}

impl MassState {
    pub const COUNT: usize = 11;
}

/// Slots of the packed interface state (`SI_*` in the kernel).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum InterfaceState {
    PInt = 0,
    ZInt = 1,
}

/// Grid shape of one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Shape {
    pub fn ncell(&self) -> usize {
        self.nx * self.ny
    }
    pub fn mass_volume(&self) -> usize {
        self.nz * self.ncell()
    }
    pub fn interface_volume(&self) -> usize {
        (self.nz + 1) * self.ncell()
    }
}

/// The 3D inputs of the state build, packed as both devices consume them.
#[derive(Clone, Debug)]
pub struct StateInputs {
    pub shape: Shape,
    /// `MassInput::COUNT` slots of `nz * ncell`; absent slots hold zeros.
    pub mass: Vec<f32>,
    /// Bit `s` set when slot `s` carries data.
    pub present: u32,
    pub ph: Vec<f32>,
    pub phb: Vec<f32>,
    /// `nz * ny * (nx + 1)`.
    pub u_stag: Vec<f32>,
    /// `nz * (ny + 1) * nx`.
    pub v_stag: Vec<f32>,
    pub mu: Vec<f32>,
    pub mub: Vec<f32>,
    pub c3f: Vec<f32>,
    pub c4f: Vec<f32>,
    pub p_top: f32,
}

impl StateInputs {
    pub fn has(&self, slot: MassInput) -> bool {
        (self.present >> slot as u32) & 1 == 1
    }
    pub fn slot(&self, slot: MassInput) -> &[f32] {
        let v = self.shape.mass_volume();
        &self.mass[slot as usize * v..(slot as usize + 1) * v]
    }
}

/// The shared column state of one frame, level-major.
#[derive(Clone, Debug, PartialEq)]
pub struct ColumnState {
    pub shape: Shape,
    /// `MassState::COUNT` slots of `nz * ncell`.
    pub mass: Vec<f32>,
    /// 2 slots of `(nz + 1) * ncell`: p_int then z_int.
    pub interface: Vec<f32>,
}

impl ColumnState {
    pub fn mass_slot(&self, slot: MassState) -> &[f32] {
        let v = self.shape.mass_volume();
        &self.mass[slot as usize * v..(slot as usize + 1) * v]
    }
    pub fn interface_slot(&self, slot: InterfaceState) -> &[f32] {
        let v = self.shape.interface_volume();
        &self.interface[slot as usize * v..(slot as usize + 1) * v]
    }
    /// One mass-level value.
    pub fn at(&self, slot: MassState, k: usize, cell: usize) -> f32 {
        self.mass_slot(slot)[k * self.shape.ncell() + cell]
    }
}

/// Raw output pointers shared by the column tasks.  Each task writes only
/// the words of its own columns, so no two tasks touch the same word.
#[derive(Clone, Copy)]
struct Out {
    mass: *mut f32,
    iface: *mut f32,
}
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

/// Build one column: the CPU twin of `woof_post_state_v1`.
///
/// # Safety
/// `out` must point at buffers of the shape's state sizes, and no other
/// caller may write column `cell` concurrently.
unsafe fn build_column(inp: &StateInputs, cell: usize, out: Out) {
    let Shape { nx, ny, nz } = inp.shape;
    let ncell = nx * ny;
    let i = cell % nx;
    let j = cell / nx;
    let vol = nz * ncell;
    let ivol = (nz + 1) * ncell;
    let m = &inp.mass;
    let st = |slot: MassState, at: usize| unsafe { out.mass.add(slot as usize * vol + at) };
    let si = |slot: InterfaceState, at: usize| unsafe { out.iface.add(slot as usize * ivol + at) };

    for k in 0..=nz {
        let at = k * ncell + cell;
        unsafe { *si(InterfaceState::ZInt, at) = (inp.ph[at] + inp.phb[at]) / G };
    }

    let mud = inp.mu[cell] + inp.mub[cell];
    let mut p_above = inp.p_top;
    let mut pd_above = inp.c3f[nz] * mud + inp.c4f[nz] + inp.p_top;
    unsafe { *si(InterfaceState::PInt, nz * ncell + cell) = p_above };
    let has_phyd = inp.has(MassInput::PHyd);
    for k in (0..nz).rev() {
        let at = k * ncell + cell;
        let pd = inp.c3f[k] * mud + inp.c4f[k] + inp.p_top;
        let mut qt = 0.0f32;
        for s in MassInput::QVapor as usize..=MassInput::QHail as usize {
            if (inp.present >> s) & 1 == 1 {
                let w = m[s * vol + at];
                if w > 0.0 {
                    qt = qt + w;
                }
            }
        }
        let weight = (pd - pd_above) * (1.0 + qt);
        let pmass = p_above + 0.5 * weight;
        p_above = p_above + weight;
        pd_above = pd;
        unsafe {
            *si(InterfaceState::PInt, at) = p_above;
            *st(MassState::P, at) =
                if has_phyd { m[MassInput::PHyd as usize * vol + at] } else { pmass };
        }
    }

    for k in 0..nz {
        let at = k * ncell + cell;
        let theta = m[MassInput::T as usize * vol + at] + 300.0;
        let rraw = m[MassInput::QVapor as usize * vol + at];
        let r = if rraw > 0.0 { rraw } else { 0.0 };
        let pfull = m[MassInput::P as usize * vol + at] + m[MassInput::PB as usize * vol + at];
        let tk = theta * powpos(pfull / P0, KAPPA);
        let urow = (k * ny + j) * (nx + 1);
        let vrow = (k * (ny + 1) + j) * nx;
        unsafe {
            *st(MassState::Theta, at) = theta;
            *st(MassState::Q, at) = r / (1.0 + r);
            *st(MassState::PFull, at) = pfull;
            *st(MassState::Tk, at) = tk;
            *st(MassState::Tv, at) = virtual_temperature(tk, r);
            *st(MassState::ZMass, at) =
                0.5 * (*si(InterfaceState::ZInt, at) + *si(InterfaceState::ZInt, at + ncell));
            *st(MassState::U, at) = 0.5 * (inp.u_stag[urow + i] + inp.u_stag[urow + i + 1]);
            *st(MassState::V, at) = 0.5 * (inp.v_stag[vrow + i] + inp.v_stag[vrow + nx + i]);
            *st(MassState::R, at) = r;
            *st(MassState::ThetaV, at) = virtual_temperature(theta, r);
        }
    }
}

/// Validate the input sizes against the shape.
pub fn check_inputs(inp: &StateInputs) -> Result<(), String> {
    let s = inp.shape;
    let checks = [
        ("mass", inp.mass.len(), MassInput::COUNT * s.mass_volume()),
        ("PH", inp.ph.len(), s.interface_volume()),
        ("PHB", inp.phb.len(), s.interface_volume()),
        ("U", inp.u_stag.len(), s.nz * s.ny * (s.nx + 1)),
        ("V", inp.v_stag.len(), s.nz * (s.ny + 1) * s.nx),
        ("MU", inp.mu.len(), s.ncell()),
        ("MUB", inp.mub.len(), s.ncell()),
        ("C3F", inp.c3f.len(), s.nz + 1),
        ("C4F", inp.c4f.len(), s.nz + 1),
    ];
    for (name, got, want) in checks {
        if got != want {
            return Err(format!("{name} has {got} words, expected {want}"));
        }
    }
    for slot in MassInput::ALL {
        if slot.required() && !inp.has(slot) {
            return Err(format!("required carrier {} is absent", slot.carrier()));
        }
    }
    Ok(())
}

/// Build the column state on the CPU (rayon over rows of columns).
pub fn build_cpu(inp: &StateInputs) -> Result<ColumnState, String> {
    check_inputs(inp)?;
    let shape = inp.shape;
    let mut mass = vec![0.0f32; MassState::COUNT * shape.mass_volume()];
    let mut interface = vec![0.0f32; 2 * shape.interface_volume()];
    let out = Out { mass: mass.as_mut_ptr(), iface: interface.as_mut_ptr() };
    (0..shape.ny).into_par_iter().for_each(|j| {
        let out = out;
        for i in 0..shape.nx {
            // SAFETY: each column index is visited by exactly one task.
            unsafe { build_column(inp, j * shape.nx + i, out) };
        }
    });
    Ok(ColumnState { shape, mass, interface })
}

/// Interface pressures of a height-coordinate frame
/// ([`crate::hybrid::CoefficientSource::HeightCoordinate`]), written over the
/// `p_int` slot of a built state, all cores (rayon over columns).
///
/// A WRF-shaped frame gathered from an MPAS history has no dry column mass,
/// so the hydrostatic integration of the state kernel has nothing to
/// integrate.  The mass-level pressure is the history's own (P + PB, carried
/// as P_HYD), and each interface takes, from the hypsometric equation
/// (Wallace and Hobbs, Atmospheric Science, 2nd ed., eq. 3.29):
///
/// * the ground (k = 0): `PSFC` where the frame carries a finite positive
///   value, else the lowest mass level reduced down by its own virtual
///   temperature, `p0 exp(g (z0 - zi0) / (Rd Tv0))`;
/// * between mass levels: log-linear in height, which is the hypsometric
///   equation with the layer's mean virtual temperature;
/// * the top (k = nz): the top mass level carried up by its own virtual
///   temperature.
pub fn fill_height_coordinate_interfaces(state: &mut ColumnState, psfc: Option<&[f32]>) {
    let Shape { nx, ny, nz } = state.shape;
    let ncell = nx * ny;
    let vol = nz * ncell;
    let ivol = (nz + 1) * ncell;
    let p = state.mass[MassState::PFull as usize * vol..(MassState::PFull as usize + 1) * vol].to_vec();
    let tv = state.mass[MassState::Tv as usize * vol..(MassState::Tv as usize + 1) * vol].to_vec();
    let zm = state.mass[MassState::ZMass as usize * vol..(MassState::ZMass as usize + 1) * vol].to_vec();
    let (pint_part, zint_part) = state.interface.split_at_mut(ivol);
    let zi: &[f32] = &zint_part[..ivol];
    let pint = pint_part;
    // Column-major work, then a level-major scatter: each task owns one column.
    let cols: Vec<Vec<f32>> = (0..ncell)
        .into_par_iter()
        .map(|c| {
            let at = |k: usize| k * ncell + c;
            let mut out = vec![f32::NAN; nz + 1];
            let ok = (0..nz).all(|k| p[at(k)].is_finite() && p[at(k)] > 0.0 && zm[at(k)].is_finite() && tv[at(k)].is_finite());
            if !ok || nz == 0 {
                return out;
            }
            let ground = psfc.map(|s| s[c]).filter(|v| v.is_finite() && *v > 0.0);
            out[0] = match ground {
                Some(v) => v,
                None => (f64::from(p[at(0)]) * ((f64::from(G) * f64::from(zm[at(0)] - zi[at(0)])) / (f64::from(RD) * f64::from(tv[at(0)]))).exp()) as f32,
            };
            for k in 1..nz {
                let (z0, z1) = (f64::from(zm[at(k - 1)]), f64::from(zm[at(k)]));
                let (l0, l1) = (f64::from(p[at(k - 1)]).ln(), f64::from(p[at(k)]).ln());
                let w = if z1 != z0 { (f64::from(zi[at(k)]) - z0) / (z1 - z0) } else { 0.5 };
                out[k] = (l0 + w * (l1 - l0)).exp() as f32;
            }
            let top = nz - 1;
            out[nz] = (f64::from(p[at(top)])
                * (-(f64::from(G) * f64::from(zi[at(nz)] - zm[at(top)])) / (f64::from(RD) * f64::from(tv[at(top)]))).exp())
                as f32;
            out
        })
        .collect();
    pint.par_chunks_mut(ncell).enumerate().for_each(|(k, row)| {
        for (c, v) in row.iter_mut().enumerate() {
            *v = cols[c][k];
        }
    });
}
