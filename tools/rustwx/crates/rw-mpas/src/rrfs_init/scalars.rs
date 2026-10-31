//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:5320-5390,
//! 7244-7302,8468-8712,8820-8850,10495-10561,10970-11135 at 1d6e370e.
//! Whole scalar interpolation/number/aerosol-selection companion to cases 7/9.
use crate::error::{MpasError,MpasResult};
use rayon::prelude::*;
use super::vinterp::{sorted_column,vertical_interp,Extrap,SURFACE_LEVEL_TAG,SURFACE_SENTINEL};

pub const COUNT:usize=15;
pub const OUTPUT:[&str;COUNT]=["qc","qr","qi","qs","qg","nc","ni","nr","ns","ng","nwfa","nifa","smoke_fine","dust_fine","dust_coarse"];
/// Fork cases.F:5046-5059,8081-8100. Snow/graupel number have no input field.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7244-8850 at 1d6e370e.
pub fn input_index(name:&str)->Option<usize> {
    Some(match name {"QC"=>0,"QR"=>1,"QI"=>2,"QS"=>3,"QG"=>4,"QNC"=>5,"QNI"=>6,"QNR"=>7,"QNWFA"=>10,"QNIFA"=>11,"SMKF"=>12,"DSTF"=>13,"DSTC"=>14,_=>return None})
}
/// Fork cases.F:7244-7302,8468-8712. Coordinates are sorted separately
/// for every scalar; the tagged surface is discarded. Below the first
/// unsorted height, case9 copies the already interpolated level above it.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7244-8850 at 1d6e370e.
pub fn interpolate(fg:&[Vec<Vec<f32>>],levels:&[f32],height:&[Vec<f32>],grid:&[Vec<f32>],lbc:bool,smoke:bool,dust:bool)->MpasResult<Vec<Vec<f32>>> {
    let nc=grid.len();let nf=levels.len();
    if nf<3 || fg.len()!=COUNT || height.len()!=nc || grid.iter().any(|g|g.len()!=grid[0].len()) {
        return Err(MpasError::Refusal("RRFS scalar interpolation needs matching cell/level arrays and at least two atmospheric levels plus surface".into()));
    }
    // One zeroed allocation per species (vec![v; n] would copy the first 14 times on one core).
    let nz=grid[0].len()-1;let mut out:Vec<Vec<f32>>=(0..COUNT).map(|_|vec![0.;nc*nz]).collect();
    for (s,field) in out.iter_mut().enumerate() {
        // Fork cases.F:7269-7270,8580-8581 explicitly zeros both
        // number slots; neither has a first-guess interpolation call.
        if s==8 || s==9 {continue;}
        if (s==12 && !smoke) || (s>=13 && !dust) {continue;}
        // The shape check the serial loop made at each cell, made for every
        // cell first; the refusal names no cell, so it is the same refusal.
        if fg[s].len()!=nc || (0..nc).into_par_iter().any(|c| height[c].len()<nf || fg[s][c].len()<nf) {
            return Err(MpasError::Refusal("RRFS scalar first-guess shape differs from level table".into()));
        }
        // Every cell's column is independent of every other's: one core per run of cells.
        field.par_chunks_mut(nz.max(1)).enumerate().for_each(|(c,column)| {
            let mut z=Vec::with_capacity(nf);let mut q=Vec::with_capacity(nf);
            for k in 0..nf {z.push(if levels[k]==SURFACE_LEVEL_TAG {SURFACE_SENTINEL} else {height[c][k]});q.push(fg[s][c][k].max(0.));}
            sorted_column(&mut z,&mut q);
            for k in (0..nz).rev() {
                let target=0.5*(grid[c][k]+grid[c][k+1]);
                let v=vertical_interp(target,&z[..nf-1],&q[..nf-1],Extrap::Constant).expect("constant interpolation cannot request lapse-rate extrapolation");
                column[k]=if s>=12 {v*1.0e9} else {v};
                if lbc && target<height[c][0] && k<nz-1 {column[k]=column[k+1];}
            }
        });
    }
    Ok(out)
}
/// Fork cases.F:10495-10561. Caller reduces maxima across its solve cells
/// before invoking this selector in distributed execution; serial owns all.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7244-8850 at 1d6e370e.
pub fn use_rap_aerosols(tempo_rap:bool,state:&[Vec<f32>])->bool {
    tempo_rap && state[10].iter().copied().fold(f32::NEG_INFINITY,f32::max)!=0.
        && state[11].iter().copied().fold(f32::NEG_INFINITY,f32::max)!=0.
}
#[derive(Clone,Copy)]
pub enum Species {Cloud,Ice,Rain,Snow,Graupel}
/// Fork cases.F:10970-11135. Constants retain default REAL rounding, then
/// the inverse-slope calculation is DOUBLE PRECISION. Single-species call
/// sites use the native optional-argument precedence unambiguously.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7244-8850 at 1d6e370e.
pub fn number_from_mass(species:Species,q:f32,density:f32)->f32 {
    const PI:f32=std::f32::consts::PI;
    const XIMS:f32=(900.0*0.523599)*((2.0*50.0e-6)*(2.0*50.0e-6)*(2.0*50.0e-6));
    // Fork cases.F:11001 folds the constant integer power before rounding
    // the parameter. The pinned strict executable promotes stored REAL
    // 0x322a032c to DOUBLE at calcnfromq ELF 0x55fdd (rodata 0x2f2410).
    // A sequence of separately rounded Rust multiplications is one ULP lower.
    const XGMS:f32=f32::from_bits(0x322a032c);
    const CW:f32=(1000.0*0.523599)*((2.0*9.0e-6)*(2.0*9.0e-6)*(2.0*9.0e-6));
    match species {
        Species::Cloud=>if q>=1.0e-12 {(0.6e9f32/1.225).min(q*(1.0/CW))} else {0.},
        Species::Ice=>if q>=1.0e-12 {q/XIMS} else {0.},
        _=> {
            if q<=1.0e-12 {return 0.;}
            let (factor,n0,g1)=match species {
                Species::Rain=>(1.0f32/((PI*1000.)*8.0e6),8.0e6f32,20.0f32),
                Species::Snow=>(1.0f32/((PI*100.)*3.0e6),3.0e6f32,(-0.8f32+2.)/(-0.8f32+1.)),
                Species::Graupel=>(1.0f32/((PI*700.)*2.0e5),2.0e5f32,20.0f32),
                _=>unreachable!(),
            };
            // Fork cases.F:11065-11105 and pristine calcnfromq ELF 0x55fcb:
            // gfortran calls DOUBLE pow with exponent 0.25, not two square roots.
            let inv=((density as f64*q as f64)*factor as f64).powf(0.25);
            let n1=inv*n0 as f64;let mut nr=(n1*g1 as f64)/20.0;
            if matches!(species,Species::Graupel) {nr=nr.min((density as f64*q as f64)/XGMS as f64);}
            (nr/density as f64) as f32
        }
    }
}
/// What a boundary time holds for rain, cloud and ice number where the
/// first guess carries the mass but not the number (a one-moment parent such
/// as GFS, whose intermediate has QR/QC/QI but no QNR/QNC/QNI).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum AbsentNumbers {
    /// The fork as shipped: cases.F:8835-8849 are commented out, so the
    /// boundary holds rain, cloud and ice mass with zero number. TEMPO reads
    /// (qr>0, nr=0) as the largest drops it allows (2.5 mm median volume
    /// diameter), so GFS drizzle (qr ~2.5e-6 kg/kg) becomes a 20-30 dBZ echo
    /// ring along every limited-area edge. Kept for the native replays.
    Zero,
    /// The same fork lines enabled: nr, nc, ni from qr, qc, qi with the
    /// fork's own calcnfromq (cases.F:10971-11135) wherever the mass is
    /// positive and the number is zero, exactly as cases.F:8825-8834 already
    /// does for snow and graupel. Points whose number arrived are untouched.
    FromMass,
}
impl AbsentNumbers {
    pub fn parse(s:&str)->Result<Self,String> {
        match s {"zero"=>Ok(Self::Zero),"from-mass"=>Ok(Self::FromMass),
            x=>Err(format!("unknown absent_numbers {x}; expected zero or from-mass"))}
    }
    pub fn label(self)->&'static str {match self {Self::Zero=>"zero",Self::FromMass=>"from-mass"}}
}
/// Fork cases.F:8820-8850. Snow and graupel number are diagnosed as the fork
/// ships; rain, cloud and ice number too under [`AbsentNumbers::FromMass`]
/// (the fork's commented-out lines 8835-8849, in their order). Returns the
/// count of points whose rain, cloud and ice number were diagnosed.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:7244-8850 at 1d6e370e.
pub fn initialize_lbc_numbers(state:&mut [Vec<f32>],density:&[f32],absent:AbsentNumbers)->[u64;3] {
    diagnose_numbers(state,density.len(),|i|density[i],absent,true)
}
/// Fork cases.F:7600-7612 (case 7, the cold-start initial state). The fork
/// ships every number diagnosis there commented out ("microphysics handles
/// it on first time step"), so a GFS start carries rain, cloud and ice mass
/// with zero number. TEMPO has no cold start for those three: it reads
/// (qc>0, nc=0) as 100 um droplets that rain out at once and (qr>0, nr=0) as
/// 2.5 mm drops, so the lam7k cut rained >=0.1 mm over 94% of its interior in
/// hour 1 (hex and native alike, 0.40 mm against MRMS 0.01 mm). Under
/// [`AbsentNumbers::FromMass`] rain, cloud and ice number are diagnosed with
/// the same calcnfromq the boundaries use. Snow and graupel number stay with
/// the scheme, as the fork intends: TEMPO's graupel_init (module_mp_tempo_main
/// .F90:1214-1242) sets graupel number AND volume on its first call, but only
/// in a column whose graupel number and volume are both zero; a number made
/// here without a volume would skip it and leave the graupel at the densest
/// class. `density` is the fork's rho_zz at those lines (pre-hydrostatic,
/// before the zz coupling), point `i` = cell `i / nz`, level `i % nz`.
pub fn initialize_init_numbers(state:&mut [Vec<f32>],n:usize,density:impl Fn(usize)->f32+Sync,absent:AbsentNumbers)->[u64;3] {
    diagnose_numbers(state,n,density,absent,false)
}
fn diagnose_numbers(state:&mut [Vec<f32>],n:usize,density:impl Fn(usize)->f32+Sync,absent:AbsentNumbers,snow_graupel:bool)->[u64;3] {
    // Element-wise (each point reads its own masses and writes its own
    // numbers), so the points run on every core with the same arithmetic.
    let (low,high)=state.split_at_mut(5);
    let (qc,qr,qi,qs,qg)=(&low[0][..n],&low[1][..n],&low[2][..n],&low[3][..n],&low[4][..n]);
    let (nc,rest)=high.split_at_mut(1);let (ni,rest)=rest.split_at_mut(1);
    let (nr,rest)=rest.split_at_mut(1);let (ns,rest)=rest.split_at_mut(1);
    let (nc,ni,nr,ns,ng)=(&mut nc[0][..n],&mut ni[0][..n],&mut nr[0][..n],&mut ns[0][..n],&mut rest[0][..n]);
    let from_mass=absent==AbsentNumbers::FromMass;
    (nc,ni,nr,ns,ng).into_par_iter().enumerate().map(|(i,(nc,ni,nr,ns,ng))| {
        let mut made=[0u64;3];
        if !snow_graupel && !from_mass {return made;}
        let rho=density(i);
        if snow_graupel {
            if qs[i]>0. && *ns==0. {*ns=number_from_mass(Species::Snow,qs[i],rho);}
            if qg[i]>0. && *ng==0. {*ng=number_from_mass(Species::Graupel,qg[i],rho);}
        }
        if from_mass {
            if qr[i]>0. && *nr==0. {*nr=number_from_mass(Species::Rain,qr[i],rho);made[0]=(*nr>0.) as u64;}
            if qc[i]>0. && *nc==0. {*nc=number_from_mass(Species::Cloud,qc[i],rho);made[1]=(*nc>0.) as u64;}
            if qi[i]>0. && *ni==0. {*ni=number_from_mass(Species::Ice,qi[i],rho);made[2]=(*ni>0.) as u64;}
        }
        made
    }).reduce(||[0;3],|a,b|[a[0]+b[0],a[1]+b[1],a[2]+b[2]])
}
#[cfg(test)]
mod absent_number_tests {
    use super::*;
    // One cell, three levels: GFS-like mass with no number at level 0,
    // a parent-supplied rain number at level 1, nothing at level 2.
    fn state()->(Vec<Vec<f32>>,Vec<f32>) {
        let mut s:Vec<Vec<f32>>=(0..COUNT).map(|_|vec![0.;3]).collect();
        s[0]=vec![1.0e-4,0.,0.];s[1]=vec![2.5e-6,3.0e-6,0.];s[2]=vec![1.0e-5,0.,0.];
        s[7]=vec![0.,5.0e3,0.];
        (s,vec![1.1,1.0,0.9])
    }
    #[test]
    fn zero_keeps_the_fork_as_shipped() {
        let (mut s,d)=state();
        assert_eq!(initialize_lbc_numbers(&mut s,&d,AbsentNumbers::Zero),[0,0,0]);
        assert_eq!((s[5][0],s[6][0],s[7][0],s[7][1]),(0.,0.,0.,5.0e3));
    }
    #[test]
    fn from_mass_fills_only_absent_numbers_with_calcnfromq() {
        let (mut s,d)=state();
        assert_eq!(initialize_lbc_numbers(&mut s,&d,AbsentNumbers::FromMass),[1,1,1]);
        assert_eq!(s[7][0],number_from_mass(Species::Rain,2.5e-6,1.1));
        assert_eq!(s[5][0],number_from_mass(Species::Cloud,1.0e-4,1.1));
        assert_eq!(s[6][0],number_from_mass(Species::Ice,1.0e-5,1.1));
        assert_eq!(s[7][1],5.0e3,"a parent-supplied number is left alone");
        assert_eq!((s[5][2],s[6][2],s[7][2]),(0.,0.,0.),"no mass, no number");
        // Marshall-Palmer drizzle: 2.5e-6 kg/kg at 1.1 kg/m3 is ~740 drops/kg
        // (mean diameter ~0.1 mm), not the 2.5 mm drops TEMPO makes of nr=0.
        assert!(s[7][0]>700. && s[7][0]<800.,"{}",s[7][0]);
    }
    #[test]
    fn init_fills_rain_cloud_ice_and_leaves_snow_graupel_to_the_scheme() {
        let (mut s,d)=state();
        s[3]=vec![1.0e-6,0.,0.];s[4]=vec![1.0e-6,0.,0.];
        assert_eq!(initialize_init_numbers(&mut s,3,|i|d[i],AbsentNumbers::FromMass),[1,1,1]);
        assert_eq!(s[7][0],number_from_mass(Species::Rain,2.5e-6,1.1));
        assert_eq!(s[5][0],number_from_mass(Species::Cloud,1.0e-4,1.1));
        assert_eq!(s[6][0],number_from_mass(Species::Ice,1.0e-5,1.1));
        assert_eq!(s[7][1],5.0e3,"a parent-supplied number is left alone");
        assert_eq!((s[8][0],s[9][0]),(0.,0.),"snow/graupel number stay with TEMPO's graupel_init");
        let (mut z,d)=state();
        assert_eq!(initialize_init_numbers(&mut z,3,|i|d[i],AbsentNumbers::Zero),[0,0,0]);
        assert_eq!((z[5][0],z[6][0],z[7][0]),(0.,0.,0.),"zero keeps the fork's case 7 as shipped");
    }
}
