//! Finite-particle scattering for the simulated radar's C/X bands.
//!
//! The PSD and weighted-melting equations are ported from the evaluated
//! polarimetric operator. Tables use rustmatrix 2.2.0 (MIT), equal-volume
//! spheroids, spherical Gaussian canting and dual Maxwell-Garnett mixing.
//! References: 10.1175/2007MWR2083.1, 10.1175/2010JAMC2363.1,
//! 10.5194/amt-11-3883-2018. Nothing in this module advances model state.

use std::f32::consts::PI;
use std::sync::OnceLock;
use sha2::{Digest,Sha256};
use crate::wrf_radar_physics::BulkContribution;

pub const ELEVATIONS: [f32; 5] = [0., 10., 30., 60., 90.];
pub const MODEL_ID: &str = "finite-particle-tmatrix-weighted-melting-v1";

pub struct Table {
    pub wavelength_mm: f32,
    diameter: Vec<f32>,
    density: Vec<f32>,
    liquid: Vec<f32>,
    values: Vec<f32>,
    log_step: f32,
}

impl Table {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 32 || &bytes[..8] != b"RWTMAT01" {
            return Err("radar scattering table has an invalid header".into());
        }
        let word = |offset| u32::from_le_bytes(bytes[offset..offset+4].try_into().unwrap());
        let [nd, nr, nw, ne, nc] = [word(8),word(12),word(16),word(20),word(24)].map(|n| n as usize);
        // Bounds prevent malformed table dimensions from overflowing offsets.
        if !(64..=4096).contains(&nd) || nr != 9 || nw != 6 || ne != 5 || nc != 7 {
            return Err("radar scattering table axes do not match the operator".into());
        }
        let count = nd + nr + nw + ne + 5*nr*nw*ne*nd*nc;
        if bytes.len() != 32 + count*4 {
            return Err("radar scattering table is truncated or has trailing values".into());
        }
        let wavelength_mm = f32::from_bits(word(28));
        if !wavelength_mm.is_finite() || wavelength_mm <= 0. {
            return Err("radar scattering table wavelength is invalid".into());
        }
        let all: Vec<f32> = bytes[32..].chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        if all.iter().any(|v| !v.is_finite()) {
            return Err("radar scattering table contains a nonfinite value".into());
        }
        let diameter = all[..nd].to_vec();
        let density = all[nd..nd+nr].to_vec();
        let liquid = all[nd+nr..nd+nr+nw].to_vec();
        if [&diameter,&density,&liquid].iter().any(|a| a.windows(2).any(|w| w[1] <= w[0]))
            || diameter[0] <= 0. || liquid[0] != 0. || liquid[nw-1] != 1.
            || all[nd+nr+nw..nd+nr+nw+ne] != ELEVATIONS {
            return Err("radar scattering table coordinates are invalid".into());
        }
        let log_min = diameter[0].ln();
        let log_step = (diameter[nd-1]/diameter[0]).ln()/(nd-1) as f32;
        if diameter.iter().enumerate().any(|(i,d)|(d.ln()-log_min-i as f32*log_step).abs()>2e-5) {
            return Err("radar scattering diameters are not logarithmically spaced".into());
        }
        let values = all[nd+nr+nw+ne..].to_vec();
        Ok(Self { wavelength_mm, diameter, density, liquid, values, log_step })
    }

    /// Add all elevation rows before deriving any logarithm or ratio.
    fn scatter(&self, sp: usize, density: f32, wet: f32, k: usize,
               count: f32, output: &mut [[f32; 7]; 5]) {
        let (ir,r) = bracket(&self.density,density);
        let (iw,w) = bracket(&self.liquid,wet);
        for a in 0..2 { for b in 0..2 {
            let weight = count * if a == 1 {r} else {1.-r} * if b == 1 {w} else {1.-w};
            let base = (((sp*9+ir+a)*6+iw+b)*5)*self.diameter.len()*7;
            for e in 0..5 {
                let offset = base+(e*self.diameter.len()+k)*7;
                for j in 0..7 { output[e][j] += weight*self.values[offset+j]; }
            }
        }}
    }
}

pub fn table(band: &str) -> Result<&'static Table, String> {
    static C: OnceLock<Result<Table,String>> = OnceLock::new();
    static X: OnceLock<Result<Table,String>> = OnceLock::new();
    let (slot, bytes, metadata): (_, &[u8], &str) = match band {
        "C" => (&C, include_bytes!("../data/scattering-C.bin"), include_str!("../data/scattering-C.json")),
        "X" => (&X, include_bytes!("../data/scattering-X.bin"), include_str!("../data/scattering-X.json")),
        _ => return Err("finite-particle table must select C or X; S retains its existing operator".into()),
    };
    slot.get_or_init(|| {
        let metadata:serde_json::Value=serde_json::from_str(metadata).map_err(|e|e.to_string())?;
        let digest=format!("{:x}",Sha256::digest(bytes));
        if metadata["native_sha256"].as_str()!=Some(&digest) || metadata["band"].as_str()!=Some(band) {
            return Err("radar table bytes do not match their band/provenance digest".into());
        }
        let table=Table::from_bytes(bytes)?;
        if table.wavelength_mm!=if band=="C" {53.5} else {33.3} {
            return Err("radar table wavelength does not match the selected band".into());
        }
        Ok(table)
    }).as_ref().map_err(Clone::clone)
}

fn bracket(axis: &[f32], x: f32) -> (usize,f32) {
    let i = axis.partition_point(|v| *v < x).saturating_sub(1).min(axis.len()-2);
    (i, ((x-axis[i])/(axis[i+1]-axis[i])).clamp(0.,1.))
}

#[derive(Clone, Copy, Default)]
pub struct Cell {
    /// rain, snow, ice, graupel, hail, per kg of dry air.
    pub mass: [f32;5],
    pub number: [f32;5],
    pub volume: [f32;2],
    pub rime: f32,
    pub rime_volume: f32,
    pub density: f32,
    pub temperature: f32,
}

#[derive(Clone, Default)]
pub struct Scattering {
    pub moments: [BulkContribution;5],
    pub mass_error: f32,
    pub clipped_mass_fraction: f32,
}

// m(D)=a D^b, N(D)=N0 D^mu exp(-lambda D^nu), SI units.
fn rows(scheme: i32) -> Result<[[f32;5];5],String> {
    let p = PI/6.;
    match scheme {
        8 => Ok([[1000.*p,3.,0.,1.,1000.],[0.069,2.,0.,1.,100.],
                 [890.*p,3.,0.,1.,890.],[400.*p,3.,0.,1.,400.],[900.*p,3.,0.,1.,900.]]),
        9 => Ok([[1000.*p,3.,0.,1.,1000.],[0.1597,2.078,0.,1.,100.],
                 [500.*p,3.,0.,1.,500.],[400.*p,3.,0.,1.,400.],[900.*p,3.,0.,1.,900.]]),
        18 => Ok([[1000.*p,3.,0.,1.,1000.],[0.106214,2.,-0.4,3.,100.],
                  [900.*p,3.,2.,3.,900.],[500.*p,3.,0.,1.,500.],[900.*p,3.,1.,1.,900.]]),
        50 | 51 => Ok([[1000.*p,3.,0.,1.,1000.],[0.069,2.,0.,1.,100.],
                       [917.*p,3.,0.,1.,917.],[400.*p,3.,0.,1.,400.],[900.*p,3.,0.,1.,900.]]),
        _ => Err(format!("radar_band_microphysics_unsupported: scheme {scheme} has no finite-particle PSD closure; using the S-band bulk closure would change the requested scattering")),
    }
}

pub fn validate_scheme(scheme: i32) -> Result<(),String> { rows(scheme).map(|_| ()) }

pub fn p3_table() -> &'static [f32] {
    static P3: OnceLock<Vec<f32>> = OnceLock::new();
    P3.get_or_init(|| include_bytes!("../data/scattering-p3.bin")
        .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

fn gamma(x: f32) -> f32 { crate::wrf_radar_physics::gamma(f64::from(x)) as f32 }

fn field3(tc: f32, m2: f32) -> f32 {
    let a = 5.065339-0.062659*tc-3.032362*3.+0.029469*tc*3.-0.000285*tc*tc
        +0.31255*9.+0.000204*tc*tc*3.+0.003199*tc*9.-0.015952*27.;
    let b = 0.476221-0.015896*tc+0.165977*3.+0.007468*tc*3.-0.000141*tc*tc
        +0.060366*9.+0.000079*tc*tc*3.+0.000594*tc*9.-0.003577*27.;
    10f32.powf(a)*m2.powf(b)
}

#[derive(Clone, Copy, Default)]
struct IceLaw { d0:f32,d1:f32,d2:f32,cg:f32,cp:f32,bp:f32 }
impl IceLaw {
    fn new(f:f32,rho:f32) -> Self {
        let mut l=Self { d0:(PI*917./(6.*0.0121)).powf(1./(1.9-3.)),
            cg:rho*PI/6.,cp:0.0121,bp:1.9,d1:1e5,d2:1e5 };
        if f<=0. { return l; }
        if f>=1. {l.d1=(0.0121/l.cg).powf(1./1.1);l.cp=l.cg;l.bp=3.;return l;}
        l.cp=0.0121/(1.-f);
        for _ in 0..256 {
            l.d1=(0.0121/l.cg).powf(1./1.1);l.d2=(l.cp/l.cg).powf(1./1.1);
            let dep=6.*0.0121/(PI*(-0.1))*(l.d2.powf(-0.1)-l.d1.powf(-0.1))/(l.d2-l.d1);
            let old=l.cg;l.cg=PI/6.*(rho*f+dep*(1.-f));
            if ((l.cg-old)/l.cg).abs()<0.01 {break;}
        }
        l
    }
    fn mass(self,d:f32)->f32 {
        if d<=self.d0 {PI/6.*917.*d*d*d} else if d<=self.d1 {0.0121*d.powf(1.9)}
        else if d<=self.d2 {self.cg*d*d*d} else {self.cp*d.powf(self.bp)}
    }
    fn exponent(self,d:f32)->f32 {
        if d<=self.d0 {3.} else if d<=self.d1 {1.9} else if d<=self.d2 {3.} else {self.bp}
    }
}

fn p3lookup(tab:&[f32],q:f32,n:f32,rim:f32,den:f32,field:usize)->f32 {
    let x=((q/n).log10()+18.)*3.444606-11.;
    let y=3.*rim;
    let z=if den<=650. {(den-50.)*0.005} else {(den-650.)*0.004+3.};
    let (x,y,z)=(x.clamp(0.,49.),y.clamp(0.,3.),z.clamp(0.,4.));
    let (ix,iy,iz)=((x as usize).min(48),(y as usize).min(2),(z as usize).min(3));
    let (x,y,z)=(x-ix as f32,y-iy as f32,z-iz as f32);
    let mut out=0.;
    for a in 0..2 {for b in 0..2 {for c in 0..2 {
        out+=if a==1 {z} else {1.-z} * if b==1 {y} else {1.-y}
            * if c==1 {x} else {1.-x} * tab[(((iz+a)*4+iy+b)*50+ix+c)*14+field];
    }}}
    out
}

fn wet_psd(d:f32,dry:f32,pm:f32,b:f32,fw:f32,rl:f32,rn:f32,share:f32)->(f32,f32) {
    let mr=PI/6.*1000.*d*d*d;
    let mm=fw*fw*mr+(1.-fw*fw)*pm;
    let dr=(6.*mm/(PI*1000.)).cbrt();
    let jac=dr*(fw*fw*3.*mr+(1.-fw*fw)*b*pm)/(3.*mm*d);
    (fw*share*(rn-rl*dr).exp()*jac+(1.-fw)*dry,mm)
}

/// PSD quadrature at every table elevation, parallelized over model cells by
/// the caller. The order within a cell is fixed and independent of worker count.
pub fn integrate(table:&Table, cell:Cell, scheme:i32, p3:&[f32])->Result<Scattering,String> {
    let rows=rows(scheme)?;
    if scheme>=50 && p3.len()!=5*4*50*14 {
        return Err("radar P3 scattering requires the packaged 5.4 two-moment PSD table".into());
    }
    let mut out=Scattering::default();
    let mut acc=[[0f32;7];5];
    let qice=cell.mass[1..].iter().map(|q|q.max(0.)).sum::<f32>();
    let rain=cell.mass[0].max(0.);
    let fw=rain/(rain+qice).max(1e-30);
    let weighted=cell.temperature>273.15 && rain>0. && qice>0.;
    let rm=(rain*cell.density).max(1e-30);
    let rn=(cell.number[0]*cell.density).max(1e-12);
    let mut rl=(PI*1000.*rn/rm).cbrt();
    let mut rlog=(rn*rl).ln();
    if scheme>=50 {rl=rl.clamp(499.99997,1e5);rlog=(rm/(PI*1000.)).ln()+4.*rl.ln();}
    for s in 0..5 {
        let mass=cell.mass[s].max(0.)*cell.density;
        if mass<1e-14 {continue;}
        let mut number=(cell.number[s]*cell.density).max(1e-12);
        let [mut a,b,mut mu,mut nu,_den]=rows[s];
        if scheme==18 && s>=3 {
            let den=if cell.volume[s-3]>0. {(cell.mass[s]/cell.volume[s-3]).clamp(if s==3 {100.} else {300.},900.)}
                else if s==3 {500.} else {900.};
            a=den*PI/6.;
            let pv=mass/(den*number.max(1e-3));
            let (lo,hi)=if s==3 {(1.413717494e-11,4.188791536e-6)} else {(1.413717581e-11,3.351033229e-5)};
            if pv<lo || pv>hi {number=mass/(den*pv.clamp(lo,hi));}
        }
        let mut lam=(a*number/mass*gamma((mu+b+1.)/nu)/gamma((mu+1.)/nu)).powf(nu/b);
        let mut logn=(number*nu).ln()+(mu+1.)/nu*lam.ln()-gamma((mu+1.)/nu).ln();
        if scheme>=50 && s==0 {lam=lam.clamp(499.99997,1e5);logn=(mass/a/6.).ln()+4.*lam.ln();}
        let snow=scheme==8 && s==1;
        let ice=scheme>=50 && s==2;
        let (m0,mrat)=if snow {let m2=mass/0.069;let m0=m2/field3((cell.temperature-273.15).min(-0.1),m2);(m0,m2*m0*m0*m0)} else {(0.,0.)};
        let mut law=IceLaw::default();
        if ice {
            let rim=(cell.rime/cell.mass[2].max(1e-30)).clamp(0.,1.);
            let rd=(cell.rime/cell.rime_volume.max(1e-30)).clamp(50.,900.);
            let mut nn=cell.number[2].max(1e-16);
            let nmin=p3lookup(p3,cell.mass[2],nn,rim,rd,6);
            let nmax=p3lookup(p3,cell.mass[2],nn,rim,rd,7);
            nn=nn.min(nmin*cell.mass[2]).max(nmax*cell.mass[2]);
            lam=p3lookup(p3,cell.mass[2],nn,rim,rd,12);
            mu=p3lookup(p3,cell.mass[2],nn,rim,rd,13);nu=1.;law=IceLaw::new(rim,rd);
            let sum:f32=table.diameter.iter().enumerate().map(|(k,mm)| {
                let d=mm*0.001;
                (mu*d.ln()-lam*d).exp()*law.mass(d)*d*table.log_step*edge(k,table.diameter.len())
            }).sum();
            logn=(mass/sum.max(1e-38)).ln();
        }
        let dry=|d:f32| if snow {mrat*(490.6*(-20.78*m0*d).exp()+17.46*(m0*d).powf(0.6357)*(-3.29*m0*d).exp())}
            else {(logn+mu*d.ln()-lam*d.powf(nu)).exp()};
        let pm=|d:f32| if ice {law.mass(d)} else {a*d.powf(b)};
        let exponent=|d:f32| if ice {law.exponent(d)} else {b};
        let mut norm=1.;
        if weighted && s>0 {
            let sum:f32=table.diameter.iter().enumerate().map(|(k,mm)| {
                let d=mm*0.001;
                let (wn,wm)=wet_psd(d,dry(d),pm(d),exponent(d),fw,rl,rlog,cell.mass[s]/qice);
                wn*wm*d*table.log_step*edge(k,table.diameter.len())
            }).sum();
            norm=(mass+rm*cell.mass[s]/qice)/sum.max(1e-38);
        }
        let (mut summass,mut clipped,mut scattered)=(0.,0.,0.);
        for (k,&mm) in table.diameter.iter().enumerate() {
            let d=mm*0.001;
            let raw=dry(d);let weight=d*table.log_step*edge(k,table.diameter.len());
            let particle=pm(d);summass+=raw*weight*particle;
            let (count,particle_density,wet,smass)=if weighted {
                if s==0 {continue;}
                let (wn,wm)=wet_psd(d,raw,particle,exponent(d),fw,rl,rlog,cell.mass[s]/qice);
                let count=wn*norm*weight;
                (count,wm/(PI/6.*d*d*d),fw,count*wm)
            } else {
                let count=raw*weight;
                (count,particle/(PI/6.*d*d*d),0.,count*particle)
            };
            scattered+=smass;
            if particle_density<table.density[0] || particle_density>table.density[8] {clipped+=smass;}
            table.scatter(s,particle_density,wet,k,count,&mut acc);
        }
        out.mass_error=out.mass_error.max((summass/mass-1.).abs());
        out.clipped_mass_fraction=out.clipped_mass_fraction.max(clipped/scattered.max(1e-30));
    }
    for (e,a) in acc.iter().enumerate() {
        if a.iter().any(|v| !v.is_finite()) {return Err("radar PSD quadrature produced nonfinite scattering".into());}
        out.moments[e]=BulkContribution {zh:a[0],zv:a[1],cov_re:a[2],cov_im:a[3],
            kdp_deg_km:a[4],ah_db_km:a[5].max(0.),av_db_km:a[6].max(0.),
            ..BulkContribution::default()};
    }
    Ok(out)
}

fn edge(k:usize,n:usize)->f32 {if k==0 || k==n-1 {0.5} else {1.}}

/// The same between-step Thompson diagnosis as the engine's init/finalize.
pub fn thompson_graupel_number(q:f32,rho:f32)->f32 {
    if q<=1e-12 {return 0.;}
    let mass=q*rho;let a=PI*400./6.;
    let power=(3.+2./7.*(mass.max(1e-9).log10()+8.)).clamp(2.,6.);
    let lambda=(10f32.powf(power)*a*6./mass).powf(0.25).clamp(3.672/0.0254,3.672/50e-6);
    let number=(mass/(6.*a)*lambda.powi(3)/rho).max(1e-6/rho);
    let lambda=(a*6.*number/q).cbrt().clamp(3.672/0.0254,3.672/50e-6);
    q/(6.*a)*lambda.powi(3)
}

pub fn at_elevation(moments:&[BulkContribution;5],elevation:f32)->BulkContribution {
    let (i,w)=bracket(&ELEVATIONS,elevation.abs());
    let mut acc=crate::wrf_radar_physics::PolarAccumulator::default();
    acc.add(1.-w,moments[i]);acc.add(w,moments[i+1]);
    let s=acc.finalize();
    BulkContribution {zh:s.zh,zv:s.zv,cov_re:s.cov_re,cov_im:s.cov_im,
        kdp_deg_km:s.kdp_deg_km,ah_db_km:s.ah_db_km,av_db_km:s.av_db_km,
        fall_speed_mps:s.fall_speed_mps,fall_speed_variance_m2s2:s.fall_speed_variance_m2s2}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tables_are_bound_to_band_and_digest() {
        assert_eq!(table("C").unwrap().wavelength_mm,53.5);
        assert_eq!(table("X").unwrap().wavelength_mm,33.3);
        assert!(table("S").is_err());
        assert!(Table::from_bytes(b"RWTMAT01").is_err());
    }
    #[test]
    fn native_rain_quadrature_closes_mass_and_retains_extinction() {
        let cell=Cell {mass:[0.001,0.,0.,0.,0.],number:[1e5,0.,0.,0.,0.],density:1.,temperature:285.,..Default::default()};
        let c=integrate(table("C").unwrap(),cell,8,&[]).unwrap();
        let x=integrate(table("X").unwrap(),cell,8,&[]).unwrap();
        assert!(c.mass_error<1e-3 && x.mass_error<1e-3);
        assert!(c.moments[0].ah_db_km>0.);
        assert!(x.moments[0].ah_db_km>c.moments[0].ah_db_km);
    }
    #[test]
    fn angular_sampling_preserves_signed_phase_and_large_extinction() {
        let mut moments=[BulkContribution {zh:10.,zv:10.,cov_re:10.,ah_db_km:3.,av_db_km:2.,kdp_deg_km:-2.,..Default::default()};5];
        moments[1].kdp_deg_km=2.;
        let sample=at_elevation(&moments,5.);
        assert_eq!(sample.kdp_deg_km,0.);
        assert_eq!(sample.ah_db_km,3.);
        assert_eq!(sample.av_db_km,2.);
    }
}
