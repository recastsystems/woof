//! Independent dry-snow microphysical model, Flanner & Zender 2006,
//! doi:10.1029/2005JD006834 equations 5,6,7,14,15,16. No fit table input.
//! The generator integrates 200 radius bins by 40 pore-spacing bins, then
//! fits the published SSA evolution law. All generation uses worker threads.
pub const TEMPERATURES:[f64;8]=[210.,220.,230.,240.,250.,260.,268.,273.];
pub const GRADIENTS:[f64;7]=[0.,10.,25.,50.,100.,200.,300.];
pub const DENSITIES:[f64;8]=[50.,100.,150.,200.,250.,300.,350.,400.];
pub const AGING_WORDS:usize=8+7+8+8*7*8*2;

fn peq(t:f64)->f64 {
    // Murphy & Koop 2005, QJRMS 131:1539 equation 7, saturation over ice.
    (9.550426-5723.265/t+3.53068*t.ln()-0.00728332*t).exp()
}
fn kelvin(t:f64,r:f64)->f64 {(2.*0.109/(461.5*t*917.*r)).exp()}

/// Generate an SSA trajectory at fixed snow T, gradient and density. Particle
/// number is fixed except when a bin fully sublimates, as in the paper.
pub fn trajectory(t:f64,gradient:f64,density:f64,nr:usize,nh:usize)->Vec<f64> {
    let r0=3./(917.*60.); let sigma=2.3_f64.ln();
    let rn=r0*(-2.5*sigma*sigma).exp();
    let pore=((4.*std::f64::consts::PI*917./(3.*density)).cbrt()-2.).max(0.);
    let mut r=Vec::new(); let mut number=Vec::new(); let mut spacing=Vec::new();
    for i in 0..nr {let z=-5.+10.*(i as f64+0.5)/nr as f64;
        for j in 0..nh {let h=-4.+8.*(j as f64+0.5)/nh as f64;
            r.push(rn*(z*sigma).exp());number.push((-0.5*(z*z+h*h)).exp());spacing.push(5.*pore*rn*(z*sigma).exp()*h);
        }
    }
    let moments=|r:&[f64]| ->(f64,f64) {
        r.iter().zip(&number).fold((0.,0.),|(s,v),(r,n)|(s+n*r*r,v+n*r*r*r))
    };
    let (s,v)=moments(&r);let re0=v/s;
    let pressure=peq(t);
    let latent=2.834e6;
    let conductivity=0.024*(t/273.15).powf(1.5);
    let diffusivity=2.12e-5*(t/273.15).powf(1.94);
    let resistance=(latent/(461.5*t)-1.)*latent/(conductivity*t)+461.5*t/(pressure*diffusivity);
    let mut result=vec![re0];
    for _hour in 0..336 {
        // Volume-weighted exponential moments. A 20-term Taylor expansion
        // evaluates the *same* Kelvin exponential at every pore temperature;
        // this avoids an O(N^2) ensemble integral. At the 1 nm numerical
        // disappearance threshold its remainder is below 1e-12.
        let (_,volume)=moments(&r);
        let a=2.*0.109/(461.5*917.);
        let mut moments_k=[0.;21];
        for (radius,n) in r.iter().zip(&number) {if *radius<=0. {continue;}
            let mut term=n*radius.powi(3)/volume;
            for (k,m) in moments_k.iter_mut().enumerate() {
                *m+=term;
                term*=a/(radius*(k+1) as f64);
            }
        }
        let eval_k=|temp:f64|->f64 {moments_k.iter().rev().fold(0.,|p,m|p/temp+m)};
        let mean_kelvin=eval_k(t);
        for j in 0..r.len() {let radius=r[j];if radius<=0. {continue;}
            let h=spacing[j]; // fixed particle-pore position, phi=5, eq14/23
            let other=(t-2.*h*gradient).clamp(150.,330.);
            let ambient_t=t-h*gradient;
            let k_other=eval_k(other);
            let ambient=0.5*ambient_t*(pressure*mean_kelvin/t+peq(other)*k_other/other);
            let dm=4.*std::f64::consts::PI*radius*(ambient/pressure-kelvin(t,radius))/resistance;
            let r3=radius.powi(3)+3600.*dm/(4.*std::f64::consts::PI/3.*917.);
            r[j]=if r3>1e-27 {r3.cbrt()} else {0.};
        }
        let (s,v)=moments(&r);result.push((v/s).max(re0));
    }
    result
}

/// Least-squares fit in log SSA. Search tau, with analytic optimal 1/kappa.
pub fn fit(values:&[f64])->[f64;2] {
    let mut best=(f64::INFINITY,1.,1.);
    for j in 0..601 {
        let tau=(0.01_f64.ln()+j as f64/600.*(10000_f64/0.01).ln()).exp();
        let (mut xy,mut xx)=(0.,0.);
        for h in (1..values.len()).step_by(3) {let x=(1.+h as f64/tau).ln();let y=(values[h]/values[0]).ln();xy+=x*y;xx+=x*x;}
        let slope=(xy/xx).max(1e-8);
        let error=(1..values.len()).step_by(3).map(|h|((values[h]/values[0]).ln()-slope*(1.+h as f64/tau).ln()).powi(2)).sum::<f64>();
        if error<best.0 {best=(error,tau,1./slope);}
    }
    [best.1,best.2]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn published_gradient_growth_and_resolution() {
        // Paper Table 4, -20 C, 200 K/m, 50 kg/m3. The generated
        // trajectory, rather than a copied table, must follow that law.
        let v=trajectory(253.15,200.,50.,200,40);
        let paper=3./(917.*60.)*(1.+336_f64/5.2).powf(1./1.9);
        assert!((v[336]/paper-1.).abs()<0.15);
        let coarse=trajectory(253.15,200.,50.,100,20);
        assert!((coarse[336]/v[336]-1.).abs()<0.02);
        let f=fit(&v);assert!((1.7..2.1).contains(&f[1]));
        assert!(v.iter().all(|x|x.is_finite()&&*x>0.));
    }
}
