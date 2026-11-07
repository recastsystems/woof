//! Optical tables generated from equations and published refractive indices.
//! No SNICAR program, aging lookup, or upstream optical table is an input.
use static_fields::{snow_optics::*,snow_shapes::*,snow_aging::*};
use std::{io::Write,path::PathBuf};
fn psd(re:f64,w:f64,m:Complex,rho:f64,host:f64)->[f64;3] {
    let mut s=[0.;4];let l=1.5_f64.ln();
    for j in 0..24 {let z=-4.+(j as f64+0.5)/3.;let r=re*(-2.5*l*l+z*l).exp();
        let n=(-0.5*z*z).exp();let q=mie(2.*std::f64::consts::PI*r*host/w,m);let a=n*r*r;
        s[0]+=a*q[0];s[1]+=a*(q[0]-q[1]);s[2]+=a*q[1]*q[2];s[3]+=n*r*r*r;
    }
    let factor=3e6/(4.*rho*s[3]);[s[0]*factor,s[1]*factor,s[2]/(s[0]-s[1])]
}
// Gauss-Legendre quadrature written directly from the Legendre recurrence.
fn quadrature(n:usize)->Vec<(f64,f64)> {
    let mut out=Vec::with_capacity(n);
    for i in 0..n {let mut z=(std::f64::consts::PI*(i as f64+0.75)/(n as f64+0.5)).cos();
        let mut dp=0.;
        for _ in 0..12 {let (mut p,mut q)=(1.,0.);for j in 1..=n {let a=p;p=((2*j-1) as f64*z*p-(j-1) as f64*q)/j as f64;q=a;}
            dp=n as f64*(z*p-q)/(z*z-1.);let step=p/dp;z-=step;if step.abs()<2e-15 {break;}}
        out.push(((1.+z)*0.5,1./((1.-z*z)*dp*dp)));
    }out
}
// Whicker 2022 equations 7--10: adjusted real refractive index
// followed by the unpolarised real Fresnel coefficients.
fn fresnel(mu:f64,n:Complex)->f64 {
    let si=1.-mu*mu; let a=n.re*n.re-n.im*n.im;
    let nr=(0.5*(a+si+((a-si).powi(2)+4.*n.re*n.re*n.im*n.im).sqrt())).sqrt();
    let ct=(1.-si/(nr*nr)).max(0.).sqrt();
    let rs=(mu-nr*ct)/(mu+nr*ct);let rp=(nr*mu-ct)/(nr*mu+ct);
    (0.5*(rs*rs+rp*rp)).clamp(0.,1.)
}
// Bin-integrated ASTM G173 incident spectra. A direct normal spectrum and
// diffuse irradiance on its 37-degree plane are independent weights. The
// reference incidence is 11.19 degrees (NLR reference geometry).
fn solar_bins(path:&std::path::Path)->Vec<[f64;2]> {
    let text=std::fs::read_to_string(path).unwrap();let points:Vec<[f64;3]>=text.lines().skip(2).map(|l| {
        let v:Vec<f64>=l.split(',').map(|v|v.parse().unwrap()).collect();
        [v[0]*0.001,v[3],(v[2]-v[3]*(11.19_f64.to_radians().cos())).max(0.)]
    }).collect();
    (0..NW).map(|i| {let lo=0.2+i as f64*0.01;let hi=lo+0.01;let mut sum=[0.;2];
        for p in points.windows(2) {let left=lo.max(p[0][0]);let right=hi.min(p[1][0]);if right<=left {continue;}
            for k in 0..2 {let f=|x:f64|p[0][k+1]+(x-p[0][0])/(p[1][0]-p[0][0])*(p[1][k+1]-p[0][k+1]);sum[k]+=0.5*(f(left)+f(right))*(right-left);}
        }sum
    }).collect()
}
fn dust_index(path:&std::path::Path)->Vec<[f64;3]> {
    std::fs::read_to_string(path).unwrap().lines().map(|l| {let v:Vec<f64>=l.split(',').take(3).map(|v|v.parse().unwrap()).collect();[v[0],v[1],v[2]]}).collect()
}
fn main() {
    let a:Vec<_>=std::env::args_os().collect();assert_eq!(a.len(),5,"snow-optics ICE.yml DUST.dat SOLAR.csv OUT.bin");
    let ice=read_index(&PathBuf::from(&a[1])).unwrap();
    let dust=dust_index(&PathBuf::from(&a[2]));let solar=solar_bins(&PathBuf::from(&a[3]));
    let radii:Vec<_>=(0..NR).map(|i|30.*50_f64.powf(i as f64/(NR-1) as f64)).collect();
    let bubbles:Vec<_>=(0..NR).map(|i|30.*(20000_f64/30.).powf(i as f64/(NR-1) as f64)).collect();
    let quad=quadrature(10000);eprintln!("quadrature ready");
    let rows=std::thread::scope(|s| {
        let jobs:Vec<_>=(0..8).map(|worker| {let ice=&ice;let radii=&radii;let bubbles=&bubbles;let quad=&quad;let dust=&dust;let solar=&solar;
            s.spawn(move || {
                let center:Vec<[f64;6]>=radii.iter().map(|r|std::array::from_fn(|i|{let v=psd(*r,CENTERS[i],index_at(ice,CENTERS[i]),917.,1.);1.-v[1]/v[0]})).collect();
                let mut rows=Vec::new();
                for iw in (worker..NW).step_by(8) {let w=0.205+iw as f64*0.01;let m=index_at(ice,w);
                    let solar=solar[iw];
                    let base:Vec<_>=radii.iter().map(|r|psd(*r,w,m,917.,1.)).collect();let mut p=base.clone();
                    for shape in 1..=3 {for ir in 0..NR {let mut v=base[ir];v[2]=shape_g(shape,radii[ir],&center[ir],w);p.push(v);}}
                    // Air bubbles: per m3 air, host-ice wavelength, nonabsorbing
                    // inclusion. Ice absorption is separately 4*pi*k/lambda.
                    for r in bubbles {let mut v=psd(*r,w,Complex::new(1./m.re,0.),1.,m.re);v[1]=0.;p.push(v);}
                    // Bond & Bergstrom 2006 strongly absorbing BC, reference
                    // radius and density of He 2017's internal-mixing fit.
                    let q=mie(2.*std::f64::consts::PI*0.1/w,Complex::new(1.95,0.79));let f=3e6/(4.*1700.*0.1);
                    p.push([q[0]*f,(q[0]-q[1])*f,q[2]]);
                    // Zhang et al. 2024 coarse-mode mean mineral mixture RI,
                    // CC-BY-4.0 doi:10.5281/zenodo.14171144. Self-written Mie,
                    // not the dataset's separately supplied bulk optics.
                    p.push(psd(1.,w,index_at(dust,w),2500.,1.));
                    let down=quad.iter().map(|(mu,q)|2.*mu*q*fresnel(*mu,m)).sum::<f64>();
                    // Reverse interface: split integral at critical angle.
                    let critical=(1.-1./(m.re*m.re)).max(0.).sqrt();
                    let up=critical*critical+quad.iter().map(|(x,q)| {let mu=critical+(1.-critical)*x;
                        2.*mu*q*(1.-critical)*fresnel(mu,Complex::new(1./m.re,0.))}).sum::<f64>();
                    let mix:Vec<_>=(0..32).map(|i| {let f=i as f64/31.;[bc_enhancement(0.01*1e6_f64.powf(f),w),dust_enhancement(0.01*1e5_f64.powf(f),w,false),dust_enhancement(0.01*1e5_f64.powf(f),w,true)]}).collect();
                    rows.push((iw,solar,p,[m.re,m.im,down,up],mix));
                    if iw%80<8 {eprintln!("optical wavelength {iw}/480");}
                } rows
            })}).collect();jobs.into_iter().flat_map(|j|j.join().unwrap()).collect::<Vec<_>>()
    });
    let mut words=vec![0_f32;TABLE_WORDS];
    for i in 0..NR {words[i]=radii[i] as f32;words[NR+i]=bubbles[i] as f32;}
    for (iw,solar,p,fr,mix) in rows {
        words[2*NR+iw]=(0.205+iw as f64*0.01) as f32;words[2*NR+NW+iw]=solar[0] as f32;words[2*NR+2*NW+iw]=solar[1] as f32;
        for (ir,v) in p.iter().enumerate() {for k in 0..3 {words[OPTICS_START+(ir*NW+iw)*3+k]=v[k] as f32;}}
        for k in 0..4 {words[FRESNEL_START+iw*4+k]=fr[k] as f32;}
        for (i,v) in mix.iter().enumerate() {for k in 0..3 {words[MIX_START+(i*NW+iw)*3+k]=v[k] as f32;}}
    }
    let aging=std::thread::scope(|s| {let jobs:Vec<_>=(0..8).map(|worker|s.spawn(move || {
        let mut values=Vec::new();for i in (worker..8*7*8).step_by(8) {let t=TEMPERATURES[i/(7*8)];let g=GRADIENTS[(i/8)%7];let d=DENSITIES[i%8];
            let v=trajectory(t,g,d,200,40);let f=fit(&v);values.push((i,f));
            if i%56<8 {eprintln!("aging {i}/448 T={t} G={g} rho={d} tau={} k={}",f[0],f[1]);}
        }values})).collect();jobs.into_iter().flat_map(|j|j.join().unwrap()).collect::<Vec<_>>()});
    let mut off=AGE_START;for axis in [&TEMPERATURES[..],&GRADIENTS[..],&DENSITIES[..]] {for v in axis {words[off]=*v as f32;off+=1;}}
    for (i,v) in aging {words[off+2*i]=v[0] as f32;words[off+2*i+1]=v[1] as f32;}
    let mut f=std::fs::File::create(&a[4]).unwrap();f.write_all(MAGIC).unwrap();for v in &words {f.write_all(&v.to_le_bytes()).unwrap();}drop(f);
    read_table(&PathBuf::from(&a[4])).unwrap();println!("generated {} words: four habits, bubbles, BC, dust, Fresnel, internal mixing, microphysical aging; eight workers",TABLE_WORDS);
}
