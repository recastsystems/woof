//! Published coefficients, independently transcribed from papers, never source.
//! He 2017 doi:10.1175/JCLI-D-17-0300.1 Table 3; Fu 2007
//! doi:10.1175/2007JAS2289.1 Tables 1/2 (smooth crystals).
pub const CENTERS: [f64; 6] = [0.475, 1.055, 1.655, 2.2, 3., 3.75];
const A: [[f64; 3]; 6] = [
    [0.976029,0.521042,-0.000266792], [0.967798,0.496181,0.00114088],
    [1.00111,0.183711,0.000237011], [1.00224,0.137082,-0.000235905],
    [0.964295,0.0550598,0.000840449], [0.997475,0.0848743,-0.000471484]];
const B: [[f64; 3]; 6] = [
    [0.7938904,-0.3987320,0.1349959], [0.8030084,-0.3723287,0.1115697],
    [0.8513932,-0.3924784,0.09853958], [0.8692241,-0.3259404,0.05557793],
    [0.7085850,0.04429054,-0.1233493], [0.6412701,-0.1726586,0.]];
const C: [[f64; 3]; 6] = [
    [0.5292852,0.1140557,0.003165543], [0.5425909,0.1143152,0.002014810],
    [0.5601598,0.1143814,0.001780838], [0.6023407,0.1071238,0.0006987734],
    [0.6473899,0.1353873,-0.01882932], [0.4634944,0.1914431,-0.02277872]];

/// Shape-preserving cubic Hermite interpolation (Fritsch/Carlson). Endpoint
/// extension is explicit: the fits cover 0.25–4 um, not the full solar range.
pub fn pchip(x: &[f64], y: &[f64], q: f64) -> f64 {
    assert_eq!(x.len(), y.len());
    let n=x.len();
    if q<=x[0] { return y[0]; }
    if q>=x[n-1] { return y[n-1]; }
    let h:Vec<_>=x.windows(2).map(|p|p[1]-p[0]).collect();
    let d:Vec<_>=y.windows(2).zip(&h).map(|(p,h)|(p[1]-p[0])/h).collect();
    let mut m=vec![0.;n];
    for i in 1..n-1 { if d[i-1]*d[i]>0. {
        let a=2.*h[i]+h[i-1]; let b=h[i]+2.*h[i-1];
        m[i]=(a+b)/(a/d[i-1]+b/d[i]);
    }}
    for (i,a,b) in [(0,0,1),(n-1,n-2,n-3)] {
        let mut v=((2.*h[a]+h[b])*d[a]-h[a]*d[b])/(h[a]+h[b]);
        if v*d[a]<=0. {v=0.;} else if d[a]*d[b]<0. && v.abs()>3.*d[a].abs() {v=3.*d[a];}
        m[i]=v;
    }
    let i=x.partition_point(|v|*v<q)-1;
    let t=(q-x[i])/h[i];
    (2.*t.powi(3)-3.*t*t+1.)*y[i]+(t.powi(3)-2.*t*t+t)*h[i]*m[i]
        +(-2.*t.powi(3)+3.*t*t)*y[i+1]+(t.powi(3)-t*t)*h[i]*m[i+1]
}

/// g at six fit bands, using equal-Re Mie omega at each band's center.
pub fn shape_g(shape: usize, radius: f64, omega: &[f64;6], wavelength: f64) -> f64 {
    assert!((1..=3).contains(&shape));
    let sf=[1.,0.929,0.788,0.712][shape];
    let ar:f64=if shape==1 {0.5} else {2.5};
    let rs=if shape==3 {radius/0.544} else {radius};
    let values:Vec<_>=(0..6).map(|i| {
        let p=if ar<=1. {B[i]} else {C[i]};
        let z=if ar<=1. {ar} else {ar.ln()};
        let gp=p[0]+p[1]*z+p[2]*z*z;
        let gh=(1.-gp)/(2.*omega[i])+gp;
        (gh*A[i][0]*(sf/0.788_f64).powf(A[i][1])*(2.*rs).powf(A[i][2])).clamp(0.,0.999999)
    }).collect();
    pchip(&CENTERS,&values,wavelength).clamp(0.,0.999999)
}

/// He 2017 Table 4 Fu–Liou spectral coalbedo enhancement, reference BC
/// radius 0.1 um, density 1700 kg/m3. Concentrations in ng/g (ppb).
pub fn bc_enhancement(ppb:f64,w:f64)->f64 {
    if ppb<=0. || w>1. {return 1.;}
    let x=[0.225,0.275,0.315,0.345,0.38,0.42,0.46,0.50,0.545,0.605,0.665,0.72,0.765,0.825,0.935];
    let d=[[2.48045,0.977209,0.395960],[4.70305,0.973317,0.204820],
        [4.68619,0.979650,0.207410],[4.67369,0.984579,0.209390],[4.65040,0.993537,0.213030],
        [2.40364,0.995955,0.418570],[0.795408,0.995218,1.29682],[0.292745,0.974284,3.75514],
        [0.0863396,0.981193,12.7372],[0.0276299,0.981239,39.3293],[0.0140864,0.955515,87.8918],
        [0.00865705,0.910491,186.969],[0.00612971,0.874196,345.600],
        [0.00445697,0.827238,708.637],[0.0306648,0.482870,1410.67]];
    let y:Vec<_>=d.iter().map(|d|(d[0]*(ppb+d[2]).powf(d[1])).max(1.).ln()).collect();
    pchip(&x,&y,w).exp()
}

/// He 2019 doi:10.1029/2019MS001737 Table 1. Clear/overcast RRTM fits.
/// This is enhancement of the *snow coalbedo*, not a multiplier on dust MAC.
pub fn dust_enhancement(ppm:f64,w:f64,overcast:bool)->f64 {
    if ppm<=0. || w>1.2422 {return 1.;}
    let x=[0.2316,0.304,0.39315,0.53325,0.7016,1.0102];
    let a=[-21.307,-15.815,if overcast {-9.2880} else {-9.1921},1.1116,1.0302,1.0185];
    let b=if overcast {[117.46,93.241,40.605,0.37389,0.014800,0.00028921]}
        else {[117.46,93.251,40.033,0.36578,0.014560,0.00026997]};
    let c=if overcast {[0.99701,0.99781,0.99848,1.0035,1.0024,1.0356]}
        else {[0.99701,0.99781,0.99849,1.0036,1.0022,1.0376]};
    // The published low-concentration UV fit can fall below unity. The
    // physical nonnegative absorption-increment constraint supplies its floor.
    let y:Vec<_>=(0..6).map(|i|(a[i]+b[i]*ppm.powf(c[i])).max(1.).ln()).collect();
    pchip(&x,&y,w).exp()
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn shape_visible_order() {
        let o=[1.;6];
        let g:Vec<_>=(1..=3).map(|s|shape_g(s,100.,&o,0.55)).collect();
        assert!(g[0]>g[1] && g[1]>g[2]);
        assert!((g[2]-0.74).abs()<0.04);
    }
    #[test] fn interpolation_preserves_monotonicity() {
        for i in 0..100 {let v=pchip(&[0.,1.,2.],&[0.,1.,1.1],i as f64/50.);assert!((0.0..=1.1).contains(&v));}
    }
}
