//! Case-7 global projections. MPAS v8.4.2 mpas_init_atm_llxy.F:1914-2222.
use super::{hinterp::LatLonProjection,metfile::{MetSlab,IPROJ_LATLON,IPROJ_GAUSS,IPROJ_POLAR}};
use crate::error::{MpasError,MpasResult};

pub enum Projection {
    LatLon(LatLonProjection),
    Gaussian {latitudes:Vec<f32>,lon1:f32,loninc:f32,nx:usize,ny:usize},
    Polar {hemi:f32,truelat:f32,stdlon:f32,rebydx:f32,polei:f32,polej:f32,nx:usize,ny:usize},
}
impl Projection {
    /// MPAS mpas_init_atm_cases.F:4816-4831; mpas_init_atm_llxy.F:437-555,702-780.
    /// The auxiliary ice path locates startlat/startlon at nx/2,ny/2 even
    /// when the WPS startloc tag says SWCORNER. Retain the native driver rule.
    pub fn auxiliary_polar(s:&MetSlab)->MpasResult<Self> {
        if s.iproj!=IPROJ_POLAR {return Err(MpasError::Refusal("native auxiliary SEAICE mapping requires polar stereographic input".into()));}
        let h=s.polar.as_ref().ok_or_else(||MpasError::Refusal("polar projection parameters are missing".into()))?;
        if h.dx<=0. || h.truelat.abs()>90. || s.start_lat.abs()>90. {
            return Err(MpasError::Refusal("polar grid spacing/latitudes violate native map_set admission".into()));
        }
        fn longitude(mut v:f32)->MpasResult<f32> {
            for _ in 0..10 {if v.abs()<=180. {break;} if v< -180. {v=v+360.;} if v>180. {v=v-360.;}}
            if v.abs()>180. {return Err(MpasError::Refusal("polar longitude exceeds native ten-wrap admission".into()));} Ok(v)
        }
        let lon1=longitude(s.start_lon)?;let stdlon=longitude(h.stdlon)?;
        let hemi=if h.truelat<0. {-1.} else {1.};let rebydx=6370000f32/h.dx;
        let radians=std::f32::consts::PI/180f32;
        let reflon=stdlon+90.;let scale=1.+hemi*(h.truelat*radians).sin();
        let ala1=s.start_lat*radians;
        let rsw=((rebydx*ala1.cos())*scale)/(1.+hemi*ala1.sin());
        let alo1=(lon1-reflon)*radians;
        let polei=s.nx as f32/2.-rsw*alo1.cos();
        let polej=s.ny as f32/2.-(hemi*rsw)*alo1.sin();
        Ok(Self::Polar {hemi,truelat:h.truelat,stdlon,rebydx,polei,polej,nx:s.nx,ny:s.ny})
    }
    /// MPAS mpas_init_atm_cases.F:3667-3685 and mpas_init_atm_llxy.F:1914-1953.
    pub fn from_met(s:&MetSlab)->MpasResult<Self> {
        if s.iproj==IPROJ_LATLON {
            if s.delta_lat==0. || s.delta_lon==0. {return Err(MpasError::Refusal("global projection increments must be nonzero".into()));}
            return Ok(Self::LatLon(LatLonProjection {lat1:s.start_lat,lon1:s.start_lon,
                latinc:s.delta_lat,loninc:s.delta_lon,knowni:1.,knownj:1.,nx:s.nx,ny:s.ny}));
        }
        if s.iproj!=IPROJ_GAUSS {return Err(MpasError::Refusal(format!("unsupported case-7 projection {}",s.iproj)));}
        let half=s.delta_lat.round() as usize;
        if half==0 || half.checked_mul(2)!=Some(s.ny) {return Err(MpasError::Refusal("Gaussian nlat must describe all slab rows".into()));}
        let mut latitudes=gaussian_latitudes(s.ny);
        if (latitudes[0]-s.start_lat).abs()>0.01 {for v in &mut latitudes {*v= -1.* *v;}}
        if (latitudes[0]-s.start_lat).abs()>0.01 {return Err(MpasError::Refusal("Gaussian starting latitude differs from native computed latitude by more than 0.01 degrees".into()));}
        Ok(Self::Gaussian {latitudes,lon1:s.start_lon,loninc:360./s.nx as f32,nx:s.nx,ny:s.ny})
    }
    /// MPAS mpas_init_atm_llxy.F:2143-2222 (llij_gauss).
    fn ij(&self,lat:f32,lon:f32)->(f32,f32) {
        match self {
            Self::Polar {hemi,truelat,stdlon,rebydx,polei,polej,..}=> {
                let radians=std::f32::consts::PI/180f32;
                let reflon= *stdlon+90.;let scale=1.+ *hemi*(*truelat*radians).sin();
                let ala=lat*radians;let rm=(*rebydx*ala.cos())*scale/(1.+ *hemi*ala.sin());
                let alo=(lon-reflon)*radians;
                (*polei+rm*alo.cos(),*polej+(*hemi*rm)*alo.sin())
            }
            Self::LatLon(p)=>((lon-p.lon1)/p.loninc+1.,(lat-p.lat1)/p.latinc+1.),
            Self::Gaussian {latitudes:g,lon1,loninc,..}=> {
                let x=(lon-*lon1)/ *loninc+1.;
                let y=if lat.abs()>g[0].abs() {
                    if (lat-g[0]).abs()<(lat-g[g.len()-1]).abs() {1.} else {g.len() as f32}
                } else {
                    let n=g.windows(2).position(|a|(a[0]-lat)*(a[1]-lat)<=0.).expect("finite latitude lies between Gaussian endpoints");
                    ((g[n]-lat)*(n+2) as f32+(lat-g[n+1])*(n+1) as f32)/(g[n]-g[n+1])
                };
                (x,y)
            }
        }
    }
    /// MPAS mpas_init_atm_cases.F:4918-4945 / 7906-7929: periodic wrap and polar clamp.
    pub fn locate(&self,lat:f32,lon:f32)->(f32,f32) {
        let (nx,ny)=match self {Self::LatLon(p)=>(p.nx,p.ny),Self::Gaussian {nx,ny,..}|Self::Polar {nx,ny,..}=>(*nx,*ny)};
        let (mut x,mut y)=self.ij(lat,lon);
        if x<0.5 {(x,y)=self.ij(lat,lon+360.);} else if x>=nx as f32+0.5 {(x,y)=self.ij(lat,lon-360.);}
        if y<0.5 {y=1.;} else if y>=ny as f32+0.5 {y=ny as f32;}
        (x,y)
    }
    /// MPAS v8.4.2 mpas_init_atm_cases.F:4654-4663. The second
    /// surface-wind pass only retries longitudes west of the first column.
    ///
    /// On a regional lat-lon first guess that one-sided retry finds no column for any cell east of the
    /// box's start longitude written west of Greenwich (the met reader folds startlon into -180..180, MPAS
    /// cell longitudes run 0..360): x lands past the slab and u10 = v10 = -1e30 on every cell.  THE BREAKAGE
    /// (hex HRRR start, Vermont 2023-07-10 09Z, 162,723 of 162,723 cells).  So a non-periodic slab also
    /// retries eastward, exactly as this routine's own 3-D pass does (`locate`).  A periodic slab keeps the
    /// native single retry, so every column the native pass found is the same column.
    pub fn locate_surface_wind(&self,lat:f32,lon:f32)->(f32,f32) {
        let (x,y)=self.ij(lat,lon);
        if x<0.5 {return self.ij(lat,lon+360.);}
        if let Self::LatLon(p)=self {
            let periodic=p.nx as f32*p.loninc.abs()>=359.0;
            if !periodic && x>=p.nx as f32+0.5 {return self.ij(lat,lon-360.);}
        }
        (x,y)
    }
    /// MPAS v8.4.2 mpas_init_atm_surface.F:363-379: the surface driver
    /// clamps y before the longitude retry, including the <= boundary.
    pub fn locate_surface(&self,lat:f32,lon:f32)->(f32,f32) {
        let (nx,ny)=match self {Self::LatLon(p)=>(p.nx,p.ny),Self::Gaussian {nx,ny,..}|Self::Polar {nx,ny,..}=>(*nx,*ny)};
        let (mut x,mut y)=self.ij(lat,lon);
        if y<=0.5 {y=1.;} else if y>=ny as f32+0.5 {y=ny as f32;}
        if x<0.5 {(x,y)=self.ij(lat,lon+360.);} else if x>=nx as f32+0.5 {(x,y)=self.ij(lat,lon-360.);}
        (x,y)
    }
}
/// MPAS mpas_init_atm_llxy.F:2095-2140 (lgord).
fn legendre(cosc:f64,n:usize)->f64 {
    let colat=cosc.acos();let mut c1=2f64.sqrt();
    for k in 1..=n {
        // The source evaluates this expression in default REAL, then promotes.
        c1=c1*((1f32-1f32/(4*k*k) as f32).sqrt() as f64);
    }
    let nn=n as f64;let mut ang=nn*colat;let mut s1=0.;let mut c4=1.;let mut a= -1.;let mut b=0.;
    for k in (0..=n).step_by(2) {
        if k==n {c4=0.5*c4;}
        s1=s1+c4*ang.cos();a=a+2.;b=b+1.;
        ang=colat*(nn-k as f64-2.);
        c4=(a*(nn-b+1.)/(b*(nn+nn-a)))*c4;
    }
    s1*c1
}
/// MPAS mpas_init_atm_llxy.F:1956-2092 (gausll and lggaus).
/// Retains REAL64 Newton iteration and the source's default-REAL pi constant.
pub fn gaussian_latitudes(n:usize)->Vec<f32> {
    let pi=std::f32::consts::PI as f64;let nn=n as f64;let fi1=nn+1.;
    let a=nn*fi1/(4.*fi1*fi1-1.).sqrt();let b=fi1*nn/(4.*nn*nn-1.).sqrt();
    let mut out=vec![0.;n];
    for i in 1..=n/2 {
        let mut cosc=(((i as f32-0.5) as f64)*pi/nn+pi*0.5).sin();
        loop {
            let g=legendre(cosc,n);let gm=legendre(cosc,n-1);let gp=legendre(cosc,n+1);
            let gt=(cosc*cosc-1.)/(a*gp-b*gm);let delta=g*gt;cosc=cosc-delta;
            if delta.abs()<=1e-14f32 as f64 {break;}
        }
        let sinc=cosc.acos().sin();let lat=(sinc.acos()*180./pi) as f32;
        out[i-1]=lat;out[n-i]= -lat;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn surface_wind_finds_a_regional_latlon_column_east_of_a_negative_start() {
        let proj = Projection::LatLon(LatLonProjection {
            lat1: 38.225, lon1: -81.38, latinc: 0.025, loninc: 0.025, knowni: 1., knownj: 1., nx: 615, ny: 397 });
        let (x, y) = proj.locate_surface_wind(43.0, 287.0);
        assert!((x - ((287.0 - 360.0 + 81.38) / 0.025 + 1.0)).abs() < 1e-2 && x > 1.0 && x < 615.0, "{x}");
        assert!(y > 1.0 && y < 397.0);
        let global = Projection::LatLon(LatLonProjection {
            lat1: -90., lon1: 0., latinc: 0.25, loninc: 0.25, knowni: 1., knownj: 1., nx: 1440, ny: 721 });
        // A periodic slab keeps the native single retry: 359.9 E stays at the slab's periodic end.
        assert!(global.locate_surface_wind(10.0, 359.9).0 > 1440.0);
    }
}
