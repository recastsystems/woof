//! RRFS sibling at RRFSx/MPAS-Model 1d6e370e.
//! Case-7 global and rotated projections: src/core_init_atmosphere/mpas_init_atm_llxy.F:273-646,702-780,1103-1314,1389-1671,1914-2222 at 1d6e370e.
use super::{hinterp::LatLonProjection,metfile::{MetSlab,IPROJ_LATLON,IPROJ_GAUSS,IPROJ_POLAR}};
use crate::error::{MpasError,MpasResult};

pub enum Projection {
    Lambert {hemi:f32,truelat:f32,stdlon:f32,cone:f32,rebydx:f32,polei:f32,polej:f32,nx:usize,ny:usize},
    Cassini {lat1:f32,lon1:f32,lat0:f32,lon0:f32,stdlon:f32,latinc:f32,loninc:f32,knowni:f32,knownj:f32,nx:usize,ny:usize},
    LatLon(LatLonProjection),
    Gaussian {latitudes:Vec<f32>,lon1:f32,loninc:f32,nx:usize,ny:usize},
    Polar {hemi:f32,truelat:f32,stdlon:f32,rebydx:f32,polei:f32,polej:f32,nx:usize,ny:usize},
}
impl Projection {
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:6930-6937; mpas_init_atm_llxy.F:273-587,702-780 at 1d6e370e.
    /// The auxiliary ice path locates startlat/startlon at nx/2,ny/2 even
    /// when the WPS startloc tag says SWCORNER. Retain the native driver rule.
    pub fn auxiliary_polar(s:&MetSlab)->MpasResult<Self> {
        if s.iproj!=IPROJ_POLAR {return Err(MpasError::Refusal("native auxiliary SEAICE mapping requires polar stereographic input".into()));}
        let h=s.polar.as_ref().ok_or_else(||MpasError::Refusal("polar projection parameters are missing".into()))?;
        if h.dx<=0. || h.truelat.abs()>90. || s.start_lat.abs()>90. {
            return Err(MpasError::Refusal("polar grid spacing/latitudes violate native map_set admission".into()));
        }
        /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:273-2222 at 1d6e370e.
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:5165-5222; mpas_init_atm_llxy.F:1914-1953 at 1d6e370e.
    pub fn from_met(s:&MetSlab)->MpasResult<Self> {
        Self::from_met_driver(s,false)
    }
    /// RRFSx/MPAS-Model mpas_init_atm_surface.F:349-374 at 1d6e370e.
    /// Retains the surface driver's ny-based knowni for Cassini.
    pub fn surface_met(s:&MetSlab)->MpasResult<Self> {Self::from_met_driver(s,true)}
    /// RRFSx/MPAS-Model mpas_init_atm_cases.F:5180-5222,
    /// mpas_init_atm_llxy.F:1103-1180,1531-1559 at 1d6e370e.
    fn from_met_driver(s:&MetSlab,surface:bool)->MpasResult<Self> {
        const RAD:f32=std::f32::consts::PI/180.;
        if s.iproj==3 {
            if s.dx<=0. {return Err(MpasError::Refusal("Lambert dx must be positive (native map_set)".into()));}
            let hemi=if s.truelat1<0. {-1.} else {1.};let cone=lambert_cone(s.truelat1,s.truelat2);
            let rebydx=6370000f32/s.dx;
            let mut delta=s.start_lon-s.stdlon;
            if delta>180. {delta-=360.;} if delta< -180. {delta+=360.;}
            let ctl=(s.truelat1*RAD).cos();
            let rsw=((rebydx*ctl)/cone)*(((90.*hemi-s.start_lat)*RAD/2.).tan()/((90.*hemi-s.truelat1)*RAD/2.).tan()).powf(cone);
            let arg=cone*(delta*RAD);
            let polei=hemi-(hemi*rsw)*arg.sin();let polej=hemi+rsw*arg.cos();
            return Ok(Self::Lambert {hemi,truelat:s.truelat1,stdlon:s.stdlon,cone,rebydx,polei,polej,nx:s.nx,ny:s.ny});
        }
        if s.iproj==6 {
            if s.delta_lat==0. || s.delta_lon==0. {return Err(MpasError::Refusal("Cassini angular increments must be nonzero (native map_set)".into()));}
            let mut lat1=s.center_lat;let mut lon1=s.center_lon;
            let global=(lat1-s.delta_lat/2.+90.).abs()<0.001 && ((lon1-s.delta_lon/2.-s.stdlon)%360.).abs()<0.001;
            if s.pole_lat.abs()!=90. && !global {
                (lat1,lon1)=rotate_coords(lat1,lon1,s.pole_lat,s.pole_lon,s.stdlon,-1);lon1+=s.stdlon;
            }
            return Ok(Self::Cassini {lat1,lon1,lat0:s.pole_lat,lon0:s.pole_lon,stdlon:s.stdlon,
                latinc:s.delta_lat,loninc:s.delta_lon,knowni:((if surface {s.ny} else {s.nx}+1)/2) as f32,
                knownj:((s.ny+1)/2) as f32,nx:s.nx,ny:s.ny});
        }
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:273-2222 at 1d6e370e.
    pub fn ij(&self,lat:f32,lon:f32)->(f32,f32) {
        match self {
            Self::Lambert {hemi,truelat,stdlon,cone,rebydx,polei,polej,..}=> {
                let rad=std::f32::consts::PI/180f32;
                let mut delta=lon-*stdlon;
                if delta>180. {delta-=360.;} if delta< -180. {delta+=360.;}
                let rm=((*rebydx*(*truelat*rad).cos())/ *cone)*(((90.* *hemi-lat)*rad/2.).tan()/((90.* *hemi-*truelat)*rad/2.).tan()).powf(*cone);
                let arg=*cone*(delta*rad);
                ((*polei+(*hemi*rm)*arg.sin())* *hemi,(*polej-rm*arg.cos())* *hemi)
            }
            Self::Cassini {lat1,lon1,lat0,lon0,stdlon,latinc,loninc,knowni,knownj,..}=> {
                let (clat,clon)=if lat0.abs()!=90. {let(a,b)=rotate_coords(lat,lon,*lat0,*lon0,*stdlon,-1);(a,b+*stdlon)} else {(lat,lon)};
                let mut delta=clon-*lon1;
                if delta<0. {delta+=360.;} if delta>360. {delta-=360.;}
                let mut i=delta/ *loninc+*knowni;let j=(clat-*lat1)/ *latinc+*knownj;
                if i<=0. {i+=360./ *loninc;} if i>360./ *loninc {i-=360./ *loninc;}
                (i,j)
            }
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
    /// MPAS cases.F:4918-4945 / 7906-7929: periodic wrap and polar clamp.
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:273-2222 at 1d6e370e.
    pub fn locate(&self,lat:f32,lon:f32)->(f32,f32) {
        let (nx,ny)=match self {Self::LatLon(p)=>(p.nx,p.ny),Self::Gaussian {nx,ny,..}|Self::Polar {nx,ny,..}|Self::Lambert {nx,ny,..}|Self::Cassini {nx,ny,..}=>(*nx,*ny)};
        let (mut x,mut y)=self.ij(lat,lon);
        if x<0.5 {(x,y)=self.ij(lat,lon+360.);} else if x>=nx as f32+0.5 {(x,y)=self.ij(lat,lon-360.);}
        if y<0.5 {y=1.;} else if y>=ny as f32+0.5 {y=ny as f32;}
        (x,y)
    }
    /// MPAS v8.4.2 mpas_init_atm_cases.F:4654-4663. The second
    /// surface-wind pass only retries longitudes west of the first column.
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:273-2222 at 1d6e370e.
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
    /// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:273-2222 at 1d6e370e.
    pub fn locate_surface(&self,lat:f32,lon:f32)->(f32,f32) {
        let (nx,ny)=match self {Self::LatLon(p)=>(p.nx,p.ny),Self::Gaussian {nx,ny,..}|Self::Polar {nx,ny,..}|Self::Lambert {nx,ny,..}|Self::Cassini {nx,ny,..}=>(*nx,*ny)};
        let (mut x,mut y)=self.ij(lat,lon);
        if y<=0.5 {y=1.;} else if y>=ny as f32+0.5 {y=ny as f32;}
        if x<0.5 {(x,y)=self.ij(lat,lon+360.);} else if x>=nx as f32+0.5 {(x,y)=self.ij(lat,lon-360.);}
        (x,y)
    }
}
/// MPAS mpas_init_atm_llxy.F:2095-2140 (lgord).
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:273-2222 at 1d6e370e.
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
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:273-2222 at 1d6e370e.
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

/// RRFSx/MPAS-Model mpas_init_atm_llxy.F:1144-1180 at 1d6e370e.
pub fn lambert_cone(a:f32,b:f32)->f32 {
    let rad=std::f32::consts::PI/180f32;
    if (a-b).abs()>0.1 {
        let cone=(a*rad).cos().log10()-(b*rad).cos().log10();
        cone/(((45.-a.abs()/2.)*rad).tan().log10()-((45.-b.abs()/2.)*rad).tan().log10())
    } else {(a.abs()*rad).sin()}
}
/// RRFSx/MPAS-Model mpas_init_atm_llxy.F:1621-1671 at 1d6e370e.
pub fn rotate_coords(lat:f32,lon:f32,lat_np:f32,lon_np:f32,lon_0:f32,direction:i32)->(f32,f32) {
    const RAD:f32=std::f32::consts::PI/180.;const DEG:f32=180./std::f32::consts::PI;
    let phi=lat_np*RAD;let lam_np=lon_np*RAD;let lam_0=lon_0*RAD;let rlat=lat*RAD;let rlon=lon*RAD;
    let dlam=if direction<0 {std::f32::consts::PI-lam_0} else {lam_np};
    let sinphi=(phi.cos()*rlat.cos())*(rlon-dlam).cos()+phi.sin()*rlat.sin();
    let cosphi=(1.-sinphi*sinphi).sqrt();
    let mut coslam=(phi.sin()*rlat.cos())*(rlon-dlam).cos()-phi.cos()*rlat.sin();
    let mut sinlam=rlat.cos()*(rlon-dlam).sin();
    if cosphi!=0. {coslam/=cosphi;sinlam/=cosphi;}
    let olat=DEG*sinphi.asin();let mut olon=DEG*(((sinlam.atan2(coslam)-dlam)-lam_0)+lam_np);
    while olon< -180. {olon+=360.;} while olon>180. {olon-=360.;}
    (olat,olon)
}
impl Projection {
    /// RRFSx/MPAS-Model mpas_init_atm_llxy.F:1497-1613 at 1d6e370e.
    pub fn inverse_cassini(&self,i:f32,j:f32)->(f32,f32) {
        let Self::Cassini {lat1,lon1,lat0,lon0,stdlon,latinc,loninc,knowni,knownj,..}=self else {unreachable!("inverse used by Cassini velocity rotation")};
        let mut iw=i-*knowni;let jw=j-*knownj;
        if iw<0. {iw+=360./ *loninc;} if iw>=360./ *loninc {iw-=360./ *loninc;}
        let clat=jw* *latinc+*lat1;let mut clon=iw* *loninc+*lon1;
        if clon< -180. {clon+=360.;} if clon>180. {clon-=360.;}
        if lat0.abs()!=90. {rotate_coords(clat,clon-*stdlon,*lat0,*lon0,*stdlon,1)} else {(clat,clon)}
    }
}
/// RRFSx/MPAS-Model mpas_init_atm_cases.F:11137-11201 at 1d6e370e.
/// The driver supplies its final met header, including auxiliary files.
pub fn rotate_velocities(header:&MetSlab,lat:&[f32],lon:&[f32],u:&mut [Vec<f32>],v:&mut [Vec<f32>])->MpasResult<()> {
    const RAD:f32=std::f32::consts::PI/180.;const DEG:f32=180./std::f32::consts::PI;
    if lat.len()!=u.len() || lon.len()!=u.len() || v.len()!=u.len() {return Err(MpasError::Refusal("velocity rotation arrays differ in edge count".into()));}
    let proj=if header.iproj==6 {Some(Projection::from_met(header)?)} else {None};
    let mut xlonc=header.stdlon*RAD;if xlonc<0. {xlonc+=4.*1.0f32.asin();}
    let cone=if header.iproj==3 {lambert_cone(header.truelat1,header.truelat2)} else {0.};
    for e in 0..u.len() {
        let mut angle=lon[e]-xlonc;
        if header.iproj==3 {angle=cone*angle;}
        else if header.iproj==6 {
            let proj=proj.as_ref().expect("Cassini projection");
            let(x,y)=proj.ij(lat[e]*DEG,lon[e]*DEG);
            let(pl,po)=proj.inverse_cassini(x,y+0.1);let(ml,mo)=proj.inverse_cassini(x,y-0.1);
            let mut dlon=po-mo;if dlon>180. {dlon-=360.;} else if dlon< -180. {dlon+=360.;}
            let dlat=pl-ml;
            angle= -((-lat[e].cos()*dlon)*RAD).atan2(dlat*RAD);
        }
        let cos=angle.cos();let sin=angle.sin();
        for k in 0..u[e].len() {let a=u[e][k];let b=v[e][k];u[e][k]=cos*a+sin*b;v[e][k]= -sin*a+cos*b;}
    }
    Ok(())
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
