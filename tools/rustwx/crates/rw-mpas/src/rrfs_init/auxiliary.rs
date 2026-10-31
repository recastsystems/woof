//! RRFS sibling at RRFSx/MPAS-Model 1d6e370e.
//! Complete optional case-7 sea-ice and ocean mixed-layer interpolation paths.
//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:6903-7114 at 1d6e370e.
use super::{metfile::{MetSlab,IPROJ_LATLON},projection::Projection,hinterp::{self,Slab,InterpContext,Method,Underflow},DEG_PER_RAD};
use crate::error::{MpasError,MpasResult};

#[derive(Clone,Copy)]
pub enum Kind {SeaIce,OceanDepth}

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_cases.F:6903-7114 at 1d6e370e. Each matching record
/// overwrites the previous one, and land cells are filled with native zero.
pub fn interpolate(slabs:&[MetSlab],kind:Kind,lat:&[f32],lon:&[f32],land:&[i32],
    destination:&mut [f32],underflow:Underflow)->MpasResult<usize> {
    if lat.len()!=land.len() || lon.len()!=land.len() || destination.len()!=land.len() {
        return Err(MpasError::Refusal("auxiliary interpolation arrays have different cell counts".into()));
    }
    let mut consumed=0;
    for s in slabs {
        let matching=match kind {Kind::SeaIce=>s.field=="SEAICE",Kind::OceanDepth=>s.field.contains("OMLD")};
        if !matching {continue;}
        let (projection,missing,methods)=match kind {
            Kind::SeaIce=>(Projection::auxiliary_polar(s)?,1.01,vec![Method::FourPoint,Method::WtAverage4]),
            Kind::OceanDepth=> {
                if s.iproj!=IPROJ_LATLON {return Err(MpasError::Refusal("native OMLD input requires regular lat-lon projection".into()));}
                (Projection::from_met(s)?,0.,vec![Method::FourPoint,Method::WtAverage4,Method::Search])
            }
        };
        // Unlike the main meteorology path, MPAS passes the auxiliary
        // rslab with bounds 1..nx, without the periodic three-column halo.
        let mut array=Slab::new(1,s.nx as i32,1,s.ny as i32);
        for j in 1..=s.ny {for i in 1..=s.nx {
            array.set(i as i32,j as i32,s.values[(j-1)*s.nx+i-1]);
        }}
        let context=InterpContext {array:&array,msgval:missing,maskval:None,mask:None,underflow,underflow_hits:None};
        for c in 0..land.len() {
            destination[c]=if land[c]==1 {0.} else {
                let (x,y)=projection.locate(lat[c]*DEG_PER_RAD,lon[c]*DEG_PER_RAD);
                let v=hinterp::interp_sequence(x,y,&context,&methods,0);
                if v==missing {0.} else {v}
            };
        }
        consumed+=1;
    }
    Ok(consumed)
}
