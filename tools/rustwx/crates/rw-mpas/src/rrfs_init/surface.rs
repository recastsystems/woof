//! RRFS sibling at RRFSx/MPAS-Model 1d6e370e.
//! Complete optional surface-input driver, RRFSx/MPAS-Model
//! src/core_init_atmosphere/mpas_init_atm_surface.F:104-423 at 1d6e370e.
use super::{metfile::{MetSlab,IPROJ_POLAR},projection::Projection,hinterp::{self,Slab,InterpContext,Method,Underflow},DEG_PER_RAD};
use crate::error::{MpasError,MpasResult};

/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_surface.F:104-423 at 1d6e370e. Ordered matching
/// records replace the destination; the last LANDSEA record supplies masks.
pub fn interpolate(slabs:&[MetSlab],lat:&[f32],lon:&[f32],land:&[i32],
    sst:&mut [f32],ice:&mut [f32],underflow:Underflow)->MpasResult<usize> {
    let n=land.len();
    if lat.len()!=n || lon.len()!=n || sst.len()!=n || ice.len()!=n {return Err(MpasError::Refusal("surface-input arrays have different cell counts".into()));}
    if land.iter().copied().max()==Some(0) {return Err(MpasError::Refusal("native interp_sfc_to_MPAS rejects a landmask whose global maximum is zero because surface interpolation would have no initialized land mask".into()));}
    let mask=slabs.iter().rev().find(|s|s.field=="LANDSEA").map(|s|Slab::from_met(&s.values,s.nx,s.ny));
    let mut consumed=0;
    for slab in slabs {
        let is_sst=slab.field=="SKINTEMP" || slab.field=="SST";
        if !is_sst && slab.field!="SEAICE" {continue;}
        let projection=if slab.iproj==IPROJ_POLAR {Projection::auxiliary_polar(slab)?} else {Projection::surface_met(slab)?};
        let array=Slab::from_met(&slab.values,slab.nx,slab.ny);
        let methods=if is_sst {vec![Method::FourPoint,Method::Search]} else {vec![Method::FourPoint,Method::WtAverage4,Method::Search]};
        let masked=if is_sst {-1} else {1};let maskval=masked as f32;
        let context=InterpContext {array:&array,msgval:-1e30f32,maskval:Some(maskval),mask:mask.as_ref(),underflow,underflow_hits:None};
        let destination=if is_sst {&mut *sst} else {&mut *ice};
        destination.fill(0.);
        for c in 0..n {
            if land[c]==masked {continue;}
            let (x,y)=projection.locate_surface(lat[c]*DEG_PER_RAD,lon[c]*DEG_PER_RAD);
            destination[c]=hinterp::interp_sequence(x,y,&context,&methods,0);
        }
        consumed+=1;
    }
    for c in 0..n {if sst[c]<271. && land[c]==0 {ice[c]=1.;}}
    for value in ice {*value=value.clamp(0.,1.);}
    Ok(consumed)
}

/// RRFSx/MPAS-Model src/core_atmosphere/physics/mpas_atmphys_initialize_real.F:866-917 at 1d6e370e.
/// Native seaice is scratch here and is rebuilt by physics_init_seaice.
pub fn initialize_sst(frac:bool,land:&[i32],sst:&[f32],skin:&mut [f32],ice:&mut [f32]) {
    let threshold=if frac {0.02f32} else {0.5f32};
    for c in 0..land.len() {
        if land[c]==0 && ice[c]<threshold {skin[c]=sst[c];}
        if (land[c]==1 && ice[c]>0.) || ice[c]>200. {ice[c]=0.;}
    }
}

/// Resolved clock and input streams. Numerical slabs never pass through JSON.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_surface.F:30-101 at 1d6e370e.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub grid_path:std::path::PathBuf,
    pub output_path:std::path::PathBuf,
    pub start_time:String,
    pub stop_time:String,
    pub interval_seconds:i64,
    pub intervals:Vec<SurfaceInput>,
    #[serde(deserialize_with="super::job::underflow")]
    pub interp_underflow:Underflow,
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurfaceInput {pub valid_time:String,pub met_path:std::path::PathBuf}
#[derive(serde::Serialize)]
pub struct SurfaceReceipt {
    pub schema:&'static str,
    pub cells:usize,
    pub records:usize,
    pub input_records_used:Vec<usize>,
    pub time_seconds:Vec<f32>,
}

/// Complete case-8 surface stream: inclusive clock, persistent SST/ice pools,
/// native record fields and timestamps, and stream close.
/// RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_surface.F:30-423
/// and Registry.xml:719-731,1204-1209,1503-1522 at 1d6e370e.
pub fn run(r:Request)->MpasResult<SurfaceReceipt> {
    use rw_store::netcdf_classic::{NcAttr,NcClassicWriter,NcData,NcDim,NcFormat,NcType,NcVarDef};
    let invalid=|message:&str|MpasError::Refusal(message.into());
    if r.grid_path==r.output_path {return Err(invalid("surface output must differ from its input grid"));}
    let elapsed=crate::rrfs_lbc::seconds_between(&r.start_time,&r.stop_time)?;
    if r.interval_seconds<=0 || elapsed<0 {return Err(invalid("surface clock requires a positive interval and stop at or after start"));}
    let records=(elapsed/r.interval_seconds+1) as usize;
    if r.intervals.len()!=records {return Err(invalid("surface input count does not cover the inclusive native clock"));}
    for (record,input) in r.intervals.iter().enumerate() {
        if crate::rrfs_lbc::seconds_between(&r.start_time,&input.valid_time)?!=record as i64*r.interval_seconds {
            return Err(invalid("surface input timestamps differ from the native clock sequence"));
        }
    }
    let grid=netcrust::File::open(&r.grid_path)?;
    let cells=grid.dimension("nCells").ok_or_else(||invalid("surface grid lacks nCells"))?.len();
    let lat:Vec<f32>=grid.read_array_f64("latCell")?.into_values().into_iter().map(|v|v as f32).collect();
    let lon:Vec<f32>=grid.read_array_f64("lonCell")?.into_values().into_iter().map(|v|v as f32).collect();
    let land:Vec<i32>=grid.read_array::<i32>("landmask")?.into_iter().collect();
    if lat.len()!=cells || lon.len()!=cells || land.len()!=cells {return Err(invalid("surface coordinates and landmask do not cover nCells"));}
    let dims=vec![NcDim::record("Time"),NcDim::fixed("nCells",cells),NcDim::fixed("StrLen",64)];
    let vars=vec![
        NcVarDef::new("xtime",NcType::Char,vec![0,2]).with_attrs(vec![NcAttr::text("units","YYYY-MM-DD_hh:mm:ss"),NcAttr::text("long_name","Model valid time")]),
        NcVarDef::new("Time",NcType::Float,vec![0]).with_attrs(vec![NcAttr::text("units",format!("seconds since {}",r.start_time.replace('_'," "))),NcAttr::text("long_name","CF-compliant valid time"),NcAttr::text("standard_name","time")]),
        NcVarDef::new("sst",NcType::Float,vec![0,1]).with_attrs(vec![NcAttr::text("units","K"),NcAttr::text("long_name","sea-surface temperature")]),
        NcVarDef::new("xice",NcType::Float,vec![0,1]).with_attrs(vec![NcAttr::text("units","unitless"),NcAttr::text("long_name","fractional area coverage of sea-ice")]),
    ];
    let tmp=r.output_path.with_extension(format!("nc.tmp-{}",std::process::id()));
    let result=(||->MpasResult<SurfaceReceipt>{
        let mut writer=NcClassicWriter::create_with_min_header(&tmp,NcFormat::Data64,dims,Vec::new(),vars,records as u64,131072)?;
        let mut sst=vec![0f32;cells];let mut ice=vec![0f32;cells];
        let mut receipt=SurfaceReceipt{schema:"rrfs-surface-stream/v1",cells,records,input_records_used:Vec::new(),time_seconds:Vec::new()};
        for (record,input) in r.intervals.iter().enumerate() {
            let slabs=super::metfile::read_surface_file(&input.met_path)?;
            receipt.input_records_used.push(interpolate(&slabs,&lat,&lon,&land,&mut sst,&mut ice,r.interp_underflow)?);
            let time=crate::rrfs_lbc::seconds_between(&r.start_time,&input.valid_time)? as f32;
            receipt.time_seconds.push(time);
            let mut xtime=vec![b' ';64];let bytes=input.valid_time.as_bytes();
            if bytes.len()>64 {return Err(invalid("surface timestamp exceeds native StrLen"));}
            xtime[..bytes.len()].copy_from_slice(bytes);
            writer.put_record("xtime",record as u64,NcData::Chars(&xtime))?;
            writer.put_record("Time",record as u64,NcData::Floats(&[time]))?;
            writer.put_record("sst",record as u64,NcData::Floats(&sst))?;
            writer.put_record("xice",record as u64,NcData::Floats(&ice))?;
        }
        writer.finish()?;std::fs::rename(&tmp,&r.output_path)?;Ok(receipt)
    })();
    if result.is_err() {let _=std::fs::remove_file(&tmp);}
    result
}
