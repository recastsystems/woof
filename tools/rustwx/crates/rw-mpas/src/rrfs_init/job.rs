//! Internal job ABI for table-selected initialization producers.
//!
//! The resolver chooses the executable by source pin. Stage names describe
//! the operation; no model or suite name selects arithmetic here. All paths,
//! numerical switches and metadata are supplied explicitly in the job.
use serde::{Deserialize, Deserializer};
use std::io::{Read, Write};

fn token<'de, D: Deserializer<'de>>(d: D) -> Result<String,D::Error> {
    String::deserialize(d)
}
pub(crate) fn extrap<'de,D:Deserializer<'de>>(d:D)->Result<super::vinterp::Extrap,D::Error> {
    super::vinterp::Extrap::parse(&token(d)?).map_err(serde::de::Error::custom)
}
pub(crate) fn underflow<'de,D:Deserializer<'de>>(d:D)->Result<super::hinterp::Underflow,D::Error> {
    super::hinterp::Underflow::parse(&token(d)?).map_err(serde::de::Error::custom)
}
pub(crate) fn absent_numbers_default()->super::scalars::AbsentNumbers {super::scalars::AbsentNumbers::FromMass}
pub(crate) fn absent_numbers<'de,D:Deserializer<'de>>(d:D)->Result<super::scalars::AbsentNumbers,D::Error> {
    super::scalars::AbsentNumbers::parse(&token(d)?).map_err(serde::de::Error::custom)
}
pub(crate) fn virtual_factor<'de,D:Deserializer<'de>>(d:D)->Result<super::dynamics::VirtualFactor,D::Error> {
    use super::dynamics::VirtualFactor;
    match token(d)?.as_str() {
        "reproduce-fortran"=>Ok(VirtualFactor::ReproduceFortran),
        "consistent"=>Ok(VirtualFactor::Consistent),
        x=>Err(serde::de::Error::custom(format!("unknown virtual_factor {x}"))),
    }
}
pub(crate) fn deep_moisture<'de,D:Deserializer<'de>>(d:D)->Result<super::soil::DeepMoisture,D::Error> {
    use super::soil::DeepMoisture;
    match token(d)?.as_str() {
        "reproduce-fortran"=>Ok(DeepMoisture::ReproduceFortran),
        "corrected"=>Ok(DeepMoisture::Corrected),
        x=>Err(serde::de::Error::custom(format!("unknown deep_moisture {x}"))),
    }
}
pub(crate) fn coordinates<'de,D:Deserializer<'de>>(d:D)->Result<crate::staticfile::coordframe::CoordinateRepresentation,D::Error> {
    let s=token(d)?;
    crate::staticfile::coordframe::CoordinateRepresentation::from_tag(&s)
        .ok_or_else(||serde::de::Error::custom(format!("unknown coordinates {s}")))
}
pub(crate) fn lineage<'de,D:Deserializer<'de>>(d:D)->Result<super::Lineage,D::Error> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Fields {model_name:String,core_name:String,version:String,git_version:String}
    let f=Fields::deserialize(d)?;
    Ok(super::Lineage {model_name:f.model_name,core_name:f.core_name,version:f.version,git_version:f.git_version})
}

pub const ABI_MARKER: &str="rw_mpas_init_v831 stdin: hex-init-job/v1 JSON; stage=case7|case9|static|terrain|vertical|artifact|prepare|gwd|surface";

#[derive(Deserialize)]
#[serde(tag="stage",content="config",rename_all="lowercase",deny_unknown_fields)]
enum Job {
    Case7(super::InitConfig),
    Case9(crate::rrfs_lbc::LbcConfig),
    Static(crate::rrfs_static::StaticBuildConfig),
    Terrain(crate::rrfs_static::StaticBuildConfig),
    Vertical(serde_json::Value),
    Artifact(super::artifact::Request),
    Prepare(super::prepare::Request),
    Gwd(crate::rrfs_static_gwd::Request),
    Surface(super::surface::Request),
}
/// Receipt JSON for file producers; native little-endian arrays for vertical.
pub fn run(reader:impl Read,mut writer:impl Write)->Result<(),String> {
    let job:Job=serde_json::from_reader(reader).map_err(|e|e.to_string())?;
    let receipt=match job {
        Job::Case7(cfg)=>serde_json::to_value(super::build_init(&cfg).map_err(|e|e.to_string())?),
        Job::Case9(cfg)=>serde_json::to_value(crate::rrfs_lbc::build_lbc(&cfg).map_err(|e|e.to_string())?),
        Job::Static(cfg)=>serde_json::to_value(crate::rrfs_static::build_static(&cfg).map_err(|e|e.to_string())?),
        Job::Terrain(cfg)=>serde_json::to_value(crate::rrfs_static::build_terrain_reporting(&cfg,&|_|{}).map_err(|e|e.to_string())?),
        Job::Artifact(cfg)=>serde_json::to_value(super::artifact::materialize(cfg).map_err(|e|e.to_string())?),
        Job::Gwd(cfg)=>serde_json::to_value(crate::rrfs_static_gwd::run(cfg).map_err(|e|e.to_string())?),
        Job::Surface(cfg)=>serde_json::to_value(super::surface::run(cfg).map_err(|e|e.to_string())?),
        Job::Prepare(cfg)=>serde_json::to_value(super::prepare::run(cfg)?),
        Job::Vertical(cfg)=>return super::vertical::run(std::io::Cursor::new(serde_json::to_vec(&cfg).map_err(|e|e.to_string())?),writer),
    }.map_err(|e|e.to_string())?;
    serde_json::to_writer_pretty(&mut writer,&receipt).map_err(|e|e.to_string())?;
    writer.write_all(b"\n").map_err(|e|e.to_string())
}
