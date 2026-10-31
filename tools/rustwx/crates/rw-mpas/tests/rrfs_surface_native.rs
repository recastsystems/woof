//! Complete surface module replay against the pinned fork, one serial process.
//! Source src/core_atmosphere/physics/mpas_atmphys_initialize_real.F:66-1056
//! at 1d6e370e. These component replays do not satisfy native init deck proof.
use serde::Deserialize;
use std::{fs,io::Write};
#[derive(Deserialize)]
struct Request {
    n:usize,nf:usize,nveg:usize,frac:bool,threshold:f32,year:i32,month:u32,day:u32,
    land:Vec<i32>,skin:Vec<f32>,soiltemp:Vec<f32>,terrain:Vec<f32>,soilz:Vec<f32>,
    snow:Vec<f32>,snowh:Vec<f32>,veg:Vec<f32>,ice:Vec<f32>,snoalb:Vec<f32>,
    shdmin:Vec<f32>,shdmax:Vec<f32>,lu:Vec<i32>,soilcat:Vec<i32>,
    st:Vec<Vec<f32>>,sm:Vec<Vec<f32>>,zs_fg:Vec<f32>,dz_fg:Vec<f32>,
    albedo:Vec<Vec<f32>>,green:Vec<Vec<f32>>,
}
#[test]
#[ignore="requires lane-owned request/output paths; invoked by native replay orchestrator"]
fn surface_probe() {
    use rw_mpas::{init::soil as shared,rrfs_init::soil};
    let mut r:Request=serde_json::from_slice(&fs::read(std::env::var("RRFS_REQUEST").unwrap()).unwrap()).unwrap();
    assert!(r.st.iter().all(|c|c.len()==r.nf));
    let nc=r.land.len();
    let mut s=shared::init_surface_companions(&r.land,&r.snow,&r.albedo,&r.green,r.year,r.month,r.day);
    if r.nveg!=1 && r.veg.iter().any(|&x|x!=0.) {s.vegetation=r.veg;s.minimum=r.shdmin;s.maximum=r.shdmax;}
    if r.snowh.iter().any(|&x|x!=0.) {s.snow_height=r.snowh;}
    let mut tmn=soil::adjust_input_soiltemps(&r.land,&r.terrain,&r.soiltemp,&r.soilz,&mut r.skin,&mut r.st);
    let(mut ts,mut sm,mut liq,mut rel,_)=soil::init_soil_layers_properties(&r.land,&r.skin,&tmn,&r.st,&r.sm,&r.dz_fg,&r.zs_fg,r.n,soil::DeepMoisture::ReproduceFortran).unwrap();
    let mut sea=vec![0.;nc];soil::physics_init_seaice(r.frac,r.threshold,soil::SeaiceCleanup::KeepInterpolated,15,&r.land,&r.skin,&mut r.ice,&mut sea,&mut r.snow,&mut s.snow_cover,&mut s.snow_height,&mut s.vegetation,&mut r.snoalb,&mut r.lu,&mut r.soilcat,&mut s.land,&mut tmn,&mut ts,&mut sm,&mut liq,&mut rel);
    let mut out=fs::File::create(std::env::var("RRFS_OUTPUT").unwrap()).unwrap();
    for v in [&r.skin,&tmn,&s.albedo,&s.vegetation,&s.minimum,&s.maximum,&r.snow,&s.snow_cover,&s.snow_height,&r.ice,&sea,&s.land,&r.snoalb] {for x in v {out.write_all(&x.to_le_bytes()).unwrap();}}
    for v in [&r.lu,&r.soilcat] {for x in v {out.write_all(&x.to_le_bytes()).unwrap();}}
    for v in [&r.st,&ts,&sm,&liq,&rel] {for col in v {for x in col {out.write_all(&x.to_le_bytes()).unwrap();}}}
    let(z,dz)=soil::layer_geometry(r.n).unwrap();for v in [&z,&dz] {for _ in 0..nc {for x in v {out.write_all(&x.to_le_bytes()).unwrap();}}}
}
