//! RRFSx/MPAS-Model src/core_init_atmosphere/mpas_init_atm_llxy.F:191-2224;
//! src/core_init_atmosphere/mpas_init_atm_cases.F:11137-11201 at 1d6e370e.
use std::{fs,io::Write,path::Path};
#[test]
#[ignore="requires pristine projection oracle and fixture"]
fn projection_probe() {
 let met=std::env::var("RRFS_MET").unwrap();let text=fs::read_to_string(std::env::var("RRFS_POINTS").unwrap()).unwrap();let mut tokens=text.split_whitespace();let mode:usize=tokens.next().unwrap().parse().unwrap();let n:usize=tokens.next().unwrap().parse().unwrap();
 let f=rw_mpas::rrfs_init::metfile::read_met_file(Path::new(&met)).unwrap().remove(0);
 use rw_mpas::rrfs_init::projection::{Projection,rotate_velocities};
 let proj=if f.iproj==5 {Projection::auxiliary_polar(&f)}else if mode==1 {Projection::surface_met(&f)}else {Projection::from_met(&f)}.unwrap();
 let mut out=fs::File::create(std::env::var("RRFS_OUTPUT").unwrap()).unwrap();
 fn put(out:&mut impl Write,values:&[f32]) {for v in values {out.write_all(&v.to_le_bytes()).unwrap();}}
 let mut lat=Vec::new();let mut lon=Vec::new();let mut u=Vec::new();let mut v=Vec::new();
 for _ in 0..n {let vals:Vec<f32>=(0..8).map(|_|tokens.next().unwrap().parse().unwrap()).collect();lat.push(vals[0]);lon.push(vals[1]);u.push(vals[2..5].to_vec());v.push(vals[5..8].to_vec());let (x,y)=proj.ij(vals[0]*(180f32/std::f32::consts::PI),vals[1]*(180f32/std::f32::consts::PI));put(&mut out,&[x,y]);if f.iproj==6 {let(a,b)=proj.inverse_cassini(x,y+0.1);put(&mut out,&[a,b]);let(a,b)=proj.inverse_cassini(x,y-0.1);put(&mut out,&[a,b]);}}
 // The cases driver rebuilds its Cassini projection with nx-based knowni.
 // The surface projection contributes coordinates above but never rotates wind.
 rotate_velocities(&f,&lat,&lon,&mut u,&mut v).unwrap();
 for row in u {put(&mut out,&row);}for row in v {put(&mut out,&row);}
}
