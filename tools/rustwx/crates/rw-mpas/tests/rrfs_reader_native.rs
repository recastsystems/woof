//! Whole pristine fork read_met.F:9-445 at 1d6e370e; every defined metadata/slab word.
use std::{fs,io::Write,path::Path};
#[test]
#[ignore="requires native reader replay inputs"]
fn reader_probe() {
    let mut out=fs::File::create(std::env::var("RRFS_OUTPUT").unwrap()).unwrap();
    let met=std::env::var("RRFS_MET").unwrap();
    fn array(w:&mut impl Write,v:&[f32]) {for x in v {w.write_all(&x.to_le_bytes()).unwrap();}}
    for f in rw_mpas::rrfs_init::metfile::read_met_file(Path::new(&met)).unwrap() {
        let native_proj=match f.iproj {1=>3,3=>1,5=>2,n=>n};
        for x in [f.format_version,native_proj,f.nx as i32,f.ny as i32,f.wind_grid_rel as i32] {out.write_all(&x.to_le_bytes()).unwrap();}
        for (s,n) in [(&f.hdate,24),(&f.map_source,32),(&f.field,9),(&f.units,25),(&f.desc,46)] {let mut b=s.as_bytes().to_vec();b.resize(n,b' ');out.write_all(&b).unwrap();}
        array(&mut out,&[f.xfcst,f.xlvl,f.start_i,f.start_j,f.start_lat,f.start_lon,f.earth_radius_km]);
        if f.iproj==0 || f.iproj==4 {array(&mut out,&[f.delta_lat,f.delta_lon]);}
        else if f.iproj==6 {array(&mut out,&[f.dx,f.dy,f.delta_lat,f.delta_lon,f.center_lat,f.center_lon,f.pole_lat,f.pole_lon,f.stdlon]);}
        else {array(&mut out,&[f.dx,f.dy,f.truelat1]);if f.iproj!=1 {array(&mut out,&[f.stdlon]);}if f.iproj==3 {array(&mut out,&[f.truelat2]);}}
        array(&mut out,&f.values);
    }
}
