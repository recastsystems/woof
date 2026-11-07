//! WRF P3 diagnostic planes. Field decode and arithmetic are Rust.
use std::{env,fs,path::Path};
use rayon::prelude::*;
fn read(f:&netcrust::File,n:&str)->Result<(Vec<f64>,Vec<usize>),Box<dyn std::error::Error>> {
 let a=f.read_array_f64(n)?;let s=a.shape().to_vec();Ok((a.into_values(),s))
}
fn planes(a:&[Vec<f64>;5], nk:usize,n:usize)->Result<[Vec<f32>;4],String>{
 if nk==0 || n==0 || a.iter().any(|v|v.len()!=nk*n || v.iter().any(|x|!x.is_finite())) {return Err("nonfinite or wrongly shaped WRF P3 field".into());}
 let columns:Vec<[f32;4]>=(0..n).into_par_iter().map(|i| {
  let mut qi=0.;let mut qr=0.;let mut p:[f32;4]=[-99.,0.,0.,0.];
  for k in 0..nk {let j=k*n+i;p[0]=p[0].max(a[0][j] as f32);p[1]=p[1].max(a[1][j] as f32);p[2]=p[2].max(a[2][j] as f32);qi+=a[1][j]+a[2][j];qr+=a[3][j]+a[4][j];}
  p[3]=if qi>0. { (qr/qi) as f32 } else {0.};p
 }).collect();
 let mut o:[Vec<f32>;4]=std::array::from_fn(|_|vec![0.;n]);
 o.par_iter_mut().enumerate().for_each(|(f,values)| {
  values.par_iter_mut().enumerate().for_each(|(i,v)|*v=columns[i][f]);
 });
 Ok(o)
}
fn main()->Result<(),Box<dyn std::error::Error>> {
 let args:Vec<_>=env::args().collect();if args.len()!=3 {return Err("usage: rw_p3_planes WRFOUT OUTPUT_DIR".into());}
 let f=netcrust::File::open(&args[1])?;
 let (_,s)=read(&f,"QICE")?;if s.len()!=4 {return Err("QICE must be Time,k,j,i".into());}
 let n=s[2]*s[3];let nk=s[1];let t=s[0]-1;
 let mut a:[Vec<f64>;5]=std::array::from_fn(|_|vec![]);
 for (i,name) in ["REFL_10CM","QICE","QICE2","QIR","QIR2"].iter().enumerate(){let(v,shape)=read(&f,name)?;if shape!=s{return Err(format!("{name} shape differs").into());}a[i]=v[t*nk*n..(t+1)*nk*n].to_vec();}
 let o=planes(&a,nk,n)?;let out=Path::new(&args[2]);fs::create_dir_all(out)?;
 let names=["P3_REFL_MAX","P3_ICE1_MAX","P3_ICE2_MAX","P3_RIME_FRACTION"];
 for (i,name) in names.iter().enumerate(){let b:Vec<u8>=o[i].iter().flat_map(|x|x.to_le_bytes()).collect();fs::write(out.join(format!("{name}.bin")),b)?;}
 fs::write(out.join("planes.json"),serde_json::to_vec_pretty(&serde_json::json!({"time_index":t,"shape":[s[2],s[3]],"names":names,"definitions":["column maximum reflectivity","column maximum category-1 mixing ratio","column maximum category-2 mixing ratio","vertical sum of both rime masses / vertical sum of both ice masses"],"units":["dBZ","kg kg-1","kg kg-1","1"]}))?)?;Ok(())
}
#[cfg(test)] mod tests {use super::*;#[test] fn distinct_categories_and_rime(){let a=[vec![1.,3.],vec![2.,4.],vec![1.,2.],vec![1.,2.],vec![0.5,1.]];let p=planes(&a,2,1).unwrap();assert_eq!(p[0],[3.]);assert_eq!(p[1],[4.]);assert_eq!(p[2],[2.]);assert_eq!(p[3],[0.5]);}#[test] fn refuses_missing_or_nonfinite(){let a=std::array::from_fn(|_|vec![f64::NAN]);assert!(planes(&a,1,1).is_err());}}
