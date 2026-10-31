//! Synthetic input fixture for cires_tauamf_data.F:82-110.
//! Rust owns NetCDF rewrite; the published strict executable is unchanged.
use std::{env, ffi::{CString,c_char,c_int},fs};
#[link(name="netcdf")] unsafe extern "C" {
 fn nc_open(p:*const c_char,mode:c_int,id:*mut c_int)->c_int;
 fn nc_close(id:c_int)->c_int;
 fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
 fn nc_inq_varndims(id:c_int,var:c_int,ndim:*mut c_int)->c_int;
 fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
 fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
 fn nc_get_var_float(id:c_int,var:c_int,out:*mut f32)->c_int;
 fn nc_put_var_float(id:c_int,var:c_int,data:*const f32)->c_int;
}
fn check(code:i32){assert_eq!(code,0,"NetCDF error {code}");}
fn main(){
 let a:Vec<String>=env::args().collect();assert_eq!(a.len(),4);
 assert!(matches!(a[3].as_str(),"before-first"|"after-last"));
 assert!(!std::path::Path::new(&a[2]).exists(),"owned output must be fresh");
 fs::copy(&a[1],&a[2]).unwrap();
 unsafe {
  let mut id=0;check(nc_open(CString::new(a[2].as_str()).unwrap().as_ptr(),1,&mut id));
  let mut var=0;check(nc_inq_varid(id,c"DAYS".as_ptr(),&mut var));
  let mut rank=0;check(nc_inq_varndims(id,var,&mut rank));assert_eq!(rank,1);
  let mut dim=0;check(nc_inq_vardimid(id,var,&mut dim));
  let mut n=0;check(nc_inq_dimlen(id,dim,&mut n));let mut values=vec![0f32;n];
  check(nc_get_var_float(id,var,values.as_mut_ptr()));
  let shift=if a[3]=="before-first" {1000f32}else{-1000f32};
  for v in &mut values{*v+=shift;}
  check(nc_put_var_float(id,var,values.as_ptr()));check(nc_close(id));
  println!("{{\"field\":\"DAYS\",\"words\":{n},\"shift\":{shift},\"mode\":\"{}\",\"first\":{},\"last\":{}}}",a[3],values[0],values[n-1]);
 }
}
