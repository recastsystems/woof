
//! Native column ice-path receipt, using the hybrid mass coefficients in history.
//! Appends IWP to a receipt copy; the model history file is not edited.
use std::{ffi::CString,os::raw::{c_char,c_int},path::Path};
#[link(name="netcdf")]
unsafe extern "C" {
 fn nc_open(p:*const c_char,mode:c_int,id:*mut c_int)->c_int;
 fn nc_close(id:c_int)->c_int;
 fn nc_redef(id:c_int)->c_int;
 fn nc_enddef(id:c_int)->c_int;
 fn nc_inq_dimid(id:c_int,name:*const c_char,dim:*mut c_int)->c_int;
 fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
 fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
 fn nc_inq_varndims(id:c_int,var:c_int,n:*mut c_int)->c_int;
 fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
 fn nc_get_var_float(id:c_int,var:c_int,data:*mut f32)->c_int;
 fn nc_def_var(id:c_int,name:*const c_char,kind:c_int,ndim:c_int,dims:*const c_int,var:*mut c_int)->c_int;
 fn nc_put_att_text(id:c_int,var:c_int,name:*const c_char,len:usize,text:*const c_char)->c_int;
 fn nc_put_vara_float(id:c_int,var:c_int,start:*const usize,count:*const usize,data:*const f32)->c_int;
}
fn c(s:&str)->CString {CString::new(s).unwrap()}
fn check(rc:c_int){assert_eq!(rc,0,"netCDF status {rc}");}
unsafe fn read(id:c_int,name:&str)->Option<Vec<f32>>{
 let mut var=0;if unsafe{nc_inq_varid(id,c(name).as_ptr(),&mut var)}!=0{return None;}
 let mut n=0;check(unsafe{nc_inq_varndims(id,var,&mut n)});let mut dims=vec![0;n as usize];check(unsafe{nc_inq_vardimid(id,var,dims.as_mut_ptr())});
 let mut count=1;for dim in dims {let mut len=0;check(unsafe{nc_inq_dimlen(id,dim,&mut len)});count*=len;}
 assert!(count<=100_000_000);let mut data=vec![0f32;count];check(unsafe{nc_get_var_float(id,var,data.as_mut_ptr())});Some(data)
}
fn integral(ice:&[&[f32]],mu:&[f32],mub:&[f32],c1:&[f32],c2:&[f32],eta:&[f32],nz:usize,cells:usize)->Vec<f32>{
 assert_eq!(eta.len(),nz+1);assert_eq!(c1.len(),nz);assert_eq!(c2.len(),nz);
 let mut out=vec![0.0f64;cells];
 for k in 0..nz {for j in 0..cells {
  let mass=(f64::from(c1[k])*(f64::from(mu[j])+f64::from(mub[j]))+f64::from(c2[k]))*(f64::from(eta[k])-f64::from(eta[k+1]))/9.81;
  let mut q=0.0;for field in ice {let v=field[k*cells+j];if !v.is_finite(){q=f64::NAN;break;}q+=f64::from(v.max(0.0));}
  out[j]+=q*mass*1000.0;
 }}out.into_iter().map(|v|v as f32).collect()
}
fn profile(data:&[f32],t:usize,n:usize)->&[f32]{if data.len()==n{data}else{&data[t*n..(t+1)*n]}}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),3,"history.nc receipt.nc");assert_ne!(Path::new(&a[1]),Path::new(&a[2]));std::fs::copy(&a[1],&a[2]).unwrap();
 unsafe {
  let mut id=0;check(nc_open(c(&a[2]).as_ptr(),1,&mut id));let mut dims=[0;4];let mut count=[0;4];
  for (i,name) in ["Time","bottom_top","south_north","west_east"].iter().enumerate(){check(nc_inq_dimid(id,c(name).as_ptr(),&mut dims[i]));check(nc_inq_dimlen(id,dims[i],&mut count[i]));}
  let [nt,nz,ny,nx]=count;let cells=ny*nx;
  let mu=read(id,"MU").unwrap();let mub=read(id,"MUB").unwrap();let c1=read(id,"C1H").expect("history must retain hybrid dry-mass coefficients");let c2=read(id,"C2H").unwrap();let eta=read(id,"ZNW").unwrap();
  let mut ice=Vec::new();for name in ["QICE","QICE2","QICE3","QSNOW","QGRAUP","QHAIL"] {if let Some(q)=read(id,name){assert_eq!(q.len(),nt*nz*cells);ice.push(q);}}
  assert!(ice.len()>=3,"native history must contain all three ice categories");
  let mut result=Vec::new();for t in 0..nt {let fields:Vec<&[f32]>=ice.iter().map(|q|&q[t*nz*cells..(t+1)*nz*cells]).collect();result.extend(integral(&fields,profile(&mu,t,cells),profile(&mub,t,cells),profile(&c1,t,nz),profile(&c2,t,nz),profile(&eta,t,nz+1),nz,cells));}
  check(nc_redef(id));let mut var=0;check(nc_def_var(id,c("IWP").as_ptr(),5,3,[dims[0],dims[2],dims[3]].as_ptr(),&mut var));
  for (name,text) in [("units","g m-2"),("description","Total ice water path from all native ice masses and hybrid dry-air layer mass")] {check(nc_put_att_text(id,var,c(name).as_ptr(),text.len(),c(text).as_ptr()));}
  check(nc_enddef(id));check(nc_put_vara_float(id,var,[0;3].as_ptr(),[nt,ny,nx].as_ptr(),result.as_ptr()));check(nc_close(id));
  let finite=result.iter().filter(|v|v.is_finite()).count();let max=result.iter().copied().filter(|v|v.is_finite()).fold(0f32,f32::max);
  println!("{{\"frames\":{nt},\"columns\":{cells},\"nz\":{nz},\"finite\":{finite},\"words\":{},\"max_g_m2\":{max},\"ice_categories\":{}}}",result.len(),ice.len());
 }
}
#[cfg(test)] mod tests {use super::integral;
 #[test] fn all_categories_and_hybrid_weights_contribute(){
  let q1=[0.001,0.002];let q2=[0.002,0.001];let q3=[0.003,0.001];
  let got=integral(&[&q1,&q2,&q3],&[981.0],&[0.0],&[1.0,0.5],&[0.0,98.1],&[1.0,0.6,0.0],2,1);
  assert!((got[0]-384.0).abs()<0.001,"{}",got[0]);
 }
}
