use crate::Result;
use rayon::prelude::*;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,io::Read,path::Path};
pub fn sha(path:&Path)->Result<String>{
    let mut h=Sha256::new();let mut f=fs::File::open(path)?;let mut b=vec![0;1024*1024];
    loop{let n=f.read(&mut b)?;if n==0{break}h.update(&b[..n]);}Ok(format!("{:x}",h.finalize()))
}
pub fn f32(out:&Path,name:&str,values:&[f32],shape:&[usize])->Result<Value>{
    if shape.iter().product::<usize>()!=values.len(){return Err(format!("shape mismatch {name}").into());}
    let mut b=vec![0;values.len()*4];b.par_chunks_mut(4).zip(values.par_iter()).for_each(|(b,v)|b.copy_from_slice(&v.to_le_bytes()));
    let file=format!("{name}.bin");fs::write(out.join(&file),&b)?;
    Ok(json!({"path":file,"dtype":"<f4","shape":shape,"sha256":format!("{:x}",Sha256::digest(&b)),"bytes":b.len()}))
}
pub fn i32(out:&Path,name:&str,values:&[i32])->Result<Value>{
    let mut b=vec![0;values.len()*4];b.par_chunks_mut(4).zip(values.par_iter()).for_each(|(b,v)|b.copy_from_slice(&v.to_le_bytes()));
    let file=format!("{name}.bin");fs::write(out.join(&file),&b)?;
    Ok(json!({"path":file,"dtype":"<i4","shape":[values.len()],"sha256":format!("{:x}",Sha256::digest(&b)),"bytes":b.len()}))
}
pub fn read_nc(path:&Path,name:&str)->Result<Vec<f64>>{Ok(netcrust::open(path).map_err(|e|e.to_string())?.read_array_f64(name).map_err(|e|e.to_string())?.values().to_vec())}
pub fn bracket(axis:&[f64],v:f64)->Option<(usize,f64)>{
    if axis.len()<2||!v.is_finite(){return None}let asc=axis[1]>axis[0];
    if v<axis[0].min(axis[axis.len()-1])||v>axis[0].max(axis[axis.len()-1]){return None}
    let(mut lo,mut hi)=(0,axis.len()-1);while hi-lo>1{let m=(hi+lo)/2;if (axis[m]<=v)==asc{lo=m}else{hi=m}}
    Some((lo,(v-axis[lo])/(axis[lo+1]-axis[lo])))
}
