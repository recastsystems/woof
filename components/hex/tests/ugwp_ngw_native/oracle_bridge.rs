//! Test transport: exact ORCL0001 words and NetCDF float32 fields, decoded in Rust.
//! ORCL0001 validation follows f1's tools/native_physics_reference/oracle/oracle_words.rs.
use std::{ffi::{CStr, c_char, c_int}, fs, slice, ptr};
pub struct Field { words: Vec<u32>, shape: Vec<usize>, ids: Vec<i32>, code: i32 }
thread_local! { static ERROR: std::cell::RefCell<std::ffi::CString> = std::cell::RefCell::new(std::ffi::CString::new("").unwrap()); }
fn fail(e: String) -> *mut Field {
    ERROR.with(|v| *v.borrow_mut() = std::ffi::CString::new(e.replace('\0', "?")).unwrap());
    ptr::null_mut()
}
#[no_mangle] pub extern "C" fn bridge_error() -> *const c_char { ERROR.with(|v| v.borrow().as_ptr()) }
// Test-only ORCL output transport. The template must be PRE: never read native
// POST values to manufacture a candidate. Retained module state is copied from
// its input; changed arrays/scalars are supplied by the CUDA boundary.
#[no_mangle] pub unsafe extern "C" fn bridge_write_record(input:*const c_char, output:*const c_char, data:*const u32, count:usize)->i32 {
    let result=(||{
        let mut raw=fs::read(name(input)?).map_err(|e|e.to_string())?;
        if raw.len()<860 || &raw[..8]!=b"ORCL0001" || String::from_utf8_lossy(&raw[348..604]).trim_end()!="pre" {
            return Err("output transport requires a PRE oracle template".into())
        }
        if !data.is_null() {
            let width=i32::from_le_bytes(raw[12..16].try_into().unwrap());
            let nids=i32::from_le_bytes(raw[72..76].try_into().unwrap());
            if width!=4 || nids<0 {return Err("replacement requires four-byte oracle words".into())}
            let start=860+nids as usize*4;
            if raw.len()!=start+count*4 {return Err("replacement oracle word count mismatch".into())}
            for (dst,value) in raw[start..].chunks_exact_mut(4).zip(slice::from_raw_parts(data,count)) {
                dst.copy_from_slice(&value.to_le_bytes());
            }
        } else if count!=0 {return Err("null replacement with a nonzero count".into())}
        raw[348..604].fill(b' ');raw[348..352].copy_from_slice(b"post");
        fs::write(name(output)?,raw).map_err(|e|e.to_string())
    })();match result{Ok(())=>0,Err(e)=>{fail(e);1}}
}
// Produce a test-only parent replay with the embedded NGW component disabled.
// The native parent/source is unchanged; all other captured arguments/state stay exact.
#[no_mangle] pub unsafe extern "C" fn bridge_disable_ngw(path:*const c_char, output:*const c_char)->i32 {
    let result=(||{
        let mut raw=fs::read(name(path)?).map_err(|e|e.to_string())?;
        if raw.len()!=864 || &raw[..8]!=b"ORCL0001" || raw[8..12]!=4i32.to_le_bytes()
            || raw[12..16]!=4i32.to_le_bytes() || raw[16..20]!=0i32.to_le_bytes()
            || String::from_utf8_lossy(&raw[604..860]).trim_end()!="ngw_scheme" {
            return Err("expected scalar logical NGW selector record".into())
        }
        raw[860..864].copy_from_slice(&0i32.to_le_bytes());
        fs::write(name(output)?,raw).map_err(|e|e.to_string())
    })();match result{Ok(())=>0,Err(e)=>{fail(e);1}}
}
fn selected(field: Field, axis: Option<usize>, wanted: &[i32]) -> Result<Field,String> {
    let Some(axis) = axis else { return Ok(field) };
    if wanted.is_empty() {return Ok(field)}
    let positions: Vec<usize> = wanted.iter().map(|id| field.ids.iter().position(|x| x==id)
        .ok_or_else(||format!("global cell ID {id} missing"))).collect::<Result<_,_>>()?;
    let stride: usize = field.shape[..axis].iter().product();
    let extent = field.shape[axis];
    let outer: usize = field.shape[axis+1..].iter().product();
    let mut words=Vec::with_capacity(stride*wanted.len()*outer);
    for o in 0..outer {for &p in &positions {words.extend_from_slice(&field.words[(o*extent+p)*stride..(o*extent+p+1)*stride]);}}
    let mut shape=field.shape; shape[axis]=wanted.len();
    Ok(Field{words,shape,ids:wanted.to_vec(),code:field.code})
}
unsafe fn name<'a>(s:*const c_char)->Result<&'a str,String>{CStr::from_ptr(s).to_str().map_err(|e|e.to_string())}
unsafe fn wanted<'a>(s:*const i32,n:usize)-> &'a [i32] {if n==0 {&[]} else {slice::from_raw_parts(s,n)}}
#[no_mangle] pub unsafe extern "C" fn bridge_oracle(path:*const c_char, ids:*const i32, n:usize)->*mut Field {
    let result=(|| {
        let raw=fs::read(name(path)?).map_err(|e|e.to_string())?;
        const FIXED:usize=860;
        if raw.len()<FIXED || &raw[..8]!=b"ORCL0001" {return Err("invalid oracle header".into())}
        let mut h=[0i32;21]; for (i,v) in h.iter_mut().enumerate(){*v=i32::from_le_bytes(raw[8+4*i..12+4*i].try_into().unwrap())}
        if ![1,3,4].contains(&h[0]) || h[1]!=4 || !(0..=5).contains(&h[2]) || h[3..8].iter().any(|x|*x<0)
            || h[16]<0 || h[17]<0 || h[17]>h[2] || h[19]!=256 || h[20]!=0 {return Err("invalid oracle type/shape/version".into())}
        let rank=h[2] as usize; let shape:Vec<usize>=h[3..3+rank].iter().map(|x|*x as usize).collect();
        if h[3+rank..8].iter().any(|x|*x!=1) {return Err("non-unit inactive dimensions".into())}
        let offset=FIXED+(h[16] as usize)*4;
        let count=shape.iter().try_fold(1usize, |a,b|a.checked_mul(*b)).ok_or("shape overflow")?;
        if raw.len()!=offset+count*4 {return Err("truncated/trailing oracle payload".into())}
        let cellids:Vec<i32>=raw[FIXED..offset].chunks_exact(4).map(|v|i32::from_le_bytes(v.try_into().unwrap())).collect();
        let axis=if h[17]>0 {Some(h[17] as usize-1)} else {None};
        if cellids.len()!=axis.map_or(0,|a|shape[a]) {return Err("oracle ID extent mismatch".into())}
        let words=raw[offset..].chunks_exact(4).map(|v|u32::from_le_bytes(v.try_into().unwrap())).collect();
        selected(Field{words,shape,ids:cellids,code:h[0]},axis,wanted(ids,n))
    })();
    match result {Ok(v)=>Box::into_raw(Box::new(v)), Err(e)=>fail(e)}
}
#[link(name="netcdf")] unsafe extern "C" {
    fn nc_open(path:*const c_char, mode:c_int, id:*mut c_int)->c_int;
    fn nc_close(id:c_int)->c_int;
    fn nc_inq_varid(id:c_int,name:*const c_char,var:*mut c_int)->c_int;
    fn nc_inq_varndims(id:c_int,var:c_int,ndim:*mut c_int)->c_int;
    fn nc_inq_vardimid(id:c_int,var:c_int,dims:*mut c_int)->c_int;
    fn nc_inq_dimlen(id:c_int,dim:c_int,len:*mut usize)->c_int;
    fn nc_get_var_float(id:c_int,var:c_int,out:*mut f32)->c_int;
    fn nc_strerror(code:c_int)->*const c_char;
}
unsafe fn check(code:c_int)->Result<(),String>{if code==0 {Ok(())}else{Err(CStr::from_ptr(nc_strerror(code)).to_string_lossy().into())}}
struct Nc(c_int); impl Drop for Nc {fn drop(&mut self){unsafe{nc_close(self.0);}}}
#[no_mangle] pub unsafe extern "C" fn bridge_netcdf(path:*const c_char, variable:*const c_char, ids:*const i32,n:usize)->*mut Field {
    let result=(||{
        let mut id=0;check(nc_open(path,0,&mut id))?;let file=Nc(id);
        let mut var=0;check(nc_inq_varid(file.0,variable,&mut var))?;
        let mut rank=0;check(nc_inq_varndims(file.0,var,&mut rank))?;
        if !(0..=5).contains(&rank){return Err("unsupported NetCDF rank".into())}
        let mut dims=vec![0;rank as usize];check(nc_inq_vardimid(file.0,var,dims.as_mut_ptr()))?;
        let mut shape=vec![0;rank as usize];for (dim,len) in dims.iter().zip(shape.iter_mut()){check(nc_inq_dimlen(file.0,*dim,len))?;}
        let count=shape.iter().try_fold(1usize,|a,b|a.checked_mul(*b)).ok_or("shape overflow")?;
        let mut values=vec![0f32;count];check(nc_get_var_float(file.0,var,values.as_mut_ptr()))?;
        // NetCDF C order reversed into the same Fortran-order metadata as ORCL0001.
        shape.reverse();
        let axis=if n>0 {if rank!=1{return Err("NetCDF cell selection requires a one-dimensional mesh field".into())}Some(0)}else{None};
        let cellids=if n>0 {(1..=count).map(|v|v as i32).collect()}else{vec![]};
        selected(Field{words:values.into_iter().map(f32::to_bits).collect(),shape,ids:cellids,code:1},axis,wanted(ids,n))
    })();match result{Ok(v)=>Box::into_raw(Box::new(v)),Err(e)=>fail(e)}
}
#[no_mangle] pub unsafe extern "C" fn bridge_data(v:*const Field)->*const u32{(*v).words.as_ptr()}
#[no_mangle] pub unsafe extern "C" fn bridge_levels(v:*const Field, levels:usize)->*mut Field{
    let f=&*v;
    if f.shape.len()!=3 || f.shape[2]!=1 || levels==0 || levels>f.shape[1]{return fail("native (i,k,1) level range mismatch".into())}
    Box::into_raw(Box::new(Field{words:f.words[..f.shape[0]*levels].to_vec(),shape:vec![f.shape[0],levels,1],ids:f.ids.clone(),code:f.code}))
}
#[no_mangle] pub unsafe extern "C" fn bridge_sample(v:*const Field, maximum:usize)->*mut Field{
    let valid:Vec<i32>=(&*v).ids.iter().copied().filter(|x|*x>0).collect();
    if valid.is_empty() || maximum==0{return fail("no valid cells or zero sample size".into())}
    let n=maximum.min(valid.len());
    let ids:Vec<i32>=(0..n).map(|i|valid[i*valid.len()/n]).collect();
    Box::into_raw(Box::new(Field{words:ids.iter().map(|x|*x as u32).collect(),shape:vec![n],ids,code:3}))
}
// f1's optional diagnostic arrays have axis=0 metadata but native (i,k,j) layout.
// Verify their first extent against the independently captured, identified u3d.
#[no_mangle] pub unsafe extern "C" fn bridge_unmarked_cells(v:*const Field, reference:*const Field, ids:*const i32,n:usize)->*mut Field{
    let field=&*v;let reference=&*reference;
    if !field.ids.is_empty() || field.shape.first()!=Some(&reference.ids.len()){
        return fail("unmarked cell axis extent mismatch".into())
    }
    let result=selected(Field{words:field.words.clone(),shape:field.shape.clone(),ids:reference.ids.clone(),code:field.code},Some(0),wanted(ids,n));
    match result {Ok(v)=>Box::into_raw(Box::new(v)),Err(e)=>fail(e)}
}
#[no_mangle] pub unsafe extern "C" fn bridge_ids(v:*const Field)->*const i32{(*v).ids.as_ptr()}
#[no_mangle] pub unsafe extern "C" fn bridge_meta(v:*const Field, which:usize)->usize{
    let f=&*v;
    match which {0=>f.words.len(),1=>f.shape.len(),2=>f.ids.len(),3=>f.code as usize,i=>f.shape.get(i-4).copied().unwrap_or(1)}
}
#[no_mangle] pub unsafe extern "C" fn bridge_free(v:*mut Field){if !v.is_null(){drop(Box::from_raw(v));}}
