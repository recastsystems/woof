//! Rust-only canonical frameset publication for device-computed TC updates.
//! Numerical transforms live in CUDA. This library checks and writes arrays.
//! Format: gpuwm-mapped-frameset-v1, matching mapped_engine_bridge.py.
use rayon::prelude::*;
mod diagnostics;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::BTreeMap, ffi::{CStr,CString,c_char},
    fs::{self,File}, io::{Read,Write}, path::PathBuf, slice};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
trait FileExt {
    fn read_exact_at(&self, buffer:&mut [u8], offset:u64)->std::io::Result<()>;
    fn write_all_at(&self, buffer:&[u8], offset:u64)->std::io::Result<()>;
}
#[cfg(windows)]
impl FileExt for File {
    fn read_exact_at(&self, mut buffer:&mut [u8], mut offset:u64)->std::io::Result<()> {
        use std::os::windows::fs::FileExt as WindowsFileExt;
        while !buffer.is_empty() {
            let n=self.seek_read(buffer,offset)?;
            if n==0{return Err(std::io::ErrorKind::UnexpectedEof.into());}
            buffer=&mut buffer[n..];offset+=n as u64;
        } Ok(())
    }
    fn write_all_at(&self, mut buffer:&[u8], mut offset:u64)->std::io::Result<()> {
        use std::os::windows::fs::FileExt as WindowsFileExt;
        while !buffer.is_empty() {
            let n=self.seek_write(buffer,offset)?;
            if n==0{return Err(std::io::ErrorKind::WriteZero.into());}
            buffer=&buffer[n..];offset+=n as u64;
        } Ok(())
    }
}
type Result<T> = std::result::Result<T,String>;
/// The mapped engine's full and windowed frameset documents (gpuwm/ingest/atmospheric_window.py), as rw_storm_merge.
const FRAMESET_SCHEMAS:[&str;2]=["gpuwm-mapped-frameset-v1","gpuwm-mapped-windowed-frameset-v1"];
thread_local! {static ERROR:RefCell<CString>=RefCell::new(CString::new("").unwrap());}
#[repr(C)]
pub struct Field { pub name:*const c_char, pub values:*const f64, pub count:usize }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request { schema:String, input:PathBuf, output:PathBuf, valid_time:String, input_manifest_sha256:String, transform_sha256:String }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadRequest { schema:String, input:PathBuf, valid_time:String, input_manifest_sha256:String }
fn err(e:impl std::fmt::Display)->String {e.to_string()}
fn hash(b:&[u8])->String {format!("{:x}",Sha256::digest(b))}
fn array_hasher(shape:&[usize])->Sha256 {
    let mut h=Sha256::new();h.update(b"<f8\0");h.update(format!("{shape:?}").as_bytes());h.update(b"\0");h
}
fn number(v:&Value,key:&str)->Result<usize>{v[key].as_u64().and_then(|v|usize::try_from(v).ok()).ok_or_else(||format!("invalid {key}"))}
fn string<'a>(v:&'a Value,key:&str)->Result<&'a str>{v[key].as_str().ok_or_else(||format!("invalid {key}"))}
struct WriteField { frame:usize,index:usize,digest:String, changed:bool }
fn publish(request:&[u8],patches:BTreeMap<String,&[f64]>)->Result<()> {
    let r:Request=serde_json::from_slice(request).map_err(err)?;
    if cfg!(target_endian="big") {return Err("canonical buffer ABI requires little endian".into())}
    if r.schema!="gpuwm-tc-frame-patch/1" || patches.is_empty() || r.transform_sha256.len()!=64 || !r.transform_sha256.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b)) {return Err("invalid patch declaration".into())}
    if r.output.exists() {return Err("output must be new; input frames are immutable".into())}
    let original=fs::read(r.input.join("frames.json")).map_err(err)?;
    if hash(&original)!=r.input_manifest_sha256 {return Err("background manifest changed after transform admission".into())}
    let mut doc:Value=serde_json::from_slice(&original).map_err(err)?;
    // TC-VI1: a mapped preparation with a full initial frame publishes the windowed schema (later frames windowed);
    // only the selected frame must be full, which the frame checks below require
    if !FRAMESET_SCHEMAS.iter().any(|v|doc["schema"]==*v) || doc["stream"]["path"]!="frames.f64" || doc["stream"]["dtype"]!="<f8" {return Err("unsupported canonical frameset".into())}
    let frames=doc["frames"].as_array().ok_or("missing frames")?;
    let selected:Vec<_>=frames.iter().enumerate().filter(|(_,f)|f["valid_time"]==r.valid_time).map(|(i,_)|i).collect();
    if selected.len()!=1 {return Err("select exactly one time and member".into())}
    let selected=selected[0];
    if frames[selected].get("atmospheric_window").is_some(){return Err("patch requires full source fields".into())}
    let source=File::open(r.input.join("frames.f64")).map_err(err)?;
    let size=source.metadata().map_err(err)?.len();
    if doc["stream"]["bytes"].as_u64()!=Some(size){return Err("stream size differs".into())}
    let inventory=frames.iter().enumerate().flat_map(|(i,f)|f["fields"].as_array().into_iter().flatten().enumerate().map(move |(j,d)|(i,j,d))).collect::<Vec<_>>();
    for name in patches.keys(){
        if inventory.iter().filter(|(i,_,d)|*i==selected&&d["name"]==*name).count()!=1 {return Err(format!("patch field absent or duplicated: {name}"))}
    }
    let mut spans=inventory.iter().map(|(_,_,d)|Ok((number(d,"offset")?,number(d,"length")?))).collect::<Result<Vec<_>>>()?;
    spans.sort_unstable();let mut end=0usize;
    for &(offset,length) in &spans {if offset!=end||length%8!=0{return Err("stream fields must cover disjoint contiguous spans".into())}end=end.checked_add(length).ok_or("field range overflow")?;}
    if end as u64!=size{return Err("field spans do not cover stream".into())}
    fs::create_dir(&r.output).map_err(err)?;
    let out=File::create(r.output.join("frames.f64")).map_err(err)?;out.set_len(size).map_err(err)?;
    // Stream bounded chunks per worker. The large forcing window never becomes
    // a second in-memory copy. A failed validation leaves no published manifest.
    let writes=inventory.par_iter().map(|&(fi,di,d)|->Result<WriteField>{
        let name=string(d,"name")?;let count=number(d,"length")?;let offset=number(d,"offset")?;
        let shape:Vec<usize>=serde_json::from_value(d["shape"].clone()).map_err(err)?;
        let elements=shape.iter().try_fold(1usize,|a,&b|a.checked_mul(b)).ok_or("shape overflow")?;
        if d["dtype"]!="<f8"||elements.checked_mul(8)!=Some(count){return Err(format!("invalid field {name}"))}
        let changed=fi==selected&&patches.contains_key(name);
        let replacement=if changed {
            let values=patches[name];if values.len()!=elements||values.par_iter().any(|v|!v.is_finite()){return Err(format!("patch shape/nonfinite failure: {name}"))}
            Some(unsafe{slice::from_raw_parts(values.as_ptr().cast::<u8>(),count)})
        } else {None};
        let mut prior_hash=array_hasher(&shape);let mut result_hash=array_hasher(&shape);
        let mut bytes=vec![0u8;count.min(8*1024*1024)];let mut cursor=0;
        while cursor<count {
            let n=bytes.len().min(count-cursor);let chunk=&mut bytes[..n];
            source.read_exact_at(chunk,(offset+cursor) as u64).map_err(err)?;
            prior_hash.update(&*chunk);
            let output=match replacement {Some(b)=>&b[cursor..cursor+n],None=>&*chunk};
            result_hash.update(output);out.write_all_at(output,(offset+cursor) as u64).map_err(err)?;
            cursor+=n;
        }
        if format!("{:x}",prior_hash.finalize())!=string(d,"sha256")? {return Err(format!("donor digest differs: {name}"))}
        Ok(WriteField{frame:fi,index:di,digest:format!("{:x}",result_hash.finalize()),changed})
    }).collect::<Result<Vec<_>>>()?;
    out.sync_all().map_err(err)?;
    for w in &writes {
        let d=&mut doc["frames"][w.frame]["fields"][w.index];d["sha256"]=json!(w.digest);
        if w.changed {d["source_references"].as_array_mut().ok_or("missing source references")?.push(json!(format!("tc-vortex:{}",r.transform_sha256)));}
    }
    let descriptors=doc["frames"][selected]["fields"].as_array().unwrap().clone();
    for header in doc["frames"][selected]["header"]["fields"].as_array_mut().ok_or("missing field headers")? {
        let name=string(header,"canonical_name")?.to_owned();
        let d=descriptors.iter().find(|d|d["name"]==name).ok_or("header field absent from payload")?;
        header["data_reference"]=json!(format!("sha256:{}",string(d,"sha256")?));
        if patches.contains_key(&name){header["source_field"]=json!(format!("{};tc-vortex:{}",string(header,"source_field")?,r.transform_sha256));}
    }
    let mut stream=File::open(r.output.join("frames.f64")).map_err(err)?;let mut h=Sha256::new();let mut b=vec![0;8*1024*1024];
    loop {let n=stream.read(&mut b).map_err(err)?;if n==0{break}h.update(&b[..n]);}
    doc["stream"]["sha256"]=json!(format!("{:x}",h.finalize()));
    fs::write(r.output.join("tc-patch-receipt.json"),serde_json::to_vec_pretty(&json!({"schema":"gpuwm-tc-frame-patch-receipt/1","request_sha256":hash(request),"background_manifest_sha256":hash(&original),"transform_sha256":r.transform_sha256,"valid_time":r.valid_time,"changed_fields":patches.keys().collect::<Vec<_>>(),"geometry":"unchanged","parallelism":"rayon per field and pwrite"})).map_err(err)?).map_err(err)?;
    let mut manifest=File::create(r.output.join("frames.json.part")).map_err(err)?;
    manifest.write_all(&serde_json::to_vec_pretty(&doc).map_err(err)?).map_err(err)?;manifest.sync_all().map_err(err)?;
    fs::rename(r.output.join("frames.json.part"),r.output.join("frames.json")).map_err(err)?;
    Ok(())
}
#[no_mangle] pub extern "C" fn tc_frames_abi_version()->u32{
    std::hint::black_box(concat!("GPUWM_BRIDGE_SOURCE_REV=",env!("GPUWM_BRIDGE_SOURCE_REV")));4
}
#[no_mangle] pub extern "C" fn tc_frames_last_error()->*const c_char{ERROR.with(|e|e.borrow().as_ptr())}
/// Exact-lattice storage overlay. No interpolation or arithmetic on fields:
/// retain the native input bits wherever canonical before/after bits agree.
/// Output is prefilled from the native input; every buffer must be disjoint.
#[no_mangle] pub unsafe extern "C" fn tc_frames_overlay(
    before:*const f64,after:*const f64,output:*mut f64,
    sx:usize,sy:usize,sk:usize,ix:*const usize,iy:*const usize,ik:*const usize,
    nx:usize,ny:usize,nk:usize)->i32{
    let result=std::panic::catch_unwind(||->Result<()>{
        if before.is_null()||after.is_null()||output.is_null()||ix.is_null()||iy.is_null()||ik.is_null(){return Err("null overlay buffer".into())}
        let source=sx.checked_mul(sy).and_then(|v|v.checked_mul(sk)).ok_or("source shape overflow")?;
        let target=nx.checked_mul(ny).and_then(|v|v.checked_mul(nk)).ok_or("target shape overflow")?;
        if source==0||target==0||source>isize::MAX as usize/8||target>isize::MAX as usize/8{return Err("invalid overlay shape".into())}
        let out_start=output as usize;let out_end=out_start.checked_add(target*8).ok_or("overlay range overflow")?;
        for (p,n) in [(before as usize,source*8),(after as usize,source*8),(ix as usize,nx*std::mem::size_of::<usize>()),(iy as usize,ny*std::mem::size_of::<usize>()),(ik as usize,nk*std::mem::size_of::<usize>())]{
            if out_start<p.checked_add(n).ok_or("overlay range overflow")? && p<out_end{return Err("overlay output aliases an input".into())}
        }
        let x=slice::from_raw_parts(ix,nx);let y=slice::from_raw_parts(iy,ny);let k=slice::from_raw_parts(ik,nk);
        if x.iter().any(|&i|i>=sx)||y.iter().any(|&i|i>=sy)||k.iter().any(|&i|i>=sk){return Err("overlay index outside source".into())}
        let a=slice::from_raw_parts(before,source);let b=slice::from_raw_parts(after,source);
        slice::from_raw_parts_mut(output,target).par_iter_mut().enumerate().try_for_each(|(i,v)|->Result<()>{
            let j=(k[i/(nx*ny)]*sy+y[i/nx%ny])*sx+x[i%nx];
            if !a[j].is_finite()||!b[j].is_finite(){return Err("nonfinite overlay source".into())}
            if a[j].to_bits()!=b[j].to_bits(){*v=b[j]}Ok(())
        })
    }).unwrap_or_else(|_|Err("Rust canonical overlay panic".into()));
    match result{Ok(())=>0,Err(e)=>{ERROR.with(|v|*v.borrow_mut()=CString::new(e.replace('\0'," ")).unwrap());1}}
}
/// Caller supplies distinct writable buffers of the declared field lengths.
/// Reads and digest checks run in parallel; no buffer is valid on failure.
#[no_mangle] pub unsafe extern "C" fn tc_frames_read(request:*const c_char,fields:*const Field,count:usize)->i32{
    let result=std::panic::catch_unwind(||->Result<()>{
        if cfg!(target_endian="big") || request.is_null()||fields.is_null()||count==0||count>1024{return Err("invalid read ABI input".into())}
        let r:ReadRequest=serde_json::from_slice(CStr::from_ptr(request).to_bytes()).map_err(err)?;
        if r.schema!="gpuwm-tc-frame-read/1"{return Err("unsupported read schema".into())}
        let bytes=fs::read(r.input.join("frames.json")).map_err(err)?;
        if hash(&bytes)!=r.input_manifest_sha256{return Err("background manifest changed before read".into())}
        let doc:Value=serde_json::from_slice(&bytes).map_err(err)?;
        if !FRAMESET_SCHEMAS.iter().any(|v|doc["schema"]==*v)||doc["stream"]["path"]!="frames.f64"||doc["stream"]["dtype"]!="<f8"{return Err("unsupported canonical frameset".into())}
        let frames=doc["frames"].as_array().ok_or("missing frames")?;
        let selected:Vec<_>=frames.iter().filter(|f|f["valid_time"]==r.valid_time).collect();
        if selected.len()!=1||selected[0].get("atmospheric_window").is_some(){return Err("read requires exactly one full time/member frame".into())}
        let inventory=selected[0]["fields"].as_array().ok_or("missing fields")?;
        let source=File::open(r.input.join("frames.f64")).map_err(err)?;
        let size=source.metadata().map_err(err)?.len();
        if doc["stream"]["bytes"].as_u64()!=Some(size){return Err("stream size differs".into())}
        let mut work=Vec::new();let mut spans=Vec::new();let mut names=std::collections::BTreeSet::new();
        for f in slice::from_raw_parts(fields,count){
            if f.name.is_null()||f.values.is_null()||f.count==0||f.count>isize::MAX as usize/8{return Err("invalid read pointer/length".into())}
            let name=CStr::from_ptr(f.name).to_str().map_err(err)?;
            if !names.insert(name.to_owned()){return Err("duplicate read field".into())}
            let matches:Vec<_>=inventory.iter().filter(|d|d["name"]==name).collect();
            if matches.len()!=1{return Err(format!("read field absent or duplicated: {name}"))}
            let d=matches[0];let shape:Vec<usize>=serde_json::from_value(d["shape"].clone()).map_err(err)?;
            let elements=shape.iter().try_fold(1usize,|a,&b|a.checked_mul(b)).ok_or("shape overflow")?;
            let length=number(d,"length")?;let offset=number(d,"offset")?;
            if d["dtype"]!="<f8"||elements!=f.count||elements.checked_mul(8)!=Some(length)||offset.checked_add(length).ok_or("field range overflow")? as u64>size{return Err(format!("invalid read shape/range: {name}"))}
            let start=f.values as usize;let end=start.checked_add(length).ok_or("read buffer range overflow")?;
            spans.push((start,end));work.push((start,length,offset,shape,string(d,"sha256")?.to_owned()));
        }
        spans.sort_unstable();if spans.windows(2).any(|w|w[0].1>w[1].0){return Err("read buffers overlap".into())}
        work.par_iter().try_for_each(|(ptr,length,offset,shape,digest)|->Result<()>{
            let target=slice::from_raw_parts_mut(*ptr as *mut u8,*length);
            source.read_exact_at(target,*offset as u64).map_err(err)?;
            let mut h=array_hasher(shape);h.update(target);
            if format!("{:x}",h.finalize())!=*digest{return Err("read field digest differs".into())}Ok(())
        })
    }).unwrap_or_else(|_|Err("Rust canonical read panic".into()));
    match result{Ok(())=>0,Err(e)=>{ERROR.with(|v|*v.borrow_mut()=CString::new(e.replace('\0'," ")).unwrap());1}}
}
/// Caller owns readable request/field buffers until return. No pointer retained.
#[no_mangle] pub unsafe extern "C" fn tc_frames_publish(request:*const c_char,fields:*const Field,count:usize)->i32{
    let result=std::panic::catch_unwind(||->Result<()>{
        if request.is_null()||fields.is_null()||count==0||count>1024{return Err("null/invalid patch ABI input".into())}
        let mut patches=BTreeMap::new();
        for f in slice::from_raw_parts(fields,count){
            if f.name.is_null()||f.values.is_null()||f.count==0||f.count>isize::MAX as usize/8{return Err("invalid patch pointer/length".into())}
            let name=CStr::from_ptr(f.name).to_str().map_err(err)?.to_owned();
            if patches.insert(name,slice::from_raw_parts(f.values,f.count)).is_some(){return Err("duplicate patch name".into())}
        }
        publish(CStr::from_ptr(request).to_bytes(),patches)
    }).unwrap_or_else(|_|Err("Rust canonical patch panic".into()));
    match result {Ok(())=>0,Err(e)=>{ERROR.with(|v|*v.borrow_mut()=CString::new(e.replace('\0'," ")).unwrap());1}}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64,Ordering};
    static SERIAL:AtomicU64=AtomicU64::new(0);
    fn setup()->(PathBuf,Value){
        let root=std::env::temp_dir().join(format!("tc-frame-io-{}-{}",std::process::id(),SERIAL.fetch_add(1,Ordering::Relaxed)));
        fs::create_dir(&root).unwrap();fs::create_dir(root.join("in")).unwrap();
        let bytes=[1.,2.,3.,4.].iter().flat_map(|v:&f64|v.to_le_bytes()).collect::<Vec<_>>();
        let mut frames=vec![];
        for (i,time) in ["2022-09-26T00:00:00","2022-09-26T03:00:00"].iter().enumerate(){
            let mut h=array_hasher(&[2]);h.update(&bytes[i*16..i*16+16]);let digest=format!("{:x}",h.finalize());
            frames.push(json!({"valid_time":time,"fields":[{"name":"air_temperature","dtype":"<f8","shape":[2],"offset":i*16,"length":16,"sha256":digest,"source_references":["fixture"]}],"header":{"fields":[{"canonical_name":"air_temperature","source_field":"fixture","data_reference":format!("sha256:{digest}")}]}}));
        }
        let doc=json!({"schema":"gpuwm-mapped-frameset-v1","stream":{"path":"frames.f64","dtype":"<f8","bytes":32,"sha256":hash(&bytes)},"frames":frames});
        let document=serde_json::to_vec(&doc).unwrap();
        fs::write(root.join("in/frames.json"),&document).unwrap();fs::write(root.join("in/frames.f64"),bytes).unwrap();
        let request=json!({"schema":"gpuwm-tc-frame-patch/1","input":root.join("in"),"output":root.join("out"),"valid_time":"2022-09-26T00:00:00","input_manifest_sha256":hash(&document),"transform_sha256":"a".repeat(64)});
        (root,request)
    }
    #[test]
    fn updates_payload_and_header_digest_preserving_later_frame(){
        let (root,request)=setup();let values=[10.,20.];let mut patch=BTreeMap::new();patch.insert("air_temperature".into(),values.as_slice());
        publish(&serde_json::to_vec(&request).unwrap(),patch).unwrap();
        let source=fs::read(root.join("in/frames.f64")).unwrap();let result=fs::read(root.join("out/frames.f64")).unwrap();assert_eq!(&source[16..],&result[16..]);
        assert_eq!(f64::from_le_bytes(result[..8].try_into().unwrap()),10.);
        let d:Value=serde_json::from_slice(&fs::read(root.join("out/frames.json")).unwrap()).unwrap();
        assert_eq!(d["frames"][0]["header"]["fields"][0]["data_reference"],format!("sha256:{}",d["frames"][0]["fields"][0]["sha256"].as_str().unwrap()));
        assert_eq!(d["stream"]["sha256"],hash(&result));
        // Files created only by this test.
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn corrupted_untouched_field_never_publishes_manifest(){
        let(root,request)=setup();let file=File::options().write(true).open(root.join("in/frames.f64")).unwrap();file.write_all_at(&[0;8],24).unwrap();
        let values=[10.,20.];let mut patch=BTreeMap::new();patch.insert("air_temperature".into(),values.as_slice());
        assert!(publish(&serde_json::to_vec(&request).unwrap(),patch).unwrap_err().contains("donor digest"));
        assert!(!root.join("out/frames.json").exists());fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn wrong_transform_background_refused_before_output(){
        let(root,mut request)=setup();request["input_manifest_sha256"]=json!("b".repeat(64));
        let values=[10.,20.];let mut patch=BTreeMap::new();patch.insert("air_temperature".into(),values.as_slice());
        assert!(publish(&serde_json::to_vec(&request).unwrap(),patch).unwrap_err().contains("manifest changed"));
        assert!(!root.join("out").exists());fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn reader_checks_payload_and_refuses_changed_bytes(){
        let(root,request)=setup();
        let request=CString::new(serde_json::to_string(&json!({"schema":"gpuwm-tc-frame-read/1","input":root.join("in"),"valid_time":request["valid_time"],"input_manifest_sha256":request["input_manifest_sha256"]})).unwrap()).unwrap();
        let name=CString::new("air_temperature").unwrap();let mut output=[0.;2];
        let field=Field{name:name.as_ptr(),values:output.as_mut_ptr(),count:2};
        assert_eq!(unsafe{tc_frames_read(request.as_ptr(),&field,1)},0);assert_eq!(output,[1.,2.]);
        File::options().write(true).open(root.join("in/frames.f64")).unwrap().write_all_at(&[0;8],0).unwrap();
        assert_ne!(unsafe{tc_frames_read(request.as_ptr(),&field,1)},0);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn overlay_retains_native_rounding_and_maps_the_longitude_cut(){
        let a=[1.,2.,3.,4.];let b=[1.,20.,3.,4.];let mut out=[3.01,4.01,1.01,2.01];
        let x=[2usize,3,0,1];let zero=[0usize];
        assert_eq!(unsafe{tc_frames_overlay(a.as_ptr(),b.as_ptr(),out.as_mut_ptr(),4,1,1,x.as_ptr(),zero.as_ptr(),zero.as_ptr(),4,1,1)},0);
        assert_eq!(out,[3.01,4.01,1.01,20.]);
        let invalid=[4usize];
        assert_ne!(unsafe{tc_frames_overlay(a.as_ptr(),b.as_ptr(),out.as_mut_ptr(),4,1,1,invalid.as_ptr(),zero.as_ptr(),zero.as_ptr(),1,1,1)},0);
    }
}
