//! MPAS native streams, with raw primitive-word handoff to the driver.
//!
//! MPAS v8.4.2 src/core_atmosphere/Registry.xml:498-1349 defines membership;
//! src/framework/mpas_io_streams.F:2430-3927 defines native file layout. Dimension order
//! is the reverse of Registry/Fortran order, including StrLen for text. This
//! bridge owns file decoding, encoding, permutation and connectivity conversion.
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};

pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

// Built only with the `system-netcdf` feature (Cargo.toml required-features),
// which links the system netCDF C library; a default `cargo test -p rw-mpas`
// builds neither this binary nor the link.
#[cfg_attr(feature = "system-netcdf", link(name = "netcdf"))]
unsafe extern "C" {
    fn nc_strerror(code: c_int) -> *const c_char;
    fn nc_open(path: *const c_char, mode: c_int, id: *mut c_int) -> c_int;
    fn nc_close(id: c_int) -> c_int;
    fn nc_enddef(id: c_int) -> c_int;
    fn nc__enddef(id: c_int, header_free: usize, variable_alignment: usize,
                 variable_free: usize, record_alignment: usize) -> c_int;
    fn nc_def_dim(id: c_int, name: *const c_char, len: usize, dim: *mut c_int) -> c_int;
    fn nc_def_var(id: c_int, name: *const c_char, kind: c_int, nd: c_int, dims: *const c_int, var: *mut c_int) -> c_int;
    fn nc_put_vara(id: c_int, var: c_int, start: *const usize, count: *const usize, data: *const c_void) -> c_int;
    fn nc_get_vara(id: c_int, var: c_int, start: *const usize, count: *const usize, data: *mut c_void) -> c_int;
    fn nc_put_att(id: c_int, var: c_int, name: *const c_char, kind: c_int, len: usize, data: *const c_void) -> c_int;
    fn nc_put_att_text(id: c_int, var: c_int, name: *const c_char, len: usize, data: *const c_char) -> c_int;
    fn nc_get_att(id: c_int, var: c_int, name: *const c_char, data: *mut c_void) -> c_int;
    fn nc_inq(id: c_int, nd: *mut c_int, nv: *mut c_int, na: *mut c_int, ud: *mut c_int) -> c_int;
    fn nc_inq_format(id: c_int, format: *mut c_int) -> c_int;
    fn nc_inq_unlimdims(id: c_int, n: *mut c_int, dims: *mut c_int) -> c_int;
    fn nc_inq_dim(id: c_int, dim: c_int, name: *mut c_char, len: *mut usize) -> c_int;
    fn nc_inq_var(id: c_int, var: c_int, name: *mut c_char, kind: *mut c_int, nd: *mut c_int, dims: *mut c_int, na: *mut c_int) -> c_int;
    fn nc_inq_attname(id: c_int, var: c_int, index: c_int, name: *mut c_char) -> c_int;
    fn nc_inq_att(id: c_int, var: c_int, name: *const c_char, kind: *mut c_int, len: *mut usize) -> c_int;
    fn nc_inq_varid(id: c_int, name: *const c_char, var: *mut c_int) -> c_int;
    fn nc__create(path: *const c_char, mode: c_int, initial_size: usize, chunk_hint: *mut usize, id: *mut c_int) -> c_int;
    fn nc__open(path: *const c_char, mode: c_int, chunk_hint: *mut usize, id: *mut c_int) -> c_int;
    fn nc_set_fill(id: c_int, mode: c_int, old_mode: *mut c_int) -> c_int;
}
// netCDF-C's classic-format I/O moves data through a buffer of this many bytes.
// The default is the file system block size (4 KiB here), so a 13.4 GB conus3km
// history frame went to disk in millions of small read-modify-write steps on one
// core. The size changes how the bytes are written, never which bytes.
const WRITE_CHUNK_BYTES: usize = 8 << 20;
const NC_NOFILL: c_int = 0x100;
const NC_FORMAT_NETCDF4: c_int = 3;
const NC_FORMAT_NETCDF4_CLASSIC: c_int = 4;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn check(code: c_int) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(unsafe { CStr::from_ptr(nc_strerror(code)) }.to_string_lossy().into_owned().into()) }
}
fn c(text: &str) -> Result<CString> { Ok(CString::new(text)?) }
struct Nc(c_int);
impl Drop for Nc { fn drop(&mut self) { unsafe { nc_close(self.0); } } }
impl Nc {
    fn open(path: &Path, mode: c_int) -> Result<Self> {
        let mut id = 0; check(unsafe { nc_open(c(&path.to_string_lossy())?.as_ptr(), mode, &mut id) })?; Ok(Self(id))
    }
    fn create(path: &Path, format: &str) -> Result<Self> {
        let flag = match format { "netcdf4" => 0x1000, "pnetcdf,cdf5" => 0x20,
            "netcdf" | "pnetcdf" => 0x200, _ => return Err(format!("unknown file format {format}").into()) };
        let mut id = 0; let mut chunk = WRITE_CHUNK_BYTES;
        check(unsafe { nc__create(c(&path.to_string_lossy())?.as_ptr(), flag | 4, 0, &mut chunk, &mut id) })?; Ok(Self(id))
    }
    fn open_for_write(path: &Path) -> Result<Self> {
        let mut id = 0; let mut chunk = WRITE_CHUNK_BYTES;
        check(unsafe { nc__open(c(&path.to_string_lossy())?.as_ptr(), 1, &mut chunk, &mut id) })?; Ok(Self(id))
    }
    fn close(self) -> Result<()> { let id = self.0; std::mem::forget(self); check(unsafe { nc_close(id) }) }
}
fn dtype(kind: i32) -> Result<(&'static str, usize)> {
    Ok(match kind { 1 => ("i1",1), 2 => ("S1",1), 3 => ("i2",2), 4 => ("i4",4),
        5 => ("f4",4), 6 => ("f8",8), 7 => ("u1",1), 8 => ("u2",2), 9 => ("u4",4),
        10 => ("i8",8), 11 => ("u8",8), _ => return Err(format!("unsupported NetCDF primitive {kind}").into()) })
}
fn kind(dtype: &str) -> Result<i32> {
    for k in 1..=11 { if self::dtype(k)?.0 == dtype { return Ok(k); } }
    Err(format!("unsupported handoff dtype {dtype}").into())
}
// Eight-byte alignment for libnetcdf's raw typed access, including doubles.
fn aligned(bytes: usize) -> Vec<u64> { vec![0; bytes.div_ceil(8)] }
fn bytes(buf: &[u64], len: usize) -> &[u8] { unsafe { std::slice::from_raw_parts(buf.as_ptr().cast(),len) } }
fn bytes_mut(buf: &mut [u64], len: usize) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast(),len) } }
fn product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |a,b| a.checked_mul(*b).ok_or_else(|| "array shape overflows usize".into()))
}
fn text_name(buf: &[c_char]) -> String { unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned() }

#[derive(Deserialize)]
struct Request {
    operation: String,
    path: PathBuf,
    #[serde(default = "default_format")] format: String,
    #[serde(default)] dimensions: BTreeMap<String,usize>,
    #[serde(default)] dimension_order: Vec<String>,
    #[serde(default)] global_attribute_order: Vec<String>,
    #[serde(default)] header_alignment: usize,
    #[serde(default)] variables: Vec<Variable>,
    #[serde(default)] global_attrs: BTreeMap<String,Value>,
    #[serde(default)] output_directory: PathBuf,
    #[serde(default)] select: Vec<String>,
    #[serde(default)] record: Option<usize>,
    #[serde(default)] record_index: Option<usize>,
    #[serde(default)] append: bool,
    #[serde(default)] layouts: BTreeMap<String,ReadLayout>,
    /// A write as lossless compressed NetCDF-4 (see `write_compressed`); None writes `format` as before.
    #[serde(default)] compression: Option<Compression>,
}
/// Lossless storage of a new file: byte shuffle + deflate, in NetCDF-4 (NETCDF4_CLASSIC).
#[derive(Deserialize)]
struct Compression {
    /// zlib level 1-9 (the history default, 4: GPUWM_HISTORY_COMPRESSION_LEVEL).
    #[serde(default = "default_level")] level: u32,
    /// Compression threads, 0 = every core (GPUWM_HISTORY_COMPRESSION_THREADS).
    #[serde(default)] threads: usize,
}
fn default_level() -> u32 { 4 }
#[derive(Deserialize)]
struct ReadLayout {
    dtype: String,
    dimensions: Vec<String>,
    expected_dimensions: Vec<String>,
    #[serde(default)] connectivity_offset: i32,
}
fn default_format() -> String { "netcdf".into() }
#[derive(Serialize,Deserialize,Clone)]
struct Variable {
    name: String,
    dtype: String,
    dimensions: Vec<String>,
    #[serde(default)] shape: Vec<usize>,
    #[serde(default)] input_dimensions: Vec<String>,
    #[serde(default)] filename: PathBuf,
    #[serde(default)] attrs: BTreeMap<String,Value>,
    #[serde(default)] attribute_order: Vec<String>,
    #[serde(default)] text: Vec<String>,
    #[serde(default)] connectivity_offset: i32,
    #[serde(default)] input_dtype: Option<String>,
}

// MPAS v8.4.2 src/framework/mpas_io_streams.F:4086-4244: native primitive/layout semantics.
fn encode_attr(value: &Value) -> Result<(i32, Vec<u8>)> {
    let (k, v) = if let Some(obj) = value.as_object() {
        (kind(obj["dtype"].as_str().ok_or("attribute dtype missing")?)?, obj.get("values").ok_or("attribute values missing")?)
    } else { (if value.is_i64() {4} else {6},value) };
    let values = if let Some(a) = v.as_array() { a.clone() } else { vec![v.clone()] };
    let mut out = Vec::new();
    for x in values {
        macro_rules! signed { ($t:ty) => { out.extend_from_slice(&<$t>::try_from(x.as_i64().ok_or("integer attribute required")?)?.to_ne_bytes()) } }
        macro_rules! unsigned { ($t:ty) => { out.extend_from_slice(&<$t>::try_from(x.as_u64().ok_or("unsigned attribute required")?)?.to_ne_bytes()) } }
        match k { 1 => signed!(i8), 3 => signed!(i16),4 => signed!(i32),10 => signed!(i64),
            7 => unsigned!(u8),8 => unsigned!(u16),9 => unsigned!(u32),11 => unsigned!(u64),
            5 => out.extend_from_slice(&(x.as_f64().ok_or("real attribute required")? as f32).to_ne_bytes()),
            6 => out.extend_from_slice(&x.as_f64().ok_or("real attribute required")?.to_ne_bytes()),
            _ => return Err("unsupported attribute primitive".into()) }
    }
    Ok((k,out))
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:4315-4363: native primitive/layout semantics.
fn ordered_keys<'a,T>(values: &'a BTreeMap<String,T>, order: &'a [String]) -> Result<Vec<&'a String>> {
    if order.is_empty() {return Ok(values.keys().collect());}
    let keys:std::collections::BTreeSet<_>=order.iter().collect();
    if keys.len()!=order.len() || keys!=values.keys().collect() {
        return Err("explicit metadata order must list each key exactly once".into());
    }
    Ok(order.iter().collect())
}
fn put_attrs(nc: &Nc, var: i32, attrs: &BTreeMap<String,Value>, order: &[String], variable_kind: Option<i32>) -> Result<()> {
    for name in ordered_keys(attrs,order)? {
        let value=&attrs[name];
        if name == "_NCProperties" { continue; }
        let cname = c(name)?;
        if let Some(s) = value.as_str() {
            check(unsafe { nc_put_att_text(nc.0,var,cname.as_ptr(),s.len(),s.as_ptr().cast()) })?;
        } else {
            // _FillValue must have exactly the variable's primitive type.
            let v = if name == "_FillValue" && !value.is_object() {
                json!({"dtype":dtype(variable_kind.ok_or("global fill value")?)?.0,"values":value})
            } else {value.clone()};
            let (k,raw)=encode_attr(&v)?; let len=raw.len()/dtype(k)?.1;
            let mut buf=aligned(raw.len()); bytes_mut(&mut buf,raw.len()).copy_from_slice(&raw);
            check(unsafe { nc_put_att(nc.0,var,cname.as_ptr(),k,len,buf.as_ptr().cast()) })?;
        }
    }
    Ok(())
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:3930-4083,4315-4363: native primitive/layout semantics.
fn get_attrs(nc: &Nc, var: i32, n: i32) -> Result<BTreeMap<String,Value>> {
    let mut attrs=BTreeMap::new();
    for i in 0..n {
        let mut name=[0;257]; check(unsafe { nc_inq_attname(nc.0,var,i,name.as_mut_ptr()) })?;
        let mut k=0; let mut len=0; check(unsafe { nc_inq_att(nc.0,var,name.as_ptr(),&mut k,&mut len) })?;
        let size=dtype(k)?.1; let mut buf=aligned(len.checked_mul(size).ok_or("attribute size overflow")?);
        check(unsafe { nc_get_att(nc.0,var,name.as_ptr(),buf.as_mut_ptr().cast()) })?;
        let raw=bytes(&buf,len*size);
        let value=if k==2 { Value::String(String::from_utf8(raw.to_vec())?) } else {
            let mut values=Vec::new();
            for chunk in raw.chunks_exact(size) {
                macro_rules! decode { ($t:ty) => { json!(<$t>::from_ne_bytes(chunk.try_into()?)) } }
                values.push(match k {1=>decode!(i8),3=>decode!(i16),4=>decode!(i32),5=>decode!(f32),6=>decode!(f64),
                    7=>decode!(u8),8=>decode!(u16),9=>decode!(u32),10=>decode!(i64),11=>decode!(u64),_=>unreachable!()});
            }
            json!({"dtype":dtype(k)?.0,"values":values})
        };
        attrs.insert(text_name(&name),value);
    }
    Ok(attrs)
}

// Compare typed attribute words and order without float/JSON conversions.
fn attribute_layout(nc: &Nc, var: i32, n: i32) -> Result<(Vec<String>,BTreeMap<String,Value>)> {
    let mut order=Vec::new();let mut words=BTreeMap::new();
    for i in 0..n {
        let mut name=[0;257];check(unsafe {nc_inq_attname(nc.0,var,i,name.as_mut_ptr())})?;
        let mut k=0;let mut len=0;check(unsafe {nc_inq_att(nc.0,var,name.as_ptr(),&mut k,&mut len)})?;
        let size=len.checked_mul(dtype(k)?.1).ok_or("attribute size overflow")?;
        let mut data=aligned(size);check(unsafe {nc_get_att(nc.0,var,name.as_ptr(),data.as_mut_ptr().cast())})?;
        let name=text_name(&name);order.push(name.clone());
        words.insert(name,json!({"dtype":dtype(k)?.0,"length":len,
            "hex":bytes(&data,size).iter().map(|v|format!("{v:02x}")).collect::<String>()}));
    }
    Ok((order,words))
}

// MPAS v8.4.2 src/framework/mpas_io_streams.F:1659-1857,2430-3927: native primitive/layout semantics.
fn prepare(v: &Variable, dims: &BTreeMap<String,usize>) -> Result<Vec<u64>> {
    if v.dimensions.iter().collect::<std::collections::BTreeSet<_>>().len()!=v.dimensions.len() {
        return Err(format!("{}: duplicate output axis",v.name).into());
    }
    let out_shape: Vec<_>=v.dimensions.iter().map(|d| dims.get(d).copied().ok_or_else(|| format!("{}: missing dimension {d}",v.name))).collect::<std::result::Result<_,_>>()?;
    let count=product(&out_shape)?; let out_kind=kind(&v.dtype)?; let size=dtype(out_kind)?.1;
    let mut out=aligned(count.checked_mul(size).ok_or("variable size overflow")?);
    if out_kind==2 && !v.text.is_empty() {
        if v.dimensions.last().map(String::as_str)!=Some("StrLen") || out_shape.last()!=Some(&64) || count/64!=v.text.len() {
            return Err(format!("{}: text shape or record count mismatch",v.name).into());
        }
        for (target,text) in bytes_mut(&mut out,count).chunks_exact_mut(64).zip(&v.text) {
            if !text.is_ascii() || text.len()>64 { return Err("MPAS text must fit 64 ASCII characters".into()); }
            target.fill(b' '); target[..text.len()].copy_from_slice(text.as_bytes());
        }
        return Ok(out);
    }
    let input_kind=kind(v.input_dtype.as_deref().unwrap_or(&v.dtype))?;
    let input_size=dtype(input_kind)?.1;
    if v.input_dimensions.len()!=v.shape.len() { return Err("input dimensions/shape mismatch".into()); }
    let mut unique=std::collections::BTreeSet::new();
    for (d,n) in v.input_dimensions.iter().zip(&v.shape) {
        if !unique.insert(d) || !v.dimensions.contains(d) || dims.get(d)!=Some(n) { return Err(format!("{}: incompatible input axis {d}",v.name).into()); }
    }
    if v.dimensions.iter().any(|d| d!="Time" && !unique.contains(d)) { return Err(format!("{}: missing input axis",v.name).into()); }
    let raw=fs::read(&v.filename)?;
    if raw.len()!=product(&v.shape)?.checked_mul(input_size).ok_or("input size overflow")? { return Err(format!("{}: raw handoff byte count mismatch",v.name).into()); }
    if count==0 { return Ok(out); }
    // The conversion is chosen once per variable, in the per-word order the
    // serial loop tested it (connectivity first, then same kind, then widening).
    let conv=if v.connectivity_offset!=0 {
        if ![4,10].contains(&input_kind) || out_kind!=4 { return Err("connectivity requires signed int32/int64 input and int32 output".into()); }
        Conv::Connectivity{wide:input_kind==10,offset:v.connectivity_offset as i64}
    } else if input_kind==out_kind { Conv::Copy }
    else if input_kind==10 && out_kind==4 { Conv::I64ToI32 }
    else if input_kind==7 && out_kind==4 { Conv::U8ToI32 }
    else if input_kind==5 && out_kind==6 { Conv::F32ToF64 }
    else if input_kind==6 && out_kind==5 { Conv::F64ToF32 }
    else { return Err(format!("{}: unsupported primitive conversion",v.name).into()); };
    // Input word stride of every output axis; an output axis the input lacks
    // (Time) repeats the input, as the serial loop's index ignored it.
    let mut strides=vec![0usize;out_shape.len()];
    for (axis,d) in v.dimensions.iter().enumerate() {
        if let Some(pos)=v.input_dimensions.iter().position(|a| a==d) { strides[axis]=v.shape[pos+1..].iter().product(); }
    }
    let target=bytes_mut(&mut out,count*size);
    permute(&raw,target,&out_shape,&strides,input_size,size,conv).map_err(|e| e.into()).map(|_| out)
}

// One word conversion, fixed per variable (MPAS v8.4.2 mpas_io_streams.F:3254-3927 primitive semantics).
#[derive(Clone,Copy)]
enum Conv { Copy, Connectivity{wide:bool,offset:i64}, I64ToI32, U8ToI32, F32ToF64, F64ToF32 }

fn convert(conv: Conv, src: &[u8], dest: &mut [u8]) -> std::result::Result<(),String> {
    match conv {
        Conv::Copy => dest.copy_from_slice(src),
        Conv::Connectivity{wide,offset} => {
            let value=if wide {i64::from_ne_bytes(src.try_into().unwrap())} else {i32::from_ne_bytes(src.try_into().unwrap()) as i64};
            let disk=i32::try_from(value.checked_add(offset).ok_or("connectivity overflow")?).map_err(|e| e.to_string())?;
            dest.copy_from_slice(&disk.to_ne_bytes());
        }
        Conv::I64ToI32 => dest.copy_from_slice(&i32::try_from(i64::from_ne_bytes(src.try_into().unwrap())).map_err(|e| e.to_string())?.to_ne_bytes()),
        Conv::U8ToI32 => dest.copy_from_slice(&(src[0] as i32).to_ne_bytes()),
        Conv::F32ToF64 => dest.copy_from_slice(&(f32::from_ne_bytes(src.try_into().unwrap()) as f64).to_ne_bytes()),
        Conv::F64ToF32 => dest.copy_from_slice(&(f64::from_ne_bytes(src.try_into().unwrap()) as f32).to_ne_bytes()),
    }
    Ok(())
}

// Threads one permute may split over; 0 is every core. A prepare_all worker sets
// its share so the variables prepared side by side do not each take every core.
thread_local! { static PERMUTE_WORKERS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
// Every variable of a write, prepared side by side. The serial loop read a 13.4 GB
// conus3km history frame's raw handoff files one after another on one thread.
// Errors keep the serial loop's order: the first failing variable in request
// order, and "duplicate variable" only when every variable before the repeat
// prepared. Each variable's bytes come from the same prepare as before.
fn prepare_all(req: &Request) -> Result<Vec<Vec<u64>>> {
    let mut names=std::collections::BTreeSet::new();
    let upto=req.variables.iter().position(|v| !names.insert(&v.name)).unwrap_or(req.variables.len());
    let vars=&req.variables[..upto];
    let cores=std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let outer=cores.min(8).min(vars.len());
    let mut slots:Vec<Option<std::result::Result<Vec<u64>,String>>>=(0..vars.len()).map(|_| None).collect();
    if outer<2 {
        for (slot,v) in slots.iter_mut().zip(vars) {
            let done=prepare(v,&req.dimensions).map_err(|e| e.to_string());
            let failed=done.is_err(); *slot=Some(done); if failed {break;}
        }
    } else {
        // Largest first, so no worker is left with a big variable at the end.
        let mut order:Vec<usize>=(0..vars.len()).collect();
        let words=|v: &Variable| v.dimensions.iter().map(|d| req.dimensions.get(d).copied().unwrap_or(0)).fold(1usize,|a,b| a.saturating_mul(b));
        order.sort_by(|a,b| words(&vars[*b]).cmp(&words(&vars[*a])).then(a.cmp(b)));
        let next=std::sync::atomic::AtomicUsize::new(0);
        let inner=(cores/outer).max(1);
        let finished:Vec<_>=std::thread::scope(|scope| {
            let handles:Vec<_>=(0..outer).map(|_| scope.spawn(|| {
                PERMUTE_WORKERS.with(|w| w.set(inner));
                let mut done=Vec::new();
                loop {
                    let k=next.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                    let Some(&i)=order.get(k) else {break};
                    done.push((i,prepare(&vars[i],&req.dimensions).map_err(|e| e.to_string())));
                }
                done
            })).collect();
            handles.into_iter().map(|h| h.join()).collect()
        });
        for worker in finished {
            let Ok(done)=worker else {return Err("stream prepare worker panicked".into())};
            for (i,result) in done {slots[i]=Some(result);}
        }
    }
    let mut prepared=Vec::with_capacity(req.variables.len());
    for slot in slots {
        match slot {
            Some(Ok(data)) => prepared.push(data),
            Some(Err(error)) => return Err(error.into()),
            None => return Err("variable was not prepared".into()),
        }
    }
    if upto<req.variables.len() {return Err("duplicate variable".into());}
    Ok(prepared)
}
// Rows of the last output axis, each a strided walk of the input. Large
// variables are split over every core in output order; the first failing word
// in output order is the error reported, as in the serial loop. Same bytes.
fn permute(raw: &[u8], target: &mut [u8], shape: &[usize], strides: &[usize], isz: usize, osz: usize,
           conv: Conv) -> std::result::Result<(),String> {
    let ndim=shape.len();
    let inner=if ndim==0 {1} else {shape[ndim-1]};
    let inner_stride=if ndim==0 {0} else {strides[ndim-1]};
    let rows=target.len()/osz/inner;
    let run=|r0: usize, chunk: &mut [u8]| -> std::result::Result<(),String> {
        for (i,row) in chunk.chunks_exact_mut(inner*osz).enumerate() {
            let mut rem=r0+i; let mut base=0;
            for axis in (0..ndim.saturating_sub(1)).rev() { base+=(rem%shape[axis])*strides[axis]; rem/=shape[axis]; }
            for (j,dest) in row.chunks_exact_mut(osz).enumerate() {
                let s=(base+j*inner_stride)*isz;
                convert(conv,&raw[s..s+isz],dest)?;
            }
        }
        Ok(())
    };
    let cap=PERMUTE_WORKERS.with(|w| w.get());
    let all=std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(32);
    let workers=if cap>0 {all.min(cap)} else {all}.min(rows);
    if workers<2 || target.len()<(4<<20) { return run(0,target); }
    let per=rows.div_ceil(workers);
    let results:Vec<_>=std::thread::scope(|scope| {
        let handles:Vec<_>=target.chunks_mut(per*inner*osz).enumerate()
            .map(|(k,chunk)| { let run=&run; scope.spawn(move || run(k*per,chunk)) }).collect();
        handles.into_iter().map(|h| h.join().unwrap_or_else(|_| Err("stream encode worker panicked".into()))).collect()
    });
    results.into_iter().collect()
}
// netCDF-C's default fill mode writes a classic-format file twice: nc_enddef fills
// every fixed variable, and the first put past the last record fills that record of
// every record variable; the puts in `write` then overwrite all of it. On a 13.4 GB
// conus3km history frame that doubled the bytes the one netCDF thread moved. When the
// puts overwrite every byte the fill would write, the file is the same without the
// fill. Classic formats only: a netCDF-4 file stores its fill setting in the file.
fn puts_cover_the_fill(nc: &Nc, req: &Request) -> Result<bool> {
    let mut format=0; check(unsafe {nc_inq_format(nc.0,&mut format)})?;
    if format==NC_FORMAT_NETCDF4 || format==NC_FORMAT_NETCDF4_CLASSIC {return Ok(false);}
    let first=req.record_index.unwrap_or(0);
    let is_record=|v: &&Variable| v.dimensions.iter().any(|d| d=="Time");
    // A classic variable is padded to four bytes, and the fill writes the padding
    // too: a byte or short variable of odd length gets its fill value there (char
    // fills with zero, as an unwritten byte reads).
    for v in &req.variables {
        let size=dtype(kind(&v.dtype)?)?.1;
        let words=v.dimensions.iter().filter(|d| *d!="Time").map(|d| req.dimensions.get(d).copied().unwrap_or(0)).fold(1usize,|a,b| a.saturating_mul(b));
        if size<4 && v.dtype!="S1" && (words*size)%4!=0 {return Ok(false);}
    }
    if !req.append {
        // This request defines every variable and writes each one whole, so the
        // only bytes left to the fill would be records before `first`.
        return Ok(first==0 || !req.variables.iter().any(|v| is_record(&v)));
    }
    let mut un=0; check(unsafe {nc_inq_unlimdims(nc.0,&mut un,std::ptr::null_mut())})?;
    if un!=1 {return Ok(false);}
    let mut unlimited=0; check(unsafe {nc_inq_unlimdims(nc.0,&mut un,&mut unlimited)})?;
    let mut name=[0;257]; let mut records=0;
    check(unsafe {nc_inq_dim(nc.0,unlimited,name.as_mut_ptr(),&mut records)})?;
    if text_name(&name)!="Time" || first>records {return Ok(false);}
    // An appended record is filled for every record variable of the file, so each
    // of them must be one this request writes.
    let written:std::collections::BTreeSet<&str>=req.variables.iter().filter(is_record).map(|v| v.name.as_str()).collect();
    let mut nv=0; check(unsafe {nc_inq(nc.0,std::ptr::null_mut(),&mut nv,std::ptr::null_mut(),std::ptr::null_mut())})?;
    for id in 0..nv {
        let (_,dims,_,_)=var_info(nc,id)?;
        if dims.first().map(String::as_str)!=Some("Time") {continue;}
        let mut name=[0;257];
        check(unsafe {nc_inq_var(nc.0,id,name.as_mut_ptr(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut())})?;
        if !written.contains(text_name(&name).as_str()) {return Ok(false);}
    }
    Ok(true)
}
fn skip_fill(nc: &Nc) -> Result<()> { let mut old=0; check(unsafe {nc_set_fill(nc.0,NC_NOFILL,&mut old)}) }
#[cfg(test)]
thread_local! { static KEEP_FILL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
fn keep_fill() -> bool { KEEP_FILL.with(|k| k.get()) }
#[cfg(not(test))]
fn keep_fill() -> bool { false }
// MPAS v8.4.2 src/framework/mpas_io_streams.F:3254-3927: native primitive/layout semantics.
fn write(req: &Request) -> Result<Value> {
    // Validate and prepare every variable before touching the destination.
    let prepared=prepare_all(req)?;
    if let Some(compression)=&req.compression { return write_compressed(req,compression,prepared); }
    let nc=if req.append {Nc::open_for_write(&req.path)?} else {Nc::create(&req.path,&req.format)?};
    // Before nc_enddef, which is where a new file's fixed variables are filled.
    if !keep_fill() && puts_cover_the_fill(&nc,req)? {skip_fill(&nc)?;}
    let mut dim_ids=BTreeMap::new();
    if !req.append {
        for name in ordered_keys(&req.dimensions,&req.dimension_order)? {
            let len=&req.dimensions[name];
            let mut id=0; check(unsafe {nc_def_dim(nc.0,c(name)?.as_ptr(),if name=="Time" {0} else {*len},&mut id)})?;
            dim_ids.insert(name.clone(),id);
        }
        put_attrs(&nc,-1,&req.global_attrs,&req.global_attribute_order,None)?;
    }
    let mut ids=Vec::new();
    for v in &req.variables {
        let mut id=0;
        if req.append {
            check(unsafe {nc_inq_varid(nc.0,c(&v.name)?.as_ptr(),&mut id)})?;
            let (k,dimensions,shape,_)=var_info(&nc,id)?;
            if k!=kind(&v.dtype)? || dimensions!=v.dimensions { return Err(format!("{}: append layout mismatch",v.name).into()); }
            for (axis,extent) in dimensions.iter().zip(shape) {
                if axis!="Time" && req.dimensions.get(axis)!=Some(&extent) {
                    return Err(format!("{}: append extent mismatch for {axis}",v.name).into());
                }
            }
        } else {
            let dims:Vec<_>=v.dimensions.iter().map(|d| dim_ids[d]).collect();
            check(unsafe {nc_def_var(nc.0,c(&v.name)?.as_ptr(),kind(&v.dtype)?,dims.len() as i32,dims.as_ptr(),&mut id)})?;
            put_attrs(&nc,id,&v.attrs,&v.attribute_order,Some(kind(&v.dtype)?))?;
        }
        ids.push(id);
    }
    if !req.append {
        if req.header_alignment > 0 {
            // SMIOL smiol.c:15,341-344 sets nc_header_align_size=131072.
            // Only the first data section is aligned. Subsequent fixed and
            // record variables retain their native four-byte packing.
            check(unsafe {nc__enddef(nc.0,0,req.header_alignment,0,4)})?;
        } else {check(unsafe {nc_enddef(nc.0)})?;}
    }
    let mut bytes_written=0;
    for ((v,id),data) in req.variables.iter().zip(ids).zip(prepared) {
        if req.append && !v.dimensions.iter().any(|d|d=="Time") {continue;}
        let mut start=vec![0;v.dimensions.len()];
        let count:Vec<_>=v.dimensions.iter().map(|d|req.dimensions[d]).collect();
        if let Some(axis)=v.dimensions.iter().position(|d|d=="Time") {start[axis]=req.record_index.unwrap_or(0);}
        check(unsafe {nc_put_vara(nc.0,id,start.as_ptr(),count.as_ptr(),data.as_ptr().cast())})?;
        bytes_written+=product(&count)?*dtype(kind(&v.dtype)?)?.1;
    }
    nc.close()?;
    Ok(json!({"schema":"mpas-native-stream/v1","variables":req.variables.len(),"bytes_written":bytes_written,"ordered_metadata":true}))
}
/// The same file as `write`, value for value (every dimension, the record dimension still unlimited, every
/// attribute and every variable in order, every word), in lossless compressed NetCDF-4: chunks byte-shuffled and
/// deflated on every core by the engine's own writer (netcdf_writer::Nc4Writer, the one ArWen's history frames use),
/// never by HDF5 on one thread. Project ruling, 2026-10-28: "Lossless compression is the default". hexcore asks for it on
/// history and diagnostics files only: native MPAS reads init, restart and LBC files through pnetcdf, which has no
/// NetCDF-4 reader. A new file only: the writer cannot extend a file already on disk, so an append (a stream whose
/// file holds several output times) or a record past the first is refused, naming why.
fn write_compressed(req: &Request, compression: &Compression, prepared: Vec<Vec<u64>>) -> Result<Value> {
    use netcdf_writer::{AttrValue, NcFormat, NcType, Nc4Options, Nc4Writer, Schema};
    if req.append || req.record_index.unwrap_or(0)!=0 {
        return Err("compressed write: a compressed file is written whole, so it cannot take an append or a record past the first (ask for the uncompressed format for a file that holds several output times)".into());
    }
    if !(1..=9).contains(&compression.level) { return Err(format!("compressed write: deflate level {} is not 1-9",compression.level).into()); }
    fn nc_type(dtype: &str) -> Result<NcType> {
        Ok(match dtype { "i1"=>NcType::Byte, "S1"=>NcType::Char, "i2"=>NcType::Short, "i4"=>NcType::Int,
            "f4"=>NcType::Float, "f8"=>NcType::Double,
            _ => return Err(format!("compressed write: {dtype} has no NETCDF4_CLASSIC type; this stream needs the uncompressed format").into()) })
    }
    fn attr(value: &Value, variable: Option<&str>) -> Result<AttrValue> {
        if let Some(s)=value.as_str() { return Ok(AttrValue::Text(s.to_owned())); }
        let (k,raw)=encode_attr(&match variable {
            Some(dtype) if !value.is_object() => json!({"dtype":dtype,"values":value}),
            _ => value.clone() })?;
        macro_rules! words { ($t:ty) => { raw.chunks_exact(std::mem::size_of::<$t>()).map(|c| <$t>::from_ne_bytes(c.try_into().unwrap())).collect() } }
        Ok(match k { 1=>AttrValue::Bytes(words!(i8)), 3=>AttrValue::Shorts(words!(i16)), 4=>AttrValue::Ints(words!(i32)),
            5=>AttrValue::Floats(words!(f32)), 6=>AttrValue::Doubles(words!(f64)),
            _ => return Err(format!("compressed write: attribute primitive {} has no NETCDF4_CLASSIC type",dtype(k)?.0).into()) })
    }
    let wrap=|e: netcdf_writer::NcWriteError| -> Box<dyn std::error::Error> { format!("compressed write: {e}").into() };
    let mut schema=Schema::new(NcFormat::Offset64);
    let mut dim_ids=BTreeMap::new();
    for name in ordered_keys(&req.dimensions,&req.dimension_order)? {
        let unlimited=name=="Time";
        dim_ids.insert(name.clone(),schema.def_dim(name,if unlimited {0} else {req.dimensions[name]},unlimited).map_err(wrap)?);
    }
    for name in ordered_keys(&req.global_attrs,&req.global_attribute_order)? {
        if name=="_NCProperties" { continue; }
        schema.put_global_attr(name,attr(&req.global_attrs[name],None)?).map_err(wrap)?;
    }
    let mut ids=Vec::new();
    for v in &req.variables {
        let ty=nc_type(&v.dtype)?;
        let dims:Vec<_>=v.dimensions.iter().map(|d| dim_ids[d]).collect();
        let id=schema.def_var(&v.name,ty,&dims).map_err(wrap)?;
        for name in ordered_keys(&v.attrs,&v.attribute_order)? {
            if name=="_NCProperties" { continue; }
            // _FillValue takes exactly the variable's primitive type (as put_attrs writes it).
            let typed=(name=="_FillValue").then_some(v.dtype.as_str());
            schema.put_var_attr(id,name,attr(&v.attrs[name],typed)?).map_err(wrap)?;
        }
        ids.push((id,ty,v.dimensions.first().map(String::as_str)==Some("Time")));
    }
    let options=Nc4Options{deflate:Some(compression.level),shuffle:true,threads:compression.threads,..Nc4Options::default()};
    let mut writer=Nc4Writer::create(&req.path,schema,options).map_err(wrap)?;
    let mut bytes_written=0;
    let outcome=(|| -> Result<()> {
        for ((v,(id,ty,record)),data) in req.variables.iter().zip(&ids).zip(&prepared) {
            let count:Vec<_>=v.dimensions.iter().map(|d|req.dimensions[d]).collect();
            let size=product(&count)?*dtype(kind(&v.dtype)?)?.1;
            let raw=bytes(data,size);
            if *record {
                // The file's records are the Time extent of the request, each written whole.
                let records=req.dimensions.get("Time").copied().unwrap_or(1);
                if records==0 { continue; }
                let slab=size/records;
                for (r,part) in raw.chunks_exact(slab.max(1)).take(records).enumerate() {
                    writer.write_record(r as u64,*id,*ty,part).map_err(wrap)?;
                }
            } else { writer.write_var(*id,*ty,raw).map_err(wrap)?; }
            bytes_written+=size;
        }
        Ok(())
    })();
    if let Err(error)=outcome { writer.abort(); return Err(error); }
    drop(prepared);
    writer.finish().map_err(wrap)?;
    Ok(json!({"schema":"mpas-native-stream/v1","variables":req.variables.len(),"bytes_written":bytes_written,"ordered_metadata":true,
        "container":"netcdf4-classic shuffle+deflate","deflate_level":compression.level,
        "file_bytes":fs::metadata(&req.path)?.len(),"compression_threads":netcdf_writer::netcdf4_stream::pool_threads()}))
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:2430-3251: native primitive/layout semantics.
fn var_info(nc: &Nc,id:i32) -> Result<(i32,Vec<String>,Vec<usize>,i32)> {
    let mut k=0; let mut nd=0; let mut na=0;
    check(unsafe {nc_inq_var(nc.0,id,std::ptr::null_mut(),&mut k,&mut nd,std::ptr::null_mut(),&mut na)})?;
    let mut dims=vec![0;nd as usize]; check(unsafe {nc_inq_var(nc.0,id,std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),dims.as_mut_ptr(),std::ptr::null_mut())})?;
    let mut dimensions=Vec::new(); let mut shape=Vec::new();
    for d in dims {let mut name=[0;257];let mut len=0;check(unsafe {nc_inq_dim(nc.0,d,name.as_mut_ptr(),&mut len)})?;dimensions.push(text_name(&name));shape.push(len);}
    Ok((k,dimensions,shape,na))
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:4315-4363: typed native attributes.
// Header-only inspection keeps forecast verification independent of data dumps.
fn inspect(req: &Request) -> Result<Value> {
    let nc=Nc::open(&req.path,0)?;
    let mut format=0;check(unsafe {nc_inq_format(nc.0,&mut format)})?;
    let mut nd=0;let mut nv=0;let mut na=0;
    check(unsafe {nc_inq(nc.0,&mut nd,&mut nv,&mut na,std::ptr::null_mut())})?;
    let mut un=0;check(unsafe {nc_inq_unlimdims(nc.0,&mut un,std::ptr::null_mut())})?;
    let mut unlimited=vec![0;un as usize];
    check(unsafe {nc_inq_unlimdims(nc.0,&mut un,unlimited.as_mut_ptr())})?;
    let mut dimensions=BTreeMap::new();let mut unlimited_names=Vec::new();let mut dimension_order=Vec::new();
    for d in 0..nd {
        let mut name=[0;257];let mut len=0;
        check(unsafe {nc_inq_dim(nc.0,d,name.as_mut_ptr(),&mut len)})?;
        let name=text_name(&name);
        if unlimited.contains(&d) {unlimited_names.push(name.clone());}
        dimension_order.push(name.clone());
        dimensions.insert(name,len);
    }
    let mut variables=BTreeMap::new();let mut variable_order=Vec::new();
    for id in 0..nv {
        let mut name=[0;257];
        check(unsafe {nc_inq_var(nc.0,id,name.as_mut_ptr(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut())})?;
        let (k,dims,shape,attrs)=var_info(&nc,id)?;
        variable_order.push(text_name(&name));
        let (attribute_order,attribute_words)=attribute_layout(&nc,id,attrs)?;
        variables.insert(text_name(&name),json!({"dtype":dtype(k)?.0,
            "dimensions":dims,"shape":shape,"attrs":get_attrs(&nc,id,attrs)?,
            "attribute_order":attribute_order,"attribute_words":attribute_words}));
    }
    let attrs=get_attrs(&nc,-1,na)?;
    let (global_attribute_order,global_attribute_words)=attribute_layout(&nc,-1,na)?;
    nc.close()?;
    Ok(json!({"schema":"mpas-native-stream/v1","format":format,"dimensions":dimensions,
        "unlimited":unlimited_names,"global_attrs":attrs,"variables":variables,
        "dimension_order":dimension_order,"variable_order":variable_order,
        "global_attribute_order":global_attribute_order,"global_attribute_words":global_attribute_words}))
}
// MPAS v8.4.2 src/framework/mpas_io_streams.F:2430-3251: native primitive/layout semantics.
// One variable of a read: its raw words (permuted by its layout, if any) in
// `output_directory/var-{id}.raw`, and its description.
fn read_one(nc: &Nc, req: &Request, id: i32, name: String) -> Result<Variable> {
    let (k,dims,mut shape,na)=var_info(nc,id)?; let mut start=vec![0;shape.len()];
    if let Some(record)=req.record {if let Some(axis)=dims.iter().position(|d|d=="Time") {if record>=shape[axis] {return Err("requested record outside file".into());} start[axis]=record;shape[axis]=1;}}
    let size=product(&shape)?.checked_mul(dtype(k)?.1).ok_or("read byte count overflow")?;
    let mut buf=aligned(size);check(unsafe {nc_get_vara(nc.0,id,start.as_ptr(),shape.as_ptr(),buf.as_mut_ptr().cast())})?;
    let filename=req.output_directory.join(format!("var-{id}.raw"));fs::write(&filename,bytes(&buf,size))?;
    let texts=if k==2 && dims.last().map(String::as_str)==Some("StrLen") {
        let width=*shape.last().ok_or("missing text width")?;
        if width==0 {return Err("zero-width native text".into());}
        bytes(&buf,size).chunks_exact(width).map(|s|String::from_utf8_lossy(s).trim_end_matches([' ','\0']).to_owned()).collect::<Vec<_>>()
    } else {Vec::new()};
    let attrs=get_attrs(nc,id,na)?;
    let (attribute_order,_)=attribute_layout(nc,id,na)?;
    if let Some(layout)=req.layouts.get(&name) {
        if dims!=layout.expected_dimensions {return Err(format!("{name}: native read dimension order mismatch").into());}
        let mut input_dimensions=dims.clone();let mut input_shape=shape.clone();
        if !layout.dimensions.iter().any(|d|d=="Time") {
            if let Some(axis)=input_dimensions.iter().position(|d|d=="Time") {
                if input_shape[axis]!=1 {return Err("authority read requires exactly one Time record".into());}
                input_dimensions.remove(axis);input_shape.remove(axis);
            }
        }
        let sizes:BTreeMap<_,_>=input_dimensions.iter().cloned().zip(input_shape.iter().copied()).collect();
        let v=Variable{name:name.clone(),dtype:layout.dtype.clone(),dimensions:layout.dimensions.clone(),
            shape:input_shape,input_dimensions,filename:filename.clone(),attrs:attrs.clone(),attribute_order,text:Vec::new(),
            connectivity_offset:layout.connectivity_offset,input_dtype:Some(dtype(k)?.0.into())};
        let data=prepare(&v,&sizes)?;
        let shape:Vec<_>=layout.dimensions.iter().map(|d|sizes[d]).collect();
        let count=product(&shape)?*dtype(kind(&layout.dtype)?)?.1;
        fs::write(&filename,bytes(&data,count))?;
        Ok(Variable{shape,input_dimensions:layout.dimensions.clone(),input_dtype:None,filename,..v})
    } else {
        Ok(Variable {name,dtype:dtype(k)?.0.into(),dimensions:dims.clone(),input_dimensions:dims,shape,filename,attrs,attribute_order,text:texts,connectivity_offset:0,input_dtype:None})
    }
}
// One worker's answer: each of its variables in order, stopping at the first
// that failed.
#[derive(Serialize,Deserialize)]
struct ShardItem { id: i32, variable: Option<Variable>, error: Option<String> }
// `rw_mpas_stream --read-shard REQUEST.json ID,ID,...`: read_one for those
// variables, in order, as JSON on stdout. The netCDF-C library is not
// thread-safe, so a read is shared out over processes, not threads.
fn read_shard(request: &Path, ids: &str) -> Result<()> {
    let req:Request=serde_json::from_slice(&fs::read(request)?)?;
    let nc=Nc::open(&req.path,0)?;
    let mut items=Vec::new();
    for id in ids.split(',').filter(|t|!t.is_empty()) {
        let id:i32=id.parse()?;
        let mut name=[0;257];
        check(unsafe {nc_inq_var(nc.0,id,name.as_mut_ptr(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut())})?;
        match read_one(&nc,&req,id,text_name(&name)) {
            Ok(v) => items.push(ShardItem{id,variable:Some(v),error:None}),
            Err(e) => {items.push(ShardItem{id,variable:None,error:Some(e.to_string())});break;}
        }
    }
    nc.close()?;
    println!("{}",serde_json::to_string(&items)?);Ok(())
}
// Every selected variable, in id order. Large reads go to worker processes,
// each opening the file itself; the variables are dealt out by size and put
// back in id order, and the first failure in id order is the one reported, so
// the response and the files are the serial reader's.
fn read_variables(nc: &Nc, req: &Request, request_path: &Path, wanted: Vec<(i32,String,usize)>) -> Result<Vec<Variable>> {
    let total: usize = wanted.iter().map(|w| w.2).sum();
    let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        .min(16).min(wanted.len());
    if workers < 2 || total < (64 << 20) {
        return wanted.into_iter().map(|(id,name,_)| read_one(nc,req,id,name)).collect();
    }
    let mut order: Vec<usize> = (0..wanted.len()).collect();
    order.sort_by(|a,b| wanted[*b].2.cmp(&wanted[*a].2).then(a.cmp(b)));
    let mut shards: Vec<(usize,Vec<usize>)> = vec![(0,Vec::new()); workers];
    for i in order {
        let lightest = (0..workers).min_by_key(|s| (shards[*s].0, *s)).expect("workers > 0");
        shards[lightest].0 += wanted[i].2; shards[lightest].1.push(i);
    }
    let exe = std::env::current_exe()?;
    let mut children = Vec::new();
    for (_, mut members) in shards.into_iter().filter(|s| !s.1.is_empty()) {
        members.sort();
        let ids = members.iter().map(|i| wanted[*i].0.to_string()).collect::<Vec<_>>().join(",");
        children.push(std::process::Command::new(&exe).arg("--read-shard").arg(request_path).arg(ids)
            .stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn()?);
    }
    let mut got: BTreeMap<i32,ShardItem> = BTreeMap::new();
    let mut crashed: Option<String> = None;
    for child in children {
        let out = child.wait_with_output()?;
        if !out.status.success() {
            let text = String::from_utf8_lossy(&out.stderr).trim().to_owned();
            crashed.get_or_insert(text.strip_prefix("rw_mpas_stream: ").unwrap_or(&text).to_owned());
            continue;
        }
        for item in serde_json::from_slice::<Vec<ShardItem>>(&out.stdout)? { got.insert(item.id,item); }
    }
    let mut variables = Vec::with_capacity(wanted.len());
    for (id,_,_) in &wanted {
        match got.remove(id) {
            Some(ShardItem{variable:Some(v),..}) => variables.push(v),
            Some(ShardItem{error:Some(e),..}) => return Err(e.into()),
            _ => return Err(crashed.unwrap_or_else(|| format!("variable {id} was not read")).into()),
        }
    }
    Ok(variables)
}
fn read(req: &Request, request_path: &Path) -> Result<Value> {
    let nc=Nc::open(&req.path,0)?;
    let mut nd=0;let mut nv=0;let mut na=0;
    check(unsafe {nc_inq(nc.0,&mut nd,&mut nv,&mut na,std::ptr::null_mut())})?;
    let mut un=0;check(unsafe {nc_inq_unlimdims(nc.0,&mut un,std::ptr::null_mut())})?;
    let mut unlimited=vec![0;un as usize];check(unsafe {nc_inq_unlimdims(nc.0,&mut un,unlimited.as_mut_ptr())})?;
    let mut dimensions=BTreeMap::new();let mut unlimited_names=Vec::new();let mut dimension_order=Vec::new();
    for d in 0..nd {let mut name=[0;257];let mut len=0;check(unsafe {nc_inq_dim(nc.0,d,name.as_mut_ptr(),&mut len)})?;let name=text_name(&name);if unlimited.contains(&d) {unlimited_names.push(name.clone());} dimension_order.push(name.clone());dimensions.insert(name,len);}
    let selected:std::collections::BTreeSet<_>=req.select.iter().cloned().collect();
    let mut found=std::collections::BTreeSet::new();let mut wanted=Vec::new();let mut variable_order=Vec::new();
    fs::create_dir_all(&req.output_directory)?;
    for id in 0..nv {
        let mut name=[0;257];check(unsafe {nc_inq_var(nc.0,id,name.as_mut_ptr(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null_mut())})?;
        let name=text_name(&name);
        if !selected.is_empty() && !selected.contains(&name) {continue;}
        found.insert(name.clone());
        variable_order.push(name.clone());
        // The bytes this variable reads, to share the work out by size.
        let (k,_,shape,_)=var_info(&nc,id)?;
        let words=shape.iter().fold(1usize,|a,b| a.saturating_mul(*b));
        wanted.push((id,name,words.saturating_mul(dtype(k).map(|d|d.1).unwrap_or(1))));
    }
    let variables=read_variables(&nc,req,request_path,wanted)?;
    if !selected.is_subset(&found) {return Err(format!("missing requested fields: {:?}",selected.difference(&found).collect::<Vec<_>>()).into());}
    if req.layouts.keys().any(|n| !found.contains(n)) {return Err("missing requested authority layout field".into());}
    let attrs=get_attrs(&nc,-1,na)?;
    let (global_attribute_order,global_attribute_words)=attribute_layout(&nc,-1,na)?;
    nc.close()?;
    Ok(json!({"schema":"mpas-native-stream/v1","dimensions":dimensions,"unlimited":unlimited_names,"global_attrs":attrs,"variables":variables,
        "dimension_order":dimension_order,"variable_order":variable_order,
        "global_attribute_order":global_attribute_order,"global_attribute_words":global_attribute_words}))
}
fn main() {
    let _stamp = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    let outcome=(|| -> Result<()> {
        let args:Vec<_>=std::env::args_os().collect();
        if args.len()==2 && args[1]=="--abi" {println!("mpas-native-stream/v1 raw-primitives native-netcdf read write permute connectivity compression");return Ok(());}
        if args.len()==4 && args[1]=="--read-shard" {return read_shard(Path::new(&args[2]),&args[3].to_string_lossy());}
        if args.len()!=2 {return Err("usage: rw_mpas_stream REQUEST.json | --abi".into());}
        let req:Request=serde_json::from_slice(&fs::read(&args[1])?)?;
        let response=match req.operation.as_str() {"write"=>write(&req)?,"read"=>read(&req,Path::new(&args[1]))?,"inspect"=>inspect(&req)?,_=>return Err("unknown operation".into())};
        println!("{}",serde_json::to_string(&response)?);Ok(())
    })();
    if let Err(error)=outcome {eprintln!("rw_mpas_stream: {error}");std::process::exit(1);}
}

#[cfg(test)]
mod tests {
    use super::*;

    // netCDF-C is not thread-safe and the test harness runs tests on threads:
    // every test that opens a file holds this.
    static NETCDF: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn netcdf() -> std::sync::MutexGuard<'static,()> { NETCDF.lock().unwrap_or_else(|e| e.into_inner()) }

    #[test]
    fn frame_round_trip_uses_rust_reader_and_preserves_words_and_header() -> Result<()> {
        let _netcdf=netcdf();
        let base=std::env::var_os("RW_MPAS_STREAM_TEST_DIRECTORY")
            .map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let root=base.join(format!("mpas-stream-roundtrip-{}-{nonce}",std::process::id()));
        fs::create_dir_all(&root)?;
        let words=[0x80000000u32,0x00000001,0x7fc12345,0x3f800000,0xff800000,0x7f7fffff];
        let raw:Vec<u8>=words.iter().flat_map(|v|v.to_ne_bytes()).collect();
        let handoff=root.join("theta.raw");fs::write(&handoff,&raw)?;
        let path=root.join("frame.nc");
        let request:Request=serde_json::from_value(json!({
            "operation":"write","path":path,"format":"pnetcdf","header_alignment":131072,
            "dimensions":{"nCells":3,"nVertLevels":2,"Time":1,"StrLen":64},
            "dimension_order":["nVertLevels","nCells","Time","StrLen"],
            "global_attrs":{"source":"MPAS","sphere_radius":{"dtype":"f4","values":6371220.0}},
            "global_attribute_order":["source","sphere_radius"],
            "variables":[{"name":"theta","dtype":"f4",
                "dimensions":["Time","nCells","nVertLevels"],
                "input_dimensions":["nVertLevels","nCells"],"shape":[2,3],"filename":handoff,
                "attrs":{"units":"K","long_name":"potential temperature"},
                "attribute_order":["units","long_name"]},
                {"name":"xtime","dtype":"S1","dimensions":["Time","StrLen"],
                 "text":["2023-03-31_12:00:00"]}]}))?;
        write(&request)?;
        let inspect_request:Request=serde_json::from_value(json!({"operation":"inspect","path":path}))?;
        let header=inspect(&inspect_request)?;
        assert_eq!(header["dimension_order"],json!(["nVertLevels","nCells","Time","StrLen"]));
        assert_eq!(header["variable_order"],json!(["theta","xtime"]));
        assert!(fs::metadata(&path)?.len()>=131072);
        assert_eq!(header["variables"]["theta"]["attribute_order"],json!(["units","long_name"]));
        assert_eq!(header["global_attribute_words"]["sphere_radius"]["dtype"],"f4");
        let read_request:Request=serde_json::from_value(json!({
            "operation":"read","path":path,"output_directory":root.join("decoded"),"record":0,
            "layouts":{"theta":{"dtype":"f4","dimensions":["nVertLevels","nCells"],
                "expected_dimensions":["Time","nCells","nVertLevels"]}}}))?;
        let decoded=read(&read_request,Path::new("unused-serial-test-request"))?;
        assert_eq!(fs::read(decoded["variables"][0]["filename"].as_str().unwrap())?,raw);
        assert_eq!(decoded["variables"][1]["text"],json!(["2023-03-31_12:00:00"]));
        assert_eq!(decoded["variables"][0]["attribute_order"],json!(["units","long_name"]));
        // Keep test processes self-contained. Runtime proofs supply their own
        // scratch root and retain hashes separately before disposing outputs.
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    // The writer skips netCDF-C's fill only where the puts overwrite every filled
    // byte: a new classic file, written whole, is the fill-mode file byte for byte,
    // in every classic format, with fixed and record variables and a widened input.
    // An append that leaves a record variable of the file unwritten keeps the fill.
    #[test]
    fn history_write_without_fill_is_the_fill_mode_file_byte_for_byte() -> Result<()> {
        let _netcdf=netcdf();
        let base=std::env::var_os("RW_MPAS_STREAM_TEST_DIRECTORY")
            .map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let root=base.join(format!("mpas-stream-nofill-{}-{nonce}",std::process::id()));
        fs::create_dir_all(&root)?;
        let (nc,nl)=(2_000usize,5usize);
        let theta:Vec<u8>=(0..nc*nl).flat_map(|i| (280.0f32+i as f32*0.25).to_ne_bytes()).collect();
        let w:Vec<u8>=(0..nc*(nl+1)).flat_map(|i| (i as f32*-0.5).to_ne_bytes()).collect();
        let ids:Vec<u8>=(0..nc as i64).flat_map(|i| (i+1).to_ne_bytes()).collect();
        for (name,bytes) in [("theta",&theta),("w",&w),("ids",&ids)] {fs::write(root.join(format!("{name}.raw")),bytes)?;}
        let request=|path: &Path, format: &str, append: bool, record: usize, with_w: bool| -> Result<Request> {
            let mut variables=vec![
                json!({"name":"xtime","dtype":"S1","dimensions":["Time","StrLen"],"text":["2023-03-31_13:00:00"]}),
                json!({"name":"theta","dtype":"f4","dimensions":["Time","nCells","nVertLevels"],
                    "input_dimensions":["nVertLevels","nCells"],"shape":[nl,nc],"filename":root.join("theta.raw"),
                    "attrs":{"units":"K"},"attribute_order":["units"]}),
                json!({"name":"indexToCellID","dtype":"i4","input_dtype":"i8","dimensions":["nCells"],
                    "input_dimensions":["nCells"],"shape":[nc],"filename":root.join("ids.raw")})];
            if with_w {variables.push(json!({"name":"w","dtype":"f4","dimensions":["Time","nCells","nVertLevelsP1"],
                "input_dimensions":["nCells","nVertLevelsP1"],"shape":[nc,nl+1],"filename":root.join("w.raw")}));}
            Ok(serde_json::from_value(json!({"operation":"write","path":path,"format":format,"header_alignment":131072,
                "dimensions":{"Time":1,"nCells":nc,"nVertLevels":nl,"nVertLevelsP1":nl+1,"StrLen":64},
                "dimension_order":["Time","nCells","nVertLevels","nVertLevelsP1","StrLen"],
                "global_attrs":{"model_name":"mpas"},"global_attribute_order":["model_name"],
                "variables":variables,"append":append,"record_index":record}))?)
        };
        for format in ["pnetcdf,cdf5","pnetcdf","netcdf"] {
            let tag=format.replace(',',"-");
            let (fast,filled)=(root.join(format!("fast-{tag}.nc")),root.join(format!("filled-{tag}.nc")));
            write(&request(&fast,format,false,0,true)?)?;
            KEEP_FILL.with(|k| k.set(true));
            write(&request(&filled,format,false,0,true)?)?;
            KEEP_FILL.with(|k| k.set(false));
            assert_eq!(fs::read(&fast)?,fs::read(&filled)?,"{format}: new file");
            // A second record appended whole: still the fill-mode bytes.
            write(&request(&fast,format,true,1,true)?)?;
            KEEP_FILL.with(|k| k.set(true));
            write(&request(&filled,format,true,1,true)?)?;
            KEEP_FILL.with(|k| k.set(false));
            assert_eq!(fs::read(&fast)?,fs::read(&filled)?,"{format}: appended record");
        }
        // An append that leaves w unwritten, or skips a record, must keep the fill.
        let path=root.join("fast-pnetcdf-cdf5.nc");
        let handle=Nc::open_for_write(&path)?;
        assert!(puts_cover_the_fill(&handle,&request(&path,"pnetcdf,cdf5",true,2,true)?)?);
        assert!(!puts_cover_the_fill(&handle,&request(&path,"pnetcdf,cdf5",true,2,false)?)?);
        assert!(!puts_cover_the_fill(&handle,&request(&path,"pnetcdf,cdf5",true,3,true)?)?);
        handle.close()?;
        let gap=Nc::create(&root.join("gap.nc"),"pnetcdf,cdf5")?;
        assert!(!puts_cover_the_fill(&gap,&request(&path,"pnetcdf,cdf5",false,1,true)?)?);
        // A short variable of odd length has fill-valued padding: keep the fill.
        let mut odd=request(&path,"pnetcdf,cdf5",false,0,true)?;
        odd.variables[2].dtype="i2".into(); odd.dimensions.insert("nCells".into(),nc+1);
        assert!(!puts_cover_the_fill(&gap,&odd)?);
        gap.close()?;
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    // Variables prepare side by side, and a failure is still the serial loop's:
    // the first failing variable in request order, then a repeated name.
    #[test]
    fn side_by_side_prepare_keeps_the_serial_bytes_and_error_order() -> Result<()> {
        let base=std::env::var_os("RW_MPAS_STREAM_TEST_DIRECTORY")
            .map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let root=base.join(format!("mpas-stream-prepare-{}-{nonce}",std::process::id()));
        fs::create_dir_all(&root)?;
        let n=300_000usize;
        let variable=|i: usize, words: usize| -> Result<Value> {
            let bytes:Vec<u8>=(0..words).flat_map(|k| ((k*31+i) as f32).to_ne_bytes()).collect();
            let file=root.join(format!("v{i}.raw")); fs::write(&file,bytes)?;
            Ok(json!({"name":format!("v{i}"),"dtype":"f4","dimensions":["Time","nCells","nVertLevels"],
                "input_dimensions":["nVertLevels","nCells"],"shape":[4,n],"filename":file}))
        };
        let mut variables=Vec::new();
        for i in 0..12 {variables.push(variable(i,4*n)?);}
        let make=|variables: &Vec<Value>| -> Result<Request> {Ok(serde_json::from_value(json!({"operation":"write",
            "path":root.join("unused.nc"),"dimensions":{"Time":1,"nCells":n,"nVertLevels":4},"variables":variables}))?)};
        let req=make(&variables)?;
        let together=prepare_all(&req)?;
        for (v,data) in req.variables.iter().zip(&together) {assert!(prepare(v,&req.dimensions)? == *data);}
        // v3 and v9 have the wrong byte count and v10 repeats v2's name: v3 is reported.
        let mut broken=variables.clone();
        broken[3]=variable(3,4*n-1)?; broken[9]=variable(9,4*n-1)?; broken[10]["name"]=json!("v2");
        let serial_error=prepare(&make(&broken)?.variables[3],&req.dimensions).unwrap_err().to_string();
        assert_eq!(prepare_all(&make(&broken)?).unwrap_err().to_string(),serial_error);
        broken[3]=variable(3,4*n)?; broken[9]=variable(9,4*n)?;
        assert_eq!(prepare_all(&make(&broken)?).unwrap_err().to_string(),"duplicate variable");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn compressed_frame_round_trip_preserves_words_order_and_attributes() -> Result<()> {
        // A history frame written lossless-compressed (hexcore asks for it on history and diagnostics files)
        // reads back through netCDF-C with the uncompressed frame's every word, dimension, variable and attribute.
        let base=std::env::var_os("RW_MPAS_STREAM_TEST_DIRECTORY").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
        let root=base.join(format!("mpas-stream-compressed-{}-{nonce}",std::process::id()));
        fs::create_dir_all(&root)?;
        let (nl,nc)=(55usize,3000usize);
        let words:Vec<u32>=(0..nl*nc).map(|i| match i%7 {0=>0x80000000,1=>0x7fc12345,2=>0x00000001,_=>(i as u32).wrapping_mul(2654435761)}).collect();
        let raw:Vec<u8>=words.iter().flat_map(|v|v.to_ne_bytes()).collect();
        let handoff=root.join("theta.raw");fs::write(&handoff,&raw)?;
        let request=|path:&Path,compression:Value| -> Result<Request> { Ok(serde_json::from_value(json!({
            "operation":"write","path":path,"format":"pnetcdf,cdf5","header_alignment":0,"compression":compression,
            "dimensions":{"nCells":nc,"nVertLevels":nl,"Time":1,"StrLen":64},
            "dimension_order":["nVertLevels","nCells","Time","StrLen"],
            "global_attrs":{"source":"MPAS","sphere_radius":{"dtype":"f4","values":6371220.0},"on_a_sphere":"YES"},
            "global_attribute_order":["source","sphere_radius","on_a_sphere"],
            "variables":[{"name":"theta","dtype":"f4",
                "dimensions":["Time","nCells","nVertLevels"],
                "input_dimensions":["nVertLevels","nCells"],"shape":[nl,nc],"filename":handoff,
                "attrs":{"units":"K","long_name":"potential temperature","_FillValue":-9999.0},
                "attribute_order":["units","long_name","_FillValue"]},
                {"name":"xtime","dtype":"S1","dimensions":["Time","StrLen"],
                 "text":["2023-03-31_12:00:00"]}]}))?) };
        let plain=root.join("plain.nc"); let packed=root.join("packed.nc");
        write(&request(&plain,Value::Null)?)?;
        let response=write(&request(&packed,json!({"level":4}))?)?;
        assert_eq!(response["container"],"netcdf4-classic shuffle+deflate");
        assert!(fs::metadata(&packed)?.len()<fs::metadata(&plain)?.len());
        let header=|path:&Path| -> Result<Value> { inspect(&serde_json::from_value(json!({"operation":"inspect","path":path}))?) };
        let (a,b)=(header(&plain)?,header(&packed)?);
        assert_eq!(b["format"],json!(4), "NC_FORMAT_NETCDF4_CLASSIC");
        for key in ["dimension_order","variable_order","dimensions","unlimited","global_attribute_order","global_attribute_words"] {
            assert_eq!(a[key],b[key],"{key}");
        }
        for key in ["dtype","dimensions","shape","attribute_order","attribute_words"] {
            assert_eq!(a["variables"]["theta"][key],b["variables"]["theta"][key],"theta.{key}");
        }
        let read_back=|path:&Path,name:&str| -> Result<Value> { read(&serde_json::from_value(json!({
            "operation":"read","path":path,"output_directory":root.join(name),"record":0}))?,Path::new("unused-serial-test-request")) };
        let (x,y)=(read_back(&plain,"plain")?,read_back(&packed,"packed")?);
        assert_eq!(fs::read(x["variables"][0]["filename"].as_str().unwrap())?,fs::read(y["variables"][0]["filename"].as_str().unwrap())?);
        assert_eq!(y["variables"][1]["text"],json!(["2023-03-31_12:00:00"]));
        // A compressed file is written whole: no append, no record past the first.
        let mut append=request(&packed,json!({"level":4}))?; append.append=true;
        assert!(write(&append).unwrap_err().to_string().contains("compressed write"));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn explicit_order_rejects_missing_duplicate_or_unknown_metadata() {
        let map=BTreeMap::from([("one".to_owned(),1),("two".to_owned(),2)]);
        for order in [vec!["one"],vec!["one","one"],vec!["one","other"]] {
            assert!(ordered_keys(&map,&order.into_iter().map(str::to_owned).collect::<Vec<_>>()).is_err());
        }
    }
}

#[cfg(test)]
mod tests_permute {
    use super::*;

    // The serial per-word loop this replaced, kept as the referee.
    fn serial(raw: &[u8], out_shape: &[usize], dims: &[&str], in_dims: &[&str], in_shape: &[usize], conv: Conv,
              isz: usize, osz: usize) -> std::result::Result<Vec<u8>,String> {
        let count: usize = out_shape.iter().product();
        let mut target = vec![0u8; count*osz];
        for flat in 0..count {
            let mut residual=flat; let mut coords=vec![0;out_shape.len()];
            for axis in (0..coords.len()).rev() { coords[axis]=residual%out_shape[axis]; residual/=out_shape[axis]; }
            let mut input_index=0;
            for (d,n) in in_dims.iter().zip(in_shape) { input_index=input_index*n+coords[dims.iter().position(|a| a==d).unwrap()]; }
            convert(conv,&raw[input_index*isz..(input_index+1)*isz],&mut target[flat*osz..(flat+1)*osz])?;
        }
        Ok(target)
    }

    fn strides(dims: &[&str], in_dims: &[&str], in_shape: &[usize]) -> Vec<usize> {
        dims.iter().map(|d| in_dims.iter().position(|a| a==d).map(|p| in_shape[p+1..].iter().product()).unwrap_or(0)).collect()
    }

    #[test]
    fn threaded_permutation_matches_the_serial_loop() {
        // (Time, nCells, nVertLevels) from (nVertLevels, nCells), large enough to split over workers.
        let (nl, nc) = (55usize, 40_000usize);
        let raw: Vec<u8> = (0..nl*nc*4).map(|i| (i*2654435761usize >> 7) as u8).collect();
        let dims = ["Time","nCells","nVertLevels"]; let in_dims = ["nVertLevels","nCells"]; let in_shape = [nl,nc];
        let shape = [1,nc,nl];
        let mut fast = vec![0u8; nl*nc*4];
        permute(&raw,&mut fast,&shape,&strides(&dims,&in_dims,&in_shape),4,4,Conv::Copy).unwrap();
        assert_eq!(fast, serial(&raw,&shape,&dims,&in_dims,&in_shape,Conv::Copy,4,4).unwrap());
        // Widening to double, and a scalar.
        let mut wide = vec![0u8; nl*nc*8];
        permute(&raw,&mut wide,&shape,&strides(&dims,&in_dims,&in_shape),4,8,Conv::F32ToF64).unwrap();
        assert_eq!(wide, serial(&raw,&shape,&dims,&in_dims,&in_shape,Conv::F32ToF64,4,8).unwrap());
        let mut one = vec![0u8; 4];
        permute(&raw[..4],&mut one,&[],&[],4,4,Conv::Copy).unwrap();
        assert_eq!(one, raw[..4]);
    }

    #[test]
    fn first_failing_word_in_output_order_is_reported() {
        // Two out-of-range connectivity words; the earlier one in output order names the error.
        let n = 2_000_000usize;
        let mut values = vec![1i64; n]; values[n-10] = i64::MAX; values[1_500_000] = i64::from(i32::MAX);
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut out = vec![0u8; n*4];
        let fast = permute(&raw,&mut out,&[n],&[1],8,4,Conv::Connectivity{wide:true,offset:1}).unwrap_err();
        let slow = serial(&raw,&[n],&["nCells"],&["nCells"],&[n],Conv::Connectivity{wide:true,offset:1},8,4).unwrap_err();
        assert_eq!(fast, slow);
    }
}
