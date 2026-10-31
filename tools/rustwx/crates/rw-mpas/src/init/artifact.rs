//! Rust container codec for the front door's constructed vertical contract.
//! Native contract: MPAS-A v8.4.2 core_init_atmosphere/Registry.xml
//! vertical_stage_out and met_stage_out. This module performs container I/O;
//! its input values come from the separately cited numerical ports.
use std::{collections::{BTreeMap, BTreeSet}, io::{Read, Write}, path::PathBuf};
use serde::{Deserialize, Serialize};
use rw_store::error::RwStoreError;
use rw_store::netcdf_classic::{NcAttr, NcAttrValue, NcClassicWriter, NcData, NcDim, NcFormat, NcOwned, NcSlab, NcType, NcVarDef};
use crate::error::{MpasError, MpasResult};

#[derive(Deserialize)]
pub struct Request {
    pub template: PathBuf,
    /// Typed producer outputs, merged in order before computed declarations.
    /// Shared dimensions and replaced field types must agree with the template.
    #[serde(default)] pub overlays: Vec<PathBuf>,
    pub output: PathBuf,
    /// None denotes the single unlimited record dimension.
    pub dimensions: BTreeMap<String, Option<usize>>,
    pub variables: Vec<Variable>,
    #[serde(default)] pub attributes: BTreeMap<String, serde_json::Value>,
}
#[derive(Deserialize)]
pub struct Variable {
    pub name: String,
    pub dimensions: Vec<String>,
    #[serde(default)] pub attributes: BTreeMap<String, serde_json::Value>,
    #[serde(default)] pub values: Option<Vec<f32>>,
    #[serde(default)] pub text: Option<String>,
    /// Preserve an existing field, including attributes, if any word is nonzero.
    #[serde(default)] pub only_if_zero: bool,
    /// Add a declaration without overwriting an existing payload.
    #[serde(default)] pub only_if_absent: bool,
}
#[derive(Default, Serialize)]
pub struct Receipt {
    pub schema: &'static str,
    pub declared: Vec<String>,
    pub computed: Vec<String>,
    pub carried: Vec<String>,
    pub overlays: Vec<PathBuf>,
}
fn refusal(message: impl Into<String>) -> MpasError {MpasError::Refusal(message.into())}
/// Stored NC_CHAR bytes, including NULs and padding. The facade's string
/// reader trims labels, so use the same vendored classic decoder directly.
fn char_words(path: &std::path::Path, name: &str) -> MpasResult<Vec<u8>> {
    use netcdf_reader::{NcFormat, classic::{header::parse_header, data::{compute_record_stride,read_record_variable,read_non_record_variable}}};
    // Mapped, not read: std::fs::read pulled the whole template (32 GB on RRFS
    // v2 conus3km) into memory once per character field, on one core.
    let file=std::fs::File::open(path)?;
    // SAFETY: the producer inputs are complete files that no stage rewrites while this one runs.
    let bytes=unsafe {memmap2::Mmap::map(&file)?};
    let format=match bytes.get(..4) {Some(b"CDF\x01")=>NcFormat::Classic,Some(b"CDF\x02")=>NcFormat::Offset64,Some(b"CDF\x05")=>NcFormat::Cdf5,_=>{
        return Ok(netcrust::File::open(path)?.read_array_f64(name)?.into_values().into_iter().map(|v|v as u8).collect());
    }};
    let error=|e:netcdf_reader::Error|MpasError::NetCdf(e.to_string());
    let header=parse_header(&bytes,format).map_err(error)?;
    let variable=header.variables.iter().find(|v|v.name==name).ok_or_else(||refusal(format!("missing character field {name}")))?;
    let data=if variable.dimensions.iter().any(|d| d.is_unlimited) {read_record_variable::<u8>(&bytes,variable,header.numrecs,compute_record_stride(&header.variables).map_err(error)?)}
        else {read_non_record_variable::<u8>(&bytes,variable)}.map_err(error)?;
    Ok(data.into_iter().collect())
}
fn json_attr(name: &str, value: &serde_json::Value) -> MpasResult<NcAttr> {
    let value=match value {
        serde_json::Value::Bool(v)=>NcAttrValue::Text(if *v {"YES"} else {"NO"}.into()),
        serde_json::Value::String(v)=>NcAttrValue::Text(v.clone()),
        serde_json::Value::Number(v) if v.is_i64()=>NcAttrValue::Ints(vec![i32::try_from(v.as_i64().unwrap())
            .map_err(|_|refusal(format!("attribute {name} exceeds the native integer range")))?]),
        serde_json::Value::Number(v)=>NcAttrValue::Floats(vec![v.as_f64().unwrap() as f32]),
        _=>return Err(refusal(format!("attribute {name} must be a scalar"))),
    };Ok(NcAttr {name:name.into(),value})
}
fn attrs(source: &[netcrust::Attribute], overrides: &BTreeMap<String, serde_json::Value>) -> MpasResult<Vec<NcAttr>> {
    let mut result=Vec::new();let mut seen=BTreeSet::new();
    for attr in source {
        seen.insert(attr.name().to_string());
        result.push(if let Some(value)=overrides.get(attr.name()) {json_attr(attr.name(),value)?}
            else {super::emit::attr_of(attr).ok_or_else(||refusal(format!("attribute {} has an unsupported type; dropping it would change the artifact contract",attr.name())))?});
    }
    for (name,value) in overrides {if !seen.contains(name) {result.push(json_attr(name,value)?);}}
    Ok(result)
}
pub fn run(input: impl Read, output: impl Write) -> MpasResult<()> {
    let request:Request=serde_json::from_reader(input).map_err(|e|refusal(e.to_string()))?;
    let receipt=materialize(request)?;
    serde_json::to_writer(output,&receipt).map_err(|e|refusal(e.to_string()))
}
pub fn materialize(request: Request) -> MpasResult<Receipt> {
    materialize_selected(request, None)
}

/// Emit only fields in a resolved native stream, dropping dimensions used only
/// by fields outside that stream. The ordinary artifact path carries all fields.
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1105-1149 at 1d6e370e.
pub(crate) fn materialize_selected(request: Request, selected: Option<&BTreeSet<String>>) -> MpasResult<Receipt> {
    let paths:Vec<_>=std::iter::once(request.template.clone()).chain(request.overlays.iter().cloned()).collect();
    if paths.contains(&request.output) {return Err(refusal("artifact output must differ from every producer input"));}
    let files=paths.iter().map(netcrust::File::open).collect::<Result<Vec<_>,_>>()?;
    // Preserve producer provenance, including native-undefined word witnesses.
    // Template attributes retain precedence; explicit request values override.
    let mut global_attributes=Vec::new();let mut global_names=BTreeSet::new();
    for source in &files {for attribute in source.attributes()? {
        if global_names.insert(attribute.name().to_string()) {global_attributes.push(attribute);}
    }}
    let selected_dims=if let Some(selected)=selected {
        let mut names=BTreeSet::new();
        for source in &files {for variable in source.variables()? {
            if selected.contains(variable.name()) {
                names.extend(variable.dimensions().iter().map(|d|d.name().to_string()));
            }
        }
        }
        for variable in &request.variables {
            if !selected.contains(&variable.name) {return Err(refusal(format!("computed field {} is outside the resolved output stream",variable.name)));}
            names.extend(variable.dimensions.iter().cloned());
        }
        Some(names)
    } else {None};
    let mut dims=Vec::new();let mut dimids=BTreeMap::new();
    for source in &files {for dim in source.dimensions()? {
        if selected_dims.as_ref().is_some_and(|names|!names.contains(dim.name())) {continue;}

        if let Some(required)=request.dimensions.get(dim.name()) {
            if dim.is_unlimited()!=required.is_none() || required.is_some_and(|n|n!=dim.len()) {
                return Err(refusal(format!("template dimension {} does not match the constructed contract",dim.name())));
            }
        }
        if dim.is_unlimited() && dim.len()>1 {return Err(refusal("an init artifact accepts one time record; truncating multiple records would discard data"));}
        if let Some(&id)=dimids.get(dim.name()) {
            let existing:&NcDim=&dims[id];
            if existing.unlimited!=dim.is_unlimited() || (!existing.unlimited && existing.len!=dim.len()) {
                return Err(refusal(format!("producer dimension {} differs from the artifact contract",dim.name())));
            }
            continue;
        }
        dimids.insert(dim.name().to_string(),dims.len());
        dims.push(if dim.is_unlimited() {NcDim::record(dim.name())} else {NcDim::fixed(dim.name(),dim.len())});
    }}
    for (name,size) in &request.dimensions {
        if !dimids.contains_key(name) {dimids.insert(name.clone(),dims.len());dims.push(match size {Some(n)=>NcDim::fixed(name,*n),None=>NcDim::record(name)});}
    }
    let mut requested=BTreeMap::new();
    for variable in &request.variables {
        if requested.insert(variable.name.clone(),variable).is_some() {return Err(refusal(format!("duplicate requested field {}",variable.name)));}
    }
    let mut definitions=Vec::new();let mut replacements=BTreeSet::new();
    let mut receipt=Receipt {schema:"hex-init-artifact-v1",overlays:request.overlays.clone(),..Default::default()};
    let mut present=BTreeSet::new();
    let mut owners=BTreeMap::new();let mut positions=BTreeMap::new();
    for (owner,source) in files.iter().enumerate() {for variable in source.variables()? {
        if selected.is_some_and(|names|!names.contains(variable.name())) {continue;}

        let name=variable.name().to_string();present.insert(name.clone());
        let ids:Vec<usize>=variable.dimensions().iter().map(|d|dimids[d.name()]).collect();
        let mut attributes=attrs(variable.attributes(),&BTreeMap::new())?;
        let ty=super::emit::nc_type_of(variable.dtype())?;
        if let Some(&position)=positions.get(&name) {
            let prior:&NcVarDef=&definitions[position];
            if prior.ty!=ty || prior.dimids!=ids {return Err(refusal(format!("overlay field {name} type or dimensions differ from its existing declaration")));}
        }
        if let Some(required)=requested.get(&name) {
            let names:Vec<_>=variable.dimensions().iter().map(|d|d.name().to_string()).collect();
            if names!=required.dimensions {return Err(refusal(format!("field {name} dimensions differ from the constructed contract")));}
            let nonzero=required.only_if_zero && source.read_array_f64(&name)?.values().iter().any(|v|*v!=0.);
            replacements.remove(&name);
            if !required.only_if_absent && !nonzero {
                let expected=if required.text.is_some() {NcType::Char} else {NcType::Float};
                if ty!=expected {return Err(refusal(format!("field {name} type {ty:?} does not match its computed payload {expected:?}")));}
                attributes=attrs(variable.attributes(),&required.attributes)?;replacements.insert(name.clone());
            }
        }
        let definition=NcVarDef::new(&name,ty,ids).with_attrs(attributes);
        if let Some(&position)=positions.get(&name) {definitions[position]=definition;}
        else {positions.insert(name.clone(),definitions.len());definitions.push(definition);}
        owners.insert(name,owner);
    }}
    for variable in &request.variables {
        if present.contains(&variable.name) {continue;}
        let ids:Vec<usize>=variable.dimensions.iter().map(|name|dimids.get(name).copied()
            .ok_or_else(||refusal(format!("field {} references missing dimension {name}",variable.name)))).collect::<MpasResult<_>>()?;
        definitions.push(NcVarDef::new(&variable.name,if variable.text.is_some() {NcType::Char} else {NcType::Float},ids)
            .with_attrs(attrs(&[],&variable.attributes)?));
        replacements.insert(variable.name.clone());receipt.declared.push(variable.name.clone());
    }
    let tmp=request.output.with_extension(format!("nc.tmp-{}",std::process::id()));
    let result=(|| {
        let mut writer=NcClassicWriter::create(&tmp,NcFormat::Data64,dims.clone(),attrs(&global_attributes,&request.attributes)?,definitions.clone(),if dims.iter().any(|d|d.unlimited) {1} else {0})?;
        // Every slab is classified and checked in declaration order first, so the
        // receipt and the computed-field refusals are the serial loop's. Then the
        // slabs are read, encoded and written at their own offsets on the cores;
        // the file holds the same bytes as writing them in turn. THE BREAKAGE IT
        // FIXES: the copy ran on one core, 412 s for the 33 GB RRFS v2 conus3km
        // boundary carrier while 127 cores waited (box V5090, 2026-10-28).
        enum Slab<'a> {Text(Vec<u8>),Floats(&'a [f32]),Zero(usize),Carried(usize,NcType,usize)}
        let mut slabs=Vec::with_capacity(definitions.len());
        let mut keys:Vec<(&str,Option<u64>)>=Vec::with_capacity(definitions.len());
        for definition in &definitions {
            let name=&definition.name;
            let record=definition.dimids.first().is_some_and(|id|dims[*id].unlimited);
            let size=definition.dimids.iter().try_fold(1usize,|n,id|n.checked_mul(if dims[*id].unlimited {1} else {dims[*id].len}))
                .ok_or_else(||refusal(format!("field {name} size overflow")))?;
            if replacements.contains(name) {
                let required=requested[name];
                if let Some(text)=&required.text {
                    if text.len()>size {return Err(refusal(format!("field {name} text exceeds its declared length")));}
                    let mut bytes=vec![b' ';size];bytes[..text.len()].copy_from_slice(text.as_bytes());slabs.push(Slab::Text(bytes));
                } else if let Some(values)=&required.values {
                    if values.len()!=size {return Err(refusal(format!("field {name} has {} values, expected {size}",values.len())));}
                    slabs.push(Slab::Floats(values));
                } else {slabs.push(Slab::Zero(size));}
                receipt.computed.push(name.clone());
            } else {
                slabs.push(Slab::Carried(owners[name],definition.ty,size));
                receipt.carried.push(name.clone());
            }
            keys.push((name.as_str(),record.then_some(0)));
        }
        // The first failure in declaration order is the one returned, as in the serial loop.
        let failure:std::sync::Mutex<Option<(usize,MpasError)>>=std::sync::Mutex::new(None);
        let fail=|i:usize,e:MpasError|->RwStoreError {
            let mut first=failure.lock().unwrap();
            if first.as_ref().map_or(true,|(j,_)|i<*j) {*first=Some((i,e));}
            RwStoreError::Format(format!("artifact field {} failed",keys[i].0))
        };
        // Bounded: a worker holds one field twice over (decoded and encoded),
        // about 1.1 GB on conus3km, so 32 workers stay near 36 GB of host memory.
        let pool=rayon::ThreadPoolBuilder::new().num_threads(rayon::current_num_threads().clamp(1,32)).build()
            .map_err(|e|refusal(format!("cannot start the artifact writers: {e}")))?;
        let written=pool.install(||writer.put_parallel(&keys,|i| {
            let name=keys[i].0;
            Ok(match &slabs[i] {
                Slab::Text(bytes)=>NcSlab::Borrowed(NcData::Chars(bytes)),
                Slab::Floats(values)=>NcSlab::Borrowed(NcData::Floats(values)),
                Slab::Zero(size)=>NcSlab::Owned(NcOwned::Floats(vec![0f32;*size])),
                Slab::Carried(owner,ty,size)=>{
                    // Stored-word reads preserve integer bits, signed zeros, NaN payloads,
                    // binary64 coordinates, and the complete character padding. Each worker
                    // opens its own (mapped) reader.
                    let source=netcrust::File::open(&paths[*owner]).map_err(|e|fail(i,MpasError::from(e)))?;
                    // A freshly read array is in standard order from offset 0, so its
                    // buffer is taken as is: no element-by-element copy on the worker
                    // (3.4 GB for zb and zb3 on conus3km, the tail of this stage).
                    macro_rules! carry {($ty:ty,$variant:ident)=>{{
                        let array=source.read_array::<$ty>(name).map_err(|e|fail(i,MpasError::from(e)))?;
                        let n=array.len();
                        let values:Vec<$ty>=if array.is_standard_layout() {
                            match array.into_raw_vec_and_offset() {
                                (values,Some(0)) if values.len()==n=>values,
                                (values,Some(offset))=>values[offset..offset+n].to_vec(),
                                (_,None)=>Vec::new(),
                            }
                        } else {array.into_iter().collect()};
                        if values.len()!=*size {return Err(fail(i,refusal(format!("carried field {name} length changed"))));}
                        NcSlab::Owned(NcOwned::$variant(values))
                    }};}
                    match ty {NcType::Char=>NcSlab::Owned(NcOwned::Chars(char_words(&paths[*owner],name).map_err(|e|fail(i,e))?)),
                        NcType::Short=>carry!(i16,Shorts),NcType::Int=>carry!(i32,Ints),NcType::Float=>carry!(f32,Floats),NcType::Double=>carry!(f64,Doubles)}
                }
            })
        }));
        if let Some((_,e))=failure.into_inner().unwrap() {return Err(e);}
        written?;
        writer.finish()?;std::fs::rename(&tmp,&request.output)?;Ok(receipt)
    })();
    if result.is_err() {let _=std::fs::remove_file(tmp);}
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn producer_overlay_preserves_words_and_rejects_incompatible_inventory() {
        let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("target/overlay-test-{}",std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let template=root.join("base.nc");let overlay=root.join("producer.nc");let output=root.join("composed.nc");
        let dims=vec![NcDim::record("Time"),NcDim::fixed("nCells",2),NcDim::fixed("StrLen",4)];
        let mut base=NcClassicWriter::create(&template,NcFormat::Data64,dims.clone(),vec![],
            vec![NcVarDef::new("terrain",NcType::Float,vec![1]),NcVarDef::new("clock",NcType::Int,vec![0])],1).unwrap();
        base.put("terrain",NcData::Floats(&[1.,2.])).unwrap();base.put_record("clock",0,NcData::Ints(&[7])).unwrap();base.finish().unwrap();
        let mut producer=NcClassicWriter::create(&overlay,NcFormat::Data64,dims,vec![NcAttr {name:"producer_provenance".into(),value:NcAttrValue::Text("retained witness".into())}],
            vec![NcVarDef::new("terrain",NcType::Float,vec![1]),NcVarDef::new("statistic",NcType::Float,vec![0,1]),
                NcVarDef::new("label",NcType::Char,vec![2]),NcVarDef::new("mask",NcType::Int,vec![1])],1).unwrap();
        producer.put("terrain",NcData::Floats(&[-0.,3.])).unwrap();
        let bits=[0x7fc12345,0x80000001];
        producer.put_record("statistic",0,NcData::Floats(&bits.map(f32::from_bits))).unwrap();
        producer.put("label",NcData::Chars(&[0,255,b' ',0])).unwrap();producer.put("mask",NcData::Ints(&[i32::MIN,i32::MAX])).unwrap();producer.finish().unwrap();
        let request=serde_json::json!({"template":template,"overlays":[overlay],"output":output,"dimensions":{},"variables":[]});
        let receipt=materialize(serde_json::from_value(request.clone()).unwrap()).unwrap();
        assert_eq!(receipt.overlays,[overlay.clone()]);
        let composed=netcrust::File::open(&output).unwrap();
        assert!(composed.attributes().unwrap().iter().any(|a|a.name()=="producer_provenance"));
        assert_eq!(composed.read_array::<f32>("statistic").unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),bits);
        assert_eq!(composed.read_array::<f32>("terrain").unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),[-0f32,3.].map(f32::to_bits));
        assert_eq!(composed.read_array::<i32>("mask").unwrap().into_iter().collect::<Vec<_>>(),[i32::MIN,i32::MAX]);
        assert_eq!(composed.read_array::<i32>("clock").unwrap().into_iter().collect::<Vec<_>>(),[7]);
        assert_eq!(char_words(&output,"label").unwrap(),[0,255,b' ',0]);drop(composed);
        let saved=std::fs::read(&output).unwrap();
        for (dimension,ty) in [(3,NcType::Float),(2,NcType::Int)] {
            let bad=root.join("bad.nc");let mut bad_writer=NcClassicWriter::create(&bad,NcFormat::Data64,
                vec![NcDim::fixed("nCells",dimension)],vec![],vec![NcVarDef::new("terrain",ty,vec![0])],0).unwrap();
            if ty==NcType::Float {bad_writer.put("terrain",NcData::Floats(&vec![0.;dimension])).unwrap();}
            else {bad_writer.put("terrain",NcData::Ints(&vec![0;dimension])).unwrap();}
            bad_writer.finish().unwrap();let mut invalid=request.clone();invalid["overlays"]=serde_json::json!([bad]);
            assert!(materialize(serde_json::from_value(invalid).unwrap()).is_err());assert_eq!(std::fs::read(&output).unwrap(),saved);
        }
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn stored_words_survive_while_constructed_fields_replace_zeros() {
        let root=PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("target/artifact-test-{}",std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let source=root.join("static.nc");let out=root.join("vertical.nc");
        let dims=vec![NcDim::fixed("n",2),NcDim::fixed("StrLen",4)];
        let defs=vec![NcVarDef::new("zero",NcType::Float,vec![0]),NcVarDef::new("coordinate",NcType::Double,vec![0]),
            NcVarDef::new("bits",NcType::Int,vec![0]),NcVarDef::new("small",NcType::Short,vec![0]),
            NcVarDef::new("label",NcType::Char,vec![1]),NcVarDef::new("special",NcType::Float,vec![0])];
        let mut writer=NcClassicWriter::create(&source,NcFormat::Data64,dims,vec![],defs,0).unwrap();
        writer.put("zero",NcData::Floats(&[0.,-0.])).unwrap();
        let coordinates=[f64::from_bits(0x3ff0000000000001),-0f64];
        let special=[f32::from_bits(0x7fc12345),-0f32];
        writer.put("coordinate",NcData::Doubles(&coordinates)).unwrap();
        writer.put("bits",NcData::Ints(&[i32::MIN,i32::MAX])).unwrap();
        writer.put("small",NcData::Shorts(&[i16::MIN,i16::MAX])).unwrap();
        writer.put("label",NcData::Chars(&[b'a',0,b' ',255])).unwrap();
        writer.put("special",NcData::Floats(&special)).unwrap();writer.finish().unwrap();
        let request=serde_json::json!({"template":source,"output":out,"dimensions":{"Time":null,"n":2},
            "variables":[{"name":"zero","dimensions":["n"],"only_if_zero":true,"values":[2.,3.]},
                {"name":"time_state","dimensions":["Time","n"]}],"attributes":{"config_yes":true}});
        let receipt=materialize(serde_json::from_value(request.clone()).unwrap()).unwrap();
        assert_eq!(receipt.declared,["time_state"]);
        let file=netcrust::File::open(&out).unwrap();
        assert_eq!(file.read_array::<f32>("zero").unwrap().into_iter().collect::<Vec<_>>(),[2.,3.]);
        assert_eq!(file.read_array::<i32>("bits").unwrap().into_iter().collect::<Vec<_>>(),[i32::MIN,i32::MAX]);
        assert_eq!(file.read_array::<i16>("small").unwrap().into_iter().collect::<Vec<_>>(),[i16::MIN,i16::MAX]);
        assert_eq!(char_words(&out,"label").unwrap(),[b'a',0,b' ',255]);
        assert_eq!(file.read_array::<f64>("coordinate").unwrap().into_iter().map(f64::to_bits).collect::<Vec<_>>(),coordinates.map(f64::to_bits));
        assert_eq!(file.read_array::<f32>("special").unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),special.map(f32::to_bits));
        assert_eq!(file.read_array::<f32>("time_state").unwrap().into_iter().collect::<Vec<_>>(),[0.,0.]);
        drop(file);
        let original=std::fs::read(&out).unwrap();let mut invalid=request.clone();invalid["variables"][0]["values"]=serde_json::json!([1.]);
        assert!(materialize(serde_json::from_value(invalid).unwrap()).is_err());
        assert_eq!(std::fs::read(&out).unwrap(),original,"failed encoding must leave the previous artifact intact");
        let mut carry=request;carry["template"]=serde_json::json!(out);carry["output"]=serde_json::json!(root.join("carried.nc"));
        let receipt=materialize(serde_json::from_value(carry).unwrap()).unwrap();assert!(receipt.carried.contains(&"zero".to_string()));
        std::fs::remove_dir_all(root).unwrap();
    }
}
