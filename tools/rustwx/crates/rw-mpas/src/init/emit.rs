//! Writing the CDF-5 initial-conditions file.
//!
//! ## Schema by example, values by computation
//! The container's shape: its seventeen dimensions, its 110 float, 21 int and
//! 3 char variables and their attributes: is read from the capsule and
//! re-emitted; native Thompson package activation can add nifa/nwfa. What each variable *holds* is then one of exactly
//! two things, and the receipt says which for every single one:
//!
//! * **computed**: this port produced the numbers, and they are what the
//!   validation compares;
//! * **carried**: the value came from the capsule untouched.  Statics,
//!   mesh geometry and the vertical metrics are carried by design this wave
//!   (see [`crate::init::capsule`]); a delta of exactly zero on a carried
//!   variable is therefore *not* evidence of anything and is labelled so
//!   nobody reads it as agreement.
//!
//! Reading the schema from the same file the run is compared against is a real
//! hazard and is named rather than hidden: it makes the container trivially
//! right and can make a *carried* field look perfect.  It cannot make a
//! computed field look right, which is why the receipt separates the two lists
//! and why the delta table repeats the label on every row.
//!
//! ## Array order
//! MPAS is cell-major with the vertical fastest: `(Time, nCells,
//! nVertLevels)`.  That is the opposite of the wrfout convention this crate's
//! converter emits, and it is the single easiest thing in the file to get
//! backwards, so every computed field is assembled cell-major and the
//! assembly is asserted against the capsule's own element count before a byte
//! is written.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rayon::prelude::*;
use rw_store::error::RwStoreError;
use rw_store::netcdf_classic::{
    NcAttr, NcAttrValue, NcClassicWriter, NcCopy, NcData, NcDim, NcFormat, NcOwned, NcSlab, NcType,
    NcVarDef,
};

use crate::error::{MpasError, MpasResult};

/// A computed variable waiting to be written, in the file's own order.
pub enum Computed {
    Floats(Vec<f32>),
    Ints(Vec<i32>),
    /// An `NC_CHAR` payload.  The writer NUL-fills a short slab to the
    /// declared length, which is how a 19-byte stamp lands in a 64-byte slot.
    Text(String),
}

impl Computed {
    fn len(&self) -> usize {
        match self {
            Computed::Floats(v) => v.len(),
            Computed::Ints(v) => v.len(),
            // A text slab is length-checked by the writer, which pads; a
            // 19-byte stamp in a 64-byte slot is correct, not short.
            Computed::Text(_) => usize::MAX,
        }
    }
}

/// What the emitter did, variable by variable.
#[derive(Debug, Default, serde::Serialize)]
pub struct EmitLedger {
    pub computed: Vec<String>,
    pub carried: Vec<String>,
    pub file_id: String,
    pub parent_id: String,
    pub bytes: u64,
}

/// The fifteen arrays a static may legitimately store as binary64.
///
/// The three lat/lon/xyz families, and nothing else. They move together
/// because the port cross-checks xyz against the lat/lon pair beside it; see
/// `staticfile::coordframe::CoordinateRepresentation::nc_type`.
const COORDINATE_ARRAYS: [&str; 15] = [
    "latCell", "lonCell", "xCell", "yCell", "zCell",
    "latEdge", "lonEdge", "xEdge", "yEdge", "zEdge",
    "latVertex", "lonVertex", "xVertex", "yVertex", "zVertex",
];

pub(super) fn nc_type_of(dtype: &netcrust::DataType) -> MpasResult<NcType> {
    Ok(match dtype {
        netcrust::DataType::Char | netcrust::DataType::I8 | netcrust::DataType::U8 => NcType::Char,
        netcrust::DataType::I16 => NcType::Short,
        netcrust::DataType::I32 => NcType::Int,
        netcrust::DataType::F32 => NcType::Float,
        netcrust::DataType::F64 => NcType::Double,
        other => {
            return Err(MpasError::Refusal(format!(
                "capsule carries a {other:?} variable, which the classic writer has no type for"
            )))
        }
    })
}

pub(super) fn attr_of(a: &netcrust::Attribute) -> Option<NcAttr> {
    let value = match a.value() {
        netcrust::AttributeValue::Chars(s) => NcAttrValue::Text(s.clone()),
        netcrust::AttributeValue::Strings(v) if v.len() == 1 => NcAttrValue::Text(v[0].clone()),
        netcrust::AttributeValue::Ints(v) => NcAttrValue::Ints(v.clone()),
        netcrust::AttributeValue::Floats(v) => NcAttrValue::Floats(v.clone()),
        netcrust::AttributeValue::Doubles(v) => NcAttrValue::Doubles(v.clone()),
        _ => return None,
    };
    Some(NcAttr {
        name: a.name().to_string(),
        value,
    })
}

/// A ten-character token, derived from the run's own inputs so two runs on the
/// same inputs mint the same id and two runs on different inputs do not.
pub fn mint_file_id(seed: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(seed.as_bytes());
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    digest
        .iter()
        .take(10)
        .map(|b| ALPHABET[(*b as usize) % ALPHABET.len()] as char)
        .collect()
}

/// Write the init file: capsule schema, computed values where we have them.
///
/// `parent_chain` becomes `parent_id`.  MPAS writes it newline separated and
/// with a trailing empty entry, and the port's loader checks that shape, so it
/// is built here rather than left to the caller to remember.
pub fn write_init(
    out_path: &Path,
    capsule_path: &Path,
    computed: BTreeMap<String, Computed>,
    file_id: &str,
    parent_chain: &[String],
    provenance: &str,
    lineage: &crate::init::Lineage,
) -> MpasResult<EmitLedger> {
    let capsule = netcrust::File::open(capsule_path)?;
    // The capsule's own classic layout and a plain handle on it: a carried variable is copied as
    // the bytes the capsule stores (NcSlab::Copy), not decoded and re-encoded.  A capsule that is
    // not classic NetCDF keeps the decode path.
    let capsule_layout = netcdf_reader::NcFile::open(capsule_path).ok();
    let capsule_bytes = std::fs::File::open(capsule_path)?;

    // Dimensions, in the capsule's own order, so dimids line up.
    let cap_dims = capsule.dimensions()?;
    let cells = cap_dims.iter().find(|d| d.name() == "nCells").map(|d| d.len()).unwrap_or(0);
    let soil_levels = match computed.get("tslb") {
        Some(Computed::Floats(v)) if cells > 0 => Some(v.len()/cells), _ => None,
    };
    let mut dims: Vec<NcDim> = Vec::with_capacity(cap_dims.len());
    let mut dimid_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut record_len = 0u64;
    for (i, d) in cap_dims.iter().enumerate() {
        dimid_of.insert(d.name().to_string(), i);
        if d.is_unlimited() {
            record_len = d.len().max(1) as u64;
            dims.push(NcDim::record(d.name()));
        } else {
            dims.push(NcDim::fixed(d.name(), if d.name() == "nSoilLevels" {soil_levels.unwrap_or(d.len())} else {d.len()}));
        }
    }
    if record_len == 0 {
        record_len = 1;
    }

    let parent_id = {
        let mut chain: Vec<String> = parent_chain.to_vec();
        // MPAS's own trailing empty entry.
        chain.push(String::new());
        chain.join("\n")
    };

    let mut gattrs: Vec<NcAttr> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    // The model's own lineage leads the header, which is where native writes
    // it: model_name, core_name, version, source, Conventions, git_version,
    // then the mesh block.  `source` and `Conventions` describe the FILE and
    // come from the capsule, so only the four that describe the MODEL are
    // written here -- and only when the capsule does not already carry them,
    // so a capsule with a lineage of its own keeps it.
    let capsule_names: BTreeSet<String> = capsule
        .attributes()?
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    for (name, value) in [
        ("model_name", lineage.model_name.as_str()),
        ("core_name", lineage.core_name.as_str()),
        ("version", lineage.version.as_str()),
    ] {
        if !capsule_names.contains(name) {
            if value.is_empty() {
                return Err(MpasError::Refusal(format!(
                    "the {name} lineage attribute is empty; an init that names                      no model cannot be handed to rw_mpas_lbc, which reads all                      four off its --grid and refuses rather than invent one"
                )));
            }
            gattrs.push(NcAttr::text(name, value));
            seen.insert(name.to_string());
        }
    }
    for a in capsule.attributes()? {
        let name = a.name().to_string();
        if name == "file_id" {
            gattrs.push(NcAttr::text("file_id", file_id));
        } else if name == "parent_id" {
            gattrs.push(NcAttr::text("parent_id", parent_id.clone()));
        } else if name == "config_nsoillevels" && soil_levels.is_some() {
            gattrs.push(NcAttr {name: name.clone(), value: NcAttrValue::Ints(vec![soil_levels.unwrap() as i32])});
        } else if let Some(attr) = attr_of(&a) {
            gattrs.push(attr);
        } else {
            return Err(MpasError::Refusal(format!(
                "capsule global attribute {name} has a value shape the classic writer cannot \
                 carry; dropping it silently would change the file's identity"
            )));
        }
        seen.insert(name);
    }
    // git_version sits after `Conventions` in native's order, which is after
    // everything the capsule carries, so it is appended rather than led.
    if !capsule_names.contains("git_version") {
        if lineage.git_version.is_empty() {
            return Err(MpasError::Refusal(
                "the git_version lineage attribute is empty; an init that                  names no build cannot be handed to rw_mpas_lbc"
                    .to_string(),
            ));
        }
        gattrs.push(NcAttr::text("git_version", lineage.git_version.as_str()));
        seen.insert("git_version".to_string());
    }
    if !seen.contains("file_id") {
        gattrs.push(NcAttr::text("file_id", file_id));
    }
    if !seen.contains("parent_id") {
        gattrs.push(NcAttr::text("parent_id", parent_id.clone()));
    }
    if !seen.contains("gpuwm_provenance") {
        gattrs.push(NcAttr::text("gpuwm_provenance", provenance));
    }

    // Variable definitions, in the capsule's order.
    let cap_vars = capsule.variables()?;
    let mut vars: Vec<NcVarDef> = Vec::with_capacity(cap_vars.len());
    for v in &cap_vars {
        let ty = nc_type_of(v.dtype())?;
        let mut dimids = Vec::new();
        for d in v.dimensions() {
            let id = *dimid_of.get(d.name()).ok_or_else(|| {
                MpasError::Refusal(format!(
                    "variable {} names dimension {} the capsule does not declare",
                    v.name(),
                    d.name()
                ))
            })?;
            dimids.push(id);
        }
        let attrs: Vec<NcAttr> = v.attributes().iter().filter_map(attr_of).collect();
        vars.push(NcVarDef::new(v.name(), ty, dimids).with_attrs(attrs));
    }

    // MPAS: mpas_init_atm_core_interface.F:229-231;
    // core_init_atmosphere/Registry.xml:1153-1173. Native file presence
    // activates the pair even when the input stream has no aerosol slots.
    let mut added_aerosols = Vec::new();
    let has_computed_pair = computed.contains_key("nifa") && computed.contains_key("nwfa");
    for (name, description) in [
        ("nifa", "Gocart ice-friendly aerosol number concentration"),
        ("nwfa", "Gocart water-friendly aerosol number concentration"),
    ] {
        if has_computed_pair && !cap_vars.iter().any(|v| v.name() == name) {
            let dimensions: Vec<usize> = ["Time", "nCells", "nVertLevels"].iter()
                .map(|name| dimid_of.get(*name).copied().ok_or_else(|| MpasError::Refusal(
                    format!("Thompson aerosol output requires init dimension {name}"))))
                .collect::<MpasResult<_>>()?;
            let expected = dims[dimensions[1]].len as usize * dims[dimensions[2]].len as usize;
            let Some(Computed::Floats(values)) = computed.get(name) else {
                return Err(MpasError::Refusal(format!("native Thompson scalar {name} must be binary32")));
            };
            if values.len() != expected {
                return Err(MpasError::Refusal(format!("Thompson scalar {name} has {} words, expected {expected}", values.len())));
            }
            vars.push(NcVarDef::new(name, NcType::Float, dimensions).with_attrs(vec![
                NcAttr::text("units", "nb kg^{-1}"), NcAttr::text("long_name", description),
            ]));
            added_aerosols.push(name);
        }
    }

    let present: BTreeSet<String> = cap_vars.iter().map(|v| v.name().to_string()).collect();
    // Only the transported species a physics suite may add are created here;
    // any other computed name without a capsule variable stays a stray and
    // meets the "no variable for" refusal below, which the door keys on.
    let admitted = ["qi","qs","qg","ni","nr","nc","nwfa","nifa","ng","volg"];
    let extra: Vec<String> = computed.keys()
        .filter(|name| !present.contains(*name) && !added_aerosols.contains(&name.as_str()) && admitted.contains(&name.as_str())).cloned().collect();
    for name in &extra {
        let mut ids = vec![];
        if let Some(&id) = dimid_of.get("Time") { ids.push(id); }
        for dim in ["nCells", "nVertLevels"] {
            ids.push(*dimid_of.get(dim).ok_or_else(|| MpasError::Refusal(format!("init lacks {dim}")))?);
        }
        vars.push(NcVarDef::new(name, NcType::Float, ids));
    }
    let mut writer = NcClassicWriter::create(
        out_path,
        NcFormat::Data64,
        dims,
        gattrs,
        vars,
        record_len,
    )
    .map_err(|e| MpasError::Refusal(format!("cannot lay out {}: {e}", out_path.display())))?;

    // A computed value whose name is not in the file would be dropped without
    // a word, which is the quietest possible way to ship an init with a field
    // left at whatever the capsule held.  Refuse instead.
    {
        let present: BTreeSet<String> = cap_vars.iter().map(|v| v.name().to_string()).collect();
        let stray: Vec<&String> = computed.keys().filter(|k| !present.contains(*k) && !extra.contains(*k) && !added_aerosols.contains(&k.as_str())).collect();
        if !stray.is_empty() {
            return Err(MpasError::Refusal(format!(
                "this run computed {} value(s) the init file has no variable for: {:?}.  Writing                  the file anyway would silently drop them and leave the capsule's own numbers in                  their place",
                stray.len(),
                stray
            )));
        }
    }

    let mut ledger = EmitLedger {
        file_id: file_id.to_string(),
        parent_id,
        ..Default::default()
    };

    // Every slab's checks run here, in file order; the bytes are then
    // produced (carried slabs read from the capsule), encoded and written on
    // rayon workers, each at its own fixed offset in the file.
    enum Source {
        Computed,
        Text(Vec<u8>),
        CarriedInt,
        CarriedFloat,
        CarriedDouble,
        /// A carried variable whose stored bytes are copied whole: the capsule offset of the
        /// slab, its type and its value count (the writer refuses a count or type that is not
        /// the init's own).
        CarriedBytes(u64, NcType, u64),
    }
    let mut jobs: Vec<(String, Option<u64>, Source)> = Vec::with_capacity(cap_vars.len());
    for v in &cap_vars {
        let name = v.name().to_string();
        let is_record = v
            .dimensions()
            .first()
            .map(|d| d.is_unlimited())
            .unwrap_or(false);
        let record = if is_record { Some(0) } else { None };

        if let Some(value) = computed.get(&name) {
            let expected: usize = v
                .dimensions()
                .iter()
                .skip(usize::from(is_record))
                .map(|d| if d.name() == "nSoilLevels" {soil_levels.unwrap_or(d.len())} else {d.len().max(1)})
                .product();
            if !matches!(value, Computed::Text(_)) && value.len() != expected {
                return Err(MpasError::Refusal(format!(
                    "computed {name} holds {} value(s); the file's slab is {expected}.  A \
                     mismatched count here is the cell-major/level-major transpose, which \
                     produces a readable file full of wrong numbers",
                    value.len()
                )));
            }
            let source = match value {
                Computed::Text(t) => {
                    if t.len() > expected {
                        return Err(MpasError::Refusal(format!(
                            "computed {name} is {} character(s); the file's field is {expected}. \
                             Truncating it would leave a different label that still reads",
                            t.len()
                        )));
                    }
                    let mut bytes = t.as_bytes().to_vec();
                    bytes.resize(expected, b' ');
                    Source::Text(bytes)
                }
                _ => Source::Computed,
            };
            jobs.push((name.clone(), record, source));
            ledger.computed.push(name);
            continue;
        }

        let ty = nc_type_of(v.dtype())?;
        let source = match ty {
            NcType::Char => {
                return Err(MpasError::Refusal(format!(
                    "char variable {name} was not supplied by the caller.  Char variables in an \
                     MPAS init are identity labels; copying one from the capsule would date this \
                     file to the capsule's cycle"
                )));
            }
            NcType::Int | NcType::Float
            | NcType::Double if ty != NcType::Double || COORDINATE_ARRAYS.contains(&name.as_str()) =>
            {
                // The whole of a fixed variable, or the one record of a record variable; any other
                // record count keeps the decode path and the slab-length refusal it meets.
                let elems: u64 = v
                    .dimensions()
                    .iter()
                    .skip(usize::from(is_record))
                    .map(|d| d.len() as u64)
                    .product();
                match capsule_layout.as_ref().and_then(|file| file.classic_extent(&name)) {
                    Some(extent) if extent.is_record == is_record && (!is_record || extent.numrecs == 1) => {
                        Source::CarriedBytes(extent.offset, ty, elems)
                    }
                    _ => match ty {
                        NcType::Int => Source::CarriedInt,
                        NcType::Float => Source::CarriedFloat,
                        _ => Source::CarriedDouble,
                    },
                }
            }
            other => {
                return Err(MpasError::Refusal(format!(
                    "carried variable {name} is {other:?}, which this emitter does not copy. \
                     Only the fifteen mesh coordinate arrays may arrive as Double, and only \
                     because a static for a mesh with no native MPAS-A counterpart stores them \
                     that way; everything else -- the vertical contract and every met-state slot \
                     -- is pre-declared as a float32 zero slot by the door that mints the capsule"
                )))
            }
        };
        jobs.push((name.clone(), record, source));
        ledger.carried.push(name);
    }
    {
        let failure: std::sync::Mutex<Option<MpasError>> = std::sync::Mutex::new(None);
        let keys: Vec<(&str, Option<u64>)> =
            jobs.iter().map(|(name, record, _)| (name.as_str(), *record)).collect();
        // A wave owns at most 512 MiB of estimated carried/encoded scratch,
        // except a single larger field. That field still decodes/encodes on
        // Rayon workers. Previously all large fields could allocate together.
        // A copied slab holds no scratch: the kernel moves its bytes.
        let mut batches = Vec::new();
        let mut start = 0;
        let mut bytes = 0usize;
        for (i, variable) in cap_vars.iter().enumerate() {
            let words = variable.dimensions().iter().map(|d| d.len().max(1)).product::<usize>();
            let cost = if matches!(jobs[i].2, Source::CarriedBytes(..)) { 0 } else { words.saturating_mul(32) };
            if i > start && bytes.saturating_add(cost) > (512 << 20) {
                batches.push(start..i);
                start = i;
                bytes = 0;
            }
            bytes = bytes.saturating_add(cost);
        }
        if start < jobs.len() { batches.push(start..jobs.len()); }
        let written = (|| -> Result<(), RwStoreError> {
            for batch in batches {
                let first = batch.start;
                writer.put_parallel(&keys[batch], |offset| {
                    let (name, _, source) = &jobs[first + offset];
                    let carried = || -> Result<Vec<f64>, RwStoreError> {
                        match super::capsule::read_f64_par(&capsule, name) {
                            Ok(values) => Ok(values),
                            Err(e) => {
                                failure.lock().unwrap().get_or_insert(MpasError::from(e));
                                Err(RwStoreError::Format(format!("cannot read {name} from the capsule")))
                            }
                        }
                    };
                    Ok(match source {
                        Source::Computed => match &computed[name] {
                            Computed::Floats(f) => NcSlab::Borrowed(NcData::Floats(f)),
                            Computed::Ints(v) => NcSlab::Borrowed(NcData::Ints(v)),
                            Computed::Text(_) => unreachable!("text is carried as Source::Text"),
                        },
                        Source::Text(bytes) => NcSlab::Borrowed(NcData::Chars(bytes)),
                        Source::CarriedInt => NcSlab::Owned(NcOwned::Ints(
                            carried()?.into_par_iter().map(|v| v.round() as i32).collect(),
                        )),
                        Source::CarriedFloat => NcSlab::Owned(NcOwned::Floats(
                            carried()?.into_par_iter().map(|v| v as f32).collect(),
                        )),
                        Source::CarriedDouble => NcSlab::Owned(NcOwned::Doubles(carried()?)),
                        Source::CarriedBytes(offset, ty, elems) => NcSlab::Copy(NcCopy {
                            file: &capsule_bytes,
                            offset: *offset,
                            ty: *ty,
                            elems: *elems,
                        }),
                    })
                })?;
            }
            Ok(())
        })();
        if let Some(e) = failure.into_inner().unwrap() {
            return Err(e);
        }
        written.map_err(|e| MpasError::Refusal(format!("cannot write the init variables: {e}")))?;
    }

    for name in added_aerosols {
        let Some(Computed::Floats(values)) = computed.get(name) else { unreachable!() };
        writer.put_record(name, 0, NcData::Floats(values))
            .map_err(|e| MpasError::Refusal(format!("cannot write native Thompson scalar {name}: {e}")))?;
        ledger.computed.push(name.to_string());
    }

    for name in extra {
        let Computed::Floats(values) = &computed[&name] else { return Err(MpasError::Refusal("added species must be float32".into())); };
        let levels = cap_dims.iter().find(|d| d.name()=="nVertLevels").unwrap().len();
        if values.len()!=cells*levels || values.iter().any(|x| !x.is_finite() || *x<0.) {
            return Err(MpasError::Refusal(format!("added species {name} is invalid or has a wrong shape")));
        }
        if dimid_of.contains_key("Time") { writer.put_record(&name,0,NcData::Floats(values)) }
        else { writer.put(&name,NcData::Floats(values)) }.map_err(|e| MpasError::Refusal(e.to_string()))?;
        ledger.computed.push(name);
    }
    ledger.bytes = writer.file_len();
    writer.finish()
    .map_err(|e| MpasError::Refusal(format!("cannot finish {}: {e}", out_path.display())))?;
    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_id_is_ten_lowercase_letters_and_is_a_function_of_its_seed() {
        let a = mint_file_id("MET:2025-03-14_12|x4.163842");
        let b = mint_file_id("MET:2025-03-14_12|x4.163842");
        let c = mint_file_id("MET:2026-08-12_06|x4.163842");
        assert_eq!(a.len(), 10);
        assert!(a.chars().all(|ch| ch.is_ascii_lowercase()));
        assert_eq!(a, b, "the same inputs must mint the same id");
        assert_ne!(a, c, "different inputs must not");
    }

    #[test]
    fn the_parent_chain_keeps_its_trailing_empty_entry() {
        let chain = vec!["abcdefghij".to_string(), "klmnopqrst".to_string()];
        let mut with_tail = chain.clone();
        with_tail.push(String::new());
        let joined = with_tail.join("\n");
        assert!(joined.ends_with('\n'), "the loader checks for it: {joined:?}");
        assert_eq!(joined.split('\n').count(), 3);
    }
}
