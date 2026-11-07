//! Subcommand orchestration: argv grammar, input staging, the three verbs.
//!
//! The invocation grammar is normative (design doc §3.1).  `--input-list`
//! is a file of one path per line rather than argv entries because the
//! 251-file icon-eu prep hit Windows' argv limit; that lesson is baked in
//! from day one, and argv here never carries an input path.
//!
//! Paths are used EXACTLY as the input list spells them.  The engine does
//! not canonicalize: provenance strings (`<path>:<index>`) must match what
//! the Python engine wrote, and Windows canonicalization would prepend the
//! `\\?\` verbatim prefix and change every reference in every receipt.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::{json, Value};

use crate::assemble::{assemble_grib, DecodedCollection};
use crate::grib::{grib2_identities, validate_grib2_envelopes, wanted_indices, GribRecord};
use crate::model::Mapping;
use crate::node::Node;
use crate::refusal::{manifest_mismatch, missing_input, usage, Refusal, Result};

/// The parsed argv of one engine run.
#[derive(Debug, Clone, Default)]
pub struct Invocation {
    pub subcommand: String,
    pub mapping: String,
    pub input_list: String,
    pub output: Option<String>,
    pub composition: Option<String>,
    pub supplements: Vec<(String, String)>,
    pub provenance: Vec<(String, String)>,
    pub contributing_mappings: Vec<(String, String)>,
    pub input_manifest: Option<String>,
    pub input_manifest_sha256: Option<String>,
    pub atmospheric_window: bool,
    /// `compose --lead-batch`: the input list is some of a window's
    /// leads, so the series rules the window answers to (at least two
    /// times on one uniform cadence) are the caller's to hold across its
    /// batches; every other check is unchanged.
    pub lead_batch: bool,
    /// `compose --unpublished-fallback FIELD=ID`: a contributing binding
    /// whose files publish no record at all for its fields is recorded
    /// with these fallbacks instead of refused, when EVERY field it binds
    /// has one.  The caller owns the field-keyed table and the start
    /// reads the fallback; a binding with any field not listed here is
    /// refused exactly as before.
    pub unpublished_fallbacks: Vec<(String, String)>,
}

pub const USAGE: &str = "usage: gpuwm_mapped_engine {decode|compose|inspect} \
--mapping MAPPING.json --input-list FILES.txt --output DIR \
[--composition COMPOSITION.json] [--supplement ROLE=PATH]... \
[--provenance ROLE=PATH]... [--contributing-mapping ROLE=PATH]... \
[--input-manifest MANIFEST.json --input-manifest-sha256 HEX] \
[--atmospheric-window stdio] [--lead-batch] [--unpublished-fallback FIELD=ID]...\n\
   or: gpuwm_mapped_engine inventory --input-list FILES.txt\n\
   or: gpuwm_mapped_engine capabilities";

/// `capabilities`: what THIS build implements, as one JSON object.
///
/// The port is arriving in stages, so "which paths does the engine
/// decode?" is a question about the binary in hand, not about the
/// release notes.  gpuwm routes a bare run on the answer, and its
/// `ENGINE_CAPABILITIES` table is checked against this output, so a
/// build that gains `compose` is picked up by the route the moment it
/// says so here, and a table that drifts from the binary fails a test
/// instead of misrouting a user.
pub fn run_capabilities() -> Value {
    json!({
        "schema": crate::CAPABILITIES_SCHEMA,
        "engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION},
        "frameset_schema": crate::FRAMESET_SCHEMA,
        "features": {"atmospheric_window": crate::window::SCHEMA,
                     "lead_batch": "gpuwm-mapped-lead-batch-v1",
                     "host_memory_budget": crate::threads::MEMORY_BUDGET_SCHEMA},
        // Per subcommand, the mapped source formats it decodes in
        // process.  An empty list means the subcommand is declared by
        // the contract and refuses `not_implemented` in this build.
        //
        // GRIB1 joined GRIB2 when `crate::grib1` landed: the ERA5 1974
        // reference object's forty-two decoded arrays, its grid
        // fingerprint and its materialization refusal are byte-identical
        // to the Python engine's through this binary, so a bare run of a
        // GRIB1 mapping decodes here rather than routing back to Python.
        //
        // NetCDF joined on the corpus fix.  It was held back because the
        // engine read one hand-made file and misread the corpus: the
        // vendored HDF5 reader enumerated ONE variable out of a
        // `netCDF4.Dataset(path, "w")` file, so a latitude selector
        // matched nothing.  Both causes are now named and fixed -- the
        // rw_wps workspace was linking the STOCK crates.io hdf5-reader
        // instead of the hardened vendored copy, and NetCDF-4 coordinate
        // variables are HDF5 dimension scales that netcrust's variable
        // index omits.  The evidence for the declaration is not the one
        // golden: it is the whole Python NetCDF suite green under
        // GPUWM_MAPPED_ENGINE=rust, on the same fixtures the Python
        // engine passes.
        //
        // `compose` joined on the parity evidence: every registered
        // composed source with staged bytes reproduces the Python
        // engine's composed answer byte for byte through this binary --
        // the frames, the alignment receipt across all three terrain
        // clock rules, and the per-binding contributing-source records
        // across both cross-source shapes.  It declares the same three
        // formats as `decode` and cannot outrun it: the manifest a
        // preparation seals asks the capability table about `decode`
        // while the composition asks about `compose`, so a format
        // declared for one and not the other would seal one decoder
        // inventory and verify against another.
        // `inventory` is the raw per-record product-identity surface: it
        // renders section octets and does not decode a field, so it is a
        // GRIB2 surface by construction -- the subprocess tool it
        // replaces (`grib2_inventory`) reads the same edition and nothing
        // else, and the archive-contract gates that consume it pin GRIB2
        // octet vocabulary (PDT, GDT, DRT).
        "subcommands": {
            "decode": ["grib1", "grib2", "netcdf"],
            "inspect": ["grib1", "grib2", "netcdf"],
            "compose": ["grib1", "grib2", "netcdf"],
            "inventory": ["grib2"],
        },
    })
}

impl Invocation {
    pub fn parse(arguments: &[String]) -> Result<Self> {
        let mut invocation = Invocation::default();
        let Some(subcommand) = arguments.first() else {
            return Err(usage("unknown or missing subcommand"));
        };
        if !matches!(
            subcommand.as_str(),
            "decode" | "compose" | "inspect" | "inventory" | "capabilities"
        ) {
            return Err(usage(format!("unknown subcommand '{subcommand}'")));
        }
        invocation.subcommand = subcommand.clone();
        // `capabilities` answers from the binary alone: it is what a
        // caller runs to find out WHICH of the others this build can do,
        // so requiring a mapping and an input list to ask would defeat
        // the question.
        if subcommand == "capabilities" {
            return Ok(invocation);
        }
        let mut position = 1usize;
        while position < arguments.len() {
            let flag = arguments[position].as_str();
            let value = || -> Result<String> {
                arguments
                    .get(position + 1)
                    .cloned()
                    .ok_or_else(|| usage(format!("{flag} needs a value")))
            };
            match flag {
                "--mapping" => invocation.mapping = value()?,
                "--input-list" => invocation.input_list = value()?,
                "--output" => invocation.output = Some(value()?),
                "--atmospheric-window" => {
                    if value()? != "stdio" {
                        return Err(usage("--atmospheric-window requires stdio"));
                    }
                    invocation.atmospheric_window = true;
                }
                "--lead-batch" => {
                    if invocation.subcommand != "compose" {
                        return Err(usage("--lead-batch belongs to compose"));
                    }
                    invocation.lead_batch = true;
                    position += 1;
                    continue;
                }
                "--unpublished-fallback" => {
                    if invocation.subcommand != "compose" {
                        return Err(usage("--unpublished-fallback belongs to compose"));
                    }
                    let entry = value()?;
                    let (field, fallback) = entry.split_once('=').ok_or_else(|| {
                        usage(format!("{flag} takes FIELD=ID; got '{entry}'"))
                    })?;
                    if field.is_empty() || fallback.is_empty() {
                        return Err(usage(format!("{flag} takes FIELD=ID; got '{entry}'")));
                    }
                    invocation
                        .unpublished_fallbacks
                        .push((field.to_owned(), fallback.to_owned()));
                }
                "--composition" => invocation.composition = Some(value()?),
                "--input-manifest" => invocation.input_manifest = Some(value()?),
                "--input-manifest-sha256" => invocation.input_manifest_sha256 = Some(value()?),
                "--supplement" | "--provenance" | "--contributing-mapping" => {
                    let binding = value()?;
                    let (role, path) = binding.split_once('=').ok_or_else(|| {
                        usage(format!("{flag} takes ROLE=PATH; got '{binding}'"))
                    })?;
                    let entry = (role.to_owned(), path.to_owned());
                    match flag {
                        "--supplement" => invocation.supplements.push(entry),
                        "--provenance" => invocation.provenance.push(entry),
                        // The third role-bound binding a cross-source
                        // composition carries: each contributing
                        // source's own sealed mapping document, whose
                        // bytes the composition pins by SHA-256.
                        // `compose` resolves every `field_sources`
                        // binding through it.
                        _ => invocation.contributing_mappings.push(entry),
                    }
                }
                other => return Err(usage(format!("unknown option '{other}'"))),
            }
            position += 2;
        }
        // `inventory` reads raw section octets; there is no mapping to
        // resolve selectors against and no frameset to write, so it takes
        // the input list alone -- demanding a mapping would make a
        // product-identity question depend on a document it never reads.
        if invocation.mapping.is_empty() && invocation.subcommand != "inventory" {
            return Err(usage("--mapping is required"));
        }
        if invocation.subcommand == "inventory" && !invocation.mapping.is_empty() {
            return Err(usage(
                "inventory reads raw record identity and takes no --mapping",
            ));
        }
        if invocation.input_list.is_empty() {
            return Err(usage("--input-list is required"));
        }
        if !matches!(invocation.subcommand.as_str(), "inspect" | "inventory")
            && invocation.output.is_none()
        {
            return Err(usage("--output is required for decode and compose"));
        }
        if invocation.input_manifest.is_some() != invocation.input_manifest_sha256.is_some() {
            return Err(usage(
                "--input-manifest and --input-manifest-sha256 are an atomic pair",
            ));
        }
        Ok(invocation)
    }
}

/// `mapped_source.read_input_list`: one UTF-8 path per line, blank lines
/// dropped, duplicates refused.
pub fn read_input_list(path: &str) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| missing_input(format!("cannot read input list {path}: {error}")))?;
    let mut files: Vec<String> = Vec::new();
    for line in text.lines() {
        // Verbatim, exactly as Python's `read_input_list` takes it: a line
        // is skipped when it is whitespace-only, and otherwise used AS
        // WRITTEN.  Trimming the surviving lines would be a divergence, not
        // a courtesy: the path is caller data, and the two engines must
        // open the same bytes.  (Both refuse identically on a list written
        // with a UTF-8 BOM, because both keep the BOM on the first line.)
        if line.trim().is_empty() {
            continue;
        }
        files.push(line.to_owned());
    }
    if files.is_empty() {
        return Err(missing_input(format!(
            "input list {path} names no files"
        )));
    }
    let mut seen = std::collections::BTreeSet::new();
    for entry in &files {
        if !seen.insert(entry.clone()) {
            return Err(usage(format!(
                "mapped source input list contains duplicates: {entry}"
            )));
        }
    }
    for entry in &files {
        if !PathBuf::from(entry).is_file() {
            return Err(missing_input(format!("no such input file: {entry}")));
        }
    }
    Ok(files)
}

/// `mapped_composition.INPUT_MANIFEST_SCHEMA`: the composition manifest,
/// which seals a mapping, a composition and every role-bound file.
pub const COMPOSITION_MANIFEST_SCHEMA: &str = "gpuwm-mapped-composition-inputs-v1";

/// `mapped_source._verify_input_manifest`, as far as the engine can see it:
/// the manifest's own digest, its schema, its mapping binding, and its
/// per-file identity.  Python still owns the authority-window recheck.
pub fn verify_input_manifest(
    manifest_path: &str,
    expected_sha256: &str,
    mapping: &Mapping,
    files: &[String],
    input_sha256: &BTreeMap<String, String>,
) -> Result<()> {
    let bytes = std::fs::read(manifest_path).map_err(|error| {
        missing_input(format!("cannot read input manifest {manifest_path}: {error}"))
    })?;
    let observed = crate::digest::bytes_sha256(&bytes);
    if observed != expected_sha256.to_ascii_lowercase() {
        return Err(manifest_mismatch(format!(
            "mapped input-manifest SHA mismatch: expected {expected_sha256}, got {observed}"
        )));
    }
    let document = Node::parse(&bytes).map_err(|error| {
        manifest_mismatch(format!("input manifest {manifest_path} is not JSON: {error}"))
    })?;
    let schema = document.get("schema").and_then(Node::as_str).unwrap_or("");
    if schema != "gpuwm-mapped-source-inputs-v1" {
        return Err(manifest_mismatch(format!(
            "unsupported mapped input manifest schema '{schema}'"
        )));
    }
    if document.get("mapping_sha256").and_then(Node::as_str) != Some(mapping.sha256.as_str()) {
        return Err(manifest_mismatch(
            "input manifest mapping SHA does not match mapping bytes",
        ));
    }
    let rows = document.get("files").map(Node::items).unwrap_or(&[]);
    if rows.len() != files.len() {
        return Err(manifest_mismatch(
            "input manifest file inventory differs from request",
        ));
    }
    for (index, (row, source)) in rows.iter().zip(files.iter()).enumerate() {
        let declared_bytes = row.get("bytes").and_then(Node::as_i64);
        let declared_sha = row.get("sha256").and_then(Node::as_str);
        let observed_size = std::fs::metadata(source)
            .map_err(|error| missing_input(format!("cannot stat {source}: {error}")))?
            .len() as i64;
        if declared_bytes != Some(observed_size) || declared_sha != input_sha256.get(source).map(String::as_str)
        {
            return Err(manifest_mismatch(format!(
                "input manifest identity differs for {source} (entry {index})"
            )));
        }
    }
    Ok(())
}

/// `mapped_composition._verify_manifest`, as far as the engine can see it.
///
/// The composition manifest is a DIFFERENT document from the decode
/// manifest above: schema `gpuwm-mapped-composition-inputs-v1`, with the
/// mapping and the composition both sealed, the primary inventory in
/// `primary_files`, and one row (or one list of rows) per declared
/// supplement, provenance and decoder role.  `_compose_through_engine`
/// already passes `--input-manifest`, so a compose that accepted the
/// argument and did not check it would report a seal it never read.
///
/// What is checked here is IDENTITY, exactly as the decode manifest is
/// checked: the manifest's own digest, the two contract digests against
/// the bytes this run loaded, and every named file's size and sha256
/// against the bytes this run is about to read.  The `path` strings are
/// deliberately not compared: they are written RELATIVE to the manifest
/// directory, and resolving them for comparison would canonicalize on
/// Windows and prepend the `\\?\` verbatim prefix this engine never
/// applies.  Python owns the path-equality half of the check and runs it
/// on the same manifest bytes moments before this call.
/// Returns the manifest's EXPLICIT ensemble-member binding when it
/// declares one: `(member, member_identity)`.  The pair is atomic --
/// gpuwm's `_verify_manifest` refuses a half-pair first, and this engine
/// refuses it again as defense for a hand-run exe -- and `compose` stamps
/// the member onto every canonical frame and into the alignment receipt,
/// which is how an archive whose product octets carry no ensemble
/// identity keeps the one its verified authority named.
#[allow(clippy::too_many_arguments)]
pub fn verify_composition_manifest(
    manifest_path: &str,
    expected_sha256: &str,
    mapping: &Mapping,
    composition_sha256: &str,
    primary: &[String],
    supplements: &BTreeMap<String, Vec<String>>,
    provenance: &BTreeMap<String, String>,
    digests: &BTreeMap<String, String>,
) -> Result<Option<(String, String)>> {
    let bytes = std::fs::read(manifest_path).map_err(|error| {
        missing_input(format!("cannot read input manifest {manifest_path}: {error}"))
    })?;
    let observed = crate::digest::bytes_sha256(&bytes);
    if observed != expected_sha256.to_ascii_lowercase() {
        return Err(manifest_mismatch(format!(
            "composition input-manifest SHA mismatch: expected {expected_sha256}, \
             got {observed}"
        )));
    }
    let document = Node::parse(&bytes).map_err(|error| {
        manifest_mismatch(format!("input manifest {manifest_path} is not JSON: {error}"))
    })?;
    let schema = document.get("schema").and_then(Node::as_str).unwrap_or("");
    if schema != COMPOSITION_MANIFEST_SCHEMA {
        return Err(manifest_mismatch(format!(
            "unsupported composition input manifest schema '{schema}'"
        )));
    }
    if document.get("mapping_sha256").and_then(Node::as_str) != Some(mapping.sha256.as_str()) {
        return Err(manifest_mismatch(
            "input manifest mapping SHA does not match mapping bytes",
        ));
    }
    if document.get("composition_sha256").and_then(Node::as_str) != Some(composition_sha256) {
        return Err(manifest_mismatch(
            "input manifest composition SHA does not match composition bytes",
        ));
    }
    let rows = document.get("primary_files").map(Node::items).unwrap_or(&[]);
    if rows.len() != primary.len() {
        return Err(manifest_mismatch(
            "manifest primary file inventory differs from the request",
        ));
    }
    for (index, (row, source)) in rows.iter().zip(primary.iter()).enumerate() {
        verify_manifest_identity(row, &format!("manifest.primary_files[{index}]"), source, digests)?;
    }
    let declared_roles = |section: &str| -> Vec<String> {
        document
            .get(section)
            .map(|node| {
                node.entries()
                    .iter()
                    .map(|(role, _)| role.clone())
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut supplement_roles = declared_roles("supplements");
    supplement_roles.sort();
    let requested: Vec<String> = supplements.keys().cloned().collect();
    if supplement_roles != requested {
        return Err(manifest_mismatch(format!(
            "manifest supplement role inventory {} differs from the request {}",
            crate::refusal::python_list_repr(&supplement_roles),
            crate::refusal::python_list_repr(&requested)
        )));
    }
    for (role, paths) in supplements {
        let node = document
            .get("supplements")
            .and_then(|section| section.get(role))
            .ok_or_else(|| manifest_mismatch(format!("manifest.supplements.{role} is absent")))?;
        // A single-file role may be written as ONE row rather than a
        // list of one; `_manifest_file_inventory` accepts both spellings
        // and manifests written before the list spelling carry the bare
        // row, so refusing it here would refuse a manifest Python
        // verifies.
        let rows: Vec<&Node> = if node.is_array() {
            node.items().iter().collect()
        } else {
            vec![node]
        };
        if rows.len() != paths.len() {
            return Err(manifest_mismatch(format!(
                "manifest.supplements.{role} file inventory differs from the request"
            )));
        }
        for (index, (row, source)) in rows.iter().zip(paths.iter()).enumerate() {
            verify_manifest_identity(
                row,
                &format!("manifest.supplements.{role}[{index}]"),
                source,
                digests,
            )?;
        }
    }
    let mut provenance_roles = declared_roles("provenance");
    provenance_roles.sort();
    let requested: Vec<String> = provenance.keys().cloned().collect();
    if provenance_roles != requested {
        return Err(manifest_mismatch(format!(
            "manifest provenance role inventory {} differs from the request {}",
            crate::refusal::python_list_repr(&provenance_roles),
            crate::refusal::python_list_repr(&requested)
        )));
    }
    for (role, source) in provenance {
        let row = document
            .get("provenance")
            .and_then(|section| section.get(role))
            .ok_or_else(|| manifest_mismatch(format!("manifest.provenance.{role} is absent")))?;
        verify_manifest_identity(row, &format!("manifest.provenance.{role}"), source, digests)?;
    }
    // The decoder section seals WHICH BINARY read the bytes.  This
    // engine decodes in process, so the run it seals has exactly one
    // decoder row and that row is this engine; a manifest naming the
    // subprocess pair was sealed for a decoder that is not running, and
    // replaying it here would produce evidence naming a binary that
    // never opened the file.
    let decoder_roles = declared_roles("decoders");
    if decoder_roles != vec![crate::ENGINE_NAME.to_owned()] {
        return Err(manifest_mismatch(format!(
            "manifest decoder inventory {} was sealed for a different decoder; \
             this run decodes in process as {}",
            crate::refusal::python_list_repr(&decoder_roles),
            crate::ENGINE_NAME
        )));
    }
    let member = document.get("member").and_then(Node::as_str);
    let member_identity = document.get("member_identity").and_then(Node::as_str);
    match (member, member_identity) {
        (None, None) => Ok(None),
        (Some(member), Some(identity))
            if !member.trim().is_empty() && !identity.trim().is_empty() =>
        {
            Ok(Some((member.to_owned(), identity.to_owned())))
        }
        _ => Err(manifest_mismatch(
            "manifest member and member_identity are an atomic pair of \
             non-empty strings",
        )),
    }
}

/// One `{path, bytes, sha256}` manifest row against the bytes in hand.
fn verify_manifest_identity(
    row: &Node,
    label: &str,
    source: &str,
    digests: &BTreeMap<String, String>,
) -> Result<()> {
    let declared_bytes = row.get("bytes").and_then(Node::as_i64);
    let declared_sha = row.get("sha256").and_then(Node::as_str);
    let observed_size = std::fs::metadata(source)
        .map_err(|error| missing_input(format!("cannot stat {source}: {error}")))?
        .len() as i64;
    if declared_bytes != Some(observed_size) || declared_sha != digests.get(source).map(String::as_str)
    {
        return Err(manifest_mismatch(format!(
            "input manifest identity differs for {source} ({label})"
        )));
    }
    Ok(())
}

/// The digests every frameset and inspection carries for its inputs.
///
/// Hashed concurrently: one file per slot, drained in document order.
/// The answer is a map keyed by path, so the schedule cannot reach it
/// even in principle, and the refusal for an unreadable file is still
/// the first one in the input list.
pub fn input_digests(files: &[String]) -> Result<BTreeMap<String, String>> {
    let slots: Vec<Result<(String, String)>> = crate::threads::install(|| {
        use rayon::prelude::*;
        files
            .par_iter()
            .map(|path| {
                crate::digest::file_sha256(path)
                    .map(|digest| (path.clone(), digest))
                    .map_err(|error| missing_input(format!("cannot hash {path}: {error}")))
            })
            .collect()
    });
    Ok(crate::threads::in_order(slots)?.into_iter().collect())
}

/// `mapped_source._decode_grib` / `_decode_netcdf`, dispatched by format.
pub fn decode_collection(
    mapping: &Mapping,
    files: &[String],
    progress: &mut dyn FnMut(Value),
) -> Result<DecodedCollection> {
    decode_collection_or_unmatched(mapping, files, progress)?
}

/// [`decode_collection`], with a GRIB2 decode that matched no record at
/// all handed back as its refusal instead of raised.
///
/// A caller with a declared answer for a field the files do not carry
/// (`fields.terrain_height.when_absent`) takes that answer only when the
/// decode matched nothing; any other failure is raised as it always was.
pub fn decode_collection_or_unmatched(
    mapping: &Mapping,
    files: &[String],
    progress: &mut dyn FnMut(Value),
) -> Result<std::result::Result<DecodedCollection, crate::refusal::Refusal>> {
    let format = mapping.format()?.to_owned();
    if format == "netcdf" {
        progress(json!({"event": "decode_netcdf", "files": files.len()}));
        return crate::ncdf::decode_netcdf(mapping, files).map(Ok);
    }
    if format == "grib1" {
        return decode_grib1_collection(mapping, files, progress).map(Ok);
    }
    let declaration = mapping.grid_declaration()?;
    let mut records: Vec<GribRecord> = Vec::new();
    let mut inventoried = 0usize;
    let mut identity_pins: Vec<crate::grib::RecordIdentity> = Vec::new();
    // One task per input object, into pre-assigned slots.  Files are
    // independent: each is read, staged through its acquisition codec,
    // parsed, inventoried and decoded on its own bytes, and the results
    // are concatenated in INPUT-LIST ORDER below -- record provenance
    // (`<path>:<index>`) and every later assembly step read that order,
    // so it is reproduced by the slot index rather than by the schedule.
    let outcomes: Vec<FileOutcome> = crate::threads::install(|| {
        use rayon::prelude::*;
        files
            .par_iter()
            .map(|source| decode_one_object(mapping, source, &declaration))
            .collect()
    });
    // Drained in order, so both the progress stream and the refusal are
    // exactly the serial engine's: an object that fails BEFORE it is
    // inventoried announces nothing (as the serial loop's `?` did), one
    // that fails after announces its inventory first.
    for (source, outcome) in files.iter().zip(outcomes) {
        let (identities, selected, decoded) = match outcome {
            FileOutcome::Unopened(refusal) => return Err(refusal),
            FileOutcome::Inventoried {
                identities,
                selected,
                records,
            } => (identities, selected, records),
        };
        inventoried += identities.len();
        progress(json!({
            "event": "inventory",
            "source": source,
            "messages": identities.len(),
            "selected": selected,
        }));
        identity_pins.extend(identities);
        records.extend(decoded?);
    }
    if records.is_empty() {
        // A total miss earns the identity diagnosis: two products under one
        // filename, separable only by the section-1 octets, must refuse by
        // naming them.
        return Ok(Err(crate::refusal::selector_unmatched(
            selector_identity_refusal(mapping, &identity_pins, files, inventoried)?,
        )));
    }
    assemble_grib(mapping, &records).map(Ok)
}

/// What ONE input object's inventory pass produced.
///
/// Deliberately holds no decoded array: the identities, the wanted
/// message indices grouped by the valid time each carries, and the
/// source cycle each valid time's messages agree on.  The parsed object
/// is dropped at the end of the pass, so a fourteen-object 3 km CONUS
/// preparation does not hold fourteen parsed objects to find out what is
/// in them.
struct ObjectInventory {
    identities: Vec<crate::grib::RecordIdentity>,
    selected: usize,
    by_key: BTreeMap<crate::assemble::TimeKey, Vec<usize>>,
    /// Wanted indices matching a field the mapping declares
    /// `cycle_invariant`: one record answers every valid time, so it is
    /// read into EVERY slice.
    invariant: Vec<usize>,
    cycles: BTreeMap<crate::assemble::TimeKey, chrono::NaiveDateTime>,
    /// Selected records read through `mapping.record_aliases`, counted
    /// per field they answer.
    aliased: BTreeMap<String, usize>,
    /// Per selected record (by message index): the cells its decode
    /// unpacks over the whole grid, and the fields it answers.  What
    /// prices the pool before anything is decoded
    /// ([`DecodeStream::first_time_estimate`]).
    answers: BTreeMap<usize, (u64, Vec<String>)>,
}

/// Staged bytes the inventory pass may hold IN FLIGHT at once, counted
/// on the supplied (still compressed) objects.
///
/// The named breakage this bound prevents is the one the serial loop was
/// written for: parsing every input at once held every staged payload of
/// a fourteen-object 3 km CONUS list resident before a single field had
/// been decoded.  A worker cap alone does not prevent it -- eight
/// half-gigabyte objects is the same accident with a smaller constant --
/// so admission is priced in BYTES and the worker cap applies inside
/// that.  The effect is that concurrency follows object size without
/// anyone naming a source: a list of small objects inventories at the
/// full pool width, a list of large ones collapses
/// to the serial pass this replaces, and a single object larger than the
/// budget is still admitted alone.
///
/// The number, chosen against measured object sizes rather than picked:
/// a compressed field-per-file object measures about 0.9 MiB, so 64 MiB
/// admits roughly seventy of them and the pass runs at the full worker
/// cap; a multi-message per-lead object measures about 20 MiB, so the
/// same budget admits three and the in-flight set stays close to the
/// serial pass this replaces.  The budget is spent on the SUPPLIED
/// bytes, so a compressed object's decompressed twin and parsed form sit
/// above it -- call the worst in-flight addition a small multiple of
/// 64 MiB, against a streamed peak of 3.9 to 4.0 GiB measured on the
/// widest source here (877 objects, seven valid times, a 10.4 GB
/// frameset).  Nothing in it names a source: object size is the
/// property, and a source that publishes small objects is exactly the
/// one whose inventory was serial.
const INVENTORY_STAGED_BYTES: u64 = 64 * 1024 * 1024;

/// The staged-byte budget one inventory batch is admitted under: the
/// [`INVENTORY_STAGED_BYTES`] floor, raised to a quarter of the memory
/// share the valid times in flight may use (a staged object's decoded
/// twin and parsed form sit at a small multiple of its supplied bytes).
///
/// Named breakage the raise removes: a per-lead 3 km CONUS object is
/// about 400 MB, so the fixed 64 MiB floor admitted ONE at a time and
/// the inventory of a 49-lead series ran serially -- one object read and
/// parsed after another, on one core -- before a single field was
/// decoded.  The floor still governs a box that cannot report its
/// memory, and a batch still runs on at most the pool's width at once.
fn inventory_budget() -> Result<u64> {
    Ok(crate::threads::usable_budget(0)?.map_or(INVENTORY_STAGED_BYTES, |bytes| bytes / 4))
}

/// How many of `sizes` the next inventory batch admits under `budget`.
///
/// Always at least one, so an object larger than the whole budget is
/// still inventoried -- alone, which is the serial pass -- rather than
/// stalling the walk.
fn inventory_batch_len(sizes: &[u64], budget: u64) -> usize {
    let mut admitted = 0u64;
    let mut taken = 0usize;
    for size in sizes {
        if taken > 0 && admitted.saturating_add(*size) > budget {
            break;
        }
        admitted = admitted.saturating_add(*size);
        taken += 1;
    }
    taken.max(1)
}

/// Inventory every input object, bounded-parallel, in DOCUMENT ORDER.
///
/// The performance breakage this replaces: the pass is what tells the
/// stream which object carries which valid time, so it runs before any
/// field is decoded, and on a compressed field-per-file source it was
/// the serial half of a decode that had otherwise gone parallel -- 37.5 s
/// of a 99.8 s preparation, against 27.6 s for the whole-decode engine
/// that had no separate pass at all.  An uncompressed multi-message
/// source never paid it (one object, one decompression that is a memcpy).
///
/// Determinism is the crate rule, not an argument made here: objects go
/// into PRE-ASSIGNED SLOTS and are drained in input-list order, so the
/// progress stream, the identity pins, the plan order and the first
/// refusal are the serial pass's regardless of the schedule.  Batches are
/// walked in order for the same reason -- a refusal in batch *n* is
/// returned before batch *n+1* is admitted, so the first refusal in
/// document order is still the one reported.
fn inventory_objects(
    mapping: &Mapping,
    files: &[String],
    invariant_fields: &std::collections::BTreeSet<String>,
) -> Result<Vec<ObjectInventory>> {
    // A path whose size cannot be read prices as zero and earns its
    // refusal from the read inside the pass, so the refusal sentence
    // stays the one the serial loop produced.
    let sizes: Vec<u64> = files
        .iter()
        .map(|path| {
            std::fs::metadata(path)
                .map(|meta| meta.len())
                .unwrap_or(0)
        })
        .collect();
    let mut inventories: Vec<ObjectInventory> = Vec::with_capacity(files.len());
    let mut start = 0usize;
    while start < files.len() {
        let budget = inventory_budget()?;
        if sizes[start] > budget {
            return Err(crate::refusal::host_memory(format!(
                "inventory of {} needs a staged object of {} bytes, but its usable staging budget is {budget} bytes; refusing before reading it",
                files[start], sizes[start])));
        }
        let end = start + inventory_batch_len(&sizes[start..], budget);
        let batch = &files[start..end];
        let slots: Vec<Result<ObjectInventory>> = crate::threads::install(|| {
            use rayon::prelude::*;
            batch
                .par_iter()
                .map(|source| inventory_one_object(mapping, source, invariant_fields))
                .collect()
        });
        inventories.extend(crate::threads::in_order(slots)?);
        start = end;
    }
    Ok(inventories)
}

/// Read, stage, parse and INVENTORY one input object; decode nothing.
fn inventory_one_object(
    mapping: &Mapping,
    source: &str,
    invariant_fields: &std::collections::BTreeSet<String>,
) -> Result<ObjectInventory> {
    let file = parse_grib2_object(source, true)?;
    let mut identities = grib2_identities(&file.messages);
    let aliased = crate::grib::alias_identities(
        &crate::grib::record_aliases(mapping)?,
        &mut identities,
    );
    let wanted = wanted_indices(mapping, &identities)?;
    let declared_levels = mapping.declared_levels()?;
    let interface_levels = mapping.interface_levels()?;
    let fields = mapping.fields()?;
    let mut inventory = ObjectInventory {
        selected: wanted.len(),
        identities,
        by_key: BTreeMap::new(),
        invariant: Vec::new(),
        cycles: BTreeMap::new(),
        aliased: BTreeMap::new(),
        answers: BTreeMap::new(),
    };
    for index in wanted {
        let identity = &inventory.identities[index];
        let mut dependent = false;
        let mut invariant = false;
        let mut answered: Vec<String> = Vec::new();
        for field in &fields {
            if field.derivation().is_some() {
                continue;
            }
            let hit = field
                .selectors()
                .iter()
                .any(|selector| crate::grib::selector_matches(selector, identity, "grib2"));
            if !hit
                || !crate::grib::declared_vertical_admits(
                    &declared_levels,
                    &interface_levels,
                    field,
                    identity.level_value,
                )?
            {
                continue;
            }
            if aliased[index] {
                *inventory.aliased.entry(field.name.clone()).or_default() += 1;
            }
            answered.push(field.name.clone());
            if invariant_fields.contains(&field.name) {
                invariant = true;
            } else {
                dependent = true;
            }
        }
        let message = &file.messages[index];
        if !answered.is_empty() {
            let grid = &message.grid;
            let cells = match u64::from(grid.nx).saturating_mul(u64::from(grid.ny)) {
                0 => u64::from(grid.num_data_points),
                cells => cells,
            };
            inventory.answers.insert(index, (cells, answered));
        }
        if invariant {
            inventory.invariant.push(index);
        }
        if !dependent {
            continue;
        }
        let valid_time = crate::grib::embedded_valid_time(
            message.reference_time,
            message.product.time_range_unit,
            message.product.forecast_time as i64,
            2,
        )?;
        let key = (valid_time, identity.member.clone());
        inventory.by_key.entry(key.clone()).or_default().push(index);
        inventory.cycles.entry(key).or_insert(message.reference_time);
    }
    Ok(inventory)
}

/// The per-object plan a streamed decode comes back to.
struct ObjectPlan {
    source: String,
    by_key: BTreeMap<crate::assemble::TimeKey, Vec<usize>>,
    invariant: Vec<usize>,
    answers: BTreeMap<usize, (u64, Vec<String>)>,
}

/// A decode that hands over ONE valid time at a time.
///
/// Named breakage, measured on real RRFS bytes (3 km CONUS, 45 pressure
/// levels): decoding the whole input list first held every object's raw
/// records AND the assembled series at once -- about 9 GiB per forcing
/// time -- so a seven-time preparation needed roughly 65 GiB of host
/// memory and the OOM reaper killed it on anything smaller.  Nothing in
/// the frameset write needs two valid times: the stream is written
/// sequentially and each frame's digests cover only its own arrays.
///
/// So the messages are INVENTORIED first -- identity octets only, no
/// unpacking -- which is what tells the stream which object carries which
/// valid time.  A valid time's messages are then read, decoded,
/// assembled, handed over and dropped before the next valid time is
/// asked for, and the peak follows ONE valid time plus the objects that
/// span it rather than the number of forcing times.
pub struct DecodeStream<'a> {
    mapping: &'a Mapping,
    kind: StreamKind,
    keys: Vec<crate::assemble::TimeKey>,
    summary: crate::frames::SeriesSummary,
    latitude: Vec<f64>,
    longitude: Vec<f64>,
    vertical_values: Vec<f64>,
    /// The resolved hybrid A/B ladder from the first slice (empty on any
    /// other vertical kind).  Kept so the cross-time check can refuse a
    /// series whose pv octets change between valid times: the
    /// whole-series assembly saw every record at once and refused that
    /// itself, and a streamed slice only ever sees its own.
    hybrid_a: Vec<f64>,
    hybrid_b: Vec<f64>,
    direct_names: std::collections::BTreeSet<String>,
    /// Selected records read through `mapping.record_aliases`, per field.
    aliased: BTreeMap<String, usize>,
    /// The atmospheric window the requester granted BEFORE the first
    /// record was decoded, when it granted one, and the fields whose
    /// records are therefore decoded over that window alone.
    decode_window: Option<crate::window::Window>,
    decode_fields: std::collections::BTreeSet<String>,
    /// Fields a pressure-level frame completes hydrostatically where the
    /// source leaves them out, watched on every window-decoded frame: a
    /// frame that leaves one out is decoded again whole
    /// ([`DecodeStream::completes_over_whole_columns`]).
    completion_watch: std::collections::BTreeSet<String>,
}

enum StreamKind {
    /// GRIB2: one valid time's messages are read when that valid time is
    /// asked for.
    Sliced {
        declaration: crate::model::GridDeclaration,
        plan: Vec<ObjectPlan>,
        parsed: BTreeMap<String, grib_core::grib2::Grib2File>,
        first: Box<Option<DecodedCollection>>,
    },
    /// Every other declared format: decoded whole once, then carved by
    /// valid time.  The carve MOVES each valid time's arrays out of the
    /// collection rather than copying them, so the writer still holds one
    /// frame at a time; what a whole-object format cannot do is avoid
    /// decoding the rest of the series first, and saying so here is
    /// cheaper than a second decoder that would drift from this one.
    Whole { collection: DecodedCollection },
}

impl<'a> DecodeStream<'a> {
    /// Inventory the inputs and establish the series header.
    pub fn open(
        mapping: &'a Mapping,
        files: &[String],
        progress: &mut dyn FnMut(Value),
    ) -> Result<Self> {
        Self::open_within(mapping, files, progress, false, &std::collections::BTreeSet::new())
    }

    /// [`DecodeStream::open`], asking the requester for the atmospheric
    /// window before any record is decoded when `request_window` is set.
    /// `keep_whole` names fields the caller reads over the whole source
    /// grid itself (a composition's terrain derivation reads the first
    /// valid time's columns beside full-grid surface fields), so they are
    /// never decoded over the window.
    pub fn open_within(
        mapping: &'a Mapping,
        files: &[String],
        progress: &mut dyn FnMut(Value),
        request_window: bool,
        keep_whole: &std::collections::BTreeSet<String>,
    ) -> Result<Self> {
        let format = mapping.format()?.to_owned();
        if format == "netcdf" {
            // A whole-object decode holds every listed object at once; it is
            // priced from the objects' own headers before any payload is read,
            // so a chained preparation can reserve and cap its later decodes
            // (box G 2026-10-23: an unpriced NetCDF decode refused the hex
            // parent's as-posted nest head until the whole parent had run).
            let price = crate::ncdf::whole_object_price(files)?;
            crate::threads::admit_width(price, &[], price)?;
            let collection = decode_collection(mapping, files, progress)?;
            return Ok(Self::whole(mapping, collection));
        }
        crate::threads::require_priced_format(&format)?;
        if format != "grib2" {
            let collection = decode_collection(mapping, files, progress)?;
            return Ok(Self::whole(mapping, collection));
        }
        let mut invariant_fields: std::collections::BTreeSet<String> =
            std::collections::BTreeSet::new();
        for field in mapping.fields()? {
            if field.time_binding() == Some("cycle_invariant") {
                invariant_fields.insert(field.name.clone());
            }
        }
        let declaration = mapping.grid_declaration()?;
        // Inventoried under a STAGED-BYTE BUDGET, not one object at a
        // time and not all of them at once.  The identity pass parses the
        // whole object, so parsing every input at once -- which is what
        // the decode-everything path did -- held every staged payload of
        // a fourteen-object 3 km CONUS list resident before a single
        // field had been decoded; the budget in
        // [`INVENTORY_STAGED_BYTES`] is what keeps that from coming back
        // while a list of small objects still inventories in parallel.
        let mut plan: Vec<ObjectPlan> = Vec::with_capacity(files.len());
        let mut identity_pins: Vec<crate::grib::RecordIdentity> = Vec::new();
        let mut inventoried = 0usize;
        let mut source_cycles: BTreeMap<crate::assemble::TimeKey, chrono::NaiveDateTime> =
            BTreeMap::new();
        let mut wanted_total = 0usize;
        let mut aliased: BTreeMap<String, usize> = BTreeMap::new();
        let inventories = inventory_objects(mapping, files, &invariant_fields)?;
        for (source, inventory) in files.iter().zip(inventories) {
            inventoried += inventory.identities.len();
            for (name, count) in &inventory.aliased {
                *aliased.entry(name.clone()).or_default() += count;
            }
            progress(json!({
                "event": "inventory",
                "source": source,
                "messages": inventory.identities.len(),
                "selected": inventory.selected,
            }));
            identity_pins.extend(inventory.identities);
            wanted_total += inventory.selected;
            for (key, cycle) in inventory.cycles {
                source_cycles.entry(key).or_insert(cycle);
            }
            plan.push(ObjectPlan {
                source: source.clone(),
                by_key: inventory.by_key,
                invariant: inventory.invariant,
                answers: inventory.answers,
            });
        }
        if wanted_total == 0 {
            return Err(crate::refusal::selector_unmatched(
                selector_identity_refusal(mapping, &identity_pins, files, inventoried)?,
            ));
        }
        if source_cycles.is_empty() {
            if invariant_fields.is_empty() {
                return Err(crate::refusal::selector_unmatched(
                    "no GRIB messages match the mapping selectors",
                ));
            }
            // Every matched record answers a `cycle_invariant` field, so
            // there is no time-dependent state to define the forcing
            // axis -- the same sentence `_broadcast_invariant_fields`
            // refuses with when it reaches the same state.
            return Err(crate::refusal::frame_invalid(
                "every decoded mapped field is declared time-invariant; there is \
                 no time-dependent state to define the forcing axis",
            ));
        }
        if !invariant_fields.is_empty() {
            // The whole-series broadcast check, made from the clock the
            // inventory read: one broadcast belongs to one cycle.
            let cycles: std::collections::BTreeSet<chrono::NaiveDateTime> =
                source_cycles.values().copied().collect();
            if cycles.len() > 1 {
                return Err(crate::refusal::frame_invalid(format!(
                    "cycle-invariant fields cannot broadcast across mixed source \
                     cycles {cycles:?}; one broadcast belongs to one cycle"
                )));
            }
        }
        let keys: Vec<crate::assemble::TimeKey> = source_cycles.keys().cloned().collect();
        let mut stream = DecodeStream {
            mapping,
            kind: StreamKind::Sliced {
                declaration,
                plan,
                parsed: BTreeMap::new(),
                first: Box::new(None),
            },
            keys,
            summary: crate::frames::SeriesSummary {
                source_cycles,
                grid_fingerprint: String::new(),
                lead_batch: false,
            },
            latitude: Vec::new(),
            longitude: Vec::new(),
            vertical_values: Vec::new(),
            hybrid_a: Vec::new(),
            hybrid_b: Vec::new(),
            direct_names: std::collections::BTreeSet::new(),
            aliased,
            decode_window: None,
            decode_fields: std::collections::BTreeSet::new(),
            completion_watch: std::collections::BTreeSet::new(),
        };
        if request_window {
            stream.request_decode_window(keep_whole)?;
        }
        // The pool is narrowed to what the first valid time fits beside
        // in memory BEFORE it is decoded, from the records the inventory
        // read: that decode is the first to run at full width, and
        // narrowing only after it was measured to hold more, not less
        // (`threads::admit_width`).
        let estimate = stream.first_time_estimate();
        let whole_estimate = stream.first_time_estimate_on(false);
        crate::threads::admit_width(stream.price(&estimate), &estimate, stream.price(&whole_estimate))?;
        // The first valid time establishes the header every cross-time
        // check and every join plan reads: the grid, the vertical ladder,
        // the fingerprint and the decoded field inventory.  It is kept
        // and handed out as the first slice, so it is decoded once.
        let first_key = stream.keys[0].clone();
        let second_key = stream.keys.get(1).cloned();
        let first = stream.decode_slice(&first_key, second_key.as_ref())?;
        stream.latitude = first.latitude.clone();
        stream.longitude = first.longitude.clone();
        stream.vertical_values = first.vertical_values.clone();
        stream.hybrid_a = first.hybrid_a.clone();
        stream.hybrid_b = first.hybrid_b.clone();
        stream.summary.grid_fingerprint = first.grid_fingerprint.clone();
        stream.direct_names = first
            .direct
            .keys()
            .map(|(_time, _member, name)| name.clone())
            .collect();
        if let StreamKind::Sliced { first: slot, .. } = &mut stream.kind {
            **slot = Some(first);
        }
        Ok(stream)
    }

    /// A stream over an ALREADY decoded collection.
    fn whole(mapping: &'a Mapping, collection: DecodedCollection) -> Self {
        let keys: Vec<crate::assemble::TimeKey> = collection.source_cycles.keys().cloned().collect();
        DecodeStream {
            mapping,
            latitude: collection.latitude.clone(),
            longitude: collection.longitude.clone(),
            vertical_values: collection.vertical_values.clone(),
            hybrid_a: collection.hybrid_a.clone(),
            hybrid_b: collection.hybrid_b.clone(),
            direct_names: collection
                .direct
                .keys()
                .map(|(_time, _member, name)| name.clone())
                .collect(),
            summary: crate::frames::SeriesSummary {
                source_cycles: collection.source_cycles.clone(),
                grid_fingerprint: collection.grid_fingerprint.clone(),
                lead_batch: false,
            },
            keys,
            aliased: BTreeMap::new(),
            decode_window: None,
            decode_fields: std::collections::BTreeSet::new(),
            completion_watch: std::collections::BTreeSet::new(),
            kind: StreamKind::Whole { collection },
        }
    }

    /// The window granted before decoding, which every frame publishes.
    pub fn decode_window(&self) -> Option<&crate::window::Window> {
        self.decode_window.as_ref()
    }

    /// Ask for the atmospheric window from Section 3 alone, before a
    /// single record is unpacked.
    ///
    /// THE BREAKAGE THIS PREVENTS: the window used to be asked for after
    /// a valid time had been decoded whole, so a source whose grid is
    /// the globe and whose domain is a few hundred kilometres decoded
    /// every cell of every level to publish the few it keeps; MSC GDPS
    /// (175 JPEG2000 fields of 2400x1201 per valid time) spent about
    /// 46 s per forecast time there on an 8 vCPU box.  The window
    /// depends only on the source axes and the target domains, and both
    /// are known now.
    ///
    /// A field is decoded over the window only when the requester
    /// granted it, it is a direct vertical/y/x field of a regular
    /// latitude/longitude grid or of the declared Lambert grid, and nothing
    /// reads it beside a full-grid field ([`decode_window_candidates`]).
    /// Anything else keeps the per-frame request and the full decode.
    ///
    /// A Lambert record is unpacked whole on its worker and cropped at
    /// once (`grib::grib2_record`), so the values are the full decode's.
    /// Named breakage this prevents: a 241x241 3 km domain on the HRRR
    /// native grid (1799x1059, 50 levels) priced 26.65 GB for one source
    /// time, because every atmospheric field was decoded and held over the
    /// whole CONUS grid before the per-frame window cropped it; a 5 GiB
    /// decode budget refused it and a 32 GiB host could not hold two.
    fn request_decode_window(&mut self, keep_whole: &std::collections::BTreeSet<String>) -> Result<()> {
        let mapping = self.mapping;
        // Mixed-grid union changes source index space. A rectangle from
        // the first record is not a rectangle on every published grid.
        // Decode these records in their own coordinates, normalize them
        // in assembly, then let the writer request its canonical crop.
        if mapping.doc.get("coordinates")
            .and_then(|n| n.get("horizontal")).and_then(|n| n.get("normalization"))
            .and_then(|n| n.as_str()).is_some_and(|v| ["coarsest_exact_subset", "largest_common_finite_rectangle"].contains(&v)) {
            return Ok(());
        }
        let first_key = self.keys[0].clone();
        let StreamKind::Sliced { declaration, plan, parsed, .. } = &mut self.kind else {
            return Ok(());
        };
        let Some(object) = plan.iter().find(|object| {
            object.by_key.get(&first_key).is_some_and(|wanted| !wanted.is_empty())
        }) else {
            return Ok(());
        };
        let index = object.by_key[&first_key][0];
        if !parsed.contains_key(&object.source) {
            parsed.insert(object.source.clone(), parse_grib2_object(&object.source, false)?);
        }
        let file = &parsed[&object.source];
        let message = &file.messages[index];
        // A projected grid's canonical axes and grid document come from
        // its declaration, checked against Section 3 here, exactly as the
        // frame header states them after the decode (`frames::source_grid`).
        let (latitude, longitude) = if declaration.is_projected() {
            let Some(axes) =
                crate::grib::declared_lambert_axes(message, index, declaration, &object.source)
            else {
                return Ok(());
            };
            axes
        } else {
            let Ok(axes) = crate::grib::regular_latlon_axes(message) else {
                return Ok(());
            };
            (axes.latitude, axes.longitude)
        };
        let (candidates, watch) = decode_window_candidates(mapping, keep_whole)?;
        let mut inventory: Vec<String> = Vec::new();
        for field in mapping.fields()? {
            if !field.dependency_only()? {
                inventory.push(field.name.clone());
            }
        }
        if candidates.is_empty() {
            return Ok(());
        }
        let grid = crate::frames::source_grid(declaration, &latitude, &longitude);
        let Some(window) = crate::window::request_for_source(
            &latitude,
            &longitude,
            &grid,
            &inventory,
            0,
            &crate::grib::grid_fingerprint(message),
        )?
        else {
            return Ok(());
        };
        if window.rows == [0, latitude.len()] && window.columns == [0, longitude.len()] {
            return Err(crate::refusal::frame_invalid(
                "an atmospheric window that covers the whole source must use full mode",
            ));
        }
        self.decode_fields = window.fields.intersection(&candidates).cloned().collect();
        self.decode_window = Some(window);
        // The watch matters only when a completion operand, or the field
        // it completes, is decoded over the window.
        let operands: std::collections::BTreeSet<&str> = crate::derive::HYPSOMETRIC_OPERANDS
            .iter()
            .map(|(name, _)| *name)
            .chain(watch.iter().map(String::as_str))
            .collect();
        if self.decode_fields.iter().any(|name| operands.contains(name.as_str())) {
            self.completion_watch = watch;
        }
        Ok(())
    }

    /// Whether a frame decoded over the window must be decoded again
    /// whole: it leaves out (at some level, or entirely) a field the
    /// frame completes hydrostatically, and that completion reads whole
    /// columns of the operands beside the full-grid surface pressure and
    /// terrain.  A frame that publishes every such field whole reads
    /// none of that, and keeps its window.
    fn completes_over_whole_columns(&self, collection: &DecodedCollection,
                                    key: &crate::assemble::TimeKey) -> bool {
        self.completion_watch.iter().any(|name| {
            !matches!(collection.direct.get(&(key.0, key.1.clone(), name.clone())),
                      Some(direct) if direct.missing_count == 0)
        })
    }

    /// Selected records read through `mapping.record_aliases`, per field
    /// they answer; empty when the mapping declares none or none applied.
    pub fn aliased(&self) -> &BTreeMap<String, usize> {
        &self.aliased
    }

    /// The first valid time's decoded collection, while it is still held
    /// for the writer.  `None` once it has been handed out, and on a
    /// whole-object format, which keeps no separate first slice.
    pub fn first_slice(&self) -> Option<&DecodedCollection> {
        match &self.kind {
            StreamKind::Sliced { first, .. } => first.as_ref().as_ref(),
            StreamKind::Whole { .. } => None,
        }
    }

    /// Hand out the first valid time's collection, decoded when the
    /// stream opened, for a writer that pulls valid times by position
    /// ([`Self::slice_detached`] for every later one).
    pub fn take_first(&mut self) -> Option<DecodedCollection> {
        match &mut self.kind {
            StreamKind::Sliced { first, .. } => first.take(),
            StreamKind::Whole { .. } => None,
        }
    }

    /// Whether every valid time can be decoded on its own, from `&self`:
    /// a GRIB2 series in which no object is read by more than one valid
    /// time and none carries a cycle-invariant record every slice needs.
    ///
    /// That is the shape of every per-lead publication (one object per
    /// forecast hour).  A series that shares objects between valid times
    /// keeps the one-lane decode, which parses a shared object once and
    /// hands it on; decoding such a series in parallel would parse the
    /// shared object once per valid time in flight.
    pub fn times_are_independent(&self) -> bool {
        match &self.kind {
            StreamKind::Sliced { plan, .. } => plan
                .iter()
                .all(|object| object.by_key.len() <= 1 && object.invariant.is_empty()),
            StreamKind::Whole { .. } => false,
        }
    }

    /// What ONE valid time holds between its decode and its write, for
    /// sizing how many run at once: twice the first valid time's decoded
    /// arrays (the unpacked records and the assembled arrays coexist
    /// until assembly returns) plus three times the largest input object
    /// (its bytes, its staged payload and its parsed form).
    ///
    /// A field decoded over the granted window is counted over the whole
    /// source grid when a frame of this stream can be decoded again
    /// whole ([`Self::completes_over_whole_columns`]).  Named breakage:
    /// sized on the first valid time's window alone, the lanes of a
    /// series whose later frames leave a watched height level out would
    /// each re-decode their frame whole at once -- about seventeen times
    /// the windowed size on a GDPS regional domain -- far past the
    /// memory share the lanes were sized to fit.
    pub fn per_time_bytes(&self) -> u64 {
        self.price(&self.field_bytes())
    }

    /// Twice `field_bytes` plus three times the largest input object:
    /// what one valid time holds, as [`Self::per_time_bytes`] states it.
    fn price(&self, field_bytes: &[u64]) -> u64 {
        let StreamKind::Sliced { plan, .. } = &self.kind else {
            return 0;
        };
        let decoded = field_bytes.iter().copied().fold(0u64, u64::saturating_add);
        let object = plan
            .iter()
            .map(|object| std::fs::metadata(&object.source).map(|meta| meta.len()).unwrap_or(0))
            .max()
            .unwrap_or(0);
        decoded.saturating_mul(2).saturating_add(object.saturating_mul(3))
    }

    /// What the first valid time's decoded arrays hold now, in bytes,
    /// while they wait to be written; 0 once they have been handed out.
    pub fn held_bytes(&self) -> u64 {
        self.first_slice().map_or(0, |collection| {
            collection
                .direct
                .values()
                .map(|value| (value.values.len() as u64).saturating_mul(8))
                .sum()
        })
    }

    /// The first valid time's decoded fields, in bytes, largest first,
    /// each counted as [`Self::per_time_bytes`] counts it (over the whole
    /// source grid when a frame of this stream can be decoded again
    /// whole); empty on a whole-object format.  What the lanes are
    /// priced with (`threads::series_price`).
    pub fn field_bytes(&self) -> Vec<u64> {
        let StreamKind::Sliced { first, .. } = &self.kind else {
            return Vec::new();
        };
        let whole = self.decode_window.as_ref().filter(|_| !self.completion_watch.is_empty());
        let mut sizes: Vec<u64> = first.as_ref().as_ref().map_or_else(Vec::new, |collection| {
            collection
                .direct
                .iter()
                .map(|((_time, _member, name), value)| {
                    let bytes = (value.values.len() as u64).saturating_mul(8);
                    let shape = value.values.shape();
                    match whole {
                        Some(window)
                            if self.decode_fields.contains(name)
                                && shape.len() >= 2
                                && shape[shape.len() - 2..]
                                    == [window.rows[1] - window.rows[0],
                                        window.columns[1] - window.columns[0]]
                                && shape[shape.len() - 2..] != window.source_shape =>
                        {
                            let cells = (shape[shape.len() - 2] * shape[shape.len() - 1]) as u64;
                            let source = (window.source_shape[0] * window.source_shape[1]) as u64;
                            (bytes / cells).saturating_mul(source)
                        }
                        _ => bytes,
                    }
                })
                .collect()
        });
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }

    /// [`Self::field_bytes`] before anything is decoded: the first valid
    /// time's records (and every cycle-invariant one, which each valid
    /// time reads) summed per field they answer, at eight bytes a cell,
    /// over the granted window for a field decoded over it and over the
    /// whole grid otherwise, as the decode will unpack them.
    pub fn first_time_estimate(&self) -> Vec<u64> {
        self.first_time_estimate_on(true)
    }

    fn first_time_estimate_on(&self, windowed: bool) -> Vec<u64> {
        let StreamKind::Sliced { plan, .. } = &self.kind else {
            return Vec::new();
        };
        let Some(first_key) = self.keys.first() else {
            return Vec::new();
        };
        let window = self
            .decode_window
            .as_ref()
            .filter(|_| windowed && self.completion_watch.is_empty())
            .map(|window| {
                ((window.rows[1] - window.rows[0]) * (window.columns[1] - window.columns[0])) as u64
            });
        let mut by_field: BTreeMap<&str, u64> = BTreeMap::new();
        for object in plan {
            let mut wanted: Vec<usize> = object.by_key.get(first_key).cloned().unwrap_or_default();
            wanted.extend(object.invariant.iter().copied());
            wanted.sort_unstable();
            wanted.dedup();
            for index in wanted {
                let Some((cells, fields)) = object.answers.get(&index) else {
                    continue;
                };
                for name in fields {
                    let cells = match window {
                        Some(window) if self.decode_fields.contains(name) => window,
                        _ => *cells,
                    };
                    let bytes = by_field.entry(name.as_str()).or_default();
                    *bytes = bytes.saturating_add(cells.saturating_mul(8));
                }
            }
        }
        let mut sizes: Vec<u64> = by_field.into_values().collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }

    /// ONE valid time decoded on its own: nothing is cached between
    /// calls, so several may run at once.  Only for a stream whose
    /// [`Self::times_are_independent`] holds; the result is the one
    /// [`Self::slice`] returns for the same key, checked against the
    /// series header the same way: the records the granted window names
    /// are decoded over it alone, and a frame that completes a watched
    /// field is decoded again whole, exactly as [`Self::decode_slice`]
    /// does on the one-lane path.
    pub fn slice_detached(&self, key: &crate::assemble::TimeKey) -> Result<DecodedCollection> {
        if !self.keys.iter().any(|candidate| candidate == key) {
            return Err(crate::refusal::frame_invalid(format!(
                "the mapped decode has no valid time {}",
                crate::frames::naive_isoformat(key.0)
            )));
        }
        let collection = self.decode_records_detached(key, true)?;
        let collection = if self.decode_window.is_some()
            && self.completes_over_whole_columns(&collection, key)
        {
            drop(collection);
            self.decode_records_detached(key, false)?
        } else {
            collection
        };
        self.require_one_series(&collection)?;
        Ok(collection)
    }

    /// [`Self::decode_records`] for [`Self::slice_detached`]: every object
    /// the valid time reads is parsed here and dropped with it, so no
    /// state is shared between valid times in flight.
    fn decode_records_detached(
        &self,
        key: &crate::assemble::TimeKey,
        windowed: bool,
    ) -> Result<DecodedCollection> {
        let StreamKind::Sliced { declaration, plan, .. } = &self.kind else {
            return Err(crate::refusal::frame_invalid(
                "a whole-object decode has no detached valid times",
            ));
        };
        let mapping = self.mapping;
        let decode_window = if windowed { self.decode_window.as_ref() } else { None };
        let aliases = crate::grib::record_aliases(mapping)?;
        let mut records: Vec<GribRecord> = Vec::new();
        let mut start = 0;
        while start < plan.len() {
            let end = start + object_batch_len(&plan[start..])?;
            let slots = crate::threads::install(|| {
                use rayon::prelude::*;
                plan[start..end].par_iter().map(|object| {
                    decode_stream_object(mapping, object, key, declaration, &aliases,
                                         decode_window, &self.decode_fields, None)
                        .map(|(_, records)| records)
                }).collect::<Vec<_>>()
            });
            for decoded in slots {
                records.extend(decoded?);
            }
            start = end;
        }
        let collection = crate::assemble::assemble_grib_within(
            mapping, &records, decode_window.map(|window| (window.rows, window.columns)))?;
        drop(records);
        Ok(collection)
    }

    /// The `(valid_time, member)` keys, in frameset order.
    pub fn keys(&self) -> &[crate::assemble::TimeKey] {
        &self.keys
    }

    /// The series facts the frameset states above its frames.
    pub fn summary(&self) -> &crate::frames::SeriesSummary {
        &self.summary
    }

    pub fn latitude(&self) -> &[f64] {
        &self.latitude
    }

    pub fn longitude(&self) -> &[f64] {
        &self.longitude
    }

    pub fn vertical_values(&self) -> &[f64] {
        &self.vertical_values
    }

    /// The field names the primary decode resolves, from the first slice.
    pub fn direct_names(&self) -> &std::collections::BTreeSet<String> {
        &self.direct_names
    }

    /// ONE valid time's decoded collection.
    pub fn slice(&mut self, key: &crate::assemble::TimeKey) -> Result<DecodedCollection> {
        let position = self
            .keys
            .iter()
            .position(|candidate| candidate == key)
            .ok_or_else(|| {
                crate::refusal::frame_invalid(format!(
                    "the mapped decode has no valid time {}",
                    crate::frames::naive_isoformat(key.0)
                ))
            })?;
        if let StreamKind::Sliced { first, .. } = &mut self.kind
            && position == 0
                && let Some(collection) = first.take() {
                    return Ok(collection);
                }
        let next = self.keys.get(position + 1).cloned();
        let collection = self.decode_slice(key, next.as_ref())?;
        self.require_one_series(&collection)?;
        Ok(collection)
    }

    /// The cross-time checks the whole-series assembly used to make.
    fn require_one_series(&self, collection: &DecodedCollection) -> Result<()> {
        if collection.grid_fingerprint != self.summary.grid_fingerprint {
            return Err(crate::refusal::frame_invalid(
                "selected GRIB fields do not share one source grid",
            ));
        }
        if collection.latitude != self.latitude || collection.longitude != self.longitude {
            return Err(crate::refusal::frame_invalid(
                "selected GRIB coordinate axes differ",
            ));
        }
        if collection.vertical_values != self.vertical_values {
            return Err(crate::refusal::frame_invalid(
                "GRIB atmospheric fields do not share one complete vertical inventory",
            ));
        }
        // The whole-series assembly saw every record's pv octets at once
        // and refused a ladder that changed between them; a streamed
        // slice resolves its own ladder alone, so the disagreement is
        // caught here, with the same sentence.
        if collection.hybrid_a != self.hybrid_a || collection.hybrid_b != self.hybrid_b {
            return Err(crate::refusal::frame_invalid(
                "selected GRIB records do not share one pv coordinate \
                 list; hybrid A/B coefficients must be identical across \
                 the source",
            ));
        }
        Ok(())
    }

    fn decode_slice(
        &mut self,
        key: &crate::assemble::TimeKey,
        next: Option<&crate::assemble::TimeKey>,
    ) -> Result<DecodedCollection> {
        let collection = self.decode_records(key, next, true)?;
        if self.decode_window.is_some() && self.completes_over_whole_columns(&collection, key) {
            // A completion would pair window-decoded columns with the
            // full-grid surface pressure and terrain, so this one frame is
            // decoded whole and cropped by the writer, as every frame was
            // before the window was granted ahead of the decode.
            drop(collection);
            return self.decode_records(key, next, false);
        }
        Ok(collection)
    }

    /// One valid time's records, assembled; with `windowed`, the records
    /// the granted window names are decoded over it alone.
    fn decode_records(
        &mut self,
        key: &crate::assemble::TimeKey,
        next: Option<&crate::assemble::TimeKey>,
        windowed: bool,
    ) -> Result<DecodedCollection> {
        let mapping = self.mapping;
        let decode_window = if windowed { self.decode_window.as_ref() } else { None };
        match &mut self.kind {
            StreamKind::Whole { collection } => Ok(carve(collection, key)),
            StreamKind::Sliced {
                declaration,
                plan,
                parsed,
                ..
            } => {
                let aliases = crate::grib::record_aliases(mapping)?;
                let mut records: Vec<GribRecord> = Vec::new();
                let mut start = 0;
                while start < plan.len() {
                    let end = start + object_batch_len(&plan[start..])?;
                    // Move each cached parse to its own task. There is no shared
                    // cache lock around decompression or message decode.
                    let jobs: Vec<_> = plan[start..end].iter()
                        .map(|object| (object, parsed.remove(&object.source))).collect();
                    let slots = crate::threads::install(|| {
                        use rayon::prelude::*;
                        jobs.into_par_iter().map(|(object, file)| {
                            decode_stream_object(mapping, object, key, declaration, &aliases,
                                                 decode_window, &self.decode_fields, file)
                        }).collect::<Vec<_>>()
                    });
                    // Drain in input order: bytes, provenance and the first
                    // refusal are the serial stream's, regardless of schedule.
                    for (object, slot) in plan[start..end].iter().zip(slots) {
                        let (file, decoded) = slot?;
                        records.extend(decoded);
                        let needed_again = next
                            .map(|next| object.by_key.contains_key(next)).unwrap_or(false)
                            || !object.invariant.is_empty();
                        if needed_again
                            && let Some(file) = file {
                                parsed.insert(object.source.clone(), file);
                            }
                    }
                    start = end;
                }
                crate::assemble::assemble_grib_within(
                    mapping, &records, decode_window.map(|window| (window.rows, window.columns)))
            }
        }
    }
}

/// Bound in-flight staged objects by both the admitted pool width and
/// the inventory byte budget. The streamed decoded records still belong
/// to one valid time, and cached multi-time objects retain their old lifetime.
fn object_batch_len(plan: &[ObjectPlan]) -> Result<usize> {
    let sizes: Vec<_> = plan.iter().take(crate::threads::width()).map(|object| {
        std::fs::metadata(&object.source).map(|meta| meta.len()).unwrap_or(0)
    }).collect();
    Ok(inventory_batch_len(&sizes, inventory_budget()?).min(sizes.len()))
}

/// A field-per-file stream used to decompress its objects serially even
/// though grib2_records_within parallelized the messages inside each one.
/// The R11 case's cold compose spent 5 s on one core here. An object is now
/// a pool task, with its parse moved back to the ordered cache when needed.
#[allow(clippy::too_many_arguments)]
fn decode_stream_object(
    mapping: &Mapping,
    object: &ObjectPlan,
    key: &crate::assemble::TimeKey,
    declaration: &crate::model::GridDeclaration,
    aliases: &[crate::grib::RecordAlias],
    decode_window: Option<&crate::window::Window>,
    decode_fields: &std::collections::BTreeSet<String>,
    cached: Option<grib_core::grib2::Grib2File>,
) -> Result<(Option<grib_core::grib2::Grib2File>, Vec<GribRecord>)> {
    let mut wanted = object.by_key.get(key).cloned().unwrap_or_default();
    wanted.extend(object.invariant.iter().copied());
    wanted.sort_unstable();
    wanted.dedup();
    if wanted.is_empty() {
        return Ok((cached, Vec::new()));
    }
    let file = match cached {
        Some(file) => file,
        None => parse_grib2_object(&object.source, false)?,
    };
    let windowed = match decode_window {
        Some(_) => windowed_records(mapping, &file, &wanted, aliases, decode_fields)?,
        None => std::collections::BTreeSet::new(),
    };
    let ladder = match decode_window {
        Some(_) => ladder_records(mapping, &file, &wanted, aliases, decode_fields)?,
        None => std::collections::BTreeSet::new(),
    };
    let record_window = decode_window.map(|window| crate::grib::RecordWindow {
        rows: window.rows, columns: window.columns, records: &windowed, ladder: &ladder,
    });
    let mut decoded = crate::grib::grib2_records_within(
        &file, &object.source, &wanted, declaration, record_window.as_ref())?;
    crate::grib::alias_records(aliases, &mut decoded);
    Ok((Some(file), decoded))
}

/// Read, stage and parse one GRIB2 object. Inventory checks its envelopes
/// and nonempty message set; streamed decodes have already inventoried it.
fn parse_grib2_object(source: &str, inventory: bool) -> Result<grib_core::grib2::Grib2File> {
    let raw = std::fs::read(source)
        .map_err(|error| missing_input(format!("cannot read {source}: {error}")))?;
    let payload = crate::codec::decoded_payload(raw, source)?;
    if inventory {
        validate_grib2_envelopes(&payload, source)?;
    }
    let file = grib_core::grib2::Grib2File::from_bytes(&payload).map_err(|error| {
        crate::refusal::decode_failed(format!("GRIB2 parse failed for {source}: {error}"))
    })?;
    if inventory && file.messages.is_empty() {
        return Err(crate::refusal::decode_failed(format!(
            "GRIB2 input {source} contains no parsed fields"
        )));
    }
    Ok(file)
}

/// The fields a granted window may decode over itself alone, and the
/// fields a frame completes hydrostatically that every window-decoded
/// frame is watched for.
///
/// A candidate is a direct vertical/y/x field that nothing reads beside a
/// full-grid field:
///
/// * no derivation reads it (a derivation's other operands keep the
///   full grid, so its operands must too);
/// * it is not `air_pressure`, whose whole plane gives the per-level
///   pressures the frame publishes (`original_pressure_hpa`), unless that
///   ladder survives the window ([`pressure_ladder_survives_window`]).
///   Named breakage: a direct pressure field decoded over the window was
///   published without that ladder, and the reader could not open the
///   frameset (ICON-D2 publishes air_pressure directly);
/// * it is not in `keep_whole`, what the caller reads whole itself;
/// * it is not an operand of a hydrostatic completion whose need the
///   window cannot see.  A completed field (geopotential height on a
///   pressure ladder) that the source publishes directly under the
///   reject or value policy leaves NaN only on the levels it does not
///   publish, which are NaN over the window too: such a field is
///   watched instead, and a frame that leaves it out is decoded whole
///   ([`DecodeStream::completes_over_whole_columns`]).  A completed
///   field that is derived, not declared, or under a policy that keeps
///   NaN cells can need its completion outside the window alone, so its
///   operands and the field itself are never decoded over the window.
///   Named breakage: the completion then read window-decoded columns
///   beside full-grid surface pressure, and every pressure-level source
///   given a window refused a frame that left a height level out.
pub fn decode_window_candidates(
    mapping: &Mapping,
    keep_whole: &std::collections::BTreeSet<String>,
) -> Result<(std::collections::BTreeSet<String>, std::collections::BTreeSet<String>)> {
    let fields = mapping.fields()?;
    let mut whole: std::collections::BTreeSet<String> = keep_whole.clone();
    if !pressure_ladder_survives_window(mapping, &fields)? {
        whole.insert("air_pressure".to_owned());
    }
    for field in &fields {
        let Some(derivation) = field.derivation() else { continue };
        let operation = mapping.derivation(derivation).ok_or_else(|| {
            crate::refusal::mapping_invalid(format!(
                "field {} names unknown derivation '{derivation}'",
                field.name
            ))
        })?;
        if let Some(names) =
            crate::derive::derivation_dependencies(operation, mapping.vertical()?, &field.name)?
        {
            whole.extend(names);
        }
    }
    let required: std::collections::BTreeSet<String> =
        mapping.required_field_names()?.into_iter().collect();
    let mut watch: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for name in crate::derive::completed_fields(mapping.vertical_kind()?, &required) {
        let declared = fields.iter().find(|field| field.name == name);
        let visible = match declared {
            Some(field) if field.derivation().is_none() => {
                matches!(field.missing_kind()?, "reject" | "value")
            }
            _ => false,
        };
        if visible {
            watch.insert(name.to_owned());
        } else {
            whole.insert(name.to_owned());
            whole.extend(
                crate::derive::HYPSOMETRIC_OPERANDS.iter().map(|(operand, _)| (*operand).to_owned()),
            );
        }
    }
    let mut candidates: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for field in &fields {
        if field.derivation().is_none()
            && field.source_axes()? == ["vertical", "y", "x"]
            && field.target_axes()? == ["vertical", "y", "x"]
            && !whole.contains(&field.name)
        {
            candidates.insert(field.name.clone());
        }
    }
    Ok((candidates, watch))
}

/// Whether `air_pressure` may be decoded over the window and still state
/// the whole-plane ladder the frame publishes: a direct field of a
/// projected grid, whose every record is unpacked whole on its worker so
/// its plane is medianed before the crop (`grib::RecordWindow::ladder`),
/// read by at most one selector and with no selector or unit scale or
/// offset, under the reject policy, so its canonical values are the
/// decoded ones and the median of the raw plane is the median the full
/// decode states.  A regular-grid window decode never unpacks the whole
/// plane, so there the field stays whole.
///
/// Named breakage: held whole, the 50-level HRRR native pressure was
/// about 2.3 GB of the 5.36 GB a 241x241 3 km domain priced for one
/// source time, so a 5 GiB decode budget still refused the preparation
/// after every other atmospheric field had been windowed.
fn pressure_ladder_survives_window(
    mapping: &Mapping,
    fields: &[crate::model::FieldSpec<'_>],
) -> Result<bool> {
    if !mapping.grid_declaration()?.is_projected() {
        return Ok(false);
    }
    let Some(field) = fields.iter().find(|field| field.name == "air_pressure") else {
        return Ok(false);
    };
    let selectors = field.selectors();
    Ok(field.derivation().is_none()
        && selectors.len() <= 1
        && selectors.iter().all(|selector| selector.field("scale").is_none())
        && field.unit_scale() == 1.0
        && field.unit_offset() == 0.0
        && field.missing_kind()? == "reject")
}

/// The windowed records of one object that answer `air_pressure`: each is
/// medianed over its whole plane as it is decoded (`grib::RecordWindow`).
fn ladder_records(
    mapping: &Mapping,
    file: &grib_core::grib2::Grib2File,
    wanted: &[usize],
    aliases: &[crate::grib::RecordAlias],
    decode_fields: &std::collections::BTreeSet<String>,
) -> Result<std::collections::BTreeSet<usize>> {
    if !decode_fields.contains("air_pressure") {
        return Ok(std::collections::BTreeSet::new());
    }
    let pressure = std::collections::BTreeSet::from(["air_pressure".to_owned()]);
    windowed_records(mapping, file, wanted, aliases, &pressure)
}

/// The wanted records of one object that answer ONLY fields decoded over
/// the window, matched the way assembly matches them (aliased identity,
/// selectors, declared levels).  A record that also answers a full-grid
/// field is decoded whole.
fn windowed_records(
    mapping: &Mapping,
    file: &grib_core::grib2::Grib2File,
    wanted: &[usize],
    aliases: &[crate::grib::RecordAlias],
    decode_fields: &std::collections::BTreeSet<String>,
) -> Result<std::collections::BTreeSet<usize>> {
    let mut identities = grib2_identities(&file.messages);
    crate::grib::alias_identities(aliases, &mut identities);
    let source_format = mapping.format()?.to_owned();
    let declared_levels = mapping.declared_levels()?;
    let interface_levels = mapping.interface_levels()?;
    let fields = mapping.fields()?;
    let mut windowed = std::collections::BTreeSet::new();
    for index in wanted {
        let identity = &identities[*index];
        let mut answers: Vec<&str> = Vec::new();
        for field in &fields {
            if field.derivation().is_some() {
                continue;
            }
            let hit = field
                .selectors()
                .iter()
                .any(|selector| crate::grib::selector_matches(selector, identity, &source_format));
            if hit
                && crate::grib::declared_vertical_admits(
                    &declared_levels,
                    &interface_levels,
                    field,
                    identity.level_value,
                )?
            {
                answers.push(field.name.as_str());
            }
        }
        if !answers.is_empty() && answers.iter().all(|name| decode_fields.contains(*name)) {
            windowed.insert(*index);
        }
    }
    Ok(windowed)
}

/// One valid time MOVED out of a whole decoded collection.
pub fn carve_valid_time(
    collection: &mut DecodedCollection,
    key: &crate::assemble::TimeKey,
) -> DecodedCollection {
    carve(collection, key)
}

/// One valid time MOVED out of a whole decoded collection.
fn carve(collection: &mut DecodedCollection, key: &crate::assemble::TimeKey) -> DecodedCollection {
    let mine: Vec<crate::assemble::DirectKey> = collection
        .direct
        .keys()
        .filter(|(time, member, _name)| (*time, member.clone()) == *key)
        .cloned()
        .collect();
    let mut direct = BTreeMap::new();
    for entry in mine {
        if let Some(value) = collection.direct.remove(&entry) {
            direct.insert(entry, value);
        }
    }
    let mut source_cycles = BTreeMap::new();
    if let Some(cycle) = collection.source_cycles.get(key) {
        source_cycles.insert(key.clone(), *cycle);
    }
    DecodedCollection {
        latitude: collection.latitude.clone(),
        longitude: collection.longitude.clone(),
        vertical_values: collection.vertical_values.clone(),
        direct,
        source_cycles,
        grid_fingerprint: collection.grid_fingerprint.clone(),
        // The carved valid time keeps the whole decode's vertical
        // identity, its hybrid coefficient ladder included: the frame
        // header reads the ladder off the collection it materializes.
        hybrid_a: collection.hybrid_a.clone(),
        hybrid_b: collection.hybrid_b.clone(),
        original_pressure_hpa: None,
    }
}

/// `mapped_source._decode_grib`'s edition-1 arm.
///
/// Deliberately NOT the GRIB2 arm above, and the differences are the
/// Python engine's, not simplifications:
///
///   * every usable message becomes a record, with no selector pre-filter.
///     `_grib1_records` hands `_assemble_grib` the whole object and lets
///     assembly do the matching, so an object none of whose messages match
///     earns assembly's "no GRIB messages match the mapping selectors"
///     rather than the GRIB2 arm's producer-identity diagnosis (which
///     reads section-1 octets edition 1 does not carry);
///   * there is no grid DECLARATION cross-check.  A GRIB1 mapping declares
///     `embedded_grid`; the lambert declaration path is GRIB2-only.
///
/// Objects are decoded one at a time, in input-list order (the messages
/// inside one object are what the work is parallel over) so record
/// provenance (`<path>:<index>`) and the first refusal are the serial
/// engine's by construction.
fn decode_grib1_collection(
    mapping: &Mapping,
    files: &[String],
    progress: &mut dyn FnMut(Value),
) -> Result<DecodedCollection> {
    let mut records: Vec<GribRecord> = Vec::new();
    for source in files {
        let raw = std::fs::read(source)
            .map_err(|error| missing_input(format!("cannot read {source}: {error}")))?;
        // Acquisition codec staging, as on the GRIB2 arm: the compressed
        // object is what the caller hashed, the decompressed twin is what
        // the parser reads, and provenance stays bound to the supplied path.
        let payload = crate::codec::decoded_payload(raw, source)?;
        let (messages, decoded) = crate::grib1::grib1_records(&payload, source)?;
        progress(json!({
            "event": "inventory",
            "source": source,
            "messages": messages,
            "selected": decoded.len(),
        }));
        records.extend(decoded);
    }
    assemble_grib(mapping, &records)
}

/// What one input object's decode task produced.
///
/// Two shapes, because the serial loop had two: an object that failed
/// before its inventory line was printed announced nothing, and an
/// object that failed after it had already announced.  Reproducing that
/// split is what keeps the progress stream identical under threads.
enum FileOutcome {
    /// Failed at read, codec staging, envelope hygiene, parse or
    /// selection, before anything was announced for this object.
    Unopened(Refusal),
    /// Inventoried.  The records either decoded or refused.
    Inventoried {
        identities: Vec<crate::grib::RecordIdentity>,
        selected: usize,
        records: Result<Vec<GribRecord>>,
    },
}

/// Read, stage, parse, inventory and decode ONE input object.
fn decode_one_object(
    mapping: &Mapping,
    source: &str,
    declaration: &crate::model::GridDeclaration,
) -> FileOutcome {
    #[allow(clippy::type_complexity)]
    let opened = (|| -> Result<(
        grib_core::grib2::Grib2File,
        Vec<crate::grib::RecordIdentity>,
        Vec<usize>,
        Vec<crate::grib::RecordAlias>,
    )> {
        let file = parse_grib2_object(source, true)?;
        let aliases = crate::grib::record_aliases(mapping)?;
        let mut identities = grib2_identities(&file.messages);
        crate::grib::alias_identities(&aliases, &mut identities);
        let wanted = wanted_indices(mapping, &identities)?;
        Ok((file, identities, wanted, aliases))
    })();
    match opened {
        Err(refusal) => FileOutcome::Unopened(refusal),
        Ok((file, identities, wanted, aliases)) => FileOutcome::Inventoried {
            selected: wanted.len(),
            identities,
            // The object is parsed ONCE and the selected messages are
            // decoded from it.  The serial path parsed the same bytes a
            // second time inside `grib2_records`, holding two parsed
            // copies of a half-gigabyte object at the peak.
            records: crate::grib::grib2_records(&file, source, &wanted, declaration).map(
                |mut records| {
                    crate::grib::alias_records(&aliases, &mut records);
                    records
                },
            ),
        },
    }
}

/// `mapped_source._selector_identity_refusal`.
fn selector_identity_refusal(
    mapping: &Mapping,
    identities: &[crate::grib::RecordIdentity],
    files: &[String],
    inventoried: usize,
) -> Result<String> {
    let keys = [
        "center",
        "subcenter",
        "master_table_version",
        "local_table_version",
    ];
    let mut pins: BTreeMap<&str, std::collections::BTreeSet<i64>> = BTreeMap::new();
    for field in mapping.fields()? {
        for selector in field.selectors() {
            for key in keys {
                if let Some(value) = selector.field(key).and_then(Node::as_i64) {
                    pins.entry(key).or_default().insert(value);
                }
            }
        }
    }
    let mut mismatched: Vec<String> = Vec::new();
    for key in keys {
        let Some(pinned) = pins.get(key).filter(|values| values.len() == 1) else {
            continue;
        };
        let observed: std::collections::BTreeSet<i64> = identities
            .iter()
            .filter_map(|identity| match key {
                "center" => identity.center,
                "subcenter" => identity.subcenter,
                "master_table_version" => identity.master_table_version,
                _ => identity.local_table_version,
            })
            .collect();
        if !observed.is_empty() && observed.intersection(pinned).count() == 0 {
            let seen: Vec<String> = observed.iter().map(i64::to_string).collect();
            mismatched.push(format!(
                "every mapping selector pins {key}={} but every supplied message \
                 observes {key}={}",
                pinned.iter().next().expect("one pinned value"),
                seen.join("/")
            ));
        }
    }
    let names: Vec<&str> = files
        .iter()
        .map(|path| {
            path.rsplit(['/', '\\'])
                .next()
                .unwrap_or(path.as_str())
        })
        .collect();
    let base = format!(
        "0 of {inventoried} GRIB message(s) in {} match this mapping's selectors",
        names.join(", ")
    );
    if mismatched.is_empty() {
        return Ok(base);
    }
    Ok(format!(
        "{base}; the producer-identity octets explain it: {} -- these bytes are \
         a DIFFERENT product line published under the same file naming, and \
         decoding them here would silently mix model versions; the profile's \
         provenance document names the front door that serves the pinned identity",
        mismatched.join("; ")
    ))
}

/// `decode`: one mapping, N inputs, one frameset directory.
pub fn run_decode(invocation: &Invocation, progress: &mut dyn FnMut(Value)) -> Result<Value> {
    let mapping = Mapping::load(&invocation.mapping)?;
    let files = read_input_list(&invocation.input_list)?;
    let digests = input_digests(&files)?;
    if let (Some(manifest), Some(expected)) = (
        invocation.input_manifest.as_ref(),
        invocation.input_manifest_sha256.as_ref(),
    ) {
        verify_input_manifest(manifest, expected, &mapping, &files, &digests)?;
    }
    let mut stream = DecodeStream::open_within(
        &mapping,
        &files,
        progress,
        invocation.atmospheric_window,
        &std::collections::BTreeSet::new(),
    )?;
    progress(json!({"event": "assembled", "valid_times": stream.keys().len()}));
    let output = PathBuf::from(invocation.output.as_ref().expect("decode requires --output"));
    // The series is decoded, materialized and written ONE VALID TIME AT
    // A TIME.  Holding the whole decode here to hand the writer was the
    // raw records plus the assembled arrays of every forcing time at
    // once -- about 9 GiB per time on a 3 km CONUS source.
    let summary = stream.summary().clone();
    let grid_fingerprint = summary.grid_fingerprint.clone();
    // A window granted before the first record was decoded is published
    // by every frame, on either writer.
    let decode_window = stream.decode_window().cloned();
    // Several valid times at once when every one decodes on its own
    // (see `compose::run_compose`), still written one at a time.
    let admitted_times = if stream.times_are_independent() { stream.keys().len() } else { 1 };
    let lanes = crate::threads::lanes(
        stream.per_time_bytes(), &stream.field_bytes(), stream.held_bytes(), admitted_times)?;
    let overlap_write = lanes == 1 && stream.keys().len() > 1 && crate::threads::overlap_write(
        stream.per_time_bytes(), &stream.field_bytes(), stream.held_bytes())?;
    let document = if lanes > 1 {
        let first = std::sync::Mutex::new(stream.take_first());
        let stream = &stream;
        crate::frames::write_frameset_lanes(
            &output, &mapping, &summary, &digests, invocation.atmospheric_window, decode_window,
            lanes, |index, key| {
                let held = if index == 0 {
                    first.lock().ok().and_then(|mut slot| slot.take())
                } else {
                    None
                };
                match held {
                    Some(collection) => Ok(collection),
                    None => stream.slice_detached(key),
                }
            },
        )?
    } else {
        crate::frames::write_frameset_one_lane(
            &output, &mapping, &summary, &digests, invocation.atmospheric_window, decode_window,
            overlap_write, |key| stream.slice(key),
        )?
    };
    let frame_count = document
        .get("frames")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    Ok(json!({
        "event": "receipt",
        "subcommand": "decode",
        "schema": document["schema"],
        "frames": frame_count,
        "output": output.display().to_string(),
        "grid_fingerprint": grid_fingerprint,
        "stream_bytes": document
            .get("stream")
            .and_then(|stream| stream.get("bytes"))
            .cloned()
            .unwrap_or(Value::Null),
    }))
}

/// `inspect`: the `gpuwm-mapped-source-inspection-v1` document.
pub fn run_inspect(invocation: &Invocation, progress: &mut dyn FnMut(Value)) -> Result<Value> {
    let mapping = Mapping::load(&invocation.mapping)?;
    let files = read_input_list(&invocation.input_list)?;
    let digests = input_digests(&files)?;
    if let (Some(manifest), Some(expected)) = (
        invocation.input_manifest.as_ref(),
        invocation.input_manifest_sha256.as_ref(),
    ) {
        verify_input_manifest(manifest, expected, &mapping, &files, &digests)?;
    }
    let collection = decode_collection(&mapping, &files, progress)?;

    let mut direct_names: Vec<String> = Vec::new();
    for field in mapping.fields()? {
        if field.derivation().is_none() {
            direct_names.push(field.name.clone());
        }
    }
    direct_names.sort();
    let mut frame_rows = Vec::new();
    for (valid_time, member) in collection.source_cycles.keys() {
        let mut decoded: BTreeMap<&str, &crate::assemble::DirectValue> = BTreeMap::new();
        for ((time, member_value, name), value) in &collection.direct {
            if time == valid_time && member_value == member {
                decoded.insert(name.as_str(), value);
            }
        }
        let unresolved: Vec<&String> = direct_names
            .iter()
            .filter(|name| !decoded.contains_key(name.as_str()))
            .collect();
        // One entry per decoded field, measured CONCURRENTLY: the
        // sha256 and the extrema are per-field work over disjoint
        // arrays.  The indexed collect keeps entry i with field i, and
        // the map is then filled in the same sorted order the serial
        // loop used.  The extrema stream over the field rather than
        // collecting its finite cells into a second array first -- on a
        // 3-km frameset that copy was a gigabyte per field for two
        // numbers.
        let rows: Vec<(&str, Value)> = crate::threads::install(|| {
            use rayon::prelude::*;
            decoded
                .iter()
                .map(|(name, value)| (*name, *value))
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(name, value)| {
                    let flat = crate::array::contiguous(&value.values);
                    (
                        name,
                        json!({
                            "axes": value.axes,
                            "shape": value.values.shape(),
                            "minimum": flat
                                .iter()
                                .copied()
                                .filter(|item| item.is_finite())
                                .reduce(f64::min),
                            "maximum": flat
                                .iter()
                                .copied()
                                .filter(|item| item.is_finite())
                                .reduce(f64::max),
                            "missing": value.missing_count,
                            "sha256": crate::digest::array_sha256(value.values.shape(), &flat),
                            "source_references": value.references,
                        }),
                    )
                })
                .collect()
        });
        let mut fields = serde_json::Map::new();
        for (name, row) in rows {
            fields.insert(name.to_owned(), row);
        }
        frame_rows.push(json!({
            "valid_time": crate::frames::naive_isoformat(*valid_time),
            "source_cycle": crate::frames::naive_isoformat(
                collection.source_cycles[&(*valid_time, member.clone())]
            ),
            "member": member,
            "decoded_direct_fields": decoded.keys().collect::<Vec<_>>(),
            "unresolved_direct_fields": unresolved,
            "fields": Value::Object(fields),
        }));
    }

    let libm_dependent = crate::portable::libm_dependent_fields(&mapping);
    let materialization = match crate::frames::materialize_frames(&mapping, &collection) {
        Ok(frames) => json!({
            "verdict": "PASS",
            "frame_count": frames.len(),
            "frame_header_sha256": frames
                .iter()
                .map(|frame| crate::digest::bytes_sha256(
                    canonical_json(&frame.header).as_bytes()
                ))
                .collect::<Vec<String>>(),
            // Beside the raw digest, never instead of it: the raw one is
            // an identity of THIS box (input paths, and a derived field's
            // libm last bits); the portable one is what another box can
            // reproduce, and therefore what a recorded golden may assert.
            "frame_header_sha256_portable": frames
                .iter()
                .map(|frame| crate::portable::portable_frame_header_sha256(
                    &frame.header, &files, &libm_dependent
                ))
                .collect::<Vec<String>>(),
            "portable_rule": crate::portable::PORTABLE_HEADER_RULE,
        }),
        Err(refusal) => json!({
            "verdict": "INCOMPLETE",
            "error_class": refusal.class,
            "error": refusal.message,
        }),
    };
    let status = if materialization["verdict"] == "PASS" {
        "CANONICAL_FRAMES_MATERIALIZED_NOT_STOCK_WRF_CERTIFIED"
    } else {
        "DECODED_INCOMPLETE_NOT_STOCK_WRF_CERTIFIED"
    };
    Ok(json!({
        "schema": crate::INSPECTION_SCHEMA,
        "status": status,
        "stock_wrf_certified": false,
        "mapping": {"path": mapping.path, "sha256": mapping.sha256},
        "inputs": files
            .iter()
            .map(|path| json!({
                "path": path,
                "bytes": std::fs::metadata(path).map(|data| data.len()).unwrap_or(0),
                "sha256": digests[path],
            }))
            .collect::<Vec<Value>>(),
        "decoders": {"engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION}},
        "source_format": mapping.format()?,
        "grid": {
            "ny": collection.latitude.len(),
            "nx": collection.longitude.len(),
            "vertical_count": collection.vertical_values.len(),
            "fingerprint": collection.grid_fingerprint,
        },
        "frames": frame_rows,
        "materialization": materialization,
    }))
}

/// `inventory`: the raw per-record GRIB2 product identity of every input.
///
/// This is the engine's answer to the one question the subprocess
/// `grib2_inventory` used to be resolved for on a composed route: WHAT
/// product is in each file, octet for octet -- authority, process,
/// time semantics, level pair, member octet, grid definition, packing.
/// Nothing is decoded; identity sections only.
///
/// Every value is a STRING in the exact spelling the subprocess tool's
/// TSV rendered (`0x40` scan modes, `true`/`false` bitmaps, `-` for an
/// absent PDT octet, shortest-round-trip floats), so an archive-contract
/// gate that moved onto this surface reads one spelling whichever
/// instrument measured it, and the two instruments can be compared
/// column for column on the same bytes.
pub fn run_inventory(invocation: &Invocation) -> Result<Value> {
    use grib_core::grib2::{level_name, parameter_name};

    let files = read_input_list(&invocation.input_list)?;
    let digests = input_digests(&files)?;
    let mut file_rows: Vec<Value> = Vec::with_capacity(files.len());
    for source in &files {
        let raw = std::fs::read(source)
            .map_err(|error| missing_input(format!("cannot read {source}: {error}")))?;
        let delivered_bytes = raw.len();
        // Acquisition codec staging, exactly as decode stages it: the
        // delivered object is what the caller hashed, the decompressed
        // twin is what the parser reads.
        let payload = crate::codec::decoded_payload(raw, source)?;
        let envelopes = validate_grib2_envelopes(&payload, source)?;
        let file = grib_core::grib2::Grib2File::from_bytes(&payload).map_err(|error| {
            crate::refusal::decode_failed(format!("GRIB2 parse failed for {source}: {error}"))
        })?;
        if file.messages.is_empty() {
            return Err(crate::refusal::decode_failed(format!(
                "GRIB2 input {source} contains no parsed fields"
            )));
        }
        let mut records: Vec<Value> = Vec::with_capacity(file.messages.len());
        for (index, message) in file.messages.iter().enumerate() {
            let absent_or = |value: Option<u8>| {
                value.map_or_else(|| "-".to_owned(), |value| value.to_string())
            };
            records.push(json!({
                "index": index.to_string(),
                "discipline": message.discipline.to_string(),
                "category": message.product.parameter_category.to_string(),
                "parameter": message.product.parameter_number.to_string(),
                "center": message.identification.center_id.to_string(),
                "subcenter": message.identification.subcenter_id.to_string(),
                "master_table_version":
                    message.identification.master_table_version.to_string(),
                "local_table_version":
                    message.identification.local_table_version.to_string(),
                "name": parameter_name(
                    message.discipline,
                    message.product.parameter_category,
                    message.product.parameter_number,
                ),
                "reference_time": message.reference_time.to_string(),
                "forecast_unit": message.product.time_range_unit.to_string(),
                "forecast_time": message.product.forecast_time.to_string(),
                "pdt": message.product.template.to_string(),
                "level_type": message.product.level_type.to_string(),
                "level_value": message.product.level_value.to_string(),
                "second_level_type":
                    message.product.second_level_type.to_string(),
                "second_level_value":
                    message.product.second_level_value.to_string(),
                "member": absent_or(message.product.perturbation_number),
                "generating_process":
                    message.product.generating_process.to_string(),
                "forecast_generating_process_id":
                    message.product.forecast_generating_process_id.to_string(),
                "level_name": level_name(message.product.level_type),
                "gdt": message.grid.template.to_string(),
                "nx": message.grid.nx.to_string(),
                "ny": message.grid.ny.to_string(),
                "lat1": message.grid.lat1.to_string(),
                "lon1": message.grid.lon1.to_string(),
                "dx": message.grid.dx.to_string(),
                "dy": message.grid.dy.to_string(),
                "latin1": message.grid.latin1.to_string(),
                "latin2": message.grid.latin2.to_string(),
                "lov": message.grid.lov.to_string(),
                "scan_mode": format!("0x{:02x}", message.grid.scan_mode),
                "shape_of_earth": message.grid.shape_of_earth.to_string(),
                "resolution_flags":
                    format!("0x{:02x}", message.grid.resolution_flags),
                "drt": message.data_rep.template.to_string(),
                "bitmap": message.bitmap.is_some().to_string(),
                "ensemble_type": absent_or(message.product.ensemble_type),
                "ensemble_size":
                    absent_or(message.product.num_forecasts_in_ensemble),
                "derived_forecast":
                    absent_or(message.product.derived_forecast_type),
            }));
        }
        file_rows.push(json!({
            "path": source,
            "bytes": delivered_bytes,
            "sha256": digests[source],
            "envelopes": envelopes,
            "records": records,
        }));
    }
    Ok(json!({
        "schema": crate::RECORD_INVENTORY_SCHEMA,
        "engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION},
        "files": file_rows,
    }))
}


/// `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False)`.
pub fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(entries) => {
            let mut keys: Vec<&String> = entries.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .iter()
                .map(|key| {
                    format!(
                        "{}:{}",
                        Value::String((*key).clone()),
                        canonical_json(&entries[*key])
                    )
                })
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A packaged mapping, edited by `edit`, loaded from a scratch copy.
    fn packaged_mapping(name: &str, edit: impl FnOnce(&mut Value)) -> Mapping {
        let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap();
        let mut document: Value = serde_json::from_slice(
            &std::fs::read(repository.join("gpuwm").join("authorities").join(name)).unwrap(),
        )
        .unwrap();
        edit(&mut document);
        let path = std::env::temp_dir().join(format!(
            "gpuwm-candidates-{}-{}-{name}",
            std::process::id(),
            document.to_string().len()
        ));
        std::fs::write(&path, document.to_string()).unwrap();
        let mapping = Mapping::load(&path.display().to_string()).unwrap();
        let _ = std::fs::remove_file(&path);
        mapping
    }

    fn names(items: &[&str]) -> std::collections::BTreeSet<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn a_projected_native_pressure_is_decoded_over_the_window_with_its_ladder() {
        let none = std::collections::BTreeSet::new();
        // HRRR native publishes pressure directly on its Lambert grid: each
        // plane is unpacked whole and medianed before its crop, so the
        // field is decoded over the window like the other operands.
        let hrrr = packaged_mapping("rw-wps-hrrr-native-grib2.mapping.json", |_| {});
        let (candidates, watch) = decode_window_candidates(&hrrr, &none).unwrap();
        for name in ["air_pressure", "air_temperature", "eastward_wind", "northward_wind",
                     "specific_humidity", "geopotential_height"] {
            assert!(candidates.contains(name), "{name} is not a window candidate");
        }
        assert!(watch.is_empty(), "a model-level frame completes nothing");
        // A scaled pressure would not median to the full decode's ladder.
        let scaled = packaged_mapping("rw-wps-hrrr-native-grib2.mapping.json", |document| {
            document["fields"]["air_pressure"]["units"]["scale"] = serde_json::json!(100.0);
        });
        let (candidates, _) = decode_window_candidates(&scaled, &none).unwrap();
        assert!(!candidates.contains("air_pressure") && candidates.contains("air_temperature"));
        // What the caller reads whole stays whole.
        let kept = names(&["air_pressure"]);
        let (candidates, _) = decode_window_candidates(&hrrr, &kept).unwrap();
        assert!(!candidates.contains("air_pressure"));
    }

    #[test]
    fn a_decode_window_keeps_whole_every_field_read_beside_the_full_grid() {
        let none = std::collections::BTreeSet::new();
        // GEM GDPS publishes its heights directly under the reject policy:
        // a height level it leaves out is NaN over the window too, so its
        // columns are decoded over the window and the heights watched.
        let gdps = packaged_mapping("rw-wps-gem-gdps-grib2.mapping.json", |_| {});
        let (candidates, watch) = decode_window_candidates(&gdps, &none).unwrap();
        assert_eq!(
            candidates,
            names(&["air_temperature", "eastward_wind", "geopotential_height",
                    "northward_wind", "specific_humidity"])
        );
        assert_eq!(watch, names(&["geopotential_height"]));
        // What the caller reads whole (a composition's terrain derivation)
        // is never decoded over the window.
        let kept = names(&["air_temperature", "geopotential_height", "specific_humidity"]);
        let (candidates, _) = decode_window_candidates(&gdps, &kept).unwrap();
        assert_eq!(candidates, names(&["eastward_wind", "northward_wind"]));
        // A pressure field the source publishes directly: its whole plane
        // gives the frame's per-level pressures, so it is decoded whole.
        let direct = packaged_mapping("rw-wps-gem-gdps-grib2.mapping.json", |document| {
            let field = &mut document["fields"]["air_pressure"];
            field.as_object_mut().unwrap().remove("derivation");
            field["selectors"] = serde_json::json!([{"format": "grib2", "discipline": 0,
                "category": 3, "parameter": 0, "level_type": 100}]);
        });
        let (candidates, _) = decode_window_candidates(&direct, &none).unwrap();
        assert!(!candidates.contains("air_pressure"));
        assert!(candidates.contains("air_temperature"));
        let icon_d2 = packaged_mapping("rw-wps-icon-d2-grib2.mapping.json", |_| {});
        let (candidates, watch) = decode_window_candidates(&icon_d2, &none).unwrap();
        assert!(!candidates.contains("air_pressure") && candidates.contains("air_temperature"));
        assert!(watch.is_empty(), "a model-level frame completes nothing");
        // ICON-EU derives its heights: whether a completion runs is not
        // visible over the window, so its operands stay whole.
        let icon_eu = packaged_mapping("rw-wps-icon-eu-regular-grib2.mapping.json", |_| {});
        let (candidates, watch) = decode_window_candidates(&icon_eu, &none).unwrap();
        assert_eq!(candidates, names(&["eastward_wind", "northward_wind"]));
        assert!(watch.is_empty());
        // A height field the mapping leaves out entirely is completed on
        // every frame, so the same holds.
        let absent = packaged_mapping("rw-wps-gem-gdps-grib2.mapping.json", |document| {
            document["fields"].as_object_mut().unwrap().remove("geopotential_height");
        });
        let (candidates, watch) = decode_window_candidates(&absent, &none).unwrap();
        assert_eq!(candidates, names(&["eastward_wind", "northward_wind"]));
        assert!(watch.is_empty());
    }

    #[test]
    fn the_manifest_pair_is_atomic() {
        let arguments: Vec<String> = [
            "decode",
            "--mapping",
            "m.json",
            "--input-list",
            "f.txt",
            "--output",
            "out",
            "--input-manifest",
            "mf.json",
        ]
        .iter()
        .map(|item| (*item).to_owned())
        .collect();
        let refusal = Invocation::parse(&arguments).unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::USAGE);
        assert!(refusal.message.contains("atomic pair"));
    }

    #[test]
    fn role_bindings_take_role_equals_path() {
        let arguments: Vec<String> = [
            "compose",
            "--mapping",
            "m.json",
            "--input-list",
            "f.txt",
            "--output",
            "out",
            "--supplement",
            "terrain=/tmp/t.nc",
            "--provenance",
            "orography=/tmp/o.json",
        ]
        .iter()
        .map(|item| (*item).to_owned())
        .collect();
        let invocation = Invocation::parse(&arguments).unwrap();
        assert_eq!(
            invocation.supplements,
            vec![("terrain".to_owned(), "/tmp/t.nc".to_owned())]
        );
        assert_eq!(
            invocation.provenance,
            vec![("orography".to_owned(), "/tmp/o.json".to_owned())]
        );
    }

    #[test]
    fn a_binding_without_an_equals_sign_refuses_with_the_grammar() {
        let arguments: Vec<String> = [
            "compose", "--mapping", "m.json", "--input-list", "f.txt", "--output", "out",
            "--supplement", "terrain",
        ]
        .iter()
        .map(|item| (*item).to_owned())
        .collect();
        let refusal = Invocation::parse(&arguments).unwrap_err();
        assert!(refusal.message.contains("ROLE=PATH"));
    }

    #[test]
    fn inspect_needs_no_output_directory() {
        let arguments: Vec<String> = ["inspect", "--mapping", "m.json", "--input-list", "f.txt"]
            .iter()
            .map(|item| (*item).to_owned())
            .collect();
        assert!(Invocation::parse(&arguments).is_ok());
    }

    #[test]
    fn small_objects_batch_together_and_large_ones_do_not() {
        // The property the bounded-parallel inventory rests on: batch
        // width follows OBJECT SIZE, so a source that publishes one
        // small object per field inventories many at a time while a
        // source that publishes one large object per lead stays close to
        // the serial pass.  Neither is named anywhere; the size decides.
        let budget = 64 * 1024 * 1024;
        let small = vec![1024 * 1024u64; 200];
        assert_eq!(inventory_batch_len(&small, budget), 64);
        let large = vec![20 * 1024 * 1024u64; 7];
        assert_eq!(inventory_batch_len(&large, budget), 3);
    }

    #[test]
    fn an_object_larger_than_the_whole_budget_is_still_admitted_alone() {
        // Named breakage: a batch that admits nothing never advances,
        // and the walk would hang on the first object bigger than the
        // budget instead of inventorying it the way the serial pass did.
        let budget = 64 * 1024 * 1024;
        let sizes = vec![512 * 1024 * 1024u64, 1024];
        assert_eq!(inventory_batch_len(&sizes, budget), 1);
        // Unreadable sizes price as zero and must not collapse the walk.
        assert_eq!(inventory_batch_len(&[0, 0, 0], budget), 3);
        assert_eq!(inventory_batch_len(&[u64::MAX, u64::MAX], budget), 1);
    }

    #[test]
    fn inventory_takes_the_input_list_alone() {
        // Raw record identity has no mapping to resolve against and no
        // frameset to write; demanding either would make a
        // product-identity question depend on documents it never reads.
        let arguments: Vec<String> = ["inventory", "--input-list", "f.txt"]
            .iter()
            .map(|item| (*item).to_owned())
            .collect();
        assert!(Invocation::parse(&arguments).is_ok());
    }

    #[test]
    fn inventory_refuses_a_mapping() {
        let arguments: Vec<String> = [
            "inventory", "--input-list", "f.txt", "--mapping", "m.json",
        ]
        .iter()
        .map(|item| (*item).to_owned())
        .collect();
        let refusal = Invocation::parse(&arguments).unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::USAGE);
        assert!(refusal.message.contains("no --mapping"), "{refusal}");
    }

    #[test]
    fn the_capabilities_document_declares_the_inventory_surface() {
        let document = run_capabilities();
        assert_eq!(
            document["subcommands"]["inventory"],
            serde_json::json!(["grib2"])
        );
    }

    #[test]
    fn canonical_json_sorts_keys_and_drops_whitespace() {
        let value = json!({"b": 1, "a": [1, {"d": 2, "c": 3}]});
        assert_eq!(canonical_json(&value), r#"{"a":[1,{"c":3,"d":2}],"b":1}"#);
    }
}
