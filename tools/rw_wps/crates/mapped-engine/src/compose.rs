//! Cross-source composition.
//!
//! `compose` mirrors `gpuwm.mapped_composition.decode_composed_source`:
//! a base mapping plus a `gpuwm-mapped-composition-v2` document, role-bound
//! supplements and provenance, terrain composition, bound-field subsetting
//! and cross-source partition decode.
//!
//! **What is here and what stays in Python.**  The BYTE WORK is here:
//! partition decode, the coordinate-subset solve, terrain composition, the
//! bound-field borrow, the union mapping, frame materialization and the two
//! evidence products only that work can produce.  Everything above it stays
//! in `mapped_composition` and runs around this call â€”  manifest
//! verification, the authority hash window, the role inventory, the
//! contributing-mapping hash pins, the primary-versus-donor field
//! agreement, the soil-layer contract against the donor table, and the two
//! post-return checks (the frames' `mapping_sha256`, and "did the engine
//! compose exactly the declared bindings").
//!
//! The document grammar below is therefore a SECOND reader of a contract
//! Python has already validated on the same bytes: through the front door
//! `load_composition` runs first and refuses first.  It is ported anyway
//! because the engine must not proceed on a contract it has not read, and
//! it is transcribed condition for condition so it can never refuse a
//! composition Python accepts â€” a stricter engine here would refuse a
//! preparation that is in fact correct.  Where Python raises `TypeError`
//! for a non-object, the engine refuses `mapping_invalid` (`ValueError`);
//! that divergence is reachable only by calling the exe directly with a
//! document the Python door would have refused first.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::engine::Invocation;
use crate::model::Mapping;
use crate::node::Node;
use crate::refusal::{mapping_invalid, missing_input, python_list_repr, python_repr, Result};

pub const COMPOSITION_SCHEMA: &str = "gpuwm-mapped-composition-v2";
/// `mapped_composition.PENDING_COMPOSITION_SCHEMA`.
pub const PENDING_COMPOSITION_SCHEMA: &str = "gpuwm-cross-source-composition-pending-v1";
/// What `mapped_engine_bridge.read_composition_evidence` reads.
pub const COMPOSITION_EVIDENCE_SCHEMA: &str = "gpuwm-mapped-composition-evidence-v1";
/// The one field a terrain supplement may provide.
pub const EXTERNAL_FIELD: &str = "terrain_height";
/// `mapped_composition._SOIL_PAIR`.
pub const SOIL_PAIR: [&str; 2] = ["soil_temperature", "volumetric_soil_moisture"];
/// `mapped_composition._BOUND_GRID_ALIGNMENTS`.
pub const BOUND_GRID_ALIGNMENTS: [&str; 4] = [
    "exact_coordinate_subset", "exact_coordinate_strided_subset", "nearest_coordinate", crate::regrid::ALIGNMENT,
];

const COMPOSITION_KEYS: [&str; 7] = [
    "schema",
    "name",
    "mapping_binding",
    "soil_layers",
    "supplements",
    "field_sources",
    // Read by gpuwm.ingest.upper_air_donor after the WPS export; compose ignores it.
    "upper_air_donor",
];
const COMPOSITION_REQUIRED: [&str; 5] = [
    "schema",
    "name",
    "mapping_binding",
    "soil_layers",
    "supplements",
];
const TERRAIN_KEYS: [&str; 8] = [
    "data_role",
    "provenance_role",
    "format",
    "field",
    "selector_authority",
    "grid_alignment",
    "time_alignment",
    "require_invariant_across_time",
];
const BINDING_KEYS: [&str; 9] = [
    "source_id",
    "mapping_role",
    "mapping_sha256",
    "data_role",
    "provenance_role",
    "fields",
    "grid_alignment",
    "time_alignment",
    "vertical_merge",
];

/// `composition.supplements.terrain_height`.
#[derive(Debug, Clone)]
pub struct TerrainSupplement {
    pub grid_alignment: String,
    pub data_role: String,
    pub provenance_role: String,
    pub time_alignment: String,
}

/// `mapped_composition._BINDING_OPTIONAL_KEYS`.
const BINDING_OPTIONAL_KEYS: [&str; 1] = ["companion"];
/// `mapped_composition._COMPANION_KEYS`.
const COMPANION_KEYS: [&str; 2] = ["fields", "vertical_extension"];
/// `mapped_composition._COMPANION_VERTICAL_EXTENSIONS`.
pub const COMPANION_VERTICAL_EXTENSIONS: [&str; 1] = ["above_primary_top"];
/// `mapped_composition._COMPANION_VERTICAL_KINDS`.
const COMPANION_VERTICAL_KINDS: [&str; 1] = ["pressure"];

/// A binding's `companion` declaration: the contributing source is a
/// declared companion of the primary (same grid and cycle) whose levels
/// above the primary's top are stacked onto the primary's own columns.
#[derive(Debug, Clone)]
pub struct Companion {
    pub vertical_extension: String,
    pub fields: Vec<String>,
}

/// One `composition.field_sources[name]` binding.
#[derive(Debug, Clone)]
pub struct Binding {
    pub name: String,
    pub source_id: String,
    pub mapping_role: String,
    pub mapping_sha256: String,
    pub data_role: String,
    pub provenance_role: String,
    pub fields: Vec<String>,
    pub grid_alignment: String,
    pub time_alignment: String,
    pub vertical_merge: Option<crate::vertical::VerticalMerge>,
    pub companion: Option<Companion>,
}

impl Binding {
    /// Every field the donor decode must carry: the borrowed fields and
    /// the companion's extended fields, in that order.
    pub fn donor_fields(&self) -> Vec<String> {
        let mut names = self.fields.clone();
        if let Some(companion) = &self.companion {
            names.extend(companion.fields.iter().cloned());
        }
        names
    }
}

/// The composition document, validated against the mapping's own bytes.
#[derive(Debug)]
pub struct Composition {
    pub doc: Node,
    pub sha256: String,
    pub path: String,
    pub terrain: Option<TerrainSupplement>,
    /// Bindings in NAME order, which is the order
    /// `decode_composed_source` composes them in (`sorted(bindings)`).
    pub bindings: Vec<Binding>,
}

/// `_object`: a closed key set, checked the way the Python engine checks it.
fn object(value: &Node, label: &str, allowed: &[&str], required: &[&str]) -> Result<()> {
    if !value.is_object() {
        return Err(mapping_invalid(format!("{label} must be an object")));
    }
    let declared: BTreeSet<&str> = value.entries().iter().map(|(key, _)| key.as_str()).collect();
    let unknown: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|key| !allowed.contains(key))
        .collect();
    if !unknown.is_empty() {
        return Err(mapping_invalid(format!(
            "{label} has unknown key(s): {}",
            python_list_repr(&unknown)
        )));
    }
    let mut missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|key| !declared.contains(key))
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(mapping_invalid(format!(
            "{label} is missing required key(s): {}",
            python_list_repr(&missing)
        )));
    }
    Ok(())
}

/// `_string`: a non-empty, non-whitespace string.
fn string<'a>(value: Option<&'a Node>, label: &str) -> Result<&'a str> {
    match value.and_then(Node::as_str) {
        Some(text) if !text.trim().is_empty() => Ok(text),
        _ => Err(mapping_invalid(format!(
            "{label} must be a non-empty string"
        ))),
    }
}

/// `_digest`: 64 lowercase hex characters.
fn digest(value: Option<&Node>, label: &str) -> Result<String> {
    let text = string(value, label)?.to_ascii_lowercase();
    if text.len() != 64 || !text.chars().all(|character| character.is_ascii_hexdigit()) {
        return Err(mapping_invalid(format!("{label} must be a SHA-256 digest")));
    }
    Ok(text)
}

impl Composition {
    /// `mapped_composition.load_composition`, against the mapping in hand.
    pub fn load(path: &str, mapping: &Mapping) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            missing_input(format!("cannot read composition {path}: {error}"))
        })?;
        let sha256 = crate::digest::bytes_sha256(&bytes);
        let doc = Node::parse(&bytes)
            .map_err(|error| mapping_invalid(format!("composition {path} is not JSON: {error}")))?;
        if doc.get("schema").and_then(Node::as_str) == Some(PENDING_COMPOSITION_SCHEMA) {
            return Err(mapping_invalid(pending_composition_refusal(&doc, path)?));
        }
        object(&doc, "composition", &COMPOSITION_KEYS, &COMPOSITION_REQUIRED)?;
        let schema = doc.get("schema").and_then(Node::as_str).unwrap_or("");
        if schema != COMPOSITION_SCHEMA {
            return Err(mapping_invalid(format!(
                "unsupported composition schema {}",
                python_repr(schema)
            )));
        }
        string(doc.get("name"), "composition.name")?;
        if doc.get("mapping_binding").and_then(Node::as_str) != Some("input_manifest_sha256") {
            return Err(mapping_invalid(
                "composition mapping_binding must be input_manifest_sha256",
            ));
        }
        let mut declared_bound: BTreeSet<String> = BTreeSet::new();
        for field in mapping.fields()? {
            if field.provider() == Some("composition_bound") {
                declared_bound.insert(field.name.clone());
            }
        }
        let bindings = validate_field_sources(
            doc.field("field_sources"),
            mapping,
            &declared_bound,
        )?;
        let mut covered: BTreeMap<&str, &str> = BTreeMap::new();
        for binding in &bindings {
            for name in &binding.fields {
                covered.insert(name.as_str(), binding.name.as_str());
            }
        }
        let unbound: Vec<&String> = declared_bound
            .iter()
            .filter(|name| !covered.contains_key(name.as_str()))
            .collect();
        if !unbound.is_empty() {
            let names: Vec<&str> = unbound.into_iter().map(String::as_str).collect();
            return Err(mapping_invalid(format!(
                "composition_bound field(s) {} have no contributing source; \
                 every declared gap must be bound in field_sources",
                python_list_repr(&names)
            )));
        }
        let soil_bindings: BTreeSet<&str> = SOIL_PAIR
            .iter()
            .filter_map(|name| covered.get(name).copied())
            .collect();
        if soil_bindings.len() > 1
            || (!soil_bindings.is_empty()
                && SOIL_PAIR.iter().any(|name| !covered.contains_key(name)))
        {
            return Err(mapping_invalid(
                "the canonical soil pair must ride one contributing source: a \
                 soil-depth contract validates against a single donor \
                 mapping's layer geometry",
            ));
        }
        // `validate_soil_layer_contract` is deliberately NOT ported: it
        // reads sealed documents rather than bytes, it validates against
        // the DONOR table once the donor mapping is pinned, and
        // `decode_composed_source` runs it on both routes.  A second
        // implementation here could only disagree with the first.
        let supplements = doc
            .get("supplements")
            .ok_or_else(|| mapping_invalid("composition.supplements must be an object"))?;
        object(
            supplements,
            "composition.supplements",
            &[EXTERNAL_FIELD],
            &[],
        )?;
        let terrain_declaration = supplements.field(EXTERNAL_FIELD);
        if terrain_declaration.is_some() && covered.contains_key(EXTERNAL_FIELD) {
            return Err(mapping_invalid(
                "terrain has two providers: the declared supplement and a \
                 contributing source binding",
            ));
        }
        if terrain_declaration.is_none() && !covered.contains_key(EXTERNAL_FIELD) {
            return Err(mapping_invalid(
                "terrain_height has no contributing source and no supplement; \
                 the composition must provide terrain through exactly one",
            ));
        }
        let terrain = match terrain_declaration {
            Some(value) => Some(validate_terrain_supplement(value, mapping)?),
            None => None,
        };
        Ok(Self {
            doc,
            sha256,
            path: path.to_owned(),
            terrain,
            bindings,
        })
    }
}

/// `mapped_composition._validate_terrain_supplement`.
fn validate_terrain_supplement(value: &Node, mapping: &Mapping) -> Result<TerrainSupplement> {
    let label = format!("composition.supplements.{EXTERNAL_FIELD}");
    object(value, &label, &TERRAIN_KEYS, &TERRAIN_KEYS)?;
    for key in [
        "data_role",
        "provenance_role",
        "format",
        "field",
        "selector_authority",
        "grid_alignment",
        "time_alignment",
    ] {
        string(value.get(key), &format!("terrain supplement.{key}"))?;
    }
    if value.get("field").and_then(Node::as_str) != Some(EXTERNAL_FIELD) {
        return Err(mapping_invalid(
            "terrain supplement must provide terrain_height",
        ));
    }
    let format = value.get("format").and_then(Node::as_str).unwrap_or("");
    if format != mapping.format()? || !matches!(format, "grib1" | "grib2" | "netcdf") {
        return Err(mapping_invalid(
            "terrain supplement format must equal the mapping format",
        ));
    }
    if value.get("selector_authority").and_then(Node::as_str) != Some("mapping_field_exact") {
        return Err(mapping_invalid(
            "terrain selector authority must be mapping_field_exact",
        ));
    }
    if !matches!(value.get("grid_alignment").and_then(Node::as_str), Some("exact_coordinate_subset" | "exact_coordinate_strided_subset")) {
        return Err(mapping_invalid(
            "terrain grid alignment must be exact_coordinate_subset or exact_coordinate_strided_subset",
        ));
    }
    let time_alignment = value
        .get("time_alignment")
        .and_then(Node::as_str)
        .unwrap_or("");
    if !crate::join::TERRAIN_TIME_ALIGNMENTS.contains(&time_alignment) {
        let mut allowed = crate::join::TERRAIN_TIME_ALIGNMENTS.to_vec();
        allowed.sort_unstable();
        return Err(mapping_invalid(format!(
            "terrain time alignment must be one of {}",
            python_list_repr(&allowed)
        )));
    }
    if value.get("require_invariant_across_time").and_then(Node::as_bool) != Some(true) {
        return Err(mapping_invalid(
            "terrain supplement must be invariant across all supplied times",
        ));
    }
    let field = mapping.field(EXTERNAL_FIELD).map_err(|_| {
        mapping_invalid("mapping terrain_height must have a direct selector")
    })?;
    if field.derivation().is_some() || field.selectors().is_empty() {
        return Err(mapping_invalid(
            "mapping terrain_height must have a direct selector",
        ));
    }
    let materialized: Vec<String> = field
        .source_axes()?
        .into_iter()
        .filter(|axis| axis != "time" && axis != "member")
        .collect();
    if materialized != ["y", "x"]
        || field.target_axes()? != ["y", "x"]
        || field.location()? != "surface"
        || field.staggering() != "none"
        || field.units_target()? != "m"
        || field.missing_kind()? != "reject"
    {
        return Err(mapping_invalid(
            "mapping terrain_height must be finite unstaggered surface metres",
        ));
    }
    Ok(TerrainSupplement {
        grid_alignment: value.get("grid_alignment").and_then(Node::as_str).unwrap().to_owned(),
        data_role: value
            .get("data_role")
            .and_then(Node::as_str)
            .unwrap_or_default()
            .to_owned(),
        provenance_role: value
            .get("provenance_role")
            .and_then(Node::as_str)
            .unwrap_or_default()
            .to_owned(),
        time_alignment: time_alignment.to_owned(),
    })
}

/// `mapped_composition._validate_field_sources`.
fn validate_field_sources(
    value: Option<&Node>,
    mapping: &Mapping,
    declared_bound: &BTreeSet<String>,
) -> Result<Vec<Binding>> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    if !value.is_object() {
        return Err(mapping_invalid("composition.field_sources must be an object"));
    }
    let mut bindings: Vec<Binding> = Vec::new();
    let mut covered: BTreeMap<String, String> = BTreeMap::new();
    let mut roles: BTreeMap<String, String> = BTreeMap::new();
    let mut companions: Vec<String> = Vec::new();
    for (binding_name, raw) in value.entries() {
        string(Some(&Node::String(binding_name.clone())), "field_sources binding name")?;
        let label = format!("composition.field_sources.{binding_name}");
        let allowed: Vec<&str> = BINDING_KEYS
            .iter()
            .chain(BINDING_OPTIONAL_KEYS.iter())
            .copied()
            .collect();
        object(raw, &label, &allowed, &BINDING_KEYS[..8])?;
        let vertical_merge = raw.get("vertical_merge").map(crate::vertical::VerticalMerge::parse).transpose()?;
        if vertical_merge.is_some() && raw.get("companion").is_some() {
            return Err(mapping_invalid(format!("{label}: companion and vertical_merge both rewrite the column ladder")));
        }
        let source_id = string(raw.get("source_id"), &format!("{label}.source_id"))?.to_owned();
        let mapping_sha256 = digest(
            raw.get("mapping_sha256"),
            &format!("{label}.mapping_sha256"),
        )?;
        let mut role_values: Vec<String> = Vec::new();
        for key in ["mapping_role", "data_role", "provenance_role"] {
            let role = string(raw.get(key), &format!("{label}.{key}"))?.to_owned();
            if let Some(previous) = roles.get(&role) {
                return Err(mapping_invalid(format!(
                    "{label}.{key} reuses role {} already claimed by {previous}; \
                     every binding role must be unique",
                    python_repr(&role)
                )));
            }
            roles.insert(role.clone(), format!("{label}.{key}"));
            role_values.push(role);
        }
        let grid_alignment = raw
            .get("grid_alignment")
            .and_then(Node::as_str)
            .unwrap_or("");
        if !BOUND_GRID_ALIGNMENTS.contains(&grid_alignment) {
            let mut allowed = BOUND_GRID_ALIGNMENTS.to_vec();
            allowed.sort_unstable();
            return Err(mapping_invalid(format!(
                "{label}.grid_alignment must be one of {}",
                python_list_repr(&allowed)
            )));
        }
        let time_alignment = raw
            .get("time_alignment")
            .and_then(Node::as_str)
            .unwrap_or("");
        if !crate::join::BOUND_TIME_ALIGNMENTS.contains(&time_alignment) {
            let mut allowed = crate::join::BOUND_TIME_ALIGNMENTS.to_vec();
            allowed.sort_unstable();
            return Err(mapping_invalid(format!(
                "{label}.time_alignment must be one of {}",
                python_list_repr(&allowed)
            )));
        }
        let companion = match raw.field("companion") {
            Some(value) => {
                let companion = validate_companion(value, mapping, &label, time_alignment)?;
                companions.push(binding_name.clone());
                if companions.len() > 1 {
                    return Err(mapping_invalid(format!(
                        "bindings {} and {} both declare a companion; two \
                         companions extending one primary ladder would leave \
                         the levels above its top with two sources",
                        python_repr(&companions[0]),
                        python_repr(binding_name)
                    )));
                }
                Some(companion)
            }
            None => None,
        };
        // A declared companion may lend nothing but the levels above the
        // primary's top; every other binding exists to lend fields.
        let raw_fields = raw.get("fields").filter(|node| node.is_array());
        let names: Vec<String> = match raw_fields {
            Some(node) if !node.items().is_empty() || companion.is_some() => node
                .items()
                .iter()
                .map(|item| {
                    string(Some(item), &format!("{label}.fields[]")).map(str::to_owned)
                })
                .collect::<Result<Vec<String>>>()?,
            _ => {
                return Err(mapping_invalid(format!(
                    "{label}.fields must be a non-empty list"
                )))
            }
        };
        if names.iter().collect::<BTreeSet<&String>>().len() != names.len() {
            return Err(mapping_invalid(format!("{label}.fields repeats a field")));
        }
        for name in &names {
            if mapping.field(name).is_err() {
                return Err(mapping_invalid(format!(
                    "{label} binds unknown field {}",
                    python_repr(name)
                )));
            }
            if !declared_bound.contains(name) && !vertical_merge.as_ref().is_some_and(|merge| merge.fields.contains_key(name)) {
                return Err(mapping_invalid(format!(
                    "field {} has two providers: the primary mapping and \
                     contributing source binding {}",
                    python_repr(name),
                    python_repr(binding_name)
                )));
            }
            if let Some(previous) = covered.get(name) {
                return Err(mapping_invalid(format!(
                    "field {} is bound by more than one contributing source \
                     ({} and {})",
                    python_repr(name),
                    python_repr(previous),
                    python_repr(binding_name)
                )));
            }
            covered.insert(name.clone(), binding_name.clone());
        }
        bindings.push(Binding {
            name: binding_name.clone(),
            source_id,
            mapping_role: role_values[0].clone(),
            mapping_sha256,
            data_role: role_values[1].clone(),
            provenance_role: role_values[2].clone(),
            fields: names,
            grid_alignment: grid_alignment.to_owned(),
            time_alignment: time_alignment.to_owned(),
            vertical_merge,
            companion,
        });
    }
    bindings.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(bindings)
}

/// `mapped_composition._companion_vertical_fields`: the primary's own
/// directly decoded fields that stack on its ladder, sorted.
fn companion_vertical_fields(mapping: &Mapping) -> Result<Vec<String>> {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for field in mapping.fields()? {
        if field.provider() != Some("composition_bound")
            && field.derivation().is_none()
            && !field.selectors().is_empty()
            && field.source_axes()?.iter().any(|axis| axis == "vertical")
        {
            names.insert(field.name.clone());
        }
    }
    Ok(names.into_iter().collect())
}

/// `mapped_composition._validate_companion`, sentence for sentence.
fn validate_companion(
    value: &Node,
    mapping: &Mapping,
    label: &str,
    time_alignment: &str,
) -> Result<Companion> {
    object(value, &format!("{label}.companion"), &COMPANION_KEYS, &COMPANION_KEYS)?;
    let extension = value
        .get("vertical_extension")
        .and_then(Node::as_str)
        .unwrap_or("");
    if !COMPANION_VERTICAL_EXTENSIONS.contains(&extension) {
        return Err(mapping_invalid(format!(
            "{label}.companion.vertical_extension must be one of {}",
            python_list_repr(&COMPANION_VERTICAL_EXTENSIONS)
        )));
    }
    if time_alignment != "valid_time_exact" {
        return Err(mapping_invalid(format!(
            "{label} declares a companion on the {} clock; a companion \
             extends every valid time's column with its own valid time, and \
             a broadcast would freeze the top of a moving column, so it \
             requires valid_time_exact",
            python_repr(time_alignment)
        )));
    }
    let kind = mapping.vertical()?.get("kind").and_then(Node::as_str);
    if !kind.is_some_and(|kind| COMPANION_VERTICAL_KINDS.contains(&kind)) {
        return Err(mapping_invalid(format!(
            "{label} declares a companion over a {} ladder; only a pressure \
             ladder says which levels lie above its top, and a hybrid ladder \
             would need its coefficients spliced as well",
            kind.map_or_else(|| "None".to_owned(), python_repr)
        )));
    }
    let names: Vec<String> = match value.get("fields").filter(|node| node.is_array()) {
        Some(node) if !node.items().is_empty() => node
            .items()
            .iter()
            .map(|item| {
                string(Some(item), &format!("{label}.companion.fields[]")).map(str::to_owned)
            })
            .collect::<Result<Vec<String>>>()?,
        _ => {
            return Err(mapping_invalid(format!(
                "{label}.companion.fields must be a non-empty list"
            )))
        }
    };
    if names.iter().collect::<BTreeSet<&String>>().len() != names.len() {
        return Err(mapping_invalid(format!(
            "{label}.companion.fields repeats a field"
        )));
    }
    let stacked = companion_vertical_fields(mapping)?;
    for name in &names {
        if !stacked.contains(name) {
            return Err(mapping_invalid(format!(
                "{label}.companion extends {}, which the primary mapping does \
                 not decode directly on its vertical ladder; a companion \
                 extends the primary's own columns and lends nothing else \
                 through this declaration",
                python_repr(name)
            )));
        }
    }
    let short: Vec<&String> = stacked.iter().filter(|name| !names.contains(name)).collect();
    if !short.is_empty() {
        return Err(mapping_invalid(format!(
            "{label}.companion leaves {} on the primary's ladder alone: those \
             fields would stop at the primary's top while the extended fields \
             reach above it, so one frame would carry columns of two heights",
            python_list_repr(&short)
        )));
    }
    Ok(Companion {
        vertical_extension: extension.to_owned(),
        fields: names,
    })
}

/// `mapped_composition._pending_composition_refusal`.
fn pending_composition_refusal(doc: &Node, path: &str) -> Result<String> {
    let file_name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let label = format!("pending composition {file_name}");
    object(
        doc,
        &label,
        &["schema", "name", "pending"],
        &["schema", "name", "pending"],
    )?;
    let name = string(doc.get("name"), "pending composition.name")?;
    let pending = doc
        .get("pending")
        .ok_or_else(|| mapping_invalid("pending composition.pending must be an object"))?;
    object(
        pending,
        "pending composition.pending",
        &["missing_canonical_state", "reason", "supply_route"],
        &["missing_canonical_state", "reason", "supply_route"],
    )?;
    let missing = pending.get("missing_canonical_state");
    let entries = match missing {
        Some(node) if node.is_array() && !node.items().is_empty() => node.items(),
        _ => {
            return Err(mapping_invalid(format!(
                "{file_name} pending.missing_canonical_state must be a \
                 non-empty list of canonical field names"
            )))
        }
    };
    let mut names: Vec<&str> = Vec::new();
    for entry in entries {
        names.push(string(Some(entry), "pending.missing_canonical_state entry")?);
    }
    let reason = string(pending.get("reason"), "pending composition.reason")?;
    let route = string(pending.get("supply_route"), "pending composition.supply_route")?;
    Ok(format!(
        "composition {} is an explicit PENDING declaration, not a runnable \
         contract: the source publishes no {}. {reason}. Initializing from \
         this source alone would invent those fields, so it is refused; \
         {route}.",
        python_repr(name),
        names.join(", ")
    ))
}

/// The operations `fields.terrain_height.when_absent` may name.
pub const WHEN_ABSENT_OPERATIONS: [&str; 1] = ["height_at_surface_pressure"];

/// The fields `height_at_surface_pressure` reads, by declaration key: the
/// three column fields first, then the three surface fields.
pub const WHEN_ABSENT_FIELDS: [&str; 6] = [
    "geopotential_height",
    "temperature",
    "specific_humidity",
    "surface_pressure",
    "surface_temperature",
    "surface_dewpoint",
];

/// `mapped_composition._derive_absent_terrain`: terrain for a source whose
/// terrain files carry none, from the primary's first valid time.
///
/// Terrain is one field per cycle (`cycle_invariant_broadcast`), so it
/// is derived once, from the analysis the primary opens with, and the
/// join broadcasts it to every valid time exactly as it broadcasts a
/// published analysis terrain.  The fields it reads are named by the
/// declaration and must be the primary's own decoded fields.
pub fn derive_absent_terrain(
    mapping: &Mapping,
    operation: &Node,
    stream: &crate::engine::DecodeStream<'_>,
) -> Result<(crate::assemble::DecodedCollection, Value)> {
    let kind = operation
        .get("operation")
        .and_then(Node::as_str)
        .unwrap_or_default();
    if !WHEN_ABSENT_OPERATIONS.contains(&kind) {
        return Err(mapping_invalid(format!(
            "fields.{EXTERNAL_FIELD}.when_absent.operation must be one of {}",
            python_list_repr(&WHEN_ABSENT_OPERATIONS)
        )));
    }
    let Some(first) = stream.first_slice() else {
        return Err(crate::refusal::frame_invalid(
            "terrain_height from surface pressure reads the primary's first \
             valid time, which this decode no longer holds",
        ));
    };
    let key = stream.keys()[0].clone();
    let mut values: Vec<&crate::assemble::DirectValue> =
        Vec::with_capacity(WHEN_ABSENT_FIELDS.len());
    let mut names: Vec<String> = Vec::with_capacity(WHEN_ABSENT_FIELDS.len());
    for (position, label) in WHEN_ABSENT_FIELDS.iter().enumerate() {
        let name = operation.get(label).and_then(Node::as_str).ok_or_else(|| {
            mapping_invalid(format!(
                "fields.{EXTERNAL_FIELD}.when_absent.{label} must name a field"
            ))
        })?;
        let value = first
            .direct
            .get(&(key.0, key.1.clone(), name.to_owned()))
            .ok_or_else(|| {
                crate::refusal::frame_invalid(format!(
                    "terrain_height from surface pressure needs {}, which the \
                     primary files do not carry at {}",
                    python_repr(name),
                    crate::frames::naive_isoformat(key.0)
                ))
            })?;
        let expected: &[&str] = if position < 3 {
            &["vertical", "y", "x"]
        } else {
            &["y", "x"]
        };
        if value.axes != expected {
            return Err(crate::refusal::frame_invalid(format!(
                "terrain_height from surface pressure needs {} on {expected:?} \
                 axes; got {:?}",
                python_repr(name),
                value.axes
            )));
        }
        values.push(value);
        names.push(name.to_owned());
    }
    let factor = match mapping.vertical()?.get("units").and_then(Node::as_str) {
        Some("Pa") => 1.0,
        Some("hPa") => 100.0,
        other => {
            return Err(mapping_invalid(format!(
                "terrain_height from surface pressure reads a pressure ladder \
                 in Pa or hPa; vertical.units is {other:?}"
            )))
        }
    };
    let levels_pa: Vec<f64> = first.vertical_values.iter().map(|level| level * factor).collect();
    let derived = crate::derive::height_at_surface_pressure(
        &levels_pa,
        &values[0].values,
        &values[1].values,
        &values[2].values,
        &values[3].values,
        &values[4].values,
        &values[5].values,
    )?;
    let source_cycle = values[3].source_cycle;
    let references: Vec<String> = values
        .iter()
        .flat_map(|value| value.references.iter().cloned())
        .collect();
    let terrain = crate::assemble::DirectValue {
        name: EXTERNAL_FIELD.to_owned(),
        valid_time: key.0,
        member: key.1.clone(),
        source_cycle,
        axes: vec!["y".to_owned(), "x".to_owned()],
        values: derived.values,
        missing_count: 0,
        references,
    };
    let collection = crate::assemble::DecodedCollection {
        latitude: first.latitude.clone(),
        longitude: first.longitude.clone(),
        vertical_values: first.vertical_values.clone(),
        direct: BTreeMap::from([(
            (key.0, key.1.clone(), EXTERNAL_FIELD.to_owned()),
            terrain,
        )]),
        source_cycles: BTreeMap::from([(key.clone(), source_cycle)]),
        grid_fingerprint: first.grid_fingerprint.clone(),
        hybrid_a: Vec::new(),
        hybrid_b: Vec::new(),
        original_pressure_hpa: None,
    };
    let receipt = json!({
        "operation": kind,
        "why": "the terrain files carry no terrain_height record",
        "valid_time": crate::frames::naive_isoformat(key.0),
        "fields": names,
        "cells": derived.cells,
        "cells_below_lowest_level": derived.below_ladder,
        "minimum_m": derived.minimum,
        "maximum_m": derived.maximum,
    });
    Ok((collection, receipt))
}

/// `mapped_composition._partition_mapping`: a decoder-only view of one
/// side of the composition.
///
/// The sealed, fully validated mapping remains the sole semantic
/// authority; partitioning only prevents a decoder from demanding a field
/// the composition contract explicitly assigns to the other product.
/// Without it the primary decode demands `terrain_height` from bytes the
/// composition assigns to the supplement, and refuses.
pub fn partition_mapping(mapping: &Mapping, terrain_only: bool) -> Result<Mapping> {
    let fields = mapping
        .doc
        .get("fields")
        .filter(|node| node.is_object())
        .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?;
    let kept = fields.object_filtered(|name, field| {
        (name == EXTERNAL_FIELD) == terrain_only
            && field.field("provider").and_then(Node::as_str) != Some("composition_bound")
    });
    let names: Vec<&str> = kept.entries().iter().map(|(name, _)| name.as_str()).collect();
    if terrain_only && names != [EXTERNAL_FIELD] {
        return Err(mapping_invalid("mapping lacks the direct terrain field"));
    }
    if !terrain_only && names.is_empty() {
        return Err(mapping_invalid(
            "mapping has no primary fields after terrain partitioning",
        ));
    }
    Ok(Mapping {
        doc: mapping.doc.with_entry("fields", kept),
        sha256: mapping.sha256.clone(),
        path: mapping.path.clone(),
    })
}

/// `mapped_composition._partition_contributing`: a decoder-only view of
/// one contributing source's bound fields.
pub fn partition_contributing(
    donor: &Mapping,
    field_names: &[String],
    binding_name: &str,
) -> Result<Mapping> {
    let fields = donor
        .doc
        .get("fields")
        .filter(|node| node.is_object())
        .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?;
    let quoted = python_repr(binding_name);
    let mut needed: BTreeSet<String> = field_names.iter().cloned().collect();
    let mut pending = field_names.to_vec();
    while let Some(name) = pending.pop() {
        let field = donor.field(&name)?;
        // Missing-value policies also have dependencies. Decode the mask
        // without binding it over the primary source's own land fraction.
        if matches!(field.missing_kind()?, "landmask_water" | "landmask_zero_water")
            && needed.insert("land_fraction".to_owned()) {
            pending.push("land_fraction".to_owned());
        }
        if let Some(operation_name) = field.derivation() {
            let operation = donor.derivation(operation_name).ok_or_else(|| mapping_invalid(format!(
                "contributing source binding {quoted} has unknown derivation {operation_name}")))?;
            let dependencies = crate::derive::derivation_dependencies(operation, donor.vertical()?, &name)?
                .ok_or_else(|| mapping_invalid(format!("cannot resolve contributing derivation {name}")))?;
            for dependency in dependencies {
                if needed.insert(dependency.clone()) { pending.push(dependency); }
            }
        } else if field.selectors().is_empty() {
            return Err(mapping_invalid(format!("contributing source binding {quoted} cannot decode {name}")));
        }
    }
    let kept: Vec<(String, Node)> = fields.entries().iter().filter(|(name, _)| needed.contains(name)).cloned().collect();
    Ok(Mapping {
        doc: donor.doc.with_entry("fields", Node::Object(kept)),
        sha256: donor.sha256.clone(),
        path: donor.path.clone(),
    })
}

/// Materialize only the requested donor fields, using their own mapping's
/// dependency closure. Donor q2 from dewpoint and q from RH follow the
/// existing derivation catalog, with no special source handling.
fn complete_donor_fields(mapping: &Mapping, mut collection: crate::assemble::DecodedCollection,
                         names: &[String]) -> Result<crate::assemble::DecodedCollection> {
    use crate::derive::{CanonicalField, evaluate_derivation};
    use rayon::prelude::*;
    let keys: Vec<_> = collection.source_cycles.keys().cloned().collect();
    let values: Vec<Result<Vec<crate::assemble::DirectValue>>> = crate::threads::install(|| keys.par_iter().map(|key| {
        let mut available: BTreeMap<String, CanonicalField> = BTreeMap::new();
        for ((time, member, name), value) in &collection.direct {
            if *time != key.0 || *member != key.1 { continue; }
            let spec = mapping.field(name)?;
            available.insert(name.clone(), CanonicalField {
                name: name.clone(), units: spec.units_target()?.into(), axes: value.axes.clone(),
                location: spec.location()?.into(), staggering: spec.staggering().into(), values: value.values.clone(),
                missing_count: value.missing_count, source_references: value.references.clone() });
        }
        let mut pending: BTreeSet<_> = mapping.fields()?.into_iter()
            .filter(|field| field.derivation().is_some() && !available.contains_key(&field.name))
            .map(|field| field.name).collect();
        while !pending.is_empty() {
            let before = pending.len();
            for name in pending.clone() {
                let field = mapping.field(&name)?;
                let operation = mapping.derivation(field.derivation().unwrap()).unwrap();
                if let Some((values, axes, references)) = evaluate_derivation(operation, &available, &collection, &field, &name, mapping.vertical()?)? {
                    let missing_count = crate::array::count_nan(&values);
                    available.insert(name.clone(), CanonicalField {
                        name: name.clone(), units: field.units_target()?.into(), axes, location: field.location()?.into(),
                        staggering: field.staggering().into(), values, missing_count, source_references: references });
                    pending.remove(&name);
                }
            }
            if pending.len() == before { return Err(mapping_invalid(format!("unresolved contributing derivations {pending:?}"))); }
        }
        names.iter().map(|name| {
            let field = available.remove(name).ok_or_else(|| mapping_invalid(format!("donor lacks requested {name}")))?;
            Ok(crate::assemble::DirectValue { name: name.clone(), valid_time: key.0, member: key.1.clone(),
                source_cycle: collection.source_cycles[key], axes: field.axes, values: field.values,
                missing_count: field.missing_count, references: field.source_references })
        }).collect()
    }).collect());
    collection.direct.clear();
    for group in crate::threads::in_order(values)? {
        for field in group { collection.direct.insert((field.valid_time, field.member.clone(), field.name.clone()), field); }
    }
    Ok(collection)
}

/// Preserve both borrowed surface fields and the companion's upper column.
/// Filtering to `binding.fields` alone drops the column after decoding it.
fn complete_binding_donor(mapping: &Mapping, collection: crate::assemble::DecodedCollection,
                          binding: &Binding) -> Result<crate::assemble::DecodedCollection> {
    complete_donor_fields(mapping, collection, &binding.donor_fields())
}

/// The UNION view a composed frameset materializes against: the primary
/// mapping's own fields, plus each borrowed field's spec taken from the
/// DONOR mapping â€” the decode authority for those octets.
///
/// Built structurally, never through JSON text: the crate's own
/// `Cargo.toml` records a one-ULP float parse that moved every cell of a
/// scaled field.  The union keeps the PRIMARY's identity (sha256 and
/// path), because a frame states the mapping the caller sealed and
/// `_compose_through_engine` re-checks exactly that digest on return.
pub fn union_mapping(
    primary: &Mapping,
    donors: &BTreeMap<String, Mapping>,
    bindings: &[Binding],
) -> Result<Mapping> {
    let mut fields = primary
        .doc
        .get("fields")
        .filter(|node| node.is_object())
        .cloned()
        .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?;
    for binding in bindings {
        let donor = &donors[&binding.name];
        let donor_fields = donor
            .doc
            .get("fields")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("contributing mapping.fields must be an object"))?;
        for name in &binding.fields {
            let spec = donor_fields.get(name).ok_or_else(|| {
                mapping_invalid(format!(
                    "contributing source binding {} borrows {} but the \
                     contributing mapping does not map it",
                    python_repr(&binding.name),
                    python_repr(name)
                ))
            })?;
            // The primary binds a borrowed field in order to publish it,
            // so the donor's own `dependency_only` does not ride along:
            // it would hold the field the composition exists to supply
            // off the composed frame.
            let spec = if spec.is_object() {
                spec.object_filtered(|key, _| key != "dependency_only" && key != "derivation")
            } else {
                spec.clone()
            };
            fields = fields.with_entry(name, spec);
        }
    }
    let mut doc = primary.doc.clone();
    for binding in bindings {
        if let Some(merge) = &binding.vertical_merge {
            let donor = &donors[&binding.name];
            merge.validate_coordinates(primary.vertical()?, donor.vertical()?)?;
            let coordinates = doc.get("coordinates").unwrap().with_entry("vertical", donor.vertical()?.clone());
            doc = doc.with_entry("coordinates", coordinates);
            fields = fields.with_entry("air_pressure", donor.field("air_pressure")?.raw.clone());
            for (output, rule) in &merge.fields {
                if output != &rule.primary_field {
                    fields = fields.object_filtered(|key, _| key != rule.primary_field);
                }
            }
        }
    }
    Ok(Mapping {
        doc: doc.with_entry("fields", fields),
        sha256: primary.sha256.clone(),
        path: primary.path.clone(),
    })
}

/// The contributing-source record of a binding whose files publish no
/// record for its fields, or `None` when any bound field has no fallback
/// (or the binding supplies terrain, which has its own `when_absent`).
///
/// Twin of `mapped_composition._unpublished_binding_record`.  The record
/// keeps the binding's mapping, data and provenance identities, so every
/// inventory check that reads it is unchanged; its alignment says
/// `UNPUBLISHED` and names each field's fallback.
#[allow(clippy::too_many_arguments)]
pub fn unpublished_record(
    binding: &Binding,
    fallbacks: &[(String, String)],
    donor_path: &str,
    donor_sha256: &str,
    donor_files: &[String],
    digests: &BTreeMap<String, String>,
    provenance_path: &String,
    provenance_digests: &BTreeMap<String, String>,
) -> Option<Value> {
    if fallbacks.is_empty() || binding.fields.iter().any(|name| name == EXTERNAL_FIELD) {
        return None;
    }
    let mut named = serde_json::Map::new();
    for name in &binding.fields {
        let (_, id) = fallbacks.iter().find(|(field, _)| field == name)?;
        named.insert(name.clone(), json!(id));
    }
    let mut fields = binding.fields.clone();
    fields.sort();
    Some(json!({
        "binding": binding.name,
        "source_id": binding.source_id,
        "mapping": {"path": donor_path, "sha256": donor_sha256},
        "data": donor_files
            .iter()
            .map(|path| json!({"path": path, "sha256": digests[path]}))
            .collect::<Vec<Value>>(),
        "provenance": {
            "path": provenance_path,
            "sha256": provenance_digests[provenance_path],
        },
        "fields": fields,
        "alignment": {
            "status": "UNPUBLISHED",
            "reason": "the supplied files publish no record for these fields",
            "fallback": Value::Object(named),
        },
    }))
}

/// `mapping` without the named fields, in `fields` and in
/// `target.required_fields`: an unpublished binding's fields are not on
/// the composed frame, so the frame does not require them.
pub fn without_fields(mapping: &Mapping, names: &BTreeSet<String>) -> Result<Mapping> {
    let fields = mapping
        .doc
        .get("fields")
        .filter(|node| node.is_object())
        .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?
        .object_filtered(|key, _| !names.contains(key));
    let mut doc = mapping.doc.with_entry("fields", fields);
    if let Some(target) = mapping.doc.get("target").filter(|node| node.is_object())
        && let Some(required) = target.get("required_fields").filter(|node| node.is_array()) {
            let kept: Vec<Node> = required
                .items()
                .iter()
                .filter(|item| {
                    item.get("name")
                        .and_then(Node::as_str)
                        .is_none_or(|name| !names.contains(name))
                })
                .cloned()
                .collect();
            doc = doc.with_entry("target", target.with_entry("required_fields", Node::Array(kept)));
        }
    Ok(Mapping {
        doc,
        sha256: mapping.sha256.clone(),
        path: mapping.path.clone(),
    })
}

/// `mapped_composition._bind_manifest_member`: stamp the manifest's
/// EXPLICIT member onto every composed record.
///
/// For archives whose product octets carry no ensemble identity: the
/// caller's verified authority named the member, the sealed input
/// manifest declares it, and this replaces ABSENT product-defined
/// membership only.  Bytes that already encode a member contradict the
/// declaration and refuse rather than being overwritten -- overwriting
/// would let one archive's frames claim another member's identity.  The
/// refusal wording is the Python engine's, verbatim, so the two engines
/// refuse the same bytes with the same sentence.
pub fn bind_manifest_member(
    collection: crate::assemble::DecodedCollection,
    member: &str,
) -> Result<crate::assemble::DecodedCollection> {
    let encoded: BTreeSet<Option<String>> = collection
        .direct
        .values()
        .map(|value| value.member.clone())
        .collect();
    if encoded != BTreeSet::from([None]) {
        let mut names: Vec<String> = encoded.into_iter().flatten().collect();
        names.sort();
        return Err(crate::refusal::manifest_mismatch(format!(
            "the composition input manifest binds explicit member {}, but \
             the decoded records already encode product-defined member(s) \
             {}; an explicit member binding replaces ABSENT product \
             membership only, so stamping it here would overwrite a \
             producer-declared identity",
            python_repr(member),
            python_list_repr(&names)
        )));
    }
    let mut direct = BTreeMap::new();
    for ((valid_time, _old_member, field_name), value) in collection.direct {
        direct.insert(
            (valid_time, Some(member.to_owned()), field_name),
            crate::assemble::DirectValue {
                member: Some(member.to_owned()),
                ..value
            },
        );
    }
    let mut source_cycles = BTreeMap::new();
    for ((valid_time, _old_member), cycle) in collection.source_cycles {
        source_cycles.insert((valid_time, Some(member.to_owned())), cycle);
    }
    Ok(crate::assemble::DecodedCollection {
        latitude: collection.latitude,
        longitude: collection.longitude,
        vertical_values: collection.vertical_values,
        direct,
        source_cycles,
        grid_fingerprint: collection.grid_fingerprint,
        hybrid_a: collection.hybrid_a,
        hybrid_b: collection.hybrid_b,
        original_pressure_hpa: collection.original_pressure_hpa,
    })
}

/// Role bindings from argv, in the shape the composition reads them.
fn role_inventory(bindings: &[(String, String)]) -> BTreeMap<String, Vec<String>> {
    let mut inventory: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (role, path) in bindings {
        inventory.entry(role.clone()).or_default().push(path.clone());
    }
    inventory
}

fn single_role(inventory: &BTreeMap<String, Vec<String>>, kind: &str) -> Result<BTreeMap<String, String>> {
    let mut single = BTreeMap::new();
    for (role, paths) in inventory {
        if paths.len() != 1 {
            return Err(crate::refusal::usage(format!(
                "--{kind} {role}= was supplied {} times; a {kind} role binds \
                 exactly one document",
                paths.len()
            )));
        }
        single.insert(role.clone(), paths[0].clone());
    }
    Ok(single)
}

/// `compose`: mapping + composition + role bindings â†’ a frameset directory
/// plus the composition evidence document.
pub fn run_compose(invocation: &Invocation, progress: &mut dyn FnMut(Value)) -> Result<Value> {
    let mapping = Mapping::load(&invocation.mapping)?;
    let Some(composition_path) = invocation.composition.as_deref() else {
        return Err(crate::refusal::usage(
            "compose requires --composition COMPOSITION.json",
        ));
    };
    let composition = Composition::load(composition_path, &mapping)?;
    let primary = crate::engine::read_input_list(&invocation.input_list)?;
    let supplements = role_inventory(&invocation.supplements);
    let provenance = single_role(&role_inventory(&invocation.provenance), "provenance")?;
    let contributing = single_role(
        &role_inventory(&invocation.contributing_mappings),
        "contributing-mapping",
    )?;

    // The role inventory the contract declares, checked against argv
    // before a byte is read.  Named breakage: a supplement bound under a
    // role the composition does not declare would be hashed into the
    // frameset's inputs and never read, so the frameset would name bytes
    // that contributed nothing.
    let mut expected_data: BTreeSet<&str> = BTreeSet::new();
    let mut expected_provenance: BTreeSet<&str> = BTreeSet::new();
    if let Some(terrain) = &composition.terrain {
        expected_data.insert(terrain.data_role.as_str());
        expected_provenance.insert(terrain.provenance_role.as_str());
    }
    let mut expected_mappings: BTreeSet<&str> = BTreeSet::new();
    for binding in &composition.bindings {
        expected_data.insert(binding.data_role.as_str());
        expected_provenance.insert(binding.provenance_role.as_str());
        expected_mappings.insert(binding.mapping_role.as_str());
    }
    for (kind, declared, supplied) in [
        ("supplement", expected_data, supplements.keys().map(String::as_str).collect::<BTreeSet<&str>>()),
        ("provenance", expected_provenance, provenance.keys().map(String::as_str).collect()),
        ("contributing mapping", expected_mappings, contributing.keys().map(String::as_str).collect()),
    ] {
        if declared != supplied {
            let declared: Vec<&str> = declared.into_iter().collect();
            let supplied: Vec<&str> = supplied.into_iter().collect();
            return Err(mapping_invalid(format!(
                "{kind} role inventory differs from the composition: expected \
                 {}, got {}",
                python_list_repr(&declared),
                python_list_repr(&supplied)
            )));
        }
    }

    // Every byte the frameset will name, hashed once.  The frames carry
    // the primary inputs AND every supplement (a borrowed field's donor
    // object is an input to the composed frame), exactly as
    // `decode_composed_source` builds `input_hashes`.
    let mut data_files: Vec<String> = primary.clone();
    for paths in supplements.values() {
        for path in paths {
            if !data_files.contains(path) {
                data_files.push(path.clone());
            }
        }
    }
    let digests = crate::engine::input_digests(&data_files)?;
    // The provenance documents are sealed by the manifest and named in
    // the composition evidence, but they are NOT frameset inputs: no
    // number in a frame came from them, and hashing them into
    // `input_sha256` would claim they did.
    let provenance_files: Vec<String> = provenance.values().cloned().collect();
    let provenance_digests = crate::engine::input_digests(&provenance_files)?;
    let mut member_binding: Option<(String, String)> = None;
    if let (Some(manifest), Some(expected)) = (
        invocation.input_manifest.as_ref(),
        invocation.input_manifest_sha256.as_ref(),
    ) {
        let mut sealed = digests.clone();
        sealed.extend(
            provenance_digests
                .iter()
                .map(|(path, value)| (path.clone(), value.clone())),
        );
        member_binding = crate::engine::verify_composition_manifest(
            manifest,
            expected,
            &mapping,
            &composition.sha256,
            &primary,
            &supplements,
            &provenance,
            &sealed,
        )?;
    }

    // The primary is STREAMED: its valid times are decoded one at a time
    // as the frameset writes them.  Every check and every receipt below
    // reads the series header -- the grid, the ladder, the clock, the
    // field inventory -- and none of them reads a primary array, which is
    // what makes the composition resolvable before the series exists.
    let partitioned = partition_mapping(&mapping, false)?;
    // A terrain derived from the primary (`when_absent`) reads the first
    // valid time's columns beside its full-grid surface fields, so the
    // fields it names are never decoded over the atmospheric window.
    let mut keep_whole: BTreeSet<String> = BTreeSet::new();
    if composition.terrain.is_some()
        && let Some(operation) = mapping.field(EXTERNAL_FIELD).ok().and_then(|field| field.when_absent()) {
            for label in WHEN_ABSENT_FIELDS {
                if let Some(name) = operation.get(label).and_then(Node::as_str) {
                    keep_whole.insert(name.to_owned());
                }
            }
        }
    let mut stream = crate::engine::DecodeStream::open_within(
        &partitioned, &primary, progress, invocation.atmospheric_window, &keep_whole)?;
    let aliased = stream.aliased().clone();
    progress(json!({
        "event": "composed_primary",
        "valid_times": stream.keys().len(),
    }));
    let mut alignment_receipt: Option<Value> = None;
    let mut terrain_plan: Option<crate::join::TerrainPlan> = None;
    // The composed field inventory, grown as each join is resolved: the
    // two-provider refusals ask what the composition has bound SO FAR,
    // which is what the accumulating collection used to answer.
    let mut composed_names: BTreeSet<String> = stream.direct_names().clone();
    if let Some(terrain) = &composition.terrain {
        let files = &supplements[&terrain.data_role];
        let mut derived: Option<Value> = None;
        let supplement = match crate::engine::decode_collection_or_unmatched(
            &partition_mapping(&mapping, true)?,
            files,
            progress,
        )? {
            Ok(collection) => collection,
            Err(unmatched) => {
                // The files carry no terrain record at all.  A mapping
                // that declares how terrain is derived then has it
                // derived from the primary's own first valid time;
                // without that declaration the miss is refused by name,
                // as it always was.
                let Some(operation) = mapping.field(EXTERNAL_FIELD)?.when_absent() else {
                    return Err(unmatched);
                };
                let (collection, receipt) = derive_absent_terrain(&mapping, operation, &stream)?;
                progress(json!({
                    "event": "derived_terrain",
                    "cells": receipt.get("cells"),
                }));
                derived = Some(receipt);
                collection
            }
        };
        let (plan, mut receipt) = crate::join::plan_terrain(
            &crate::join::PrimaryHeader {
                latitude: stream.latitude(),
                longitude: stream.longitude(),
                vertical_values: stream.vertical_values(),
                source_cycles: &stream.summary().source_cycles,
                direct_names: &composed_names,
            },
            &supplement,
            &terrain.time_alignment,
            &terrain.grid_alignment,
        )?;
        if let (Some(derived), Some(object)) = (derived, receipt.as_object_mut()) {
            // Recorded only when terrain was derived, so every receipt of
            // a source that carries its terrain is unchanged.
            object.insert("derived".into(), derived);
        }
        terrain_plan = Some(plan);
        composed_names.insert(EXTERNAL_FIELD.to_owned());
        alignment_receipt = Some(receipt);
        progress(json!({"event": "composed_terrain", "objects": files.len()}));
    }

    // The borrowed donors stay resident because the frames borrow from
    // them one valid time at a time; a donor is one contributing
    // source's own decode, not the forcing series.
    let mut borrowed: Vec<(crate::join::BindingPlan, crate::assemble::DecodedCollection)> =
        Vec::new();
    let mut donors: BTreeMap<String, Mapping> = BTreeMap::new();
    let mut contributing_records: Vec<Value> = Vec::new();
    let mut terrain_binding_receipt: Option<Value> = None;
    // Fields of bindings recorded as unpublished: off the composed frame.
    let mut unpublished: BTreeSet<String> = BTreeSet::new();
    // The ladder the composed slices carry: the primary's own until a
    // declared companion stacks its levels above the top, after which
    // every later binding is judged against the composed ladder, exactly
    // as `decode_composed_source` judges it against the composed decode.
    let mut composed_ladder: Vec<f64> = stream.vertical_values().to_vec();
    let mut extension_bytes: u64 = 0;
    for binding in &composition.bindings {
        let donor_path = &contributing[&binding.mapping_role];
        let donor = Mapping::load(donor_path)?;
        if donor.sha256 != binding.mapping_sha256 {
            return Err(mapping_invalid(format!(
                "contributing source binding {} authority hash mismatch: the \
                 composition pins {}, the supplied mapping bytes hash to {}",
                python_repr(&binding.name),
                binding.mapping_sha256,
                donor.sha256
            )));
        }
        if donor.name()? != binding.source_id {
            return Err(mapping_invalid(format!(
                "contributing source binding {} names source_id {} but the \
                 pinned mapping is {}",
                python_repr(&binding.name),
                python_repr(&binding.source_id),
                python_repr(donor.name()?)
            )));
        }
        if binding.companion.is_some() {
            // Twin of the companion check in `decode_composed_source`.
            for aspect in ["kind", "units"] {
                let ours = mapping.vertical()?.get(aspect).and_then(Node::as_str);
                let theirs = donor.vertical()?.get(aspect).and_then(Node::as_str);
                if ours != theirs {
                    let spell = |value: Option<&str>| value.map_or_else(|| "None".to_owned(), python_repr);
                    return Err(mapping_invalid(format!(
                        "contributing source binding {} is a companion whose \
                         vertical {aspect} is {} where the primary's is {}; \
                         levels on two coordinates cannot be stacked into one \
                         column",
                        python_repr(&binding.name),
                        spell(theirs),
                        spell(ours)
                    )));
                }
            }
        }
        let donor_files = &supplements[&binding.data_role];
        let mut decode_fields = binding.donor_fields();
        if binding.grid_alignment == crate::regrid::ALIGNMENT
            && !decode_fields.iter().any(|name| name == "land_fraction") {
            decode_fields.push("land_fraction".to_owned());
        }
        let donor_partition = partition_contributing(&donor, &decode_fields, &binding.name)?;
        let donor_collection = match crate::engine::decode_collection_or_unmatched(
            &donor_partition,
            donor_files,
            progress,
        )? {
            Ok(collection) => collection,
            Err(unmatched) => {
                if binding.companion.is_some() {
                    // Land fallbacks cannot supply the absent upper column.
                    return Err(unmatched);
                }
                // The files publish no record at all for this binding's
                // fields (HRRR wrfsfc before 2020-12-02 carries no VEG).
                // When the caller gave a fallback for EVERY bound field the
                // binding is recorded as unpublished with those fallbacks
                // and composes nothing; otherwise the miss is refused by
                // name, as it always was.
                let Some(record) = unpublished_record(
                    binding,
                    &invocation.unpublished_fallbacks,
                    donor_path,
                    &donor.sha256,
                    donor_files,
                    &digests,
                    &provenance[&binding.provenance_role],
                    &provenance_digests,
                ) else {
                    return Err(unmatched);
                };
                contributing_records.push(record);
                unpublished.extend(binding.fields.iter().cloned());
                donors.insert(binding.name.clone(), donor);
                progress(json!({
                    "event": "unpublished_binding",
                    "binding": binding.name,
                    "fields": binding.fields.len(),
                }));
                continue;
            }
        };
        let mut donor_collection = complete_donor_fields(&donor_partition, donor_collection, &decode_fields)?;
        crate::join::register_projected_subset(&mapping, &donor, &mut donor_collection)?;
        let (mut donor_collection, regrid) = if binding.grid_alignment == crate::regrid::ALIGNMENT {
            let (donor, receipt) = crate::regrid::align(donor_collection, stream.latitude(), stream.longitude())?;
            progress(json!({"event": "horizontal_regrid", "binding": binding.name, "regrid": receipt}));
            (donor, Some(receipt))
        } else { (donor_collection, None) };
        donor_collection.direct.retain(|(_, _, name), _| binding.fields.contains(name));
        let (plan, mut receipt) = crate::join::plan_bound_fields(
            &crate::join::PrimaryHeader {
                latitude: stream.latitude(),
                longitude: stream.longitude(),
                vertical_values: &composed_ladder,
                source_cycles: &stream.summary().source_cycles,
                direct_names: &composed_names,
            },
            &donor_collection,
            binding,
        )?;
        if let Some(regrid) = regrid {
            receipt["coordinate_match"] = json!(crate::regrid::ALIGNMENT);
            receipt["donor_shape"] = regrid["source_shape"].clone();
            receipt["horizontal_regrid"] = regrid;
            receipt.as_object_mut().unwrap().remove("latitude_index_range");
            receipt.as_object_mut().unwrap().remove("longitude_index_range");
        }
        if let Some(ladder) = plan.composed_ladder() {
            composed_ladder = ladder.to_vec();
        }
        extension_bytes = extension_bytes.saturating_add(plan.extension_bytes());
        borrowed.push((plan, donor_collection));
        composed_names.extend(binding.fields.iter().cloned());
        let provenance_path = &provenance[&binding.provenance_role];
        let mut fields = binding.fields.clone();
        fields.sort();
        contributing_records.push(json!({
            "binding": binding.name,
            "source_id": binding.source_id,
            "mapping": {"path": donor_path, "sha256": donor.sha256},
            "data": donor_files
                .iter()
                .map(|path| json!({"path": path, "sha256": digests[path]}))
                .collect::<Vec<Value>>(),
            "provenance": {
                "path": provenance_path,
                "sha256": provenance_digests[provenance_path],
            },
            "fields": fields,
            "alignment": receipt,
        }));
        if binding.fields.iter().any(|name| name == EXTERNAL_FIELD) {
            terrain_binding_receipt = Some(
                contributing_records
                    .last()
                    .expect("just pushed")
                    .get("alignment")
                    .cloned()
                    .expect("the record carries its alignment"),
            );
        }
        donors.insert(binding.name.clone(), donor);
        progress(json!({
            "event": "composed_binding",
            "binding": binding.name,
            "fields": binding.fields.len(),
        }));
    }

    let mut series = stream.summary().clone();
    series.lead_batch = invocation.lead_batch;
    if let Some((member, _identity)) = &member_binding {
        // The composed clock is restated under the bound member so the
        // frameset's keys and the slices it pulls agree; the records
        // themselves are stamped as each slice is composed.
        let stamped: BTreeMap<crate::assemble::TimeKey, chrono::NaiveDateTime> = series
            .source_cycles
            .into_iter()
            .map(|((valid_time, _member), cycle)| {
                ((valid_time, Some(member.clone())), cycle)
            })
            .collect();
        series.source_cycles = stamped;
    }

    let union = union_mapping(&mapping, &donors, &composition.bindings)?;
    let union = if unpublished.is_empty() {
        union
    } else {
        without_fields(&union, &unpublished)?
    };
    let output = std::path::PathBuf::from(
        invocation
            .output
            .as_ref()
            .expect("compose requires --output"),
    );
    // One valid time end to end: the primary slice is decoded, the
    // composition's joins are applied to it, the frame is materialized
    // and written, and all of it is dropped.  Several valid times do
    // this at once when the series allows it (`lanes`), each on its own
    // decode, and they are still written one at a time, in order.
    let source_keys: Vec<crate::assemble::TimeKey> = stream.keys().to_vec();
    // A window granted before the first record was decoded is published
    // by every frame, on either writer (`DecodeStream::decode_window`).
    let decode_window = stream.decode_window().cloned();
    // The frameset's key and the decode's key differ only in the
    // member the manifest binds; if they ever differed in the VALID
    // TIME, the composition would be applied to a different forcing
    // time than the frame being written, so it refuses by naming both
    // rather than composing the wrong one.
    let same_time = |key: &crate::assemble::TimeKey, source_key: &crate::assemble::TimeKey| {
        if key.0 != source_key.0 {
            return Err(mapping_invalid(format!(
                "the composed frameset asked for {} while the decode is at \
                 {}; the frame order and the decode order have diverged",
                crate::frames::naive_isoformat(key.0),
                crate::frames::naive_isoformat(source_key.0)
            )));
        }
        Ok(())
    };
    let composition_window = decode_window.clone();
    let splice_receipts = std::sync::Mutex::new(Vec::<Value>::new());
    let compose_slice = |mut slice: crate::assemble::DecodedCollection,
                         source_key: &crate::assemble::TimeKey| {
        if let Some(plan) = &terrain_plan {
            crate::join::apply_terrain(plan, &mut slice, source_key)?;
        }
        for (plan, donor) in &borrowed {
            if let Some(receipt) = crate::join::apply_bound_fields(plan, donor, &mut slice, source_key, composition_window.as_ref())? {
                splice_receipts.lock().unwrap().push(receipt);
            }
        }
        // The manifest's explicit member binding lands LAST, after every
        // join: the composition operates on the identity the bytes carry
        // (donor alignment refuses member-bearing donors on its own
        // terms).  Twin of `mapped_composition._bind_manifest_member`.
        if let Some((member, _identity)) = &member_binding {
            slice = bind_manifest_member(slice, member)?;
        }
        Ok(slice)
    };
    // How many valid times are in flight is not announced: the progress
    // stream is part of the byte-identity gate (docs/dev/decode-vendor-
    // design.md, "THE GATE") and must not change with the worker count.
    // Shared-object streams remain serial but still recheck retained donor
    // memory before dispatch. A forced single lane is not an admission bypass.
    let admitted_times = if stream.times_are_independent() { source_keys.len() } else { 1 };
    // A declared companion's levels ride every valid time on top of its
    // decode, so they are priced with it.
    let lanes = crate::threads::lanes(
        stream.per_time_bytes().saturating_add(extension_bytes), &stream.field_bytes(), stream.held_bytes(), admitted_times)?;
    let overlap_write = lanes == 1 && source_keys.len() > 1 && crate::threads::overlap_write(
        stream.per_time_bytes().saturating_add(extension_bytes), &stream.field_bytes(), stream.held_bytes())?;
    let document = if lanes > 1 {
        let first = std::sync::Mutex::new(stream.take_first());
        let stream = &stream;
        crate::frames::write_frameset_lanes(
            &output, &union, &series, &digests, invocation.atmospheric_window, decode_window,
            lanes, |index, key| {
                let source_key = &source_keys[index];
                same_time(key, source_key)?;
                let held = if index == 0 {
                    first.lock().ok().and_then(|mut slot| slot.take())
                } else {
                    None
                };
                let slice = match held {
                    Some(collection) => collection,
                    None => stream.slice_detached(source_key)?,
                };
                compose_slice(slice, source_key)
            },
        )?
    } else {
        let mut position = 0usize;
        crate::frames::write_frameset_one_lane(
            &output, &union, &series, &digests, invocation.atmospheric_window, decode_window,
            overlap_write, |key| {
            let source_key = source_keys[position].clone();
            position += 1;
            same_time(key, &source_key)?;
            let slice = stream.slice(&source_key)?;
            compose_slice(slice, &source_key)
        })?
    };
    for record in &mut contributing_records {
        let binding_name = record["binding"].as_str().unwrap_or("");
        let mut receipts: Vec<Value> = splice_receipts.lock().unwrap().iter()
            .filter(|receipt| receipt["binding"] == binding_name).cloned().collect();
        receipts.sort_by_key(|receipt| receipt["valid_time"].as_str().unwrap_or("").to_owned());
        if !receipts.is_empty() {
            record["alignment"]["vertical_merge"] = json!(receipts);
        }
    }
    let frame_count = document
        .get("frames")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    // The alignment receipt a composed bundle is judged on: the terrain
    // supplement's when there is one, otherwise the receipt of the
    // binding that supplies terrain.  `MappedSourceBundle` requires one,
    // and the grammar above proved terrain has exactly one provider.
    let mut alignment_receipt = alignment_receipt.or(terrain_binding_receipt).ok_or_else(|| {
        mapping_invalid(
            "composed source has neither a terrain supplement nor a terrain \
             field binding",
        )
    })?;
    if let Some((member, identity)) = &member_binding {
        // The sealed alignment receipt keeps the ensemble identity the
        // manifest bound, beside the alignment it describes.  Recorded
        // only under an explicit binding, so every receipt written
        // before the binding existed is byte-identical today.
        if let Some(receipt) = alignment_receipt.as_object_mut() {
            receipt.insert("member".to_owned(), json!(member));
            receipt.insert("member_identity".to_owned(), json!(identity));
        }
    }
    let mut evidence = json!({
        "schema": COMPOSITION_EVIDENCE_SCHEMA,
        "engine": {"name": crate::ENGINE_NAME, "version": crate::ENGINE_VERSION},
        "mapping": {"path": mapping.path, "sha256": mapping.sha256},
        "composition": {"path": composition.path, "sha256": composition.sha256},
        "alignment_receipt": alignment_receipt,
        "contributing_sources": contributing_records,
    });
    if !aliased.is_empty() {
        // Present only when records were read through
        // `mapping.record_aliases`, so every other evidence document is
        // unchanged: the primary records each field read that way.
        evidence["record_aliases"] = json!(aliased);
    }
    let evidence_path = output.join("composition.json");
    std::fs::write(
        &evidence_path,
        serde_json::to_vec_pretty(&evidence).unwrap_or_default(),
    )
    .map_err(|error| {
        crate::refusal::write_error(
            &format!("write the composition evidence {}", evidence_path.display()),
            &error,
            None,
        )
    })?;
    Ok(json!({
        "event": "receipt",
        "subcommand": "compose",
        "schema": document["schema"],
        "frames": frame_count,
        "output": output.display().to_string(),
        "grid_fingerprint": series.grid_fingerprint,
        "bindings": composition.bindings.len(),
        "stream_bytes": document
            .get("stream")
            .and_then(|stream| stream.get("bytes"))
            .cloned()
            .unwrap_or(Value::Null),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, body: &str) -> String {
        // One folder per test process: a fixed name let two test binaries
        // running at once on one machine overwrite each other's documents
        // mid-read ("c5.json is not JSON: EOF", test-host-1, 2026-10-09).
        let directory = std::env::temp_dir()
            .join(format!("gpuwm-mapped-engine-compose-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join(name);
        std::fs::write(&path, body).unwrap();
        path.to_str().unwrap().to_owned()
    }

    const TERRAIN_FIELD: &str = r#"{
        "selectors": [{"parameter": "HGT"}],
        "source_axes": ["time", "y", "x"],
        "target_axes": ["y", "x"],
        "location": "surface",
        "units": {"target": "m"},
        "missing": {"kind": "reject"}
    }"#;

    fn mapping_document(extra_fields: &str) -> String {
        format!(
            r#"{{"schema": "rw-wps.mapping.v1", "format": "grib2", "name": "m",
                "fields": {{"terrain_height": {TERRAIN_FIELD}{extra_fields}}}}}"#
        )
    }

    fn composition_document(supplements: &str, field_sources: &str) -> String {
        format!(
            r#"{{"schema": "gpuwm-mapped-composition-v2", "name": "c",
                "mapping_binding": "input_manifest_sha256",
                "soil_layers": {{}},
                "supplements": {supplements}{field_sources}}}"#
        )
    }

    const TERRAIN_SUPPLEMENT: &str = r#"{"terrain_height": {
        "data_role": "terrain_data", "provenance_role": "terrain_provenance",
        "format": "grib2", "field": "terrain_height",
        "selector_authority": "mapping_field_exact",
        "grid_alignment": "exact_coordinate_subset",
        "time_alignment": "valid_time_exact",
        "require_invariant_across_time": true}}"#;

    #[test]
    fn a_composition_with_the_wrong_schema_refuses_by_naming_it() {
        let mapping = Mapping::load(&write("m1.json", &mapping_document(""))).unwrap();
        let path = write(
            "c1.json",
            &composition_document(TERRAIN_SUPPLEMENT, "")
                .replace(COMPOSITION_SCHEMA, "something-else"),
        );
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID);
        assert_eq!(
            refusal.message,
            "unsupported composition schema 'something-else'"
        );
    }

    #[test]
    fn a_document_missing_a_required_key_names_every_one_it_lacks() {
        // The closed key set is checked BEFORE the schema string, exactly
        // as `_object` runs before the schema comparison: a document that
        // is neither a v2 composition nor anything else earns the
        // structural refusal, not a schema sentence.
        let mapping = Mapping::load(&write("m1b.json", &mapping_document(""))).unwrap();
        let path = write("c1b.json", r#"{"schema": "something-else"}"#);
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert_eq!(
            refusal.message,
            "composition is missing required key(s): ['mapping_binding', \
             'name', 'soil_layers', 'supplements']"
        );
    }

    #[test]
    fn a_terrain_supplement_composition_parses_its_roles() {
        let mapping = Mapping::load(&write("m2.json", &mapping_document(""))).unwrap();
        let path = write("c2.json", &composition_document(TERRAIN_SUPPLEMENT, ""));
        let composition = Composition::load(&path, &mapping).unwrap();
        let terrain = composition.terrain.expect("a declared supplement");
        assert_eq!(terrain.data_role, "terrain_data");
        assert_eq!(terrain.time_alignment, "valid_time_exact");
        assert!(composition.bindings.is_empty());
    }

    fn companion_mapping(kind: &str, extra_fields: &str) -> String {
        format!(
            r#"{{"schema": "rw-wps.mapping.v1", "format": "grib2", "name": "m",
                "coordinates": {{"vertical": {{"kind": "{kind}", "units": "Pa"}}}},
                "fields": {{"terrain_height": {TERRAIN_FIELD},
                    "air_temperature": {{"selectors": [{{"parameter": "TMP"}}],
                        "source_axes": ["vertical", "y", "x"],
                        "target_axes": ["vertical", "y", "x"],
                        "location": "mass", "units": {{"target": "K"}},
                        "missing": {{"kind": "reject"}}}},
                    "land_fraction": {{"provider": "composition_bound",
                        "source_axes": ["y", "x"], "target_axes": ["y", "x"],
                        "location": "surface", "units": {{"target": "1"}},
                        "missing": {{"kind": "reject"}}}}{extra_fields}}}}}"#
        )
    }

    fn companion_sources(clock: &str, extended: &str) -> String {
        format!(
            r#", "field_sources": {{"companion": {{
                "source_id": "d", "mapping_role": "companion_mapping",
                "mapping_sha256": "{zeros}", "data_role": "companion_data",
                "provenance_role": "companion_provenance",
                "fields": ["land_fraction"],
                "grid_alignment": "exact_coordinate_subset",
                "time_alignment": "{clock}",
                "companion": {{"vertical_extension": "above_primary_top",
                               "fields": [{extended}]}}}}}}"#,
            zeros = "0".repeat(64)
        )
    }

    #[test]
    fn a_declared_companion_parses_as_composition_rows() {
        let mapping = Mapping::load(&write("mc1.json", &companion_mapping("pressure", ""))).unwrap();
        let path = write(
            "cc1.json",
            &composition_document(
                TERRAIN_SUPPLEMENT,
                &companion_sources("valid_time_exact", r#""air_temperature""#),
            ),
        );
        let composition = Composition::load(&path, &mapping).unwrap();
        let binding = &composition.bindings[0];
        let companion = binding.companion.as_ref().unwrap();
        assert_eq!(companion.fields, vec!["air_temperature".to_owned()]);
        assert_eq!(
            binding.donor_fields(),
            vec!["land_fraction".to_owned(), "air_temperature".to_owned()]
        );
    }

    #[test]
    fn companion_refusals_are_the_python_sentences() {
        let short = Mapping::load(&write(
            "mc2.json",
            &companion_mapping(
                "pressure",
                r#", "specific_humidity": {"selectors": [{"parameter": "SPFH"}],
                    "source_axes": ["vertical", "y", "x"],
                    "target_axes": ["vertical", "y", "x"],
                    "location": "mass", "units": {"target": "kg kg-1"},
                    "missing": {"kind": "reject"}}"#,
            ),
        ))
        .unwrap();
        let path = write(
            "cc2.json",
            &composition_document(
                TERRAIN_SUPPLEMENT,
                &companion_sources("valid_time_exact", r#""air_temperature""#),
            ),
        );
        let refusal = Composition::load(&path, &short).unwrap_err();
        assert_eq!(
            refusal.message,
            "composition.field_sources.companion.companion leaves \
             ['specific_humidity'] on the primary's ladder alone: those fields \
             would stop at the primary's top while the extended fields reach \
             above it, so one frame would carry columns of two heights"
        );

        let hybrid =
            Mapping::load(&write("mc3.json", &companion_mapping("hybrid_sigma_pressure", ""))).unwrap();
        let refusal = Composition::load(&path, &hybrid).unwrap_err();
        assert!(refusal.message.contains("only a pressure ladder"), "{refusal}");

        let mapping = Mapping::load(&write("mc4.json", &companion_mapping("pressure", ""))).unwrap();
        let path = write(
            "cc4.json",
            &composition_document(
                TERRAIN_SUPPLEMENT,
                &companion_sources("cycle_invariant_broadcast", r#""air_temperature""#),
            ),
        );
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert!(refusal.message.contains("freeze the top of a moving column"), "{refusal}");

        let path = write(
            "cc5.json",
            &composition_document(
                TERRAIN_SUPPLEMENT,
                &companion_sources("valid_time_exact", r#""air_temperature", "land_fraction""#),
            ),
        );
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert!(
            refusal.message.contains("does not decode directly on its vertical ladder"),
            "{refusal}"
        );
    }

    #[test]
    fn a_composition_bound_field_with_no_contributing_source_refuses() {
        // The named breakage: the mapping declares a gap, the composition
        // fills none of it, and materialization would then emit a frame
        // whose bound field silently came from the base source.
        let mapping = Mapping::load(&write(
            "m3.json",
            &mapping_document(
                r#", "skin_temperature": {"provider": "composition_bound",
                    "source_axes": ["y", "x"], "target_axes": ["y", "x"],
                    "location": "surface", "units": {"target": "K"},
                    "missing": {"kind": "reject"}}"#,
            ),
        ))
        .unwrap();
        let path = write("c3.json", &composition_document(TERRAIN_SUPPLEMENT, ""));
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert!(
            refusal
                .message
                .contains("composition_bound field(s) ['skin_temperature'] have no"),
            "{refusal}"
        );
    }

    #[test]
    fn terrain_with_two_providers_refuses() {
        let mapping = Mapping::load(&write(
            "m4.json",
            &mapping_document("").replace(
                r#""selectors": [{"parameter": "HGT"}]"#,
                r#""provider": "composition_bound", "selectors": []"#,
            ),
        ))
        .unwrap();
        let path = write(
            "c4.json",
            &composition_document(
                TERRAIN_SUPPLEMENT,
                r#", "field_sources": {"donor": {
                    "source_id": "d", "mapping_role": "donor_mapping",
                    "mapping_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    "data_role": "donor_data", "provenance_role": "donor_provenance",
                    "fields": ["terrain_height"],
                    "grid_alignment": "exact_coordinate_subset",
                    "time_alignment": "valid_time_exact"}}"#,
            ),
        );
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert!(refusal.message.contains("terrain has two providers"), "{refusal}");
    }

    #[test]
    fn a_pending_declaration_refuses_by_naming_the_state_it_lacks() {
        let mapping = Mapping::load(&write("m5.json", &mapping_document(""))).unwrap();
        let path = write(
            "c5.json",
            r#"{"schema": "gpuwm-cross-source-composition-pending-v1",
                "name": "atmosphere-only",
                "pending": {"missing_canonical_state": ["soil_temperature"],
                            "reason": "the producer publishes no land surface",
                            "supply_route": "bind a physical analysis donor"}}"#,
        );
        let refusal = Composition::load(&path, &mapping).unwrap_err();
        assert_eq!(
            refusal.message,
            "composition 'atmosphere-only' is an explicit PENDING declaration, \
             not a runnable contract: the source publishes no soil_temperature. \
             the producer publishes no land surface. Initializing from this \
             source alone would invent those fields, so it is refused; bind a \
             physical analysis donor."
        );
    }

    #[test]
    fn partitioning_splits_terrain_from_the_primary_fields() {
        let mapping = Mapping::load(&write(
            "m6.json",
            &mapping_document(
                r#", "surface_pressure": {"selectors": [{"parameter": "PRES"}],
                    "source_axes": ["y", "x"], "target_axes": ["y", "x"],
                    "location": "surface", "units": {"target": "Pa"},
                    "missing": {"kind": "reject"}}"#,
            ),
        ))
        .unwrap();
        assert_eq!(
            partition_mapping(&mapping, true).unwrap().field_names().unwrap(),
            vec!["terrain_height".to_owned()]
        );
        assert_eq!(
            partition_mapping(&mapping, false).unwrap().field_names().unwrap(),
            vec!["surface_pressure".to_owned()]
        );
    }

    #[test]
    fn a_borrowed_field_is_published_even_where_its_donor_only_reads_it() {
        let primary = Mapping::load(&write(
            "m8.json",
            &mapping_document(r#", "soil_temperature": {"provider": "composition_bound"}"#),
        ))
        .unwrap();
        let donor = Mapping::load(&write(
            "m8-donor.json",
            &mapping_document(
                r#", "soil_temperature": {"selectors": [{"parameter": "TSOIL"}],
                    "source_axes": ["soil", "y", "x"], "target_axes": ["soil", "y", "x"],
                    "location": "soil", "units": {"target": "K"},
                    "missing": {"kind": "reject"}, "dependency_only": true}"#,
            ),
        ))
        .unwrap();
        assert!(donor.field("soil_temperature").unwrap().dependency_only().unwrap());
        let binding = Binding {
            name: "donor".to_owned(),
            source_id: "d".to_owned(),
            mapping_role: "donor_mapping".to_owned(),
            mapping_sha256: donor.sha256.clone(),
            data_role: "donor_data".to_owned(),
            provenance_role: "donor_provenance".to_owned(),
            fields: vec!["soil_temperature".to_owned()],
            grid_alignment: "exact_coordinate_subset".to_owned(),
            time_alignment: "valid_time_exact".to_owned(),
            vertical_merge: None,
            companion: None,
        };
        let donors = BTreeMap::from([("donor".to_owned(), donor)]);
        let union = union_mapping(&primary, &donors, &[binding]).unwrap();
        let field = union.field("soil_temperature").unwrap();
        assert_eq!(field.location().unwrap(), "soil");
        assert!(!field.dependency_only().unwrap());
    }

    #[test]
    fn a_binding_whose_files_publish_nothing_is_recorded_only_with_a_fallback_for_every_field() {
        // HRRR wrfsfc before 2020-12-02 carries no VEG record: the
        // vegetation binding is recorded UNPUBLISHED with the caller's
        // fallback and its field leaves the composed frame.  A binding with
        // any field lacking a fallback, or one that supplies terrain, is
        // refused as before (None).
        let binding = |fields: &[&str]| Binding {
            name: "vegetation_surface".to_owned(),
            source_id: "v".to_owned(),
            mapping_role: "vegetation_surface_mapping".to_owned(),
            mapping_sha256: "ab".repeat(32),
            data_role: "vegetation_surface_data".to_owned(),
            provenance_role: "vegetation_surface_provenance".to_owned(),
            fields: fields.iter().map(|name| (*name).to_owned()).collect(),
            grid_alignment: "exact_coordinate_subset".to_owned(),
            time_alignment: "source_cycle_analysis_broadcast".to_owned(),
            vertical_merge: None,
            companion: None,
        };
        let fallbacks = vec![(
            "vegetation_fraction".to_owned(),
            "static-greenfrac-monthly-climatology".to_owned(),
        )];
        let files = vec!["/d/sfc.grib2".to_owned()];
        let digests = BTreeMap::from([("/d/sfc.grib2".to_owned(), "cd".repeat(32))]);
        let provenance = "/d/p.json".to_owned();
        let provenance_digests = BTreeMap::from([(provenance.clone(), "ef".repeat(32))]);
        let record = |binding: &Binding, fallbacks: &[(String, String)]| {
            unpublished_record(
                binding, fallbacks, "/d/m.json", &"12".repeat(32), &files,
                &digests, &provenance, &provenance_digests,
            )
        };
        let recorded = record(&binding(&["vegetation_fraction"]), &fallbacks).unwrap();
        assert_eq!(recorded["alignment"]["status"], "UNPUBLISHED");
        assert_eq!(
            recorded["alignment"]["fallback"]["vegetation_fraction"],
            "static-greenfrac-monthly-climatology"
        );
        assert_eq!(recorded["data"][0]["sha256"], "cd".repeat(32));
        assert_eq!(recorded["provenance"]["sha256"], "ef".repeat(32));
        assert!(record(&binding(&["vegetation_fraction"]), &[]).is_none());
        assert!(record(&binding(&["vegetation_fraction", "land_fraction"]), &fallbacks).is_none());
        let terrain = vec![(EXTERNAL_FIELD.to_owned(), "x".to_owned())];
        assert!(record(&binding(&[EXTERNAL_FIELD]), &terrain).is_none());

        let mapping = Mapping::load(&write(
            "m9.json",
            &mapping_document(r#", "vegetation_fraction": {"provider": "composition_bound"}"#),
        ))
        .unwrap();
        assert!(mapping.field("vegetation_fraction").is_ok());
        let names = BTreeSet::from(["vegetation_fraction".to_owned()]);
        let reduced = without_fields(&mapping, &names).unwrap();
        assert!(reduced.field("vegetation_fraction").is_err());
        assert_eq!(reduced.sha256, mapping.sha256);
    }

    #[test]
    fn a_partition_keeps_the_sealed_identity_of_the_mapping_it_came_from() {
        // The frameset states the mapping the caller sealed.  A partition
        // that re-hashed its own edited document would bind the frames to
        // a document nobody verified.
        let mapping = Mapping::load(&write("m7.json", &mapping_document(""))).unwrap();
        let partition = partition_mapping(&mapping, true).unwrap();
        assert_eq!(partition.sha256, mapping.sha256);
        assert_eq!(partition.path, mapping.path);
    }

    fn member_collection(member: Option<&str>) -> crate::assemble::DecodedCollection {
        use chrono::NaiveDateTime;
        use ndarray::ArrayD;

        let valid = NaiveDateTime::parse_from_str(
            "1932-03-21 00:00:00", "%Y-%m-%d %H:%M:%S",
        )
        .unwrap();
        let member = member.map(str::to_owned);
        let mut direct = std::collections::BTreeMap::new();
        direct.insert(
            (valid, member.clone(), "surface_pressure".to_owned()),
            crate::assemble::DirectValue {
                name: "surface_pressure".to_owned(),
                valid_time: valid,
                member: member.clone(),
                source_cycle: valid,
                axes: vec!["y".to_owned(), "x".to_owned()],
                values: ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), 100000.0),
                missing_count: 0,
                references: vec!["fixture".to_owned()],
            },
        );
        let mut source_cycles = std::collections::BTreeMap::new();
        source_cycles.insert((valid, member), valid);
        crate::assemble::DecodedCollection {
            latitude: vec![10.0, 10.5],
            longitude: vec![20.0, 20.5],
            vertical_values: vec![100000.0],
            direct,
            source_cycles,
            grid_fingerprint: "fixture-grid".to_owned(),
            original_pressure_hpa: None,
            hybrid_a: Vec::new(),
            hybrid_b: Vec::new(),
        }
    }

    #[test]
    fn donor_completion_preserves_the_companion_upper_column() {
        let mapping = Mapping::load(&write("mc-complete.json", &companion_mapping("pressure", ""))).unwrap();
        let path = write("cc-complete.json", &composition_document(TERRAIN_SUPPLEMENT,
            &companion_sources("valid_time_exact", r#""air_temperature""#)));
        let composition = Composition::load(&path, &mapping).unwrap();
        let mut collection = member_collection(None);
        let mut value = collection.direct.values().next().unwrap().clone();
        collection.direct.clear();
        value.name = "land_fraction".into();
        value.values.fill(1.0);
        collection.direct.insert((value.valid_time, None, value.name.clone()), value.clone());
        value.name = "air_temperature".into();
        value.axes = vec!["vertical".into(), "y".into(), "x".into()];
        value.values = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 2, 2]), 250.0);
        collection.direct.insert((value.valid_time, None, value.name.clone()), value);
        let completed = complete_binding_donor(&mapping, collection, &composition.bindings[0]).unwrap();
        assert_eq!(completed.direct.len(), 2);
        let column = completed.direct.values().find(|v| v.name == "air_temperature").unwrap();
        assert_eq!(column.values.shape(), &[1, 2, 2]);
        assert!(column.values.iter().all(|v| *v == 250.0));
    }

    #[test]
    fn the_manifest_member_replaces_absent_product_membership() {
        let bound = bind_manifest_member(member_collection(None), "072").unwrap();
        let keys: Vec<_> = bound.direct.keys().cloned().collect();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].1.as_deref(), Some("072"));
        assert!(bound
            .source_cycles
            .keys()
            .all(|(_time, member)| member.as_deref() == Some("072")));
        assert!(bound
            .direct
            .values()
            .all(|value| value.member.as_deref() == Some("072")));
    }

    #[test]
    fn a_producer_declared_member_refuses_the_explicit_binding() {
        // The Python engine's sentence, verbatim: the two engines must
        // refuse the same bytes the same way.
        let refusal = bind_manifest_member(member_collection(Some("9")), "072").unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::MANIFEST_MISMATCH);
        assert_eq!(
            refusal.message,
            "the composition input manifest binds explicit member '072', but \
             the decoded records already encode product-defined member(s) \
             ['9']; an explicit member binding replaces ABSENT product \
             membership only, so stamping it here would overwrite a \
             producer-declared identity"
        );
    }
}
