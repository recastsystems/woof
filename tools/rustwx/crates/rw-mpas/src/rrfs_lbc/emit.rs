//! RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1280-1376 and
//! src/core_init_atmosphere/registry.chemistry.xml:96-108 at 1d6e370e.
//! Writing the CDF-5 `lbc` stream file.
//!
//! ## Schema by table, not by donor
//! Unlike the init emitter, which re-emits a native capsule's container, the
//! lbc container follows the fork registry (`lbc` stream,
//! `lbcs` package: the three `moist` scalars, `lbc_u`, `lbc_w`, `lbc_rho`,
//! `lbc_theta`, `xtime`, `Time`), so its dimensions, variables and attributes
//! are declared here as a table.  There is no donor file to make the
//! container trivially right; what this writer emits is checkable against a
//! native file only because the table was transcribed from the registry and
//! measured against the reference files.
//!
//! ## Header identity
//! Core identity is stamped for the v8.3.1 fork producer, while mesh metadata and
//! ancestry follow native bootstrapping, including its missing-attribute defaults.
//! `config_*` values come from producer switches or registry defaults, with
//! caller-supplied overrides for metadata only. `history`, `git_version`,
//! `file_id` and `gpuwm_provenance` describe this Rust producer truthfully;
//! header differences are reported separately from every stream field.
//!
//! ## Layout identity
//! The header is zero-padded so the data section starts at the same offset
//! SMIOL reserves ([`SMIOL_HEADER_RESERVE_BYTES`]), making the data sections
//! of a produced file and a native file comparable byte-for-byte at equal
//! offsets.

use std::collections::BTreeMap;
use std::path::Path;

use rw_store::netcdf_classic::{
    NcAttr, NcAttrValue, NcClassicWriter, NcData, NcDim, NcFormat, NcSlab, NcType, NcVarDef,
};

use crate::error::{MpasError, MpasResult};
use crate::rrfs_lbc::LbcConfig;

/// The header space SMIOL reserves ahead of the data section, measured on the
/// native v8.4.1 `init_atmosphere_model` lbc and init writes (data begins at
/// 0x20000 in both).  Reproduced so data offsets line up file-to-file.
pub const SMIOL_HEADER_RESERVE_BYTES: u64 = 131_072;

/// `lbc.$Y-$M-$D_$h.$m.$s.nc` from a `YYYY-MM-DD_HH:MM:SS` stamp: the
/// stream's own filename template.
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
pub fn lbc_file_name(valid_time: &str) -> String {
    format!("lbc.{}.nc", valid_time.replace(':', "."))
}

/// Native bootstrapped mesh attributes and the complete parent chain.
#[derive(Debug)]
pub struct HeaderSource {
    carried: Vec<NcAttr>,
    pub parent_id: String,
}

const CARRIED_LEADING: [&str; 6] = [
    "model_name",
    "core_name",
    "version",
    "source",
    "Conventions",
    "git_version",
];
const CARRIED_MESH: [&str; 5] = [
    "on_a_sphere",
    "sphere_radius",
    "is_periodic",
    "x_period",
    "y_period",
];

/// MPAS v8.4.2 src/framework/mpas_bootstrapping.F:285-381:
/// failed text attribute reads take the native bootstrap default.
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn mesh_text(file: &netcrust::File, name: &str, default: &str) -> String {
    file.attribute(name)
        .and_then(|a| a.as_string().map(|s| s.trim_end_matches(' ').to_string()))
        .unwrap_or_else(|| default.to_string())
}

/// MPAS v8.4.2 src/framework/mpas_bootstrapping.F:300-308,364-381:
/// the input attribute is converted to RKIND, not carried in its donor type.
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn mesh_real(file: &netcrust::File, name: &str, default: f32) -> f32 {
    file.attribute(name)
        .map(|a| match a.value() {
            netcrust::AttributeValue::Floats(v) if v.len() == 1 => v[0],
            netcrust::AttributeValue::Doubles(v) if v.len() == 1 => v[0] as f32,
            netcrust::AttributeValue::Ints(v) if v.len() == 1 => v[0] as f32,
            _ => default,
        })
        .unwrap_or(default)
}

impl HeaderSource {
    /// MPAS v8.4.2 src/framework/mpas_bootstrapping.F:285-381 and
    /// src/driver/mpas_subdriver.F:513-530. Core metadata is supplied by the
    /// producer (core_init_atmosphere/mpas_init_atm_core_interface.F:38-61),
    /// not required in its input mesh. A bare regional mesh is valid input.
    /// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
    pub fn from_grid(grid_path: &Path) -> MpasResult<HeaderSource> {
        let file = netcrust::File::open(grid_path)?;
        let on_a_sphere = mesh_text(&file, "on_a_sphere", "YES").contains("YES");
        let periodic = !on_a_sphere && mesh_text(&file, "is_periodic", "NO").contains("YES");
        let mut carried = vec![
            NcAttr::text("model_name", "mpas"),
            NcAttr::text("core_name", "init_atmosphere"),
            NcAttr::text("version", "8.3.1"),
            NcAttr::text("source", "MPAS"),
            NcAttr::text("Conventions", "MPAS"),
            NcAttr::text(
                "git_version",
                concat!("gpuwm-", env!("GPUWM_BRIDGE_SOURCE_REV")),
            ),
        ];
        for (name, value) in [
            ("on_a_sphere", logical(on_a_sphere)),
            (
                "sphere_radius",
                float(mesh_real(&file, "sphere_radius", 1.0)),
            ),
            ("is_periodic", logical(periodic)),
            (
                "x_period",
                float(if periodic {
                    mesh_real(&file, "x_period", 0.0)
                } else {
                    0.0
                }),
            ),
            (
                "y_period",
                float(if periodic {
                    mesh_real(&file, "y_period", 0.0)
                } else {
                    0.0
                }),
            ),
            ("mesh_spec", text(&mesh_text(&file, "mesh_spec", "0.0"))),
        ] {
            carried.push(NcAttr {
                name: name.to_string(),
                value,
            });
        }
        let file_id = mesh_text(&file, "file_id", "");
        let previous_parent = mesh_text(&file, "parent_id", "");
        let parent_id = if file_id.is_empty() {
            previous_parent
        } else {
            format!("{file_id}\n{previous_parent}")
        };
        Ok(HeaderSource { carried, parent_id })
    }
}

/// One row of the config-attribute table.
enum ConfigValue {
    /// Producer switch: the value comes from [`LbcConfig`], never from an
    /// override.
    Knob(NcAttrValue),
    /// Namelist metadata: the v8.4.2 registry default, overridable by name.
    Default(NcAttrValue),
}

/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn text(v: &str) -> NcAttrValue {
    NcAttrValue::Text(v.to_string())
}
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn int(v: i32) -> NcAttrValue {
    NcAttrValue::Ints(vec![v])
}
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn float(v: f32) -> NcAttrValue {
    NcAttrValue::Floats(vec![v])
}
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn logical(v: bool) -> NcAttrValue {
    text(if v { "YES" } else { "NO" })
}

/// The `config_*` block, in the registry's declaration order, which is the
/// order the native writer stamps it in. Defaults are the v8.4.2
/// `core_init_atmosphere` Registry.xml `default_value`s, transcribed; rows
/// marked `Knob` take the producer's actual switch instead and refuse an
/// override, because the header must not claim a configuration the run did
/// not use.
/// MPAS v8.4.2 src/core_init_atmosphere/Registry.xml:54-370.
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
pub fn config_attributes(cfg: &LbcConfig, n_vert_levels: usize) -> MpasResult<Vec<NcAttr>> {
    use ConfigValue::{Default, Knob};
    let rows: Vec<(&str,ConfigValue)>=vec![
        ("config_init_case", Knob(int(9))),
        ("config_calendar_type", Default(text("gregorian"))),
        ("config_start_time", Knob(text(&cfg.start_time))),
        ("config_stop_time", Knob(text(&cfg.stop_time))),
        ("config_theta_adv_order", Knob(int(cfg.theta_adv_order))),
        ("config_coef_3rd_order", Knob(float(cfg.coef_3rd_order))),
        ("config_num_halos", Default(int(2))),
        ("config_interface_projection", Default(text("linear_interpolation"))),
        ("config_nvertlevels", Knob(int(n_vert_levels as i32))),
        ("config_nsoillevels", Default(int(4))),
        ("config_nfglevels", Knob(int(cfg.n_fg_levels as i32))),
        ("config_nfgsoillevels", Default(int(4))),
        ("config_gocartlevels", Knob(int(cfg.gocart_levels as i32))),
        ("config_months", Default(int(12))),
        ("config_nsoilcat", Default(int(16))),
        ("config_nvegopt", Default(int(1))),
        ("config_geog_data_path", Default(text("/glade/work/wrfhelp/WPS_GEOG/"))),
        ("config_met_prefix", Default(text("CFSR"))),
        ("config_sfc_prefix", Default(text("SST"))),
        ("config_fg_interval", Knob(int(cfg.fg_interval_seconds as i32))),
        ("config_landuse_data", Default(text("MODIFIED_IGBP_MODIS_NOAH"))),
        ("config_soilcat_data", Default(text("STATSGO"))),
        ("config_topo_data", Default(text("GMTED2010"))),
        ("config_vegfrac_data", Default(text("MODIS"))),
        ("config_albedo_data", Default(text("MODIS"))),
        ("config_maxsnowalbedo_data", Default(text("MODIS"))),
        ("config_supersample_factor", Default(int(3))),
        ("config_lu_supersample_factor", Default(int(1))),
        ("config_30s_supersample_factor", Default(int(1))),
        ("config_noahmp_static", Default(logical(true))),
        ("config_use_spechumd", Knob(logical(cfg.use_spechumd))),
        ("config_lai_data", Default(text("MODIS"))),
        ("config_ztop", Default(float(30000.0))),
        ("config_nsmterrain", Default(int(1))),
        ("config_smooth_surfaces", Default(logical(true))),
        ("config_dzmin", Default(float(0.3))),
        ("config_nsm", Default(int(30))),
        ("config_tc_vertical_grid", Default(logical(true))),
        ("config_specified_zeta_levels", Default(text(""))),
        ("config_blend_bdy_terrain", Default(logical(false))),
        ("config_extrap_airtemp", Knob(text(cfg.extrap_airtemp.label()))),
        ("config_static_interp", Default(logical(true))),
        ("config_native_gwd_static", Default(logical(true))),
        ("config_native_gwd_gsl_static", Default(logical(false))),
        ("config_gwd_cell_scaling", Default(float(1.0))),
        ("config_vertical_grid", Default(logical(true))),
        ("config_met_interp", Default(logical(true))),
        ("config_tempo_rap", Knob(logical(cfg.tempo_rap))),
        ("config_input_sst", Default(logical(false))),
        ("config_frac_seaice", Default(logical(true))),
        ("config_tsk_seaice_threshold", Default(float(100.0))),
        ("config_lbc_hydrometeors_rrfs", Knob(logical(cfg.hydrometeors_rrfs))),
        ("config_lbc_hydrometeors_gfs", Knob(logical(cfg.hydrometeors_gfs))),
        ("config_pio_num_iotasks", Default(int(0))),
        ("config_pio_stride", Default(int(1))),
        ("config_block_decomp_file_prefix", Default(text("x1.40962.graph.info.part."))),
        ("config_smoke_scheme", Knob(text(if cfg.smoke {"on"} else {"off"}))),
        ("config_dust_scheme", Knob(text(if cfg.dust {"on"} else {"off"}))),
        ("config_number_of_blocks", Default(int(0))),
        ("config_explicit_proc_decomp", Default(logical(false))),
        ("config_proc_decomp_file_prefix", Default(text("graph.info.part."))),
    ];

    let known: BTreeMap<&str, bool> = rows
        .iter()
        .map(|(n, v)| (*n, matches!(v, ConfigValue::Knob(_))))
        .collect();
    for (name, _) in cfg.attr_overrides.iter() {
        match known.get(name.as_str()) {
            None => {
                return Err(MpasError::Refusal(format!(
                    "--config-attrs names {name}, which is not a v8.4.2 init_atmosphere \
                     namelist attribute this table declares; a misspelled name would silently \
                     leave the default in the header"
                )))
            }
            Some(true) => {
                return Err(MpasError::Refusal(format!(
                    "--config-attrs tries to override {name}, which is a producer switch; the \
                     header must record the value the run actually used, so set the switch \
                     itself instead"
                )))
            }
            Some(false) => {}
        }
    }

    let mut out = Vec::with_capacity(rows.len());
    for (name, value) in rows {
        let resolved = match value {
            ConfigValue::Knob(v) => v,
            ConfigValue::Default(v) => match cfg.attr_overrides.get(name) {
                None => v,
                Some(json) => override_value(name, &v, json)?,
            },
        };
        out.push(NcAttr {
            name: name.to_string(),
            value: resolved,
        });
    }
    Ok(out)
}

/// Apply one override, typed by the default it replaces.
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn override_value(
    name: &str,
    default: &NcAttrValue,
    json: &serde_json::Value,
) -> MpasResult<NcAttrValue> {
    let refuse = |wanted: &str| {
        MpasError::Refusal(format!(
            "--config-attrs value for {name} is {json}; this attribute takes {wanted}"
        ))
    };
    Ok(match default {
        NcAttrValue::Text(current) => {
            // Logical attributes are stored as YES/NO text; accept JSON
            // booleans for them and plain strings for the rest.
            if let Some(b) = json.as_bool() {
                if current == "YES" || current == "NO" {
                    NcAttrValue::Text(if b { "YES" } else { "NO" }.to_string())
                } else {
                    return Err(refuse("a string"));
                }
            } else {
                NcAttrValue::Text(json.as_str().ok_or_else(|| refuse("a string"))?.to_string())
            }
        }
        NcAttrValue::Ints(_) => NcAttrValue::Ints(vec![json
            .as_i64()
            .ok_or_else(|| refuse("an integer"))?
            as i32]),
        NcAttrValue::Floats(_) => {
            NcAttrValue::Floats(vec![json.as_f64().ok_or_else(|| refuse("a number"))? as f32])
        }
        other => {
            return Err(MpasError::Refusal(format!(
                "attribute {name} has an unoverridable default shape {other:?}"
            )))
        }
    })
}

/// The seven field slabs of one boundary time, flat in file order (point
/// major, vertical fastest).
pub struct LbcFields<'a> {
    pub qv: &'a [f32],
    pub extra: BTreeMap<String,&'a [f32]>,
    pub u: &'a [f32],
    pub w: &'a [f32],
    pub rho: &'a [f32],
    pub theta: &'a [f32],
}

/// Mint the deterministic per-file id, same alphabet and length as the init
/// emitter's.
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn mint_file_id(seed: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(seed.as_bytes());
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
    digest
        .iter()
        .take(10)
        .map(|b| ALPHABET[(*b as usize) % ALPHABET.len()] as char)
        .collect()
}

/// Write one boundary time's file.
#[allow(clippy::too_many_arguments)]
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
pub fn write_lbc_file(
    out_path: &Path,
    cfg: &LbcConfig,
    header: &HeaderSource,
    n_cells: usize,
    n_edges: usize,
    nz: usize,
    valid_time: &str,
    time_seconds: f32,
    fields: &LbcFields<'_>,
) -> MpasResult<std::fs::File> {
    let nzp1 = nz + 1;
    let check = |name: &str, got: usize, want: usize| -> MpasResult<()> {
        if got != want {
            return Err(MpasError::Refusal(format!(
                "computed {name} holds {got} value(s); the file's slab is {want}.  A mismatched \
                 count here is the cell-major/level-major transpose, which produces a readable \
                 file full of wrong numbers"
            )));
        }
        Ok(())
    };
    check("lbc_qv", fields.qv.len(), n_cells * nz)?;
    for (name,values) in &fields.extra {check(name,values.len(),n_cells*nz)?;}
    check("lbc_u", fields.u.len(), n_edges * nz)?;
    check("lbc_w", fields.w.len(), n_cells * nzp1)?;
    check("lbc_rho", fields.rho.len(), n_cells * nz)?;
    check("lbc_theta", fields.theta.len(), n_cells * nz)?;

    // Dimensions in the native file's order.
    let dims = vec![
        NcDim::fixed("nVertLevels", nz),
        NcDim::fixed("nCells", n_cells),
        NcDim::record("Time"),
        NcDim::fixed("nEdges", n_edges),
        NcDim::fixed("nVertLevelsP1", nzp1),
        NcDim::fixed("StrLen", 64),
    ];
    const D_NZ: usize = 0;
    const D_CELLS: usize = 1;
    const D_TIME: usize = 2;
    const D_EDGES: usize = 3;
    const D_NZP1: usize = 4;
    const D_STRLEN: usize = 5;

    let cell_var = |name: &str, units: &str, long_name: &str| -> NcVarDef {
        NcVarDef::new(name, NcType::Float, vec![D_TIME, D_CELLS, D_NZ]).with_attrs(vec![
            NcAttr::text("units", units),
            NcAttr::text("long_name", long_name),
        ])
    };
    let mut vars = vec![
        cell_var(
            "lbc_qv",
            "kg kg^{-1}",
            "Water vapor mixing ratio on lateral boundary cells",
        ),
        NcVarDef::new("lbc_u", NcType::Float, vec![D_TIME, D_EDGES, D_NZ]).with_attrs(vec![
            NcAttr::text("units", "m s^{-1}"),
            NcAttr::text(
                "long_name",
                "Horizontal normal velocity on domain lateral boundary edges",
            ),
        ]),
        NcVarDef::new("lbc_w", NcType::Float, vec![D_TIME, D_CELLS, D_NZP1]).with_attrs(vec![
            NcAttr::text("units", "m s^{-1}"),
            NcAttr::text(
                "long_name",
                "Vertical velocity on domain lateral boundary vertical cell faces",
            ),
        ]),
        cell_var(
            "lbc_rho",
            "kg m^{-3}",
            "Dry air density on lateral boundary cells",
        ),
        cell_var(
            "lbc_theta",
            "K",
            "Potential temperature on lateral boundary cells",
        ),
        NcVarDef::new("xtime", NcType::Char, vec![D_TIME, D_STRLEN]).with_attrs(vec![
            NcAttr::text("units", "YYYY-MM-DD_hh:mm:ss"),
            NcAttr::text("long_name", "Model valid time"),
        ]),
        NcVarDef::new("Time", NcType::Float, vec![D_TIME]).with_attrs(vec![
            NcAttr::text(
                "units",
                format!("seconds since {}", space_separated(&cfg.start_time)?),
            ),
            NcAttr::text("long_name", "CF-compliant valid time"),
            NcAttr::text("standard_name", "time"),
        ]),
    ];

    let mut scalar_vars=Vec::new();
    if fields.extra.contains_key("lbc_qc") { scalar_vars.push(cell_var("lbc_qc","kg kg^{-1}","Cloud water mixing ratio on lateral boundary cells")); }
    if fields.extra.contains_key("lbc_qr") { scalar_vars.push(cell_var("lbc_qr","kg kg^{-1}","Rain water mixing ratio on lateral boundary cells")); }
    if fields.extra.contains_key("lbc_qi") { scalar_vars.push(cell_var("lbc_qi","kg kg^{-1}","Lateral boundary tendency of ice mixing ratio")); }
    if fields.extra.contains_key("lbc_qs") { scalar_vars.push(cell_var("lbc_qs","kg kg^{-1}","Lateral boundary tendency of snow mixing ratio")); }
    if fields.extra.contains_key("lbc_qg") { scalar_vars.push(cell_var("lbc_qg","kg kg^{-1}","Lateral boundary tendency of graupel mixing ratio")); }
    if fields.extra.contains_key("lbc_qh") { scalar_vars.push(cell_var("lbc_qh","kg kg^{-1} s^{-1}","Lateral boundary tendency of hail mixing ratio")); }
    if fields.extra.contains_key("lbc_nr") { scalar_vars.push(cell_var("lbc_nr","kg^{-1}","Lateral boundary tendency of rain number concentration")); }
    if fields.extra.contains_key("lbc_ni") { scalar_vars.push(cell_var("lbc_ni","kg^{-1}","Lateral boundary tendency of ice number concentration")); }
    if fields.extra.contains_key("lbc_nc") { scalar_vars.push(cell_var("lbc_nc","kg^{-1}","Lateral boundary tendency of droplet number concentration")); }
    if fields.extra.contains_key("lbc_ns") { scalar_vars.push(cell_var("lbc_ns","kg^{-1}","Lateral boundary tendency of snow number concentration")); }
    if fields.extra.contains_key("lbc_ng") { scalar_vars.push(cell_var("lbc_ng","kg^{-1}","Lateral boundary tendency of graupel number concentration")); }
    if fields.extra.contains_key("lbc_nh") { scalar_vars.push(cell_var("lbc_nh","m^{-3} s^{-1}","Lateral boundary tendency of hail number concentration")); }
    if fields.extra.contains_key("lbc_nccn") { scalar_vars.push(cell_var("lbc_nccn","m^{-3} s^{-1}","Lateral boundary tendency of CCN number concentration")); }
    if fields.extra.contains_key("lbc_volg") { scalar_vars.push(cell_var("lbc_volg","m^{3} s^{-1}","Lateral boundary tendency of graupel particle volume")); }
    if fields.extra.contains_key("lbc_volh") { scalar_vars.push(cell_var("lbc_volh","m^{3} s^{-1}","Lateral boundary tendency of hail particle volume")); }
    if fields.extra.contains_key("lbc_nifa") { scalar_vars.push(cell_var("lbc_nifa","nb kg^{-1}","Gocart ice-friendly aerosol number concentration on lateral boundary cells")); }
    if fields.extra.contains_key("lbc_nwfa") { scalar_vars.push(cell_var("lbc_nwfa","nb kg^{-1}","Gocart water-friendly aerosol number concentration on lateral boundary cells")); }
    // RRFSx/MPAS-Model src/core_init_atmosphere/registry.chemistry.xml:96-108
    // at 1d6e370e. Optional chemistry scalars extend the same lbc scalar array;
    // every selected computed field must have a declared file slab.
    for (name,description) in [
        ("lbc_smoke_fine", "Lateral boundary tendency of fine smoke mixing ratio"),
        ("lbc_dust_fine", "Lateral boundary tendency of fine dust mixing ratio"),
        ("lbc_dust_coarse", "Lateral boundary tendency of coarse dust mixing ratio"),
    ] {
        if fields.extra.contains_key(name) {
            scalar_vars.push(cell_var(name,"kg kg^{-1} s^{-1}",description));
        }
    }
    vars.splice(1..1,scalar_vars);

    // Global attributes: carried lineage, owned identity, stamped config.
    let mut gattrs: Vec<NcAttr> = Vec::new();
    // model_name .. y_period.
    for a in header
        .carried
        .iter()
        .take(CARRIED_LEADING.len() + CARRIED_MESH.len())
    {
        gattrs.push(a.clone());
    }
    gattrs.push(NcAttr::text("history", "rw_mpas_lbc"));
    gattrs.push(NcAttr::text("parent_id", header.parent_id.clone()));
    // mesh_spec, carried last.
    gattrs.push(
        header
            .carried
            .last()
            .expect("HeaderSource always carries mesh_spec")
            .clone(),
    );
    gattrs.extend(config_attributes(cfg, nz)?);
    let seed = format!(
        "{}|{}|{}",
        cfg.grid_path.display(),
        cfg.start_time,
        valid_time
    );
    gattrs.push(NcAttr::text("file_id", mint_file_id(&seed)));
    gattrs.push(NcAttr::text("gpuwm_provenance", match cfg.absent_numbers {
        crate::rrfs_init::scalars::AbsentNumbers::Zero=>cfg.provenance.clone(),
        crate::rrfs_init::scalars::AbsentNumbers::FromMass=>format!(
            "{}; absent rain/cloud/ice number diagnosed from mass (fork calcnfromq, cases.F:8835-8849 enabled)",
            cfg.provenance),
    }));

    let mut writer = NcClassicWriter::create_with_min_header(
        out_path,
        NcFormat::Data64,
        dims,
        gattrs,
        vars,
        1,
        SMIOL_HEADER_RESERVE_BYTES,
    )
    .map_err(|e| MpasError::Refusal(format!("cannot lay out {}: {e}", out_path.display())))?;

    // Every boundary field at once, each encoded and written at its own offset
    // on the cores; the file holds the same bytes as writing them in turn.
    // THE BREAKAGE IT FIXES: one core encoded the 13 GB of an RRFS v2 conus3km
    // boundary hour field by field (box V5090, 2026-10-28).
    let mut named: Vec<(&str, &[f32])> = vec![("lbc_qv", fields.qv)];
    for (name, values) in &fields.extra {named.push((name.as_str(), *values));}
    named.extend([("lbc_u", fields.u), ("lbc_w", fields.w), ("lbc_rho", fields.rho), ("lbc_theta", fields.theta)]);
    let keys: Vec<(&str, Option<u64>)> = named.iter().map(|(name, _)| (*name, Some(0))).collect();
    writer
        .put_parallel(&keys, |i| Ok(NcSlab::Borrowed(NcData::Floats(named[i].1))))
        .map_err(|e| MpasError::Refusal(format!("cannot write the boundary fields: {e}")))?;

    // xtime is space-padded to StrLen, the way the native stream writes it.
    let mut xtime = valid_time.as_bytes().to_vec();
    if xtime.len() > 64 {
        return Err(MpasError::Refusal(format!(
            "valid time '{valid_time}' is {} byte(s); xtime holds 64",
            xtime.len()
        )));
    }
    xtime.resize(64, b' ');
    writer
        .put_record("xtime", 0, NcData::Chars(&xtime))
        .map_err(|e| MpasError::Refusal(format!("cannot write xtime: {e}")))?;
    writer
        .put_record("Time", 0, NcData::Floats(&[time_seconds]))
        .map_err(|e| MpasError::Refusal(format!("cannot write Time: {e}")))?;

    // The fsync is the caller's: build_lbc runs it beside the next boundary
    // hour, and waits for it before the run reports success.
    writer
        .finish_unsynced()
        .map_err(|e| MpasError::Refusal(format!("cannot finish {}: {e}", out_path.display())))
}

/// `YYYY-MM-DD_HH:MM:SS` to the `YYYY-MM-DD HH:MM:SS` form the `Time`
/// units string uses.
/// RRFSx/MPAS-Model src/core_init_atmosphere/Registry.xml:1282-1370 at 1d6e370e.
fn space_separated(stamp: &str) -> MpasResult<String> {
    if stamp.len() < 19 || stamp.as_bytes().get(10) != Some(&b'_') {
        return Err(MpasError::Refusal(format!(
            "start time '{stamp}' is not YYYY-MM-DD_HH:MM:SS"
        )));
    }
    let mut s = stamp.to_string();
    s.replace_range(10..11, " ");
    Ok(s)
}
