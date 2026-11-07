//! The `rw-wps.mapping.v1` document, as the decode path reads it.
//!
//! This is the port of the READ side of `gpuwm.mapped_source.load_mapping`:
//! the accessors the data path uses, plus the structural refusals a
//! decoder cannot proceed without.  The full authoring grammar (allowed
//! key sets, cross-field policy, the vocabulary a mapping author is held
//! to) stays in Python: `load_mapping` is the mapping's validator of
//! record and lane 3 re-runs it around every engine call, so duplicating
//! all 670 lines of it here would create a second authority that could
//! disagree with the first.  What lives here is what the DECODE needs to
//! be well defined, and every read of a malformed value refuses by name.

use crate::node::Node;
use crate::refusal::{mapping_invalid, Result};

pub const MAPPING_SCHEMA: &str = "rw-wps.mapping.v1";

pub const GRID_FAMILY_REGULAR: &str = "regular_latitude_longitude";
pub const GRID_FAMILY_LAMBERT: &str = "lambert_conformal";
/// GRIB2 GDT 3.1: a regular latitude/longitude lattice in a ROTATED
/// spherical frame (ECCC's HRDPS and RDPS publish on one).  Projected
/// like Lambert: its canonical axes are distances along the rotated
/// lattice in [`PROJECTED_AXIS_UNIT_M`] units, never degrees, so no
/// consumer can read a rotated coordinate as a geographic one.
pub const GRID_FAMILY_ROTATED: &str = "rotated_latitude_longitude";

/// Axis unit for projected regular grids: one unit = 100 km along the
/// projection plane (`mapped_source.PROJECTED_AXIS_UNIT_M`).
pub const PROJECTED_AXIS_UNIT_M: f64 = 100_000.0;

/// Canonical wind pairs a grid-relative source must rotate to the earth
/// basis at decode time (`mapped_source._ROTATED_WIND_PAIRS`).
pub const ROTATED_WIND_PAIRS: [(&str, &str); 2] = [
    ("eastward_wind", "northward_wind"),
    ("eastward_wind_10m", "northward_wind_10m"),
];

/// `resolution_flags` bit 0x08: vector components are grid-relative.
pub const GRIB2_GRID_RELATIVE_WIND_BIT: u8 = 0x08;

/// One mapping document plus the identity the frames carry.
pub struct Mapping {
    pub doc: Node,
    pub sha256: String,
    pub path: String,
}

impl Mapping {
    pub fn load(path: &str) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|error| {
            crate::refusal::missing_input(format!("cannot read mapping {path}: {error}"))
        })?;
        let sha256 = crate::digest::bytes_sha256(&bytes);
        let doc = Node::parse(&bytes)
            .map_err(|error| mapping_invalid(format!("mapping {path} is not JSON: {error}")))?;
        if !doc.is_object() {
            return Err(mapping_invalid(format!(
                "mapping {path} must be a JSON object"
            )));
        }
        let mapping = Self {
            doc,
            sha256,
            path: path.to_owned(),
        };
        let schema = mapping.string("schema")?;
        if schema != MAPPING_SCHEMA {
            return Err(mapping_invalid(format!(
                "mapping {path} declares schema {schema:?}; this engine \
                 implements {MAPPING_SCHEMA}"
            )));
        }
        let format = mapping.format()?;
        if !matches!(format, "grib1" | "grib2" | "netcdf") {
            return Err(mapping_invalid(format!(
                "unsupported mapped source format {format:?}"
            )));
        }
        Ok(mapping)
    }

    pub fn string(&self, key: &str) -> Result<&str> {
        self.doc
            .get(key)
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid(format!("mapping.{key} must be a string")))
    }

    pub fn format(&self) -> Result<&str> {
        self.string("format")
    }

    pub fn name(&self) -> Result<&str> {
        self.string("name")
    }

    pub fn target(&self) -> Result<&Node> {
        self.doc
            .get("target")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.target must be an object"))
    }

    pub fn coordinates(&self) -> Result<&Node> {
        self.doc
            .get("coordinates")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.coordinates must be an object"))
    }

    pub fn vertical(&self) -> Result<&Node> {
        self.coordinates()?
            .get("vertical")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.coordinates.vertical must be an object"))
    }

    /// `mapping.coordinates.vertical.kind`.
    pub fn vertical_kind(&self) -> Result<&str> {
        self.vertical()?
            .get("kind")
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid("mapping.coordinates.vertical.kind must be a string"))
    }

    /// Inline `vertical.hybrid_a`/`hybrid_b` literal arrays: the
    /// declared-data fallback coefficient channel.  `None` when neither
    /// is declared; both-or-neither is the contract.
    pub fn hybrid_literals(&self) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
        let vertical = self.vertical()?;
        let read = |key: &str| -> Result<Option<Vec<f64>>> {
            let Some(node) = vertical.field(key) else {
                return Ok(None);
            };
            node.items()
                .iter()
                .map(|item| {
                    item.as_f64().ok_or_else(|| {
                        mapping_invalid(format!(
                            "vertical.{key} must be a non-empty numeric list"
                        ))
                    })
                })
                .collect::<Result<Vec<f64>>>()
                .map(Some)
        };
        match (read("hybrid_a")?, read("hybrid_b")?) {
            (None, None) => Ok(None),
            (Some(a), Some(b)) => Ok(Some((a, b))),
            _ => Err(mapping_invalid(
                "vertical.hybrid_a and vertical.hybrid_b must be declared \
                 together; one half of a coefficient pair prices no pressure",
            )),
        }
    }

    /// The N+1 interface ladder, empty when the mapping declares none.
    pub fn interface_levels(&self) -> Result<Vec<f64>> {
        let Some(values) = self.vertical()?.get("interface_levels").filter(|v| v.is_array()) else {
            return Ok(Vec::new());
        };
        values.items().iter().map(|value| value.as_f64().ok_or_else(|| mapping_invalid("interface_levels must be numeric"))).collect()
    }

    /// The declared vertical ladder, in the mapping's own order.  Empty
    /// means "take the levels the file offers" (`_assemble_grib`).
    pub fn declared_levels(&self) -> Result<Vec<f64>> {
        let vertical = self.vertical()?;
        let Some(levels) = vertical.field("levels") else {
            return Ok(Vec::new());
        };
        levels
            .items()
            .iter()
            .map(|item| {
                item.as_f64().ok_or_else(|| {
                    mapping_invalid("mapping.coordinates.vertical.levels must be numbers")
                })
            })
            .collect()
    }

    /// `vertical.era_ladders`: the whole ladders other publications of
    /// the product carry, each drawn from `levels`.  Empty when the
    /// mapping declares none (`mapped_source._carried_ladder`).
    pub fn era_ladders(&self) -> Result<Vec<Vec<f64>>> {
        let vertical = self.vertical()?;
        let Some(ladders) = vertical.field("era_ladders") else {
            return Ok(Vec::new());
        };
        ladders
            .items()
            .iter()
            .map(|ladder| {
                ladder
                    .items()
                    .iter()
                    .map(|item| {
                        item.as_f64().ok_or_else(|| {
                            mapping_invalid(
                                "mapping.coordinates.vertical.era_ladders must be \
                                 lists of numbers",
                            )
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Field names in declared order.
    pub fn field_names(&self) -> Result<Vec<String>> {
        let fields = self
            .doc
            .get("fields")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.fields must be an object"))?;
        Ok(fields
            .entries()
            .iter()
            .map(|(name, _)| name.clone())
            .collect())
    }

    pub fn field(&self, name: &str) -> Result<FieldSpec<'_>> {
        let raw = self
            .doc
            .get("fields")
            .and_then(|fields| fields.get(name))
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid(format!("mapping.fields.{name} must be an object")))?;
        Ok(FieldSpec {
            name: name.to_owned(),
            raw,
        })
    }

    pub fn fields(&self) -> Result<Vec<FieldSpec<'_>>> {
        self.field_names()?
            .into_iter()
            .map(|name| self.field(&name))
            .collect()
    }

    pub fn derivation(&self, name: &str) -> Option<&Node> {
        self.doc
            .field("derivations")?
            .items()
            .iter()
            .find(|item| item.get("name").and_then(Node::as_str) == Some(name))
    }

    pub fn required_field_names(&self) -> Result<Vec<String>> {
        let target = self.target()?;
        let required = target
            .get("required_fields")
            .filter(|node| node.is_array())
            .ok_or_else(|| mapping_invalid("mapping.target.required_fields must be an array"))?;
        required
            .items()
            .iter()
            .map(|item| {
                item.get("name")
                    .and_then(Node::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        mapping_invalid("mapping.target.required_fields[].name must be a string")
                    })
            })
            .collect()
    }

    pub fn soil_layer_count(&self) -> Result<Option<i64>> {
        let Some(node) = self.target()?.field("soil_layer_count") else {
            return Ok(None);
        };
        node.as_i64()
            .map(Some)
            .ok_or_else(|| mapping_invalid("mapping.target.soil_layer_count must be an integer"))
    }

    /// The mapping's normalized grid declaration, defaulting to regular
    /// (`mapped_source._mapping_grid_declaration`).
    pub fn grid_declaration(&self) -> Result<GridDeclaration> {
        let Some(grid) = self.doc.field("grid") else {
            return Ok(GridDeclaration::regular());
        };
        let family = grid
            .get("family")
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid("mapping.grid.family must be a string"))?;
        let same_grid_pairing = match grid.get("same_grid_pairing") {
            None | Some(Node::Null) => None,
            Some(Node::String(value)) if value == "identity" && family == GRID_FAMILY_LAMBERT => {
                Some(value.clone())
            }
            _ => return Err(mapping_invalid(
                "mapping.grid.same_grid_pairing must be 'identity' on a \
                 declared Lambert grid; other sources retain projected pairing",
            )),
        };
        if family == GRID_FAMILY_REGULAR {
            return Ok(GridDeclaration::regular());
        }
        if family != GRID_FAMILY_LAMBERT && family != GRID_FAMILY_ROTATED {
            return Err(mapping_invalid(format!(
                "unsupported mapping.grid.family {family:?}; supported: \
                 {GRID_FAMILY_REGULAR}, {GRID_FAMILY_LAMBERT}, {GRID_FAMILY_ROTATED}"
            )));
        }
        let wind_basis = grid
            .get("wind_basis")
            .and_then(Node::as_str)
            .unwrap_or_default();
        if !matches!(wind_basis, "earth_relative" | "grid_relative_with_rotation") {
            return Err(mapping_invalid(format!(
                "a {family} grid must declare wind_basis \
                 'earth_relative' or 'grid_relative_with_rotation'"
            )));
        }
        let parameters = grid
            .get("parameters")
            .filter(|node| node.is_object())
            .ok_or_else(|| mapping_invalid("mapping.grid.parameters must be an object"))?;
        if family == GRID_FAMILY_ROTATED {
            if grid.get("published_first_points").is_some() {
                return Err(mapping_invalid(
                    "mapping.grid.published_first_points is a Lambert admission; \
                     a rotated_latitude_longitude grid declares one first point",
                ));
            }
            let rotated = RotatedParameters::from_node(parameters)?;
            return Ok(GridDeclaration {
                family: GRID_FAMILY_ROTATED,
                wind_basis: wind_basis.to_owned(),
                parameters: None,
                rotated: Some(rotated),
                same_grid_pairing: None,
                published_first_points: Vec::new(),
            });
        }
        let number = |key: &str| -> Result<f64> {
            parameters
                .get(key)
                .and_then(Node::as_f64)
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    mapping_invalid(format!(
                        "mapping.grid.parameters.{key} must be a finite number"
                    ))
                })
        };
        let integer = |key: &str| -> Result<i64> {
            parameters.get(key).and_then(Node::as_i64).ok_or_else(|| {
                mapping_invalid(format!("mapping.grid.parameters.{key} must be an integer"))
            })
        };
        let lambert = LambertParameters {
            latin1: number("latin1")?,
            latin2: number("latin2")?,
            lov: number("lov")?,
            lat1: number("lat1")?,
            lon1: number("lon1")?,
            dx_m: number("dx_m")?,
            dy_m: number("dy_m")?,
            nx: integer("nx")?,
            ny: integer("ny")?,
            earth_radius_m: number("earth_radius_m")?,
            shape_of_earth: integer("shape_of_earth")?,
        };
        if lambert.dx_m <= 0.0 || lambert.dy_m <= 0.0 {
            return Err(mapping_invalid(
                "mapping.grid.parameters dx_m/dy_m must be positive",
            ));
        }
        if !(6.2e6..=6.5e6).contains(&lambert.earth_radius_m) {
            return Err(mapping_invalid(
                "mapping.grid.parameters.earth_radius_m must be a plausible \
                 Earth radius in metres",
            ));
        }
        let published_first_points = published_first_points(grid, &lambert)?;
        let span_units = (lambert.nx as f64 * lambert.dx_m).max(lambert.ny as f64 * lambert.dy_m)
            / PROJECTED_AXIS_UNIT_M;
        if span_units >= 180.0 {
            return Err(mapping_invalid(format!(
                "declared projected grid spans {:.0} km; the projected-axis \
                 representation covers grids below 18,000 km",
                span_units * PROJECTED_AXIS_UNIT_M / 1000.0
            )));
        }
        Ok(GridDeclaration {
            family: GRID_FAMILY_LAMBERT,
            wind_basis: wind_basis.to_owned(),
            parameters: Some(lambert),
            rotated: None,
            same_grid_pairing,
            published_first_points,
        })
    }
}

/// `mapping.grid.published_first_points`: other encodings of the DECLARED
/// first grid point that a publisher writes for this same grid
/// (`mapped_source._validate_published_first_points`).
///
/// HRRRv1 wrfprs and wrfnat files (to 2016-08-23) publish HRRR's corner
/// rounded to millidegrees, 21.138, 237.28, under an otherwise identical
/// grid definition; current files publish 21.138123, 237.280472.  The
/// declaration admits such a record and decodes nothing differently: the
/// frame keeps the declared parameters.  An encoding farther than a tenth
/// of a cell from the declared point is a different grid, not a spelling
/// of this one, and is refused here.
fn published_first_points(grid: &Node, lambert: &LambertParameters) -> Result<Vec<(f64, f64)>> {
    let Some(node) = grid.get("published_first_points") else {
        return Ok(Vec::new());
    };
    if !node.is_array() {
        return Err(mapping_invalid(
            "mapping.grid.published_first_points must be an array of {lat1, lon1}",
        ));
    }
    let tenth_cell_deg = 0.1 * lambert.dx_m.min(lambert.dy_m) / 111_000.0;
    let mut points = Vec::new();
    for item in node.items() {
        let coordinate = |key: &str| -> Result<f64> {
            item.get(key)
                .and_then(Node::as_f64)
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    mapping_invalid(format!(
                        "mapping.grid.published_first_points[].{key} must be a finite number"
                    ))
                })
        };
        let (lat1, lon1) = (coordinate("lat1")?, coordinate("lon1")?);
        if (lat1 - lambert.lat1).abs() > tenth_cell_deg
            || (lon1 - lambert.lon1).abs() > tenth_cell_deg
        {
            return Err(mapping_invalid(format!(
                "mapping.grid.published_first_points entry ({lat1}, {lon1}) is                  more than a tenth of a cell from the declared first point                  ({}, {}): that is a different grid, not an encoding of this one",
                lambert.lat1, lambert.lon1
            )));
        }
        points.push((lat1, lon1));
    }
    Ok(points)
}

/// The normalized `mapping.grid` block.
#[derive(Debug, Clone)]
pub struct GridDeclaration {
    pub family: &'static str,
    pub wind_basis: String,
    pub parameters: Option<LambertParameters>,
    /// The declared rotated lattice, for [`GRID_FAMILY_ROTATED`] only.
    pub rotated: Option<RotatedParameters>,
    pub same_grid_pairing: Option<String>,
    /// Other encodings of the declared first point (lat1, lon1) that a
    /// publisher writes for this same grid; admission only.
    pub published_first_points: Vec<(f64, f64)>,
}

impl GridDeclaration {
    pub fn regular() -> Self {
        Self {
            family: GRID_FAMILY_REGULAR,
            wind_basis: "earth_relative".to_owned(),
            parameters: None,
            rotated: None,
            same_grid_pairing: None,
            published_first_points: Vec::new(),
        }
    }

    pub fn is_lambert(&self) -> bool {
        self.family == GRID_FAMILY_LAMBERT
    }

    pub fn is_rotated(&self) -> bool {
        self.family == GRID_FAMILY_ROTATED
    }

    /// Lambert or rotated: canonical axes are plane distances, not degrees.
    pub fn is_projected(&self) -> bool {
        self.is_lambert() || self.is_rotated()
    }

    /// `(y_axis, x_axis)` of the declared projected grid, in 100-km units.
    pub fn projected_axes(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        if let Some(parameters) = &self.parameters {
            return Some(parameters.projected_axes());
        }
        self.rotated.as_ref().map(RotatedParameters::projected_axes)
    }

    /// `(ny, nx)` of the declared projected grid.
    pub fn projected_shape(&self) -> Option<(usize, usize)> {
        if let Some(parameters) = &self.parameters {
            return Some((parameters.ny as usize, parameters.nx as usize));
        }
        self.rotated.as_ref().map(|r| (r.ny as usize, r.nx as usize))
    }

    pub fn rotates_winds(&self) -> bool {
        self.wind_basis == "grid_relative_with_rotation"
    }
}

/// The declared Lambert source grid, in GRIB2's own vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct LambertParameters {
    pub latin1: f64,
    pub latin2: f64,
    pub lov: f64,
    pub lat1: f64,
    pub lon1: f64,
    pub dx_m: f64,
    pub dy_m: f64,
    pub nx: i64,
    pub ny: i64,
    pub earth_radius_m: f64,
    pub shape_of_earth: i64,
}

/// The declared rotated latitude/longitude lattice (GRIB2 GDT 3.1), in
/// GRIB2's own vocabulary: the rotated first point and increments in
/// degrees, the geographic position of the rotated frame's SOUTHERN pole
/// and the angle of rotation about the new polar axis.
#[derive(Debug, Clone, PartialEq)]
pub struct RotatedParameters {
    pub lat1: f64,
    pub lon1: f64,
    pub dlat_deg: f64,
    pub dlon_deg: f64,
    pub nx: i64,
    pub ny: i64,
    pub south_pole_lat: f64,
    pub south_pole_lon: f64,
    pub rotation_angle: f64,
    pub earth_radius_m: f64,
    pub shape_of_earth: i64,
}

impl RotatedParameters {
    fn from_node(parameters: &Node) -> Result<Self> {
        let number = |key: &str| -> Result<f64> {
            parameters
                .get(key)
                .and_then(Node::as_f64)
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    mapping_invalid(format!(
                        "mapping.grid.parameters.{key} must be a finite number"
                    ))
                })
        };
        let integer = |key: &str| -> Result<i64> {
            parameters.get(key).and_then(Node::as_i64).ok_or_else(|| {
                mapping_invalid(format!("mapping.grid.parameters.{key} must be an integer"))
            })
        };
        let rotated = RotatedParameters {
            lat1: number("lat1")?,
            lon1: number("lon1")?,
            dlat_deg: number("dlat_deg")?,
            dlon_deg: number("dlon_deg")?,
            nx: integer("nx")?,
            ny: integer("ny")?,
            south_pole_lat: number("south_pole_lat")?,
            south_pole_lon: number("south_pole_lon")?,
            rotation_angle: number("rotation_angle")?,
            earth_radius_m: number("earth_radius_m")?,
            shape_of_earth: integer("shape_of_earth")?,
        };
        if rotated.dlat_deg <= 0.0 || rotated.dlon_deg <= 0.0 {
            return Err(mapping_invalid(
                "mapping.grid.parameters dlat_deg/dlon_deg must be positive",
            ));
        }
        if rotated.nx < 2 || rotated.ny < 2 {
            return Err(mapping_invalid("mapping.grid.parameters nx/ny must be at least 2"));
        }
        if !(6.2e6..=6.5e6).contains(&rotated.earth_radius_m) {
            return Err(mapping_invalid(
                "mapping.grid.parameters.earth_radius_m must be a plausible \
                 Earth radius in metres",
            ));
        }
        if !(-90.0..=90.0).contains(&rotated.south_pole_lat)
            || !(-90.0..=90.0).contains(&rotated.lat1)
        {
            return Err(mapping_invalid(
                "mapping.grid.parameters south_pole_lat/lat1 must be within [-90, 90]",
            ));
        }
        let span_units = (rotated.nx as f64 * rotated.dx_m()).max(rotated.ny as f64 * rotated.dy_m())
            / PROJECTED_AXIS_UNIT_M;
        if span_units >= 180.0 {
            return Err(mapping_invalid(format!(
                "declared projected grid spans {:.0} km; the projected-axis \
                 representation covers grids below 18,000 km",
                span_units * PROJECTED_AXIS_UNIT_M / 1000.0
            )));
        }
        Ok(rotated)
    }

    /// Lattice spacing along the rotated equator, in metres.
    pub fn dx_m(&self) -> f64 {
        self.dlon_deg.to_radians() * self.earth_radius_m
    }

    /// Lattice spacing along the rotated meridians, in metres.
    pub fn dy_m(&self) -> f64 {
        self.dlat_deg.to_radians() * self.earth_radius_m
    }

    /// `(y_axis, x_axis)` in 100-km units: index times the rotated-equator
    /// spacing, zero on the first point (`mapped_source._projected_axes`).
    pub fn projected_axes(&self) -> (Vec<f64>, Vec<f64>) {
        let (dy, dx) = (self.dy_m(), self.dx_m());
        let y = (0..self.ny).map(|index| index as f64 * dy / PROJECTED_AXIS_UNIT_M).collect();
        let x = (0..self.nx).map(|index| index as f64 * dx / PROJECTED_AXIS_UNIT_M).collect();
        (y, x)
    }
}

impl LambertParameters {
    /// `(y_axis, x_axis)` of the declared projected grid, in 100-km units
    /// (`mapped_source._projected_axes`).
    pub fn projected_axes(&self) -> (Vec<f64>, Vec<f64>) {
        let y = (0..self.ny)
            .map(|index| index as f64 * self.dy_m / PROJECTED_AXIS_UNIT_M)
            .collect();
        let x = (0..self.nx)
            .map(|index| index as f64 * self.dx_m / PROJECTED_AXIS_UNIT_M)
            .collect();
        (y, x)
    }
}

/// One `mapping.fields[name]` entry.
#[derive(Debug, Clone)]
pub struct FieldSpec<'a> {
    pub name: String,
    pub raw: &'a Node,
}

impl<'a> FieldSpec<'a> {
    pub fn optional(&self) -> Result<bool> {
        match self.raw.get("optional") {
            None => Ok(false),
            Some(value) => value.as_bool().ok_or_else(|| mapping_invalid(format!("fields.{}.optional must be true or false",self.name))),
        }
    }
    pub fn selectors(&self) -> &'a [Node] {
        self.raw.field("selectors").map(Node::items).unwrap_or(&[])
    }

    pub fn derivation(&self) -> Option<&'a str> {
        self.raw.field("derivation").and_then(Node::as_str)
    }

    pub fn provider(&self) -> Option<&'a str> {
        self.raw.field("provider").and_then(Node::as_str)
    }

    /// `fields.<name>.when_absent`: how the field is derived when the
    /// files that should carry it carry none of its records.
    pub fn when_absent(&self) -> Option<&'a Node> {
        self.raw.field("when_absent").filter(|node| node.is_object())
    }

    pub fn time_binding(&self) -> Option<&'a str> {
        self.raw.field("time_binding").and_then(Node::as_str)
    }

    /// `fields.<name>.dependency_only`: the field is decoded only so a
    /// derivation can read it, and the frame does not publish it.
    pub fn dependency_only(&self) -> Result<bool> {
        match self.raw.field("dependency_only") {
            None => Ok(false),
            Some(node) => node.as_bool().ok_or_else(|| {
                mapping_invalid(format!(
                    "fields.{}.dependency_only must be true or false",
                    self.name
                ))
            }),
        }
    }

    /// `fields.<name>.published_levels`: the declared ladder levels the
    /// publisher writes this field on (`mapped_source._published_levels`).
    /// On the other declared levels the field is zero when the file
    /// leaves it out.  `None` when the field declares no subset.  The
    /// authoring rules (which fields, which formats, a strict subset of
    /// `vertical.levels`) are `load_mapping`'s; what the decode needs is
    /// that the value is the one it implements.
    /// `fields.<name>.repeated_record`: `"last"` declares that a record
    /// the publisher writes twice at one level and valid time is read from
    /// its later message (`mapped_source` reads the same key).  Absent, a
    /// repeat is refused.
    pub fn repeated_record_last(&self) -> Result<bool> {
        match self.raw.field("repeated_record") {
            None => Ok(false),
            Some(node) if node.as_str() == Some("last") => Ok(true),
            Some(_) => Err(mapping_invalid(format!(
                "fields.{}.repeated_record must be 'last'",
                self.name
            ))),
        }
    }

    pub fn published_levels(&self) -> Result<Option<Vec<f64>>> {
        let Some(declaration) = self.raw.field("published_levels") else {
            return Ok(None);
        };
        if !matches!(declaration.get("absent").and_then(Node::as_str), Some("zero") | Some("nearest_upper_boundary")) {
            return Err(mapping_invalid(format!(
                "fields.{}.published_levels.absent must be zero or nearest_upper_boundary",
                self.name
            )));
        }
        let levels = declaration
            .get("levels")
            .map(Node::items)
            .filter(|items| !items.is_empty())
            .ok_or_else(|| {
                mapping_invalid(format!(
                    "fields.{}.published_levels.levels must be a non-empty numeric list",
                    self.name
                ))
            })?;
        levels
            .iter()
            .map(|item| {
                item.as_f64().ok_or_else(|| {
                    mapping_invalid(format!(
                        "fields.{}.published_levels.levels must be numbers",
                        self.name
                    ))
                })
            })
            .collect::<Result<Vec<f64>>>()
            .map(Some)
    }

    pub fn selector_stack_axis(&self) -> Option<&'a str> {
        self.raw.field("selector_stack_axis").and_then(Node::as_str)
    }

    pub fn location(&self) -> Result<&'a str> {
        self.raw
            .get("location")
            .and_then(Node::as_str)
            .ok_or_else(|| mapping_invalid(format!("fields.{}.location must be a string", self.name)))
    }

    pub fn staggering(&self) -> &'a str {
        self.raw
            .field("staggering")
            .and_then(Node::as_str)
            .unwrap_or("none")
    }

    pub fn source_axes(&self) -> Result<Vec<String>> {
        self.axes("source_axes")
    }

    pub fn target_axes(&self) -> Result<Vec<String>> {
        self.axes("target_axes")
    }

    fn axes(&self, key: &str) -> Result<Vec<String>> {
        self.raw
            .get(key)
            .and_then(Node::as_string_list)
            .ok_or_else(|| {
                mapping_invalid(format!("fields.{}.{key} must be a list of axis names", self.name))
            })
    }

    pub fn units_target(&self) -> Result<&'a str> {
        self.raw
            .get("units")
            .and_then(|units| units.get("target"))
            .and_then(Node::as_str)
            .ok_or_else(|| {
                mapping_invalid(format!("fields.{}.units.target must be a string", self.name))
            })
    }

    pub fn unit_scale(&self) -> f64 {
        self.raw
            .get("units")
            .and_then(|units| units.field("scale"))
            .and_then(Node::as_f64)
            .unwrap_or(1.0)
    }

    pub fn unit_offset(&self) -> f64 {
        self.raw
            .get("units")
            .and_then(|units| units.field("offset"))
            .and_then(Node::as_f64)
            .unwrap_or(0.0)
    }

    /// The declared missing policy: `reject`, `allow`, `value`, or
    /// `landmask_water` (`mapped_source` missing-policy grammar).
    pub fn missing_kind(&self) -> Result<&'a str> {
        self.raw
            .get("missing")
            .and_then(|missing| missing.get("kind"))
            .and_then(Node::as_str)
            .ok_or_else(|| {
                mapping_invalid(format!("fields.{}.missing.kind must be a string", self.name))
            })
    }

    pub fn missing_value(&self) -> Result<f64> {
        self.raw
            .get("missing")
            .and_then(|missing| missing.get("value"))
            .and_then(Node::as_f64)
            .ok_or_else(|| {
                mapping_invalid(format!("fields.{}.missing.value must be a number", self.name))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping_from(text: &str) -> Mapping {
        let bytes = text.as_bytes().to_vec();
        Mapping {
            sha256: crate::digest::bytes_sha256(&bytes),
            doc: Node::parse(&bytes).unwrap(),
            path: "<test>".to_owned(),
        }
    }

    #[test]
    fn declared_field_order_is_the_document_order_not_alphabetical() {
        let mapping = mapping_from(
            r#"{"fields": {"zulu": {}, "alpha": {}, "mike": {}}}"#,
        );
        assert_eq!(
            mapping.field_names().unwrap(),
            vec!["zulu".to_owned(), "alpha".to_owned(), "mike".to_owned()]
        );
    }

    #[test]
    fn an_absent_grid_block_declares_the_regular_family() {
        let mapping = mapping_from(r#"{"fields": {}}"#);
        let declaration = mapping.grid_declaration().unwrap();
        assert_eq!(declaration.family, GRID_FAMILY_REGULAR);
        assert!(!declaration.rotates_winds());
        assert!(declaration.same_grid_pairing.is_none());
    }

    #[test]
    fn exact_grid_pairing_refuses_unknown_and_nonprojected_declarations() {
        for text in [
            r#"{"grid":{"family":"regular_latitude_longitude","same_grid_pairing":"identity"}}"#,
            r#"{"grid":{"family":"lambert_conformal","same_grid_pairing":"unknown"}}"#,
            r#"{"grid":{"family":"lambert_conformal","same_grid_pairing":true}}"#,
        ] {
            let refusal = mapping_from(text).grid_declaration().unwrap_err();
            assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID);
            assert!(refusal.message.contains("same_grid_pairing"));
        }
    }

    #[test]
    fn a_published_first_point_far_from_the_declared_one_is_a_different_grid() {
        let grid = |points: &str| {
            format!(
                r#"{{"grid": {{"family": "lambert_conformal",
                "wind_basis": "grid_relative_with_rotation",
                "parameters": {{"latin1": 38.5, "latin2": 38.5, "lov": 262.5,
                  "lat1": 21.138123, "lon1": 237.280472, "dx_m": 3000.0,
                  "dy_m": 3000.0, "nx": 1799, "ny": 1059,
                  "earth_radius_m": 6371229.0, "shape_of_earth": 6}},
                "published_first_points": {points}}}}}"#
            )
        };
        let accepted = mapping_from(&grid(r#"[{"lat1": 21.138, "lon1": 237.28}]"#))
            .grid_declaration()
            .unwrap();
        assert_eq!(accepted.published_first_points, vec![(21.138, 237.28)]);
        for points in [
            r#"[{"lat1": 21.2, "lon1": 237.28}]"#,
            r#"[{"lat1": 21.138}]"#,
            r#"{"lat1": 21.138, "lon1": 237.28}"#,
        ] {
            let refusal = mapping_from(&grid(points)).grid_declaration().unwrap_err();
            assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID);
            assert!(refusal.message.contains("published_first_points"), "{}", refusal.message);
        }
    }

    #[test]
    fn a_repeated_record_declaration_is_last_or_refused() {
        let mapping = mapping_from(
            r#"{"fields": {"a": {}, "b": {"repeated_record": "last"}, "c": {"repeated_record": "first"}}}"#,
        );
        assert!(!mapping.field("a").unwrap().repeated_record_last().unwrap());
        assert!(mapping.field("b").unwrap().repeated_record_last().unwrap());
        let refusal = mapping.field("c").unwrap().repeated_record_last().unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID);
        assert!(refusal.message.contains("repeated_record"));
    }

    #[test]
    fn a_lambert_grid_without_a_wind_basis_refuses() {
        let mapping = mapping_from(
            r#"{"grid": {"family": "lambert_conformal", "parameters": {}}}"#,
        );
        let refusal = mapping.grid_declaration().unwrap_err();
        assert_eq!(refusal.class, crate::refusal::class::MAPPING_INVALID);
        assert!(refusal.message.contains("wind_basis"));
    }

    #[test]
    fn projected_axes_are_in_hundred_kilometre_units() {
        let parameters = LambertParameters {
            latin1: 50.0,
            latin2: 50.0,
            lov: 253.0,
            lat1: 1.0,
            lon1: 214.5,
            dx_m: 32463.0,
            dy_m: 32463.0,
            nx: 3,
            ny: 2,
            earth_radius_m: 6371229.0,
            shape_of_earth: 6,
        };
        let (y, x) = parameters.projected_axes();
        assert_eq!(y.len(), 2);
        assert_eq!(x.len(), 3);
        assert_eq!(x[1], 32463.0 / 100_000.0);
    }
}
