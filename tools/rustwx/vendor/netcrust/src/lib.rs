//! WRF-focused, pure-Rust NetCDF reader facade.
//!
//! `netcrust` is intentionally smaller than the full C-backed `netcdf` crate
//! API. It exposes the read surface used by weather workflows here: dimensions,
//! global attributes, variable metadata, promoted `f64` reads, and the WRF
//! convention of reading the first time record for 3-D-or-higher variables.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hdf5_reader::messages::datatype::Datatype as H5Datatype;
use hdf5_reader::{Hdf5File, SliceInfo as H5SliceInfo, SliceInfoElem as H5SliceInfoElem};
use ndarray::ArrayD;
pub use netcdf_reader::NcReadable;
pub use netcdf_reader::{NcFormat, NcMetadataMode, NcOpenOptions, NcSliceInfo, NcSliceInfoElem};

use netcdf_reader::{NcAttrValue, NcDimension, NcFile, NcType, NcVariable};

/// HDF5/NetCDF4 signature bytes.
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0D, 0x0A, 0x1A, 0x0A];

/// Maximum values in an intermediate hyperslab, independent of dataset size.
/// Dense results retain their complete shape and use fallible allocation.
const READ_CHUNK_ELEMENTS: usize = 1024 * 1024;

/// Result type used by `netcrust`.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced by `netcrust`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("NetCDF read error: {0}")]
    Netcdf(#[from] netcdf_reader::Error),

    #[error("HDF5 read error: {0}")]
    Hdf5(String),

    #[error("variable not found: {0}")]
    VariableNotFound(String),

    #[error("dimension size for {name} exceeds usize: {size}")]
    DimensionTooLarge { name: String, size: u64 },

    #[error("unlimited dimension {name} has an unresolved zero length")]
    UnresolvedDimension { name: String },

    #[error("array shape for {name} overflows its element count: {shape:?}")]
    ArrayShapeOverflow { name: String, shape: Vec<u64> },

    #[error("cannot allocate {elements} values for array {name}: {reason}")]
    Allocation {
        name: String,
        elements: usize,
        reason: String,
    },

    #[error("invalid selection for {name}: {reason}")]
    InvalidSelection { name: String, reason: String },

    #[error(
        "cannot select a record from {name} with shape {shape:?}: no explicit leading time axis"
    )]
    UnprovenRecordAxis { name: String, shape: Vec<u64> },
}

/// DEFLATE streams (NetCDF-4 zlib chunks) this process has inflated; see
/// `hdf5_reader::filters::deflate::inflated_streams`.
pub fn inflated_streams() -> u64 {
    hdf5_reader::filters::deflate::inflated_streams()
}

/// Open a NetCDF file.
pub fn open(path: impl AsRef<Path>) -> Result<File> {
    File::open(path)
}

/// Returns true when `bytes` starts with the HDF5/NetCDF4 signature.
pub fn looks_like_hdf5(bytes: &[u8]) -> bool {
    bytes.len() >= HDF5_SIGNATURE.len() && bytes[..HDF5_SIGNATURE.len()] == HDF5_SIGNATURE
}

/// Returns true when `bytes` starts with a supported NetCDF classic or HDF5 signature.
pub fn looks_like_netcdf(bytes: &[u8]) -> bool {
    looks_like_hdf5(bytes) || matches!(bytes.get(..4), Some([b'C', b'D', b'F', 1 | 2 | 5]))
}

/// Opened NetCDF file.
#[derive(Clone)]
pub struct File {
    inner: Arc<NcFile>,
    hdf5: Option<Arc<Hdf5File>>,
    path: Option<PathBuf>,
    dimension_overrides: Arc<HashMap<String, usize>>,
}

/// Dataset metadata read directly from the HDF5 root group.
///
/// NetCDF-4 readers normally expose the same objects through their NetCDF
/// index. Keeping this small fallback surface lets callers enumerate a valid
/// dataset that the NetCDF index omitted without constructing synthetic
/// dimensions or weakening the normal [`Variable`] API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hdf5DatasetMetadata {
    name: String,
    shape: Vec<u64>,
    max_dims: Option<Vec<u64>>,
}

impl Hdf5DatasetMetadata {
    /// Root-relative dataset name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Stored HDF5 dimensions.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Whether HDF5 metadata proves an explicit leading record axis.
    pub fn has_leading_record_axis(&self) -> bool {
        hdf5_metadata_has_leading_record_axis(&self.shape, self.max_dims.as_deref())
    }
}

impl File {
    /// Open a NetCDF file from disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = NcFile::open(path)?;
        let hdf5 = Hdf5File::open(path).ok().map(Arc::new);
        let dimension_overrides = infer_dimension_overrides(&inner, hdf5.as_deref());
        Ok(Self {
            inner: Arc::new(inner),
            hdf5,
            path: Some(path.to_path_buf()),
            dimension_overrides: Arc::new(dimension_overrides),
        })
    }

    /// Open a NetCDF file from disk with custom reader options.
    pub fn open_with_options(path: impl AsRef<Path>, options: NcOpenOptions) -> Result<Self> {
        let path = path.as_ref();
        let inner = NcFile::open_with_options(path, options)?;
        let hdf5 = Hdf5File::open(path).ok().map(Arc::new);
        let dimension_overrides = infer_dimension_overrides(&inner, hdf5.as_deref());
        Ok(Self {
            inner: Arc::new(inner),
            hdf5,
            path: Some(path.to_path_buf()),
            dimension_overrides: Arc::new(dimension_overrides),
        })
    }

    /// Open a NetCDF file from in-memory bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = NcFile::from_bytes(bytes)?;
        let hdf5 = Hdf5File::from_bytes(bytes).ok().map(Arc::new);
        let dimension_overrides = infer_dimension_overrides(&inner, hdf5.as_deref());
        Ok(Self {
            inner: Arc::new(inner),
            hdf5,
            path: None,
            dimension_overrides: Arc::new(dimension_overrides),
        })
    }

    /// Open a NetCDF file from in-memory bytes with custom reader options.
    pub fn from_bytes_with_options(bytes: &[u8], options: NcOpenOptions) -> Result<Self> {
        let inner = NcFile::from_bytes_with_options(bytes, options)?;
        let hdf5 = Hdf5File::from_bytes(bytes).ok().map(Arc::new);
        let dimension_overrides = infer_dimension_overrides(&inner, hdf5.as_deref());
        Ok(Self {
            inner: Arc::new(inner),
            hdf5,
            path: None,
            dimension_overrides: Arc::new(dimension_overrides),
        })
    }

    /// Source path when the file came from disk.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Detected NetCDF format.
    pub fn format(&self) -> NcFormat {
        self.inner.format()
    }

    /// Root-group dimensions.
    pub fn dimensions(&self) -> Result<Vec<Dimension>> {
        self.inner
            .dimensions()?
            .iter()
            .map(|dim| Dimension::try_from(dim, &self.dimension_overrides))
            .collect()
    }

    /// Find a dimension by name or root-relative path.
    pub fn dimension(&self, name: &str) -> Option<Dimension> {
        self.inner
            .dimension(name)
            .ok()
            .and_then(|dim| Dimension::try_from(dim, &self.dimension_overrides).ok())
    }

    /// Root-group variables.
    pub fn variables(&self) -> Result<Vec<Variable>> {
        self.inner
            .variables()?
            .iter()
            .map(|var| {
                Variable::try_from_reader(
                    self.inner.clone(),
                    self.hdf5.clone(),
                    self.dimension_overrides.clone(),
                    var,
                )
            })
            .collect()
    }

    /// Root-group dataset metadata from the raw HDF5 index.
    ///
    /// Returns an empty list for classic NetCDF inputs. For NetCDF-4/HDF5,
    /// every reported shape uses checked 64-bit element counts. No values
    /// are allocated or read while enumerating metadata.
    pub fn hdf5_root_datasets(&self) -> Result<Vec<Hdf5DatasetMetadata>> {
        let Some(hdf5) = self.hdf5.as_ref() else {
            return Ok(Vec::new());
        };
        let root = hdf5
            .root_group()
            .map_err(|err| Error::Hdf5(format!("cannot open HDF5 root group: {err}")))?;
        root.datasets()
            .map_err(|err| Error::Hdf5(format!("cannot enumerate HDF5 root datasets: {err}")))?
            .into_iter()
            .map(|dataset| {
                checked_array_elements(dataset.name(), dataset.shape())?;
                Ok(Hdf5DatasetMetadata {
                    name: dataset.name().to_string(),
                    shape: dataset.shape().to_vec(),
                    max_dims: dataset.max_dims().map(ToOwned::to_owned),
                })
            })
            .collect()
    }

    /// Read one scalar string attribute directly from an HDF5 dataset.
    /// This is the metadata counterpart to the raw-HDF5 by-name data fallback.
    pub fn hdf5_dataset_attribute_string(&self, name: &str, attribute: &str) -> Option<String> {
        self.hdf5
            .as_ref()?
            .dataset(name)
            .ok()?
            .attribute(attribute)
            .ok()?
            .read_string()
            .ok()
    }

    /// Whether raw HDF5 lookup resolves a root-relative dataset name.
    pub fn has_hdf5_dataset(&self, name: &str) -> bool {
        self.hdf5
            .as_ref()
            .is_some_and(|hdf5| hdf5.dataset(name).is_ok())
    }

    /// Find a variable by name or root-relative path.
    pub fn variable(&self, name: &str) -> Option<Variable> {
        self.inner.variable(name).ok().and_then(|var| {
            Variable::try_from_reader(
                self.inner.clone(),
                self.hdf5.clone(),
                self.dimension_overrides.clone(),
                var,
            )
            .ok()
        })
    }

    /// Find a root-group/global attribute by name or root-relative path.
    pub fn attribute(&self, name: &str) -> Option<Attribute> {
        self.inner
            .global_attribute(name)
            .ok()
            .map(Attribute::from_reader)
    }

    /// Root-group/global attributes.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        Ok(self
            .inner
            .global_attributes()?
            .iter()
            .map(Attribute::from_reader)
            .collect())
    }

    /// Read stored numeric words without promotion or CF interpretation.
    ///
    /// Membership masks and diagnostic replay require exact integer and
    /// floating words. Promoting UInt64 to f64 discards member bits above 53.
    pub fn read_array<T: NcReadable>(&self, name: &str) -> Result<ArrayD<T>> {
        let variable = self.inner.variable(name)?;
        let shape = nc_variable_shape(&variable, &self.dimension_overrides)?;
        checked_array_elements(name, &shape)?;
        if nc_variable_uses_overrides(&variable, &self.dimension_overrides) {
            return self.read_array_slice::<T>(name, &full_selection(shape.len()));
        }
        read_in_slabs(name, &shape, &full_selection(shape.len()), |selection| {
            Ok(self.inner.read_variable_slice::<T>(name, selection)?)
        })
    }

    /// Read a bounded stored-word hyperslab without numeric promotion.
    pub fn read_array_slice<T: NcReadable>(
        &self,
        name: &str,
        selection: &NcSliceInfo,
    ) -> Result<ArrayD<T>> {
        let variable = self.inner.variable(name)?;
        let shape = nc_variable_shape(&variable, &self.dimension_overrides)?;
        checked_selection_elements(name, &shape, selection)?;
        if nc_variable_uses_overrides(&variable, &self.dimension_overrides) {
            // WRF may leave the unlimited dimension scale at extent zero
            // while its variable datasets already contain records. Validate
            // against that independent stored extent, then read typed HDF5
            // words directly, preserving NaNs, signed zeros and integer bits.
            let hdf5 = self.hdf5.as_ref().ok_or_else(||
                Error::Hdf5(format!("no stored extent for {name}")))?;
            let dataset = hdf5.dataset(name).map_err(|e| Error::Hdf5(e.to_string()))?;
            checked_selection_elements(name, dataset.shape(), selection)?;
            return dataset.read_slice_parallel::<T>(&hdf5_selection(selection))
                .map_err(|e| Error::Hdf5(e.to_string()));
        }
        read_in_slabs(name, &shape, selection, |selection| {
            Ok(self.inner.read_variable_slice::<T>(name, selection)?)
        })
    }

    /// Read a variable as promoted `f64` values with shape metadata.
    pub fn read_array_f64(&self, name: &str) -> Result<DataArray> {
        let Ok(variable) = self.inner.variable(name) else {
            return self.read_hdf5_dataset_all(name);
        };
        let shape = nc_variable_shape(&variable, &self.dimension_overrides)?;
        self.read_array_f64_slice(name, &full_selection(shape.len()))
    }

    /// Read a char (or string) variable as one string per leading element.
    ///
    /// Classic char arrays interpret the last dimension as the string
    /// length and flatten the leading dimensions; NetCDF-4 files answer
    /// through the HDF5 arm.  This exists because `read_array_f64` only
    /// promotes char through the raw-HDF5 fallback: on a CLASSIC container
    /// there is no HDF5 handle to fall back to, so a char variable such as
    /// WRF's `Times` was readable from an HDF5 tape and refused with
    /// "expected numeric type, got Char" from a CDF-1/2/5 one.  Text is
    /// read as text in both containers instead.
    pub fn read_strings(&self, name: &str) -> Result<Vec<String>> {
        Ok(self.inner.read_variable_as_strings(name)?)
    }

    /// Read a hyperslab selection as promoted `f64` values with shape metadata.
    pub fn read_array_f64_slice(&self, name: &str, selection: &NcSliceInfo) -> Result<DataArray> {
        if let Ok(variable) = self.inner.variable(name) {
            let shape = nc_variable_shape(&variable, &self.dimension_overrides)?;
            checked_selection_elements(name, &shape, selection)?;
            if nc_variable_uses_overrides(&variable, &self.dimension_overrides) {
                let hdf5 = self.hdf5.as_ref().ok_or_else(|| {
                    Error::Hdf5(format!(
                        "cannot validate overridden dimensions for dataset {name}"
                    ))
                })?;
                let dataset = hdf5.dataset(name).map_err(|err| {
                    Error::Hdf5(format!(
                        "cannot validate overridden dimensions for dataset {name}: {err}"
                    ))
                })?;
                checked_selection_elements(name, dataset.shape(), selection)?;
                return self.read_hdf5_dataset_slice(name, selection);
            }
            if matches!(self.inner.format(), NcFormat::Nc4 | NcFormat::Nc4Classic) {
                // One pass over the selection, each overlapping chunk
                // decompressed once and in parallel.  The row-major slabs
                // below cut through chunks: NCAR's 2.5 km restart stores
                // U_2 as 1 x 49 x 600 x 750 zlib chunks (88 MB each,
                // larger than the 64 MiB chunk cache, so never cached),
                // and the ~1M-element slabs re-inflated every chunk once
                // per level per slab, ~60 s for one 3D variable against
                // ~1 s for netCDF-C.  The slab loop bounds only the
                // intermediate typed array; the f64 result is the full
                // selection either way.
                checked_selection_elements(name, &shape, selection)?;
                return match self.inner.read_variable_slice_as_f64_parallel(name, selection) {
                    Ok(array) => Ok(DataArray::from_ndarray(array)),
                    Err(err) => self
                        .read_hdf5_dataset_slice(name, selection)
                        .map_err(|_| err.into()),
                };
            }
            return match read_in_slabs(name, &shape, selection, |selection| {
                Ok(self.inner.read_variable_slice_as_f64(name, selection)?)
            }) {
                Ok(array) => Ok(DataArray::from_ndarray(array)),
                Err(err) => self
                    .read_hdf5_dataset_slice(name, selection)
                    .map_err(|_| err),
            };
        }
        self.read_hdf5_dataset_slice(name, selection)
    }

    /// Read a variable as promoted flat `f64` values.
    pub fn read_f64(&self, name: &str) -> Result<Vec<f64>> {
        Ok(self.read_array_f64(name)?.into_values())
    }

    /// Read the first WRF time record when metadata proves a leading time
    /// dimension; ambiguous rank >= 3 data fails closed.
    ///
    /// This mirrors the behavior used by the current `rustwx-wrf` reader for
    /// WRF variables shaped like `[Time, south_north, west_east]` or
    /// `[Time, bottom_top, south_north, west_east]`.
    pub fn read_array_f64_first_record_or_all(&self, name: &str) -> Result<DataArray> {
        self.read_array_f64_record_or_all(name, 0)
    }

    /// Read one indexed WRF time record only when metadata proves that the
    /// leading axis is time; otherwise read all values for rank < 3, or for a
    /// lossless singleton leading axis at index zero, and fail closed for any
    /// other ambiguous rank >= 3 dataset. Unlike constructing a slice from
    /// listed metadata at the caller, this retains a guarded raw-HDF5 by-name
    /// fallback for datasets omitted from netcdf-reader's index.
    pub fn read_array_f64_record_or_all(&self, name: &str, time_index: u64) -> Result<DataArray> {
        let variable = match self.inner.variable(name) {
            Ok(variable) => variable,
            Err(_) => return self.read_hdf5_dataset_record_or_all(name, time_index),
        };

        if variable.ndim() >= 3 && listed_variable_has_leading_time_axis(&variable) {
            let selection = record_selection(variable.ndim(), time_index);
            self.read_array_f64_slice(name, &selection)
        } else if variable.ndim() >= 3 {
            Err(Error::UnprovenRecordAxis {
                name: name.to_string(),
                shape: nc_variable_shape(&variable, &self.dimension_overrides)?,
            })
        } else {
            self.read_array_f64(name)
        }
    }

    fn read_hdf5_dataset_all(&self, name: &str) -> Result<DataArray> {
        let Some(hdf5) = self.hdf5.as_ref() else {
            return Err(Error::VariableNotFound(name.to_string()));
        };
        let dataset = hdf5
            .dataset(name)
            .map_err(|_| Error::VariableNotFound(name.to_string()))?;
        checked_array_elements(name, dataset.shape())?;
        read_hdf5_dataset_as_f64(&dataset, None)
    }

    fn read_hdf5_dataset_slice(&self, name: &str, selection: &NcSliceInfo) -> Result<DataArray> {
        let Some(hdf5) = self.hdf5.as_ref() else {
            return Err(Error::VariableNotFound(name.to_string()));
        };
        let dataset = hdf5
            .dataset(name)
            .map_err(|_| Error::VariableNotFound(name.to_string()))?;
        checked_selection_elements(name, dataset.shape(), selection)?;
        let selection = hdf5_selection(selection);
        read_hdf5_dataset_as_f64(&dataset, Some(&selection))
    }

    fn read_hdf5_dataset_record_or_all(&self, name: &str, time_index: u64) -> Result<DataArray> {
        let Some(hdf5) = self.hdf5.as_ref() else {
            return Err(Error::VariableNotFound(name.to_string()));
        };
        let dataset = hdf5
            .dataset(name)
            .map_err(|_| Error::VariableNotFound(name.to_string()))?;
        let selection = if dataset.ndim() >= 3
            && hdf5_metadata_has_leading_record_axis(dataset.shape(), dataset.max_dims())
        {
            checked_record_elements(name, dataset.shape(), time_index)?;
            Some(hdf5_record_selection(dataset.ndim(), time_index))
        } else if dataset.ndim() >= 3 && dataset.shape().first() == Some(&1) {
            if time_index != 0 {
                return Err(Error::InvalidSelection {
                    name: name.to_string(),
                    reason: format!(
                        "singleton leading axis has only record 0, cannot read record {time_index}"
                    ),
                });
            }
            // A singleton ambiguous axis can be read in full without losing
            // or relabeling any data. Keep the axis in DataArray metadata;
            // callers that consume the last 2-D plane see the same values.
            checked_array_elements(name, dataset.shape())?;
            None
        } else if dataset.ndim() >= 3 {
            return Err(Error::UnprovenRecordAxis {
                name: name.to_string(),
                shape: dataset.shape().to_vec(),
            });
        } else {
            checked_array_elements(name, dataset.shape())?;
            None
        };
        read_hdf5_dataset_as_f64(&dataset, selection.as_ref())
    }

    /// Read first WRF time record or all values as flat promoted `f64` values.
    pub fn read_f64_first_record_or_all(&self, name: &str) -> Result<Vec<f64>> {
        Ok(self.read_array_f64_first_record_or_all(name)?.into_values())
    }
}

/// Dimension metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dimension {
    name: String,
    len: usize,
    unlimited: bool,
}

impl Dimension {
    fn try_from(dim: &NcDimension, overrides: &HashMap<String, usize>) -> Result<Self> {
        let size = match overrides.get(&dim.name) {
            Some(len) => u64::try_from(*len).map_err(|_| Error::DimensionTooLarge {
                name: dim.name.clone(),
                size: u64::MAX,
            })?,
            None => dim.size,
        };
        if dim.is_unlimited && size == 0 {
            return Err(Error::UnresolvedDimension {
                name: dim.name.clone(),
            });
        }
        Ok(Self {
            name: dim.name.clone(),
            len: usize::try_from(size).map_err(|_| Error::DimensionTooLarge {
                name: dim.name.clone(),
                size,
            })?,
            unlimited: dim.is_unlimited,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_unlimited(&self) -> bool {
        self.unlimited
    }
}

/// Variable metadata and read helpers.
#[derive(Clone)]
pub struct Variable {
    file: Arc<NcFile>,
    hdf5: Option<Arc<Hdf5File>>,
    shape_was_overridden: bool,
    name: String,
    dimensions: Vec<Dimension>,
    dtype: DataType,
    attributes: Vec<Attribute>,
}

impl Variable {
    fn try_from_reader(
        file: Arc<NcFile>,
        hdf5: Option<Arc<Hdf5File>>,
        dimension_overrides: Arc<HashMap<String, usize>>,
        var: &NcVariable,
    ) -> Result<Self> {
        let shape_was_overridden = nc_variable_uses_overrides(var, &dimension_overrides);
        Ok(Self {
            file,
            hdf5,
            shape_was_overridden,
            name: var.name().to_string(),
            dimensions: var
                .dimensions()
                .iter()
                .map(|dim| Dimension::try_from(dim, &dimension_overrides))
                .collect::<Result<_>>()?,
            dtype: DataType::from(var.dtype()),
            attributes: var
                .attributes()
                .iter()
                .map(Attribute::from_reader)
                .collect(),
        })
    }

    fn validate_overridden_hdf_shape(&self, selection: Option<&NcSliceInfo>) -> Result<()> {
        if !self.shape_was_overridden {
            return Ok(());
        }
        let hdf5 = self.hdf5.as_ref().ok_or_else(|| {
            Error::Hdf5(format!(
                "cannot validate overridden dimensions for dataset {}",
                self.name
            ))
        })?;
        let dataset = hdf5.dataset(&self.name).map_err(|err| {
            Error::Hdf5(format!(
                "cannot validate overridden dimensions for dataset {}: {err}",
                self.name
            ))
        })?;
        if let Some(selection) = selection {
            checked_selection_elements(&self.name, dataset.shape(), selection)?;
        } else {
            checked_array_elements(&self.name, dataset.shape())?;
        }
        Ok(())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    pub fn shape(&self) -> Vec<usize> {
        self.dimensions.iter().map(Dimension::len).collect()
    }

    pub fn ndim(&self) -> usize {
        self.dimensions.len()
    }

    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }

    pub fn attributes(&self) -> &[Attribute] {
        &self.attributes
    }

    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|attr| attr.name() == name)
    }

    /// Read this variable as promoted `f64` values with shape metadata.
    pub fn array_f64(&self) -> Result<DataArray> {
        self.array_f64_slice(&full_selection(self.dimensions.len()))
    }

    /// Read a hyperslab selection as promoted `f64` values with shape metadata.
    pub fn array_f64_slice(&self, selection: &NcSliceInfo) -> Result<DataArray> {
        checked_selection_elements(&self.name, &dimension_shape(&self.dimensions), selection)?;
        self.validate_overridden_hdf_shape(Some(selection))?;
        if self.shape_was_overridden {
            let dataset = self
                .hdf5
                .as_ref()
                .ok_or_else(|| Error::VariableNotFound(self.name.clone()))?
                .dataset(&self.name)
                .map_err(|err| Error::Hdf5(err.to_string()))?;
            return read_hdf5_dataset_as_f64(&dataset, Some(&hdf5_selection(selection)));
        }
        let shape = dimension_shape(&self.dimensions);
        let array = read_in_slabs(&self.name, &shape, selection, |selection| {
            Ok(self
                .file
                .read_variable_slice_as_f64(&self.name, selection)?)
        })?;
        Ok(DataArray::from_ndarray(array))
    }

    /// Read this variable as flat promoted `f64` values.
    pub fn values_f64(&self) -> Result<Vec<f64>> {
        Ok(self.array_f64()?.into_values())
    }

    /// Read the first WRF time record only for a named leading time dimension;
    /// rank >= 3 without that evidence fails closed. Lower-rank data reads all.
    pub fn array_f64_first_record_or_all(&self) -> Result<DataArray> {
        let has_leading_time = self
            .dimensions
            .first()
            .is_some_and(|dimension| is_time_dimension_name(&dimension.name));
        if self.ndim() >= 3 && has_leading_time {
            let selection = first_record_selection(self.ndim());
            checked_selection_elements(&self.name, &dimension_shape(&self.dimensions), &selection)?;
            self.validate_overridden_hdf_shape(Some(&selection))?;
            self.array_f64_slice(&selection)
        } else if self.ndim() >= 3 {
            Err(Error::UnprovenRecordAxis {
                name: self.name.clone(),
                shape: dimension_shape(&self.dimensions),
            })
        } else {
            self.array_f64()
        }
    }

    /// Read first WRF time record or all values as flat promoted `f64` values.
    pub fn values_f64_first_record_or_all(&self) -> Result<Vec<f64>> {
        Ok(self.array_f64_first_record_or_all()?.into_values())
    }
}

/// Supported public datatype names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    I8,
    Char,
    I16,
    I32,
    F32,
    F64,
    U8,
    U16,
    U32,
    I64,
    U64,
    String,
    Compound,
    Opaque,
    Array,
    VLen,
}

impl From<&NcType> for DataType {
    fn from(value: &NcType) -> Self {
        match value {
            NcType::Byte => Self::I8,
            NcType::Char => Self::Char,
            NcType::Short => Self::I16,
            NcType::Int => Self::I32,
            NcType::Float => Self::F32,
            NcType::Double => Self::F64,
            NcType::UByte => Self::U8,
            NcType::UShort => Self::U16,
            NcType::UInt => Self::U32,
            NcType::Int64 => Self::I64,
            NcType::UInt64 => Self::U64,
            NcType::String => Self::String,
            NcType::Compound { .. } => Self::Compound,
            NcType::Opaque { .. } => Self::Opaque,
            NcType::Array { .. } => Self::Array,
            NcType::VLen { .. } => Self::VLen,
        }
    }
}

/// Root-group/global attribute metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    name: String,
    value: AttributeValue,
}

impl Attribute {
    fn from_reader(attr: &netcdf_reader::NcAttribute) -> Self {
        Self {
            name: attr.name.clone(),
            value: AttributeValue::from(&attr.value),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn value(&self) -> &AttributeValue {
        &self.value
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.value.as_f64()
    }

    pub fn as_string(&self) -> Option<&str> {
        self.value.as_string()
    }
}

/// Attribute values.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Bytes(Vec<i8>),
    Chars(String),
    Shorts(Vec<i16>),
    Ints(Vec<i32>),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    UBytes(Vec<u8>),
    UShorts(Vec<u16>),
    UInts(Vec<u32>),
    Int64s(Vec<i64>),
    UInt64s(Vec<u64>),
    Strings(Vec<String>),
}

impl AttributeValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::Chars(value) => Some(value),
            Self::Strings(values) if values.len() == 1 => values.first().map(String::as_str),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Bytes(values) => values.first().map(|&value| value as f64),
            Self::Shorts(values) => values.first().map(|&value| value as f64),
            Self::Ints(values) => values.first().map(|&value| value as f64),
            Self::Floats(values) => values.first().map(|&value| value as f64),
            Self::Doubles(values) => values.first().copied(),
            Self::UBytes(values) => values.first().map(|&value| value as f64),
            Self::UShorts(values) => values.first().map(|&value| value as f64),
            Self::UInts(values) => values.first().map(|&value| value as f64),
            Self::Int64s(values) => values.first().map(|&value| value as f64),
            Self::UInt64s(values) => values.first().map(|&value| value as f64),
            Self::Chars(_) | Self::Strings(_) => None,
        }
    }

    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            Self::Bytes(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::Shorts(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::Ints(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::Floats(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::Doubles(values) => Some(values.clone()),
            Self::UBytes(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::UShorts(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::UInts(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::Int64s(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::UInt64s(values) => Some(values.iter().map(|&value| value as f64).collect()),
            Self::Chars(_) | Self::Strings(_) => None,
        }
    }
}

impl From<&NcAttrValue> for AttributeValue {
    fn from(value: &NcAttrValue) -> Self {
        match value {
            NcAttrValue::Bytes(values) => Self::Bytes(values.clone()),
            NcAttrValue::Chars(value) => Self::Chars(value.clone()),
            NcAttrValue::Shorts(values) => Self::Shorts(values.clone()),
            NcAttrValue::Ints(values) => Self::Ints(values.clone()),
            NcAttrValue::Floats(values) => Self::Floats(values.clone()),
            NcAttrValue::Doubles(values) => Self::Doubles(values.clone()),
            NcAttrValue::UBytes(values) => Self::UBytes(values.clone()),
            NcAttrValue::UShorts(values) => Self::UShorts(values.clone()),
            NcAttrValue::UInts(values) => Self::UInts(values.clone()),
            NcAttrValue::Int64s(values) => Self::Int64s(values.clone()),
            NcAttrValue::UInt64s(values) => Self::UInt64s(values.clone()),
            NcAttrValue::Strings(values) => Self::Strings(values.clone()),
        }
    }
}

/// Dense numeric variable data.
#[derive(Debug, Clone, PartialEq)]
pub struct DataArray {
    shape: Vec<usize>,
    values: Vec<f64>,
}

impl DataArray {
    fn from_ndarray(array: ArrayD<f64>) -> Self {
        let shape = array.shape().to_vec();
        let values = if array.is_standard_layout() {
            let (mut values, offset) = array.into_raw_vec_and_offset();
            let offset = offset.unwrap_or(0);
            if offset > 0 {
                values.drain(..offset);
            }
            values.truncate(shape.iter().product());
            values
        } else {
            array.iter().copied().collect()
        };
        Self { shape, values }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn into_values(self) -> Vec<f64> {
        self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

fn record_selection(ndim: usize, time_index: u64) -> NcSliceInfo {
    let mut selections = Vec::with_capacity(ndim);
    selections.push(NcSliceInfoElem::Index(time_index));
    selections.extend((1..ndim).map(|_| NcSliceInfoElem::Slice {
        start: 0,
        end: u64::MAX,
        step: 1,
    }));
    NcSliceInfo { selections }
}

fn is_time_dimension_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "time" | "times" | "xtime" | "valid_time" | "forecast_time" | "t"
    ) || lower.starts_with("time_")
}

fn listed_variable_has_leading_time_axis(variable: &NcVariable) -> bool {
    variable
        .dimensions()
        .first()
        .is_some_and(|dimension| is_time_dimension_name(&dimension.name))
}

/// A dataset missing from the NetCDF index has no dimension labels available
/// here. HDF5 still proves a record axis when exactly its leading dataspace
/// dimension is explicitly unlimited; rank alone is never evidence of time.
fn hdf5_metadata_has_leading_record_axis(shape: &[u64], max_dims: Option<&[u64]>) -> bool {
    shape.len() >= 3
        && max_dims.is_some_and(|max_dims| {
            max_dims.len() == shape.len()
                && max_dims[0] == u64::MAX
                && max_dims[1..].iter().all(|maximum| *maximum != u64::MAX)
        })
}

fn first_record_selection(ndim: usize) -> NcSliceInfo {
    record_selection(ndim, 0)
}

fn hdf5_record_selection(ndim: usize, time_index: u64) -> H5SliceInfo {
    let mut selections = Vec::with_capacity(ndim);
    selections.push(H5SliceInfoElem::Index(time_index));
    selections.extend((1..ndim).map(|_| H5SliceInfoElem::Slice {
        start: 0,
        end: u64::MAX,
        step: 1,
    }));
    H5SliceInfo { selections }
}

fn hdf5_selection(selection: &NcSliceInfo) -> H5SliceInfo {
    H5SliceInfo {
        selections: selection
            .selections
            .iter()
            .map(|element| match element {
                NcSliceInfoElem::Index(index) => H5SliceInfoElem::Index(*index),
                NcSliceInfoElem::Slice { start, end, step } => H5SliceInfoElem::Slice {
                    start: *start,
                    end: *end,
                    step: *step,
                },
            })
            .collect(),
    }
}

fn read_hdf5_dataset_as_f64(
    dataset: &hdf5_reader::Dataset,
    selection: Option<&H5SliceInfo>,
) -> Result<DataArray> {
    let selection = selection
        .map(|selection| NcSliceInfo {
            selections: selection
                .selections
                .iter()
                .map(|element| match element {
                    H5SliceInfoElem::Index(index) => NcSliceInfoElem::Index(*index),
                    H5SliceInfoElem::Slice { start, end, step } => NcSliceInfoElem::Slice {
                        start: *start,
                        end: *end,
                        step: *step,
                    },
                })
                .collect(),
        })
        .unwrap_or_else(|| full_selection(dataset.ndim()));
    // A chunked dataset is read in one pass so each chunk is decompressed
    // once (see `File::read_array_f64_slice`); other layouts keep the slabs.
    let chunked = dataset.chunks().is_some();
    let read_whole = |name: &str, shape: &[u64], selection: &NcSliceInfo, read: &mut dyn FnMut(&NcSliceInfo) -> Result<ArrayD<f64>>| {
        if chunked {
            checked_selection_elements(name, shape, selection)?;
            read(selection)
        } else {
            read_in_slabs(name, shape, selection, |slab| read(slab))
        }
    };
    let array = read_whole(dataset.name(), dataset.shape(), &selection, &mut |selection| {
        let selection = hdf5_selection(selection);
        // Integer decoders validate width but not signedness. Select the
        // storage type so every numeric width and sign promotes correctly.
        macro_rules! read_numeric {
            ($ty:ty) => {
                read_hdf5_numeric::<$ty>(dataset, Some(&selection), |value| value as f64)
            };
        }
        match dataset.dtype() {
            H5Datatype::FloatingPoint { size: 8, .. } => read_numeric!(f64),
            H5Datatype::FloatingPoint { size: 4, .. } => read_numeric!(f32),
            H5Datatype::FixedPoint { size: 1, signed: true, .. } => read_numeric!(i8),
            H5Datatype::FixedPoint { size: 1, signed: false, .. } => read_numeric!(u8),
            H5Datatype::FixedPoint { size: 2, signed: true, .. } => read_numeric!(i16),
            H5Datatype::FixedPoint { size: 2, signed: false, .. } => read_numeric!(u16),
            H5Datatype::FixedPoint { size: 4, signed: true, .. } => read_numeric!(i32),
            H5Datatype::FixedPoint { size: 4, signed: false, .. } => read_numeric!(u32),
            H5Datatype::FixedPoint { size: 8, signed: true, .. } => read_numeric!(i64),
            H5Datatype::FixedPoint { size: 8, signed: false, .. } => read_numeric!(u64),
            datatype => Err(hdf5_reader::error::Error::TypeMismatch {
                expected: "numeric type".into(),
                actual: format!("{datatype:?}"),
            }),
        }
        .map_err(|err| Error::Hdf5(err.to_string()))
    })?;
    Ok(DataArray::from_ndarray(array))
}

fn read_hdf5_array<T: hdf5_reader::H5Type>(
    dataset: &hdf5_reader::Dataset,
    selection: Option<&H5SliceInfo>,
) -> std::result::Result<ArrayD<T>, hdf5_reader::error::Error> {
    match selection {
        Some(selection) => dataset.read_slice_parallel::<T>(selection),
        None => dataset.read_array_parallel::<T>(),
    }
}

fn read_hdf5_numeric<T>(
    dataset: &hdf5_reader::Dataset,
    selection: Option<&H5SliceInfo>,
    promote: impl Fn(T) -> f64,
) -> std::result::Result<ArrayD<f64>, hdf5_reader::error::Error>
where
    T: hdf5_reader::H5Type + Copy,
{
    let array = read_hdf5_array::<T>(dataset, selection)?;
    Ok(array.mapv(promote))
}

fn dimension_shape(dimensions: &[Dimension]) -> Vec<u64> {
    dimensions
        .iter()
        .map(|dimension| dimension.len as u64)
        .collect()
}

fn nc_variable_shape(
    variable: &NcVariable,
    overrides: &HashMap<String, usize>,
) -> Result<Vec<u64>> {
    variable
        .dimensions()
        .iter()
        .map(|dimension| {
            Dimension::try_from(dimension, overrides).map(|dimension| dimension.len as u64)
        })
        .collect()
}

fn nc_variable_uses_overrides(variable: &NcVariable, overrides: &HashMap<String, usize>) -> bool {
    variable
        .dimensions()
        .iter()
        .any(|dimension| overrides.contains_key(&dimension.name))
}

fn checked_array_elements(name: &str, shape: &[u64]) -> Result<usize> {
    let elements = shape.iter().try_fold(1u64, |product, &size| {
        product
            .checked_mul(size)
            .ok_or_else(|| Error::ArrayShapeOverflow {
                name: name.to_string(),
                shape: shape.to_vec(),
            })
    })?;
    usize::try_from(elements).map_err(|_| Error::ArrayShapeOverflow {
        name: name.to_string(),
        shape: shape.to_vec(),
    })
}

fn full_selection(ndim: usize) -> NcSliceInfo {
    NcSliceInfo {
        selections: (0..ndim)
            .map(|_| NcSliceInfoElem::Slice {
                start: 0,
                end: u64::MAX,
                step: 1,
            })
            .collect(),
    }
}

/// Row-major slabs keep all trailing selected axes intact. Earlier selected
/// axes become indexes, so even a one-dimensional array can be streamed.
struct SlabSelections {
    selection: NcSliceInfo,
    counts: Vec<u64>,
    ordinals: Vec<u64>,
    split_axis: Option<usize>,
    split_count: u64,
    done: bool,
}

impl SlabSelections {
    fn new(name: &str, shape: &[u64], selection: &NcSliceInfo) -> Result<Self> {
        let elements = checked_selection_elements(name, shape, selection)?;
        let mut selection = selection.clone();
        let mut counts = Vec::with_capacity(shape.len());
        for (element, &size) in selection.selections.iter_mut().zip(shape) {
            counts.push(match element {
                NcSliceInfoElem::Index(_) => 1,
                NcSliceInfoElem::Slice { start, end, step } => {
                    *end = (*end).min(size);
                    if *start >= *end {
                        0
                    } else {
                        (*end - *start).div_ceil(*step)
                    }
                }
            });
        }
        let mut split_axis = None;
        let mut split_count = 0;
        if elements > READ_CHUNK_ELEMENTS {
            let mut trailing = 1usize;
            for axis in (0..counts.len()).rev() {
                if matches!(selection.selections[axis], NcSliceInfoElem::Index(_)) {
                    continue;
                }
                let count =
                    usize::try_from(counts[axis]).map_err(|_| Error::ArrayShapeOverflow {
                        name: name.to_string(),
                        shape: shape.to_vec(),
                    })?;
                if count > READ_CHUNK_ELEMENTS / trailing {
                    split_axis = Some(axis);
                    split_count = (READ_CHUNK_ELEMENTS / trailing) as u64;
                    break;
                }
                trailing *= count;
            }
        }
        Ok(Self {
            ordinals: vec![0; counts.len()],
            selection,
            counts,
            split_axis,
            split_count,
            done: elements == 0,
        })
    }

    fn next(&mut self) -> Option<NcSliceInfo> {
        if self.done {
            return None;
        }
        let Some(split_axis) = self.split_axis else {
            self.done = true;
            return Some(self.selection.clone());
        };
        let mut slab = self.selection.clone();
        for axis in 0..split_axis {
            if let NcSliceInfoElem::Slice { start, step, .. } = self.selection.selections[axis] {
                // Validation proves each selected coordinate is inside its
                // u64 extent, so these products and sums cannot overflow.
                slab.selections[axis] = NcSliceInfoElem::Index(start + self.ordinals[axis] * step);
            }
        }
        if let NcSliceInfoElem::Slice { start, end, step } = self.selection.selections[split_axis] {
            let ordinal = self.ordinals[split_axis];
            let count = self.split_count.min(self.counts[split_axis] - ordinal);
            let slab_start = start + ordinal * step;
            let slab_end = if ordinal + count == self.counts[split_axis] {
                end
            } else {
                slab_start + count * step
            };
            slab.selections[split_axis] = NcSliceInfoElem::Slice {
                start: slab_start,
                end: slab_end,
                step,
            };
            self.ordinals[split_axis] += count;
        }
        if self.ordinals[split_axis] == self.counts[split_axis] {
            self.ordinals[split_axis] = 0;
            self.done = true;
            for axis in (0..split_axis).rev() {
                if matches!(
                    self.selection.selections[axis],
                    NcSliceInfoElem::Slice { .. }
                ) {
                    self.ordinals[axis] += 1;
                    if self.ordinals[axis] < self.counts[axis] {
                        self.done = false;
                        break;
                    }
                    self.ordinals[axis] = 0;
                }
            }
        }
        Some(slab)
    }
}

fn read_in_slabs<T, F>(
    name: &str,
    shape: &[u64],
    selection: &NcSliceInfo,
    mut read: F,
) -> Result<ArrayD<T>>
where
    F: FnMut(&NcSliceInfo) -> Result<ArrayD<T>>,
{
    let elements = checked_selection_elements(name, shape, selection)?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| Error::Allocation {
            name: name.to_string(),
            elements,
            reason: "result byte size exceeds addressable allocation".to_string(),
        })?;
    let mut slabs = SlabSelections::new(name, shape, selection)?;
    let result_shape = selection
        .selections
        .iter()
        .zip(shape)
        .filter_map(|(element, &size)| match element {
            NcSliceInfoElem::Index(_) => None,
            NcSliceInfoElem::Slice { start, end, step } => {
                let end = (*end).min(size);
                Some(if *start >= end {
                    0
                } else {
                    (end - *start).div_ceil(*step) as usize
                })
            }
        })
        .collect::<Vec<_>>();
    if elements != 0 && elements <= READ_CHUNK_ELEMENTS {
        return read(selection);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|err| Error::Allocation {
            name: name.to_string(),
            elements,
            reason: format!("{bytes} bytes: {err}"),
        })?;
    while let Some(slab) = slabs.next() {
        let array = read(&slab)?;
        let expected = checked_selection_elements(name, shape, &slab)?;
        if array.len() != expected {
            return Err(Error::InvalidSelection {
                name: name.to_string(),
                reason: format!("slab returned {} values, expected {expected}", array.len()),
            });
        }
        values.extend(array);
    }
    ArrayD::from_shape_vec(ndarray::IxDyn(&result_shape), values).map_err(|err| {
        Error::InvalidSelection {
            name: name.to_string(),
            reason: err.to_string(),
        }
    })
}

fn checked_selection_elements(name: &str, shape: &[u64], selection: &NcSliceInfo) -> Result<usize> {
    if selection.selections.len() != shape.len() {
        return Err(Error::InvalidSelection {
            name: name.to_string(),
            reason: format!(
                "selection has {} dimensions but variable has {}",
                selection.selections.len(),
                shape.len()
            ),
        });
    }
    let mut selected_shape = Vec::with_capacity(shape.len());
    for (axis, (element, &size)) in selection.selections.iter().zip(shape).enumerate() {
        match element {
            NcSliceInfoElem::Index(index) => {
                if *index >= size {
                    return Err(Error::InvalidSelection {
                        name: name.to_string(),
                        reason: format!(
                            "index {index} is out of bounds for axis {axis} with length {size}"
                        ),
                    });
                }
            }
            NcSliceInfoElem::Slice { start, end, step } => {
                if *step == 0 {
                    return Err(Error::InvalidSelection {
                        name: name.to_string(),
                        reason: format!("axis {axis} has a zero slice step"),
                    });
                }
                if *start > size {
                    return Err(Error::InvalidSelection {
                        name: name.to_string(),
                        reason: format!(
                            "slice start {start} is out of bounds for axis {axis} with length {size}"
                        ),
                    });
                }
                let actual_end = if *end == u64::MAX {
                    size
                } else {
                    (*end).min(size)
                };
                let count = if *start >= actual_end {
                    0
                } else {
                    (actual_end - *start).div_ceil(*step)
                };
                selected_shape.push(count);
            }
        }
    }
    checked_array_elements(name, &selected_shape)
}

fn checked_record_elements(name: &str, shape: &[u64], time_index: u64) -> Result<usize> {
    checked_selection_elements(name, shape, &record_selection(shape.len(), time_index))
}

fn consistent_inferred_extent(extents: impl IntoIterator<Item = u64>) -> Option<usize> {
    let mut inferred = None::<u64>;
    for extent in extents {
        if extent == 0 {
            return None;
        }
        match inferred {
            Some(existing) if existing != extent => return None,
            Some(_) => {}
            None => inferred = Some(extent),
        }
    }
    inferred.and_then(|extent| usize::try_from(extent).ok())
}

fn infer_dimension_overrides(file: &NcFile, hdf5: Option<&Hdf5File>) -> HashMap<String, usize> {
    let Ok(dimensions) = file.dimensions() else {
        return HashMap::new();
    };
    let zero_dims = dimensions
        .iter()
        .filter(|dim| dim.size == 0)
        .map(|dim| dim.name.clone())
        .collect::<Vec<_>>();
    if zero_dims.is_empty() {
        return HashMap::new();
    }
    let Some(hdf5) = hdf5 else {
        // Classic NetCDF has no independent dataset extent metadata. Do not
        // decode an arbitrary payload merely to guess a broken unlimited
        // dimension; leaving it at zero makes callers fail closed.
        return HashMap::new();
    };

    let Ok(variables) = file.variables() else {
        return HashMap::new();
    };

    let mut overrides = HashMap::new();
    for dim_name in zero_dims {
        let mut candidates = variables
            .iter()
            .filter_map(|var| {
                let axis = var
                    .dimensions()
                    .iter()
                    .position(|dim| dim.name == dim_name)?;
                let other_shape = var
                    .dimensions()
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| *index != axis)
                    .map(|(_, dimension)| dimension.size.max(1))
                    .collect::<Vec<_>>();
                let other_elements = checked_array_elements(var.name(), &other_shape).ok()?;
                let priority = if var.name() == dim_name {
                    0u8
                } else if matches!(
                    var.name().to_ascii_lowercase().as_str(),
                    "time" | "times" | "xtime" | "valid_time" | "forecast_time"
                ) {
                    1
                } else {
                    2
                };
                Some((priority, other_elements, axis, var))
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(priority, other_elements, _, _)| (*priority, *other_elements));

        let mut chosen_key = None::<(u8, usize)>;
        let mut extents = Vec::<u64>::new();
        let mut invalid_witness = false;
        for (priority, other_elements, axis, var) in candidates {
            let key = (priority, other_elements);
            if chosen_key.is_some_and(|chosen| chosen != key) {
                break;
            }
            let Ok(dataset) = hdf5.dataset(var.name()) else {
                continue;
            };
            chosen_key.get_or_insert(key);
            let shape = dataset.shape();
            if shape.len() != var.dimensions().len()
                || checked_array_elements(var.name(), shape).is_err()
            {
                invalid_witness = true;
                break;
            }
            let Some(&extent) = shape.get(axis) else {
                invalid_witness = true;
                break;
            };
            extents.push(extent);
        }
        if invalid_witness {
            continue;
        }

        // Prefer an authoritative coordinate/time dataset, otherwise the
        // smallest metadata-only witness. Equally ranked witnesses must agree;
        // conflicting or unusable metadata leaves the dimension unresolved.
        // No variable data is read here.
        if let Some(len) = consistent_inferred_extent(extents) {
            overrides.insert(dim_name, len);
        }
    }

    overrides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_netcdf_signatures() {
        assert!(looks_like_hdf5(&HDF5_SIGNATURE));
        assert!(looks_like_netcdf(&HDF5_SIGNATURE));
        assert!(looks_like_netcdf(b"CDF\x01extra"));
        assert!(looks_like_netcdf(b"CDF\x02extra"));
        assert!(looks_like_netcdf(b"CDF\x05extra"));
        assert!(!looks_like_netcdf(b"CDF\x09extra"));
        assert!(!looks_like_netcdf(b"NOPE"));
    }

    #[test]
    fn first_record_selection_drops_leading_axis() {
        let selection = first_record_selection(4);
        assert_eq!(selection.selections.len(), 4);
        assert!(matches!(selection.selections[0], NcSliceInfoElem::Index(0)));
        for elem in &selection.selections[1..] {
            assert!(matches!(
                elem,
                NcSliceInfoElem::Slice {
                    start: 0,
                    end: u64::MAX,
                    step: 1
                }
            ));
        }
    }

    #[test]
    fn record_axis_requires_named_time_or_explicit_hdf_unlimited_metadata() {
        for name in ["Time", "time", "Times", "XTIME", "forecast_time"] {
            assert!(is_time_dimension_name(name), "{name} should prove time");
        }
        for name in ["bottom_top", "south_north", "phony_dim_0", "level"] {
            assert!(!is_time_dimension_name(name), "{name} is not a time axis");
        }

        let shape = [2, 40, 50];
        assert!(hdf5_metadata_has_leading_record_axis(
            &shape,
            Some(&[u64::MAX, 40, 50])
        ));
        assert!(!hdf5_metadata_has_leading_record_axis(&shape, None));
        assert!(!hdf5_metadata_has_leading_record_axis(
            &shape,
            Some(&[2, 40, 50])
        ));
        assert!(!hdf5_metadata_has_leading_record_axis(
            &shape,
            Some(&[2, u64::MAX, 50])
        ));
        assert!(!hdf5_metadata_has_leading_record_axis(
            &shape,
            Some(&[u64::MAX, 40])
        ));
    }

    #[test]
    fn netcdf_selection_maps_losslessly_to_hdf5_selection() {
        let source = NcSliceInfo {
            selections: vec![
                NcSliceInfoElem::Index(2),
                NcSliceInfoElem::Slice {
                    start: 3,
                    end: 19,
                    step: 4,
                },
            ],
        };
        let mapped = hdf5_selection(&source);
        assert!(matches!(mapped.selections[0], H5SliceInfoElem::Index(2)));
        assert!(matches!(
            mapped.selections[1],
            H5SliceInfoElem::Slice {
                start: 3,
                end: 19,
                step: 4
            }
        ));
    }

    #[test]
    fn metadata_products_use_checked_u64_without_a_grid_ceiling() {
        assert_eq!(checked_array_elements("field", &[2, 3, 4]).unwrap(), 24);
        assert_eq!(
            checked_array_elements("field", &[2695, 1585, 50]).unwrap(),
            213_578_750
        );
        assert_eq!(
            checked_array_elements("field", &[134_217_737]).unwrap(),
            134_217_737
        );
        assert!(matches!(
            checked_array_elements("field", &[u64::MAX, 2]),
            Err(Error::ArrayShapeOverflow { .. })
        ));
    }

    #[test]
    fn selections_count_only_the_requested_hyperslab() {
        let record = record_selection(3, 1);
        assert_eq!(
            checked_selection_elements("field", &[10, 20, 30], &record).unwrap(),
            600
        );

        let strided = NcSliceInfo {
            selections: vec![NcSliceInfoElem::Slice {
                start: 1,
                end: 10,
                step: 3,
            }],
        };
        assert_eq!(
            checked_selection_elements("field", &[10], &strided).unwrap(),
            3
        );

        let invalid = NcSliceInfo {
            selections: vec![NcSliceInfoElem::Index(10)],
        };
        assert!(matches!(
            checked_selection_elements("field", &[10], &invalid),
            Err(Error::InvalidSelection { .. })
        ));
    }

    #[test]
    fn unlimited_extent_inference_requires_consistent_sane_metadata() {
        assert_eq!(consistent_inferred_extent([4, 4, 4]), Some(4));
        assert_eq!(consistent_inferred_extent([4, 5]), None);
        assert_eq!(consistent_inferred_extent([0, 4]), None);
        assert_eq!(consistent_inferred_extent([134_217_737]), Some(134_217_737));
    }

    #[test]
    fn slab_plan_is_bounded_and_preserves_strided_row_order() {
        let shape = [3, 9, 1_048_577];
        let selection = NcSliceInfo {
            selections: vec![
                NcSliceInfoElem::Slice {
                    start: 0,
                    end: 3,
                    step: 2,
                },
                NcSliceInfoElem::Slice {
                    start: 1,
                    end: 9,
                    step: 3,
                },
                NcSliceInfoElem::Slice {
                    start: 1,
                    end: 1_048_577,
                    step: 1,
                },
            ],
        };
        let mut plan = SlabSelections::new("field", &shape, &selection).unwrap();
        let mut coordinates = Vec::new();
        while let Some(slab) = plan.next() {
            assert!(
                checked_selection_elements("field", &shape, &slab).unwrap() <= READ_CHUNK_ELEMENTS
            );
            if let (NcSliceInfoElem::Index(a), NcSliceInfoElem::Slice { start: b, .. }) =
                (&slab.selections[0], &slab.selections[1])
            {
                coordinates.push((*a, *b));
            } else {
                panic!("unexpected slab plan");
            }
        }
        assert_eq!(
            coordinates,
            [(0, 1), (0, 4), (0, 7), (2, 1), (2, 4), (2, 7)]
        );
    }

    #[test]
    fn oversized_dense_byte_request_fails_before_reading() {
        let mut read_called = false;
        let selection = full_selection(1);
        let result = read_in_slabs::<f64, _>("field", &[u64::MAX / 8], &selection, |_| {
            read_called = true;
            unreachable!()
        });
        assert!(matches!(result, Err(Error::Allocation { .. })));
        assert!(!read_called);
    }

    #[test]
    fn attribute_value_promotes_numeric_scalars() {
        assert_eq!(AttributeValue::Ints(vec![42]).as_f64(), Some(42.0));
        assert_eq!(AttributeValue::Floats(vec![1.5]).as_f64(), Some(1.5));
        assert_eq!(
            AttributeValue::Chars("Lambert".to_string()).as_string(),
            Some("Lambert")
        );
    }
}
