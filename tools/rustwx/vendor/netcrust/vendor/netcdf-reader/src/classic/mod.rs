//! Classic (CDF-1/2/5) NetCDF file format support.
//!
//! This module handles the original NetCDF binary format (CDF-1 classic, CDF-2
//! 64-bit offset, and CDF-5 64-bit data). All multi-byte values are big-endian.

pub mod data;
pub mod header;
pub mod types;
pub mod variable;

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::Result;
use crate::types::NcGroup;
use crate::NcFormat;

/// Backing storage for a classic NetCDF file.
pub(crate) enum ClassicData {
    Mmap(Mmap),
    Bytes(Vec<u8>),
}

impl ClassicData {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            ClassicData::Mmap(m) => m,
            ClassicData::Bytes(b) => b,
        }
    }
}

/// An opened classic-format NetCDF file (CDF-1, CDF-2, or CDF-5).
pub struct ClassicFile {
    pub(crate) format: NcFormat,
    pub(crate) root_group: NcGroup,
    pub(crate) data: ClassicData,
    pub(crate) numrecs: u64,
}

impl ClassicFile {
    /// Open a classic NetCDF file from disk using memory-mapping.
    pub fn open(path: &Path, format: NcFormat) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: read-only mapping; caller must not modify the file concurrently.
        let mmap = unsafe { Mmap::map(&file)? };
        let header = header::parse_header(&mmap, format)?;

        let root_group = NcGroup {
            name: "/".to_string(),
            dimensions: header.dimensions,
            variables: header.variables,
            attributes: header.global_attributes,
            groups: Vec::new(), // Classic format has no sub-groups.
        };

        Ok(ClassicFile {
            format,
            root_group,
            data: ClassicData::Mmap(mmap),
            numrecs: header.numrecs,
        })
    }

    /// Open a classic NetCDF file from in-memory bytes.
    pub fn from_bytes(bytes: &[u8], format: NcFormat) -> Result<Self> {
        let header = header::parse_header(bytes, format)?;

        let root_group = NcGroup {
            name: "/".to_string(),
            dimensions: header.dimensions,
            variables: header.variables,
            attributes: header.global_attributes,
            groups: Vec::new(),
        };

        Ok(ClassicFile {
            format,
            root_group,
            data: ClassicData::Bytes(bytes.to_vec()),
            numrecs: header.numrecs,
        })
    }

    /// Open a classic NetCDF file from an existing memory map (avoids double mmap).
    pub fn from_mmap(mmap: Mmap, format: NcFormat) -> Result<Self> {
        let header = header::parse_header(&mmap, format)?;

        let root_group = NcGroup {
            name: "/".to_string(),
            dimensions: header.dimensions,
            variables: header.variables,
            attributes: header.global_attributes,
            groups: Vec::new(),
        };

        Ok(ClassicFile {
            format,
            root_group,
            data: ClassicData::Mmap(mmap),
            numrecs: header.numrecs,
        })
    }

    /// The file format (Classic, Offset64, or Cdf5).
    pub fn format(&self) -> NcFormat {
        self.format
    }

    /// The root group containing all dimensions, variables, and global attributes.
    pub fn root_group(&self) -> &NcGroup {
        &self.root_group
    }

    /// Number of records in the unlimited dimension.
    pub fn numrecs(&self) -> u64 {
        self.numrecs
    }

    /// Where `name`'s stored bytes lie in the file: the same offset and record stride the
    /// readers in [`data`] use, for a caller that copies a variable's big-endian bytes whole.
    pub fn extent(&self, name: &str) -> Result<ClassicExtent> {
        let variable = self
            .root_group
            .variable(name)
            .ok_or_else(|| crate::Error::VariableNotFound(name.to_string()))?;
        let record_stride = if variable.is_record_var {
            data::compute_record_stride(&self.root_group.variables)?
        } else {
            0
        };
        Ok(ClassicExtent {
            offset: variable.data_offset,
            record_stride,
            is_record: variable.is_record_var,
            numrecs: self.numrecs,
        })
    }
}

/// The stored position of one classic variable ([`ClassicFile::extent`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassicExtent {
    /// File offset of the first stored value (record 0 of a record variable).
    pub offset: u64,
    /// Bytes from record `r` to record `r + 1`; 0 for a fixed-size variable.
    pub record_stride: u64,
    /// Whether the variable is laid out along the unlimited dimension.
    pub is_record: bool,
    /// Records the file holds.
    pub numrecs: u64,
}
