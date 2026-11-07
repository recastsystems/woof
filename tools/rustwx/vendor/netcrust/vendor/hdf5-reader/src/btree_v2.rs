//! HDF5 B-tree Version 2.
//!
//! V2 B-trees are used by newer-style groups and datasets for indexed link
//! storage, attribute storage, and chunked dataset indexing. The header
//! (`BTHD`) describes the tree parameters. Internal nodes (`BTIN`) and leaf
//! nodes (`BTLF`) contain the actual records.
//!
//! This module provides the header parse, record types, and a traversal
//! function that collects all records from a tree.

use crate::checksum::jenkins_lookup3;
use crate::error::{Error, Result};
use crate::io::Cursor;
use crate::storage::Storage;

// ---------------------------------------------------------------------------
// Signatures
// ---------------------------------------------------------------------------

const BTHD_SIGNATURE: [u8; 4] = *b"BTHD";
const BTIN_SIGNATURE: [u8; 4] = *b"BTIN";
const BTLF_SIGNATURE: [u8; 4] = *b"BTLF";

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// Parsed B-tree v2 header.
#[derive(Debug, Clone)]
pub struct BTreeV2Header {
    /// B-tree type (determines the record format).
    pub btree_type: u8,
    /// Size in bytes of each B-tree node (both internal and leaf).
    pub node_size: u32,
    /// Size in bytes of each record.
    pub record_size: u16,
    /// Depth of the tree (0 = root is a leaf).
    pub depth: u16,
    /// Percent full at which to split a node.
    pub split_percent: u8,
    /// Percent full at which to merge a node.
    pub merge_percent: u8,
    /// Address of the root node.
    pub root_node_address: u64,
    /// Number of records in the root node.
    pub num_records_in_root: u16,
    /// Total number of records in the entire tree.
    pub total_records: u64,
}

impl BTreeV2Header {
    /// Parse a B-tree v2 header at the current cursor position.
    ///
    /// Format:
    /// - Signature: `BTHD` (4 bytes)
    /// - Version: 0 (1 byte)
    /// - B-tree type (u8)
    /// - Node size (u32 LE)
    /// - Record size (u16 LE)
    /// - Depth (u16 LE)
    /// - Split percent (u8)
    /// - Merge percent (u8)
    /// - Root node address (`offset_size` bytes)
    /// - Number of records in root node (u16 LE)
    /// - Total number of records in tree (`length_size` bytes)
    /// - Checksum (u32 LE)
    pub fn parse(cursor: &mut Cursor, offset_size: u8, length_size: u8) -> Result<Self> {
        let start = cursor.position();

        let sig = cursor.read_bytes(4)?;
        if sig != BTHD_SIGNATURE {
            return Err(Error::InvalidBTreeV2Signature { context: "header" });
        }

        let version = cursor.read_u8()?;
        if version != 0 {
            return Err(Error::UnsupportedBTreeVersion(version));
        }

        let btree_type = cursor.read_u8()?;
        let node_size = cursor.read_u32_le()?;
        let record_size = cursor.read_u16_le()?;
        let depth = cursor.read_u16_le()?;
        let split_percent = cursor.read_u8()?;
        let merge_percent = cursor.read_u8()?;
        let root_node_address = cursor.read_offset(offset_size)?;
        let num_records_in_root = cursor.read_u16_le()?;
        let total_records = cursor.read_length(length_size)?;

        // Checksum covers everything from signature through total_records.
        let checksum_end = cursor.position();
        let stored_checksum = cursor.read_u32_le()?;

        let computed = jenkins_lookup3(&cursor.data()[start as usize..checksum_end as usize]);
        if computed != stored_checksum {
            return Err(Error::ChecksumMismatch {
                expected: stored_checksum,
                actual: computed,
            });
        }

        Ok(BTreeV2Header {
            btree_type,
            node_size,
            record_size,
            depth,
            split_percent,
            merge_percent,
            root_node_address,
            num_records_in_root,
            total_records,
        })
    }

    /// Parse a B-tree v2 header from random-access storage.
    pub fn parse_at_storage(
        storage: &dyn Storage,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self> {
        let header_len = 4
            + 1
            + 1
            + 4
            + 2
            + 2
            + 1
            + 1
            + usize::from(offset_size)
            + 2
            + usize::from(length_size)
            + 4;
        let bytes = storage.read_range(address, header_len)?;
        let mut cursor = Cursor::new(bytes.as_ref());
        Self::parse(&mut cursor, offset_size, length_size)
    }
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// A record from a B-tree v2.
///
/// The record format depends on the B-tree type field in the header.
#[derive(Debug, Clone)]
pub enum BTreeV2Record {
    /// Type 5: Link name for indexed group (hashed).
    LinkNameHash { hash: u32, heap_id: Vec<u8> },
    /// Type 6: Creation order for indexed group.
    CreationOrder { order: u64, heap_id: Vec<u8> },
    /// Type 8: Attribute name for indexed group (hashed).
    AttributeNameHash {
        hash: u32,
        flags: u8,
        creation_order: u32,
        heap_id: Vec<u8>,
    },
    /// Type 9: Attribute creation order.
    AttributeCreationOrder { order: u32, heap_id: Vec<u8> },
    /// Type 10: Non-filtered chunked dataset record (v2 chunk index).
    ChunkedNonFiltered { address: u64, offsets: Vec<u64> },
    /// Type 11: Filtered chunked dataset record (v2 chunk index).
    ChunkedFiltered {
        address: u64,
        chunk_size: u64,
        filter_mask: u32,
        offsets: Vec<u64>,
    },
    /// Unknown/unsupported record type — raw bytes preserved.
    Unknown { record_type: u8, data: Vec<u8> },
}

// ---------------------------------------------------------------------------
// Record parsing
// ---------------------------------------------------------------------------

/// Parse a single record of the given B-tree type.
fn parse_record(
    cursor: &mut Cursor,
    btree_type: u8,
    record_size: u16,
    offset_size: u8,
    length_size: u8,
    _ndims: Option<u32>,
    heap_id_len: usize,
) -> Result<BTreeV2Record> {
    let record_start = cursor.position();

    let record = match btree_type {
        // Type 5: link name hash
        5 => {
            let hash = cursor.read_u32_le()?;
            let heap_id = cursor.read_bytes(heap_id_len)?.to_vec();
            BTreeV2Record::LinkNameHash { hash, heap_id }
        }

        // Type 6: creation order
        6 => {
            let order = cursor.read_u64_le()?;
            let heap_id = cursor.read_bytes(heap_id_len)?.to_vec();
            BTreeV2Record::CreationOrder { order, heap_id }
        }

        // Type 8: attribute name hash
        //
        // Indexed attribute records store the heap ID first, unlike indexed
        // link records. Layout: heap_id, message flags, creation order, hash.
        8 => {
            let heap_id = cursor.read_bytes(heap_id_len)?.to_vec();
            let flags = cursor.read_u8()?;
            let creation_order = cursor.read_u32_le()?;
            let hash = cursor.read_u32_le()?;
            BTreeV2Record::AttributeNameHash {
                hash,
                flags,
                creation_order,
                heap_id,
            }
        }

        // Type 9: attribute creation order
        //
        // Layout: heap_id, message flags, creation order.
        9 => {
            let heap_id = cursor.read_bytes(heap_id_len)?.to_vec();
            let _flags = cursor.read_u8()?;
            let order = cursor.read_u32_le()?;
            BTreeV2Record::AttributeCreationOrder { order, heap_id }
        }

        // Type 10: non-filtered chunk
        10 => {
            let address = cursor.read_offset(offset_size)?;
            // Chunk offsets are encoded as scaled 64-bit values.
            // The number of offset dimensions is calculated from the record size.
            // Each offset is 8 bytes in a type-10 record.
            let offset_bytes_available = record_size as usize - offset_size as usize;
            let num_offsets = offset_bytes_available / 8;
            let mut offsets = Vec::with_capacity(num_offsets);
            for _ in 0..num_offsets {
                offsets.push(cursor.read_u64_le()?);
            }
            BTreeV2Record::ChunkedNonFiltered { address, offsets }
        }

        // Type 11: filtered chunk
        11 => {
            let address = cursor.read_offset(offset_size)?;
            // nbytes (chunk size on disk) is encoded using length_size bytes.
            let nbytes_size = length_size as usize;
            let chunk_size = cursor.read_length(length_size)?;
            let filter_mask = cursor.read_u32_le()?;
            let remaining = record_size as usize - offset_size as usize - nbytes_size - 4; // filter_mask
            let num_offsets = remaining / 8;
            let mut offsets = Vec::with_capacity(num_offsets);
            for _ in 0..num_offsets {
                offsets.push(cursor.read_u64_le()?);
            }
            BTreeV2Record::ChunkedFiltered {
                address,
                chunk_size,
                filter_mask,
                offsets,
            }
        }

        // Unknown type — read raw bytes.
        _ => {
            let data = cursor.read_bytes(record_size as usize)?.to_vec();
            return Ok(BTreeV2Record::Unknown {
                record_type: btree_type,
                data,
            });
        }
    };

    // Ensure we consumed exactly record_size bytes (skip any remaining).
    let consumed = (cursor.position() - record_start) as usize;
    if consumed < record_size as usize {
        cursor.skip(record_size as usize - consumed)?;
    }

    Ok(record)
}

fn record_matches_chunk_bounds(
    record: &BTreeV2Record,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> bool {
    let Some((first_chunk, last_chunk)) = chunk_bounds else {
        return true;
    };

    let offsets = match record {
        BTreeV2Record::ChunkedNonFiltered { offsets, .. }
        | BTreeV2Record::ChunkedFiltered { offsets, .. } => offsets,
        _ => return true,
    };

    offsets.iter().enumerate().all(|(dim, offset)| {
        let chunk_index = *offset / u64::from(chunk_dims[dim]);
        chunk_index >= first_chunk[dim] && chunk_index <= last_chunk[dim]
    })
}

// ---------------------------------------------------------------------------
// Node parsing
// ---------------------------------------------------------------------------

/// Compute the number of bytes needed to represent `max_records` as an
/// unsigned integer (used for child-node record counts in internal nodes).
fn num_records_size(max_records: u64) -> usize {
    ((64 - max_records.leading_zeros()) as usize).div_ceil(8).max(1)
}

/// Compute the maximum number of records that fit in a leaf node.
fn max_leaf_records(node_size: u32, record_size: u16) -> u64 {
    // Leaf node overhead: signature(4) + version(1) + type(1) + checksum(4) = 10
    let overhead = 10u32;
    if node_size <= overhead || record_size == 0 {
        return 0;
    }
    ((node_size - overhead) / record_size as u32) as u64
}

/// Encoded child counts in HDF5 v2 B-trees use capacity-derived widths,
/// independent of the superblock's length width. The immediate count uses
/// the leaf capacity at every level; a deeper pointer also stores the
/// cumulative capacity of its entire child subtree.
fn child_count_sizes(header: &BTreeV2Header, offset_size: u8, depth: u16) -> Result<(usize, usize)> {
    let leaf_max = max_leaf_records(header.node_size, header.record_size);
    let nrec_bytes = num_records_size(leaf_max);
    let mut cumulative_max = leaf_max;
    for level in 1..depth {
        let total_bytes = if level > 1 { num_records_size(cumulative_max) } else { 0 };
        let pointer_bytes = u64::from(offset_size) + nrec_bytes as u64 + total_bytes as u64;
        let available = u64::from(header.node_size).saturating_sub(10);
        let node_max = available.saturating_sub(pointer_bytes)
            / (u64::from(header.record_size) + pointer_bytes);
        cumulative_max = node_max.checked_add(1)
            .and_then(|children| children.checked_mul(cumulative_max))
            .and_then(|children| children.checked_add(node_max))
            .ok_or_else(|| Error::InvalidData("B-tree v2 subtree capacity overflow".into()))?;
    }
    Ok((nrec_bytes, if depth > 1 { num_records_size(cumulative_max) } else { 0 }))
}

/// Parse a leaf node and collect its records.
#[allow(clippy::too_many_arguments)]
fn parse_leaf_node(
    cursor: &mut Cursor,
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
    ndims: Option<u32>,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
    num_records: u16,
    heap_id_len: usize,
    records: &mut Vec<BTreeV2Record>,
) -> Result<()> {
    let start = cursor.position();

    let sig = cursor.read_bytes(4)?;
    if sig != BTLF_SIGNATURE {
        return Err(Error::InvalidBTreeV2Signature {
            context: "leaf node",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::UnsupportedBTreeVersion(version));
    }

    let node_type = cursor.read_u8()?;
    if node_type != header.btree_type {
        return Err(Error::InvalidData(format!(
            "B-tree v2 leaf node type mismatch: header says {}, node says {}",
            header.btree_type, node_type
        )));
    }

    for _ in 0..num_records {
        let record = parse_record(
            cursor,
            header.btree_type,
            header.record_size,
            offset_size,
            length_size,
            ndims,
            heap_id_len,
        )?;
        if record_matches_chunk_bounds(&record, chunk_dims, chunk_bounds) {
            records.push(record);
        }
    }

    // Verify checksum: covers signature through the end of records.
    let checksum_data_end = cursor.position();
    let stored_checksum = cursor.read_u32_le()?;
    let computed = jenkins_lookup3(&cursor.data()[start as usize..checksum_data_end as usize]);
    if computed != stored_checksum {
        return Err(Error::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed,
        });
    }

    Ok(())
}

/// Parse an internal node, collecting child addresses and recursing.
#[allow(clippy::too_many_arguments)]
fn parse_internal_node(
    data: &[u8],
    address: u64,
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
    ndims: Option<u32>,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
    num_records: u16,
    depth: u16,
    heap_id_len: usize,
    records: &mut Vec<BTreeV2Record>,
) -> Result<()> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let start = cursor.position();

    let sig = cursor.read_bytes(4)?;
    if sig != BTIN_SIGNATURE {
        return Err(Error::InvalidBTreeV2Signature {
            context: "internal node",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::UnsupportedBTreeVersion(version));
    }

    let node_type = cursor.read_u8()?;
    if node_type != header.btree_type {
        return Err(Error::InvalidData(format!(
            "B-tree v2 internal node type mismatch: header says {}, node says {}",
            header.btree_type, node_type
        )));
    }

    let (nrec_bytes, total_nrec_bytes) = child_count_sizes(header, offset_size, depth)?;

    // Read records and child pointers interleaved:
    // child[0], record[0], child[1], record[1], ..., record[n-1], child[n]
    // Plus total_records counts for each child.
    //
    // Actually per the HDF5 spec the layout is:
    // record[0], record[1], ..., record[n-1],
    // child_ptr[0], nrec[0], total[0], child_ptr[1], nrec[1], total[1], ..., child_ptr[n], nrec[n], total[n]
    //
    // Records first, then child pointers with their metadata.

    // Read all records.
    // Internal-node records are real B-tree records, not duplicate separator
    // keys. Preserve their slots so traversal can emit child[i], record[i]
    // in HDF5's in-order sequence even when chunk-bound filtering excludes a
    // particular record.
    let mut node_records = Vec::with_capacity(num_records as usize);
    for _ in 0..num_records {
        let record = parse_record(
            &mut cursor,
            header.btree_type,
            header.record_size,
            offset_size,
            length_size,
            ndims,
            heap_id_len,
        )?;
        node_records
            .push(record_matches_chunk_bounds(&record, chunk_dims, chunk_bounds).then_some(record));
    }

    // Read child node pointers (num_records + 1 of them).
    let num_children = num_records as usize + 1;

    // Whether to include a "total records" field for each child.
    // This is present when depth > 1.
    let has_total_records = depth > 1;
    let mut child_addresses = Vec::with_capacity(num_children);
    let mut child_nrecords = Vec::with_capacity(num_children);

    for _ in 0..num_children {
        let child_addr = cursor.read_offset(offset_size)?;
        child_addresses.push(child_addr);
        let nrec = cursor.read_uvar(nrec_bytes)?;
        child_nrecords.push(nrec as u16);
        if has_total_records {
            // Skip total records count
            cursor.read_uvar(total_nrec_bytes)?;
        }
    }

    // Verify checksum.
    let checksum_data_end = cursor.position();
    let stored_checksum = cursor.read_u32_le()?;
    let computed = jenkins_lookup3(&cursor.data()[start as usize..checksum_data_end as usize]);
    if computed != stored_checksum {
        return Err(Error::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed,
        });
    }

    // Recurse in order and emit each internal record between its adjacent
    // children. Omitting these records makes dense groups lose exactly one
    // object per internal record, so dataset lookup and enumeration can both
    // silently miss valid variables.
    let child_depth = depth - 1;
    for (i, &child_addr) in child_addresses.iter().enumerate() {
        if !Cursor::is_undefined_offset(child_addr, offset_size) {
            let child_nrec = child_nrecords[i];
            if child_depth == 0 {
                // Child is a leaf.
                let mut child_cursor = Cursor::new(data);
                child_cursor.set_position(child_addr);
                parse_leaf_node(
                    &mut child_cursor,
                    header,
                    offset_size,
                    length_size,
                    ndims,
                    chunk_dims,
                    chunk_bounds,
                    child_nrec,
                    heap_id_len,
                    records,
                )?;
            } else {
                // Child is another internal node.
                parse_internal_node(
                    data,
                    child_addr,
                    header,
                    offset_size,
                    length_size,
                    ndims,
                    chunk_dims,
                    chunk_bounds,
                    child_nrec,
                    child_depth,
                    heap_id_len,
                    records,
                )?;
            }
        }
        if let Some(record) = node_records.get_mut(i).and_then(|record| record.take()) {
            records.push(record);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public traversal
// ---------------------------------------------------------------------------

/// Collect all records from a B-tree v2 by traversing from the root.
///
/// `header` must be a previously parsed `BTreeV2Header`. `data` is the full
/// file buffer. `ndims` is needed for chunk index record types (10 and 11).
pub fn collect_btree_v2_records(
    data: &[u8],
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
    ndims: Option<u32>,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Result<Vec<BTreeV2Record>> {
    if header.total_records == 0 && header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }
    if Cursor::is_undefined_offset(header.root_node_address, offset_size) {
        return Err(Error::InvalidData(
            "non-empty B-tree v2 has an undefined root address".into(),
        ));
    }
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Err(Error::InvalidData(format!(
            "inconsistent B-tree v2 counts: root has {}, tree declares {} total",
            header.num_records_in_root, header.total_records
        )));
    }

    // Determine heap_id_len from the record_size and btree_type.
    let heap_id_len = compute_heap_id_len(header);

    let mut records = Vec::new();

    if header.depth == 0 {
        // Root is a leaf node.
        let mut cursor = Cursor::new(data);
        cursor.set_position(header.root_node_address);
        parse_leaf_node(
            &mut cursor,
            header,
            offset_size,
            length_size,
            ndims,
            chunk_dims,
            chunk_bounds,
            header.num_records_in_root,
            heap_id_len,
            &mut records,
        )?;
    } else {
        // Root is an internal node.
        parse_internal_node(
            data,
            header.root_node_address,
            header,
            offset_size,
            length_size,
            ndims,
            chunk_dims,
            chunk_bounds,
            header.num_records_in_root,
            header.depth,
            heap_id_len,
            &mut records,
        )?;
    }

    if chunk_bounds.is_none() && u64::try_from(records.len()).ok() != Some(header.total_records) {
        return Err(Error::InvalidData(format!(
            "B-tree v2 traversal collected {} records, header declares {}",
            records.len(),
            header.total_records
        )));
    }

    Ok(records)
}

/// Collect all records from a B-tree v2 using random-access storage.
pub fn collect_btree_v2_records_storage(
    storage: &dyn Storage,
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
    ndims: Option<u32>,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Result<Vec<BTreeV2Record>> {
    if header.total_records == 0 && header.num_records_in_root == 0 {
        return Ok(Vec::new());
    }
    if Cursor::is_undefined_offset(header.root_node_address, offset_size) {
        return Err(Error::InvalidData(
            "non-empty B-tree v2 has an undefined root address".into(),
        ));
    }
    if header.total_records == 0 || header.num_records_in_root == 0 {
        return Err(Error::InvalidData(format!(
            "inconsistent B-tree v2 counts: root has {}, tree declares {} total",
            header.num_records_in_root, header.total_records
        )));
    }

    let heap_id_len = compute_heap_id_len(header);
    let mut records = Vec::new();

    if header.depth == 0 {
        parse_leaf_node_storage(
            storage,
            header.root_node_address,
            header,
            offset_size,
            length_size,
            ndims,
            chunk_dims,
            chunk_bounds,
            header.num_records_in_root,
            heap_id_len,
            &mut records,
        )?;
    } else {
        parse_internal_node_storage(
            storage,
            header.root_node_address,
            header,
            offset_size,
            length_size,
            ndims,
            chunk_dims,
            chunk_bounds,
            header.num_records_in_root,
            header.depth,
            heap_id_len,
            &mut records,
        )?;
    }

    if chunk_bounds.is_none() && u64::try_from(records.len()).ok() != Some(header.total_records) {
        return Err(Error::InvalidData(format!(
            "B-tree v2 traversal collected {} records, header declares {}",
            records.len(),
            header.total_records
        )));
    }

    Ok(records)
}

#[allow(clippy::too_many_arguments)]
fn parse_leaf_node_storage(
    storage: &dyn Storage,
    address: u64,
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
    ndims: Option<u32>,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
    num_records: u16,
    heap_id_len: usize,
    records: &mut Vec<BTreeV2Record>,
) -> Result<()> {
    let node_len = usize::try_from(header.node_size).map_err(|_| {
        Error::InvalidData("B-tree v2 node size exceeds platform usize capacity".into())
    })?;
    let node_bytes = storage.read_range(address, node_len)?;
    let mut cursor = Cursor::new(node_bytes.as_ref());
    parse_leaf_node(
        &mut cursor,
        header,
        offset_size,
        length_size,
        ndims,
        chunk_dims,
        chunk_bounds,
        num_records,
        heap_id_len,
        records,
    )
}

#[allow(clippy::too_many_arguments)]
fn parse_internal_node_storage(
    storage: &dyn Storage,
    address: u64,
    header: &BTreeV2Header,
    offset_size: u8,
    length_size: u8,
    ndims: Option<u32>,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
    num_records: u16,
    depth: u16,
    heap_id_len: usize,
    records: &mut Vec<BTreeV2Record>,
) -> Result<()> {
    let node_len = usize::try_from(header.node_size).map_err(|_| {
        Error::InvalidData("B-tree v2 node size exceeds platform usize capacity".into())
    })?;
    let node_bytes = storage.read_range(address, node_len)?;
    let mut cursor = Cursor::new(node_bytes.as_ref());
    let start = cursor.position();

    let sig = cursor.read_bytes(4)?;
    if sig != BTIN_SIGNATURE {
        return Err(Error::InvalidBTreeV2Signature {
            context: "internal node",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::UnsupportedBTreeVersion(version));
    }

    let node_type = cursor.read_u8()?;
    if node_type != header.btree_type {
        return Err(Error::InvalidData(format!(
            "B-tree v2 internal node type mismatch: header says {}, node says {}",
            header.btree_type, node_type
        )));
    }

    let (nrec_bytes, total_nrec_bytes) = child_count_sizes(header, offset_size, depth)?;

    let mut node_records = Vec::with_capacity(num_records as usize);
    for _ in 0..num_records {
        let record = parse_record(
            &mut cursor,
            header.btree_type,
            header.record_size,
            offset_size,
            length_size,
            ndims,
            heap_id_len,
        )?;
        node_records
            .push(record_matches_chunk_bounds(&record, chunk_dims, chunk_bounds).then_some(record));
    }

    let num_children = usize::from(num_records) + 1;
    let has_total_records = depth > 1;
    let mut child_addresses = Vec::with_capacity(num_children);
    let mut child_nrecords = Vec::with_capacity(num_children);
    for _ in 0..num_children {
        child_addresses.push(cursor.read_offset(offset_size)?);
        child_nrecords.push(cursor.read_uvar(nrec_bytes)? as u16);
        if has_total_records {
            cursor.read_uvar(total_nrec_bytes)?;
        }
    }

    let checksum_data_end = cursor.position();
    let stored_checksum = cursor.read_u32_le()?;
    let computed = jenkins_lookup3(&cursor.data()[start as usize..checksum_data_end as usize]);
    if computed != stored_checksum {
        return Err(Error::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed,
        });
    }

    let child_depth = depth - 1;
    for (i, &child_addr) in child_addresses.iter().enumerate() {
        if !Cursor::is_undefined_offset(child_addr, offset_size) {
            let child_nrec = child_nrecords[i];
            if child_depth == 0 {
                parse_leaf_node_storage(
                    storage,
                    child_addr,
                    header,
                    offset_size,
                    length_size,
                    ndims,
                    chunk_dims,
                    chunk_bounds,
                    child_nrec,
                    heap_id_len,
                    records,
                )?;
            } else {
                parse_internal_node_storage(
                    storage,
                    child_addr,
                    header,
                    offset_size,
                    length_size,
                    ndims,
                    chunk_dims,
                    chunk_bounds,
                    child_nrec,
                    child_depth,
                    heap_id_len,
                    records,
                )?;
            }
        }
        if let Some(record) = node_records.get_mut(i).and_then(|record| record.take()) {
            records.push(record);
        }
    }

    Ok(())
}

/// Compute the heap ID length from the record size and tree type.
///
/// For link/attribute B-trees (types 5, 6, 8, 9), the heap ID occupies
/// the remaining bytes after the fixed fields. For chunk types (10, 11)
/// or unknown types, return 0 (heap_id is not used).
fn compute_heap_id_len(header: &BTreeV2Header) -> usize {
    let rs = header.record_size as usize;
    match header.btree_type {
        5 => rs.saturating_sub(4),         // hash(4)
        6 => rs.saturating_sub(8),         // order(8)
        8 => rs.saturating_sub(1 + 4 + 4), // flags(1) + creation_order(4) + hash(4)
        9 => rs.saturating_sub(1 + 4),     // flags(1) + order(4)
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal BTHD with the given parameters.
    #[allow(clippy::too_many_arguments)]
    fn build_header(
        btree_type: u8,
        node_size: u32,
        record_size: u16,
        depth: u16,
        root_node_address: u64,
        num_records_in_root: u16,
        total_records: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BTHD");
        buf.push(0); // version
        buf.push(btree_type);
        buf.extend_from_slice(&node_size.to_le_bytes());
        buf.extend_from_slice(&record_size.to_le_bytes());
        buf.extend_from_slice(&depth.to_le_bytes());
        buf.push(75); // split percent
        buf.push(40); // merge percent
        match offset_size {
            4 => buf.extend_from_slice(&(root_node_address as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&root_node_address.to_le_bytes()),
            _ => panic!("unsupported"),
        }
        buf.extend_from_slice(&num_records_in_root.to_le_bytes());
        match length_size {
            4 => buf.extend_from_slice(&(total_records as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&total_records.to_le_bytes()),
            _ => panic!("unsupported"),
        }
        // Compute and append checksum.
        let checksum = jenkins_lookup3(&buf);
        buf.extend_from_slice(&checksum.to_le_bytes());
        buf
    }

    #[test]
    fn test_parse_header() {
        let data = build_header(5, 4096, 12, 0, 0x1000, 3, 3, 8, 8);
        let mut cursor = Cursor::new(&data);
        let hdr = BTreeV2Header::parse(&mut cursor, 8, 8).unwrap();

        assert_eq!(hdr.btree_type, 5);
        assert_eq!(hdr.node_size, 4096);
        assert_eq!(hdr.record_size, 12);
        assert_eq!(hdr.depth, 0);
        assert_eq!(hdr.split_percent, 75);
        assert_eq!(hdr.merge_percent, 40);
        assert_eq!(hdr.root_node_address, 0x1000);
        assert_eq!(hdr.num_records_in_root, 3);
        assert_eq!(hdr.total_records, 3);
    }

    #[test]
    fn test_bad_signature() {
        let mut data = build_header(5, 4096, 12, 0, 0x1000, 0, 0, 8, 8);
        data[0] = b'X';
        let mut cursor = Cursor::new(&data);
        assert!(matches!(
            BTreeV2Header::parse(&mut cursor, 8, 8),
            Err(Error::InvalidBTreeV2Signature { .. })
        ));
    }

    #[test]
    fn test_bad_checksum() {
        let mut data = build_header(5, 4096, 12, 0, 0x1000, 0, 0, 8, 8);
        // Corrupt a byte in the middle.
        data[6] = 0xFF;
        let mut cursor = Cursor::new(&data);
        assert!(matches!(
            BTreeV2Header::parse(&mut cursor, 8, 8),
            Err(Error::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn test_collect_empty_tree() {
        let header = BTreeV2Header {
            btree_type: 5,
            node_size: 4096,
            record_size: 12,
            depth: 0,
            split_percent: 75,
            merge_percent: 40,
            root_node_address: u64::MAX,
            num_records_in_root: 0,
            total_records: 0,
        };
        let data = vec![0u8; 100];
        let records = collect_btree_v2_records(&data, &header, 8, 8, None, &[], None).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn test_compute_heap_id_len() {
        // Type 5: record_size - 4
        let h5 = BTreeV2Header {
            btree_type: 5,
            record_size: 12,
            node_size: 0,
            depth: 0,
            split_percent: 0,
            merge_percent: 0,
            root_node_address: 0,
            num_records_in_root: 0,
            total_records: 0,
        };
        assert_eq!(compute_heap_id_len(&h5), 8);

        // Type 8: record_size - 9
        let h8 = BTreeV2Header {
            btree_type: 8,
            record_size: 17,
            ..h5
        };
        assert_eq!(compute_heap_id_len(&h8), 8);
    }

    #[test]
    fn test_max_leaf_records() {
        // node_size=4096, record_size=12, overhead=10
        // => (4096 - 10) / 12 = 340
        assert_eq!(max_leaf_records(4096, 12), 340);
    }

    #[test]
    fn test_num_records_size() {
        assert_eq!(num_records_size(0), 1);
        assert_eq!(num_records_size(255), 1);
        assert_eq!(num_records_size(256), 2);
        assert_eq!(num_records_size(65535), 2);
        assert_eq!(num_records_size(65536), 3);
        assert_eq!(num_records_size(0x1000000), 4);
    }

    #[test]
    fn test_parse_leaf_with_type5_records() {
        // Build a leaf node with 2 type-5 records (link name hash).
        // record_size = 12 (hash=4 + heap_id=8)
        let record_size: u16 = 12;
        let node_size: u32 = 4096;

        let header = BTreeV2Header {
            btree_type: 5,
            node_size,
            record_size,
            depth: 0,
            split_percent: 75,
            merge_percent: 40,
            root_node_address: 0, // will point to our leaf
            num_records_in_root: 2,
            total_records: 2,
        };

        // Build the leaf node.
        let mut leaf = Vec::new();
        leaf.extend_from_slice(b"BTLF"); // signature
        leaf.push(0); // version
        leaf.push(5); // type

        // Record 1: hash=0xAABBCCDD, heap_id=[1,2,3,4,5,6,7,8]
        leaf.extend_from_slice(&0xAABBCCDDu32.to_le_bytes());
        leaf.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        // Record 2: hash=0x11223344, heap_id=[9,10,11,12,13,14,15,16]
        leaf.extend_from_slice(&0x11223344u32.to_le_bytes());
        leaf.extend_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);

        // Checksum covers everything so far.
        let checksum = jenkins_lookup3(&leaf);
        leaf.extend_from_slice(&checksum.to_le_bytes());

        // Pad to node_size (not strictly needed for parsing, but realistic).
        leaf.resize(node_size as usize, 0);

        let mut records = Vec::new();
        let mut cursor = Cursor::new(&leaf);
        parse_leaf_node(
            &mut cursor,
            &header,
            8,
            8,
            None,
            &[],
            None,
            2,
            8, // heap_id_len
            &mut records,
        )
        .unwrap();

        assert_eq!(records.len(), 2);
        match &records[0] {
            BTreeV2Record::LinkNameHash { hash, heap_id } => {
                assert_eq!(*hash, 0xAABBCCDD);
                assert_eq!(heap_id, &[1, 2, 3, 4, 5, 6, 7, 8]);
            }
            _ => panic!("expected LinkNameHash"),
        }
        match &records[1] {
            BTreeV2Record::LinkNameHash { hash, heap_id } => {
                assert_eq!(*hash, 0x11223344);
                assert_eq!(heap_id, &[9, 10, 11, 12, 13, 14, 15, 16]);
            }
            _ => panic!("expected LinkNameHash"),
        }
    }

    fn type5_record(hash: u32, heap_byte: u8) -> Vec<u8> {
        let mut record = Vec::with_capacity(12);
        record.extend_from_slice(&hash.to_le_bytes());
        record.extend_from_slice(&[heap_byte; 8]);
        record
    }

    fn type5_leaf(hash: u32, heap_byte: u8, node_size: usize) -> Vec<u8> {
        let mut leaf = Vec::with_capacity(node_size);
        leaf.extend_from_slice(b"BTLF");
        leaf.push(0);
        leaf.push(5);
        leaf.extend_from_slice(&type5_record(hash, heap_byte));
        let checksum = jenkins_lookup3(&leaf);
        leaf.extend_from_slice(&checksum.to_le_bytes());
        leaf.resize(node_size, 0);
        leaf
    }

    fn type5_depth_one_tree() -> (Vec<u8>, BTreeV2Header) {
        const NODE_SIZE: usize = 128;
        const LEFT_ADDRESS: u64 = NODE_SIZE as u64;
        const RIGHT_ADDRESS: u64 = (NODE_SIZE * 2) as u64;

        let header = BTreeV2Header {
            btree_type: 5,
            node_size: NODE_SIZE as u32,
            record_size: 12,
            depth: 1,
            split_percent: 75,
            merge_percent: 40,
            root_node_address: 0,
            num_records_in_root: 1,
            total_records: 3,
        };

        let mut internal = Vec::with_capacity(NODE_SIZE);
        internal.extend_from_slice(b"BTIN");
        internal.push(0);
        internal.push(5);
        internal.extend_from_slice(&type5_record(20, 2));
        // A depth-one internal node stores address + one-byte child record
        // count for each of its two leaf children.
        internal.extend_from_slice(&LEFT_ADDRESS.to_le_bytes());
        internal.push(1);
        internal.extend_from_slice(&RIGHT_ADDRESS.to_le_bytes());
        internal.push(1);
        let checksum = jenkins_lookup3(&internal);
        internal.extend_from_slice(&checksum.to_le_bytes());
        internal.resize(NODE_SIZE, 0);

        let mut data = internal;
        data.extend_from_slice(&type5_leaf(10, 1, NODE_SIZE));
        data.extend_from_slice(&type5_leaf(30, 3, NODE_SIZE));
        (data, header)
    }

    fn link_hashes(records: &[BTreeV2Record]) -> Vec<u32> {
        records
            .iter()
            .map(|record| match record {
                BTreeV2Record::LinkNameHash { hash, .. } => *hash,
                other => panic!("expected link-name record, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn internal_records_are_collected_in_order_from_buffered_tree() {
        let (data, header) = type5_depth_one_tree();
        let records = collect_btree_v2_records(&data, &header, 8, 8, None, &[], None).unwrap();

        assert_eq!(link_hashes(&records), vec![10, 20, 30]);
        assert_eq!(records.len() as u64, header.total_records);
    }

    #[test]
    fn full_traversal_fails_closed_when_header_count_disagrees() {
        let (data, mut header) = type5_depth_one_tree();
        header.total_records = 4;
        let error = collect_btree_v2_records(&data, &header, 8, 8, None, &[], None)
            .expect_err("record-count disagreement must not return a partial index");

        assert!(
            error.to_string().contains("collected 3 records")
                && error.to_string().contains("declares 4"),
            "{error}"
        );
    }

    #[test]
    fn internal_records_are_collected_in_order_from_random_access_tree() {
        let (data, header) = type5_depth_one_tree();
        let storage = crate::storage::BytesStorage::new(data);
        let records =
            collect_btree_v2_records_storage(&storage, &header, 8, 8, None, &[], None).unwrap();

        assert_eq!(link_hashes(&records), vec![10, 20, 30]);
        assert_eq!(records.len() as u64, header.total_records);
    }
    fn type5_depth_two_tree() -> (Vec<u8>, BTreeV2Header) {
        const N: usize = 512;
        // Independent HDF5 wire encoding: 41 records fit in a leaf; 23
        // fit in a depth-one node, giving a subtree capacity of 1007.
        // Thus a root pointer stores address(8), nrec(1), subtree(2).
        fn node(hash: u32, children: [(u64, u8); 2], root: bool) -> Vec<u8> {
            let mut b = b"BTIN\0\x05".to_vec();
            b.extend(type5_record(hash, hash as u8));
            for (address, count) in children {
                b.extend(address.to_le_bytes());
                b.push(count);
                if root { b.extend(3u16.to_le_bytes()); }
            }
            b.extend(jenkins_lookup3(&b).to_le_bytes());
            b.resize(N, 0);
            b
        }
        let header = BTreeV2Header {
            btree_type: 5, node_size: N as u32, record_size: 12, depth: 2,
            split_percent: 75, merge_percent: 40, root_node_address: 0,
            num_records_in_root: 1, total_records: 7,
        };
        let mut data = node(40, [(N as u64, 1), (2*N as u64, 1)], true);
        data.extend(node(20, [(3*N as u64, 1), (4*N as u64, 1)], false));
        data.extend(node(60, [(5*N as u64, 1), (6*N as u64, 1)], false));
        for hash in [10, 30, 50, 70] { data.extend(type5_leaf(hash, hash as u8, N)); }
        (data, header)
    }

    #[test]
    fn depth_two_counts_use_capacity_width_in_both_readers() {
        let (data, header) = type5_depth_two_tree();
        let records = collect_btree_v2_records(&data, &header, 8, 8, None, &[], None).unwrap();
        assert_eq!(link_hashes(&records), vec![10,20,30,40,50,60,70]);
        let storage = crate::storage::BytesStorage::new(data);
        let records = collect_btree_v2_records_storage(&storage, &header, 8, 8, None, &[], None).unwrap();
        assert_eq!(link_hashes(&records), vec![10,20,30,40,50,60,70]);
    }

    #[test]
    fn depth_two_checksum_corruption_still_fails_closed() {
        let (mut data, header) = type5_depth_two_tree();
        data[12] ^= 1;
        assert!(matches!(collect_btree_v2_records(&data, &header, 8, 8, None, &[], None), Err(Error::ChecksumMismatch { .. })));
        let storage = crate::storage::BytesStorage::new(data);
        assert!(matches!(collect_btree_v2_records_storage(&storage, &header, 8, 8, None, &[], None), Err(Error::ChecksumMismatch { .. })));
    }

}
