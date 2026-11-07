use std::{collections::HashMap, sync::Arc};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use radar_scattering::{AdditiveScattering, OutputError, Sha256Digest};
use thiserror::Error;

use crate::{
    CudaDeviceInfo, MINIMUM_COMPUTE_CAPABILITY,
    prepared::{CUDA_MAX_ACTIVE_AXES, CudaLutNodePlan},
};

pub(crate) const CUDA_LUT_COMPONENT_COUNT: usize = AdditiveScattering::COMPONENT_COUNT;
const KERNEL_NAME: &str = "bowecho_p3_lut_segments_v1";
const CUDA_SEGMENT_THREADS_PER_BLOCK: u32 = 128;
const KERNEL_PTX: &str = include_str!("../kernels/p3_lut_segments.ptx");
#[cfg(test)]
const KERNEL_MANIFEST: &str = include_str!("../kernels/p3_lut_segments.manifest.json");

#[derive(Clone, Copy)]
enum KernelImage {
    Cubin(&'static [u8]),
    Ptx(&'static str),
}

#[derive(Clone, Copy)]
struct KernelArtifact {
    label: &'static str,
    image: KernelImage,
}

macro_rules! cubin_artifact {
    ($architecture:literal) => {
        KernelArtifact {
            label: concat!("sm_", stringify!($architecture), " CUBIN"),
            image: KernelImage::Cubin(include_bytes!(concat!(
                "../kernels/p3_lut_segments_sm",
                stringify!($architecture),
                ".cubin"
            ))),
        }
    };
}

fn kernel_artifact_for_compute_capability(major: i32, minor: i32) -> KernelArtifact {
    match major * 10 + minor {
        75 => cubin_artifact!(75),
        80 => cubin_artifact!(80),
        86 => cubin_artifact!(86),
        87 => cubin_artifact!(87),
        88 => cubin_artifact!(88),
        89 => cubin_artifact!(89),
        90 => cubin_artifact!(90),
        100 => cubin_artifact!(100),
        103 => cubin_artifact!(103),
        110 => cubin_artifact!(110),
        120 => cubin_artifact!(120),
        121 => cubin_artifact!(121),
        _ => KernelArtifact {
            label: "compute_75 PTX fallback",
            image: KernelImage::Ptx(KERNEL_PTX),
        },
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaLutSegment {
    pub(crate) first_node: u32,
    pub(crate) node_count: u32,
}

struct UploadedLut {
    point_count: usize,
    values: cudarc::driver::CudaSlice<f64>,
}

#[derive(Default)]
struct HostBatchStaging {
    base_indices: Vec<u64>,
    upper_offsets: Vec<u64>,
    upper_fractions: Vec<f64>,
    active_counts: Vec<u32>,
    number_concentrations: Vec<f64>,
    fall_speeds: Vec<f64>,
    segment_starts: Vec<u32>,
    segment_counts: Vec<u32>,
}

struct DeviceBatchBuffers {
    node_capacity: usize,
    segment_capacity: usize,
    base_indices: cudarc::driver::CudaSlice<u64>,
    upper_offsets: cudarc::driver::CudaSlice<u64>,
    upper_fractions: cudarc::driver::CudaSlice<f64>,
    active_counts: cudarc::driver::CudaSlice<u32>,
    number_concentrations: cudarc::driver::CudaSlice<f64>,
    fall_speeds: cudarc::driver::CudaSlice<f64>,
    segment_starts: cudarc::driver::CudaSlice<u32>,
    segment_counts: cudarc::driver::CudaSlice<u32>,
    outputs: cudarc::driver::CudaSlice<f64>,
    error_codes: cudarc::driver::CudaSlice<u32>,
    error_nodes: cudarc::driver::CudaSlice<u32>,
}

/// Loaded deterministic CUDA kernel bound to one NVIDIA primary context.
/// Table payloads become persistent in the next adapter layer; this low-level
/// executor intentionally accepts an explicit validated payload so its ABI and
/// parity can be tested independently.
pub(crate) struct CudaLutExecutor {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    device: CudaDeviceInfo,
    kernel_artifact: &'static str,
    uploaded_luts: HashMap<Sha256Digest, UploadedLut>,
    host_batch: HostBatchStaging,
    device_batch: Option<DeviceBatchBuffers>,
}

impl std::fmt::Debug for CudaLutExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaLutExecutor")
            .field("device", &self.device)
            .field("kernel", &KERNEL_NAME)
            .field("kernel_artifact", &self.kernel_artifact)
            .finish_non_exhaustive()
    }
}

impl CudaLutExecutor {
    pub fn open(ordinal: usize) -> Result<Self, CudaSegmentExecutionError> {
        let context = CudaContext::new(ordinal)
            .map_err(|error| driver_error("retain CUDA primary context", error))?;
        let name = context
            .name()
            .map_err(|error| driver_error("read CUDA device name", error))?;
        let (compute_capability_major, compute_capability_minor) = context
            .compute_capability()
            .map_err(|error| driver_error("read CUDA compute capability", error))?;
        if !cudarc::gpuwm_hip::requested() && (compute_capability_major, compute_capability_minor) < MINIMUM_COMPUTE_CAPABILITY {
            return Err(CudaSegmentExecutionError::UnsupportedComputeCapability {
                actual_major: compute_capability_major,
                actual_minor: compute_capability_minor,
                minimum_major: MINIMUM_COMPUTE_CAPABILITY.0,
                minimum_minor: MINIMUM_COMPUTE_CAPABILITY.1,
            });
        }
        let total_memory_bytes = context
            .total_mem()
            .map_err(|error| driver_error("read CUDA device memory", error))?;
        let stream = context.default_stream();
        let artifact = kernel_artifact_for_compute_capability(
            compute_capability_major,
            compute_capability_minor,
        );
        let image = if cudarc::gpuwm_hip::requested() {
            Ptx::from_binary(cudarc::gpuwm_hip::compile(include_str!("../kernels/p3_lut_segments.cu"),
                &["--std=c++17".into(),"--fmad=false".into(),"--ftz=false".into()])
                .map_err(|e| CudaSegmentExecutionError::Driver {operation:"compile HIP LUT source",detail:e})?)
        } else {match artifact.image {
            KernelImage::Cubin(bytes) => Ptx::from_binary(bytes.to_vec()),
            KernelImage::Ptx(source) => Ptx::from_src(source),
        }};
        let module = context
            .load_module(image)
            .map_err(|error| driver_error("load BowEcho CUDA kernel", error))?;
        let function = module
            .load_function(KERNEL_NAME)
            .map_err(|error| driver_error("resolve BowEcho CUDA kernel", error))?;
        Ok(Self {
            _context: context,
            stream,
            function,
            device: CudaDeviceInfo {
                ordinal,
                name,
                compute_capability_major,
                compute_capability_minor,
                total_memory_bytes,
            },
            kernel_artifact: if cudarc::gpuwm_hip::requested() {"canonical HIP source"} else {artifact.label},
            uploaded_luts: HashMap::new(),
            host_batch: HostBatchStaging::default(),
            device_batch: None,
        })
    }

    #[must_use]
    pub(crate) const fn device(&self) -> &CudaDeviceInfo {
        &self.device
    }

    #[must_use]
    pub(crate) const fn kernel_artifact(&self) -> &'static str {
        self.kernel_artifact
    }

    pub(crate) fn preload_lut(
        &mut self,
        table_file_sha256: Sha256Digest,
        table_values: &[AdditiveScattering],
    ) -> Result<usize, CudaSegmentExecutionError> {
        if let Some(uploaded) = self.uploaded_luts.get(&table_file_sha256) {
            if uploaded.point_count != table_values.len() {
                return Err(CudaSegmentExecutionError::InvalidInput(format!(
                    "cached LUT {table_file_sha256} has {} points, caller supplied {}",
                    uploaded.point_count,
                    table_values.len()
                )));
            }
            return Ok(uploaded.point_count);
        }
        let table_flat = table_values
            .iter()
            .flat_map(|point| point.components())
            .collect::<Vec<_>>();
        let values = self.copy_to_device("upload persistent LUT payload", &table_flat)?;
        self.uploaded_luts.insert(
            table_file_sha256,
            UploadedLut {
                point_count: table_values.len(),
                values,
            },
        );
        Ok(table_values.len())
    }

    /// Evaluate ordered PSD reduction segments against a previously uploaded
    /// table. One CUDA thread owns one segment and accumulates all nine
    /// components in node order.
    pub(crate) fn evaluate_preloaded_segments(
        &mut self,
        table_file_sha256: Sha256Digest,
        table_point_count: usize,
        nodes: &[CudaLutNodePlan],
        segments: &[CudaLutSegment],
    ) -> Result<Vec<AdditiveScattering>, CudaSegmentExecutionError> {
        let uploaded = self.uploaded_luts.get(&table_file_sha256).ok_or_else(|| {
            CudaSegmentExecutionError::InvalidInput(format!(
                "LUT {table_file_sha256} was not preloaded"
            ))
        })?;
        if uploaded.point_count != table_point_count {
            return Err(CudaSegmentExecutionError::InvalidInput(format!(
                "preloaded LUT {table_file_sha256} has {} points, token declares {table_point_count}",
                uploaded.point_count
            )));
        }
        validate_batch(table_point_count, nodes, segments)?;
        if segments.is_empty() {
            return Ok(Vec::new());
        }
        self.stage_host_batch(nodes, segments);
        self.ensure_device_batch_capacity(nodes.len(), segments.len())?;
        let stream = Arc::clone(&self.stream);
        let host = &self.host_batch;
        let device = self
            .device_batch
            .as_mut()
            .expect("nonempty CUDA batch allocated reusable device buffers");
        copy_to_existing(
            &stream,
            "upload LUT base indices",
            &host.base_indices,
            &mut device.base_indices,
        )?;
        copy_to_existing(
            &stream,
            "upload LUT upper offsets",
            &host.upper_offsets,
            &mut device.upper_offsets,
        )?;
        copy_to_existing(
            &stream,
            "upload LUT upper fractions",
            &host.upper_fractions,
            &mut device.upper_fractions,
        )?;
        copy_to_existing(
            &stream,
            "upload LUT active-axis counts",
            &host.active_counts,
            &mut device.active_counts,
        )?;
        copy_to_existing(
            &stream,
            "upload LUT population concentrations",
            &host.number_concentrations,
            &mut device.number_concentrations,
        )?;
        copy_to_existing(
            &stream,
            "upload LUT fall speeds",
            &host.fall_speeds,
            &mut device.fall_speeds,
        )?;
        copy_to_existing(
            &stream,
            "upload segment starts",
            &host.segment_starts,
            &mut device.segment_starts,
        )?;
        copy_to_existing(
            &stream,
            "upload segment counts",
            &host.segment_counts,
            &mut device.segment_counts,
        )?;

        let table_point_count = table_point_count as u64;
        let node_count = nodes.len() as u32;
        let segment_count = segments.len() as u32;
        let mut launch = self.stream.launch_builder(&self.function);
        let table_device = &self
            .uploaded_luts
            .get(&table_file_sha256)
            .expect("LUT upload was inserted before launch")
            .values;
        launch
            .arg(table_device)
            .arg(&table_point_count)
            .arg(&device.base_indices)
            .arg(&device.upper_offsets)
            .arg(&device.upper_fractions)
            .arg(&device.active_counts)
            .arg(&device.number_concentrations)
            .arg(&device.fall_speeds)
            .arg(&node_count)
            .arg(&device.segment_starts)
            .arg(&device.segment_counts)
            .arg(&segment_count)
            .arg(&mut device.outputs)
            .arg(&mut device.error_codes)
            .arg(&mut device.error_nodes);
        let grid_blocks = segment_count.div_ceil(CUDA_SEGMENT_THREADS_PER_BLOCK);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (grid_blocks, 1, 1),
                block_dim: (CUDA_SEGMENT_THREADS_PER_BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_error("launch CUDA LUT segment kernel", error))?;

        let error_codes = stream
            .clone_dtoh(&device.error_codes.slice(..segments.len()))
            .map_err(|error| driver_error("download CUDA error codes", error))?;
        let error_nodes = stream
            .clone_dtoh(&device.error_nodes.slice(..segments.len()))
            .map_err(|error| driver_error("download CUDA error nodes", error))?;
        if let Some((segment, &code)) = error_codes.iter().enumerate().find(|(_, code)| **code != 0)
        {
            return Err(CudaSegmentExecutionError::Kernel {
                segment,
                node: error_nodes[segment] as usize,
                code,
            });
        }
        let output = stream
            .clone_dtoh(
                &device
                    .outputs
                    .slice(..segments.len() * CUDA_LUT_COMPONENT_COUNT),
            )
            .map_err(|error| driver_error("download CUDA segment output", error))?;
        output
            .as_chunks::<CUDA_LUT_COMPONENT_COUNT>()
            .0
            .iter()
            .enumerate()
            .map(|(segment, values)| {
                let components = *values;
                AdditiveScattering::from_components(components)
                    .map_err(|source| CudaSegmentExecutionError::InvalidOutput { segment, source })
            })
            .collect()
    }

    fn stage_host_batch(&mut self, nodes: &[CudaLutNodePlan], segments: &[CudaLutSegment]) {
        let host = &mut self.host_batch;
        host.base_indices.clear();
        host.upper_offsets.clear();
        host.upper_fractions.clear();
        host.active_counts.clear();
        host.number_concentrations.clear();
        host.fall_speeds.clear();
        host.segment_starts.clear();
        host.segment_counts.clear();
        host.base_indices
            .extend(nodes.iter().map(|node| node.base_point_index));
        host.upper_offsets
            .extend(nodes.iter().flat_map(|node| node.upper_point_offsets));
        host.upper_fractions
            .extend(nodes.iter().flat_map(|node| node.upper_fractions));
        host.active_counts
            .extend(nodes.iter().map(|node| node.active_axis_count));
        host.number_concentrations
            .extend(nodes.iter().map(|node| node.number_concentration_m3));
        host.fall_speeds
            .extend(nodes.iter().map(|node| node.positive_down_fall_speed_m_s));
        host.segment_starts
            .extend(segments.iter().map(|segment| segment.first_node));
        host.segment_counts
            .extend(segments.iter().map(|segment| segment.node_count));
    }

    fn ensure_device_batch_capacity(
        &mut self,
        node_count: usize,
        segment_count: usize,
    ) -> Result<(), CudaSegmentExecutionError> {
        let sufficient = self.device_batch.as_ref().is_some_and(|buffers| {
            buffers.node_capacity >= node_count && buffers.segment_capacity >= segment_count
        });
        if sufficient {
            return Ok(());
        }
        let node_capacity = self
            .device_batch
            .as_ref()
            .map_or(node_count, |buffers| buffers.node_capacity.max(node_count))
            .next_power_of_two();
        let segment_capacity = self
            .device_batch
            .as_ref()
            .map_or(segment_count, |buffers| {
                buffers.segment_capacity.max(segment_count)
            })
            .next_power_of_two();
        let allocate_u64 = |operation| {
            self.stream
                .alloc_zeros::<u64>(node_capacity)
                .map_err(|error| driver_error(operation, error))
        };
        let allocate_node_f64 = |operation| {
            self.stream
                .alloc_zeros::<f64>(node_capacity)
                .map_err(|error| driver_error(operation, error))
        };
        let allocate_segment_u32 = |operation| {
            self.stream
                .alloc_zeros::<u32>(segment_capacity)
                .map_err(|error| driver_error(operation, error))
        };
        self.device_batch = Some(DeviceBatchBuffers {
            node_capacity,
            segment_capacity,
            base_indices: allocate_u64("allocate CUDA LUT base indices")?,
            upper_offsets: self
                .stream
                .alloc_zeros::<u64>(node_capacity * CUDA_MAX_ACTIVE_AXES)
                .map_err(|error| driver_error("allocate CUDA LUT upper offsets", error))?,
            upper_fractions: self
                .stream
                .alloc_zeros::<f64>(node_capacity * CUDA_MAX_ACTIVE_AXES)
                .map_err(|error| driver_error("allocate CUDA LUT upper fractions", error))?,
            active_counts: self
                .stream
                .alloc_zeros::<u32>(node_capacity)
                .map_err(|error| driver_error("allocate CUDA LUT active-axis counts", error))?,
            number_concentrations: allocate_node_f64(
                "allocate CUDA LUT population concentrations",
            )?,
            fall_speeds: allocate_node_f64("allocate CUDA LUT fall speeds")?,
            segment_starts: allocate_segment_u32("allocate CUDA segment starts")?,
            segment_counts: allocate_segment_u32("allocate CUDA segment counts")?,
            outputs: self
                .stream
                .alloc_zeros::<f64>(segment_capacity * CUDA_LUT_COMPONENT_COUNT)
                .map_err(|error| driver_error("allocate CUDA segment output", error))?,
            error_codes: allocate_segment_u32("allocate CUDA error codes")?,
            error_nodes: allocate_segment_u32("allocate CUDA error nodes")?,
        });
        Ok(())
    }

    fn copy_to_device<T: cudarc::driver::DeviceRepr>(
        &self,
        operation: &'static str,
        values: &[T],
    ) -> Result<cudarc::driver::CudaSlice<T>, CudaSegmentExecutionError> {
        self.stream
            .clone_htod(values)
            .map_err(|error| driver_error(operation, error))
    }
}

fn copy_to_existing<T: cudarc::driver::DeviceRepr>(
    stream: &Arc<CudaStream>,
    operation: &'static str,
    values: &[T],
    device: &mut cudarc::driver::CudaSlice<T>,
) -> Result<(), CudaSegmentExecutionError> {
    stream
        .memcpy_htod(values, device)
        .map_err(|error| driver_error(operation, error))
}

fn validate_batch(
    table_point_count: usize,
    nodes: &[CudaLutNodePlan],
    segments: &[CudaLutSegment],
) -> Result<(), CudaSegmentExecutionError> {
    if table_point_count == 0 && !nodes.is_empty() {
        return Err(CudaSegmentExecutionError::InvalidInput(
            "nonempty node batch requires a LUT payload".to_owned(),
        ));
    }
    for (index, node) in nodes.iter().enumerate() {
        let active = node.active_axis_count as usize;
        if active > CUDA_MAX_ACTIVE_AXES {
            return Err(CudaSegmentExecutionError::InvalidInput(format!(
                "node {index} has {active} active axes; maximum is {CUDA_MAX_ACTIVE_AXES}"
            )));
        }
        if !node.number_concentration_m3.is_finite() || node.number_concentration_m3 < 0.0 {
            return Err(CudaSegmentExecutionError::InvalidInput(format!(
                "node {index} has invalid number concentration {}",
                node.number_concentration_m3
            )));
        }
        if !node.positive_down_fall_speed_m_s.is_finite()
            || node.positive_down_fall_speed_m_s <= 0.0
        {
            return Err(CudaSegmentExecutionError::InvalidInput(format!(
                "node {index} has invalid fall speed {}",
                node.positive_down_fall_speed_m_s
            )));
        }
        let mut maximum_point = node.base_point_index;
        for axis in 0..CUDA_MAX_ACTIVE_AXES {
            let fraction = node.upper_fractions[axis];
            let offset = node.upper_point_offsets[axis];
            if axis < active {
                if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) || offset == 0 {
                    return Err(CudaSegmentExecutionError::InvalidInput(format!(
                        "node {index} active axis {axis} has invalid fraction/offset"
                    )));
                }
                maximum_point = maximum_point.checked_add(offset).ok_or_else(|| {
                    CudaSegmentExecutionError::InvalidInput(format!(
                        "node {index} point index overflows"
                    ))
                })?;
            } else if fraction != 0.0 || offset != 0 {
                return Err(CudaSegmentExecutionError::InvalidInput(format!(
                    "node {index} inactive axis {axis} is not zero-filled"
                )));
            }
        }
        if maximum_point >= table_point_count as u64 {
            return Err(CudaSegmentExecutionError::InvalidInput(format!(
                "node {index} addresses LUT point {maximum_point}, but payload has {} points",
                table_point_count
            )));
        }
    }
    for (index, segment) in segments.iter().enumerate() {
        let end = segment
            .first_node
            .checked_add(segment.node_count)
            .ok_or_else(|| {
                CudaSegmentExecutionError::InvalidInput(format!(
                    "segment {index} node range overflows"
                ))
            })?;
        if end as usize > nodes.len() {
            return Err(CudaSegmentExecutionError::InvalidInput(format!(
                "segment {index} ends at node {end}, but batch has {} nodes",
                nodes.len()
            )));
        }
    }
    Ok(())
}

fn driver_error(
    operation: &'static str,
    error: cudarc::driver::DriverError,
) -> CudaSegmentExecutionError {
    CudaSegmentExecutionError::Driver {
        operation,
        detail: error.to_string(),
    }
}

#[derive(Debug, Error)]
pub enum CudaSegmentExecutionError {
    #[error(
        "CUDA compute capability {actual_major}.{actual_minor} is below required {minimum_major}.{minimum_minor}"
    )]
    UnsupportedComputeCapability {
        actual_major: i32,
        actual_minor: i32,
        minimum_major: i32,
        minimum_minor: i32,
    },
    #[error("invalid CUDA LUT batch: {0}")]
    InvalidInput(String),
    #[error("{operation}: {detail}")]
    Driver {
        operation: &'static str,
        detail: String,
    },
    #[error("CUDA LUT kernel failed in segment {segment}, node {node}, code {code}")]
    Kernel {
        segment: usize,
        node: usize,
        code: u32,
    },
    #[error("CUDA LUT segment {segment} produced invalid additive output: {source}")]
    InvalidOutput {
        segment: usize,
        #[source]
        source: OutputError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn checked_in_artifact(file: &str) -> Option<&'static [u8]> {
        Some(match file {
            "p3_lut_segments_sm75.cubin" => include_bytes!("../kernels/p3_lut_segments_sm75.cubin"),
            "p3_lut_segments_sm80.cubin" => include_bytes!("../kernels/p3_lut_segments_sm80.cubin"),
            "p3_lut_segments_sm86.cubin" => include_bytes!("../kernels/p3_lut_segments_sm86.cubin"),
            "p3_lut_segments_sm87.cubin" => include_bytes!("../kernels/p3_lut_segments_sm87.cubin"),
            "p3_lut_segments_sm88.cubin" => include_bytes!("../kernels/p3_lut_segments_sm88.cubin"),
            "p3_lut_segments_sm89.cubin" => include_bytes!("../kernels/p3_lut_segments_sm89.cubin"),
            "p3_lut_segments_sm90.cubin" => include_bytes!("../kernels/p3_lut_segments_sm90.cubin"),
            "p3_lut_segments_sm100.cubin" => {
                include_bytes!("../kernels/p3_lut_segments_sm100.cubin")
            }
            "p3_lut_segments_sm103.cubin" => {
                include_bytes!("../kernels/p3_lut_segments_sm103.cubin")
            }
            "p3_lut_segments_sm110.cubin" => {
                include_bytes!("../kernels/p3_lut_segments_sm110.cubin")
            }
            "p3_lut_segments_sm120.cubin" => {
                include_bytes!("../kernels/p3_lut_segments_sm120.cubin")
            }
            "p3_lut_segments_sm121.cubin" => {
                include_bytes!("../kernels/p3_lut_segments_sm121.cubin")
            }
            "p3_lut_segments.ptx" => KERNEL_PTX.as_bytes(),
            _ => return None,
        })
    }

    fn ordered_cpu_segments(
        table: &[AdditiveScattering],
        nodes: &[CudaLutNodePlan],
        segments: &[CudaLutSegment],
    ) -> Vec<AdditiveScattering> {
        segments
            .iter()
            .map(|segment| {
                let mut accumulated = AdditiveScattering::default();
                for node in &nodes[segment.first_node as usize
                    ..(segment.first_node + segment.node_count) as usize]
                {
                    let mut interpolated = [0.0; CUDA_LUT_COMPONENT_COUNT];
                    for corner in 0..(1_u32 << node.active_axis_count) {
                        let mut point = node.base_point_index as usize;
                        let mut weight = 1.0;
                        for axis in 0..node.active_axis_count as usize {
                            let upper = ((corner >> axis) & 1) == 1;
                            let fraction = node.upper_fractions[axis];
                            if upper {
                                weight *= fraction;
                                point += node.upper_point_offsets[axis] as usize;
                            } else {
                                weight *= 1.0 - fraction;
                            }
                        }
                        let point_components = table[point].components();
                        for component in 0..CUDA_LUT_COMPONENT_COUNT {
                            interpolated[component] += weight * point_components[component];
                        }
                    }
                    let speed = node.positive_down_fall_speed_m_s;
                    interpolated[7] = interpolated[0] * speed;
                    interpolated[8] = interpolated[0] * speed * speed;
                    let contribution = AdditiveScattering::from_components(interpolated)
                        .unwrap()
                        .checked_scale(node.number_concentration_m3)
                        .unwrap();
                    accumulated = accumulated.checked_add(contribution).unwrap();
                }
                accumulated
            })
            .collect()
    }

    fn valid_point(zh: f64) -> AdditiveScattering {
        AdditiveScattering::from_components([
            zh,
            zh,
            0.5 * zh,
            0.0,
            0.01 * zh,
            0.001 * zh,
            0.0008 * zh,
            zh,
            zh,
        ])
        .unwrap()
    }

    fn interior_node(scale: f64, speed: f64) -> CudaLutNodePlan {
        let mut offsets = [0; CUDA_MAX_ACTIVE_AXES];
        offsets[0] = 2;
        offsets[1] = 1;
        let mut fractions = [0.0; CUDA_MAX_ACTIVE_AXES];
        fractions[0] = 0.25;
        fractions[1] = 0.75;
        CudaLutNodePlan {
            base_point_index: 0,
            upper_point_offsets: offsets,
            upper_fractions: fractions,
            active_axis_count: 2,
            number_concentration_m3: scale,
            positive_down_fall_speed_m_s: speed,
        }
    }

    fn five_axis_node(scale: f64, speed: f64) -> CudaLutNodePlan {
        let mut offsets = [0; CUDA_MAX_ACTIVE_AXES];
        offsets[..5].copy_from_slice(&[16, 8, 4, 2, 1]);
        let mut fractions = [0.0; CUDA_MAX_ACTIVE_AXES];
        fractions[..5].copy_from_slice(&[0.25, 0.75, 0.375, 0.625, 0.5]);
        CudaLutNodePlan {
            base_point_index: 0,
            upper_point_offsets: offsets,
            upper_fractions: fractions,
            active_axis_count: 5,
            number_concentration_m3: scale,
            positive_down_fall_speed_m_s: speed,
        }
    }

    #[test]
    fn input_validation_rejects_out_of_bounds_plan_before_launch() {
        let mut node = interior_node(1.0, 2.0);
        node.base_point_index = 3;
        let error = validate_batch(
            4,
            &[node],
            &[CudaLutSegment {
                first_node: 0,
                node_count: 1,
            }],
        )
        .unwrap_err();
        assert!(error.to_string().contains("addresses LUT point"));
    }

    #[test]
    fn checked_in_kernel_manifest_authenticates_source_and_every_artifact() {
        let manifest: serde_json::Value = serde_json::from_str(KERNEL_MANIFEST).unwrap();
        let source = include_bytes!("../kernels/p3_lut_segments.cu");
        assert_eq!(manifest["abi_revision"], 1);
        assert_eq!(manifest["kernel"], KERNEL_NAME);
        assert_eq!(
            manifest["source_sha256"].as_str().unwrap(),
            format!("{:x}", Sha256::digest(source))
        );
        let artifacts = manifest["artifacts"].as_array().unwrap();
        assert_eq!(artifacts.len(), 13);
        for artifact in artifacts {
            let file = artifact["file"].as_str().unwrap();
            let bytes = checked_in_artifact(file)
                .unwrap_or_else(|| panic!("manifest names unchecked artifact {file}"));
            assert_eq!(artifact["bytes"].as_u64().unwrap(), bytes.len() as u64);
            assert_eq!(
                artifact["sha256"].as_str().unwrap(),
                format!("{:x}", Sha256::digest(bytes)),
                "{file}"
            );
        }
    }

    #[test]
    fn actual_gpu_segment_is_deterministic_and_matches_ordered_cpu_math() {
        let availability = crate::probe_cuda();
        let Some(device) = availability.preferred_device() else {
            eprintln!("skipping CUDA execution parity: {availability:?}");
            return;
        };
        let mut executor = CudaLutExecutor::open(device.ordinal).unwrap();
        assert_eq!(
            executor.kernel_artifact(),
            kernel_artifact_for_compute_capability(
                device.compute_capability_major,
                device.compute_capability_minor
            )
            .label
        );
        let table = [
            valid_point(10.0),
            valid_point(20.0),
            valid_point(30.0),
            valid_point(40.0),
        ];
        let nodes = [interior_node(2.0, 3.0), interior_node(0.5, 4.0)];
        let segments = [CudaLutSegment {
            first_node: 0,
            node_count: 2,
        }];
        let table_digest = Sha256Digest::compute(b"deterministic synthetic CUDA LUT");
        let table_point_count = executor.preload_lut(table_digest, &table).unwrap();

        // Fractions are exact binary values. Use the literal scalar operation
        // sequence owned by OfflineLut + checked_scale + checked_add as the
        // oracle; decimal literals can differ by one ULP from that sequence.
        let expected = ordered_cpu_segments(&table, &nodes, &segments);
        assert_eq!(expected[0].components()[0], 56.25);
        let first = executor
            .evaluate_preloaded_segments(table_digest, table_point_count, &nodes, &segments)
            .unwrap();
        assert_eq!(
            first[0].components().map(f64::to_bits),
            expected[0].components().map(f64::to_bits)
        );
        let expected_bits = first[0].components().map(f64::to_bits);
        for _ in 0..20 {
            let repeated = executor
                .evaluate_preloaded_segments(table_digest, table_point_count, &nodes, &segments)
                .unwrap();
            assert_eq!(repeated[0].components().map(f64::to_bits), expected_bits);
        }

        // Grow every reusable host/device staging array, then deliberately
        // submit a much smaller batch with different values. Only the active
        // prefixes may be copied, launched, and downloaded; stale output/error
        // tails from the larger launch must remain unreachable.
        let large_nodes = vec![interior_node(1.25, 2.5); 64];
        let large_segments = (0..large_nodes.len())
            .map(|node| CudaLutSegment {
                first_node: node as u32,
                node_count: 1,
            })
            .collect::<Vec<_>>();
        let large_expected = ordered_cpu_segments(&table, &large_nodes, &large_segments);
        let large = executor
            .evaluate_preloaded_segments(
                table_digest,
                table_point_count,
                &large_nodes,
                &large_segments,
            )
            .unwrap();
        assert_eq!(large, large_expected);
        assert_eq!(
            executor.device_batch.as_ref().unwrap().node_capacity,
            large_nodes.len()
        );
        assert_eq!(
            executor.device_batch.as_ref().unwrap().segment_capacity,
            large_segments.len()
        );

        let small_nodes = [interior_node(0.125, 7.0)];
        let small_segments = [CudaLutSegment {
            first_node: 0,
            node_count: 1,
        }];
        let small_expected = ordered_cpu_segments(&table, &small_nodes, &small_segments);
        let small = executor
            .evaluate_preloaded_segments(
                table_digest,
                table_point_count,
                &small_nodes,
                &small_segments,
            )
            .unwrap();
        assert_eq!(small.len(), 1);
        assert_eq!(small, small_expected);
        assert_eq!(
            executor.device_batch.as_ref().unwrap().node_capacity,
            large_nodes.len(),
            "smaller follow-up must reuse the existing node buffers"
        );
        assert_eq!(
            executor.device_batch.as_ref().unwrap().segment_capacity,
            large_segments.len(),
            "smaller follow-up must reuse the existing output/error buffers"
        );
    }

    #[test]
    #[ignore = "manual low-level CUDA launch benchmark"]
    fn manual_gpu_65k_one_node_segment_throughput() {
        const NODE_COUNT: usize = 65_536;
        const SAMPLES: usize = 5;

        let availability = crate::probe_cuda();
        let Some(device) = availability.preferred_device() else {
            eprintln!("skipping CUDA launch benchmark: {availability:?}");
            return;
        };
        let mut executor = CudaLutExecutor::open(device.ordinal).unwrap();
        let table = (0..32)
            .map(|point| valid_point(10.0 + f64::from(point)))
            .collect::<Vec<_>>();
        let nodes = vec![five_axis_node(1.25, 2.5); NODE_COUNT];
        let segments = (0..NODE_COUNT)
            .map(|node| CudaLutSegment {
                first_node: node as u32,
                node_count: 1,
            })
            .collect::<Vec<_>>();
        let digest = Sha256Digest::compute(b"manual CUDA launch benchmark LUT");
        let point_count = executor.preload_lut(digest, &table).unwrap();
        let expected = ordered_cpu_segments(&table, &nodes, &segments);
        let mut samples = Vec::with_capacity(SAMPLES);
        let mut actual = Vec::new();
        for _ in 0..SAMPLES {
            let started = std::time::Instant::now();
            actual = executor
                .evaluate_preloaded_segments(digest, point_count, &nodes, &segments)
                .unwrap();
            samples.push(started.elapsed());
        }
        assert_eq!(actual, expected);
        let steady = samples
            .iter()
            .skip(1)
            .map(|sample| sample.as_secs_f64())
            .sum::<f64>()
            / (SAMPLES - 1) as f64;
        eprintln!(
            "CUDA low-level 65k one-node segments: cold {:.6} s; steady {:.6} s ({:.0} nodes/s); samples {:?}",
            samples[0].as_secs_f64(),
            steady,
            NODE_COUNT as f64 / steady,
            samples
                .iter()
                .map(std::time::Duration::as_secs_f64)
                .collect::<Vec<_>>()
        );
    }
}
