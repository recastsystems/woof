//! Optional NVIDIA CUDA execution support for synthetic radar.
//!
//! This crate owns the CUDA boundary so the rest of BowEcho remains portable.
//! It links no CUDA libraries: on Windows/Linux the NVIDIA Driver API is
//! discovered at runtime, while macOS and other targets compile a deterministic
//! unsupported-platform stub. A missing/broken driver must therefore remain a
//! normal CPU-fallback condition, never an application startup failure.

use std::sync::OnceLock;

#[cfg(any(windows, target_os = "linux"))]
use thiserror::Error;

mod prepared;

#[cfg(any(windows, target_os = "linux"))]
mod adapter;

#[cfg(any(windows, target_os = "linux"))]
mod kernel;

pub use prepared::{CudaLutNodePreparationError, CudaPreparedTMatrixNode};

#[cfg(any(windows, target_os = "linux"))]
pub use adapter::{
    CudaPreloadedTMatrixTable, CudaSegmentLayoutError, CudaTMatrixExecutionError,
    CudaTMatrixExecutor, CudaTMatrixSegment,
};

/// Minimum architecture supported by the first deterministic f64 P3 kernel.
pub const MINIMUM_COMPUTE_CAPABILITY: (i32, i32) = (7, 5);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub total_memory_bytes: usize,
}

impl CudaDeviceInfo {
    #[must_use]
    pub fn supports_p3_kernel(&self) -> bool {
        if cudarc::gpuwm_hip::requested() {return true;}
        (self.compute_capability_major, self.compute_capability_minor) >= MINIMUM_COMPUTE_CAPABILITY
    }

    #[must_use]
    pub fn compute_capability_label(&self) -> String {
        if cudarc::gpuwm_hip::requested() {
            return format!("HIP {}",cudarc::gpuwm_hip::architecture(self.ordinal).unwrap_or_else(|_|"unknown ISA".into()));
        }
        format!(
            "{}.{}",
            self.compute_capability_major, self.compute_capability_minor
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaAvailability {
    Available {
        driver_api_version: i32,
        devices: Vec<CudaDeviceInfo>,
    },
    UnsupportedPlatform,
    DriverUnavailable,
    NoDevice {
        driver_api_version: i32,
    },
    ProbeFailed(String),
}

impl CudaAvailability {
    #[must_use]
    pub fn supported_devices(&self) -> &[CudaDeviceInfo] {
        match self {
            Self::Available { devices, .. } => devices,
            _ => &[],
        }
    }

    #[must_use]
    pub fn preferred_device(&self) -> Option<&CudaDeviceInfo> {
        self.supported_devices()
            .iter()
            .filter(|device| device.supports_p3_kernel())
            .max_by_key(|device| device.total_memory_bytes)
    }

    #[must_use]
    pub fn fallback_reason(&self) -> Option<String> {
        match self {
            Self::Available { devices, .. }
                if devices.iter().any(CudaDeviceInfo::supports_p3_kernel) =>
            {
                None
            }
            Self::Available { .. } => Some(format!(
                "CUDA devices are older than compute capability {}.{}",
                MINIMUM_COMPUTE_CAPABILITY.0, MINIMUM_COMPUTE_CAPABILITY.1
            )),
            Self::UnsupportedPlatform => {
                Some("NVIDIA CUDA acceleration is available on Windows and Linux".to_owned())
            }
            Self::DriverUnavailable => Some("NVIDIA CUDA driver not found".to_owned()),
            Self::NoDevice { .. } => Some("NVIDIA driver reported no CUDA devices".to_owned()),
            Self::ProbeFailed(error) => Some(format!("CUDA probe failed: {error}")),
        }
    }
}

/// Cached process-wide capability probe. The UI can call this every frame
/// without repeatedly creating primary contexts or loading the driver DLL.
pub fn probe_cuda_cached() -> &'static CudaAvailability {
    static PROBE: OnceLock<CudaAvailability> = OnceLock::new();
    PROBE.get_or_init(probe_cuda)
}

#[cfg(any(windows, target_os = "linux"))]
pub fn probe_cuda() -> CudaAvailability {
    // cudarc's generated loader deliberately panics when asked to resolve a
    // symbol without a CUDA library. Check first, then contain malformed/old
    // driver symbol tables as a fallback condition too.
    let library_present = unsafe { cudarc::driver::sys::is_culib_present() };
    if !library_present {
        return CudaAvailability::DriverUnavailable;
    }
    match std::panic::catch_unwind(probe_cuda_driver) {
        Ok(Ok(availability)) => availability,
        Ok(Err(error)) => CudaAvailability::ProbeFailed(error.to_string()),
        Err(_) => CudaAvailability::ProbeFailed(
            "NVIDIA driver library could not resolve the required Driver API".to_owned(),
        ),
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn probe_cuda() -> CudaAvailability {
    CudaAvailability::UnsupportedPlatform
}

#[cfg(any(windows, target_os = "linux"))]
fn probe_cuda_driver() -> Result<CudaAvailability, CudaProbeError> {
    use cudarc::driver::CudaContext;

    let count = CudaContext::device_count().map_err(|source| CudaProbeError::Driver {
        operation: "enumerate CUDA devices",
        detail: source.to_string(),
    })?;
    let driver_api_version = driver_api_version()?;
    if count <= 0 {
        return Ok(CudaAvailability::NoDevice { driver_api_version });
    }

    let mut devices = Vec::with_capacity(count as usize);
    for ordinal in 0..count as usize {
        let context = CudaContext::new(ordinal).map_err(|source| CudaProbeError::Driver {
            operation: "retain CUDA primary context",
            detail: source.to_string(),
        })?;
        let name = context.name().map_err(|source| CudaProbeError::Driver {
            operation: "read CUDA device name",
            detail: source.to_string(),
        })?;
        let (compute_capability_major, compute_capability_minor) = context
            .compute_capability()
            .map_err(|source| CudaProbeError::Driver {
                operation: "read CUDA compute capability",
                detail: source.to_string(),
            })?;
        let total_memory_bytes = context
            .total_mem()
            .map_err(|source| CudaProbeError::Driver {
                operation: "read CUDA device memory",
                detail: source.to_string(),
            })?;
        devices.push(CudaDeviceInfo {
            ordinal,
            name,
            compute_capability_major,
            compute_capability_minor,
            total_memory_bytes,
        });
    }
    Ok(CudaAvailability::Available {
        driver_api_version,
        devices,
    })
}

#[cfg(any(windows, target_os = "linux"))]
fn driver_api_version() -> Result<i32, CudaProbeError> {
    use cudarc::driver::sys;

    let mut version = 0_i32;
    let result = unsafe { sys::cuDriverGetVersion(&mut version) };
    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(version)
    } else {
        Err(CudaProbeError::Driver {
            operation: "read CUDA driver API version",
            detail: format!("{result:?}"),
        })
    }
}

#[cfg(any(windows, target_os = "linux"))]
#[derive(Clone, Debug, Error, Eq, PartialEq)]
enum CudaProbeError {
    #[error("{operation}: {detail}")]
    Driver {
        operation: &'static str,
        detail: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_floor_is_explicit() {
        let make = |major, minor| CudaDeviceInfo {
            ordinal: 0,
            name: "test".to_owned(),
            compute_capability_major: major,
            compute_capability_minor: minor,
            total_memory_bytes: 1,
        };
        assert!(!make(7, 4).supports_p3_kernel());
        assert!(make(7, 5).supports_p3_kernel());
        assert!(make(12, 0).supports_p3_kernel());
    }

    #[test]
    fn preferred_device_is_supported_device_with_most_memory() {
        let availability = CudaAvailability::Available {
            driver_api_version: 13_000,
            devices: vec![
                CudaDeviceInfo {
                    ordinal: 0,
                    name: "old".to_owned(),
                    compute_capability_major: 5,
                    compute_capability_minor: 2,
                    total_memory_bytes: 64,
                },
                CudaDeviceInfo {
                    ordinal: 1,
                    name: "small".to_owned(),
                    compute_capability_major: 8,
                    compute_capability_minor: 6,
                    total_memory_bytes: 16,
                },
                CudaDeviceInfo {
                    ordinal: 2,
                    name: "large".to_owned(),
                    compute_capability_major: 12,
                    compute_capability_minor: 0,
                    total_memory_bytes: 32,
                },
            ],
        };
        assert_eq!(
            availability.preferred_device().map(|device| device.ordinal),
            Some(2)
        );
        assert_eq!(availability.fallback_reason(), None);
    }

    #[test]
    fn runtime_probe_is_a_normal_value_on_every_machine() {
        let availability = probe_cuda();
        eprintln!("CUDA availability: {availability:?}");
        assert!(matches!(
            availability,
            CudaAvailability::Available { .. }
                | CudaAvailability::UnsupportedPlatform
                | CudaAvailability::DriverUnavailable
                | CudaAvailability::NoDevice { .. }
                | CudaAvailability::ProbeFailed(_)
        ));
    }
}
