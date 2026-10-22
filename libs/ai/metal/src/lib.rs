//! THE Metal store: shaders + metallib build, device/runtime, try_* shims,
//! and the host GpuTensor path. Compiled-graph execution lives in
//! makepad-ai-llm (the graph is that model).
//!
//! Makepad's own kernels run on the Metal stream (`stream`): Cutie's ops
//! (`cutie_ops`), the generic ops Cutie calls (`generic_ops`), and the fp32
//! GEMM's users, `linear` and `attention` (RationalVisions
//! docs/cutie-metal.md).

pub mod affine;
#[cfg(target_os = "macos")]
pub mod attention;
pub mod backend_kind;
pub mod cutie_ops;
pub mod generic_ops;
pub mod gpu_types;
#[cfg(target_os = "macos")]
pub mod gpu_tensor;
#[cfg(target_os = "macos")]
pub mod linear;
#[cfg(target_os = "macos")]
mod objc_util;
pub mod residency;
pub mod runtime;
pub mod shim;
pub mod stream;
pub mod rife;

pub use affine::*;
pub use backend_kind::*;
pub use gpu_types::{GpuLinearPart, GpuTensor};
pub use runtime::*;
pub use shim::*;
