//! The Metal stream and the kernel seam (RationalVisions docs/cutie-metal.md §3 and §8): Makepad's
//! own kernels (`shaders/makepad/`) launched on a per-thread stream of command buffers, written
//! once against this seam and run unchanged by both of its bindings.
//!
//! - **Host binding** (now, until step 12): tensors are today's host `GpuTensor`s. An input is
//!   copied into a Shared buffer (kept by content id, so a weight is copied once), an output gets
//!   a pooled buffer, and every launch commits, waits and copies its outputs back. Correct and
//!   slow: one wait a launch.
//! - **Resident binding** (step 12 on): a tensor is storage plus an offset, bound where it lies;
//!   ops enqueue and return, and only a read of the host side waits.
//!
//! The kernel-side types (libraries, kernels, function constants, params, grids) and the checks a
//! launch makes compile on every OS and are unit-tested anywhere; the stream itself is macOS only
//! and is checked by the Mac's `tests/stream*.rs`.
//!
//! Rules for kernel authors:
//! 1. Do every shape check and u32 conversion before [`with_metal_stream`], with CUDA's messages.
//!    A failure returns `Err` with nothing encoded.
//! 2. Write only tensors this op allocated ([`GpuTensor::alloc`] inside the op), each bound
//!    `Out` or `InOut`.
//! 3. Inputs may be views at any 4-byte offset; use `float4` only behind `is_aligned16()`.
//! 4. Follow D8 (exactness in MSL).
//! 5. Temporaries may drop inside the closure while their dispatches are still pending.
//! 6. A library that fails to compile is an `Err` from the op, never a silent fallback.

// The crate-private helpers in these modules serve the macOS stream; elsewhere only the tests use
// them.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub mod grid;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub mod kernel;
pub mod kernels;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub mod source;

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod mirror;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod plan;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod weight_map;

#[cfg(target_os = "macos")]
mod ctx;
#[cfg(target_os = "macos")]
mod seam;
#[cfg(target_os = "macos")]
mod storage;
#[cfg(target_os = "macos")]
pub mod tensor_ops;

#[cfg(test)]
mod tests;

pub use grid::{Grid, KernelLimits};
pub use kernel::{FnConst, Kernel, KernelLib, KernelParams};
pub use plan::{MAX_BINDS, MAX_SET_BYTES};

#[cfg(target_os = "macos")]
pub use seam::{
    available, download, download_u32, evict_weights, from_host, library_available, pool_cap_override,
    pool_clear, pool_trim, stats, synchronize, upload, upload_u32, with_host, with_metal_stream, Bind, Gemm,
    GemmOperand, MetalStream, MetalWeights,
};

/// What every stream call returns: `Err` carries a message naming the op.
pub type MetalResult<T> = Result<T, String>;

/// The stream's counters since they were last reset, and the device's memory. Pool and ring
/// counters are the calling thread's.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StreamStats {
    /// `recommendedMaxWorkingSetSize`.
    pub mem_total_bytes: u64,
    /// The working set less `currentAllocatedSize`.
    pub mem_free_bytes: u64,
    /// Bytes written into the weight cache (CUDA's `weight_stream_bytes`).
    pub weight_bytes: u64,
    pub pool_fresh_alloc_count: u64,
    pub pool_fresh_alloc_bytes: u64,
    pub pool_overcap_free_bytes: u64,
    pub pool_oom_clears: u64,
    /// Command buffers committed.
    pub commits: u64,
    /// Times the host blocked on a command buffer.
    pub waits: u64,
    /// GPU time of the command buffers retired.
    pub gpu_ms: f64,
}
