//! The Metal stream and the kernel seam (RationalVisions docs/cutie-metal.md §3 and §8): Makepad's
//! own kernels (`shaders/makepad/`) launched on a per-thread stream of command buffers, written
//! once against this seam and run unchanged by both of its bindings.
//!
//! - **Host binding** (steps 6 to 11, gone since step 12): tensors were host `Vec`s copied into
//!   Shared buffers for each launch, and every launch committed, waited and copied its outputs
//!   back. Correct and slow: one wait a launch. Step 11's reports from it are the golden the
//!   resident binding is held to, bit for bit.
//! - **Resident binding** (step 12 on): a tensor is storage plus an offset, bound where it lies;
//!   ops enqueue and return, and only a read of the host side waits, for that tensor's writer.
//!
//! A launch is checked against what its kernel reads (from the pipeline's reflection) as well as
//! its own bindings, and host calls inside an op are `Err`.
//!
//! The kernel-side types (libraries, kernels, function constants, params, grids) and the checks a
//! launch makes compile on every OS and are unit-tested anywhere; the stream itself is macOS only
//! and is checked by the Mac's `tests/stream*.rs`.
//!
//! Rules for kernel authors:
//! 1. Do every shape check and u32 conversion before [`with_metal_stream`], with CUDA's messages.
//!    A failure returns `Err` with nothing encoded. Inside the closure call no `gpu_*` function,
//!    no host op on a `GpuTensor` and no download: host data comes in through `weights`,
//!    `upload_bytes` or `Bind::Bytes`.
//! 2. Write only tensors this op allocated ([`GpuTensor::alloc`] inside the op), each bound
//!    `Out` or `InOut`, and bind nothing else in that launch that overlaps it (not the tensor
//!    again, not a view of it): a launch is refused otherwise. A launch may write part of an
//!    output; later launches in the op see what it wrote.
//! 3. A new tensor's contents are undefined (a pooled buffer as it was left; NaN with
//!    MAKEPAD_METAL_POISON=1): write every element before it is read, and fill before
//!    accumulating.
//! 4. Bind every buffer the kernel reads (`Bind::Absent` for a slot it does not), pass params when
//!    it reads `[[buffer(0)]]`, and give every `[[threadgroup(i)]]` it uses its memory.
//! 5. Inputs may be views at any 4-byte offset; use `float4` only behind `is_aligned16()`
//!    (MAKEPAD_METAL_MISALIGN=1 tries the other path). Buffer arguments are `device` pointers,
//!    never `constant`, apart from `Bind::Bytes` tables: Metal wants a `constant` buffer's offset
//!    256-byte aligned on macOS.
//! 6. Follow D8 (exactness in MSL).
//! 7. Temporaries may drop inside the closure while their dispatches are still pending.
//! 8. A library that fails to compile is an `Err` from the op, never a silent fallback.

// The crate-private helpers in these modules serve the macOS stream; elsewhere only the tests use
// them.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub mod gemm;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub mod grid;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub mod kernel;
pub mod kernels;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub mod source;

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod plan;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod weight_map;

#[cfg(target_os = "macos")]
mod ctx;
#[cfg(target_os = "macos")]
mod mps;
#[cfg(target_os = "macos")]
mod seam;
#[cfg(target_os = "macos")]
pub(crate) mod storage;
#[cfg(target_os = "macos")]
pub mod tensor_ops;

#[cfg(test)]
mod tests;

pub use grid::{Grid, KernelLimits};
pub use kernel::{FnConst, Kernel, KernelLib, KernelParams};
pub use plan::{MAX_BINDS, MAX_SET_BYTES};

#[cfg(target_os = "macos")]
pub use seam::{
    available, copy_into, download, download_u32, evict_weights, from_host, library_available,
    library_ready, pool_cap_override, pool_clear, pool_trim, reshape, slice_rows, stats, synchronize, upload, upload_into,
    upload_u32, with_host, with_metal_stream, Bind, Gemm, GemmOperand, HostSlice, MetalStream, MetalWeights, MpsConv, MpsGraphRun, MpsTransformerRun,
};
#[cfg(target_os = "macos")]
#[doc(hidden)]
pub use seam::{flush, ring_state, residency, set_poison_allocs, Residency, RingState};
#[cfg(target_os = "macos")]
pub(crate) use seam::{host_slice, upload_as, view};

/// What every stream call returns: `Err` carries a message naming the op.
pub type MetalResult<T> = Result<T, String>;

/// The word `GpuTensor::alloc` fills a new tensor with when allocations are poisoned
/// (MAKEPAD_METAL_POISON=1, `set_poison_allocs`): a quiet NaN (payload 0xdead), so a kernel that
/// reads an element before writing it, or an op that leaves one unwritten, shows NaN where a 0
/// would have looked right. Otherwise a new tensor starts as its pooled buffer was left.
pub const UNWRITTEN_BITS: u32 = 0x7fc0_dead;

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
