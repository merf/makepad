//! The tensor handle of the `gpu_*` surface on Metal (RationalVisions docs/cutie-metal.md §2).
//!
//! On macOS a tensor is GPU-resident: shared storage (a pooled Shared `MTLBuffer`, the stream that
//! owns it, and the serials of its last use and write) plus an offset in f32 words, as CUDA's is a
//! buffer plus an offset. Row views (`gpu_slice_rows`, `gpu_reshape`, adjacent row concatenations)
//! clone the storage; ops enqueue their work and return, and only a read of the host side waits,
//! for the tensor's own writer (`stream::with_host`). Elsewhere it stays the host struct that
//! `rife.rs` and the shim's `DecLinearRef` compile against.

#[cfg(target_os = "macos")]
use std::cell::Cell;
#[cfg(not(target_os = "macos"))]
use std::cell::RefCell;
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
static NEXT_TENSOR_ID: AtomicU64 = AtomicU64::new(1);

/// A process-unique identity for a tensor's current contents.
#[cfg(target_os = "macos")]
pub(crate) fn fresh_tensor_id() -> u64 {
    NEXT_TENSOR_ID.fetch_add(1, Ordering::Relaxed)
}

/// A `[rows, cols]` f32 tensor (or `[len, 1]` u32 words, read as u32, as on CUDA) at `offset` words
/// into storage it may share with its views. `Send`, not `Sync`: a tensor belongs to the thread
/// that uses it, and one handed to another thread is adopted there once its owner's work on it
/// has finished (`gpu_synchronize` on the owner first).
#[cfg(target_os = "macos")]
pub struct GpuTensor {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// The buffer, shared with every view of it.
    pub(crate) storage: Arc<crate::stream::storage::MtlStorage>,
    /// Where the tensor starts in the storage, in f32 words.
    pub(crate) offset: usize,
    /// Identity of the contents, kept only for the legacy two-way layer's weight cache
    /// (shim.rs), keyed by it: fresh at creation, for every view and after every in-place write, so
    /// that cache can never serve a stale weight (a pointer can be reused; an id cannot).
    pub(crate) id: Cell<u64>,
}

/// The host tensor of the non-macOS builds, where nothing runs it: `rife.rs` and the shim's
/// `DecLinearRef` compile against it.
#[cfg(not(target_os = "macos"))]
pub struct GpuTensor {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: RefCell<Vec<f32>>,
}

impl GpuTensor {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn is_half(&self) -> bool {
        false
    }
}

pub struct GpuLinearPart<'a> {
    pub bt_ggml_type: u32,
    pub n: usize,
    pub cache_key: &'a str,
    pub bytes: &'a [u8],
}
