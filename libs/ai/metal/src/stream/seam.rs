//! The seam on macOS (cutie-metal.md §8) in the resident binding (step 12): a tensor is storage
//! plus an offset (`gpu_types.rs`), bound where it lies. A launch claims and checks its tensors,
//! stamps their storage with the open command buffer's serial (its last use, and for `Out` and
//! `InOut` its last write) and encodes; nothing waits. A read on the host waits for the tensor's
//! own writer only, and a host write goes into storage no queued work uses, or through a staging
//! buffer and a copy kernel queued after that work.
//!
//! - `GpuTensor::alloc` takes a pooled buffer as it was left (`take_gpu`): its contents are
//!   undefined. MAKEPAD_METAL_POISON=1 (or `set_poison_allocs` on a thread) fills each with
//!   [`UNWRITTEN_BITS`] instead, a NaN, from a buffer no queued work uses, so what an op leaves
//!   unwritten shows.
//! - A launch never waits and never acknowledges a failure: a failed command buffer is reported by
//!   the next wait at or after it (a download, `with_host`, the in-flight cap, `synchronize`), and
//!   what was written from it until a synchronize acknowledges it is refused as an input or a host
//!   read (`Ring::check_written`).
//! - MAKEPAD_METAL_MISALIGN=1 makes every tensor the host writes (`upload`, `from_host`, the host
//!   ops' results) start one word into its buffer, and `is_aligned16()` false for every tensor, so
//!   kernels' unaligned paths run on real unaligned bindings as well as on row views.

use super::ctx::{encode_dispatch, Slot, StreamCtx};
use super::grid::{check_grid, Grid, KernelLimits};
use super::kernel::{FnConst, Kernel, KernelLib, KernelParams};
use super::kernels::{TensorCopy2d, TensorFill, TENSOR_COPY2D, TENSOR_FILL};
use super::plan::{check_args, check_slots, SlotKind, SlotPlan};
use super::storage::{self, MtlStorage};
use super::{MetalResult, StreamStats, UNWRITTEN_BITS};
use crate::gpu_types::{fresh_tensor_id, GpuTensor};
use crate::objc_util::{AutoreleasePool, MtlBuf};
use crate::residency::{env_usize, Storage};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

/// The calling thread's stream: not started yet, unavailable (with the reason), or running.
enum StreamState {
    Unstarted,
    Unavailable(String),
    Running(StreamCtx),
}

thread_local! {
    static STREAM: RefCell<StreamState> = const { RefCell::new(StreamState::Unstarted) };
    /// The content ids of the tensors the op running on this thread allocated, and so may write;
    /// `None` outside `with_metal_stream`.
    static OP_ALLOCS: RefCell<Option<Vec<u64>>> = const { RefCell::new(None) };
    /// The storage (by address) of every tensor this thread is reading on the host right now
    /// (a live [`HostSlice`]): an in-place write into one is refused.
    static HOST_READS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// `set_poison_allocs`' override of MAKEPAD_METAL_POISON for this thread.
    static POISON: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Runs `f` on the calling thread's stream, starting it first when `start` is true (a failed
/// start leaves the thread without one, and every later call is `Err` with the reason).
/// `Ok(None)` when the thread has no stream and `start` is false. While `f` runs the stream is
/// borrowed: a nested call is `Err`, never a panic.
fn with_ctx<T>(op: &str, start: bool, f: impl FnOnce(&mut StreamCtx) -> MetalResult<T>) -> MetalResult<Option<T>> {
    STREAM
        .try_with(|cell| {
            let Ok(mut state) = cell.try_borrow_mut() else {
                return Err(format!(
                    "{op}: the thread's Metal stream is in use (a gpu_* or host call inside with_metal_stream)"
                ));
            };
            if matches!(*state, StreamState::Unstarted) {
                if !start {
                    return Ok(None);
                }
                *state = match StreamCtx::new() {
                    Ok(ctx) => StreamState::Running(ctx),
                    Err(e) => {
                        eprintln!("[metal stream] unavailable on this thread: {e}");
                        StreamState::Unavailable(e)
                    }
                };
            }
            match &mut *state {
                StreamState::Running(ctx) => f(ctx).map(Some),
                StreamState::Unavailable(e) => Err(format!("{op}: no Metal stream on this thread: {e}")),
                StreamState::Unstarted => Ok(None),
            }
        })
        .unwrap_or_else(|_| Err(format!("{op}: the thread is exiting")))
}

/// [`with_ctx`], starting the stream: `f`'s result, or `Err` when the thread has none.
fn with_started<T>(op: &str, f: impl FnOnce(&mut StreamCtx) -> MetalResult<T>) -> MetalResult<T> {
    with_ctx(op, true, f)?.ok_or_else(|| format!("{op}: no Metal stream on this thread"))
}

/// Marks the span of one op on this thread: tensors allocated inside it may be written by it.
struct OpScope;

impl OpScope {
    fn begin() -> OpScope {
        OP_ALLOCS.with(|ids| *ids.borrow_mut() = Some(Vec::new()));
        OpScope
    }
}

impl Drop for OpScope {
    fn drop(&mut self) {
        let _ = OP_ALLOCS.try_with(|ids| *ids.borrow_mut() = None);
    }
}

fn note_alloc(id: u64) {
    let _ = OP_ALLOCS.try_with(|ids| {
        if let Some(ids) = ids.borrow_mut().as_mut() {
            ids.push(id);
        }
    });
}

fn allocated_here(id: u64) -> bool {
    OP_ALLOCS.try_with(|ids| ids.borrow().as_ref().is_some_and(|ids| ids.contains(&id))).unwrap_or(false)
}

fn inside_op() -> bool {
    OP_ALLOCS.try_with(|ids| ids.borrow().is_some()).unwrap_or(false)
}

/// `Err` when called inside `with_metal_stream`: a host call may wait for the stream, whose
/// context the op holds. The seam's host side and the host ops on `GpuTensor` (`gpu_tensor.rs`,
/// `rife.rs`) all ask.
pub(crate) fn host_call(op: &str) -> MetalResult<()> {
    if inside_op() {
        return Err(format!("{op}: a host call inside with_metal_stream; make it before or after the op"));
    }
    Ok(())
}

/// The address `HOST_READS` and the launch's alias check know storage by.
fn storage_key(storage: &Arc<MtlStorage>) -> usize {
    Arc::as_ptr(storage) as usize
}

/// `Err` when this thread is reading `t`'s storage on the host (a live [`HostSlice`] or a
/// `with_host` closure): an in-place write would change what it reads.
fn check_not_host_read(op: &str, t: &GpuTensor) -> MetalResult<()> {
    let key = storage_key(&t.storage);
    if HOST_READS.try_with(|reads| reads.borrow().contains(&key)).unwrap_or(false) {
        return Err(format!("{op}: the tensor is being read on the host (with_host); write it after the read"));
    }
    Ok(())
}

/// MAKEPAD_METAL_MISALIGN=1: tensors the host writes start one word into their buffer and
/// `is_aligned16()` is false for every tensor, so kernels' unaligned paths run, and are checked.
/// Read once.
fn misaligned_inputs() -> bool {
    static MISALIGN: OnceLock<bool> = OnceLock::new();
    *MISALIGN.get_or_init(|| env_usize("MAKEPAD_METAL_MISALIGN", 0) != 0)
}

/// The word a tensor the host writes starts at: 1 with MAKEPAD_METAL_MISALIGN, else 0.
fn host_lead() -> usize {
    usize::from(misaligned_inputs())
}

/// Whether `GpuTensor::alloc` fills new tensors with [`UNWRITTEN_BITS`] on this thread:
/// `set_poison_allocs`' override, else MAKEPAD_METAL_POISON (read once).
fn poison_allocs() -> bool {
    static POISON_ENV: OnceLock<bool> = OnceLock::new();
    POISON
        .try_with(Cell::get)
        .ok()
        .flatten()
        .unwrap_or_else(|| *POISON_ENV.get_or_init(|| env_usize("MAKEPAD_METAL_POISON", 0) != 0))
}

/// `t`'s element count as a kernel's u32, or `Err` naming `op`.
pub(crate) fn element_count(op: &str, t: &GpuTensor) -> MetalResult<u32> {
    let n = t.rows().checked_mul(t.cols()).ok_or_else(|| format!("{op}: {}x{} overflows", t.rows(), t.cols()))?;
    u32::try_from(n).map_err(|_| format!("{op}: {n} elements is more than a u32 index reaches"))
}

/// Runs one op on the calling thread's stream (created lazily): an autorelease pool, the open
/// command buffer and its serial encoder, then a budget or SYNC commit on exit. `op` names the op
/// in errors and command-buffer labels. Inside `f`, call no gpu_* function and no
/// with_host/download (the context is borrowed: Err, not a panic).
pub fn with_metal_stream<T>(
    op: &'static str,
    f: impl FnOnce(&mut MetalStream<'_>) -> MetalResult<T>,
) -> MetalResult<T> {
    with_started(op, |ctx| {
        let _pool = AutoreleasePool::new();
        let _scope = OpScope::begin();
        let mut stream = MetalStream { ctx, op };
        let result = f(&mut stream);
        let end = stream.ctx.end_op();
        let value = result?;
        end?;
        Ok(value)
    })
}

/// One op's view of the stream, inside [`with_metal_stream`].
pub struct MetalStream<'ctx> {
    ctx: &'ctx mut StreamCtx,
    op: &'static str,
}

/// Cached device weights (a retained MTLBuffer); cheap to clone.
#[derive(Clone)]
pub struct MetalWeights {
    buf: MtlBuf,
    floats: usize,
}

impl MetalWeights {
    pub fn floats(&self) -> usize {
        self.floats
    }
}

/// Buffer arguments at [[buffer(1)]], [[buffer(2)]], ... in slice order. Every one but `Bytes` is a
/// `device` pointer in the kernel, never `constant`: tensors and weights are bound at any 4-byte
/// offset, and Metal wants a `constant` buffer's offset 256-byte aligned on macOS.
#[derive(Clone, Copy)]
pub enum Bind<'a> {
    /// Read at the tensor's offset.
    In(&'a GpuTensor),
    /// Written; must be a tensor this op allocated. A launch may write part of it (a GEMM's
    /// `c_offset`, one part of a concat): the rest keeps what earlier launches wrote, or `alloc`'s
    /// undefined contents.
    Out(&'a GpuTensor),
    /// Read and written; allocated by this op.
    InOut(&'a GpuTensor),
    /// Read, at a float offset (a bias after its weights); `device const float*` in the kernel.
    Weights(&'a MetalWeights, usize),
    /// setBytes, at most 4096 bytes (ECA taps, small tables).
    Bytes(&'a [u8]),
    /// A 16-byte dummy, so slot indices never shift.
    Absent,
}

impl Bind<'_> {
    fn plan(&self) -> SlotPlan {
        let tensor = |kind: SlotKind, t: &GpuTensor, writes: bool| SlotPlan {
            kind,
            tensor: t as *const GpuTensor as usize,
            storage: storage_key(&t.storage),
            words: (t.offset, t.offset.saturating_add(t.rows.saturating_mul(t.cols))),
            allocated_here: writes && allocated_here(t.id.get()),
        };
        let other = |kind: SlotKind| SlotPlan { kind, tensor: 0, storage: 0, words: (0, 0), allocated_here: false };
        match *self {
            Bind::In(t) => tensor(SlotKind::In, t, false),
            Bind::Out(t) => tensor(SlotKind::Out, t, true),
            Bind::InOut(t) => tensor(SlotKind::InOut, t, true),
            Bind::Weights(w, offset) => other(SlotKind::Weights { offset, floats: w.floats }),
            Bind::Bytes(bytes) => other(SlotKind::Bytes(bytes.len())),
            Bind::Absent => other(SlotKind::Absent),
        }
    }
}

/// One GEMM operand: where it is bound, its float offset, leading dimension, whether it is
/// transposed, and the stride between batches.
#[derive(Clone, Copy)]
pub struct GemmOperand<'a> {
    pub src: Bind<'a>,
    pub offset: usize,
    pub ld: usize,
    pub trans: bool,
    pub batch_stride: usize,
}

/// C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i], i < batch; fp32 multiply-accumulate, never
/// half-staged; k == 0 writes beta * C. `c` is Bind::Out (beta 0) or Bind::InOut.
pub struct Gemm<'a> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    pub alpha: f32,
    pub beta: f32,
    pub a: GemmOperand<'a>,
    pub b: GemmOperand<'a>,
    pub c: Bind<'a>,
    pub c_offset: usize,
    pub ldc: usize,
    pub c_batch_stride: usize,
}

/// The slot that binds `t` where it lies.
fn tensor_slot(t: &GpuTensor) -> Slot {
    Slot::Buffer { id: t.storage.buf().as_id(), offset: (t.offset * 4) as u64 }
}

impl MetalStream<'_> {
    /// Encodes `kernel` (specialised with `consts`) over `grid`, with `params` at [[buffer(0)]]
    /// and `binds` from [[buffer(1)]], into the open command buffer; nothing waits. Checks the
    /// params, the bindings and the grid first, and that they give the kernel everything it reads
    /// (params, every buffer index, every threadgroup index it uses); then that every tensor may be
    /// used here: one another thread's stream still uses is `Err` (`Storage::claim`), and an input
    /// written at or after a failed command buffer no synchronize has acknowledged, or between a
    /// failure and its acknowledgement, is `Err` (`Ring::check_written`). An empty grid encodes
    /// nothing (the library is still compiled and the checks still made, so a broken launch is an
    /// `Err` whatever the sizes).
    pub fn launch<P: KernelParams>(
        &mut self,
        kernel: &Kernel,
        consts: &[FnConst],
        params: &P,
        binds: &[Bind<'_>],
        grid: Grid,
    ) -> MetalResult<()> {
        let op = self.op;
        let plans: Vec<SlotPlan> = binds.iter().map(Bind::plan).collect();
        check_slots(op, kernel.function, std::mem::size_of::<P>(), &plans)?;
        let pipeline = self.ctx.pipeline(kernel, consts).map_err(|e| format!("{op}: {e}"))?;
        if let Some(args) = &pipeline.args {
            check_args(op, kernel.function, args, std::mem::size_of::<P>(), binds.len(), &grid.threadgroup_bytes)?;
        }
        check_grid(
            op,
            kernel.function,
            &grid,
            pipeline.limits,
            pipeline.static_threadgroup_bytes,
            self.ctx.max_threadgroup_bytes(),
        )?;
        if grid.is_empty() {
            return Ok(());
        }
        let mut slots = Vec::with_capacity(binds.len());
        // Each tensor's storage, and whether the dispatch writes it.
        let mut used: Vec<(&MtlStorage, bool)> = Vec::with_capacity(binds.len());
        for bind in binds {
            let slot = match *bind {
                Bind::In(t) => {
                    self.claim(t, true)?;
                    used.push((&t.storage, false));
                    tensor_slot(t)
                }
                Bind::Out(t) => {
                    self.claim(t, false)?;
                    used.push((&t.storage, true));
                    tensor_slot(t)
                }
                Bind::InOut(t) => {
                    self.claim(t, true)?;
                    used.push((&t.storage, true));
                    tensor_slot(t)
                }
                Bind::Weights(weights, offset) => {
                    Slot::Buffer { id: weights.buf.as_id(), offset: (offset * 4) as u64 }
                }
                Bind::Bytes(bytes) if !bytes.is_empty() => {
                    Slot::Bytes { ptr: bytes.as_ptr().cast(), len: bytes.len() as u64 }
                }
                Bind::Bytes(_) | Bind::Absent => Slot::Buffer { id: self.ctx.dummy(), offset: 0 },
            };
            slots.push(slot);
        }
        let (encoder, serial) = self.ctx.begin_dispatch(op).map_err(|e| format!("{op}: {e}"))?;
        for (storage, writes) in used {
            if writes {
                storage.mark_write(serial);
            } else {
                storage.mark_use(serial);
            }
        }
        encode_dispatch(
            encoder,
            pipeline.state,
            params as *const P as *const c_void,
            std::mem::size_of::<P>(),
            &slots,
            &grid,
        );
        Ok(())
    }

    /// Makes `t`'s storage this stream's to bind (adopting it from a thread whose stream has
    /// finished with it, or `Err` while that stream still uses it), and when the launch `reads`
    /// it, checks that what last wrote it may be trusted.
    fn claim(&mut self, t: &GpuTensor, reads: bool) -> MetalResult<()> {
        let op = self.op;
        t.storage.claim(self.ctx.clock(), op)?;
        if reads {
            self.ctx.check_written(t.storage.last_write()).map_err(|e| format!("{op}: {e}"))?;
        }
        Ok(())
    }

    /// Lets this op write `t` though it did not allocate it: `copy_into`'s in-place copy, after
    /// its checks (strict shapes, storage no view shares).
    fn write_in_place(&mut self, t: &GpuTensor) {
        note_alloc(t.id.get());
    }

    /// `kernel`'s pipeline limits with `consts` (compiling it if it is not yet).
    pub fn limits(&mut self, kernel: &Kernel, consts: &[FnConst]) -> MetalResult<KernelLimits> {
        let op = self.op;
        self.ctx.pipeline(kernel, consts).map(|p| p.limits).map_err(|e| format!("{op}: {e}"))
    }

    /// Fills `dst` with the 32-bit word `bits` (dst allocated by this op).
    pub fn fill(&mut self, dst: &GpuTensor, bits: u32) -> MetalResult<()> {
        let n = element_count(self.op, dst)?;
        self.launch(&TENSOR_FILL, &[], &TensorFill { n, bits }, &[Bind::Out(dst)], Grid::linear(n as usize, 256)?)
    }

    /// Copies `src` into `dst` word for word (equal element counts; dst allocated by this op).
    pub fn copy(&mut self, src: &GpuTensor, dst: &GpuTensor) -> MetalResult<()> {
        let n = element_count(self.op, src)?;
        let into = element_count(self.op, dst)?;
        if n != into {
            return Err(format!("{}: copy of {n} elements into a tensor of {into}", self.op));
        }
        let params = TensorCopy2d { rows: 1, cols: n, src_ld: n, dst_ld: n, src_offset: 0, dst_offset: 0 };
        let grid = Grid::cover([n as usize, 1, 1], [256, 1, 1])?;
        self.launch(&TENSOR_COPY2D, &[], &params, &[Bind::In(src), Bind::Out(dst)], grid)
    }

    /// C = alpha * op(A) @ op(B) + beta * C in fp32: checked and planned by `gemm.rs` (what it
    /// refuses, the naive or tiled kernel, MAKEPAD_METAL_GEMM), run by `gemm.metal`.
    pub fn gemm(&mut self, g: &Gemm<'_>) -> MetalResult<()> {
        let op = self.op;
        super::gemm::run(self, op, g)
    }

    /// "{namespace}::{key}::{suffix}"; `fill` runs only on a miss or a size change, writing
    /// straight into Shared memory; it must not call gpu_*.
    pub fn weights(
        &mut self,
        qualified_key: &str,
        floats: usize,
        fill: impl FnOnce(&mut [f32]) -> MetalResult<()>,
    ) -> MetalResult<MetalWeights> {
        if let Some(weights) = self.ctx.weights.get(qualified_key, floats) {
            return Ok(weights.clone());
        }
        let op = self.op;
        let bytes = floats.checked_mul(4).ok_or_else(|| format!("{op}: {floats} weights overflow"))?;
        let device = self.ctx.device();
        // One word more than the weights, so a binding at their end (an empty bias after them)
        // is still inside the buffer, as Metal's validation wants. As the pool does, a failed
        // allocation releases the idle buffers and tries once more.
        let buf = MtlBuf::new_shared(device, bytes + 4)
            .or_else(|| {
                storage::trim();
                MtlBuf::new_shared(device, bytes + 4)
            })
            .ok_or_else(|| format!("{op}: could not allocate {bytes} bytes for weights '{qualified_key}'"))?;
        // SAFETY: a new Shared buffer of `bytes + 4`, page-aligned, that no GPU work uses yet.
        let values = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f32, floats) };
        fill(values)?;
        let weights = MetalWeights { buf, floats };
        let replaced = self.ctx.weights.insert(qualified_key, weights.clone(), floats);
        drop(replaced);
        self.ctx.weight_bytes += bytes as u64;
        Ok(weights)
    }

    /// Host bytes as [ceil(len/4), 1] u32 words, written before this returns, read by later
    /// kernels: a buffer no queued work uses, written on the host.
    pub fn upload_bytes(&mut self, bytes: &[u8]) -> MetalResult<GpuTensor> {
        let op = self.op;
        let words = bytes.len().div_ceil(4);
        let completed = self.ctx.retire();
        let storage = storage::take_host(words * 4, completed).map_err(|e| format!("{op}: {e}"))?;
        // SAFETY: the buffer holds at least `words * 4` bytes (it was taken for them), no queued
        // work uses it, and nothing else refers to it yet.
        unsafe {
            let dst = storage.buf().contents();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            std::ptr::write_bytes(dst.add(bytes.len()), 0, words * 4 - bytes.len());
        }
        Ok(GpuTensor { rows: words, cols: 1, storage, offset: 0, id: Cell::new(fresh_tensor_id()) })
    }
}

impl GpuTensor {
    /// A `[rows, cols]` tensor for the running op to write, `Out` or `InOut`. Callable inside
    /// `with_metal_stream`. Its contents are undefined, a pooled buffer as it was left: write an
    /// element before reading it, and fill before accumulating. With poisoned allocations
    /// (MAKEPAD_METAL_POISON=1, `set_poison_allocs`) every word is [`UNWRITTEN_BITS`], a NaN, so a
    /// read too early shows.
    pub fn alloc(rows: usize, cols: usize) -> MetalResult<GpuTensor> {
        const OP: &str = "GpuTensor::alloc";
        let floats = rows.checked_mul(cols).ok_or_else(|| format!("{OP}: {rows}x{cols} overflows"))?;
        let bytes = floats.checked_mul(4).ok_or_else(|| format!("{OP}: {rows}x{cols} overflows"))?;
        if !storage::has_device() {
            // Outside an op on a thread whose stream has not started (inside one it has).
            with_started(OP, |_| Ok(()))?;
        }
        let storage = if poison_allocs() {
            let storage = storage::take_host_now(bytes).map_err(|e| format!("{OP}: {e}"))?;
            // SAFETY: the buffer holds at least `floats` words, and no queued work uses it.
            let words = unsafe { std::slice::from_raw_parts_mut(storage.buf().contents() as *mut u32, floats) };
            words.fill(UNWRITTEN_BITS);
            storage
        } else {
            storage::take_gpu(bytes).map_err(|e| format!("{OP}: {e}"))?
        };
        let tensor = GpuTensor { rows, cols, storage, offset: 0, id: Cell::new(fresh_tensor_id()) };
        note_alloc(tensor.id.get());
        Ok(tensor)
    }

    /// `[len, 1]` words read as u32, undefined as [`GpuTensor::alloc`]'s.
    pub fn alloc_u32(len: usize) -> MetalResult<GpuTensor> {
        GpuTensor::alloc(len, 1)
    }

    /// Whether the tensor starts on a 16-byte boundary, so a kernel may read it as `float4`: its
    /// buffer is page-aligned, so whether its offset is a multiple of 4 words. Always false with
    /// MAKEPAD_METAL_MISALIGN=1.
    pub fn is_aligned16(&self) -> bool {
        !misaligned_inputs() && self.offset % 4 == 0
    }
}

/// A new tensor of `rows x cols` the host is about to write: a buffer no queued work uses,
/// starting at [`host_lead`]. Nothing else refers to it yet.
fn new_host_tensor(op: &str, rows: usize, cols: usize) -> MetalResult<GpuTensor> {
    let floats = rows.checked_mul(cols).ok_or_else(|| format!("{op}: {rows}x{cols} overflows"))?;
    let lead = host_lead();
    let bytes = floats
        .checked_add(lead)
        .and_then(|words| words.checked_mul(4))
        .ok_or_else(|| format!("{op}: {rows}x{cols} overflows"))?;
    let storage = with_started(op, |ctx| {
        let completed = ctx.retire();
        storage::take_host(bytes, completed).map_err(|e| format!("{op}: {e}"))
    })?;
    Ok(GpuTensor { rows, cols, storage, offset: lead, id: Cell::new(fresh_tensor_id()) })
}

/// The values of a tensor made by [`new_host_tensor`], for the host to write.
///
/// # Safety
/// `t` came from `new_host_tensor` and nothing else refers to its storage (no view, no binding):
/// the slice is the only reference to that memory while it lives.
#[allow(clippy::mut_from_ref)]
unsafe fn host_values_mut(t: &GpuTensor) -> &mut [f32] {
    let base = t.storage.buf().contents() as *mut f32;
    std::slice::from_raw_parts_mut(base.add(t.offset), t.rows * t.cols)
}

/// Copies `values` into `storage`'s buffer from float `at` on.
///
/// # Safety
/// The buffer holds at least `at + values.len()` floats, no queued work uses that memory, and
/// nothing on the host reads it while this runs.
unsafe fn write_storage(storage: &MtlStorage, at: usize, values: &[f32]) {
    debug_assert!(storage.bytes() >= (at + values.len()) * 4);
    let dst = (storage.buf().contents() as *mut f32).add(at);
    std::ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
}

/// A new `[rows, cols]` tensor holding `values`. No GPU work, never waits.
pub fn upload(values: &[f32], rows: usize, cols: usize) -> MetalResult<GpuTensor> {
    upload_as("gpu_upload", values, rows, cols)
}

/// [`upload`], with `op` naming the call in errors (the host ops' results, `gpu_upload`).
pub(crate) fn upload_as(op: &str, values: &[f32], rows: usize, cols: usize) -> MetalResult<GpuTensor> {
    host_call(op)?;
    if rows.checked_mul(cols) != Some(values.len()) {
        return Err(format!("{op} shape mismatch: {} values for {rows}x{cols}", values.len()));
    }
    let t = new_host_tensor(op, rows, cols)?;
    // SAFETY: a new host tensor, nothing else refers to it.
    unsafe { host_values_mut(&t) }.copy_from_slice(values);
    Ok(t)
}

/// A new `[len, 1]` tensor holding u32 words (at least one row, as on CUDA; the one word of an
/// empty upload is 0).
pub fn upload_u32(values: &[u32]) -> MetalResult<GpuTensor> {
    const OP: &str = "gpu_upload_u32";
    host_call(OP)?;
    let t = new_host_tensor(OP, values.len().max(1), 1)?;
    // SAFETY: a new host tensor, nothing else refers to it.
    let words = unsafe { host_values_mut(&t) };
    words.fill(0.0);
    for (word, &value) in words.iter_mut().zip(values) {
        *word = f32::from_bits(value);
    }
    Ok(t)
}

/// A new `[rows, cols]` tensor whose values `fill` writes, starting from zero, straight into the
/// buffer. `fill` may read other tensors (`with_host`); no context is held while it runs.
pub fn from_host(
    rows: usize,
    cols: usize,
    fill: impl FnOnce(&mut [f32]) -> MetalResult<()>,
) -> MetalResult<GpuTensor> {
    const OP: &str = "from_host";
    host_call(OP)?;
    let t = new_host_tensor(OP, rows, cols)?;
    // SAFETY: a new host tensor, nothing else refers to it until it is returned.
    let values = unsafe { host_values_mut(&t) };
    values.fill(0.0);
    fill(values)?;
    Ok(t)
}

/// A tensor's values read on the host where they lie (Shared memory), once the command buffer that
/// last wrote them has finished: what `with_host` gives its closure, and what the legacy host ops
/// read through. While one is alive, an in-place write into its tensor's storage on this thread
/// (`upload_into`, `copy_into`) is refused; ops only ever write tensors they allocate, so nothing
/// else can change it.
pub struct HostSlice<'a> {
    values: &'a [f32],
    storage: usize,
    /// Registered in this thread's `HOST_READS`: it must drop on this thread.
    _thread: PhantomData<*const ()>,
}

impl Deref for HostSlice<'_> {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        self.values
    }
}

impl Drop for HostSlice<'_> {
    fn drop(&mut self) {
        let key = self.storage;
        let _ = HOST_READS.try_with(|reads| {
            let mut reads = reads.borrow_mut();
            if let Some(at) = reads.iter().position(|&k| k == key) {
                reads.swap_remove(at);
            }
        });
    }
}

/// `t`'s values on the host (`op` names the call in errors): waits for the command buffer that
/// last wrote it, and only that one (committing it if it is still open). `Err` inside an op, while
/// another thread's stream still uses the tensor, or when what last wrote it may not be trusted
/// (a failed command buffer, `Ring::check_written`).
pub(crate) fn host_slice<'a>(op: &str, t: &'a GpuTensor) -> MetalResult<HostSlice<'a>> {
    host_call(op)?;
    let floats = t.rows.checked_mul(t.cols).ok_or_else(|| format!("{op}: {}x{} overflows", t.rows, t.cols))?;
    with_started(op, |ctx| {
        t.storage.claim(ctx.clock(), op)?;
        let last_write = t.storage.last_write();
        if last_write > ctx.retire() {
            let _pool = AutoreleasePool::new();
            ctx.wait(last_write).map_err(|e| format!("{op}: {e}"))?;
        }
        ctx.check_written(last_write).map_err(|e| format!("{op}: {e}"))
    })?;
    let key = storage_key(&t.storage);
    HOST_READS.with(|reads| reads.borrow_mut().push(key));
    // SAFETY: the storage holds the tensor's `floats` words from its offset (a tensor never
    // reaches past its storage), its last writer has finished, and the borrow of `t` keeps the
    // storage alive. Nothing writes it while the slice lives: kernels write only tensors their op
    // allocates, and in-place writes from this thread are refused (HOST_READS), from others
    // refused while a view shares the storage (`Storage::check_unshared`).
    let values = unsafe {
        std::slice::from_raw_parts((t.storage.buf().contents() as *const f32).add(t.offset), floats)
    };
    Ok(HostSlice { values, storage: key, _thread: PhantomData })
}

/// Runs `f` on `t`'s values on the host; waits for t's writer only. No context is held while `f`
/// runs, so it may read other tensors.
pub fn with_host<R>(t: &GpuTensor, f: impl FnOnce(&[f32]) -> R) -> MetalResult<R> {
    let values = host_slice("with_host", t)?;
    Ok(f(&values))
}

/// `t`'s values, once its writer has finished (`gpu_download`; CUDA's drains the whole stream:
/// timing code calls `synchronize`).
pub fn download(t: &GpuTensor) -> MetalResult<Vec<f32>> {
    Ok(host_slice("gpu_download", t)?.to_vec())
}

/// The tensor's words as u32 (as `upload_u32` and `alloc_u32` tensors hold them).
pub fn download_u32(t: &GpuTensor) -> MetalResult<Vec<u32>> {
    Ok(host_slice("gpu_download_u32", t)?.iter().map(|v| v.to_bits()).collect())
}

/// Overwrites `t` with `values` (CUDA's `gpu_upload_into`): exactly its element count. Storage no
/// queued work uses takes them in place; busy storage gets them through a staging buffer and a copy
/// kernel queued after the work that uses it, so that work still reads what it was given. `Err`
/// when `t` shares its storage with a view (the write would change the view behind its back) or
/// is being read on the host. `t` gets a new content id.
pub fn upload_into(t: &GpuTensor, values: &[f32]) -> MetalResult<()> {
    const OP: &str = "gpu_upload_into";
    host_call(OP)?;
    if values.len() != t.rows * t.cols {
        return Err(format!("{OP} shape mismatch: {} values for {}x{}", values.len(), t.rows, t.cols));
    }
    check_not_host_read(OP, t)?;
    Storage::check_unshared(&t.storage, OP)?;
    let staging = with_started(OP, |ctx| {
        t.storage.claim(ctx.clock(), OP)?;
        let completed = ctx.retire();
        if t.storage.host_idle(completed) {
            return Ok(None);
        }
        storage::take_host(values.len() * 4, completed).map(Some).map_err(|e| format!("{OP}: {e}"))
    })?;
    match staging {
        None => {
            // SAFETY: the storage holds the tensor's words from its offset, no queued work uses
            // it (host idle), and no host read of it is live (checked above).
            unsafe { write_storage(&t.storage, t.offset, values) };
            t.storage.mark_host_write();
        }
        Some(staging) => {
            // SAFETY: taken for `values.len()` words, no queued work uses it, nothing else refers
            // to it.
            unsafe { write_storage(&staging, 0, values) };
            let src = GpuTensor { rows: t.rows, cols: t.cols, storage: staging, offset: 0, id: Cell::new(fresh_tensor_id()) };
            with_metal_stream(OP, |s| {
                s.write_in_place(t);
                s.copy(&src, t)
            })?;
        }
    }
    t.id.set(fresh_tensor_id());
    Ok(())
}

/// Copies `src` into `dst` on the GPU (CUDA's `gpu_copy_into`): equal shapes, queued after the
/// work that uses either. `Err` when `dst` shares its storage with a view or is being read on the
/// host. `dst` gets a new content id.
pub fn copy_into(src: &GpuTensor, dst: &GpuTensor) -> MetalResult<()> {
    const OP: &str = "gpu_copy_into";
    host_call(OP)?;
    if src.rows != dst.rows || src.cols != dst.cols {
        return Err(format!("{OP} shape mismatch: {}x{} vs {}x{}", src.rows, src.cols, dst.rows, dst.cols));
    }
    if std::ptr::eq(src, dst) {
        return Ok(());
    }
    check_not_host_read(OP, dst)?;
    Storage::check_unshared(&dst.storage, OP)?;
    with_metal_stream(OP, |s| {
        s.write_in_place(dst);
        s.copy(src, dst)
    })?;
    dst.id.set(fresh_tensor_id());
    Ok(())
}

/// A view of `x`'s storage: `rows x cols` from `words` past `x`'s start, with a content id of its
/// own. The caller has checked it lies inside `x`.
pub(crate) fn view(x: &GpuTensor, words: usize, rows: usize, cols: usize) -> GpuTensor {
    GpuTensor { rows, cols, storage: x.storage.clone(), offset: x.offset + words, id: Cell::new(fresh_tensor_id()) }
}

/// Rows `start..start + len` of `x`, a view of its storage (CUDA's `gpu_slice_rows`): no copy.
pub fn slice_rows(x: &GpuTensor, start: usize, len: usize) -> MetalResult<GpuTensor> {
    if start.checked_add(len).is_none_or(|end| end > x.rows) {
        return Err("gpu_slice_rows out of range".to_string());
    }
    Ok(view(x, start * x.cols, len, x.cols))
}

/// `x` as `[rows, cols]`, the same storage and offset (CUDA's `gpu_reshape`), with a content id of
/// its own.
pub fn reshape(x: GpuTensor, rows: usize, cols: usize) -> MetalResult<GpuTensor> {
    let src = x.rows.checked_mul(x.cols);
    if src.is_none() || src != rows.checked_mul(cols) {
        return Err("gpu_reshape element count mismatch".to_string());
    }
    Ok(GpuTensor { rows, cols, storage: x.storage, offset: x.offset, id: Cell::new(fresh_tensor_id()) })
}

/// Commits what the calling thread's stream has open and waits for all of it; returns a GPU
/// failure no synchronize has reported yet, and acknowledges it: later waits are Ok again (until
/// then every wait at or after the failure returns it), and what was written from the failure up
/// to here stays untrusted. Ok at once on a thread with no stream.
pub fn synchronize() -> MetalResult<()> {
    host_call("synchronize")?;
    with_ctx("synchronize", false, |ctx| {
        let _pool = AutoreleasePool::new();
        ctx.synchronize()
    })
    .map(|_| ())
}

/// Whether the calling thread's stream starts (the device and a queue).
pub fn available() -> bool {
    with_ctx("available", true, |_| Ok(())).is_ok()
}

/// Whether `lib` compiles on the calling thread's stream (compiling it now if it has not).
pub fn library_available(lib: &'static KernelLib) -> bool {
    library_ready(lib).is_ok()
}

/// [`library_available`] with the reason: `Err` with Metal's compile error, or why the thread has
/// no stream.
pub fn library_ready(lib: &'static KernelLib) -> MetalResult<()> {
    with_started("library_ready", |ctx| ctx.library(lib).map(|_| ()))
}

/// Drops the calling thread's cached weights whose key starts with `prefix`; returns how many.
/// With `only_if_loaded`, a thread with no stream is left without one (Ok(0)), as on CUDA.
pub fn evict_weights(prefix: &str, only_if_loaded: bool) -> MetalResult<usize> {
    with_ctx("evict_weights", !only_if_loaded, |ctx| Ok(ctx.weights.evict_prefix(prefix))).map(|n| n.unwrap_or(0))
}

/// Releases the calling thread's idle buffers.
pub fn pool_clear() {
    pool_trim();
}

/// [`pool_clear`], returning the bytes released.
pub fn pool_trim() -> usize {
    storage::trim()
}

/// Overrides the calling thread's idle-pool cap (`None`: MAKEPAD_METAL_POOL_CAP_MB again).
pub fn pool_cap_override(bytes: Option<usize>) {
    storage::set_cap_override(bytes);
}

/// The counters (the calling thread's pool and ring) and the device's memory, zeroed afterwards
/// when `reset`. Starts no stream.
pub fn stats(reset: bool) -> StreamStats {
    let pool = storage::stats(reset);
    let mut stats = StreamStats {
        pool_fresh_alloc_count: pool.pool_fresh_alloc_count,
        pool_fresh_alloc_bytes: pool.pool_fresh_alloc_bytes,
        pool_overcap_free_bytes: pool.pool_overcap_free_bytes,
        pool_oom_clears: pool.pool_oom_clears,
        ..StreamStats::default()
    };
    let _ = with_ctx("stats", false, |ctx| {
        let (total, allocated) = ctx.memory();
        stats.mem_total_bytes = total;
        stats.mem_free_bytes = total.saturating_sub(allocated);
        stats.weight_bytes = ctx.weight_bytes;
        if reset {
            ctx.weight_bytes = 0;
        }
        let ring = ctx.ring_stats(reset);
        stats.commits = ring.commits;
        stats.waits = ring.waits;
        stats.gpu_ms = ring.gpu_ms;
        Ok(())
    });
    stats
}

/// Commits the calling thread's open command buffer without waiting for it (for tests and
/// benches that need work in a command buffer of its own). Ok with nothing open or no stream.
#[doc(hidden)]
pub fn flush() -> MetalResult<()> {
    host_call("flush")?;
    with_ctx("flush", false, |ctx| {
        let _pool = AutoreleasePool::new();
        ctx.commit_open();
        Ok(())
    })
    .map(|_| ())
}

/// Where a tensor lies and what its storage records (for the Mac's residency tests).
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency {
    /// The `MTLBuffer`'s address: two tensors on one buffer share storage.
    pub buffer: usize,
    /// The buffer's size (its pool bucket).
    pub bytes: usize,
    /// The tensor's offset in words.
    pub offset: usize,
    /// The id of the stream that owns the storage.
    pub owner: u64,
    pub last_use: u64,
    pub last_write: u64,
}

#[doc(hidden)]
pub fn residency(t: &GpuTensor) -> Residency {
    Residency {
        buffer: t.storage.buf().as_id() as usize,
        bytes: t.storage.bytes(),
        offset: t.offset,
        owner: t.storage.owner(),
        last_use: t.storage.last_use(),
        last_write: t.storage.last_write(),
    }
}

/// The calling thread's stream as it stands, after retiring what has finished (for the Mac's
/// residency tests). All zero on a thread with no stream; starts none.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingState {
    /// The stream's id (a storage's `owner` when this thread's stream owns it).
    pub stream: u64,
    /// The serial of the command buffer being encoded, if any.
    pub open: Option<u64>,
    pub last_committed: u64,
    /// The last serial retired.
    pub completed: u64,
    /// Committed, not yet retired.
    pub in_flight: usize,
    /// The bytes the thread's pool holds idle.
    pub idle_bytes: usize,
}

#[doc(hidden)]
pub fn ring_state() -> RingState {
    let mut state = RingState { idle_bytes: storage::idle_bytes(), ..RingState::default() };
    let _ = with_ctx("ring_state", false, |ctx| {
        let completed = ctx.retire();
        state.stream = ctx.clock().id();
        state.open = ctx.open_serial();
        state.last_committed = ctx.last_committed();
        state.completed = completed;
        state.in_flight = ctx.in_flight();
        Ok(())
    });
    state
}

/// Poisons (`Some(true)`) or not (`Some(false)`) the calling thread's new tensors, whatever
/// MAKEPAD_METAL_POISON says; `None` goes back to it.
#[doc(hidden)]
pub fn set_poison_allocs(on: Option<bool>) {
    let _ = POISON.try_with(|poison| poison.set(on));
}
