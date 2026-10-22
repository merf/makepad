//! The seam on macOS (cutie-metal.md §8), in the host binding: the kernel code written against it
//! runs over today's host `GpuTensor` (a `RefCell<Vec<f32>>`) until step 12 makes tensors
//! resident, and needs no change then.
//!
//! In the host binding an `In` is copied into a Shared buffer (kept by the tensor's content id in
//! the context's mirror, so weights are copied once). An `Out` or `InOut` gets a host-written copy
//! of its tensor, so what a launch does not write keeps its value, as it will when the tensor's own
//! storage is bound. Every launch commits, waits, and copies `Out` and `InOut` back into their
//! tensors (giving each a new content id); a GPU failure is the launch's `Err`, with nothing copied
//! back. `alloc` fills a tensor with [`UNWRITTEN_BITS`], a NaN. `is_aligned16()` is true, unless
//! MAKEPAD_METAL_MISALIGN=1 binds every input 4 bytes into its buffer to try kernels' unaligned
//! paths.

use super::ctx::{encode_dispatch, Slot, StreamCtx};
use super::grid::{check_grid, Grid, KernelLimits};
use super::kernel::{FnConst, Kernel, KernelLib, KernelParams};
use super::kernels::{TensorCopy2d, TensorFill, TENSOR_COPY2D, TENSOR_FILL};
use super::plan::{check_args, check_slots, SlotKind, SlotPlan};
use super::storage::{self, MtlStorage};
use super::{MetalResult, StreamStats, UNWRITTEN_BITS};
use crate::gpu_tensor;
use crate::gpu_types::{fresh_tensor_id, GpuTensor};
use crate::objc_util::{AutoreleasePool, MtlBuf};
use crate::residency::env_usize;
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

/// The calling thread's stream: not started yet, unavailable (with the reason), or running.
enum StreamState {
    Unstarted,
    Unavailable(String),
    Running(StreamCtx),
}

thread_local! {
    static STREAM: RefCell<StreamState> = const { RefCell::new(StreamState::Unstarted) };
    /// The content ids of the tensors the op running on this thread allocated (and wrote); `None`
    /// outside `with_metal_stream`.
    static OP_ALLOCS: RefCell<Option<Vec<u64>>> = const { RefCell::new(None) };
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

/// `Err` when called inside `with_metal_stream`: host calls wait for the stream in the resident
/// binding, so they are refused here too, where they would have worked. The seam's host side and
/// the host ops on `GpuTensor` (`gpu_tensor.rs`, `rife.rs`) all ask.
pub(crate) fn host_call(op: &str) -> MetalResult<()> {
    if inside_op() {
        return Err(format!("{op}: a host call inside with_metal_stream; make it before or after the op"));
    }
    Ok(())
}

/// MAKEPAD_METAL_MISALIGN=1: the host binding binds every input 4 bytes into its buffer and
/// `is_aligned16()` is false, so kernels' unaligned paths run, and are checked, before residency
/// makes real views. Read once.
fn misaligned_inputs() -> bool {
    static MISALIGN: OnceLock<bool> = OnceLock::new();
    *MISALIGN.get_or_init(|| env_usize("MAKEPAD_METAL_MISALIGN", 0) != 0)
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
    let out = with_ctx(op, true, |ctx| {
        let _pool = AutoreleasePool::new();
        let _scope = OpScope::begin();
        let mut stream = MetalStream { ctx, op };
        let result = f(&mut stream);
        let end = stream.ctx.end_op();
        let value = result?;
        end?;
        Ok(value)
    })?;
    out.ok_or_else(|| format!("{op}: no Metal stream on this thread"))
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
            allocated_here: writes && allocated_here(t.id.get()),
        };
        match *self {
            Bind::In(t) => tensor(SlotKind::In, t, false),
            Bind::Out(t) => tensor(SlotKind::Out, t, true),
            Bind::InOut(t) => tensor(SlotKind::InOut, t, true),
            Bind::Weights(w, offset) => SlotPlan {
                kind: SlotKind::Weights { offset, floats: w.floats },
                tensor: 0,
                allocated_here: false,
            },
            Bind::Bytes(bytes) => SlotPlan { kind: SlotKind::Bytes(bytes.len()), tensor: 0, allocated_here: false },
            Bind::Absent => SlotPlan { kind: SlotKind::Absent, tensor: 0, allocated_here: false },
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

/// A launch's buffers in the host binding: the slots to encode, the input copies to keep alive
/// until the work is done, and the written buffers with the binding index of their tensor and
/// their float count, to copy back.
struct HostBound {
    slots: Vec<Slot>,
    read: Vec<Arc<MtlStorage>>,
    written: Vec<(usize, Arc<MtlStorage>, usize)>,
}

impl MetalStream<'_> {
    /// Encodes `kernel` (specialised with `consts`) over `grid`, with `params` at [[buffer(0)]]
    /// and `binds` from [[buffer(1)]]. Checks the params, the bindings and the grid first, and
    /// that they give the kernel everything it reads (params, every buffer index, every
    /// threadgroup index it uses); an empty grid encodes nothing (the library is still compiled
    /// and the checks still made, so a broken launch is an `Err` whatever the sizes). Host
    /// binding: waits for the work and copies the outputs back; a GPU failure is this launch's
    /// `Err`.
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
        let bound = self.bind_host(binds)?;
        let (encoder, serial) = self.ctx.begin_dispatch(op).map_err(|e| format!("{op}: {e}"))?;
        for storage in &bound.read {
            storage.mark_use(serial);
        }
        for (_, storage, _) in &bound.written {
            storage.mark_write(serial);
        }
        encode_dispatch(
            encoder,
            pipeline.state,
            params as *const P as *const c_void,
            std::mem::size_of::<P>(),
            &bound.slots,
            &grid,
        );
        // The host binding: wait, then copy the outputs back into their tensors. Nothing else is
        // in flight, so a failure is this launch's alone and, with nothing copied back, nothing
        // holds what it wrote: it is acknowledged here, and the stream carries on.
        if let Err(e) = self.ctx.wait(serial) {
            self.ctx.acknowledge_failure();
            return Err(e);
        }
        for (index, storage, floats) in &bound.written {
            let (Bind::Out(t) | Bind::InOut(t)) = binds[*index] else {
                unreachable!("only Out and InOut are written");
            };
            let mut data = t
                .data
                .try_borrow_mut()
                .map_err(|_| format!("{op}: an output tensor is borrowed; it cannot be written back"))?;
            // SAFETY: the buffer holds at least `floats` f32s (it was taken for them), the GPU
            // has finished with it (waited above), and Shared contents are page-aligned.
            let values = unsafe { std::slice::from_raw_parts(storage.buf().contents() as *const f32, *floats) };
            data.copy_from_slice(values);
            drop(data);
            let id = fresh_tensor_id();
            t.id.set(id);
            note_alloc(id);
        }
        Ok(())
    }

    /// The buffers for `binds` in the host binding.
    fn bind_host(&mut self, binds: &[Bind<'_>]) -> MetalResult<HostBound> {
        let op = self.op;
        let mut bound = HostBound { slots: Vec::with_capacity(binds.len()), read: Vec::new(), written: Vec::new() };
        for (index, bind) in binds.iter().enumerate() {
            let slot = match *bind {
                Bind::In(t) => {
                    let (storage, lead) = self.mirror(t)?;
                    let slot = Slot::Buffer { id: storage.buf().as_id(), offset: (lead * 4) as u64 };
                    bound.read.push(storage);
                    slot
                }
                // Out starts from the tensor's values as InOut does: the resident binding binds the
                // tensor's own storage, so what a launch does not write keeps what earlier
                // launches wrote there (or alloc's NaNs). Only the rule checks tell them apart.
                Bind::Out(t) | Bind::InOut(t) => {
                    let data = t.data.try_borrow().map_err(|_| borrowed(op))?;
                    let completed = self.ctx.retire();
                    let storage = storage::take_host(data.len() * 4, completed).map_err(|e| format!("{op}: {e}"))?;
                    write_storage(&storage, 0, &data);
                    let slot = Slot::Buffer { id: storage.buf().as_id(), offset: 0 };
                    bound.written.push((index, storage, data.len()));
                    slot
                }
                Bind::Weights(weights, offset) => {
                    Slot::Buffer { id: weights.buf.as_id(), offset: (offset * 4) as u64 }
                }
                Bind::Bytes(bytes) if !bytes.is_empty() => {
                    Slot::Bytes { ptr: bytes.as_ptr().cast(), len: bytes.len() as u64 }
                }
                Bind::Bytes(_) | Bind::Absent => Slot::Buffer { id: self.ctx.dummy(), offset: 0 },
            };
            bound.slots.push(slot);
        }
        Ok(bound)
    }

    /// The device copy of input `t`, and the float it starts at (1 with MAKEPAD_METAL_MISALIGN,
    /// else 0): the mirror's, kept by content id, or a new one.
    fn mirror(&mut self, t: &GpuTensor) -> MetalResult<(Arc<MtlStorage>, usize)> {
        let op = self.op;
        let data = t.data.try_borrow().map_err(|_| borrowed(op))?;
        let lead = usize::from(misaligned_inputs());
        let bytes = (data.len() + lead) * 4;
        let id = t.id.get();
        if let Some(storage) = self.ctx.mirror.get(id) {
            if storage.bytes() >= bytes {
                return Ok((storage.clone(), lead));
            }
        }
        let completed = self.ctx.retire();
        let storage = storage::take_host(bytes, completed).map_err(|e| format!("{op}: {e}"))?;
        write_storage(&storage, lead, &data);
        let dropped = self.ctx.mirror.insert(id, storage.clone(), storage.bytes());
        drop(dropped);
        Ok((storage, lead))
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
    /// kernels.
    pub fn upload_bytes(&mut self, bytes: &[u8]) -> MetalResult<GpuTensor> {
        let words: Vec<f32> = bytes
            .chunks(4)
            .map(|chunk| {
                let mut word = [0u8; 4];
                word[..chunk.len()].copy_from_slice(chunk);
                f32::from_bits(u32::from_le_bytes(word))
            })
            .collect();
        Ok(gpu_tensor::tensor(words.len(), 1, words))
    }
}

fn borrowed(op: &str) -> String {
    format!("{op}: a bound tensor is mutably borrowed")
}

/// Copies `values` into `storage`'s buffer from float `at` on; the host may write the buffer
/// (taken with `take_host`).
fn write_storage(storage: &MtlStorage, at: usize, values: &[f32]) {
    debug_assert!(storage.bytes() >= (at + values.len()) * 4);
    // SAFETY: the buffer holds at least `at + values.len()` f32s (it was taken for them), no
    // queued work uses it, and Shared contents are page-aligned.
    unsafe {
        let dst = (storage.buf().contents() as *mut f32).add(at);
        std::ptr::copy_nonoverlapping(values.as_ptr(), dst, values.len());
    }
}

impl GpuTensor {
    /// A `[rows, cols]` tensor for the running op to write, `Out` or `InOut`. Callable inside
    /// `with_metal_stream`. Its contents are undefined: write an element before reading it, and
    /// fill before accumulating. The host binding fills it with [`UNWRITTEN_BITS`], a NaN, so a
    /// read too early shows; the resident binding hands out a pooled buffer as it was left.
    pub fn alloc(rows: usize, cols: usize) -> MetalResult<GpuTensor> {
        let floats = rows.checked_mul(cols).ok_or_else(|| format!("GpuTensor::alloc: {rows}x{cols} overflows"))?;
        let tensor = gpu_tensor::tensor(rows, cols, vec![f32::from_bits(UNWRITTEN_BITS); floats]);
        note_alloc(tensor.id.get());
        Ok(tensor)
    }

    /// `[len, 1]` words read as u32, undefined as [`GpuTensor::alloc`]'s.
    pub fn alloc_u32(len: usize) -> MetalResult<GpuTensor> {
        GpuTensor::alloc(len, 1)
    }

    /// Whether the view starts on a 16-byte boundary, so a kernel may read it as `float4`. Always
    /// true in the host binding, which binds every tensor at the start of its copy, unless
    /// MAKEPAD_METAL_MISALIGN=1, which binds inputs 4 bytes in and makes it false for every tensor.
    pub fn is_aligned16(&self) -> bool {
        !misaligned_inputs()
    }
}

/// A new `[rows, cols]` tensor holding `values`.
pub fn upload(values: &[f32], rows: usize, cols: usize) -> MetalResult<GpuTensor> {
    host_call("upload")?;
    gpu_tensor::upload(values, rows, cols)
}

/// A new `[len, 1]` tensor holding u32 words (at least one row, as on CUDA).
pub fn upload_u32(values: &[u32]) -> MetalResult<GpuTensor> {
    host_call("upload_u32")?;
    gpu_tensor::upload_u32(values)
}

/// A new `[rows, cols]` tensor whose values `fill` writes (starting from zero).
pub fn from_host(
    rows: usize,
    cols: usize,
    fill: impl FnOnce(&mut [f32]) -> MetalResult<()>,
) -> MetalResult<GpuTensor> {
    host_call("from_host")?;
    let floats = rows.checked_mul(cols).ok_or_else(|| format!("from_host: {rows}x{cols} overflows"))?;
    let mut values = vec![0.0f32; floats];
    fill(&mut values)?;
    Ok(gpu_tensor::tensor(rows, cols, values))
}

/// Runs `f` on `t`'s values on the host; waits for t's writer only.
pub fn with_host<R>(t: &GpuTensor, f: impl FnOnce(&[f32]) -> R) -> MetalResult<R> {
    read_host("with_host", t, f)
}

fn read_host<R>(op: &str, t: &GpuTensor, f: impl FnOnce(&[f32]) -> R) -> MetalResult<R> {
    host_call(op)?;
    let data = t.data.try_borrow().map_err(|_| format!("{op}: the tensor is mutably borrowed"))?;
    Ok(f(&data))
}

pub fn download(t: &GpuTensor) -> MetalResult<Vec<f32>> {
    read_host("download", t, |values| values.to_vec())
}

/// The tensor's words as u32 (as `upload_u32` and `alloc_u32` tensors hold them).
pub fn download_u32(t: &GpuTensor) -> MetalResult<Vec<u32>> {
    read_host("download_u32", t, |values| values.iter().map(|v| v.to_bits()).collect())
}

/// Commits what the calling thread's stream has open and waits for all of it; returns a GPU
/// failure no synchronize has reported yet, and acknowledges it: later waits are Ok again (in the
/// resident binding, until then every wait at or after the failure returns it, and what was
/// written from the failure up to here stays untrusted). Ok at once on a thread with no stream.
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
    with_ctx("library_available", true, |ctx| ctx.library(lib).map(|_| ())).is_ok()
}

/// Drops the calling thread's cached weights whose key starts with `prefix`; returns how many.
/// With `only_if_loaded`, a thread with no stream is left without one (Ok(0)), as on CUDA.
pub fn evict_weights(prefix: &str, only_if_loaded: bool) -> MetalResult<usize> {
    with_ctx("evict_weights", !only_if_loaded, |ctx| Ok(ctx.weights.evict_prefix(prefix))).map(|n| n.unwrap_or(0))
}

/// Releases the calling thread's idle buffers (and the host binding's input copies).
pub fn pool_clear() {
    pool_trim();
}

/// [`pool_clear`], returning the bytes released.
pub fn pool_trim() -> usize {
    // The copies go back to the pool first (outside the context's borrow), then the pool empties.
    let copies = with_ctx("pool_trim", false, |ctx| Ok(ctx.mirror.clear())).ok().flatten();
    drop(copies);
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
