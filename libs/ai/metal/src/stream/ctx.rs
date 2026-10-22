//! The calling thread's Metal stream (cutie-metal.md §3): its own device handle and queue, the
//! kernel libraries and pipelines it compiled, the command buffer it is encoding into (one serial
//! compute encoder, so dispatch order is execution order), the ring of command buffers it
//! committed, its weight cache, and the host binding's device copies of tensors.
//!
//! Separate from the shim's context (D1): the shim waits after every op and its batches serve the
//! host-slice `try_*` users; the stream batches and waits only where it must.

use super::grid::{Grid, KernelLimits};
use super::kernel::{consts_match, FnConst, FnConstKey, Kernel, KernelLib};
use super::mirror::{Lru, DEFAULT_MIRROR_CAP_MB};
use super::source::{kernels_src_dir, library_source};
use super::storage::{self, MtlStorage};
use super::weight_map::WeightMap;
use super::MetalWeights;
use crate::objc_util::{
    ns_error_string, responds_to, set_label, system_device, AutoreleasePool, MtlBuf, StrongId,
};
use crate::residency::{env_usize, CbStatus, CommandBuf, Ring};
use makepad_objc_sys::runtime::{nil, ObjcId, NO};
use makepad_objc_sys::{class, msg_send, sel, sel_impl};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

/// Dispatches a command buffer collects before the end of an op commits it, when
/// MAKEPAD_METAL_STREAM_OPS is unset. Apple GPUs preempt only between command buffers, and RV
/// draws on the same GPU.
pub const DEFAULT_STREAM_OPS: usize = 64;

const MTL_DATA_TYPE_FLOAT: u64 = 3;
const MTL_DATA_TYPE_INT: u64 = 29;
const MTL_DATA_TYPE_UINT: u64 = 33;
const MTL_DATA_TYPE_BOOL: u64 = 53;

/// `MTLCommandBufferStatus`: Completed and Error; below Completed it is still pending.
const MTL_COMMAND_BUFFER_STATUS_COMPLETED: u64 = 4;
const MTL_COMMAND_BUFFER_STATUS_ERROR: u64 = 5;

#[repr(C)]
#[derive(Clone, Copy)]
struct MtlSize {
    width: u64,
    height: u64,
    depth: u64,
}

/// A compiled pipeline and its limits.
struct Pipeline {
    state: StrongId,
    limits: KernelLimits,
    static_threadgroup_bytes: u32,
}

/// A pipeline as a launch uses it: the raw handle stays valid while the context holds the
/// pipeline, which is for the context's life.
#[derive(Clone, Copy)]
pub(crate) struct PipelineRef {
    pub state: ObjcId,
    pub limits: KernelLimits,
    pub static_threadgroup_bytes: u32,
}

/// The command buffer being encoded into.
struct OpenCb {
    cb: StrongId,
    encoder: StrongId,
    serial: u64,
    dispatches: usize,
    first_op: &'static str,
    last_op: &'static str,
}

impl OpenCb {
    /// The label errors and GPU captures show: the ops it holds.
    fn label(&self) -> String {
        let plural = if self.dispatches == 1 { "" } else { "es" };
        if self.first_op == self.last_op {
            format!("{} ({} dispatch{plural})", self.first_op, self.dispatches)
        } else {
            format!("{} .. {} ({} dispatch{plural})", self.first_op, self.last_op, self.dispatches)
        }
    }
}

/// A committed `MTLCommandBuffer`, as the ring sees it.
pub(crate) struct MetalCb(StrongId);

impl CommandBuf for MetalCb {
    fn status(&self) -> CbStatus {
        let status: u64 = unsafe { msg_send![self.0.as_id(), status] };
        match status {
            MTL_COMMAND_BUFFER_STATUS_COMPLETED => CbStatus::Completed,
            MTL_COMMAND_BUFFER_STATUS_ERROR => {
                let _pool = AutoreleasePool::new();
                let error: ObjcId = unsafe { msg_send![self.0.as_id(), error] };
                CbStatus::Failed(ns_error_string(error))
            }
            _ => CbStatus::Pending,
        }
    }

    fn wait_until_completed(&self) {
        unsafe {
            let _: () = msg_send![self.0.as_id(), waitUntilCompleted];
        }
    }

    fn gpu_ms(&self) -> f64 {
        let start: f64 = unsafe { msg_send![self.0.as_id(), GPUStartTime] };
        let end: f64 = unsafe { msg_send![self.0.as_id(), GPUEndTime] };
        if start.is_finite() && end.is_finite() && end > start {
            (end - start) * 1e3
        } else {
            0.0
        }
    }
}

/// One binding as the encoder sets it.
#[derive(Clone, Copy)]
pub(crate) enum Slot {
    /// `setBuffer:offset:atIndex:`.
    Buffer { id: ObjcId, offset: u64 },
    /// `setBytes:length:atIndex:` (at most 4096 bytes, never empty).
    Bytes { ptr: *const c_void, len: u64 },
}

/// The calling thread's stream.
pub(crate) struct StreamCtx {
    device: StrongId,
    queue: StrongId,
    max_threadgroup_bytes: u64,
    libraries: HashMap<&'static str, Result<StrongId, String>>,
    pipelines: HashMap<(&'static str, &'static str), Vec<(Box<[FnConstKey]>, Pipeline)>>,
    open: Option<OpenCb>,
    next_serial: u64,
    ring: Ring<MetalCb>,
    /// Weights by qualified key.
    pub(crate) weights: WeightMap<MetalWeights>,
    /// Bytes written into the weight cache since the counters were last reset.
    pub(crate) weight_bytes: u64,
    /// Bound at `Bind::Absent` and at an empty `Bind::Bytes`, so slot indices never shift.
    dummy: MtlBuf,
    /// The host binding's device copies of input tensors, by content id.
    pub(crate) mirror: Lru<Arc<MtlStorage>>,
    budget: usize,
    sync: bool,
    /// MAKEPAD_METAL_MATH_MODE, as `MTLMathMode`: `None` leaves the compiler's default (fast math
    /// off is always set).
    math_mode: Option<u64>,
}

impl StreamCtx {
    pub(crate) fn new() -> Result<StreamCtx, String> {
        let _pool = AutoreleasePool::new();
        let device = system_device()?;
        let queue: ObjcId = unsafe { msg_send![device.as_id(), newCommandQueue] };
        let queue = unsafe { StrongId::from_owned(queue) }.ok_or("newCommandQueue returned nil")?;
        set_label(queue.as_id(), "makepad metal stream");
        let dummy = MtlBuf::new_shared(device.as_id(), 16).ok_or("could not allocate the 16-byte dummy buffer")?;
        let max_threadgroup_bytes: u64 = unsafe { msg_send![device.as_id(), maxThreadgroupMemoryLength] };
        let ring = Ring::from_env();
        storage::attach(ring.clock().id(), &device);
        Ok(StreamCtx {
            device,
            queue,
            max_threadgroup_bytes,
            libraries: HashMap::new(),
            pipelines: HashMap::new(),
            open: None,
            next_serial: 1,
            ring,
            weights: WeightMap::new(),
            weight_bytes: 0,
            dummy,
            mirror: Lru::new(env_usize("MAKEPAD_METAL_MIRROR_CAP_MB", DEFAULT_MIRROR_CAP_MB).saturating_mul(1 << 20)),
            budget: env_usize("MAKEPAD_METAL_STREAM_OPS", DEFAULT_STREAM_OPS).max(1),
            sync: env_usize("MAKEPAD_METAL_STREAM_SYNC", 0) != 0,
            math_mode: math_mode_from_env(),
        })
    }

    pub(crate) fn device(&self) -> ObjcId {
        self.device.as_id()
    }

    pub(crate) fn dummy(&self) -> ObjcId {
        self.dummy.as_id()
    }

    /// `lib` compiled for this thread: at first use from its source (or MAKEPAD_METAL_KERNELS_SRC),
    /// fast math off; a failure is kept and returned for every later use.
    pub(crate) fn library(&mut self, lib: &'static KernelLib) -> Result<ObjcId, String> {
        if !self.libraries.contains_key(lib.name) {
            let dir = kernels_src_dir();
            let compiled = library_source(lib, dir.as_deref())
                .and_then(|source| compile_library(self.device.as_id(), &source, self.math_mode))
                .map_err(|e| format!("Metal kernel library '{}': {e}", lib.name));
            if let (Some(dir), Ok(_)) = (&dir, &compiled) {
                eprintln!("[metal stream] kernel library '{}' compiled from {}", lib.name, dir.display());
            }
            self.libraries.insert(lib.name, compiled);
        }
        match &self.libraries[lib.name] {
            Ok(library) => Ok(library.as_id()),
            Err(e) => Err(e.clone()),
        }
    }

    /// The pipeline for `kernel` specialised with `consts`, compiled at first use.
    pub(crate) fn pipeline(&mut self, kernel: &Kernel, consts: &[FnConst]) -> Result<PipelineRef, String> {
        let key = (kernel.lib.name, kernel.function);
        if let Some(found) = self.pipelines.get(&key).and_then(|entries| {
            entries.iter().find(|(keys, _)| consts_match(consts, keys)).map(|(_, pipeline)| pipeline)
        }) {
            return Ok(PipelineRef {
                state: found.state.as_id(),
                limits: found.limits,
                static_threadgroup_bytes: found.static_threadgroup_bytes,
            });
        }
        let library = self.library(kernel.lib)?;
        let pipeline = compile_pipeline(self.device.as_id(), library, kernel, consts)?;
        let found = PipelineRef {
            state: pipeline.state.as_id(),
            limits: pipeline.limits,
            static_threadgroup_bytes: pipeline.static_threadgroup_bytes,
        };
        let keys: Box<[FnConstKey]> = consts.iter().map(|c| c.key()).collect();
        self.pipelines.entry(key).or_default().push((keys, pipeline));
        Ok(found)
    }

    pub(crate) fn max_threadgroup_bytes(&self) -> u64 {
        self.max_threadgroup_bytes
    }

    /// The open command buffer's serial encoder, opening both if none is: one more dispatch of
    /// `op`. Returns the encoder and the serial its work runs under.
    pub(crate) fn begin_dispatch(&mut self, op: &'static str) -> Result<(ObjcId, u64), String> {
        if self.open.is_none() {
            let cb: ObjcId = unsafe { msg_send![self.queue.as_id(), commandBuffer] };
            let cb = unsafe { StrongId::retain(cb) }.ok_or("commandBuffer returned nil")?;
            let encoder: ObjcId = unsafe { msg_send![cb.as_id(), computeCommandEncoder] };
            let encoder = unsafe { StrongId::retain(encoder) }.ok_or("computeCommandEncoder returned nil")?;
            let serial = self.next_serial;
            self.next_serial += 1;
            self.open = Some(OpenCb { cb, encoder, serial, dispatches: 0, first_op: op, last_op: op });
        }
        let open = self.open.as_mut().expect("opened above");
        open.dispatches += 1;
        open.last_op = op;
        Ok((open.encoder.as_id(), open.serial))
    }

    /// Ends the open command buffer's encoder and commits it, after making room in the ring (it
    /// waits on the oldest when the ring is full). Nothing to do with none open. Never waits on
    /// this one.
    pub(crate) fn commit_open(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        let label = open.label();
        unsafe {
            let _: () = msg_send![open.encoder.as_id(), endEncoding];
        }
        set_label(open.cb.as_id(), &label);
        self.ring.make_room();
        unsafe {
            let _: () = msg_send![open.cb.as_id(), commit];
        }
        self.ring.push(open.serial, MetalCb(open.cb), label);
    }

    /// Waits until the command buffer with `serial` (committing it if it is the open one) and
    /// every one before it have finished; returns the first failure retired since the last wait.
    pub(crate) fn wait(&mut self, serial: u64) -> Result<(), String> {
        if self.open.as_ref().is_some_and(|open| open.serial <= serial) {
            self.commit_open();
        }
        self.ring.wait(serial)
    }

    /// Commits the open command buffer and waits for everything committed.
    pub(crate) fn wait_all(&mut self) -> Result<(), String> {
        self.commit_open();
        self.ring.wait_all()
    }

    /// Retires the finished command buffers without blocking; returns the completed watermark.
    pub(crate) fn retire(&mut self) -> u64 {
        self.ring.retire();
        self.ring.completed()
    }

    /// The end of a `with_metal_stream` call: with MAKEPAD_METAL_STREAM_SYNC, commit and wait;
    /// otherwise commit once the open buffer holds the budget's dispatches.
    pub(crate) fn end_op(&mut self) -> Result<(), String> {
        if self.sync {
            return self.wait_all();
        }
        if self.open.as_ref().is_some_and(|open| open.dispatches >= self.budget) {
            self.commit_open();
        }
        Ok(())
    }

    /// The ring's counters, zeroed afterwards when `reset`.
    pub(crate) fn ring_stats(&mut self, reset: bool) -> crate::residency::RingStats {
        self.ring.stats(reset)
    }

    /// The device's memory: (`recommendedMaxWorkingSetSize`, `currentAllocatedSize`).
    pub(crate) fn memory(&self) -> (u64, u64) {
        let total: u64 = unsafe { msg_send![self.device.as_id(), recommendedMaxWorkingSetSize] };
        let allocated: u64 = unsafe { msg_send![self.device.as_id(), currentAllocatedSize] };
        (total, allocated)
    }
}

impl Drop for StreamCtx {
    /// Commits what is open (an encoder released without `endEncoding` trips Metal's validation)
    /// and waits for everything, before the clock leaves the registry with the ring.
    fn drop(&mut self) {
        // Only when something is pending: this runs among the thread's exit destructors, where
        // fewer Objective-C calls are better.
        if self.open.is_some() || self.ring.in_flight() > 0 {
            let _pool = AutoreleasePool::new();
            if let Err(e) = self.wait_all() {
                eprintln!("[metal stream] at thread exit: {e}");
            }
        }
        let copies = self.mirror.clear();
        drop(copies);
        self.weights.clear();
    }
}

/// Encodes one dispatch on `encoder`: the pipeline, the params at index 0 (none when empty), the
/// slots at 1, 2, ..., the grid's threadgroup memory, and the threadgroups.
pub(crate) fn encode_dispatch(
    encoder: ObjcId,
    pipeline: ObjcId,
    params: *const c_void,
    params_len: usize,
    slots: &[Slot],
    grid: &Grid,
) {
    unsafe {
        let _: () = msg_send![encoder, setComputePipelineState: pipeline];
        if params_len > 0 {
            let _: () = msg_send![encoder, setBytes: params length: params_len as u64 atIndex: 0u64];
        }
        for (index, slot) in slots.iter().enumerate() {
            let index = index as u64 + 1;
            match *slot {
                Slot::Buffer { id, offset } => {
                    let _: () = msg_send![encoder, setBuffer: id offset: offset atIndex: index];
                }
                Slot::Bytes { ptr, len } => {
                    let _: () = msg_send![encoder, setBytes: ptr length: len atIndex: index];
                }
            }
        }
        for (index, &bytes) in grid.threadgroup_bytes.iter().enumerate() {
            if bytes > 0 {
                let _: () = msg_send![encoder, setThreadgroupMemoryLength: bytes as u64 atIndex: index as u64];
            }
        }
        let groups = MtlSize {
            width: grid.groups[0] as u64,
            height: grid.groups[1] as u64,
            depth: grid.groups[2] as u64,
        };
        let threads = MtlSize {
            width: grid.threads[0] as u64,
            height: grid.threads[1] as u64,
            depth: grid.threads[2] as u64,
        };
        let _: () = msg_send![encoder, dispatchThreadgroups: groups threadsPerThreadgroup: threads];
    }
}

/// MAKEPAD_METAL_MATH_MODE=safe|relaxed|fast as `MTLMathMode` (0, 1, 2); anything else leaves
/// the default. Step 7's probe compares the modes.
fn math_mode_from_env() -> Option<u64> {
    match std::env::var("MAKEPAD_METAL_MATH_MODE").ok()?.trim().to_ascii_lowercase().as_str() {
        "safe" => Some(0),
        "relaxed" => Some(1),
        "fast" => Some(2),
        _ => None,
    }
}

/// Compiles `source` with fast math off (and `math_mode` where the OS has `setMathMode:`).
fn compile_library(device: ObjcId, source: &str, math_mode: Option<u64>) -> Result<StrongId, String> {
    let _pool = AutoreleasePool::new();
    let options: ObjcId = unsafe { msg_send![class!(MTLCompileOptions), new] };
    let options = unsafe { StrongId::from_owned(options) }.ok_or("MTLCompileOptions new returned nil")?;
    unsafe {
        let _: () = msg_send![options.as_id(), setFastMathEnabled: NO];
    }
    if let Some(mode) = math_mode {
        if responds_to(options.as_id(), sel!(setMathMode:)) {
            unsafe {
                let _: () = msg_send![options.as_id(), setMathMode: mode];
            }
        } else {
            eprintln!("[metal stream] MAKEPAD_METAL_MATH_MODE ignored: this macOS has no MTLCompileOptions.mathMode");
        }
    }
    let source = crate::objc_util::ns_string(source).ok_or("could not make an NSString of the source")?;
    let mut error: ObjcId = nil;
    let library: ObjcId = unsafe {
        msg_send![device, newLibraryWithSource: source.as_id() options: options.as_id() error: &mut error]
    };
    unsafe { StrongId::from_owned(library) }.ok_or_else(|| ns_error_string(error))
}

/// A compute pipeline for `kernel.function` in `library`, specialised with `consts`.
fn compile_pipeline(device: ObjcId, library: ObjcId, kernel: &Kernel, consts: &[FnConst]) -> Result<Pipeline, String> {
    let _pool = AutoreleasePool::new();
    let name = crate::objc_util::ns_string(kernel.function).ok_or("could not make an NSString of the function name")?;
    let mut error: ObjcId = nil;
    let function: ObjcId = if consts.is_empty() {
        unsafe { msg_send![library, newFunctionWithName: name.as_id()] }
    } else {
        let values: ObjcId = unsafe { msg_send![class!(MTLFunctionConstantValues), new] };
        let values = unsafe { StrongId::from_owned(values) }.ok_or("MTLFunctionConstantValues new returned nil")?;
        for constant in consts {
            set_function_constant(values.as_id(), *constant);
        }
        unsafe {
            msg_send![library, newFunctionWithName: name.as_id() constantValues: values.as_id() error: &mut error]
        }
    };
    let function = unsafe { StrongId::from_owned(function) }.ok_or_else(|| {
        let why = if error.is_null() { "no such function".to_string() } else { ns_error_string(error) };
        format!("Metal kernel '{}' in library '{}': {why}", kernel.function, kernel.lib.name)
    })?;
    let mut error: ObjcId = nil;
    let state: ObjcId = unsafe {
        msg_send![device, newComputePipelineStateWithFunction: function.as_id() error: &mut error]
    };
    let state = unsafe { StrongId::from_owned(state) }.ok_or_else(|| {
        format!(
            "Metal pipeline for '{}' in library '{}': {}",
            kernel.function,
            kernel.lib.name,
            ns_error_string(error)
        )
    })?;
    let max_threads: u64 = unsafe { msg_send![state.as_id(), maxTotalThreadsPerThreadgroup] };
    let simd_width: u64 = unsafe { msg_send![state.as_id(), threadExecutionWidth] };
    let static_threadgroup: u64 = unsafe { msg_send![state.as_id(), staticThreadgroupMemoryLength] };
    Ok(Pipeline {
        state,
        limits: KernelLimits { max_threads: max_threads as u32, simd_width: simd_width as u32 },
        static_threadgroup_bytes: static_threadgroup as u32,
    })
}

fn set_function_constant(values: ObjcId, constant: FnConst) {
    unsafe {
        match constant {
            FnConst::Bool(index, value) => {
                let value = value as u8;
                let _: () = msg_send![
                    values,
                    setConstantValue: &value as *const u8 as *const c_void
                    type: MTL_DATA_TYPE_BOOL
                    atIndex: index as u64
                ];
            }
            FnConst::I32(index, value) => {
                let _: () = msg_send![
                    values,
                    setConstantValue: &value as *const i32 as *const c_void
                    type: MTL_DATA_TYPE_INT
                    atIndex: index as u64
                ];
            }
            FnConst::U32(index, value) => {
                let _: () = msg_send![
                    values,
                    setConstantValue: &value as *const u32 as *const c_void
                    type: MTL_DATA_TYPE_UINT
                    atIndex: index as u64
                ];
            }
            FnConst::F32(index, value) => {
                let _: () = msg_send![
                    values,
                    setConstantValue: &value as *const f32 as *const c_void
                    type: MTL_DATA_TYPE_FLOAT
                    atIndex: index as u64
                ];
            }
        }
    }
}
