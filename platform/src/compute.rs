//! Compute kernels, the app-facing side (milestone 16 of the fork's compute work): a kernel is
//! declared in Splash as a draw shader is (`mod.draw.ComputeKernel{ workgroup_size: 64 ... main:
//! fn() { ... } }`), compiled when it is applied for the dialect the HAL runs (HLSL on D3D11,
//! MSL on Metal once its printer exists), and its pipeline made with the draw shaders'
//! (`Cx::fb_compile_kernels`). Its bindings (`KernelBindings`) say where each param, texture,
//! image and buffer goes, by name, for the dialect it was compiled for.

use crate::{
    cx::Cx,
    makepad_live_id::*,
    makepad_script::shader_backend::ShaderBackend,
    makepad_script::shader_ir_compute::{compile_compute_kernel_for, KernelBindings, KernelCode},
    makepad_script::*,
    script::vm::*,
};
use std::collections::{BTreeSet, HashMap};

/// A kernel as Cx keeps it: an index into `Cx::compute_kernels`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeKernelId(pub(crate) usize);

/// A compute kernel, declared in Splash; compiled when applied. `#[live] kernel: ComputeKernel`
/// on a widget, its Splash object a `mod.draw.ComputeKernel{ ... }`.
#[derive(Script)]
pub struct ComputeKernel {
    #[rust]
    pub kernel_id: Option<ComputeKernelId>,
}

impl ScriptHook for ComputeKernel {
    fn on_after_apply(&mut self, vm: &mut ScriptVm, apply: &Apply, _scope: &mut Scope, value: ScriptValue) {
        if apply.is_default() || apply.is_animate() {
            return;
        }
        if let Some(object) = value.as_object() {
            self.kernel_id = Cx::compile_kernel(vm, object);
        }
    }
}

impl ComputeKernel {
    /// Where its params, textures, images and buffers go, as compiled for the running backend;
    /// `None` before it is compiled or when it failed.
    pub fn bindings<'a>(&self, cx: &'a Cx) -> Option<&'a KernelBindings> {
        cx.compute_kernels.get(self.kernel_id?)?.code.as_ref()?.bindings()
    }

    /// Its workgroup size, as declared (`workgroup_size`, `_y`, `_z`).
    pub fn workgroup(&self, cx: &Cx) -> Option<[u32; 3]> {
        self.bindings(cx).map(|b| b.workgroup)
    }

    /// Whether the HAL has made its pipeline (dispatches wait for it until then).
    pub fn is_ready(&self, cx: &Cx) -> bool {
        let Some(k) = self.kernel_id.and_then(|id| cx.compute_kernels.get(id)) else { return false };
        k.pipeline.is_some_and(|p| cx.hal.frames.pipeline_is_ready(p))
    }

    /// Why it does not run: it did not compile, or its pipeline failed.
    pub fn error<'a>(&self, cx: &'a Cx) -> Option<&'a str> {
        cx.compute_kernels.get(self.kernel_id?)?.error.as_deref()
    }
}

/// One kernel compiled for the running backend.
pub struct CxComputeKernel {
    /// its type's name in Splash, for messages
    pub debug_id: LiveId,
    pub code: Option<KernelCode>,
    /// its pipeline, an index into the frame builder's pipelines, once made
    pub pipeline: Option<usize>,
    pub error: Option<String>,
}

/// Every kernel compiled: by the object it was compiled from (with its heap), and by the code
/// it compiled to, so the same kernel applied twice shares one pipeline.
#[derive(Default)]
pub struct CxComputeKernels {
    pub kernels: Vec<CxComputeKernel>,
    /// kernels whose pipeline is still to be made (`Cx::fb_compile_kernels`)
    pub compile_set: BTreeSet<usize>,
    cache_object: HashMap<(usize, ScriptObject), usize>,
    cache_code: HashMap<u64, usize>,
    /// why dispatches did not run, each said once
    pub(crate) reported: std::collections::HashSet<String>,
}

impl CxComputeKernels {
    pub fn get(&self, id: ComputeKernelId) -> Option<&CxComputeKernel> {
        self.kernels.get(id.0)
    }

    /// Forgets every compiled object (a live reload): the code cache keeps their pipelines.
    pub fn forget_objects(&mut self) {
        self.cache_object.clear();
    }
}

/// The shader backend a kernel compiles for under HAL dialect `dialect`, and whether it is
/// Vulkan's. Headless (no dialect) compiles HLSL, the slot-bound form, which it validates and
/// does not run.
pub(crate) fn kernel_backend(dialect: u32) -> (ShaderBackend, bool) {
    match dialect {
        makepad_hal::dialect::HLSL | makepad_hal::dialect::NONE => (ShaderBackend::Hlsl, false),
        makepad_hal::dialect::MSL => (ShaderBackend::Metal, false),
        makepad_hal::dialect::SPIRV => (ShaderBackend::Glsl, true),
        _ => (ShaderBackend::Glsl, false),
    }
}

/// A hash of what a kernel compiled to, for sharing its pipeline.
fn code_key(code: &KernelCode) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match code {
        KernelCode::Hlsl(k) => {
            0u8.hash(&mut h);
            k.source.hash(&mut h);
        }
        KernelCode::Spirv(k) => {
            1u8.hash(&mut h);
            k.spirv.hash(&mut h);
        }
    }
    h.finish()
}

impl Cx {
    /// Compiles kernel `object` for the backend the HAL runs, or finds it compiled; its id, or
    /// `None` when it did not compile (logged, with the reason).
    pub(crate) fn compile_kernel(vm: &mut ScriptVm, object: ScriptObject) -> Option<ComputeKernelId> {
        let heap_key = vm.heap_key();
        if let Some(&index) = vm.host.cx().compute_kernels.cache_object.get(&(heap_key, object)) {
            return Some(ComputeKernelId(index));
        }
        let debug_id = vm.heap().object_type_name_in_chain(object).unwrap_or(LiveId(0));
        let dialect = vm.host.cx().hal.info.map_or(makepad_hal::dialect::NONE, |i| i.dialect);
        let (backend, vulkan) = kernel_backend(dialect);
        let compiled = compile_compute_kernel_for(vm, object, backend, vulkan);
        let cx = vm.host.cx_mut();
        let kernels = &mut cx.compute_kernels;
        let index = match compiled {
            Ok(code) => {
                let key = code_key(&code);
                match kernels.cache_code.get(&key) {
                    Some(&index) => index,
                    None => {
                        let index = kernels.kernels.len();
                        kernels.kernels.push(CxComputeKernel { debug_id, code: Some(code), pipeline: None, error: None });
                        kernels.cache_code.insert(key, index);
                        kernels.compile_set.insert(index);
                        index
                    }
                }
            }
            Err(error) => {
                let error = format!("compute kernel '{debug_id}' did not compile and will NOT run: {error}");
                crate::shader_error::note(error.clone());
                crate::error!("{}", error);
                let index = kernels.kernels.len();
                kernels.kernels.push(CxComputeKernel { debug_id, code: None, pipeline: None, error: Some(error) });
                index
            }
        };
        kernels.cache_object.insert((heap_key, object), index);
        Some(ComputeKernelId(index))
    }
}

/// A value for one of a kernel's params (`shader.uniform(type)`), by its type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KernelParam {
    F32(f32),
    Vec2f([f32; 2]),
    Vec3f([f32; 3]),
    Vec4f([f32; 4]),
    U32(u32),
    Vec2u([u32; 2]),
    Vec3u([u32; 3]),
    Vec4u([u32; 4]),
    I32(i32),
    Vec2i([i32; 2]),
    Vec4i([i32; 4]),
}

impl KernelParam {
    /// Its scalar type and lane count, as a kernel's param slot says them.
    pub(crate) fn kind(&self) -> (crate::makepad_script::shader_ir::IrScalar, u32) {
        use crate::makepad_script::shader_ir::IrScalar::*;
        match self {
            KernelParam::F32(_) => (F32, 1),
            KernelParam::Vec2f(_) => (F32, 2),
            KernelParam::Vec3f(_) => (F32, 3),
            KernelParam::Vec4f(_) => (F32, 4),
            KernelParam::U32(_) => (U32, 1),
            KernelParam::Vec2u(_) => (U32, 2),
            KernelParam::Vec3u(_) => (U32, 3),
            KernelParam::Vec4u(_) => (U32, 4),
            KernelParam::I32(_) => (I32, 1),
            KernelParam::Vec2i(_) => (I32, 2),
            KernelParam::Vec4i(_) => (I32, 4),
        }
    }

    /// Its bytes, little-endian, as the parameter block holds them.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        fn f(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
        fn u(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
        fn i(v: &[i32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
        match self {
            KernelParam::F32(v) => f(&[*v]),
            KernelParam::Vec2f(v) => f(v),
            KernelParam::Vec3f(v) => f(v),
            KernelParam::Vec4f(v) => f(v),
            KernelParam::U32(v) => u(&[*v]),
            KernelParam::Vec2u(v) => u(v),
            KernelParam::Vec3u(v) => u(v),
            KernelParam::Vec4u(v) => u(v),
            KernelParam::I32(v) => i(&[*v]),
            KernelParam::Vec2i(v) => i(v),
            KernelParam::Vec4i(v) => i(v),
        }
    }
}

impl From<f32> for KernelParam { fn from(v: f32) -> Self { KernelParam::F32(v) } }
impl From<u32> for KernelParam { fn from(v: u32) -> Self { KernelParam::U32(v) } }
impl From<i32> for KernelParam { fn from(v: i32) -> Self { KernelParam::I32(v) } }
impl From<[f32; 2]> for KernelParam { fn from(v: [f32; 2]) -> Self { KernelParam::Vec2f(v) } }
impl From<[f32; 3]> for KernelParam { fn from(v: [f32; 3]) -> Self { KernelParam::Vec3f(v) } }
impl From<[f32; 4]> for KernelParam { fn from(v: [f32; 4]) -> Self { KernelParam::Vec4f(v) } }
impl From<[u32; 2]> for KernelParam { fn from(v: [u32; 2]) -> Self { KernelParam::Vec2u(v) } }
impl From<[u32; 3]> for KernelParam { fn from(v: [u32; 3]) -> Self { KernelParam::Vec3u(v) } }
impl From<[u32; 4]> for KernelParam { fn from(v: [u32; 4]) -> Self { KernelParam::Vec4u(v) } }
impl From<[i32; 2]> for KernelParam { fn from(v: [i32; 2]) -> Self { KernelParam::Vec2i(v) } }
impl From<[i32; 4]> for KernelParam { fn from(v: [i32; 4]) -> Self { KernelParam::Vec4i(v) } }
impl From<crate::makepad_math::Vec2f> for KernelParam { fn from(v: crate::makepad_math::Vec2f) -> Self { KernelParam::Vec2f([v.x, v.y]) } }
impl From<crate::makepad_math::Vec4f> for KernelParam { fn from(v: crate::makepad_math::Vec4f) -> Self { KernelParam::Vec4f([v.x, v.y, v.z, v.w]) } }

/// How many workgroups a dispatch runs: given as is, or as threads (each axis rounded up to
/// whole workgroups of the kernel's size when the frame is built).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchSize {
    Groups([u32; 3]),
    Threads([u32; 3]),
}

/// One dispatch of a kernel: its params, sampled textures, written images and buffers by the
/// names the kernel declares them by, and how many groups run. Recorded on the pass that reads
/// what it writes (`Cx::dispatch`); names are resolved when the frame is built, against the
/// kernel as compiled for the running backend.
#[derive(Clone, Debug)]
pub struct ComputeDispatch {
    pub(crate) kernel: Option<ComputeKernelId>,
    pub(crate) size: DispatchSize,
    pub(crate) params: Vec<(LiveId, KernelParam)>,
    pub(crate) textures: Vec<(LiveId, crate::texture::Texture)>,
    pub(crate) images: Vec<(LiveId, crate::texture::Texture)>,
    pub(crate) buffers: Vec<(LiveId, crate::storage_buffer::StorageBuffer)>,
}

impl ComputeDispatch {
    pub fn new(kernel: &ComputeKernel) -> Self {
        ComputeDispatch { kernel: kernel.kernel_id, size: DispatchSize::Groups([1, 1, 1]), params: Vec::new(), textures: Vec::new(), images: Vec::new(), buffers: Vec::new() }
    }
    /// Sets param `name` (`shader.uniform(type)`); a param never set reads 0.
    pub fn param(mut self, name: LiveId, value: impl Into<KernelParam>) -> Self {
        self.params.retain(|(n, _)| *n != name);
        self.params.push((name, value.into()));
        self
    }
    /// Binds texture `name` (`shader.texture_2d(float)`), sampled.
    pub fn texture(mut self, name: LiveId, texture: &crate::texture::Texture) -> Self {
        self.textures.push((name, texture.clone()));
        self
    }
    /// Binds storage image `name` (`shader.storage_2d(type)`), written: a `Texture::new_storage`.
    pub fn image(mut self, name: LiveId, texture: &crate::texture::Texture) -> Self {
        self.images.push((name, texture.clone()));
        self
    }
    /// Binds buffer `name` (`shader.buffer` read, `shader.buffer_rw` written).
    pub fn buffer(mut self, name: LiveId, buffer: &crate::storage_buffer::StorageBuffer) -> Self {
        self.buffers.push((name, buffer.clone()));
        self
    }
    /// Runs `x` by `y` by `z` workgroups.
    pub fn groups(mut self, x: u32, y: u32, z: u32) -> Self {
        self.size = DispatchSize::Groups([x, y, z]);
        self
    }
    /// Runs at least `x` by `y` by `z` threads: whole workgroups of the kernel's size.
    pub fn threads(mut self, x: u32, y: u32, z: u32) -> Self {
        self.size = DispatchSize::Threads([x, y, z]);
        self
    }
}

/// The most dispatches a pass keeps waiting (one that never paints would grow without end): past
/// it the oldest go, with a warning.
const MOST_QUEUED: usize = 256;

impl Cx {
    /// Runs `dispatch` before pass `pass` draws, in the order recorded, once: what it writes is
    /// there for that pass and the passes that read it after (its parents). Marks the pass to
    /// paint. A per-frame simulation records its steps each frame.
    pub fn dispatch(&mut self, pass: crate::draw_pass::DrawPassId, dispatch: ComputeDispatch) {
        let queue = &mut self.passes[pass].dispatches;
        if queue.len() >= MOST_QUEUED {
            crate::warning!("a pass has {MOST_QUEUED} compute dispatches waiting; the oldest is dropped (does the pass paint?)");
            queue.remove(0);
        }
        queue.push(dispatch);
        self.passes[pass].paint_dirty = true;
    }

    /// How many dispatches wait to run before pass `pass` draws.
    pub fn dispatches_pending(&self, pass: crate::draw_pass::DrawPassId) -> usize {
        self.passes[pass].dispatches.len()
    }

    /// Drops the dispatches waiting on pass `pass`.
    pub fn clear_dispatches(&mut self, pass: crate::draw_pass::DrawPassId) {
        self.passes[pass].dispatches.clear();
    }
}
