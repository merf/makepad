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
