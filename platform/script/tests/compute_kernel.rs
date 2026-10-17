//! Splash compute kernels compile to SPIR-V that validates (spirv-val with
//! Vulkan 1.2 rules, when the tool is installed), with the push-constant
//! layout their host binds them by.
use makepad_script::shader_ir_compute::*;
use makepad_script::*;

fn kernel(body: &str) -> Result<CompiledKernel, String> {
    let host = Box::leak(Box::new(ScriptVmHost::new(0i32, ())));
    let mut vm = ScriptVm { host: host.host_context(), bx: Box::new(ScriptVmBase::new()) };
    vm.bx.captured_errors = Some(Vec::new());
    let code = format!("use mod.pod.*\nuse mod.math.*\nuse mod.shader\nlet k = {{\n{body}\n}}\nk");
    let value = vm.with_instruction_limit(50_000_000, |vm| vm.eval(ScriptMod::new("", "", "x", 0, 0, &code, &[])));
    let errors = vm.take_errors();
    assert!(errors.is_empty(), "{errors:?}");
    let obj = value.as_object().expect("kernel object");
    let r = compile_compute_kernel(&mut vm, obj);
    let errors = vm.take_errors();
    if !errors.is_empty() {
        return Err(format!("{errors:?}"));
    }
    r
}

fn validate(k: &CompiledKernel) {
    let mut bytes = Vec::new();
    for w in &k.spirv {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let path = std::env::temp_dir().join(format!("makepad-kernel-{}-{}.spv", std::process::id(), k.spirv.len()));
    std::fs::write(&path, &bytes).unwrap();
    // MAKEPAD_KERNEL_DUMP=dir keeps the modules for spirv-dis.
    if let Ok(dir) = std::env::var("MAKEPAD_KERNEL_DUMP") {
        std::fs::write(std::path::Path::new(&dir).join(format!("kernel-{}.spv", k.spirv.len())), &bytes).unwrap();
    }
    let out = std::process::Command::new("spirv-val").arg("--target-env").arg("vulkan1.2").arg(&path).output();
    if let Ok(o) = out {
        assert!(o.status.success(), "spirv-val: {}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    }
}

#[test]
fn scale_kernel_compiles_and_validates() {
    let k = kernel(
        "workgroup_size: 64\n\
         src: shader.buffer(f32)\n\
         dst: shader.buffer_rw(f32)\n\
         n: shader.uniform(u32)\n\
         scale: shader.uniform(f32)\n\
         main: fn() {\n\
             let i = global_id().x\n\
             if i < self.n { self.dst[i] = self.src[i] * self.scale }\n\
         }",
    )
    .unwrap();
    assert_eq!(k.layout.workgroup, [64, 1, 1]);
    let names: Vec<String> = k.layout.slots.iter().map(|s| format!("{}@{}", s.name, s.offset)).collect();
    assert_eq!(names, ["dst@0", "src@8", "n@16", "scale@20"].map(String::from).to_vec(), "{:?}", k.layout);
    assert_eq!(k.layout.push_size, 24);
    validate(&k);
}

#[test]
fn row_sum_with_workgroup_memory_and_subgroups_validates() {
    let k = kernel(
        "workgroup_size: 128\n\
         x: shader.buffer(f32)\n\
         y: shader.buffer_rw(f32)\n\
         cols: shader.uniform(u32)\n\
         partial: shader.workgroup(f32, 32)\n\
         main: fn() {\n\
             let row = group_id().x\n\
             let t = local_id().x\n\
             var acc = 0.0\n\
             var c = t\n\
             while c < self.cols {\n\
                 acc += self.x[row * self.cols + c]\n\
                 c += self.workgroup_size\n\
             }\n\
             acc = subgroup_add(acc)\n\
             if subgroup_lane() == 0 { self.partial[subgroup_id()] = acc }\n\
             barrier()\n\
             if t == 0 {\n\
                 var s = 0.0\n\
                 for k in 0..self.workgroup_size / subgroup_size() { s += self.partial[k] }\n\
                 self.y[row] = s\n\
             }\n\
         }",
    )
    .unwrap();
    validate(&k);
}

#[test]
fn packed_words_unpack_and_dot() {
    let k = kernel(
        "workgroup_size: 32\n\
         q: shader.buffer(u32)\n\
         h: shader.buffer(u32)\n\
         out: shader.buffer_rw(f32)\n\
         main: fn() {\n\
             let i = global_id().x\n\
             let w = self.q[i]\n\
             let lo = extract_bits(w, 0, 4)\n\
             let d = unpack_half2(self.h[i]).x\n\
             self.out[i] = d * float(lo) + float(dot4_u8(w, w))\n\
         }",
    )
    .unwrap();
    validate(&k);
}

#[test]
fn writing_a_read_only_buffer_is_refused() {
    let r = kernel(
        "src: shader.buffer(f32)\n\
         main: fn() { self.src[0] = 1.0 }",
    );
    assert!(r.is_err(), "{r:?}");
}

#[test]
fn cooperative_matrices_validate_and_reject_invalid_operands() {
    let host = Box::leak(Box::new(ScriptVmHost::new(0i32, ())));
    let mut vm = ScriptVm { host: host.host_context(), bx: Box::new(ScriptVmBase::new()) };
    vm.bx.captured_errors = Some(Vec::new());
    let source = include_str!("../../../libs/ai/vulkan/kernels/cooperative_test.splash");
    let value = vm.eval(ScriptMod::new("", "", "cooperative_test.splash", 0, 0, source, &[]));
    assert!(vm.take_errors().is_empty());
    for field in vm.bx.heap.map_entries_ordered(value.as_object().unwrap()) {
        let name = format!("{}", field.key.as_id().unwrap());
        let compiled = compile_compute_kernel(&mut vm, field.value.as_object().unwrap());
        let errors = vm.take_errors();
        if name.starts_with("reject_") {
            assert!(compiled.is_err() || !errors.is_empty(), "{name} compiled");
        } else {
            assert!(errors.is_empty(), "{name}: {errors:?}");
            let k = compiled.unwrap_or_else(|e| panic!("{name}: {e}"));
            validate(&k);
            assert!(!k.layout.cooperative_matrices.is_empty());
            if name == "float_matrices" {
                let mut output = shader_output::ShaderOutput::default();
                output.mode = shader::ShaderMode::Vertex;
                let obj = field.value.as_object().unwrap();
                let main = vm.bx.heap.object_method(obj, makepad_live_id::id!(main).into(), NoTrap).as_object().unwrap();
                shader_lower::IrFnCompiler::compile_shader_def(&mut vm, &mut output, NoTrap, makepad_live_id::id!(main), main, shader::ShaderType::IoSelf(obj), vec![]);
                assert!(output.has_errors && output.compiled.errors.iter().any(|e| e.contains("cooperative matrices exist only in compute kernels")));
                vm.take_errors();
            }
        }
    }
}

// ---- HLSL (D3D11): slots and a constant buffer instead of device addresses

fn eval_kernel(body: &str) -> (ScriptVm<'static>, ScriptObject) {
    eval_kernel_after("", body)
}

/// The kernel `body` after `prelude` (the pod structs its buffers hold).
fn eval_kernel_after(prelude: &str, body: &str) -> (ScriptVm<'static>, ScriptObject) {
    let host = Box::leak(Box::new(ScriptVmHost::new(0i32, ())));
    let mut vm = ScriptVm { host: host.host_context(), bx: Box::new(ScriptVmBase::new()) };
    vm.bx.captured_errors = Some(Vec::new());
    let code = format!("use mod.pod.*\nuse mod.math.*\nuse mod.shader\n{prelude}\nlet k = {{\n{body}\n}}\nk");
    let value = vm.with_instruction_limit(50_000_000, |vm| vm.eval(ScriptMod::new("", "", "x", 0, 0, &code, &[])));
    let errors = vm.take_errors();
    assert!(errors.is_empty(), "{errors:?}");
    let obj = value.as_object().expect("kernel object");
    (vm, obj)
}

fn hlsl_kernel(body: &str) -> Result<HlslKernel, String> {
    hlsl_kernel_after("", body)
}

fn hlsl_kernel_after(prelude: &str, body: &str) -> Result<HlslKernel, String> {
    let (mut vm, obj) = eval_kernel_after(prelude, body);
    let r = compile_compute_kernel_hlsl(&mut vm, obj);
    let errors = vm.take_errors();
    if !errors.is_empty() {
        return Err(format!("{errors:?}"));
    }
    r
}

/// FXC from the Windows SDK, when it is installed (`MAKEPAD_FXC` names another).
fn fxc() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("MAKEPAD_FXC") {
        return Some(p.into());
    }
    let kits = std::path::Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut found: Vec<_> = std::fs::read_dir(kits).ok()?.filter_map(|e| Some(e.ok()?.path().join("x64").join("fxc.exe"))).filter(|p| p.exists()).collect();
    found.sort();
    found.pop()
}

/// The source compiles as D3D11 does it (`cs_5_0`, entry `kernel_main`), when FXC is at hand.
fn validate_hlsl(k: &HlslKernel) {
    let Some(fxc) = fxc() else { return };
    let path = std::env::temp_dir().join(format!("makepad-kernel-{}-{}.hlsl", std::process::id(), k.source.len()));
    std::fs::write(&path, &k.source).unwrap();
    let out = std::process::Command::new(fxc).args(["/nologo", "/T", "cs_5_0", "/E", "kernel_main", "/Fo"]).arg(path.with_extension("dxbc")).arg(&path).output().unwrap();
    let numbered: String = k.source.lines().enumerate().map(|(i, l)| format!("{}:{l}\n", i + 1)).collect();
    assert!(out.status.success(), "fxc: {}{}\n{numbered}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}

#[test]
fn hlsl_scale_kernel_binds_buffers_to_slots_and_params_to_a_constant_buffer() {
    let k = hlsl_kernel(
        "workgroup_size: 64\n\
         src: shader.buffer(f32)\n\
         dst: shader.buffer_rw(f32)\n\
         n: shader.uniform(u32)\n\
         scale: shader.uniform(f32)\n\
         offset: shader.uniform(vec2f)\n\
         tint: shader.uniform(vec4f)\n\
         main: fn() {\n\
             let i = global_id().x\n\
             if i < self.n { self.dst[i] = self.src[i] * self.scale + self.offset.x + self.tint.w }\n\
         }",
    )
    .unwrap();
    let b = &k.bindings;
    assert_eq!(b.workgroup, [64, 1, 1]);
    // the buffers in the order the kernel first uses them, each in its own register file
    let buffers: Vec<String> = b.buffers.iter().map(|x| format!("{}:{}{}/{}", x.name, if x.writable { 'u' } else { 't' }, x.register, x.stride)).collect();
    assert_eq!(buffers, ["dst:u0/4", "src:t0/4"]);
    // params at natural alignment, which is the constant buffer's packing
    let params: Vec<String> = b.params.iter().map(|s| format!("{}@{}", s.name, s.offset)).collect();
    assert_eq!(params, ["n@0", "scale@4", "offset@8", "tint@16"]);
    assert_eq!(b.params_size, 32);
    assert!(k.source.contains("[numthreads(64, 1, 1)]"), "{}", k.source);
    assert!(k.source.contains("RWStructuredBuffer<float> kernel_dst : register(u0);"));
    assert!(k.source.contains("StructuredBuffer<float> kernel_src : register(t0);"));
    assert!(k.source.contains("packoffset(c1.x)"), "tint starts the second row");
    validate_hlsl(&k);
}

#[test]
fn hlsl_kernel_writes_a_storage_image_and_reads_a_texture() {
    let k = hlsl_kernel(
        "workgroup_size: 8\n\
         workgroup_size_y: 8\n\
         src: shader.texture_2d(float)\n\
         out: shader.storage_2d(vec4f)\n\
         counts: shader.buffer_rw(u32)\n\
         size: shader.uniform(vec2u)\n\
         main: fn() {\n\
             let p = global_id().xy\n\
             if p.x < self.size.x && p.y < self.size.y {\n\
                 let c = self.src.sample_lod(vec2(float(p.x), float(p.y)) / vec2(float(self.size.x), float(self.size.y)), 0.0)\n\
                 self.out[p] = vec4(c.x, float(p.x) / 255.0, float(group_count().x), 1.0)\n\
                 let old = atomic_add(self.counts[0], 1)\n\
                 atomic_add(self.counts[1], old)\n\
             }\n\
         }",
    )
    .unwrap();
    let b = &k.bindings;
    assert_eq!(b.workgroup, [8, 8, 1]);
    assert_eq!(b.textures.len(), 1);
    assert_eq!((b.images.len(), b.images[0].lanes), (1, 4));
    // the image takes u0, the written buffer the u register after it
    assert_eq!((b.buffers[0].writable, b.buffers[0].register, b.buffers[0].stride), (true, 1, 4));
    assert!(k.source.contains("RWTexture2D<float4> kernel_out : register(u0);"), "{}", k.source);
    assert!(k.source.contains("register(b13)"), "group_count reads the dispatch's counts");
    assert!(k.source.contains("InterlockedAdd(kernel_counts[0u], 1u, "), "{}", k.source);
    validate_hlsl(&k);
}

#[test]
fn hlsl_workgroup_memory_and_barriers() {
    let k = hlsl_kernel(
        "workgroup_size: 64\n\
         x: shader.buffer(f32)\n\
         y: shader.buffer_rw(f32)\n\
         partial: shader.workgroup(f32, 64)\n\
         main: fn() {\n\
             let t = local_id().x\n\
             self.partial[t] = self.x[global_id().x]\n\
             barrier()\n\
             if t == 0 {\n\
                 var s = 0.0\n\
                 for k in 0..64 { s += self.partial[k] }\n\
                 self.y[group_id().x] = s + float(extract_bits(local_index(), 0, 4)) + unpack_half2(pack_half2(vec2(s, s))).x\n\
             }\n\
         }",
    )
    .unwrap();
    assert!(k.source.contains("groupshared float kernel_partial[64];"), "{}", k.source);
    assert!(k.source.contains("GroupMemoryBarrierWithGroupSync();"));
    validate_hlsl(&k);
}

#[test]
fn hlsl_buffers_of_structs_keep_the_shared_layout() {
    let k = hlsl_kernel_after(
        "let Copy = struct{ pos: vec4f, scale: f32, id: u32, pad: vec2f }",
        "copies: shader.buffer_rw(Copy)\n\
         main: fn() { let i = global_id().x\n self.copies[i].scale = self.copies[i].scale * 2.0 }",
    )
    .unwrap();
    assert_eq!(k.bindings.buffers[0].stride, 32);
    validate_hlsl(&k);
    // a vec3 packs at 12 bytes in HLSL's structured buffers and 16 in the others: refused
    let r = hlsl_kernel_after(
        "let Copy = struct{ pos: vec3f, id: u32, scale: vec2f }",
        "copies: shader.buffer_rw(Copy)\n\
         main: fn() { self.copies[global_id().x].id = 1 }",
    );
    assert!(r.as_ref().is_err_and(|e| e.contains("packs differently")), "{r:?}");
    let r = hlsl_kernel(
        "points: shader.buffer_rw(vec3f)\n\
         main: fn() { self.points[global_id().x] = vec3(1.0, 2.0, 3.0) }",
    );
    assert!(r.as_ref().is_err_and(|e| e.contains("packs differently")), "{r:?}");
}

#[test]
fn what_hlsl_and_the_other_dialects_cannot_run_is_refused() {
    // subgroups need shader model 6
    let r = hlsl_kernel(
        "y: shader.buffer_rw(f32)\n\
         main: fn() { self.y[0] = subgroup_add(1.0) }",
    );
    assert!(r.as_ref().is_err_and(|e| e.contains("shader model 6")), "{r:?}");
    // a storage image is written, not read, at a 2D coordinate, and only in kernels
    let r = hlsl_kernel(
        "out: shader.storage_2d(vec4f)\n\
         y: shader.buffer_rw(vec4f)\n\
         main: fn() { self.y[0] = self.out[global_id().xy] }",
    );
    assert!(r.as_ref().is_err_and(|e| e.contains("write-only")), "{r:?}");
    let r = hlsl_kernel(
        "out: shader.storage_2d(vec4f)\n\
         main: fn() { self.out[global_id().x] = vec4(1.0) }",
    );
    assert!(r.as_ref().is_err_and(|e| e.contains("vec2u or vec2i")), "{r:?}");
    let r = hlsl_kernel(
        "out: shader.storage_2d(vec3f)\n\
         main: fn() { self.out[global_id().xy] = vec3(1.0) }",
    );
    assert!(r.is_err(), "{r:?}");
    // SPIR-V kernels have no storage images yet
    let (mut vm, obj) = eval_kernel(
        "out: shader.storage_2d(vec4f)\n\
         main: fn() { self.out[global_id().xy] = vec4(1.0) }",
    );
    let r = compile_compute_kernel(&mut vm, obj);
    assert!(r.as_ref().is_err_and(|e| e.contains("SPIR-V kernels have no storage images")), "{r:?}");
    // GLSL, WGSL and Metal say why they compile none
    for (backend, why) in [(shader_backend::ShaderBackend::Glsl, "GL ES 3.0"), (shader_backend::ShaderBackend::Wgsl, "WGSL"), (shader_backend::ShaderBackend::Metal, "Metal")] {
        let r = compile_compute_kernel_for(&mut vm, obj, backend, false);
        assert!(r.as_ref().is_err_and(|e| e.contains(why)), "{backend:?}: {r:?}");
    }
    assert!(matches!(compile_compute_kernel_for(&mut vm, obj, shader_backend::ShaderBackend::Hlsl, false), Ok(KernelCode::Hlsl(_))));
}
