//! Metal Shading Language from the shader IR: one library holding
//! `vertex_main` and `fragment_main`, and the pick variant of the same
//! shader (`ShaderOutput::pick`); and a compute kernel's library
//! (`metal_compute`).
//!
//! Bindings: the geometry at `buffer(0)`, the instance records at
//! `buffer(1)`, the dyn uniforms at `buffer(2)`, uniform buffers at their
//! `buffer_index`, the scope uniforms after them; textures at `texture(i)`
//! in io order (the pick twins after them, the pick ids at `buffer(30)`);
//! samplers are `constexpr`.
//!
//! MSL has no mutable program-scope variables, so the per-invocation state
//! the IR keeps in globals (records, varyings, outputs, resources) is the
//! `Mp` struct each entry point fills and every function takes as
//! `thread Mp &_mp`.

use crate::shader_ir::*;
use crate::shader_ir_compute::{ComputeSlot, ComputeSlotKind, KernelBindings, KernelBuffer, KernelImage, MetalKernel};
use crate::shader_ir_draw::{build_metal_entries, MetalDrawInfo};
use crate::shader_ir_text::*;
use crate::shader_output::*;
use std::fmt::Write;

/// The library source of a draw shader compiled to IR; `pick` prints its
/// pick variant.
pub fn metal_draw_source(output: &ShaderOutput, types: &impl crate::shader_types::ShaderTypes, pick: bool) -> String {
    metal_draw(output, types, pick).0
}

pub fn metal_draw(output: &ShaderOutput, types: &impl crate::shader_types::ShaderTypes, pick: bool) -> (String, MetalDrawInfo) {
    let mut ir = output.ir.clone();
    crate::shader_ir_precise::lower_precise(&mut ir, false);
    let info = build_metal_entries(output, &mut ir, types);
    let mut io_tys = Vec::new();
    for io in &output.io {
        io_tys.push(match io.kind {
            ShaderIoKind::Texture(_) | ShaderIoKind::Sampler(_) => TY_VOID,
            _ => ir.pod_ty(types, io.ty),
        });
    }
    let vertex_main = ir.function_by_name("vertex_main").unwrap();
    let fragment_main = ir.function_by_name("fragment_main").unwrap();
    let funcs = callees_first(&ir, &[vertex_main, fragment_main]);
    let used = ir.globals_used(&funcs);
    let mut cx = TextCx::new(&ir, Dialect::Metal, false);

    // The pick twins: the shader's 2D textures among its first 32.
    let mut twins = Vec::new();
    if pick {
        let mut k = 0u32;
        for io in &output.io {
            if let ShaderIoKind::Texture(tt) = io.kind {
                if k < 32 && tt == TextureType::Texture2d {
                    if let Some(g) = ir.find_global(IrGlobalKind::Io(IrIo::Texture(tt)), io.name) {
                        cx.pick_twins.push((g, k));
                    }
                    twins.push((k, io.name));
                }
                k += 1;
            }
        }
    }

    let mut bodies = String::new();
    for f in &funcs {
        if *f == vertex_main || *f == fragment_main {
            continue;
        }
        cx.function(*f, &mut bodies);
    }

    let mut out = String::new();
    // precise math: no fp contraction here; the HAL compiles the pipeline without fast math
    if ir.precise {
        out.push_str("#pragma clang fp contract(off)\n");
    }
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    let mut done = vec![false; ir.structs.len()];
    for s in 0..ir.structs.len() {
        struct_def(&cx, s as u32, &mut done, &mut out);
    }
    let mut scope_buffer = None;
    for io in &output.io {
        if let (ShaderIoKind::UniformBuffer, Some(b)) = (&io.kind, io.buffer_index) {
            scope_buffer = Some(scope_buffer.map_or(b + 1, |m: usize| m.max(b + 1)));
        }
    }
    let scope_buffer = scope_buffer.unwrap_or(3);
    let mut has = [false, false];
    for (kind, block, prefix) in [(0, "MpUniforms", "uni_"), (1, "MpScopeUniforms", "su_")] {
        let mut first = true;
        for (i, io) in output.io.iter().enumerate() {
            let wanted = match io.kind {
                ShaderIoKind::Uniform => kind == 0,
                ShaderIoKind::ScopeUniform => kind == 1,
                _ => false,
            };
            if !wanted {
                continue;
            }
            if first {
                writeln!(out, "struct {} {{", block).ok();
                first = false;
            }
            writeln!(out, "    {};", cx.decl(io_tys[i], &format!("{}{}", prefix, io.name))).ok();
        }
        if !first {
            out.push_str("};\n");
            has[kind] = true;
        }
    }
    sampler_decls(output, &mut out);

    // The invocation's state.
    out.push_str("struct Mp {\n    uint _mp_iter;\n");
    if has[0] {
        out.push_str("    constant MpUniforms *u;\n");
    }
    if has[1] {
        out.push_str("    constant MpScopeUniforms *su;\n");
    }
    for (i, io) in output.io.iter().enumerate() {
        match io.kind {
            ShaderIoKind::UniformBuffer => {
                writeln!(out, "    constant {} *unibuf_{};", cx.ty_name(io_tys[i]), io.name).ok();
            }
            ShaderIoKind::Texture(tt) => {
                writeln!(out, "    {} tex_{};", metal_texture_type(tt), io.name).ok();
            }
            _ => {}
        }
    }
    for (g, glob) in ir.globals.iter().enumerate() {
        if !used.get(g).copied().unwrap_or(false) {
            continue;
        }
        let name = cx.plain_global_name(g as GlobalId);
        match glob.kind {
            IrGlobalKind::Io(IrIo::Geometry | IrIo::DynInstance | IrIo::RustInstance | IrIo::Varying | IrIo::FragmentOutput(_) | IrIo::FragmentDepth)
            | IrGlobalKind::VertexPosition
            | IrGlobalKind::InstanceIndex => {
                writeln!(out, "    {};", cx.decl(glob.ty, &name)).ok();
            }
            IrGlobalKind::InstanceBuffer => {
                writeln!(out, "    constant uint *{};", word_buffer_name(glob.name)).ok();
            }
            _ => {}
        }
    }
    for (_, name) in &twins {
        writeln!(out, "    texture2d<float> tex_{}__pick;", name).ok();
    }
    if pick {
        out.push_str("    uint _pick_mask;\n    uint _pick_id;\n    float _pick_w;\n");
    }
    out.push_str("};\n");
    out.push_str(METAL_HELPERS);
    if pick {
        out.push_str(METAL_PICK_HELPERS);
    }
    out.push_str(&bodies);

    // Entry points: the resources both take, and how they fill `_mp`.
    let mut resources = String::new();
    if has[0] {
        resources.push_str(",\n    constant MpUniforms *u [[buffer(2)]]");
    }
    for (i, io) in output.io.iter().enumerate() {
        if let (ShaderIoKind::UniformBuffer, Some(b)) = (&io.kind, io.buffer_index) {
            write!(resources, ",\n    constant {} *unibuf_{} [[buffer({})]]", cx.ty_name(io_tys[i]), io.name, b).ok();
        }
    }
    if has[1] {
        write!(resources, ",\n    constant MpScopeUniforms *su [[buffer({})]]", scope_buffer).ok();
    }
    let mut t = 0;
    for io in &output.io {
        if let ShaderIoKind::Texture(tt) = io.kind {
            write!(resources, ",\n    {} tex_{} [[texture({})]]", metal_texture_type(tt), io.name, t).ok();
            t += 1;
        }
    }
    let mut fill = String::from("    Mp _mp;\n    _mp._mp_iter = 0u;\n");
    if has[0] {
        fill.push_str("    _mp.u = u;\n");
    }
    if has[1] {
        fill.push_str("    _mp.su = su;\n");
    }
    for io in &output.io {
        match io.kind {
            ShaderIoKind::UniformBuffer => {
                writeln!(fill, "    _mp.unibuf_{0} = unibuf_{0};", io.name).ok();
            }
            ShaderIoKind::Texture(_) => {
                writeln!(fill, "    _mp.tex_{0} = tex_{0};", io.name).ok();
            }
            _ => {}
        }
    }
    for (g, glob) in ir.globals.iter().enumerate() {
        if !used.get(g).copied().unwrap_or(false) {
            continue;
        }
        match glob.kind {
            IrGlobalKind::InstanceBuffer => {
                writeln!(fill, "    _mp.{0} = {0};", word_buffer_name(glob.name)).ok();
            }
            // The invocation's state starts zeroed: a varying a vertex
            // function writes on some paths only passes 0 on the others.
            IrGlobalKind::Io(IrIo::Geometry | IrIo::DynInstance | IrIo::RustInstance | IrIo::Varying | IrIo::FragmentOutput(_) | IrIo::FragmentDepth)
            | IrGlobalKind::VertexPosition
            | IrGlobalKind::InstanceIndex => {
                writeln!(fill, "    _mp.{} = {{}};", cx.plain_global_name(g as GlobalId)).ok();
            }
            _ => {}
        }
    }

    let vf = &ir.functions[vertex_main as usize];
    let ve = vf.entry.as_ref().unwrap();
    out.push_str("struct VertexOut {\n");
    for io in &ve.outputs {
        match io.binding {
            IrIoBinding::Builtin(IrBuiltinIo::Position) => writeln!(out, "    float4 {} [[position]];", io.name).ok(),
            // The instance id: an integer varying is flat.
            _ if io.ty == TY_U32 => writeln!(out, "    {} [[flat]];", cx.decl(io.ty, &io.name)).ok(),
            _ => writeln!(out, "    {};", cx.decl(io.ty, &io.name)).ok(),
        };
    }
    out.push_str("};\n");
    out.push_str("vertex VertexOut vertex_main(\n    constant uint *_mp_geom [[buffer(0)]],\n    constant uint *_mp_inst [[buffer(1)]]");
    out.push_str(&resources);
    out.push_str(",\n    uint vertex_index [[vertex_id]],\n    uint instance_index [[instance_id]]\n) {\n");
    out.push_str(&fill);
    if pick {
        out.push_str("    _mp._pick_mask = 0u;\n    _mp._pick_id = 0u;\n    _mp._pick_w = 0.0;\n");
    }
    out.push_str("    VertexOut out_data = {};\n");
    cx.stmts(vf, &vf.body, 1, &mut out);
    out.push_str("}\n");

    let ff = &ir.functions[fragment_main as usize];
    let fe = ff.entry.as_ref().unwrap();
    out.push_str("struct FragmentOut {\n");
    let mut depth_out = None;
    for io in &fe.outputs {
        match io.binding {
            IrIoBinding::Location(l) => {
                writeln!(out, "    {} [[color({})]];", cx.decl(io.ty, &io.name), l).ok();
            }
            IrIoBinding::Builtin(bi @ (IrBuiltinIo::FragDepth | IrBuiltinIo::FragDepthGreater | IrBuiltinIo::FragDepthLess)) => {
                writeln!(out, "    float {} {};", io.name, metal_depth_attribute(bi)).ok();
                depth_out = Some((io.name.clone(), bi));
            }
            _ => {}
        }
    }
    out.push_str("};\n");
    let pick_out = pick && !fe.outputs.is_empty();
    if pick_out {
        out.push_str("struct PickOut {\n");
        for io in &fe.outputs {
            if let IrIoBinding::Location(l) = io.binding {
                writeln!(out, "    float4 {} [[color({})]];", io.name, l).ok();
            }
        }
        // The id target keeps the written depth: picking resolves occlusion
        // against raster geometry exactly like the colour pass.
        if let Some((name, bi)) = &depth_out {
            writeln!(out, "    float {} {};", name, metal_depth_attribute(*bi)).ok();
        }
        out.push_str("};\n");
    }
    let ret_struct = if pick_out { "PickOut" } else { "FragmentOut" };
    write!(out, "fragment {} fragment_main(\n    VertexOut input [[stage_in]],\n    constant uint *_mp_geom [[buffer(0)]],\n    constant uint *_mp_inst [[buffer(1)]]", ret_struct).ok();
    out.push_str(&resources);
    let ntex = output.io.iter().filter(|io| matches!(io.kind, ShaderIoKind::Texture(_))).count();
    if pick {
        for (k, name) in &twins {
            write!(out, ",\n    texture2d<float> tex_{}__pick [[texture({})]]", name, ntex + *k as usize).ok();
        }
        out.push_str(",\n    constant uint4 *_pk [[buffer(30)]]");
    }
    out.push_str("\n) {\n");
    out.push_str(&fill);
    if pick {
        out.push_str("    _mp._pick_mask = _pk->y;\n    _mp._pick_id = 0u;\n    _mp._pick_w = 0.0;\n");
        for (_, name) in &twins {
            writeln!(out, "    _mp.tex_{0}__pick = tex_{0}__pick;", name).ok();
        }
    }
    out.push_str("    FragmentOut out_data = {};\n");
    if pick_out {
        // The body's `return out_data;` returns the colours; the pick
        // variant turns them into the id of what is drawn there.
        let mut body = String::new();
        cx.stmts(ff, &ff.body, 1, &mut body);
        out.push_str(&body.replace("return out_data;", "return _mp_pick_out(_mp, out_data, _pk->x);"));
    } else {
        cx.stmts(ff, &ff.body, 1, &mut out);
    }
    out.push_str("}\n");
    if pick_out {
        // Defined after use is not allowed: move it before the entry.
        let at = out.find("fragment PickOut fragment_main(").unwrap();
        let mut f = String::new();
        writeln!(f, "PickOut _mp_pick_out(thread Mp &_mp, FragmentOut c, uint draw_id) {{").ok();
        // The coverage: the first colour target's (a data target of one
        // channel covers).
        let first = fe.outputs.iter().find(|io| matches!(io.binding, IrIoBinding::Location(_)));
        let cover = match first {
            Some(io) if *ir.get(io.ty) == IrTy::Vec(4, IrScalar::F32) => format!("_mp_pick_cover(c.{})", io.name),
            _ => "1.0".to_string(),
        };
        writeln!(f, "    uint pid = _mp._pick_w > 0.0 ? _mp._pick_id : draw_id;").ok();
        writeln!(f, "    if ({} < 0.1 || pid == 0u) discard_fragment();", cover).ok();
        writeln!(f, "    float4 pc = float4(float(pid & 255u), float((pid >> 8) & 255u), float((pid >> 16) & 255u), 255.0) / 255.0;").ok();
        writeln!(f, "    PickOut o;").ok();
        for io in &fe.outputs {
            if let IrIoBinding::Location(_) = io.binding {
                writeln!(f, "    o.{} = pc;", io.name).ok();
            }
        }
        if let Some((name, _)) = &depth_out {
            writeln!(f, "    o.{0} = c.{0};", name).ok();
        }
        writeln!(f, "    return o;\n}}").ok();
        out.insert_str(at, &f);
    }
    (out, info)
}

/// The attribute of a fragment depth output (conservative where promised).
fn metal_depth_attribute(bi: IrBuiltinIo) -> &'static str {
    match bi {
        IrBuiltinIo::FragDepthGreater => "[[depth(greater)]]",
        IrBuiltinIo::FragDepthLess => "[[depth(less)]]",
        _ => "[[depth(any)]]",
    }
}

fn struct_def(cx: &TextCx, s: u32, done: &mut Vec<bool>, out: &mut String) {
    if done[s as usize] {
        return;
    }
    done[s as usize] = true;
    let st = &cx.m.structs[s as usize];
    if st.synth.is_some() {
        return;
    }
    for f in &st.fields {
        let mut t = f.ty;
        while let IrTy::Array(e, _) = cx.m.get(t) {
            t = *e;
        }
        if let IrTy::Struct(inner) = cx.m.get(t) {
            struct_def(cx, *inner, done, out);
        }
    }
    let sty = cx.m.types.iter().position(|t| *t == IrTy::Struct(s)).unwrap_or(0) as TyId;
    writeln!(out, "struct {} {{", cx.struct_name(s)).ok();
    for (k, f) in st.fields.iter().enumerate() {
        writeln!(out, "    {};", cx.decl(f.ty, &cx.field_name(sty, k as u32))).ok();
    }
    out.push_str("};\n");
}

fn sampler_decls(output: &ShaderOutput, out: &mut String) {
    use crate::shader_output::{SamplerAddress, SamplerCoord, SamplerFilter};
    for (idx, sampler) in output.samplers.iter().enumerate() {
        let filter = match sampler.filter {
            SamplerFilter::Nearest => "nearest",
            SamplerFilter::Linear => "linear",
        };
        let address = match sampler.address {
            SamplerAddress::Repeat => "repeat",
            SamplerAddress::ClampToEdge => "clamp_to_edge",
            SamplerAddress::ClampToZero => "clamp_to_zero",
            SamplerAddress::MirroredRepeat => "mirrored_repeat",
        };
        let coord = match sampler.coord {
            SamplerCoord::Normalized => "normalized",
            SamplerCoord::Pixel => "pixel",
        };
        writeln!(
            out,
            "constexpr sampler _s{}(filter::{}, mip_filter::linear, address::{}, coord::{}{});",
            idx,
            filter,
            address,
            coord,
            if sampler.compare { ", compare_func::less_equal" } else { "" }
        )
        .ok();
    }
}

const METAL_HELPERS: &str = "inline float2 _mp_unpack2f16(float x) { half2 h = as_type<half2>(as_type<uint>(x)); return float2(h.x, h.y); }
inline float4 _mp_unpack4u8(float x) { uint u = as_type<uint>(x); return float4(float(u & 0xffu), float((u >> 8) & 0xffu), float((u >> 16) & 0xffu), float((u >> 24) & 0xffu)) * (1.0 / 255.0); }
inline float4x4 _mp_inverse(float4x4 m) {
    float a00 = m[0][0]; float a01 = m[0][1]; float a02 = m[0][2]; float a03 = m[0][3];
    float a10 = m[1][0]; float a11 = m[1][1]; float a12 = m[1][2]; float a13 = m[1][3];
    float a20 = m[2][0]; float a21 = m[2][1]; float a22 = m[2][2]; float a23 = m[2][3];
    float a30 = m[3][0]; float a31 = m[3][1]; float a32 = m[3][2]; float a33 = m[3][3];
    float b00 = a00 * a11 - a01 * a10; float b01 = a00 * a12 - a02 * a10;
    float b02 = a00 * a13 - a03 * a10; float b03 = a01 * a12 - a02 * a11;
    float b04 = a01 * a13 - a03 * a11; float b05 = a02 * a13 - a03 * a12;
    float b06 = a20 * a31 - a21 * a30; float b07 = a20 * a32 - a22 * a30;
    float b08 = a20 * a33 - a23 * a30; float b09 = a21 * a32 - a22 * a31;
    float b10 = a21 * a33 - a23 * a31; float b11 = a22 * a33 - a23 * a32;
    float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (det == 0.0) {
        return float4x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 0.0, 0.0, 1.0));
    }
    float idet = 1.0 / det;
    return float4x4(
        float4((a11 * b11 - a12 * b10 + a13 * b09) * idet, (a02 * b10 - a01 * b11 - a03 * b09) * idet, (a31 * b05 - a32 * b04 + a33 * b03) * idet, (a22 * b04 - a21 * b05 - a23 * b03) * idet),
        float4((a12 * b08 - a10 * b11 - a13 * b07) * idet, (a00 * b11 - a02 * b08 + a03 * b07) * idet, (a32 * b02 - a30 * b05 - a33 * b01) * idet, (a20 * b05 - a22 * b02 + a23 * b01) * idet),
        float4((a10 * b10 - a11 * b08 + a13 * b06) * idet, (a01 * b08 - a00 * b10 - a03 * b06) * idet, (a30 * b04 - a31 * b02 + a33 * b00) * idet, (a21 * b02 - a20 * b04 - a23 * b00) * idet),
        float4((a11 * b07 - a10 * b09 - a12 * b06) * idet, (a00 * b09 - a01 * b07 + a02 * b06) * idet, (a31 * b01 - a30 * b03 - a32 * b00) * idet, (a20 * b03 - a21 * b01 + a22 * b00) * idet));
}
";

/// The pick variant's sample: where the colour shows more than any earlier
/// sample's (its alpha, or its light where it adds), the id in the
/// texture's twin under the place, else one beside it (a filtered sample
/// sees its neighbours too).
const METAL_PICK_HELPERS: &str = "inline float _mp_pick_cover(float4 c) { return max(c.w, max(c.x, max(c.y, c.z))); }
inline float4 _mp_pick_sample(thread Mp &_mp, uint b, texture2d<float> tw, float4 c, float2 uv) {
    float w = _mp_pick_cover(c);
    if ((_mp._pick_mask & (1u << b)) != 0u && w > _mp._pick_w) {
        float2 sz = float2(tw.get_width(), tw.get_height());
        int2 q = int2(clamp(uv, float2(0.0), float2(0.99999)) * sz);
        int2 hi = int2(sz) - 1;
        uint id = 0u;
        for (int k = 0; k < 5 && id == 0u; k++) {
            int2 o = k == 0 ? int2(0) : (k == 1 ? int2(1, 0) : (k == 2 ? int2(-1, 0) : (k == 3 ? int2(0, 1) : int2(0, -1))));
            float4 e = tw.read(uint2(clamp(q + o, int2(0), hi)));
            id = uint(e.x * 255.0 + 0.5) | (uint(e.y * 255.0 + 0.5) << 8) | (uint(e.z * 255.0 + 0.5) << 16);
        }
        if (id != 0u) { _mp._pick_w = w; _mp._pick_id = id; }
    }
    return c;
}
";

/// The Metal buffer index of a kernel's parameter block (the last argument-table slot).
pub const METAL_KERNEL_PARAMS_BUFFER: u32 = 30;

/// A compute kernel (`entry`, made by `build_compute_entry`) as MSL, entry `kernel_main`, and
/// what the host binds it by.
///
/// Indices, each list in the IR's global order (the order a kernel first uses its resources, the
/// order HLSL lists them in too): sampled textures at `texture(0..)`, storage images
/// (`texture2d<T, access::write>`) at the texture indices after them; the storage buffers at
/// `buffer(0..)` in the order a dispatch names them, `const device T*` when read-only and
/// `device T*` when written (each `KernelBuffer::register` is its index); the params in one
/// `constant` struct at `buffer(METAL_KERNEL_PARAMS_BUFFER)`, at the offsets HLSL packs them at;
/// samplers are `constexpr`. The group count comes from `threadgroups_per_grid`, so a dispatch
/// binds nothing for it. Workgroup arrays are `threadgroup` arrays of the entry. The workgroup
/// size is not in MSL: the host passes it at each dispatch (`KernelBindings::workgroup`).
///
/// Refused, with every reason: cooperative matrices, f16 and bool in buffers or params, and
/// buffer elements whose MSL layout differs from the one the other dialects use (a vec3 followed
/// by a field in its padding: pad it).
pub fn metal_compute(output: &ShaderOutput, entry: FuncId) -> Result<MetalKernel, String> {
    let ir = &output.ir;
    let mut errors = Vec::new();
    let funcs = callees_first(ir, &[entry]);
    let mut cx = TextCx::new(ir, Dialect::Metal, false);
    let mut bodies = String::new();
    for f in &funcs {
        if *f != entry {
            cx.function(*f, &mut bodies);
        }
    }
    let mut main = String::new();
    let f = &ir.functions[entry as usize];
    cx.stmts(f, &f.body, 1, &mut main);
    errors.append(&mut cx.errors);

    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    let mut done = vec![false; ir.structs.len()];
    for s in 0..ir.structs.len() {
        struct_def(&cx, s as u32, &mut done, &mut out);
    }

    let wg = ir.workgroup;
    let mut b = KernelBindings { workgroup: [wg[0].max(1), wg[1].max(1), wg[2].max(1)], samplers: output.samplers.clone(), ..Default::default() };
    // the parameter block: each param at its natural offset, padding spelled out (a vec3 is
    // packed, 12 bytes as HLSL has it)
    let mut offset = 0u32;
    let mut params = String::new();
    for (i, g) in ir.globals.iter().enumerate() {
        if g.kind != IrGlobalKind::Io(IrIo::Uniform) {
            continue;
        }
        let (scalar, lanes) = match ir.get(g.ty) {
            IrTy::Scalar(s) => (*s, 1),
            IrTy::Vec(n, s) => (*s, *n as u32),
            other => {
                errors.push(format!("kernel param {} must be a scalar or a vector, not {:?}", g.name, other));
                continue;
            }
        };
        if matches!(scalar, IrScalar::Bool | IrScalar::F16) {
            errors.push(format!("kernel param {} cannot be a bool or an f16 (use u32 / f32)", g.name));
            continue;
        }
        let (size, align) = ir_layout(ir, g.ty);
        let at = round_up(offset, align);
        if at > offset {
            writeln!(params, "    uchar _mp_pad{}[{}];", offset, at - offset).ok();
        }
        let name = cx.plain_global_name(i as GlobalId);
        if lanes == 3 {
            writeln!(params, "    packed_{} {};", cx.ty_name(g.ty), name).ok();
        } else {
            writeln!(params, "    {};", cx.decl(g.ty, &name)).ok();
        }
        b.params.push(ComputeSlot { name: g.name, global: i as GlobalId, kind: ComputeSlotKind::Param { scalar, lanes }, offset: at, size });
        offset = at + size;
    }
    if !b.params.is_empty() {
        b.params_size = round_up(offset, 16);
        writeln!(out, "struct MpUniforms {{\n{}}};", params).ok();
    }
    sampler_decls(output, &mut out);

    // the invocation's state, the entry's arguments, and how it fills `_mp`
    let mut fields = String::from("    uint _mp_iter;\n    uint3 _mp_gid;\n    uint3 _mp_lid;\n    uint3 _mp_grp;\n    uint3 _mp_groups;\n    uint _mp_lindex;\n    uint _mp_sg_size;\n    uint _mp_sg_lane;\n    uint _mp_sg_id;\n");
    let mut args = Vec::new();
    let mut fill = String::from("    Mp _mp;\n    _mp._mp_iter = 0u;\n    _mp._mp_gid = _mp_gid_in;\n    _mp._mp_lid = _mp_lid_in;\n    _mp._mp_grp = _mp_grp_in;\n    _mp._mp_groups = _mp_groups_in;\n    _mp._mp_lindex = _mp_lindex_in;\n    _mp._mp_sg_size = _mp_sg_size_in;\n    _mp._mp_sg_lane = _mp_sg_lane_in;\n    _mp._mp_sg_id = _mp_sg_id_in;\n");
    if !b.params.is_empty() {
        fields.push_str("    constant MpUniforms *u;\n");
        args.push(format!("constant MpUniforms *u [[buffer({})]]", METAL_KERNEL_PARAMS_BUFFER));
        fill.push_str("    _mp.u = u;\n");
    }
    let mut t = 0u32;
    for (i, g) in ir.globals.iter().enumerate() {
        if let IrGlobalKind::Io(IrIo::Texture(tt)) = g.kind {
            let name = cx.plain_global_name(i as GlobalId);
            writeln!(fields, "    {} {};", metal_texture_type(tt), name).ok();
            args.push(format!("{} {} [[texture({})]]", metal_texture_type(tt), name, t));
            writeln!(fill, "    _mp.{0} = {0};", name).ok();
            b.textures.push((g.name, tt));
            t += 1;
        }
    }
    for (i, g) in ir.globals.iter().enumerate() {
        if g.kind != IrGlobalKind::Io(IrIo::StorageImage) {
            continue;
        }
        let IrTy::Array(texel, _) = *ir.get(g.ty) else { continue };
        let (scalar, lanes) = match ir.get(texel) {
            IrTy::Scalar(s) => (*s, 1),
            IrTy::Vec(n, s) => (*s, *n as u32),
            _ => (IrScalar::Bool, 0),
        };
        if !matches!(scalar, IrScalar::F32 | IrScalar::U32 | IrScalar::I32) || lanes == 3 {
            errors.push(format!("storage image {}: a texel of 1, 2 or 4 lanes of f32, u32 or i32", g.name));
        }
        let name = cx.plain_global_name(i as GlobalId);
        let ty = format!("texture2d<{}, access::write>", cx.scalar_name(scalar));
        writeln!(fields, "    {} {};", ty, name).ok();
        args.push(format!("{} {} [[texture({})]]", ty, name, t));
        writeln!(fill, "    _mp.{0} = {0};", name).ok();
        b.images.push(KernelImage { name: g.name, scalar, lanes });
        t += 1;
    }
    let mut index = 0u32;
    for (i, g) in ir.globals.iter().enumerate() {
        let IrGlobalKind::Io(IrIo::Storage(writable)) = g.kind else { continue };
        let IrTy::Array(elem, _) = *ir.get(g.ty) else { continue };
        let stride = match msl_layout(ir, elem) {
            Some((size, _)) if size == ir_array_stride(ir, elem) => size,
            _ => {
                errors.push(format!(
                    "buffer {}: its element lays out differently in MSL (an f16 or a bool, or a field in a vec3's padding); pad it", g.name));
                0
            }
        };
        if index >= METAL_KERNEL_PARAMS_BUFFER {
            errors.push(format!("buffer {}: a Metal kernel binds at most {} buffers", g.name, METAL_KERNEL_PARAMS_BUFFER));
        }
        let name = cx.plain_global_name(i as GlobalId);
        let ty = format!("{}device {}*", if writable { "" } else { "const " }, cx.ty_name(elem));
        writeln!(fields, "    {} {};", ty, name).ok();
        args.push(format!("{} {} [[buffer({})]]", ty, name, index));
        writeln!(fill, "    _mp.{0} = {0};", name).ok();
        b.buffers.push(KernelBuffer { name: g.name, writable, register: index, stride });
        index += 1;
    }
    let mut shared = String::new();
    for (i, g) in ir.globals.iter().enumerate() {
        if g.kind != IrGlobalKind::Io(IrIo::Workgroup) {
            continue;
        }
        let IrTy::Array(elem, n) = *ir.get(g.ty) else { continue };
        let name = cx.plain_global_name(i as GlobalId);
        writeln!(fields, "    threadgroup {} *{};", cx.ty_name(elem), name).ok();
        writeln!(shared, "    threadgroup {} {}[{}];", cx.ty_name(elem), name, n.max(1)).ok();
        writeln!(fill, "    _mp.{0} = {0};", name).ok();
    }
    writeln!(out, "struct Mp {{\n{}}};", fields).ok();
    out.push_str(METAL_HELPERS);
    out.push_str(METAL_KERNEL_HELPERS);
    out.push_str(&bodies);
    args.extend([
        "uint3 _mp_gid_in [[thread_position_in_grid]]",
        "uint3 _mp_lid_in [[thread_position_in_threadgroup]]",
        "uint3 _mp_grp_in [[threadgroup_position_in_grid]]",
        "uint3 _mp_groups_in [[threadgroups_per_grid]]",
        "uint _mp_lindex_in [[thread_index_in_threadgroup]]",
        "uint _mp_sg_size_in [[threads_per_simdgroup]]",
        "uint _mp_sg_lane_in [[thread_index_in_simdgroup]]",
        "uint _mp_sg_id_in [[simdgroup_index_in_threadgroup]]",
    ].map(String::from));
    writeln!(out, "kernel void kernel_main(\n    {}\n) {{", args.join(",\n    ")).ok();
    out.push_str(&shared);
    out.push_str(&fill);
    out.push_str(&main);
    out.push_str("}\n");
    if !errors.is_empty() {
        errors.dedup();
        return Err(errors.join("\n"));
    }
    Ok(MetalKernel { source: out, bindings: b })
}

/// Size and alignment of a type in an MSL buffer, when every field sits where `ir_layout` puts
/// it; None otherwise (and for f16 and bool, refused in buffers as HLSL refuses them).
fn msl_layout(m: &IrModule, ty: TyId) -> Option<(u32, u32)> {
    match m.get(ty) {
        IrTy::Scalar(IrScalar::F16 | IrScalar::Bool) | IrTy::Vec(_, IrScalar::F16 | IrScalar::Bool) => None,
        IrTy::Scalar(_) => Some((4, 4)),
        IrTy::Vec(n, _) => {
            let s = if *n == 3 { 16 } else { 4 * *n as u32 };
            Some((s, s))
        }
        IrTy::Mat(c, r) => {
            let col = if *r == 2 { 8 } else { 16 };
            Some((*c as u32 * col, col))
        }
        IrTy::Array(e, n) if *n > 0 => {
            let (es, ea) = msl_layout(m, *e)?;
            (es == ir_array_stride(m, *e)).then_some((es * n, ea))
        }
        IrTy::Struct(s) => {
            let (mut offset, mut align) = (0u32, 1u32);
            for (f, ir_offset) in m.structs[*s as usize].fields.iter().zip(ir_struct_offsets(m, *s)) {
                let (size, a) = msl_layout(m, f.ty)?;
                offset = round_up(offset, a);
                if offset != ir_offset {
                    return None;
                }
                offset += size;
                align = align.max(a);
            }
            let size = round_up(offset.max(1), align);
            (size == ir_layout(m, ty).0).then_some((size, align))
        }
        _ => None,
    }
}

const METAL_KERNEL_HELPERS: &str = "inline uint _mp_extract_bits(uint x, uint o, uint c) { return c == 0u ? 0u : (x >> o) & (0xffffffffu >> (32u - c)); }
inline uint _mp_dot4_u8(uint a, uint b) { return (a & 255u) * (b & 255u) + ((a >> 8) & 255u) * ((b >> 8) & 255u) + ((a >> 16) & 255u) * ((b >> 16) & 255u) + (a >> 24) * (b >> 24); }
";

pub fn metal_draw_source_into(output: &ShaderOutput, types: &impl crate::shader_types::ShaderTypes, pick: bool, out: &mut String) {
    out.push_str(&metal_draw_source(output, types, pick));
}
