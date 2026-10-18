//! Splash kernels on Metal: compiled to MSL (`compile_compute_kernel_metal`) and built by the
//! Metal device as the backend builds them (every feature the compiler prints), and one end to
//! end: made a pipeline by the frame builder's `metal_kernel_pipeline_blob`, its params laid out
//! by the bindings the compiler reported, dispatched through the HAL into a storage image that a
//! draw in the next pass samples.
#![cfg(target_os = "macos")]

use makepad_platform::makepad_hal::backend::apple::metal::compile_msl_kernel;
use makepad_platform::makepad_hal::*;
use makepad_platform::makepad_script::shader_ir_compute::{compile_compute_kernel_metal, MetalKernel};
use makepad_platform::makepad_script::{ScriptMod, ScriptVm, ScriptVmBase, ScriptVmHost};
use makepad_platform::makepad_live_id::{id, LiveId};
use makepad_platform::metal_kernel_pipeline_blob;
use std::time::{Duration, Instant};

const IMAGE: Id = Id::new(1, 1);
const TARGET: Id = Id::new(2, 1);
const COLORS: Id = Id::new(1, 1);
const QUAD: Id = Id::new(2, 1);
const QUAD_INDEX: Id = Id::new(3, 1);
const KERNEL: Id = Id::new(1, 1);
const SAMPLE: Id = Id::new(2, 1);

/// Four vertical bands of `colors`, tinted, into `out`.
const BANDS: &str = "workgroup_size: 8
workgroup_size_y: 8
colors: shader.buffer(vec4f)
out: shader.storage_2d(vec4f)
size: shader.uniform(vec2u)
tint: shader.uniform(vec4f)
main: fn() {
    let p = global_id().xy
    if p.x < self.size.x && p.y < self.size.y {
        self.out[p] = self.colors[p.x * 4 / self.size.x] * self.tint
    }
}";

/// A full-target quad sampling texture 0 texel for texel (point, clamped).
const SAMPLE_MSL: &str = "#include <metal_stdlib>
using namespace metal;
struct VertexOut { float4 p [[position]]; float2 uv; };
constexpr sampler s0(filter::nearest, address::clamp_to_edge);
vertex VertexOut vertex_main(constant float4 *g [[buffer(0)]], uint vid [[vertex_id]]) {
    VertexOut o; o.p = g[vid]; o.uv = g[vid].xy * float2(0.5, -0.5) + 0.5; return o;
}
fragment float4 fragment_main(VertexOut v [[stage_in]], texture2d<float> tex [[texture(0)]]) { return tex.sample(s0, v.uv); }
";

fn metal_kernel(prelude: &str, body: &str) -> MetalKernel {
    let host = Box::leak(Box::new(ScriptVmHost::new(0i32, ())));
    let mut vm = ScriptVm { host: host.host_context(), bx: Box::new(ScriptVmBase::new()) };
    vm.bx.captured_errors = Some(Vec::new());
    let code = format!("use mod.pod.*\nuse mod.math.*\nuse mod.shader\n{prelude}\nlet k = {{\n{body}\n}}\nk");
    let value = vm.with_instruction_limit(50_000_000, |vm| vm.eval(ScriptMod::new("", "", "kernel", 0, 0, &code, &[])));
    assert!(vm.take_errors().is_empty());
    let kernel = compile_compute_kernel_metal(&mut vm, value.as_object().unwrap()).unwrap();
    assert!(vm.take_errors().is_empty());
    kernel
}

fn until(hal: &Hal, sid: u32, mut accept: impl FnMut(HalEvent<'_>) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut store = EventStore::new(sid, 64, 65536);
    loop {
        assert!(Instant::now() < deadline, "Metal event deadline");
        let mut buf = store.buf();
        let n = hal.pump(10_000_000, &mut buf);
        for (_, event) in store.reader(n, buf.arena_len) {
            if let HalEvent::PipelineFailed { log, .. } = event {
                panic!("pipeline: {}", String::from_utf8_lossy(log));
            }
            if accept(event) {
                return;
            }
        }
    }
}

/// What the compiler prints for every kernel feature is MSL the device builds a compute
/// pipeline from: buffers, params, a sampled texture and a storage image, atomics on buffers and
/// workgroup memory, barriers, subgroups, packing helpers, structs in buffers, helper functions.
#[test]
fn every_kernel_feature_builds_on_the_device() {
    let kernels = [
        ("", "workgroup_size: 64
src: shader.buffer(f32)
dst: shader.buffer_rw(f32)
n: shader.uniform(u32)
scale: shader.uniform(f32)
offset: shader.uniform(vec2f)
tint: shader.uniform(vec4f)
dir: shader.uniform(vec3f)
main: fn() {
    let i = global_id().x
    if i < self.n { self.dst[i] = self.src[i] * self.scale + self.offset.x + self.tint.w + self.dir.z }
}"),
        ("", "workgroup_size: 8
workgroup_size_y: 8
src: shader.texture_2d(float)
out: shader.storage_2d(vec4f)
mask: shader.storage_2d(f32)
ids: shader.storage_2d(vec2u)
counts: shader.buffer_rw(u32)
size: shader.uniform(vec2u)
main: fn() {
    let p = global_id().xy
    if p.x < self.size.x && p.y < self.size.y {
        let c = self.src.sample_lod(vec2(float(p.x), float(p.y)) / vec2(float(self.size.x), float(self.size.y)), 0.0)
        self.out[p] = vec4(c.x, float(p.x) / 255.0, float(group_count().x), 1.0)
        self.mask[p] = c.y
        self.ids[p] = p
        let old = atomic_add(self.counts[0], 1)
        atomic_add(self.counts[1], old)
    }
}"),
        ("", "workgroup_size: 64
x: shader.buffer(f32)
y: shader.buffer_rw(f32)
partial: shader.workgroup(f32, 64)
hits: shader.workgroup(u32, 1)
main: fn() {
    let t = local_id().x
    self.partial[t] = self.x[global_id().x]
    atomic_add(self.hits[0], 1)
    barrier()
    if t == 0 {
        var s = 0.0
        for k in 0..64 { s += self.partial[k] }
        self.y[group_id().x] = s + float(extract_bits(local_index(), 0, 4)) + unpack_half2(pack_half2(vec2(s, s))).x
            + subgroup_add(1.0) + subgroup_max(s) + subgroup_shuffle(s, 1) + subgroup_shuffle_xor(s, 1) + subgroup_first(s)
            + float(subgroup_size() + subgroup_lane() + subgroup_id() + self.hits[0] + dot4_u8(7, 9))
    }
}"),
        ("let Copy = struct{ pos: vec4f, scale: f32, id: u32, pad: vec2f }\nlet grow = fn(c: Copy) -> f32 { return c.scale * 2.0 }", "copies: shader.buffer_rw(Copy)
points: shader.buffer_rw(vec3f)
total: shader.buffer_rw(i32)
main: fn() {
    let i = global_id().x
    self.copies[i].scale = grow(self.copies[i])
    self.points[i] = self.copies[i].pos.xyz
    atomic_add(self.total[0], -1)
}"),
    ];
    for (prelude, body) in kernels {
        let k = metal_kernel(prelude, body);
        let numbered: String = k.source.lines().enumerate().map(|(i, l)| format!("{}:{l}\n", i + 1)).collect();
        if let Err(e) = compile_msl_kernel(&k.source) {
            panic!("{e}\n{numbered}");
        }
    }
}

#[test]
fn a_splash_kernel_writes_an_image_a_draw_then_samples() {
    // the kernel, compiled as the app would for a Metal HAL
    let kernel = metal_kernel("", BANDS);
    let b = &kernel.bindings;
    assert_eq!((b.images.len(), b.buffers.len(), b.buffers[0].stride), (1, 1, 16));

    // the params at the offsets the compiler reported
    let mut params = vec![0u8; b.params_size as usize];
    let size = b.param(id!(size)).unwrap().offset as usize;
    let tint = b.param(id!(tint)).unwrap().offset as usize;
    params[size..size + 8].copy_from_slice(&[32u32.to_le_bytes(), 32u32.to_le_bytes()].concat());
    for (k, v) in [1.0f32, 1.0, 0.5, 1.0].into_iter().enumerate() {
        params[tint + 4 * k..tint + 4 * k + 4].copy_from_slice(&v.to_le_bytes());
    }

    let hal = Hal::new();
    let info = hal.open(&OpenReq { backend: backend_id::METAL, flags: open_flags::OFFSCREEN_WINDOWS, ..Default::default() });
    assert_eq!(info.backend, backend_id::METAL, "Metal HAL initialization failed");
    let sid = info.session;
    let mut w = CmdWriter::new(sid);
    w.texture_create(IMAGE, usage::STORAGE | usage::SAMPLED, format::RGBA8, 32, 32, [1, 1], Attr::default());
    // BGRA8: what a Metal pipeline is built for when it declares no format (RGBA8 is format 0)
    w.texture_create(TARGET, usage::RENDER_TARGET | usage::SAMPLED, format::BGRA8, 32, 32, [1, 1], Attr::default());
    let colors: Vec<u8> = [[1.0f32, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]].into_iter().flatten().flat_map(f32::to_le_bytes).collect();
    w.buffer_create(COLORS, buffer_kind::STORAGE, colors.len() as u64, b.buffers[0].stride, Attr::default());
    w.buffer_update(COLORS, 0, &colors);
    let quad: Vec<u8> = [-1.0f32, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0].into_iter().flat_map(f32::to_le_bytes).collect();
    w.buffer_create(QUAD, buffer_kind::VERTEX, quad.len() as u64, 0, Attr::default());
    w.buffer_update(QUAD, 0, &quad);
    w.buffer_create(QUAD_INDEX, buffer_kind::INDEX, 12, 0, Attr::default());
    w.buffer_update(QUAD_INDEX, 0, &[0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 3, 0]);
    w.pipeline_create(KERNEL, dialect::MSL, Attr::default(), &metal_kernel_pipeline_blob(&kernel, 0, 0).unwrap());
    let none = pipeline_binding::NONE;
    let desc = PipelineDesc { outputs: 1, color_formats: [format::BGRA8, 0, 0, 0], ..Default::default() };
    w.pipeline_create(SAMPLE, dialect::MSL, Attr::default(), &CmdWriter::pipeline_blob(desc, SAMPLE_MSL.as_bytes(), &[none, none, none, none, none, 1], &[16, 0, 3, 4]));
    hal.submit(&w.buf());
    let mut ready = Vec::new();
    until(&hal, sid, |e| {
        if let HalEvent::PipelineReady { pipeline } = e {
            ready.push(pipeline);
        }
        ready.contains(&KERNEL) && ready.contains(&SAMPLE)
    });

    w.clear();
    let images = w.put_ids(&[IMAGE]);
    let buffers = w.put_ids(&[COLORS]);
    let tex = w.put_ids(&[IMAGE]);
    let wg = b.workgroup;
    w.frame_begin(1, FrameMeta::default());
    w.pass(Pass::compute(0));
    w.dispatch(Dispatch { pipeline: KERNEL, groups: [32 / wg[0], 32 / wg[1], 1], uniforms: Slice::from_bytes(&params), images, buffers, ..Default::default() });
    w.pass(Pass { color: [TARGET, Id::NONE, Id::NONE, Id::NONE], load: [pass_load::CLEAR; 5], viewport: [0.0, 0.0, 32.0, 32.0], dpi: 1.0, ..Default::default() });
    w.draw(Draw { pipeline: SAMPLE, vb: QUAD, ib: QUAD_INDEX, index: [0, 6], inst: Inst::inline(Slice::EMPTY, 1), tex, scissor: [f32::NAN; 4], ..Default::default() });
    w.frame_end(TARGET, target::TEXTURE);
    hal.submit(&w.buf());
    until(&hal, sid, |e| matches!(e, HalEvent::FrameDone { frame: 1, .. }));
    let st = hal.with_world(|w| w.session(sid).unwrap().stats);
    assert_eq!((st.dispatches, st.skipped_dispatches, st.bad_args, st.skipped_draws), (1, 0, 0, 0));

    w.clear();
    w.texture_readback(TARGET, 1, [0, 0, 32, 32]);
    hal.submit(&w.buf());
    let mut image = Vec::new();
    until(&hal, sid, |e| match e {
        HalEvent::Readback { req: 1, status, pixels, .. } => {
            assert_eq!(status, readback::OK);
            image = pixels.to_vec();
            true
        }
        _ => false,
    });
    let at = |x: usize, y: usize| image[(y * 32 + x) * 4..][..4].to_vec();
    let half = at(20, 5)[2];
    assert!((127..=128).contains(&half), "{:?}", at(20, 5));
    assert_eq!(at(0, 0), [255, 0, 0, 255]);
    assert_eq!(at(12, 31), [0, 255, 0, 255]);
    assert_eq!(at(20, 5), [0, 0, half, 255]);
    assert_eq!(at(31, 16), [255, 255, half, 255]);
    assert!(hal.close(sid));
}
