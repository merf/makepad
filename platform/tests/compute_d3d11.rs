//! A Splash kernel end to end on D3D11: compiled to HLSL (`compile_compute_kernel_hlsl`), made a
//! pipeline by the frame builder's `hlsl_kernel_pipeline_blob`, its params laid out by the
//! bindings the compiler reported, dispatched through the HAL into a storage image that a draw in
//! the next pass samples.
#![cfg(windows)]

use makepad_platform::hlsl_kernel_pipeline_blob;
use makepad_platform::makepad_hal::hlsl::HlslPipeline;
use makepad_platform::makepad_hal::*;
use makepad_platform::makepad_script::shader_ir_compute::compile_compute_kernel_hlsl;
use makepad_platform::makepad_script::{ScriptMod, ScriptVm, ScriptVmBase, ScriptVmHost};
use makepad_platform::makepad_live_id::{id, LiveId};
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

/// A full-target quad sampling texture t0 texel for texel (point, clamped).
const SAMPLE_HLSL: &str = "Texture2D tex : register(t0);
SamplerState s0 : register(s0);
struct VertexIn { float4 g : GEOMA0; };
struct VertexOut { float4 p : SV_POSITION; float2 uv : TEXCOORD0; };
VertexOut vertex_main(VertexIn i) { VertexOut o; o.p = i.g; o.uv = i.g.xy * float2(0.5, -0.5) + 0.5; return o; }
float4 pixel_main(VertexOut v) : SV_TARGET0 { return tex.SampleLevel(s0, v.uv, 0.0); }
";

fn until(hal: &Hal, sid: u32, mut accept: impl FnMut(HalEvent<'_>) -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut store = EventStore::new(sid, 64, 65536);
    loop {
        assert!(Instant::now() < deadline, "D3D11 event deadline");
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

#[test]
fn a_splash_kernel_writes_an_image_a_draw_then_samples() {
    // the kernel, compiled as the app would for a D3D11 HAL
    let host = Box::leak(Box::new(ScriptVmHost::new(0i32, ())));
    let mut vm = ScriptVm { host: host.host_context(), bx: Box::new(ScriptVmBase::new()) };
    vm.bx.captured_errors = Some(Vec::new());
    let code = format!("use mod.pod.*\nuse mod.math.*\nuse mod.shader\nlet k = {{\n{BANDS}\n}}\nk");
    let value = vm.with_instruction_limit(50_000_000, |vm| vm.eval(ScriptMod::new("", "", "bands", 0, 0, &code, &[])));
    assert!(vm.take_errors().is_empty());
    let kernel = compile_compute_kernel_hlsl(&mut vm, value.as_object().unwrap()).unwrap();
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
    let info = hal.open(&OpenReq { backend: backend_id::D3D11, flags: open_flags::OFFSCREEN_WINDOWS, ..Default::default() });
    assert_eq!(info.backend, backend_id::D3D11, "D3D11 HAL initialization failed");
    let sid = info.session;
    let mut w = CmdWriter::new(sid);
    w.texture_create(IMAGE, usage::STORAGE | usage::SAMPLED, format::RGBA8, 32, 32, [1, 1], Attr::default());
    w.texture_create(TARGET, usage::RENDER_TARGET | usage::SAMPLED, format::RGBA8, 32, 32, [1, 1], Attr::default());
    let colors: Vec<u8> = [[1.0f32, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]].into_iter().flatten().flat_map(f32::to_le_bytes).collect();
    w.buffer_create(COLORS, buffer_kind::STORAGE, colors.len() as u64, b.buffers[0].stride, Attr::default());
    w.buffer_update(COLORS, 0, &colors);
    let quad: Vec<u8> = [-1.0f32, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0].into_iter().flat_map(f32::to_le_bytes).collect();
    w.buffer_create(QUAD, buffer_kind::VERTEX, quad.len() as u64, 0, Attr::default());
    w.buffer_update(QUAD, 0, &quad);
    w.buffer_create(QUAD_INDEX, buffer_kind::INDEX, 12, 0, Attr::default());
    w.buffer_update(QUAD_INDEX, 0, &[0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 3, 0]);
    w.pipeline_create(KERNEL, dialect::HLSL, Attr::default(), &hlsl_kernel_pipeline_blob(&kernel, 0, 0).unwrap());
    let point = makepad_platform::makepad_hal::spirv::SpirvSampler { filter: 0, address: 1, compare: false, pixel: false };
    let code = HlslPipeline { source: SAMPLE_HLSL.into(), textures: vec![0], samplers: vec![point] }.encode();
    let none = pipeline_binding::NONE;
    w.pipeline_create(SAMPLE, dialect::HLSL, Attr::default(), &CmdWriter::pipeline_blob(PipelineDesc { outputs: 1, ..Default::default() }, &code, &[none, none, none, none, none, 1], &[16, 0, 3, 4]));
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
