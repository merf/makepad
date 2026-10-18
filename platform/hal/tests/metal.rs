//! Native Metal coverage (macOS, the system's default device, offscreen windows): compute passes
//! as tests/d3d11.rs runs them, with the kernels written by hand in MSL.
#![cfg(target_os = "macos")]

use makepad_hal::*;
use std::time::{Duration, Instant};

const IMAGE: Id = Id::new(3, 1);
const TEX: Id = Id::new(4, 1);
const COLORS: Id = Id::new(3, 1);
const COUNT: Id = Id::new(4, 1);
const QUAD: Id = Id::new(5, 1);
const QUAD_INDEX: Id = Id::new(6, 1);
const BANDS: Id = Id::new(3, 1);
const PATCH: Id = Id::new(4, 1);
const SAMPLE: Id = Id::new(5, 1);
const SPIN: Id = Id::new(6, 1);
const SCRATCH: Id = Id::new(7, 1);

/// Four vertical bands from a read-only buffer, scaled by the params, into a storage image; each
/// invocation counts itself into a written buffer. The shape the shader compiler prints
/// (shader_ir_metal::metal_compute): buffers at their table registers, params at buffer(30).
const BANDS_MSL: &str = "#include <metal_stdlib>
using namespace metal;
struct Params { float4 tint; };
kernel void kernel_main(constant Params *p [[buffer(30)]], const device float4 *colors [[buffer(0)]], device uint *count [[buffer(1)]],
    texture2d<float, access::write> image [[texture(0)]], uint2 id [[thread_position_in_grid]]) {
    uint w = image.get_width(), h = image.get_height();
    if (id.x >= w || id.y >= h) return;
    image.write(colors[id.x * 4 / w] * p->tint, id);
    atomic_fetch_add_explicit((device atomic_uint *)&count[0], 1u, memory_order_relaxed);
}
";

/// Reads the count the first kernel wrote and marks texel (0, 0): blue when every invocation
/// counted and this dispatch is one group.
const PATCH_MSL: &str = "#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(const device uint *count [[buffer(0)]], texture2d<float, access::write> image [[texture(0)]], uint3 groups [[threadgroups_per_grid]]) {
    image.write(count[0] == 1024u && groups.x == 1u ? float4(0.0, 0.0, 1.0, 1.0) : float4(1.0, 0.0, 1.0, 1.0), uint2(0, 0));
}
";

/// Keeps the GPU busy for a while, so the frame is still running when the next one is sent.
const SPIN_MSL: &str = "#include <metal_stdlib>
using namespace metal;
kernel void kernel_main(device float *scratch [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    float x = float(id);
    for (uint i = 0; i < 200000u; i++) { x = fract(x * 1.0001 + 0.37); }
    scratch[id] = x;
}
";

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

fn open() -> (std::sync::Arc<Hal>, HalInfo) {
    let hal = Hal::new();
    let info = hal.open(&OpenReq { backend: backend_id::METAL, flags: open_flags::OFFSCREEN_WINDOWS, ..Default::default() });
    assert_eq!(info.backend, backend_id::METAL, "Metal HAL initialization failed");
    assert_eq!(info.dialect, dialect::MSL);
    (hal, info)
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

fn frame_done(hal: &Hal, sid: u32, frame: u64) {
    until(hal, sid, |e| matches!(e, HalEvent::FrameDone { frame: done, .. } if done == frame));
}

fn pixels(hal: &Hal, sid: u32, req: u32) -> Vec<u8> {
    let mut out = Vec::new();
    until(hal, sid, |e| match e {
        HalEvent::Readback { req: r, status, pixels, .. } if r == req => {
            assert_eq!(status, readback::OK);
            out = pixels.to_vec();
            true
        }
        _ => false,
    });
    out
}

fn kernel_blob(source: &str, table: &[u32]) -> Vec<u8> {
    CmdWriter::pipeline_blob(PipelineDesc { kind: pipeline_kind::COMPUTE, ..Default::default() }, source.as_bytes(), table, &[])
}

/// The kernels, the image they write, the buffers, and the draw that samples the image.
fn compute_resources(hal: &Hal, sid: u32) {
    let mut w = CmdWriter::new(sid);
    w.texture_create(IMAGE, usage::STORAGE | usage::SAMPLED, format::RGBA8, 32, 32, [1, 1], Attr::default());
    // BGRA8: what a Metal pipeline is built for when it declares no format (RGBA8 is format 0)
    w.texture_create(TEX, usage::RENDER_TARGET | usage::SAMPLED, format::BGRA8, 32, 32, [1, 1], Attr::default());
    let colors: Vec<u8> = [[1.0f32, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]].into_iter().flatten().flat_map(f32::to_le_bytes).collect();
    w.buffer_create(COLORS, buffer_kind::STORAGE, colors.len() as u64, 16, Attr::default());
    w.buffer_update(COLORS, 0, &colors);
    w.buffer_create(COUNT, buffer_kind::STORAGE, 16, 4, Attr::default());
    w.buffer_create(SCRATCH, buffer_kind::STORAGE, 4 * 4096, 4, Attr::default());
    let quad: Vec<u8> = [-1.0f32, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0].into_iter().flat_map(f32::to_le_bytes).collect();
    w.buffer_create(QUAD, buffer_kind::VERTEX, quad.len() as u64, 0, Attr::default());
    w.buffer_update(QUAD, 0, &quad);
    w.buffer_create(QUAD_INDEX, buffer_kind::INDEX, 12, 0, Attr::default());
    w.buffer_update(QUAD_INDEX, 0, &[0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 3, 0]);
    let none = compute_binding::NONE;
    w.pipeline_create(BANDS, dialect::MSL, Attr::default(), &kernel_blob(BANDS_MSL, &[30, 0, 1, 2, 0, 16, 1 | compute_binding::WRITABLE, 4, 8, 8, 1]));
    w.pipeline_create(PATCH, dialect::MSL, Attr::default(), &kernel_blob(PATCH_MSL, &[none, 0, 1, 1, 0, 4, 1, 1, 1]));
    w.pipeline_create(SPIN, dialect::MSL, Attr::default(), &kernel_blob(SPIN_MSL, &[none, 0, 0, 1, 1 | compute_binding::WRITABLE, 4, 64, 1, 1]));
    let none = pipeline_binding::NONE;
    let desc = PipelineDesc { outputs: 1, color_formats: [format::BGRA8, 0, 0, 0], ..Default::default() };
    w.pipeline_create(SAMPLE, dialect::MSL, Attr::default(), &CmdWriter::pipeline_blob(desc, SAMPLE_MSL.as_bytes(), &[none, none, none, none, none, 1], &[16, 0, 3, 4]));
    hal.submit(&w.buf());
    let mut ready = Vec::new();
    until(hal, sid, |e| {
        if let HalEvent::PipelineReady { pipeline } = e {
            ready.push(pipeline);
        }
        [BANDS, PATCH, SAMPLE, SPIN].iter().all(|p| ready.contains(p))
    });
    assert_eq!(hal.with_world(|w| w.session(sid).unwrap().stats.bad_args), 0);
}

/// One frame: the counter cleared, the two dispatches, then the pass that samples the image.
/// `matching`: the second dispatch reads the counter, else a buffer of other elements.
fn compute_then_draw(hal: &Hal, sid: u32, frame: u64, matching: bool) {
    submit_compute_then_draw(hal, sid, frame, matching);
    frame_done(hal, sid, frame);
}

fn submit_compute_then_draw(hal: &Hal, sid: u32, frame: u64, matching: bool) {
    submit_frame(hal, sid, frame, matching, false);
}

/// `spin`: a slow dispatch first, after the counter was cleared.
fn submit_frame(hal: &Hal, sid: u32, frame: u64, matching: bool, spin: bool) {
    let mut w = CmdWriter::new(sid);
    w.buffer_update(COUNT, 0, &[0; 16]);
    let images = w.put_ids(&[IMAGE]);
    let bands_buffers = w.put_ids(&[COLORS, COUNT]);
    let patch_buffers = w.put_ids(&[if matching { COUNT } else { COLORS }]);
    let tex = w.put_ids(&[IMAGE]);
    let scratch = w.put_ids(&[SCRATCH]);
    let tint = [1.0f32, 1.0, 0.5, 1.0];
    w.frame_begin(frame, FrameMeta::default());
    w.pass(Pass::compute(0));
    if spin {
        w.dispatch(Dispatch { pipeline: SPIN, groups: [64, 1, 1], buffers: scratch, ..Default::default() });
    }
    w.dispatch(Dispatch { pipeline: BANDS, groups: [4, 4, 1], uniforms: Slice::of(&tint), images, buffers: bands_buffers, ..Default::default() });
    w.dispatch(Dispatch { pipeline: PATCH, groups: [1, 1, 1], images, buffers: patch_buffers, ..Default::default() });
    w.pass(Pass { color: [TEX, Id::NONE, Id::NONE, Id::NONE], load: [pass_load::CLEAR; 5], viewport: [0.0, 0.0, 32.0, 32.0], dpi: 1.0, ..Default::default() });
    w.draw(Draw { pipeline: SAMPLE, vb: QUAD, ib: QUAD_INDEX, index: [0, 6], inst: Inst::inline(Slice::EMPTY, 1), tex, scissor: [f32::NAN; 4], ..Default::default() });
    w.frame_end(TEX, target::TEXTURE);
    hal.submit(&w.buf());
}

/// Milestone 16's first proof on Metal: a kernel writes a texture that a draw in a later pass of
/// the same frame samples. On the way: a read-only buffer, params in a constant block, an atomic
/// count into a written buffer that the next dispatch reads, the group count, the workgroup size
/// from the binding table, and the counter cleared on the queue between frames (the frame before
/// may still be writing it when the clear arrives).
#[test]
fn a_kernel_writes_an_image_a_draw_then_samples() {
    let (hal, info) = open();
    let sid = info.session;
    compute_resources(&hal, sid);
    for frame in 1..=2 {
        compute_then_draw(&hal, sid, frame, true);
        let st = hal.with_world(|w| w.session(sid).unwrap().stats);
        assert_eq!((st.dispatches, st.skipped_dispatches, st.bad_args, st.stale_ids, st.skipped_draws), (2 * frame, 0, 0, 0, 0));
    }
    let mut w = CmdWriter::new(sid);
    w.texture_readback(TEX, 1, [0, 0, 32, 32]);
    hal.submit(&w.buf());
    let image = pixels(&hal, sid, 1);
    // read back as RGBA
    let at = |x: usize, y: usize| image[(y * 32 + x) * 4..][..4].to_vec();
    // the bands, the blue channel through the params' half
    let half = at(20, 31)[2];
    assert!((127..=128).contains(&half), "{:?}", at(20, 31));
    assert_eq!(at(4, 9), [255, 0, 0, 255]);
    assert_eq!(at(12, 20), [0, 255, 0, 255]);
    assert_eq!(at(20, 31), [0, 0, half, 255]);
    assert_eq!(at(28, 0), [255, 255, half, 255]);
    // the second dispatch saw all 1024 counts, in a dispatch of one group
    assert_eq!(at(0, 0), [0, 0, 255, 255]);

    // frames sent back to back: each clear of the counter reaches the GPU after the dispatches
    // of the frame before (still running when it is sent) and before its own
    for frame in 3..=6 {
        submit_frame(&hal, sid, frame, true, true);
    }
    frame_done(&hal, sid, 6);
    let mut w = CmdWriter::new(sid);
    w.texture_readback(TEX, 2, [0, 0, 32, 32]);
    hal.submit(&w.buf());
    let image = pixels(&hal, sid, 2);
    assert_eq!(image[..4], [0, 0, 255, 255]);

    // a buffer whose elements are not the kernel's is refused, not bound
    compute_then_draw(&hal, sid, 7, false);
    let st = hal.with_world(|w| w.session(sid).unwrap().stats);
    assert_eq!((st.dispatches, st.skipped_dispatches, st.bad_args), (17, 1, 1));
    assert!(hal.close(sid));
}

/// A kernel's table without its workgroup size is refused: Metal could not dispatch it.
#[test]
fn a_kernel_without_its_workgroup_size_fails() {
    let (hal, info) = open();
    let sid = info.session;
    let mut w = CmdWriter::new(sid);
    w.pipeline_create(PATCH, dialect::MSL, Attr::default(), &kernel_blob(PATCH_MSL, &[compute_binding::NONE, 0, 1, 1, 0, 4]));
    hal.submit(&w.buf());
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut store = EventStore::new(sid, 64, 65536);
    let mut failed = None;
    while failed.is_none() {
        assert!(Instant::now() < deadline, "no PipelineFailed");
        let mut buf = store.buf();
        let n = hal.pump(10_000_000, &mut buf);
        for (_, event) in store.reader(n, buf.arena_len) {
            if let HalEvent::PipelineFailed { log, .. } = event {
                failed = Some(String::from_utf8_lossy(log).to_string());
            }
        }
    }
    assert!(failed.unwrap().contains("workgroup"));
    assert!(hal.close(sid));
}
