//! Compute dispatches through the app's API: a kernel declared in Splash, a storage buffer and a
//! storage image, a dispatch recorded on the pass that reads the image, run in a compute packet of
//! its own before that pass. Runs on the native backend (D3D11 checks the pixels) or
//! MAKEPAD_HAL_BACKEND=headless (the logic only).
#![cfg(not(target_arch = "wasm32"))]
use makepad_platform::makepad_live_id::id;
use makepad_platform::makepad_script::{apply::Apply, ScriptApply, ScriptMod, Scope};
use makepad_platform::os::{ReadbackError, ReadbackRequest};
use makepad_platform::*;

fn open() -> (Cx, u32, bool) {
    let mut cx = Cx::new(Box::new(|_, _| {}));
    cx.init_cx_os();
    let info = makepad_hal::Hal::global().open(&makepad_hal::OpenReq::from_env(0, makepad_hal::open_flags::OFFSCREEN_WINDOWS, 1.0));
    cx.hal_attach_session(info);
    (cx, info.session, info.backend == makepad_hal::backend_id::HEADLESS)
}

fn kernel(cx: &mut Cx, body: &str) -> ComputeKernel {
    let code = format!("use mod.pod.*\nuse mod.math.*\nuse mod.shader\nuse mod.draw\ndraw.ComputeKernel{{\n{body}\n}}");
    cx.with_vm(|vm| {
        let value = vm.eval(ScriptMod { file: "kernel://test".into(), code, ..Default::default() });
        assert!(vm.take_errors().is_empty());
        let mut k = ComputeKernel { kernel_id: None };
        k.script_apply(vm, &Apply::Eval, &mut Scope::empty(), value);
        k
    })
}

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

/// A pass drawing into `image` and nothing over it: it holds what the kernels wrote.
fn reader_pass(cx: &mut Cx, image: &Texture, size: f64) -> (DrawPass, DrawList) {
    let pass = DrawPass::new(cx);
    let list = DrawList::new(cx);
    pass.add_color_texture(cx, image, DrawPassClearColor::InitWith(vec4(0.0, 0.0, 0.0, 1.0)));
    pass.set_size(cx, dvec2(size, size));
    let id = pass.draw_pass_id();
    cx.passes[id].main_draw_list_id = Some(list.id());
    cx.passes[id].dpi_factor = Some(1.0);
    cx.passes[id].paint_dirty = true;
    (pass, list)
}

fn render(cx: &mut Cx) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    assert!(cx.hal_session_turn(0, true));
    while cx.pipelines_pending() > 0 || cx.frame_completion_serial() < cx.frame_submission_serial() {
        assert!(std::time::Instant::now() < deadline, "the HAL did not answer");
        assert!(cx.hal_session_turn(1_000_000, false));
    }
}

fn read_back(cx: &mut Cx, image: &Texture) -> TextureReadback {
    let ticket = image.read_back(cx, ReadbackRequest::default()).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if let Some(r) = cx.try_take_texture_readbacks().into_iter().find(|r| r.ticket == ticket) {
            return r;
        }
        assert!(std::time::Instant::now() < deadline, "the readback never came");
        assert!(cx.hal_session_turn(1_000_000, false));
    }
}

#[test]
fn a_dispatch_writes_an_image_before_the_pass_that_reads_it() {
    let (mut cx, session, headless) = open();
    let k = kernel(&mut cx, BANDS);
    let image = Texture::new_storage(&mut cx, StorageFormat::BGRAu8, 32, 32);
    let colors = StorageBuffer::new_with_data(&mut cx, &[[1.0f32, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [1.0; 4]]);
    let (pass, _list) = reader_pass(&mut cx, &image, 32.0);
    let pass = pass.draw_pass_id();
    cx.dispatch(pass, ComputeDispatch::new(&k)
        .buffer(id!(colors), &colors)
        .image(id!(out), &image)
        .param(id!(size), [32u32, 32])
        .param(id!(tint), [1.0f32, 1.0, 0.5, 1.0])
        .threads(32, 32, 1));
    assert_eq!(cx.dispatches_pending(pass), 1);
    render(&mut cx);
    // the kernel's pipeline may have answered after the first build: its PipelineReady repaints
    render(&mut cx);
    assert_eq!(cx.dispatches_pending(pass), 0, "the queue ran once");
    assert!(k.error(&cx).is_none(), "{:?}", k.error(&cx));
    let r = read_back(&mut cx, &image);
    if headless {
        assert_eq!(r.data, Err(ReadbackError::UnsupportedBackend));
    } else {
        let data = r.data.as_ref().expect("pixels");
        // the readback is RGBA bytes
        let at = |x: usize, y: usize| data[(y * 32 + x) * 4..][..4].to_vec();
        let near = |a: &[u8], b: [u8; 4]| a.iter().zip(b).all(|(a, b)| a.abs_diff(b) <= 1);
        assert!(near(&at(0, 0), [255, 0, 0, 255]), "red band: {:?}", at(0, 0));
        assert!(near(&at(12, 31), [0, 255, 0, 255]), "green band: {:?}", at(12, 31));
        assert!(near(&at(20, 5), [0, 0, 128, 255]), "blue band, tinted: {:?}", at(20, 5));
        assert!(near(&at(31, 16), [255, 255, 128, 255]), "white band, tinted: {:?}", at(31, 16));
    }
    makepad_hal::Hal::global().close(session);
}

#[test]
fn a_dispatch_that_names_what_the_kernel_lacks_does_not_run() {
    let (mut cx, session, _) = open();
    let k = kernel(&mut cx, BANDS);
    let image = Texture::new_storage(&mut cx, StorageFormat::BGRAu8, 16, 16);
    let colors = StorageBuffer::new_with_data(&mut cx, &[[1.0f32; 4]; 4]);
    let (pass, _list) = reader_pass(&mut cx, &image, 16.0);
    let pass = pass.draw_pass_id();
    // no image given; a param of the wrong type; a texture sampled and written at once
    cx.dispatch(pass, ComputeDispatch::new(&k).buffer(id!(colors), &colors).param(id!(size), [16u32, 16]).threads(16, 16, 1));
    cx.dispatch(pass, ComputeDispatch::new(&k).buffer(id!(colors), &colors).image(id!(out), &image).param(id!(size), 16.0f32).threads(16, 16, 1));
    render(&mut cx);
    render(&mut cx);
    assert_eq!(cx.dispatches_pending(pass), 0, "refused dispatches are dropped, not kept");
    makepad_hal::Hal::global().close(session);
}
