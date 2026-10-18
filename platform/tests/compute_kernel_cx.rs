//! A compute kernel declared in Splash (`mod.draw.ComputeKernel{ ... }`) compiles for the backend
//! the HAL runs when it is applied, reports its bindings by name, and gets a pipeline beside the
//! draw shaders'. Runs on the native backend, or MAKEPAD_HAL_BACKEND=headless.
#![cfg(not(target_arch = "wasm32"))]
use makepad_platform::makepad_live_id::id;
use makepad_platform::makepad_script::{apply::Apply, ScriptApply, ScriptMod, Scope};
use makepad_platform::*;

fn open() -> (Cx, u32) {
    let mut cx = Cx::new(Box::new(|_, _| {}));
    cx.init_cx_os();
    let info = makepad_hal::Hal::global().open(&makepad_hal::OpenReq::from_env(0, makepad_hal::open_flags::OFFSCREEN_WINDOWS, 1.0));
    cx.hal_attach_session(info);
    (cx, info.session)
}

/// Kernel `body` applied to a ComputeKernel, as a widget's `#[live] kernel` is.
fn kernel(cx: &mut Cx, body: &str) -> ComputeKernel {
    let code = format!("use mod.pod.*\nuse mod.math.*\nuse mod.shader\nuse mod.draw\ndraw.ComputeKernel{{\n{body}\n}}");
    cx.with_vm(|vm| {
        let value = vm.eval(ScriptMod { file: "kernel://test".into(), code, ..Default::default() });
        let errors = vm.take_errors();
        assert!(errors.is_empty(), "{errors:?}");
        let mut k = ComputeKernel { kernel_id: None };
        k.script_apply(vm, &Apply::Eval, &mut Scope::empty(), value);
        k
    })
}

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

fn turn_until_pipelines_answer(cx: &mut Cx) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    assert!(cx.hal_session_turn(0, true));
    while cx.pipelines_pending() > 0 {
        assert!(std::time::Instant::now() < deadline, "the kernel's pipeline never answered");
        assert!(cx.hal_session_turn(1_000_000, false));
    }
}

#[test]
fn a_splash_kernel_compiles_when_applied_and_gets_a_pipeline() {
    let (mut cx, session) = open();
    let k = kernel(&mut cx, BANDS);
    assert!(k.error(&cx).is_none(), "{:?}", k.error(&cx));
    let b = k.bindings(&cx).expect("bindings for a slot-bound backend");
    assert_eq!(b.workgroup, [8, 8, 1]);
    assert!(b.param(id!(size)).is_some() && b.param(id!(tint)).is_some());
    assert_eq!((b.images.len(), b.buffers.len(), b.buffers[0].stride), (1, 1, 16));
    // The same kernel applied again is the same kernel, one pipeline.
    let again = kernel(&mut cx, BANDS);
    assert_eq!(again.kernel_id, k.kernel_id);
    turn_until_pipelines_answer(&mut cx);
    assert!(k.is_ready(&cx), "{:?}", k.error(&cx));
    makepad_hal::Hal::global().close(session);
}

#[test]
fn a_kernel_that_does_not_compile_says_why_and_never_runs() {
    let (mut cx, session) = open();
    let k = kernel(&mut cx, "src: shader.buffer(f32)\nmain: fn() { self.src[0] = 1.0 }");
    assert!(k.error(&cx).is_some_and(|e| e.contains("will NOT run")), "{:?}", k.error(&cx));
    assert!(k.bindings(&cx).is_none());
    assert!(!k.is_ready(&cx));
    makepad_hal::Hal::global().close(session);
}
