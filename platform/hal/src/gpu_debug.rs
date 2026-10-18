//! GPU debugging aids for frame captures and GPU profilers (Xcode, RenderDoc, Nsight, PIX).
//!
//! Every pass is labelled with its name (`Pass::name`) on the backends that can (a Metal encoder's
//! label, a D3D11 annotation event). Per-draw labels, the shader each draw runs, cost a string per
//! draw and are on only with `MAKEPAD_GPU_LABELS` set, [`set_draw_labels`], or while a capture
//! is being taken.
//!
//! A capture of the next frames is asked for with [`request_capture`] (any thread). Metal writes
//! a `.gputrace` into the directory given, which Xcode opens (with its shader profiler);
//! D3D11, when the process runs under RenderDoc, has RenderDoc capture them. Each result is
//! printed to stderr, `[gpu-capture] ...`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static DRAW_LABELS: AtomicBool = AtomicBool::new(false);
static PENDING: AtomicBool = AtomicBool::new(false);
static REQUEST: Mutex<Option<(u32, PathBuf)>> = Mutex::new(None);

/// Whether each draw is labelled with its shader: `MAKEPAD_GPU_LABELS` set, or
/// [`set_draw_labels`]. One relaxed load after the first call.
pub fn draw_labels() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("MAKEPAD_GPU_LABELS").is_some()) || DRAW_LABELS.load(Ordering::Relaxed)
}

pub fn set_draw_labels(on: bool) {
    DRAW_LABELS.store(on, Ordering::Relaxed);
}

/// Capture the next `frames` window frames (at least one) into `dir`. A request made while
/// another waits replaces it; one made during a capture waits for it to end.
pub fn request_capture(frames: u32, dir: impl Into<PathBuf>) {
    *REQUEST.lock().unwrap_or_else(|e| e.into_inner()) = Some((frames.max(1), dir.into()));
    PENDING.store(true, Ordering::Release);
}

/// The waiting request, taken (backends). One relaxed load when there is none.
pub fn take_capture_request() -> Option<(u32, PathBuf)> {
    if !PENDING.load(Ordering::Acquire) {
        return None;
    }
    PENDING.store(false, Ordering::Release);
    REQUEST.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// A file name for a capture taken now: `<stem>-<unix seconds>.<ext>`.
pub fn capture_path(dir: &std::path::Path, ext: &str) -> PathBuf {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let stem = std::env::current_exe().ok().and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned())).unwrap_or_else(|| "frames".into());
    dir.join(format!("{stem}-{secs}.{ext}"))
}

/// A draw's or dispatch's label: its shader's name and where its script is (`file:line:col`,
/// the file without its directories), since generated shaders and kernels share names.
pub fn shader_label(name: &str, source: &str) -> String {
    let file = source.rsplit(['/', '\\']).next().unwrap_or("");
    match (name.is_empty(), file.is_empty()) {
        (_, true) => if name.is_empty() { "draw".into() } else { name.into() },
        (true, false) => file.into(),
        (false, false) => format!("{name} {file}"),
    }
}
