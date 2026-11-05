//! Frame captures on request (`gpu_debug::request_capture`) through RenderDoc's in-application
//! API, when the process runs under RenderDoc (launched from it, or with its DLL injected):
//! RenderDoc captures the next window frames into the directory asked for, which RenderDoc opens.
//! Without RenderDoc the request is answered with how to capture by hand (RenderDoc, Nsight
//! Graphics, PIX), each of which shows the passes by the annotation events `frame.rs` opens.

use super::D3d11;
use makepad_windows_sys::core::{PCSTR, PCWSTR};
use makepad_windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use std::ffi::c_void;
use std::sync::OnceLock;

/// `eRENDERDOC_API_Version_1_1_2`: the first with `TriggerMultiFrameCapture`.
const RENDERDOC_API_1_1_2: u32 = 10102;
/// Slots of `RENDERDOC_API_1_1_2`'s function table (renderdoc_app.h).
const SET_CAPTURE_FILE_PATH_TEMPLATE: usize = 11;
const TRIGGER_MULTI_FRAME_CAPTURE: usize = 22;

/// RenderDoc's function table, when the process runs under RenderDoc.
fn renderdoc() -> Option<&'static [*const c_void; 23]> {
    struct Api(Option<&'static [*const c_void; 23]>);
    // SAFETY: a table of function pointers RenderDoc keeps for the life of the process
    unsafe impl Send for Api {}
    unsafe impl Sync for Api {}
    static API: OnceLock<Api> = OnceLock::new();
    API.get_or_init(|| unsafe {
        let name: Vec<u16> = "renderdoc.dll".encode_utf16().chain(Some(0)).collect();
        let Ok(module) = GetModuleHandleW(Some(PCWSTR(name.as_ptr()))) else { return Api(None) };
        let Some(get_api) = GetProcAddress(module, PCSTR(b"RENDERDOC_GetAPI\0".as_ptr())) else { return Api(None) };
        let get_api: extern "C" fn(u32, *mut *mut c_void) -> i32 = std::mem::transmute(get_api);
        let mut table: *mut c_void = std::ptr::null_mut();
        if get_api(RENDERDOC_API_1_1_2, &mut table) != 1 || table.is_null() {
            return Api(None);
        }
        Api(Some(&*(table as *const [*const c_void; 23])))
    })
    .0
}

impl D3d11 {
    /// Hands a waiting capture request to RenderDoc, or says how to capture without it.
    pub(super) fn poll_gpu_capture(&mut self) {
        let Some((frames, dir)) = crate::gpu_debug::take_capture_request() else { return };
        let Some(api) = renderdoc() else {
            eprintln!("[gpu-capture] not running under RenderDoc: launch from RenderDoc (or Nsight Graphics, PIX) and capture there");
            return;
        };
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("[gpu-capture] cannot make {}: {e}", dir.display());
            return;
        }
        let template = crate::gpu_debug::capture_path(&dir, "rdc").with_extension("");
        let template = std::ffi::CString::new(template.to_string_lossy().into_owned()).unwrap_or_default();
        unsafe {
            let set_path: extern "C" fn(*const std::ffi::c_char) = std::mem::transmute(api[SET_CAPTURE_FILE_PATH_TEMPLATE]);
            let trigger: extern "C" fn(u32) = std::mem::transmute(api[TRIGGER_MULTI_FRAME_CAPTURE]);
            set_path(template.as_ptr());
            trigger(frames);
        }
        crate::gpu_debug::set_draw_labels(true);
        eprintln!("[gpu-capture] RenderDoc captures the next {frames} frame(s) as {}_frame*.rdc", template.to_string_lossy());
    }
}
