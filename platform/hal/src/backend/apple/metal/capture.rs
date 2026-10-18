//! Frame captures on request (`gpu_debug::request_capture`): MTLCaptureManager writes the next
//! window frames' command buffers into a `.gputrace`, which Xcode opens (its shader profiler,
//! per-pass and per-draw costs, the labels `frame.rs` sets). Metal allows a capture outside
//! Xcode only with `MTL_CAPTURE_ENABLED=1` in the environment when the process starts.

use super::Metal;
use makepad_apple_sys::*;
use std::path::PathBuf;

/// `MTLCaptureDestinationGPUTraceDocument`
const GPU_TRACE_DOCUMENT: i64 = 2;

/// A capture being taken: the window frames still to come and where it goes.
pub(super) struct GpuCapture {
    frames: u32,
    path: PathBuf,
}

impl Metal {
    /// Starts a waiting capture request, on a command buffer of its own (the one open now is
    /// committed first, so the capture holds whole frames).
    pub(super) fn poll_gpu_capture(&mut self) {
        if self.gpu_capture.is_some() {
            return;
        }
        let Some((frames, dir)) = crate::gpu_debug::take_capture_request() else { return };
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("[gpu-capture] cannot make {}: {e}", dir.display());
            return;
        }
        let path = crate::gpu_debug::capture_path(&dir, "gputrace");
        self.commit_batch();
        unsafe {
            let manager: ObjcId = msg_send![class!(MTLCaptureManager), sharedCaptureManager];
            let supported: bool = msg_send![manager, supportsDestination: GPU_TRACE_DOCUMENT];
            if !supported {
                eprintln!("[gpu-capture] Metal refuses to capture to a file: launch with MTL_CAPTURE_ENABLED=1 in the environment (or run from Xcode)");
                return;
            }
            let descriptor: ObjcId = msg_send![class!(MTLCaptureDescriptor), new];
            let () = msg_send![descriptor, setCaptureObject: self.dev.queue];
            let () = msg_send![descriptor, setDestination: GPU_TRACE_DOCUMENT];
            let url: ObjcId = msg_send![class!(NSURL), fileURLWithPath: str_to_nsstring(&path.to_string_lossy())];
            let () = msg_send![descriptor, setOutputURL: url];
            let mut error: ObjcId = nil;
            let started: bool = msg_send![manager, startCaptureWithDescriptor: descriptor error: &mut error];
            let () = msg_send![descriptor, release];
            if !started {
                let why = if error == nil { "no reason given".to_string() } else { nsstring_to_string(msg_send![error, localizedDescription]) };
                eprintln!("[gpu-capture] the capture did not start: {why}");
                return;
            }
        }
        crate::gpu_debug::set_draw_labels(true);
        eprintln!("[gpu-capture] capturing {frames} frame(s) into {}", path.display());
        self.gpu_capture = Some(GpuCapture { frames, path });
    }

    /// A window frame was encoded: the capture ends after its last one (committed first).
    pub(super) fn gpu_capture_frame(&mut self) {
        let Some(capture) = self.gpu_capture.as_mut() else { return };
        capture.frames = capture.frames.saturating_sub(1);
        if capture.frames > 0 {
            return;
        }
        let capture = self.gpu_capture.take().expect("capture");
        self.commit_batch();
        unsafe {
            let manager: ObjcId = msg_send![class!(MTLCaptureManager), sharedCaptureManager];
            let () = msg_send![manager, stopCapture];
        }
        if std::env::var_os("MAKEPAD_GPU_LABELS").is_none() {
            crate::gpu_debug::set_draw_labels(false);
        }
        eprintln!("[gpu-capture] wrote {} (open it in Xcode)", capture.path.display());
    }
}
