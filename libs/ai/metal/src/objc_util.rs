//! Retained Objective-C handles for the Metal stream (`stream/`): strong references released on
//! drop, autorelease pools, and the shared `MTLBuffer`s tensors and weights live in. The shim and
//! the runtime keep their own private copies of the first two; this one is the stream's.
//!
//! Apple's rules, as this file applies them:
//! - Methods named `new...`, `alloc`/`init...` and `copy...` return +1: wrap with
//!   [`StrongId::from_owned`]. Everything else returns +0 (autoreleased or borrowed): wrap with
//!   [`StrongId::retain`] to keep it past the current autorelease pool.
//! - Every call that can autorelease runs inside an [`AutoreleasePool`] on the calling thread; the
//!   stream opens one around every op, commit and wait.

use crate::residency::PoolBuf;
use makepad_objc_sys::runtime::{nil, ObjcId, Object, Sel};
use makepad_objc_sys::{class, msg_send, sel, sel_impl};
use std::ffi::{c_char, c_void, CStr};
use std::ptr::NonNull;
use std::thread;
use std::time::Duration;

#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> ObjcId;
    fn MTLCopyAllDevices() -> ObjcId;
}

#[link(name = "Foundation", kind = "framework")]
extern "C" {}

/// `MTLResourceStorageModeShared` with the default CPU cache mode and hazard tracking.
pub(crate) const MTL_RESOURCE_STORAGE_MODE_SHARED: u64 = 0;

const UTF8_ENCODING: u64 = 4;

/// How often, and how far apart, the system device is asked for before giving up (the shim's
/// numbers: the device can be briefly unavailable while the window server starts).
const DEVICE_ATTEMPTS: usize = 12;
const DEVICE_RETRY_DELAY_MS: u64 = 100;

/// A strong (+1) reference to an Objective-C object, released on drop; cloning retains.
pub(crate) struct StrongId(NonNull<Object>);

impl StrongId {
    /// Takes over an object returned at +1. `None` for nil.
    ///
    /// # Safety
    /// `id` is nil or an object the caller owns a reference to, which this handle now releases.
    pub(crate) unsafe fn from_owned(id: ObjcId) -> Option<Self> {
        NonNull::new(id).map(Self)
    }

    /// Retains an object returned at +0. `None` for nil.
    ///
    /// # Safety
    /// `id` is nil or a live object.
    pub(crate) unsafe fn retain(id: ObjcId) -> Option<Self> {
        let id = NonNull::new(id)?;
        let _: ObjcId = msg_send![id.as_ptr(), retain];
        Some(Self(id))
    }

    pub(crate) fn as_id(&self) -> ObjcId {
        self.0.as_ptr()
    }
}

impl Clone for StrongId {
    fn clone(&self) -> Self {
        unsafe {
            let _: ObjcId = msg_send![self.as_id(), retain];
        }
        Self(self.0)
    }
}

impl Drop for StrongId {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.as_id(), release];
        }
    }
}

/// An `NSAutoreleasePool` for the current scope: objects autoreleased while it lives are released
/// when it drops. Pools nest; drop them in reverse order of creation (a local does).
pub(crate) struct AutoreleasePool(ObjcId);

impl AutoreleasePool {
    pub(crate) fn new() -> Self {
        Self(unsafe { msg_send![class!(NSAutoreleasePool), new] })
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let _: () = msg_send![self.0, release];
            }
        }
    }
}

/// A retained `MTLBuffer` in Shared storage, and the size it was asked for (its pool bucket, or a
/// weight's exact size). Retain and release are thread-safe and the stream touches `contents`
/// only under its serial rules (residency/), so the handle is `Send` and `Sync`.
pub(crate) struct MtlBuf {
    id: NonNull<Object>,
    bytes: usize,
}

// SAFETY: MTLBuffer objects may be retained, released and bound from any thread; their contents
// are read and written only as the stream's rules allow (`residency::Storage`).
unsafe impl Send for MtlBuf {}
unsafe impl Sync for MtlBuf {}

impl MtlBuf {
    /// A new Shared buffer of `bytes` (at least 4: Metal refuses zero), or `None` when the device
    /// cannot allocate it, its only out-of-memory signal.
    pub(crate) fn new_shared(device: ObjcId, bytes: usize) -> Option<MtlBuf> {
        let _pool = AutoreleasePool::new();
        let id: ObjcId = unsafe {
            msg_send![
                device,
                newBufferWithLength: bytes.max(4) as u64
                options: MTL_RESOURCE_STORAGE_MODE_SHARED
            ]
        };
        NonNull::new(id).map(|id| MtlBuf { id, bytes })
    }

    pub(crate) fn as_id(&self) -> ObjcId {
        self.id.as_ptr()
    }

    /// The CPU address of the buffer's first byte (Shared storage: the same memory the GPU uses).
    pub(crate) fn contents(&self) -> *mut u8 {
        let ptr: *mut c_void = unsafe { msg_send![self.as_id(), contents] };
        ptr.cast()
    }
}

impl Clone for MtlBuf {
    fn clone(&self) -> Self {
        unsafe {
            let _: ObjcId = msg_send![self.as_id(), retain];
        }
        Self { id: self.id, bytes: self.bytes }
    }
}

impl Drop for MtlBuf {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.as_id(), release];
        }
    }
}

impl PoolBuf for MtlBuf {
    fn bytes(&self) -> usize {
        self.bytes
    }
}

/// The system's Metal device, found the way the shim finds it: the default device, else the first
/// of all devices, asked for a few times before giving up.
pub(crate) fn system_device() -> Result<StrongId, String> {
    for attempt in 0..DEVICE_ATTEMPTS {
        let _pool = AutoreleasePool::new();
        if let Some(device) = unsafe { StrongId::from_owned(MTLCreateSystemDefaultDevice()) } {
            return Ok(device);
        }
        let all = unsafe { MTLCopyAllDevices() };
        if !all.is_null() {
            let count: u64 = unsafe { msg_send![all, count] };
            let first: ObjcId = if count > 0 { unsafe { msg_send![all, objectAtIndex: 0u64] } } else { nil };
            let device = unsafe { StrongId::retain(first) };
            unsafe {
                let _: () = msg_send![all, release];
            }
            if let Some(device) = device {
                return Ok(device);
            }
        }
        if attempt + 1 < DEVICE_ATTEMPTS {
            thread::sleep(Duration::from_millis(DEVICE_RETRY_DELAY_MS));
        }
    }
    Err("no Metal device (MTLCreateSystemDefaultDevice and MTLCopyAllDevices returned none)".to_string())
}

/// A new `NSString` holding `s`.
pub(crate) fn ns_string(s: &str) -> Option<StrongId> {
    unsafe {
        let alloc: ObjcId = msg_send![class!(NSString), alloc];
        if alloc.is_null() {
            return None;
        }
        let string: ObjcId = msg_send![
            alloc,
            initWithBytes: s.as_ptr() as *const c_void
            length: s.len() as u64
            encoding: UTF8_ENCODING
        ];
        StrongId::from_owned(string)
    }
}

/// The text of an `NSString` (empty for nil).
pub(crate) fn string_from_ns(string: ObjcId) -> String {
    if string.is_null() {
        return String::new();
    }
    unsafe {
        let utf8: *const c_char = msg_send![string, UTF8String];
        if utf8.is_null() {
            return String::new();
        }
        CStr::from_ptr(utf8).to_string_lossy().into_owned()
    }
}

/// An `NSError`'s localized description. Call inside an autorelease pool.
pub(crate) fn ns_error_string(error: ObjcId) -> String {
    if error.is_null() {
        return "unknown Metal error".to_string();
    }
    let description: ObjcId = unsafe { msg_send![error, localizedDescription] };
    string_from_ns(description)
}

/// Whether `object` responds to `selector` (an API newer than the oldest macOS supported).
pub(crate) fn responds_to(object: ObjcId, selector: Sel) -> bool {
    let answer: u8 = unsafe { msg_send![object, respondsToSelector: selector] };
    answer != 0
}

/// Sets an object's `label` (for Xcode's GPU captures and Metal's error reports).
pub(crate) fn set_label(object: ObjcId, label: &str) {
    if let Some(label) = ns_string(label) {
        unsafe {
            let _: () = msg_send![object, setLabel: label.as_id()];
        }
    }
}
