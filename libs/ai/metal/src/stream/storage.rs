//! The residency core (`crate::residency`) over Metal: storage is a retained Shared `MTLBuffer`,
//! and each thread's pool lives in its own thread-local beside the stream's context, so storage
//! dropped while the context is borrowed (a temporary inside an op) still goes back to the pool.
//! The device rides along with the pool, so allocating needs no context borrow either.

use crate::objc_util::{MtlBuf, StrongId};
use crate::residency::{Home, Pool, PoolStats, Storage, ThreadHome};
use std::cell::RefCell;
use std::sync::Arc;

/// Storage behind a tensor and its views (cutie-metal.md §2).
pub(crate) type MtlStorage = Storage<MtlBuf>;

/// One thread's pool and the stream id it pools for, and the device its buffers come from (none
/// until the thread's stream starts).
struct ThreadPool {
    home: Home<MtlBuf>,
    device: Option<StrongId>,
}

thread_local! {
    static THREAD_POOL: RefCell<ThreadPool> = RefCell::new(ThreadPool {
        home: Home::new(Pool::from_env()),
        device: None,
    });
}

impl ThreadHome for MtlBuf {
    fn with_home<R>(f: impl FnOnce(&mut Home<Self>) -> R) -> Option<R> {
        THREAD_POOL
            .try_with(|pool| pool.try_borrow_mut().ok().map(|mut pool| f(&mut pool.home)))
            .ok()
            .flatten()
    }
}

/// Runs `f` on the calling thread's pool. `None` while it is borrowed or torn down.
fn with_pool<R>(f: impl FnOnce(&mut ThreadPool) -> R) -> Option<R> {
    THREAD_POOL.try_with(|pool| pool.try_borrow_mut().ok().map(|mut pool| f(&mut pool))).ok().flatten()
}

/// Makes the calling thread's pool pool for stream `stream`, allocating from `device`.
pub(crate) fn attach(stream: u64, device: &StrongId) {
    with_pool(|pool| {
        pool.home.stream = stream;
        pool.device = Some(device.clone());
    });
}

/// Storage of at least `bytes` that the GPU will write (`Pool::take_gpu`): `GpuTensor::alloc` in
/// the resident binding (step 12). The host binding seeds every output from its tensor, so it
/// takes `take_host` buffers only.
#[allow(dead_code)]
pub(crate) fn take_gpu(bytes: usize) -> Result<Arc<MtlStorage>, String> {
    with_pool(|pool| {
        let ThreadPool { home, device } = pool;
        let device = device.as_ref().ok_or_else(no_stream)?.as_id();
        home.take_gpu(bytes, |n| MtlBuf::new_shared(device, n))
    })
    .unwrap_or_else(|| Err(pool_busy()))
}

/// Storage of at least `bytes` that the host writes now, no queued work using it
/// (`Pool::take_host`; `completed` is the stream's watermark after retiring). Never waits.
pub(crate) fn take_host(bytes: usize, completed: u64) -> Result<Arc<MtlStorage>, String> {
    with_pool(|pool| {
        let ThreadPool { home, device } = pool;
        let device = device.as_ref().ok_or_else(no_stream)?.as_id();
        home.take_host(bytes, completed, |n| MtlBuf::new_shared(device, n))
    })
    .unwrap_or_else(|| Err(pool_busy()))
}

/// Releases every idle buffer; returns the bytes released.
pub(crate) fn trim() -> usize {
    with_pool(|pool| pool.home.pool.trim()).unwrap_or(0)
}

pub(crate) fn set_cap_override(bytes: Option<usize>) {
    with_pool(|pool| pool.home.pool.set_cap_override(bytes));
}

/// The pool's counters, zeroed afterwards when `reset`.
pub(crate) fn stats(reset: bool) -> PoolStats {
    with_pool(|pool| pool.home.pool.stats(reset)).unwrap_or_default()
}

fn no_stream() -> String {
    "Metal stream: this thread's pool has no device (its stream did not start)".to_string()
}

fn pool_busy() -> String {
    "Metal stream: this thread's buffer pool is busy or gone".to_string()
}
