//! The residency core (`crate::residency`) over Metal: storage is a retained Shared `MTLBuffer`,
//! and each thread's pool lives in its own thread-local beside the stream's context, so storage
//! dropped while the context is borrowed (a temporary inside an op) still goes back to the pool.
//! The device and the stream's clock ride along with the pool, so allocating needs no context
//! borrow either: `GpuTensor::alloc` runs inside an op.

use crate::objc_util::{MtlBuf, StrongId};
use crate::residency::{Home, Pool, PoolStats, Storage, StreamClock, ThreadHome};
use std::cell::RefCell;
use std::sync::Arc;

/// Storage behind a tensor and its views (cutie-metal.md §2).
pub(crate) type MtlStorage = Storage<MtlBuf>;

/// One thread's pool and the stream id it pools for, and the device its buffers come from and the
/// stream's clock (none until the thread's stream starts).
struct ThreadPool {
    home: Home<MtlBuf>,
    device: Option<StrongId>,
    clock: Option<Arc<StreamClock>>,
}

thread_local! {
    static THREAD_POOL: RefCell<ThreadPool> = RefCell::new(ThreadPool {
        home: Home::new(Pool::from_env()),
        device: None,
        clock: None,
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

/// Makes the calling thread's pool pool for the stream with `clock`, allocating from `device`.
pub(crate) fn attach(clock: &Arc<StreamClock>, device: &StrongId) {
    with_pool(|pool| {
        pool.home.stream = clock.id();
        pool.device = Some(device.clone());
        pool.clock = Some(clock.clone());
    });
}

/// Whether the calling thread's pool has a device: its stream has started.
pub(crate) fn has_device() -> bool {
    with_pool(|pool| pool.device.is_some()).unwrap_or(false)
}

/// Storage of at least `bytes` that the GPU will write (`Pool::take_gpu`): `GpuTensor::alloc`. It
/// may be a buffer queued work still uses; the stream's serial encoder and Metal's hazard tracking
/// order the new writes after that use.
pub(crate) fn take_gpu(bytes: usize) -> Result<Arc<MtlStorage>, String> {
    with_pool(|pool| {
        let ThreadPool { home, device, .. } = pool;
        let device = device.as_ref().ok_or_else(no_stream)?.as_id();
        home.take_gpu(bytes, |n| MtlBuf::new_shared(device, n))
    })
    .unwrap_or_else(|| Err(pool_busy()))
}

/// Storage of at least `bytes` that the host writes now, no queued work using it
/// (`Pool::take_host`; `completed` is the stream's watermark after retiring). Never waits.
pub(crate) fn take_host(bytes: usize, completed: u64) -> Result<Arc<MtlStorage>, String> {
    with_pool(|pool| {
        let ThreadPool { home, device, .. } = pool;
        let device = device.as_ref().ok_or_else(no_stream)?.as_id();
        home.take_host(bytes, completed, |n| MtlBuf::new_shared(device, n))
    })
    .unwrap_or_else(|| Err(pool_busy()))
}

/// [`take_host`] against the clock's watermark as it stands (nothing retired first, so it may pass
/// over a buffer that has just finished): for host writes made inside an op, where the stream's
/// context is borrowed.
pub(crate) fn take_host_now(bytes: usize) -> Result<Arc<MtlStorage>, String> {
    let completed = with_pool(|pool| pool.clock.as_ref().map_or(0, |clock| clock.completed())).unwrap_or(0);
    take_host(bytes, completed)
}

/// Releases every idle buffer; returns the bytes released.
pub(crate) fn trim() -> usize {
    with_pool(|pool| pool.home.pool.trim()).unwrap_or(0)
}

pub(crate) fn set_cap_override(bytes: Option<usize>) {
    with_pool(|pool| pool.home.pool.set_cap_override(bytes));
}

/// The bytes the calling thread's pool holds idle.
pub(crate) fn idle_bytes() -> usize {
    with_pool(|pool| pool.home.pool.idle_bytes()).unwrap_or(0)
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
