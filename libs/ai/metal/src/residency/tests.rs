//! The residency rules over fake buffers and command buffers, so they run on any OS.

use super::*;
use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const MB: usize = 1024 * 1024;

/// A fake device buffer: its size, a number, and the device's list of released numbers.
struct FakeBuf {
    bytes: usize,
    id: usize,
    released: Arc<Mutex<Vec<usize>>>,
}

impl Drop for FakeBuf {
    fn drop(&mut self) {
        self.released.lock().unwrap().push(self.id);
    }
}

impl PoolBuf for FakeBuf {
    fn bytes(&self) -> usize {
        self.bytes
    }
}

thread_local! {
    static HOME: RefCell<Option<Home<FakeBuf>>> = const { RefCell::new(None) };
}

impl ThreadHome for FakeBuf {
    fn with_home<R>(f: impl FnOnce(&mut Home<Self>) -> R) -> Option<R> {
        HOME.try_with(|home| {
            let mut home = home.try_borrow_mut().ok()?;
            home.as_mut().map(f)
        })
        .ok()
        .flatten()
    }
}

/// Gives the calling thread a home for stream `stream` with an idle cap of `cap` bytes.
fn set_home(stream: u64, cap: usize) {
    HOME.with(|home| *home.borrow_mut() = Some(Home { stream, pool: Pool::new(cap) }));
}

/// Runs `f` on the calling thread's home, which must exist.
fn home<R>(f: impl FnOnce(&mut Home<FakeBuf>) -> R) -> R {
    FakeBuf::with_home(f).expect("this thread has a home")
}

/// A fake device: numbers its buffers from 1, can be told to fail the next allocations, and
/// records which buffers were released.
struct Device {
    allocated: Cell<usize>,
    fail_next: Cell<usize>,
    released: Arc<Mutex<Vec<usize>>>,
}

impl Device {
    fn new() -> Self {
        Self {
            allocated: Cell::new(0),
            fail_next: Cell::new(0),
            released: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn alloc(&self, bytes: usize) -> Option<FakeBuf> {
        assert!(bytes >= pool::MIN_BUCKET, "the pool asks only for whole buckets");
        if self.fail_next.get() > 0 {
            self.fail_next.set(self.fail_next.get() - 1);
            return None;
        }
        self.allocated.set(self.allocated.get() + 1);
        Some(FakeBuf {
            bytes,
            id: self.allocated.get(),
            released: self.released.clone(),
        })
    }

    fn released(&self) -> Vec<usize> {
        self.released.lock().unwrap().clone()
    }
}

/// A fake command buffer: pending until the test finishes it, or until waited on.
#[derive(Clone)]
struct FakeCb(Arc<FakeCbState>);

struct FakeCbState {
    status: Mutex<CbStatus>,
    waits: AtomicUsize,
    gpu_ms: f64,
}

impl FakeCb {
    fn pending(gpu_ms: f64) -> Self {
        Self(Arc::new(FakeCbState {
            status: Mutex::new(CbStatus::Pending),
            waits: AtomicUsize::new(0),
            gpu_ms,
        }))
    }

    fn finish(&self, status: CbStatus) {
        *self.0.status.lock().unwrap() = status;
    }

    fn waits(&self) -> usize {
        self.0.waits.load(Ordering::SeqCst)
    }
}

impl CommandBuf for FakeCb {
    fn status(&self) -> CbStatus {
        self.0.status.lock().unwrap().clone()
    }

    fn wait_until_completed(&self) {
        self.0.waits.fetch_add(1, Ordering::SeqCst);
        let mut status = self.0.status.lock().unwrap();
        if *status == CbStatus::Pending {
            *status = CbStatus::Completed;
        }
    }

    fn gpu_ms(&self) -> f64 {
        self.0.gpu_ms
    }
}

/// Pushes a pending command buffer `serial` onto `ring` and returns a handle to finish it.
fn commit(ring: &mut Ring<FakeCb>, serial: u64) -> FakeCb {
    let cb = FakeCb::pending(1.0);
    ring.make_room();
    ring.push(serial, cb.clone(), format!("op {serial}"));
    cb
}

#[test]
fn bucket_rounds_as_cuda_does() {
    assert_eq!(bucket(0), 256);
    assert_eq!(bucket(1), 256);
    assert_eq!(bucket(256), 256);
    assert_eq!(bucket(257), 512);
    assert_eq!(bucket(3 * MB), 4 * MB);
    assert_eq!(bucket(256 * MB), 256 * MB);
    assert_eq!(bucket(256 * MB + 1), 512 * MB);
    assert_eq!(bucket(700 * MB), 768 * MB);
    assert_eq!(bucket(1024 * MB), 1024 * MB);
}

#[test]
fn take_gpu_is_the_best_fit_below_twice_the_bucket() {
    let device = Device::new();
    let mut pool = Pool::new(usize::MAX);
    for bytes in [4096, 1024, 2048] {
        let (buf, _) = pool.take_gpu(bytes, |b| device.alloc(b)).unwrap();
        pool.release(buf, 0);
    }
    assert_eq!(pool.idle_count(), 3);
    // 1000 rounds to 1024, which is idle.
    let (buf, _) = pool.take_gpu(1000, |b| device.alloc(b)).unwrap();
    assert_eq!(buf.bytes(), 1024);
    // 1500 rounds to 2048: [2048, 4096) holds the 2048 only.
    let (buf, _) = pool.take_gpu(1500, |b| device.alloc(b)).unwrap();
    assert_eq!(buf.bytes(), 2048);
    // 600 rounds to 1024: [1024, 2048) is empty now, and the 4096 is too big.
    let (buf, _) = pool.take_gpu(600, |b| device.alloc(b)).unwrap();
    assert_eq!(buf.bytes(), 1024);
    assert_eq!(device.allocated.get(), 4);
    // 3000 rounds to 4096, which is idle.
    let (buf, _) = pool.take_gpu(3000, |b| device.alloc(b)).unwrap();
    assert_eq!(buf.bytes(), 4096);
    assert_eq!(device.allocated.get(), 4);
    assert_eq!(pool.idle_count(), 0);
    assert_eq!(pool.idle_bytes(), 0);
    // Below 256 MB, [r, 2r) holds one bucket; above, several: the smallest that fits comes first.
    let (gb, _) = pool.take_gpu(1024 * MB, |b| device.alloc(b)).unwrap();
    let (three_quarters, _) = pool.take_gpu(768 * MB, |b| device.alloc(b)).unwrap();
    pool.release(gb, 0);
    pool.release(three_quarters, 0);
    let (buf, _) = pool.take_gpu(600 * MB, |b| device.alloc(b)).unwrap();
    assert_eq!(buf.bytes(), 768 * MB);
    let (buf, _) = pool.take_gpu(600 * MB, |b| device.alloc(b)).unwrap();
    assert_eq!(buf.bytes(), 1024 * MB);
    let (buf, _) = pool.take_gpu(600 * MB, |b| device.alloc(b)).unwrap();
    assert_eq!((buf.id, buf.bytes()), (7, 768 * MB));
}

#[test]
fn take_gpu_reuses_busy_buffers_and_keeps_their_last_use() {
    let device = Device::new();
    let mut pool = Pool::new(usize::MAX);
    let (older, _) = pool.take_gpu(1024, |b| device.alloc(b)).unwrap();
    let (newer, _) = pool.take_gpu(1024, |b| device.alloc(b)).unwrap();
    pool.release(older, 3);
    pool.release(newer, 7);
    // The most recently released first, whatever its last use.
    let (buf, last_use) = pool.take_gpu(1024, |b| device.alloc(b)).unwrap();
    assert_eq!((buf.id, last_use), (2, 7));
    let (buf, last_use) = pool.take_gpu(1024, |b| device.alloc(b)).unwrap();
    assert_eq!((buf.id, last_use), (1, 3));
    // A fresh buffer has never been used.
    let (_, last_use) = pool.take_gpu(1024, |b| device.alloc(b)).unwrap();
    assert_eq!(last_use, 0);
}

#[test]
fn take_host_only_takes_buffers_whose_last_use_has_completed() {
    let device = Device::new();
    let mut pool = Pool::new(usize::MAX);
    let (busy, _) = pool.take_gpu(500 * MB, |b| device.alloc(b)).unwrap();
    let (idle_larger, _) = pool.take_gpu(700 * MB, |b| device.alloc(b)).unwrap();
    pool.release(idle_larger, 2);
    pool.release(busy, 5);
    // Completed up to 4: the 512 MB buffer is still used by command buffer 5, so the host gets
    // the idle 768 MB one, the next best fit.
    let (buf, last_use) = pool.take_host(500 * MB, 4, |b| device.alloc(b)).unwrap();
    assert_eq!((buf.id, buf.bytes(), last_use), (2, 768 * MB, 2));
    // Nothing idle fits: a fresh buffer rather than the busy one, and no waiting.
    let (buf, _) = pool.take_host(500 * MB, 4, |b| device.alloc(b)).unwrap();
    assert_eq!(buf.id, 3);
    assert_eq!(pool.idle_count(), 1);
    // Once 5 completes the busy buffer is the host's.
    let (buf, last_use) = pool.take_host(500 * MB, 5, |b| device.alloc(b)).unwrap();
    assert_eq!((buf.id, last_use), (1, 5));
}

#[test]
fn over_the_cap_the_least_recently_released_go_first() {
    let device = Device::new();
    let mut pool = Pool::new(4096);
    let mut bufs = Vec::new();
    for bytes in [1024, 1024, 2048, 1024] {
        bufs.push(pool.take_gpu(bytes, |b| device.alloc(b)).unwrap().0);
    }
    let mut bufs = bufs.into_iter();
    for _ in 0..3 {
        pool.release(bufs.next().unwrap(), 0);
    }
    assert_eq!(pool.idle_bytes(), 4096);
    assert!(device.released().is_empty());
    // A fourth 1024 is over the cap by 1024: buffer 1, released first, goes.
    pool.release(bufs.next().unwrap(), 0);
    assert_eq!(device.released(), vec![1]);
    assert_eq!(pool.idle_bytes(), 4096);
    assert_eq!(pool.stats(false).pool_overcap_free_bytes, 1024);
    // A buffer taken and given back counts as released last: 3 is now newer than 2 and 4, so
    // making room for another 2048 releases those two.
    let (buf, _) = pool.take_gpu(2048, |b| device.alloc(b)).unwrap();
    assert_eq!(buf.id, 3);
    pool.release(buf, 0);
    pool.release(device.alloc(2048).unwrap(), 0);
    assert_eq!(device.released(), vec![1, 2, 4]);
    assert_eq!(pool.idle_bytes(), 4096);
    assert_eq!(pool.stats(false).pool_overcap_free_bytes, 3072);
}

#[test]
fn a_buffer_larger_than_the_cap_is_released_at_once() {
    let device = Device::new();
    let mut pool = Pool::new(4096);
    let (small, _) = pool.take_gpu(1024, |b| device.alloc(b)).unwrap();
    pool.release(small, 0);
    let (big, _) = pool.take_gpu(8192, |b| device.alloc(b)).unwrap();
    pool.release(big, 0);
    assert_eq!(device.released(), vec![2]);
    assert_eq!(pool.idle_bytes(), 1024);
    assert_eq!(pool.stats(false).pool_overcap_free_bytes, 8192);
}

#[test]
fn lowering_the_cap_shrinks_the_pool_at_once() {
    let device = Device::new();
    let mut pool = Pool::new(1 << 20);
    let bufs: Vec<_> = (0..4)
        .map(|_| pool.take_gpu(1024, |b| device.alloc(b)).unwrap().0)
        .collect();
    for buf in bufs {
        pool.release(buf, 0);
    }
    pool.set_cap_override(Some(2048));
    assert_eq!(pool.cap(), 2048);
    assert_eq!(device.released(), vec![1, 2]);
    assert_eq!(pool.idle_bytes(), 2048);
    pool.set_cap_override(None);
    assert_eq!(pool.cap(), 1 << 20);
    assert_eq!(pool.idle_bytes(), 2048);
}

#[test]
fn a_failed_allocation_clears_the_pool_and_retries_once() {
    let device = Device::new();
    let mut pool = Pool::new(usize::MAX);
    let (idle, _) = pool.take_gpu(1024, |b| device.alloc(b)).unwrap();
    pool.release(idle, 0);
    device.fail_next.set(1);
    let (buf, last_use) = pool.take_gpu(1 << 20, |b| device.alloc(b)).unwrap();
    assert_eq!((buf.id, buf.bytes(), last_use), (2, 1 << 20, 0));
    assert_eq!(device.released(), vec![1]);
    assert_eq!(pool.idle_count(), 0);
    let stats = pool.stats(true);
    assert_eq!(stats.pool_oom_clears, 1);
    assert_eq!(stats.pool_fresh_alloc_count, 2);
    assert_eq!(stats.pool_fresh_alloc_bytes, (1024 + (1 << 20)) as u64);
    assert_eq!(pool.stats(false), PoolStats::default());
    // Failing twice is an error, not a third try.
    device.fail_next.set(2);
    let error = pool.take_host(64, 0, |b| device.alloc(b)).err().unwrap();
    assert!(error.contains("256-byte"), "{error}");
    assert_eq!(device.fail_next.get(), 0);
    device.fail_next.set(3);
    assert!(pool.take_host(64, 0, |b| device.alloc(b)).is_err());
    assert_eq!(device.fail_next.get(), 1);
}

#[test]
fn clear_and_trim_release_every_idle_buffer() {
    let device = Device::new();
    let mut pool = Pool::new(usize::MAX);
    for bytes in [300, 5000] {
        let (buf, _) = pool.take_gpu(bytes, |b| device.alloc(b)).unwrap();
        pool.release(buf, 9);
    }
    assert_eq!(pool.trim(), 512 + 8192);
    assert_eq!(device.released(), vec![1, 2]);
    assert_eq!((pool.idle_count(), pool.idle_bytes(), pool.trim()), (0, 0, 0));
}

#[test]
fn the_ring_retires_in_order_without_blocking() {
    let mut ring: Ring<FakeCb> = Ring::new(8);
    let cbs: Vec<_> = (1..=3).map(|serial| commit(&mut ring, serial)).collect();
    ring.retire();
    assert_eq!((ring.completed(), ring.in_flight()), (0, 3));
    // 1 and 3 finish, 2 is still running: only 1 retires.
    cbs[0].finish(CbStatus::Completed);
    cbs[2].finish(CbStatus::Completed);
    ring.retire();
    assert_eq!((ring.completed(), ring.in_flight()), (1, 2));
    assert_eq!(stream_completed(ring.clock().id()), Some(1));
    cbs[1].finish(CbStatus::Completed);
    ring.retire();
    assert_eq!((ring.completed(), ring.in_flight()), (3, 0));
    assert!(cbs.iter().all(|cb| cb.waits() == 0), "retiring never waits");
    let stats = ring.stats(true);
    assert_eq!((stats.commits, stats.waits, stats.gpu_ms), (3, 0, 3.0));
    assert_eq!(ring.stats(false), RingStats::default());
}

#[test]
fn a_wait_blocks_up_to_its_serial_only() {
    let mut ring: Ring<FakeCb> = Ring::new(8);
    let cbs: Vec<_> = (1..=3).map(|serial| commit(&mut ring, serial)).collect();
    ring.wait(2).unwrap();
    assert_eq!(ring.completed(), 2);
    assert_eq!(cbs.iter().map(FakeCb::waits).collect::<Vec<_>>(), vec![1, 1, 0]);
    // Retired serials and the host's serial 0 do not block.
    ring.wait(1).unwrap();
    ring.wait(0).unwrap();
    assert_eq!(ring.stats(false).waits, 1);
    // Not committed yet: the stream must commit before it waits.
    let error = ring.wait(4).err().unwrap();
    assert!(error.contains("not committed"), "{error}");
    ring.wait_all().unwrap();
    assert_eq!((ring.completed(), ring.in_flight()), (3, 0));
}

#[test]
fn a_failure_is_returned_by_the_next_wait_with_its_label() {
    let mut ring: Ring<FakeCb> = Ring::new(8);
    let cbs: Vec<_> = (1..=3).map(|serial| commit(&mut ring, serial)).collect();
    cbs[0].finish(CbStatus::Failed("first".into()));
    cbs[1].finish(CbStatus::Failed("second".into()));
    // Retired without blocking: the buffers it bound are free again, the error is kept.
    ring.retire();
    assert_eq!(ring.completed(), 2);
    let error = ring.wait(3).err().unwrap();
    assert!(error.contains("command buffer 1 (op 1) failed: first"), "{error}");
    assert_eq!(ring.completed(), 3);
    // Returned once; the second failure is not the first.
    ring.wait(3).unwrap();
}

#[test]
fn the_in_flight_cap_waits_on_the_oldest_before_a_commit() {
    let mut ring: Ring<FakeCb> = Ring::new(2);
    let first = commit(&mut ring, 1);
    let second = commit(&mut ring, 2);
    assert_eq!((first.waits(), ring.in_flight()), (0, 2));
    // A third commit waits on 1 only.
    let third = commit(&mut ring, 3);
    assert_eq!((first.waits(), second.waits()), (1, 0));
    assert_eq!((ring.completed(), ring.in_flight()), (1, 2));
    // Already finished: making room retires it without waiting.
    second.finish(CbStatus::Completed);
    commit(&mut ring, 4);
    assert_eq!((second.waits(), third.waits()), (0, 0));
    assert_eq!((ring.completed(), ring.in_flight()), (2, 2));
    assert_eq!(ring.stats(false).waits, 1);
}

#[test]
#[should_panic(expected = "committed after")]
fn serials_must_increase() {
    let mut ring: Ring<FakeCb> = Ring::new(8);
    commit(&mut ring, 2);
    commit(&mut ring, 2);
}

#[test]
fn a_dropped_ring_waits_for_its_work_and_leaves_the_registry() {
    let mut ring: Ring<FakeCb> = Ring::new(8);
    let cb = commit(&mut ring, 1);
    let id = ring.clock().id();
    assert_eq!(stream_completed(id), Some(0));
    drop(ring);
    assert_eq!(cb.waits(), 1);
    assert_eq!(stream_completed(id), None);
}

#[test]
fn clocks_have_unique_ids_and_their_watermark_only_rises() {
    let a = StreamClock::register();
    let b = StreamClock::register();
    assert_ne!(a.id(), b.id());
    assert_ne!(a.id(), 0);
    a.advance(5);
    a.advance(3);
    assert_eq!(stream_completed(a.id()), Some(5));
    assert_eq!(stream_completed(b.id()), Some(0));
    // The registry's own temporary handle on a clock never outlives it.
    let id = a.id();
    let weak = Arc::downgrade(&a);
    drop(a);
    assert!(weak.upgrade().is_none());
    assert_eq!(stream_completed(id), None);
    assert_eq!(stream_completed(0), None);
}

#[test]
fn storage_dropped_on_its_owners_thread_goes_back_with_its_last_use() {
    let device = Device::new();
    let mut ring: Ring<FakeCb> = Ring::new(8);
    set_home(ring.clock().id(), usize::MAX);
    let storage = home(|h| h.take_gpu(1024, |b| device.alloc(b))).unwrap();
    assert_eq!(storage.owner(), ring.clock().id());
    let cb = commit(&mut ring, 1);
    storage.mark_write(1);
    assert_eq!((storage.last_use(), storage.last_write()), (1, 1));
    assert!(!storage.host_idle(ring.completed()));
    assert!(!storage.written(ring.completed()));
    drop(storage);
    assert!(device.released().is_empty());
    // Pending: the host cannot have it, but the GPU can.
    let completed = ring.completed();
    let fresh = home(|h| h.take_host(1024, completed, |b| device.alloc(b))).unwrap();
    assert_eq!(fresh.buf().id, 2);
    let reused = home(|h| h.take_gpu(1024, |b| device.alloc(b))).unwrap();
    assert_eq!((reused.buf().id, reused.last_use(), reused.last_write()), (1, 1, 0));
    // Dropped unused, it keeps that last use: still not the host's until 1 completes.
    drop(reused);
    let completed = ring.completed();
    let fresh_again = home(|h| h.take_host(1024, completed, |b| device.alloc(b))).unwrap();
    assert_eq!(fresh_again.buf().id, 3);
    cb.finish(CbStatus::Completed);
    ring.retire();
    let completed = ring.completed();
    let now_idle = home(|h| h.take_host(1024, completed, |b| device.alloc(b))).unwrap();
    assert_eq!(now_idle.buf().id, 1);
    assert!(now_idle.host_idle(completed) && now_idle.written(completed));
}

#[test]
fn storage_dropped_while_the_home_is_borrowed_is_released() {
    let device = Device::new();
    set_home(7, usize::MAX);
    let storage = home(|h| h.take_gpu(1024, |b| device.alloc(b))).unwrap();
    home(move |_| drop(storage));
    assert_eq!(device.released(), vec![1]);
    assert_eq!(home(|h| h.pool.idle_count()), 0);
}

#[test]
fn storage_dropped_on_another_thread_is_released() {
    let device = Device::new();
    let ring: Ring<FakeCb> = Ring::new(8);
    set_home(ring.clock().id(), usize::MAX);
    let storage = home(|h| h.take_gpu(1024, |b| device.alloc(b))).unwrap();
    let view = storage.clone();
    std::thread::spawn(move || {
        let other: Ring<FakeCb> = Ring::new(8);
        set_home(other.clock().id(), usize::MAX);
        drop(view);
        drop(storage);
        assert_eq!(home(|h| h.pool.idle_count()), 0);
    })
    .join()
    .unwrap();
    assert_eq!(device.released(), vec![1]);
    assert_eq!(home(|h| h.pool.idle_count()), 0);
}

#[test]
fn another_thread_adopts_storage_once_its_use_completes() {
    let device = Device::new();
    let mut ring: Ring<FakeCb> = Ring::new(8);
    let owner = ring.clock().id();
    set_home(owner, usize::MAX);
    let storage = home(|h| h.take_gpu(1024, |b| device.alloc(b))).unwrap();
    let cb = commit(&mut ring, 1);
    storage.mark_write(1);
    // The owner's own claims are free.
    storage.claim(ring.clock(), "gpu_test").unwrap();

    let (to_other, from_owner) = std::sync::mpsc::channel::<()>();
    let (to_owner, from_other) = std::sync::mpsc::channel::<()>();
    let handed = storage.clone();
    let other = std::thread::spawn(move || {
        let other: Ring<FakeCb> = Ring::new(8);
        set_home(other.clock().id(), usize::MAX);
        let error = handed.claim(other.clock(), "gpu_test").err().unwrap();
        assert!(error.starts_with("gpu_test: "), "{error}");
        assert!(error.contains("gpu_synchronize"), "{error}");
        assert_eq!(handed.owner(), owner);
        to_owner.send(()).unwrap();
        from_owner.recv().unwrap();
        // Command buffer 1 has completed on the owner's thread: adopted, serials 0.
        handed.claim(other.clock(), "gpu_test").unwrap();
        assert_eq!(handed.owner(), other.clock().id());
        assert_eq!((handed.last_use(), handed.last_write()), (0, 0));
        // Now the other thread's: dropped there (the last reference), it joins that pool.
        drop(handed);
        assert_eq!(home(|h| h.pool.idle_count()), 1);
    });
    from_other.recv().unwrap();
    cb.finish(CbStatus::Completed);
    ring.retire();
    // The owner drops its reference first, so the other thread's is the last.
    drop(storage);
    to_other.send(()).unwrap();
    other.join().unwrap();
    // The adopting thread's pool held the buffer and released it when that thread ended.
    assert_eq!(device.released(), vec![1]);
}

#[test]
fn storage_whose_owner_is_gone_is_adopted() {
    let device = Arc::new(Mutex::new(Device::new()));
    let released = device.lock().unwrap().released.clone();
    let made_on = device.clone();
    let (storage, dead) = std::thread::spawn(move || {
        let device = made_on.lock().unwrap();
        let mut ring: Ring<FakeCb> = Ring::new(8);
        set_home(ring.clock().id(), usize::MAX);
        let storage = home(|h| h.take_gpu(1024, |b| device.alloc(b))).unwrap();
        commit(&mut ring, 1);
        storage.mark_use(1);
        // The thread ends with command buffer 1 pending: its ring waits for it as it drops.
        (storage, ring.clock().id())
    })
    .join()
    .unwrap();
    assert_eq!(stream_completed(dead), None);
    assert_eq!(storage.last_use(), 1);
    let ring: Ring<FakeCb> = Ring::new(8);
    set_home(ring.clock().id(), usize::MAX);
    storage.claim(ring.clock(), "gpu_test").unwrap();
    assert_eq!((storage.owner(), storage.last_use()), (ring.clock().id(), 0));
    drop(storage);
    assert_eq!(home(|h| h.pool.idle_count()), 1);
    assert!(released.lock().unwrap().is_empty());
}

#[test]
fn in_place_writes_need_storage_no_view_shares() {
    let device = Device::new();
    set_home(9, usize::MAX);
    let storage = home(|h| h.take_host(1024, 0, |b| device.alloc(b))).unwrap();
    Storage::check_unshared(&storage, "gpu_upload_into").unwrap();
    let view = storage.clone();
    let error = Storage::check_unshared(&storage, "gpu_upload_into").err().unwrap();
    assert!(error.starts_with("gpu_upload_into: "), "{error}");
    assert!(Storage::check_unshared(&view, "gpu_copy_into").is_err());
    drop(view);
    Storage::check_unshared(&storage, "gpu_upload_into").unwrap();
}
