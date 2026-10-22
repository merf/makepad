//! The storage behind a tensor and its views: a pooled buffer, the stream that owns it, and the
//! serials of the last command buffers that used and wrote it. On macOS a `GpuTensor` is an
//! `Arc<Storage<MtlBuf>>` plus an offset; views clone the `Arc`.

use super::clock::{stream_completed, StreamClock};
use super::pool::{Pool, PoolBuf};
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A buffer type whose threads each have one [`Home`]: the thread's pool and the id of the
/// thread's stream. The home is its own thread-local, apart from the stream's context, so storage
/// dropped while the context is borrowed (a temporary inside an op) still goes back to the pool.
pub trait ThreadHome: PoolBuf + Sized + Send + Sync + 'static {
    /// Runs `f` on the calling thread's home. `None` when the thread has none, it is already
    /// borrowed, or the thread's locals are being torn down: storage dropping then releases its
    /// buffer.
    fn with_home<R>(f: impl FnOnce(&mut Home<Self>) -> R) -> Option<R>;
}

/// One thread's pool, and the id of the thread's stream (0 until it has one).
pub struct Home<B> {
    pub stream: u64,
    pub pool: Pool<B>,
}

impl<B: PoolBuf> Home<B> {
    /// A home with `pool` and no stream yet.
    pub fn new(pool: Pool<B>) -> Self {
        Self { stream: 0, pool }
    }
}

impl<B: ThreadHome> Home<B> {
    /// Storage of at least `bytes` for this thread's stream to write on the GPU
    /// ([`Pool::take_gpu`]). It keeps the last use of the buffer it reuses: dropped unused, it is
    /// still busy until that use completes.
    pub fn take_gpu(
        &mut self,
        bytes: usize,
        alloc: impl FnMut(usize) -> Option<B>,
    ) -> Result<Arc<Storage<B>>, String> {
        let (buf, last_use) = self.pool.take_gpu(bytes, alloc)?;
        Ok(Storage::new(buf, self.stream, last_use))
    }

    /// Storage of at least `bytes` for the host to write now, no queued work using it
    /// ([`Pool::take_host`]; `completed` is the stream's watermark). Never waits.
    pub fn take_host(
        &mut self,
        bytes: usize,
        completed: u64,
        alloc: impl FnMut(usize) -> Option<B>,
    ) -> Result<Arc<Storage<B>>, String> {
        let (buf, last_use) = self.pool.take_host(bytes, completed, alloc)?;
        Ok(Storage::new(buf, self.stream, last_use))
    }
}

/// A buffer shared by a tensor and its views.
///
/// `stream`, `last_use` and `last_write` are atomics so the storage is `Sync` and a thread that
/// does not own it can read them; only the owning stream writes them, apart from an adoption
/// ([`Storage::claim`]).
pub struct Storage<B: ThreadHome> {
    buf: ManuallyDrop<B>,
    /// The id of the stream that owns it.
    stream: AtomicU64,
    /// Serial of the last command buffer that bound it; 0 if none has.
    last_use: AtomicU64,
    /// Serial of the last command buffer that wrote it; 0 if the host wrote it last.
    last_write: AtomicU64,
}

impl<B: ThreadHome> Storage<B> {
    /// Storage over `buf`, owned by stream `stream`, last used by command buffer `last_use` (a
    /// reused buffer's, from the pool), written by the host.
    pub fn new(buf: B, stream: u64, last_use: u64) -> Arc<Self> {
        Arc::new(Self {
            buf: ManuallyDrop::new(buf),
            stream: AtomicU64::new(stream),
            last_use: AtomicU64::new(last_use),
            last_write: AtomicU64::new(0),
        })
    }

    pub fn buf(&self) -> &B {
        &self.buf
    }

    /// The buffer's size: its pool bucket.
    pub fn bytes(&self) -> usize {
        self.buf.bytes()
    }

    /// The id of the stream that owns it.
    pub fn owner(&self) -> u64 {
        self.stream.load(Ordering::Acquire)
    }

    pub fn last_use(&self) -> u64 {
        self.last_use.load(Ordering::Acquire)
    }

    pub fn last_write(&self) -> u64 {
        self.last_write.load(Ordering::Acquire)
    }

    /// Records that the owning stream's command buffer `serial` binds it.
    pub fn mark_use(&self, serial: u64) {
        self.last_use.store(serial, Ordering::Release);
    }

    /// Records that the owning stream's command buffer `serial` writes it (and so uses it).
    pub fn mark_write(&self, serial: u64) {
        self.last_use.store(serial, Ordering::Release);
        self.last_write.store(serial, Ordering::Release);
    }

    /// Records that the host has just written it in place (`upload_into` into idle storage): its
    /// contents are the host's again, trusted whatever wrote them before, and its last use stands.
    pub fn mark_host_write(&self) {
        self.last_write.store(0, Ordering::Release);
    }

    /// True when no queued work uses it any more, given the owner's watermark `completed`: the
    /// host may write it.
    pub fn host_idle(&self, completed: u64) -> bool {
        self.last_use() <= completed
    }

    /// True when its last writer has finished, given the owner's watermark `completed`: the host
    /// may read it.
    pub fn written(&self, completed: u64) -> bool {
        self.last_write() <= completed
    }

    /// Called before the stream with `clock` binds this storage or reads it on the host. Ok at
    /// once when that stream owns it. Otherwise the storage is adopted (owner: this stream,
    /// serials 0) if its owner is gone or has completed its last use, and refused while that use
    /// is pending: tensors belong to the thread that made them (Cutie's own contract), and a
    /// thread handing one over synchronises first.
    ///
    /// This checks a handover; it does not make sharing safe. The owner's serials and watermark
    /// are read one after the other, so a thread that claims storage while its owner is still
    /// binding it (two threads using one tensor, or two threads adopting it at once and one
    /// starting work on it) can see an old `last_use` and adopt it while that work is pending.
    /// The compare-exchange only stops two adoptions from both succeeding against the same
    /// owner. The other stream's failures are not seen here either: its synchronize reports them.
    pub fn claim(&self, clock: &StreamClock, op: &str) -> Result<(), String> {
        let mine = clock.id();
        let mut owner = self.stream.load(Ordering::Acquire);
        loop {
            if owner == mine {
                return Ok(());
            }
            let last_use = self.last_use.load(Ordering::Acquire);
            let finished = match stream_completed(owner) {
                None => true,
                Some(completed) => last_use <= completed,
            };
            if !finished {
                return Err(format!(
                    "{op}: the tensor is pending on another thread's Metal stream; \
                     gpu_synchronize there before using it here"
                ));
            }
            match self
                .stream
                .compare_exchange(owner, mine, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    self.last_use.store(0, Ordering::Release);
                    self.last_write.store(0, Ordering::Release);
                    return Ok(());
                }
                Err(now) => owner = now,
            }
        }
    }

    /// The rule for in-place writes (`upload_into`, `copy_into`): refused when the storage is
    /// shared with a view, since a write through one would change the other behind its back
    /// (and leave its content id stale for the id-keyed cache).
    pub fn check_unshared(this: &Arc<Self>, op: &str) -> Result<(), String> {
        if Arc::strong_count(this) > 1 {
            return Err(format!(
                "{op}: in-place write into storage shared with a view; write into a new tensor"
            ));
        }
        Ok(())
    }
}

impl<B: ThreadHome> Drop for Storage<B> {
    /// On the owning stream's thread, the buffer goes back to the pool with its last use.
    /// Anywhere else (or with no pool left), it is released: command buffers retain the buffers
    /// they bind, so work still in flight keeps it alive.
    fn drop(&mut self) {
        // SAFETY: `buf` is taken once, here, and the storage is never touched again.
        let buf = unsafe { ManuallyDrop::take(&mut self.buf) };
        let owner = *self.stream.get_mut();
        let last_use = *self.last_use.get_mut();
        let mut buf = Some(buf);
        if owner != 0 {
            B::with_home(|home| {
                if home.stream == owner {
                    if let Some(buf) = buf.take() {
                        home.pool.release(buf, last_use);
                    }
                }
            });
        }
        drop(buf);
    }
}
