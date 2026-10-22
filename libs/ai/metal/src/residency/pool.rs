//! The per-thread buffer pool: idle buffers kept by size bucket and handed out again by best fit,
//! with a cap on idle bytes. These are CUDA's activation pool's rules (`cuda/src/launch.rs`,
//! `gpu_pool_round`, `gpu_pool_acquire`, `gpu_pool_release`), plus one of the stream's own: every
//! idle buffer keeps the serial of the last command buffer that used it, and the host is only
//! given a buffer whose last use has completed, so the CPU never writes memory that queued GPU
//! work still reads (the Flux bug noted in `runtime.rs`).
//!
//! The pool owns no device: the caller passes the allocation as a closure (`newBufferWithLength`,
//! Shared, on macOS) that returns `None` when the device is out of memory.

use super::env_usize;
use std::collections::{BTreeMap, VecDeque};

/// A device buffer the pool can hold.
pub trait PoolBuf {
    /// The buffer's size in bytes: the bucket it was allocated for.
    fn bytes(&self) -> usize;
}

/// The smallest bucket.
pub const MIN_BUCKET: usize = 256;

/// Above this, buckets step by this much instead of doubling.
pub const BIG_STEP: usize = 256 * 1024 * 1024;

/// The idle cap per thread when `MAKEPAD_METAL_POOL_CAP_MB` is unset.
pub const DEFAULT_CAP_MB: usize = 1024;

/// The bucket a request of `bytes` is allocated and pooled at: at least 256 B, powers of two up
/// to 256 MB, then multiples of 256 MB (CUDA's rule; doubling there would waste up to half of
/// the largest planes). Never zero, so a zero-byte request still gets a buffer. One function, so
/// the Mac bench can try finer steps.
pub fn bucket(bytes: usize) -> usize {
    let bytes = bytes.max(MIN_BUCKET);
    if bytes <= BIG_STEP {
        bytes.next_power_of_two()
    } else {
        bytes.div_ceil(BIG_STEP).saturating_mul(BIG_STEP)
    }
}

/// The pool's counters since they were last reset, as CUDA's `GpuPerfStats` names them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PoolStats {
    /// Allocations the pool could not serve from idle buffers.
    pub pool_fresh_alloc_count: u64,
    pub pool_fresh_alloc_bytes: u64,
    /// Bytes released for real because idle buffers would have gone over the cap.
    pub pool_overcap_free_bytes: u64,
    /// Failed allocations that cleared the pool and retried.
    pub pool_oom_clears: u64,
}

/// An idle buffer.
struct Idle<B> {
    buf: B,
    /// Serial of the last command buffer that used it; 0 if none has since the host had it.
    last_use: u64,
    /// When it was released, counting releases, for least-recently-released eviction.
    released: u64,
}

/// One thread's idle buffers. Each bucket's queue is in release order, oldest first.
pub struct Pool<B> {
    idle: BTreeMap<usize, VecDeque<Idle<B>>>,
    idle_bytes: usize,
    cap: usize,
    cap_override: Option<usize>,
    releases: u64,
    stats: PoolStats,
}

impl<B: PoolBuf> Pool<B> {
    /// An empty pool holding at most `cap_bytes` idle.
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            idle: BTreeMap::new(),
            idle_bytes: 0,
            cap: cap_bytes,
            cap_override: None,
            releases: 0,
            stats: PoolStats::default(),
        }
    }

    /// An empty pool with the cap from `MAKEPAD_METAL_POOL_CAP_MB` (1024 by default), per thread.
    pub fn from_env() -> Self {
        let cap_mb = env_usize("MAKEPAD_METAL_POOL_CAP_MB", DEFAULT_CAP_MB);
        Self::new(cap_mb.saturating_mul(1024 * 1024))
    }

    /// The idle cap in bytes: the override if one is set.
    pub fn cap(&self) -> usize {
        self.cap_override.unwrap_or(self.cap)
    }

    pub fn idle_bytes(&self) -> usize {
        self.idle_bytes
    }

    pub fn idle_count(&self) -> usize {
        self.idle.values().map(VecDeque::len).sum()
    }

    /// A buffer for at least `bytes` that the GPU will write: the best fit among idle buffers
    /// whose bucket is in `[r, 2r)` for `r = bucket(bytes)`, the most recently released in that
    /// bucket, whatever its last use (reuse on the GPU is ordered by the stream's serial encoder
    /// and Metal's hazard tracking); otherwise a fresh one. Returns the buffer and its last use,
    /// which the storage made from it must keep: dropped unused, it is still busy until then.
    pub fn take_gpu(
        &mut self,
        bytes: usize,
        alloc: impl FnMut(usize) -> Option<B>,
    ) -> Result<(B, u64), String> {
        let rounded = bucket(bytes);
        if let Some(hit) = self.take_idle(rounded, u64::MAX) {
            return Ok(hit);
        }
        self.fresh(rounded, alloc)
    }

    /// A buffer for at least `bytes` that the host will write now: the best fit as in
    /// [`Pool::take_gpu`], but only among idle buffers whose last use is at most `completed` (the
    /// stream's completed watermark, after retiring what has finished); otherwise a fresh one.
    /// Never waits.
    pub fn take_host(
        &mut self,
        bytes: usize,
        completed: u64,
        alloc: impl FnMut(usize) -> Option<B>,
    ) -> Result<(B, u64), String> {
        let rounded = bucket(bytes);
        if let Some(hit) = self.take_idle(rounded, completed) {
            return Ok(hit);
        }
        self.fresh(rounded, alloc)
    }

    /// The smallest bucket in `[rounded, 2*rounded)` with an entry last used at or before
    /// `completed`, and in it the most recently released such entry.
    fn take_idle(&mut self, rounded: usize, completed: u64) -> Option<(B, u64)> {
        let limit = rounded.saturating_mul(2);
        let (size, index) = self.idle.range(rounded..limit).find_map(|(size, entries)| {
            entries
                .iter()
                .rposition(|entry| entry.last_use <= completed)
                .map(|index| (*size, index))
        })?;
        let entries = self.idle.get_mut(&size)?;
        let entry = entries.remove(index)?;
        if entries.is_empty() {
            self.idle.remove(&size);
        }
        self.idle_bytes -= size;
        Some((entry.buf, entry.last_use))
    }

    /// A fresh buffer of `rounded` bytes. A failed allocation is most likely the device out of
    /// memory: every idle buffer is released and the allocation tried once more.
    fn fresh(
        &mut self,
        rounded: usize,
        mut alloc: impl FnMut(usize) -> Option<B>,
    ) -> Result<(B, u64), String> {
        self.stats.pool_fresh_alloc_count += 1;
        self.stats.pool_fresh_alloc_bytes += rounded as u64;
        if let Some(buf) = alloc(rounded) {
            return Ok((buf, 0));
        }
        self.stats.pool_oom_clears += 1;
        self.clear();
        match alloc(rounded) {
            Some(buf) => Ok((buf, 0)),
            None => Err(format!(
                "Metal: could not allocate a {rounded}-byte buffer, even after clearing the pool"
            )),
        }
    }

    /// Gives `buf` back, idle, last used by command buffer `last_use` (0: by none). Over the cap,
    /// the least recently released idle buffers are released for real until it fits; a buffer
    /// larger than the whole cap is released at once.
    pub fn release(&mut self, buf: B, last_use: u64) {
        let size = buf.bytes();
        let cap = self.cap();
        if size > cap {
            self.stats.pool_overcap_free_bytes += size as u64;
            return; // `buf` drops here: released
        }
        if self.idle_bytes + size > cap {
            let freed = self.evict_lru(self.idle_bytes + size - cap);
            self.stats.pool_overcap_free_bytes += freed as u64;
        }
        self.releases += 1;
        self.idle.entry(size).or_default().push_back(Idle {
            buf,
            last_use,
            released: self.releases,
        });
        self.idle_bytes += size;
    }

    /// Releases idle buffers for real, least recently released first, until at least `need`
    /// bytes are freed or none is left. Returns the bytes freed.
    fn evict_lru(&mut self, need: usize) -> usize {
        let mut freed = 0;
        while freed < need {
            let oldest = self
                .idle
                .iter()
                .filter_map(|(size, entries)| entries.front().map(|entry| (entry.released, *size)))
                .min();
            let Some((_, size)) = oldest else {
                break;
            };
            if let Some(entries) = self.idle.get_mut(&size) {
                entries.pop_front();
                if entries.is_empty() {
                    self.idle.remove(&size);
                }
            }
            self.idle_bytes -= size;
            freed += size;
        }
        freed
    }

    /// Overrides the idle cap (`None` goes back to the one the pool was made with). Lowering it
    /// shrinks the idle buffers at once, least recently released first.
    pub fn set_cap_override(&mut self, bytes: Option<usize>) {
        self.cap_override = bytes;
        let cap = self.cap();
        if self.idle_bytes > cap {
            let freed = self.evict_lru(self.idle_bytes - cap);
            self.stats.pool_overcap_free_bytes += freed as u64;
        }
    }

    /// Releases every idle buffer.
    pub fn clear(&mut self) {
        self.idle.clear();
        self.idle_bytes = 0;
    }

    /// Releases every idle buffer and returns how many bytes that gave back.
    pub fn trim(&mut self) -> usize {
        let freed = self.idle_bytes;
        self.clear();
        freed
    }

    /// The counters, zeroed afterwards when `reset`.
    pub fn stats(&mut self, reset: bool) -> PoolStats {
        let stats = self.stats;
        if reset {
            self.stats = PoolStats::default();
        }
        stats
    }
}
