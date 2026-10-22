//! Each stream's clock, and its ring of committed command buffers.
//!
//! A stream numbers its command buffers from 1 (serial 0 means "the host"), commits them in
//! order and retires them in order. Its clock's `completed` is the serial of the last command
//! buffer retired, so any storage last used at or before it is idle. Other threads read a
//! stream's watermark through a global registry, a slow path taken only when a thread touches
//! storage another thread's stream owns ([`super::Storage::claim`]).

use super::env_usize;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The cap on committed command buffers not yet retired when `MAKEPAD_METAL_MAX_IN_FLIGHT` is
/// unset. Apple GPUs preempt only between command buffers, and RV draws on the same GPU.
pub const DEFAULT_MAX_IN_FLIGHT: usize = 3;

static NEXT_STREAM_ID: AtomicU64 = AtomicU64::new(1);

/// Every live stream's clock by id (`STREAM_CLOCKS` in the plan).
static STREAM_CLOCKS: Mutex<BTreeMap<u64, Weak<StreamClock>>> = Mutex::new(BTreeMap::new());

/// A stream's id and its completed watermark, readable from any thread.
pub struct StreamClock {
    id: u64,
    completed: AtomicU64,
}

impl StreamClock {
    /// A new stream's clock, with a process-unique id (never 0), registered so other threads can
    /// read its watermark until it drops.
    pub fn register() -> Arc<StreamClock> {
        let clock = Arc::new(StreamClock {
            id: NEXT_STREAM_ID.fetch_add(1, Ordering::Relaxed),
            completed: AtomicU64::new(0),
        });
        STREAM_CLOCKS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(clock.id, Arc::downgrade(&clock));
        clock
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// The serial of the last command buffer this stream retired.
    pub fn completed(&self) -> u64 {
        self.completed.load(Ordering::Acquire)
    }

    /// Moves the watermark up to `serial` (never down). Only the stream that owns the clock
    /// calls this, as it retires command buffers.
    pub fn advance(&self, serial: u64) {
        self.completed.fetch_max(serial, Ordering::AcqRel);
    }
}

impl Drop for StreamClock {
    fn drop(&mut self) {
        STREAM_CLOCKS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.id);
    }
}

/// The completed watermark of stream `id`, or `None` once that stream is gone (or if it never
/// was).
pub fn stream_completed(id: u64) -> Option<u64> {
    // The lock is released at the end of this statement, before the clock can drop here: a
    // clock's drop takes the lock too.
    let clock = STREAM_CLOCKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&id)
        .and_then(Weak::upgrade);
    clock.map(|clock| clock.completed())
}

/// What a committed command buffer says about itself without blocking.
#[derive(Clone, Debug, PartialEq)]
pub enum CbStatus {
    /// Enqueued, committed or scheduled: Metal's status below 4.
    Pending,
    /// Metal's status 4.
    Completed,
    /// Metal's status 5, with the error's description.
    Failed(String),
}

/// A committed command buffer as the ring sees it.
pub trait CommandBuf {
    /// Its status, without blocking.
    fn status(&self) -> CbStatus;
    /// Blocks until the GPU has finished with it, completed or failed (`waitUntilCompleted`).
    fn wait_until_completed(&self);
    /// GPU time in milliseconds once finished (`GPUEndTime - GPUStartTime`); 0 when unknown.
    fn gpu_ms(&self) -> f64;
}

/// The ring's counters since they were last reset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RingStats {
    /// Command buffers pushed (committed).
    pub commits: u64,
    /// Times the host blocked on a command buffer: waits and the in-flight cap.
    pub waits: u64,
    /// GPU time of the command buffers retired.
    pub gpu_ms: f64,
}

/// A failed command buffer not yet acknowledged: its serial, its error with its label, and how
/// many more failed after it.
struct Failure {
    serial: u64,
    message: String,
    more: usize,
}

impl Failure {
    fn describe(&self) -> String {
        match self.more {
            0 => self.message.clone(),
            1 => format!("{} (and 1 more command buffer after it)", self.message),
            more => format!("{} (and {more} more command buffers after it)", self.message),
        }
    }
}

/// A stream's committed command buffers not yet retired, oldest first, and its clock.
///
/// A failed command buffer's error is sticky: every wait for it or for a later serial returns it
/// until [`Ring::acknowledge`] (a synchronize). Metal runs later work after a fault, and that work
/// may have read what the failed buffer wrote, so what was written from the failure up to the
/// acknowledgement stays untrusted afterwards too ([`Ring::check_written`]). Waits for work
/// committed before the failure are Ok.
pub struct Ring<C: CommandBuf> {
    clock: Arc<StreamClock>,
    /// (serial, command buffer, label) in commit order.
    in_flight: VecDeque<(u64, C, String)>,
    last_committed: u64,
    max_in_flight: usize,
    /// The first failure retired and not yet acknowledged.
    failure: Option<Failure>,
    /// Acknowledged failures: the failed serial, the last serial committed when it was
    /// acknowledged, and the error. Storage last written in one of these spans is untrusted.
    failed_spans: Vec<(u64, u64, String)>,
    stats: RingStats,
}

impl<C: CommandBuf> Ring<C> {
    /// An empty ring holding at most `max_in_flight` (at least 1) command buffers, with a newly
    /// registered clock.
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            clock: StreamClock::register(),
            in_flight: VecDeque::new(),
            last_committed: 0,
            max_in_flight: max_in_flight.max(1),
            failure: None,
            failed_spans: Vec::new(),
            stats: RingStats::default(),
        }
    }

    /// An empty ring with the cap from `MAKEPAD_METAL_MAX_IN_FLIGHT` (3 by default).
    pub fn from_env() -> Self {
        Self::new(env_usize("MAKEPAD_METAL_MAX_IN_FLIGHT", DEFAULT_MAX_IN_FLIGHT))
    }

    pub fn clock(&self) -> &Arc<StreamClock> {
        &self.clock
    }

    /// The serial of the last command buffer retired.
    pub fn completed(&self) -> u64 {
        self.clock.completed()
    }

    /// The serial of the last command buffer pushed; 0 before the first.
    pub fn last_committed(&self) -> u64 {
        self.last_committed
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Pops the finished command buffers off the front without blocking, stopping at the first
    /// still pending, and advances the clock to the last one popped (a failed buffer is finished
    /// too: the GPU is done with what it bound). A failure is kept, the first with its serial and
    /// label and the later ones counted, until [`Ring::acknowledge`]; each one's GPU time goes
    /// into the stats. The stream calls this before any host-idle acquire.
    pub fn retire(&mut self) {
        while let Some((serial, cb, label)) = self.in_flight.front() {
            match cb.status() {
                CbStatus::Pending => break,
                CbStatus::Completed => {}
                CbStatus::Failed(message) => match &mut self.failure {
                    Some(failure) => failure.more += 1,
                    None => {
                        self.failure = Some(Failure {
                            serial: *serial,
                            message: format!("Metal command buffer {serial} ({label}) failed: {message}"),
                            more: 0,
                        })
                    }
                },
            }
            self.stats.gpu_ms += cb.gpu_ms();
            let serial = *serial;
            self.in_flight.pop_front();
            self.clock.advance(serial);
        }
    }

    /// Called before committing a command buffer: retires what has finished, then, while the ring
    /// holds `max_in_flight`, waits on the oldest. A failure found here is kept for the waits
    /// after it.
    pub fn make_room(&mut self) {
        self.retire();
        while self.in_flight.len() >= self.max_in_flight {
            let Some(oldest) = self.in_flight.front().map(|entry| entry.0) else {
                break;
            };
            self.block_until(oldest);
            if self.in_flight.front().map(|entry| entry.0) == Some(oldest) {
                // Still pending after waitUntilCompleted: never spin on it.
                break;
            }
        }
    }

    /// Records command buffer `serial`, just committed, labelled `label` for errors. Serials
    /// increase by commit.
    pub fn push(&mut self, serial: u64, cb: C, label: String) {
        assert!(
            serial > self.last_committed,
            "Metal stream: command buffer {serial} committed after {}",
            self.last_committed
        );
        self.last_committed = serial;
        self.in_flight.push_back((serial, cb, label));
        self.stats.commits += 1;
    }

    /// Blocks until command buffer `serial` and every one before it have finished, and retires
    /// them. `Err` when one of them failed and the failure is not yet acknowledged (sticky: every
    /// wait at or after the failed serial returns it). `serial` must be committed: the stream
    /// commits its open command buffer before waiting on it. Serial 0 (the host) and serials
    /// already retired do not block.
    pub fn wait(&mut self, serial: u64) -> Result<(), String> {
        if serial > self.last_committed {
            return Err(format!(
                "Metal stream: waiting for command buffer {serial}, which is not committed (last \
                 committed: {})",
                self.last_committed
            ));
        }
        if serial > self.completed() {
            self.block_until(serial);
        }
        if let Some(failure) = self.failure.as_ref().filter(|failure| failure.serial <= serial) {
            return Err(failure.describe());
        }
        if serial > self.completed() {
            return Err(format!(
                "Metal stream: command buffer {serial} still pending after waitUntilCompleted"
            ));
        }
        Ok(())
    }

    /// [`Ring::wait`] for the last command buffer committed.
    pub fn wait_all(&mut self) -> Result<(), String> {
        self.wait(self.last_committed)
    }

    /// Ends a failure's stickiness: `Err` with it once more if one was kept, after which waits
    /// are Ok again. Storage written from the failed serial up to the last one committed now
    /// stays untrusted ([`Ring::check_written`]). Called by a synchronize, after it has committed
    /// and waited for everything; Ok when there is nothing to acknowledge.
    pub fn acknowledge(&mut self) -> Result<(), String> {
        let Some(failure) = self.failure.take() else {
            return Ok(());
        };
        let message = failure.describe();
        self.failed_spans.push((failure.serial, self.last_committed, message.clone()));
        Err(message)
    }

    /// Whether the host may trust storage last written by command buffer `serial`: `Err` when
    /// that buffer failed, or ran after a failure that is not yet acknowledged, or between a
    /// failure and its acknowledgement. Serial 0 (written by the host) is always Ok. For reads of
    /// written storage (download, with_host, binding an input) once tensors are resident; it does
    /// not wait.
    pub fn check_written(&self, serial: u64) -> Result<(), String> {
        if serial == 0 {
            return Ok(());
        }
        if let Some(failure) = self.failure.as_ref().filter(|failure| failure.serial <= serial) {
            return Err(failure.describe());
        }
        for (first, last, message) in &self.failed_spans {
            if (*first..=*last).contains(&serial) {
                return Err(format!(
                    "data written by Metal command buffer {serial} is not trusted: command buffers \
                     {first} to {last} ran at or after a failure, reported by a synchronize since: \
                     {message}"
                ));
            }
        }
        Ok(())
    }

    /// Waits on every command buffer up to `serial`, in order, then retires.
    fn block_until(&mut self, serial: u64) {
        let mut waited = false;
        for (entry_serial, cb, _) in &self.in_flight {
            if *entry_serial > serial {
                break;
            }
            cb.wait_until_completed();
            waited = true;
        }
        if waited {
            self.stats.waits += 1;
        }
        self.retire();
    }

    /// The counters, zeroed afterwards when `reset`.
    pub fn stats(&mut self, reset: bool) -> RingStats {
        let stats = self.stats;
        if reset {
            self.stats = RingStats::default();
        }
        stats
    }
}

impl<C: CommandBuf> Drop for Ring<C> {
    /// Waits for everything committed. Once the clock leaves the registry, other threads adopt
    /// this stream's storage as finished, so it must be.
    fn drop(&mut self) {
        let last = self.last_committed;
        self.block_until(last);
    }
}
