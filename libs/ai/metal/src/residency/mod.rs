//! The platform-free core of the Metal stream's residency: the per-thread buffer pool, each
//! stream's clock and its ring of committed command buffers, and the storage a tensor and its
//! views share. The macOS stream (`stream/`) fills these in with retained `MTLBuffer`s and
//! Metal's command buffers; everything here is generic over both, so it compiles on every OS and
//! its rules are unit-tested with fake buffers on any machine (`cargo test -p makepad-ai-metal
//! --lib residency::`).
//!
//! The rules, in one place:
//! - A stream numbers its command buffers from 1, commits them in order and retires them in
//!   order. Serial 0 means "the host": storage last used or written by serial 0 is idle.
//! - Every storage records the stream that owns it and the serials of the last command buffers
//!   that used and wrote it. The host writes only storage whose last use has completed
//!   ([`Pool::take_host`]); it reads only storage whose last writer has completed.
//! - Storage dropped on its owner's thread goes back to that thread's pool with its last use;
//!   dropped anywhere else, its buffer is released (command buffers retain what they bind, so
//!   work still in flight keeps it alive).
//! - Another thread's stream adopts storage once its owner is gone or has completed its last use,
//!   and is refused while that use is pending ([`Storage::claim`]).
//! - An in-place write into storage shared with a view is refused
//!   ([`Storage::check_unshared`]).

pub mod clock;
pub mod pool;
pub mod storage;

#[cfg(test)]
mod tests;

pub use clock::{stream_completed, CbStatus, CommandBuf, Ring, RingStats, StreamClock};
pub use pool::{bucket, Pool, PoolBuf, PoolStats};
pub use storage::{Home, Storage, ThreadHome};

/// The environment variable `name` as a whole number, or `default` when it is unset or not one.
pub fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}
