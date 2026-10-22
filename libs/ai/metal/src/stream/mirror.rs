//! The host binding's device copies of host tensors (cutie-metal.md §8): until the tensor itself
//! lives on the device (step 12), a tensor bound as an input is copied into a Shared buffer, kept
//! here by its content id, so a weight read every frame is copied once. Least recently used
//! copies go first once their bytes pass the cap (MAKEPAD_METAL_MIRROR_CAP_MB, 512 by default).
//! Ids are content identities (`gpu_types.rs`): a write gives a tensor a new one, so an entry is
//! never stale, only unused.

use std::collections::HashMap;

/// The cap when MAKEPAD_METAL_MIRROR_CAP_MB is unset.
pub const DEFAULT_MIRROR_CAP_MB: usize = 512;

struct Entry<V> {
    value: V,
    bytes: usize,
    used: u64,
}

/// Values by id with their sizes, the least recently used dropped first over the cap.
pub(crate) struct Lru<V> {
    entries: HashMap<u64, Entry<V>>,
    bytes: usize,
    cap: usize,
    tick: u64,
}

impl<V> Lru<V> {
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self { entries: HashMap::new(), bytes: 0, cap: cap_bytes, tick: 0 }
    }

    /// The value for `key`, now the most recently used.
    pub(crate) fn get(&mut self, key: u64) -> Option<&V> {
        self.tick += 1;
        let tick = self.tick;
        self.entries.get_mut(&key).map(|entry| {
            entry.used = tick;
            &entry.value
        })
    }

    /// Keeps `value` (`bytes` big) for `key`, the most recently used. Returns what it no longer
    /// keeps, for the caller to drop when it is safe to: the value it replaces, the least recently
    /// used values dropped to fit under the cap, or `value` itself when it alone is over the cap.
    pub(crate) fn insert(&mut self, key: u64, value: V, bytes: usize) -> Vec<V> {
        let mut dropped = Vec::new();
        if let Some(old) = self.entries.remove(&key) {
            self.bytes -= old.bytes;
            dropped.push(old.value);
        }
        if bytes > self.cap {
            dropped.push(value);
            return dropped;
        }
        while self.bytes + bytes > self.cap {
            let Some(oldest) = self.entries.iter().min_by_key(|(_, entry)| entry.used).map(|(key, _)| *key) else {
                break;
            };
            if let Some(entry) = self.entries.remove(&oldest) {
                self.bytes -= entry.bytes;
                dropped.push(entry.value);
            }
        }
        self.tick += 1;
        self.entries.insert(key, Entry { value, bytes, used: self.tick });
        self.bytes += bytes;
        dropped
    }

    /// Takes every value out.
    pub(crate) fn clear(&mut self) -> Vec<V> {
        self.bytes = 0;
        self.entries.drain().map(|(_, entry)| entry.value).collect()
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}
