//! The stream's weight cache by qualified key, `{namespace}::{key}::{suffix}` as on CUDA
//! (`::cutie_conv` holds a convolution's weights then its bias, `::cutie_ln` gamma then beta).
//! A lookup is size-checked: an entry of another size is a miss, replaced on fill.

use std::collections::HashMap;

pub(crate) struct WeightMap<W> {
    entries: HashMap<String, (W, usize)>,
}

impl<W> WeightMap<W> {
    pub(crate) fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// The entry for `key` if it holds exactly `floats` values.
    pub(crate) fn get(&self, key: &str, floats: usize) -> Option<&W> {
        self.entries.get(key).filter(|(_, held)| *held == floats).map(|(weights, _)| weights)
    }

    /// Keeps `weights` (`floats` values) for `key`; returns the entry it replaces, which work
    /// still in flight may hold on to.
    pub(crate) fn insert(&mut self, key: &str, weights: W, floats: usize) -> Option<W> {
        self.entries.insert(key.to_string(), (weights, floats)).map(|(old, _)| old)
    }

    /// Drops every entry whose key starts with `prefix`; returns how many.
    pub(crate) fn evict_prefix(&mut self, prefix: &str) -> usize {
        let before = self.entries.len();
        self.entries.retain(|key, _| !key.starts_with(prefix));
        before - self.entries.len()
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}
