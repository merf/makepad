//! Cutie's memory (`inference/memory_manager.MemoryManager`, `inference/kv_memory_store.
//! KeyValueMemoryStore`): working memory, long-term memory, sensory memory and the object memory.
//!
//! One bucket: the processor always adds every object together (the worker's way: a fixed number
//! of object slots, all memorised from the first frame), so every object shares one key store.
//!
//! - **Working memory**: keys [64, N], shrinkage [1, N], selection [64, N] (temporary part only)
//!   and each object's values [256, N], one column a token (h*w a memorised frame). The permanent
//!   part comes first (`perm_end_pt` columns): `as_permanent` 'all' (a commit) prepends to it,
//!   'first' makes the first frame permanent only while there is none. Temporary columns have a
//!   use count and a life count (f32, life starts at 1e-7). Each read adds the read's usage to the
//!   use counts and 1 to the life counts (temporary part only).
//! - **Long-term memory**: when working memory's temporary part reaches (max_mem_frames - 1) * h*w
//!   tokens after an add, all but the last (min_mem_frames - 1) frames of it consolidate: if
//!   long-term's own tokens are at least max_long_tokens - num_prototypes, it first keeps only its
//!   max_long_tokens - num_prototypes - buffer_tokens most used (use / life, top-k); then the
//!   candidates' num_prototypes most used (usage normalised by life, top-k, sorted) become
//!   prototype keys and selections; similarity of every candidate to each prototype
//!   (`gpu_cutie_similarity` with the candidates' shrinkage and the prototypes' selection),
//!   a full softmax over the candidates (`gpu_cutie_softmax_axis0`), and the prototypes' values and
//!   shrinkage read through it (candidates' [c, M] @ affinity [M, P]); the candidates leave
//!   working memory and the prototypes join long-term memory (appended, use and life counts 0 and
//!   1e-7). Long-term memory has no selection and no permanent part.
//! - **Reading** (each segmented frame): keys = [long-term, working] concatenated (long-term
//!   first), shrinkage likewise; similarity with the query's key and selection
//!   (`gpu_cutie_similarity`); top-k softmax (`gpu_cutie_topk_softmax`, k = top_k) giving the
//!   affinity and each token's usage (split: long-term's part to its counts when counted, working
//!   memory's to its); each object's values [long-term, working] concatenated, read out
//!   ([256, N] @ affinity [N, h*w], all objects at once [n*256, N]); then `net.pixel_fusion` with
//!   the sensory memory and the last mask, then `net.readout_query` with the object memory.
//! - **Sensory memory**: [256, h*w] per object, zeros when first needed, replaced by `segment`
//!   and `encode_mask` (deep update).
//! - **Object memory**: [16, 257] per object, a running sum of the summaries (and areas).
//!
//! The oracle dumps this state before and after each tapped step (`state.before.*`,
//! `state.after.*`): `work_mem.b0.key`, `.shrinkage`, `.selection`, `.use_count`, `.life_count`,
//! `.value.obj{o}`, `long_mem.b0.*` likewise (no selection), `sensory.obj{o}`,
//! `obj_memory.obj{o}`, `last_mask`.

use crate::CutieConfig;
use crate::model::{not_ported, CutieNet, Feat, Keys, Tap};
use crate::oracle::{Oracle, Report};
use crate::model::Weights;
use crate::Result;
use makepad_ai_common::gpu::GpuTensor;

/// How an add joins the permanent part (`KeyValueMemoryStore.add`'s `as_permanent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Permanence {
    No,
    First,
    All,
}

pub(crate) struct MemoryManager {}

impl MemoryManager {
    pub(crate) fn new(_config: &CutieConfig) -> Self {
        Self {}
    }

    /// Whether anything has been memorised.
    pub(crate) fn engaged(&self) -> bool {
        false
    }

    /// `MemoryManager.read`: the memory readout for every object (n, 256 at stride 16), from the
    /// frame's `pix_feat` and `keys` and the `last_mask` (n images of 1 channel at the padded H,
    /// W). Updates the usage counts. Taps `memory.read.memory_key`, `.memory_shrinkage`,
    /// `.query_key` ([1, 64, h*w]), `.query_selection`, `.similarity`, `.affinity`, `.usage`,
    /// `.readout` ([1, n, 256, h*w]), then the network's own taps (pixel fusion, transformer).
    pub(crate) fn read(&mut self, _net: &CutieNet, _pix_feat: &Feat, _keys: &Keys, _last_mask: &Feat, _tap: &mut Tap) -> Result<Feat> {
        Err(not_ported("the memory read"))
    }

    /// `MemoryManager.add_memory`: memorises a frame's key, shrinkage and selection with each
    /// object's `value` (n, 256 at stride 16) and summaries `obj_value` [n*16, 257], then
    /// consolidates if working memory is full. Taps the consolidation as
    /// `memory.consolidate.memory_key`, `.memory_shrinkage`, `.query_key`, `.query_selection`,
    /// `.similarity`, `.affinity`, `.readout` (each object's prototype values, then the
    /// shrinkage as `.readout#n`).
    pub(crate) fn add_memory(&mut self, _keys: &Keys, _value: &Feat, _obj_value: &GpuTensor, _as_permanent: Permanence, _tap: &mut Tap) -> Result<()> {
        Err(not_ported("adding memory"))
    }

    /// Zero sensory memory for `objects` objects at the key's size, if there is none yet.
    pub(crate) fn initialize_sensory_if_needed(&mut self, _keys: &Keys, _objects: usize) -> Result<()> {
        Err(not_ported("sensory memory"))
    }

    /// The sensory memory, (n, 256 at stride 16).
    pub(crate) fn sensory(&self) -> Result<Feat> {
        Err(not_ported("sensory memory"))
    }

    pub(crate) fn update_sensory(&mut self, _sensory: Feat) -> Result<()> {
        Err(not_ported("sensory memory"))
    }

    /// Drops working and long-term memory's temporary parts (the permanent part stays).
    pub(crate) fn clear_non_permanent_memory(&mut self) {}

    pub(crate) fn clear_sensory_memory(&mut self) {}

    /// The state as the oracle's `state.<prefix>.*` taps.
    pub(crate) fn dump(&self, _prefix: &str, _tap: &mut Tap) -> Result<()> {
        Err(not_ported("dumping memory"))
    }
}

/// Stage `memory`: from the oracle's `state.before.*` of each tapped step, the step's memory work
/// with the oracle's own inputs (keys, `encode_mask.out.*`, `pixel_fusion`/`readout_query` outputs
/// where the read needs the network): `memory.read.*` compared, `add_memory` and consolidation
/// compared through `state.after.*`.
pub(crate) fn check(_oracle: &Oracle, _report: &mut Report, _weights: &mut Weights) -> Result<()> {
    Err(not_ported("the memory check"))
}
