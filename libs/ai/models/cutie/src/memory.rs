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
//!   use count and a life count (f32 on the device, life starts at 1e-7). Each read adds the
//!   read's usage to the use counts and 1 to the life counts (temporary part only), on the
//!   device as PyTorch does; they are downloaded only where the host chooses tokens by them
//!   (consolidation, removing obsolete tokens).
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
//!
//! Ties: `torch.topk` does not say which of equal values it keeps. Every top-k here (the read's
//! per-pixel top-k in `gpu_cutie_topk_softmax`, the prototypes, long-term memory's survivors)
//! keeps the lower index first. That is what the oracles made on CUDA did wherever equal
//! similarities met at the read's k-th place: working memory holds the first frame twice (the
//! commit's permanent memory, then the propagation's first temporary frame), so pairs of
//! identical keys tie. The memory check holds the read to it on those oracles: its row
//! `memory.read.affinity ties` fails where the port kept another of a tied pair
//! (`oracle::TieOrder`). Oracles made on the CPU keep either token of a pair (about half the
//! higher), which is as right: there the read's picks at an exact tie are judged as a set
//! (`oracle::TopKTies`). The oracles have no tie among the prototypes or survivors (the first
//! frame's permanent copy has no counts, so it never ties there), so the rule is unverified there.
//! Long-term memory's survivors are checked by Solege-03-long (`long_term.max_num_tokens=700`,
//! `buffer_tokens=100`), whose step `prop_f165` keeps the 472 most used of 640 before the
//! consolidation adds 128: no tie, the closest neighbours in that order 9.7e-8 apart, and the
//! teacher-forced survivors are the oracle's in its order.

use crate::model::{dev, CutieNet, Feat, Keys, Tap, Weights};
use crate::oracle::{Comparer, Oracle, OracleTap, Report, TieJudgement, Tol, TopKTies};
use crate::processor::{StepPlan, StepSchedule};
use crate::{CutieConfig, DiffusionError, Result};
use makepad_ai_common::gpu::{
    gpu_add, gpu_concat_cols, gpu_concat_rows_many, gpu_cutie_matmul, gpu_cutie_similarity, gpu_cutie_softmax_axis0,
    gpu_cutie_topk_softmax, gpu_download, gpu_gather_cols, gpu_rife_fill, gpu_slice_cols, gpu_slice_rows, gpu_upload, GpuTensor,
};

/// Channels of each object's values in working and long-term memory (`model.value_dim`).
pub(crate) const VALUE_DIM: usize = 256;
/// Channels of each object's sensory memory (`model.sensory_dim`).
pub(crate) const SENSORY_DIM: usize = 256;
/// The life count a new temporary token starts with (`KeyValueMemoryStore.add`: zeros + 1e-7).
const NEW_LIFE: f32 = 1e-7;

/// How an add joins the permanent part (`KeyValueMemoryStore.add`'s `as_permanent`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Permanence {
    No,
    First,
    All,
}

/// One `KeyValueMemoryStore` holding the one bucket: working memory (selection and usage kept)
/// or long-term memory (usage kept). Tensors are None while the store has no tokens (device
/// tensors cannot be empty). Columns are tokens: the permanent part first (`perm_end` columns),
/// then the temporary part, whose selection and counts are kept alone.
pub(crate) struct KeyValueStore {
    /// [64, N].
    pub(crate) key: Option<GpuTensor>,
    /// [1, N].
    pub(crate) shrinkage: Option<GpuTensor>,
    /// Every object's values, object-major rows: [n*256, N].
    pub(crate) values: Option<GpuTensor>,
    /// The temporary part's selection [64, N - perm_end] (`save_selection`).
    pub(crate) selection: Option<GpuTensor>,
    /// The temporary part's use and life counts [1, N - perm_end] (`save_usage`), f32 on the
    /// device as PyTorch accumulates them; None while the temporary part is empty.
    pub(crate) use_count: Option<GpuTensor>,
    pub(crate) life_count: Option<GpuTensor>,
    /// Columns of the permanent part (`perm_end_pt`).
    pub(crate) perm_end: usize,
    save_selection: bool,
    save_usage: bool,
}

/// What `KeyValueStore::candidates` gives: the temporary tokens that consolidate, with their
/// normalised usage (`get_all_sliced`).
pub(crate) struct Candidates {
    pub(crate) key: GpuTensor,
    pub(crate) shrinkage: GpuTensor,
    pub(crate) selection: GpuTensor,
    pub(crate) values: GpuTensor,
    pub(crate) usage: Vec<f32>,
}

impl KeyValueStore {
    pub(crate) fn new(save_selection: bool, save_usage: bool) -> Self {
        Self {
            key: None,
            shrinkage: None,
            values: None,
            selection: None,
            use_count: None,
            life_count: None,
            perm_end: 0,
            save_selection,
            save_usage,
        }
    }

    /// Tokens held (`size`).
    pub(crate) fn size(&self) -> usize {
        self.key.as_ref().map_or(0, |k| k.cols())
    }

    /// Tokens in the temporary part (`non_perm_size`).
    pub(crate) fn non_perm_size(&self) -> usize {
        self.size() - self.perm_end
    }

    /// The columns of `parts` that are there, one after another (a view when only one is);
    /// None when none are.
    pub(crate) fn cat(parts: &[Option<&GpuTensor>]) -> Result<Option<GpuTensor>> {
        let present: Vec<&GpuTensor> = parts.iter().flatten().copied().collect();
        match present.len() {
            0 => Ok(None),
            1 => gpu_slice_rows(present[0], 0, present[0].rows()).map(Some).map_err(|e| dev("memory cat", e)),
            _ => gpu_concat_cols(&present).map(Some).map_err(|e| dev("memory cat", e)),
        }
    }

    /// `old` with `new`'s columns after it, or before it when `prepend` (`_add_last_dim`).
    fn joined(old: Option<&GpuTensor>, new: &GpuTensor, prepend: bool) -> Result<GpuTensor> {
        let parts = if prepend { [Some(new), old] } else { [old, Some(new)] };
        Self::cat(&parts)?.ok_or_else(|| DiffusionError::model("cutie memory add: nothing"))
    }

    /// Columns `start..start + len` of `t` (a copy), None when empty.
    fn cols(t: &GpuTensor, start: usize, len: usize) -> Result<Option<GpuTensor>> {
        if len == 0 {
            return Ok(None);
        }
        gpu_slice_cols(t, start, len).map(Some).map_err(|e| dev("memory slice", e))
    }

    /// `KeyValueMemoryStore.add` (one bucket): `key` [64, ne], `shrinkage` [1, ne], `selection`
    /// [64, ne] (kept for the temporary part when the store keeps selection), `values` [n*256, ne].
    /// Keys and shrinkage go first when they join the permanent part, values only on 'all', as
    /// PyTorch does (the two differ only for 'first' on a store with temporary tokens and no
    /// permanent part, which the processor never makes).
    pub(crate) fn add(
        &mut self, key: &GpuTensor, shrinkage: &GpuTensor, selection: Option<&GpuTensor>, values: &GpuTensor, as_permanent: Permanence,
    ) -> Result<()> {
        let ne = key.cols();
        let add_as_permanent = match as_permanent {
            Permanence::All => {
                self.perm_end += ne;
                true
            }
            Permanence::First if self.perm_end == 0 => {
                self.perm_end = ne;
                true
            }
            _ => false,
        };
        self.values = Some(Self::joined(self.values.as_ref(), values, as_permanent == Permanence::All)?);
        self.key = Some(Self::joined(self.key.as_ref(), key, add_as_permanent)?);
        self.shrinkage = Some(Self::joined(self.shrinkage.as_ref(), shrinkage, add_as_permanent)?);
        if !add_as_permanent {
            if self.save_selection {
                let selection = selection.ok_or_else(|| DiffusionError::model("cutie memory add: working memory wants the selection"))?;
                self.selection = Some(Self::joined(self.selection.as_ref(), selection, false)?);
            }
            if self.save_usage {
                let fill = |value: f32| gpu_rife_fill(1, ne, value).map_err(|e| dev("memory counts", e));
                self.use_count = Some(Self::joined(self.use_count.as_ref(), &fill(0.0)?, false)?);
                self.life_count = Some(Self::joined(self.life_count.as_ref(), &fill(0.0 + NEW_LIFE)?, false)?);
            }
        }
        Ok(())
    }

    /// `update_bucket_usage`: this store's share of a read's usage is columns `start..start +
    /// size` of `usage` [1, N] (the permanent part first); the temporary tokens' part is added to
    /// their use counts and 1 to their life counts, on the device.
    pub(crate) fn update_usage(&mut self, usage: &GpuTensor, start: usize) -> Result<()> {
        let temp = self.non_perm_size();
        if !self.save_usage || temp == 0 {
            return Ok(());
        }
        let (Some(use_count), Some(life_count)) = (&self.use_count, &self.life_count) else {
            return Err(DiffusionError::model("cutie memory: temporary tokens without counts"));
        };
        let usage = gpu_slice_cols(usage, start + self.perm_end, temp).map_err(|e| dev("memory usage", e))?;
        let one = gpu_rife_fill(1, temp, 1.0).map_err(|e| dev("memory usage", e))?;
        self.use_count = Some(gpu_add(use_count, &usage).map_err(|e| dev("memory usage", e))?);
        self.life_count = Some(gpu_add(life_count, &one).map_err(|e| dev("memory usage", e))?);
        Ok(())
    }

    /// `get_usage`: the temporary tokens' use counts normalised by their lives, downloaded for
    /// the host's top-k (empty while the temporary part is). A store that does not count usage
    /// has none to give (PyTorch raises too).
    pub(crate) fn usage(&self) -> Result<Vec<f32>> {
        if !self.save_usage {
            return Err(DiffusionError::model("cutie memory: this store does not count usage"));
        }
        let (Some(use_count), Some(life_count)) = (&self.use_count, &self.life_count) else {
            return Ok(Vec::new());
        };
        let use_count = gpu_download(use_count).map_err(|e| dev("memory usage", e))?;
        let life_count = gpu_download(life_count).map_err(|e| dev("memory usage", e))?;
        Ok(use_count.iter().zip(&life_count).map(|(u, l)| u / l).collect())
    }

    /// The indices of the `k` largest of `values`, largest first; equal values by lower index
    /// (`torch.topk(.., sorted=True)`). The tie rule copies what CUDA's oracles showed for the
    /// read's top-k; it is unverified here, no oracle having an exact tie in its usage there (the
    /// memory check records it: `OracleMemory::record_usage_ties`).
    /// k 0 gives none, as torch.topk does; more than there are is an error, as there.
    pub(crate) fn most_used(values: &[f32], k: usize) -> Result<Vec<u32>> {
        if k > values.len() {
            return Err(DiffusionError::model(format!("cutie memory: top {k} of {} tokens", values.len())));
        }
        let mut order: Vec<u32> = (0..values.len() as u32).collect();
        order.sort_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]).then(a.cmp(&b)));
        order.truncate(k);
        Ok(order)
    }

    /// `get_all_sliced(0, -keep_last)`: the temporary tokens but the last `keep_last`.
    pub(crate) fn candidates(&self, keep_last: usize) -> Result<Candidates> {
        let (Some(key), Some(shrinkage), Some(values), Some(selection)) = (&self.key, &self.shrinkage, &self.values, &self.selection) else {
            return Err(DiffusionError::model("cutie memory: consolidating a store without temporary tokens"));
        };
        let m = self.non_perm_size().saturating_sub(keep_last);
        let none = || DiffusionError::model("cutie memory: nothing to consolidate");
        let mut usage = self.usage()?;
        if usage.len() < m {
            return Err(DiffusionError::model("cutie memory: consolidating tokens without counts"));
        }
        usage.truncate(m);
        Ok(Candidates {
            key: Self::cols(key, self.perm_end, m)?.ok_or_else(none)?,
            shrinkage: Self::cols(shrinkage, self.perm_end, m)?.ok_or_else(none)?,
            selection: Self::cols(selection, 0, m)?.ok_or_else(none)?,
            values: Self::cols(values, self.perm_end, m)?.ok_or_else(none)?,
            usage,
        })
    }

    /// `sieve_by_range(0, -keep_last, min_size=keep_last)`: keeps the permanent part and the last
    /// `keep_last` temporary tokens; a temporary part of at most `keep_last` tokens stays whole.
    /// `keep_last` 0 drops the temporary part (`clear_non_permanent_memory`).
    pub(crate) fn sieve(&mut self, keep_last: usize) -> Result<()> {
        let temp = self.non_perm_size();
        if temp <= keep_last {
            return Ok(());
        }
        let (size, perm) = (self.size(), self.perm_end);
        let keep = |t: &Option<GpuTensor>| -> Result<Option<GpuTensor>> {
            match t {
                Some(t) => Self::cat(&[Self::cols(t, 0, perm)?.as_ref(), Self::cols(t, size - keep_last, keep_last)?.as_ref()]),
                None => Ok(None),
            }
        };
        // The temporary part's own tensors (selection, counts): their last `keep_last` columns.
        let keep_temp = |t: &Option<GpuTensor>| -> Result<Option<GpuTensor>> {
            match t {
                Some(t) => Self::cols(t, temp - keep_last, keep_last),
                None => Ok(None),
            }
        };
        self.key = keep(&self.key)?;
        self.shrinkage = keep(&self.shrinkage)?;
        self.values = keep(&self.values)?;
        self.selection = keep_temp(&self.selection)?;
        self.use_count = keep_temp(&self.use_count)?;
        self.life_count = keep_temp(&self.life_count)?;
        Ok(())
    }

    /// `remove_obsolete_features` (long-term memory: no permanent part, no selection): keeps the
    /// `max_size` most used tokens, most used first; none when `max_size` is 0 (torch.topk with
    /// k 0: the store empties, and the consolidation after it adds its prototypes alone).
    pub(crate) fn remove_obsolete(&mut self, max_size: usize) -> Result<()> {
        if self.perm_end != 0 {
            return Err(DiffusionError::model("cutie memory: long-term memory has a permanent part"));
        }
        let survivors = Self::most_used(&self.usage()?, max_size)?;
        if survivors.is_empty() {
            // Device tensors cannot be empty: an empty store has none.
            *self = Self::new(self.save_selection, self.save_usage);
            return Ok(());
        }
        let gather = |t: &Option<GpuTensor>| -> Result<Option<GpuTensor>> {
            t.as_ref().map(|t| gpu_gather_cols(t, &survivors)).transpose().map_err(|e| dev("memory survivors", e))
        };
        self.key = gather(&self.key)?;
        self.shrinkage = gather(&self.shrinkage)?;
        self.values = gather(&self.values)?;
        self.selection = gather(&self.selection)?;
        self.use_count = gather(&self.use_count)?;
        self.life_count = gather(&self.life_count)?;
        Ok(())
    }

    /// The store as the oracle's `<prefix>.key`, `.shrinkage`, `.selection`, `.use_count`,
    /// `.life_count`, `.value.obj{o}` (what it has).
    pub(crate) fn dump(&self, prefix: &str, objects: usize, tap: &mut Tap) -> Result<()> {
        let (Some(key), Some(shrinkage), Some(values)) = (&self.key, &self.shrinkage, &self.values) else {
            return Ok(());
        };
        tap.put(&format!("{prefix}.key"), key)?;
        tap.put(&format!("{prefix}.shrinkage"), shrinkage)?;
        if let Some(selection) = &self.selection {
            tap.put(&format!("{prefix}.selection"), selection)?;
        }
        if let (Some(use_count), Some(life_count)) = (&self.use_count, &self.life_count) {
            tap.put(&format!("{prefix}.use_count"), use_count)?;
            tap.put(&format!("{prefix}.life_count"), life_count)?;
        }
        let per = values.rows() / objects.max(1);
        for o in 0..objects {
            tap.put(&format!("{prefix}.value.obj{}", o + 1), &gpu_slice_rows(values, o * per, per).map_err(|e| dev("memory dump", e))?)?;
        }
        Ok(())
    }
}

/// `MemoryManager` for the one bucket, with Cutie's settings.
pub(crate) struct MemoryManager {
    top_k: usize,
    use_long_term: bool,
    count_long_term_usage: bool,
    /// Frames of working memory's temporary part that make it consolidate (or, without
    /// long-term memory, that it keeps): `max_mem_frames - 1`, the permanent part not counted.
    max_mem_frames: usize,
    /// Frames of the temporary part a consolidation leaves: `min_mem_frames - 1`.
    min_mem_frames: usize,
    num_prototypes: usize,
    max_long_tokens: usize,
    buffer_tokens: usize,
    /// A memorised frame's size at stride 16 (w, h), from the first add or sensory memory (`H`,
    /// `W`): every frame stepped must have it until the memory is cleared.
    size: Option<(usize, usize)>,
    pub(crate) work: KeyValueStore,
    pub(crate) long: KeyValueStore,
    /// Every object's sensory memory (n, 256 at stride 16).
    sensory: Option<Feat>,
    /// Every object's summaries summed, [n*16, 257] (`obj_v`).
    obj_memory: Option<GpuTensor>,
    /// The objects held (the processor's slots), from the first add or sensory memory.
    objects: usize,
    engaged: bool,
    /// How many times working memory has consolidated (the run check's schedule).
    pub(crate) consolidations: usize,
    /// How many times long-term memory has dropped its least used tokens before a consolidation
    /// (`remove_obsolete_features`; the schedule's `obsolete_removed`).
    pub(crate) obsolete_removals: usize,
}

impl MemoryManager {
    pub(crate) fn new(config: &CutieConfig) -> Self {
        Self {
            top_k: config.top_k,
            use_long_term: config.use_long_term,
            count_long_term_usage: config.count_long_term_usage,
            max_mem_frames: if config.use_long_term { config.max_mem_frames } else { config.max_mem_frames_without_long_term }.saturating_sub(1),
            min_mem_frames: config.min_mem_frames.saturating_sub(1),
            num_prototypes: config.num_prototypes,
            max_long_tokens: config.max_long_tokens,
            buffer_tokens: config.buffer_tokens,
            size: None,
            work: KeyValueStore::new(config.use_long_term, config.use_long_term),
            long: KeyValueStore::new(false, config.count_long_term_usage),
            sensory: None,
            obj_memory: None,
            objects: 0,
            engaged: false,
            consolidations: 0,
            obsolete_removals: 0,
        }
    }

    /// Whether anything has been memorised.
    pub(crate) fn engaged(&self) -> bool {
        self.engaged
    }

    /// The stride-16 size (w, h) the memory holds frames at, once anything is memorised or the
    /// sensory memory made (until `clear_memory`, which makes a new manager).
    pub(crate) fn frame_size(&self) -> Option<(usize, usize)> {
        self.size
    }

    /// Working memory's tokens, its permanent part's, and long-term memory's (the schedule's
    /// `work_tokens`, `work_permanent`, `long_tokens`).
    pub(crate) fn sizes(&self) -> (usize, usize, usize) {
        (self.work.size(), self.work.perm_end, if self.use_long_term { self.long.size() } else { 0 })
    }

    /// `MemoryManager.read`: the memory readout for every object (n, 256 at stride 16), from the
    /// frame's `pix_feat` and `keys` and the `last_mask` (n images of 1 channel at the padded H,
    /// W). Updates the usage counts. Taps `memory.read.memory_key`, `.memory_shrinkage`,
    /// `.query_key` ([1, 64, h*w]), `.query_selection`, `.similarity`, `.affinity`, `.usage`,
    /// `.readout` ([1, n, 256, h*w]), then the network's own taps (pixel fusion, transformer).
    pub(crate) fn read(&mut self, net: &CutieNet, pix_feat: &Feat, keys: &Keys, last_mask: &Feat, tap: &mut Tap) -> Result<Feat> {
        let affinity = self.read_affinity(keys, tap)?;
        let visual_readout = self.read_values(&affinity, keys, tap)?;
        let sensory = self.sensory()?;
        let pixel = net.pixel_fusion(pix_feat, &visual_readout, &sensory, last_mask, tap)?;
        let obj_memory = self.obj_memory.as_ref().ok_or_else(|| DiffusionError::model("cutie memory read: no object memory"))?;
        let (readout, _aux_logits) = net.readout_query(&pixel, obj_memory, tap)?;
        Ok(readout)
    }

    /// The read's first half: the query's similarity to every memory token ([long-term,
    /// working]), the top-k softmax, and the usage counted. Out: the affinity [N, h*w].
    pub(crate) fn read_affinity(&mut self, keys: &Keys, tap: &mut Tap) -> Result<GpuTensor> {
        let (affinity, usage) = self.read_top_k(keys, tap)?;
        self.count_usage(&affinity, &usage, tap)?;
        Ok(affinity)
    }

    /// Long-term memory's tokens in a read (none without long-term memory).
    fn long_size(&self) -> usize {
        if self.use_long_term {
            self.long.size()
        } else {
            0
        }
    }

    /// The query's similarity to every memory token ([long-term, working]) and the top-k
    /// softmax: the affinity [N, h*w] and each token's usage [1, N]. Taps the read up to the
    /// similarity.
    pub(crate) fn read_top_k(&self, keys: &Keys, tap: &mut Tap) -> Result<(GpuTensor, GpuTensor)> {
        let long_size = self.long_size();
        let (long_key, long_shrinkage) = if long_size > 0 { (self.long.key.as_ref(), self.long.shrinkage.as_ref()) } else { (None, None) };
        let memory_key = KeyValueStore::cat(&[long_key, self.work.key.as_ref()])?;
        let memory_shrinkage = KeyValueStore::cat(&[long_shrinkage, self.work.shrinkage.as_ref()])?;
        let (Some(memory_key), Some(memory_shrinkage)) = (memory_key, memory_shrinkage) else {
            return Err(DiffusionError::model("cutie memory read: nothing memorised"));
        };
        // `torch.topk` raises when memory holds fewer tokens than top_k (a frame of under 30
        // tokens at stride 16); `gpu_cutie_topk_softmax` would take them all instead.
        if self.top_k > memory_key.cols() {
            return Err(DiffusionError::model(format!("cutie memory read: top {} of {} memory tokens", self.top_k, memory_key.cols())));
        }
        tap.put("memory.read.memory_key", &memory_key)?;
        tap.put("memory.read.memory_shrinkage", &memory_shrinkage)?;
        tap.put("memory.read.query_key", &keys.key.t)?;
        tap.put("memory.read.query_selection", &keys.selection.t)?;
        let similarity = gpu_cutie_similarity(&memory_key, Some(&memory_shrinkage), &keys.key.t, Some(&keys.selection.t))
            .map_err(|e| dev("memory similarity", e))?;
        tap.put("memory.read.similarity", &similarity)?;
        gpu_cutie_topk_softmax(&similarity, self.top_k).map_err(|e| dev("memory top-k", e))
    }

    /// Taps the read's affinity and usage (`read_top_k`'s) and counts the usage: working
    /// memory's share, and long-term memory's when it is counted.
    pub(crate) fn count_usage(&mut self, affinity: &GpuTensor, usage: &GpuTensor, tap: &mut Tap) -> Result<()> {
        tap.put("memory.read.affinity", affinity)?;
        if self.use_long_term {
            let long_size = self.long_size();
            tap.put("memory.read.usage", usage)?;
            self.work.update_usage(usage, long_size)?;
            if long_size > 0 && self.count_long_term_usage {
                self.long.update_usage(usage, 0)?;
            }
        }
        Ok(())
    }

    /// The read's second half: every object's values ([long-term, working]) read through the
    /// affinity, all objects at once. Out: the visual readout (n, 256 at the key's size).
    pub(crate) fn read_values(&self, affinity: &GpuTensor, keys: &Keys, tap: &mut Tap) -> Result<Feat> {
        let long_values = if self.use_long_term { self.long.values.as_ref() } else { None };
        let values = KeyValueStore::cat(&[long_values, self.work.values.as_ref()])?
            .ok_or_else(|| DiffusionError::model("cutie memory read: no values"))?;
        let readout = gpu_cutie_matmul(&values, false, affinity, false).map_err(|e| dev("memory readout", e))?;
        tap.put("memory.read.readout", &readout)?;
        Feat::new(readout, values.rows() / VALUE_DIM, VALUE_DIM, keys.key.w, keys.key.h)
    }

    /// `MemoryManager.add_memory`: memorises a frame's key, shrinkage and selection with each
    /// object's `value` (n, 256 at stride 16) and summaries `obj_value` [n*16, 257], then
    /// consolidates if working memory is full. Taps the consolidation as
    /// `memory.consolidate.memory_key`, `.memory_shrinkage`, `.query_key`, `.query_selection`,
    /// `.similarity`, `.affinity`, `.readout` (each object's prototype values, then the
    /// shrinkage as `.readout#n`).
    pub(crate) fn add_memory(&mut self, keys: &Keys, value: &Feat, obj_value: &GpuTensor, as_permanent: Permanence, tap: &mut Tap) -> Result<()> {
        self.engaged = true;
        let (w, h) = *self.size.get_or_insert((value.w, value.h));
        let plane = w * h;
        self.objects = value.n;
        // The object memory: a running sum of each object's summaries and areas.
        self.obj_memory = Some(match &self.obj_memory {
            Some(sum) => gpu_add(sum, obj_value).map_err(|e| dev("object memory", e))?,
            None => gpu_slice_rows(obj_value, 0, obj_value.rows()).map_err(|e| dev("object memory", e))?,
        });
        self.work.add(&keys.key.t, &keys.shrinkage.t, Some(&keys.selection.t), &value.t, as_permanent)?;
        let max_work_tokens = self.max_mem_frames * plane;
        if self.use_long_term {
            if self.work.non_perm_size() >= max_work_tokens {
                let reserve = self.max_long_tokens.saturating_sub(self.num_prototypes);
                if self.long.non_perm_size() >= reserve {
                    // What survives, max_num_tokens - num_prototypes - buffer_tokens, may be 0
                    // (every token goes); below that torch.topk refuses, and so does this.
                    let keep = self.max_long_tokens as i64 - self.num_prototypes as i64 - self.buffer_tokens as i64;
                    let keep = usize::try_from(keep).map_err(|_| {
                        DiffusionError::model(format!(
                            "cutie memory: long-term memory would keep {keep} tokens (max_long_tokens under num_prototypes + buffer_tokens)"
                        ))
                    })?;
                    self.long.remove_obsolete(keep)?;
                    self.obsolete_removals += 1;
                }
                self.consolidate(self.min_mem_frames * plane, tap)?;
            }
        } else {
            // FIFO (`remove_old_memory`).
            self.work.sieve(max_work_tokens)?;
        }
        Ok(())
    }

    /// `compress_features` and `consolidation`: working memory's temporary tokens but the last
    /// `min_work_tokens` become `num_prototypes` long-term tokens.
    fn consolidate(&mut self, min_work_tokens: usize, tap: &mut Tap) -> Result<()> {
        let c = self.work.candidates(min_work_tokens)?;
        let prototypes = KeyValueStore::most_used(&c.usage, self.num_prototypes)?;
        let prototype_key = gpu_gather_cols(&c.key, &prototypes).map_err(|e| dev("prototypes", e))?;
        let prototype_selection = gpu_gather_cols(&c.selection, &prototypes).map_err(|e| dev("prototypes", e))?;
        tap.put("memory.consolidate.memory_key", &c.key)?;
        tap.put("memory.consolidate.memory_shrinkage", &c.shrinkage)?;
        tap.put("memory.consolidate.query_key", &prototype_key)?;
        tap.put("memory.consolidate.query_selection", &prototype_selection)?;
        // Potentiation: each prototype reads every candidate, a full softmax over the candidates.
        let similarity = gpu_cutie_similarity(&c.key, Some(&c.shrinkage), &prototype_key, Some(&prototype_selection))
            .map_err(|e| dev("consolidation similarity", e))?;
        tap.put("memory.consolidate.similarity", &similarity)?;
        let affinity = gpu_cutie_softmax_axis0(&similarity).map_err(|e| dev("consolidation softmax", e))?;
        tap.put("memory.consolidate.affinity", &affinity)?;
        let values = gpu_cutie_matmul(&c.values, false, &affinity, false).map_err(|e| dev("consolidation readout", e))?;
        for o in 0..values.rows() / VALUE_DIM {
            tap.put("memory.consolidate.readout", &gpu_slice_rows(&values, o * VALUE_DIM, VALUE_DIM).map_err(|e| dev("consolidation", e))?)?;
        }
        let shrinkage = gpu_cutie_matmul(&c.shrinkage, false, &affinity, false).map_err(|e| dev("consolidation shrinkage", e))?;
        tap.put("memory.consolidate.readout", &shrinkage)?;
        self.work.sieve(min_work_tokens)?;
        self.long.add(&prototype_key, &shrinkage, None, &values, Permanence::No)?;
        self.consolidations += 1;
        Ok(())
    }

    /// Zero sensory memory for `objects` objects at the key's size, if there is none yet.
    pub(crate) fn initialize_sensory_if_needed(&mut self, keys: &Keys, objects: usize) -> Result<()> {
        if self.sensory.is_none() {
            let (w, h) = (keys.key.w, keys.key.h);
            let zeros = gpu_rife_fill(objects * SENSORY_DIM, w * h, 0.0).map_err(|e| dev("sensory memory", e))?;
            self.sensory = Some(Feat::new(zeros, objects, SENSORY_DIM, w, h)?);
            self.objects = objects;
            self.size.get_or_insert((w, h));
        }
        Ok(())
    }

    /// The sensory memory, (n, 256 at stride 16).
    pub(crate) fn sensory(&self) -> Result<Feat> {
        self.sensory.as_ref().ok_or_else(|| DiffusionError::model("cutie memory: no sensory memory"))?.view()
    }

    pub(crate) fn update_sensory(&mut self, sensory: Feat) -> Result<()> {
        if sensory.c != SENSORY_DIM {
            return Err(DiffusionError::model(format!("cutie sensory memory: {} channels", sensory.c)));
        }
        self.objects = sensory.n;
        self.sensory = Some(sensory);
        Ok(())
    }

    /// The object memory, [n*16, 257].
    pub(crate) fn obj_memory(&self) -> Option<&GpuTensor> {
        self.obj_memory.as_ref()
    }

    /// Drops working and long-term memory's temporary parts (the permanent part stays).
    pub(crate) fn clear_non_permanent_memory(&mut self) -> Result<()> {
        self.work.sieve(0)?;
        if self.use_long_term {
            self.long.sieve(0)?;
        }
        Ok(())
    }

    pub(crate) fn clear_sensory_memory(&mut self) {
        self.sensory = None;
    }

    /// The state as the oracle's `state.<prefix>.*` taps (the processor adds `last_mask`).
    pub(crate) fn dump(&self, prefix: &str, tap: &mut Tap) -> Result<()> {
        let objects = self.objects;
        self.work.dump(&format!("{prefix}.work_mem.b0"), objects, tap)?;
        if self.use_long_term {
            self.long.dump(&format!("{prefix}.long_mem.b0"), objects, tap)?;
        }
        if let Some(sensory) = &self.sensory {
            for o in 0..sensory.n {
                tap.put(&format!("{prefix}.sensory.obj{}", o + 1), &sensory.image(o)?.t)?;
            }
        }
        if let Some(obj_memory) = &self.obj_memory {
            let per = obj_memory.rows() / objects.max(1);
            for o in 0..objects {
                tap.put(&format!("{prefix}.obj_memory.obj{}", o + 1), &gpu_slice_rows(obj_memory, o * per, per).map_err(|e| dev("memory dump", e))?)?;
            }
        }
        Ok(())
    }
}

// --- The check ---

/// The oracle's memory state and inputs, loaded for the teacher-forced check.
struct OracleMemory;

impl OracleMemory {
    /// A memory as the oracle's `state.before.*` of `step` has it, at a stride-16 size of w x h.
    fn manager(oracle: &Oracle, step: &str, config: &CutieConfig, w: usize, h: usize) -> Result<MemoryManager> {
        let mut m = MemoryManager::new(config);
        let objects = oracle.objects;
        m.objects = objects;
        let (save_selection, save_usage) = (m.work.save_selection, m.work.save_usage);
        if let Some(work) = Self::store(oracle, step, "state.before.work_mem.b0", objects, save_selection, save_usage)? {
            m.work = work;
            m.engaged = true;
            m.size = Some((w, h));
        }
        let save_usage = m.long.save_usage;
        if let Some(long) = Self::store(oracle, step, "state.before.long_mem.b0", objects, false, save_usage)? {
            m.long = long;
        }
        if oracle.has(step, "state.before.sensory.obj1") {
            let t = Self::objects(oracle, step, "state.before.sensory", objects, SENSORY_DIM, w * h)?;
            m.sensory = Some(Feat::new(t, objects, SENSORY_DIM, w, h)?);
        }
        if oracle.has(step, "state.before.obj_memory.obj1") {
            let shape = oracle.shape(step, "state.before.obj_memory.obj1")?;
            let (rows, cols) = (shape[shape.len() - 2], shape[shape.len() - 1]);
            m.obj_memory = Some(Self::objects(oracle, step, "state.before.obj_memory", objects, rows, cols)?);
        }
        Ok(m)
    }

    /// A store as the oracle's `<prefix>.*` taps have it, None when the oracle has none. The
    /// permanent part is what has no counts. Zero-size taps (PyTorch keeps emptied tensors) are
    /// absent tensors here.
    fn store(oracle: &Oracle, step: &str, prefix: &str, objects: usize, save_selection: bool, save_usage: bool) -> Result<Option<KeyValueStore>> {
        if !oracle.has(step, &format!("{prefix}.key")) {
            return Ok(None);
        }
        let mut s = KeyValueStore::new(save_selection, save_usage);
        let key = oracle.tap(step, &format!("{prefix}.key"))?;
        let n = *key.shape.last().unwrap_or(&0);
        if n == 0 {
            return Ok(None);
        }
        let ck = key.values.len() / n;
        s.key = Some(gpu_upload(&key.values, ck, n).map_err(|e| dev("oracle state", e))?);
        s.shrinkage = Some(Self::upload(oracle, step, &format!("{prefix}.shrinkage"), 1, n)?);
        s.values = Some(Self::objects(oracle, step, &format!("{prefix}.value"), objects, VALUE_DIM, n)?);
        // The temporary part's own tensors, [rows, temporary tokens].
        let temp = |name: &str, rows: usize| -> Result<Option<GpuTensor>> {
            let name = format!("{prefix}.{name}");
            if !oracle.has(step, &name) {
                return Ok(None);
            }
            let t = oracle.tap(step, &name)?;
            let cols = t.values.len() / rows;
            if cols == 0 {
                return Ok(None);
            }
            gpu_upload(&t.values, rows, cols).map(Some).map_err(|e| dev("oracle state", e))
        };
        s.selection = temp("selection", ck)?;
        s.use_count = temp("use_count", 1)?;
        s.life_count = temp("life_count", 1)?;
        s.perm_end = n - s.use_count.as_ref().map_or(0, |c| c.cols());
        Ok(Some(s))
    }

    /// A tap uploaded as [rows, cols].
    fn upload(oracle: &Oracle, step: &str, name: &str, rows: usize, cols: usize) -> Result<GpuTensor> {
        let t = oracle.tap(step, name)?;
        gpu_upload(&t.values, rows, cols).map_err(|e| dev(name, e))
    }

    /// Each object's `<prefix>.obj{o}` [rows, cols] stacked object-major: [objects*rows, cols].
    fn objects(oracle: &Oracle, step: &str, prefix: &str, objects: usize, rows: usize, cols: usize) -> Result<GpuTensor> {
        let parts = (1..=objects).map(|o| Self::upload(oracle, step, &format!("{prefix}.obj{o}"), rows, cols)).collect::<Result<Vec<_>>>()?;
        let refs: Vec<&GpuTensor> = parts.iter().collect();
        gpu_concat_rows_many(&refs).map_err(|e| dev(prefix, e))
    }

    /// The step's `key.*`, as the key projection gives them.
    fn keys(oracle: &Oracle, step: &str) -> Result<Keys> {
        Ok(Keys {
            key: oracle.feat(step, "key.key")?,
            shrinkage: oracle.feat(step, "key.shrinkage")?,
            selection: oracle.feat(step, "key.selection")?,
        })
    }

    /// The read's top-k (`read_top_k`'s affinity and usage) with its exact ties broken as the
    /// oracle broke them (`TopKTies`): where the port kept other tied tokens than the oracle, the
    /// affinity with their weights moved onto the oracle's tokens and the usage likewise,
    /// uploaded, so that the usage counted, the readout and the consolidation after them go on
    /// from the oracle's picks. The port's own tensors where it kept the same tokens.
    fn break_ties_as_oracle(
        ties: &TopKTies, expected: &OracleTap, affinity: GpuTensor, usage: GpuTensor,
    ) -> Result<(GpuTensor, GpuTensor, TieJudgement)> {
        let mut aligned = gpu_download(&affinity).map_err(|e| dev("memory ties", e))?;
        let judgement = ties.align(&mut aligned, &expected.values);
        if judgement.moved == 0 {
            return Ok((affinity, usage, judgement));
        }
        let mut sums = gpu_download(&usage).map_err(|e| dev("memory ties", e))?;
        judgement.align_sums(&mut sums);
        let affinity = gpu_upload(&aligned, affinity.rows(), affinity.cols()).map_err(|e| dev("memory ties", e))?;
        let usage = gpu_upload(&sums, usage.rows(), usage.cols()).map_err(|e| dev("memory ties", e))?;
        Ok((affinity, usage, judgement))
    }

    /// A store's temporary tokens' normalised usage as the oracle's PyTorch had it at the step's
    /// add (`get_usage` after the read's `update_bucket_usage`: use + usage and life + 1, then
    /// use / life, each in f32 as there): from `<prefix>.use_count` and `.life_count` before the
    /// step and, when the step read and counted this store, its share of `memory.read.usage`
    /// from column `start`. None when the oracle has no counts for the store.
    fn usage(oracle: &Oracle, step: &str, prefix: &str, start: Option<usize>) -> Result<Option<Vec<f32>>> {
        let (use_name, life_name) = (format!("{prefix}.use_count"), format!("{prefix}.life_count"));
        if !oracle.has(step, &use_name) || !oracle.has(step, &life_name) {
            return Ok(None);
        }
        let mut use_count = oracle.tap(step, &use_name)?.values;
        let mut life_count = oracle.tap(step, &life_name)?.values;
        if use_count.len() != life_count.len() {
            let counts = format!("{} use and {} life counts", use_count.len(), life_count.len());
            return Err(DiffusionError::model(format!("cutie oracle {step}: {prefix} has {counts}")));
        }
        if let Some(start) = start.filter(|_| oracle.has(step, "memory.read.usage") && !use_count.is_empty()) {
            let usage = oracle.tap(step, "memory.read.usage")?.values;
            let share = usage
                .get(start..start + use_count.len())
                .ok_or_else(|| DiffusionError::model(format!("cutie oracle {step}: the read's usage has no columns {start}.. for {prefix}")))?;
            for ((u, l), s) in use_count.iter_mut().zip(life_count.iter_mut()).zip(share) {
                *u += s;
                *l += 1.0;
            }
        }
        Ok(Some(use_count.iter().zip(&life_count).map(|(u, l)| u / l).collect()))
    }

    /// The exact ties a top `k` of `usage` (`torch.topk(.., sorted=True)`, whose order among
    /// equal values is open as at the read) meets: equal neighbours in its order down to the
    /// (k+1)-th, which inside the top k would order the picks and at the cut choose them; and the
    /// smallest gap between those neighbours.
    fn usage_ties(usage: &[f32], k: usize) -> (usize, f32) {
        let mut sorted = usage.to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let end = (k + 1).min(sorted.len());
        let ties = sorted[..end].windows(2).filter(|w| w[0] == w[1]).count();
        let gap = sorted[..end].windows(2).map(|w| w[0] - w[1]).fold(f32::INFINITY, f32::min);
        (ties, gap)
    }

    /// Rows recording the exact ties in the step's top-k by usage, in the oracle's own usage
    /// (`usage`): long-term memory's survivors when it dropped its least used tokens, the
    /// consolidation's prototypes when working memory consolidated. Neither has had one: the
    /// first frame held twice never ties here (its permanent copy has no counts), and in every
    /// oracle the closest neighbours are 9.7e-8 apart or more. Each row passes, its note saying
    /// what it found; with a tie the comparisons after it stay strict (the port keeps the lower
    /// index), so a failure there that this row explains is torch's order, not the port.
    fn record_usage_ties(oracle: &Oracle, report: &mut Report, step: &str, memory: &MemoryManager) -> Result<()> {
        let config = &oracle.config;
        let cols = |name: &str| -> usize {
            if oracle.has(step, name) {
                oracle.shape(step, name).ok().and_then(|s| s.last().copied()).unwrap_or(0)
            } else {
                0
            }
        };
        let long_size = if config.use_long_term { cols("state.before.long_mem.b0.key") } else { 0 };
        let mut record = |name: &str, what: &str, usage: &[f32], k: usize| {
            let (ties, gap) = Self::usage_ties(usage, k);
            let of = usage.len();
            let note = format!(
                "{ties} exact ties among the top {k} of {of} {what} by usage and at the cut (the oracle's own usage); closest neighbours {gap:.3e} apart"
            );
            report.record("memory", step, name, true, note);
        };
        if memory.obsolete_removals > 0 {
            let start = config.count_long_term_usage.then_some(0);
            if let Some(usage) = Self::usage(oracle, step, "state.before.long_mem.b0", start)? {
                let keep = config.max_long_tokens.saturating_sub(config.num_prototypes + config.buffer_tokens);
                record("long_mem.survivors usage ties", "long-term tokens", &usage, keep);
            }
        }
        if memory.consolidations > 0 && oracle.has(step, "memory.consolidate.memory_key") {
            let perm_end = cols("state.before.work_mem.b0.key").saturating_sub(cols("state.before.work_mem.b0.use_count"));
            if let Some(mut usage) = Self::usage(oracle, step, "state.before.work_mem.b0", Some(long_size + perm_end))? {
                // The candidates: the temporary tokens but the last frames kept; when none are
                // kept, the step's own frame too, whose tokens start unused (0 / 1e-7).
                usage.resize(cols("memory.consolidate.memory_key"), 0.0);
                record("memory.consolidate.usage ties", "candidates", &usage, config.num_prototypes);
            }
        }
        Ok(())
    }
}

/// What no oracle reaches, on small stores (step `edge`): long-term memory told to keep no tokens
/// (`long_term.max_num_tokens` = num_prototypes + buffer_tokens: torch.topk with k 0) empties,
/// and the consolidation's prototypes then join it alone; told to keep fewer than none, the add
/// fails, as torch.topk does.
fn check_edges(report: &mut Report) -> Result<()> {
    let fill = |rows: usize, cols: usize, value: f32| gpu_rife_fill(rows, cols, value).map_err(|e| dev("memory edge", e));
    let mut long = KeyValueStore::new(false, true);
    long.add(&fill(64, 5, 1.0)?, &fill(1, 5, 1.0)?, None, &fill(VALUE_DIM, 5, 0.5)?, Permanence::No)?;
    long.remove_obsolete(0)?;
    let emptied = long.size() == 0 && long.values.is_none() && long.use_count.is_none() && long.life_count.is_none();
    long.add(&fill(64, 3, 2.0)?, &fill(1, 3, 1.0)?, None, &fill(VALUE_DIM, 3, 0.5)?, Permanence::No)?;
    let refilled = long.size() == 3 && long.use_count.as_ref().map(|c| c.cols()) == Some(3);
    let note = format!("emptied {emptied}, then {} tokens with {:?} counts", long.size(), long.use_count.as_ref().map(|c| c.cols()));
    report.record("memory", "edge", "long-term memory keeping no tokens", emptied && refilled, note);

    // max_num_tokens under num_prototypes + buffer_tokens: the add that consolidates fails.
    let config = CutieConfig { max_mem_frames: 2, min_mem_frames: 1, num_prototypes: 2, max_long_tokens: 2, buffer_tokens: 1, ..CutieConfig::default() };
    let mut m = MemoryManager::new(&config);
    let keys = Keys {
        key: Feat::new(fill(64, 4, 1.0)?, 1, 64, 2, 2)?,
        shrinkage: Feat::new(fill(1, 4, 1.0)?, 1, 1, 2, 2)?,
        selection: Feat::new(fill(64, 4, 0.5)?, 1, 64, 2, 2)?,
    };
    let value = Feat::new(fill(VALUE_DIM, 4, 0.5)?, 1, VALUE_DIM, 2, 2)?;
    let obj_value = fill(16, 257, 1.0)?;
    m.add_memory(&keys, &value, &obj_value, Permanence::All, &mut Tap::none())?;
    // The first temporary frame consolidates (max_mem_frames 2: one temporary frame); long-term
    // memory, empty, already counts as full (2 - 2 = 0 tokens).
    let refused = m.add_memory(&keys, &value, &obj_value, Permanence::First, &mut Tap::none()).is_err();
    report.record("memory", "edge", "long-term memory keeping fewer than none", refused, format!("the add refused: {refused}"));
    Ok(())
}

/// Stage `memory`: from the oracle's `state.before.*` of each tapped step, the step's memory work
/// with the oracle's own inputs: the read's memory half (similarity, top-k, usage counted,
/// readout) on the oracle's `key.*`, compared with `memory.read.*`; the sensory and object memory
/// the network is given (`pixel_fusion.in.sensory`, `object_transformer.in.obj_memory`,
/// `encode_mask.in.sensory`); the segment's sensory update (`segment.out.sensory`, kept when the
/// step's plan from `StepSchedule` says so, which the next sensory comparison checks); `add_memory`
/// with `encode_mask.out.*` and the step's permanence, its consolidation compared with
/// `memory.consolidate.*`; then the whole state against `state.after.*` (counts, long-term
/// prototypes, the sieve) and the schedule's sizes; then the processor's own work without the
/// network (`processor::check_without_network`: the whole run's schedule, each tapped step's
/// frame and mask handling). The network's half of the read (pixel fusion, the
/// object transformer) is the transformer stage's.
///
/// Some comparisons only check how the oracle's own tensors are laid out and stacked, being
/// made from them: the read's `memory_key`, `memory_shrinkage`, `query_key`, `query_selection`;
/// `pixel_fusion.in.sensory`, `object_transformer.in.obj_memory` and `encode_mask.in.sensory`
/// (but for the sensory update's decision); `state.after.*` of what a step leaves alone (the
/// sensory memory a memorising step replaces with `encode_mask.out.sensory`, keys and values on
/// steps that add nothing); `step.out.prob` on segmenting steps. The rest is computed here.
///
/// Every tap at `Tol::STAGE`, the affinity too. Where the oracle's own similarity has an exact
/// tie at the top k's cut (the first frame held twice: `TopKTies`), the port's picks among the
/// tied tokens are judged as a set: where it kept others than the oracle (CPU torch breaks those
/// ties in an order of its own), its weights are moved onto the oracle's tokens before the
/// affinity and usage are compared, and the usage counted and the readout go on from them, so the
/// consolidation after them is checked against the oracle's own picks. A row a read,
/// `memory.read.affinity ties`, says how many columns were tied and how many were moved. On an
/// oracle made on CUDA, whose torch.topk keeps the lower index as the port does, that row fails
/// where any column was moved or kept another number of the tied tokens (`TieOrder`), so the
/// port's rule stays checked against CUDA's torch. Every other column, and every token outside a
/// tie, is compared as it is: a near tie is judged strictly. The closest pairs at the k-th place
/// that are not exact ties are 15 ulps apart in the CUDA oracles (7.2e-6 at -5.93, lawn_cut
/// prop_f5; 33 ulps, 3.9e-6, at Solege-03 prop_f40) and as close as 3 ulps in oracles made on the
/// CPU (1.4e-6, the PC's CPU lawn_cut prop_f5); the port's similarity (up to 1.5e-5 off in this
/// stage) changes no column's top k in any of the five CUDA oracles, nor in the PC's CPU ones. The
/// margin is thin: a similarity kernel must keep near-tied entries within those few ulps, and a
/// read failing on a CPU oracle at a near tie is to be checked against the oracle's gap there
/// before it is taken for the port's error. The top-k by usage (the consolidation's prototypes,
/// long-term memory's survivors) is recorded likewise (`OracleMemory::record_usage_ties`) and
/// judged strictly.
pub(crate) fn check(oracle: &Oracle, report: &mut Report, _weights: &mut Weights) -> Result<()> {
    check_edges(report)?;
    let config = &oracle.config;
    let schedule = oracle.schedule();
    let objects = oracle.objects;
    let tol = |_: &str| Tol::STAGE;
    // Each row's plan, as the processor decides it over the worker's run.
    let mut replay = StepSchedule::new(config);
    let plans: Vec<StepPlan> = schedule.iter().map(|row| replay.replay(row)).collect();
    for step in oracle.steps() {
        let (index, row) = schedule
            .iter()
            .enumerate()
            .find(|(_, r)| r.tap.as_deref() == Some(step))
            .ok_or_else(|| DiffusionError::model(format!("cutie oracle: {step} is not in the schedule")))?;
        let plan = plans[index];
        let keys = OracleMemory::keys(oracle, step)?;
        let (w, h) = (keys.key.w, keys.key.h);
        let mut memory = OracleMemory::manager(oracle, step, config, w, h)?;
        let mut cmp = Comparer::new(oracle, report, "memory", step, &tol);
        // A step that segments reads memory: without its read's taps the read would go
        // unchecked (and the state after it with it).
        if oracle.has(step, "segment.out.prob") && !oracle.has(step, "memory.read.similarity") {
            cmp.report.fail("memory", step, "memory.read.similarity", "not in the oracle, which segments here");
        }
        if oracle.has(step, "memory.read.similarity") {
            let ties = TopKTies::find(&oracle.tap(step, "memory.read.similarity")?, config.top_k)?;
            let expected_affinity = if oracle.has(step, "memory.read.affinity") { Some(oracle.tap(step, "memory.read.affinity")?) } else { None };
            let judgement = cmp.with(|tap| {
                tap.put("pixel_fusion.in.sensory", &memory.sensory()?.t)?;
                if let Some(obj_memory) = memory.obj_memory() {
                    tap.put("object_transformer.in.obj_memory", obj_memory)?;
                }
                let (affinity, usage) = memory.read_top_k(&keys, tap)?;
                let (affinity, usage, judgement) = match &expected_affinity {
                    Some(expected) => OracleMemory::break_ties_as_oracle(&ties, expected, affinity, usage)?,
                    None => (affinity, usage, TieJudgement::default()),
                };
                memory.count_usage(&affinity, &usage, tap)?;
                memory.read_values(&affinity, &keys, tap)?;
                Ok(judgement)
            })?;
            let order = oracle.tie_order;
            cmp.report.record("memory", step, "memory.read.affinity ties", judgement.pass(order, true), judgement.note(config.top_k, order));
            if plan.update_sensory {
                memory.update_sensory(oracle.feat_as(step, "segment.out.sensory", objects, SENSORY_DIM)?)?;
            }
        }
        let memorise = plan.memorise || row.force_permanent;
        if memorise != oracle.has(step, "encode_mask.out.value") {
            cmp.report.fail("memory", step, "schedule.memorise", format!("memorise {memorise}, the oracle {}", !memorise));
        }
        if oracle.has(step, "encode_mask.out.value") {
            memory.initialize_sensory_if_needed(&keys, objects)?;
            cmp.with(|tap| tap.put("encode_mask.in.sensory", &memory.sensory()?.t))?;
            let value = oracle.feat_as(step, "encode_mask.out.value", objects, VALUE_DIM)?;
            let summaries = oracle.tap(step, "encode_mask.out.obj_summaries")?;
            let cols = *summaries.shape.last().unwrap_or(&1);
            let obj_value = gpu_upload(&summaries.values, summaries.values.len() / cols, cols).map_err(|e| dev("oracle summaries", e))?;
            let as_permanent = if row.force_permanent { Permanence::All } else { Permanence::First };
            cmp.with(|tap| memory.add_memory(&keys, &value, &obj_value, as_permanent, tap))?;
            memory.update_sensory(oracle.feat_as(step, "encode_mask.out.sensory", objects, SENSORY_DIM)?)?;
            OracleMemory::record_usage_ties(oracle, cmp.report, step, &memory)?;
        }
        cmp.with(|tap| memory.dump("state.after", tap))?;
        // Every memory tap the oracle has must have been compared (the last mask is the
        // processor's), and every tap put must be in the oracle: a store with counts or a
        // selection where the oracle's has none (a permanent part made temporary) fails.
        cmp.check_coverage(&["memory.", "state.after.work_mem.", "state.after.long_mem.", "state.after.sensory.", "state.after.obj_memory."], &[]);
        // The schedule's sizes after the step, and whether it consolidated and dropped long-term
        // memory's least used tokens (not compared where the oracle did not record it).
        let sizes = memory.sizes();
        let consolidated = memory.consolidations > 0;
        let obsolete_removed = memory.obsolete_removals > 0;
        let expected = (row.work_tokens, row.work_permanent, row.long_tokens);
        let pass = sizes == expected && consolidated == row.consolidated && row.obsolete_removed.is_none_or(|e| e == obsolete_removed);
        let expected_removed = row.obsolete_removed.map_or("-".to_string(), |e| e.to_string());
        let note = format!(
            "tokens {sizes:?} consolidated {consolidated} obsolete removed {obsolete_removed} (oracle {expected:?} {} {expected_removed})",
            row.consolidated
        );
        report.record("memory", step, "schedule", pass, note);
    }
    crate::processor::check_without_network(oracle, report)
}
