//! Cutie's per-frame step (`inference/inference_core.InferenceCore`), driving the network and
//! the memory.
//!
//! A step (`step(image, mask, force_permanent, end)`), with `curr_ti` counting steps since the
//! last clear and `last_mem_ti` the last memorised one:
//! - resize the image (and mask) when its shorter side exceeds `max_internal_size` (bilinear,
//!   align_corners false, bit for bit as PyTorch's CUDA kernel: `gpu_cutie_resize_bilinear`),
//!   then pad to a multiple of 16, centred (left = (W'-W)/2 rounded down, the rest right;
//!   likewise top and bottom), the mask too;
//! - `curr_ti += 1`; memorise when `curr_ti - last_mem_ti >= mem_every` or a mask is given, and
//!   not `end`; segment when there is no mask (with a mask for every object, never); update the
//!   sensory memory when `curr_ti - last_mem_ti` is in the stagger set (1..=mem_every when
//!   stagger_updates >= mem_every, else round(linspace(1, mem_every, stagger_updates))), and not
//!   `end`;
//! - `encode_image`, `transform_key`; segmenting: the memory read, `segment` (its sensory kept
//!   when updating); with a mask: prob = softmax(aggregate(mask)) over background and objects;
//! - `last_mask` = prob without the background; memorising (or `force_permanent`): sensory
//!   memory zeroed if new, `encode_mask` (deep update) on the image with last_mask, `add_memory`
//!   (`Permanence::All` when forced, else `First`), `last_mem_ti = curr_ti`, sensory replaced;
//! - out: prob unpadded (and resized back when it was resized), [(n+1), h*w].
//!
//! `clear_memory` drops everything; `clear_non_permanent_memory` and `clear_sensory_memory`
//! reset `curr_ti` to -1 and `last_mem_ti` to 0 and drop that memory.
//!
//! The objects are fixed for a processor (the worker's slots): every mask given has a channel for
//! each, and memory holds them all in one bucket. The image comes in as the worker gives it to
//! Cutie: RGB bytes scaled to 0..1 as PyTorch's CUDA `x / 255` does it (times the f32 reciprocal).

use crate::memory::{MemoryManager, Permanence, VALUE_DIM};
use crate::model::{dev, CutieNet, Feat, ImageFeatures, Keys, Tap, Weights};
use crate::oracle::{Comparer, Oracle, OracleTap, Report, ScheduleRow, Tol};
use crate::{CutieConfig, DiffusionError, Result};
use makepad_ai_common::gpu::{
    gpu_cutie_aggregate, gpu_cutie_crop, gpu_cutie_pad, gpu_cutie_resize_bilinear, gpu_cutie_rgb_planes, gpu_cutie_softmax_axis0,
    gpu_download, gpu_slice_rows, gpu_upload, GpuTensor,
};

/// Cutie pads every frame to a multiple of this (the network's stride).
const PAD_TO: usize = 16;

/// `pad_divide_by`'s padding: the frame centred in the next multiple of `PAD_TO`, the odd pixel
/// right and bottom.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Padding {
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) top: usize,
    pub(crate) bottom: usize,
}

impl Padding {
    pub(crate) fn divide_by(w: usize, h: usize, d: usize) -> Self {
        let extra = |s: usize| if s % d > 0 { d - s % d } else { 0 };
        let (ew, eh) = (extra(w), extra(h));
        Self { left: ew / 2, right: ew - ew / 2, top: eh / 2, bottom: eh - eh / 2 }
    }

    /// Planar [planes, h*w] padded with zeros.
    fn pad(&self, x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor> {
        gpu_cutie_pad(x, w, h, self.left, self.right, self.top, self.bottom).map_err(|e| dev("pad", e))
    }

    /// `unpad`: planar [planes, H*W] at the padded size back to w x h.
    fn unpad(&self, x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor> {
        gpu_cutie_crop(x, w + self.left + self.right, h + self.top + self.bottom, self.left, self.top, w, h).map_err(|e| dev("unpad", e))
    }
}

/// A frame's sizes through a step: as given (width x height), inside Cutie (w x h, resized when
/// the shorter side is over `max_internal_size`), and padded to the stride.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frame {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) pad: Padding,
}

impl Frame {
    /// `step`'s sizes: the new size is int(side / shorter side * max_internal_size), in f64 as
    /// Python computes it.
    pub(crate) fn new(width: usize, height: usize, max_internal_size: usize) -> Self {
        let min_side = width.min(height);
        let (w, h) = if max_internal_size > 0 && min_side > max_internal_size {
            let scale = |side: usize| (side as f64 / min_side as f64 * max_internal_size as f64) as usize;
            (scale(width), scale(height))
        } else {
            (width, height)
        };
        Self { width, height, w, h, pad: Padding::divide_by(w, h, PAD_TO) }
    }

    pub(crate) fn resized(&self) -> bool {
        (self.w, self.h) != (self.width, self.height)
    }

    /// The padded size inside Cutie.
    pub(crate) fn padded(&self) -> (usize, usize) {
        (self.w + self.pad.left + self.pad.right, self.h + self.pad.top + self.pad.bottom)
    }

    /// RGB bytes (height x width x 3) as Cutie's image: planes [3, width*height] in 0..1, x / 255
    /// as PyTorch's CUDA division by a scalar does it (times the f32 reciprocal), made on the
    /// device from the bytes.
    pub(crate) fn image(&self, rgb: &[u8]) -> Result<GpuTensor> {
        let plane = self.width * self.height;
        if rgb.len() != plane * 3 {
            return Err(DiffusionError::model(format!("cutie step: {} bytes for a {}x{} RGB frame", rgb.len(), self.width, self.height)));
        }
        gpu_cutie_rgb_planes(rgb, plane, 1.0f32 / 255.0f32).map_err(|e| dev("step image", e))
    }

    /// Planar [planes, width*height] at the size given, to the padded size inside Cutie: resized
    /// (`gpu_cutie_resize_bilinear`) if need be, then padded with zeros.
    pub(crate) fn to_internal(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if self.resized() {
            let x = gpu_cutie_resize_bilinear(x, self.width, self.height, self.w, self.h).map_err(|e| dev("step resize", e))?;
            self.pad.pad(&x, self.w, self.h)
        } else {
            self.pad.pad(x, self.w, self.h)
        }
    }

    /// Planar at the padded size back to the size given: unpadded, then resized back.
    pub(crate) fn to_given(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let x = self.pad.unpad(x, self.w, self.h)?;
        if self.resized() {
            gpu_cutie_resize_bilinear(&x, self.w, self.h, self.width, self.height).map_err(|e| dev("step resize", e))
        } else {
            Ok(x)
        }
    }
}

/// The step counters and what they decide (`InferenceCore`'s `curr_ti`, `last_mem_ti` and
/// `stagger_ti`).
pub(crate) struct StepSchedule {
    /// Steps since the last clear (-1 before the first).
    pub(crate) curr_ti: i64,
    /// The last memorised step.
    pub(crate) last_mem_ti: i64,
    mem_every: i64,
    /// The values of `curr_ti - last_mem_ti` on which segmenting updates the sensory memory.
    stagger_ti: Vec<i64>,
}

/// What one step does besides segmenting or taking its mask.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepPlan {
    /// The step is a memory frame (`is_mem_frame`).
    pub(crate) memorise: bool,
    /// Segmenting keeps the decoder's sensory memory.
    pub(crate) update_sensory: bool,
}

impl StepSchedule {
    pub(crate) fn new(config: &CutieConfig) -> Self {
        Self { curr_ti: -1, last_mem_ti: 0, mem_every: config.mem_every as i64, stagger_ti: Self::stagger_ti(config.mem_every, config.stagger_updates) }
    }

    /// `InferenceCore.stagger_ti`: 1..=mem_every, or numpy's round(linspace(1, mem_every,
    /// stagger_updates)) (round half to even) when fewer updates are asked for.
    fn stagger_ti(mem_every: usize, stagger_updates: usize) -> Vec<i64> {
        if stagger_updates >= mem_every {
            return (1..=mem_every as i64).collect();
        }
        let (start, stop) = (1.0f64, mem_every as f64);
        let div = stagger_updates.saturating_sub(1).max(1) as f64;
        let step = (stop - start) / div;
        let mut set: Vec<i64> = (0..stagger_updates)
            .map(|i| if i == stagger_updates - 1 && stagger_updates > 1 { stop } else { i as f64 * step + start })
            .map(|v| v.round_ties_even() as i64)
            .collect();
        set.dedup();
        set
    }

    /// Starts a step (`curr_ti += 1`): memorised when `mem_every` steps have passed since the
    /// last memory frame or a mask is given, the sensory memory updated on the stagger steps;
    /// neither at the `end`.
    pub(crate) fn next(&mut self, has_mask: bool, end: bool) -> StepPlan {
        self.curr_ti += 1;
        let since = self.curr_ti - self.last_mem_ti;
        StepPlan { memorise: (since >= self.mem_every || has_mask) && !end, update_sensory: self.stagger_ti.contains(&since) && !end }
    }

    /// This step was memorised.
    pub(crate) fn memorised(&mut self) {
        self.last_mem_ti = self.curr_ti;
    }

    /// What every clear does to the counters.
    pub(crate) fn reset(&mut self) {
        self.curr_ti = -1;
        self.last_mem_ti = 0;
    }

    /// Whether the worker clears the sensory memory before a row of the oracle's run
    /// (`raw/schedule.tsv`): a mask on a row that is not the commit starts the propagation.
    pub(crate) fn restarts(row: &ScheduleRow) -> bool {
        row.with_mask && !row.force_permanent
    }

    /// A row of the oracle's run as the processor schedules it: the counters reset when the
    /// propagation starts, then the step's plan, memorised when it says so or forced.
    pub(crate) fn replay(&mut self, row: &ScheduleRow) -> StepPlan {
        if Self::restarts(row) {
            self.reset();
        }
        let plan = self.next(row.with_mask, false);
        if plan.memorise || row.force_permanent {
            self.memorised();
        }
        plan
    }
}

pub struct CutieProcessor<'a> {
    pub(crate) net: &'a CutieNet,
    pub(crate) config: CutieConfig,
    pub(crate) objects: usize,
    pub(crate) memory: MemoryManager,
    pub(crate) schedule: StepSchedule,
    /// Each object's probability on the last step, at its padded size (n, 1): `last_mask`. Clears
    /// leave it.
    pub(crate) last_mask: Option<Feat>,
}

impl<'a> CutieProcessor<'a> {
    pub(crate) fn new(net: &'a CutieNet, config: CutieConfig, objects: usize) -> Self {
        let memory = MemoryManager::new(&config);
        let schedule = StepSchedule::new(&config);
        Self { net, config, objects, memory, schedule, last_mask: None }
    }

    /// One frame: `image` RGB bytes, `height` x `width` x 3; `mask` each object's probability
    /// (objects x height x width) when the frame is given one; `force_permanent` memorises it as
    /// a memory frame (Cutie's permanent memory); `end` skips memorising. Out: the probability of
    /// background then each object, (objects + 1) x height x width.
    pub fn step(&mut self, image: &[u8], width: usize, height: usize, mask: Option<&[f32]>, force_permanent: bool, end: bool) -> Result<Vec<f32>> {
        self.step_tapped(image, width, height, mask, force_permanent, end, &mut Tap::none())
    }

    /// `step`, with taps under the oracle's names (`step.in.image` [3, h, w], `step.in.mask`,
    /// `state.before.*`, every part's own, `step.out.prob`, `step.out.mask` (the argmax),
    /// `step.last_mask`, `state.after.*`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_tapped(
        &mut self, image: &[u8], width: usize, height: usize, mask: Option<&[f32]>, force_permanent: bool, end: bool, tap: &mut Tap,
    ) -> Result<Vec<f32>> {
        let n = self.objects;
        let frame = Frame::new(width, height, self.config.max_internal_size);
        if mask.is_some_and(|m| m.len() != n * width * height) {
            return Err(DiffusionError::model(format!("cutie step: the mask is not {n} objects at {width}x{height}")));
        }
        if tap.active() {
            self.dump("state.before", tap)?;
        }
        let image = frame.image(image)?;
        tap.put("step.in.image", &image)?;
        let mask = mask.map(|m| gpu_upload(m, n, width * height)).transpose().map_err(|e| dev("step mask", e))?;
        if let Some(mask) = &mask {
            tap.put("step.in.mask", mask)?;
        }

        let plan = self.schedule.next(mask.is_some(), end);
        let (pw, ph) = frame.padded();
        let image = Feat::new(frame.to_internal(&image)?, 1, 3, pw, ph)?;
        tap.put("encode_image.in.image", &image.t)?;
        let feats = self.net.encode_image(&image, tap)?;
        let keys = self.net.transform_key(&feats.f16, tap)?;

        // Segmented from memory without a mask; a mask (which has every object, so nothing is
        // left to segment) is taken as it is.
        let prob = match &mask {
            None => self.segment(&feats, &keys, plan.update_sensory, tap)?,
            Some(mask) => Self::mask_prob(&frame.to_internal(mask)?)?,
        };
        if prob.rows() != n + 1 {
            return Err(DiffusionError::model(format!("cutie step: {} probabilities for {n} objects", prob.rows())));
        }
        self.last_mask = Some(Self::last_mask(&prob, pw, ph)?);

        if plan.memorise || force_permanent {
            self.add_memory(&feats, &keys, force_permanent, tap)?;
        }

        let out = frame.to_given(&prob)?;
        let values = gpu_download(&out).map_err(|e| dev("step out", e))?;
        if tap.active() {
            tap.put("step.out.prob", &out)?;
            tap.put("step.out.mask", &Self::labels_tensor(&values, n + 1, width * height)?)?;
            if let Some(last_mask) = &self.last_mask {
                tap.put("step.last_mask", &last_mask.t)?;
            }
            self.dump("state.after", tap)?;
        }
        Ok(values)
    }

    /// `_segment`: the memory read, then the decoder; the sensory memory kept when
    /// `update_sensory`. Out: the probabilities [n+1, H*W] at the padded size.
    fn segment(&mut self, feats: &ImageFeatures, keys: &Keys, update_sensory: bool, tap: &mut Tap) -> Result<GpuTensor> {
        if !self.memory.engaged() {
            // PyTorch warns and returns one plane of zeros, which nothing after it can use.
            return Err(DiffusionError::model("cutie step: segmenting without any memory"));
        }
        let net = self.net;
        let last_mask = self.last_mask.as_ref().ok_or_else(|| DiffusionError::model("cutie step: no last mask"))?;
        let readout = self.memory.read(net, &feats.pix_feat, keys, last_mask, tap)?;
        let (sensory, _logits, prob) = net.segment(feats, &readout, &self.memory.sensory()?, update_sensory, tap)?;
        if update_sensory {
            self.memory.update_sensory(sensory)?;
        }
        Ok(prob)
    }

    /// `_add_memory`: the last mask encoded with the frame (deep update) and memorised.
    fn add_memory(&mut self, feats: &ImageFeatures, keys: &Keys, force_permanent: bool, tap: &mut Tap) -> Result<()> {
        let as_permanent = if force_permanent { Permanence::All } else { Permanence::First };
        self.memory.initialize_sensory_if_needed(keys, self.objects)?;
        let last_mask = self.last_mask.as_ref().ok_or_else(|| DiffusionError::model("cutie step: no mask to memorise"))?;
        let (value, sensory, obj_value) = self.net.encode_mask(&feats.image, &feats.pix_feat, &self.memory.sensory()?, last_mask, true, tap)?;
        self.memory.add_memory(keys, &value, &obj_value, as_permanent, tap)?;
        self.schedule.memorised();
        self.memory.update_sensory(sensory)
    }

    /// A given mask's probabilities (objects at the padded size): softmax(aggregate(mask)) over
    /// the background and the objects, [n+1, H*W].
    pub(crate) fn mask_prob(mask: &GpuTensor) -> Result<GpuTensor> {
        let logits = gpu_cutie_aggregate(mask).map_err(|e| dev("step aggregate", e))?;
        gpu_cutie_softmax_axis0(&logits).map_err(|e| dev("step softmax", e))
    }

    /// The probabilities without the background (a view), as the next step's `last_mask`.
    pub(crate) fn last_mask(prob: &GpuTensor, pw: usize, ph: usize) -> Result<Feat> {
        let n = prob.rows() - 1;
        Feat::new(gpu_slice_rows(prob, 1, n).map_err(|e| dev("last mask", e))?, n, 1, pw, ph)
    }

    /// `labels` as the oracle's `step.out.mask` tap [1, plane].
    pub(crate) fn labels_tensor(prob: &[f32], classes: usize, plane: usize) -> Result<GpuTensor> {
        let labels: Vec<f32> = Self::labels(prob, classes, plane).into_iter().map(f32::from).collect();
        gpu_upload(&labels, 1, plane).map_err(|e| dev("step labels", e))
    }

    /// Each pixel's class (0 the background), the first of equal probabilities
    /// (`torch.argmax`): `prob` is classes x plane.
    pub(crate) fn labels(prob: &[f32], classes: usize, plane: usize) -> Vec<u8> {
        (0..plane)
            .map(|i| {
                let mut best = 0;
                for c in 1..classes {
                    if prob[c * plane + i] > prob[best * plane + i] {
                        best = c;
                    }
                }
                best as u8
            })
            .collect()
    }

    /// The memory and the last mask as the oracle's `<prefix>.*` taps.
    fn dump(&self, prefix: &str, tap: &mut Tap) -> Result<()> {
        self.memory.dump(prefix, tap)?;
        if let Some(last_mask) = &self.last_mask {
            tap.put(&format!("{prefix}.last_mask"), &last_mask.t)?;
        }
        Ok(())
    }

    pub fn clear_memory(&mut self) {
        self.schedule.reset();
        self.memory = MemoryManager::new(&self.config);
    }

    pub fn clear_non_permanent_memory(&mut self) -> Result<()> {
        self.schedule.reset();
        self.memory.clear_non_permanent_memory()
    }

    pub fn clear_sensory_memory(&mut self) {
        self.schedule.reset();
        self.memory.clear_sensory_memory();
    }
}

// --- The check ---

/// What a step's add did to memory: working memory consolidated, and long-term memory dropped its
/// least used tokens first.
#[derive(Clone, Copy, Debug)]
struct AddOutcome {
    consolidated: bool,
    obsolete_removed: bool,
}

impl AddOutcome {
    /// What happened between `before` (the memory's counts then) and now.
    fn since(before: (usize, usize), memory: &MemoryManager) -> Self {
        Self { consolidated: memory.consolidations != before.0, obsolete_removed: memory.obsolete_removals != before.1 }
    }

    /// The memory's counts, for `since`.
    fn counts(memory: &MemoryManager) -> (usize, usize) {
        (memory.consolidations, memory.obsolete_removals)
    }
}

/// A step's counters and memory sizes against its row of `raw/schedule.tsv`, and whether it
/// consolidated and dropped long-term memory's least used tokens (the latter only where the oracle
/// recorded it). The mask areas, when given, are noted: the label maps are judged by IoU.
#[allow(clippy::too_many_arguments)]
fn record_schedule(
    report: &mut Report, stage: &str, step: &str, name: &str, row: &ScheduleRow, schedule: &StepSchedule, memory: &MemoryManager, outcome: AddOutcome,
    area: Option<&[usize]>,
) {
    let AddOutcome { consolidated, obsolete_removed } = outcome;
    let (work, permanent, long) = memory.sizes();
    let memorised = schedule.last_mem_ti == schedule.curr_ti;
    let mut differ = Vec::new();
    if schedule.curr_ti != row.ti {
        differ.push(format!("ti {} (oracle {})", schedule.curr_ti, row.ti));
    }
    if memorised != row.memorised {
        differ.push(format!("memorised {memorised} (oracle {})", row.memorised));
    }
    if consolidated != row.consolidated {
        differ.push(format!("consolidated {consolidated} (oracle {})", row.consolidated));
    }
    if row.obsolete_removed.is_some_and(|e| e != obsolete_removed) {
        differ.push(format!("obsolete removed {obsolete_removed} (oracle {})", !obsolete_removed));
    }
    if (work, permanent, long) != (row.work_tokens, row.work_permanent, row.long_tokens) {
        differ.push(format!("tokens {work}/{permanent}/{long} (oracle {}/{}/{})", row.work_tokens, row.work_permanent, row.long_tokens));
    }
    let mut note = if differ.is_empty() {
        let removed = if row.obsolete_removed.is_some() { format!(" obsolete removed {obsolete_removed}") } else { String::new() };
        format!("ti {} memorised {memorised} consolidated {consolidated}{removed} tokens {work}/{permanent}/{long}", row.ti)
    } else {
        differ.join("; ")
    };
    if let Some(area) = area {
        note += &format!("; mask area {area:?} (oracle {:?})", row.mask_area);
    }
    report.record(stage, step, name, differ.is_empty(), note);
}

/// What the worker gives Cutie in the oracle's run besides the frames: masks as label maps
/// (`frame_index_mask`), one-hot without the background (`index_numpy_to_one_hot_torch`, `[1:]`).
struct WorkerMasks {
    objects: usize,
    plane: usize,
    /// The seed masks as a label map (a later object over an earlier one).
    seed: Vec<u8>,
}

impl WorkerMasks {
    fn new(oracle: &Oracle) -> Result<Self> {
        let (objects, plane) = (oracle.objects, oracle.width * oracle.height);
        let masks = oracle.seed()?;
        if masks.len() != objects * plane {
            return Err(DiffusionError::model("cutie oracle: raw/seed.u8 has the wrong size"));
        }
        let mut seed = vec![0u8; plane];
        for o in 0..objects {
            for (i, label) in seed.iter_mut().enumerate() {
                if masks[o * plane + i] != 0 {
                    *label = (o + 1) as u8;
                }
            }
        }
        Ok(Self { objects, plane, seed })
    }

    /// A label map one-hot: objects x plane.
    fn one_hot(&self, labels: &[u8]) -> Vec<f32> {
        let mut one_hot = vec![0.0f32; self.objects * self.plane];
        for (i, &l) in labels.iter().enumerate() {
            if l > 0 {
                one_hot[(l as usize - 1) * self.plane + i] = 1.0;
            }
        }
        one_hot
    }
}

/// The processor's own work checked without the network (part of stage `memory`):
///
/// - The whole run's schedule: the worker's steps (the commit, the sensory memory cleared, the
///   propagation) through `StepSchedule` and a `MemoryManager` fed zeros at the oracle's sizes,
///   every step's counters, memorising, consolidation and token counts against
///   `raw/schedule.tsv`. Nothing is read: usage only chooses which tokens become prototypes,
///   never when or how many.
/// - Each tapped step's frame and mask handling, given the network's output: the frame scaled and
///   padded (`step.in.image`, `encode_image.in.image`), the worker's mask (`step.in.mask`) through
///   softmax(aggregate) or the oracle's `segment.out.prob`, then `step.last_mask`,
///   `step.out.prob` (unpadded) and `step.out.mask` (the argmax), at `Tol::STAGE`.
pub(crate) fn check_without_network(oracle: &Oracle, report: &mut Report) -> Result<()> {
    /// Object summaries a frame (`num_summaries`) and each one's width (the embedding, then the area).
    const SUMMARIES: usize = 16;
    const SUMMARY_WIDTH: usize = 257;
    let config = &oracle.config;
    let n = oracle.objects;
    let frame = Frame::new(oracle.width, oracle.height, config.max_internal_size);
    let (pw, ph) = frame.padded();
    let (w, h) = (pw / PAD_TO, ph / PAD_TO);
    let zeros = |n: usize, c: usize| Feat::upload(&vec![0.0; n * c * w * h], n, c, w, h);
    let keys = Keys { key: zeros(1, 64)?, shrinkage: zeros(1, 1)?, selection: zeros(1, 64)? };
    let value = zeros(n, VALUE_DIM)?;
    let obj_value = gpu_upload(&vec![0.0; n * SUMMARIES * SUMMARY_WIDTH], n * SUMMARIES, SUMMARY_WIDTH).map_err(|e| dev("schedule", e))?;
    let mut schedule = StepSchedule::new(config);
    let mut memory = MemoryManager::new(config);
    for row in oracle.schedule() {
        let step = row.tap.clone().unwrap_or_else(|| format!("s{}", row.step));
        if StepSchedule::restarts(row) {
            memory.clear_sensory_memory();
        }
        let counts = AddOutcome::counts(&memory);
        let plan = schedule.replay(row);
        if plan.memorise || row.force_permanent {
            let as_permanent = if row.force_permanent { Permanence::All } else { Permanence::First };
            memory.initialize_sensory_if_needed(&keys, n)?;
            memory.add_memory(&keys, &value, &obj_value, as_permanent, &mut Tap::none())?;
        }
        let outcome = AddOutcome::since(counts, &memory);
        record_schedule(report, "memory", &step, "schedule.without_network", row, &schedule, &memory, outcome, None);
    }

    let masks = WorkerMasks::new(oracle)?;
    let labels = oracle.labels()?;
    let plane = oracle.width * oracle.height;
    let tol = |_: &str| Tol::STAGE;
    for row in oracle.schedule().iter().filter(|r| r.tap.is_some()) {
        let step = row.tap.as_deref().unwrap_or_default();
        let mut cmp = Comparer::new(oracle, report, "processor", step, &tol);
        cmp.with(|tap| {
            let image = frame.image(&oracle.frame_rgb(row.frame)?)?;
            tap.put("step.in.image", &image)?;
            tap.put("encode_image.in.image", &frame.to_internal(&image)?)?;
            let prob = if row.with_mask {
                // The seed on the commit; after it, the commit's output read back (the oracle's
                // label map of the frame).
                let label_map = if row.force_permanent { &masks.seed[..] } else { &labels[row.frame * plane..(row.frame + 1) * plane] };
                let mask = gpu_upload(&masks.one_hot(label_map), n, plane).map_err(|e| dev("step mask", e))?;
                tap.put("step.in.mask", &mask)?;
                CutieProcessor::mask_prob(&frame.to_internal(&mask)?)?
            } else {
                let prob = oracle.tap(step, "segment.out.prob")?;
                gpu_upload(&prob.values, n + 1, pw * ph).map_err(|e| dev("oracle prob", e))?
            };
            tap.put("step.last_mask", &CutieProcessor::last_mask(&prob, pw, ph)?.t)?;
            let out = frame.to_given(&prob)?;
            tap.put("step.out.prob", &out)?;
            let values = gpu_download(&out).map_err(|e| dev("step out", e))?;
            tap.put("step.out.mask", &CutieProcessor::labels_tensor(&values, n + 1, plane)?)
        })?;
    }
    Ok(())
}

/// Stage `run`: the oracle's whole run natively, as the worker drives Cutie (commit frame 0 with
/// the seed masks as a label map one-hot, `force_permanent`; clear the sensory memory; frame 0
/// again with the same mask; frames 1.. with none): every frame's label map against the oracle's
/// (IoU per object, 0.998 or better), the schedule (memorised, consolidated, token counts)
/// against `raw/schedule.tsv`, the tapped steps' taps (`Tol::RUN`).
///
/// The propagation's start is given the commit's output read back as a label map, as the worker
/// does (it saves each step's masks and loads them again). `step.out.mask` (the argmax labels) is
/// reported, not failed: one pixel's label flipping fails any tolerance on label numbers, so the
/// label maps are judged by IoU instead. Taps whose columns or rows follow long-term memory's
/// order are reported, not failed, too (`run_tol`); long-term memory is compared without its
/// order instead (`compare_long_term`). The steps without taps are timed (a note): what a step
/// costs as the worker runs it, the tapped steps' downloads and comparisons left out.
pub(crate) fn check(oracle: &Oracle, report: &mut Report, weights: &mut Weights) -> Result<()> {
    const MIN_IOU: f64 = 0.998;
    let net = CutieNet::load(weights)?;
    let (w, h, n) = (oracle.width, oracle.height, oracle.objects);
    let plane = w * h;
    let mut processor = CutieProcessor::new(&net, oracle.config.clone(), n);
    let schedule = oracle.schedule();
    let expected_labels = oracle.labels()?;
    if expected_labels.len() != oracle.frames * plane {
        return Err(DiffusionError::model("cutie oracle: raw/labels.u8 has the wrong size"));
    }
    // Each frame's latest label map, frame 0's the seed's to begin with.
    let masks = WorkerMasks::new(oracle)?;
    let mut frame_labels: Vec<Option<Vec<u8>>> = vec![None; oracle.frames];
    frame_labels[0] = Some(masks.seed.clone());
    // The steps without taps, timed: what a step costs as the worker runs it.
    let (mut untapped, mut untapped_time) = (0usize, std::time::Duration::ZERO);

    for row in schedule {
        let step = row.tap.clone().unwrap_or_else(|| format!("s{}", row.step));
        // The propagation starts from its memory frame again, with the sensory memory cleared.
        if StepSchedule::restarts(row) {
            processor.clear_sensory_memory();
        }
        let frame = oracle.frame_rgb(row.frame)?;
        let mask = if row.with_mask {
            let labels = frame_labels[row.frame].as_ref().ok_or_else(|| DiffusionError::model(format!("cutie run: no mask for frame {}", row.frame)))?;
            Some(masks.one_hot(labels))
        } else {
            None
        };
        let counts = AddOutcome::counts(&processor.memory);
        let prob = match &row.tap {
            Some(tapped) => {
                compare_long_term(oracle, report, tapped, "state.before", &processor.memory)?;
                let long_term = oracle.has(tapped, "state.before.long_mem.b0.key");
                let tol = |name: &str| run_tol(name, long_term);
                let mut cmp = Comparer::new(oracle, report, "run", tapped, &tol);
                let prob = cmp.with(|tap| processor.step_tapped(&frame, w, h, mask.as_deref(), row.force_permanent, false, tap))?;
                // Taps no part puts (e.g. a convolution whose BatchNorm is folded) are noted.
                let missing = cmp.uncompared(&[""]);
                if !missing.is_empty() {
                    cmp.report.note(format!("run/{tapped}: {} oracle taps never put: {}", missing.len(), missing.join(", ")));
                }
                compare_long_term(oracle, report, tapped, "state.after", &processor.memory)?;
                prob
            }
            None => {
                let started = std::time::Instant::now();
                let prob = processor.step(&frame, w, h, mask.as_deref(), row.force_permanent, false)?;
                untapped += 1;
                untapped_time += started.elapsed();
                prob
            }
        };
        let labels = CutieProcessor::labels(&prob, n + 1, plane);

        // The label map against the oracle's, object by object.
        let expected = &expected_labels[row.frame * plane..(row.frame + 1) * plane];
        for o in 1..=n as u8 {
            let (mut both, mut either) = (0usize, 0usize);
            for (&a, &e) in labels.iter().zip(expected) {
                both += (a == o && e == o) as usize;
                either += (a == o || e == o) as usize;
            }
            let iou = if either == 0 { 1.0 } else { both as f64 / either as f64 };
            report.record("run", &step, &format!("labels.f{}.obj{o}", row.frame), iou >= MIN_IOU, format!("iou={iou:.6}"));
        }

        let outcome = AddOutcome::since(counts, &processor.memory);
        let area: Vec<usize> = (1..=n as u8).map(|o| labels.iter().filter(|&&l| l == o).count()).collect();
        record_schedule(report, "run", &step, "schedule", row, &processor.schedule, &processor.memory, outcome, Some(&area));
        frame_labels[row.frame] = Some(labels);
    }
    let ms = untapped_time.as_secs_f64() * 1e3;
    report.note(format!("run: {untapped} steps without taps in {ms:.1} ms, {:.2} ms a step", ms / untapped.max(1) as f64));
    Ok(())
}

/// Stage `run`'s tolerance for a tap (`long_term`: the step reads through long-term memory):
/// `Tol::RUN`, but reported only (`Tol::INFO`) for
/// - `step.out.mask` (the label maps are judged by IoU);
/// - the consolidation's prototype side (`memory.consolidate.query_key`, `.query_selection`,
///   `.similarity`, `.affinity`, `.readout`), one column a prototype in order of use: the
///   oracle's neighbours in that order are as close as 3.7e-5 apart, so a native run's own usage
///   reorders them;
/// - long-term memory in that order (`*.long_mem.*`), compared by `compare_long_term` instead;
/// - with long-term memory, the read's `memory_key`, `memory_shrinkage`, `similarity`, `affinity`
///   and `usage`, whose first rows are long-term memory's tokens in that order. The read is judged
///   by what it gives, `memory.read.readout`, a sum over the tokens that does not depend on
///   their order;
/// - always, the read's `affinity` and `usage`: each pixel's top-k picks, which a near tie swaps
///   (one swapped pick is an entry's whole weight, YouTubeComp-15 at 960x540 rel 7%, cosine
///   0.99996); the readout judges them too.
fn run_tol(name: &str, long_term: bool) -> Tol {
    const PROTOTYPE_ORDER: [&str; 5] = [
        "memory.consolidate.query_key",
        "memory.consolidate.query_selection",
        "memory.consolidate.similarity",
        "memory.consolidate.affinity",
        "memory.consolidate.readout",
    ];
    const LONG_TERM_ROWS: [&str; 5] =
        ["memory.read.memory_key", "memory.read.memory_shrinkage", "memory.read.similarity", "memory.read.affinity", "memory.read.usage"];
    let base = name.split('#').next().unwrap_or(name);
    let in_long_term_order = PROTOTYPE_ORDER.contains(&base) || base.contains(".long_mem.") || (long_term && LONG_TERM_ROWS.contains(&base));
    let picks = base == "memory.read.affinity" || base == "memory.read.usage";
    if name == "step.out.mask" || in_long_term_order || picks {
        Tol::INFO
    } else {
        Tol::RUN
    }
}

/// Stage `run`: long-term memory (`<prefix>.long_mem.b0.*`) against the oracle's without its
/// order. Its tokens are each consolidation's most used, sorted by use: in the oracle, neighbours
/// in that order are as close as 3.7e-5 apart, and 14 to 16 candidates are within 1% of the cut
/// at `num_prototypes` (prop_f40). A native run's own usage (top-k near ties in its reads)
/// reorders them or swaps a few across the cut: the columns permute, and a swapped token is
/// another one about as used. So each oracle token is paired with the native token of the
/// nearest key when that one's nearest oracle token is it in turn (distinct prototypes' keys
/// differ by 0.7% of the largest key or more); the pairs' keys, shrinkage, values and counts are
/// compared at `Tol::RUN`, and the tokens left without a pair are counted, failing when more
/// than half are (a wrong choice of prototypes, not near ties). Quadratic in the tokens (128 in
/// the oracle).
fn compare_long_term(oracle: &Oracle, report: &mut Report, step: &str, prefix: &str, memory: &MemoryManager) -> Result<()> {
    let name = |part: &str| format!("{prefix}.long_mem.b0.{part}");
    let long = &memory.long;
    let expected_key = if oracle.has(step, &name("key")) { Some(oracle.tap(step, &name("key"))?) } else { None };
    let n = expected_key.as_ref().map_or(0, |t| t.shape.last().copied().unwrap_or(0));
    if n != long.size() {
        report.fail("run", step, &name("key"), format!("{} long-term tokens, the oracle {n}", long.size()));
        return Ok(());
    }
    let (Some(expected_key), Some(key)) = (expected_key, &long.key) else {
        return Ok(());
    };
    let download = |t: &GpuTensor| gpu_download(t).map_err(|e| dev("long-term memory", e));
    let native_key = download(key)?;
    let ck = native_key.len() / n;
    // Squared distance from column i of `a` to column j of `b`, both [ck, n].
    let distance = |a: &[f32], i: usize, b: &[f32], j: usize| -> f64 {
        (0..ck)
            .map(|c| {
                let d = a[c * n + i] as f64 - b[c * n + j] as f64;
                d * d
            })
            .sum()
    };
    let nearest = |a: &[f32], i: usize, b: &[f32]| (0..n).min_by(|&j, &k| distance(a, i, b, j).total_cmp(&distance(a, i, b, k))).unwrap_or(0);
    let pairs: Vec<(usize, usize)> = (0..n)
        .map(|i| (i, nearest(&expected_key.values, i, &native_key)))
        .filter(|&(i, j)| nearest(&native_key, j, &expected_key.values) == i)
        .collect();
    let unpaired = n - pairs.len();
    let note = format!("{} of {n} tokens paired by key, {unpaired} unpaired (swapped at a cut)", pairs.len());
    report.record("run", step, &name("tokens (paired)"), unpaired * 2 <= n, note);

    // The pairs' columns of a tensor [rows, n], the oracle's against their native partners.
    let compare = |report: &mut Report, part: &str, native: &[f32], expected: &OracleTap| {
        let rows = expected.values.len() / n;
        let (mut actual, mut values) = (Vec::with_capacity(rows * pairs.len()), Vec::with_capacity(rows * pairs.len()));
        for r in 0..rows {
            for &(i, j) in &pairs {
                actual.push(native[r * n + j]);
                values.push(expected.values[r * n + i]);
            }
        }
        let expected = OracleTap { shape: vec![rows, pairs.len()], dtype: expected.dtype.clone(), values };
        report.compare("run", step, &format!("{} (paired)", name(part)), &actual, &expected, Tol::RUN);
    };
    compare(report, "key", &native_key, &expected_key);
    let mut native: Vec<(String, Option<Vec<f32>>)> = vec![
        ("shrinkage".into(), long.shrinkage.as_ref().map(download).transpose()?),
        ("use_count".into(), long.use_count.as_ref().map(download).transpose()?),
        ("life_count".into(), long.life_count.as_ref().map(download).transpose()?),
    ];
    let values = long.values.as_ref().map(download).transpose()?;
    for o in 0..oracle.objects {
        let per = VALUE_DIM * n;
        native.push((format!("value.obj{}", o + 1), values.as_ref().and_then(|v| v.get(o * per..(o + 1) * per)).map(<[f32]>::to_vec)));
    }
    for (part, native) in &native {
        let expected = if oracle.has(step, &name(part)) { Some(oracle.tap(step, &name(part))?) } else { None };
        match (native, expected.filter(|t| !t.values.is_empty())) {
            (Some(native), Some(expected)) => compare(report, part, native, &expected),
            (None, None) => {}
            (native, _) => report.fail("run", step, &name(part), if native.is_some() { "not in the oracle" } else { "missing" }),
        }
    }
    Ok(())
}
