//! Cutie's per-frame step (`inference/inference_core.InferenceCore`), driving the network and
//! the memory.
//!
//! A step (`step(image, mask, force_permanent, end)`), with `curr_ti` counting steps since the
//! last clear and `last_mem_ti` the last memorised one:
//! - resize the image (and mask) when its shorter side exceeds `max_internal_size` (bilinear,
//!   align_corners false), then pad to a multiple of 16, centred (left = (W'-W)/2 rounded down,
//!   the rest right; likewise top and bottom), the mask too;
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
use crate::oracle::{Comparer, Oracle, Report, Row, ScheduleRow, Tol};
use crate::{CutieConfig, DiffusionError, Result};
use makepad_ai_common::gpu::{
    gpu_birefnet_resize_bilinear, gpu_cutie_aggregate, gpu_cutie_crop, gpu_cutie_pad, gpu_cutie_softmax_axis0, gpu_download, gpu_slice_rows,
    gpu_upload, GpuTensor,
};
use std::collections::HashSet;

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
    /// as PyTorch's CUDA division by a scalar does it (times the f32 reciprocal).
    pub(crate) fn image(&self, rgb: &[u8]) -> Result<GpuTensor> {
        let plane = self.width * self.height;
        if rgb.len() != plane * 3 {
            return Err(DiffusionError::model(format!("cutie step: {} bytes for a {}x{} RGB frame", rgb.len(), self.width, self.height)));
        }
        let scale = 1.0f32 / 255.0f32;
        let mut planes = vec![0.0f32; 3 * plane];
        for (i, px) in rgb.chunks_exact(3).enumerate() {
            for c in 0..3 {
                planes[c * plane + i] = px[c] as f32 * scale;
            }
        }
        gpu_upload(&planes, 3, plane).map_err(|e| dev("step image", e))
    }

    /// Planar [planes, width*height] at the size given, to the padded size inside Cutie: resized
    /// (bilinear, align_corners false) if need be, then padded with zeros.
    pub(crate) fn to_internal(&self, x: &GpuTensor) -> Result<GpuTensor> {
        if self.resized() {
            let x = gpu_birefnet_resize_bilinear(x, self.width, self.height, self.w, self.h, false).map_err(|e| dev("step resize", e))?;
            self.pad.pad(&x, self.w, self.h)
        } else {
            self.pad.pad(x, self.w, self.h)
        }
    }

    /// Planar at the padded size back to the size given: unpadded, then resized back.
    pub(crate) fn to_given(&self, x: &GpuTensor) -> Result<GpuTensor> {
        let x = self.pad.unpad(x, self.w, self.h)?;
        if self.resized() {
            gpu_birefnet_resize_bilinear(&x, self.w, self.h, self.width, self.height, false).map_err(|e| dev("step resize back", e))
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

/// A check's row that is not a tensor comparison (a label map's IoU, the schedule): `score` 1 is
/// a match, shown as the cosine.
pub(crate) fn record(report: &mut Report, stage: &str, step: &str, name: &str, pass: bool, score: f64, note: String) {
    if !pass {
        report.failures += 1;
    }
    report.rows.push(Row {
        stage: stage.into(),
        step: step.into(),
        name: name.into(),
        n: 1,
        max_abs: (1.0 - score) as f32,
        max_ref: 1.0,
        rel: (1.0 - score) as f32,
        cos: score,
        pass,
        note,
    });
}

/// The names of the step's oracle taps under any of `prefixes` that nothing put (the oracle's
/// `#n` names counted from the comparer's repeats).
pub(crate) fn uncompared(cmp: &Comparer, prefixes: &[&str]) -> Vec<String> {
    let mut put = HashSet::new();
    for (name, &count) in &cmp.seen {
        put.insert(name.clone());
        put.extend((1..count).map(|i| format!("{name}#{i}")));
    }
    cmp.oracle
        .names(&cmp.step)
        .into_iter()
        .filter(|name| prefixes.iter().any(|p| name.starts_with(p)) && !put.contains(*name))
        .map(String::from)
        .collect()
}

/// A step's counters and memory sizes against its row of `raw/schedule.tsv` (the mask areas, when
/// given, are noted: the label maps are judged by IoU).
#[allow(clippy::too_many_arguments)]
fn record_schedule(
    report: &mut Report, stage: &str, step: &str, name: &str, row: &ScheduleRow, schedule: &StepSchedule, memory: &MemoryManager, consolidated: bool,
    area: Option<&[usize]>,
) {
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
    if (work, permanent, long) != (row.work_tokens, row.work_permanent, row.long_tokens) {
        differ.push(format!("tokens {work}/{permanent}/{long} (oracle {}/{}/{})", row.work_tokens, row.work_permanent, row.long_tokens));
    }
    let mut note = if differ.is_empty() {
        format!("ti {} memorised {memorised} consolidated {consolidated} tokens {work}/{permanent}/{long}", row.ti)
    } else {
        differ.join("; ")
    };
    if let Some(area) = area {
        note += &format!("; mask area {area:?} (oracle {:?})", row.mask_area);
    }
    let pass = differ.is_empty();
    record(report, stage, step, name, pass, if pass { 1.0 } else { 0.0 }, note);
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
    let config = CutieConfig::default();
    let n = oracle.objects;
    let frame = Frame::new(oracle.width, oracle.height, config.max_internal_size);
    let (pw, ph) = frame.padded();
    let (w, h) = (pw / PAD_TO, ph / PAD_TO);
    let zeros = |n: usize, c: usize| Feat::upload(&vec![0.0; n * c * w * h], n, c, w, h);
    let keys = Keys { key: zeros(1, 64)?, shrinkage: zeros(1, 1)?, selection: zeros(1, 64)? };
    let value = zeros(n, VALUE_DIM)?;
    let obj_value = gpu_upload(&vec![0.0; n * SUMMARIES * SUMMARY_WIDTH], n * SUMMARIES, SUMMARY_WIDTH).map_err(|e| dev("schedule", e))?;
    let mut schedule = StepSchedule::new(&config);
    let mut memory = MemoryManager::new(&config);
    for row in &oracle.schedule()? {
        let step = row.tap.clone().unwrap_or_else(|| format!("s{}", row.step));
        if row.with_mask && !row.force_permanent {
            schedule.reset();
            memory.clear_sensory_memory();
        }
        let consolidations = memory.consolidations;
        let plan = schedule.next(row.with_mask, false);
        if plan.memorise || row.force_permanent {
            let as_permanent = if row.force_permanent { Permanence::All } else { Permanence::First };
            memory.initialize_sensory_if_needed(&keys, n)?;
            memory.add_memory(&keys, &value, &obj_value, as_permanent, &mut Tap::none())?;
            schedule.memorised();
        }
        record_schedule(report, "memory", &step, "schedule.without_network", row, &schedule, &memory, memory.consolidations != consolidations, None);
    }

    let masks = WorkerMasks::new(oracle)?;
    let labels = oracle.labels()?;
    let plane = oracle.width * oracle.height;
    let tol = |_: &str| Tol::STAGE;
    for row in oracle.schedule()?.iter().filter(|r| r.tap.is_some()) {
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
/// label maps are judged by IoU instead.
pub(crate) fn check(oracle: &Oracle, report: &mut Report, weights: &mut Weights) -> Result<()> {
    const MIN_IOU: f64 = 0.998;
    let net = CutieNet::load(weights)?;
    let (w, h, n) = (oracle.width, oracle.height, oracle.objects);
    let plane = w * h;
    let mut processor = CutieProcessor::new(&net, CutieConfig::default(), n);
    let schedule = oracle.schedule()?;
    let expected_labels = oracle.labels()?;
    if expected_labels.len() != oracle.frames * plane {
        return Err(DiffusionError::model("cutie oracle: raw/labels.u8 has the wrong size"));
    }
    // Each frame's latest label map, frame 0's the seed's to begin with.
    let masks = WorkerMasks::new(oracle)?;
    let mut frame_labels: Vec<Option<Vec<u8>>> = vec![None; oracle.frames];
    frame_labels[0] = Some(masks.seed.clone());
    let tol = |name: &str| if name == "step.out.mask" { Tol::INFO } else { Tol::RUN };

    for row in &schedule {
        let step = row.tap.clone().unwrap_or_else(|| format!("s{}", row.step));
        // The propagation starts from its memory frame again, with the sensory memory cleared.
        if row.with_mask && !row.force_permanent {
            processor.clear_sensory_memory();
        }
        let frame = oracle.frame_rgb(row.frame)?;
        let mask = if row.with_mask {
            let labels = frame_labels[row.frame].as_ref().ok_or_else(|| DiffusionError::model(format!("cutie run: no mask for frame {}", row.frame)))?;
            Some(masks.one_hot(labels))
        } else {
            None
        };
        let consolidations = processor.memory.consolidations;
        let prob = match &row.tap {
            Some(tapped) => {
                let mut cmp = Comparer::new(oracle, report, "run", tapped, &tol);
                let prob = cmp.with(|tap| processor.step_tapped(&frame, w, h, mask.as_deref(), row.force_permanent, false, tap))?;
                // Taps no part puts (e.g. a convolution whose BatchNorm is folded) are noted.
                let missing = uncompared(&cmp, &[""]);
                if !missing.is_empty() {
                    cmp.report.note(format!("run/{tapped}: {} oracle taps never put: {}", missing.len(), missing.join(", ")));
                }
                prob
            }
            None => processor.step(&frame, w, h, mask.as_deref(), row.force_permanent, false)?,
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
            record(report, "run", &step, &format!("labels.f{}.obj{o}", row.frame), iou >= MIN_IOU, iou, format!("iou={iou:.6}"));
        }

        let consolidated = processor.memory.consolidations != consolidations;
        let area: Vec<usize> = (1..=n as u8).map(|o| labels.iter().filter(|&&l| l == o).count()).collect();
        record_schedule(report, "run", &step, "schedule", row, &processor.schedule, &processor.memory, consolidated, Some(&area));
        frame_labels[row.frame] = Some(labels);
    }
    Ok(())
}
