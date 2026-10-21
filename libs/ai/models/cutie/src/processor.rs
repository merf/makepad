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
//! each, and memory holds them all in one bucket.

use crate::CutieConfig;
use crate::memory::MemoryManager;
use crate::model::{not_ported, CutieNet, Tap, Weights};
use crate::oracle::{Oracle, Report};
use crate::Result;

pub struct CutieProcessor<'a> {
    pub(crate) net: &'a CutieNet,
    pub(crate) config: CutieConfig,
    pub(crate) objects: usize,
    pub(crate) memory: MemoryManager,
}

impl<'a> CutieProcessor<'a> {
    pub(crate) fn new(net: &'a CutieNet, config: CutieConfig, objects: usize) -> Self {
        let memory = MemoryManager::new(&config);
        Self { net, config, objects, memory }
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
        &mut self, _image: &[u8], _width: usize, _height: usize, _mask: Option<&[f32]>, _force_permanent: bool, _end: bool, _tap: &mut Tap,
    ) -> Result<Vec<f32>> {
        Err(not_ported("the processor step"))
    }

    pub fn clear_memory(&mut self) {
        self.memory = MemoryManager::new(&self.config);
    }

    pub fn clear_non_permanent_memory(&mut self) {
        self.memory.clear_non_permanent_memory();
    }

    pub fn clear_sensory_memory(&mut self) {
        self.memory.clear_sensory_memory();
    }
}

/// Stage `run`: the oracle's whole run natively, as the worker drives Cutie (commit frame 0 with
/// the seed masks as a label map one-hot, `force_permanent`; clear the sensory memory; frame 0
/// again with the same mask; frames 1.. with none): every frame's label map against the oracle's
/// (IoU per object, 0.998 or better), the schedule (memorised, consolidated, token counts)
/// against `raw/schedule.tsv`, the tapped steps' taps (`Tol::RUN`).
pub(crate) fn check(_oracle: &Oracle, _report: &mut Report, _weights: &mut Weights) -> Result<()> {
    Err(not_ported("the run check"))
}
