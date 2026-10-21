//! Cutie's mask decoder and sensory update (`big_modules.MaskDecoder`, `modules.*`,
//! `CUTIE.segment`).
//!
//! `decoder_feat_proc`: 1x1 convs on f8 (512 -> 256) and f4 (256 -> 128). `up_16_8` and `up_8_4`
//! (MaskUpsampleBlock): the objects' features bilinear x2 (align_corners false,
//! `gpu_birefnet_resize_bilinear`), the image's skip added to each object, then a GroupResBlock
//! (256 -> 128, then 128 -> 128). `pred`: 3x3 p1 on ReLU(p4) to one logit per object at stride 4.
//! The sensory update (SensoryUpdater): `g16_conv` 1x1 on p16, `g8_conv` 1x1 on p8
//! area-downsampled by 2, `g4_conv` 1x1 on [p4, logit] (129) area-downsampled by 4, summed; then
//! `transform` 3x3 p1 on [that, sensory] (512 -> 768) and `gpu_cutie_gru`.
//!
//! `segment`: sigmoid of the logits, `gpu_cutie_aggregate` (background first), bilinear x4 to the
//! padded frame size, softmax over the n+1 (`gpu_cutie_softmax_axis0`).

use crate::model::{not_ported, CutieNet, Feat, ImageFeatures, Tap, Weights};
use crate::oracle::{Oracle, Report};
use crate::Result;
use makepad_ai_common::gpu::GpuTensor;

pub(crate) struct MaskDecoder {}

impl MaskDecoder {
    pub(crate) fn load(_w: &mut Weights) -> Result<Self> {
        Err(not_ported("the mask decoder"))
    }
}

impl CutieNet {
    /// `CUTIE.segment`: the decoder on `memory_readout` (n, 256 at stride 16) with the image's
    /// f8 and f4 and the `sensory` memory (n, 256), then the probabilities. Returns the new
    /// sensory memory (`sensory` itself, as a view, when not `update_sensory`), the logits
    /// [n+1, H*W] and the probabilities [n+1, H*W] (background first, at the padded size).
    ///
    /// Taps: `segment.in.memory_readout`, `.in.sensory`, `net.mask_decoder.decoder_feat_proc.0`,
    /// `.1`, `net.mask_decoder.up_16_8`, `.up_8_4`, `.pred`, `.sensory_update`,
    /// `segment.out.sensory`, `.out.logits`, `.out.prob`.
    pub(crate) fn segment(&self, _feats: &ImageFeatures, _memory_readout: &Feat, _sensory: &Feat, _update_sensory: bool, _tap: &mut Tap) -> Result<(Feat, GpuTensor, GpuTensor)> {
        Err(not_ported("segment"))
    }
}

/// Stage `decoder`: for every oracle step that segments, the oracle's `encode_image.out.f8`,
/// `.f4` and `segment.in.*` through `segment`, every tap compared.
pub(crate) fn check(_oracle: &Oracle, _report: &mut Report, _weights: &mut Weights) -> Result<()> {
    Err(not_ported("the decoder check"))
}
