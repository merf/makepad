//! Cutie's mask encoder, sensory deep update and object summarizer (`big_modules.MaskEncoder`,
//! `modules.SensoryDeepUpdater`, `transformer/object_summarizer.ObjectSummarizer`,
//! `CUTIE.encode_mask`).
//!
//! Mask encoder, per object: the normalised image and [mask, others] stacked (5 channels; others
//! = the other objects' masks summed and clamped to 0..1, `gpu_cutie_others`), through a
//! ResNet-18 (`model/utils/resnet.py` BasicBlocks): conv1 7x7 s2 p3 (5 -> 64, no bias) + bn1,
//! then max-pool 3x3 s2 p1, then ReLU (in that order), layer1 (2 blocks, 64), layer2 (2, 128, s2),
//! layer3 (2, 256, s2). A BasicBlock: conv1 3x3 (stride) + bn1 + ReLU, conv2 3x3 + bn2, plus the
//! residual (`downsample.0` 1x1 stride + `downsample.1` BN where the shape changes), ReLU. Then
//! `fuser` (a `FusionBlock`, pix_feat 256 and the 256 features -> 256: the value), and when
//! deep-updating, `sensory_update` (SensoryDeepUpdater: `transform` 3x3 p1 on [value, sensory]
//! (512 -> 768), then `gpu_cutie_gru` with the old sensory).
//!
//! Object summarizer: each object's mask area-downsampled to the value's size (`gpu_pixal_pool`
//! is PyTorch's area interpolation), masks repeated for the first 8 summaries and 1 - mask for the
//! last 8; the value (tokens [h*w, 256]) through `input_proj` (Linear 256 -> 256) plus the sine
//! positional encoding (`PositionalEncoding(256, scale=32, temperature=128)`, channel-last: x
//! encodings in the first 128 channels, y in the next, sin/cos interleaved, positions normalised
//! by the last index); then `feature_pred` (Linear, ReLU, Linear) and `weights_pred` (Linear,
//! ReLU, Linear to 16); weights = sigmoid(logits) * masks; sums = weights^T @ feature [16, 256],
//! area = each summary's weights summed; summaries = [sums, area] [16, 257] per object.

use crate::model::{not_ported, CutieNet, Feat, Tap, Weights};
use crate::oracle::{Oracle, Report};
use crate::Result;
use makepad_ai_common::gpu::GpuTensor;

pub(crate) struct MaskEncoder {}

impl MaskEncoder {
    pub(crate) fn load(_w: &mut Weights) -> Result<Self> {
        Err(not_ported("the mask encoder"))
    }
}

pub(crate) struct ObjectSummarizer {}

impl ObjectSummarizer {
    pub(crate) fn load(_w: &mut Weights) -> Result<Self> {
        Err(not_ported("the object summarizer"))
    }
}

impl CutieNet {
    /// `CUTIE.encode_mask`: `image` the normalised frame (one image, [3, H*W], as
    /// `ImageFeatures::image`), `pix_feat` (one image, 256 at stride 16), `sensory` (n objects,
    /// 256 at stride 16), `masks` (n images of 1 channel at H, W: each object's probability).
    /// Returns the value (n, 256 at stride 16), the new sensory memory (the old one unchanged
    /// when not `deep_update`) and the object summaries [n*16, 257] (object-major).
    ///
    /// Taps: `encode_mask.in.pix_feat`, `.in.sensory`, `.in.masks`, `net.mask_encoder.distributor`
    /// (the 5-channel input), `net.mask_encoder.bn1`, `.maxpool`, `.relu`, `.layer1.{i}`,
    /// `.layer1`, `.layer2.{i}`, `.layer2`, `.layer3.{i}`, `.layer3`, `.fuser.distributor`,
    /// `.fuser.block1`, `.fuser.block2`, `net.mask_encoder.sensory_update`,
    /// `encode_mask.out.value`, `.out.sensory`, `.out.obj_summaries`.
    pub(crate) fn encode_mask(
        &self, _image: &Feat, _pix_feat: &Feat, _sensory: &Feat, _masks: &Feat, _deep_update: bool, _tap: &mut Tap,
    ) -> Result<(Feat, Feat, GpuTensor)> {
        Err(not_ported("encode_mask"))
    }
}

/// Stage `mask`: for every oracle step with `encode_mask` taps, the oracle's inputs
/// (`encode_image.normalised`, `encode_mask.in.*`) through `encode_mask`, every tap compared.
pub(crate) fn check(_oracle: &Oracle, _report: &mut Report, _weights: &mut Weights) -> Result<()> {
    Err(not_ported("the mask check"))
}
