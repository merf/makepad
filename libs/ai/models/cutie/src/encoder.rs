//! Cutie's pixel encoder and key projection (`big_modules.PixelEncoder`, `KeyProjection`,
//! `CUTIE.encode_image`, `CUTIE.transform_key`).
//!
//! The pixel encoder is torchvision's ResNet-50 (`model/utils/resnet.py`) to its third stage:
//! conv1 7x7 s2 p3 (3 -> 64, no bias) + bn1 + ReLU + max-pool 3x3 s2 p1, then `res2` (layer1: 3
//! bottlenecks, 256 out, stride 1), `layer2` (4, 512, stride 2), `layer3` (6, 1024, stride 2).
//! A bottleneck: conv1 1x1 + bn1 + ReLU, conv2 3x3 (stride on this one) + bn2 + ReLU, conv3 1x1
//! (x4 width) + bn3, plus the residual (block 0 of each stage: `downsample.0` 1x1 conv with the
//! stride + `downsample.1` BatchNorm), then ReLU. Every BatchNorm folds into its conv
//! (`Weights::conv_bn`). `pix_feat_proj` is a 1x1 conv 1024 -> 256 with bias on f16.
//!
//! The key projection from f16: `key_proj.pix_feat_proj` 1x1 1024 -> 256, then three 3x3 p1
//! convs on that: `key_proj` (64, the key), `d_proj` (1; shrinkage = d^2 + 1), `e_proj` (64;
//! selection = sigmoid(e)).

use crate::model::{not_ported, CutieNet, Feat, ImageFeatures, Keys, Tap, Weights};
use crate::oracle::{Oracle, Report};
use crate::Result;

pub(crate) struct PixelEncoder {}

impl PixelEncoder {
    pub(crate) fn load(_w: &mut Weights) -> Result<Self> {
        Err(not_ported("the pixel encoder"))
    }
}

pub(crate) struct KeyProjection {}

impl KeyProjection {
    pub(crate) fn load(_w: &mut Weights) -> Result<Self> {
        Err(not_ported("the key projection"))
    }
}

impl CutieNet {
    /// `CUTIE.encode_image`: `image` is one image [3, H*W] in 0..1, already padded to a multiple
    /// of 16 (the oracle's `encode_image.in.image`). Normalises it (ImageNet mean 0.485, 0.456,
    /// 0.406, std 0.229, 0.224, 0.225: (x - mean) / std), runs the pixel encoder and
    /// `pix_feat_proj`.
    ///
    /// Taps: `encode_image.normalised`, `net.pixel_encoder.bn1` (conv1 + bn1, before the ReLU),
    /// `net.pixel_encoder.maxpool`, `net.pixel_encoder.res2.{i}` and `net.pixel_encoder.res2`,
    /// the same for `layer2` and `layer3`, `net.pix_feat_proj`, `encode_image.out.f16`, `.f8`,
    /// `.f4`, `.pix_feat`. (`net.pixel_encoder.conv1` is the conv before its BatchNorm, which is
    /// folded here: not tapped.)
    pub(crate) fn encode_image(&self, _image: &Feat, _tap: &mut Tap) -> Result<ImageFeatures> {
        Err(not_ported("encode_image"))
    }

    /// `CUTIE.transform_key` on f16 (one image, 1024 at stride 16): key (64), shrinkage (1),
    /// selection (64). Taps: `net.key_proj.pix_feat_proj`, `key.key`, `key.shrinkage`,
    /// `key.selection`.
    pub(crate) fn transform_key(&self, _f16: &Feat, _tap: &mut Tap) -> Result<Keys> {
        Err(not_ported("transform_key"))
    }
}

/// Stage `encoder`: for every oracle step, the oracle's `encode_image.in.image` through
/// `encode_image` and the oracle's `encode_image.out.f16` through `transform_key`, every tap
/// compared (`Tol::STAGE`).
pub(crate) fn check(_oracle: &Oracle, _report: &mut Report, _weights: &mut Weights) -> Result<()> {
    Err(not_ported("the encoder check"))
}
