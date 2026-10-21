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

use crate::model::{add, dev, relu, Conv2d, CutieNet, Feat, ImageFeatures, Keys, Tap, Weights};
use crate::oracle::{Comparer, Oracle, OracleTap, Report, Tol, STEPS};
use crate::{DiffusionError, Result};
use makepad_ai_common::gpu::{
    cutie_conv_out, gpu_add, gpu_cutie_maxpool3x3s2, gpu_cutie_sigmoid, gpu_mul, gpu_rife_fill, gpu_upload,
};
use std::collections::{HashMap, HashSet};

/// Cutie's image normalisation (`model_cfg.pixel_mean`, `pixel_std`: ImageNet's), per RGB channel.
pub(crate) const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub(crate) const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A ResNet bottleneck (`resnet.Bottleneck`, expansion 4), its BatchNorms folded into the convs.
pub(crate) struct Bottleneck {
    /// 1x1, in -> planes, + bn1.
    pub(crate) conv1: Conv2d,
    /// 3x3 p1 with the block's stride, planes -> planes, + bn2.
    pub(crate) conv2: Conv2d,
    /// 1x1, planes -> 4 * planes, + bn3.
    pub(crate) conv3: Conv2d,
    /// `downsample.0` (1x1 with the stride) + `downsample.1`, on the stage's first block only.
    pub(crate) downsample: Option<Conv2d>,
}

impl Bottleneck {
    pub(crate) fn load(w: &mut Weights, name: &str, in_c: usize, planes: usize, stride: usize, downsample: bool) -> Result<Self> {
        let out_c = planes * 4;
        Ok(Self {
            conv1: w.conv_bn(&format!("{name}.conv1"), &format!("{name}.bn1"), in_c, planes, 1, 0, 1)?,
            conv2: w.conv_bn(&format!("{name}.conv2"), &format!("{name}.bn2"), planes, planes, 3, 1, stride)?,
            conv3: w.conv_bn(&format!("{name}.conv3"), &format!("{name}.bn3"), planes, out_c, 1, 0, 1)?,
            downsample: if downsample {
                Some(w.conv_bn(&format!("{name}.downsample.0"), &format!("{name}.downsample.1"), in_c, out_c, 1, 0, stride)?)
            } else {
                None
            },
        })
    }

    /// relu(bn3(conv3(relu(bn2(conv2(relu(bn1(conv1(x)))))))) + residual), the residual x or
    /// its downsample.
    pub(crate) fn forward(&self, x: &Feat) -> Result<Feat> {
        let out = relu(&self.conv1.forward(x)?)?;
        let out = relu(&self.conv2.forward(&out)?)?;
        let out = self.conv3.forward(&out)?;
        match &self.downsample {
            Some(d) => relu(&add(&out, &d.forward(x)?)?),
            None => relu(&add(&out, x)?),
        }
    }
}

/// A ResNet stage (`ResNet._make_layer`): bottlenecks one after another, the first with the
/// stride and the downsample.
pub(crate) struct ResLayer {
    /// The oracle's module name, `net.pixel_encoder.<stage>`.
    pub(crate) tap_name: String,
    pub(crate) blocks: Vec<Bottleneck>,
}

impl ResLayer {
    /// `{name}.0` .. `{name}.{blocks-1}`, `in_c` channels in, `planes * 4` out.
    pub(crate) fn load(w: &mut Weights, name: &str, in_c: usize, planes: usize, blocks: usize, stride: usize) -> Result<Self> {
        let mut layer = Self { tap_name: format!("net.{name}"), blocks: Vec::with_capacity(blocks) };
        for i in 0..blocks {
            let block_in = if i == 0 { in_c } else { planes * 4 };
            let block_stride = if i == 0 { stride } else { 1 };
            layer.blocks.push(Bottleneck::load(w, &format!("{name}.{i}"), block_in, planes, block_stride, i == 0)?);
        }
        Ok(layer)
    }

    /// Taps each block's output (`{tap_name}.{i}`) and the stage's (`{tap_name}`).
    pub(crate) fn forward(&self, x: &Feat, tap: &mut Tap) -> Result<Feat> {
        let mut out = self.blocks[0].forward(x)?;
        tap.put(&format!("{}.0", self.tap_name), &out.t)?;
        for (i, block) in self.blocks.iter().enumerate().skip(1) {
            out = block.forward(&out)?;
            tap.put(&format!("{}.{i}", self.tap_name), &out.t)?;
        }
        tap.put(&self.tap_name, &out.t)?;
        Ok(out)
    }
}

/// The pixel encoder (ResNet-50 to stride 16) and `CUTIE.pix_feat_proj` on its f16.
pub(crate) struct PixelEncoder {
    /// conv1 7x7 s2 p3 + bn1.
    pub(crate) conv1: Conv2d,
    pub(crate) res2: ResLayer,
    pub(crate) layer2: ResLayer,
    pub(crate) layer3: ResLayer,
    /// 1x1 1024 -> 256 with bias (`pix_feat_proj`, a module of `CUTIE` itself).
    pub(crate) pix_feat_proj: Conv2d,
}

impl PixelEncoder {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        Ok(Self {
            conv1: w.conv_bn("pixel_encoder.conv1", "pixel_encoder.bn1", 3, 64, 7, 3, 2)?,
            res2: ResLayer::load(w, "pixel_encoder.res2", 64, 64, 3, 1)?,
            layer2: ResLayer::load(w, "pixel_encoder.layer2", 256, 128, 4, 2)?,
            layer3: ResLayer::load(w, "pixel_encoder.layer3", 512, 256, 6, 2)?,
            pix_feat_proj: w.conv("pix_feat_proj", 1024, 256, 1, 0, 1, true)?,
        })
    }

    /// `(image - mean) / std` per channel, in that order and in f32, as `CUTIE.encode_image`.
    /// `image` is one image of 3 channels.
    // HOST FALLBACK: wants a kernel (per-row (x - a[r]) / b[r] on planar [rows, cols], a true
    // division: multiplying by 1 / std is up to 1 ulp off). It costs a download (with a stream
    // sync) and an upload of 3 x H x W floats a frame: 2.8 MB each way at the worker's 640x360
    // working frames (padded 640x368), 21.7 MB at max_internal_size (1008x1792 padded). The
    // subtraction alone is exact on the device today (`gpu_add_rows_broadcast` with -mean).
    pub(crate) fn normalise(image: &Feat) -> Result<Feat> {
        if image.n != 1 || image.c != 3 {
            return Err(DiffusionError::model(format!(
                "cutie encode_image: {} images of {} channels, wants one RGB image",
                image.n, image.c
            )));
        }
        let mut v = image.download()?;
        let plane = image.plane();
        for (c, values) in v.chunks_mut(plane).enumerate() {
            for x in values {
                *x = (*x - PIXEL_MEAN[c]) / PIXEL_STD[c];
            }
        }
        Feat::new(gpu_upload(&v, 3, plane).map_err(|e| dev("normalise", e))?, 1, 3, image.w, image.h)
    }

    /// `CUTIE.encode_image` (see `CutieNet::encode_image`).
    pub(crate) fn encode(&self, image: &Feat, tap: &mut Tap) -> Result<ImageFeatures> {
        let image = Self::normalise(image)?;
        tap.put("encode_image.normalised", &image.t)?;
        let x = self.conv1.forward(&image)?;
        tap.put("net.pixel_encoder.bn1", &x.t)?;
        let x = relu(&x)?;
        let (w, h) = (cutie_conv_out(x.w, 3, 1, 2), cutie_conv_out(x.h, 3, 1, 2));
        let pooled = gpu_cutie_maxpool3x3s2(&x.t, x.w, x.h).map_err(|e| dev("pixel encoder maxpool", e))?;
        let x = Feat::new(pooled, x.n, x.c, w, h)?;
        tap.put("net.pixel_encoder.maxpool", &x.t)?;
        let f4 = self.res2.forward(&x, tap)?;
        let f8 = self.layer2.forward(&f4, tap)?;
        let f16 = self.layer3.forward(&f8, tap)?;
        let pix_feat = self.pix_feat_proj.forward(&f16)?;
        tap.put("net.pix_feat_proj", &pix_feat.t)?;
        tap.put("encode_image.out.f16", &f16.t)?;
        tap.put("encode_image.out.f8", &f8.t)?;
        tap.put("encode_image.out.f4", &f4.t)?;
        tap.put("encode_image.out.pix_feat", &pix_feat.t)?;
        Ok(ImageFeatures { image, f16, f8, f4, pix_feat })
    }
}

/// `KeyProjection`: f16 to the key, shrinkage and selection.
pub(crate) struct KeyProjection {
    /// 1x1 1024 -> 256 with bias (`key_proj.pix_feat_proj`, not `CUTIE.pix_feat_proj`).
    pub(crate) pix_feat_proj: Conv2d,
    /// 3x3 p1 256 -> 64: the key.
    pub(crate) key_proj: Conv2d,
    /// 3x3 p1 256 -> 1: shrinkage = d^2 + 1.
    pub(crate) d_proj: Conv2d,
    /// 3x3 p1 256 -> 64: selection = sigmoid(e).
    pub(crate) e_proj: Conv2d,
}

impl KeyProjection {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        Ok(Self {
            pix_feat_proj: w.conv("key_proj.pix_feat_proj", 1024, 256, 1, 0, 1, true)?,
            key_proj: w.conv("key_proj.key_proj", 256, 64, 3, 1, 1, true)?,
            d_proj: w.conv("key_proj.d_proj", 256, 1, 3, 1, 1, true)?,
            e_proj: w.conv("key_proj.e_proj", 256, 64, 3, 1, 1, true)?,
        })
    }

    /// `CUTIE.transform_key` (see `CutieNet::transform_key`).
    pub(crate) fn forward(&self, f16: &Feat, tap: &mut Tap) -> Result<Keys> {
        let x = self.pix_feat_proj.forward(f16)?;
        tap.put("net.key_proj.pix_feat_proj", &x.t)?;
        // PyTorch's `d ** 2` is d * d (pow with exponent 2), then + 1.
        let d = self.d_proj.forward(&x)?;
        let squared = gpu_mul(&d.t, &d.t).map_err(|e| dev("shrinkage", e))?;
        let ones = gpu_rife_fill(squared.rows(), squared.cols(), 1.0).map_err(|e| dev("shrinkage", e))?;
        let shrinkage = Feat::new(gpu_add(&squared, &ones).map_err(|e| dev("shrinkage", e))?, d.n, d.c, d.w, d.h)?;
        let e = self.e_proj.forward(&x)?;
        let selection = Feat::new(gpu_cutie_sigmoid(&e.t).map_err(|err| dev("selection", err))?, e.n, e.c, e.w, e.h)?;
        let key = self.key_proj.forward(&x)?;
        tap.put("key.key", &key.t)?;
        tap.put("key.shrinkage", &shrinkage.t)?;
        tap.put("key.selection", &selection.t)?;
        Ok(Keys { key, shrinkage, selection })
    }
}

impl CutieNet {
    /// `CUTIE.encode_image`: `image` is one image [3, H*W] in 0..1, already padded to a multiple
    /// of 16 (the oracle's `encode_image.in.image`). Normalises it (ImageNet mean 0.485, 0.456,
    /// 0.406, std 0.229, 0.224, 0.225: (x - mean) / std), runs the pixel encoder and
    /// `pix_feat_proj`.
    ///
    /// To match bit for bit, the 0..1 image must be made as the worker makes it
    /// (`torch.from_numpy(chw).float().to(cuda) / 255`): PyTorch's CUDA division by a CPU scalar
    /// multiplies by its reciprocal, so each value is `u8 as f32 * (1.0f32 / 255.0)` in f32, not
    /// `u8 as f32 / 255.0` (which differs in 126 of the 256 byte values). The padding is zeros.
    ///
    /// Taps: `encode_image.normalised`, `net.pixel_encoder.bn1` (conv1 + bn1, before the ReLU),
    /// `net.pixel_encoder.maxpool`, `net.pixel_encoder.res2.{i}` and `net.pixel_encoder.res2`,
    /// the same for `layer2` and `layer3`, `net.pix_feat_proj`, `encode_image.out.f16`, `.f8`,
    /// `.f4`, `.pix_feat`. (`net.pixel_encoder.conv1` is the conv before its BatchNorm, which is
    /// folded here: not tapped.)
    pub(crate) fn encode_image(&self, image: &Feat, tap: &mut Tap) -> Result<ImageFeatures> {
        self.pixel_encoder.encode(image, tap)
    }

    /// `CUTIE.transform_key` on f16 (one image, 1024 at stride 16): key (64), shrinkage (1),
    /// selection (64). Taps: `net.key_proj.pix_feat_proj`, `key.key`, `key.shrinkage`,
    /// `key.selection`.
    pub(crate) fn transform_key(&self, f16: &Feat, tap: &mut Tap) -> Result<Keys> {
        self.key_proj.forward(f16, tap)
    }
}

/// An oracle tap of shape [n, c, h, w] on the device as a `Feat`. (General: wanted in `oracle.rs`
/// as `OracleTap::feat` at the merge, so the other parts' checks need no copy.)
fn oracle_feat(t: &OracleTap) -> Result<Feat> {
    let [n, c, h, w] = t.shape[..] else {
        return Err(DiffusionError::model(format!("cutie oracle: {:?} is not [n, c, h, w]", t.shape)));
    };
    Feat::upload(&t.values, n, c, w, h)
}

/// The oracle names this stage owns: everything `encode_image` and `transform_key` tap.
fn stage_owns(name: &str) -> bool {
    let base = name.split('#').next().unwrap_or(name);
    base == "net.pix_feat_proj"
        || ["encode_image.", "net.pixel_encoder.", "net.key_proj.", "key."].iter().any(|p| base.starts_with(p))
}

/// Names the stage owns but does not compare: its input, and conv1 before bn1 (folded here).
const NOT_COMPARED: [&str; 2] = ["encode_image.in.image", "net.pixel_encoder.conv1"];

/// Fails every name the stage owns that one side has and the other lacks, so a renamed tap or a
/// regenerated oracle cannot drop comparisons and still pass. `seen` is the `Comparer`'s count of
/// each name put (a name put again is the oracle's `name#1`, ...).
fn check_coverage(oracle: &Oracle, report: &mut Report, step: &str, seen: &HashMap<String, usize>) {
    let mut put = HashSet::new();
    for (name, &count) in seen {
        put.insert(name.clone());
        put.extend((1..count).map(|i| format!("{name}#{i}")));
    }
    for name in oracle.names(step) {
        if stage_owns(name) && !NOT_COMPARED.contains(&name) && !put.contains(name) {
            report.fail("encoder", step, name, "in the oracle, not put by the port");
        }
    }
    let mut extra: Vec<&String> = put.iter().filter(|name| !oracle.has(step, name)).collect();
    extra.sort();
    for name in extra {
        report.fail("encoder", step, name, "put by the port, not in the oracle");
    }
}

/// Stage `encoder`: for every oracle step, the oracle's `encode_image.in.image` through
/// `encode_image` and the oracle's `encode_image.out.f16` through `transform_key`, every tap
/// compared (`Tol::STAGE`) and the names checked both ways (`check_coverage`). The parts are
/// loaded alone (not as a `CutieNet`), so the stage runs before the other parts are ported.
pub(crate) fn check(oracle: &Oracle, report: &mut Report, weights: &mut Weights) -> Result<()> {
    let pixel_encoder = PixelEncoder::load(weights)?;
    let key_proj = KeyProjection::load(weights)?;
    let tol = |_: &str| Tol::STAGE;
    for step in STEPS {
        // Every step encodes its frame (the commit and each propagation): an oracle step without
        // it has lost the stage's coverage.
        if !oracle.has(step, "encode_image.in.image") {
            report.fail("encoder", step, "encode_image.in.image", "not in the oracle: every step encodes its frame");
            continue;
        }
        let image = oracle_feat(&oracle.tap(step, "encode_image.in.image")?)?;
        let f16 = oracle_feat(&oracle.tap(step, "encode_image.out.f16")?)?;
        let mut comparer = Comparer::new(oracle, report, "encoder", step, &tol);
        comparer.with(|tap| pixel_encoder.encode(&image, tap))?;
        comparer.with(|tap| key_proj.forward(&f16, tap))?;
        let seen = std::mem::take(&mut comparer.seen);
        check_coverage(oracle, report, step, &seen);
    }
    Ok(())
}
