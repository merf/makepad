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

use crate::model::{dev, relu, Conv2d, CutieNet, Feat, ImageFeatures, Keys, Tap, Weights};
use crate::oracle::{Comparer, Oracle, Report, Tol};
use crate::{DiffusionError, Result};
use makepad_ai_common::gpu::{
    cutie_conv_out, gpu_add, gpu_cutie_maxpool3x3s2, gpu_cutie_normalise_rows, gpu_cutie_sigmoid, gpu_mul, gpu_rife_fill,
    gpu_upload, GpuTensor,
};

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
    /// The ReLUs and the residual's add are in the convolutions' bias passes
    /// (`Conv2d::forward_fused`).
    pub(crate) fn forward(&self, x: &Feat) -> Result<Feat> {
        let out = self.conv1.forward_relu(x)?;
        let out = self.conv2.forward_relu(&out)?;
        match &self.downsample {
            Some(d) => self.conv3.forward_fused(&out, Some(&d.forward(x)?), true),
            None => self.conv3.forward_fused(&out, Some(x), true),
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
    /// `PIXEL_MEAN` and `PIXEL_STD` on the device, [3, 1].
    pub(crate) pixel_mean: GpuTensor,
    pub(crate) pixel_std: GpuTensor,
}

impl PixelEncoder {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        let upload = |values: &[f32; 3]| gpu_upload(values, 3, 1).map_err(|e| dev("pixel normalisation", e));
        Ok(Self {
            pixel_mean: upload(&PIXEL_MEAN)?,
            pixel_std: upload(&PIXEL_STD)?,
            conv1: w.conv_bn("pixel_encoder.conv1", "pixel_encoder.bn1", 3, 64, 7, 3, 2)?,
            res2: ResLayer::load(w, "pixel_encoder.res2", 64, 64, 3, 1)?,
            layer2: ResLayer::load(w, "pixel_encoder.layer2", 256, 128, 4, 2)?,
            layer3: ResLayer::load(w, "pixel_encoder.layer3", 512, 256, 6, 2)?,
            pix_feat_proj: w.conv("pix_feat_proj", 1024, 256, 1, 0, 1, true)?,
        })
    }

    /// `(image - mean) / std` per channel, in that order and in f32 (a true division), as
    /// `CUTIE.encode_image`. `image` is one image of 3 channels.
    pub(crate) fn normalise(&self, image: &Feat) -> Result<Feat> {
        if image.n != 1 || image.c != 3 {
            return Err(DiffusionError::model(format!(
                "cutie encode_image: {} images of {} channels, wants one RGB image",
                image.n, image.c
            )));
        }
        let t = gpu_cutie_normalise_rows(&image.t, &self.pixel_mean, &self.pixel_std).map_err(|e| dev("normalise", e))?;
        Feat::new(t, 1, 3, image.w, image.h)
    }

    /// In an fp16 step where the backend runs graphs (`GPU_CUTIE_GRAPH`, macOS): the ResNet from
    /// conv1 to layer3 as one graph in half (`gpu_cutie_graph_f16`), its activations half between
    /// the convolutions, f4, f8 and f16 out in fp32. MAKEPAD_CUTIE_GRAPH_ENCODER=0 keeps a
    /// convolution at a time. `None` when not used.
    fn encode_graph(&self, image: &Feat) -> Result<Option<(Feat, Feat, Feat)>> {
        use makepad_ai_common::gpu::{gpu_cutie_graph_f16, CutieGraphOp, GPU_CUTIE_GRAPH};
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *ON.get_or_init(|| std::env::var("MAKEPAD_CUTIE_GRAPH_ENCODER").map_or(true, |v| v.trim() != "0"));
        if !GPU_CUTIE_GRAPH || !enabled || !crate::model::HalfConvs::on() {
            return Ok(None);
        }
        fn push<'a>(ops: &mut Vec<CutieGraphOp<'a>>, op: CutieGraphOp<'a>) -> usize {
            ops.push(op);
            ops.len()
        }
        let mut ops: Vec<CutieGraphOp<'_>> = Vec::new();
        let mut x = push(&mut ops, CutieGraphOp::Conv { src: 0, conv: self.conv1.graph_conv(), residual: None, relu: true });
        x = push(&mut ops, CutieGraphOp::MaxPool3x3s2 { src: x });
        let mut stages = [0usize; 3];
        for (s, layer) in [&self.res2, &self.layer2, &self.layer3].into_iter().enumerate() {
            for block in &layer.blocks {
                let a = push(&mut ops, CutieGraphOp::Conv { src: x, conv: block.conv1.graph_conv(), residual: None, relu: true });
                let b = push(&mut ops, CutieGraphOp::Conv { src: a, conv: block.conv2.graph_conv(), residual: None, relu: true });
                let residual = match &block.downsample {
                    Some(d) => push(&mut ops, CutieGraphOp::Conv { src: x, conv: d.graph_conv(), residual: None, relu: false }),
                    None => x,
                };
                x = push(&mut ops, CutieGraphOp::Conv { src: b, conv: block.conv3.graph_conv(), residual: Some(residual), relu: true });
            }
            stages[s] = x;
        }
        let key = format!("{}::pixel_encoder", self.conv1.key);
        let outs = gpu_cutie_graph_f16(&image.t, image.w, image.h, &key, &ops, &stages).map_err(|e| dev("pixel encoder graph", e))?;
        let [f4, f8, f16]: [GpuTensor; 3] = outs.try_into().map_err(|_| dev("pixel encoder graph", "three outputs".into()))?;
        let feat = |t: GpuTensor, c: usize, div: usize| Feat::new(t, 1, c, image.w / div, image.h / div);
        Ok(Some((feat(f4, 256, 4)?, feat(f8, 512, 8)?, feat(f16, 1024, 16)?)))
    }

    /// `CUTIE.encode_image` (see `CutieNet::encode_image`).
    pub(crate) fn encode(&self, image: &Feat, tap: &mut Tap) -> Result<ImageFeatures> {
        let image = self.normalise(image)?;
        tap.put("encode_image.normalised", &image.t)?;
        if let Some((f4, f8, f16)) = self.encode_graph(&image)? {
            let pix_feat = self.pix_feat_proj.forward(&f16)?;
            return Ok(ImageFeatures { image, f16, f8, f4, pix_feat });
        }
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

/// The oracle names this stage owns: everything `encode_image` and `transform_key` tap.
const OWNED: [&str; 5] = ["net.pix_feat_proj", "encode_image.", "net.pixel_encoder.", "net.key_proj.", "key."];

/// Names the stage owns but does not compare: its input, and conv1 before bn1 (folded here).
const NOT_COMPARED: [&str; 2] = ["encode_image.in.image", "net.pixel_encoder.conv1"];

/// Stage `encoder`: for every oracle step, the oracle's `encode_image.in.image` through
/// `encode_image` and the oracle's `encode_image.out.f16` through `transform_key`, every tap
/// compared (`Tol::STAGE`) and the names checked both ways (`Comparer::check_coverage`). The
/// parts are loaded alone (not as a `CutieNet`), so the stage runs before the other parts are
/// ported.
pub(crate) fn check(oracle: &Oracle, report: &mut Report, weights: &mut Weights) -> Result<()> {
    let pixel_encoder = PixelEncoder::load(weights)?;
    let key_proj = KeyProjection::load(weights)?;
    let tol = |_: &str| Tol::STAGE;
    for step in oracle.steps() {
        // Every step encodes its frame (the commit and each propagation): an oracle step without
        // it has lost the stage's coverage.
        if !oracle.has(step, "encode_image.in.image") {
            report.fail("encoder", step, "encode_image.in.image", "not in the oracle: every step encodes its frame");
            continue;
        }
        let image = oracle.feat(step, "encode_image.in.image")?;
        let f16 = oracle.feat(step, "encode_image.out.f16")?;
        let mut comparer = Comparer::new(oracle, report, "encoder", step, &tol);
        comparer.with(|tap| pixel_encoder.encode(&image, tap))?;
        comparer.with(|tap| key_proj.forward(&f16, tap))?;
        comparer.check_coverage(&OWNED, &NOT_COMPARED);
    }
    Ok(())
}
