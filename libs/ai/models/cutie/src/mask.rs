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
//! positional encoding (`model::PositionalEncoding`, channel-last); then `feature_pred` (Linear,
//! ReLU, Linear) and `weights_pred` (Linear,
//! ReLU, Linear to 16); weights = sigmoid(logits) * masks; sums = weights^T @ feature [16, 256],
//! area = each summary's weights summed; summaries = [sums, area] [16, 257] per object.

use crate::model::{dev, relu, Conv2d, CutieNet, Feat, FusionBlock, Linear, PositionalEncoding, Tap, Weights};
use crate::oracle::{Comparer, Oracle, OracleTap, Report, Tol};
use crate::{DiffusionError, Result};
use makepad_ai_common::gpu::{
    cutie_conv_out, gpu_birefnet_relu, gpu_concat_cols, gpu_concat_rows_many, gpu_cutie_add_broadcast, gpu_cutie_gru,
    gpu_cutie_matmul, gpu_cutie_maxpool3x3s2, gpu_cutie_others, gpu_cutie_sigmoid, gpu_download, gpu_mul, gpu_pixal_pool,
    gpu_planar_tokens_transpose, gpu_realesrgan_scale_add, gpu_reshape, gpu_rife_fill, gpu_slice_rows, GpuTensor,
};

/// The pixel features' width (`model/base.yaml`).
const PIXEL_DIM: usize = 256;
/// The value's width.
const VALUE_DIM: usize = 256;
/// The sensory memory's width.
const SENSORY_DIM: usize = 256;
/// The mask encoder's last stage's width (`mask_encoder.final_dim`).
const FINAL_DIM: usize = 256;
/// The object summarizer's width.
const EMBED_DIM: usize = 256;
/// Summaries per object: the first half pool over the mask, the second over its complement.
const NUM_SUMMARIES: usize = 16;
/// What the object transformer adds to a summary's area before dividing its sums by it
/// (`object_transformer.py`: `obj_sums / (obj_area + 1e-4)`).
const AREA_EPS: f32 = 1e-4;

/// A ResNet BasicBlock (`model/utils/resnet.py`), its BatchNorms folded:
/// relu(conv2(relu(conv1(x))) + residual), the residual through `downsample` (1x1 with the
/// stride, + BN) where the shape changes.
pub(crate) struct BasicBlock {
    pub(crate) conv1: Conv2d,
    pub(crate) conv2: Conv2d,
    pub(crate) downsample: Option<Conv2d>,
}

impl BasicBlock {
    fn load(w: &mut Weights, name: &str, in_c: usize, out_c: usize, stride: usize) -> Result<Self> {
        Ok(Self {
            conv1: w.conv_bn(&format!("{name}.conv1"), &format!("{name}.bn1"), in_c, out_c, 3, 1, stride)?,
            conv2: w.conv_bn(&format!("{name}.conv2"), &format!("{name}.bn2"), out_c, out_c, 3, 1, 1)?,
            downsample: if stride != 1 || in_c != out_c {
                Some(w.conv_bn(&format!("{name}.downsample.0"), &format!("{name}.downsample.1"), in_c, out_c, 1, 0, stride)?)
            } else {
                None
            },
        })
    }

    /// The ReLUs and the residual's add are in the convolutions' bias passes
    /// (`Conv2d::forward_fused`).
    fn forward(&self, x: &Feat) -> Result<Feat> {
        let out = self.conv1.forward_relu(x)?;
        match &self.downsample {
            Some(d) => self.conv2.forward_fused(&out, Some(&d.forward(x)?), true),
            None => self.conv2.forward_fused(&out, Some(x), true),
        }
    }
}

/// A ResNet stage (`layer1`..`layer3`): its BasicBlocks in order, the first carrying the stride.
pub(crate) struct ResStage {
    /// The stage's module name in `CUTIE` (`mask_encoder.layer1`): its taps are `net.{name}.{i}`
    /// after each block and `net.{name}` after the last.
    name: String,
    pub(crate) blocks: Vec<BasicBlock>,
}

impl ResStage {
    fn load(w: &mut Weights, name: &str, in_c: usize, out_c: usize, stride: usize, blocks: usize) -> Result<Self> {
        let blocks = (0..blocks)
            .map(|i| {
                let (block_in, block_stride) = if i == 0 { (in_c, stride) } else { (out_c, 1) };
                BasicBlock::load(w, &format!("{name}.{i}"), block_in, out_c, block_stride)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { name: name.to_string(), blocks })
    }

    fn forward(&self, x: &Feat, tap: &mut Tap) -> Result<Feat> {
        let mut x = self.blocks[0].forward(x)?;
        tap.put(&format!("net.{}.0", self.name), &x.t)?;
        for (i, block) in self.blocks.iter().enumerate().skip(1) {
            x = block.forward(&x)?;
            tap.put(&format!("net.{}.{i}", self.name), &x.t)?;
        }
        tap.put(&format!("net.{}", self.name), &x.t)?;
        Ok(x)
    }
}

/// The mask encoder (ResNet-18 on [image, mask, others] to stride 16, fused with the pixel
/// features) and its sensory deep update.
pub(crate) struct MaskEncoder {
    /// conv1 (7x7 s2 p3, 5 -> 64) with bn1 folded in.
    pub(crate) conv1: Conv2d,
    pub(crate) layers: [ResStage; 3],
    pub(crate) fuser: FusionBlock,
    /// SensoryDeepUpdater's `transform`: 3x3 p1 on [value, sensory] (512 -> 768: the forget,
    /// update and new-value gates).
    pub(crate) sensory_transform: Conv2d,
}

impl MaskEncoder {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        Ok(Self {
            conv1: w.conv_bn("mask_encoder.conv1", "mask_encoder.bn1", 5, 64, 7, 3, 2)?,
            layers: [
                ResStage::load(w, "mask_encoder.layer1", 64, 64, 1, 2)?,
                ResStage::load(w, "mask_encoder.layer2", 64, 128, 2, 2)?,
                ResStage::load(w, "mask_encoder.layer3", 128, FINAL_DIM, 2, 2)?,
            ],
            fuser: FusionBlock::load(w, "mask_encoder.fuser", PIXEL_DIM, FINAL_DIM, VALUE_DIM)?,
            sensory_transform: w.conv(
                "mask_encoder.sensory_update.transform",
                VALUE_DIM + SENSORY_DIM,
                SENSORY_DIM * 3,
                3,
                1,
                1,
                true,
            )?,
        })
    }

    /// `MaskEncoder.forward` (one chunk: every object at once): the value (n, 256 at stride 16)
    /// and the sensory memory, deep-updated or the old one. Inputs as `CutieNet::encode_mask`.
    fn forward(&self, image: &Feat, pix_feat: &Feat, sensory: &Feat, masks: &Feat, deep_update: bool, tap: &mut Tap) -> Result<(Feat, Feat)> {
        let n = masks.n;
        // Each object's [R, G, B, mask, others] (MainToGroupDistributor's cat: image first).
        let others = gpu_cutie_others(&masks.t).map_err(|e| dev("mask others", e))?;
        let mut rows = Vec::with_capacity(n * 2);
        for o in 0..n {
            rows.push(gpu_slice_rows(&masks.t, o, 1).map_err(|e| dev("mask input", e))?);
            rows.push(gpu_slice_rows(&others, o, 1).map_err(|e| dev("mask input", e))?);
        }
        let mut parts: Vec<&GpuTensor> = Vec::with_capacity(n * 3);
        for pair in rows.chunks(2) {
            parts.extend([&image.t, &pair[0], &pair[1]]);
        }
        let g = Feat::new(gpu_concat_rows_many(&parts).map_err(|e| dev("mask input", e))?, n, 5, image.w, image.h)?;
        tap.put("net.mask_encoder.distributor", &g.t)?;

        let g = self.conv1.forward(&g)?;
        tap.put("net.mask_encoder.bn1", &g.t)?;
        let (w, h) = (cutie_conv_out(g.w, 3, 1, 2), cutie_conv_out(g.h, 3, 1, 2));
        let g = Feat::new(gpu_cutie_maxpool3x3s2(&g.t, g.w, g.h).map_err(|e| dev("mask maxpool", e))?, n, g.c, w, h)?;
        tap.put("net.mask_encoder.maxpool", &g.t)?;
        let mut g = relu(&g)?;
        tap.put("net.mask_encoder.relu", &g.t)?;
        for layer in &self.layers {
            g = layer.forward(&g, tap)?;
        }
        if (g.w, g.h) != (pix_feat.w, pix_feat.h) {
            return Err(DiffusionError::model(format!(
                "cutie mask encoder: features at {}x{}, pix_feat at {}x{}",
                g.w, g.h, pix_feat.w, pix_feat.h
            )));
        }

        let value = self.fuser.forward(pix_feat, &g, "net.mask_encoder.fuser", tap)?;
        let sensory = if deep_update {
            let gates = self.sensory_transform.forward(&value.cat_channels(sensory)?)?;
            let new = gpu_cutie_gru(&sensory.t, &gates.t, n, SENSORY_DIM).map_err(|e| dev("mask sensory update", e))?;
            let new = Feat::new(new, n, SENSORY_DIM, value.w, value.h)?;
            tap.put("net.mask_encoder.sensory_update", &new.t)?;
            new
        } else {
            sensory.view()?
        };
        Ok((value, sensory))
    }

    /// `CUTIE.encode_mask` with this encoder and `summarizer`: what `CutieNet::encode_mask`
    /// runs, and what the check runs without the rest of the network.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mask(
        &self, summarizer: &ObjectSummarizer, image: &Feat, pix_feat: &Feat, sensory: &Feat, masks: &Feat, deep_update: bool,
        tap: &mut Tap,
    ) -> Result<(Feat, Feat, GpuTensor)> {
        if masks.n == 0 {
            // PyTorch never gets here: InferenceCore._add_memory returns early with no objects.
            return Err(DiffusionError::model("cutie encode_mask: no objects (the processor skips memorising then)"));
        }
        if image.n != 1 || image.c != 3 || pix_feat.n != 1 || pix_feat.c != PIXEL_DIM || masks.c != 1 {
            return Err(DiffusionError::model("cutie encode_mask: wants one 3-channel image, one pix_feat and 1-channel masks"));
        }
        if (masks.w, masks.h) != (image.w, image.h) || sensory.n != masks.n || sensory.c != SENSORY_DIM {
            return Err(DiffusionError::model(format!(
                "cutie encode_mask: image {}x{}, masks {} at {}x{}, sensory {}x{}",
                image.w, image.h, masks.n, masks.w, masks.h, sensory.n, sensory.c
            )));
        }
        tap.put("encode_mask.in.pix_feat", &pix_feat.t)?;
        tap.put("encode_mask.in.sensory", &sensory.t)?;
        tap.put("encode_mask.in.masks", &masks.t)?;
        let (value, sensory) = self.forward(image, pix_feat, sensory, masks, deep_update, tap)?;
        let summaries = summarizer.forward(masks, &value)?;
        tap.put("encode_mask.out.value", &value.t)?;
        tap.put("encode_mask.out.sensory", &sensory.t)?;
        tap.put("encode_mask.out.obj_summaries", &summaries)?;
        Ok((value, sensory, summaries))
    }
}

/// The object summarizer: each object's value pooled into 16 weighted summaries and their areas.
pub(crate) struct ObjectSummarizer {
    input_proj: Linear,
    /// Linear, ReLU, Linear (256 -> 256 -> 256): what each pixel contributes.
    feature_pred: [Linear; 2],
    /// Linear, ReLU, Linear (256 -> 256 -> 16): each pixel's weight logit for each summary.
    weights_pred: [Linear; 2],
    /// `pos_enc`.
    pos_enc: PositionalEncoding,
}

impl ObjectSummarizer {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        Ok(Self {
            input_proj: w.linear("object_summarizer.input_proj", VALUE_DIM, EMBED_DIM, true)?,
            feature_pred: [
                w.linear("object_summarizer.feature_pred.0", EMBED_DIM, EMBED_DIM, true)?,
                w.linear("object_summarizer.feature_pred.2", EMBED_DIM, EMBED_DIM, true)?,
            ],
            weights_pred: [
                w.linear("object_summarizer.weights_pred.0", EMBED_DIM, EMBED_DIM, true)?,
                w.linear("object_summarizer.weights_pred.2", EMBED_DIM, NUM_SUMMARIES, true)?,
            ],
            pos_enc: PositionalEncoding::load(w, "object_summarizer.pos_enc", EMBED_DIM)?,
        })
    }

    /// What the object transformer reads from summaries [.., 257] (sums, then area): each
    /// summary's sums divided by its area + 1e-4, [.., 256], in f32 as `object_transformer.py`
    /// does (summed over memory frames first; one frame here). For the check: the summaries with
    /// small areas are the ones this magnifies.
    fn normalised(summaries: &[f32]) -> Vec<f32> {
        summaries
            .chunks_exact(EMBED_DIM + 1)
            .flat_map(|row| {
                let area = row[EMBED_DIM] + AREA_EPS;
                row[..EMBED_DIM].iter().map(move |s| s / area)
            })
            .collect()
    }

    /// `ObjectSummarizer.forward`: `masks` (n images of 1 channel, each object's probability at
    /// the image's size) and `value` (n, 256 at stride 16). Out [n*16, 257], object-major: each
    /// summary's weighted sum of the features, then its total weight (the area).
    fn forward(&self, masks: &Feat, value: &Feat) -> Result<GpuTensor> {
        let (n, tokens) = (value.n, value.plane());
        if value.c != VALUE_DIM || masks.n != n {
            return Err(DiffusionError::model("cutie object summarizer: value and masks disagree"));
        }
        let pooled = gpu_pixal_pool(&masks.t, masks.w, masks.h, value.w, value.h).map_err(|e| dev("summary masks", e))?;
        let allowed = Self::summary_masks(&pooled)?;

        // Every object's value token-major, [n*h*w, 256], then projected and the encoding added.
        let per_object = (0..n)
            .map(|o| gpu_slice_rows(&value.t, o * value.c, value.c).and_then(|v| gpu_planar_tokens_transpose(&v)))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| dev("summary tokens", e))?;
        let parts: Vec<&GpuTensor> = per_object.iter().collect();
        let x = gpu_concat_rows_many(&parts).map_err(|e| dev("summary tokens", e))?;
        let x = self.input_proj.forward(&x)?;
        let pe = self.pos_enc.tokens(value.w, value.h)?;
        let x = gpu_cutie_add_broadcast(&pe, &x).map_err(|e| dev("summary encoding", e))?;

        let hidden = gpu_birefnet_relu(&self.feature_pred[0].forward(&x)?).map_err(|e| dev("summary features", e))?;
        let feature = self.feature_pred[1].forward(&hidden)?;
        let hidden = gpu_birefnet_relu(&self.weights_pred[0].forward(&x)?).map_err(|e| dev("summary weights", e))?;
        let logits = self.weights_pred[1].forward(&hidden)?;
        let sigmoid = gpu_cutie_sigmoid(&logits).map_err(|e| dev("summary weights", e))?;
        let weights = gpu_mul(&sigmoid, &allowed).map_err(|e| dev("summary weights", e))?;

        // [feature, 1], so that one product gives each summary's sums and its area together.
        let ones = gpu_rife_fill(n * tokens, 1, 1.0).map_err(|e| dev("summary pooling", e))?;
        let feature = gpu_concat_cols(&[&feature, &ones]).map_err(|e| dev("summary pooling", e))?;
        let summaries = (0..n)
            .map(|o| {
                let w = gpu_slice_rows(&weights, o * tokens, tokens)?;
                let f = gpu_slice_rows(&feature, o * tokens, tokens)?;
                gpu_cutie_matmul(&w, true, &f, false)
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| dev("summary pooling", e))?;
        let parts: Vec<&GpuTensor> = summaries.iter().collect();
        gpu_concat_rows_many(&parts).map_err(|e| dev("summary pooling", e))
    }

    /// Which pixels each summary may pool, token-major [n*h*w, 16] from the pooled masks
    /// [n, h*w]: the mask for the first 8 summaries, 1 - mask for the last 8.
    fn summary_masks(pooled: &GpuTensor) -> Result<GpuTensor> {
        let (n, tokens) = (pooled.rows(), pooled.cols());
        let half = NUM_SUMMARIES / 2;
        let ones = gpu_rife_fill(1, tokens, 1.0).map_err(|e| dev("summary masks", e))?;
        let mut per_object = Vec::with_capacity(n);
        for o in 0..n {
            let mask = gpu_slice_rows(pooled, o, 1).map_err(|e| dev("summary masks", e))?;
            // fma(-1, m, 1): the product is exact, so this rounds once, as PyTorch's 1 - m.
            let inverse = gpu_realesrgan_scale_add(&ones, &mask, -1.0).map_err(|e| dev("summary masks", e))?;
            // A row [1, h*w] is the same memory as a column [h*w, 1].
            let mask = gpu_reshape(mask, tokens, 1).map_err(|e| dev("summary masks", e))?;
            let inverse = gpu_reshape(inverse, tokens, 1).map_err(|e| dev("summary masks", e))?;
            let cols: Vec<&GpuTensor> = std::iter::repeat(&mask).take(half).chain(std::iter::repeat(&inverse).take(half)).collect();
            per_object.push(gpu_concat_cols(&cols).map_err(|e| dev("summary masks", e))?);
        }
        let parts: Vec<&GpuTensor> = per_object.iter().collect();
        gpu_concat_rows_many(&parts).map_err(|e| dev("summary masks", e))
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
        &self, image: &Feat, pix_feat: &Feat, sensory: &Feat, masks: &Feat, deep_update: bool, tap: &mut Tap,
    ) -> Result<(Feat, Feat, GpuTensor)> {
        self.mask_encoder.encode_mask(&self.object_summarizer, image, pix_feat, sensory, masks, deep_update, tap)
    }
}

/// Stage `mask`: for every oracle step with `encode_mask` taps, the oracle's inputs
/// (`encode_image.normalised`, `encode_mask.in.*`) through `encode_mask`, every tap compared and
/// the names checked both ways (`Comparer::check_coverage`).
/// The deep update runs where the oracle's did (where it has `net.mask_encoder.sensory_update`).
///
/// The `encode_mask.in.*` taps are the oracle's own inputs read back, so they are not compared
/// here. The summaries are compared as they are and also as the object transformer reads them,
/// sums / (area + 1e-4): compared whole, the first is measured against the largest areas and
/// cannot see an error in a summary with a small one, which the division magnifies.
pub(crate) fn check(oracle: &Oracle, report: &mut Report, weights: &mut Weights) -> Result<()> {
    const INPUTS: [&str; 4] = ["encode_image.normalised", "encode_mask.in.pix_feat", "encode_mask.in.sensory", "encode_mask.in.masks"];
    const SUMMARIES: &str = "encode_mask.out.obj_summaries";
    let encoder = MaskEncoder::load(weights)?;
    let summarizer = ObjectSummarizer::load(weights)?;
    let tol = |_: &str| Tol::STAGE;
    let n = oracle.objects;
    // Each step's inputs, to say where a step repeats an earlier one's encode.
    let mut encoded: Vec<(&str, bool, Vec<Vec<f32>>)> = Vec::new();
    let mut distinct = 0;
    for step in oracle.steps() {
        if !oracle.has(step, "encode_mask.in.masks") {
            // A step with the mask encoder's taps but not its input has lost the stage's coverage.
            if oracle.names(step).iter().any(|name| name.starts_with("encode_mask.") || name.starts_with("net.mask_encoder.")) {
                report.fail("mask", step, "encode_mask.in.masks", "not in the oracle, which encodes a mask here");
            }
            continue;
        }
        let [image, pix_feat, sensory, masks] = INPUTS.map(|name| oracle.tap(step, name));
        let taps = [image?, pix_feat?, sensory?, masks?];
        let feat = |i: usize, n: usize, c: usize| {
            taps[i].feat_as(n, c).map_err(|e| DiffusionError::model(format!("{step}/{}: {e}", INPUTS[i])))
        };
        let image = feat(0, 1, 3)?;
        let pix_feat = feat(1, 1, PIXEL_DIM)?;
        let sensory = feat(2, n, SENSORY_DIM)?;
        let masks = feat(3, n, 1)?;
        let deep_update = oracle.has(step, "net.mask_encoder.sensory_update");
        let inputs: Vec<Vec<f32>> = taps.into_iter().map(|t| t.values).collect();
        match encoded.iter().find(|(_, d, v)| *d == deep_update && *v == inputs) {
            Some((earlier, ..)) => report.note(format!("mask/{step}: the same inputs as {earlier}, so the same encode again")),
            None => distinct += 1,
        }
        encoded.push((step, deep_update, inputs));

        let summaries = {
            let mut comparer = Comparer::new(oracle, report, "mask", step, &tol);
            let summaries = {
                let mut sink = |name: &str, t: &GpuTensor| if INPUTS.contains(&name) { Ok(()) } else { comparer.take(name, t) };
                let mut tap = Tap::new(&mut sink);
                encoder.encode_mask(&summarizer, &image, &pix_feat, &sensory, &masks, deep_update, &mut tap)?.2
            };
            // Every oracle tap of the mask encoder compared, every tap put in the oracle: but the
            // inputs fed, and conv1 before bn1 (folded into it here).
            comparer.check_coverage(&["encode_mask.", "net.mask_encoder."], &[INPUTS[1], INPUTS[2], INPUTS[3], "net.mask_encoder.conv1"]);
            summaries
        };

        let actual = gpu_download(&summaries).map_err(|e| dev("mask check", e))?;
        let expected = oracle.tap(step, SUMMARIES)?;
        let expected = OracleTap {
            shape: vec![expected.values.len() / (EMBED_DIM + 1), EMBED_DIM],
            dtype: expected.dtype,
            values: ObjectSummarizer::normalised(&expected.values),
        };
        let name = format!("{SUMMARIES} sums / (area + 1e-4)");
        report.compare("mask", step, &name, &ObjectSummarizer::normalised(&actual), &expected, Tol::STAGE);
    }
    if encoded.is_empty() {
        report.fail("mask", "-", "(stage)", "no oracle step encodes a mask");
    } else {
        report.note(format!(
            "mask: {} encodes, {distinct} distinct; encode_mask.in.* not compared (the oracle's own inputs read back); \
             net.mask_encoder.conv1 not compared (bn1 is folded into it: its output is net.mask_encoder.bn1)",
            encoded.len()
        ));
    }
    Ok(())
}
