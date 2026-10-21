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
//!
//! Cutie's decoder can run the objects in chunks (`chunk_size`); every object comes out the same
//! either way, so this always runs them all at once (Cutie's fast path).

use crate::model::{add, dev, relu, Conv2d, CutieNet, Feat, GroupResBlock, ImageFeatures, Tap, Weights};
use crate::oracle::{Comparer, Oracle, Report, Tol, STEPS};
use crate::{DiffusionError, Result};
use makepad_ai_common::gpu::{
    gpu_birefnet_resize_bilinear, gpu_cutie_aggregate, gpu_cutie_gru, gpu_cutie_sigmoid, gpu_cutie_softmax_axis0,
    gpu_download, gpu_pixal_pool, GpuTensor,
};

/// Cutie's MaskUpsampleBlock: the objects' features bilinear x2 (align_corners false), the
/// image's skip features (one image) added to each object, then a GroupResBlock.
pub(crate) struct MaskUpsampleBlock {
    pub(crate) out_conv: GroupResBlock,
}

impl MaskUpsampleBlock {
    pub(crate) fn load(w: &mut Weights, name: &str, in_c: usize, out_c: usize) -> Result<Self> {
        Ok(Self { out_conv: GroupResBlock::load(w, &format!("{name}.out_conv"), in_c, out_c)? })
    }

    /// `g` the objects' features (n images), `skip` the image's at twice the size (one image).
    pub(crate) fn forward(&self, g: &Feat, skip: &Feat) -> Result<Feat> {
        let (w, h) = (g.w * 2, g.h * 2);
        if skip.n != 1 || skip.c != g.c || skip.w != w || skip.h != h {
            return Err(DiffusionError::model(format!(
                "cutie mask upsample: skip {}x{} at {}x{} for {} channels at {w}x{h}",
                skip.n, skip.c, skip.w, skip.h, g.c
            )));
        }
        let up = gpu_birefnet_resize_bilinear(&g.t, g.w, g.h, w, h, false).map_err(|e| dev("mask upsample", e))?;
        let up = Feat::new(up, g.n, g.c, w, h)?;
        let g = if g.n == 1 { add(skip, &up)? } else { add(&skip.repeat(g.n)?, &up)? };
        self.out_conv.forward(&g)
    }
}

/// Cutie's SensoryUpdater (the decoder's): the decoder's three scales through 1x1 convs to the
/// sensory width at stride 16 (strides 8 and 4 area-downsampled first) and summed, then
/// `transform` on [that, sensory] to the GRU's gates and `gpu_cutie_gru`.
pub(crate) struct SensoryUpdater {
    pub(crate) g16_conv: Conv2d,
    pub(crate) g8_conv: Conv2d,
    pub(crate) g4_conv: Conv2d,
    pub(crate) transform: Conv2d,
}

impl SensoryUpdater {
    pub(crate) fn load(w: &mut Weights, name: &str, g_dims: [usize; 3], mid: usize, sensory: usize) -> Result<Self> {
        Ok(Self {
            g16_conv: w.conv(&format!("{name}.g16_conv"), g_dims[0], mid, 1, 0, 1, true)?,
            g8_conv: w.conv(&format!("{name}.g8_conv"), g_dims[1], mid, 1, 0, 1, true)?,
            g4_conv: w.conv(&format!("{name}.g4_conv"), g_dims[2], mid, 1, 0, 1, true)?,
            transform: w.conv(&format!("{name}.transform"), mid + sensory, sensory * 3, 3, 1, 1, true)?,
        })
    }

    /// `p16`, `p8` and `p4` the decoder's features at strides 16, 8 and 4 (n images each; `p4`
    /// with the logit as its last channel), `h` the sensory memory (n, at stride 16).
    pub(crate) fn forward(&self, p16: &Feat, p8: &Feat, p4: &Feat, h: &Feat) -> Result<Feat> {
        let p8 = Self::area_down(p8, p16.w, p16.h)?;
        let p4 = Self::area_down(p4, p16.w, p16.h)?;
        // (g16 + g8) + g4, PyTorch's order.
        let g = add(&add(&self.g16_conv.forward(p16)?, &self.g8_conv.forward(&p8)?)?, &self.g4_conv.forward(&p4)?)?;
        let values = self.transform.forward(&g.cat_channels(h)?)?;
        let new_h = gpu_cutie_gru(&h.t, &values.t, h.n, h.c).map_err(|e| dev("sensory gru", e))?;
        Feat::new(new_h, h.n, h.c, h.w, h.h)
    }

    /// PyTorch's area interpolation (`F.interpolate(mode='area')`, adaptive average pooling) of
    /// every channel of `x` to `w` x `h`.
    fn area_down(x: &Feat, w: usize, h: usize) -> Result<Feat> {
        let t = gpu_pixal_pool(&x.t, x.w, x.h, w, h).map_err(|e| dev("area downsample", e))?;
        Feat::new(t, x.n, x.c, w, h)
    }
}

/// Cutie's MaskDecoder (`cutie-base-mega`: up dims 256, 128, 128; the image's f8 512 wide, f4
/// 256; sensory 256).
pub(crate) struct MaskDecoder {
    /// `decoder_feat_proc.transforms.0`: f8, 512 -> 256.
    pub(crate) feat_proc8: Conv2d,
    /// `decoder_feat_proc.transforms.1`: f4, 256 -> 128.
    pub(crate) feat_proc4: Conv2d,
    pub(crate) up_16_8: MaskUpsampleBlock,
    pub(crate) up_8_4: MaskUpsampleBlock,
    pub(crate) pred: Conv2d,
    pub(crate) sensory_update: SensoryUpdater,
}

impl MaskDecoder {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        let (f8_dim, f4_dim) = (512, 256);
        let up = [256, 128, 128];
        let sensory = 256;
        Ok(Self {
            feat_proc8: w.conv("mask_decoder.decoder_feat_proc.transforms.0", f8_dim, up[0], 1, 0, 1, true)?,
            feat_proc4: w.conv("mask_decoder.decoder_feat_proc.transforms.1", f4_dim, up[1], 1, 0, 1, true)?,
            up_16_8: MaskUpsampleBlock::load(w, "mask_decoder.up_16_8", up[0], up[1])?,
            up_8_4: MaskUpsampleBlock::load(w, "mask_decoder.up_8_4", up[1], up[2])?,
            pred: w.conv("mask_decoder.pred", up[2], 1, 3, 1, 1, true)?,
            sensory_update: SensoryUpdater::load(w, "mask_decoder.sensory_update", [up[0], up[1], up[2] + 1], sensory, sensory)?,
        })
    }

    /// `CutieNet::segment` from the image's `f8` and `f4` (one image each) on: the decoder, the
    /// sensory update when `update_sensory`, the probabilities. Taps as `CutieNet::segment`.
    pub(crate) fn segment(
        &self, f8: &Feat, f4: &Feat, memory_readout: &Feat, sensory: &Feat, update_sensory: bool, tap: &mut Tap,
    ) -> Result<(Feat, GpuTensor, GpuTensor)> {
        tap.put("segment.in.memory_readout", &memory_readout.t)?;
        tap.put("segment.in.sensory", &sensory.t)?;
        let f8 = self.feat_proc8.forward(f8)?;
        tap.put("net.mask_decoder.decoder_feat_proc.0", &f8.t)?;
        let f4 = self.feat_proc4.forward(f4)?;
        tap.put("net.mask_decoder.decoder_feat_proc.1", &f4.t)?;

        let p16 = memory_readout;
        let p8 = self.up_16_8.forward(p16, &f8)?;
        tap.put("net.mask_decoder.up_16_8", &p8.t)?;
        let p4 = self.up_8_4.forward(&p8, &f4)?;
        tap.put("net.mask_decoder.up_8_4", &p4.t)?;
        // One logit per object at stride 4, from p4 through a ReLU; p4 itself goes on to the
        // sensory update.
        let logits = self.pred.forward(&relu(&p4)?)?;
        tap.put("net.mask_decoder.pred", &logits.t)?;

        let new_sensory = if update_sensory {
            let s = self.sensory_update.forward(p16, &p8, &p4.cat_channels(&logits)?, sensory)?;
            tap.put("net.mask_decoder.sensory_update", &s.t)?;
            s
        } else {
            sensory.view()?
        };

        // CUTIE.segment: sigmoid, the background aggregated in, x4 to the padded frame, softmax
        // over background and objects.
        let prob = gpu_cutie_sigmoid(&logits.t).map_err(|e| dev("segment sigmoid", e))?;
        let agg = gpu_cutie_aggregate(&prob).map_err(|e| dev("segment aggregate", e))?;
        let (w, h) = (logits.w * 4, logits.h * 4);
        let logits = gpu_birefnet_resize_bilinear(&agg, logits.w, logits.h, w, h, false).map_err(|e| dev("segment upsample", e))?;
        let prob = gpu_cutie_softmax_axis0(&logits).map_err(|e| dev("segment softmax", e))?;
        tap.put("segment.out.sensory", &new_sensory.t)?;
        tap.put("segment.out.logits", &logits)?;
        tap.put("segment.out.prob", &prob)?;
        Ok((new_sensory, logits, prob))
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
    pub(crate) fn segment(&self, feats: &ImageFeatures, memory_readout: &Feat, sensory: &Feat, update_sensory: bool, tap: &mut Tap) -> Result<(Feat, GpuTensor, GpuTensor)> {
        self.mask_decoder.segment(&feats.f8, &feats.f4, memory_readout, sensory, update_sensory, tap)
    }
}

/// An oracle tap as planar images: [n, c, h, w], or Cutie's grouped [1, n, c, h, w].
fn oracle_feat(oracle: &Oracle, step: &str, name: &str) -> Result<Feat> {
    let tap = oracle.tap(step, name)?;
    let (n, c, h, w) = match tap.shape[..] {
        [n, c, h, w] | [1, n, c, h, w] => (n, c, h, w),
        _ => return Err(DiffusionError::model(format!("cutie oracle {step}/{name}: shape {:?} is not planar", tap.shape))),
    };
    Feat::upload(&tap.values, n, c, w, h)
}

/// Stage `decoder`: for every oracle step that segments, the oracle's `encode_image.out.f8`,
/// `.f4` and `segment.in.*` through `segment`, every tap compared. The sensory memory is updated
/// where the oracle's was (it tapped `net.mask_decoder.sensory_update`); where it was, the step
/// runs again without the update too, against the same outputs.
pub(crate) fn check(oracle: &Oracle, report: &mut Report, weights: &mut Weights) -> Result<()> {
    let decoder = MaskDecoder::load(weights)?;
    let tol = |_: &str| Tol::STAGE;
    let mut steps = 0;
    for step in STEPS {
        if !oracle.has(step, "segment.in.memory_readout") {
            continue;
        }
        steps += 1;
        let f8 = oracle_feat(oracle, step, "encode_image.out.f8")?;
        let f4 = oracle_feat(oracle, step, "encode_image.out.f4")?;
        let readout = oracle_feat(oracle, step, "segment.in.memory_readout")?;
        let sensory = oracle_feat(oracle, step, "segment.in.sensory")?;
        let update_sensory = oracle.has(step, "net.mask_decoder.sensory_update");
        let mut cmp = Comparer::new(oracle, report, "decoder", step, &tol);
        cmp.with(|tap| decoder.segment(&f8, &f4, &readout, &sensory, update_sensory, tap).map(|_| ()))?;
        if update_sensory {
            // The oracle updates the sensory memory on every step it segments (the worker's
            // stagger set is every step). Without the update (Cutie's `end` frame) the logits and
            // probabilities are the same and the sensory memory comes back as it went in.
            let (kept, logits, prob) = decoder.segment(&f8, &f4, &readout, &sensory, false, &mut Tap::none())?;
            let logits = gpu_download(&logits).map_err(|e| dev("download", e))?;
            let prob = gpu_download(&prob).map_err(|e| dev("download", e))?;
            let expected_sensory = oracle.tap(step, "segment.in.sensory")?;
            report.compare("decoder", step, "segment.out.sensory (no update)", &kept.download()?, &expected_sensory, Tol::STAGE);
            report.compare("decoder", step, "segment.out.logits (no update)", &logits, &oracle.tap(step, "segment.out.logits")?, Tol::STAGE);
            report.compare("decoder", step, "segment.out.prob (no update)", &prob, &oracle.tap(step, "segment.out.prob")?, Tol::STAGE);
        }
    }
    if steps == 0 {
        report.fail("decoder", "-", "(stage)", "no oracle step segments");
    }
    Ok(())
}
