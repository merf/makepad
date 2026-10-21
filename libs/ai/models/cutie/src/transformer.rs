//! Cutie's pixel fusion and object transformer (`big_modules.PixelFeatureFuser`,
//! `CUTIE.pixel_fusion`, `transformer/object_transformer.QueryTransformer`,
//! `transformer/transformer_layers.*`, `CUTIE.readout_query`).
//!
//! Pixel fusion, per object: the last mask area-downsampled to stride 16 (`gpu_pixal_pool`) and
//! its others (`gpu_cutie_others`); `sensory_compress` (1x1, 258 -> 256) on [sensory, mask,
//! others]; added to the visual readout; then `fuser` (a `FusionBlock` with pix_feat).
//!
//! The object transformer (embed 256, 8 heads, 16 queries, 3 blocks, FFN 2048): queries from the
//! object memory (sums / (area + 1e-4), through `summary_to_query_init` and `_emb`, added to the
//! learnt `query_init` / `query_emb`); pixels through `pixel_init_proj` (the running pixel) and
//! `pixel_emb_proj` plus the sine encoding (`spatial_pe`, PositionalEncoding(256, scale 32,
//! temperature 128), the pixel PE). Before each block and after it, `mask_pred[i]` (ReLU, 1x1 to
//! 1 logit) gives the aux logits; the attention mask for the next block from them
//! (`gpu_cutie_attn_mask`, additive). A block: `read_from_pixel` (queries norm, then masked
//! cross-attention to the pixels, PE added to q and k only, residual), `self_attn` (norm, self
//! attention with PE on q and k, residual), `ffn` (norm, Linear, ReLU, Linear, residual),
//! `read_from_query` (pixels attend to the queries, PE on q and k, no norm, residual),
//! `pixel_ffn` (a CAResBlock on the pixels). nn.MultiheadAttention: `in_proj_weight` [768, 256]
//! (q, k, v rows), `out_proj`; attention scaled 1/sqrt(32). Fp32 attention:
//! `gpu_attention_packed_cross_bias` (with the mask) and `gpu_attention_packed_cross_f32`.

use crate::model::{not_ported, CutieNet, Feat, Tap, Weights};
use crate::oracle::{Oracle, Report};
use crate::Result;
use makepad_ai_common::gpu::GpuTensor;

pub(crate) struct PixelFuser {}

impl PixelFuser {
    pub(crate) fn load(_w: &mut Weights) -> Result<Self> {
        Err(not_ported("pixel fusion"))
    }
}

pub(crate) struct QueryTransformer {}

impl QueryTransformer {
    pub(crate) fn load(_w: &mut Weights) -> Result<Self> {
        Err(not_ported("the object transformer"))
    }
}

impl CutieNet {
    /// `CUTIE.pixel_fusion`: `pix_feat` (one image, 256 at stride 16), `visual_readout` (n, 256
    /// at stride 16), `sensory` (n, 256), `last_mask` (n images of 1 channel at the padded H, W:
    /// each object's probability last frame). Out (n, 256 at stride 16).
    ///
    /// Taps: `pixel_fusion.in.visual_readout`, `.in.sensory`, `.in.last_mask`,
    /// `net.pixel_fuser.sensory_compress`, `net.pixel_fuser.fuser.distributor`, `.block1`,
    /// `.block2`, `pixel_fusion.out`.
    pub(crate) fn pixel_fusion(&self, _pix_feat: &Feat, _visual_readout: &Feat, _sensory: &Feat, _last_mask: &Feat, _tap: &mut Tap) -> Result<Feat> {
        Err(not_ported("pixel_fusion"))
    }

    /// `CUTIE.readout_query`: the object transformer on `pixel` (n, 256 at stride 16) with
    /// `obj_memory` [n*16, 257] (each object's summed summaries, then the area). Out: the pixels
    /// (n, 256 at stride 16) and the aux logits of each prediction ([n, h*w], 4 of them).
    ///
    /// Taps: `object_transformer.in.pixel`, `.in.obj_memory`,
    /// `net.object_transformer.summary_to_query_init`, `_emb`, `.pixel_init_proj`,
    /// `.pixel_emb_proj`, `.spatial_pe` ([1, h, w, 256]), `.mask_pred.{i}`, and per block b
    /// `.blocks.{b}.attn_mask` (bool [n*8, 16, h*w], 1 = blocked), `.blocks.{b}.read_from_pixel.0`,
    /// `.self_attn`, `.ffn`, `.read_from_query.0` ([n, h*w, 256], token-major),
    /// `.pixel_ffn`, `.blocks.{b}.0` (queries), `.blocks.{b}.1` (pixels);
    /// `object_transformer.out.pixel`, `.out.aux_logits.{i}`.
    pub(crate) fn readout_query(&self, _pixel: &Feat, _obj_memory: &GpuTensor, _tap: &mut Tap) -> Result<(Feat, Vec<GpuTensor>)> {
        Err(not_ported("readout_query"))
    }
}

/// Stages `fusion` and `transformer`: for every oracle step that segments (`prop_f1`, `prop_f5`,
/// `prop_f40`, `prop_f41`), the oracle's `pixel_fusion.in.*` (and `encode_image.out.pix_feat`)
/// through `pixel_fusion`, and `object_transformer.in.*` through `readout_query`, every tap
/// compared. The attention masks are compared exactly; feeding each block the oracle's own mask
/// as well as the native one separates a flipped mask bit from a wrong block.
pub(crate) fn check(_oracle: &Oracle, _report: &mut Report, _weights: &mut Weights) -> Result<()> {
    Err(not_ported("the fusion and transformer check"))
}
