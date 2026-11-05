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
//! `pixel_emb_proj` plus the sine encoding (`spatial_pe`, `model::PositionalEncoding`, the pixel
//! PE). Before each block and after it, `mask_pred[i]` (ReLU, 1x1 to
//! 1 logit) gives the aux logits; the attention mask for the next block from them
//! (`gpu_cutie_attn_mask`, additive). A block: `read_from_pixel` (queries norm, then masked
//! cross-attention to the pixels, PE added to q and k only, residual), `self_attn` (norm, self
//! attention with PE on q and k, residual), `ffn` (norm, Linear, ReLU, Linear, residual),
//! `read_from_query` (pixels attend to the queries, PE on q and k, no norm, residual),
//! `pixel_ffn` (a CAResBlock on the pixels). nn.MultiheadAttention: `in_proj_weight` [768, 256]
//! (q, k, v rows), `out_proj`; attention scaled 1/sqrt(32). Fp32 attention:
//! `gpu_attention_packed_cross_bias` (with the mask) and `gpu_attention_packed_cross_f32`.
//!
//! Inside the transformer the objects' tokens are interleaved: queries [16*n, 256] with row
//! q*n + o, pixels [h*w*n, 256] with row p*n + o. Read as [len, n*256], that is every object's
//! 256 channels side by side, so each attention runs all objects at once with n*8 heads of 32
//! (object o's head h at columns (o*8 + h)*32, its mask rows at (o*8 + h)*16, as
//! `gpu_cutie_attn_mask` lays them out); the linears and LayerNorms work row by row and do not
//! care; and a planar batch [n*256, h*w] becomes these tokens with one transpose. Taps are put
//! object-major, PyTorch's order.

use crate::model::{add, dev, relu, CAResBlock, Conv2d, CutieNet, Feat, FusionBlock, LayerNorm, Linear, PositionalEncoding, Tap, Weights};
use crate::oracle::{Comparer, Oracle, OracleTap, Report, Tol};
use crate::{DiffusionError, Result};
use makepad_ai_common::gpu::{
    gpu_add, gpu_add_cols_broadcast, gpu_attention_packed_cross_bias, gpu_attention_packed_cross_f32, gpu_birefnet_relu,
    gpu_concat_cols, gpu_concat_rows_many, gpu_cutie_attn_mask, gpu_cutie_object_values, gpu_cutie_others, gpu_download,
    gpu_pixal_pool, gpu_planar_tokens_transpose, gpu_reshape, gpu_slice_cols, gpu_slice_rows, gpu_upload, GpuTensor,
};

/// The object transformer's width (`embed_dim`, also the pixel and value width), heads, queries
/// (`num_queries`, also the object summaries), blocks and FFN width (`config/model/base.yaml`).
const EMBED: usize = 256;
const HEADS: usize = 8;
const HEAD_DIM: usize = EMBED / HEADS;
const QUERIES: usize = 16;
const BLOCKS: usize = 3;
const FF_DIM: usize = 2048;
/// Added to each summary's area before dividing by it (`QueryTransformer.forward`).
const AREA_EPS: f32 = 1e-4;

// --- Pixel fusion ---

/// `PixelFeatureFuser` with several objects: `sensory_compress` on each object's [sensory, last
/// mask, others], added to the visual readout, then `fuser` with the image's pix_feat.
pub(crate) struct PixelFuser {
    sensory_compress: Conv2d,
    fuser: FusionBlock,
}

impl PixelFuser {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        Ok(Self {
            sensory_compress: w.conv("pixel_fuser.sensory_compress", EMBED + 2, EMBED, 1, 0, 1, true)?,
            fuser: FusionBlock::load(w, "pixel_fuser.fuser", EMBED, EMBED, EMBED)?,
        })
    }

    /// `CUTIE.pixel_fusion` (see `CutieNet::pixel_fusion`).
    pub(crate) fn forward(&self, pix_feat: &Feat, visual_readout: &Feat, sensory: &Feat, last_mask: &Feat, tap: &mut Tap) -> Result<Feat> {
        tap.put("pixel_fusion.in.visual_readout", &visual_readout.t)?;
        tap.put("pixel_fusion.in.sensory", &sensory.t)?;
        tap.put("pixel_fusion.in.last_mask", &last_mask.t)?;
        let n = sensory.n;
        if visual_readout.n != n || last_mask.n != n || last_mask.c != 1 || sensory.c != EMBED {
            return Err(DiffusionError::model(format!(
                "cutie pixel fusion: {} readouts, {} sensory of {} channels, {} masks of {} channels",
                visual_readout.n, n, sensory.c, last_mask.n, last_mask.c
            )));
        }
        let fail = |e: String| dev("pixel fusion", e);
        // F.interpolate(mode='area') to the sensory memory's size, then each object's others.
        let mask = gpu_pixal_pool(&last_mask.t, last_mask.w, last_mask.h, sensory.w, sensory.h).map_err(fail)?;
        let others = gpu_cutie_others(&mask).map_err(fail)?;
        // cat([sensory, stack([mask, others])]) along the channels: 258 channels an object.
        let mut views = Vec::with_capacity(3 * n);
        for o in 0..n {
            views.push(gpu_slice_rows(&sensory.t, o * sensory.c, sensory.c).map_err(fail)?);
            views.push(gpu_slice_rows(&mask, o, 1).map_err(fail)?);
            views.push(gpu_slice_rows(&others, o, 1).map_err(fail)?);
        }
        let parts: Vec<&GpuTensor> = views.iter().collect();
        let stacked = Feat::new(gpu_concat_rows_many(&parts).map_err(fail)?, n, sensory.c + 2, sensory.w, sensory.h)?;
        let sensory_readout = self.sensory_compress.forward(&stacked)?;
        tap.put("net.pixel_fuser.sensory_compress", &sensory_readout.t)?;
        let p16 = add(visual_readout, &sensory_readout)?;
        let out = self.fuser.forward(pix_feat, &p16, "net.pixel_fuser.fuser", tap)?;
        tap.put("pixel_fusion.out", &out.t)?;
        Ok(out)
    }
}

// --- The object transformer ---

/// nn.MultiheadAttention (batch first, 8 heads of 32, no dropout), its in-projection split into
/// the q, k and v linears (rows 0..256, 256..512, 512..768 of `in_proj_weight`).
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl Attention {
    fn graph(&self) -> makepad_ai_common::gpu::CutieGraphAttention<'_> {
        makepad_ai_common::gpu::CutieGraphAttention {
            q: self.q.graph_linear(),
            k: self.k.graph_linear(),
            v: self.v.graph_linear(),
            out: self.out.graph_linear(),
        }
    }

    fn load(w: &mut Weights, name: &str) -> Result<Self> {
        let weight = w.f32(&format!("{name}.in_proj_weight"), &[3 * EMBED, EMBED])?;
        let bias = w.f32(&format!("{name}.in_proj_bias"), &[3 * EMBED])?;
        let part = |i: usize| {
            Linear::new(&weight[i * EMBED * EMBED..(i + 1) * EMBED * EMBED], Some(&bias[i * EMBED..(i + 1) * EMBED]), EMBED, EMBED)
        };
        Ok(Self { q: part(0)?, k: part(1)?, v: part(2)?, out: w.linear(&format!("{name}.out_proj"), EMBED, EMBED, true)? })
    }

    /// Attention of `query` to `key` and `value` for n objects, all interleaved tokens; each
    /// object attends only to its own tokens (its own heads). `mask`: additive, [n*8*q_len,
    /// kv_len] (object, head, query, key), or none.
    fn forward(&self, query: &GpuTensor, key: &GpuTensor, value: &GpuTensor, n: usize, mask: Option<&GpuTensor>) -> Result<GpuTensor> {
        let fail = |e: String| dev("attention", e);
        let (q_len, kv_len) = (query.rows() / n, key.rows() / n);
        let q = gpu_reshape(self.q.forward(query)?, q_len, n * EMBED).map_err(fail)?;
        let k = gpu_reshape(self.k.forward(key)?, kv_len, n * EMBED).map_err(fail)?;
        let v = gpu_reshape(self.v.forward(value)?, kv_len, n * EMBED).map_err(fail)?;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let heads = match mask {
            Some(mask) => gpu_attention_packed_cross_bias(&q, &k, &v, n * HEADS, scale, mask),
            None => gpu_attention_packed_cross_f32(&q, &k, &v, n * HEADS, scale),
        }
        .map_err(fail)?;
        self.out.forward(&gpu_reshape(heads, q_len * n, EMBED).map_err(fail)?)
    }
}

/// A `QueryTransformerBlock`.
struct Block {
    read_from_pixel: Attention,
    read_from_pixel_norm: LayerNorm,
    self_attn: Attention,
    self_attn_norm: LayerNorm,
    ffn_norm: LayerNorm,
    ffn1: Linear,
    ffn2: Linear,
    read_from_query: Attention,
    pixel_ffn: CAResBlock,
}

impl Block {
    fn graph(&self) -> makepad_ai_common::gpu::CutieGraphBlock<'_> {
        makepad_ai_common::gpu::CutieGraphBlock {
            read_from_pixel: self.read_from_pixel.graph(),
            read_from_pixel_norm: self.read_from_pixel_norm.graph_norm(),
            self_attn: self.self_attn.graph(),
            self_attn_norm: self.self_attn_norm.graph_norm(),
            ffn_norm: self.ffn_norm.graph_norm(),
            ffn1: self.ffn1.graph_linear(),
            ffn2: self.ffn2.graph_linear(),
            read_from_query: self.read_from_query.graph(),
            pixel_conv1: self.pixel_ffn.conv1.graph_conv(),
            pixel_conv2: self.pixel_ffn.conv2.graph_conv(),
            eca: &self.pixel_ffn.eca,
        }
    }

    fn load(w: &mut Weights, name: &str) -> Result<Self> {
        Ok(Self {
            read_from_pixel: Attention::load(w, &format!("{name}.read_from_pixel.cross_attn"))?,
            read_from_pixel_norm: w.layer_norm(&format!("{name}.read_from_pixel.norm"), EMBED)?,
            self_attn: Attention::load(w, &format!("{name}.self_attn.self_attn"))?,
            self_attn_norm: w.layer_norm(&format!("{name}.self_attn.norm"), EMBED)?,
            ffn_norm: w.layer_norm(&format!("{name}.ffn.norm"), EMBED)?,
            ffn1: w.linear(&format!("{name}.ffn.linear1"), EMBED, FF_DIM, true)?,
            ffn2: w.linear(&format!("{name}.ffn.linear2"), FF_DIM, EMBED, true)?,
            read_from_query: Attention::load(w, &format!("{name}.read_from_query.cross_attn"))?,
            pixel_ffn: CAResBlock::load(w, &format!("{name}.pixel_ffn.conv"), EMBED)?,
        })
    }

    /// Block `b` on the queries `x` (interleaved tokens) and `pixel` (planar), with the queries'
    /// and pixels' PE (interleaved) and the attention mask for `read_from_pixel`. Returns the new
    /// queries and pixels.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self, b: usize, x: &GpuTensor, pixel: &Feat, query_pe: &GpuTensor, pixel_pe: &GpuTensor, mask: &GpuTensor, tap: &mut Tap,
    ) -> Result<(GpuTensor, Feat)> {
        let n = pixel.n;
        let name = |part: &str| format!("net.object_transformer.blocks.{b}.{part}");
        let pixel_flat = to_tokens(pixel)?;
        // The block's input pixels plus their PE: read_from_pixel's k and read_from_query's q
        // (PyTorch adds them twice, to the same values).
        let pixel_with_pe = add_tokens(&pixel_flat, pixel_pe)?;

        // read_from_pixel (CrossAttention): the queries normalised; q = that + PE, k = pixels +
        // PE, v = pixels; the residual is the normalised queries.
        let xn = self.read_from_pixel_norm.forward(x)?;
        let q = add_tokens(&xn, query_pe)?;
        let x = add_tokens(&xn, &self.read_from_pixel.forward(&q, &pixel_with_pe, &pixel_flat, n, Some(mask))?)?;
        put_tokens(tap, &name("read_from_pixel.0"), &x, n)?;

        // self_attn (SelfAttention): normalised; q = k = that + PE, v = that; the residual is
        // the normalised queries.
        let xn = self.self_attn_norm.forward(&x)?;
        let qk = add_tokens(&xn, query_pe)?;
        let x = add_tokens(&xn, &self.self_attn.forward(&qk, &qk, &xn, n, None)?)?;
        put_tokens(tap, &name("self_attn"), &x, n)?;

        // ffn (FFN): the residual is the queries before the norm.
        let h = self.ffn1.forward(&self.ffn_norm.forward(&x)?)?;
        let h = self.ffn2.forward(&relu_tokens(&h)?)?;
        let x = add_tokens(&x, &h)?;
        put_tokens(tap, &name("ffn"), &x, n)?;

        // read_from_query (CrossAttention without a norm): q = pixels + PE, k = queries + PE, v =
        // queries; the residual is the pixels.
        let k = add_tokens(&x, query_pe)?;
        let pixel_flat = add_tokens(&pixel_flat, &self.read_from_query.forward(&pixel_with_pe, &k, &x, n, None)?)?;
        put_tokens(tap, &name("read_from_query.0"), &pixel_flat, n)?;

        // pixel_ffn (PixelFFN): the CAResBlock on read_from_query's pixels (the block's input
        // pixels are not used).
        let pixel = self.pixel_ffn.forward(&to_planar(&pixel_flat, n, pixel.w, pixel.h)?)?;
        tap.put(&name("pixel_ffn"), &pixel.t)?;
        put_tokens(tap, &name("0"), &x, n)?;
        tap.put(&name("1"), &pixel.t)?;
        Ok((x, pixel))
    }
}

/// `QueryTransformer`.
pub(crate) struct QueryTransformer {
    /// The learnt `query_init` and `query_emb` [16, 256], shared by the objects.
    query_init: GpuTensor,
    query_emb: GpuTensor,
    /// The same on the host, for the graph's constants.
    host_query_init: Vec<f32>,
    host_query_emb: Vec<f32>,
    summary_to_query_init: Linear,
    summary_to_query_emb: Linear,
    pixel_init_proj: Conv2d,
    pixel_emb_proj: Conv2d,
    /// The pixels' sine encoding.
    spatial_pe: PositionalEncoding,
    blocks: Vec<Block>,
    /// `mask_pred[i]`'s 1x1 conv (256 -> 1), after a ReLU; one before the blocks and one after
    /// each.
    mask_pred: Vec<Conv2d>,
}

impl QueryTransformer {
    pub(crate) fn load(w: &mut Weights) -> Result<Self> {
        let p = "object_transformer";
        let host_query_init = w.f32(&format!("{p}.query_init.weight"), &[QUERIES, EMBED])?;
        let host_query_emb = w.f32(&format!("{p}.query_emb.weight"), &[QUERIES, EMBED])?;
        let upload = |values: &[f32]| gpu_upload(values, QUERIES, EMBED).map_err(|e| dev("object transformer", e));
        Ok(Self {
            query_init: upload(&host_query_init)?,
            query_emb: upload(&host_query_emb)?,
            host_query_init,
            host_query_emb,
            summary_to_query_init: w.linear(&format!("{p}.summary_to_query_init"), EMBED, EMBED, true)?,
            summary_to_query_emb: w.linear(&format!("{p}.summary_to_query_emb"), EMBED, EMBED, true)?,
            pixel_init_proj: w.conv(&format!("{p}.pixel_init_proj"), EMBED, EMBED, 1, 0, 1, true)?,
            pixel_emb_proj: w.conv(&format!("{p}.pixel_emb_proj"), EMBED, EMBED, 1, 0, 1, true)?,
            spatial_pe: PositionalEncoding::load(w, &format!("{p}.spatial_pe"), EMBED)?,
            blocks: (0..BLOCKS).map(|b| Block::load(w, &format!("{p}.blocks.{b}"))).collect::<Result<_>>()?,
            mask_pred: (0..=BLOCKS).map(|i| w.conv(&format!("{p}.mask_pred.{i}.1"), EMBED, 1, 1, 0, 1, true)).collect::<Result<_>>()?,
        })
    }

    /// `CUTIE.readout_query` (see `CutieNet::readout_query`). `forced_masks`: the attention mask
    /// for each block instead of the one made from the aux logits (the check feeds the oracle's);
    /// the made one is still tapped.
    pub(crate) fn forward(&self, pixel: &Feat, obj_memory: &GpuTensor, forced_masks: Option<&[GpuTensor]>, tap: &mut Tap) -> Result<(Feat, Vec<GpuTensor>)> {
        tap.put("object_transformer.in.pixel", &pixel.t)?;
        tap.put("object_transformer.in.obj_memory", obj_memory)?;
        let n = pixel.n;
        if pixel.c != EMBED {
            return Err(DiffusionError::model(format!("cutie object transformer: pixel of {} channels", pixel.c)));
        }
        if let Some(masks) = forced_masks.filter(|m| m.len() != BLOCKS) {
            return Err(DiffusionError::model(format!("cutie object transformer: {} forced masks for {BLOCKS} blocks", masks.len())));
        }
        if forced_masks.is_none() && !tap.active() {
            if let Some(out) = self.forward_graph(pixel, obj_memory)? {
                return Ok(out);
            }
        }
        let p = "net.object_transformer";

        // The queries and their PE from the object memory.
        let values = object_values(obj_memory, n)?;
        let init = self.summary_to_query_init.forward(&values)?;
        put_tokens(tap, &format!("{p}.summary_to_query_init"), &init, n)?;
        let emb = self.summary_to_query_emb.forward(&values)?;
        put_tokens(tap, &format!("{p}.summary_to_query_emb"), &emb, n)?;
        let mut x = add_tokens(&per_object(&self.query_init, n)?, &init)?;
        let query_pe = add_tokens(&per_object(&self.query_emb, n)?, &emb)?;

        // The pixels and their PE: the sine encoding added to every object's pixel_emb while
        // planar, where it is one row added to each object's [256*h*w] (the same single add as
        // PyTorch's on tokens), then made tokens.
        let pixel_init = self.pixel_init_proj.forward(pixel)?;
        tap.put(&format!("{p}.pixel_init_proj"), &pixel_init.t)?;
        let pixel_emb = self.pixel_emb_proj.forward(pixel)?;
        tap.put(&format!("{p}.pixel_emb_proj"), &pixel_emb.t)?;
        if tap.active() {
            put_spatial_pe(tap, &format!("{p}.spatial_pe"), &self.spatial_pe.tokens(pixel.w, pixel.h)?, n)?;
        }
        let pixel_pe = to_tokens(&add_planar_pe(&pixel_emb, &self.spatial_pe.planar(pixel.w, pixel.h)?)?)?;

        // The blocks, each masked by the aux logits before it. (PyTorch also makes a mask from
        // the last aux logits; nothing reads it, so it is not made here.)
        let mut pixel = pixel_init;
        let mut aux = Vec::with_capacity(BLOCKS + 1);
        aux.push(self.aux_logits(0, &pixel, tap)?);
        for (b, block) in self.blocks.iter().enumerate() {
            let mask = gpu_cutie_attn_mask(&aux[b], HEADS, QUERIES).map_err(|e| dev("attention mask", e))?;
            tap.put(&format!("{p}.blocks.{b}.attn_mask"), &mask)?;
            let mask = forced_masks.map_or(&mask, |m| &m[b]);
            let (next_x, next_pixel) = block.forward(b, &x, &pixel, &query_pe, &pixel_pe, mask, tap)?;
            x = next_x;
            pixel = next_pixel;
            aux.push(self.aux_logits(b + 1, &pixel, tap)?);
        }
        tap.put("object_transformer.out.pixel", &pixel.t)?;
        for (i, logits) in aux.iter().enumerate() {
            tap.put(&format!("object_transformer.out.aux_logits.{i}"), logits)?;
        }
        Ok((pixel, aux))
    }

    /// In an fp16 step where the backend runs graphs (`GPU_CUTIE_GRAPH`, macOS): the whole
    /// transformer as one graph in half (`gpu_cutie_transformer_graph_f16`), the pixels and the
    /// aux logits out in fp32. MAKEPAD_CUTIE_GRAPH_TRANSFORMER=0 keeps it an op at a time. `None`
    /// when not used.
    fn forward_graph(&self, pixel: &Feat, obj_memory: &GpuTensor) -> Result<Option<(Feat, Vec<GpuTensor>)>> {
        use makepad_ai_common::gpu::{gpu_cutie_transformer_graph_f16, CutieGraphTransformer, GPU_CUTIE_GRAPH};
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled = *ON.get_or_init(|| std::env::var("MAKEPAD_CUTIE_GRAPH_TRANSFORMER").map_or(true, |v| v.trim() != "0"));
        if !GPU_CUTIE_GRAPH || !enabled || !crate::model::HalfConvs::on() {
            return Ok(None);
        }
        let graph = CutieGraphTransformer {
            key: &self.pixel_init_proj.key,
            embed: EMBED,
            heads: HEADS,
            queries: QUERIES,
            area_eps: AREA_EPS,
            query_init: &self.host_query_init,
            query_emb: &self.host_query_emb,
            summary_to_query_init: self.summary_to_query_init.graph_linear(),
            summary_to_query_emb: self.summary_to_query_emb.graph_linear(),
            pixel_init_proj: self.pixel_init_proj.graph_conv(),
            pixel_emb_proj: self.pixel_emb_proj.graph_conv(),
            blocks: self.blocks.iter().map(Block::graph).collect(),
            mask_pred: self.mask_pred.iter().map(Conv2d::graph_conv).collect(),
        };
        let pe = self.spatial_pe.planar(pixel.w, pixel.h)?;
        let (out, aux) = gpu_cutie_transformer_graph_f16(&graph, &pixel.t, pixel.n, pixel.w, pixel.h, obj_memory, &pe)
            .map_err(|e| dev("object transformer graph", e))?;
        Ok(Some((Feat::new(out, pixel.n, EMBED, pixel.w, pixel.h)?, aux)))
    }

    /// `mask_pred[i]` on the pixels: [n, h*w], one logit a pixel and object.
    fn aux_logits(&self, i: usize, pixel: &Feat, tap: &mut Tap) -> Result<GpuTensor> {
        let logits = self.mask_pred[i].forward(&relu(pixel)?)?;
        tap.put(&format!("net.object_transformer.mask_pred.{i}"), &logits.t)?;
        Ok(logits.t)
    }
}

/// Object-shared rows [len, 256] as interleaved tokens [len*n, 256], each row repeated per object
/// (n copies side by side are [len, n*256]); a view for one object.
fn per_object(rows: &GpuTensor, n: usize) -> Result<GpuTensor> {
    let fail = |e: String| dev("per object", e);
    let wide = if n == 1 { gpu_slice_rows(rows, 0, rows.rows()) } else { gpu_concat_cols(&vec![rows; n]) }.map_err(fail)?;
    gpu_reshape(wide, rows.rows() * n, EMBED).map_err(fail)
}

/// A planar batch (n images of 256 channels) plus the sine encoding (planar [256, h*w]) in each
/// image: read as [n, 256*h*w], the encoding is one row added to every row.
fn add_planar_pe(x: &Feat, pe: &GpuTensor) -> Result<Feat> {
    let fail = |e: String| dev("add pe", e);
    let rows = gpu_reshape(gpu_slice_rows(&x.t, 0, x.t.rows()).map_err(fail)?, x.n, x.c * x.plane()).map_err(fail)?;
    let sum = gpu_add_cols_broadcast(&rows, pe).map_err(fail)?;
    Feat::new(gpu_reshape(sum, x.n * x.c, x.plane()).map_err(fail)?, x.n, x.c, x.w, x.h)
}

/// Puts the sine encoding (tokens [h*w, 256]) as the oracle has it, one copy an object [n, h, w,
/// 256].
fn put_spatial_pe(tap: &mut Tap, name: &str, tokens: &GpuTensor, n: usize) -> Result<()> {
    tap.put(name, &gpu_concat_rows_many(&vec![tokens; n]).map_err(|e| dev("tap", e))?)
}

/// The queries' source: each summary's sums over its area, `obj_sums / (obj_area + 1e-4)`, from
/// `obj_memory` [n*16, 257] (object-major, the area last; in inference its time axis is already
/// summed) to interleaved [16*n, 256].
fn object_values(obj_memory: &GpuTensor, n: usize) -> Result<GpuTensor> {
    let width = EMBED + 1;
    if obj_memory.rows() != n * QUERIES || obj_memory.cols() != width {
        return Err(DiffusionError::model(format!(
            "cutie object transformer: object memory {}x{}, wants {}x{width}",
            obj_memory.rows(),
            obj_memory.cols(),
            n * QUERIES
        )));
    }
    gpu_cutie_object_values(obj_memory, n, AREA_EPS).map_err(|e| dev("object values", e))
}

// --- Token layouts ---

/// A planar batch (n images of 256 channels) as interleaved tokens [h*w*n, 256]: [n*256, h*w]
/// transposed is [h*w, n*256].
fn to_tokens(x: &Feat) -> Result<GpuTensor> {
    let fail = |e: String| dev("to tokens", e);
    gpu_reshape(gpu_planar_tokens_transpose(&x.t).map_err(fail)?, x.plane() * x.n, x.c).map_err(fail)
}

/// Interleaved tokens [h*w*n, 256] back to a planar batch.
fn to_planar(t: &GpuTensor, n: usize, w: usize, h: usize) -> Result<Feat> {
    let fail = |e: String| dev("to planar", e);
    let wide = gpu_reshape(gpu_slice_rows(t, 0, t.rows()).map_err(fail)?, w * h, n * EMBED).map_err(fail)?;
    Feat::new(gpu_planar_tokens_transpose(&wide).map_err(fail)?, n, EMBED, w, h)
}

fn add_tokens(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor> {
    gpu_add(a, b).map_err(|e| dev("add", e))
}

fn relu_tokens(x: &GpuTensor) -> Result<GpuTensor> {
    gpu_birefnet_relu(x).map_err(|e| dev("relu", e))
}

/// Puts interleaved tokens [len*n, 256] as the oracle has them, object-major [n, len, 256]
/// (a copy, made only while a check runs and n > 1).
fn put_tokens(tap: &mut Tap, name: &str, t: &GpuTensor, n: usize) -> Result<()> {
    if !tap.active() {
        return Ok(());
    }
    if n == 1 {
        return tap.put(name, t);
    }
    let fail = |e: String| dev("tap", e);
    let len = t.rows() / n;
    let wide = gpu_reshape(gpu_slice_rows(t, 0, t.rows()).map_err(fail)?, len, n * EMBED).map_err(fail)?;
    let objects = (0..n).map(|o| gpu_slice_cols(&wide, o * EMBED, EMBED)).collect::<std::result::Result<Vec<_>, _>>().map_err(fail)?;
    let parts: Vec<&GpuTensor> = objects.iter().collect();
    tap.put(name, &gpu_concat_rows_many(&parts).map_err(fail)?)
}

impl CutieNet {
    /// `CUTIE.pixel_fusion`: `pix_feat` (one image, 256 at stride 16), `visual_readout` (n, 256
    /// at stride 16), `sensory` (n, 256), `last_mask` (n images of 1 channel at the padded H, W:
    /// each object's probability last frame). Out (n, 256 at stride 16). The n objects are the
    /// whole set each object's others are taken over: one bucket, in one chunk (PyTorch's
    /// `MemoryManager.read` passes each `chunk_size` chunk alone; the port always passes every
    /// object, as `chunk_size` -1 does).
    ///
    /// Taps: `pixel_fusion.in.visual_readout`, `.in.sensory`, `.in.last_mask`,
    /// `net.pixel_fuser.sensory_compress`, `net.pixel_fuser.fuser.distributor`, `.block1`,
    /// `.block2`, `pixel_fusion.out`.
    pub(crate) fn pixel_fusion(&self, pix_feat: &Feat, visual_readout: &Feat, sensory: &Feat, last_mask: &Feat, tap: &mut Tap) -> Result<Feat> {
        self.pixel_fuser.forward(pix_feat, visual_readout, sensory, last_mask, tap)
    }

    /// `CUTIE.readout_query`: the object transformer on `pixel` (n, 256 at stride 16) with
    /// `obj_memory` [n*16, 257] (each object's summed summaries, then the area). Out: the pixels
    /// (n, 256 at stride 16) and the aux logits of each prediction ([n, h*w], 4 of them). The n
    /// objects are the whole set the attention masks' aggregate is taken over, as for
    /// `pixel_fusion`: one bucket, in one chunk.
    ///
    /// Taps: `object_transformer.in.pixel`, `.in.obj_memory`,
    /// `net.object_transformer.summary_to_query_init`, `_emb`, `.pixel_init_proj`,
    /// `.pixel_emb_proj`, `.spatial_pe` ([n, h, w, 256]), `.mask_pred.{i}`, and per block b
    /// `.blocks.{b}.attn_mask` (bool [n*8, 16, h*w], 1 = blocked; put additive, -inf blocked),
    /// `.blocks.{b}.read_from_pixel.0`, `.self_attn`, `.ffn`, `.read_from_query.0` ([n, h*w,
    /// 256], token-major), `.pixel_ffn`, `.blocks.{b}.0` (queries), `.blocks.{b}.1` (pixels);
    /// `object_transformer.out.pixel`, `.out.aux_logits.{i}`.
    pub(crate) fn readout_query(&self, pixel: &Feat, obj_memory: &GpuTensor, tap: &mut Tap) -> Result<(Feat, Vec<GpuTensor>)> {
        self.object_transformer.forward(pixel, obj_memory, None, tap)
    }
}

// --- The check ---

/// Stage `transformer`: for every oracle step that segments (`prop_f1`, `prop_f5`, `prop_f40`,
/// `prop_f41`), the oracle's `pixel_fusion.in.*` (and `encode_image.out.pix_feat`) through
/// `pixel_fusion` (rows `fusion`), and `object_transformer.in.*` through `readout_query` (rows
/// `transformer`), every tap compared. The attention masks are compared exactly; the transformer
/// runs again with the oracle's own mask fed to each block (rows `oracle-mask`), its native
/// masks still compared, which separates a flipped mask bit (a near-tie in the aux logits) from
/// a wrong block; and once with a mask row that would block every pixel (row `edge`, under
/// `transformer`). The `*.in.*` rows are the fed inputs put back: they check layout and shape,
/// not the port.
///
/// Every step that segments (any with a `segment.*`, `pixel_fusion.*` or `object_transformer.*`
/// tap) must have the transformer's inputs, every oracle tap under the stage's names must be
/// compared and every tap put must be in the oracle (`Comparer::check_coverage`), and some step
/// must have been checked: a renamed tap on either side fails instead of passing with fewer
/// comparisons.
pub(crate) fn check(oracle: &Oracle, report: &mut Report, weights: &mut Weights) -> Result<()> {
    const FUSION: [&str; 2] = ["pixel_fusion.", "net.pixel_fuser."];
    const TRANSFORMER: [&str; 2] = ["object_transformer.", "net.object_transformer."];
    let fuser = PixelFuser::load(weights)?;
    let transformer = QueryTransformer::load(weights)?;
    let n = oracle.objects;
    let tol = |_: &str| Tol::STAGE;
    let mut checked = 0;
    for step in oracle.steps() {
        let segments = oracle.names(step).iter().any(|name| ["segment.", FUSION[0], FUSION[1], TRANSFORMER[0], TRANSFORMER[1]].iter().any(|p| name.starts_with(p)));
        if !segments {
            continue;
        }
        if !oracle.has(step, "object_transformer.in.pixel") {
            report.fail("transformer", step, "object_transformer.in.pixel", "not in the oracle, which segments here");
            continue;
        }
        checked += 1;
        let pix_feat = oracle.feat_as(step, "encode_image.out.pix_feat", 1, EMBED)?;
        let visual_readout = oracle.feat_as(step, "pixel_fusion.in.visual_readout", n, EMBED)?;
        let sensory = oracle.feat_as(step, "pixel_fusion.in.sensory", n, EMBED)?;
        let last_mask = oracle.feat_as(step, "pixel_fusion.in.last_mask", n, 1)?;
        let mut cmp = Comparer::new(oracle, report, "fusion", step, &tol);
        cmp.with(|tap| fuser.forward(&pix_feat, &visual_readout, &sensory, &last_mask, tap))?;
        cmp.check_coverage(&FUSION, &[]);

        let pixel = oracle.feat_as(step, "object_transformer.in.pixel", n, EMBED)?;
        let memory = oracle.tap(step, "object_transformer.in.obj_memory")?;
        let obj_memory = gpu_upload(&memory.values, n * QUERIES, EMBED + 1).map_err(|e| dev("check", e))?;
        let mut cmp = Comparer::new(oracle, report, "transformer", step, &tol);
        cmp.with(|tap| transformer.forward(&pixel, &obj_memory, None, tap))?;
        cmp.check_coverage(&TRANSFORMER, &[]);

        let masks = (0..BLOCKS).map(|b| oracle_mask(oracle, step, b)).collect::<Result<Vec<_>>>()?;
        let mut cmp = Comparer::new(oracle, report, "oracle-mask", step, &tol);
        cmp.with(|tap| transformer.forward(&pixel, &obj_memory, Some(&masks), tap))?;
        cmp.check_coverage(&TRANSFORMER, &[]);

        // On the first step checked (prop_f1 in every oracle so far).
        if checked == 1 {
            check_open_rows(report, &transformer, &pixel, &obj_memory)?;
        }
    }
    if checked == 0 {
        report.fail("transformer", "-", "(stage)", "no oracle step reads memory through the object transformer");
    }
    Ok(())
}

/// What the oracle never reaches: a mask row that would block every pixel (an object with no
/// foreground blocks its foreground queries everywhere; one that is foreground everywhere, its
/// background queries) sees every pixel instead (`_get_aux_mask`'s last line), so no softmax row
/// is all -inf. Aux logits with object 0 foreground everywhere and the others nowhere must give
/// a mask open everywhere, and the transformer run with it finite outputs (row `edge`).
fn check_open_rows(report: &mut Report, transformer: &QueryTransformer, pixel: &Feat, obj_memory: &GpuTensor) -> Result<()> {
    let (n, p) = (pixel.n, pixel.plane());
    let fail = |e: String| dev("check", e);
    let logits: Vec<f32> = (0..n * p).map(|i| if i < p { 10.0 } else { -10.0 }).collect();
    let mask = gpu_cutie_attn_mask(&gpu_upload(&logits, n, p).map_err(fail)?, HEADS, QUERIES).map_err(fail)?;
    let expected = OracleTap { shape: vec![n * HEADS, QUERIES, p], dtype: "bool".into(), values: vec![0.0; n * HEADS * QUERIES * p] };
    report.compare("transformer", "edge", "attn_mask with every row open", &gpu_download(&mask).map_err(fail)?, &expected, Tol::STAGE);
    let masks: Vec<GpuTensor> = (0..BLOCKS).map(|_| gpu_slice_rows(&mask, 0, mask.rows())).collect::<std::result::Result<_, _>>().map_err(fail)?;
    let (out, _) = transformer.forward(pixel, obj_memory, Some(&masks), &mut Tap::none())?;
    let nonfinite = out.download()?.iter().filter(|v| !v.is_finite()).count();
    if nonfinite > 0 {
        report.fail("transformer", "edge", "out.pixel with every row open", format!("{nonfinite} non-finite values"));
    } else {
        report.note("transformer/edge: out.pixel is finite with every mask row open");
    }
    Ok(())
}

/// The oracle's attention mask for block `b` (bool [n*8, 16, h*w], 1 = blocked) as the additive
/// mask the attention takes.
fn oracle_mask(oracle: &Oracle, step: &str, b: usize) -> Result<GpuTensor> {
    let tap = oracle.tap(step, &format!("net.object_transformer.blocks.{b}.attn_mask"))?;
    let cols = tap.shape.last().copied().unwrap_or(0);
    let values: Vec<f32> = tap.values.iter().map(|&v| if v != 0.0 { f32::NEG_INFINITY } else { 0.0 }).collect();
    gpu_upload(&values, values.len() / cols.max(1), cols).map_err(|e| dev("check", e))
}
