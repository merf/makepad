//! Cutie's object transformer as one MPSGraph (an fp16 step's, macOS): the queries from the
//! object memory, the pixel projections and their PE, the aux logits and each block's attention
//! mask, and the three blocks (`read_from_pixel`, `self_attn`, `ffn`, `read_from_query`,
//! `pixel_ffn`), in one graph instead of about 150 dispatches.
//!
//! Precision as PyTorch's autocast runs it: the linears, convolutions and attention matmuls in
//! half; the LayerNorms, the softmax, the mask and the residual streams in fp32 (the queries'
//! stream is fp32 under autocast too, LayerNorm's output; the pixels' is half there, fp32 here).
//! The mask compares the clamped probabilities rather than their logits (the logit is
//! increasing), and blocks with -1e30 rather than -inf (no row is ever all blocked).
//!
//! In the graph each object is a batch entry: queries [n, 16, 256], pixel tokens [n, h*w, 256],
//! pixels planar [n, 256, h, w]. The types compile on every OS; the op runs on macOS only.

use super::graph::CutieGraphConv;
use crate::stream::MetalResult;

/// nn.Linear: PyTorch's [out][in] weight and the bias.
#[derive(Clone, Copy)]
pub struct CutieGraphLinear<'a> {
    pub weight: &'a [f32],
    pub bias: Option<&'a [f32]>,
    pub in_f: usize,
    pub out_f: usize,
}

/// nn.LayerNorm over the last dimension.
#[derive(Clone, Copy)]
pub struct CutieGraphNorm<'a> {
    pub gamma: &'a [f32],
    pub beta: &'a [f32],
    pub eps: f32,
}

/// nn.MultiheadAttention's projections (its in-projection split into q, k and v).
#[derive(Clone, Copy)]
pub struct CutieGraphAttention<'a> {
    pub q: CutieGraphLinear<'a>,
    pub k: CutieGraphLinear<'a>,
    pub v: CutieGraphLinear<'a>,
    pub out: CutieGraphLinear<'a>,
}

/// A `QueryTransformerBlock`; `pixel_ffn` is the CAResBlock's two 3x3 convolutions and its ECA
/// taps.
#[derive(Clone, Copy)]
pub struct CutieGraphBlock<'a> {
    pub read_from_pixel: CutieGraphAttention<'a>,
    pub read_from_pixel_norm: CutieGraphNorm<'a>,
    pub self_attn: CutieGraphAttention<'a>,
    pub self_attn_norm: CutieGraphNorm<'a>,
    pub ffn_norm: CutieGraphNorm<'a>,
    pub ffn1: CutieGraphLinear<'a>,
    pub ffn2: CutieGraphLinear<'a>,
    pub read_from_query: CutieGraphAttention<'a>,
    pub pixel_conv1: CutieGraphConv<'a>,
    pub pixel_conv2: CutieGraphConv<'a>,
    pub eca: &'a [f32],
}

/// The `QueryTransformer`'s weights and sizes. `key` names its graph (cached under it, the
/// objects and the size).
pub struct CutieGraphTransformer<'a> {
    pub key: &'a str,
    pub embed: usize,
    pub heads: usize,
    pub queries: usize,
    pub area_eps: f32,
    /// The learnt `query_init` and `query_emb`, [queries][embed].
    pub query_init: &'a [f32],
    pub query_emb: &'a [f32],
    pub summary_to_query_init: CutieGraphLinear<'a>,
    pub summary_to_query_emb: CutieGraphLinear<'a>,
    pub pixel_init_proj: CutieGraphConv<'a>,
    pub pixel_emb_proj: CutieGraphConv<'a>,
    pub blocks: Vec<CutieGraphBlock<'a>>,
    /// One more than the blocks: before the first and after each.
    pub mask_pred: Vec<CutieGraphConv<'a>>,
}

fn check_linear(what: &str, l: &CutieGraphLinear<'_>, in_f: usize, out_f: usize) -> MetalResult<()> {
    if l.in_f != in_f || l.out_f != out_f || l.weight.len() != in_f * out_f || l.bias.is_some_and(|b| b.len() != out_f) {
        return Err(format!("cutie transformer graph: {what} is {}x{} with {} weights, wants {out_f}x{in_f}", l.out_f, l.in_f, l.weight.len()));
    }
    Ok(())
}

fn check_conv(what: &str, c: &CutieGraphConv<'_>, in_c: usize, out_c: usize, k: usize) -> MetalResult<()> {
    if c.in_channels != in_c || c.out_channels != out_c || c.k != k || c.pad != k / 2 || c.stride != 1 || c.weights.len() != out_c * in_c * k * k
        || c.bias.is_some_and(|b| b.len() != out_c)
    {
        return Err(format!("cutie transformer graph: {what} ({}) is not a {in_c} to {out_c} {k}x{k} convolution", c.key));
    }
    Ok(())
}

fn check_norm(what: &str, n: &CutieGraphNorm<'_>, dim: usize) -> MetalResult<()> {
    if n.gamma.len() != dim || n.beta.len() != dim {
        return Err(format!("cutie transformer graph: {what} has {} and {} affines for {dim}", n.gamma.len(), n.beta.len()));
    }
    Ok(())
}

impl CutieGraphTransformer<'_> {
    /// `Err` naming the first weight whose size is not the transformer's.
    pub fn check(&self) -> MetalResult<()> {
        let (e, q) = (self.embed, self.queries);
        if self.heads == 0 || e % self.heads != 0 || q % 2 != 0 || self.query_init.len() != q * e || self.query_emb.len() != q * e {
            return Err(format!("cutie transformer graph: embed {e}, {} heads, {q} queries", self.heads));
        }
        if self.mask_pred.len() != self.blocks.len() + 1 {
            return Err(format!("cutie transformer graph: {} mask predictions for {} blocks", self.mask_pred.len(), self.blocks.len()));
        }
        check_linear("summary_to_query_init", &self.summary_to_query_init, e, e)?;
        check_linear("summary_to_query_emb", &self.summary_to_query_emb, e, e)?;
        check_conv("pixel_init_proj", &self.pixel_init_proj, e, e, 1)?;
        check_conv("pixel_emb_proj", &self.pixel_emb_proj, e, e, 1)?;
        for m in &self.mask_pred {
            check_conv("mask_pred", m, e, 1, 1)?;
        }
        for b in &self.blocks {
            for (what, a) in [("read_from_pixel", &b.read_from_pixel), ("self_attn", &b.self_attn), ("read_from_query", &b.read_from_query)] {
                for l in [&a.q, &a.k, &a.v, &a.out] {
                    check_linear(what, l, e, e)?;
                }
            }
            check_norm("read_from_pixel.norm", &b.read_from_pixel_norm, e)?;
            check_norm("self_attn.norm", &b.self_attn_norm, e)?;
            check_norm("ffn.norm", &b.ffn_norm, e)?;
            check_linear("ffn.linear1", &b.ffn1, e, b.ffn1.out_f)?;
            check_linear("ffn.linear2", &b.ffn2, b.ffn1.out_f, e)?;
            check_conv("pixel_ffn.conv1", &b.pixel_conv1, e, e, 3)?;
            check_conv("pixel_ffn.conv2", &b.pixel_conv2, e, e, 3)?;
            if b.eca.len() % 2 == 0 || b.eca.len() > e {
                return Err(format!("cutie transformer graph: {} ECA taps", b.eca.len()));
            }
        }
        Ok(())
    }
}

/// The object transformer on `pixel` ([n*embed, h*w] planar, n = `objects`) with `obj_memory`
/// ([n*queries, embed + 1], each object's summed summaries then the area) and the pixels' sine
/// encoding `pe` ([embed, h*w] planar), as one graph in half: the pixels out ([n*embed, h*w]) and
/// the aux logits of each prediction ([n, h*w], one more than the blocks), in fp32.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_transformer_graph_f16(
    t: &CutieGraphTransformer<'_>, pixel: &crate::gpu_types::GpuTensor, objects: usize, width: usize, height: usize,
    obj_memory: &crate::gpu_types::GpuTensor, pe: &crate::gpu_types::GpuTensor,
) -> MetalResult<(crate::gpu_types::GpuTensor, Vec<crate::gpu_types::GpuTensor>)> {
    use crate::gpu_types::GpuTensor;
    use crate::stream::{with_metal_stream, MpsTransformerRun};
    let op = "gpu_cutie_transformer_graph_f16";
    t.check()?;
    let (e, plane) = (t.embed, width * height);
    if objects == 0 || pixel.rows() != objects * e || pixel.cols() != plane {
        return Err(format!("{op}: pixel {}x{} for {objects} objects of {e} at {width}x{height}", pixel.rows(), pixel.cols()));
    }
    if obj_memory.rows() != objects * t.queries || obj_memory.cols() != e + 1 {
        return Err(format!("{op}: object memory {}x{}, wants {}x{}", obj_memory.rows(), obj_memory.cols(), objects * t.queries, e + 1));
    }
    if pe.rows() != e || pe.cols() != plane {
        return Err(format!("{op}: encoding {}x{}, wants {e}x{plane}", pe.rows(), pe.cols()));
    }
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(objects * e, plane)?;
        let aux: Vec<GpuTensor> = (0..t.mask_pred.len()).map(|_| GpuTensor::alloc(objects, plane)).collect::<MetalResult<_>>()?;
        let mut outs: Vec<&GpuTensor> = vec![&out];
        outs.extend(aux.iter());
        s.transformer_mpsgraph(&MpsTransformerRun {
            t,
            objects,
            width,
            height,
            inputs: &[pixel, obj_memory, pe],
            outs: &outs,
            copy: &super::matmul_conv::GRAPH_TO_PLANAR,
        })?;
        Ok((out, aux))
    })
}
