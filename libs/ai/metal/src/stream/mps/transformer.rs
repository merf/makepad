//! Cutie's object transformer built as one MPSGraph (`cutie_ops::transformer_graph`, which says
//! what runs in half and what in fp32). Inputs and outputs are fp32 buffers of the cache's,
//! NCHW from their start: the pixels [n, embed, h, w], the object memory [n, queries, embed + 1]
//! and the sine encoding [1, embed, h, w] in; the pixels and each aux logits [n, 1, h, w] out.

use super::{graph_shape, GraphConv, MpsCache, GRAPH_LAYOUT_NCHW, GRAPH_LAYOUT_OIHW, GRAPH_PADDING_EXPLICIT, MPS_DATA_TYPE_FLOAT16, MPS_DATA_TYPE_FLOAT32};
use crate::cutie_ops::graph::CutieGraphConv;
use crate::cutie_ops::transformer_graph::{CutieGraphAttention, CutieGraphBlock, CutieGraphLinear, CutieGraphNorm, CutieGraphTransformer};
use crate::objc_util::StrongId;
use crate::stream::MetalResult;
use makepad_objc_sys::runtime::ObjcId;
use makepad_objc_sys::{class, msg_send, sel, sel_impl};

const F16: u32 = MPS_DATA_TYPE_FLOAT16;
const F32: u32 = MPS_DATA_TYPE_FLOAT32;
/// What a blocked attention entry adds (in fp32): exp of it is 0, as -inf's is, without the NaN
/// a 0 times -inf would make.
const BLOCKED: f64 = -1e30;
/// The aggregate's clamp (`aggregate`'s, before the logit).
const PROB_EPS: f64 = 1e-7;

/// The cache's buffer roles for the transformer: inputs from 20, outputs from 30.
const ROLE_IN: u8 = 20;
const ROLE_OUT: u8 = 30;

/// The graph being built, its tensors autoreleased (the graph keeps them).
struct G(ObjcId);

fn nil() -> ObjcId {
    std::ptr::null_mut()
}

impl G {
    unsafe fn constant(&self, data: &[f32], dims: &[usize]) -> ObjcId {
        let bytes: ObjcId = msg_send![class!(NSData), dataWithBytes: data.as_ptr() as *const std::ffi::c_void length: (data.len() * 4) as u64];
        msg_send![self.0, constantWithData: bytes shape: graph_shape(dims) dataType: F32]
    }
    unsafe fn scalar(&self, v: f64, ty: u32) -> ObjcId {
        msg_send![self.0, constantWithScalar: v dataType: ty]
    }
    unsafe fn cast(&self, x: ObjcId, ty: u32) -> ObjcId {
        msg_send![self.0, castTensor: x toType: ty name: nil()]
    }
    unsafe fn half(&self, x: ObjcId) -> ObjcId {
        self.cast(x, F16)
    }
    unsafe fn float(&self, x: ObjcId) -> ObjcId {
        self.cast(x, F32)
    }
    unsafe fn add(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, additionWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    unsafe fn sub(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, subtractionWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    unsafe fn mul(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, multiplicationWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    unsafe fn div(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, divisionWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    unsafe fn maximum(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, maximumWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    unsafe fn relu(&self, x: ObjcId) -> ObjcId {
        msg_send![self.0, reLUWithTensor: x name: nil()]
    }
    unsafe fn sigmoid(&self, x: ObjcId) -> ObjcId {
        msg_send![self.0, sigmoidWithTensor: x name: nil()]
    }
    unsafe fn reshape(&self, x: ObjcId, dims: &[usize]) -> ObjcId {
        msg_send![self.0, reshapeTensor: x withShape: graph_shape(dims) name: nil()]
    }
    unsafe fn transpose(&self, x: ObjcId, a: usize, b: usize) -> ObjcId {
        msg_send![self.0, transposeTensor: x dimension: a as u64 withDimension: b as u64 name: nil()]
    }
    unsafe fn matmul(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, matrixMultiplicationWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    unsafe fn mean(&self, x: ObjcId, axes: &[usize]) -> ObjcId {
        msg_send![self.0, meanOfTensor: x axes: graph_shape(axes) name: nil()]
    }
    unsafe fn softmax(&self, x: ObjcId, axis: usize) -> ObjcId {
        msg_send![self.0, softMaxWithTensor: x axis: axis as i64 name: nil()]
    }
    unsafe fn max_along(&self, x: ObjcId, axis: usize) -> ObjcId {
        msg_send![self.0, reductionMaximumWithTensor: x axis: axis as i64 name: nil()]
    }
    unsafe fn prod_along(&self, x: ObjcId, axis: usize) -> ObjcId {
        msg_send![self.0, reductionProductWithTensor: x axis: axis as i64 name: nil()]
    }
    unsafe fn clamp(&self, x: ObjcId, lo: f64, hi: f64) -> ObjcId {
        let (lo, hi) = (self.scalar(lo, F32), self.scalar(hi, F32));
        msg_send![self.0, clampWithTensor: x minValueTensor: lo maxValueTensor: hi name: nil()]
    }
    unsafe fn ge(&self, a: ObjcId, b: ObjcId) -> ObjcId {
        msg_send![self.0, greaterThanOrEqualToWithPrimaryTensor: a secondaryTensor: b name: nil()]
    }
    unsafe fn broadcast(&self, x: ObjcId, dims: &[usize]) -> ObjcId {
        msg_send![self.0, broadcastTensor: x toShape: graph_shape(dims) name: nil()]
    }
    unsafe fn concat(&self, a: ObjcId, b: ObjcId, dim: usize) -> ObjcId {
        msg_send![self.0, concatTensor: a withTensor: b dimension: dim as i64 name: nil()]
    }
    unsafe fn slice(&self, x: ObjcId, dim: usize, start: usize, len: usize) -> ObjcId {
        msg_send![self.0, sliceTensor: x dimension: dim as u64 start: start as i64 length: len as i64 name: nil()]
    }

    /// A same-size convolution on a half NCHW tensor, its bias added and the ReLU taken in half.
    unsafe fn conv(&self, x: ObjcId, c: &CutieGraphConv<'_>, relu: bool) -> MetalResult<ObjcId> {
        let (o, i, k, pad) = (c.out_channels, c.in_channels, c.k, c.pad as u64);
        let w = self.half(self.constant(c.weights, &[o, i, k, k]));
        let desc: ObjcId = msg_send![
            class!(MPSGraphConvolution2DOpDescriptor),
            descriptorWithStrideInX: c.stride as u64
            strideInY: c.stride as u64
            dilationRateInX: 1u64
            dilationRateInY: 1u64
            groups: 1u64
            paddingLeft: pad
            paddingRight: pad
            paddingTop: pad
            paddingBottom: pad
            paddingStyle: GRAPH_PADDING_EXPLICIT
            dataLayout: GRAPH_LAYOUT_NCHW
            weightsLayout: GRAPH_LAYOUT_OIHW
        ];
        if desc.is_null() {
            return Err(format!("MPSGraph transformer: the descriptor for {} returned nil", c.key));
        }
        let mut y: ObjcId = msg_send![self.0, convolution2DWithSourceTensor: x weightsTensor: w descriptor: desc name: nil()];
        if let Some(bias) = c.bias {
            y = self.add(y, self.half(self.constant(bias, &[1, o, 1, 1])));
        }
        Ok(if relu { self.relu(y) } else { y })
    }

    /// nn.Linear on half rows [batch, len, in]: half [batch, len, out].
    unsafe fn linear(&self, x: ObjcId, l: &CutieGraphLinear<'_>, batch: usize, len: usize) -> ObjcId {
        let w = self.transpose(self.half(self.constant(l.weight, &[l.out_f, l.in_f])), 0, 1);
        let mut y = self.matmul(self.reshape(x, &[batch * len, l.in_f]), w);
        if let Some(bias) = l.bias {
            y = self.add(y, self.half(self.constant(bias, &[1, l.out_f])));
        }
        self.reshape(y, &[batch, len, l.out_f])
    }

    /// nn.LayerNorm over the last axis (`axis`) of an fp32 tensor, in fp32.
    unsafe fn layer_norm(&self, x: ObjcId, n: &CutieGraphNorm<'_>, axis: usize) -> ObjcId {
        let mean = self.mean(x, &[axis]);
        let var: ObjcId = msg_send![self.0, varianceOfTensor: x meanTensor: mean axes: graph_shape(&[axis]) name: nil()];
        let inv: ObjcId = msg_send![self.0, reverseSquareRootWithTensor: self.add(var, self.scalar(n.eps as f64, F32)) name: nil()];
        let y = self.mul(self.sub(x, mean), inv);
        self.add(self.mul(y, self.constant(n.gamma, &[n.gamma.len()])), self.constant(n.beta, &[n.beta.len()]))
    }

    /// Multi-head attention of fp32 `query` [n, lq, e] to `key` and `value` [n, lk, e], `mask`
    /// additive [n, 1, lq, lk] or none: fp32 [n, lq, e]. Projections and matmuls in half, the
    /// softmax in fp32.
    #[allow(clippy::too_many_arguments)]
    unsafe fn attention(
        &self, a: &CutieGraphAttention<'_>, query: ObjcId, key: ObjcId, value: ObjcId, n: usize, lq: usize, lk: usize, heads: usize, e: usize,
        mask: Option<ObjcId>,
    ) -> ObjcId {
        let d = e / heads;
        let split = |x: ObjcId, len: usize| self.transpose(self.reshape(x, &[n, len, heads, d]), 1, 2);
        let q = split(self.linear(self.half(query), &a.q, n, lq), lq);
        let k = split(self.linear(self.half(key), &a.k, n, lk), lk);
        let v = split(self.linear(self.half(value), &a.v, n, lk), lk);
        let q = self.mul(q, self.scalar(1.0 / (d as f64).sqrt(), F16));
        let mut scores = self.float(self.matmul(q, self.transpose(k, 2, 3)));
        if let Some(mask) = mask {
            scores = self.add(scores, mask);
        }
        let probs = self.half(self.softmax(scores, 3));
        let heads_out = self.reshape(self.transpose(self.matmul(probs, v), 1, 2), &[n, lq, e]);
        self.float(self.linear(heads_out, &a.out, n, lq))
    }

    /// Planar [n, e, h, w] as tokens [n, h*w, e], and back.
    unsafe fn tokens(&self, x: ObjcId, n: usize, e: usize, plane: usize) -> ObjcId {
        self.transpose(self.reshape(x, &[n, e, plane]), 1, 2)
    }
    unsafe fn planar(&self, x: ObjcId, n: usize, e: usize, w: usize, h: usize) -> ObjcId {
        self.reshape(self.transpose(x, 1, 2), &[n, e, h, w])
    }

    /// `mask_pred[i]` on fp32 pixels: fp32 logits [n, 1, h, w].
    unsafe fn aux(&self, pixel: ObjcId, c: &CutieGraphConv<'_>) -> MetalResult<ObjcId> {
        Ok(self.float(self.conv(self.half(self.relu(pixel)), c, false)?))
    }

    /// `gpu_cutie_attn_mask` on fp32 logits [n, 1, h, w]: additive [n, 1, queries, h*w], the first
    /// half of the queries foreground's, each row that would block every pixel open.
    unsafe fn attn_mask(&self, logits: ObjcId, n: usize, plane: usize, queries: usize) -> ObjcId {
        let prob = self.sigmoid(self.reshape(logits, &[n, plane]));
        let one = self.scalar(1.0, F32);
        let bg = self.prod_along(self.sub(one, prob), 0);
        let (prob, bg) = (self.clamp(prob, PROB_EPS, 1.0 - PROB_EPS), self.clamp(bg, PROB_EPS, 1.0 - PROB_EPS));
        let top = self.maximum(bg, self.max_along(prob, 0));
        let fg = self.float(self.ge(prob, top));
        let not_fg = self.sub(one, fg);
        let (any_fg, any_bg) = (self.max_along(fg, 1), self.max_along(not_fg, 1));
        let blocked_fg = self.reshape(self.mul(not_fg, any_fg), &[n, 1, 1, plane]);
        let blocked_bg = self.reshape(self.mul(fg, any_bg), &[n, 1, 1, plane]);
        let blocked = self.broadcast(self.concat(blocked_fg, blocked_bg, 1), &[n, 2, queries / 2, plane]);
        self.mul(self.reshape(blocked, &[n, 1, queries, plane]), self.scalar(BLOCKED, F32))
    }

    /// The CAResBlock on fp32 pixels [n, e, h, w]: conv2(relu(conv1(relu(x)))) scaled per channel
    /// by ECA (the plane's mean, the taps across the channels as one banded matrix, the sigmoid),
    /// plus x, in fp32.
    unsafe fn ca_res_block(&self, x: ObjcId, b: &CutieGraphBlock<'_>, n: usize, e: usize) -> MetalResult<ObjcId> {
        let h = self.conv(self.half(self.relu(x)), &b.pixel_conv1, true)?;
        let h = self.float(self.conv(h, &b.pixel_conv2, false)?);
        let means = self.reshape(self.mean(h, &[2, 3]), &[n, e]);
        // y[ch] = sum_j taps[j] * means[ch + j - pad]: means [n, e] times band [e (src), e (ch)].
        let (k, pad) = (b.eca.len(), b.eca.len() / 2);
        let mut band = vec![0.0f32; e * e];
        for ch in 0..e {
            for (j, &tap) in b.eca.iter().enumerate().take(k) {
                let src = ch as isize + j as isize - pad as isize;
                if src >= 0 && (src as usize) < e {
                    band[src as usize * e + ch] = tap;
                }
            }
        }
        let scale = self.sigmoid(self.matmul(means, self.constant(&band, &[e, e])));
        Ok(self.add(self.mul(h, self.reshape(scale, &[n, e, 1, 1])), x))
    }

    /// A `QueryTransformerBlock` on the fp32 queries [n, queries, e] and pixels [n, e, h, w]:
    /// the new queries and pixels.
    #[allow(clippy::too_many_arguments)]
    unsafe fn block(
        &self, b: &CutieGraphBlock<'_>, t: &CutieGraphTransformer<'_>, x: ObjcId, pixel: ObjcId, query_pe: ObjcId, pixel_pe: ObjcId,
        mask: ObjcId, n: usize, w: usize, h: usize,
    ) -> MetalResult<(ObjcId, ObjcId)> {
        let (e, q, heads, plane) = (t.embed, t.queries, t.heads, w * h);
        let pixel_flat = self.tokens(pixel, n, e, plane);
        let pixel_with_pe = self.add(pixel_flat, pixel_pe);

        let xn = self.layer_norm(x, &b.read_from_pixel_norm, 2);
        let qx = self.add(xn, query_pe);
        let x = self.add(xn, self.attention(&b.read_from_pixel, qx, pixel_with_pe, pixel_flat, n, q, plane, heads, e, Some(mask)));

        let xn = self.layer_norm(x, &b.self_attn_norm, 2);
        let qk = self.add(xn, query_pe);
        let x = self.add(xn, self.attention(&b.self_attn, qk, qk, xn, n, q, q, heads, e, None));

        let hidden = self.relu(self.linear(self.half(self.layer_norm(x, &b.ffn_norm, 2)), &b.ffn1, n, q));
        let x = self.add(x, self.float(self.linear(hidden, &b.ffn2, n, q)));

        let k = self.add(x, query_pe);
        let pixel_flat = self.add(pixel_flat, self.attention(&b.read_from_query, pixel_with_pe, k, x, n, plane, q, heads, e, None));
        let pixel = self.ca_res_block(self.planar(pixel_flat, n, e, w, h), b, n, e)?;
        Ok((x, pixel))
    }
}

/// The transformer's graph: its three input placeholders and its outputs (the pixels, then each
/// aux logits).
unsafe fn build(g: &G, t: &CutieGraphTransformer<'_>, n: usize, w: usize, h: usize) -> MetalResult<([ObjcId; 3], Vec<ObjcId>)> {
    let (e, q, plane) = (t.embed, t.queries, w * h);
    let pixel_in: ObjcId = msg_send![g.0, placeholderWithShape: graph_shape(&[n, e, h, w]) dataType: F32 name: nil()];
    let memory_in: ObjcId = msg_send![g.0, placeholderWithShape: graph_shape(&[n, q, e + 1]) dataType: F32 name: nil()];
    let pe_in: ObjcId = msg_send![g.0, placeholderWithShape: graph_shape(&[1, e, h, w]) dataType: F32 name: nil()];

    // The queries and their PE: sums / (area + eps) through the summary linears, plus the learnt
    // init and emb.
    let area = g.add(g.slice(memory_in, 2, e, 1), g.scalar(t.area_eps as f64, F32));
    let values = g.half(g.div(g.slice(memory_in, 2, 0, e), area));
    let init = g.float(g.linear(values, &t.summary_to_query_init, n, q));
    let emb = g.float(g.linear(values, &t.summary_to_query_emb, n, q));
    let mut x = g.add(init, g.constant(t.query_init, &[1, q, e]));
    let query_pe = g.add(emb, g.constant(t.query_emb, &[1, q, e]));

    // The pixels and their PE.
    let pixel_half = g.half(pixel_in);
    let mut pixel = g.float(g.conv(pixel_half, &t.pixel_init_proj, false)?);
    let pixel_emb = g.add(g.float(g.conv(pixel_half, &t.pixel_emb_proj, false)?), pe_in);
    let pixel_pe = g.tokens(pixel_emb, n, e, plane);

    let mut aux = vec![g.aux(pixel, &t.mask_pred[0])?];
    for (b, block) in t.blocks.iter().enumerate() {
        let mask = g.attn_mask(aux[b], n, plane, q);
        let (next_x, next_pixel) = g.block(block, t, x, pixel, query_pe, pixel_pe, mask, n, w, h)?;
        x = next_x;
        pixel = next_pixel;
        aux.push(g.aux(pixel, &t.mask_pred[b + 1])?);
    }
    let mut outs = vec![pixel];
    outs.extend(aux);
    Ok(([pixel_in, memory_in, pe_in], outs))
}

/// The graph's cache key: its weights key, the objects and the size.
fn graph_key(t: &CutieGraphTransformer<'_>, n: usize, w: usize, h: usize) -> super::GraphKey {
    (format!("transformer::{}", t.key), n, w, h)
}

impl MpsCache {
    /// The transformer's graph for `n` objects at `w` x `h`, made at first use (it compiles at its
    /// first encode), and the fp32 buffers it reads (pixels, object memory, encoding) and writes
    /// (pixels, then each aux logits).
    pub(crate) fn transformer_graph(
        &mut self, device: ObjcId, t: &CutieGraphTransformer<'_>, n: usize, w: usize, h: usize,
    ) -> MetalResult<(Vec<ObjcId>, Vec<ObjcId>)> {
        let (e, q, plane) = (t.embed, t.queries, w * h);
        let in_sizes = [n * e * plane, n * q * (e + 1), e * plane];
        let in_shapes = [vec![n, e, h, w], vec![n, q, e + 1], vec![1, e, h, w]];
        let mut in_bufs = Vec::with_capacity(3);
        for (i, &size) in in_sizes.iter().enumerate() {
            in_bufs.push(self.graph_buffer(device, ROLE_IN + i as u8, size * 4)?);
        }
        let mut out_sizes = vec![n * e * plane];
        let mut out_shapes = vec![vec![n, e, h, w]];
        for _ in 0..t.mask_pred.len() {
            out_sizes.push(n * plane);
            out_shapes.push(vec![n, 1, h, w]);
        }
        let mut out_bufs = Vec::with_capacity(out_sizes.len());
        for (i, &size) in out_sizes.iter().enumerate() {
            out_bufs.push(self.graph_buffer(device, ROLE_OUT + i as u8, size * 4)?);
        }
        let key = graph_key(t, n, w, h);
        if !self.graphs.contains_key(&key) {
            let built = unsafe {
                let graph: ObjcId = msg_send![class!(MPSGraph), new];
                let graph = StrongId::from_owned(graph).ok_or("MPSGraph new returned nil")?;
                let g = G(graph.as_id());
                let (inputs, outputs) = build(&g, t, n, w, h)?;
                let data = |buf: ObjcId, dims: &[usize]| -> MetalResult<StrongId> {
                    let d: ObjcId = msg_send![class!(MPSGraphTensorData), alloc];
                    let d: ObjcId = msg_send![d, initWithMTLBuffer: buf shape: graph_shape(dims) dataType: F32];
                    StrongId::from_owned(d).ok_or_else(|| "MPSGraphTensorData init returned nil".to_string())
                };
                let in_data = (0..3).map(|i| data(in_bufs[i], &in_shapes[i])).collect::<MetalResult<Vec<_>>>()?;
                let out_data = (0..outputs.len()).map(|i| data(out_bufs[i], &out_shapes[i])).collect::<MetalResult<Vec<_>>>()?;
                let in_objects: Vec<ObjcId> = in_data.iter().map(StrongId::as_id).collect();
                let out_objects: Vec<ObjcId> = out_data.iter().map(StrongId::as_id).collect();
                let feeds: ObjcId =
                    msg_send![class!(NSDictionary), dictionaryWithObjects: in_objects.as_ptr() forKeys: inputs.as_ptr() count: 3u64];
                let results: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObjects: out_objects.as_ptr() forKeys: outputs.as_ptr() count: outputs.len() as u64];
                let feeds = StrongId::retain(feeds).ok_or("NSDictionary returned nil")?;
                let results = StrongId::retain(results).ok_or("NSDictionary returned nil")?;
                GraphConv { graph, feeds, results, input: inputs[0], output: outputs[0] }
            };
            self.graphs.insert(key, built);
        }
        Ok((in_bufs, out_bufs))
    }

    /// The transformer's graph `transformer_graph` made for `n` objects at `w` x `h`.
    pub(crate) fn transformer(&self, t: &CutieGraphTransformer<'_>, n: usize, w: usize, h: usize) -> Option<&GraphConv> {
        self.graphs.get(&graph_key(t, n, w, h))
    }
}
