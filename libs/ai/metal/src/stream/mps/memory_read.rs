//! The similarity of Cutie's memory read built as one MPSGraph (`cutie_ops::memory_graph`, which
//! says what runs in half). Inputs are half buffers of the cache's, row-major from their start:
//! memory's key [ck, tokens] and shrinkage [1, tokens], the query's key and selection [ck, p].
//! Out: the similarity [tokens, p] in fp32.

use super::ops::{nil, G, F16, F32};
use super::{graph_shape, GraphConv, GraphKey, MpsCache};
use crate::objc_util::StrongId;
use crate::stream::MetalResult;
use makepad_objc_sys::runtime::ObjcId;
use makepad_objc_sys::{class, msg_send, sel, sel_impl};

/// The cache's buffer roles for the read: inputs from 40, the output 50.
const ROLE_IN: u8 = 40;
const ROLE_OUT: u8 = 50;

/// The similarity's sizes: the key's channels, the pixels and the memory tokens.
#[derive(Clone, Copy)]
pub(crate) struct MemoryShape {
    pub ck: usize,
    pub p: usize,
    pub tokens: usize,
}

impl MemoryShape {
    fn key(&self) -> GraphKey {
        (format!("memory_similarity::{}", self.ck), self.p, self.tokens, 0)
    }
    /// Input dims: key, shrinkage, query key, query selection.
    fn inputs(&self) -> [Vec<usize>; 4] {
        let (ck, p, n) = (self.ck, self.p, self.tokens);
        [vec![ck, n], vec![1, n], vec![ck, p], vec![ck, p]]
    }
    fn output(&self) -> Vec<usize> {
        vec![self.tokens, self.p]
    }
}

/// The similarity's graph: its input placeholders and its output. In fp32 from the half inputs:
/// -a_sq + 2 ab as one product, [mk^2; mk] against [-qe; 2 qk qe], then - b_sq, times the
/// shrinkage over sqrt(ck). (A half product sums in half: its 128 large terms of both signs
/// cancel to nothing like the fp32 sum, and the masks went far off.)
unsafe fn build(g: &G, s: &MemoryShape) -> ([ObjcId; 4], ObjcId) {
    let dims = s.inputs();
    let input = |i: usize| -> ObjcId { msg_send![g.0, placeholderWithShape: graph_shape(&dims[i]) dataType: F16 name: nil()] };
    let placeholders = [input(0), input(1), input(2), input(3)];
    let [mk, ms, qk, qe] = placeholders.map(|t| g.float(t));
    let a = g.concat(g.mul(mk, mk), mk, 0);
    let b = g.concat(g.neg(qe), g.mul(g.scalar(2.0, F32), g.mul(qk, qe)), 0);
    let products = g.matmul(g.transpose(a, 0, 1), b);
    let b_sq = g.sum_along(g.mul(qe, g.mul(qk, qk)), 0);
    let scale = g.scalar(1.0 / (s.ck as f64).sqrt(), F32);
    let similarity = g.mul(g.mul(g.sub(products, b_sq), g.transpose(ms, 0, 1)), scale);
    (placeholders, similarity)
}

impl MpsCache {
    /// The similarity's graph for `s`, made at first use (it compiles at its first encode), and
    /// its input and output buffers.
    pub(crate) fn memory_similarity_graph(&mut self, device: ObjcId, s: &MemoryShape) -> MetalResult<(Vec<ObjcId>, ObjcId)> {
        let (in_dims, out_dims) = (s.inputs(), s.output());
        let count = |d: &[usize]| d.iter().product::<usize>();
        let mut in_bufs = Vec::with_capacity(4);
        for (i, d) in in_dims.iter().enumerate() {
            in_bufs.push(self.graph_buffer(device, ROLE_IN + i as u8, count(d) * 2)?);
        }
        let out_buf = self.graph_buffer(device, ROLE_OUT, count(&out_dims) * 4)?;
        let key = s.key();
        if !self.graphs.contains_key(&key) {
            let built = unsafe {
                let graph: ObjcId = msg_send![class!(MPSGraph), new];
                let graph = StrongId::from_owned(graph).ok_or("MPSGraph new returned nil")?;
                let g = G(graph.as_id());
                let (inputs, output) = build(&g, s);
                let data = |buf: ObjcId, dims: &[usize], ty: u32| -> MetalResult<StrongId> {
                    let d: ObjcId = msg_send![class!(MPSGraphTensorData), alloc];
                    let d: ObjcId = msg_send![d, initWithMTLBuffer: buf shape: graph_shape(dims) dataType: ty];
                    StrongId::from_owned(d).ok_or_else(|| "MPSGraphTensorData init returned nil".to_string())
                };
                let in_data = (0..4).map(|i| data(in_bufs[i], &in_dims[i], F16)).collect::<MetalResult<Vec<_>>>()?;
                let out_data = data(out_buf, &out_dims, F32)?;
                let in_objects: Vec<ObjcId> = in_data.iter().map(StrongId::as_id).collect();
                let feeds: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObjects: in_objects.as_ptr() forKeys: inputs.as_ptr() count: 4u64];
                let results: ObjcId = msg_send![class!(NSDictionary), dictionaryWithObject: out_data.as_id() forKey: output];
                let feeds = StrongId::retain(feeds).ok_or("NSDictionary returned nil")?;
                let results = StrongId::retain(results).ok_or("NSDictionary returned nil")?;
                GraphConv { graph, feeds, results, input: inputs[0], output }
            };
            self.graphs.insert(key, built);
        }
        Ok((in_bufs, out_buf))
    }

    /// The similarity's graph `memory_similarity_graph` made for `s`.
    pub(crate) fn memory_similarity(&self, s: &MemoryShape) -> Option<&GraphConv> {
        self.graphs.get(&s.key())
    }
}
