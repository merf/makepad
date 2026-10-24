//! The similarity of Cutie's memory read as one MPSGraph (an fp16 step's, macOS): the query's
//! similarity to every memory token, instead of `gpu_cutie_similarity`, Cutie's own exact-order
//! kernel. The top-k softmax and the readout stay the kernels'.
//!
//! Precision: memory's key and shrinkage and the query are read in half; the similarity (its
//! product and sums) runs in fp32, since MPSGraph's half product sums in half and the
//! similarity's terms cancel (in half it moved the masks far off: worst IoU 0.91 against fp32).
//!
//! Tried and not kept (a probe at 2040 pixels and 16k tokens, `tests/memory_read_probe.rs`): the
//! whole read in one graph, memory padded to buckets of tokens. MPSGraph's top-k alone took
//! about 13 ms there against the kernel's 7, and its gather or dense readout more again, so the
//! whole graph was slower than the kernels below 16k tokens. The similarity alone is fast.
//!
//! The graph is made for each size of memory (it grows by a frame at a time); each compiles in a
//! few ms. The types compile on every OS; the op runs on macOS only.

use crate::stream::MetalResult;

/// The similarity's memory and query: memory in parts side by side ([long-term, working]), each
/// part a key [ck, n_i] and a shrinkage [1, n_i]; the query's key and selection [ck, p].
pub struct CutieMemorySimilarity<'a, T> {
    pub keys: &'a [&'a T],
    pub shrinkages: &'a [&'a T],
    pub query_key: &'a T,
    pub query_selection: &'a T,
}

#[cfg(target_os = "macos")]
impl CutieMemorySimilarity<'_, crate::gpu_types::GpuTensor> {
    /// The key's channels, the pixels and the memory tokens; `Err` naming what does not fit.
    pub fn sizes(&self) -> MetalResult<(usize, usize, usize)> {
        let op = "gpu_cutie_memory_similarity_f16";
        let parts = self.keys.len();
        if parts == 0 || self.shrinkages.len() != parts {
            return Err(format!("{op}: {parts} key parts, {} shrinkage", self.shrinkages.len()));
        }
        let (ck, p) = (self.query_key.rows(), self.query_key.cols());
        if self.query_selection.rows() != ck || self.query_selection.cols() != p {
            return Err(format!("{op}: selection {}x{} for a key {ck}x{p}", self.query_selection.rows(), self.query_selection.cols()));
        }
        let mut tokens = 0;
        for i in 0..parts {
            let n = self.keys[i].cols();
            if self.keys[i].rows() != ck || self.shrinkages[i].rows() * self.shrinkages[i].cols() != n {
                return Err(format!("{op}: memory part {i} does not line up"));
            }
            tokens += n;
        }
        if tokens == 0 || p == 0 {
            return Err(format!("{op}: {tokens} tokens, {p} pixels"));
        }
        Ok((ck, p, tokens))
    }
}

/// The memory read's similarity (see the module) for an fp16 step: [tokens, p] in fp32, as
/// `gpu_cutie_similarity` lays it out.
#[cfg(target_os = "macos")]
pub fn gpu_cutie_memory_similarity_f16(
    r: &CutieMemorySimilarity<'_, crate::gpu_types::GpuTensor>,
) -> MetalResult<crate::gpu_types::GpuTensor> {
    use crate::gpu_types::GpuTensor;
    use crate::stream::{with_metal_stream, MpsMemoryRun};
    let (ck, p, tokens) = r.sizes()?;
    with_metal_stream("gpu_cutie_memory_similarity_f16", |s| {
        let similarity = GpuTensor::alloc(tokens, p)?;
        s.memory_similarity_mpsgraph(&MpsMemoryRun {
            read: r,
            ck,
            p,
            tokens,
            similarity: &similarity,
            pad: &super::matmul_conv::PAD_COLS_TO_HALF,
            to_half: &super::matmul_conv::PLANAR_TO_HALF,
            copy: &super::matmul_conv::GRAPH_TO_PLANAR,
        })?;
        Ok(similarity)
    })
}
