//! Packed cross-attention in fp32 on the stream's GEMM (RationalVisions docs/cutie-metal.md §7):
//! CUDA's `gpu_attention_packed_cross_f32` and `gpu_attention_packed_cross_bias`
//! (cuda/src/launch.rs) step for step, with their checks and messages:
//! 1. per head, scores = scale · Q_h K_hᵀ: one strided-batched GEMM over the heads, Q and K read
//!    in place (ld the hidden width, a head's columns `head_dim` apart), into a `[heads·rows, kv]`
//!    scores tensor laid out `[h][q][k]`;
//! 2. each row's precise softmax in place (CUDA's `makepad_cuda_softmax_rows_precise_f32`), the
//!    additive bias `[heads·q_len, kv]` (same layout, −inf allowed) added first, as CUDA's
//!    `makepad_cuda_add_f32_precise` pass does;
//! 3. per head, out_h = P_h V_h: one strided-batched GEMM writing each head's columns of the
//!    output in place.
//!
//! Every product, sum and the softmax are fp32 (`gemm.metal`); exp is `precise::exp` where CUDA
//! has `expf`, so results are CUDA's to fp32 rounding and exp's, not its bits. Without the bias the
//! queries run in chunks so the scores stay within 64M floats (256 MB), as CUDA's do; with it, in
//! one pass, as CUDA's do.
//!
//! macOS. Until the wiring declares it in lib.rs it compiles as `stream::gemm::attention`.

use crate::gpu_types::GpuTensor;
use crate::stream::gemm::{check_simd_width, gemm_dim, GemmSoftmaxRows, GEMM_SOFTMAX_ROWS, SIMD_WIDTH};
use crate::stream::{with_metal_stream, Bind, Gemm, GemmOperand, Grid, MetalResult, MetalStream};

/// The most floats the scores of one chunk of queries hold: CUDA's bound (256 MB).
const MAX_SCORE_ELEMS: usize = 64 * 1024 * 1024;

/// The shapes both attentions share, checked.
#[derive(Clone, Copy)]
struct Cross {
    q_len: usize,
    kv_len: usize,
    hidden: usize,
    heads: usize,
    head_dim: usize,
}

/// softmax(scale · Q_h K_hᵀ) V_h for every head, Q [q_len, hidden] and K, V [kv_len, hidden] with
/// the heads packed along the columns; out [q_len, hidden]. CUDA's
/// `gpu_attention_packed_cross_f32`: fp32 GEMMs and an fp32 softmax.
pub fn attention_packed_cross_f32(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    head_count: usize,
    scale: f32,
) -> MetalResult<GpuTensor> {
    attention_packed_cross_f32_capped(q, k, v, head_count, scale, MAX_SCORE_ELEMS)
}

/// [`attention_packed_cross_f32`] with the scores of a chunk of queries bounded at
/// `max_score_elems` floats instead of CUDA's 64M, so a test reaches the chunked path with small
/// tensors.
#[doc(hidden)]
pub fn attention_packed_cross_f32_capped(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    head_count: usize,
    scale: f32,
    max_score_elems: usize,
) -> MetalResult<GpuTensor> {
    const OP: &str = "gpu_attention_packed_cross_f32";
    let cross = check_cross(q, k, v, head_count, "gpu_attention_cross")?;
    let chunk_rows = (max_score_elems / (head_count * cross.kv_len).max(1)).clamp(1, cross.q_len.max(1));
    check_sizes(OP, &cross, chunk_rows)?;
    with_metal_stream(OP, |s| run(s, OP, &cross, q, k, v, scale, None, chunk_rows))
}

/// [`attention_packed_cross_f32`] with an additive score bias `[heads·q_len, kv_len]` (row-major
/// `h, q, k`; −inf blocks a key) added before the softmax. CUDA's
/// `gpu_attention_packed_cross_bias` (SAM 3.1's box-RPB decoder attention, Cutie's masked
/// attention).
pub fn attention_packed_cross_bias(
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    head_count: usize,
    scale: f32,
    bias: &GpuTensor,
) -> MetalResult<GpuTensor> {
    const OP: &str = "gpu_attention_packed_cross_bias";
    let cross = check_cross(q, k, v, head_count, "gpu_attention_cross_bias")?;
    if bias.is_half() {
        return Err("gpu_attention_cross_bias is f32-only".to_string());
    }
    if bias.rows() * bias.cols() != head_count * cross.q_len * cross.kv_len {
        return Err("gpu_attention_cross_bias bias length mismatch".to_string());
    }
    let chunk_rows = cross.q_len.max(1);
    check_sizes(OP, &cross, chunk_rows)?;
    with_metal_stream(OP, |s| run(s, OP, &cross, q, k, v, scale, Some(bias), chunk_rows))
}

/// CUDA's checks, in its order, with its messages (`what` names the op as CUDA's do).
fn check_cross(q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, head_count: usize, what: &str) -> MetalResult<Cross> {
    let (q_len, kv_len, hidden) = (q.rows(), k.rows(), q.cols());
    if k.cols() != hidden || v.cols() != hidden || v.rows() != kv_len {
        return Err(format!("{what} shape mismatch"));
    }
    if head_count == 0 || hidden % head_count != 0 {
        return Err(format!("{what} head mismatch"));
    }
    if q.is_half() || k.is_half() || v.is_half() {
        return Err(format!("{what} is f32-only"));
    }
    Ok(Cross { q_len, kv_len, hidden, heads: head_count, head_dim: hidden / head_count })
}

/// Every size the GEMMs and the softmax take, before anything is encoded: dimensions within
/// cuBLAS's ints, and the tensors' floats and the scores' rows and floats within the kernels' 32
/// bits.
fn check_sizes(op: &str, c: &Cross, chunk_rows: usize) -> MetalResult<()> {
    for (what, v) in [("q_len", c.q_len), ("kv_len", c.kv_len), ("hidden", c.hidden), ("heads", c.heads)] {
        gemm_dim(op, what, v)?;
    }
    let score_rows = c.heads.checked_mul(chunk_rows);
    let scores = score_rows.and_then(|rows| rows.checked_mul(c.kv_len));
    let floats = [c.q_len.checked_mul(c.hidden), c.kv_len.checked_mul(c.hidden), score_rows, scores];
    for floats in floats {
        match floats {
            Some(n) if n <= u32::MAX as usize => {}
            _ => return Err(format!("{op}: a tensor past the kernels' 32-bit offsets")),
        }
    }
    Ok(())
}

/// The composite inside one op: the scores tensor, then per chunk of queries the scores GEMM, the
/// softmax and the output GEMM (writing its rows of `out`).
#[allow(clippy::too_many_arguments)]
fn run(
    s: &mut MetalStream<'_>,
    op: &'static str,
    c: &Cross,
    q: &GpuTensor,
    k: &GpuTensor,
    v: &GpuTensor,
    scale: f32,
    bias: Option<&GpuTensor>,
    chunk_rows: usize,
) -> MetalResult<GpuTensor> {
    let out = GpuTensor::alloc(c.q_len, c.hidden)?;
    let scores = GpuTensor::alloc(c.heads * chunk_rows, c.kv_len)?;
    let threads = softmax_threads(s, op, c.kv_len)?;
    let mut start = 0;
    while start < c.q_len {
        let rows = chunk_rows.min(c.q_len - start);
        // scores[h][i][j] = scale * sum_d q[start+i][h][d] * k[j][h][d]
        s.gemm(&Gemm {
            m: rows,
            n: c.kv_len,
            k: c.head_dim,
            batch: c.heads,
            alpha: scale,
            beta: 0.0,
            a: GemmOperand { src: Bind::In(q), offset: start * c.hidden, ld: c.hidden, trans: false, batch_stride: c.head_dim },
            b: GemmOperand { src: Bind::In(k), offset: 0, ld: c.hidden, trans: true, batch_stride: c.head_dim },
            c: Bind::Out(&scores),
            c_offset: 0,
            ldc: c.kv_len,
            c_batch_stride: rows * c.kv_len,
        })?;
        softmax_rows(s, &scores, c.heads * rows, c.kv_len, bias, threads)?;
        // out[start+i][h][d] = sum_j probs[h][i][j] * v[j][h][d]
        s.gemm(&Gemm {
            m: rows,
            n: c.head_dim,
            k: c.kv_len,
            batch: c.heads,
            alpha: 1.0,
            beta: 0.0,
            a: GemmOperand { src: Bind::In(&scores), offset: 0, ld: c.kv_len, trans: false, batch_stride: rows * c.kv_len },
            b: GemmOperand { src: Bind::In(v), offset: 0, ld: c.hidden, trans: false, batch_stride: c.head_dim },
            c: Bind::Out(&out),
            c_offset: start * c.hidden,
            ldc: c.hidden,
            c_batch_stride: c.head_dim,
        })?;
        start += rows;
    }
    Ok(out)
}

/// The softmax's threads a row: CUDA's 256, or 1024 from 1024 keys on, at most what the
/// pipeline takes (in whole SIMD groups). The count sets the order of the sums, not their
/// precision.
fn softmax_threads(s: &mut MetalStream<'_>, op: &str, cols: usize) -> MetalResult<u32> {
    check_simd_width(s, op, &GEMM_SOFTMAX_ROWS)?;
    let limits = s.limits(&GEMM_SOFTMAX_ROWS, &[])?;
    let want = if cols < 1024 { 256 } else { 1024 };
    Ok(want.min(limits.max_threads / SIMD_WIDTH * SIMD_WIDTH))
}

/// The softmax of each of `rows` rows of `scores` ([rows, cols] at its start), in place, the bias
/// (laid out as the scores) added first. Nothing to do without keys, as CUDA's.
fn softmax_rows(
    s: &mut MetalStream<'_>,
    scores: &GpuTensor,
    rows: usize,
    cols: usize,
    bias: Option<&GpuTensor>,
    threads: u32,
) -> MetalResult<()> {
    let params = GemmSoftmaxRows { rows: rows as u32, row_stride: cols as u32, cols: cols as u32, has_bias: bias.is_some() as u32 };
    let groups = if cols == 0 { 0 } else { rows };
    let grid = Grid::groups([groups, 1, 1], [threads, 1, 1])?;
    let bias = bias.map_or(Bind::Absent, Bind::In);
    s.launch(&GEMM_SOFTMAX_ROWS, &[], &params, &[Bind::InOut(scores), bias], grid)
}
