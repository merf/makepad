//! `linear_f32_resident` on the stream's fp32 GEMM (RationalVisions docs/cutie-metal.md §7 and §9):
//! x @ W^T + b, as CUDA's `gpu_linear_f32_resident` (cuda/src/launch.rs) computes it: an SGEMM in
//! fp32, then the bias added on the device in a pass of its own (CUDA's
//! `makepad_cuda_add_rows_vec_f32`), so each output is the product rounded, then one fp32 add.
//! Its checks and messages are CUDA's. It replaces `gpu_tensor::linear_f32_resident`, whose ggml
//! mul_mm stages its tiles in half, for every caller: Body's linears become fp32 (deliberately),
//! and Cutie's `gpu_cutie_linear` needs no host sgemm.
//!
//! In the host binding W's device copy is a stream weight keyed by W's content id
//! (`gpu_linear_f32_resident::t{id}::w`), copied once and kept, as ggml's `resident_f32` cache
//! kept it before: not through the input mirror, whose least recently used copies go past its cap
//! (512 MB) and which never keeps a copy bigger than that, so Body's 664 MB pose correctives would
//! be copied again on every call. `gpu_weight_cache_evict_prefix("gpu_linear_f32_resident")`
//! drops them. Once tensors are resident (step 12), W is bound from its own storage instead and
//! this cache goes.
//!
//! macOS.

use crate::gpu_types::GpuTensor;
use crate::stream::gemm::{gemm_dim, GemmAddRows, GEMM_ADD_ROWS};
use crate::stream::{with_metal_stream, Bind, Gemm, GemmOperand, Grid, MetalResult};

const OP: &str = "gpu_linear_f32_resident";

/// The bias pass's threadgroup: 32 columns by 8 rows.
const ADD_ROWS_THREADS: [u32; 3] = [32, 8, 1];

/// x [m, k] @ W^T (W [n, k]) + bias [n] into a new [m, n], in fp32 (CUDA's
/// `gpu_linear_f32_resident`). k = 0 gives the bias alone (0 without one).
pub fn linear_f32_resident(x: &GpuTensor, weight: &GpuTensor, bias: Option<&GpuTensor>) -> MetalResult<GpuTensor> {
    if x.is_half() || weight.is_half() {
        return Err("gpu_linear_f32_resident expects f32 tensors".to_string());
    }
    if weight.cols() != x.cols() {
        return Err(format!(
            "gpu_linear_f32_resident k mismatch: x {}x{}, w {}x{}",
            x.rows(),
            x.cols(),
            weight.rows(),
            weight.cols()
        ));
    }
    let (m, k, n) = (x.rows(), x.cols(), weight.rows());
    if let Some(bias) = bias {
        if bias.rows() * bias.cols() != n {
            return Err("gpu_linear_f32_resident bias mismatch".to_string());
        }
    }
    let rows = gemm_dim(OP, "m", m)?;
    let cols = gemm_dim(OP, "n", n)?;
    gemm_dim(OP, "k", k)?;
    let weight_key = format!("{OP}::t{}::w", weight.id.get());
    with_metal_stream(OP, |s| {
        // The host binding's copy of W, by content id (module docs). The fill reads W's host
        // values, which only the host binding has; step 12 binds W itself (Bind::In).
        let w = s.weights(&weight_key, n * k, |dst| {
            let values = weight.data.try_borrow().map_err(|_| format!("{OP}: the weight tensor is mutably borrowed"))?;
            dst.copy_from_slice(&values);
            Ok(())
        })?;
        let out = GpuTensor::alloc(m, n)?;
        s.gemm(&Gemm {
            m,
            n,
            k,
            batch: 1,
            alpha: 1.0,
            beta: 0.0,
            a: GemmOperand { src: Bind::In(x), offset: 0, ld: k, trans: false, batch_stride: 0 },
            // W [n, k] read as its transpose, op(B) [k, n].
            b: GemmOperand { src: Bind::Weights(&w, 0), offset: 0, ld: k, trans: true, batch_stride: 0 },
            c: Bind::Out(&out),
            c_offset: 0,
            ldc: n,
            c_batch_stride: 0,
        })?;
        if let Some(bias) = bias {
            let grid = Grid::cover([n, m, 1], ADD_ROWS_THREADS)?;
            s.launch(&GEMM_ADD_ROWS, &[], &GemmAddRows { rows, cols }, &[Bind::InOut(&out), Bind::In(bias)], grid)?;
        }
        Ok(out)
    })
}
