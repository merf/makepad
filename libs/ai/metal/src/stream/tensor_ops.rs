//! Tensor ops on the stream's own kernels (`tensor.metal`): fills and copies, and `gpu_add`,
//! `gpu_mul` and `gpu_birefnet_relu` on macOS for every caller (since step 12; ggml's add and mul
//! and a host ReLU before). Shapes are checked before anything is encoded, with CUDA's rules (`add`
//! and `mul` take equal shapes: no broadcast, no truncation). One IEEE operation an element, so the
//! bits are CUDA's and the host's.

use super::kernels::{TensorCount, TENSOR_ADD, TENSOR_MUL, TENSOR_RELU};
use super::seam::element_count;
use super::{with_metal_stream, Bind, Grid, Kernel, MetalResult};
use crate::gpu_types::GpuTensor;

/// Threads to a threadgroup for the elementwise kernels.
const THREADS: u32 = 256;

/// A new `[rows, cols]` tensor whose every word is `bits`.
pub fn fill(rows: usize, cols: usize, bits: u32) -> MetalResult<GpuTensor> {
    with_metal_stream("metal fill", |s| {
        let out = GpuTensor::alloc(rows, cols)?;
        s.fill(&out, bits)?;
        Ok(out)
    })
}

/// A copy of `src`, word for word.
pub fn copy(src: &GpuTensor) -> MetalResult<GpuTensor> {
    with_metal_stream("metal copy", |s| {
        let out = GpuTensor::alloc(src.rows(), src.cols())?;
        s.copy(src, &out)?;
        Ok(out)
    })
}

/// `a + b` elementwise (CUDA's `gpu_add`).
pub fn add(a: &GpuTensor, b: &GpuTensor) -> MetalResult<GpuTensor> {
    binary("gpu_add", &TENSOR_ADD, a, b)
}

/// `a * b` elementwise (CUDA's `gpu_mul`).
pub fn mul(a: &GpuTensor, b: &GpuTensor) -> MetalResult<GpuTensor> {
    binary("gpu_mul", &TENSOR_MUL, a, b)
}

/// `fmax(x, 0)` (CUDA's `gpu_birefnet_relu`).
pub fn relu(x: &GpuTensor) -> MetalResult<GpuTensor> {
    let n = element_count("gpu_birefnet_relu", x)?;
    with_metal_stream("gpu_birefnet_relu", |s| {
        let out = GpuTensor::alloc(x.rows(), x.cols())?;
        let grid = Grid::linear(n as usize, THREADS)?;
        s.launch(&TENSOR_RELU, &[], &TensorCount { n }, &[Bind::In(x), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}

fn binary(op: &'static str, kernel: &'static Kernel, a: &GpuTensor, b: &GpuTensor) -> MetalResult<GpuTensor> {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return Err(format!(
            "{op} shape mismatch {}x{} vs {}x{}",
            a.rows(),
            a.cols(),
            b.rows(),
            b.cols()
        ));
    }
    let n = element_count(op, a)?;
    with_metal_stream(op, |s| {
        let out = GpuTensor::alloc(a.rows(), a.cols())?;
        let grid = Grid::linear(n as usize, THREADS)?;
        s.launch(kernel, &[], &TensorCount { n }, &[Bind::In(a), Bind::In(b), Bind::Out(&out)], grid)?;
        Ok(out)
    })
}
