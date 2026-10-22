// Cutie's ops on builds without their CUDA kernels (cuda/src/cutie_ops.rs): each runs its host
// reference (cuda/src/cutie_reference.rs), whose convolutions, adds and resizes go through this
// backend's gpu_* ops, so on macOS Cutie runs on the Metal tensors, correct and slow. Included
// inside gpu.rs's `mod imp` for those builds; the signatures are cutie_ops.rs's.

/// Whether Cutie's ops run here: on macOS when the Metal device is there (the host references
/// then run on it); elsewhere without the kernels, never.
pub fn gpu_cutie_available() -> bool {
    cfg!(target_os = "macos") && gpu_device_available()
}

/// Whether `gpu_cutie_X` are kernels of their own (cuda/src/cutie_ops.rs) or, as here, the host
/// references.
pub const GPU_CUTIE_KERNELS: bool = false;

/// The CUDA top-k kernel's largest k, refused here too so a step fails alike on every build.
const CUTIE_TOPK_MAX: usize = 64;

include!("../../../cuda/src/cutie_reference.rs");

/// x @ W^T + b in fp32 (cutie_ops.rs's `gpu_cutie_linear`): on macOS Accelerate's sgemm, since
/// Metal's `gpu_linear_f32_resident` stages its tiles in half (rel 3e-4, past `Tol::STAGE`).
pub fn gpu_cutie_linear(x: &GpuTensor, w: &GpuTensor, bias: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    {
        return makepad_ai_metal::gpu_tensor::linear_f32_exact(x, w, bias);
    }
    #[cfg(not(target_os = "macos"))]
    {
        gpu_linear_f32_resident(x, w, bias)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
) -> Result<GpuTensor, String> {
    gpu_cutie_conv2d_reference(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride)
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
    residual: Option<&GpuTensor>, relu: bool,
) -> Result<GpuTensor, String> {
    gpu_cutie_conv2d_fused_reference(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, residual, relu)
}

pub fn gpu_cutie_maxpool3x3s2(x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor, String> {
    gpu_cutie_maxpool3x3s2_reference(x, w, h)
}

pub fn gpu_cutie_sigmoid(x: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_sigmoid_reference(x)
}

pub fn gpu_cutie_gru(h: &GpuTensor, values: &GpuTensor, n: usize, c: usize) -> Result<GpuTensor, String> {
    gpu_cutie_gru_reference(h, values, n, c)
}

pub fn gpu_cutie_eca(x: &GpuTensor, residual: &GpuTensor, n: usize, c: usize, conv_w: &[f32]) -> Result<GpuTensor, String> {
    gpu_cutie_eca_reference(x, residual, n, c, conv_w)
}

pub fn gpu_cutie_others(masks: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_others_reference(masks)
}

pub fn gpu_cutie_pad(x: &GpuTensor, w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize) -> Result<GpuTensor, String> {
    gpu_cutie_pad_reference(x, w, h, left, right, top, bottom)
}

pub fn gpu_cutie_crop(x: &GpuTensor, w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    gpu_cutie_crop_reference(x, w, h, left, top, out_w, out_h)
}

pub fn gpu_cutie_matmul(a: &GpuTensor, trans_a: bool, b: &GpuTensor, trans_b: bool) -> Result<GpuTensor, String> {
    gpu_cutie_matmul_reference(a, trans_a, b, trans_b)
}

pub fn gpu_cutie_similarity(mk: &GpuTensor, ms: Option<&GpuTensor>, qk: &GpuTensor, qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    gpu_cutie_similarity_reference(mk, ms, qk, qe)
}

pub fn gpu_cutie_topk_softmax(sim: &GpuTensor, k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    if k > CUTIE_TOPK_MAX {
        return Err(format!("gpu_cutie_topk_softmax: k {k} over the kernel's {CUTIE_TOPK_MAX}"));
    }
    gpu_cutie_topk_softmax_reference(sim, k)
}

pub fn gpu_cutie_softmax_axis0(x: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_softmax_axis0_reference(x)
}

pub fn gpu_cutie_aggregate(prob: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_aggregate_reference(prob)
}

pub fn gpu_cutie_attn_mask(aux: &GpuTensor, heads: usize, queries: usize) -> Result<GpuTensor, String> {
    gpu_cutie_attn_mask_reference(aux, heads, queries)
}

pub fn gpu_cutie_add_broadcast(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_add_broadcast_reference(a, b)
}

pub fn gpu_cutie_normalise_rows(x: &GpuTensor, sub: &GpuTensor, div: &GpuTensor) -> Result<GpuTensor, String> {
    gpu_cutie_normalise_rows_reference(x, sub, div)
}

pub fn gpu_cutie_rgb_planes(rgb: &[u8], plane: usize, scale: f32) -> Result<GpuTensor, String> {
    gpu_cutie_rgb_planes_reference(rgb, plane, scale)
}

pub fn gpu_cutie_resize_bilinear(x: &GpuTensor, in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    gpu_cutie_resize_bilinear_reference(x, in_w, in_h, out_w, out_h)
}

pub fn gpu_cutie_object_values(memory: &GpuTensor, objects: usize, eps: f32) -> Result<GpuTensor, String> {
    gpu_cutie_object_values_reference(memory, objects, eps)
}

pub fn gpu_cutie_sine_encoding(inv_freq: &[f32], w: usize, h: usize, scale: f32, eps: f32) -> Result<GpuTensor, String> {
    gpu_cutie_sine_encoding_reference(inv_freq, w, h, scale, eps)
}

pub fn gpu_cutie_layer_norm(
    x: &GpuTensor, cache_namespace: &str, key: &str, gamma: &[f32], beta: &[f32], eps: f32,
) -> Result<GpuTensor, String> {
    gpu_cutie_layer_norm_reference(x, cache_namespace, key, gamma, beta, eps)
}
