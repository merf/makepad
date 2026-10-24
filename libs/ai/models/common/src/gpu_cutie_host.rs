// Cutie's ops on builds without their CUDA kernels (cuda/src/cutie_ops.rs), at its signatures.
// On macOS each runs Makepad's Metal kernel (makepad_ai_metal::cutie_ops, RationalVisions
// docs/cutie-metal.md §8), with CUDA's checks and messages, or its host reference
// (cuda/src/cutie_reference.rs, included below) where MAKEPAD_CUTIE_METAL_REF names it (`all`, or
// ops by name: gpu.rs's `MetalReference`), so the reference stays one variable away. Elsewhere
// every op is its reference, and `gpu_cutie_available` says false. The references' convolutions,
// adds and resizes go through this backend's gpu_* ops. Included inside gpu.rs's `mod imp` for
// those builds.

/// Whether Cutie's ops run here: on macOS when the calling thread's Metal stream starts (the device
/// check: `gpu_device_available` is true on every Mac), every kernel library Cutie's ops launch
/// compiles there and the kernels written for 32-wide SIMD groups get them
/// (`makepad_ai_metal::cutie_ops::cutie_ready`, which compiles them now on this thread, so ask on
/// Cutie's; a failure is logged with Metal's message); elsewhere without the kernels, never.
pub fn gpu_cutie_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        if let Err(e) = makepad_ai_metal::cutie_ops::cutie_ready() {
            eprintln!("[cutie] Cutie's ops do not run on Metal here: {e}");
            return false;
        }
        true
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Whether `gpu_cutie_X` are CUDA's kernels (cuda/src/cutie_ops.rs), whose expf, logf, sinf and
/// cosf are PyTorch's, so `cutie-validate ops` holds them to PyTorch's bits. Not here: Metal's
/// kernels call `precise::` functions, which round otherwise, so the pins that depend on them are
/// held at the references' tolerance (cutie-metal.md D8); everything else is the same.
pub const GPU_CUTIE_KERNELS: bool = false;

/// The CUDA top-k kernel's largest k, refused on the reference too so a step fails alike on every
/// build and either path.
const CUTIE_TOPK_MAX: usize = 64;

include!("../../../cuda/src/cutie_reference.rs");

/// CUDA's rule for the top-k's k: at most the rows (`k.min(m)`), then at most the kernel's 64,
/// with CUDA's message.
fn cutie_topk_check(sim: &GpuTensor, k: usize) -> Result<(), String> {
    let m = sim.rows();
    let k = k.min(m);
    if k > CUTIE_TOPK_MAX {
        return Err(format!(
            "gpu_cutie_topk_softmax: top {k} of {m} rows; the kernel takes at most {CUTIE_TOPK_MAX} (Cutie's top_k is 30)"
        ));
    }
    Ok(())
}

/// x @ W^T + b in fp32 (cutie_ops.rs's `gpu_cutie_linear`, CUDA's `gpu_linear_f32_resident`): on
/// macOS the stream's fp32 GEMM, its reference Accelerate's sgemm.
pub fn gpu_cutie_linear(x: &GpuTensor, w: &GpuTensor, bias: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    {
        if super::cutie_metal_ref("linear")? {
            return makepad_ai_metal::gpu_tensor::linear_f32_exact(x, w, bias);
        }
        return makepad_ai_metal::linear::linear_f32_resident(x, w, bias);
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
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("conv2d")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_conv2d(
            x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride,
        );
    }
    gpu_cutie_conv2d_reference(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride)
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
    residual: Option<&GpuTensor>, relu: bool,
) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("conv2d_fused")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_conv2d_fused(
            x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, residual, relu,
        );
    }
    gpu_cutie_conv2d_fused_reference(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, residual, relu)
}

/// CUDA's half convolution for fp16 steps. On macOS, `MPSCNNConvolution` (its weights in half, its
/// sums in fp32: metal's `gpu_cutie_conv2d_fused_f16`); elsewhere without CUDA's kernels there is
/// none, so it is the fp32 one, `GPU_CUTIE_HALF` says so, and Cutie reports fp32.
#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused_f16(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
    residual: Option<&GpuTensor>, relu: bool,
) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("conv2d_fused")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_conv2d_fused_f16(
            x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, residual, relu,
        );
    }
    gpu_cutie_conv2d_fused(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, residual, relu)
}

/// Whether `gpu_cutie_graph_f16` runs here: on macOS (MPSGraph), not on the other builds without
/// CUDA's kernels.
pub const GPU_CUTIE_GRAPH: bool = cfg!(target_os = "macos");

/// A chain of convolutions as one graph in half (metal's `gpu_cutie_graph_f16`) on macOS; an
/// error elsewhere (`GPU_CUTIE_GRAPH`).
pub fn gpu_cutie_graph_f16(
    x: &GpuTensor, width: usize, height: usize, key: &str, ops: &[super::CutieGraphOp<'_>], outputs: &[usize],
) -> Result<Vec<GpuTensor>, String> {
    #[cfg(target_os = "macos")]
    {
        return makepad_ai_metal::cutie_ops::gpu_cutie_graph_f16(x, width, height, key, ops, outputs);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (x, width, height, key, ops, outputs);
        Err("gpu_cutie_graph_f16: macOS only".into())
    }
}

/// The object transformer as one graph in half (metal's `gpu_cutie_transformer_graph_f16`) on
/// macOS; an error elsewhere (`GPU_CUTIE_GRAPH`).
pub fn gpu_cutie_transformer_graph_f16(
    t: &super::CutieGraphTransformer<'_>, pixel: &GpuTensor, objects: usize, width: usize, height: usize, obj_memory: &GpuTensor,
    pe: &GpuTensor,
) -> Result<(GpuTensor, Vec<GpuTensor>), String> {
    #[cfg(target_os = "macos")]
    {
        return makepad_ai_metal::cutie_ops::gpu_cutie_transformer_graph_f16(t, pixel, objects, width, height, obj_memory, pe);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (t, pixel, objects, width, height, obj_memory, pe);
        Err("gpu_cutie_transformer_graph_f16: macOS only".into())
    }
}

/// The memory read's similarity as one graph (metal's `gpu_cutie_memory_similarity_f16`) on
/// macOS; an error elsewhere (`GPU_CUTIE_GRAPH`).
pub fn gpu_cutie_memory_similarity_f16(r: &super::CutieMemorySimilarity<'_, GpuTensor>) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    {
        return makepad_ai_metal::cutie_ops::gpu_cutie_memory_similarity_f16(r);
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = r;
        Err("gpu_cutie_memory_similarity_f16: macOS only".into())
    }
}

/// Whether `gpu_cutie_conv2d_fused_f16` really runs in half: on macOS (MPSCNNConvolution), not on
/// the other builds without CUDA's kernels.
pub const GPU_CUTIE_HALF: bool = cfg!(target_os = "macos");

/// Whether an fp16 step runs even the convolutions Cutie keeps in fp32 under autocast (`pred`, the
/// sensory updates' `transform`) in half: on macOS, where MPSCNN makes them a sixth of a frame
/// faster at 960x540 with the masks as close (CUDA follows autocast).
pub const GPU_CUTIE_HALF_EVERY_CONV: bool = cfg!(target_os = "macos");

pub fn gpu_cutie_maxpool3x3s2(x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("maxpool3x3s2")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_maxpool3x3s2(x, w, h);
    }
    gpu_cutie_maxpool3x3s2_reference(x, w, h)
}

pub fn gpu_cutie_sigmoid(x: &GpuTensor) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("sigmoid")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_sigmoid(x);
    }
    gpu_cutie_sigmoid_reference(x)
}

pub fn gpu_cutie_gru(h: &GpuTensor, values: &GpuTensor, n: usize, c: usize) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("gru")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_gru(h, values, n, c);
    }
    gpu_cutie_gru_reference(h, values, n, c)
}

/// More taps than the kernel takes by value run the reference, as on CUDA (Cutie's ECA has 3 or 5).
pub fn gpu_cutie_eca(x: &GpuTensor, residual: &GpuTensor, n: usize, c: usize, conv_w: &[f32]) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("eca")? && conv_w.len() <= makepad_ai_metal::cutie_ops::CUTIE_ECA_MAX_TAPS {
        return makepad_ai_metal::cutie_ops::gpu_cutie_eca(x, residual, n, c, conv_w);
    }
    gpu_cutie_eca_reference(x, residual, n, c, conv_w)
}

pub fn gpu_cutie_others(masks: &GpuTensor) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("others")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_others(masks);
    }
    gpu_cutie_others_reference(masks)
}

pub fn gpu_cutie_pad(x: &GpuTensor, w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("pad")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_pad(x, w, h, left, right, top, bottom);
    }
    gpu_cutie_pad_reference(x, w, h, left, right, top, bottom)
}

pub fn gpu_cutie_crop(x: &GpuTensor, w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("crop")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_crop(x, w, h, left, top, out_w, out_h);
    }
    gpu_cutie_crop_reference(x, w, h, left, top, out_w, out_h)
}

pub fn gpu_cutie_matmul(a: &GpuTensor, trans_a: bool, b: &GpuTensor, trans_b: bool) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("matmul")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_matmul(a, trans_a, b, trans_b);
    }
    gpu_cutie_matmul_reference(a, trans_a, b, trans_b)
}

pub fn gpu_cutie_similarity(mk: &GpuTensor, ms: Option<&GpuTensor>, qk: &GpuTensor, qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("similarity")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_similarity(mk, ms, qk, qe);
    }
    gpu_cutie_similarity_reference(mk, ms, qk, qe)
}

pub fn gpu_cutie_topk_softmax(sim: &GpuTensor, k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("topk_softmax")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_topk_softmax(sim, k);
    }
    cutie_topk_check(sim, k)?;
    gpu_cutie_topk_softmax_reference(sim, k)
}

pub fn gpu_cutie_softmax_axis0(x: &GpuTensor) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("softmax_axis0")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_softmax_axis0(x);
    }
    gpu_cutie_softmax_axis0_reference(x)
}

pub fn gpu_cutie_aggregate(prob: &GpuTensor) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("aggregate")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_aggregate(prob);
    }
    gpu_cutie_aggregate_reference(prob)
}

pub fn gpu_cutie_attn_mask(aux: &GpuTensor, heads: usize, queries: usize) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("attn_mask")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_attn_mask(aux, heads, queries);
    }
    gpu_cutie_attn_mask_reference(aux, heads, queries)
}

pub fn gpu_cutie_add_broadcast(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("add_broadcast")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_add_broadcast(a, b);
    }
    gpu_cutie_add_broadcast_reference(a, b)
}

pub fn gpu_cutie_normalise_rows(x: &GpuTensor, sub: &GpuTensor, div: &GpuTensor) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("normalise_rows")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_normalise_rows(x, sub, div);
    }
    gpu_cutie_normalise_rows_reference(x, sub, div)
}

pub fn gpu_cutie_rgb_planes(rgb: &[u8], plane: usize, scale: f32) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("rgb_planes")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_rgb_planes(rgb, plane, scale);
    }
    gpu_cutie_rgb_planes_reference(rgb, plane, scale)
}

pub fn gpu_cutie_resize_bilinear(x: &GpuTensor, in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("resize_bilinear")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_resize_bilinear(x, in_w, in_h, out_w, out_h);
    }
    gpu_cutie_resize_bilinear_reference(x, in_w, in_h, out_w, out_h)
}

pub fn gpu_cutie_object_values(memory: &GpuTensor, objects: usize, eps: f32) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("object_values")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_object_values(memory, objects, eps);
    }
    gpu_cutie_object_values_reference(memory, objects, eps)
}

pub fn gpu_cutie_sine_encoding(inv_freq: &[f32], w: usize, h: usize, scale: f32, eps: f32) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("sine_encoding")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_sine_encoding(inv_freq, w, h, scale, eps);
    }
    gpu_cutie_sine_encoding_reference(inv_freq, w, h, scale, eps)
}

/// The reference is `gpu_layer_norm_pytorch` with the affine uploaded, which on macOS runs this
/// op's kernel too; MAKEPAD_CUTIE_METAL_REF forces `layer_norm` and `layer_norm_pytorch`
/// together, so the reference it forces is the host's all the way down.
pub fn gpu_cutie_layer_norm(
    x: &GpuTensor, cache_namespace: &str, key: &str, gamma: &[f32], beta: &[f32], eps: f32,
) -> Result<GpuTensor, String> {
    #[cfg(target_os = "macos")]
    if !super::cutie_metal_ref("layer_norm")? {
        return makepad_ai_metal::cutie_ops::gpu_cutie_layer_norm(x, cache_namespace, key, gamma, beta, eps);
    }
    gpu_cutie_layer_norm_reference(x, cache_namespace, key, gamma, beta, eps)
}
