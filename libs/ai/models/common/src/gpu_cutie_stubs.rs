// Cutie's device ops for builds without CUDA kernels: the signatures of cuda/src/cutie_ops.rs,
// each unavailable. Included inside gpu.rs's stub `mod imp`.

/// A convolution's output size along one axis (PyTorch's floor rule).
pub fn cutie_conv_out(size: usize, k: usize, pad: usize, stride: usize) -> usize {
    (size + 2 * pad - k) / stride + 1
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d(
    _x: &GpuTensor, _batch: usize, _in_w: usize, _in_h: usize, _cache_namespace: &str, _key: &str,
    _weights: &[f32], _bias: Option<&[f32]>, _out_channels: usize, _k: usize, _pad: usize, _stride: usize,
) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_reference(
    _x: &GpuTensor, _batch: usize, _in_w: usize, _in_h: usize, _cache_namespace: &str, _key: &str,
    _weights: &[f32], _bias: Option<&[f32]>, _out_channels: usize, _k: usize, _pad: usize, _stride: usize,
) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused(
    _x: &GpuTensor, _batch: usize, _in_w: usize, _in_h: usize, _cache_namespace: &str, _key: &str,
    _weights: &[f32], _bias: Option<&[f32]>, _out_channels: usize, _k: usize, _pad: usize, _stride: usize,
    _residual: Option<&GpuTensor>, _relu: bool,
) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused_reference(
    _x: &GpuTensor, _batch: usize, _in_w: usize, _in_h: usize, _cache_namespace: &str, _key: &str,
    _weights: &[f32], _bias: Option<&[f32]>, _out_channels: usize, _k: usize, _pad: usize, _stride: usize,
    _residual: Option<&GpuTensor>, _relu: bool,
) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_maxpool3x3s2(_x: &GpuTensor, _w: usize, _h: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_maxpool3x3s2_reference(_x: &GpuTensor, _w: usize, _h: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_sigmoid(_x: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_sigmoid_reference(_x: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_gru(_h: &GpuTensor, _values: &GpuTensor, _n: usize, _c: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_gru_reference(_h: &GpuTensor, _values: &GpuTensor, _n: usize, _c: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_eca(_x: &GpuTensor, _residual: &GpuTensor, _n: usize, _c: usize, _conv_w: &[f32]) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_eca_reference(_x: &GpuTensor, _residual: &GpuTensor, _n: usize, _c: usize, _conv_w: &[f32]) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_others(_masks: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_others_reference(_masks: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_pad(_x: &GpuTensor, _w: usize, _h: usize, _left: usize, _right: usize, _top: usize, _bottom: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_pad_reference(_x: &GpuTensor, _w: usize, _h: usize, _left: usize, _right: usize, _top: usize, _bottom: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_crop(_x: &GpuTensor, _w: usize, _h: usize, _left: usize, _top: usize, _out_w: usize, _out_h: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_crop_reference(_x: &GpuTensor, _w: usize, _h: usize, _left: usize, _top: usize, _out_w: usize, _out_h: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_matmul(_a: &GpuTensor, _trans_a: bool, _b: &GpuTensor, _trans_b: bool) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_matmul_reference(_a: &GpuTensor, _trans_a: bool, _b: &GpuTensor, _trans_b: bool) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_similarity(_mk: &GpuTensor, _ms: Option<&GpuTensor>, _qk: &GpuTensor, _qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_similarity_reference(_mk: &GpuTensor, _ms: Option<&GpuTensor>, _qk: &GpuTensor, _qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_topk_softmax(_sim: &GpuTensor, _k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_topk_softmax_reference(_sim: &GpuTensor, _k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_softmax_axis0(_x: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_softmax_axis0_reference(_x: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_aggregate(_prob: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_aggregate_reference(_prob: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_attn_mask(_aux: &GpuTensor, _heads: usize, _queries: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_attn_mask_reference(_aux: &GpuTensor, _heads: usize, _queries: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_add_broadcast(_a: &GpuTensor, _b: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_add_broadcast_reference(_a: &GpuTensor, _b: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_normalise_rows(_x: &GpuTensor, _sub: &GpuTensor, _div: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_normalise_rows_reference(_x: &GpuTensor, _sub: &GpuTensor, _div: &GpuTensor) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_rgb_planes(_rgb: &[u8], _plane: usize, _scale: f32) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_rgb_planes_reference(_rgb: &[u8], _plane: usize, _scale: f32) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_resize_bilinear(_x: &GpuTensor, _in_w: usize, _in_h: usize, _out_w: usize, _out_h: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_resize_bilinear_reference(_x: &GpuTensor, _in_w: usize, _in_h: usize, _out_w: usize, _out_h: usize) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_object_values(_memory: &GpuTensor, _objects: usize, _eps: f32) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_object_values_reference(_memory: &GpuTensor, _objects: usize, _eps: f32) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_sine_encoding(_inv_freq: &[f32], _w: usize, _h: usize, _scale: f32, _eps: f32) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_sine_encoding_reference(_inv_freq: &[f32], _w: usize, _h: usize, _scale: f32, _eps: f32) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_layer_norm(
    _x: &GpuTensor, _cache_namespace: &str, _key: &str, _gamma: &[f32], _beta: &[f32], _eps: f32,
) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}

pub fn gpu_cutie_layer_norm_reference(
    _x: &GpuTensor, _cache_namespace: &str, _key: &str, _gamma: &[f32], _beta: &[f32], _eps: f32,
) -> Result<GpuTensor, String> {
    Err(GPU_UNAVAILABLE.to_string())
}
