// Cutie's device ops (models/cutie), included inside launch.rs's CUDA `mod imp` so they can use
// its private helpers. The stubs for builds without kernels are in
// models/common/src/gpu_cutie_stubs.rs, with the same signatures.
//
// Every op is f32 with exact fp32 arithmetic: Cutie is checked against a PyTorch fp32 oracle with
// TF32 off. Each `gpu_cutie_X` has a host reference `gpu_cutie_X_reference` (download, compute on
// the host, upload): obviously correct and slow. `gpu_cutie_X` runs its kernel in
// kernels/cutie.cu (or cuBLAS SGEMM, which on this backend's handle is never TF32: no math mode is
// ever set); `cutie-validate ops` checks every kernel against its reference.
//
// Layouts: tensors are row-major [rows, cols]. A batch of planar images is [batch*channels,
// height*width], each image's channels contiguous rows, which is the memory order of PyTorch's
// contiguous (batch, channels, height, width).

unsafe extern "C" {
    fn makepad_cuda_cutie_maxpool3x3s2_f32(
        x: *const f32, out: *mut f32, width: u32, height: u32, ow: u32, oh: u32, planes: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_im2col_f32(
        x: *const f32, col: *mut f32, images: u32, channels: u32, width: u32, height: u32, ow: u32, oh: u32,
        k: u32, pad: u32, stride: u32, side_by_side: u32, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_conv_epilogue_f32(
        out: *mut f32, bias: *const f32, residual: *const f32, relu: u32, channels: u32, plane: usize, rows: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_unstack_epilogue_f32(
        src: *const f32, out: *mut f32, bias: *const f32, residual: *const f32, relu: u32, images: u32, channels: u32,
        plane: usize, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_sigmoid_f32(x: *const f32, out: *mut f32, n: usize, stream: cudaStream_t) -> cudaError_t;
    fn makepad_cuda_cutie_gru_f32(
        h: *const f32, values: *const f32, out: *mut f32, objects: u32, channels: u32, plane: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_eca_f32(
        x: *const f32, residual: *const f32, out: *mut f32, scratch: *mut f32, taps: *const f32, k: u32,
        objects: u32, channels: u32, plane: usize, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_others_f32(masks: *const f32, out: *mut f32, n: u32, p: usize, stream: cudaStream_t) -> cudaError_t;
    fn makepad_cuda_cutie_pad_f32(
        x: *const f32, out: *mut f32, planes: u32, w: u32, h: u32, left: u32, right: u32, top: u32, bottom: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_crop_f32(
        x: *const f32, out: *mut f32, planes: u32, w: u32, h: u32, left: u32, top: u32, out_w: u32, out_h: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_similarity_prep_f32(
        mk: *const f32, qk: *const f32, qe: *const f32, mk_sq: *mut f32, qkqe: *mut f32, b_sq: *mut f32,
        a_sq: *mut f32, ck: u32, m: u32, p: u32, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_similarity_epilogue_f32(
        sim: *mut f32, two_ab: *const f32, b_sq: *const f32, a_sq: *const f32, ms: *const f32, inv_scale: f32,
        m: u32, p: u32, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_topk_scratch_entries(m: u32, p: u32, k: u32) -> usize;
    fn makepad_cuda_cutie_topk_softmax_f32(
        sim: *const f32, aff: *mut f32, usage: *mut f32, keys: *mut u32, idx: *mut u32, m: u32, p: u32, k: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_softmax_axis0_f32(x: *const f32, out: *mut f32, m: u32, p: u32, stream: cudaStream_t) -> cudaError_t;
    fn makepad_cuda_cutie_aggregate_f32(prob: *const f32, out: *mut f32, n: u32, p: usize, stream: cudaStream_t) -> cudaError_t;
    fn makepad_cuda_cutie_attn_mask_f32(
        aux: *const f32, out: *mut f32, scratch: *mut u32, n: u32, p: usize, heads: u32, queries: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_normalise_rows_f32(
        x: *const f32, out: *mut f32, sub: *const f32, div: *const f32, rows: u32, plane: usize, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_rgb_planes_f32(rgb: *const u8, out: *mut f32, plane: usize, scale: f32, stream: cudaStream_t) -> cudaError_t;
    fn makepad_cuda_cutie_resize_bilinear_f32(
        x: *const f32, out: *mut f32, planes: u32, in_w: u32, in_h: u32, out_w: u32, out_h: u32, scale_x: f32,
        scale_y: f32, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_add_broadcast_f32(
        a: *const f32, b: *const f32, out: *mut f32, a_len: usize, n: usize, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_object_values_f32(
        memory: *const f32, out: *mut f32, objects: u32, queries: u32, width: u32, eps: f32, stream: cudaStream_t,
    ) -> cudaError_t;
    fn makepad_cuda_cutie_sine_encoding_f32(
        inv_freq: *const f32, out: *mut f32, freqs: u32, w: u32, h: u32, scale: f32, eps: f32, stream: cudaStream_t,
    ) -> cudaError_t;
}

/// The most taps `makepad_cuda_cutie_eca_f32` takes by value (Cutie's ECA has 3 or 5).
const CUTIE_ECA_MAX_TAPS: usize = 15;
/// The largest k of the top-k kernel (Cutie reads 30).
const CUTIE_TOPK_MAX: usize = 64;
/// The im2col scratch a convolution may hold at once; batches beyond it run in groups.
const CUTIE_CONV_SCRATCH_BYTES: usize = 256 << 20;
/// Batches of images this small (output pixels each) run as one GEMM, side by side.
const CUTIE_CONV_SIDE_BY_SIDE_PIXELS: usize = 4096;

fn cutie_check_f32(op: &str, x: &GpuTensor) -> Result<(), String> {
    if x.half {
        return Err(format!("{op} is f32-only"));
    }
    Ok(())
}

/// A dimension as the kernels take it.
fn cutie_u32(op: &str, v: usize) -> Result<u32, String> {
    u32::try_from(v).map_err(|_| format!("{op}: {v} does not fit the kernel's 32-bit sizes"))
}

/// Zeroes `count` f32 at `ptr` on the backend's stream.
fn cutie_zero(backend: &CudaDenseLinearBackend, ptr: *mut f32, count: usize) -> Result<(), String> {
    if count == 0 {
        return Ok(());
    }
    let ptr = NonNull::new(ptr.cast::<c_void>()).ok_or("cutie zero: null pointer")?;
    unsafe { crate::memset_async(ptr, 0, count * size_of::<f32>(), backend.stream) }.map_err(|e| e.to_string())
}

/// c [m, n] = op(a) @ op(b), every matrix row-major f32 on the device (`a_cols`, `b_cols` the
/// stored widths): cuBLAS SGEMM, fp32 multiply-accumulate (the handle's default math mode,
/// never TF32). Row-major C is column-major C^T = op(b)^T op(a)^T, so b goes first.
#[allow(clippy::too_many_arguments)]
fn cutie_sgemm(
    backend: &CudaDenseLinearBackend, a: *const f32, a_cols: usize, trans_a: bool, b: *const f32, b_cols: usize,
    trans_b: bool, m: usize, n: usize, k: usize, c: *mut f32,
) -> Result<(), String> {
    if m == 0 || n == 0 {
        return Ok(());
    }
    if k == 0 {
        return cutie_zero(backend, c, m * n);
    }
    let dim = |v: usize| i32::try_from(v).map_err(|_| format!("cutie sgemm: {v} exceeds cuBLAS's sizes"));
    let op = |t: bool| if t { crate::CUBLAS_OP_T } else { crate::CUBLAS_OP_N };
    crate::cublas_sgemm(
        backend.blas, op(trans_b), op(trans_a), dim(n)?, dim(m)?, dim(k)?, &1.0f32, b, dim(b_cols)?, a, dim(a_cols)?,
        &0.0f32, c, dim(n)?,
    )
    .map_err(|e| e.to_string())
}

/// PyTorch's order for a sum or product over a dimension that is not the innermost (Cutie's
/// objects, the top-k picks), as the kernels' `cutie_reduce4`: four accumulators from `ident`,
/// term o into o % 4, then ((a0 op a1) op a2) op a3 (Reduce.cuh's thread_reduce_impl with
/// vt0 = 4, one thread an output, which holds for fewer than 64 terms). From 5 terms on it rounds
/// differently from a sequential sum, and PyTorch's results are this order's to the bit.
fn cutie_reduce4(terms: impl IntoIterator<Item = f32>, ident: f32, op: impl Fn(f32, f32) -> f32) -> f32 {
    let mut acc = [ident; 4];
    for (o, t) in terms.into_iter().enumerate() {
        acc[o % 4] = op(acc[o % 4], t);
    }
    op(op(op(acc[0], acc[1]), acc[2]), acc[3])
}

/// 1 / sqrt(ck) as PyTorch applies `x / math.sqrt(ck)` on CUDA (div_true with a CPU scalar): the
/// reciprocal of the Python float taken in f64, rounded to f32, then a multiply (torch 2.11 gives
/// exactly this for every ck tried; dividing by sqrt(ck) in f32 differs at every one but 64).
fn cutie_inv_sqrt(ck: usize) -> f32 {
    (1.0 / (ck as f64).sqrt()) as f32
}

/// The larger of two values, a NaN in either winning (PyTorch's `max_propagate_nan`).
fn cutie_max_nan(a: f32, b: f32) -> f32 {
    if b.is_nan() {
        b
    } else if a.is_nan() {
        a
    } else {
        a.max(b)
    }
}

/// PyTorch's radix key of a float (`TopKTypeConfig<float>::convert`), as the top-k kernel's: the
/// keys' unsigned order is torch.topk's order of the values, so -0 ranks below +0 and every NaN
/// (either sign) above every number.
fn cutie_topk_key(v: f32) -> u32 {
    if v.is_nan() {
        return u32::MAX;
    }
    let x = v.to_bits();
    x ^ if x & 0x8000_0000 != 0 { u32::MAX } else { 0x8000_0000 }
}

/// A convolution's output size along one axis (PyTorch's floor rule).
pub fn cutie_conv_out(size: usize, k: usize, pad: usize, stride: usize) -> usize {
    (size + 2 * pad - k) / stride + 1
}

// --- Convolution ---

/// Conv2d over a batch of planar images: `x` is [batch*in_channels, in_h*in_w], `weights` PyTorch's
/// [out][in][k][k], `bias` per output channel (None: no bias). Square kernel, symmetric padding and
/// stride. Out: [batch*out_channels, out_h*out_w], out_h = (in_h + 2*pad - k)/stride + 1. Weights
/// then bias are cached on the device as one buffer under `{cache_namespace}::{key}::cutie_conv`
/// (one key per layer).
///
/// The images' columns (f32 im2col) times the [out, in*k*k] weights in cuBLAS (CUDA_R_32F,
/// CUBLAS_COMPUTE_32F, so fp32 multiply-accumulate), then the bias. Several small images (stride
/// 8 and 16) go side by side through one GEMM, then back to their own planes; otherwise one
/// strided-batched GEMM over the images (a 1x1 stride-1 convolution's input already is its
/// columns).
pub fn gpu_cutie_conv2d(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
) -> Result<GpuTensor, String> {
    gpu_cutie_conv2d_fused(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride, None, false)
}

/// `gpu_cutie_conv2d`, then `residual` added (when given: the output's shape, [batch*out_channels,
/// out_h*out_w]) and a ReLU (when `relu`), in the pass that adds the bias: relu((conv + bias) +
/// residual), each add rounded as `gpu_add` rounds it and the ReLU `gpu_birefnet_relu`'s (fmaxf(x,
/// 0)), so the result is those ops' to the bit, the output read and written once instead of three
/// times (a ResNet block's tail). The reference is the plain convolution, then those two ops.
#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
    residual: Option<&GpuTensor>, relu: bool,
) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_conv2d";
    cutie_check_f32(op, x)?;
    if batch == 0 || x.rows == 0 || x.rows % batch != 0 || x.cols != in_w * in_h || k == 0 || stride == 0 || in_w + 2 * pad < k || in_h + 2 * pad < k {
        return Err(format!("{op} {key}: x {}x{} for batch {batch} at {in_w}x{in_h}, k {k} pad {pad} stride {stride}", x.rows, x.cols));
    }
    let in_channels = x.rows / batch;
    let kdim = in_channels * k * k;
    if weights.len() != out_channels * kdim || bias.is_some_and(|b| b.len() != out_channels) {
        return Err(format!("{op} {key}: {} weights for {out_channels}x{in_channels}x{k}x{k}", weights.len()));
    }
    let (ow, oh) = (cutie_conv_out(in_w, k, pad, stride), cutie_conv_out(in_h, k, pad, stride));
    let opix = ow * oh;
    if let Some(r) = residual {
        cutie_check_f32(op, r)?;
        if (r.rows, r.cols) != (batch * out_channels, opix) {
            return Err(format!("{op} {key}: residual {}x{} for an output {}x{opix}", r.rows, r.cols, batch * out_channels));
        }
    }
    let dim = |v: usize| i32::try_from(v).map_err(|_| format!("{op} {key}: {v} exceeds cuBLAS's sizes"));
    let (m, n, kd) = (dim(opix)?, dim(out_channels)?, dim(kdim)?);
    let d = |v: usize| cutie_u32(op, v);
    let (channels_u, w_u, h_u, ow_u, oh_u, k_u, pad_u, stride_u) =
        (d(in_channels)?, d(in_w)?, d(in_h)?, d(ow)?, d(oh)?, d(k)?, d(pad)?, d(stride)?);
    let (batch_u, out_channels_u) = (d(batch)?, d(out_channels)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        // Weights then bias, one cached buffer: no host upload after the first call.
        let qualified_key = format!("{cache_namespace}::{key}::cutie_conv");
        let floats = weights.len() + bias.map_or(0, |b| b.len());
        backend.cached_weight_buffer(&qualified_key, floats * size_of::<f32>(), || {
            let mut raw = Vec::with_capacity(floats * size_of::<f32>());
            for v in weights.iter().chain(bias.into_iter().flatten()) {
                raw.extend_from_slice(&v.to_le_bytes());
            }
            Ok(raw)
        })?;
        let w_ptr = backend
            .weight_buffers
            .get(&qualified_key)
            .ok_or_else(|| format!("missing cached cutie conv weights {qualified_key}"))?
            .ptr
            .as_ptr()
            .cast::<f32>();
        let out = GpuTensor::from_pool(batch * out_channels, opix)?;
        let x_ptr = x.device_ptr()?;
        let out_ptr = out.device_ptr()?;
        let bias_ptr = if bias.is_some() { unsafe { w_ptr.add(weights.len()) } } else { std::ptr::null() };
        let residual_ptr = match residual {
            Some(r) => r.device_ptr()?.cast_const(),
            None => std::ptr::null(),
        };
        let per_image = kdim * opix;
        let side_by_side =
            batch > 1 && opix <= CUTIE_CONV_SIDE_BY_SIDE_PIXELS && batch * per_image * size_of::<f32>() <= CUTIE_CONV_SCRATCH_BYTES;
        if side_by_side {
            // Small images: one GEMM over every image's columns side by side keeps cuBLAS's
            // tiles full, then each image's channels are put back in place with the epilogue.
            let wide = dim(batch * opix)?;
            let col = GpuTensor::from_pool(kdim, batch * opix)?;
            let stacked = GpuTensor::from_pool(out_channels, batch * opix)?;
            gpu_check(unsafe {
                makepad_cuda_cutie_im2col_f32(
                    x_ptr, col.device_ptr()?, batch_u, channels_u, w_u, h_u, ow_u, oh_u, k_u, pad_u, stride_u, 1,
                    backend.stream,
                )
            })?;
            unsafe {
                crate::cublas_gemm_ex(
                    backend.blas, crate::CUBLAS_OP_N, crate::CUBLAS_OP_N, wide, n, kd, &1.0f32,
                    col.device_ptr()?.cast::<c_void>(), crate::CUDA_R_32F, wide, w_ptr.cast::<c_void>(), crate::CUDA_R_32F, kd,
                    &0.0f32, stacked.device_ptr()?.cast::<c_void>(), crate::CUDA_R_32F, wide, crate::CUBLAS_COMPUTE_32F,
                    crate::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(|e| format!("{op} {key}: {e}"))?;
            gpu_check(unsafe {
                makepad_cuda_cutie_unstack_epilogue_f32(
                    stacked.device_ptr()?, out_ptr, bias_ptr, residual_ptr, relu as u32, batch_u, out_channels_u, opix,
                    backend.stream,
                )
            })?;
            return Ok(out);
        }
        let direct = k == 1 && stride == 1 && pad == 0;
        // Images per GEMM: all of them, unless their columns outgrow the scratch.
        let group = if direct { batch } else { (CUTIE_CONV_SCRATCH_BYTES / size_of::<f32>() / per_image.max(1)).clamp(1, batch) };
        let col = if direct { None } else { Some(GpuTensor::from_pool(group * kdim, opix)?) };
        let mut first = 0;
        while first < batch {
            let count = group.min(batch - first);
            let src = unsafe { x_ptr.add(first * in_channels * in_w * in_h) };
            let a = match &col {
                Some(col) => {
                    gpu_check(unsafe {
                        makepad_cuda_cutie_im2col_f32(
                            src, col.device_ptr()?, count as u32, channels_u, w_u, h_u, ow_u, oh_u, k_u, pad_u, stride_u, 0,
                            backend.stream,
                        )
                    })?;
                    col.device_ptr()?
                }
                None => src,
            };
            unsafe {
                crate::cublas_gemm_strided_batched_ex(
                    backend.blas, crate::CUBLAS_OP_N, crate::CUBLAS_OP_N, m, n, kd, &1.0f32,
                    a.cast::<c_void>(), crate::CUDA_R_32F, m, per_image as i64,
                    w_ptr.cast::<c_void>(), crate::CUDA_R_32F, kd, 0,
                    &0.0f32, out_ptr.add(first * out_channels * opix).cast::<c_void>(), crate::CUDA_R_32F, m,
                    (out_channels * opix) as i64, count as i32, crate::CUBLAS_COMPUTE_32F, crate::CUBLAS_GEMM_DEFAULT,
                )
            }
            .map_err(|e| format!("{op} {key}: {e}"))?;
            first += count;
        }
        if bias.is_some() || residual.is_some() || relu {
            gpu_check(unsafe {
                makepad_cuda_cutie_conv_epilogue_f32(
                    out_ptr, bias_ptr, residual_ptr, relu as u32, out_channels_u, opix, batch * out_channels, backend.stream,
                )
            })?;
        }
        Ok(out)
    })
}

/// Reference: the existing direct fp32 kernel (`gpu_conv2d_planar_strided`) once per image.
pub fn gpu_cutie_conv2d_reference(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_conv2d", x)?;
    if batch == 0 || x.rows % batch != 0 || x.cols != in_w * in_h {
        return Err(format!("gpu_cutie_conv2d {key}: x {}x{} for batch {batch} at {in_w}x{in_h}", x.rows, x.cols));
    }
    let in_channels = x.rows / batch;
    if weights.len() != out_channels * in_channels * k * k {
        return Err(format!("gpu_cutie_conv2d {key}: {} weights for {out_channels}x{in_channels}x{k}x{k}", weights.len()));
    }
    let zeros;
    let bias = match bias {
        Some(b) => b,
        None => {
            zeros = vec![0.0f32; out_channels];
            &zeros
        }
    };
    let (ow, oh) = (cutie_conv_out(in_w, k, pad, stride), cutie_conv_out(in_h, k, pad, stride));
    let mut outs = Vec::with_capacity(batch);
    for b in 0..batch {
        let view = gpu_slice_rows(x, b * in_channels, in_channels)?;
        outs.push(gpu_conv2d_planar_strided(
            &view, in_w, in_h, ow, oh, cache_namespace, key, weights, bias, out_channels, k, k, pad, pad, stride, stride,
        )?);
    }
    if batch == 1 {
        return Ok(outs.pop().unwrap());
    }
    let parts: Vec<&GpuTensor> = outs.iter().collect();
    gpu_concat_rows_many(&parts)
}

/// Reference: `gpu_cutie_conv2d_reference`, then `gpu_add` with the residual and
/// `gpu_birefnet_relu`, the separate ops the fused epilogue replaces.
#[allow(clippy::too_many_arguments)]
pub fn gpu_cutie_conv2d_fused_reference(
    x: &GpuTensor, batch: usize, in_w: usize, in_h: usize, cache_namespace: &str, key: &str,
    weights: &[f32], bias: Option<&[f32]>, out_channels: usize, k: usize, pad: usize, stride: usize,
    residual: Option<&GpuTensor>, relu: bool,
) -> Result<GpuTensor, String> {
    let mut out = gpu_cutie_conv2d_reference(x, batch, in_w, in_h, cache_namespace, key, weights, bias, out_channels, k, pad, stride)?;
    if let Some(r) = residual {
        out = gpu_add(&out, r)?;
    }
    if relu {
        out = gpu_birefnet_relu(&out)?;
    }
    Ok(out)
}

// --- Pooling ---

/// MaxPool2d(3, stride 2, padding 1) on planar [planes, h*w]. Out [planes, oh*ow], oh = (h-1)/2+1.
/// A NaN in a window wins, as in PyTorch.
pub fn gpu_cutie_maxpool3x3s2(x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_maxpool3x3s2", x)?;
    if x.cols != w * h || w == 0 || h == 0 {
        return Err("gpu_cutie_maxpool3x3s2 shape mismatch".into());
    }
    let (ow, oh) = (cutie_conv_out(w, 3, 1, 2), cutie_conv_out(h, 3, 1, 2));
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, ow * oh)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_maxpool3x3s2_f32(
                x.device_ptr()?, out.device_ptr()?, w as u32, h as u32, ow as u32, oh as u32, x.rows as u32,
                backend.stream,
            )
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_maxpool3x3s2_reference(x: &GpuTensor, w: usize, h: usize) -> Result<GpuTensor, String> {
    let v = gpu_download(x)?;
    let (ow, oh) = (cutie_conv_out(w, 3, 1, 2), cutie_conv_out(h, 3, 1, 2));
    let mut out = vec![0.0f32; x.rows * ow * oh];
    for p in 0..x.rows {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut best = f32::NEG_INFINITY;
                for ky in 0..3 {
                    let y = (oy * 2 + ky) as isize - 1;
                    if y < 0 || y >= h as isize {
                        continue;
                    }
                    for kx in 0..3 {
                        let xx = (ox * 2 + kx) as isize - 1;
                        if xx < 0 || xx >= w as isize {
                            continue;
                        }
                        let tap = v[p * w * h + y as usize * w + xx as usize];
                        if tap > best || tap.is_nan() {
                            best = tap;
                        }
                    }
                }
                out[p * ow * oh + oy * ow + ox] = best;
            }
        }
    }
    gpu_upload(&out, x.rows, ow * oh)
}

// --- Elementwise ---

/// 1 / (1 + exp(-x)), elementwise.
pub fn gpu_cutie_sigmoid(x: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_sigmoid", x)?;
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, x.cols)?;
        gpu_check(unsafe { makepad_cuda_cutie_sigmoid_f32(x.device_ptr()?, out.device_ptr()?, x.rows * x.cols, backend.stream) })?;
        Ok(out)
    })
}

pub fn gpu_cutie_sigmoid_reference(x: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_sigmoid", x)?;
    let v: Vec<f32> = gpu_download(x)?.into_iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
    gpu_upload(&v, x.rows, x.cols)
}

/// Cutie's GRU-like sensory update, per object: `values` [n*3c, p] holds each object's forget,
/// update and new-value rows (c each); `h` [n*c, p]. new_h = sigmoid(f)*h*(1-sigmoid(u)) +
/// sigmoid(u)*tanh(v), evaluated in that order.
pub fn gpu_cutie_gru(h: &GpuTensor, values: &GpuTensor, n: usize, c: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_gru", h)?;
    cutie_check_f32("gpu_cutie_gru", values)?;
    if h.rows != n * c || values.rows != n * 3 * c || h.cols != values.cols {
        return Err("gpu_cutie_gru shape mismatch".into());
    }
    let (n_u, c_u) = (cutie_u32("gpu_cutie_gru", n)?, cutie_u32("gpu_cutie_gru", c)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(h.rows, h.cols)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_gru_f32(h.device_ptr()?, values.device_ptr()?, out.device_ptr()?, n_u, c_u, h.cols, backend.stream)
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_gru_reference(h: &GpuTensor, values: &GpuTensor, n: usize, c: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_gru", h)?;
    if h.rows != n * c || values.rows != n * 3 * c || h.cols != values.cols {
        return Err("gpu_cutie_gru shape mismatch".into());
    }
    let p = h.cols;
    let hv = gpu_download(h)?;
    let vv = gpu_download(values)?;
    let mut out = vec![0.0f32; n * c * p];
    for o in 0..n {
        for ch in 0..c {
            for i in 0..p {
                let f = 1.0 / (1.0 + (-vv[((o * 3 * c) + ch) * p + i]).exp());
                let u = 1.0 / (1.0 + (-vv[((o * 3 * c) + c + ch) * p + i]).exp());
                let nv = vv[((o * 3 * c) + 2 * c + ch) * p + i].tanh();
                let hh = hv[(o * c + ch) * p + i];
                out[(o * c + ch) * p + i] = f * hh * (1.0 - u) + u * nv;
            }
        }
    }
    gpu_upload(&out, n * c, p)
}

/// The tail of Cutie's channel-attention block (ECA): per image b of `x` [n*c, p], the mean of
/// each channel, a 1-D convolution over channels (`conv_w`, odd length k, zero padding (k-1)/2,
/// no bias), a sigmoid, then x * that + `residual` ([n*c, p]).
pub fn gpu_cutie_eca(x: &GpuTensor, residual: &GpuTensor, n: usize, c: usize, conv_w: &[f32]) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_eca", x)?;
    cutie_check_f32("gpu_cutie_eca", residual)?;
    if x.rows != n * c || residual.rows != x.rows || residual.cols != x.cols || conv_w.len() % 2 == 0 {
        return Err("gpu_cutie_eca shape mismatch".into());
    }
    if conv_w.len() > CUTIE_ECA_MAX_TAPS {
        // Never for Cutie (3 or 5 taps): the kernel takes its taps by value.
        return gpu_cutie_eca_reference(x, residual, n, c, conv_w);
    }
    let (n_u, c_u) = (cutie_u32("gpu_cutie_eca", n)?, cutie_u32("gpu_cutie_eca", c)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, x.cols)?;
        // Each channel's mean, then its scale.
        let scratch = GpuTensor::from_pool(2, x.rows.max(1))?;
        gpu_check(unsafe {
            makepad_cuda_cutie_eca_f32(
                x.device_ptr()?, residual.device_ptr()?, out.device_ptr()?, scratch.device_ptr()?, conv_w.as_ptr(),
                conv_w.len() as u32, n_u, c_u, x.cols, backend.stream,
            )
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_eca_reference(x: &GpuTensor, residual: &GpuTensor, n: usize, c: usize, conv_w: &[f32]) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_eca", x)?;
    if x.rows != n * c || residual.rows != x.rows || residual.cols != x.cols || conv_w.len() % 2 == 0 {
        return Err("gpu_cutie_eca shape mismatch".into());
    }
    let p = x.cols;
    let xv = gpu_download(x)?;
    let rv = gpu_download(residual)?;
    let k = conv_w.len();
    let pad = (k - 1) / 2;
    let mut out = vec![0.0f32; n * c * p];
    for o in 0..n {
        let means: Vec<f32> = (0..c)
            .map(|ch| {
                let row = &xv[(o * c + ch) * p..(o * c + ch + 1) * p];
                (row.iter().map(|&v| v as f64).sum::<f64>() / p as f64) as f32
            })
            .collect();
        for ch in 0..c {
            let mut y = 0.0f32;
            for j in 0..k {
                let src = ch as isize + j as isize - pad as isize;
                if src >= 0 && (src as usize) < c {
                    y += conv_w[j] * means[src as usize];
                }
            }
            let s = 1.0 / (1.0 + (-y).exp());
            for i in 0..p {
                let at = (o * c + ch) * p + i;
                out[at] = xv[at] * s + rv[at];
            }
        }
    }
    gpu_upload(&out, n * c, p)
}

/// For each object's mask in `masks` [n, p], the sum of the other objects' masks, clamped to 0..1
/// (Cutie's `_get_others`: all objects summed in PyTorch's order, `cutie_reduce4`, then the
/// object's own subtracted).
pub fn gpu_cutie_others(masks: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_others", masks)?;
    let n = cutie_u32("gpu_cutie_others", masks.rows)?;
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(masks.rows, masks.cols)?;
        gpu_check(unsafe { makepad_cuda_cutie_others_f32(masks.device_ptr()?, out.device_ptr()?, n, masks.cols, backend.stream) })?;
        Ok(out)
    })
}

pub fn gpu_cutie_others_reference(masks: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_others", masks)?;
    let (n, p) = (masks.rows, masks.cols);
    let v = gpu_download(masks)?;
    let mut out = vec![0.0f32; n * p];
    for i in 0..p {
        let sum = cutie_reduce4((0..n).map(|o| v[o * p + i]), 0.0, |a, b| a + b);
        for o in 0..n {
            out[o * p + i] = (sum - v[o * p + i]).clamp(0.0, 1.0);
        }
    }
    gpu_upload(&out, n, p)
}

// --- Shapes ---

/// Zero padding of planar [planes, h*w] by (left, right, top, bottom). Out [planes, (h+t+b)*(w+l+r)].
pub fn gpu_cutie_pad(x: &GpuTensor, w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_pad", x)?;
    if x.cols != w * h {
        return Err("gpu_cutie_pad shape mismatch".into());
    }
    let (nw, nh) = (w + left + right, h + top + bottom);
    let d = |v: usize| cutie_u32("gpu_cutie_pad", v);
    let (planes, wu, hu, l, r, t, b) = (d(x.rows)?, d(w)?, d(h)?, d(left)?, d(right)?, d(top)?, d(bottom)?);
    d(nw)?;
    d(nh)?;
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, nw * nh)?;
        gpu_check(unsafe { makepad_cuda_cutie_pad_f32(x.device_ptr()?, out.device_ptr()?, planes, wu, hu, l, r, t, b, backend.stream) })?;
        Ok(out)
    })
}

pub fn gpu_cutie_pad_reference(x: &GpuTensor, w: usize, h: usize, left: usize, right: usize, top: usize, bottom: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_pad", x)?;
    if x.cols != w * h {
        return Err("gpu_cutie_pad shape mismatch".into());
    }
    let (nw, nh) = (w + left + right, h + top + bottom);
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; x.rows * nw * nh];
    for p in 0..x.rows {
        for y in 0..h {
            let src = &v[p * w * h + y * w..p * w * h + (y + 1) * w];
            let at = p * nw * nh + (y + top) * nw + left;
            out[at..at + w].copy_from_slice(src);
        }
    }
    gpu_upload(&out, x.rows, nw * nh)
}

/// A window of planar [planes, h*w]: `out_w` x `out_h` from (`left`, `top`).
pub fn gpu_cutie_crop(x: &GpuTensor, w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_crop", x)?;
    if x.cols != w * h || left + out_w > w || top + out_h > h {
        return Err("gpu_cutie_crop shape mismatch".into());
    }
    let d = |v: usize| cutie_u32("gpu_cutie_crop", v);
    let (planes, wu, hu, l, t, owu, ohu) = (d(x.rows)?, d(w)?, d(h)?, d(left)?, d(top)?, d(out_w)?, d(out_h)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, out_w * out_h)?;
        gpu_check(unsafe { makepad_cuda_cutie_crop_f32(x.device_ptr()?, out.device_ptr()?, planes, wu, hu, l, t, owu, ohu, backend.stream) })?;
        Ok(out)
    })
}

pub fn gpu_cutie_crop_reference(x: &GpuTensor, w: usize, h: usize, left: usize, top: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_crop", x)?;
    if x.cols != w * h || left + out_w > w || top + out_h > h {
        return Err("gpu_cutie_crop shape mismatch".into());
    }
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; x.rows * out_w * out_h];
    for p in 0..x.rows {
        for y in 0..out_h {
            let src = p * w * h + (y + top) * w + left;
            let at = p * out_w * out_h + y * out_w;
            out[at..at + out_w].copy_from_slice(&v[src..src + out_w]);
        }
    }
    gpu_upload(&out, x.rows, out_w * out_h)
}

// --- Matrix products ---

/// op(a) @ op(b), op transposing when asked: op(a) [m, k], op(b) [k, n]; out [m, n]. fp32
/// multiply-accumulate (no TF32).
pub fn gpu_cutie_matmul(a: &GpuTensor, trans_a: bool, b: &GpuTensor, trans_b: bool) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_matmul", a)?;
    cutie_check_f32("gpu_cutie_matmul", b)?;
    let (m, k) = if trans_a { (a.cols, a.rows) } else { (a.rows, a.cols) };
    let (k2, n) = if trans_b { (b.cols, b.rows) } else { (b.rows, b.cols) };
    if k != k2 {
        return Err(format!("gpu_cutie_matmul inner dims {k} != {k2}"));
    }
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(m, n)?;
        cutie_sgemm(backend, a.device_ptr()?, a.cols, trans_a, b.device_ptr()?, b.cols, trans_b, m, n, k, out.device_ptr()?)?;
        Ok(out)
    })
}

pub fn gpu_cutie_matmul_reference(a: &GpuTensor, trans_a: bool, b: &GpuTensor, trans_b: bool) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_matmul", a)?;
    cutie_check_f32("gpu_cutie_matmul", b)?;
    let (m, k) = if trans_a { (a.cols, a.rows) } else { (a.rows, a.cols) };
    let (k2, n) = if trans_b { (b.cols, b.rows) } else { (b.rows, b.cols) };
    if k != k2 {
        return Err(format!("gpu_cutie_matmul inner dims {k} != {k2}"));
    }
    let av = gpu_download(a)?;
    let bv = gpu_download(b)?;
    let at = |i: usize, j: usize| if trans_a { av[j * a.cols + i] } else { av[i * a.cols + j] };
    let bt = |i: usize, j: usize| if trans_b { bv[j * b.cols + i] } else { bv[i * b.cols + j] };
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for l in 0..k {
                s += at(i, l) as f64 * bt(l, j) as f64;
            }
            out[i * n + j] = s as f32;
        }
    }
    gpu_upload(&out, m, n)
}

// --- Memory reading ---

/// Cutie's anisotropic L2 similarity (XMem's appendix): memory keys `mk` [ck, m], shrinkage `ms`
/// [1, m] (None: 1), query keys `qk` [ck, p], selection `qe` [ck, p] (None: plain L2 as STCN).
/// Out [m, p]: (-a_sq + two_ab - b_sq) * ms / sqrt(ck), with a_sq = mk^2 . qe, two_ab =
/// 2 mk . (qk*qe), b_sq = sum_c qe*qk^2 (without qe: a_sq = |mk|^2, two_ab = 2 mk . qk, no b_sq).
///
/// As PyTorch forms it: a_sq and two_ab are GEMMs over the channels (cuBLAS SGEMM), each rounded
/// before they are combined, so cancellation between them is PyTorch's own; the division by the
/// Python scalar sqrt(ck) is PyTorch's multiply by its reciprocal (`cutie_inv_sqrt`), the same as
/// dividing at Cutie's ck of 64. The channel sums (b_sq, and a_sq without qe) are
/// sequential, PyTorch's split over warps: fp32 rounding apart.
pub fn gpu_cutie_similarity(mk: &GpuTensor, ms: Option<&GpuTensor>, qk: &GpuTensor, qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_similarity";
    cutie_check_f32(op, mk)?;
    cutie_check_f32(op, qk)?;
    for t in ms.iter().chain(qe.iter()) {
        cutie_check_f32(op, t)?;
    }
    let (ck, m, p) = (mk.rows, mk.cols, qk.cols);
    if qk.rows != ck || qe.is_some_and(|e| e.rows != ck || e.cols != p) || ms.is_some_and(|s| s.rows * s.cols != m) {
        return Err("gpu_cutie_similarity shape mismatch".into());
    }
    let (ck_u, m_u, p_u) = (cutie_u32(op, ck)?, cutie_u32(op, m)?, cutie_u32(op, p)?);
    let inv_scale = cutie_inv_sqrt(ck);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(m, p)?;
        let ms_ptr = match ms {
            Some(s) => s.device_ptr()?.cast_const(),
            None => std::ptr::null(),
        };
        let null = std::ptr::null_mut::<f32>();
        match qe {
            Some(qe) => {
                let mk_sq = GpuTensor::from_pool(ck, m)?;
                let qkqe = GpuTensor::from_pool(ck, p)?;
                let b_sq = GpuTensor::from_pool(1, p)?;
                let two_ab = GpuTensor::from_pool(m, p)?;
                gpu_check(unsafe {
                    makepad_cuda_cutie_similarity_prep_f32(
                        mk.device_ptr()?, qk.device_ptr()?, qe.device_ptr()?, mk_sq.device_ptr()?, qkqe.device_ptr()?,
                        b_sq.device_ptr()?, null, ck_u, m_u, p_u, backend.stream,
                    )
                })?;
                // a_sq = (mk^2)^T qe into the output; two_ab's product mk^T (qk*qe) beside it.
                cutie_sgemm(backend, mk_sq.device_ptr()?, m, true, qe.device_ptr()?, p, false, m, p, ck, out.device_ptr()?)?;
                cutie_sgemm(backend, mk.device_ptr()?, m, true, qkqe.device_ptr()?, p, false, m, p, ck, two_ab.device_ptr()?)?;
                gpu_check(unsafe {
                    makepad_cuda_cutie_similarity_epilogue_f32(
                        out.device_ptr()?, two_ab.device_ptr()?, b_sq.device_ptr()?, std::ptr::null(), ms_ptr, inv_scale,
                        m_u, p_u, backend.stream,
                    )
                })?;
            }
            None => {
                let a_sq = GpuTensor::from_pool(1, m)?;
                gpu_check(unsafe {
                    makepad_cuda_cutie_similarity_prep_f32(
                        mk.device_ptr()?, qk.device_ptr()?, std::ptr::null(), null, null, null, a_sq.device_ptr()?, ck_u,
                        m_u, p_u, backend.stream,
                    )
                })?;
                // mk^T qk into the output.
                cutie_sgemm(backend, mk.device_ptr()?, m, true, qk.device_ptr()?, p, false, m, p, ck, out.device_ptr()?)?;
                gpu_check(unsafe {
                    makepad_cuda_cutie_similarity_epilogue_f32(
                        out.device_ptr()?, std::ptr::null(), std::ptr::null(), a_sq.device_ptr()?, ms_ptr, inv_scale, m_u,
                        p_u, backend.stream,
                    )
                })?;
            }
        }
        Ok(out)
    })
}

pub fn gpu_cutie_similarity_reference(mk: &GpuTensor, ms: Option<&GpuTensor>, qk: &GpuTensor, qe: Option<&GpuTensor>) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_similarity", mk)?;
    let (ck, m, p) = (mk.rows, mk.cols, qk.cols);
    if qk.rows != ck || qe.is_some_and(|e| e.rows != ck || e.cols != p) || ms.is_some_and(|s| s.rows * s.cols != m) {
        return Err("gpu_cutie_similarity shape mismatch".into());
    }
    let mkv = gpu_download(mk)?;
    let qkv = gpu_download(qk)?;
    let qev = qe.map(gpu_download).transpose()?;
    let msv = ms.map(gpu_download).transpose()?;
    let inv_scale = cutie_inv_sqrt(ck);
    let mut out = vec![0.0f32; m * p];
    for j in 0..m {
        for i in 0..p {
            let sim = if let Some(qev) = &qev {
                let (mut a_sq, mut two_ab, mut b_sq) = (0.0f64, 0.0f64, 0.0f64);
                for c in 0..ck {
                    let (mkc, qkc, qec) = (mkv[c * m + j] as f64, qkv[c * p + i] as f64, qev[c * p + i] as f64);
                    a_sq += mkc * mkc * qec;
                    two_ab += mkc * (qkc * qec);
                    b_sq += qec * qkc * qkc;
                }
                -(a_sq as f32) + 2.0 * (two_ab as f32) - b_sq as f32
            } else {
                let (mut a_sq, mut two_ab) = (0.0f64, 0.0f64);
                for c in 0..ck {
                    let (mkc, qkc) = (mkv[c * m + j] as f64, qkv[c * p + i] as f64);
                    a_sq += mkc * mkc;
                    two_ab += mkc * qkc;
                }
                -(a_sq as f32) + 2.0 * (two_ab as f32)
            };
            out[j * p + i] = match &msv {
                Some(s) => sim * s[j] * inv_scale,
                None => sim * inv_scale,
            };
        }
    }
    gpu_upload(&out, m, p)
}

/// Cutie's top-k softmax down each column of `sim` [m, p]: the k largest (k = min(k, m)), their
/// exp normalised to sum 1 (no max subtracted, as Cutie's `do_softmax`), every other entry 0.
/// Also each row's sum of the result ([1, m]): how much each memory token was used.
///
/// The picks are torch.topk's: values in PyTorch's radix-key order (`cutie_topk_key`: -0 below
/// +0, a NaN of either sign above every number, so a NaN similarity is picked and its column's
/// affinities come out NaN, as PyTorch's do), equal keys by the lower row (torch 2.11 takes the
/// same set at 336 to 30000 rows). The denominator sums the picks best first in PyTorch's order
/// (`cutie_reduce4`), so for k under 64 (Cutie's is 30) the affinities are PyTorch's to the bit
/// given the same similarities. The usage sums each row in another order than PyTorch's innermost
/// reduce: fp32 rounding apart.
///
/// The kernel splits each column's rows into slices, keeps each slice's top k (one thread a
/// column and slice, reads coalesced across columns), merges them per column, then the softmax
/// and the usage.
pub fn gpu_cutie_topk_softmax(sim: &GpuTensor, k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    let op = "gpu_cutie_topk_softmax";
    cutie_check_f32(op, sim)?;
    let (m, p) = (sim.rows, sim.cols);
    let k = k.min(m);
    if k > CUTIE_TOPK_MAX {
        // Never for Cutie (top_k 30): the kernel keeps at most 64 a column in registers.
        return gpu_cutie_topk_softmax_reference(sim, k);
    }
    let (m_u, p_u, k_u) = (cutie_u32(op, m)?, cutie_u32(op, p)?, k as u32);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let aff = GpuTensor::from_pool(m, p)?;
        let usage = GpuTensor::from_pool(1, m)?;
        cutie_zero(backend, aff.device_ptr()?, m * p)?;
        // Each slice's best k per column: their keys and rows (u32 in f32-sized buffers).
        let entries = unsafe { makepad_cuda_cutie_topk_scratch_entries(m_u, p_u, k_u) }.max(1);
        let keys = GpuTensor::from_pool(entries, 1)?;
        let idx = GpuTensor::from_pool(entries, 1)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_topk_softmax_f32(
                sim.device_ptr()?, aff.device_ptr()?, usage.device_ptr()?, keys.device_ptr()?.cast::<u32>(),
                idx.device_ptr()?.cast::<u32>(), m_u, p_u, k_u, backend.stream,
            )
        })?;
        Ok((aff, usage))
    })
}

pub fn gpu_cutie_topk_softmax_reference(sim: &GpuTensor, k: usize) -> Result<(GpuTensor, GpuTensor), String> {
    cutie_check_f32("gpu_cutie_topk_softmax", sim)?;
    let (m, p) = (sim.rows, sim.cols);
    let k = k.min(m);
    let v = gpu_download(sim)?;
    let mut aff = vec![0.0f32; m * p];
    let mut col: Vec<(u32, usize)> = Vec::with_capacity(m);
    for i in 0..p {
        col.clear();
        col.extend((0..m).map(|j| (cutie_topk_key(v[j * p + i]), j)));
        // torch.topk's order: the largest key first, equal keys by the lower row.
        col.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let exps: Vec<f32> = col[..k].iter().map(|&(_, j)| v[j * p + i].exp()).collect();
        let sum = cutie_reduce4(exps.iter().copied(), 0.0, |a, b| a + b);
        for (e, &(_, j)) in exps.iter().zip(&col[..k]) {
            aff[j * p + i] = e / sum;
        }
    }
    let usage: Vec<f32> = (0..m).map(|j| aff[j * p..(j + 1) * p].iter().sum()).collect();
    Ok((gpu_upload(&aff, m, p)?, gpu_upload(&usage, 1, m)?))
}

/// Softmax down each column of `x` [m, p] (over the rows), the max subtracted first.
///
/// Over a few rows (the decoder's softmax over background and objects) one thread a column sums
/// in row order, as PyTorch's spatial softmax does. Over more than 64 rows and fewer than 8192
/// columns (consolidation's candidates by prototypes) a block a tile of columns sums in a fixed
/// tree: neither that nor PyTorch's order there is sequential, so it is fp32 rounding apart.
pub fn gpu_cutie_softmax_axis0(x: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_softmax_axis0";
    cutie_check_f32(op, x)?;
    let (m, p) = (cutie_u32(op, x.rows)?, cutie_u32(op, x.cols)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, x.cols)?;
        gpu_check(unsafe { makepad_cuda_cutie_softmax_axis0_f32(x.device_ptr()?, out.device_ptr()?, m, p, backend.stream) })?;
        Ok(out)
    })
}

/// Reference: exp(x - max) summed in f64 and rounded once, so it stands for the exact softmax
/// whichever order the kernel sums in (a sequential f32 sum of 35k peaked terms is itself off
/// by 5e-5).
pub fn gpu_cutie_softmax_axis0_reference(x: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_softmax_axis0", x)?;
    let (m, p) = (x.rows, x.cols);
    let v = gpu_download(x)?;
    let mut out = vec![0.0f32; m * p];
    for i in 0..p {
        let max = (0..m).map(|j| v[j * p + i]).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for j in 0..m {
            let e = (v[j * p + i] - max).exp();
            out[j * p + i] = e;
            sum += e as f64;
        }
        let sum = sum as f32;
        for j in 0..m {
            out[j * p + i] /= sum;
        }
    }
    gpu_upload(&out, m, p)
}

// --- Objects ---

/// Cutie's `aggregate`: objects' probabilities `prob` [n, p] to logits with a background row first
/// ([n+1, p]): the background is prod(1 - prob) (in PyTorch's order, `cutie_reduce4`), every row
/// clamped to 1e-7..1-1e-7, then log(q / (1 - q)).
pub fn gpu_cutie_aggregate(prob: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_aggregate", prob)?;
    let n = cutie_u32("gpu_cutie_aggregate", prob.rows)?;
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(prob.rows + 1, prob.cols)?;
        gpu_check(unsafe { makepad_cuda_cutie_aggregate_f32(prob.device_ptr()?, out.device_ptr()?, n, prob.cols, backend.stream) })?;
        Ok(out)
    })
}

pub fn gpu_cutie_aggregate_reference(prob: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_aggregate", prob)?;
    let (n, p) = (prob.rows, prob.cols);
    let v = gpu_download(prob)?;
    let mut out = vec![0.0f32; (n + 1) * p];
    let logit = |q: f32| {
        let q = q.clamp(1e-7, 1.0 - 1e-7);
        (q / (1.0 - q)).ln()
    };
    for i in 0..p {
        let bg = cutie_reduce4((0..n).map(|o| 1.0 - v[o * p + i]), 1.0, |a, b| a * b);
        out[i] = logit(bg);
        for o in 0..n {
            out[(o + 1) * p + i] = logit(v[o * p + i]);
        }
    }
    gpu_upload(&out, n + 1, p)
}

/// The object transformer's attention mask from its auxiliary logits `aux` [n, p] (one row per
/// object): sigmoid, aggregated with a background (as `gpu_cutie_aggregate`); a pixel is an
/// object's foreground where its logit is the largest of all rows (a NaN makes the largest NaN,
/// as `logits.max` does, and the pixel no object's foreground). For each object, each of
/// `heads` heads and each of `queries` queries a row of p: the first half of the queries may only
/// see the object's foreground, the second half only the rest; a row that would see nothing sees
/// everything. Out [n*heads*queries, p], additive: 0 where seen, -inf where blocked.
pub fn gpu_cutie_attn_mask(aux: &GpuTensor, heads: usize, queries: usize) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_attn_mask";
    cutie_check_f32(op, aux)?;
    let (n, p) = (aux.rows, aux.cols);
    let (n_u, heads_u, queries_u) = (cutie_u32(op, n)?, cutie_u32(op, heads)?, cutie_u32(op, queries)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(n * heads * queries, p)?;
        // Each object's foreground flags [n, p], then whether it has any foreground and any
        // background (zeroed: the kernel only sets them).
        let scratch = GpuTensor::from_pool(n * p + 2 * n, 1)?;
        let scratch_ptr = scratch.device_ptr()?;
        cutie_zero(backend, unsafe { scratch_ptr.add(n * p) }, 2 * n)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_attn_mask_f32(aux.device_ptr()?, out.device_ptr()?, scratch_ptr.cast::<u32>(), n_u, p, heads_u, queries_u, backend.stream)
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_attn_mask_reference(aux: &GpuTensor, heads: usize, queries: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_attn_mask", aux)?;
    let (n, p) = (aux.rows, aux.cols);
    let prob = gpu_cutie_sigmoid_reference(aux)?;
    let agg = gpu_download(&gpu_cutie_aggregate_reference(&prob)?)?;
    let mut fg = vec![false; n * p];
    for i in 0..p {
        let max = (0..=n).map(|r| agg[r * p + i]).fold(f32::NEG_INFINITY, cutie_max_nan);
        for o in 0..n {
            fg[o * p + i] = agg[(o + 1) * p + i] >= max;
        }
    }
    let mut out = vec![0.0f32; n * heads * queries * p];
    for o in 0..n {
        let any_fg = (0..p).any(|i| fg[o * p + i]);
        let any_bg = (0..p).any(|i| !fg[o * p + i]);
        for h in 0..heads {
            for q in 0..queries {
                let foreground_query = q < queries / 2;
                // A row blocked everywhere is let through everywhere.
                let open_all = if foreground_query { !any_fg } else { !any_bg };
                let row = ((o * heads + h) * queries + q) * p;
                for i in 0..p {
                    let blocked = !open_all && (fg[o * p + i] != foreground_query);
                    out[row + i] = if blocked { f32::NEG_INFINITY } else { 0.0 };
                }
            }
        }
    }
    gpu_upload(&out, n * heads * queries, p)
}

/// The sum of a block `a` [r, c] and each of the k blocks of `b` [k*r, c]: out = a + b_i, one
/// rounding each (Cutie's x + g, one image's features broadcast over the objects, without
/// repeating them first).
pub fn gpu_cutie_add_broadcast(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_add_broadcast";
    cutie_check_f32(op, a)?;
    cutie_check_f32(op, b)?;
    if a.rows == 0 || a.cols != b.cols || b.rows % a.rows != 0 {
        return Err(format!("{op}: {}x{} over {}x{}", a.rows, a.cols, b.rows, b.cols));
    }
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(b.rows, b.cols)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_add_broadcast_f32(
                a.device_ptr()?, b.device_ptr()?, out.device_ptr()?, a.rows * a.cols, b.rows * b.cols, backend.stream,
            )
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_add_broadcast_reference(a: &GpuTensor, b: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_add_broadcast", a)?;
    if a.rows == 0 || a.cols != b.cols || b.rows % a.rows != 0 {
        return Err("gpu_cutie_add_broadcast shape mismatch".into());
    }
    let av = gpu_download(a)?;
    let bv = gpu_download(b)?;
    let out: Vec<f32> = bv.iter().enumerate().map(|(i, &g)| av[i % av.len()] + g).collect();
    gpu_upload(&out, b.rows, b.cols)
}

// --- The per-frame step ---

/// (x - sub[r]) / div[r] for each row r of `x` [rows, cols], `sub` and `div` one value a row on
/// the device: CUTIE.encode_image's (image - pixel_mean) / pixel_std, each op rounded, a true
/// division (multiplying by 1 / div would be up to an ulp off).
pub fn gpu_cutie_normalise_rows(x: &GpuTensor, sub: &GpuTensor, div: &GpuTensor) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_normalise_rows";
    for t in [x, sub, div] {
        cutie_check_f32(op, t)?;
    }
    if sub.rows * sub.cols != x.rows || div.rows * div.cols != x.rows {
        return Err(format!("{op}: {} and {} values for {} rows", sub.rows * sub.cols, div.rows * div.cols, x.rows));
    }
    let rows = cutie_u32(op, x.rows)?;
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, x.cols)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_normalise_rows_f32(
                x.device_ptr()?, out.device_ptr()?, sub.device_ptr()?, div.device_ptr()?, rows, x.cols, backend.stream,
            )
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_normalise_rows_reference(x: &GpuTensor, sub: &GpuTensor, div: &GpuTensor) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_normalise_rows", x)?;
    if sub.rows * sub.cols != x.rows || div.rows * div.cols != x.rows {
        return Err("gpu_cutie_normalise_rows shape mismatch".into());
    }
    let (s, d) = (gpu_download(sub)?, gpu_download(div)?);
    let mut v = gpu_download(x)?;
    for (r, row) in v.chunks_mut(x.cols.max(1)).enumerate() {
        for value in row {
            *value = (*value - s[r]) / d[r];
        }
    }
    gpu_upload(&v, x.rows, x.cols)
}

/// RGB bytes (`plane` pixels, 3 bytes each) to planar f32 [3, plane], each byte times `scale`.
/// The bytes are what goes to the device. Cutie's image is the worker's `x / 255` in PyTorch,
/// which on CUDA multiplies by the f32 reciprocal: `scale` 1.0f32 / 255.0.
pub fn gpu_cutie_rgb_planes(rgb: &[u8], plane: usize, scale: f32) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_rgb_planes";
    if plane == 0 || rgb.len() != plane * 3 {
        return Err(format!("{op}: {} bytes for {plane} RGB pixels", rgb.len()));
    }
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        // The bytes in a pooled buffer of whole f32 slots.
        let bytes = GpuTensor::from_pool(rgb.len().div_ceil(size_of::<f32>()), 1)?;
        bytes.buf.as_ref().ok_or("cutie rgb: buffer released")?.write(rgb, backend.stream)?;
        let out = GpuTensor::from_pool(3, plane)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_rgb_planes_f32(bytes.device_ptr()?.cast::<u8>().cast_const(), out.device_ptr()?, plane, scale, backend.stream)
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_rgb_planes_reference(rgb: &[u8], plane: usize, scale: f32) -> Result<GpuTensor, String> {
    if plane == 0 || rgb.len() != plane * 3 {
        return Err("gpu_cutie_rgb_planes shape mismatch".into());
    }
    let mut planes = vec![0.0f32; 3 * plane];
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        for c in 0..3 {
            planes[c * plane + i] = px[c] as f32 * scale;
        }
    }
    gpu_upload(&planes, 3, plane)
}

/// `gpu_cutie_resize_bilinear`'s scale for one axis: Python's in / out (a double), made f32 as
/// PyTorch's CUDA multiply takes a scalar.
fn cutie_bilinear_scale(size_in: usize, size_out: usize) -> f32 {
    (size_in as f64 / size_out as f64) as f32
}

/// One axis of `gpu_cutie_resize_bilinear` as torch._decomp's `_upsample_linear` computes it in
/// f32 (size given, align_corners false): for each output index, its lower and upper source
/// indices and the upper one's lambda.
fn cutie_bilinear_axis(size_in: usize, size_out: usize) -> Vec<(usize, usize, f32)> {
    let scale = cutie_bilinear_scale(size_in, size_out);
    (0..size_out)
        .map(|dst| {
            let source = scale * (dst as f32 + 0.5) - 0.5;
            let source = if source < 0.0 { 0.0 } else { source };
            let lower = source as usize;
            let upper = if lower + 1 < size_in { lower + 1 } else { lower };
            (lower, upper, source - lower as f32)
        })
        .collect()
}

/// `F.interpolate(mode='bilinear', align_corners=False)` of planar [planes, in_h*in_w] to
/// [planes, out_h*out_w], bit for bit as PyTorch computes it with deterministic algorithms on, as
/// the oracle runs it: F.interpolate on CUDA then runs torch._decomp's `_upsample_linear`, every
/// step its own fp32 op. Scale = in / out (a double made f32); source = scale * (dst + 0.5) -
/// 0.5, not fused, clamped at 0; the lower index the source truncated, the upper the next one
/// but at the edge; the lambda source - lower; then lerps a + (b - a) * t, along the width first
/// (each source row), then the height. Bit-exact against torch 2.11's deterministic CUDA output
/// on random data at 1920x1080 to 1777x1000 and back, 384x216 to 320x180 (the oracle
/// Solege-03-resize's), the decoder's x2 and x4, 7x5 to 5x8 and odd sizes, and against every
/// resize in the oracles. Torch's default CUDA kernel (`upsample_bilinear2d_out_frame`, its sums
/// of weighted taps fused by nvcc), which the worker runs without deterministic algorithms,
/// differs by a few ulps on about a fifth of the entries. `gpu_birefnet_resize_bilinear` computes
/// the source position in another order, up to 1.2e-4 off on 0..1 data.
pub fn gpu_cutie_resize_bilinear(x: &GpuTensor, in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_resize_bilinear";
    cutie_check_f32(op, x)?;
    if x.cols != in_w * in_h || in_w == 0 || in_h == 0 || out_w == 0 || out_h == 0 {
        return Err(format!("{op}: {}x{} for {in_w}x{in_h} to {out_w}x{out_h}", x.rows, x.cols));
    }
    let d = |v: usize| cutie_u32(op, v);
    let (planes, iw, ih, ow, oh) = (d(x.rows)?, d(in_w)?, d(in_h)?, d(out_w)?, d(out_h)?);
    let (scale_x, scale_y) = (cutie_bilinear_scale(in_w, out_w), cutie_bilinear_scale(in_h, out_h));
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(x.rows, out_w * out_h)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_resize_bilinear_f32(
                x.device_ptr()?, out.device_ptr()?, planes, iw, ih, ow, oh, scale_x, scale_y, backend.stream,
            )
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_resize_bilinear_reference(x: &GpuTensor, in_w: usize, in_h: usize, out_w: usize, out_h: usize) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_resize_bilinear", x)?;
    if x.cols != in_w * in_h || in_w == 0 || in_h == 0 || out_w == 0 || out_h == 0 {
        return Err("gpu_cutie_resize_bilinear shape mismatch".into());
    }
    let (ax, ay) = (cutie_bilinear_axis(in_w, out_w), cutie_bilinear_axis(in_h, out_h));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let src = gpu_download(x)?;
    let (in_plane, out_plane) = (in_w * in_h, out_w * out_h);
    let mut out = vec![0.0f32; x.rows * out_plane];
    for (p, dst) in out.chunks_exact_mut(out_plane).enumerate() {
        let src = &src[p * in_plane..(p + 1) * in_plane];
        for (&(y0, y1, h), dst) in ay.iter().zip(dst.chunks_exact_mut(out_w)) {
            let (row0, row1) = (&src[y0 * in_w..][..in_w], &src[y1 * in_w..][..in_w]);
            for (&(x0, x1, w), d) in ax.iter().zip(dst.iter_mut()) {
                *d = lerp(lerp(row0[x0], row0[x1], w), lerp(row1[x0], row1[x1], w), h);
            }
        }
    }
    gpu_upload(&out, x.rows, out_plane)
}

/// The object transformer's values from the object memory `memory` [objects*queries, width+1]
/// (object-major, each summary's sums then its area): sums / (area + `eps`) as
/// `object_transformer.py`'s `obj_sums / (obj_area + 1e-4)` rounds it (the add, then a true
/// division), interleaved [queries*objects, width] (row q*objects + o).
pub fn gpu_cutie_object_values(memory: &GpuTensor, objects: usize, eps: f32) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_object_values";
    cutie_check_f32(op, memory)?;
    if objects == 0 || memory.rows % objects != 0 || memory.cols < 2 {
        return Err(format!("{op}: {}x{} for {objects} objects", memory.rows, memory.cols));
    }
    let (queries, width) = (memory.rows / objects, memory.cols - 1);
    let d = |v: usize| cutie_u32(op, v);
    let (objects_u, queries_u, width_u) = (d(objects)?, d(queries)?, d(width)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let out = GpuTensor::from_pool(memory.rows, width)?;
        gpu_check(unsafe {
            makepad_cuda_cutie_object_values_f32(memory.device_ptr()?, out.device_ptr()?, objects_u, queries_u, width_u, eps, backend.stream)
        })?;
        Ok(out)
    })
}

pub fn gpu_cutie_object_values_reference(memory: &GpuTensor, objects: usize, eps: f32) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_object_values", memory)?;
    if objects == 0 || memory.rows % objects != 0 || memory.cols < 2 {
        return Err("gpu_cutie_object_values shape mismatch".into());
    }
    let (queries, width) = (memory.rows / objects, memory.cols - 1);
    let v = gpu_download(memory)?;
    let mut out = vec![0.0f32; objects * queries * width];
    for o in 0..objects {
        for q in 0..queries {
            let row = &v[(o * queries + q) * (width + 1)..(o * queries + q + 1) * (width + 1)];
            let area = row[width] + eps;
            for (dst, &s) in out[(q * objects + o) * width..(q * objects + o + 1) * width].iter_mut().zip(row) {
                *dst = s / area;
            }
        }
    }
    gpu_upload(&out, objects * queries, width)
}

/// Cutie's `PositionalEncoding` (channel-last, normalised) at `w` x `h` from its frequencies
/// `inv_freq` (F of them): tokens [h*w, 4F], token y*w + x, the x encoding in channels 0..2F and
/// the y encoding in 2F..4F, sin and cos interleaved per frequency. Each axis' positions are
/// i / ((len - 1) + `eps`) * `scale`, times each frequency, every op rounded as PyTorch's tensor
/// ops round it; then CUDA's own sinf and cosf, which is what torch.sin and torch.cos give on CUDA
/// (bit for bit against the oracle's encodings at 24x14 and 60x34).
pub fn gpu_cutie_sine_encoding(inv_freq: &[f32], w: usize, h: usize, scale: f32, eps: f32) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_sine_encoding";
    if inv_freq.is_empty() || w == 0 || h == 0 {
        return Err(format!("{op}: {} frequencies at {w}x{h}", inv_freq.len()));
    }
    let d = |v: usize| cutie_u32(op, v);
    let (freqs, wu, hu) = (d(inv_freq.len())?, d(w)?, d(h)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let freq_buf = gpu_upload_small(backend, inv_freq)?;
        let out = GpuTensor::from_pool(w * h, 4 * inv_freq.len())?;
        let status = unsafe {
            makepad_cuda_cutie_sine_encoding_f32(
                freq_buf.ptr.as_ptr().cast::<f32>(), out.device_ptr()?, freqs, wu, hu, scale, eps, backend.stream,
            )
        };
        gpu_pool_release(freq_buf);
        gpu_check(status)?;
        Ok(out)
    })
}

/// Reference: sin and cos in f64, rounded once. CUDA's sinf and cosf are not correctly rounded
/// (about 1 entry in 7 of Cutie's encodings is 1 ulp from this), so the two are fp32 rounding
/// apart.
pub fn gpu_cutie_sine_encoding_reference(inv_freq: &[f32], w: usize, h: usize, scale: f32, eps: f32) -> Result<GpuTensor, String> {
    if inv_freq.is_empty() || w == 0 || h == 0 {
        return Err("gpu_cutie_sine_encoding shape mismatch".into());
    }
    let half = inv_freq.len() * 2;
    let axis = |len: usize| -> Vec<f32> {
        let last = (len - 1) as f32 + eps;
        let mut emb = Vec::with_capacity(len * half);
        for i in 0..len {
            let pos = i as f32 / last * scale;
            for &f in inv_freq {
                let angle = (pos * f) as f64;
                emb.extend([angle.sin() as f32, angle.cos() as f32]);
            }
        }
        emb
    };
    let (ex, ey) = (axis(w), axis(h));
    let mut pe = Vec::with_capacity(h * w * 2 * half);
    for y in 0..h {
        for x in 0..w {
            pe.extend_from_slice(&ex[x * half..(x + 1) * half]);
            pe.extend_from_slice(&ey[y * half..(y + 1) * half]);
        }
    }
    gpu_upload(&pe, h * w, 2 * half)
}

/// nn.LayerNorm over each row of `x` [rows, cols] (cols % 4 == 0), the kernel of
/// `gpu_layer_norm_pytorch` (PyTorch's CUDA order), with `gamma` and `beta` kept on the device
/// under `{cache_namespace}::{key}::cutie_ln` instead of uploaded on every call.
pub fn gpu_cutie_layer_norm(
    x: &GpuTensor, cache_namespace: &str, key: &str, gamma: &[f32], beta: &[f32], eps: f32,
) -> Result<GpuTensor, String> {
    let op = "gpu_cutie_layer_norm";
    cutie_check_f32(op, x)?;
    if gamma.len() != x.cols || beta.len() != x.cols || x.cols % 4 != 0 {
        return Err(format!("{op} {key}: {} and {} affine values for {} columns", gamma.len(), beta.len(), x.cols));
    }
    let (rows, cols) = (cutie_u32(op, x.rows)?, cutie_u32(op, x.cols)?);
    with_dense_linear_backend(|backend| {
        backend.prepare_device()?;
        let qualified_key = format!("{cache_namespace}::{key}::cutie_ln");
        backend.cached_weight_buffer(&qualified_key, 2 * x.cols * size_of::<f32>(), || {
            Ok(gamma.iter().chain(beta).flat_map(|v| v.to_le_bytes()).collect())
        })?;
        let gamma_ptr = backend
            .weight_buffers
            .get(&qualified_key)
            .ok_or_else(|| format!("missing cached cutie layer norm {qualified_key}"))?
            .ptr
            .as_ptr()
            .cast::<f32>();
        let beta_ptr = unsafe { gamma_ptr.add(x.cols) };
        let out = GpuTensor::from_pool(x.rows, x.cols)?;
        gpu_check(unsafe {
            makepad_cuda_layer_norm_pytorch_f32(x.device_ptr()?, gamma_ptr, beta_ptr, out.device_ptr()?, rows, cols, eps, backend.stream)
        })?;
        Ok(out)
    })
}

/// Reference: `gpu_layer_norm_pytorch` itself (the same kernel, its affine uploaded per call).
pub fn gpu_cutie_layer_norm_reference(
    x: &GpuTensor, _cache_namespace: &str, _key: &str, gamma: &[f32], beta: &[f32], eps: f32,
) -> Result<GpuTensor, String> {
    cutie_check_f32("gpu_cutie_layer_norm", x)?;
    gpu_layer_norm_pytorch(x, gamma, beta, eps)
}
