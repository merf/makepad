// Cutie (video object segmentation) device ops, in exact fp32: the native port is checked against
// a PyTorch fp32 oracle with TF32 off (models/cutie). The Rust wrappers are in src/cutie_ops.rs;
// each op there has a host reference beside it that these kernels must match.
//
// Elementwise arithmetic that PyTorch does as separate tensor ops (each rounded) is written with
// the _rn intrinsics, so nvcc cannot contract it into an FMA that PyTorch never does.
#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <math_constants.h>

// A 1-D launch: enough blocks to cover n with grid-stride loops, capped.
static unsigned cutie_blocks(size_t n, unsigned threads) {
    const size_t blocks = (n + threads - 1) / threads;
    return (unsigned)(blocks < 65536 ? (blocks ? blocks : 1) : 65536);
}

// PyTorch's sigmoid for f32 (precise expf, IEEE division).
static __device__ __forceinline__ float cutie_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Cutie's aggregate logit: clamp to 1e-7..1-1e-7 (NaN passes through, as torch.clamp), then
// log(q / (1 - q)).
static __device__ __forceinline__ float cutie_logit(float q) {
    const float lo = 1e-7f, hi = 1.0f - 1e-7f;
    q = q < lo ? lo : (q > hi ? hi : q);
    return logf(q / (1.0f - q));
}

// PyTorch's order for a sum or product over a dimension that is not the innermost (Cutie's
// objects, the top-k picks): four accumulators, term o into o % 4, then ((a0 . a1) . a2) . a3
// (Reduce.cuh's thread_reduce_impl with vt0 = 4, one thread an output, which holds for fewer than
// 64 terms). From 5 terms on it rounds differently from a sequential sum.
template <bool Product, typename Term>
static __device__ __forceinline__ float cutie_reduce4(uint32_t n, Term term) {
    const float ident = Product ? 1.0f : 0.0f;
    float acc[4] = {ident, ident, ident, ident};
    for (uint32_t o = 0; o < n; o += 4) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            if (o + j < n) acc[j] = Product ? __fmul_rn(acc[j], term(o + j)) : __fadd_rn(acc[j], term(o + j));
        }
    }
    return Product ? __fmul_rn(__fmul_rn(__fmul_rn(acc[0], acc[1]), acc[2]), acc[3])
                   : __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
}

// The larger of two values, a NaN in either winning (PyTorch's max_propagate_nan).
static __device__ __forceinline__ float cutie_max_nan(float a, float b) {
    return b != b ? b : (a != a ? a : fmaxf(a, b));
}

// MaxPool2d(kernel 3, stride 2, padding 1) on planar [planes, height*width]: padding never wins
// (PyTorch pads with -inf), so a window takes the max of its in-bounds taps; a NaN tap wins, as
// in PyTorch's max_pool2d (`val > maxval || isnan(val)`).
static __global__ void cutie_maxpool3x3s2_kernel(
    const float* __restrict__ x, float* __restrict__ out,
    uint32_t width, uint32_t height, uint32_t ow, uint32_t oh, uint32_t planes) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = size_t(ow) * oh;
    if (i >= plane * planes) return;
    const size_t p = i / plane;
    const int ox = int(i % ow), oy = int((i % plane) / ow);
    const float* src = x + p * size_t(width) * height;
    float best = -CUDART_INF_F;
    for (int ky = 0; ky < 3; ++ky) {
        const int y = oy * 2 - 1 + ky;
        if (y < 0 || y >= int(height)) continue;
        for (int kx = 0; kx < 3; ++kx) {
            const int xx = ox * 2 - 1 + kx;
            if (xx < 0 || xx >= int(width)) continue;
            const float v = src[size_t(y) * width + xx];
            if (v > best || v != v) best = v;
        }
    }
    out[i] = best;
}

extern "C" cudaError_t makepad_cuda_cutie_maxpool3x3s2_f32(
    const float* x, float* out, uint32_t width, uint32_t height, uint32_t ow, uint32_t oh,
    uint32_t planes, cudaStream_t stream) {
    const size_t n = size_t(ow) * oh * planes;
    if (!n) return cudaSuccess;
    cutie_maxpool3x3s2_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(x, out, width, height, ow, oh, planes);
    return cudaGetLastError();
}

// --- Convolution: f32 im2col, then cuBLAS SGEMM (in the Rust wrapper), then the bias ---

// The columns of `images` planar images [images*channels, height*width] for a square k x k
// window with stride and zero padding, row r = c*k*k + ky*k + kx and column q = oy*ow + ox: the
// layout PyTorch's [out][in][k][k] weights multiply as a [out, in*k*k] matrix. Per image
// (col[b][r][q], one GEMM an image) or, `side_by_side`, the images' columns next to each other
// (col[r][b][q], one GEMM for them all).
static __global__ void cutie_im2col_kernel(
    const float* __restrict__ x, float* __restrict__ col, uint32_t images, uint32_t channels, uint32_t width,
    uint32_t height, uint32_t ow, uint32_t oh, uint32_t k, uint32_t pad, uint32_t stride, size_t rows,
    bool side_by_side) {
    const size_t opix = size_t(ow) * oh;
    const size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= opix) return;
    const int oy = int(q / ow), ox = int(q % ow);
    const uint32_t kk = k * k, kdim = channels * kk;
    for (size_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const size_t b = row / kdim;
        const uint32_t r = uint32_t(row % kdim);
        const uint32_t c = r / kk, ky = (r % kk) / k, kx = r % k;
        const int iy = oy * int(stride) - int(pad) + int(ky);
        const int ix = ox * int(stride) - int(pad) + int(kx);
        float v = 0.0f;
        if (iy >= 0 && iy < int(height) && ix >= 0 && ix < int(width)) {
            v = x[((b * channels + c) * height + iy) * size_t(width) + ix];
        }
        const size_t at = side_by_side ? size_t(r) * images + b : row;
        col[at * opix + q] = v;
    }
}

extern "C" cudaError_t makepad_cuda_cutie_im2col_f32(
    const float* x, float* col, uint32_t images, uint32_t channels, uint32_t width, uint32_t height,
    uint32_t ow, uint32_t oh, uint32_t k, uint32_t pad, uint32_t stride, uint32_t side_by_side, cudaStream_t stream) {
    const size_t opix = size_t(ow) * oh;
    const size_t rows = size_t(images) * channels * k * k;
    if (!opix || !rows) return cudaSuccess;
    const dim3 block(256, 1, 1);
    const dim3 grid((unsigned)((opix + 255) / 256), (unsigned)(rows < 65535 ? rows : 65535), 1);
    cutie_im2col_kernel<<<grid, block, 0, stream>>>(
        x, col, images, channels, width, height, ow, oh, k, pad, stride, rows, side_by_side != 0);
    return cudaGetLastError();
}

// The side-by-side GEMM's result [channels][images][plane] to the images' planar layout
// [images][channels][plane], plus the bias (when given).
static __global__ void cutie_unstack_bias_kernel(
    const float* __restrict__ src, float* __restrict__ out, const float* __restrict__ bias, uint32_t images,
    uint32_t channels, size_t plane, size_t rows) {
    const size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= plane) return;
    for (size_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const size_t b = row / channels, c = row % channels;
        const float v = src[(c * images + b) * plane + q];
        out[row * plane + q] = bias ? __fadd_rn(v, bias[c]) : v;
    }
}

extern "C" cudaError_t makepad_cuda_cutie_unstack_bias_f32(
    const float* src, float* out, const float* bias, uint32_t images, uint32_t channels, size_t plane,
    cudaStream_t stream) {
    const size_t rows = size_t(images) * channels;
    if (!plane || !rows) return cudaSuccess;
    const dim3 block(256, 1, 1);
    const dim3 grid((unsigned)((plane + 255) / 256), (unsigned)(rows < 65535 ? rows : 65535), 1);
    cutie_unstack_bias_kernel<<<grid, block, 0, stream>>>(src, out, bias, images, channels, plane, rows);
    return cudaGetLastError();
}

// out[row][q] += bias[row % channels] over [rows, plane] (rows = images*channels).
static __global__ void cutie_add_channel_bias_kernel(
    float* __restrict__ out, const float* __restrict__ bias, uint32_t channels, size_t plane, size_t rows) {
    const size_t q = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= plane) return;
    for (size_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const size_t at = row * plane + q;
        out[at] = __fadd_rn(out[at], bias[row % channels]);
    }
}

extern "C" cudaError_t makepad_cuda_cutie_add_channel_bias_f32(
    float* out, const float* bias, uint32_t channels, size_t plane, size_t rows, cudaStream_t stream) {
    if (!plane || !rows) return cudaSuccess;
    const dim3 block(256, 1, 1);
    const dim3 grid((unsigned)((plane + 255) / 256), (unsigned)(rows < 65535 ? rows : 65535), 1);
    cutie_add_channel_bias_kernel<<<grid, block, 0, stream>>>(out, bias, channels, plane, rows);
    return cudaGetLastError();
}

// --- Elementwise ---

static __global__ void cutie_sigmoid_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        out[i] = cutie_sigmoid(x[i]);
    }
}

extern "C" cudaError_t makepad_cuda_cutie_sigmoid_f32(const float* x, float* out, size_t n, cudaStream_t stream) {
    if (!n) return cudaSuccess;
    cutie_sigmoid_kernel<<<cutie_blocks(n, 256), 256, 0, stream>>>(x, out, n);
    return cudaGetLastError();
}

// Cutie's sensory GRU: per object o, values rows [o*3c, o*3c+c) forget, then update, then new;
// new_h = forget * h * (1 - update) + update * tanh(new), each product and sum rounded as
// PyTorch's separate tensor ops round them.
static __global__ void cutie_gru_kernel(
    const float* __restrict__ h, const float* __restrict__ values, float* __restrict__ out,
    size_t cp, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const size_t o = i / cp, rem = i % cp;
        const float* v = values + o * 3 * cp;
        const float forget = cutie_sigmoid(v[rem]);
        const float update = cutie_sigmoid(v[cp + rem]);
        const float fresh = tanhf(v[2 * cp + rem]);
        const float kept = __fmul_rn(__fmul_rn(forget, h[i]), __fsub_rn(1.0f, update));
        out[i] = __fadd_rn(kept, __fmul_rn(update, fresh));
    }
}

extern "C" cudaError_t makepad_cuda_cutie_gru_f32(
    const float* h, const float* values, float* out, uint32_t objects, uint32_t channels, size_t plane,
    cudaStream_t stream) {
    const size_t cp = size_t(channels) * plane, n = size_t(objects) * cp;
    if (!n) return cudaSuccess;
    cutie_gru_kernel<<<cutie_blocks(n, 256), 256, 0, stream>>>(h, values, out, cp, n);
    return cudaGetLastError();
}

// The mean of each row of [rows, plane]: one block a row, a warp-shuffle tree.
static __global__ void cutie_row_mean_kernel(const float* __restrict__ x, float* __restrict__ out, size_t plane) {
    const float* row = x + size_t(blockIdx.x) * plane;
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < plane; i += blockDim.x) sum += row[i];
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    __shared__ float warp_sums[32];
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        if (lane == 0) out[blockIdx.x] = sum / float(plane);
    }
}

// ECA's taps travel by value (no upload): Cutie's are 3 or 5 long.
#define CUTIE_ECA_MAX_TAPS 15
struct CutieEcaTaps {
    float w[CUTIE_ECA_MAX_TAPS];
};

// Per image and channel: sigmoid of the 1-D convolution over the channel means (zero padding).
static __global__ void cutie_eca_scale_kernel(
    const float* __restrict__ means, float* __restrict__ scale, CutieEcaTaps taps, uint32_t k,
    uint32_t channels, uint32_t rows) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int ch = int(row % channels), base = int(row - row % channels), pad = int(k - 1) / 2;
    float y = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        const int src = ch + int(j) - pad;
        if (src >= 0 && src < int(channels)) y = __fadd_rn(y, __fmul_rn(taps.w[j], means[base + src]));
    }
    scale[row] = cutie_sigmoid(y);
}

// out = x * scale[row] + residual.
static __global__ void cutie_eca_apply_kernel(
    const float* __restrict__ x, const float* __restrict__ residual, const float* __restrict__ scale,
    float* __restrict__ out, size_t plane, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        out[i] = __fadd_rn(__fmul_rn(x[i], scale[i / plane]), residual[i]);
    }
}

extern "C" cudaError_t makepad_cuda_cutie_eca_f32(
    const float* x, const float* residual, float* out, float* scratch, const float* taps, uint32_t k,
    uint32_t objects, uint32_t channels, size_t plane, cudaStream_t stream) {
    const uint32_t rows = objects * channels;
    if (!rows || !plane) return cudaSuccess;
    if (k == 0 || k > CUTIE_ECA_MAX_TAPS || k % 2 == 0) return cudaErrorInvalidValue;
    CutieEcaTaps t;
    for (uint32_t j = 0; j < CUTIE_ECA_MAX_TAPS; ++j) t.w[j] = j < k ? taps[j] : 0.0f;
    float* means = scratch;
    float* scale = scratch + rows;
    cutie_row_mean_kernel<<<rows, 256, 0, stream>>>(x, means, plane);
    cutie_eca_scale_kernel<<<(rows + 255) / 256, 256, 0, stream>>>(means, scale, t, k, channels, rows);
    const size_t n = size_t(rows) * plane;
    cutie_eca_apply_kernel<<<cutie_blocks(n, 256), 256, 0, stream>>>(x, residual, scale, out, plane, n);
    return cudaGetLastError();
}

// For each object's mask [n, p], the others' sum (all objects summed in PyTorch's order, then the
// object's own subtracted), clamped to 0..1 with NaN passing through.
static __global__ void cutie_others_kernel(const float* __restrict__ masks, float* __restrict__ out, uint32_t n, size_t p) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < p; i += size_t(gridDim.x) * blockDim.x) {
        const float sum = cutie_reduce4<false>(n, [&](uint32_t o) { return masks[o * p + i]; });
        for (uint32_t o = 0; o < n; ++o) {
            const float d = __fsub_rn(sum, masks[o * p + i]);
            out[o * p + i] = d < 0.0f ? 0.0f : (d > 1.0f ? 1.0f : d);
        }
    }
}

extern "C" cudaError_t makepad_cuda_cutie_others_f32(const float* masks, float* out, uint32_t n, size_t p, cudaStream_t stream) {
    if (!n || !p) return cudaSuccess;
    cutie_others_kernel<<<cutie_blocks(p, 256), 256, 0, stream>>>(masks, out, n, p);
    return cudaGetLastError();
}

// --- Shapes ---

// Zero padding of planar [planes, h*w] to [planes, nh*nw], the image at (left, top).
static __global__ void cutie_pad_kernel(
    const float* __restrict__ x, float* __restrict__ out, uint32_t w, uint32_t h, uint32_t nw, uint32_t nh,
    uint32_t left, uint32_t top, size_t n) {
    const size_t plane = size_t(nw) * nh;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const size_t p = i / plane, at = i % plane;
        const int y = int(at / nw) - int(top), xx = int(at % nw) - int(left);
        out[i] = (y >= 0 && y < int(h) && xx >= 0 && xx < int(w)) ? x[(p * h + y) * size_t(w) + xx] : 0.0f;
    }
}

extern "C" cudaError_t makepad_cuda_cutie_pad_f32(
    const float* x, float* out, uint32_t planes, uint32_t w, uint32_t h, uint32_t left, uint32_t right,
    uint32_t top, uint32_t bottom, cudaStream_t stream) {
    const uint32_t nw = w + left + right, nh = h + top + bottom;
    const size_t n = size_t(nw) * nh * planes;
    if (!n) return cudaSuccess;
    cutie_pad_kernel<<<cutie_blocks(n, 256), 256, 0, stream>>>(x, out, w, h, nw, nh, left, top, n);
    return cudaGetLastError();
}

// A window of planar [planes, h*w]: out_w x out_h from (left, top).
static __global__ void cutie_crop_kernel(
    const float* __restrict__ x, float* __restrict__ out, uint32_t w, uint32_t h, uint32_t left, uint32_t top,
    uint32_t out_w, uint32_t out_h, size_t n) {
    const size_t plane = size_t(out_w) * out_h;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const size_t p = i / plane, at = i % plane;
        const size_t y = at / out_w + top, xx = at % out_w + left;
        out[i] = x[(p * h + y) * size_t(w) + xx];
    }
}

extern "C" cudaError_t makepad_cuda_cutie_crop_f32(
    const float* x, float* out, uint32_t planes, uint32_t w, uint32_t h, uint32_t left, uint32_t top,
    uint32_t out_w, uint32_t out_h, cudaStream_t stream) {
    const size_t n = size_t(out_w) * out_h * planes;
    if (!n) return cudaSuccess;
    cutie_crop_kernel<<<cutie_blocks(n, 256), 256, 0, stream>>>(x, out, w, h, left, top, out_w, out_h, n);
    return cudaGetLastError();
}

// --- Memory reading ---

// The similarity's operands before its GEMMs, as PyTorch forms them: mk^2 [ck, m] and, per
// query pixel, qk * qe [ck, p] and b_sq = sum_c qe * qk^2 [p]; without a selection, a_sq =
// sum_c mk^2 per memory token [m].
static __global__ void cutie_similarity_prep_kernel(
    const float* __restrict__ mk, const float* __restrict__ qk, const float* __restrict__ qe,
    float* __restrict__ mk_sq, float* __restrict__ qkqe, float* __restrict__ b_sq, float* __restrict__ a_sq,
    uint32_t ck, uint32_t m, uint32_t p) {
    const size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (qe) {
        if (t < m) {
            for (uint32_t c = 0; c < ck; ++c) {
                const float v = mk[size_t(c) * m + t];
                mk_sq[size_t(c) * m + t] = __fmul_rn(v, v);
            }
        }
        if (t < p) {
            float sum = 0.0f;
            for (uint32_t c = 0; c < ck; ++c) {
                const float k = qk[size_t(c) * p + t], e = qe[size_t(c) * p + t];
                qkqe[size_t(c) * p + t] = __fmul_rn(k, e);
                sum = __fadd_rn(sum, __fmul_rn(e, __fmul_rn(k, k)));
            }
            b_sq[t] = sum;
        }
    } else if (t < m) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < ck; ++c) {
            const float v = mk[size_t(c) * m + t];
            sum = __fadd_rn(sum, __fmul_rn(v, v));
        }
        a_sq[t] = sum;
    }
}

extern "C" cudaError_t makepad_cuda_cutie_similarity_prep_f32(
    const float* mk, const float* qk, const float* qe, float* mk_sq, float* qkqe, float* b_sq, float* a_sq,
    uint32_t ck, uint32_t m, uint32_t p, cudaStream_t stream) {
    const uint32_t n = m > p ? m : p;
    if (!n) return cudaSuccess;
    cutie_similarity_prep_kernel<<<(n + 255) / 256, 256, 0, stream>>>(mk, qk, qe, mk_sq, qkqe, b_sq, a_sq, ck, m, p);
    return cudaGetLastError();
}

// The similarity from its GEMMs, in the reference's order: with a selection, `sim` holds a_sq
// and `two_ab` mk^T (qk*qe): ((-a_sq + 2 two_ab) - b_sq); without, `sim` holds mk^T qk and
// a_sq is per token: (-a_sq + 2 sim). Then * ms (when given), then * inv_scale (PyTorch divides
// by the scalar sqrt(CK) as a multiply by its reciprocal). In place in `sim`.
static __global__ void cutie_similarity_epilogue_kernel(
    float* __restrict__ sim, const float* __restrict__ two_ab, const float* __restrict__ b_sq,
    const float* __restrict__ a_sq, const float* __restrict__ ms, float inv_scale, uint32_t p, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const size_t j = i / p, px = i % p;
        float s;
        if (two_ab) {
            s = __fsub_rn(__fadd_rn(-sim[i], __fmul_rn(2.0f, two_ab[i])), b_sq[px]);
        } else {
            s = __fadd_rn(-a_sq[j], __fmul_rn(2.0f, sim[i]));
        }
        if (ms) s = __fmul_rn(s, ms[j]);
        sim[i] = __fmul_rn(s, inv_scale);
    }
}

extern "C" cudaError_t makepad_cuda_cutie_similarity_epilogue_f32(
    float* sim, const float* two_ab, const float* b_sq, const float* a_sq, const float* ms, float inv_scale,
    uint32_t m, uint32_t p, cudaStream_t stream) {
    const size_t n = size_t(m) * p;
    if (!n) return cudaSuccess;
    cutie_similarity_epilogue_kernel<<<cutie_blocks(n, 256), 256, 0, stream>>>(sim, two_ab, b_sq, a_sq, ms, inv_scale, p, n);
    return cudaGetLastError();
}

// Top-k down the columns of sim [m, p] in torch.topk's order: values compared by PyTorch's radix
// key (TopKTypeConfig<float>::convert), so -0 ranks below +0 and every NaN above every number;
// equal keys by the lower row, the rows torch.topk takes at the k-th place.
#define CUTIE_TOPK_MAX 64
#define CUTIE_TOPK_COLS 32
#define CUTIE_TOPK_SLICES_PER_BLOCK 4
#define CUTIE_TOPK_SLICE_ROWS 256
// An empty place in a list: key 0 is below every float's key (-inf's is 0x007fffff), so every
// row beats it and it is never chosen while k <= m.
#define CUTIE_TOPK_EMPTY_KEY 0u
#define CUTIE_TOPK_EMPTY_ROW 0xffffffffu

// A float's radix key: the keys' unsigned order is torch.topk's order of the values.
static __device__ __forceinline__ uint32_t cutie_topk_key(float v) {
    const uint32_t x = __float_as_uint(v);
    if (v != v) return 0xffffffffu;
    return x ^ ((x & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

// The value back from its key (every NaN comes back as the one NaN 0x7fffffff).
static __device__ __forceinline__ float cutie_topk_value(uint32_t key) {
    return __uint_as_float((key & 0x80000000u) ? (key ^ 0x80000000u) : ~key);
}

static __device__ __forceinline__ bool cutie_topk_before(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) {
    return ka > kb || (ka == kb && ia < ib);
}

// Each thread takes one column's rows [slice*chunk, (slice+1)*chunk) and keeps the best K of
// them in registers as keys, sorted (a new entry is carried down the list, swapping past every
// entry it beats); reads are coalesced across the 32 columns of a block row. The first k of
// each list go to keys/idx [slices][k][p].
template <int K>
static __global__ void __launch_bounds__(CUTIE_TOPK_COLS * CUTIE_TOPK_SLICES_PER_BLOCK) cutie_topk_partial_kernel(
    const float* __restrict__ sim, uint32_t* __restrict__ keys, uint32_t* __restrict__ idx,
    uint32_t m, uint32_t p, uint32_t k, uint32_t chunk, uint32_t slices) {
    const uint32_t col = blockIdx.x * CUTIE_TOPK_COLS + threadIdx.x;
    const uint32_t slice = blockIdx.y * CUTIE_TOPK_SLICES_PER_BLOCK + threadIdx.y;
    if (col >= p || slice >= slices) return;
    uint32_t tk[K];
    uint32_t ti[K];
#pragma unroll
    for (int t = 0; t < K; ++t) {
        tk[t] = CUTIE_TOPK_EMPTY_KEY;
        ti[t] = CUTIE_TOPK_EMPTY_ROW;
    }
    const uint32_t start = slice * chunk;
    const uint32_t end = min(m, start + chunk);
    const float* src = sim + col;
    // Rows are read four at a time so the loads are in flight together.
    for (uint32_t j = start; j < end; j += 4) {
        uint32_t batch[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) batch[u] = j + u < end ? cutie_topk_key(src[size_t(j + u) * p]) : CUTIE_TOPK_EMPTY_KEY;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            uint32_t row = j + u;
            uint32_t key = batch[u];
            if (row >= end || !cutie_topk_before(key, row, tk[K - 1], ti[K - 1])) continue;
#pragma unroll
            for (int t = 0; t < K; ++t) {
                const bool before = cutie_topk_before(key, row, tk[t], ti[t]);
                const uint32_t sk = tk[t];
                const uint32_t si = ti[t];
                tk[t] = before ? key : sk;
                ti[t] = before ? row : si;
                key = before ? sk : key;
                row = before ? si : row;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < K; ++t) {
        if (uint32_t(t) < k) {
            keys[(size_t(slice) * k + t) * p + col] = tk[t];
            idx[(size_t(slice) * k + t) * p + col] = ti[t];
        }
    }
}

// One warp a column: lane l holds the heads of slices l and l+32. Each of the k picks is the
// warp's best head (a shuffle reduction in the key order; rows are unique, so one lane owns it
// and advances). Then the softmax as Cutie's do_softmax: exp of the picked values (no max
// subtracted), summed in PyTorch's order over the picks (best first, cutie_reduce4's four
// accumulators), normalised, scattered into aff [m, p] (zeroed before).
static __global__ void cutie_topk_merge_kernel(
    const uint32_t* __restrict__ keys, const uint32_t* __restrict__ idx, float* __restrict__ aff,
    uint32_t m, uint32_t p, uint32_t k, uint32_t slices) {
    const uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const uint32_t lane = threadIdx.x % 32;
    if (col >= p) return;
    const uint32_t s0 = lane, s1 = lane + 32;
    uint32_t h0 = 0, h1 = 0;
    uint32_t key0 = CUTIE_TOPK_EMPTY_KEY, key1 = CUTIE_TOPK_EMPTY_KEY;
    uint32_t i0 = CUTIE_TOPK_EMPTY_ROW, i1 = CUTIE_TOPK_EMPTY_ROW;
    if (s0 < slices) {
        key0 = keys[size_t(s0) * k * p + col];
        i0 = idx[size_t(s0) * k * p + col];
    }
    if (s1 < slices) {
        key1 = keys[size_t(s1) * k * p + col];
        i1 = idx[size_t(s1) * k * p + col];
    }
    // Lane l keeps picks l and l+32: their exp and row.
    float e0 = 0.0f, e1 = 0.0f;
    uint32_t r0 = CUTIE_TOPK_EMPTY_ROW, r1 = CUTIE_TOPK_EMPTY_ROW;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint32_t t = 0; t < k; ++t) {
        const bool second = cutie_topk_before(key1, i1, key0, i0);
        uint32_t bk = second ? key1 : key0;
        uint32_t bi = second ? i1 : i0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const uint32_t ok = __shfl_xor_sync(0xffffffffu, bk, offset);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, offset);
            if (cutie_topk_before(ok, oi, bk, bi)) {
                bk = ok;
                bi = oi;
            }
        }
        if (bi == i0 && s0 < slices) {
            ++h0;
            key0 = h0 < k ? keys[(size_t(s0) * k + h0) * p + col] : CUTIE_TOPK_EMPTY_KEY;
            i0 = h0 < k ? idx[(size_t(s0) * k + h0) * p + col] : CUTIE_TOPK_EMPTY_ROW;
        } else if (bi == i1 && s1 < slices) {
            ++h1;
            key1 = h1 < k ? keys[(size_t(s1) * k + h1) * p + col] : CUTIE_TOPK_EMPTY_KEY;
            i1 = h1 < k ? idx[(size_t(s1) * k + h1) * p + col] : CUTIE_TOPK_EMPTY_ROW;
        }
        const float e = expf(cutie_topk_value(bk));
        const uint32_t slot = t & 3u;
        a0 = slot == 0 ? __fadd_rn(a0, e) : a0;
        a1 = slot == 1 ? __fadd_rn(a1, e) : a1;
        a2 = slot == 2 ? __fadd_rn(a2, e) : a2;
        a3 = slot == 3 ? __fadd_rn(a3, e) : a3;
        if (t % 32 == lane) {
            if (t < 32) {
                e0 = e;
                r0 = bi;
            } else {
                e1 = e;
                r1 = bi;
            }
        }
    }
    const float sum = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
    // A pick is always a real row while k <= m; the bound keeps a stray one off memory.
    if (lane < k && r0 < m) aff[size_t(r0) * p + col] = e0 / sum;
    if (lane + 32 < k && r1 < m) aff[size_t(r1) * p + col] = e1 / sum;
}

// The sum of each row of [rows, cols]: one warp a row. Not PyTorch's order for affinity.sum(dim=2)
// (a vectorised innermost reduce), so the usage is fp32 rounding apart from PyTorch's.
static __global__ void cutie_row_sum_kernel(const float* __restrict__ x, float* __restrict__ out, uint32_t rows, uint32_t cols) {
    const uint32_t row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* src = x + size_t(row) * cols;
    float sum = 0.0f;
    for (uint32_t i = lane; i < cols; i += 32) sum += src[i];
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    if (lane == 0) out[row] = sum;
}

// The slices one column is split into: about CUTIE_TOPK_SLICE_ROWS rows each, at most 64 (the
// merge's warp holds two slices a lane).
static uint32_t cutie_topk_slices(uint32_t m) {
    uint32_t slices = (m + CUTIE_TOPK_SLICE_ROWS - 1) / CUTIE_TOPK_SLICE_ROWS;
    if (slices > 64) slices = 64;
    return slices ? slices : 1;
}

extern "C" size_t makepad_cuda_cutie_topk_scratch_entries(uint32_t m, uint32_t p, uint32_t k) {
    return size_t(cutie_topk_slices(m)) * p * k;
}

// aff [m, p] must be zeroed; keys and idx hold makepad_cuda_cutie_topk_scratch_entries each.
extern "C" cudaError_t makepad_cuda_cutie_topk_softmax_f32(
    const float* sim, float* aff, float* usage, uint32_t* keys, uint32_t* idx, uint32_t m, uint32_t p, uint32_t k,
    cudaStream_t stream) {
    if (k > CUTIE_TOPK_MAX || k > m) return cudaErrorInvalidValue;
    if (!m || !p) return cudaSuccess;
    if (k) {
        const uint32_t slices = cutie_topk_slices(m);
        const uint32_t chunk = (m + slices - 1) / slices;
        const dim3 block(CUTIE_TOPK_COLS, CUTIE_TOPK_SLICES_PER_BLOCK, 1);
        const dim3 grid((p + CUTIE_TOPK_COLS - 1) / CUTIE_TOPK_COLS,
                        (slices + CUTIE_TOPK_SLICES_PER_BLOCK - 1) / CUTIE_TOPK_SLICES_PER_BLOCK, 1);
        if (k <= 32) {
            cutie_topk_partial_kernel<32><<<grid, block, 0, stream>>>(sim, keys, idx, m, p, k, chunk, slices);
        } else {
            cutie_topk_partial_kernel<64><<<grid, block, 0, stream>>>(sim, keys, idx, m, p, k, chunk, slices);
        }
        cutie_topk_merge_kernel<<<(p + 3) / 4, 128, 0, stream>>>(keys, idx, aff, m, p, k, slices);
    }
    cutie_row_sum_kernel<<<(m + 7) / 8, 256, 0, stream>>>(aff, usage, m, p);
    return cudaGetLastError();
}

// Softmax down each column of [m, p], one thread a column: the max, exp(x - max) summed in row
// order, divided. PyTorch's spatial softmax over a few rows (the decoder's F.softmax over
// background and objects) does exactly this.
static __global__ void cutie_softmax_axis0_kernel(const float* __restrict__ x, float* __restrict__ out, uint32_t m, uint32_t p) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= p) return;
    float max = -CUDART_INF_F;
    for (uint32_t j = 0; j < m; ++j) max = fmaxf(max, x[size_t(j) * p + col]);
    float sum = 0.0f;
    for (uint32_t j = 0; j < m; ++j) {
        const float e = expf(__fsub_rn(x[size_t(j) * p + col], max));
        out[size_t(j) * p + col] = e;
        sum = __fadd_rn(sum, e);
    }
    for (uint32_t j = 0; j < m; ++j) out[size_t(j) * p + col] = out[size_t(j) * p + col] / sum;
}

// The same for few columns over many rows (memory consolidation: candidates by 128 prototypes),
// where a thread a column would leave the GPU idle: a block a tile of CUTIE_SOFTMAX_TILE_COLS
// columns, its CUTIE_SOFTMAX_TILE_ROWS thread rows striding down them (each warp reads a few
// rows' sectors), max and sum reduced across the thread rows in a fixed tree. PyTorch's own
// order there (torch.sum over the candidates) is a multi-block reduce: fp32 rounding apart
// either way.
#define CUTIE_SOFTMAX_TILE_COLS 8
#define CUTIE_SOFTMAX_TILE_ROWS 128
static __global__ void __launch_bounds__(CUTIE_SOFTMAX_TILE_COLS * CUTIE_SOFTMAX_TILE_ROWS) cutie_softmax_axis0_tile_kernel(
    const float* __restrict__ x, float* __restrict__ out, uint32_t m, uint32_t p) {
    __shared__ float shared[CUTIE_SOFTMAX_TILE_ROWS][CUTIE_SOFTMAX_TILE_COLS];
    const uint32_t tx = threadIdx.x, ty = threadIdx.y;
    const uint32_t col = blockIdx.x * CUTIE_SOFTMAX_TILE_COLS + tx;
    const bool live = col < p;
    float max = -CUDART_INF_F;
    if (live) {
        for (uint32_t j = ty; j < m; j += CUTIE_SOFTMAX_TILE_ROWS) max = fmaxf(max, x[size_t(j) * p + col]);
    }
    shared[ty][tx] = max;
    __syncthreads();
    for (uint32_t s = CUTIE_SOFTMAX_TILE_ROWS / 2; s > 0; s >>= 1) {
        if (ty < s) shared[ty][tx] = fmaxf(shared[ty][tx], shared[ty + s][tx]);
        __syncthreads();
    }
    max = shared[0][tx];
    __syncthreads();
    float sum = 0.0f;
    if (live) {
        for (uint32_t j = ty; j < m; j += CUTIE_SOFTMAX_TILE_ROWS) {
            const float e = expf(__fsub_rn(x[size_t(j) * p + col], max));
            out[size_t(j) * p + col] = e;
            sum = __fadd_rn(sum, e);
        }
    }
    shared[ty][tx] = sum;
    __syncthreads();
    for (uint32_t s = CUTIE_SOFTMAX_TILE_ROWS / 2; s > 0; s >>= 1) {
        if (ty < s) shared[ty][tx] = __fadd_rn(shared[ty][tx], shared[ty + s][tx]);
        __syncthreads();
    }
    sum = shared[0][tx];
    if (live) {
        for (uint32_t j = ty; j < m; j += CUTIE_SOFTMAX_TILE_ROWS) out[size_t(j) * p + col] = out[size_t(j) * p + col] / sum;
    }
}

extern "C" cudaError_t makepad_cuda_cutie_softmax_axis0_f32(const float* x, float* out, uint32_t m, uint32_t p, cudaStream_t stream) {
    if (!m || !p) return cudaSuccess;
    if (m > 64 && p < 8192) {
        const dim3 block(CUTIE_SOFTMAX_TILE_COLS, CUTIE_SOFTMAX_TILE_ROWS, 1);
        cutie_softmax_axis0_tile_kernel<<<(p + CUTIE_SOFTMAX_TILE_COLS - 1) / CUTIE_SOFTMAX_TILE_COLS, block, 0, stream>>>(x, out, m, p);
    } else {
        cutie_softmax_axis0_kernel<<<(p + 127) / 128, 128, 0, stream>>>(x, out, m, p);
    }
    return cudaGetLastError();
}

// --- Objects ---

// Cutie's aggregate: [n, p] probabilities to [n+1, p] logits, the background prod(1 - prob)
// (multiplied in PyTorch's order) first.
static __global__ void cutie_aggregate_kernel(const float* __restrict__ prob, float* __restrict__ out, uint32_t n, size_t p) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < p; i += size_t(gridDim.x) * blockDim.x) {
        const float bg = cutie_reduce4<true>(n, [&](uint32_t o) { return __fsub_rn(1.0f, prob[o * p + i]); });
        for (uint32_t o = 0; o < n; ++o) out[(o + 1) * p + i] = cutie_logit(prob[o * p + i]);
        out[i] = cutie_logit(bg);
    }
}

extern "C" cudaError_t makepad_cuda_cutie_aggregate_f32(const float* prob, float* out, uint32_t n, size_t p, cudaStream_t stream) {
    if (!p) return cudaSuccess;
    cutie_aggregate_kernel<<<cutie_blocks(p, 256), 256, 0, stream>>>(prob, out, n, p);
    return cudaGetLastError();
}

// The attention mask's foreground: per pixel, each object's aggregated logit (of the sigmoid of
// its aux logit) against the largest of all rows, the background's included. fg [n, p] (0/1);
// seen[2o] set when object o has any foreground, seen[2o+1] when it has any background.
static __global__ void cutie_attn_foreground_kernel(
    const float* __restrict__ aux, uint32_t* __restrict__ fg, uint32_t* __restrict__ seen, uint32_t n, size_t p) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < p; i += size_t(gridDim.x) * blockDim.x) {
        const float bg = cutie_reduce4<true>(n, [&](uint32_t o) { return __fsub_rn(1.0f, cutie_sigmoid(aux[o * p + i])); });
        // A NaN anywhere makes the max NaN (logits.max propagates it), and then no row is >= it.
        float max = cutie_logit(bg);
        for (uint32_t o = 0; o < n; ++o) max = cutie_max_nan(max, cutie_logit(cutie_sigmoid(aux[o * p + i])));
        for (uint32_t o = 0; o < n; ++o) {
            const uint32_t is_fg = cutie_logit(cutie_sigmoid(aux[o * p + i])) >= max ? 1u : 0u;
            fg[o * p + i] = is_fg;
            atomicOr(&seen[2 * o + (is_fg ? 0 : 1)], 1u);
        }
    }
}

// The additive mask [n*heads*queries, p]: the first half of each object's queries see only its
// foreground, the rest only the rest; a row that would see nothing sees everything.
static __global__ void cutie_attn_mask_kernel(
    const uint32_t* __restrict__ fg, const uint32_t* __restrict__ seen, float* __restrict__ out,
    uint32_t heads, uint32_t queries, size_t p, size_t rows) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= p) return;
    for (size_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const size_t o = row / (size_t(heads) * queries);
        const uint32_t q = uint32_t(row % queries);
        const bool foreground_query = q < queries / 2;
        const bool open_all = foreground_query ? !seen[2 * o] : !seen[2 * o + 1];
        const bool blocked = !open_all && ((fg[o * p + i] != 0) != foreground_query);
        out[row * p + i] = blocked ? -CUDART_INF_F : 0.0f;
    }
}

// scratch: n*p + 2n u32, the last 2n zeroed by the caller.
extern "C" cudaError_t makepad_cuda_cutie_attn_mask_f32(
    const float* aux, float* out, uint32_t* scratch, uint32_t n, size_t p, uint32_t heads, uint32_t queries,
    cudaStream_t stream) {
    const size_t rows = size_t(n) * heads * queries;
    if (!rows || !p) return cudaSuccess;
    uint32_t* fg = scratch;
    uint32_t* seen = scratch + size_t(n) * p;
    cutie_attn_foreground_kernel<<<cutie_blocks(p, 256), 256, 0, stream>>>(aux, fg, seen, n, p);
    const dim3 grid((unsigned)((p + 255) / 256), (unsigned)(rows < 65535 ? rows : 65535), 1);
    cutie_attn_mask_kernel<<<grid, 256, 0, stream>>>(fg, seen, out, heads, queries, p, rows);
    return cudaGetLastError();
}
