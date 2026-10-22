// Cutie's softmax down columns, aggregate, ECA and sine encoding (RationalVisions
// docs/cutie-metal.md, family (c) of step 10): ports of cuda/kernels/cutie.cu's
// cutie_softmax_axis0_kernel, cutie_softmax_axis0_tile_kernel, cutie_aggregate_kernel,
// cutie_row_mean_kernel, cutie_eca_scale_kernel, cutie_eca_apply_kernel and
// cutie_sine_encoding_kernel, in the same order of evaluation, launched by
// metal/src/cutie_ops/softmax_norm.rs. Each op's host reference is in cuda/src/cutie_reference.rs.
// The family's layer norm runs PyTorch's LayerNorm kernel, which gpu_layer_norm_pytorch shares and
// every model calls, so it is a library of its own (layer_norm.metal).
//
// Exactness (D8): fast math is off.
// - Where cutie.cu writes __fmul_rn and __fadd_rn (ECA's taps, and x * scale + residual) the
//   product and the sum are separate statements, so no contraction can fuse them (step 7's
//   setting in makepad_common.h stops it outright).
// - exp, log, sin and cos are precise::, and a division is IEEE's. CUDA's expf is ex2.approx
//   underneath and its sinf and cosf are not correctly rounded, so the softmax, aggregate, ECA and
//   sine are fp32 rounding from CUDA's (the ops check's TIGHT).
// - The reductions keep CUDA's trees: the softmax tile's tree over its thread rows and the ECA row
//   mean's butterfly (simd_shuffle_xor). The butterfly takes CUDA's 32 lanes: the host refuses a
//   GPU whose SIMD-groups are another width.
// Indices that multiply two sizes are ulong, as CUDA's size_t. No 16-bit float anywhere.
#include "makepad_common.h"

// This library's helpers, in a namespace of their own so they never meet makepad_common.h's.
namespace softmax_norm {

// PyTorch's sigmoid for f32 (precise exp, IEEE division): cutie.cu's cutie_sigmoid.
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + precise::exp(-x));
}

// Cutie's aggregate logit: clamp to 1e-7..1-1e-7 (NaN passes through, as torch.clamp), then
// log(q / (1 - q)): cutie.cu's cutie_logit.
static inline float logit(float q) {
    const float lo = 1e-7f;
    const float hi = 1.0f - 1e-7f;
    q = q < lo ? lo : (q > hi ? hi : q);
    return precise::log(q / (1.0f - q));
}

}  // namespace softmax_norm

// --- Softmax down columns ---

// x [rows, cols]: softmax over the rows of each column.
struct CutieSoftmaxAxis0 {
    uint rows;
    uint cols;
};

// One thread a column (the decoder's background and objects, a few rows): the largest, exp(x -
// largest) summed in row order, divided. PyTorch's spatial softmax over a few rows does exactly
// this. Threadgroups of 128 threads.
kernel void cutie_softmax_axis0_f32(constant CutieSoftmaxAxis0& p [[buffer(0)]],
                                    device const float* x [[buffer(1)]],
                                    device float* out [[buffer(2)]],
                                    uint col [[thread_position_in_grid]]) {
    if (col >= p.cols) {
        return;
    }
    float largest = -INFINITY;
    for (uint j = 0; j < p.rows; ++j) {
        largest = fmax(largest, x[ulong(j) * p.cols + col]);
    }
    float sum = 0.0f;
    for (uint j = 0; j < p.rows; ++j) {
        const ulong at = ulong(j) * p.cols + col;
        const float e = precise::exp(x[at] - largest);
        out[at] = e;
        sum = sum + e;
    }
    for (uint j = 0; j < p.rows; ++j) {
        const ulong at = ulong(j) * p.cols + col;
        out[at] = out[at] / sum;
    }
}

#define CUTIE_SOFTMAX_TILE_COLS 8
#define CUTIE_SOFTMAX_TILE_ROWS 128

// The same for few columns over many rows (memory consolidation: candidates by 128 prototypes),
// where a thread a column would leave the GPU idle: a threadgroup [TILE_COLS, TILE_ROWS] a tile of
// TILE_COLS columns, its TILE_ROWS thread rows striding down them, the largest and the sum reduced
// across the thread rows in a fixed tree, each step pairing thread row ty with ty + s.
kernel void cutie_softmax_axis0_tile_f32(constant CutieSoftmaxAxis0& p [[buffer(0)]],
                                         device const float* x [[buffer(1)]],
                                         device float* out [[buffer(2)]],
                                         uint2 group [[threadgroup_position_in_grid]],
                                         uint2 thread_in_group [[thread_position_in_threadgroup]]) {
    threadgroup float shared[CUTIE_SOFTMAX_TILE_ROWS][CUTIE_SOFTMAX_TILE_COLS];
    const uint tx = thread_in_group.x;
    const uint ty = thread_in_group.y;
    const uint col = group.x * CUTIE_SOFTMAX_TILE_COLS + tx;
    const bool live = col < p.cols;
    float largest = -INFINITY;
    if (live) {
        for (uint j = ty; j < p.rows; j += CUTIE_SOFTMAX_TILE_ROWS) {
            largest = fmax(largest, x[ulong(j) * p.cols + col]);
        }
    }
    shared[ty][tx] = largest;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = CUTIE_SOFTMAX_TILE_ROWS / 2; s > 0; s >>= 1) {
        if (ty < s) {
            shared[ty][tx] = fmax(shared[ty][tx], shared[ty + s][tx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    largest = shared[0][tx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    if (live) {
        for (uint j = ty; j < p.rows; j += CUTIE_SOFTMAX_TILE_ROWS) {
            const ulong at = ulong(j) * p.cols + col;
            const float e = precise::exp(x[at] - largest);
            out[at] = e;
            sum = sum + e;
        }
    }
    shared[ty][tx] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = CUTIE_SOFTMAX_TILE_ROWS / 2; s > 0; s >>= 1) {
        if (ty < s) {
            shared[ty][tx] = shared[ty][tx] + shared[ty + s][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum = shared[0][tx];
    if (live) {
        for (uint j = ty; j < p.rows; j += CUTIE_SOFTMAX_TILE_ROWS) {
            const ulong at = ulong(j) * p.cols + col;
            out[at] = out[at] / sum;
        }
    }
}

// --- Objects ---

// prob [objects, pixels] to out [objects + 1, pixels].
struct CutieAggregate {
    uint objects;
    uint pixels;
};

// Cutie's aggregate, one thread a pixel: the background prod(1 - prob) in PyTorch's order for a
// reduction that is not the innermost (cutie.cu's cutie_reduce4<true>: four accumulators, term o
// into o % 4, then ((a0 * a1) * a2) * a3) first, then every row's logit.
kernel void cutie_aggregate_f32(constant CutieAggregate& p [[buffer(0)]],
                                device const float* prob [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p.pixels) {
        return;
    }
    float acc[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    for (uint o = 0; o < p.objects; o += 4) {
        for (uint j = 0; j < 4; ++j) {
            if (o + j < p.objects) {
                const float rest = 1.0f - prob[ulong(o + j) * p.pixels + i];
                acc[j] = acc[j] * rest;
            }
        }
    }
    const float first_two = acc[0] * acc[1];
    const float first_three = first_two * acc[2];
    const float background = first_three * acc[3];
    for (uint o = 0; o < p.objects; ++o) {
        out[ulong(o + 1) * p.pixels + i] = softmax_norm::logit(prob[ulong(o) * p.pixels + i]);
    }
    out[i] = softmax_norm::logit(background);
}

// --- ECA ---

// x [rows, plane] to means [rows].
struct CutieEcaRowMean {
    uint plane;
};

// The mean of each row, a threadgroup of 256 threads a row: each thread's strided sum, a
// simd_shuffle_xor butterfly in each SIMD-group (cutie.cu's __shfl_xor_sync), the groups' sums
// through threadgroup memory, the same butterfly in the first group (lanes past the groups add
// 0), then the sum over the plane.
kernel void cutie_eca_row_mean_f32(constant CutieEcaRowMean& p [[buffer(0)]],
                                   device const float* x [[buffer(1)]],
                                   device float* means [[buffer(2)]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint thread_in_group [[thread_position_in_threadgroup]],
                                   uint group_size [[threads_per_threadgroup]]) {
    threadgroup float warp_sums[32];
    device const float* values = x + ulong(row) * p.plane;
    float sum = 0.0f;
    for (uint i = thread_in_group; i < p.plane; i += group_size) {
        sum = sum + values[i];
    }
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        sum = sum + simd_shuffle_xor(sum, offset);
    }
    const uint warp = thread_in_group / 32;
    const uint lane = thread_in_group % 32;
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0) {
        sum = lane < (group_size + 31) / 32 ? warp_sums[lane] : 0.0f;
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            sum = sum + simd_shuffle_xor(sum, offset);
        }
        if (lane == 0) {
            means[row] = sum / float(p.plane);
        }
    }
}

// ECA's taps travel by value, as on CUDA: Cutie's are 3 or 5 long; the rest are 0.
#define CUTIE_ECA_MAX_TAPS 15

// means [rows] to scale [rows]; rows = images * channels.
struct CutieEcaScale {
    float taps[CUTIE_ECA_MAX_TAPS];
    uint k;
    uint channels;
    uint rows;
};

// Per image and channel, a thread each: the sigmoid of the 1-D convolution over its image's
// channel means (zero padding (k - 1) / 2), each tap's product rounded before it is added.
kernel void cutie_eca_scale_f32(constant CutieEcaScale& p [[buffer(0)]],
                                device const float* means [[buffer(1)]],
                                device float* scale [[buffer(2)]],
                                uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) {
        return;
    }
    const int ch = int(row % p.channels);
    const int base = int(row - row % p.channels);
    const int pad = int(p.k - 1) / 2;
    float y = 0.0f;
    for (uint j = 0; j < p.k; ++j) {
        const int src = ch + int(j) - pad;
        if (src >= 0 && src < int(p.channels)) {
            const float term = p.taps[j] * means[base + src];
            y = y + term;
        }
    }
    scale[row] = softmax_norm::sigmoid(y);
}

// x, residual and out [rows, plane]; n = rows * plane.
struct CutieEcaApply {
    uint plane;
    uint n;
};

// out = x * scale[row] + residual, the product rounded before the sum.
kernel void cutie_eca_apply_f32(constant CutieEcaApply& p [[buffer(0)]],
                                device const float* x [[buffer(1)]],
                                device const float* residual [[buffer(2)]],
                                device const float* scale [[buffer(3)]],
                                device float* out [[buffer(4)]],
                                uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const float scaled = x[i] * scale[i / p.plane];
    out[i] = scaled + residual[i];
}

// --- Sine encoding ---

// inv_freq [freqs] to out [h * w, 4 * freqs]; n = w * h * 4 * freqs.
struct CutieSineEncoding {
    uint freqs;
    uint w;
    uint h;
    float scale;
    float eps;
    uint n;
};

// Cutie's PositionalEncoding (channel-last, normalised), a thread an entry: token y * w + x, the x
// encoding in the first 2 * freqs channels and the y encoding in the rest, sin and cos
// interleaved per frequency. Each axis' position i / ((len - 1) + eps) * scale, times the
// frequency, every op rounded as PyTorch's tensor ops round it; then precise::sin and cos where
// CUDA calls its own sinf and cosf.
kernel void cutie_sine_encoding_f32(constant CutieSineEncoding& p [[buffer(0)]],
                                    device const float* inv_freq [[buffer(1)]],
                                    device float* out [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint dim = 4 * p.freqs;
    const uint axis_dim = 2 * p.freqs;
    const uint token = i / dim;
    const uint ch = i % dim;
    const bool along_x = ch < axis_dim;
    const uint len = along_x ? p.w : p.h;
    const uint at = along_x ? token % p.w : token / p.w;
    const uint axis_ch = along_x ? ch : ch - axis_dim;
    const float last = float(len - 1) + p.eps;
    const float ratio = float(at) / last;
    const float pos = ratio * p.scale;
    const float angle = pos * inv_freq[axis_ch / 2];
    out[i] = (axis_ch & 1) ? precise::cos(angle) : precise::sin(angle);
}
