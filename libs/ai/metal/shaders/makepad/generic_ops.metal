// The generic gpu_* ops' own kernels (metal/src/generic_ops.rs names and launches them; slices and
// concatenations of columns use tensor.metal's copy): CUDA's pixal_pool_kernel
// (cuda/kernels/pixal.cu), makepad_cuda_paint_transpose_f32_kernel (paint_extras.cu) and
// makepad_cuda_realesrgan_scale_add_f32_kernel (diffusion_ops.cu), one thread an output with
// CUDA's indexing and CUDA's arithmetic, so each result is CUDA's to the bit. Tensors are bound at
// any 4-byte offset, so they are `device` pointers, never `constant`, read one word at a time.
// Nothing narrower than fp32 anywhere (D8).
#include "makepad_common.h"

struct GenericPixalPool {
    uint width;
    uint height;
    uint out_width;
    uint out_height;
    uint n;
};

struct GenericTranspose {
    uint rows;
    uint cols;
    uint n;
};

struct GenericScaleAdd {
    uint n;
    float scale;
};

// Adaptive average pooling of planar [channels, height*width] to [channels, out_height*out_width]:
// each output the sum of its window [floor(i*in/out), ceil((i+1)*in/out)) in row order, then one
// division by the window's area. Adds and a division only, so nothing can be contracted. Every
// size is at most 65535 (the host checks, as CUDA's does), so (px+1)*width + out_width - 1 fits a
// uint, as in CUDA's.
kernel void generic_pixal_pool_f32(constant GenericPixalPool& p [[buffer(0)]],
                                   device const float* x [[buffer(1)]],
                                   device float* out [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint plane = p.out_width * p.out_height;
    const uint c = i / plane;
    const uint px = i % p.out_width;
    const uint py = (i % plane) / p.out_width;
    const uint x0 = px * p.width / p.out_width;
    const uint x1 = ((px + 1) * p.width + p.out_width - 1) / p.out_width;
    const uint y0 = py * p.height / p.out_height;
    const uint y1 = ((py + 1) * p.height + p.out_height - 1) / p.out_height;
    float sum = 0.0f;
    for (uint y = y0; y < y1; ++y) {
        const ulong row = (ulong(c) * ulong(p.height) + ulong(y)) * ulong(p.width);
        for (uint xx = x0; xx < x1; ++xx) {
            sum += x[row + ulong(xx)];
        }
    }
    out[i] = sum / float((x1 - x0) * (y1 - y0));
}

// [rows, cols] to [cols, rows] (planar [C, HW] to tokens [HW, C] and back): a word moved, so every
// bit pattern (NaN payloads included) arrives unchanged.
kernel void generic_transpose_u32(constant GenericTranspose& p [[buffer(0)]],
                                  device const uint* x [[buffer(1)]],
                                  device uint* out [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= p.n) {
        return;
    }
    const uint r = i / p.cols;
    const uint c = i - r * p.cols;
    out[ulong(c) * ulong(p.rows) + ulong(r)] = x[i];
}

// base + scale * delta with one rounding: CUDA's fmaf(scale, delta, base), written as fma() so no
// contraction setting changes it.
kernel void generic_scale_add_f32(constant GenericScaleAdd& p [[buffer(0)]],
                                  device const float* base [[buffer(1)]],
                                  device const float* delta [[buffer(2)]],
                                  device float* out [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i < p.n) {
        out[i] = fma(p.scale, delta[i], base[i]);
    }
}
